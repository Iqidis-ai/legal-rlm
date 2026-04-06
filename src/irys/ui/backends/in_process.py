"""In-process backend — runs RLM engine directly in the Gradio process.

Dev fallback when the FastAPI service is not running. Uses the Irys class
(which wires the matter model) and exposes the same interface as HttpBackend.
Not suitable for production — only one Gradio worker, no deployment isolation.
"""

import asyncio
import heapq
import os
import queue
import threading
from typing import Any, AsyncIterator, Optional

from ...api import Irys, IrysConfig
from ...matter.enums import BeliefState
from ...rlm.state import InvestigationState, StepType, ThinkingStep
from .base import UIBackend


class InProcessBackend(UIBackend):
    """In-process backend for local dev without a running service."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._irys: Optional[Irys] = None
        # Active run tracking: run_id → event queue
        self._event_queues: dict[str, queue.Queue] = {}
        self._active_runs: dict[str, dict] = {}

    def _get_irys(self) -> Irys:
        if self._irys is None:
            self._irys = Irys(
                config=IrysConfig(
                    api_key=self.api_key,
                    enable_matter_model=True,
                )
            )
        return self._irys

    def _get_matter_model(self, matter_id: str):
        irys = self._get_irys()
        for model in irys._matter_models.values():
            if model.matter_id == matter_id:
                return model
        raise ValueError(f"Matter '{matter_id}' not in active models")

    # ------------------------------------------------------------------ #
    # Investigation control                                                #
    # ------------------------------------------------------------------ #

    async def start_investigation(
        self,
        repo_path: str,
        query: str,
        matter_id: Optional[str] = None,
    ) -> dict:
        """Start an investigation in-process. Returns after investigation completes."""
        irys = self._get_irys()
        result = await irys.investigate(query, repo_path)
        state = result.state
        # Get matter_id and run_id from the matter model if available
        engine = irys._engine
        mm = engine._matter_model if engine else None
        resolved_matter_id = mm.matter_id if mm else (matter_id or "unknown")
        run_id = getattr(state, "_run_id", None) or "unknown"
        return {
            "matter_id": resolved_matter_id,
            "run_id": run_id,
            "output": result.output,
            "state": state,
            "citations": [
                {"document": c.document, "page": c.page, "text": c.text}
                for c in state.citations
            ],
        }

    async def stop_run(self, matter_id: str, run_id: str) -> dict:
        try:
            model = self._get_matter_model(matter_id)
            model.ledger.request_stop(run_id)
            return {"status": "stop_requested", "run_id": run_id}
        except Exception as exc:
            return {"status": "error", "detail": str(exc)}

    # ------------------------------------------------------------------ #
    # Overview / dashboard                                                  #
    # ------------------------------------------------------------------ #

    async def get_overview(self, matter_id: str) -> dict:
        model = self._get_matter_model(matter_id)
        stats = model.stats()
        # Fetch coverage once — reused by both get_so_metrics() and weakest_issues.
        coverage_report: list = []
        try:
            coverage_report = model.get_issue_coverage_report()
        except Exception:
            pass
        so: dict = {}
        try:
            so = model.get_so_metrics(_coverage_report=coverage_report)
        except Exception:
            pass
        weakest: list = (
            heapq.nsmallest(5, coverage_report, key=lambda r: float(r.get("coverage_fraction", 0.0)))
            if coverage_report else []
        )
        top_gaps: list = []
        try:
            top_gaps = model.gaps.open_gaps(limit=5)
        except Exception:
            pass
        clarifications: list = []
        try:
            clarifications = model.clarifications.get_pending(limit=5)
        except Exception:
            pass
        return {
            "matter_id": matter_id,
            "stats": stats,
            "so_metrics": so,
            "weakest_issues": weakest,
            "top_gaps": top_gaps,
            "pending_clarifications": clarifications,
        }

    # ------------------------------------------------------------------ #
    # Live ledger streaming                                                 #
    # ------------------------------------------------------------------ #

    async def stream_run_events(
        self, matter_id: str, run_id: str, after_seq: int = -1
    ) -> AsyncIterator[dict]:
        """Poll the DB for new ledger events during a live run."""
        model = self._get_matter_model(matter_id)
        last_seq = after_seq
        terminal_statuses = {"completed", "failed", "interrupted"}

        # Validate run_id exists before entering the poll loop to prevent
        # infinite polling on bad/stale run IDs (MEDIUM 4 fix).
        try:
            run_check = model.db.execute(
                "SELECT id FROM run_session WHERE id=?", (run_id,)
            ).fetchone()
        except Exception as exc:
            yield {"error": str(exc)}
            return
        if run_check is None:
            yield {"error": f"run_id {run_id!r} not found"}
            return

        while True:
            try:
                rows = model.db.execute(
                    """SELECT id, run_id, seq_no, event_type, summary, why,
                              created_at
                       FROM ledger_event
                       WHERE run_id=? AND seq_no > ?
                       ORDER BY seq_no""",
                    (run_id, last_seq),
                ).fetchall()
                for row in rows:
                    event = dict(row)
                    last_seq = event["seq_no"]
                    yield event
            except Exception:
                pass

            # Check terminal status
            try:
                run_row = model.db.execute(
                    "SELECT status FROM run_session WHERE id=?", (run_id,)
                ).fetchone()
                if run_row and run_row["status"] in terminal_statuses:
                    yield {"event": "run_terminal", "status": run_row["status"]}
                    break
            except Exception:
                pass

            await asyncio.sleep(0.5)

    # ------------------------------------------------------------------ #
    # Matter model data                                                    #
    # ------------------------------------------------------------------ #

    async def list_runs(self, matter_id: str, limit: int = 10) -> list[dict]:
        model = self._get_matter_model(matter_id)
        return model.ledger.recent_runs(limit=limit)

    async def get_run_events(self, matter_id: str, run_id: str) -> list[dict]:
        model = self._get_matter_model(matter_id)
        return model.ledger.get_events(run_id)

    async def list_issues(self, matter_id: str) -> list[dict]:
        # No try/except — let exceptions propagate so AppState.load_issues()
        # can display "Error loading issues: <detail>" instead of a silent empty table.
        model = self._get_matter_model(matter_id)
        return model.get_issue_coverage_report()

    async def list_assertions(
        self, matter_id: str, limit: int = 50, offset: int = 0
    ) -> list[dict]:
        model = self._get_matter_model(matter_id)
        return model.assertions.list_recent(limit=limit, offset=offset)

    async def list_gaps(self, matter_id: str, limit: int = 50) -> list[dict]:
        model = self._get_matter_model(matter_id)
        return model.gaps.open_gaps(limit=limit)

    async def list_clarifications(self, matter_id: str, limit: int = 20) -> list[dict]:
        model = self._get_matter_model(matter_id)
        return model.clarifications.get_pending(limit=limit)

    # ------------------------------------------------------------------ #
    # User steering                                                        #
    # ------------------------------------------------------------------ #

    async def correct_assertion(
        self,
        matter_id: str,
        assertion_id: str,
        new_state: str,
        reason: str,
    ) -> dict:
        model = self._get_matter_model(matter_id)
        try:
            belief_state = BeliefState(new_state)
        except ValueError:
            return {"status": "error", "detail": f"Invalid belief state: {new_state!r}"}
        try:
            result = model.correct_assertion(assertion_id, belief_state, note=reason)
            resp = {"status": "corrected", "assertion_id": assertion_id}
            if getattr(result, "propagation_truncated", False):
                resp["warning"] = "Belief revision truncated — proof state will refresh on next run"
            return resp
        except Exception as exc:
            return {"status": "error", "detail": str(exc)}

    async def redirect_run(
        self, matter_id: str, run_id: str, issue_id: str
    ) -> dict:
        model = self._get_matter_model(matter_id)
        try:
            run = model.ledger.get_run(run_id)
            if run is None:
                return {"status": "error", "detail": f"Run '{run_id}' not found"}
            if run.status != "running":
                return {"status": "error", "detail": f"Run is not active (status: {run.status})"}
            issue = model.issues.get_issue(issue_id)
            if issue is None:
                return {"status": "error", "detail": f"Issue '{issue_id}' not found"}
            model.ledger.request_redirect(run_id, issue_id)
            return {
                "status": "redirect_requested",
                "run_id": run_id,
                "issue_id": issue_id,
                "issue_title": issue.get("title"),
            }
        except Exception as exc:
            return {"status": "error", "detail": str(exc)}

    # ------------------------------------------------------------------ #
    # SO-3 / SO-6 supplemental surfaces                                   #
    # ------------------------------------------------------------------ #

    async def get_steering_surface(
        self, matter_id: str, run_id: Optional[str] = None
    ) -> list[dict]:
        model = self._get_matter_model(matter_id)
        return model.get_ledger_steering_surface(run_id=run_id)

    async def get_quant_summary(self, matter_id: str) -> dict:
        model = self._get_matter_model(matter_id)
        return {
            "payment_reconciliation": model.reconcile_payment_chain(),
            "damages_waterfall": model.get_damages_waterfall(),
        }

    # ------------------------------------------------------------------ #
    # Streaming investigation (InProcessBackend-specific)                 #
    # ------------------------------------------------------------------ #

    def run_investigation_thread(
        self,
        query: str,
        repo_path: str,
        update_q: "queue.Queue",
        thinking: list,
        citations: list,
        on_irys_created=None,
        on_step=None,
        set_current_run_id=None,
        set_current_matter_id=None,
        set_final_output=None,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        """Run investigation in the calling thread (which must be a daemon Thread).

        Encapsulates irys.investigate() + UI update queue logic. AppState._run_thread
        delegates here so the Run tab goes through the backend, not around it.
        Callbacks let AppState track run_id, matter_id, and irys ref for stop/redirect.

        stop_event: if set before or during thread startup, cancels before investigate()
        begins (handles early-stop race where run_session doesn't exist yet).
        """
        import queue as _queue

        async def _inner():
            # Early-stop check: if user pressed Stop before investigation starts, bail.
            if stop_event is not None and stop_event.is_set():
                update_q.put(("error", "Investigation stopped before it began."))
                return
            irys = self._get_irys()
            if on_irys_created is not None:
                on_irys_created(irys)
            if on_step is not None:
                irys.on_step(on_step)
            # Second check after irys ref is set (covers the on_irys_created window).
            if stop_event is not None and stop_event.is_set():
                update_q.put(("error", "Investigation stopped before it began."))
                return
            try:
                result = await irys.investigate(query, repo_path)
                state = result.state
                engine = irys._engine
                mm = engine._matter_model if engine else None
                if set_current_matter_id is not None:
                    set_current_matter_id(mm.matter_id if mm else None)
                if set_current_run_id is not None:
                    set_current_run_id(getattr(state, "_run_id", None))
                citations.extend(
                    f"[{i+1}] {c.document}" + (f", p.{c.page}" if c.page else "")
                    for i, c in enumerate(state.citations)
                )
                if set_final_output is not None:
                    set_final_output(result.output)
                update_q.put(("complete", state))
            except Exception as exc:
                update_q.put(("error", str(exc)))

        asyncio.run(_inner())
