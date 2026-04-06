"""In-process backend — runs RLM engine directly in the Gradio process.

Dev fallback when the FastAPI service is not running. Uses the Irys class
(which wires the matter model) and exposes the same interface as HttpBackend.
Not suitable for production — only one Gradio worker, no deployment isolation.
"""

import asyncio
import os
import queue
import threading
from typing import Any, AsyncIterator, Optional

from ...api import Irys, IrysConfig
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
        so: dict = {}
        try:
            so = model.get_so_metrics()
        except Exception:
            pass
        recent = model.ledger.recent_runs(limit=5)
        weakest: list = []
        try:
            coverage = model.get_issue_coverage_report()
            if coverage:
                weakest = sorted(coverage, key=lambda r: float(r.get("coverage_fraction", 0.0)))[:5]
        except Exception:
            pass
        top_gaps: list = []
        try:
            top_gaps = model.gaps.get_open(limit=5)
        except Exception:
            try:
                top_gaps = model.gaps.list_open()[:5]
            except Exception:
                pass
        clarifications: list = []
        try:
            clarifications = model.clarifications.get_pending()[:5]
        except Exception:
            pass
        return {
            "matter_id": matter_id,
            "stats": stats,
            "so_metrics": so,
            "recent_runs": recent,
            "weakest_issues": weakest,
            "top_gaps": top_gaps,
            "pending_clarifications": clarifications,
            "source_role_summary": {},
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
        model = self._get_matter_model(matter_id)
        try:
            return model.issues.list_open()
        except Exception:
            return []

    async def list_assertions(
        self, matter_id: str, limit: int = 50, offset: int = 0,
        issue_id: Optional[str] = None
    ) -> list[dict]:
        model = self._get_matter_model(matter_id)
        try:
            return model.assertions.list_recent(limit=limit, offset=offset)
        except Exception:
            return []

    async def list_gaps(self, matter_id: str, limit: int = 50) -> list[dict]:
        model = self._get_matter_model(matter_id)
        try:
            return model.gaps.get_open(limit=limit)
        except Exception:
            try:
                return model.gaps.list_open()[:limit]
            except Exception:
                return []

    async def list_clarifications(self, matter_id: str) -> list[dict]:
        model = self._get_matter_model(matter_id)
        try:
            return model.clarifications.get_pending()
        except Exception:
            return []

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
            model.correct_assertion(assertion_id, new_state, reason=reason)
            return {"status": "corrected", "assertion_id": assertion_id}
        except Exception as exc:
            return {"status": "error", "detail": str(exc)}

    async def redirect_run(
        self, matter_id: str, run_id: str, issue_id: str
    ) -> dict:
        model = self._get_matter_model(matter_id)
        try:
            model.ledger.request_redirect(run_id, issue_id)
            return {"status": "redirect_requested", "issue_id": issue_id}
        except Exception as exc:
            return {"status": "error", "detail": str(exc)}
