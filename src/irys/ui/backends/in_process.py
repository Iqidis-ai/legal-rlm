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

import logging as _logging
_log = _logging.getLogger(__name__)


def _do_one_in_process_flush(matter_id: str, model) -> None:
    """Execute a single flush pass. Called from within the flush loop with _bg_flush_running held."""
    from ...matter.runtime import MatterRuntimeAdapter
    with model._flush_lock:
        flush_run_id = model.start_run(
            "Background flush", objective="background_flush",
            operation_type="maintenance", trigger="system",
        )
        try:
            adapter = MatterRuntimeAdapter(model, run_id=flush_run_id)
            adapter._flush_revisions_locked()
        except Exception as exc:
            try:
                model.fail_run(flush_run_id, str(exc))
            except Exception as fe:
                _log.warning("in_process bg_flush fail_run failed for %s run %s: %s", matter_id, flush_run_id, fe)
            raise
        else:
            try:
                model.complete_run(flush_run_id)
            except Exception as ce:
                try:
                    model.fail_run(flush_run_id, str(ce))
                except Exception as fe:
                    _log.warning("in_process bg_flush terminal close failed for %s run %s: %s", matter_id, flush_run_id, fe)


def _in_process_background_flush_loop(matter_id: str, model) -> None:
    """Flush loop body — runs while _bg_flush_event is set, then releases _bg_flush_running."""
    try:
        while model._bg_flush_event.is_set():
            model._bg_flush_event.clear()
            try:
                _do_one_in_process_flush(matter_id, model)
            except Exception as exc:
                _log.warning("in_process background_flush failed for matter %s: %s", matter_id, exc)
                break
    finally:
        model._bg_flush_running.release()
        # Final race: work enqueued between last event check and release
        if model._bg_flush_event.is_set():
            if model._bg_flush_running.acquire(blocking=False):
                try:
                    threading.Thread(
                        target=_in_process_background_flush_loop,
                        args=(matter_id, model),
                        daemon=False,
                    ).start()
                except Exception:
                    model._bg_flush_running.release()


class InProcessBackend(UIBackend):
    """In-process backend for local dev without a running service."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._irys: Optional[Irys] = None

    def _get_irys(self) -> Irys:
        if self._irys is None:
            import os
            import tempfile
            from pathlib import Path as _Path
            # Default checkpoint_dir so stop→resume works out of the box.
            # Uses IRYS_CHECKPOINT_DIR env var if set, otherwise a persistent
            # tempdir path (survives for the session).
            ckpt = os.environ.get("IRYS_CHECKPOINT_DIR") or str(
                _Path(tempfile.gettempdir()) / "irys" / "checkpoints"
            )
            self._irys = Irys(
                config=IrysConfig(
                    api_key=self.api_key,
                    enable_matter_model=True,
                    checkpoint_dir=ckpt,
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
        research_mode: Optional[str] = None,
        conversation_history: Optional[list[dict[str, str]]] = None,
    ) -> dict:
        """Start an investigation in-process. Returns after investigation completes."""
        irys = self._get_irys()
        result = await irys.investigate(
            query,
            repo_path,
            research_mode=research_mode,
            conversation_history=conversation_history,
        )
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

    def _resolve_active_run_id(self, model: Any, run_id: str) -> str:
        """HIGH adv#035/r82: if run_id points to an interrupted run, find the actually
        running run that was started to resume it. This handles the UI steerability gap
        where current_run_id still points to the old interrupted run during a resumed
        investigation (do_resume() updates current_run_id only after completion).

        Uses the resumed_from lineage column (r84 HIGH) to prove the running run is
        the direct continuation of this interrupted run — not just "the only running
        run in the matter". Falls back to run_id if no lineage match is found.
        """
        run = model.ledger.get_run(run_id)
        if run is not None and run.status == "interrupted":
            row = model.db.execute(
                "SELECT id FROM run_session"
                " WHERE matter_id=? AND status='running' AND resumed_from=?",
                (model.matter_id, run_id),
            ).fetchone()
            if row is not None:
                return row["id"]
        return run_id

    async def stop_run(self, matter_id: str, run_id: str) -> dict:
        try:
            model = self._get_matter_model(matter_id)
            run_id = self._resolve_active_run_id(model, run_id)
            run = model.ledger.get_run(run_id)
            if run is not None and run.matter_id != model.matter_id:
                return {"status": "error", "detail": f"Run '{run_id}' does not belong to matter '{matter_id}'"}
            if run and run.objective in ("manual_flush", "background_flush"):
                return {"status": "error", "detail": f"Run '{run_id}' is a utility flush run and cannot be stopped"}
            applied = model.ledger.request_stop(run_id)
            if not applied:
                return {"status": "error", "detail": f"Run '{run_id}' completed before stop could be applied"}
            return {"status": "stop_requested", "run_id": run_id}
        except Exception as exc:
            return {"status": "error", "detail": str(exc)}

    async def resume_run(
        self,
        matter_id: str,
        run_id: str,
        follow_up_query: Optional[str] = None,
        research_mode: Optional[str] = None,
        conversation_history: Optional[list[dict[str, str]]] = None,
    ) -> dict:
        """Resume an interrupted run from its checkpoint (InProcessBackend)."""
        try:
            from pathlib import Path as _Path
            model = self._get_matter_model(matter_id)
            run = model.ledger.get_run(run_id)
            if run is None:
                return {"status": "error", "detail": f"Run '{run_id}' not found"}
            if run.status != "interrupted":
                return {"status": "error", "detail": f"Run '{run_id}' is not interrupted (status={run.status})"}
            if run.objective in ("manual_flush", "background_flush"):
                return {"status": "error", "detail": f"Run '{run_id}' is a utility flush run and cannot be resumed"}
            checkpoint_path = run.next_action
            if not checkpoint_path:
                return {"status": "error", "detail": f"Run '{run_id}' has no checkpoint — was stopped before first checkpoint interval"}
            if not _Path(checkpoint_path).exists():
                return {"status": "error", "detail": f"Checkpoint file not found: {checkpoint_path}"}
            # Guard against concurrent-resume double-click: mirror the service gate —
            # indexed fetchone instead of materializing N rows. (MEDIUM r69, LOW r70)
            running = model.db.execute(
                "SELECT id FROM run_session WHERE matter_id=? AND status='running'"
                " AND id != ?"
                " AND (objective IS NULL OR objective NOT IN ('manual_flush','background_flush'))",
                (model.matter_id, run_id),
            ).fetchone()
            if running:
                return {"status": "error", "detail": "Another run is already active for this matter — wait for it to complete or stop it before resuming"}
            irys = self._get_irys()
            # Wire the same matter model so ledger entries go to the correct DB
            irys._ensure_initialized()
            irys._engine._matter_model = model
            result = await irys.resume_investigation(
                checkpoint_path,
                original_run_id=run_id,
                follow_up_query=follow_up_query,
                research_mode=research_mode,
                conversation_history=conversation_history,
            )
            new_run_id = getattr(result.state, "_run_id", None)
            return {"status": "resumed", "run_id": run_id, "new_run_id": new_run_id}
        except Exception as exc:
            return {"status": "error", "detail": str(exc)}

    # ------------------------------------------------------------------ #
    # Overview / dashboard                                                  #
    # ------------------------------------------------------------------ #

    async def get_overview(self, matter_id: str) -> dict:
        model = self._get_matter_model(matter_id)
        import sqlite3 as _sqlite3
        stats = model.stats()
        # Fetch coverage once — reused by both get_so_metrics() and weakest_issues.
        # DB errors are survivable (overview panel renders with
        # whatever it can); every other exception is a logic bug
        # and should propagate.
        coverage_report: list = []
        try:
            coverage_report = model.get_issue_coverage_report()
        except _sqlite3.Error as _exc:
            _log.warning("overview: coverage_report load failed: %s", _exc)
        so: dict = {}
        try:
            so = model.get_so_metrics(_coverage_report=coverage_report)
        except _sqlite3.Error as _exc:
            _log.warning("overview: so_metrics load failed: %s", _exc)
        weakest: list = (
            heapq.nsmallest(5, coverage_report, key=lambda r: float(r.get("coverage_fraction", 0.0)))
            if coverage_report else []
        )
        top_gaps: list = []
        try:
            top_gaps = model.gaps.open_gaps(limit=5)
        except _sqlite3.Error as _exc:
            _log.warning("overview: top gaps load failed: %s", _exc)
        clarifications: list = []
        try:
            clarifications = model.clarifications.get_pending(limit=5)
        except _sqlite3.Error as _exc:
            _log.warning("overview: pending clarifications load failed: %s", _exc)
            pass
        domain_composition: dict = {}
        try:
            facets, tw, primary = model._read_matter_domain_composition()
            domain_composition = {
                "facets": facets,
                "composed_trust_weights": tw,
                "primary_domain_profile_id": primary,
            }
        except _sqlite3.Error as _exc:
            _log.warning("overview: domain_composition load failed: %s", _exc)
        contradiction_count = 0
        try:
            contradiction_count = len(model.assertions.find_contradictions())
        except _sqlite3.Error as _exc:
            _log.warning("overview: contradiction count load failed: %s", _exc)
        version_chain_count = 0
        try:
            version_chain_count = len(model.list_version_families())
        except _sqlite3.Error as _exc:
            _log.warning("overview: version chain count load failed: %s", _exc)
        return {
            "matter_id": matter_id,
            "stats": stats,
            "so_metrics": so,
            "coverage_report": coverage_report,
            "weakest_issues": weakest,
            "top_gaps": top_gaps,
            "pending_clarifications": clarifications,
            "domain_composition": domain_composition,
            "contradiction_count": contradiction_count,
            "version_chain_count": version_chain_count,
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
            except Exception as _exc:
                _log.warning("stream_run_events: event fetch failed: %s", _exc)

            # Check terminal status
            try:
                run_row = model.db.execute(
                    "SELECT status FROM run_session WHERE id=?", (run_id,)
                ).fetchone()
                if run_row and run_row["status"] in terminal_statuses:
                    yield {"event": "run_terminal", "status": run_row["status"]}
                    break
            except Exception as _exc:
                _log.warning("stream_run_events: terminal status check failed: %s", _exc)

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
        """Returns recent assertions with verification metadata joined
        per-row. UI surfaces render trust pills from this payload."""
        model = self._get_matter_model(matter_id)
        assertions = model.assertions.list_recent(limit=limit, offset=offset)
        ids = [a.get("id") for a in assertions if a.get("id")]
        if not ids:
            return assertions
        placeholders = ",".join("?" * len(ids))
        rows = model.db.execute(
            f"""SELECT target_id, status, reviewed_by_kind, reviewed_at,
                       rejection_reason
                FROM verification_state
                WHERE matter_id=? AND target_kind='assertion'
                  AND target_id IN ({placeholders})""",
            [model.matter_id, *ids],
        ).fetchall()
        by_id = {r["target_id"]: dict(r) for r in rows}
        for a in assertions:
            vs = by_id.get(a.get("id"))
            if vs is None:
                a["verification_status"] = "candidate"
                a["reviewed_by_kind"] = None
                a["reviewed_at"] = None
                a["rejection_reason"] = None
            else:
                a["verification_status"] = vs.get("status")
                a["reviewed_by_kind"] = vs.get("reviewed_by_kind")
                a["reviewed_at"] = vs.get("reviewed_at")
                a["rejection_reason"] = vs.get("rejection_reason")
        return assertions

    async def get_issue_assertions(self, matter_id: str, issue_id: str) -> list[dict]:
        model = self._get_matter_model(matter_id)
        return model.assertions.get_assertions_for_issue(issue_id)

    async def get_source_agreement(self, matter_id: str, issue_id: str) -> list[dict]:
        model = self._get_matter_model(matter_id)
        return model.get_source_agreement_for_issue(issue_id)

    async def get_assertion_graph(self, matter_id: str, issue_id: str) -> dict:
        model = self._get_matter_model(matter_id)
        return model.get_assertion_graph_for_issue(issue_id)

    async def get_issue_closure_workbench(self, matter_id: str, issue_id: str) -> dict:
        model = self._get_matter_model(matter_id)
        return model.get_issue_closure_workbench(issue_id)

    async def get_issue_authorities(self, matter_id: str, issue_id: str) -> list[dict]:
        model = self._get_matter_model(matter_id)
        return model.authority.list_for_issue(issue_id)

    async def list_gaps(self, matter_id: str, limit: int = 50) -> list[dict]:
        model = self._get_matter_model(matter_id)
        return model.gaps.open_gaps(limit=limit)

    async def get_gap_workbench(self, matter_id: str, limit: int = 50, min_materiality: float = 0.0) -> dict:
        model = self._get_matter_model(matter_id)
        return model.get_gap_workbench(limit=limit, min_materiality=min_materiality)

    async def get_investigation_readiness(self, matter_id: str) -> dict:
        model = self._get_matter_model(matter_id)
        return model.get_investigation_readiness()

    async def get_assertion_trace(self, matter_id: str, assertion_id: str) -> dict:
        model = self._get_matter_model(matter_id)
        return model.get_assertion_trace(assertion_id)

    async def resolve_gap(self, matter_id: str, gap_id: str, resolution_note: str = "") -> bool:
        model = self._get_matter_model(matter_id)
        return model.gaps.resolve_gap(gap_id, resolution_note)

    async def escalate_gap(self, matter_id: str, gap_id: str) -> bool:
        model = self._get_matter_model(matter_id)
        return model.escalate_gap(gap_id)

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
        run_id: "str | None" = None,
    ) -> dict:
        model = self._get_matter_model(matter_id)
        try:
            belief_state = BeliefState(new_state)
        except ValueError:
            return {"status": "error", "detail": f"Invalid belief state: {new_state!r}"}
        try:
            result = model.correct_assertion(assertion_id, belief_state, run_id=run_id, note=reason)
        except Exception as exc:
            return {"status": "error", "detail": str(exc)}
        resp = {"status": "corrected", "assertion_id": assertion_id}
        if getattr(result, "propagation_truncated", False):
            # Signal that a flush is needed. Only spawn a thread if no loop is active.
            # daemon=False: non-daemon so shutdown waits for the flush to finish.
            model._bg_flush_event.set()
            if model._bg_flush_running.acquire(blocking=False):
                try:
                    threading.Thread(
                        target=_in_process_background_flush_loop,
                        args=(matter_id, model),
                        daemon=False,
                    ).start()
                    resp["warning"] = "Belief revision truncated — background flush scheduled"
                except Exception as te:
                    model._bg_flush_running.release()
                    _log.warning("could not start background flush thread for %s: %s", matter_id, te)
                    resp["warning"] = "Belief revision truncated — proof state will refresh on next run"
            else:
                # Loop already running; it will see _bg_flush_event and do another pass.
                resp["warning"] = "Belief revision truncated — background flush scheduled"
        return resp

    async def redirect_run(
        self, matter_id: str, run_id: str, issue_id: str
    ) -> dict:
        model = self._get_matter_model(matter_id)
        try:
            run_id = self._resolve_active_run_id(model, run_id)
            run = model.ledger.get_run(run_id)
            if run is None:
                return {"status": "error", "detail": f"Run '{run_id}' not found"}
            if run.matter_id != model.matter_id:
                return {"status": "error", "detail": f"Run '{run_id}' does not belong to matter '{matter_id}'"}
            if run.status not in ("running", "interrupted"):
                return {"status": "error", "detail": f"Run is not active or interrupted (status: {run.status})"}
            if run.objective in ("manual_flush", "background_flush"):
                return {"status": "error", "detail": f"Run '{run_id}' is a utility flush run and cannot be redirected"}
            issue = model.issues.get_issue(issue_id)
            if issue is None:
                return {"status": "error", "detail": f"Issue '{issue_id}' not found"}
            applied = model.ledger.request_redirect(run_id, issue_id)
            if not applied:
                return {"status": "error", "detail": f"Run '{run_id}' completed before redirect could be applied"}
            # Log the steering event so the reasoning trail reflects the user action (SO-3).
            # Service API does the same at service/api.py:1606.
            from irys.matter.enums import LedgerEventType
            model.ledger.append_event(
                run_id=run_id,
                event_type=LedgerEventType.USER_REDIRECTED,
                summary=f"User redirected to issue: {issue.get('title', issue_id)[:80]}",
                why="User-initiated redirect via local UI",
                branch_issue_id=issue_id,
            )
            return {
                "status": "redirect_requested",
                "run_id": run_id,
                "issue_id": issue_id,
                "issue_title": issue.get("title"),
                # SO-3: redirect is consumed at the next iteration boundary inside the engine
                # loop. If the engine is mid-synthesis or mid-verification, the redirect will
                # take effect at the start of the following investigation cycle, not immediately.
                # Stop is similarly best-effort: in-flight LLM SDK calls cannot be cancelled
                # mid-request; the engine will stop at the next loop check point.
                "note": "Redirect takes effect at the next iteration boundary. May be delayed if the engine is currently in synthesis or verification phase.",
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
            "invoice_reconciliation": model.reconcile_invoice_chain(),
            "amount_conflicts": model.get_amount_conflicts(),
            "damages_waterfall": model.get_damages_waterfall(),
        }

    async def list_assumptions(self, matter_id: str, limit: int = 30) -> list[dict]:
        model = self._get_matter_model(matter_id)
        assumptions = model.assumptions.get_all(max_rows=limit)
        for a in assumptions:
            targets = model.assumptions.get_linked_targets(a["id"])
            a["linked_target_count"] = len(targets)
            a["linked_targets"] = targets[:5]
        return assumptions

    async def update_assumption_status(self, matter_id: str, assumption_id: str, status: str, reason: str = "") -> bool:
        model = self._get_matter_model(matter_id)
        return model.assumptions.set_status(assumption_id, status, reason or None)

    async def get_quant_facts(self, matter_id: str, limit: int = 200) -> dict:
        model = self._get_matter_model(matter_id)
        return model.get_quant_fact_workbench(limit=limit)

    async def get_decision_leverage(self, matter_id: str, top_n: int = 15) -> dict:
        model = self._get_matter_model(matter_id)
        return model.get_decision_leverage_map(top_n=top_n)

    async def get_output_quality(self, matter_id: str, run_id: str | None = None) -> dict:
        model = self._get_matter_model(matter_id)
        return model.get_output_quality_workbench(run_id=run_id)

    async def get_deliverable_workbench(self, matter_id: str) -> dict:
        model = self._get_matter_model(matter_id)
        return model.get_deliverable_workbench()

    async def get_scenario_workbench(self, matter_id: str) -> dict:
        model = self._get_matter_model(matter_id)
        return model.get_scenario_workbench()

    async def create_scenario_branch(self, matter_id: str, payload: dict) -> dict:
        model = self._get_matter_model(matter_id)
        return model.create_scenario_branch(
            name=payload.get("name", "Untitled"),
            assumptions=payload.get("assumptions", []),
            objective_ids=payload.get("objective_ids"),
            source_branch_id=payload.get("source_branch_id"),
            notes=payload.get("notes", ""),
        )

    async def archive_scenario_branch(self, matter_id: str, branch_id: str) -> dict:
        model = self._get_matter_model(matter_id)
        found = model.archive_scenario_branch(branch_id)
        if not found:
            return {"error": "Scenario branch not found"}
        return {"status": "archived", "branch_id": branch_id}

    async def apply_scenario_delta(self, matter_id: str, branch_id: str,
                                   target_kind: str, target_id: str,
                                   operation: str, payload: dict) -> dict:
        model = self._get_matter_model(matter_id)
        return model.apply_scenario_delta(branch_id, target_kind, target_id, operation, payload)

    async def list_scenario_deltas(self, matter_id: str, branch_id: str) -> list[dict]:
        model = self._get_matter_model(matter_id)
        return model.list_scenario_deltas(branch_id)

    async def compute_scenario_snapshot(self, matter_id: str, branch_id: str) -> dict:
        model = self._get_matter_model(matter_id)
        return model.compute_scenario_snapshot(branch_id)

    async def compare_scenario_to_baseline(self, matter_id: str, branch_id: str) -> dict:
        model = self._get_matter_model(matter_id)
        return model.compare_scenario_to_baseline(branch_id)

    async def list_scenario_snapshots(
        self, matter_id: str, branch_id: str, limit: int = 10,
    ) -> dict:
        model = self._get_matter_model(matter_id)
        snapshots = model.list_scenario_snapshots(branch_id, limit=limit)
        return {"branch_id": branch_id, "snapshots": snapshots, "count": len(snapshots)}

    async def get_alternative_theory_portfolio(self, matter_id: str, objective_id: str | None = None) -> dict:
        model = self._get_matter_model(matter_id)
        return model.get_alternative_theory_portfolio(objective_id=objective_id)

    async def get_dependency_manifest_inspector(self, matter_id: str) -> dict:
        model = self._get_matter_model(matter_id)
        return model.get_dependency_manifest_inspector()

    async def get_steering_impact_preview(self, matter_id: str, action_type: str, payload: dict) -> dict:
        model = self._get_matter_model(matter_id)
        return model.get_steering_impact_preview(action_type=action_type, payload=payload)

    async def get_domain_investigation_readiness(self, matter_id: str) -> dict:
        model = self._get_matter_model(matter_id)
        return model.evaluate_domain_investigation_readiness()

    async def compile_issue_brief(self, matter_id: str) -> dict:
        model = self._get_matter_model(matter_id)
        return model.compile_issue_brief()

    async def get_objective_coverage(self, matter_id: str) -> dict:
        model = self._get_matter_model(matter_id)
        return model.get_objective_coverage_workbench()

    async def set_criterion_status(
        self, matter_id: str, predicate_id: str, status: str, reason: str = "",
    ) -> dict:
        model = self._get_matter_model(matter_id)
        valid = ("open", "resolved", "contested", "blocked")
        if status not in valid:
            return {"error": f"Invalid status '{status}'. Must be one of: {', '.join(valid)}"}
        updated = model.issues.set_predicate_status(predicate_id, status, reason or None)
        if not updated:
            return {"error": f"Criterion '{predicate_id}' not found"}
        return {"updated": True, "predicate_id": predicate_id, "status": status}

    async def add_criterion(
        self, matter_id: str, objective_id: str, description: str, burden_side: str = "",
    ) -> dict:
        model = self._get_matter_model(matter_id)
        pid = model.issues.add_predicate(objective_id, description, burden_side or None)
        return {"predicate_id": pid, "objective_id": objective_id}

    async def get_assumption_review(self, matter_id: str) -> dict:
        model = self._get_matter_model(matter_id)
        return model.get_assumption_review_workbench()

    async def review_assumption(
        self, matter_id: str, assumption_id: str, decision: str, reason: str = "",
    ) -> dict:
        model = self._get_matter_model(matter_id)
        return model.review_assumption(
            assumption_id=assumption_id, decision=decision, reason=reason or None,
        )

    _PRIORITY_MAP = {"critical": 0.95, "high": 0.8, "medium": 0.5, "low": 0.2}

    async def set_issue_priority(self, matter_id: str, issue_id: str, priority: str) -> bool:
        model = self._get_matter_model(matter_id)
        materiality = self._PRIORITY_MAP.get(priority.lower(), 0.5)
        return model.issues.set_materiality(issue_id, materiality)

    async def get_timeline(
        self,
        matter_id: str,
        limit: int = 80,
        policy_audience: str = "clean",
    ) -> list[dict]:
        """Adversarial #9 fix: default to clean audience so privileged
        events render as "[withheld]" in the UI. Internal-audience
        callers (e.g. attorney reviewing their own workspace) must
        opt in explicitly."""
        model = self._get_matter_model(matter_id)
        return model.get_timeline(limit=limit, policy_audience=policy_audience)

    async def get_document_console(self, matter_id: str, document_ref: str) -> dict:
        model = self._get_matter_model(matter_id)
        return model.get_document_console(document_ref)

    async def get_quant_ontology(self, matter_id: str) -> dict:
        model = self._get_matter_model(matter_id)
        return model.get_quant_ontology_workbench()

    async def approve_metric_alias(
        self, matter_id: str, raw_label: str, canonical_metric: str, unit: str | None = None,
    ) -> bool:
        model = self._get_matter_model(matter_id)
        return model.approve_metric_alias(
            raw_label=raw_label,
            canonical_metric=canonical_metric,
            unit=unit,
        )

    async def get_answer_audits(
        self, matter_id: str, manifest_hash: str | None = None,
    ) -> dict:
        model = self._get_matter_model(matter_id)
        return model.get_answer_audit_workbench(manifest_hash=manifest_hash)

    async def resolve_contradiction(
        self, matter_id: str, attacker_id: str, attacked_id: str,
        decision: str, rationale: str = "",
    ) -> dict:
        model = self._get_matter_model(matter_id)
        return model.resolve_contradiction(
            attacker_id=attacker_id, attacked_id=attacked_id,
            decision=decision, rationale=rationale,
        )

    async def get_knowledge_seeds(self, matter_id: str) -> dict:
        model = self._get_matter_model(matter_id)
        return model.get_knowledge_seed_workbench()

    async def review_knowledge_seed(
        self, matter_id: str, seed_id: str, decision: str,
        review_note: str = "",
    ) -> dict:
        model = self._get_matter_model(matter_id)
        return model.review_knowledge_seed(
            seed_id=seed_id, decision=decision, review_note=review_note or None,
        )

    async def promote_knowledge_seed(
        self, matter_id: str, seed_kind: str, domain_profile_id: str,
        payload_json: str, source_matter_id: str | None = None,
    ) -> dict:
        model = self._get_matter_model(matter_id)
        return model.promote_knowledge_seed(
            seed_kind=seed_kind, domain_profile_id=domain_profile_id,
            payload_json=payload_json, source_matter_id=source_matter_id,
        )

    async def list_reviewable_documents(self, matter_id: str) -> list[dict]:
        """Document picker feed for the bulk-verify dropdown — every
        doc in the matter with pending/verified counts so reviewers
        can triage at a glance."""
        model = self._get_matter_model(matter_id)
        return model.list_reviewable_documents()

    async def get_evidence_matrix(
        self,
        matter_id: str,
        policy_audience: str = "clean",
    ) -> dict:
        """Adversarial #9 fix: default to clean audience so
        privileged source columns collapse to "[withheld]"."""
        model = self._get_matter_model(matter_id)
        return model.get_evidence_matrix(policy_audience=policy_audience)

    async def get_communication_map(self, matter_id: str) -> dict:
        model = self._get_matter_model(matter_id)
        return model.get_communication_map()

    async def get_authority_network(self, matter_id: str) -> dict:
        model = self._get_matter_model(matter_id)
        issue_titles = {
            i["id"]: i.get("title", i["id"])
            for i in model.issues.get_open_issues()
        }
        return model.authority.get_network(issue_titles=issue_titles)

    async def upsert_authority(
        self,
        matter_id: str,
        citation: str,
        *,
        authority_type: str = "case",
        name: str | None = None,
        jurisdiction: str | None = None,
        weight: str = "persuasive",
    ) -> dict:
        model = self._get_matter_model(matter_id)
        aid, is_new = model.authority.upsert(
            citation=citation,
            authority_type=authority_type,
            name=name,
            jurisdiction=jurisdiction,
            weight=weight,
        )
        return {"authority_id": aid, "is_new": is_new}

    async def link_authority_to_issue(
        self,
        matter_id: str,
        authority_id: str,
        issue_id: str,
        relevance: str = "supporting",
    ) -> dict:
        model = self._get_matter_model(matter_id)
        model.authority.link_to_issue(authority_id, issue_id, relevance=relevance)
        return {"status": "linked"}

    async def unlink_authority_from_issue(
        self, matter_id: str, authority_id: str, issue_id: str
    ) -> dict:
        model = self._get_matter_model(matter_id)
        model.authority.unlink_from_issue(authority_id, issue_id)
        return {"status": "unlinked"}

    async def search_authorities(
        self, matter_id: str, query: str, limit: int = 20
    ) -> list[dict]:
        model = self._get_matter_model(matter_id)
        return model.authority.search(query, limit=limit)

    async def get_document_intelligence(self, matter_id: str) -> dict:
        model = self._get_matter_model(matter_id)
        cards = model.document_cards.list_candidates(limit=200)
        total_docs = model.inventory.count()
        ingested_paths = model.inventory.get_ingested_paths()
        return {
            "cards": cards,
            "total_inventory": total_docs,
            "ingested_count": len(ingested_paths),
        }

    async def get_proof_state_summary(self, matter_id: str) -> dict:
        model = self._get_matter_model(matter_id)
        summary = model.proof_state.get_summary()
        all_states = model.proof_state.get_all()
        issue_titles = {
            i["id"]: i.get("title", i["id"])
            for i in model.issues.get_open_issues()
        }
        for ps in all_states:
            ps["issue_title"] = issue_titles.get(ps.get("issue_id", ""), ps.get("issue_id", "?"))
        return {"matter_id": matter_id, "summary": summary, "issues": all_states}

    async def list_belief_revisions(self, matter_id: str, limit: int = 100) -> list[dict]:
        model = self._get_matter_model(matter_id)
        return model.list_belief_revisions(limit=limit)

    async def get_contradictions(self, matter_id: str, limit: int = 100) -> list[dict]:
        model = self._get_matter_model(matter_id)
        return model.assertions.find_contradictions(limit=limit)

    async def get_document_versions(self, matter_id: str) -> list[dict]:
        model = self._get_matter_model(matter_id)
        return model.list_version_families()

    async def get_operative_document_version(self, matter_id: str, doc_id: str) -> dict:
        model = self._get_matter_model(matter_id)
        operative_id = model.get_operative_document_version(doc_id)
        return {
            "doc_id": doc_id,
            "operative_doc_id": operative_id,
            "is_operative": doc_id == operative_id,
        }

    async def mine_contradictions(self, matter_id: str) -> list[dict]:
        model = self._get_matter_model(matter_id)
        return model.mine_contradictions()

    async def refresh_document_families(self, matter_id: str) -> list[dict]:
        model = self._get_matter_model(matter_id)
        return model.refresh_document_families()

    async def get_assertion_health(self, matter_id: str, assertion_id: str) -> dict:
        model = self._get_matter_model(matter_id)
        return model.get_assertion_health(assertion_id)

    async def get_assertion_history(self, matter_id: str, assertion_id: str, limit: int = 20) -> dict:
        model = self._get_matter_model(matter_id)
        return model.list_assertion_history(assertion_id, limit=limit)

    async def list_content_policy_decisions(self, matter_id: str, limit: int = 50) -> list[dict]:
        model = self._get_matter_model(matter_id)
        return model.list_content_policy_decisions(limit=limit)

    async def get_quant_thresholds(
        self, matter_id: str, currency: str = "USD",
        *, exposure_high: float = 10_000.0, disputed_fraction_min: float = 0.10,
    ) -> list[dict]:
        model = self._get_matter_model(matter_id)
        return model.compute_quant_thresholds(
            currency=currency, exposure_high=exposure_high,
            disputed_fraction_min=disputed_fraction_min,
        )

    async def get_amount_conflicts(self, matter_id: str) -> list[dict]:
        model = self._get_matter_model(matter_id)
        return model.get_amount_conflicts()

    async def detect_quant_conflicts(self, matter_id: str) -> list[str]:
        model = self._get_matter_model(matter_id)
        return model.detect_quant_conflicts()

    async def get_reconciliation(self, matter_id: str, currency: str = "USD") -> dict:
        model = self._get_matter_model(matter_id)
        return model.reconcile_payment_chain(currency)

    async def get_invoice_chain(self, matter_id: str, currency: str = "USD") -> list:
        model = self._get_matter_model(matter_id)
        return model.reconcile_invoice_chain(currency)

    async def get_damages_waterfall(self, matter_id: str, currency: str = "USD") -> list[dict]:
        model = self._get_matter_model(matter_id)
        return model.get_damages_waterfall(currency=currency)

    async def get_system_health(self, matter_id: str) -> dict:
        model = self._get_matter_model(matter_id)
        return model.get_system_health()

    async def compute_proof_state(self, matter_id: str) -> dict:
        model = self._get_matter_model(matter_id)
        states = model.proof_state.compute_all()
        return {"matter_id": matter_id, "updated_count": len(states), "states": states}

    async def compute_issue_proof_state(self, matter_id: str, issue_id: str) -> dict:
        model = self._get_matter_model(matter_id)
        return model.proof_state.compute_and_store(issue_id)

    async def flush_pending(self, matter_id: str) -> dict:
        model = self._get_matter_model(matter_id)
        from ...matter.runtime import MatterRuntimeAdapter
        with model._flush_lock:
            flush_run_id = model.start_run(
                "UI flush", objective="manual_flush",
                operation_type="maintenance", trigger="ui",
            )
            try:
                adapter = MatterRuntimeAdapter(model, run_id=flush_run_id)
                revised = adapter._flush_revisions_locked()
            except Exception as exc:
                try:
                    model.fail_run(flush_run_id, str(exc))
                except Exception as fe:
                    _log.warning("flush_pending fail_run failed for %s run %s: %s", matter_id, flush_run_id, fe)
                raise
            try:
                model.complete_run(flush_run_id)
            except Exception as ce:
                try:
                    model.fail_run(flush_run_id, str(ce))
                except Exception as fe:
                    _log.warning("flush_pending terminal close failed for %s run %s: %s", matter_id, flush_run_id, fe)
                raise
        return {"status": "ok", "revised_count": revised}

    async def get_so_scorecard(self, matter_id: str) -> dict:
        model = self._get_matter_model(matter_id)
        return model.get_so_metrics()

    async def get_domain_profile_summary(
        self, matter_id: str, profile_id: str | None = None
    ) -> dict:
        model = self._get_matter_model(matter_id)
        return model.get_domain_profile_summary(profile_id=profile_id)

    async def get_domain_composition(self, matter_id: str) -> dict:
        model = self._get_matter_model(matter_id)
        facets, tw, primary = model._read_matter_domain_composition()
        detection_events = []
        try:
            rows = model.db.execute(
                """SELECT id, target_kind, target_id, candidate_profile_id,
                          confidence, signals_json, evidence_refs_json, created_at
                   FROM domain_detection_event
                   WHERE matter_id=?
                   ORDER BY created_at DESC LIMIT 50""",
                (model.matter_id,),
            ).fetchall()
            detection_events = [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("domain_detection_event query failed: %s", exc)
        return {
            "matter_id": model.matter_id,
            "primary_domain_profile_id": primary,
            "facets": facets,
            "composed_trust_weights": tw,
            "detection_events": detection_events,
        }

    async def list_documents_needing_profile(
        self, matter_id: str, limit: int = 50
    ) -> list[dict]:
        model = self._get_matter_model(matter_id)
        return model.list_documents_needing_profile(limit=limit)

    async def get_taint_summary(
        self, matter_id: str, limit: int = 50
    ) -> dict:
        model = self._get_matter_model(matter_id)
        return model.summarize_taint(limit=limit)

    async def answer_clarification(
        self, matter_id: str, question_id: str, answer_text: str
    ) -> bool:
        model = self._get_matter_model(matter_id)
        return model.answer_clarification(question_id, answer_text)

    async def list_trust_overrides(self, matter_id: str) -> list[dict]:
        model = self._get_matter_model(matter_id)
        return model.trust_overrides.list_all()

    async def set_trust_override(
        self, matter_id: str, document_pattern: str, trust_level: str, note: str = ""
    ) -> str:
        model = self._get_matter_model(matter_id)
        return model.set_trust_override(document_pattern, trust_level, note=note or None)

    async def delete_trust_override(self, matter_id: str, document_pattern: str) -> bool:
        model = self._get_matter_model(matter_id)
        return model.delete_trust_override(document_pattern)

    async def generate_clarifications(self, matter_id: str, top_n: int = 3) -> list[str]:
        model = self._get_matter_model(matter_id)
        return model.generate_clarifications_from_gaps(top_n=top_n)

    async def search_assertions(self, matter_id: str, query: str, limit: int = 20) -> list[dict]:
        model = self._get_matter_model(matter_id)
        return model.search_assertions([query], limit=limit)

    async def find_duplicate_actors(self, matter_id: str, min_prefix_len: int = 6) -> list[dict]:
        model = self._get_matter_model(matter_id)
        return model.actors.find_possible_duplicates(min_prefix_len=min_prefix_len)

    async def merge_actors(self, matter_id: str, keep_id: str, merge_id: str) -> dict:
        model = self._get_matter_model(matter_id)
        model.actors.merge_actors(keep_id=keep_id, merge_id=merge_id)
        return {"keep_id": keep_id, "merged_id": merge_id, "status": "merged"}

    async def resolve_actor(self, matter_id: str, name: str) -> dict:
        model = self._get_matter_model(matter_id)
        actor_id = model.actors.resolve_by_name(name)
        if actor_id is None:
            return {"actor_id": None}
        actor = next(
            (a for a in model.actors.list_actors() if a["id"] == actor_id), None
        )
        return {"actor_id": actor_id, "actor": actor}

    async def get_decision_context(self, matter_id: str) -> "dict | None":
        model = self._get_matter_model(matter_id)
        return model.decision_context.get()

    async def set_decision_context(
        self, matter_id: str,
        decision_maker_type: Optional[str] = None,
        decision_maker_name: Optional[str] = None,
        objective: Optional[str] = None,
        strategic_notes: Optional[str] = None,
        scope_narrow: bool = False,
    ) -> str:
        model = self._get_matter_model(matter_id)
        return model.decision_context.set(
            decision_maker_type=decision_maker_type,
            decision_maker_name=decision_maker_name,
            objective=objective,
            strategic_notes=strategic_notes,
            scope_narrow=scope_narrow,
        )

    async def clear_decision_context(self, matter_id: str) -> bool:
        model = self._get_matter_model(matter_id)
        model.decision_context.clear()
        return True

    async def list_annotations(self, matter_id: str, document: Optional[str] = None) -> list[dict]:
        model = self._get_matter_model(matter_id)
        if document:
            return model.annotations.get_for_document(document)
        return model.annotations.list_recent()

    async def add_annotation(
        self, matter_id: str, document_pattern: str,
        annotation_text: str, annotation_type: str = "strategic",
    ) -> str:
        model = self._get_matter_model(matter_id)
        return model.annotations.add(document_pattern, annotation_text, annotation_type)

    async def delete_annotation(self, matter_id: str, annotation_id: str) -> bool:
        model = self._get_matter_model(matter_id)
        return model.annotations.delete(annotation_id)

    async def export_matter_summary(self, matter_id: str) -> dict:
        model = self._get_matter_model(matter_id)
        return model.export_matter_summary()

    async def list_llm_calls(
        self,
        matter_id: str,
        run_id: Optional[str] = None,
        limit: int = 120,
    ) -> list[dict]:
        model = self._get_matter_model(matter_id)
        return model.list_llm_calls(run_id=run_id, limit=limit)

    async def get_cost_breakdown(
        self,
        matter_id: str,
        run_id: Optional[str] = None,
    ) -> dict:
        model = self._get_matter_model(matter_id)
        return model.get_cost_breakdown(run_id=run_id)

    async def get_cost_anomalies(
        self,
        matter_id: str,
        limit: int = 10,
        run_id: Optional[str] = None,
    ) -> list[dict]:
        model = self._get_matter_model(matter_id)
        return model.get_cost_anomalies(limit=limit, run_id=run_id)

    # ------------------------------------------------------------------ #
    # P0.3 Review Queue — attorney review workflow (SO-3)                 #
    # ------------------------------------------------------------------ #

    async def get_review_queue(
        self, matter_id: str, limit: int = 50, offset: int = 0,
        target_kind: Optional[str] = None,
    ) -> list[dict]:
        model = self._get_matter_model(matter_id)
        return model.get_review_queue(
            limit=limit, offset=offset, target_kind=target_kind,
        )

    async def count_review_queue(self, matter_id: str) -> dict:
        model = self._get_matter_model(matter_id)
        return model.count_review_queue()

    async def verify_target(
        self, matter_id: str, target_kind: str, target_id: str,
        *, reviewed_by_kind: str = "user",
        reviewed_by_id: Optional[str] = None,
        review_note: Optional[str] = None,
        review_scope: str = "extraction_correct",
    ) -> str:
        model = self._get_matter_model(matter_id)
        return model.verify_target(
            target_kind, target_id,
            reviewed_by_kind=reviewed_by_kind,
            reviewed_by_id=reviewed_by_id,
            review_note=review_note,
            review_scope=review_scope,
        )

    async def reject_target(
        self, matter_id: str, target_kind: str, target_id: str,
        *, rejection_reason: str,
        reviewed_by_kind: str = "user",
        reviewed_by_id: Optional[str] = None,
        review_note: Optional[str] = None,
    ) -> str:
        model = self._get_matter_model(matter_id)
        return model.reject_target(
            target_kind, target_id,
            reviewed_by_kind=reviewed_by_kind,
            reviewed_by_id=reviewed_by_id,
            rejection_reason=rejection_reason,
            review_note=review_note,
        )

    async def bulk_verify_by_document(
        self, matter_id: str, document_ref: str,
        *, reviewed_by_kind: str = "user",
        reviewed_by_id: Optional[str] = None,
    ) -> list[str]:
        model = self._get_matter_model(matter_id)
        return model.bulk_verify_by_document(
            document_ref,
            reviewed_by_kind=reviewed_by_kind,
            reviewed_by_id=reviewed_by_id,
        )

    async def list_candidate_assertions_for_document(
        self, matter_id: str, document_ref: str,
    ) -> list[dict]:
        """Return candidate assertions + metadata for the
        review-before-verify flow. Attorney inspects the list,
        unchecks anything they don't want to approve, then submits
        the subset via bulk_verify_assertion_ids."""
        model = self._get_matter_model(matter_id)
        return model.list_candidate_assertions_for_document(document_ref)

    async def bulk_verify_by_span(
        self, matter_id: str, span_id: str,
        *, reviewed_by_kind: str = "user",
        reviewed_by_id: Optional[str] = None,
        review_note: Optional[str] = None,
        review_scope: str = "extraction_correct",
    ) -> list[str]:
        model = self._get_matter_model(matter_id)
        return model.bulk_verify_by_span(
            span_id,
            reviewed_by_kind=reviewed_by_kind,
            reviewed_by_id=reviewed_by_id,
            review_note=review_note,
            review_scope=review_scope,
        )

    async def bulk_verify_assertion_ids(
        self, matter_id: str, assertion_ids: list[str],
        *, reviewed_by_kind: str = "user",
        reviewed_by_id: Optional[str] = None,
        review_note: Optional[str] = None,
    ) -> list[str]:
        model = self._get_matter_model(matter_id)
        return model.bulk_verify_assertion_ids(
            assertion_ids,
            reviewed_by_kind=reviewed_by_kind,
            reviewed_by_id=reviewed_by_id,
            review_note=review_note,
        )

    async def reclassify_document_sensitivity(
        self, matter_id: str, doc_id: str, privilege_flag: bool,
        reviewed_by_kind: str = "user", reviewed_by_id: str | None = None,
    ) -> dict:
        model = self._get_matter_model(matter_id)
        staled = model.reclassify_privilege(
            doc_id, privilege_flag,
            reviewed_by_kind=reviewed_by_kind,
            reviewed_by_id=reviewed_by_id,
        )
        return {"doc_id": doc_id, "staled_count": staled}

    async def mark_document_stale(
        self, matter_id: str, doc_id: str, reason: str = "manual_stale",
    ) -> dict:
        model = self._get_matter_model(matter_id)
        staled = model.mark_document_stale(doc_id, reason)
        return {"doc_id": doc_id, "staled_count": staled}

    async def mark_span_stale(
        self, matter_id: str, span_id: str, reason: str = "manual_span_stale",
    ) -> dict:
        model = self._get_matter_model(matter_id)
        staled = model.mark_span_stale(span_id, reason)
        return {"span_id": span_id, "staled_count": staled}

    async def get_verification_events(
        self, matter_id: str, target_kind: Optional[str] = None,
        target_id: Optional[str] = None, limit: int = 50,
    ) -> list[dict]:
        model = self._get_matter_model(matter_id)
        return model.get_verification_events(
            target_kind=target_kind, target_id=target_id, limit=limit,
        )

    async def get_provenance(
        self, matter_id: str, target_kind: str, target_id: str,
        limit: int = 50,
    ) -> list[dict]:
        """P0.1: return the provenance_event trail for one target.

        Each row attributes the AI write: run, tier, prompt version,
        source document + span, llm_call_id — everything the attorney
        needs to answer "where did this come from?"."""
        model = self._get_matter_model(matter_id)
        rows = model.get_provenance(target_kind, target_id, limit=limit)
        return [{k: v for k, v in row.items() if k != "model_id"} for row in rows]

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
        research_mode: Optional[str] = None,
        conversation_history: Optional[list[dict[str, str]]] = None,
        resume_matter_id: Optional[str] = None,
        resume_run_id: Optional[str] = None,
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
                result = None
                if resume_matter_id and resume_run_id:
                    try:
                        model = self._get_matter_model(resume_matter_id)
                        run = model.ledger.get_run(resume_run_id)
                        checkpoint_path = getattr(run, "next_action", None) if run else None
                        if (
                            run is not None
                            and run.status == "interrupted"
                            and checkpoint_path
                        ):
                            irys._engine._matter_model = model
                            result = await irys.resume_investigation(
                                checkpoint_path,
                                original_run_id=resume_run_id,
                                follow_up_query=query,
                                research_mode=research_mode,
                                conversation_history=conversation_history,
                            )
                    except Exception as _exc:
                        _log.warning("resume_investigation failed, falling back to fresh investigate: %s", _exc)
                        result = None
                if result is None:
                    result = await irys.investigate(
                        query,
                        repo_path,
                        research_mode=research_mode,
                        conversation_history=conversation_history,
                    )
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
                    set_final_output(state.findings.get("final_output") or result.output)
                update_q.put(("complete", state))
            except Exception as exc:
                update_q.put(("error", str(exc)))

        asyncio.run(_inner())

    async def get_query_context(self, matter_id: str) -> dict:
        from dataclasses import asdict
        model = self._get_matter_model(matter_id)
        return asdict(model.build_query_context())

    async def get_source_calibration(self, matter_id: str) -> dict:
        model = self._get_matter_model(matter_id)
        return {
            "evidence_matrix": model.get_evidence_matrix(policy_audience="clean"),
            "coverage_report": model.get_issue_coverage_report(policy_audience="clean"),
            "domain_profile": model.get_domain_profile_summary(),
            "gap_workbench": model.get_gap_workbench(),
            "reviewable_documents": model.list_reviewable_documents(limit=200),
        }
