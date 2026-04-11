"""ReasoningLedgerStore — structured, user-facing reasoning trace.

The ledger is append-only ordered events that record what the system did
and why. It is NOT raw chain-of-thought — it is a structured log safe for
users to read, interrupt, and redirect.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from .db import SQLiteMatterDB
from .enums import RunStatus, LedgerEventType
from .models import RunSessionRecord


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id() -> str:
    return uuid.uuid4().hex


class ReasoningLedgerStore:
    """
    Manages run sessions and their ledger events.

    A run session corresponds to one call to RLMEngine.investigate().
    Ledger events are appended in real time during the investigation.
    """

    def __init__(self, db: SQLiteMatterDB, matter_id: str):
        self.db = db
        self.matter_id = matter_id
        # seq_no cache: avoids a SELECT per append_event call (initialized lazily
        # from DB on first write per run, incremented in-memory thereafter).
        self._seq_cache: dict[str, int] = {}

    def start_run(
        self,
        query: str,
        objective: Optional[str] = None,
        assertions_at_start: Optional[int] = None,
        resumed_from: Optional[str] = None,
        operation_type: str = "query",
        trigger: str = "user",
        research_mode: str = "deep",
    ) -> str:
        """Start a new run session. Returns run_id.

        ``assertions_at_start`` should be the assertion count snapshotted
        immediately before calling this method so that ``reuse_rate`` can be
        computed when the run completes.

        ``resumed_from`` is the interrupted run_id this run was started to
        resume, if any.  Stored for lineage-based run resolution (r84 HIGH).

        ``operation_type`` classifies the session: 'query', 'revise', 'ingest',
        'lint', or 'maintenance'.

        ``trigger`` indicates who/what started this: 'user', 'system', 'api'.
        """
        run_id = _id()
        now = _now()
        with self.db.transaction():
            self.db.execute(
                """INSERT INTO run_session
                   (id, matter_id, query, objective, status, started_at,
                    assertions_at_start, resumed_from, operation_type, trigger, research_mode)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id, self.matter_id, query, objective,
                    RunStatus.RUNNING.value, now,
                    assertions_at_start, resumed_from,
                    operation_type, trigger, research_mode,
                ),
            )
            # Seed the first ledger event
            self._append_event(
                run_id=run_id,
                event_type=LedgerEventType.RUN_STARTED,
                summary=f"Run started for query: {query[:120]}",
                why="User initiated investigation",
            )
        return run_id

    def append_event(
        self,
        run_id: str,
        event_type: LedgerEventType,
        summary: str,
        why: Optional[str] = None,
        branch_issue_id: Optional[str] = None,
        changed_object_type: Optional[str] = None,
        changed_object_id: Optional[str] = None,
        snapshot_json: Optional[str] = None,
    ) -> str:
        """Append a ledger event. Returns event_id."""
        with self.db.transaction():
            return self._append_event(
                run_id=run_id,
                event_type=event_type,
                summary=summary,
                why=why,
                branch_issue_id=branch_issue_id,
                changed_object_type=changed_object_type,
                changed_object_id=changed_object_id,
                snapshot_json=snapshot_json,
            )

    def _append_event(
        self,
        run_id: str,
        event_type: LedgerEventType,
        summary: str,
        why: Optional[str] = None,
        branch_issue_id: Optional[str] = None,
        changed_object_type: Optional[str] = None,
        changed_object_id: Optional[str] = None,
        snapshot_json: Optional[str] = None,
    ) -> str:
        """Internal: append event (caller must hold transaction)."""
        # Get next sequence number — initialize from DB once, then increment in-memory
        # to avoid a SELECT round-trip on every event append.
        if run_id not in self._seq_cache:
            row = self.db.execute(
                "SELECT COALESCE(MAX(seq_no), -1) + 1 FROM ledger_event WHERE run_id=?",
                (run_id,),
            ).fetchone()
            self._seq_cache[run_id] = row[0]
        seq_no = self._seq_cache[run_id]
        self._seq_cache[run_id] = seq_no + 1

        event_id = _id()
        now = _now()
        self.db.execute(
            """INSERT INTO ledger_event
               (id, run_id, seq_no, event_type, why, summary,
                branch_issue_id, changed_object_type, changed_object_id,
                snapshot_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                event_id, run_id, seq_no,
                event_type.value if hasattr(event_type, "value") else str(event_type),
                why, summary,
                branch_issue_id, changed_object_type, changed_object_id,
                snapshot_json, now,
            ),
        )
        return event_id

    def complete_run(
        self,
        run_id: str,
        summary: Optional[str] = None,
        reuse_rate: Optional[float] = None,
        llm_calls_avoided: Optional[int] = None,
        llm_calls_required: Optional[int] = None,
    ) -> None:
        """Mark a run session as completed.

        ``reuse_rate`` is the fraction of the final assertion count that
        pre-existed when the run started (assertions_at_start / assertions_at_end).
        ``llm_calls_avoided`` / ``llm_calls_required``: SO-1 real reuse telemetry
        (avoided + required = total LLM opportunities; true reuse = avoided/total).

        Clears next_action so completed runs are not shown as resumable.
        Also atomically clears stop_requested and redirect_requested so no
        stale steering flags can strand on a completed run (r91 TOCTOU fix).
        """
        now = _now()
        with self.db.transaction():
            self.db.execute(
                "UPDATE run_session"
                " SET status=?, completed_at=?, reuse_rate=?, next_action=NULL,"
                "     stop_requested=0, redirect_requested=0,"
                "     llm_calls_avoided=COALESCE(?, llm_calls_avoided),"
                "     llm_calls_required=COALESCE(?, llm_calls_required)"
                " WHERE id=? AND matter_id=?",
                (RunStatus.COMPLETED.value, now, reuse_rate,
                 llm_calls_avoided, llm_calls_required,
                 run_id, self.matter_id),
            )
            self._append_event(
                run_id=run_id,
                event_type=LedgerEventType.RUN_COMPLETED,
                summary=summary or "Run completed successfully",
            )
        self._seq_cache.pop(run_id, None)  # r96 MEDIUM: release dead cache entry

    def fail_run(self, run_id: str, reason: str) -> None:
        """Mark a run session as failed.

        Clears next_action (failed runs are not resumable) and atomically
        clears stop_requested/redirect_requested for consistency with
        complete_run() — stale steering flags must not survive terminal status
        (r94 LOW fix).
        """
        now = _now()
        with self.db.transaction():
            self.db.execute(
                "UPDATE run_session SET status=?, completed_at=?, next_action=NULL,"
                " stop_requested=0, redirect_requested=0"
                " WHERE id=? AND matter_id=?",
                (RunStatus.FAILED.value, now, run_id, self.matter_id),
            )
            self._append_event(
                run_id=run_id,
                event_type=LedgerEventType.RUN_FAILED,
                summary=f"Run failed: {reason[:200]}",
            )
        self._seq_cache.pop(run_id, None)  # r96 MEDIUM: release dead cache entry

    def interrupt_run(self, run_id: str) -> None:
        """Mark a run session as interrupted by user stop."""
        now = _now()
        with self.db.transaction():
            self.db.execute(
                "UPDATE run_session SET status=?, completed_at=? WHERE id=? AND matter_id=?",
                (RunStatus.INTERRUPTED.value, now, run_id, self.matter_id),
            )
            self._append_event(
                run_id=run_id,
                event_type=LedgerEventType.USER_INTERRUPTED,
                summary="Run interrupted by user stop request",
            )
        self._seq_cache.pop(run_id, None)  # r96 MEDIUM: release dead cache entry

    def set_next_action(self, run_id: str, next_action: str) -> None:
        """Store checkpoint path in run_session.next_action for SO-3 resume."""
        self.db.execute(
            "UPDATE run_session SET next_action=? WHERE id=? AND matter_id=?",
            (next_action, run_id, self.matter_id),
        )

    def clear_next_action(self, run_id: str) -> bool:
        """Clear next_action when a run completes/fails — only interrupted runs stay resumable.

        Returns True if the field was actually cleared (rowcount > 0), False if it was
        already NULL. Callers that use this as a CAS fence (resume_investigation) must
        check the return value: rowcount==0 means a concurrent call beat them to it.
        """
        # HIGH adv#035: also require status='interrupted' so direct engine callers
        # cannot claim a RUNNING run's checkpoint via resume. Running runs write
        # next_action at each checkpoint, so IS NOT NULL alone is insufficient.
        cur = self.db.execute(
            "UPDATE run_session SET next_action=NULL"
            " WHERE id=? AND matter_id=? AND next_action IS NOT NULL"
            " AND status='interrupted'",
            (run_id, self.matter_id),
        )
        return cur.rowcount > 0

    def request_stop(self, run_id: str) -> bool:
        """Set stop_requested flag — checked by the engine between iterations.

        Silently ignores utility flush runs (manual_flush / background_flush) so
        user stop actions cannot accidentally target a flush instead of a real
        investigation. Also requires status='running' to close the TOCTOU window
        where the API validates state before the UPDATE.

        Returns True if the flag was actually set (run was still running), False if
        the run had already completed/failed/was utility — allowing callers to detect
        the race.
        """
        cur = self.db.execute(
            "UPDATE run_session SET stop_requested=1 WHERE id=? AND status='running'"
            " AND (objective IS NULL OR objective NOT IN ('manual_flush','background_flush'))",
            (run_id,),
        )
        return cur.rowcount > 0

    def is_stop_requested(self, run_id: str) -> bool:
        row = self.db.execute(
            "SELECT stop_requested FROM run_session WHERE id=?", (run_id,)
        ).fetchone()
        return bool(row["stop_requested"]) if row else False

    def request_redirect(self, run_id: str, issue_id: str) -> bool:
        """Signal the engine to redirect focus to the given issue on the next iteration.

        Silently ignores utility flush runs (manual_flush / background_flush).
        Accepts 'running' and 'interrupted' status: an interrupted run can have a
        redirect focus set before resume, which is then propagated to the new run by
        resume_investigation(). For interrupted runs, also requires next_action IS NOT NULL
        (i.e. the run has not been resumed yet — resume clears next_action on the original
        run to prevent zombie steerability: post-resume redirects to the old run_id).

        Returns True if the flag was actually set, False if run not found or utility.
        """
        cur = self.db.execute(
            "UPDATE run_session SET redirect_requested=1, active_branch_issue_id=? WHERE id=?"
            " AND (status='running'"
            "      OR (status='interrupted' AND next_action IS NOT NULL))"
            " AND (objective IS NULL OR objective NOT IN ('manual_flush','background_flush'))",
            (issue_id, run_id),
        )
        return cur.rowcount > 0

    def is_redirect_requested(self, run_id: str) -> bool:
        row = self.db.execute(
            "SELECT redirect_requested FROM run_session WHERE id=?", (run_id,)
        ).fetchone()
        return bool(row["redirect_requested"]) if row else False

    def get_redirect_issue_id(self, run_id: str) -> Optional[str]:
        row = self.db.execute(
            "SELECT active_branch_issue_id FROM run_session WHERE id=?", (run_id,)
        ).fetchone()
        return row["active_branch_issue_id"] if row else None

    def clear_redirect(self, run_id: str) -> None:
        """Clear the redirect flag after the engine has processed it."""
        self.db.execute(
            "UPDATE run_session SET redirect_requested=0 WHERE id=?", (run_id,)
        )

    def get_events(self, run_id: str) -> list[dict]:
        """Fetch all ledger events for a run in sequence order."""
        rows = self.db.execute(
            "SELECT * FROM ledger_event WHERE run_id=? ORDER BY seq_no",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_run(self, run_id: str) -> Optional[RunSessionRecord]:
        row = self.db.execute(
            "SELECT * FROM run_session WHERE id=?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        return RunSessionRecord(
            id=d["id"],
            matter_id=d["matter_id"],
            query=d["query"],
            status=d["status"],
            started_at=d["started_at"],
            objective=d.get("objective"),
            active_branch_issue_id=d.get("active_branch_issue_id"),
            stop_requested=bool(d.get("stop_requested", 0)),
            redirect_requested=bool(d.get("redirect_requested", 0)),
            next_action=d.get("next_action"),
            completed_at=d.get("completed_at"),
            assertions_at_start=d.get("assertions_at_start"),
            reuse_rate=d.get("reuse_rate"),
            resumed_from=d.get("resumed_from"),
            research_mode=d.get("research_mode"),
            llm_input_tokens=d.get("llm_input_tokens"),
            llm_cache_read_tokens=d.get("llm_cache_read_tokens"),
            llm_output_tokens=d.get("llm_output_tokens"),
            llm_request_count=d.get("llm_request_count"),
            llm_estimated_cost_usd=d.get("llm_estimated_cost_usd"),
        )

    def recent_runs(self, limit: int = 10) -> list[dict]:
        rows = self.db.execute(
            """SELECT * FROM run_session WHERE matter_id=?
               ORDER BY started_at DESC LIMIT ?""",
            (self.matter_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]
