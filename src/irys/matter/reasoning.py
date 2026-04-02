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

    def start_run(self, query: str, objective: Optional[str] = None) -> str:
        """Start a new run session. Returns run_id."""
        run_id = _id()
        now = _now()
        with self.db.transaction():
            self.db.execute(
                """INSERT INTO run_session
                   (id, matter_id, query, objective, status, started_at)
                   VALUES (?,?,?,?,?,?)""",
                (run_id, self.matter_id, query, objective, RunStatus.RUNNING.value, now),
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
        # Get next sequence number
        row = self.db.execute(
            "SELECT COALESCE(MAX(seq_no), -1) + 1 FROM ledger_event WHERE run_id=?",
            (run_id,),
        ).fetchone()
        seq_no = row[0]

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

    def complete_run(self, run_id: str, summary: Optional[str] = None) -> None:
        """Mark a run session as completed."""
        now = _now()
        with self.db.transaction():
            self.db.execute(
                "UPDATE run_session SET status=?, completed_at=? WHERE id=?",
                (RunStatus.COMPLETED.value, now, run_id),
            )
            self._append_event(
                run_id=run_id,
                event_type=LedgerEventType.RUN_COMPLETED,
                summary=summary or "Run completed successfully",
            )

    def fail_run(self, run_id: str, reason: str) -> None:
        """Mark a run session as failed."""
        now = _now()
        with self.db.transaction():
            self.db.execute(
                "UPDATE run_session SET status=?, completed_at=? WHERE id=?",
                (RunStatus.FAILED.value, now, run_id),
            )
            self._append_event(
                run_id=run_id,
                event_type=LedgerEventType.RUN_FAILED,
                summary=f"Run failed: {reason[:200]}",
            )

    def interrupt_run(self, run_id: str) -> None:
        """Mark a run session as interrupted by user stop."""
        now = _now()
        with self.db.transaction():
            self.db.execute(
                "UPDATE run_session SET status=?, completed_at=? WHERE id=?",
                (RunStatus.INTERRUPTED.value, now, run_id),
            )
            self._append_event(
                run_id=run_id,
                event_type=LedgerEventType.USER_INTERRUPTED,
                summary="Run interrupted by user stop request",
            )

    def request_stop(self, run_id: str) -> None:
        """Set stop_requested flag — checked by the engine between iterations."""
        self.db.execute(
            "UPDATE run_session SET stop_requested=1 WHERE id=?", (run_id,)
        )

    def is_stop_requested(self, run_id: str) -> bool:
        row = self.db.execute(
            "SELECT stop_requested FROM run_session WHERE id=?", (run_id,)
        ).fetchone()
        return bool(row["stop_requested"]) if row else False

    def request_redirect(self, run_id: str, issue_id: str) -> None:
        """Signal the engine to redirect focus to the given issue on the next iteration."""
        self.db.execute(
            "UPDATE run_session SET redirect_requested=1, active_branch_issue_id=? WHERE id=?",
            (issue_id, run_id),
        )

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
        )

    def recent_runs(self, limit: int = 10) -> list[dict]:
        rows = self.db.execute(
            """SELECT * FROM run_session WHERE matter_id=?
               ORDER BY started_at DESC LIMIT ?""",
            (self.matter_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]
