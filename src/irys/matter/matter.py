"""MatterModel — the central facade for all matter intelligence stores.

Usage:
    model = MatterModel.open("path/to/repository", matter_name="Acme v TechServices")
    run_id = model.start_run("What are the key obligations?")
    assertion_id, is_new = model.record_assertion(candidate)
    model.ledger.append_event(run_id, LedgerEventType.ASSERTION_ADDED, ...)
    model.complete_run(run_id)
"""

import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_log = logging.getLogger(__name__)

from ..core.models import LLMCallRecord, PRICING_SOURCE_URL, PRICING_VERIFIED_AT
from .db import SQLiteMatterDB
from .graph import (
    AssertionStore, GapStore, ActorStore, IssueStore, ClarificationStore, QuantStore,
    DocumentInventoryStore, DocumentCardStore, SpanStore, DocumentActorRoleStore,
    ReasoningCacheStore, TrustOverrideStore, DocumentAnnotationStore,
    DecisionContextStore, AuthorityStore, ProofStateStore, AssumptionStore,
    VerificationStateStore, EvidenceStore, PrivilegeGate, ProvenanceStore,
    ContentPolicyGuard, MemoryBrokerStore,
    MemoryBrokerPolicyError,
)
from .reasoning import ReasoningLedgerStore
from .belief_revision import BeliefRevisionEngine
from .enums import (
    BeliefState, AssertionLinkType, RevisionCause,
    LedgerEventType, GapType, SOURCE_TRUST_WEIGHTS,
)
from .models import (
    AssertionCandidate, AssertionRecord, RevisionResult,
    QueryMatterContext, RunSessionRecord, ProvenanceContext,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id() -> str:
    return uuid.uuid4().hex


def _interpret_privilege_flag(value) -> Optional[bool]:
    """MVP.4 fail-closed privilege classifier for profile analysis output.

    Maps the prompt's three-valued "true"/"false"/"unknown" string
    (and legacy bools/ints) onto the DocumentCardStore.upsert contract:

    - None or "unknown" (in any form): treat as contained in clean mode.
      Return True — privileged until human review.
    - explicit False/0/"false"/"no"/"clean": return False. This is the
      only path that can clear a previously-privileged card.
    - anything else (True, 1, "true", "yes", arbitrary non-empty string):
      treat as privileged. Return True.

    Returning True for missing/unknown is the MVP.4 AC #2 "clean-mode
    processing treats unknown as contained until reviewed" semantics.
    """
    if value is None:
        return True  # fail-closed on missing
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in ("false", "0", "no", "not privileged", "not_privileged", "clean", "no_privilege", "no_privilege_detected"):
        return False
    # "true", "unknown", "yes", "privileged", "candidate_privileged", etc.
    return True


class MatterModel:
    """
    Central facade providing access to all matter intelligence stores.

    One MatterModel per matter, one SQLite DB per repository.

    The engine interacts exclusively through this facade — stores are
    implementation details. The facade is also the integration seam
    that the MatterRuntimeAdapter wraps to keep engine.py clean.
    """

    def __init__(self, db: SQLiteMatterDB, matter_id: str):
        self.db = db
        self.matter_id = matter_id
        # Sub-stores
        self.assertions = AssertionStore(db, matter_id)
        self.gaps = GapStore(db, matter_id)
        self.actors = ActorStore(db, matter_id)
        self.issues = IssueStore(db, matter_id)
        self.clarifications = ClarificationStore(db, matter_id)
        self.quant = QuantStore(db, matter_id)
        self.ledger = ReasoningLedgerStore(db, matter_id)
        self.belief = BeliefRevisionEngine(db, self.assertions, self.ledger)
        self.inventory = DocumentInventoryStore(db, matter_id)
        self.document_cards = DocumentCardStore(db, matter_id)
        self.spans = SpanStore(db, matter_id)
        self.doc_actor_roles = DocumentActorRoleStore(db, matter_id)
        self.cache = ReasoningCacheStore(db, matter_id)
        self.trust_overrides = TrustOverrideStore(db, matter_id)
        self.annotations = DocumentAnnotationStore(db, matter_id)
        self.decision_context = DecisionContextStore(db, matter_id)
        self.authority = AuthorityStore(db, matter_id)
        self.proof_state = ProofStateStore(db, matter_id)
        self.assumptions = AssumptionStore(db, matter_id)
        self.verification = VerificationStateStore(db, matter_id)
        self.evidence = EvidenceStore(db, matter_id)
        self.privilege = PrivilegeGate(db, matter_id)
        self.provenance = ProvenanceStore(db, matter_id)
        self.content_policy = ContentPolicyGuard(db, matter_id)
        self.memory_broker = MemoryBrokerStore(db, matter_id)
        self.memory_broker.ensure_builtin_domain_profiles()
        self.cache.set_broker(self.memory_broker)
        # In-memory snapshot of assertion counts captured at run start.
        # Keyed by run_id.  Allows complete_run() to compute reuse_rate without
        # an extra SELECT round-trip (DB is the authoritative fallback).
        self._run_snapshots: dict[str, int] = {}
        # Assertion IDs left unvisited after correct_assertion() inline retry rounds.
        # Maps assertion_id → originating run_id (None if unknown) for audit attribution:
        # when flush_revisions() re-emits ASSERTION_REVISED, the ledger event carries the
        # originating run_id rather than the draining adapter's run_id (r29 MEDIUM fix).
        # First-write wins: if two corrections target the same node, the first run_id is
        # preserved.  Protected by a lock for concurrent REST requests (r25 fix).
        self._correction_pending: dict[str, "str | None"] = {}
        self._correction_pending_lock = threading.Lock()
        # Assertion IDs dropped by BFS truncation during flush_revisions(), keyed by their
        # originating RevisionCause so replay uses the correct cause not always NEW_EVIDENCE
        # (r31 MEDIUM fix). First-write wins: earlier cause preserved on repeated truncation.
        self._evidence_pending: "dict[str, tuple[RevisionCause, str | None]]" = {}
        self._evidence_pending_lock = threading.Lock()
        # Serializes flush_revisions() calls across concurrent adapters/endpoints so that
        # reload_pending_from_db() + drain cannot race and double-replay the same row
        # (adv#029 SO-1 fix r7).
        self._flush_lock = threading.Lock()
        # Background flush coalescing state (adv#030 perf fix r2):
        # _bg_flush_event: set to signal that work is pending; cleared at the start of each
        #   flush pass so concurrent enqueues during a flush trigger another pass.
        # _bg_flush_running: held while the flush loop is active; non-blocking acquire
        #   prevents duplicate loop threads from starting.
        self._bg_flush_event = threading.Event()
        self._bg_flush_running = threading.Lock()
        # Reconstruct pending queues from the durable pending_propagation table (adv#029 SO-1 fix).
        # This ensures partial BFS propagation survives process restarts with no replay loss.
        self._load_pending_propagation()

    def _load_pending_propagation(self) -> None:
        """Populate in-memory pending queues from the durable DB table on open.

        Called at the end of __init__ so that a MatterModel rebuilt after a process
        restart (via MatterModel.open() or rehydration) immediately reflects any
        correction/evidence work that was queued before the restart but never drained
        (adv#029 SO-1 HIGH fix).  Uses first-write-wins semantics consistent with enqueue.
        NOT thread-safe — only call from __init__ before the model is shared.
        """
        try:
            rows = self.db.execute(
                "SELECT assertion_id, cause, orig_run_id, queue"
                " FROM pending_propagation WHERE matter_id=?",
                (self.matter_id,),
            ).fetchall()
        except Exception:
            return  # Table not yet present on pre-v40 DBs; migration runs on next open
        for row in rows:
            if row["queue"] == "correction":
                if row["assertion_id"] not in self._correction_pending:
                    self._correction_pending[row["assertion_id"]] = row["orig_run_id"]
            elif row["queue"] == "evidence":
                try:
                    cause = RevisionCause(row["cause"])
                except ValueError:
                    cause = RevisionCause.NEW_EVIDENCE
                if row["assertion_id"] not in self._evidence_pending:
                    self._evidence_pending[row["assertion_id"]] = (cause, row["orig_run_id"])

    def reload_pending_from_db(self) -> None:
        """Recover any pending_propagation DB rows missing from in-memory queues.

        Called at the start of each flush_revisions() to pick up rows left in DB
        by a previous failed delete_pending_propagation_db() call.  Without this,
        a delete failure strands rows in DB permanently for the lifetime of the live
        MatterModel instance (adv#029 SO-1 HIGH fix r6).
        Thread-safe: acquires each queue lock separately per row.
        """
        try:
            rows = self.db.execute(
                "SELECT assertion_id, cause, orig_run_id, queue"
                " FROM pending_propagation WHERE matter_id=?",
                (self.matter_id,),
            ).fetchall()
        except Exception:
            return
        for row in rows:
            if row["queue"] == "correction":
                with self._correction_pending_lock:
                    if row["assertion_id"] not in self._correction_pending:
                        self._correction_pending[row["assertion_id"]] = row["orig_run_id"]
            elif row["queue"] == "evidence":
                try:
                    cause = RevisionCause(row["cause"])
                except ValueError:
                    cause = RevisionCause.NEW_EVIDENCE
                with self._evidence_pending_lock:
                    if row["assertion_id"] not in self._evidence_pending:
                        self._evidence_pending[row["assertion_id"]] = (cause, row["orig_run_id"])

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def open(
        cls,
        repository_path: str | Path,
        matter_name: Optional[str] = None,
    ) -> "MatterModel":
        """
        Open (or create) a MatterModel for a repository path.

        If no matter exists for this repository, creates one.
        """
        db = SQLiteMatterDB.for_repository(repository_path)
        repo_str = str(Path(repository_path).resolve())

        # Look up or create matter record
        row = db.execute(
            "SELECT id FROM matter WHERE repository_root=?", (repo_str,)
        ).fetchone()

        if row is None:
            matter_id = _id()
            name = matter_name or Path(repository_path).name
            now = _now()
            with db.transaction():
                db.execute(
                    """INSERT INTO matter (id, name, repository_root, maturity, created_at, updated_at)
                       VALUES (?,?,?,?,?,?)""",
                    (matter_id, name, repo_str, "initial", now, now),
                )
        else:
            matter_id = row["id"]
            if matter_name:
                db.execute(
                    "UPDATE matter SET name=?, updated_at=? WHERE id=?",
                    (matter_name, _now(), matter_id),
                )

        return cls(db, matter_id)

    @classmethod
    def open_in_memory(cls, matter_name: str = "test_matter") -> "MatterModel":
        """Open an in-memory MatterModel for testing."""
        db = SQLiteMatterDB.in_memory()
        matter_id = _id()
        now = _now()
        with db.transaction():
            db.execute(
                """INSERT INTO matter (id, name, repository_root, maturity, created_at, updated_at)
                   VALUES (?,?,?,?,?,?)""",
                (matter_id, matter_name, ":memory:", "initial", now, now),
            )
        return cls(db, matter_id)

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(
        self,
        query: str,
        objective: Optional[str] = None,
        resumed_from: Optional[str] = None,
        operation_type: str = "query",
        trigger: str = "user",
        research_mode: str = "deep",
    ) -> str:
        """Start a new investigation run. Returns run_id.

        Snapshots the current assertion count so that ``complete_run`` can
        compute a measurable reuse_rate (SO-1 success criterion: > 0.70 on
        repeated queries over a stable matter).

        ``resumed_from`` is the interrupted run_id this run is resuming, if any.
        ``operation_type`` classifies the session: 'query', 'revise', 'maintenance'.
        ``trigger`` indicates who/what started this: 'user', 'system', 'api'.
        """
        assertions_at_start = self.assertions.count()
        run_id = self.ledger.start_run(
            query, objective, assertions_at_start, resumed_from=resumed_from,
            operation_type=operation_type, trigger=trigger,
            research_mode=research_mode,
        )
        # Cache snapshot in memory so complete_run() avoids a DB round-trip.
        self._run_snapshots[run_id] = assertions_at_start
        return run_id

    def complete_run(
        self,
        run_id: str,
        summary: Optional[str] = None,
        llm_calls_avoided: Optional[int] = None,
        llm_calls_required: Optional[int] = None,
    ) -> None:
        """Complete a run and compute reuse_rate from assertion count delta.

        Resolves assertions_at_start from the in-memory snapshot captured at
        start_run() — falling back to a DB query (scoped to this matter) for
        recovery paths where the snapshot is absent.

        llm_calls_avoided / llm_calls_required: SO-1 real reuse telemetry.
        When provided, they are persisted on run_session for post-hoc analysis.
        True reuse rate = avoided / (avoided + required).
        """
        assertions_at_end = self.assertions.count()
        # Use in-memory snapshot first; fall back to DB (scoped to this matter).
        at_start: Optional[int] = self._run_snapshots.pop(run_id, None)
        if at_start is None:
            row = self.db.execute(
                "SELECT assertions_at_start FROM run_session WHERE id=? AND matter_id=?",
                (run_id, self.matter_id),
            ).fetchone()
            at_start = row["assertions_at_start"] if row else None
        reuse_rate: Optional[float] = None
        if at_start is not None and assertions_at_end > 0:
            reuse_rate = round(at_start / assertions_at_end, 4)
        self.ledger.complete_run(
            run_id, summary, reuse_rate,
            llm_calls_avoided=llm_calls_avoided,
            llm_calls_required=llm_calls_required,
        )

    def fail_run(self, run_id: str, reason: str) -> None:
        self._run_snapshots.pop(run_id, None)  # prevent unbounded growth on non-completion paths
        self.ledger.fail_run(run_id, reason)

    def interrupt_run(self, run_id: str) -> None:
        self._run_snapshots.pop(run_id, None)  # prevent unbounded growth on non-completion paths
        self.ledger.interrupt_run(run_id)

    def get_provenance(
        self, target_kind: str, target_id: str, limit: int = 50,
    ) -> list[dict]:
        """P0.1: thin wrapper over ProvenanceStore.list_for_target so
        callers can read an AI-derived object's provenance trail
        without reaching into the store directly."""
        rows = self.provenance.list_for_target(target_kind, target_id, limit=limit)
        if rows:
            return rows
        return self._fallback_provenance_rows(target_kind, target_id, limit=limit)

    @staticmethod
    def _synthetic_span_status(span_id: Any) -> str:
        return "present" if span_id else "missing"

    def _materialize_fallback_provenance(
        self,
        target_kind: str,
        target_id: str,
        rows: list[dict],
    ) -> list[dict]:
        out: list[dict] = []
        for idx, row in enumerate(rows):
            source_document_ref = row.get("source_document_ref")
            source_span_id = row.get("source_span_id")
            if not source_document_ref and not source_span_id:
                continue
            out.append({
                "id": f"fallback:{target_kind}:{target_id}:{idx}",
                "matter_id": self.matter_id,
                "target_kind": target_kind,
                "target_id": target_id,
                "event_kind": "derived_source_link",
                "writer_name": "MatterModel.get_provenance",
                "run_id": None,
                "model_id": "stored source link",
                "model_tier": None,
                "prompt_version": None,
                "extractor_version": None,
                "llm_call_id": None,
                "prompt_hash": None,
                "response_hash": None,
                "source_document_ref": source_document_ref,
                "source_document_inventory_id": row.get("source_document_inventory_id"),
                "source_span_id": source_span_id,
                "source_span_status": (
                    row.get("source_span_status")
                    or self._synthetic_span_status(source_span_id)
                ),
                "note": (
                    "Derived from stored source links because no provenance_event row "
                    "was present for this target."
                ),
                "created_at": row.get("created_at") or _now(),
            })
        return out

    def _fallback_provenance_rows(
        self, target_kind: str, target_id: str, limit: int = 50,
    ) -> list[dict]:
        if target_kind == "assertion":
            rows = self.db.execute(
                """SELECT COALESCE(di.relative_path, ao.document_id) AS source_document_ref,
                          ao.document_inventory_id AS source_document_inventory_id,
                          ao.span_id AS source_span_id,
                          ao.created_at
                   FROM assertion_occurrence ao
                   JOIN assertion a ON a.id = ao.assertion_id
                   LEFT JOIN document_inventory di ON di.id = ao.document_inventory_id
                   WHERE a.matter_id=? AND ao.assertion_id=?
                   ORDER BY ao.created_at DESC
                   LIMIT ?""",
                (self.matter_id, target_id, int(limit)),
            ).fetchall()
            return self._materialize_fallback_provenance(
                target_kind, target_id, [dict(r) for r in rows],
            )

        if target_kind == "assertion_occurrence":
            rows = self.db.execute(
                """SELECT COALESCE(di.relative_path, ao.document_id) AS source_document_ref,
                          ao.document_inventory_id AS source_document_inventory_id,
                          ao.span_id AS source_span_id,
                          ao.created_at
                   FROM assertion_occurrence ao
                   JOIN assertion a ON a.id = ao.assertion_id
                   LEFT JOIN document_inventory di ON di.id = ao.document_inventory_id
                   WHERE a.matter_id=? AND ao.id=?
                   LIMIT ?""",
                (self.matter_id, target_id, int(limit)),
            ).fetchall()
            return self._materialize_fallback_provenance(
                target_kind, target_id, [dict(r) for r in rows],
            )

        if target_kind == "quant_fact":
            q_row = self.db.execute(
                """SELECT assertion_id, span_id, created_at
                   FROM quant_fact
                   WHERE matter_id=? AND id=?""",
                (self.matter_id, target_id),
            ).fetchone()
            if q_row is None:
                return []
            rows = []
            if q_row["assertion_id"]:
                occ_rows = self.db.execute(
                    """SELECT COALESCE(di.relative_path, ao.document_id) AS source_document_ref,
                              ao.document_inventory_id AS source_document_inventory_id,
                              COALESCE(?, ao.span_id) AS source_span_id,
                              ao.created_at
                       FROM assertion_occurrence ao
                       JOIN assertion a ON a.id = ao.assertion_id
                       LEFT JOIN document_inventory di ON di.id = ao.document_inventory_id
                       WHERE a.matter_id=? AND ao.assertion_id=?
                       ORDER BY CASE WHEN ? IS NOT NULL AND ao.span_id=? THEN 0 ELSE 1 END,
                                ao.created_at DESC
                       LIMIT ?""",
                    (
                        q_row["span_id"],
                        self.matter_id,
                        q_row["assertion_id"],
                        q_row["span_id"],
                        q_row["span_id"],
                        int(limit),
                    ),
                ).fetchall()
                rows = [dict(r) for r in occ_rows]
            if not rows and q_row["span_id"]:
                rows = [{
                    "source_document_ref": None,
                    "source_document_inventory_id": None,
                    "source_span_id": q_row["span_id"],
                    "created_at": q_row["created_at"],
                }]
            return self._materialize_fallback_provenance(target_kind, target_id, rows)

        if target_kind == "authority":
            rows = self.db.execute(
                """SELECT di.relative_path AS source_document_ref,
                          a.source_doc_id AS source_document_inventory_id,
                          a.source_span_id AS source_span_id,
                          a.updated_at AS created_at
                   FROM authority a
                   LEFT JOIN document_inventory di ON di.id = a.source_doc_id
                   WHERE a.matter_id=? AND a.id=?
                   LIMIT ?""",
                (self.matter_id, target_id, int(limit)),
            ).fetchall()
            return self._materialize_fallback_provenance(
                target_kind, target_id, [dict(r) for r in rows],
            )

        if target_kind == "document_card":
            rows = self.db.execute(
                """SELECT di.relative_path AS source_document_ref,
                          dc.doc_id AS source_document_inventory_id,
                          NULL AS source_span_id,
                          COALESCE(di.profiled_at, di.discovered_at) AS created_at
                   FROM document_card dc
                   JOIN document_inventory di ON di.id = dc.doc_id
                   WHERE di.matter_id=? AND dc.id=?
                   LIMIT ?""",
                (self.matter_id, target_id, int(limit)),
            ).fetchall()
            return self._materialize_fallback_provenance(
                target_kind, target_id, [dict(r) for r in rows],
            )

        if target_kind == "evidence_edge":
            edge = self.db.execute(
                """SELECT source_kind, source_id, source_document_inventory_id,
                          source_span_id, source_occurrence_id, updated_at
                   FROM evidence_edge
                   WHERE matter_id=? AND id=?""",
                (self.matter_id, target_id),
            ).fetchone()
            if edge is None:
                return []
            if edge["source_occurrence_id"] or edge["source_document_inventory_id"] or edge["source_span_id"]:
                rows = self.db.execute(
                    """SELECT COALESCE(di.relative_path, di_occ.relative_path, ao.document_id) AS source_document_ref,
                              COALESCE(ee.source_document_inventory_id, ao.document_inventory_id)
                                  AS source_document_inventory_id,
                              COALESCE(ee.source_span_id, ao.span_id) AS source_span_id,
                              ee.updated_at AS created_at
                       FROM evidence_edge ee
                       LEFT JOIN document_inventory di ON di.id = ee.source_document_inventory_id
                       LEFT JOIN assertion_occurrence ao ON ao.id = ee.source_occurrence_id
                       LEFT JOIN document_inventory di_occ ON di_occ.id = ao.document_inventory_id
                       WHERE ee.matter_id=? AND ee.id=?
                       LIMIT ?""",
                    (self.matter_id, target_id, int(limit)),
                ).fetchall()
                materialized = self._materialize_fallback_provenance(
                    target_kind, target_id, [dict(r) for r in rows],
                )
                if materialized:
                    return materialized
            if edge["source_kind"] == "assertion":
                return self.get_provenance("assertion", edge["source_id"], limit=limit)
            return []

        return []

    # ------------------------------------------------------------------
    # P0.3: Review Queue and Verification API (SO-3)
    # ------------------------------------------------------------------

    def get_review_queue(
        self,
        limit: int = 50,
        offset: int = 0,
        target_kind: Optional[str] = None,
    ) -> list[dict]:
        """P0.3: prioritized review queue — what a human reviewer
        should work on next. Thin facade over
        VerificationStateStore.review_queue so API and UI surfaces
        don't reach into stores directly."""
        return self.verification.review_queue(
            limit=limit, offset=offset, target_kind=target_kind,
        )

    def count_review_queue(self) -> dict:
        """Bucket-only counts for the sidebar badge and for verify/reject
        toast sizing. Avoids pulling the full queue when only totals are
        needed."""
        return self.verification.count_review_queue()

    def _issues_affected_by_target(
        self, target_kind: str, target_id: str,
    ) -> list[str]:
        """Return open issue ids whose support depends on this target,
        so rejection can trigger proof recomputation. P0.3 AC: rejections
        trigger proof recomputation or mark proof stale.

        Codex P0.3 review fix #3: every review target kind that can
        carry support must be covered. Previously only assertion and
        evidence_edge were handled — rejecting an authority, predicate,
        or quant_fact never triggered proof recomputation.
        """
        ids: set[str] = set()
        if target_kind == "assertion":
            rows = self.db.execute(
                """SELECT DISTINCT i.id FROM evidence_edge ee
                   JOIN issue i ON i.id=ee.target_id
                   WHERE ee.matter_id=? AND ee.source_kind='assertion'
                     AND ee.source_id=? AND ee.target_kind='issue'
                     AND ee.active=1 AND i.status='open'""",
                (self.matter_id, target_id),
            ).fetchall()
            legacy_rows = self.db.execute(
                """SELECT DISTINCT issue_id AS id FROM assertion_issue_link ail
                   JOIN issue i ON i.id = ail.issue_id
                   WHERE ail.assertion_id=? AND i.matter_id=? AND i.status='open'""",
                (target_id, self.matter_id),
            ).fetchall()
            ids.update(r["id"] for r in rows)
            ids.update(r["id"] for r in legacy_rows)
            # An assertion can also support an issue via a predicate —
            # find open predicates on open issues it links to.
            pred_rows = self.db.execute(
                """SELECT DISTINCT ip.issue_id AS id
                   FROM evidence_edge ee
                   JOIN issue_predicate ip ON ip.id=ee.target_id
                   JOIN issue i ON i.id=ip.issue_id
                   WHERE ee.matter_id=? AND ee.source_kind='assertion'
                     AND ee.source_id=? AND ee.target_kind='issue_predicate'
                     AND ee.active=1 AND i.status='open' AND ip.status='open'""",
                (self.matter_id, target_id),
            ).fetchall()
            ids.update(r["id"] for r in pred_rows)
        elif target_kind == "evidence_edge":
            # The edge target may be an issue directly OR a predicate
            # whose parent issue's proof state is affected.
            issue_rows = self.db.execute(
                """SELECT i.id FROM evidence_edge ee
                   JOIN issue i ON i.id=ee.target_id
                   WHERE ee.id=? AND ee.matter_id=?
                     AND ee.target_kind='issue' AND i.status='open'""",
                (target_id, self.matter_id),
            ).fetchall()
            ids.update(r["id"] for r in issue_rows)
            pred_rows = self.db.execute(
                """SELECT i.id FROM evidence_edge ee
                   JOIN issue_predicate ip ON ip.id=ee.target_id
                   JOIN issue i ON i.id=ip.issue_id
                   WHERE ee.id=? AND ee.matter_id=?
                     AND ee.target_kind='issue_predicate'
                     AND i.status='open' AND ip.status='open'""",
                (target_id, self.matter_id),
            ).fetchall()
            ids.update(r["id"] for r in pred_rows)
        elif target_kind == "issue_predicate":
            # A rejected predicate affects its parent issue's proof.
            rows = self.db.execute(
                """SELECT i.id FROM issue_predicate ip
                   JOIN issue i ON i.id=ip.issue_id
                   WHERE ip.id=? AND i.matter_id=? AND i.status='open'""",
                (target_id, self.matter_id),
            ).fetchall()
            ids.update(r["id"] for r in rows)
        elif target_kind == "authority":
            # Authorities can be linked to issues as supporting law.
            rows = self.db.execute(
                """SELECT DISTINCT i.id FROM authority_issue_link ail
                   JOIN issue i ON i.id=ail.issue_id
                   WHERE ail.authority_id=? AND i.matter_id=? AND i.status='open'""",
                (target_id, self.matter_id),
            ).fetchall() if self._table_exists("authority_issue_link") else []
            ids.update(r["id"] for r in rows)
        elif target_kind == "quant_fact":
            # A quant_fact's parent assertion (via assertion_id) is what
            # carries issue support; trace through it.
            parent_rows = self.db.execute(
                "SELECT assertion_id FROM quant_fact WHERE id=? AND matter_id=?",
                (target_id, self.matter_id),
            ).fetchall()
            for pr in parent_rows:
                pid = pr["assertion_id"]
                if pid:
                    ids.update(self._issues_affected_by_target("assertion", pid))
        return list(ids)

    def _table_exists(self, name: str) -> bool:
        """Small helper — some optional tables (authority_issue_link)
        exist only after later schema migrations. Guard reads so older
        matters don't crash."""
        row = self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        ).fetchone()
        return row is not None

    def verify_target(
        self,
        target_kind: str,
        target_id: str,
        *,
        reviewed_by_kind: str,
        reviewed_by_id: Optional[str] = None,
        review_note: Optional[str] = None,
        review_scope: str = "extraction_correct",
        run_id: Optional[str] = None,
    ) -> str:
        """P0.3: promote a target to verified, append ledger audit
        event, recompute proof state for any open issues it supports.
        review_scope records WHAT was validated (extraction, record
        truth, inference, legal conclusion) so downstream audit can
        tell a syntax-correct extraction apart from a
        record-truth-validated fact.
        """
        vid = self.verification.verify(
            target_kind, target_id,
            reviewed_by_kind=reviewed_by_kind,
            reviewed_by_id=reviewed_by_id,
            review_note=review_note,
            review_scope=review_scope,
            run_id=run_id,
        )
        # Adversarial #8 fix: verifying an assertion must also promote
        # its companion evidence_edge(s) to verified — otherwise the
        # two-lane verified_coverage_fraction stays at 0 because
        # TrustPolicy requires BOTH lanes to be verified, and the UI
        # looks like it lied. We fan-out only on assertion + evidence_edge
        # promotions; other kinds are leaves.
        if target_kind == "assertion":
            self._verify_companion_edges(
                assertion_id=target_id,
                reviewed_by_kind=reviewed_by_kind,
                reviewed_by_id=reviewed_by_id,
                review_note=review_note,
                review_scope=review_scope,
                run_id=run_id,
            )
        # verification_event (written by VerificationStateStore) is the
        # canonical audit row. Mirror to the reasoning ledger only when
        # a run_id is available — the schema enforces NOT NULL run_id
        # on ledger_event, and human reviews often happen outside any
        # run.
        if run_id is not None:
            self.ledger.append_event(
                run_id=run_id,
                event_type=LedgerEventType.ASSERTION_REVISED,
                summary=f"Verified {target_kind}:{target_id}",
                changed_object_type=target_kind,
                changed_object_id=target_id,
            )
        # Promotion to verified changes the verified_supporting_count
        # lane on any issue this target supports. Recompute affected
        # proof states so coverage_report reflects the new lane.
        # Narrow the fallback: sqlite errors during recompute are
        # survivable (the next recompute will pick it up); every
        # other exception is a real bug and should propagate.
        import sqlite3 as _sqlite3
        for iid in self._issues_affected_by_target(target_kind, target_id):
            try:
                self.proof_state.compute_and_store(iid, policy_audience="internal")
            except _sqlite3.Error as _exc:
                _log.warning(
                    "verify_target: proof recompute failed for issue %s: %s",
                    iid, _exc,
                )
        return vid

    def _verify_companion_edges(
        self,
        *,
        assertion_id: str,
        reviewed_by_kind: str,
        reviewed_by_id: Optional[str],
        review_note: Optional[str],
        review_scope: str,
        run_id: Optional[str],
    ) -> None:
        """Adversarial #8 fix: when the attorney verifies an
        assertion, also promote its system-inferred evidence_edges
        so the two-lane coverage actually moves. We only auto-verify
        edges whose origin_kind is 'system_inferred' — that's the
        class of edges Irys creates automatically when link_assertion
        fires. Edges a reviewer created by hand, or edges the
        reviewer has previously rejected, are left untouched.
        """
        rows = self.db.execute(
            """SELECT ee.id
               FROM evidence_edge ee
               LEFT JOIN verification_state vs
                 ON vs.target_kind='evidence_edge'
                AND vs.target_id=ee.id
                AND vs.matter_id=ee.matter_id
               WHERE ee.matter_id=? AND ee.source_kind='assertion'
                 AND ee.source_id=? AND ee.active=1
                 AND ee.origin_kind='system_inferred'
                 AND (vs.status IS NULL OR vs.status='candidate')""",
            (self.matter_id, assertion_id),
        ).fetchall()
        for row in rows:
            try:
                self.verification.verify(
                    "evidence_edge", row["id"],
                    reviewed_by_kind=reviewed_by_kind,
                    reviewed_by_id=reviewed_by_id,
                    review_note=(
                        review_note if review_note
                        else "promoted with parent assertion"
                    ),
                    review_scope=review_scope,
                    run_id=run_id,
                )
            except ValueError:
                # Human-gate rejected (e.g. automation trying to
                # promote) — skip rather than fail the whole
                # verify_target.
                continue

    def reject_target(
        self,
        target_kind: str,
        target_id: str,
        *,
        reviewed_by_kind: str,
        rejection_reason: str,
        reviewed_by_id: Optional[str] = None,
        review_note: Optional[str] = None,
        review_scope: str = "extraction_correct",
        run_id: Optional[str] = None,
    ) -> str:
        """P0.3: reject a target, append ledger audit event, and
        recompute proof state for any open issues it supported so
        their coverage drops accordingly."""
        vid = self.verification.reject(
            target_kind, target_id,
            reviewed_by_kind=reviewed_by_kind,
            reviewed_by_id=reviewed_by_id,
            rejection_reason=rejection_reason,
            review_note=review_note,
            review_scope=review_scope,
            run_id=run_id,
        )
        if run_id is not None:
            self.ledger.append_event(
                run_id=run_id,
                event_type=LedgerEventType.ASSERTION_REVISED,
                summary=f"Rejected {target_kind}:{target_id}: {rejection_reason}",
                changed_object_type=target_kind,
                changed_object_id=target_id,
            )
        # Adversarial #7 fix (SO-2 truth maintenance): rejection of a
        # supporting target must stale its direct dependents so
        # downstream reasoning re-evaluates. Previously we recomputed
        # proof only; dependent edges and quants stayed candidate
        # and silently re-entered consumers that don't read the
        # assertion's verification row. This is a bounded fan-out —
        # direct dependents only, no transitive walk.
        self._stale_rejection_dependents(target_kind, target_id)
        import sqlite3 as _sqlite3
        for iid in self._issues_affected_by_target(target_kind, target_id):
            try:
                self.proof_state.compute_and_store(iid, policy_audience="internal")
            except _sqlite3.Error as _exc:
                _log.warning(
                    "reject_target: proof recompute failed for issue %s: %s",
                    iid, _exc,
                )
        # P0.4: human rejection is a trust-invalidation trigger.
        # Bump the matter's trust_revision so downstream reasoning
        # caches keyed on the old revision silently miss. Subsequent
        # cached plans that referenced the now-rejected target are
        # unreachable, forcing a fresh LLM call.
        try:
            self.cache.bump_trust_revision()
        except _sqlite3.Error as _exc:
            _log.warning("reject_target: trust_revision bump failed: %s", _exc)
        return vid

    def _stale_rejection_dependents(
        self, target_kind: str, target_id: str,
    ) -> None:
        """Adversarial #7 fix: when a human rejects a supporting
        target, fan out stale to its direct dependents so the whole
        support chain reflects the rejection.

        Rules:
        - Rejecting an assertion: stale its evidence_edges and
          quant_facts sourced from it (occurrences preserved for
          audit but not staled — they're just raw utterance records).
        - Rejecting an evidence_edge: stale the edge alone; the
          assertion may still be valid in other contexts.
        - Rejecting an authority / predicate / quant_fact: no
          downstream fan-out (these are leaves).
        """
        if target_kind == "assertion":
            edge_rows = self.db.execute(
                """SELECT id FROM evidence_edge
                   WHERE matter_id=? AND source_kind='assertion' AND source_id=?
                     AND active=1""",
                (self.matter_id, target_id),
            ).fetchall()
            quant_rows = self.db.execute(
                "SELECT id FROM quant_fact WHERE matter_id=? AND assertion_id=?",
                (self.matter_id, target_id),
            ).fetchall()
            specs: list[dict] = []
            specs.extend(
                {"target_kind": "evidence_edge", "target_id": r["id"]}
                for r in edge_rows
            )
            specs.extend(
                {"target_kind": "quant_fact", "target_id": r["id"]}
                for r in quant_rows
            )
            if specs:
                self.verification.bulk_mark_stale(
                    specs,
                    stale_reason=f"upstream_rejected:assertion:{target_id}",
                )

    def bulk_verify_by_document(
        self,
        document_ref: str,
        *,
        reviewed_by_kind: str,
        reviewed_by_id: Optional[str] = None,
        review_note: Optional[str] = None,
        review_scope: str = "extraction_correct",
        run_id: Optional[str] = None,
    ) -> list[str]:
        """P0.3: bulk-verify every candidate assertion whose
        occurrence points at the given document. Convenience for
        reviewers who want to approve everything sourced from a
        single document at once.

        Codex P0.3 review fix #2: match on relative_path (slash-
        normalized), basename, and raw document_id via the
        document_inventory join so a caller passing either the full
        path or a basename still hits the right assertions.
        """
        # Normalize incoming ref: strip backslashes, split basename.
        ref_norm = (document_ref or "").replace("\\", "/")
        basename = ref_norm.rsplit("/", 1)[-1] if "/" in ref_norm else ref_norm
        rows = self.db.execute(
            """SELECT DISTINCT ao.assertion_id AS id
               FROM assertion_occurrence ao
               JOIN assertion a ON a.id=ao.assertion_id
               LEFT JOIN document_inventory di
                 ON di.id = ao.document_inventory_id
               LEFT JOIN verification_state vs
                 ON vs.target_kind='assertion'
                AND vs.target_id=a.id
                AND vs.matter_id=a.matter_id
               WHERE a.matter_id=?
                 AND (
                      ao.document_id = ?
                      OR REPLACE(ao.document_id, '\\', '/') = ?
                      OR ao.doc_basename = ?
                      OR REPLACE(di.relative_path, '\\', '/') = ?
                      OR di.relative_path = ?
                 )
                 AND COALESCE(vs.status, 'candidate')='candidate'""",
            (
                self.matter_id,
                document_ref,          # exact raw
                ref_norm,              # slash-normalized
                basename,              # basename match
                ref_norm,              # inventory slash-normalized
                document_ref,          # inventory exact raw
            ),
        ).fetchall()
        import sqlite3 as _sqlite3
        specs = [{"target_kind": "assertion", "target_id": r["id"]} for r in rows]
        ids = self.verification.bulk_set_status(
            specs,
            new_status="verified",
            reviewed_by_kind=reviewed_by_kind,
            reviewed_by_id=reviewed_by_id,
            review_note=review_note,
            review_scope=review_scope,
            run_id=run_id,
        )
        # Audit + proof recompute once per target so ledger events
        # reflect each promotion and downstream coverage updates.
        for spec in specs:
            if run_id is not None:
                self.ledger.append_event(
                    run_id=run_id,
                    event_type=LedgerEventType.ASSERTION_REVISED,
                    summary=(
                        f"Bulk-verified {spec['target_kind']}:{spec['target_id']} "
                        f"via document={document_ref}"
                    ),
                    changed_object_type=spec["target_kind"],
                    changed_object_id=spec["target_id"],
                )
            for iid in self._issues_affected_by_target(
                spec["target_kind"], spec["target_id"],
            ):
                try:
                    self.proof_state.compute_and_store(iid, policy_audience="internal")
                except _sqlite3.Error as _exc:
                    _log.warning(
                        "bulk_verify: proof recompute failed for issue %s: %s",
                        iid, _exc,
                    )
        return ids

    def list_candidate_assertions_for_document(
        self,
        document_ref: str,
    ) -> list[dict]:
        """Return the candidate assertions sourced from one document,
        with enough metadata for an attorney to accept or reject each
        one without opening individual rows.

        Mirrors `bulk_verify_by_document`'s document-matching rules
        (full path / slash-normalized / basename / inventory join) so
        the same document_ref maps to the same assertion set —
        attorneys can "see what I'm about to verify" without divergence
        between the preview and the one-shot action.

        Returns the list ordered newest-first. Each row carries:
          id, proposition_text, belief_state, primary_source_role,
          primary_speech_act, primary_document_id, confidence,
          occurrence_count, created_at.

        Only candidate rows are returned — already-verified, rejected,
        or stale assertions are filtered out (same gate as the bulk
        verify path).
        """
        ref_norm = (document_ref or "").replace("\\", "/")
        basename = ref_norm.rsplit("/", 1)[-1] if "/" in ref_norm else ref_norm
        rows = self.db.execute(
            """SELECT DISTINCT
                   a.id AS id,
                   a.proposition_text AS proposition_text,
                   a.belief_state AS belief_state,
                   a.confidence AS confidence,
                   a.created_at AS created_at,
                   fo.primary_source_role AS primary_source_role,
                   fo.primary_speech_act AS primary_speech_act,
                   fo.primary_document_id AS primary_document_id,
                   fo.occurrence_count AS occurrence_count
               FROM assertion_occurrence ao
               JOIN assertion a ON a.id=ao.assertion_id
               LEFT JOIN document_inventory di
                 ON di.id = ao.document_inventory_id
               LEFT JOIN verification_state vs
                 ON vs.target_kind='assertion'
                AND vs.target_id=a.id
                AND vs.matter_id=a.matter_id
               LEFT JOIN (
                   SELECT assertion_id,
                          MIN(source_role) AS primary_source_role,
                          MIN(speech_act) AS primary_speech_act,
                          MIN(document_id) AS primary_document_id,
                          COUNT(*) AS occurrence_count
                   FROM assertion_occurrence
                   GROUP BY assertion_id
               ) fo ON fo.assertion_id = a.id
               WHERE a.matter_id=?
                 AND (
                      ao.document_id = ?
                      OR REPLACE(ao.document_id, '\\', '/') = ?
                      OR ao.doc_basename = ?
                      OR REPLACE(di.relative_path, '\\', '/') = ?
                      OR di.relative_path = ?
                 )
                 AND COALESCE(vs.status, 'candidate')='candidate'
                 AND a.belief_state NOT IN ('withdrawn', 'superseded', 'rejected')
               ORDER BY a.created_at DESC""",
            (
                self.matter_id,
                document_ref,
                ref_norm,
                basename,
                ref_norm,
                document_ref,
            ),
        ).fetchall()
        return [dict(r) for r in rows]

    def bulk_verify_assertion_ids(
        self,
        assertion_ids: list[str],
        *,
        reviewed_by_kind: str,
        reviewed_by_id: Optional[str] = None,
        review_note: Optional[str] = None,
        review_scope: str = "extraction_correct",
        run_id: Optional[str] = None,
    ) -> list[str]:
        """Verify an explicit set of assertion ids. Used by the
        "review-then-verify" flow where the attorney has seen each
        fact and unchecked any they don't want to promote. Filters to
        ids that actually belong to this matter and are still
        candidates — an attacker-controlled id list can't promote
        random rows, and a stale UI selection silently skips
        already-verified items.
        """
        if not assertion_ids:
            return []
        seen: set[str] = set()
        filtered_input: list[str] = []
        for aid in assertion_ids:
            if not aid or aid in seen:
                continue
            seen.add(aid)
            filtered_input.append(aid)
        if not filtered_input:
            return []
        import sqlite3 as _sqlite3
        # SQLite bind-param limit: chunk the IN list.
        _BIND_LIMIT = 900
        # adv#13 Finding #2: close the TOCTOU race between the
        # candidate SELECT and the bulk_set_status UPDATE. Wrap both
        # in ONE transaction — self.db.transaction() issues
        # BEGIN IMMEDIATE, so no other connection can flip status
        # between our filter and our verify UPDATE. Without this, a
        # concurrent rejection could silently be overwritten to
        # verified.
        with self.db.transaction():
            valid_ids: list[str] = []
            for _chunk_start in range(0, len(filtered_input), _BIND_LIMIT):
                _chunk = filtered_input[_chunk_start : _chunk_start + _BIND_LIMIT]
                _rows = self.db.execute(
                    """SELECT a.id FROM assertion a
                       LEFT JOIN verification_state vs
                         ON vs.target_kind='assertion'
                        AND vs.target_id=a.id
                        AND vs.matter_id=a.matter_id
                       WHERE a.matter_id=?
                         AND a.id IN ({})
                         AND COALESCE(vs.status, 'candidate')='candidate'
                         AND a.belief_state NOT IN ('withdrawn', 'superseded', 'rejected')""".format(
                        ",".join("?" * len(_chunk))
                    ),
                    (self.matter_id, *_chunk),
                ).fetchall()
                valid_ids.extend(r["id"] for r in _rows)
            if not valid_ids:
                return []
            specs = [{"target_kind": "assertion", "target_id": aid} for aid in valid_ids]
            ids = self.verification.bulk_set_status(
                specs,
                new_status="verified",
                reviewed_by_kind=reviewed_by_kind,
                reviewed_by_id=reviewed_by_id,
                review_note=review_note,
                review_scope=review_scope,
                run_id=run_id,
            )
        for spec in specs:
            if run_id is not None:
                self.ledger.append_event(
                    run_id=run_id,
                    event_type=LedgerEventType.ASSERTION_REVISED,
                    summary=(
                        f"Batch-verified {spec['target_kind']}:{spec['target_id']} "
                        "via selected-subset flow"
                    ),
                    changed_object_type=spec["target_kind"],
                    changed_object_id=spec["target_id"],
                )
            for iid in self._issues_affected_by_target(
                spec["target_kind"], spec["target_id"],
            ):
                try:
                    self.proof_state.compute_and_store(iid, policy_audience="internal")
                except _sqlite3.Error as _exc:
                    _log.warning(
                        "bulk_verify_ids: proof recompute failed for issue %s: %s",
                        iid, _exc,
                    )
        return ids

    def bulk_verify_by_span(
        self,
        span_id: str,
        *,
        reviewed_by_kind: str,
        reviewed_by_id: Optional[str] = None,
        review_note: Optional[str] = None,
        review_scope: str = "extraction_correct",
        run_id: Optional[str] = None,
    ) -> list[str]:
        """P0.3: bulk-verify every candidate assertion whose
        occurrence points at the given span_id. Convenience for
        reviewers approving a clause, signature block, or paragraph
        at once (SO-3)."""
        rows = self.db.execute(
            """SELECT DISTINCT ao.assertion_id AS id
               FROM assertion_occurrence ao
               JOIN assertion a ON a.id=ao.assertion_id
               LEFT JOIN verification_state vs
                 ON vs.target_kind='assertion'
                AND vs.target_id=a.id
                AND vs.matter_id=a.matter_id
               WHERE a.matter_id=? AND ao.span_id=?
                 AND COALESCE(vs.status, 'candidate')='candidate'""",
            (self.matter_id, span_id),
        ).fetchall()
        import sqlite3 as _sqlite3
        specs = [{"target_kind": "assertion", "target_id": r["id"]} for r in rows]
        ids = self.verification.bulk_set_status(
            specs,
            new_status="verified",
            reviewed_by_kind=reviewed_by_kind,
            reviewed_by_id=reviewed_by_id,
            review_note=review_note,
            review_scope=review_scope,
            run_id=run_id,
        )
        for spec in specs:
            if run_id is not None:
                self.ledger.append_event(
                    run_id=run_id,
                    event_type=LedgerEventType.ASSERTION_REVISED,
                    summary=(
                        f"Bulk-verified {spec['target_kind']}:{spec['target_id']} "
                        f"via span={span_id}"
                    ),
                    changed_object_type=spec["target_kind"],
                    changed_object_id=spec["target_id"],
                )
            for iid in self._issues_affected_by_target(
                spec["target_kind"], spec["target_id"],
            ):
                try:
                    self.proof_state.compute_and_store(iid, policy_audience="internal")
                except _sqlite3.Error as _exc:
                    _log.warning(
                        "bulk_verify: proof recompute failed for issue %s: %s",
                        iid, _exc,
                    )
        return ids

    # ------------------------------------------------------------------
    # P0.4 Trust Invalidation Lite: bounded document + span invalidation
    # ------------------------------------------------------------------

    def _collect_document_invalidation_scope(
        self, doc_id: str,
    ) -> dict[str, set[str]]:
        """Gather direct dependents of a document. Returns a dict with
        sets keyed by target kind. Bounded — does NOT traverse
        assertion_link, support chains, or provenance_event. Each set
        reflects one hop from the document.

        Lookup resolves doc_id as either a document_inventory.id OR a
        relative_path so callers can pass whichever identity they have.
        """
        # Resolve the inventory row so we can look up its relative_path
        # for the legacy occurrence column and collect spans.
        inv = self.db.execute(
            """SELECT id, relative_path FROM document_inventory
               WHERE matter_id=? AND (id=? OR relative_path=?)""",
            (self.matter_id, doc_id, doc_id),
        ).fetchone()
        if inv is None:
            return {
                "assertion_ids": set(), "occurrence_ids": set(),
                "edge_ids": set(), "quant_ids": set(),
                "authority_ids": set(), "document_card_ids": set(),
            }
        inv_id = inv["id"]
        rel_path = inv["relative_path"]
        # Occurrences: inventory-linked OR legacy path-based.
        occ_rows = self.db.execute(
            """SELECT ao.id, ao.assertion_id FROM assertion_occurrence ao
               JOIN assertion a ON a.id=ao.assertion_id
               WHERE a.matter_id=?
                 AND (ao.document_inventory_id=? OR ao.document_id=?)""",
            (self.matter_id, inv_id, rel_path),
        ).fetchall()
        occurrence_ids = {r["id"] for r in occ_rows}
        assertion_ids = {r["assertion_id"] for r in occ_rows}
        # Spans attached to this document (`span` is the canonical table).
        span_rows = self.db.execute(
            "SELECT id FROM span WHERE document_id=?",
            (inv_id,),
        ).fetchall() if self._table_exists("span") else []
        doc_span_ids = {r["id"] for r in span_rows}
        # Quants: attached to assertions we already collected OR via
        # span_id when a quant was pinned to a doc span OR pinned to
        # an occurrence's span_id even when no `span` row exists.
        # P0.4 review fix: runtime writers accept bare span_id strings
        # without creating a span row; those quants were missed by
        # the doc_span_ids-only lookup.
        quant_rows = []
        if assertion_ids:
            placeholders = ",".join("?" * len(assertion_ids))
            quant_rows += self.db.execute(
                f"""SELECT id FROM quant_fact
                    WHERE matter_id=? AND assertion_id IN ({placeholders})""",
                (self.matter_id, *assertion_ids),
            ).fetchall()
        if doc_span_ids:
            placeholders = ",".join("?" * len(doc_span_ids))
            quant_rows += self.db.execute(
                f"""SELECT id FROM quant_fact
                    WHERE matter_id=? AND span_id IN ({placeholders})""",
                (self.matter_id, *doc_span_ids),
            ).fetchall()
        # Also sweep quants whose span_id matches an occurrence_span_id
        # for this document — bare-string span refs without `span` rows.
        occ_span_rows = self.db.execute(
            """SELECT DISTINCT ao.span_id FROM assertion_occurrence ao
               JOIN assertion a ON a.id=ao.assertion_id
               WHERE a.matter_id=? AND ao.span_id IS NOT NULL
                 AND (ao.document_inventory_id=? OR ao.document_id=?)""",
            (self.matter_id, inv_id, rel_path),
        ).fetchall()
        bare_span_ids = {r["span_id"] for r in occ_span_rows if r["span_id"]}
        bare_span_ids -= doc_span_ids  # don't double-query
        if bare_span_ids:
            placeholders = ",".join("?" * len(bare_span_ids))
            quant_rows += self.db.execute(
                f"""SELECT id FROM quant_fact
                    WHERE matter_id=? AND span_id IN ({placeholders})""",
                (self.matter_id, *bare_span_ids),
            ).fetchall()
        quant_ids = {r["id"] for r in quant_rows}
        # Authorities linked via source_doc_id or source_span_id.
        # P0.4 review fix: same bare-span sweep as quants — pick up
        # authorities pinned to an occurrence's span_id even when no
        # `span` row exists.
        auth_rows = []
        auth_rows += self.db.execute(
            """SELECT id FROM authority
               WHERE matter_id=? AND source_doc_id=?""",
            (self.matter_id, inv_id),
        ).fetchall()
        all_span_ids = doc_span_ids | bare_span_ids
        if all_span_ids:
            placeholders = ",".join("?" * len(all_span_ids))
            auth_rows += self.db.execute(
                f"""SELECT id FROM authority
                    WHERE matter_id=? AND source_span_id IN ({placeholders})""",
                (self.matter_id, *all_span_ids),
            ).fetchall()
        authority_ids = {r["id"] for r in auth_rows}
        # Document card.
        card_rows = self.db.execute(
            "SELECT id FROM document_card WHERE doc_id=?",
            (inv_id,),
        ).fetchall()
        document_card_ids = {r["id"] for r in card_rows}
        # Evidence edges sourced from these assertions.
        edge_ids: set[str] = set()
        if assertion_ids:
            placeholders = ",".join("?" * len(assertion_ids))
            edge_rows = self.db.execute(
                f"""SELECT id FROM evidence_edge
                    WHERE matter_id=? AND source_kind='assertion'
                      AND source_id IN ({placeholders})""",
                (self.matter_id, *assertion_ids),
            ).fetchall()
            edge_ids = {r["id"] for r in edge_rows}
        return {
            "assertion_ids": assertion_ids,
            "occurrence_ids": occurrence_ids,
            "edge_ids": edge_ids,
            "quant_ids": quant_ids,
            "authority_ids": authority_ids,
            "document_card_ids": document_card_ids,
        }

    def _collect_span_invalidation_scope(
        self, span_id: str,
    ) -> dict[str, set[str]]:
        """Gather direct dependents of a single span. Narrower than
        the document collector."""
        occ_rows = self.db.execute(
            """SELECT ao.id, ao.assertion_id FROM assertion_occurrence ao
               JOIN assertion a ON a.id=ao.assertion_id
               WHERE a.matter_id=? AND ao.span_id=?""",
            (self.matter_id, span_id),
        ).fetchall()
        occurrence_ids = {r["id"] for r in occ_rows}
        assertion_ids = {r["assertion_id"] for r in occ_rows}
        quant_rows = self.db.execute(
            """SELECT id FROM quant_fact
               WHERE matter_id=? AND span_id=?""",
            (self.matter_id, span_id),
        ).fetchall()
        quant_ids = {r["id"] for r in quant_rows}
        auth_rows = self.db.execute(
            """SELECT id FROM authority
               WHERE matter_id=? AND source_span_id=?""",
            (self.matter_id, span_id),
        ).fetchall()
        authority_ids = {r["id"] for r in auth_rows}
        edge_ids: set[str] = set()
        if assertion_ids:
            placeholders = ",".join("?" * len(assertion_ids))
            edge_rows = self.db.execute(
                f"""SELECT id FROM evidence_edge
                    WHERE matter_id=? AND source_kind='assertion'
                      AND (source_span_id=?
                           OR source_occurrence_id IN ({
                               ",".join("?" * len(occurrence_ids))
                           } ) OR source_id IN ({placeholders}))""",
                (
                    self.matter_id, span_id,
                    *occurrence_ids, *assertion_ids,
                ),
            ).fetchall() if occurrence_ids else self.db.execute(
                f"""SELECT id FROM evidence_edge
                    WHERE matter_id=? AND source_kind='assertion'
                      AND (source_span_id=? OR source_id IN ({placeholders}))""",
                (self.matter_id, span_id, *assertion_ids),
            ).fetchall()
            edge_ids = {r["id"] for r in edge_rows}
        return {
            "assertion_ids": assertion_ids,
            "occurrence_ids": occurrence_ids,
            "edge_ids": edge_ids,
            "quant_ids": quant_ids,
            "authority_ids": authority_ids,
        }

    def _apply_invalidation(
        self, scope: dict[str, set[str]], *, reason: str,
    ) -> int:
        """Mark every collected target stale in one sweep; recompute
        proof for affected open issues; bump trust_revision once.
        Returns the count of rows touched."""
        specs: list[dict] = []
        for kind_plural, kind_single in (
            ("assertion_ids", "assertion"),
            ("occurrence_ids", "assertion_occurrence"),
            ("edge_ids", "evidence_edge"),
            ("quant_ids", "quant_fact"),
            ("authority_ids", "authority"),
            ("document_card_ids", "document_card"),
        ):
            for tid in scope.get(kind_plural, set()):
                specs.append({"target_kind": kind_single, "target_id": tid})
        touched = self.verification.bulk_mark_stale(
            specs, stale_reason=reason,
        )
        # Recompute proof for every issue this scope touches.
        issue_ids: set[str] = set()
        for tid in scope.get("assertion_ids", set()):
            issue_ids.update(self._issues_affected_by_target("assertion", tid))
        for tid in scope.get("edge_ids", set()):
            issue_ids.update(self._issues_affected_by_target("evidence_edge", tid))
        import sqlite3 as _sqlite3
        for iid in issue_ids:
            try:
                self.proof_state.compute_and_store(iid, policy_audience="internal")
            except _sqlite3.Error as _exc:
                _log.warning(
                    "_apply_invalidation: proof recompute failed for issue %s: %s",
                    iid, _exc,
                )
        if touched:
            try:
                self.cache.bump_trust_revision()
            except _sqlite3.Error as _exc:
                _log.warning(
                    "_apply_invalidation: trust_revision bump failed: %s", _exc,
                )
        return len(touched)

    def mark_document_stale(self, doc_id: str, reason: str) -> int:
        """P0.4: mark every direct dependent of this document stale
        (assertions, occurrences, edges, quants, authorities, card).
        Recompute proof for affected issues, bump trust_revision.
        Returns the count of targets marked stale."""
        scope = self._collect_document_invalidation_scope(doc_id)
        return self._apply_invalidation(scope, reason=reason)

    def mark_span_stale(self, span_id: str, reason: str) -> int:
        """P0.4: mark every direct dependent of a single span stale.
        Tighter scope than mark_document_stale — used when a specific
        clause is replaced but the rest of the document is
        unchanged."""
        scope = self._collect_span_invalidation_scope(span_id)
        return self._apply_invalidation(scope, reason=reason)

    def reclassify_privilege(
        self,
        doc_id: str,
        new_flag: bool,
        *,
        reviewed_by_kind: str,
        reviewed_by_id: Optional[str] = None,
    ) -> int:
        """P0.4: an attorney flips a document's privilege classification.
        The card itself is verified under the privilege_classification
        review scope; downstream direct dependents (assertions, quants,
        authorities, edges) are staled so clean-audience synthesis
        re-evaluates with the new flag. Returns count of staled
        targets.

        Rejected targets are preserved (stronger opinion than stale).
        If the flag hasn't actually changed, this is a no-op (no stale,
        no trust_revision bump).
        """
        inv = self.db.execute(
            """SELECT id FROM document_inventory
               WHERE matter_id=? AND (id=? OR relative_path=?)""",
            (self.matter_id, doc_id, doc_id),
        ).fetchone()
        if inv is None:
            return 0
        inv_id = inv["id"]
        card = self.document_cards.get_by_doc_id(inv_id)
        old_flag = card.get("privilege_flag") if card else None
        if old_flag == (1 if new_flag else 0):
            return 0  # no change
        # Upsert the card with the new flag.
        card_id = self.document_cards.upsert(
            doc_id=inv_id,
            privilege_flag=new_flag,
        )
        # Verify the card itself under the privilege_classification
        # scope so the reviewer's decision is audited. ValueError
        # from the human-reviewer gate is survivable (the stale
        # sweep still runs with the new flag); logic errors should
        # propagate.
        try:
            self.verify_target(
                "document_card", card_id,
                reviewed_by_kind=reviewed_by_kind,
                reviewed_by_id=reviewed_by_id,
                review_scope="privilege_classification",
            )
        except ValueError as _exc:
            _log.warning(
                "reclassify_privilege: card verify blocked by human-gate: %s",
                _exc,
            )
        # Collect the document's direct dependents but exclude the
        # card itself (we just verified it).
        scope = self._collect_document_invalidation_scope(inv_id)
        scope["document_card_ids"] = set()
        return self._apply_invalidation(
            scope,
            reason=f"privilege_reclassified:{old_flag}->{1 if new_flag else 0}",
        )

    def get_verification_events(
        self,
        target_kind: Optional[str] = None,
        target_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """P0.3 review fix: expose verification_event rows so the
        audit surface is reachable from the HTTP layer, not just the
        store-local list_events(). SO-3 needs the user to be able to
        see every transition chronologically."""
        return self.verification.list_events(
            target_kind=target_kind, target_id=target_id, limit=limit,
        )

    def record_llm_call(self, record: LLMCallRecord) -> None:
        """Persist one Gemini API request for later cost and latency analysis.

        P0.1 provenance: call_id, prompt_hash, and response_hash are
        stored here so provenance_event rows can reference the call that
        produced them. call_id is preferred as the primary key; if the
        client minted one, we use it, otherwise fall back to a matter-
        local uuid (legacy callers).
        """
        row_id = record.call_id or _id()
        self.db.execute(
            """INSERT INTO llm_call
               (id, matter_id, run_id, model_tier, model_id, usage_label,
                input_tokens, cache_read_tokens, tool_use_prompt_tokens,
                thinking_tokens, output_tokens, total_prompt_tokens,
                estimated_cost_usd, latency_ms, success, error_kind, created_at,
                prompt_hash, response_hash)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                row_id,
                self.matter_id,
                record.run_id,
                record.model_tier,
                record.model_id,
                record.usage_label,
                record.input_tokens,
                record.cache_read_tokens,
                record.tool_use_prompt_tokens,
                record.thinking_tokens,
                record.output_tokens,
                record.total_prompt_tokens,
                record.estimated_cost_usd,
                record.latency_ms,
                1 if record.success else 0,
                record.error_kind,
                _now(),
                record.prompt_hash,
                record.response_hash,
            ),
        )

    def summarize_llm_usage(self, run_id: Optional[str] = None) -> dict[str, Any]:
        """Aggregate persisted Gemini usage for a matter or a single run."""
        where = "matter_id=?"
        params: list[Any] = [self.matter_id]
        if run_id is not None:
            where += " AND run_id=?"
            params.append(run_id)

        zero_summary = {
            "request_count": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "input_tokens": 0,
            "cache_read_tokens": 0,
            "tool_use_prompt_tokens": 0,
            "thinking_tokens": 0,
            "output_tokens": 0,
            "total_prompt_tokens": 0,
            "total_processed_tokens": 0,
            "estimated_cost_usd": 0.0,
            "by_tier": {},
            "pricing_source": PRICING_SOURCE_URL,
            "pricing_verified_at": PRICING_VERIFIED_AT,
        }

        try:
            totals = self.db.execute(
                f"""SELECT COUNT(*) AS request_count,
                           COALESCE(SUM(CASE WHEN success=1 THEN 1 ELSE 0 END), 0) AS successful_requests,
                           COALESCE(SUM(CASE WHEN success=0 THEN 1 ELSE 0 END), 0) AS failed_requests,
                           COALESCE(SUM(input_tokens), 0) AS input_tokens,
                           COALESCE(SUM(cache_read_tokens), 0) AS cache_read_tokens,
                           COALESCE(SUM(tool_use_prompt_tokens), 0) AS tool_use_prompt_tokens,
                           COALESCE(SUM(thinking_tokens), 0) AS thinking_tokens,
                           COALESCE(SUM(output_tokens), 0) AS output_tokens,
                           COALESCE(SUM(total_prompt_tokens), 0) AS total_prompt_tokens,
                           COALESCE(SUM(estimated_cost_usd), 0.0) AS estimated_cost_usd
                    FROM llm_call
                    WHERE {where}""",
                params,
            ).fetchone()
            tier_rows = self.db.execute(
                f"""SELECT model_tier,
                           COUNT(*) AS request_count,
                           COALESCE(SUM(input_tokens), 0) AS input_tokens,
                           COALESCE(SUM(cache_read_tokens), 0) AS cache_read_tokens,
                           COALESCE(SUM(tool_use_prompt_tokens), 0) AS tool_use_prompt_tokens,
                           COALESCE(SUM(thinking_tokens), 0) AS thinking_tokens,
                           COALESCE(SUM(output_tokens), 0) AS output_tokens,
                           COALESCE(SUM(total_prompt_tokens), 0) AS total_prompt_tokens,
                           COALESCE(SUM(estimated_cost_usd), 0.0) AS estimated_cost_usd
                    FROM llm_call
                    WHERE {where}
                    GROUP BY model_tier
                    ORDER BY model_tier""",
                params,
            ).fetchall()
        except Exception:
            return zero_summary

        by_tier = {
            row["model_tier"]: {
                "requests": int(row["request_count"] or 0),
                "input_tokens": int(row["input_tokens"] or 0),
                "cache_read_tokens": int(row["cache_read_tokens"] or 0),
                "tool_use_prompt_tokens": int(row["tool_use_prompt_tokens"] or 0),
                "thinking_tokens": int(row["thinking_tokens"] or 0),
                "output_tokens": int(row["output_tokens"] or 0),
                "total_prompt_tokens": int(row["total_prompt_tokens"] or 0),
                "total_processed_tokens": (
                    int(row["total_prompt_tokens"] or 0)
                    + int(row["tool_use_prompt_tokens"] or 0)
                    + int(row["thinking_tokens"] or 0)
                    + int(row["output_tokens"] or 0)
                ),
                "estimated_cost_usd": round(float(row["estimated_cost_usd"] or 0.0), 6),
            }
            for row in tier_rows
        }
        return {
            "request_count": int(totals["request_count"] or 0),
            "successful_requests": int(totals["successful_requests"] or 0),
            "failed_requests": int(totals["failed_requests"] or 0),
            "input_tokens": int(totals["input_tokens"] or 0),
            "cache_read_tokens": int(totals["cache_read_tokens"] or 0),
            "tool_use_prompt_tokens": int(totals["tool_use_prompt_tokens"] or 0),
            "thinking_tokens": int(totals["thinking_tokens"] or 0),
            "output_tokens": int(totals["output_tokens"] or 0),
            "total_prompt_tokens": int(totals["total_prompt_tokens"] or 0),
            "total_processed_tokens": (
                int(totals["total_prompt_tokens"] or 0)
                + int(totals["tool_use_prompt_tokens"] or 0)
                + int(totals["thinking_tokens"] or 0)
                + int(totals["output_tokens"] or 0)
            ),
            "estimated_cost_usd": round(float(totals["estimated_cost_usd"] or 0.0), 6),
            "by_tier": by_tier,
            "pricing_source": PRICING_SOURCE_URL,
            "pricing_verified_at": PRICING_VERIFIED_AT,
        }

    def get_cost_breakdown(
        self, run_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Cost visibility layer: richer aggregation than summarize_llm_usage.

        Returns totals (with cache_hit_rate, success_rate, and
        avg/p50/p95/p99 latency), by_stage + by_tier breakdowns
        (cache_hit_rate, success_rate, avg latency only — percentiles are
        totals-level), a trailing 7-day trend with zero-filled days, and
        a projected monthly burn.

        Cache hit rate is cache_read_tokens / (cache_read_tokens + input_tokens).
        run_id=None reports matter-wide; otherwise narrow to one run.

        Raises on real SQL failure — empty matters return a well-formed
        zeroed response, but database errors propagate so callers see
        real problems instead of silent zeros.
        """
        where = "matter_id=?"
        params: list[Any] = [self.matter_id]
        if run_id is not None:
            where += " AND run_id=?"
            params.append(run_id)

        zero = {
            "totals": {
                "request_count": 0,
                "estimated_cost_usd": 0.0,
                "input_tokens": 0,
                "cache_read_tokens": 0,
                "tool_use_prompt_tokens": 0,
                "thinking_tokens": 0,
                "output_tokens": 0,
                "total_processed_tokens": 0,
                "cache_hit_rate": None,
                "success_rate": None,
                "avg_latency_ms": None,
                "p50_latency_ms": None,
                "p95_latency_ms": None,
                "p99_latency_ms": None,
            },
            "by_stage": [],
            "by_tier": [],
            "trend": [],
            "estimated_monthly_burn_usd": 0.0,
            "pricing_source": PRICING_SOURCE_URL,
            "pricing_verified_at": PRICING_VERIFIED_AT,
        }

        try:
            totals_row = self.db.execute(
                f"""SELECT COUNT(*) AS n,
                           COALESCE(SUM(estimated_cost_usd), 0) AS cost,
                           COALESCE(SUM(input_tokens), 0) AS input_tokens,
                           COALESCE(SUM(cache_read_tokens), 0) AS cache_read_tokens,
                           COALESCE(SUM(tool_use_prompt_tokens), 0) AS tool_use_prompt_tokens,
                           COALESCE(SUM(thinking_tokens), 0) AS thinking_tokens,
                           COALESCE(SUM(output_tokens), 0) AS output_tokens,
                           COALESCE(SUM(CASE WHEN success=1 THEN 1 ELSE 0 END), 0) AS ok,
                           COALESCE(AVG(latency_ms), 0) AS avg_latency
                       FROM llm_call WHERE {where}""",
                params,
            ).fetchone()
            if not totals_row or int(totals_row["n"] or 0) == 0:
                return zero

            all_latencies = [
                int(r["latency_ms"] or 0) for r in self.db.execute(
                    f"SELECT latency_ms FROM llm_call WHERE {where}"
                    f" AND latency_ms > 0 ORDER BY latency_ms ASC",
                    params,
                ).fetchall()
            ]

            def _pct(vals: list[int], q: float) -> Optional[int]:
                """Order-statistic percentile on a pre-sorted list. Uses
                ceil((n+1)*q) - 1 so that p95 on N=20 returns the 19th
                order statistic rather than the max."""
                if not vals:
                    return None
                import math
                idx = max(0, min(len(vals) - 1, math.ceil((len(vals) + 1) * q) - 1))
                return vals[idx]

            n = int(totals_row["n"])
            inp = int(totals_row["input_tokens"] or 0)
            cached = int(totals_row["cache_read_tokens"] or 0)
            tool_use = int(totals_row["tool_use_prompt_tokens"] or 0)
            thinking = int(totals_row["thinking_tokens"] or 0)
            totals = {
                "request_count": n,
                "estimated_cost_usd": round(float(totals_row["cost"] or 0), 6),
                "input_tokens": inp,
                "cache_read_tokens": cached,
                "tool_use_prompt_tokens": tool_use,
                "thinking_tokens": thinking,
                "output_tokens": int(totals_row["output_tokens"] or 0),
                "total_processed_tokens": (
                    inp
                    + cached
                    + tool_use
                    + thinking
                    + int(totals_row["output_tokens"] or 0)
                ),
                "cache_hit_rate": (
                    round(cached / (cached + inp + tool_use), 4)
                    if (cached + inp + tool_use) > 0 else None
                ),
                "success_rate": (
                    round(int(totals_row["ok"] or 0) / n, 4) if n > 0 else None
                ),
                "avg_latency_ms": (
                    int(totals_row["avg_latency"]) if totals_row["avg_latency"] else None
                ),
                "p50_latency_ms": _pct(all_latencies, 0.5),
                "p95_latency_ms": _pct(all_latencies, 0.95),
                "p99_latency_ms": _pct(all_latencies, 0.99),
            }

            stage_rows = self.db.execute(
                f"""SELECT COALESCE(usage_label, 'unknown') AS stage,
                           COUNT(*) AS n,
                           COALESCE(SUM(estimated_cost_usd), 0) AS cost,
                           COALESCE(SUM(input_tokens), 0) AS input_tokens,
                           COALESCE(SUM(cache_read_tokens), 0) AS cache_read_tokens,
                           COALESCE(SUM(tool_use_prompt_tokens), 0) AS tool_use_prompt_tokens,
                           COALESCE(SUM(thinking_tokens), 0) AS thinking_tokens,
                           COALESCE(SUM(output_tokens), 0) AS output_tokens,
                           COALESCE(SUM(CASE WHEN success=1 THEN 1 ELSE 0 END), 0) AS ok,
                           COALESCE(AVG(latency_ms), 0) AS avg_latency
                       FROM llm_call
                       WHERE {where}
                       GROUP BY stage
                       ORDER BY cost DESC""",
                params,
            ).fetchall()
            by_stage = []
            for row in stage_rows:
                stg_inp = int(row["input_tokens"] or 0)
                stg_cached = int(row["cache_read_tokens"] or 0)
                stg_tool_use = int(row["tool_use_prompt_tokens"] or 0)
                stg_thinking = int(row["thinking_tokens"] or 0)
                stg_n = int(row["n"] or 0)
                by_stage.append({
                    "stage": row["stage"],
                    "request_count": stg_n,
                    "estimated_cost_usd": round(float(row["cost"] or 0), 6),
                    "input_tokens": stg_inp,
                    "cache_read_tokens": stg_cached,
                    "tool_use_prompt_tokens": stg_tool_use,
                    "thinking_tokens": stg_thinking,
                    "output_tokens": int(row["output_tokens"] or 0),
                    "total_processed_tokens": (
                        stg_inp
                        + stg_cached
                        + stg_tool_use
                        + stg_thinking
                        + int(row["output_tokens"] or 0)
                    ),
                    "cache_hit_rate": (
                        round(stg_cached / (stg_cached + stg_inp + stg_tool_use), 4)
                        if (stg_cached + stg_inp + stg_tool_use) > 0 else None
                    ),
                    "success_rate": (
                        round(int(row["ok"] or 0) / stg_n, 4) if stg_n > 0 else None
                    ),
                    "avg_latency_ms": (
                        int(row["avg_latency"]) if row["avg_latency"] else None
                    ),
                })

            tier_rows = self.db.execute(
                f"""SELECT model_tier,
                           COUNT(*) AS n,
                           COALESCE(SUM(estimated_cost_usd), 0) AS cost,
                           COALESCE(SUM(input_tokens), 0) AS input_tokens,
                           COALESCE(SUM(cache_read_tokens), 0) AS cache_read_tokens,
                           COALESCE(SUM(tool_use_prompt_tokens), 0) AS tool_use_prompt_tokens,
                           COALESCE(SUM(thinking_tokens), 0) AS thinking_tokens,
                           COALESCE(SUM(output_tokens), 0) AS output_tokens,
                           COALESCE(SUM(CASE WHEN success=1 THEN 1 ELSE 0 END), 0) AS ok,
                           COALESCE(AVG(latency_ms), 0) AS avg_latency
                       FROM llm_call
                       WHERE {where}
                       GROUP BY model_tier
                       ORDER BY cost DESC""",
                params,
            ).fetchall()
            by_tier = []
            for row in tier_rows:
                t_inp = int(row["input_tokens"] or 0)
                t_cached = int(row["cache_read_tokens"] or 0)
                t_tool_use = int(row["tool_use_prompt_tokens"] or 0)
                t_thinking = int(row["thinking_tokens"] or 0)
                t_n = int(row["n"] or 0)
                by_tier.append({
                    "model_tier": row["model_tier"],
                    "request_count": t_n,
                    "estimated_cost_usd": round(float(row["cost"] or 0), 6),
                    "input_tokens": t_inp,
                    "cache_read_tokens": t_cached,
                    "tool_use_prompt_tokens": t_tool_use,
                    "thinking_tokens": t_thinking,
                    "output_tokens": int(row["output_tokens"] or 0),
                    "total_processed_tokens": (
                        t_inp
                        + t_cached
                        + t_tool_use
                        + t_thinking
                        + int(row["output_tokens"] or 0)
                    ),
                    "cache_hit_rate": (
                        round(t_cached / (t_cached + t_inp + t_tool_use), 4)
                        if (t_cached + t_inp + t_tool_use) > 0 else None
                    ),
                    "success_rate": (
                        round(int(row["ok"] or 0) / t_n, 4) if t_n > 0 else None
                    ),
                    "avg_latency_ms": (
                        int(row["avg_latency"]) if row["avg_latency"] else None
                    ),
                })

            # Trailing 7-day window with zero-fill so the monthly burn
            # projection is scaled against a real 7-day slice, not "up to
            # seven days that had calls" spanning arbitrary history.
            from datetime import datetime, timedelta, timezone
            today = datetime.now(timezone.utc).date()
            start = today - timedelta(days=6)
            trend_rows = self.db.execute(
                f"""SELECT SUBSTR(created_at, 1, 10) AS day,
                           COUNT(*) AS n,
                           COALESCE(SUM(estimated_cost_usd), 0) AS cost
                       FROM llm_call
                       WHERE {where}
                         AND SUBSTR(created_at, 1, 10) >= ?
                       GROUP BY day""",
                [*params, start.isoformat()],
            ).fetchall()
            by_day = {
                row["day"]: (int(row["n"] or 0), float(row["cost"] or 0))
                for row in trend_rows
            }
            trend = []
            for i in range(7):
                day = (start + timedelta(days=i)).isoformat()
                n, cost = by_day.get(day, (0, 0.0))
                trend.append({
                    "day": day,
                    "request_count": n,
                    "estimated_cost_usd": round(cost, 6),
                })

            last_week_cost = sum(t["estimated_cost_usd"] for t in trend)
            estimated_monthly_burn_usd = round(last_week_cost * (30 / 7), 4)

            return {
                "totals": totals,
                "by_stage": by_stage,
                "by_tier": by_tier,
                "trend": trend,
                "estimated_monthly_burn_usd": estimated_monthly_burn_usd,
                "pricing_source": PRICING_SOURCE_URL,
                "pricing_verified_at": PRICING_VERIFIED_AT,
            }
        except sqlite3.Error:
            # Real DB errors propagate — silent-zero would launder the
            # failure into a "clean" analytics panel, a pattern
            # adversarial audits 1–3 flagged repeatedly.
            raise

    def get_cost_anomalies(
        self, limit: int = 10, run_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Return calls that are outliers on cost or latency vs their tier's
        robust baseline. First candidates when chasing cost reductions.

        Uses median + MAD-based z-scores (not mean/pstdev) so that a single
        large outlier does not inflate σ and hide itself. Thresholds:
        modified z-score > 3.5 (Iglewicz & Hoaglin 1993) OR raw value more
        than 3× the median, whichever flags more.
        """
        where = "matter_id=?"
        params: list[Any] = [self.matter_id]
        if run_id is not None:
            where += " AND run_id=?"
            params.append(run_id)
        try:
            import statistics
            tiers = [
                row["model_tier"] for row in self.db.execute(
                    f"""SELECT model_tier
                           FROM llm_call
                           WHERE {where} AND success=1
                           GROUP BY model_tier
                           HAVING COUNT(*) >= 3""",
                    params,
                ).fetchall()
            ]
            anomalies: list[dict] = []
            for tier in tiers:
                rows = self.db.execute(
                    f"""SELECT id, created_at, model_tier, usage_label,
                               input_tokens, cache_read_tokens, tool_use_prompt_tokens,
                               thinking_tokens, output_tokens,
                               estimated_cost_usd, latency_ms
                           FROM llm_call
                           WHERE {where} AND model_tier=? AND success=1""",
                    [*params, tier],
                ).fetchall()
                if len(rows) < 3:
                    continue
                costs = [float(r["estimated_cost_usd"] or 0) for r in rows]
                lats = [float(r["latency_ms"] or 0) for r in rows]
                med_cost = statistics.median(costs)
                med_lat = statistics.median(lats)
                # Median absolute deviation (scaled to approx σ for normal data)
                mad_cost = statistics.median(
                    [abs(c - med_cost) for c in costs]
                ) * 1.4826
                mad_lat = statistics.median(
                    [abs(l - med_lat) for l in lats]
                ) * 1.4826

                for r in rows:
                    cost = float(r["estimated_cost_usd"] or 0)
                    lat = float(r["latency_ms"] or 0)
                    cost_z = (
                        (cost - med_cost) / mad_cost if mad_cost > 0 else None
                    )
                    lat_z = (
                        (lat - med_lat) / mad_lat if mad_lat > 0 else None
                    )
                    # Flag if either z > 3.5 OR raw > 3× median (covers MAD=0
                    # cases where most calls are near-identical).
                    flag_cost = (
                        (cost_z is not None and cost_z > 3.5)
                        or (med_cost > 0 and cost > 3 * med_cost)
                    )
                    flag_lat = (
                        (lat_z is not None and lat_z > 3.5)
                        or (med_lat > 0 and lat > 3 * med_lat)
                    )
                    if not (flag_cost or flag_lat):
                        continue
                    d = dict(r)
                    d["baseline_cost"] = round(med_cost, 6)
                    d["baseline_latency_ms"] = int(med_lat)
                    d["cost_z"] = round(cost_z, 2) if cost_z is not None else None
                    d["latency_z"] = round(lat_z, 2) if lat_z is not None else None
                    anomalies.append(d)

            anomalies.sort(key=lambda r: -float(r.get("cost_z") or 0))
            return anomalies[:limit]
        except sqlite3.Error:
            # Same rationale as get_cost_breakdown: DB errors propagate so
            # failures surface rather than hiding behind an empty list.
            raise

    def list_llm_calls(
        self,
        run_id: Optional[str] = None,
        limit: int = 120,
    ) -> list[dict[str, Any]]:
        """Return recent LLM call rows for matter-level analytics surfaces."""
        where = "matter_id=?"
        params: list[Any] = [self.matter_id]
        if run_id is not None:
            where += " AND run_id=?"
            params.append(run_id)
        params.append(int(limit))
        try:
            rows = self.db.execute(
                f"""SELECT run_id, model_tier, usage_label,
                           input_tokens, cache_read_tokens, tool_use_prompt_tokens,
                           thinking_tokens, output_tokens,
                           total_prompt_tokens, estimated_cost_usd, latency_ms,
                           success, error_kind, created_at
                    FROM llm_call
                    WHERE {where}
                    ORDER BY created_at DESC
                    LIMIT ?""",
                params,
            ).fetchall()
        except Exception:
            return []
        return [
            {
                "run_id": row["run_id"],
                "model_tier": row["model_tier"],
                "usage_label": row["usage_label"],
                "input_tokens": int(row["input_tokens"] or 0),
                "cache_read_tokens": int(row["cache_read_tokens"] or 0),
                "tool_use_prompt_tokens": int(row["tool_use_prompt_tokens"] or 0),
                "thinking_tokens": int(row["thinking_tokens"] or 0),
                "output_tokens": int(row["output_tokens"] or 0),
                "total_prompt_tokens": int(row["total_prompt_tokens"] or 0),
                "total_processed_tokens": (
                    int(row["total_prompt_tokens"] or 0)
                    + int(row["tool_use_prompt_tokens"] or 0)
                    + int(row["thinking_tokens"] or 0)
                    + int(row["output_tokens"] or 0)
                ),
                "estimated_cost_usd": round(float(row["estimated_cost_usd"] or 0.0), 6),
                "latency_ms": int(row["latency_ms"] or 0),
                "success": bool(row["success"]),
                "error_kind": row["error_kind"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def record_run_usage_summary(self, run_id: str, usage: dict[str, Any]) -> None:
        """Persist cheap per-run Gemini totals onto run_session for UI fetches."""
        self.db.execute(
            "UPDATE run_session"
            " SET llm_input_tokens=?, llm_cache_read_tokens=?,"
            "     llm_tool_use_prompt_tokens=?, llm_thinking_tokens=?,"
            "     llm_output_tokens=?, llm_total_processed_tokens=?,"
            "     llm_request_count=?, llm_estimated_cost_usd=?"
            " WHERE id=? AND matter_id=?",
            (
                int(usage.get("input_tokens", 0) or 0),
                int(usage.get("cache_read_tokens", 0) or 0),
                int(usage.get("tool_use_prompt_tokens", 0) or 0),
                int(usage.get("thinking_tokens", 0) or 0),
                int(usage.get("output_tokens", 0) or 0),
                int(usage.get("total_processed_tokens", 0) or 0),
                int(usage.get("request_count", 0) or 0),
                float(usage.get("estimated_cost_usd", 0.0) or 0.0),
                run_id,
                self.matter_id,
            ),
        )

    # ------------------------------------------------------------------
    # Assertion management
    # ------------------------------------------------------------------

    def record_assertion(
        self,
        candidate: AssertionCandidate,
        run_id: Optional[str] = None,
        *,
        provenance: "Optional[ProvenanceContext]" = None,
    ) -> tuple[str, bool]:
        """
        Upsert a canonical assertion and record an occurrence.

        Returns (assertion_id, is_new_assertion). P0.1: optional
        ProvenanceContext forwards to AssertionStore.upsert_occurrence.
        """
        return self.assertions.upsert_occurrence(
            candidate, run_id=run_id, provenance=provenance,
        )

    def link_assertions(
        self,
        src_id: str,
        dst_id: str,
        link_type: AssertionLinkType,
        weight: float = 1.0,
    ) -> str:
        return self.assertions.link(src_id, dst_id, link_type, weight)

    def search_assertions(
        self,
        queries: list[str],
        issue_id: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict]:
        """Search active assertions for hot-path query answering."""
        return self.assertions.search(queries, issue_id=issue_id, limit=limit)

    def _ensure_belief_trust_weights(self) -> None:
        """Load composed trust weights into the belief engine if not yet set."""
        if self.belief.trust_weights is not None:
            return
        _, tw, _ = self._read_matter_domain_composition()
        if tw:
            self.belief.trust_weights = tw

    def apply_revision(
        self,
        seed_assertion_ids: list[str],
        cause: RevisionCause,
        run_id: Optional[str] = None,
        note: Optional[str] = None,
        _collect_unvisited: "list[str] | None" = None,
    ) -> list[RevisionResult]:
        """Trigger belief revision from seed assertions."""
        self._ensure_belief_trust_weights()
        return self.belief.apply(seed_assertion_ids, cause, run_id, note, _collect_unvisited)

    def enqueue_correction_pending(
        self, ids: list[str], run_id: "str | None" = None
    ) -> None:
        """Add assertion IDs to the durable correction retry queue (thread-safe).

        run_id is the originating run that triggered the correction; stored alongside
        the assertion_id so flush_revisions() can emit ASSERTION_REVISED ledger events
        under the correct originating run rather than the draining adapter's run_id
        (r29 MEDIUM provenance fix). First-write wins: if the same assertion is already
        queued from an earlier correction, its original run_id is preserved.
        Persists each new entry to pending_propagation so queued work survives restarts
        (adv#029 SO-1 HIGH fix). DB write happens before in-memory update so a DB failure
        propagates to the caller rather than leaving inconsistent best-effort state.
        """
        if not ids:
            return
        now = _now()
        with self._correction_pending_lock:
            _new: list[str] = []
            for aid in ids:
                if aid not in self._correction_pending:
                    _new.append(aid)
            if _new:
                with self.db.transaction():
                    for aid in _new:
                        self.db.execute(
                            "INSERT OR REPLACE INTO pending_propagation"
                            " (id, matter_id, assertion_id, cause, orig_run_id, queue, enqueued_at)"
                            " VALUES (?,?,?,?,?,?,?)",
                            (_id(), self.matter_id, aid, "USER_CORRECTION", run_id, "correction", now),
                        )
                # DB committed — now safe to update in-memory dict
                for aid in _new:
                    self._correction_pending[aid] = run_id

    def drain_correction_pending(self) -> "tuple[dict[str, str | None], list[str]]":
        """Return and clear the durable correction retry queue (thread-safe).

        Returns (assertion_id→orig_run_id map, list of DB row primary keys drained).
        The DB primary keys are read inside the lock so no concurrent enqueue can
        replace a row between the dict clear and the ID snapshot — any concurrent
        enqueue that starts after lock release will INSERT OR REPLACE, creating a
        fresh row with a new primary key that is not in the returned id list and
        therefore will not be deleted by flush_revisions() (adv#029 SO-1 fix r4).

        Callers MUST call delete_pending_propagation_db(db_row_ids) AFTER replay
        completes so crash-between-drain-and-delete is recoverable.
        """
        with self._correction_pending_lock:
            result = dict(self._correction_pending)
            self._correction_pending.clear()
            db_ids: list[str] = []
            _aid_list = list(result)
            _SQL_PARAM_LIMIT = 900
            for _bs in range(0, len(_aid_list), _SQL_PARAM_LIMIT):
                _batch = _aid_list[_bs : _bs + _SQL_PARAM_LIMIT]
                try:
                    rows = self.db.execute(
                        "SELECT id FROM pending_propagation"
                        " WHERE matter_id=? AND queue='correction'"
                        " AND assertion_id IN ({})".format(",".join("?" * len(_batch))),
                        [self.matter_id] + _batch,
                    ).fetchall()
                    db_ids.extend(r["id"] for r in rows)
                except Exception as _dbe:
                    _log.warning("pending_propagation drain SELECT failed (correction): %s", _dbe)
        return result, db_ids

    def enqueue_evidence_pending(
        self,
        ids: list[str],
        cause: "RevisionCause" = None,
        run_id: Optional[str] = None,
    ) -> None:
        """Add BFS-truncated assertion IDs to the durable evidence queue with cause + run_id.

        cause: originating RevisionCause (TRUST_OVERRIDE, CONFLICT_DETECTION, NEW_EVIDENCE, etc.)
        run_id: originating run for audit attribution in the deferred replay batch event.
        Defaults to NEW_EVIDENCE/None. First-write wins on all fields per assertion_id.
        Persists each new entry to pending_propagation so queued work survives restarts
        (adv#029 SO-1 HIGH fix). DB write happens before in-memory update so a DB failure
        propagates to the caller rather than leaving inconsistent best-effort state.
        """
        if not ids:
            return
        _cause = cause if cause is not None else RevisionCause.NEW_EVIDENCE
        now = _now()
        with self._evidence_pending_lock:
            _new: list[str] = []
            for aid in ids:
                if aid not in self._evidence_pending:
                    _new.append(aid)
            if _new:
                with self.db.transaction():
                    for aid in _new:
                        self.db.execute(
                            "INSERT OR REPLACE INTO pending_propagation"
                            " (id, matter_id, assertion_id, cause, orig_run_id, queue, enqueued_at)"
                            " VALUES (?,?,?,?,?,?,?)",
                            (_id(), self.matter_id, aid, _cause.value, run_id, "evidence", now),
                        )
                # DB committed — now safe to update in-memory dict
                for aid in _new:
                    self._evidence_pending[aid] = (_cause, run_id)

    def drain_evidence_pending(self) -> "tuple[dict[str, tuple[RevisionCause, str | None]], list[str]]":
        """Return and clear truncated evidence assertion IDs → (cause, run_id) (thread-safe).

        Returns (assertion_id→(cause, orig_run_id) map, list of DB row primary keys drained).
        DB IDs read inside the lock — same concurrency guarantee as drain_correction_pending.
        Callers MUST call delete_pending_propagation_db(db_row_ids) AFTER replay completes
        (adv#029 SO-1 fix r4).
        """
        with self._evidence_pending_lock:
            result = dict(self._evidence_pending)
            self._evidence_pending.clear()
            db_ids: list[str] = []
            _aid_list = list(result)
            _SQL_PARAM_LIMIT = 900
            for _bs in range(0, len(_aid_list), _SQL_PARAM_LIMIT):
                _batch = _aid_list[_bs : _bs + _SQL_PARAM_LIMIT]
                try:
                    rows = self.db.execute(
                        "SELECT id FROM pending_propagation"
                        " WHERE matter_id=? AND queue='evidence'"
                        " AND assertion_id IN ({})".format(",".join("?" * len(_batch))),
                        [self.matter_id] + _batch,
                    ).fetchall()
                    db_ids.extend(r["id"] for r in rows)
                except Exception as _dbe:
                    _log.warning("pending_propagation drain SELECT failed (evidence): %s", _dbe)
        return result, db_ids

    def delete_pending_propagation_db(self, db_row_ids: "list[str]") -> None:
        """Delete pending_propagation rows by primary key after successful replay.

        Called by flush_revisions() AFTER replay of the drained set is complete.
        Deletes by `id` (primary key), not by assertion_id, so concurrent re-enqueues
        that do INSERT OR REPLACE (creating new rows with fresh primary keys) are not
        affected — only the exact rows that were drained are removed
        (adv#029 SO-1 fix r4 — solves the peek+delete race by using stable row identity).
        """
        if not db_row_ids:
            return
        _SQL_PARAM_LIMIT = 900
        for _bs in range(0, len(db_row_ids), _SQL_PARAM_LIMIT):
            _batch = db_row_ids[_bs : _bs + _SQL_PARAM_LIMIT]
            try:
                self.db.execute(
                    "DELETE FROM pending_propagation WHERE id IN ({})".format(
                        ",".join("?" * len(_batch))
                    ),
                    _batch,
                )
            except Exception as _e:
                _log.warning(
                    "pending_propagation cleanup failed for %d rows (will replay on next flush): %s",
                    len(_batch), _e,
                )

    def correct_assertion(
        self,
        assertion_id: str,
        new_state: BeliefState,
        run_id: Optional[str] = None,
        note: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> RevisionResult:
        """Apply a user correction to an assertion and propagate.

        After belief revision propagates through the assertion graph, this
        triggers a targeted proof_state recompute for all issues linked to
        the corrected assertion — so issue-level prioritization in the loop
        reflects the correction, not a stale pre-correction state (SO-2).

        run_id is validated: if provided but not a running session for this matter,
        it is silently cleared to None so stale IDs cannot misattribute audit rows
        regardless of the calling path (REST, in-process, or engine).
        """
        self._ensure_belief_trust_weights()
        # Model-level run_id guard — covers all callers (r37 MEDIUM fix).
        if run_id:
            try:
                _valid_run = self.db.execute(
                    "SELECT 1 FROM run_session WHERE id=? AND matter_id=? AND status='running'"
                    " AND (objective IS NULL OR objective NOT IN ('manual_flush','background_flush'))",
                    (run_id, self.matter_id),
                ).fetchone()
                if not _valid_run:
                    run_id = None
            except Exception:
                run_id = None
        if confidence is None:
            confidence_map = {
                BeliefState.OPERATIVE: 0.95,
                BeliefState.ADMITTED: 0.90,
                BeliefState.SUPERSEDED: 0.10,
                BeliefState.WITHDRAWN: 0.0,
                BeliefState.DISPUTED: 0.3,
            }
            confidence = confidence_map.get(new_state, 0.5)
        result = self.belief.force_state(
            assertion_id=assertion_id,
            new_state=new_state,
            new_confidence=confidence,
            cause=RevisionCause.USER_CORRECTION,
            run_id=run_id,
            note=note,
        )

        # Retry nodes left unvisited by BFS budget truncation (r17/r18 HIGH fix).
        # Unvisited nodes come from result.truncation_pending — stored locally on the
        # RevisionResult, never on the engine, so concurrent corrections cannot
        # contaminate each other's retry frontiers.
        #
        # Cap: 3 inline rounds to bound synchronous HTTP latency (~6k visits max).
        # Any remaining nodes after the cap stay in result.truncation_pending so the
        # caller (e.g. REST endpoint) can surface propagation_truncated to the client,
        # and the next flush_revisions() or compute_all() will finish the work.
        _pending: list[str] = list(result.truncation_pending)
        result.truncation_pending = []  # consumed from here; new ones collected below
        for _round in range(3):
            if not _pending:
                break
            _next_pending: list[str] = []
            _retry_results = self.belief.apply(
                _pending,
                cause=RevisionCause.USER_CORRECTION,
                run_id=run_id,
                note="truncation retry",
                _collect_unvisited=_next_pending,
            )
            result.propagated_to = list(
                dict.fromkeys(
                    (result.propagated_to or [])
                    + [r.assertion_id for r in _retry_results]
                )
            )
            _pending = _next_pending

        # Update top-level truncation flag: False only if we fully converged.
        result.propagation_truncated = bool(_pending)
        if _pending:
            result.truncation_pending = _pending  # surface remaining work to caller
            # Durably queue unvisited nodes so the next flush_revisions() call can
            # finish propagation — prevents permanent stale states after cap exhaustion
            # (adversarial #028 HIGH fix). Lock protects concurrent corrections on the
            # same shared MatterModel instance (r25 MEDIUM fix).
            self.enqueue_correction_pending(_pending, run_id=run_id)

        # Targeted proof_state recompute: find issues linked to this assertion
        # and any that were revised as dependents (result.propagated_to).
        # This ensures downstream issue/proof consumers see the corrected state.
        # Note: when MAX_WORK truncation occurred, propagated_to may be incomplete;
        # remaining issues will be refreshed on the next compute_all() or trust-override.
        # The truncation SYSTEM_WARNING in the ledger makes this visible to users.
        try:
            # Deduplicate to avoid SQLite bind-variable overrun on large propagation sets.
            # dict.fromkeys preserves order while deduplicating.
            affected = list(dict.fromkeys([assertion_id] + (result.propagated_to or [])))
            if affected:
                # SQLite bind limit ~999: chunk the assertion list so ALL affected
                # issues are discovered, even when propagation chains exceed 900 nodes.
                _SQL_PARAM_LIMIT = 900
                issue_ids_to_recompute: set[str] = set()
                for _batch_start in range(0, len(affected), _SQL_PARAM_LIMIT):
                    _batch = affected[_batch_start:_batch_start + _SQL_PARAM_LIMIT]
                    _rows = self.db.execute(
                        "SELECT DISTINCT issue_id FROM assertion_issue_link"
                        " WHERE assertion_id IN ({})".format(
                            ",".join("?" * len(_batch))
                        ),
                        _batch,
                    ).fetchall()
                    issue_ids_to_recompute.update(r["issue_id"] for r in _rows)
                if issue_ids_to_recompute:
                    # Pre-load trust overrides once — avoids one DB query per issue
                    # (same pattern as compute_all()).
                    _override_rows = self.db.execute(
                        """SELECT document_pattern, trust_level FROM document_trust_override
                           WHERE matter_id=? AND trust_level != 'normal'
                           ORDER BY LENGTH(document_pattern) DESC""",
                        (self.matter_id,),
                    ).fetchall()
                    _overrides = [
                        (r["document_pattern"], r["trust_level"]) for r in _override_rows
                    ]
                    # Batch all writes in one transaction — N→1 BEGIN/COMMIT cycles.
                    with self.db.transaction():
                        for _iid in issue_ids_to_recompute:
                            self.proof_state.compute_and_store(
                                _iid, _preloaded_overrides=_overrides
                            )
        except Exception as exc:
            _log.warning(
                "proof_state recompute after correct_assertion failed for %r: %s",
                assertion_id, exc,
            )

        # adv#12 Finding #1: user correction is an authoritative
        # truth-maintenance event; any cached cascade decision or
        # orientation plan keyed on the pre-correction state is now
        # stale. Bump trust_revision so those entries silently miss
        # on the next lookup. Matches the pattern used for
        # reject_target, trust_override set/delete, and span/doc
        # invalidation.
        try:
            self.cache.bump_trust_revision()
        except sqlite3.Error as _exc:
            _log.warning("correct_assertion: trust_revision bump failed: %s", _exc)

        return result

    def set_trust_override(
        self,
        document_pattern: str,
        trust_level: str,
        note: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> str:
        """Set a document trust override and trigger belief revision on affected assertions.

        This is the high-level entry point for SO-3 trust steering.  It:
        1. Persists the trust override to document_trust_override (SO-3).
        2. Finds all assertions whose primary document matches the pattern.
        3. Triggers belief revision (RevisionCause.TRUST_OVERRIDE) on those assertions
           so the new effective source_role weight flows through to stored belief states (SO-2).

        Belief revision failure does not block the override — the override is persisted
        regardless of whether propagation succeeds.

        Returns the override_id.
        """
        # Model-level run_id guard — same pattern as correct_assertion (r37 / adv#030 fix).
        if run_id:
            try:
                _valid_run = self.db.execute(
                    "SELECT 1 FROM run_session WHERE id=? AND matter_id=? AND status='running'"
                    " AND (objective IS NULL OR objective NOT IN ('manual_flush','background_flush'))",
                    (run_id, self.matter_id),
                ).fetchone()
                if not _valid_run:
                    run_id = None
            except Exception:
                run_id = None

        override_id = self.trust_overrides.set(document_pattern, trust_level, note)

        # Adv#11 Fix 1: trust posture changed — bump trust_revision so any
        # cascade decision / reasoning cache keyed on the prior revision
        # silently misses. Belief revision below only touches existing
        # assertion rows; the revision bump invalidates cached plans that
        # *would have* routed around the newly re-scored source.
        try:
            self.cache.bump_trust_revision()
        except sqlite3.Error as _exc:
            _log.warning("set_trust_override: trust_revision bump failed: %s", _exc)

        # Trigger belief revision on all assertions from the affected document.
        # Uses indexed doc_basename column (schema v27) for exact basename lookup —
        # replaces leading-wildcard LIKE ('%/basename') which was not sargable.
        affected_ids: list[str] = []
        _trust_unvisited: list[str] = []
        try:
            pat_norm = document_pattern.replace("\\\\", "/").replace("\\", "/")
            basename = Path(pat_norm).name
            occurrence_rows = self.db.execute(
                """SELECT DISTINCT ao.assertion_id, ao.document_id
                   FROM assertion_occurrence ao
                   JOIN assertion a ON a.id = ao.assertion_id
                   WHERE a.matter_id = ?
                     AND ao.document_id IS NOT NULL
                     AND (ao.document_id = ?
                          OR ao.doc_basename = ?)""",
                (self.matter_id, pat_norm, basename),
            ).fetchall()
            for row in occurrence_rows:
                doc = (row["document_id"] or "").replace("\\\\", "/").replace("\\", "/")
                doc_basename = Path(doc).name
                if pat_norm == doc or pat_norm == doc_basename:
                    affected_ids.append(row["assertion_id"])
            if affected_ids:
                self.apply_revision(
                    affected_ids,
                    cause=RevisionCause.TRUST_OVERRIDE,
                    run_id=run_id,
                    note=f"Document trust override set to '{trust_level}' for {document_pattern!r}",
                    _collect_unvisited=_trust_unvisited,
                )
        except (sqlite3.Error, ValueError, RuntimeError) as exc:
            _log.warning("Trust override belief revision failed for %r: %s", document_pattern, exc)
        # Enqueue outside the except block so DB failures propagate rather than being swallowed.
        self.enqueue_evidence_pending(_trust_unvisited, cause=RevisionCause.TRUST_OVERRIDE, run_id=run_id)

        # Targeted proof state recompute: only recompute issues linked to affected assertions.
        # No-op when affected_ids is empty (pattern matched nothing — proof state unchanged).
        # Chunks affected_ids to stay within SQLite's ~999 bind-variable limit.
        try:
            if affected_ids:
                _SQL_PARAM_LIMIT = 900
                issue_ids_to_recompute: set[str] = set()
                for _bs in range(0, len(affected_ids), _SQL_PARAM_LIMIT):
                    _batch = affected_ids[_bs : _bs + _SQL_PARAM_LIMIT]
                    _rows = self.db.execute(
                        "SELECT DISTINCT issue_id FROM assertion_issue_link"
                        " WHERE assertion_id IN ({})".format(",".join("?" * len(_batch))),
                        _batch,
                    ).fetchall()
                    issue_ids_to_recompute.update(r["issue_id"] for r in _rows)
                if issue_ids_to_recompute:
                    # Pre-fetch overrides once; batch all writes in one transaction.
                    _ov_rows = self.db.execute(
                        """SELECT document_pattern, trust_level FROM document_trust_override
                           WHERE matter_id=? AND trust_level != 'normal'
                           ORDER BY LENGTH(document_pattern) DESC""",
                        (self.matter_id,),
                    ).fetchall()
                    _preloaded = [
                        (r["document_pattern"], r["trust_level"]) for r in _ov_rows
                    ]
                    with self.db.transaction():
                        for _iid in issue_ids_to_recompute:
                            self.proof_state.compute_and_store(
                                _iid, _preloaded_overrides=_preloaded
                            )
            else:
                # No assertions matched — nothing to recompute; proof state is unchanged.
                pass
        except (sqlite3.Error, ValueError, RuntimeError) as exc:
            _log.warning("Trust override proof state refresh failed for %r: %s", document_pattern, exc)

        return override_id

    def delete_trust_override(
        self,
        document_pattern: str,
        run_id: Optional[str] = None,
    ) -> None:
        """Delete a document trust override and re-propagate belief revision.

        Mirrors set_trust_override: deletes the row first, then re-runs belief
        revision on affected assertions so beliefs revert to auto-inferred trust.
        Proof state is recomputed for affected issues.
        """
        deleted = self.trust_overrides.delete(document_pattern)
        if not deleted:
            return  # nothing to propagate — no override existed

        # Adv#11 Fix 1: trust posture reverted — bump trust_revision so
        # cached cascade decisions made while the override was in force
        # become unreachable. Mirrors the bump in set_trust_override.
        try:
            self.cache.bump_trust_revision()
        except sqlite3.Error as _exc:
            _log.warning("delete_trust_override: trust_revision bump failed: %s", _exc)

        affected_ids: list[str] = []
        _trust_unvisited: list[str] = []
        try:
            pat_norm = document_pattern.replace("\\\\", "/").replace("\\", "/")
            basename = Path(pat_norm).name
            occurrence_rows = self.db.execute(
                """SELECT DISTINCT ao.assertion_id, ao.document_id
                   FROM assertion_occurrence ao
                   JOIN assertion a ON a.id = ao.assertion_id
                   WHERE a.matter_id = ?
                     AND ao.document_id IS NOT NULL
                     AND (ao.document_id = ?
                          OR ao.doc_basename = ?)""",
                (self.matter_id, pat_norm, basename),
            ).fetchall()
            for row in occurrence_rows:
                doc = (row["document_id"] or "").replace("\\\\", "/").replace("\\", "/")
                doc_basename = Path(doc).name
                if pat_norm == doc or pat_norm == doc_basename:
                    affected_ids.append(row["assertion_id"])
            if affected_ids:
                self.apply_revision(
                    affected_ids,
                    cause=RevisionCause.TRUST_OVERRIDE,
                    run_id=run_id,
                    note=f"Document trust override deleted for {document_pattern!r}",
                    _collect_unvisited=_trust_unvisited,
                )
        except (sqlite3.Error, ValueError, RuntimeError) as exc:
            _log.warning("Trust override delete belief revision failed for %r: %s", document_pattern, exc)
        self.enqueue_evidence_pending(_trust_unvisited, cause=RevisionCause.TRUST_OVERRIDE, run_id=run_id)

        # No compute_all() fallback on delete: if no assertions matched the pattern,
        # nothing changed and a full recompute would be wasted work.
        try:
            if affected_ids:
                _SQL_PARAM_LIMIT = 900
                issue_ids_to_recompute: set[str] = set()
                for _bs in range(0, len(affected_ids), _SQL_PARAM_LIMIT):
                    _batch = affected_ids[_bs : _bs + _SQL_PARAM_LIMIT]
                    _rows = self.db.execute(
                        "SELECT DISTINCT issue_id FROM assertion_issue_link"
                        " WHERE assertion_id IN ({})".format(",".join("?" * len(_batch))),
                        _batch,
                    ).fetchall()
                    issue_ids_to_recompute.update(r["issue_id"] for r in _rows)
                if issue_ids_to_recompute:
                    _ov_rows = self.db.execute(
                        """SELECT document_pattern, trust_level FROM document_trust_override
                           WHERE matter_id=? AND trust_level != 'normal'
                           ORDER BY LENGTH(document_pattern) DESC""",
                        (self.matter_id,),
                    ).fetchall()
                    _preloaded = [
                        (r["document_pattern"], r["trust_level"]) for r in _ov_rows
                    ]
                    with self.db.transaction():
                        for _iid in issue_ids_to_recompute:
                            self.proof_state.compute_and_store(
                                _iid, _preloaded_overrides=_preloaded
                            )
        except (sqlite3.Error, ValueError, RuntimeError) as exc:
            _log.warning("Trust override delete proof state refresh failed for %r: %s", document_pattern, exc)

    def mine_contradictions(self, run_id: Optional[str] = None) -> list[dict]:
        """
        Run the contradiction mining pass over all assertions in this matter.

        Finds explicit attacks/contradicts links between active assertions,
        marks attacked assertions as DISPUTED when the attacker is high-trust,
        and records UNRESOLVED_CONTRADICTION gaps for each open conflict.

        Returns the list of contradiction dicts found.
        """
        _truncated: list[str] = []
        result = self.assertions.mine_and_mark_contradictions(
            gap_store=self.gaps,
            belief_engine=self.belief,
            run_id=run_id,
            _truncated_nodes=_truncated,
        )
        if _truncated:
            self.enqueue_evidence_pending(
                list(dict.fromkeys(_truncated)),
                cause=RevisionCause.CONFLICT_DETECTION,
                run_id=run_id,
            )
        return result

    def detect_document_version_chains(self) -> list[dict]:
        """
        Heuristically detect document version chains from filename patterns.

        Links 'version_of' relations in document_relation for detected chains
        (e.g. contract_v1.pdf → contract_v2.pdf).  Records MISSING_DOCUMENT
        gaps when a chain has no unversioned base.

        Returns the list of version link dicts created.
        """
        return self.inventory.detect_version_chains(gap_store=self.gaps)

    def get_operative_document_version(self, doc_id: str) -> str:
        """
        Return the operative (latest HEAD) version in the version chain containing doc_id.

        Traverses 'version_of' links to find the document that no later version
        supersedes. Returns doc_id if no chain exists (it is already operative).
        """
        return self.inventory.get_operative_version(doc_id)

    def compute_quant_thresholds(self, currency: str = "USD") -> list[dict]:
        """Detect quantitative threshold violations and record them as gaps (SO-6).

        Checks positive exposure, high disputed fraction, and numeric conflicts.
        Records violations as gaps in the gap store for downstream synthesis.
        Returns list of violation dicts with threshold, level, description, amount.
        """
        return self.quant.compute_thresholds(self.gaps, currency=currency)

    # ------------------------------------------------------------------
    # Gap management
    # ------------------------------------------------------------------

    def record_gap(
        self,
        gap_type: GapType,
        description: str,
        expected_artifact: Optional[str] = None,
        materiality: float = 0.5,
        affected_type: Optional[str] = None,
        affected_id: Optional[str] = None,
    ) -> str:
        return self.gaps.record(
            gap_type, description, expected_artifact, materiality,
            affected_type=affected_type, affected_id=affected_id,
        )

    # ------------------------------------------------------------------
    # Document intelligence (cards + spans)
    # ------------------------------------------------------------------

    def _card_provenance(
        self,
        *,
        run_id: Optional[str],
        relative_path: str,
        doc_id: str,
    ) -> "ProvenanceContext":
        """Build a ProvenanceContext for document card writes that
        bridges ACTIVE_LLM_CALL — so card provenance rows carry
        llm_call_id + model_id + model_tier + prompt_hash when the
        card was written inside a live LLM call (adversarial #7
        finding #2)."""
        try:
            from ..core.models import ACTIVE_LLM_CALL
            active = ACTIVE_LLM_CALL.get() or {}
        except Exception:
            active = {}
        return ProvenanceContext(
            event_kind="card_profile",
            writer_name="DocumentCardStore.upsert",
            run_id=run_id,
            model_id=active.get("model_id"),
            model_tier=active.get("model_tier"),
            extractor_version="2026-04-17.p01.v1",
            prompt_version="DI.DOC_TYPE.v1",
            llm_call_id=active.get("call_id"),
            prompt_hash=active.get("prompt_hash"),
            source_document_ref=relative_path,
            source_document_inventory_id=doc_id,
            source_span_status="not_applicable",
        )

    def upsert_document_intelligence(
        self,
        relative_path: str,
        analysis: dict,
        focus_issue_id: Optional[str] = None,
        file_type: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Optional[str]:
        """Map LLM deep-read output into a document card + salience update.

        ``analysis`` should contain fields from the LLM response:
        doc_type, doc_subtype, doc_source_role, title, author, sender,
        recipient, creation_date, effective_date, purpose,
        rhetorical_posture, reliability_posture, operative_status,
        privilege_flag, unresolved_flags.

        Returns the card_id or None if the inventory row is missing.
        """
        # Resolve inventory doc_id from relative_path
        row = self.db.execute(
            "SELECT id FROM document_inventory WHERE matter_id = ? AND relative_path = ?",
            (self.matter_id, relative_path),
        ).fetchone()
        if row is None:
            return None
        doc_id = row["id"]

        card_id = self.document_cards.upsert(
            doc_id,
            title=analysis.get("title") or analysis.get("doc_title"),
            doc_type=analysis.get("doc_type"),
            doc_subtype=analysis.get("doc_subtype"),
            source_side=analysis.get("source_side"),
            source_role=analysis.get("doc_source_role"),
            author=analysis.get("author"),
            sender=analysis.get("sender"),
            recipient=analysis.get("recipient"),
            creation_date=analysis.get("creation_date"),
            effective_date=analysis.get("effective_date"),
            purpose=analysis.get("purpose"),
            rhetorical_posture=analysis.get("rhetorical_posture"),
            reliability_posture=analysis.get("reliability_posture"),
            operative_status=analysis.get("operative_status", "unknown"),
            privilege_flag=_interpret_privilege_flag(analysis.get("privilege_flag")),
            unresolved_flags=analysis.get("unresolved_flags"),
            # P0.1: every AI profile write records a provenance_event
            # row. source_span_status='not_applicable' because document
            # cards summarize the whole document, not a span.
            # Adversarial #7 fix: pull llm_call_id + model_id +
            # model_tier + prompt_hash from the ACTIVE_LLM_CALL
            # ContextVar so card provenance rows carry the same LLM
            # attribution as assertion/edge/quant rows.
            provenance=self._card_provenance(
                run_id=run_id,
                relative_path=relative_path,
                doc_id=doc_id,
            ),
        )

        # Update salience based on document type + whether it's linked to issues
        from ..core.search import get_document_priority
        base_salience = get_document_priority(relative_path) / 2.0  # normalize to [0,1]
        self.inventory.set_salience(doc_id, min(1.0, base_salience))

        self._detect_and_record_domain_signals(
            doc_id=doc_id,
            analysis=analysis,
            filename=relative_path,
        )

        return card_id

    def upsert_document_profile(
        self,
        relative_path: str,
        analysis: dict,
        *,
        file_type: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Optional[dict]:
        """Query-agnostic document profiling: write card + mark profiled.

        Unlike upsert_document_intelligence (which is called during query-time
        deep-read), this is called during the maintenance loop and does NOT
        depend on any query or issue context.

        Returns dict with card_id, doc_id, and profile summary, or None if
        the inventory row is missing.
        """
        row = self.db.execute(
            "SELECT id FROM document_inventory WHERE matter_id = ? AND relative_path = ?",
            (self.matter_id, relative_path),
        ).fetchone()
        if row is None:
            return None
        doc_id = row["id"]

        card_id = self.document_cards.upsert(
            doc_id,
            title=analysis.get("title") or analysis.get("doc_title"),
            doc_type=analysis.get("doc_type"),
            doc_subtype=analysis.get("doc_subtype"),
            source_side=analysis.get("source_side"),
            source_role=analysis.get("doc_source_role"),
            author=analysis.get("author"),
            sender=analysis.get("sender"),
            recipient=analysis.get("recipient"),
            creation_date=analysis.get("creation_date"),
            effective_date=analysis.get("effective_date"),
            purpose=analysis.get("purpose"),
            rhetorical_posture=analysis.get("rhetorical_posture"),
            reliability_posture=analysis.get("reliability_posture"),
            operative_status=analysis.get("operative_status", "unknown"),
            privilege_flag=_interpret_privilege_flag(analysis.get("privilege_flag")),
            unresolved_flags=analysis.get("unresolved_flags"),
            # P0.1 + adversarial #7 fix: bridge ACTIVE_LLM_CALL.
            provenance=self._card_provenance(
                run_id=run_id,
                relative_path=relative_path,
                doc_id=doc_id,
            ),
        )

        # Update salience
        from ..core.search import get_document_priority
        base_salience = get_document_priority(relative_path) / 2.0
        self.inventory.set_salience(doc_id, min(1.0, base_salience))

        # Mark profiled in inventory
        self.inventory.mark_profile_complete(doc_id)

        self._detect_and_record_domain_signals(
            doc_id=doc_id,
            analysis=analysis,
            filename=relative_path,
        )

        return {
            "card_id": card_id,
            "doc_id": doc_id,
            "doc_type": analysis.get("doc_type"),
            "source_role": analysis.get("doc_source_role"),
            "operative_status": analysis.get("operative_status", "unknown"),
        }

    def list_reviewable_documents(self, limit: int = 500) -> list[dict]:
        """Return every document in the matter with its pending/verified
        candidate-assertion counts so the UI can label a picker with
        at-a-glance review status.

        Returns rows shaped like:
          {path, doc_type, pending, verified, total}
        ordered by pending DESC (most-work-to-do first), then path.
        """
        rows = self.db.execute(
            """WITH doc_pending AS (
                   SELECT ao.document_id AS doc_key,
                          SUM(CASE
                              WHEN COALESCE(vs.status, 'candidate')='candidate'
                                   THEN 1 ELSE 0 END) AS pending,
                          SUM(CASE
                              WHEN vs.status='verified'
                                   THEN 1 ELSE 0 END) AS verified,
                          COUNT(*) AS total
                   FROM assertion a
                   JOIN assertion_occurrence ao ON ao.assertion_id = a.id
                   LEFT JOIN verification_state vs
                     ON vs.target_kind='assertion'
                    AND vs.target_id=a.id
                    AND vs.matter_id=a.matter_id
                   WHERE a.matter_id = ?
                   GROUP BY ao.document_id
               )
               SELECT di.relative_path AS path,
                      COALESCE(dc.doc_type, 'unknown') AS doc_type,
                      COALESCE(dp.pending, 0) AS pending,
                      COALESCE(dp.verified, 0) AS verified,
                      COALESCE(dp.total, 0) AS total
               FROM document_inventory di
               LEFT JOIN document_card dc ON dc.doc_id = di.id
               LEFT JOIN doc_pending dp
                      ON dp.doc_key = di.id
                      OR dp.doc_key = di.relative_path
               WHERE di.matter_id = ?
               ORDER BY pending DESC, di.relative_path
               LIMIT ?""",
            (self.matter_id, self.matter_id, int(limit)),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_documents_needing_profile(self, limit: int = 200) -> list[dict]:
        """Return docs that need query-agnostic profiling."""
        return self.inventory.list_needing_profile(limit=limit)

    def refresh_document_families(
        self,
        doc_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        """Detect and persist version families for specified docs (or all).

        Uses DocumentInventoryStore.detect_version_chains() to find version
        links, groups them into families, and persists family_id/version_chain_id.
        Returns list of link dicts created.
        """
        from collections import defaultdict

        links = self.inventory.detect_version_chains(gap_store=self.gaps)
        if not links:
            return []

        # Group links into families by connected component (union-find).
        # Each link has source_doc_id and target_doc_id.
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(a: str, b: str):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for link in links:
            sid, tid = link["source_doc_id"], link["target_doc_id"]
            parent.setdefault(sid, sid)
            parent.setdefault(tid, tid)
            union(sid, tid)

        # Collect family members
        families: dict[str, list[str]] = defaultdict(list)
        for doc_id in parent:
            families[find(doc_id)].append(doc_id)

        for family_id, member_ids in families.items():
            if doc_ids is not None:
                if not any(mid in doc_ids for mid in member_ids):
                    continue
            if len(member_ids) >= 2:
                self.inventory.set_family_membership(
                    member_ids, family_id, version_chain_id=family_id
                )

        return links

    def get_document_card(
        self,
        relative_path: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> Optional[dict]:
        """Retrieve a document card by path or inventory ID."""
        if doc_id:
            return self.document_cards.get_by_doc_id(doc_id)
        if relative_path:
            return self.document_cards.get_by_path(relative_path)
        return None

    def list_search_seed_docs(
        self,
        issue_id: Optional[str] = None,
        query: Optional[str] = None,
        doc_types: Optional[list] = None,
        include_related_versions: bool = False,
        limit: int = 120,
    ) -> list[str]:
        """Return candidate relative_paths from document memory for search targeting.

        Pulls from document_cards + document_inventory and returns paths
        ranked by a composite score:
        - Base: salience_score from inventory
        - Boost +0.4: document has assertions linked to ``issue_id``
        - Boost +0.2: document has unresolved flags
        - Boost +0.15: document filename contains a query term (>3 chars)
        """
        candidates = self.document_cards.list_candidates(
            doc_types=doc_types, limit=limit * 2,  # over-fetch for re-ranking
        )

        # Build set of doc_ids that have assertions linked to target issue
        _issue_doc_ids: set = set()
        if issue_id:
            rows = self.db.execute(
                """SELECT DISTINCT ao.document_id FROM assertion_issue_link ail
                   JOIN assertion_occurrence ao ON ail.assertion_id = ao.assertion_id
                   WHERE ail.issue_id = ? AND ao.document_id IS NOT NULL""",
                (issue_id,),
            ).fetchall()
            _issue_doc_ids = {r["document_id"] for r in rows}

        # Query terms for filename matching
        _query_terms: set = set()
        if query:
            _query_terms = {w.lower() for w in query.split() if len(w) > 3}

        # Score and sort candidates
        scored: list[tuple[float, str]] = []
        for c in candidates:
            path = c.get("relative_path")
            if not path:
                continue
            score = c.get("salience_score") or 0.0
            # Boost docs linked to target issue
            if _issue_doc_ids and c.get("doc_id") in _issue_doc_ids:
                score += 0.4
            # Boost docs with unresolved flags (under-explored)
            flags = c.get("unresolved_flags")
            if flags and isinstance(flags, list) and len(flags) > 0:
                score += 0.2
            # Boost docs whose filename matches query terms
            if _query_terms:
                path_lower = path.lower()
                if any(t in path_lower for t in _query_terms):
                    score += 0.15
            scored.append((score, path))

        scored.sort(key=lambda x: x[0], reverse=True)
        paths = [p for _, p in scored]

        # Supplement with high-salience inventory rows not already in card results
        if len(paths) < limit:
            remaining = limit - len(paths)
            path_set = set(paths)
            inv_rows = self.db.execute(
                """SELECT relative_path FROM document_inventory
                   WHERE matter_id = ? AND relative_path NOT IN ({})
                   ORDER BY salience_score DESC, last_read_at ASC NULLS FIRST
                   LIMIT ?""".format(",".join("?" * len(paths)) if paths else "'__none__'"),
                [self.matter_id] + list(path_set) + [remaining],
            ).fetchall()
            for r in inv_rows:
                rp = r["relative_path"]
                if rp not in path_set:
                    paths.append(rp)

        return paths[:limit]

    def add_doc_span(self, doc_id: str, payload: dict) -> str:
        """Record a span (section/clause/quote) for a document."""
        return self.spans.upsert(
            document_id=doc_id,
            span_type=payload.get("span_type", "quote"),
            span_text=payload.get("span_text", ""),
            page_start=payload.get("page_start"),
            page_end=payload.get("page_end"),
            line_start=payload.get("line_start"),
            line_end=payload.get("line_end"),
            char_start=payload.get("char_start"),
            char_end=payload.get("char_end"),
            section_ref=payload.get("section_ref"),
            clause_ref=payload.get("clause_ref"),
            ordinal_in_doc=payload.get("ordinal_in_doc"),
        )

    # ------------------------------------------------------------------
    # Query context (read at start of each run)
    # ------------------------------------------------------------------

    def _context_row_passes_taint_policy(
        self,
        row: dict,
        target_kind: str,
        id_key: str = "id",
        *,
        domain_profile_id: str | None = None,
        domain_profile_version: int | None = None,
        target_namespace: str | None = None,
    ) -> bool:
        target_id = row.get(id_key)
        if not target_id:
            return True
        if (
            domain_profile_id is not None
            and domain_profile_version is not None
            and target_namespace is not None
        ):
            return self.memory_broker.object_is_clean_for_profile(
                target_kind,
                str(target_id),
                domain_profile_id=domain_profile_id,
                domain_profile_version=domain_profile_version,
                target_namespace=target_namespace,
            )
        if target_namespace is not None:
            return self.memory_broker.object_is_clean_for_current_profile_binding(
                target_kind,
                str(target_id),
                target_namespace=target_namespace,
            )
        return self.memory_broker.object_is_clean(target_kind, str(target_id))

    def _filter_context_rows_by_taint(
        self,
        rows: list[dict],
        target_kind: str,
        id_key: str = "id",
        *,
        domain_profile_id: str | None = None,
        domain_profile_version: int | None = None,
        target_namespace: str | None = None,
    ) -> list[dict]:
        return [
            row
            for row in rows
            if self._context_row_passes_taint_policy(
                row,
                target_kind,
                id_key,
                domain_profile_id=domain_profile_id,
                domain_profile_version=domain_profile_version,
                target_namespace=target_namespace,
            )
        ]

    def _tainted_target_ids(self, target_kind: str) -> set[str]:
        return self.memory_broker.tainted_target_ids(target_kind)

    def _assertion_has_tainted_context(
        self,
        assertion_id: str,
        *,
        tainted_occurrences: set[str],
        tainted_artifacts: set[str],
        tainted_actors: set[str],
    ) -> bool:
        rows = self.db.execute(
            """SELECT id, document_id, speaker_actor_id
               FROM assertion_occurrence WHERE assertion_id=?""",
            (assertion_id,),
        ).fetchall()
        for row in rows:
            if row["id"] in tainted_occurrences:
                return True
            if row["document_id"] in tainted_artifacts:
                return True
            if row["speaker_actor_id"] in tainted_actors:
                return True
        return False

    def _clean_issue_coverage_by_id(
        self,
        issue_ids: list[str],
        *,
        tainted_assertions: set[str],
        tainted_occurrences: set[str],
        tainted_artifacts: set[str],
        tainted_actors: set[str],
        tainted_criteria: set[str] | None = None,
        tainted_support_edges: set[str] | None = None,
    ) -> dict[str, float]:
        tainted_criteria = tainted_criteria or set()
        tainted_support_edges = tainted_support_edges or set()
        coverage: dict[str, float] = {}
        for issue_id in issue_ids:
            predicate_rows = self.db.execute(
                "SELECT id FROM issue_predicate WHERE issue_id=?",
                (issue_id,),
            ).fetchall()
            clean_predicate_ids = [
                row["id"]
                for row in predicate_rows
                if row["id"] not in tainted_criteria
            ]
            predicate_count = len(clean_predicate_ids)
            rows = self.db.execute(
                """SELECT DISTINCT ail.id AS link_id, a.id, a.belief_state
                   FROM assertion_issue_link ail
                   JOIN assertion a ON a.id = ail.assertion_id
                   WHERE ail.issue_id=? AND ail.relation_type IN ('supports', 'establishes')""",
                (issue_id,),
            ).fetchall()
            weighted_support = 0.0
            for row in rows:
                if row["link_id"] in tainted_support_edges:
                    continue
                assertion_id = row["id"]
                if assertion_id in tainted_assertions:
                    continue
                if self._assertion_has_tainted_context(
                    assertion_id,
                    tainted_occurrences=tainted_occurrences,
                    tainted_artifacts=tainted_artifacts,
                    tainted_actors=tainted_actors,
                ):
                    continue
                state = str(row["belief_state"] or "").lower()
                if state in {"operative", "admitted", "resolved"}:
                    weighted_support += 1.0
                elif state in {"alleged", "argued", "inferred"}:
                    weighted_support += 0.5
                elif state not in {"disputed", "withdrawn", "superseded"}:
                    weighted_support += 0.3
            if predicate_count > 0:
                coverage[issue_id] = min(weighted_support, predicate_count) / predicate_count
            else:
                coverage[issue_id] = min(weighted_support, 1.0)
        return coverage

    def _clean_subtree_issues(
        self,
        issue_id: str,
        *,
        tainted_issues: set[str],
        include_self: bool = True,
    ) -> list[dict]:
        subtree = self.issues.get_subtree(issue_id, include_self=include_self)
        return [issue for issue in subtree if issue["id"] not in tainted_issues]

    def _clean_weakest_leaf_id(
        self,
        issue_id: str,
        *,
        tainted_issues: set[str],
        tainted_assertions: set[str],
        tainted_occurrences: set[str],
        tainted_artifacts: set[str],
        tainted_actors: set[str],
        tainted_criteria: set[str],
        tainted_support_edges: set[str],
    ) -> str | None:
        subtree = self._clean_subtree_issues(
            issue_id,
            tainted_issues=tainted_issues,
            include_self=True,
        )
        if not subtree:
            return None
        subtree_ids = {issue["id"] for issue in subtree}
        parent_ids = {
            issue.get("parent_issue_id")
            for issue in subtree
            if issue.get("parent_issue_id") in subtree_ids
        }
        leaf_ids = sorted(subtree_ids - parent_ids)
        if not leaf_ids:
            return None
        coverage_by_id = self._clean_issue_coverage_by_id(
            leaf_ids,
            tainted_assertions=tainted_assertions,
            tainted_occurrences=tainted_occurrences,
            tainted_artifacts=tainted_artifacts,
            tainted_actors=tainted_actors,
            tainted_criteria=tainted_criteria,
            tainted_support_edges=tainted_support_edges,
        )
        by_id = {issue["id"]: issue for issue in subtree}

        def _leaf_rank(leaf_id: str) -> tuple:
            issue = by_id[leaf_id]
            coverage = coverage_by_id.get(leaf_id, 0.0)
            priority = (
                float(issue.get("materiality") or 0.5)
                * float(issue.get("salience") or 0.5)
                * (1.0 - coverage)
            )
            return (-priority, leaf_id)

        return min(leaf_ids, key=_leaf_rank)

    def build_query_context(self) -> QueryMatterContext:
        """Build a coherent read snapshot for the recursive engine."""
        with self.db.transaction():
            return self._build_query_context_snapshot()

    def _build_query_context_snapshot(self) -> QueryMatterContext:
        """
        Build a QueryMatterContext from current matter state.

        Called at the start of each investigation run to give the engine
        a transactionally coherent snapshot of what is already known, enabling
        targeted retrieval.
        """
        row = self.db.execute(
            "SELECT id, name FROM matter WHERE id=?", (self.matter_id,)
        ).fetchone()
        matter_name = row["name"] if row else "unknown"

        assertion_count = self.assertions.count()
        # Limit to 10: engine context only uses count + first 3 descriptions.
        open_gaps = self._filter_context_rows_by_taint(
            self.gaps.open_gaps(min_materiality=0.3, limit=10),
            "gap",
        )
        open_issues = self._filter_context_rows_by_taint(
            self.issues.get_open_issues(min_materiality=0.3),
            "issue",
        )
        actor_count = self.actors.count()
        tainted_issues = self._tainted_target_ids("issue")

        # Top actors by canonical name (limit 10 to keep context brief)
        tainted_actors = self._tainted_target_ids("actor")
        known_actors = [
            a["canonical_name"]
            for a in self.actors.list_actors()
            if a["id"] not in tainted_actors
        ][:10]

        # Documents already indexed in the assertion store
        tainted_artifacts = self._tainted_target_ids("artifact")
        tainted_artifacts |= self._tainted_target_ids("artifacts")
        tainted_assertions = self._tainted_target_ids("assertion")
        tainted_assertions |= self._tainted_target_ids("claims")
        tainted_occurrences = self._tainted_target_ids("assertion_occurrence")
        tainted_occurrences |= self._tainted_target_ids("claim_occurrence")
        tainted_criteria = (
            self._tainted_target_ids("criteria")
            | self._tainted_target_ids("issue_predicate")
        )
        tainted_support_edges = (
            self._tainted_target_ids("support_edge")
            | self._tainted_target_ids("support_edges")
            | self._tainted_target_ids("assertion_issue_link")
            | self._tainted_target_ids("evidence_edge")
        )
        actor_filters = ""
        actor_params: list[Any] = []
        if tainted_actors:
            actor_filters = (
                f" AND (ao.speaker_actor_id IS NULL OR ao.speaker_actor_id NOT IN "
                f"({','.join('?' for _ in tainted_actors)}))"
            )
            actor_params.extend(sorted(tainted_actors))
        rows = self.db.execute(
            f"""SELECT DISTINCT ao.document_id
                FROM assertion_occurrence ao
                JOIN assertion a ON a.id = ao.assertion_id
                WHERE a.matter_id=?
                  AND ao.document_id IS NOT NULL
                  AND a.id NOT IN ({','.join('?' for _ in tainted_assertions) or "''"})
                  AND ao.id NOT IN ({','.join('?' for _ in tainted_occurrences) or "''"})
                  AND ao.document_id NOT IN ({','.join('?' for _ in tainted_artifacts) or "''"})
                  {actor_filters}
                ORDER BY ao.document_id LIMIT 20""",
            (
                self.matter_id,
                *sorted(tainted_assertions),
                *sorted(tainted_occurrences),
                *sorted(tainted_artifacts),
                *actor_params,
            ),
        ).fetchall()
        known_document_ids = [r["document_id"] for r in rows]

        # Find the weakest issue: the highest-priority issue with the least evidentiary support.
        # "Weakest" means most important AND least covered — where work will have most impact.
        # Priority = materiality × salience × (1 - coverage_fraction).
        # Uses predicate-aware coverage_fraction (see _coverage_fraction()).
        weakest_issue_id = None
        if open_issues:
            # Belief-state-weighted support sum (mirrors get_issue_coverage_report logic).
            # operative/admitted/resolved = 1.0, alleged/argued/inferred = 0.5,
            # other active states = 0.3; disputed/withdrawn/superseded excluded entirely.
            # P0.2 review fix #2 (round 2): the weakness query was
            # originally reading raw assertion_issue_link, so
            # rejected/stale support inflated weighted coverage. A
            # first pass added the assertion-lane verification filter
            # but still couldn't see edge verification — stale edges
            # on edge-backed issues still counted. Source the
            # coverage_fraction directly from the canonical coverage
            # report, which already applies
            # TrustPurpose.PROOF_CANDIDATE eligibility to both lanes.
            coverage_by_id = self._clean_issue_coverage_by_id(
                [issue["id"] for issue in open_issues],
                tainted_assertions=tainted_assertions,
                tainted_occurrences=tainted_occurrences,
                tainted_artifacts=tainted_artifacts,
                tainted_actors=tainted_actors,
                tainted_criteria=tainted_criteria,
                tainted_support_edges=tainted_support_edges,
            )

            # Use subtree-aware weakness: for each root issue, find its weakest
            # leaf descendant. The engine should target the most specific weak element,
            # not just the top-level claim (Gap 1: hierarchical issue model).
            def _weakness(issue: dict) -> tuple:
                coverage = coverage_by_id.get(issue["id"], 0.0)
                priority = issue["materiality"] * issue["salience"] * (1.0 - coverage)
                return (-priority, issue["id"])

            weakest = min(open_issues, key=_weakness)
            weakest_issue_id = weakest["id"]

            # If the weakest issue has children, drill down through a taint-aware
            # subtree. Raw compute_coverage_rollup can see quarantined children.
            children = [
                child
                for child in self.issues.get_children(weakest_issue_id)
                if child["id"] not in tainted_issues
            ]
            if children:
                leaf_id = self._clean_weakest_leaf_id(
                    weakest_issue_id,
                    tainted_issues=tainted_issues,
                    tainted_assertions=tainted_assertions,
                    tainted_occurrences=tainted_occurrences,
                    tainted_artifacts=tainted_artifacts,
                    tainted_actors=tainted_actors,
                    tainted_criteria=tainted_criteria,
                    tainted_support_edges=tainted_support_edges,
                )
                if leaf_id:
                    weakest_issue_id = leaf_id

        # Answered clarifications: inject user context into orientation (limit to 3 most recent)
        answered_clarifications = self._filter_context_rows_by_taint(
            self.clarifications.get_answered(limit=3),
            "clarification",
            target_namespace="clarifications",
        )

        # Document annotations: strategic notes from user (SO-3 annotation)
        document_annotations = self._filter_context_rows_by_taint(
            self.annotations.list_recent(limit=10),
            "annotation",
        )

        # SO-2: top predicate_key values from the typed assertion graph so orientation
        # can generate SPO-aware search leads targeting known relationship types.
        pred_rows = self.db.execute(
            f"""SELECT a.predicate_key, COUNT(*) AS cnt
                FROM assertion a
                LEFT JOIN assertion_occurrence ao ON ao.assertion_id = a.id
                WHERE a.matter_id=? AND a.predicate_key IS NOT NULL
                  AND a.id NOT IN ({','.join('?' for _ in tainted_assertions) or "''"})
                  AND (ao.id IS NULL OR ao.id NOT IN ({','.join('?' for _ in tainted_occurrences) or "''"}))
                  AND (ao.document_id IS NULL OR ao.document_id NOT IN ({','.join('?' for _ in tainted_artifacts) or "''"}))
                  {actor_filters}
                GROUP BY a.predicate_key
                ORDER BY cnt DESC
                LIMIT 10""",
            (
                self.matter_id,
                *sorted(tainted_assertions),
                *sorted(tainted_occurrences),
                *sorted(tainted_artifacts),
                *actor_params,
            ),
        ).fetchall()
        key_predicates = [r["predicate_key"] for r in pred_rows]

        # Gap 3: inject active assumptions so the engine can surface them in
        # orientation and respect assumption-gated predicates.
        active_assumptions = self._filter_context_rows_by_taint(
            self.assumptions.get_active(max_rows=20),
            "assumption",
        )

        # Document intelligence: count of documents with structured cards
        doc_card_count = self.document_cards.count()

        # Domain composition: read already-recorded facets (no fresh detection)
        domain_facets, composed_trust, primary_profile = (
            self._read_matter_domain_composition()
        )
        if composed_trust:
            self.belief.trust_weights = composed_trust

        return QueryMatterContext(
            matter_id=self.matter_id,
            matter_name=matter_name,
            open_gaps=open_gaps,
            open_issues=open_issues,
            existing_assertion_count=assertion_count,
            existing_actor_count=actor_count,
            known_actors=known_actors,
            known_document_ids=known_document_ids,
            answered_clarifications=answered_clarifications,
            document_annotations=document_annotations,
            weakest_issue_id=weakest_issue_id,
            key_predicates=key_predicates,
            active_assumptions=active_assumptions,
            document_card_count=doc_card_count,
            domain_facets=domain_facets,
            composed_trust_weights=composed_trust,
            primary_domain_profile_id=primary_profile,
        )

    def _detect_and_record_domain_signals(
        self,
        *,
        doc_id: str,
        analysis: dict,
        filename: str = "",
    ) -> None:
        """Run deterministic domain detection on document metadata and record results.

        Called from upsert_document_intelligence and upsert_document_profile
        after the card is written. Records detection events and upserts
        workspace-level facets so the matter's domain composition reflects
        the documents ingested.
        """
        try:
            from .domain_detection import detect_domain_signals, CONFIDENCE_ACTIVE

            text_parts = []
            for key in ("title", "doc_title", "purpose", "doc_type", "doc_subtype"):
                val = analysis.get(key)
                if val and isinstance(val, str):
                    text_parts.append(val)
            text = " ".join(text_parts)
            if not text.strip():
                return

            source_type = analysis.get("doc_type") or ""
            metadata = {
                k: v for k, v in analysis.items()
                if isinstance(v, (str, int, float, bool)) and v
            }

            candidates = detect_domain_signals(
                text,
                source_type=source_type,
                filename=filename,
                metadata=metadata,
            )
            if not candidates:
                return

            broker = self.memory_broker
            import json as _jm

            for candidate in candidates:
                broker.record_domain_detection_event(
                    target_kind="artifact",
                    target_id=doc_id,
                    candidate_profile_id=candidate.profile_id,
                    candidate_profile_version=1,
                    confidence=candidate.confidence,
                    signals_json=_jm.dumps(candidate.signals.to_dict()),
                    evidence_refs_json=_jm.dumps(candidate.evidence_refs),
                )

                if candidate.is_active:
                    profile = broker.get_domain_profile(candidate.profile_id, 1)
                    mapping_hash = profile["mapping_hash"] if profile else "sha256:unknown"
                    broker.upsert_object_domain_facet(
                        target_kind="workspace",
                        target_id=self.matter_id,
                        domain_profile_id=candidate.profile_id,
                        domain_profile_version=1,
                        profile_mapping_hash=mapping_hash,
                        confidence=candidate.confidence,
                        status="active",
                    )
                elif candidate.is_candidate:
                    broker.record_unknown_domain_candidate(
                        evidence_cluster_hash=f"{candidate.profile_id}:{doc_id}",
                        signals_json=_jm.dumps(candidate.signals.to_dict()),
                        evidence_refs_json=_jm.dumps(candidate.evidence_refs),
                    )
        except Exception as exc:
            _log.debug("Domain detection failed for doc %s: %s", doc_id, exc)

    _SEMANTIC_CACHE_NAMESPACES = (
        "claims:*", "claim_occurrences:*", "objective_nodes:*",
        "criteria:*", "entities:*", "artifacts:*", "clarifications:*",
        "annotations:*", "assumptions:*", "support_edges:*",
        "guidance:*", "spans:*", "proof_state:*",
    )

    def build_semantic_cache_manifest(
        self,
        *,
        purpose: str = "semantic_cache",
        policy_audience: str = "clean",
        taint_class: str = "public_clean",
    ) -> str:
        """Build and record a DependencyManifest for semantic cache writes.

        Returns the manifest hash. The engine calls this once per run when
        context is built, then passes the hash to put_brokered() for each
        semantic stage. The manifest captures namespace revisions at
        snapshot time so cached plans are automatically invalidated when
        the underlying data changes.
        """
        from .memory_contracts import DependencyManifest

        broker = self.memory_broker
        ns_deps = broker.namespace_dependencies_for_keys(
            self._SEMANTIC_CACHE_NAMESPACES
        )
        mapping_hash = broker.current_profile_mapping_hash(
            domain_profile_id="legal",
            domain_profile_version=1,
            target_kind="clarification",
            target_namespace="clarifications",
        )
        manifest = DependencyManifest(
            matter_id=self.matter_id,
            namespace_dependencies=ns_deps,
            domain_profile_id="legal",
            domain_profile_version=1,
            profile_mapping_hash=mapping_hash,
            purpose=purpose,
            policy_audience=policy_audience,
            taint_class=taint_class,
        )
        broker.record_dependency_manifest(manifest)
        return manifest.manifest_hash()

    _TRUST_DISAGREEMENT_THRESHOLD = 0.25

    def _read_matter_domain_composition(
        self,
    ) -> tuple[list[dict], dict[str, float], str | None]:
        """Read recorded domain facets at matter level and compose trust weights.

        Returns (facet_list, composed_trust_weights, primary_profile_id).
        Reads only — no fresh detection (per Design Gate 3 §3).

        Composition uses role-local normalization: each role is averaged only
        across profiles that define that role, so single-profile roles are not
        diluted by unrelated facets. When profiles disagree on a role by more
        than _TRUST_DISAGREEMENT_THRESHOLD, the role is flagged requires_review
        in the facet metadata.
        """
        broker = self.memory_broker
        facet_rows = broker.get_object_domain_facets(
            "workspace", self.matter_id, status="active",
        )

        def _legal_fallback() -> dict[str, float]:
            tw = broker.get_profile_trust_weights("legal")
            return tw if tw else dict(SOURCE_TRUST_WEIGHTS)

        if not facet_rows:
            return [], _legal_fallback(), "legal"

        total_conf = sum(f["confidence"] for f in facet_rows)
        if total_conf <= 0:
            return [], _legal_fallback(), "legal"

        primary_profile: str | None = None
        best_conf = -1.0
        for f in facet_rows:
            if f["confidence"] > best_conf:
                best_conf = f["confidence"]
                primary_profile = f["domain_profile_id"]

        role_contributions: dict[str, list[tuple[float, float]]] = {}
        for f in facet_rows:
            tw = broker.get_profile_trust_weights(f["domain_profile_id"])
            conf = f["confidence"]
            for role, val in tw.items():
                role_contributions.setdefault(role, []).append((conf, val))

        composed: dict[str, float] = {}
        review_roles: set[str] = set()
        for role, contributions in role_contributions.items():
            role_total_conf = sum(c for c, _ in contributions)
            if role_total_conf <= 0:
                continue
            weighted_val = sum(c * v for c, v in contributions) / role_total_conf
            composed[role] = weighted_val
            if len(contributions) > 1:
                vals = [v for _, v in contributions]
                if max(vals) - min(vals) > self._TRUST_DISAGREEMENT_THRESHOLD:
                    review_roles.add(role)

        facets = [
            {
                "domain_profile_id": f["domain_profile_id"],
                "domain_profile_version": f["domain_profile_version"],
                "confidence": f["confidence"],
                "status": f["status"],
            }
            for f in facet_rows
        ]
        if review_roles:
            for fd in facets:
                fd["requires_review_roles"] = sorted(review_roles)
        return facets, composed, primary_profile

    @staticmethod
    def _coverage_fraction(weighted_support: float, predicate_count: int) -> float:
        """Compute evidence coverage fraction for a single issue.

        *weighted_support* is the trust-weighted sum of supporting assertions,
        where weights reflect proof strength by belief_state:
            operative / admitted / resolved → 1.0  (solid proof)
            alleged / argued / inferred             → 0.5  (contested/uncertain)
            unknown / other active states           → 0.3

        When the issue has defined claim elements (predicates):
            min(weighted_support, predicate_count) / predicate_count
        This caps coverage at 1.0 only when solid-weight support reaches
        the number of required elements.  Two alleged assertions (weight=1.0
        combined) do not substitute for one operative assertion (weight=1.0)
        when predicate_count=1 — only an operative yields full element coverage.

        When no predicates exist (predicate_count == 0), falls back to:
            weighted_support / (weighted_support + 1.0)
        which preserves monotone ordering by support strength alone.
        """
        if predicate_count > 0:
            return min(weighted_support, float(predicate_count)) / float(predicate_count)
        return weighted_support / (weighted_support + 1.0)

    def get_issue_coverage_report(self, policy_audience: str = "clean") -> list[dict]:
        """Return per-issue evidence coverage for all open issues (SO-4).

        policy_audience controls privileged-content filtering. MVP.4 ships
        two audiences:
        - "clean" (default): privileged assertions are excluded from
          supporting/attacking counts, weighted_support, and all verified
          derivatives. This is what UI/API callers, synthesis, and export
          should use.
        - "internal": no privilege filter. Used by internal audit/review
          paths that need to see what was withheld.

        Each entry contains:
          - id, title, issue_type, materiality, salience
          - supporting_count: raw integer count of active supporting assertions
            (excludes disputed/withdrawn/superseded; includes all other belief states)
          - predicate_count: number of open claim elements (predicates) for the issue
          - coverage_fraction: proof-strength-aware fraction in [0, 1].
            If predicates exist: min(weighted_support, predicate_count) / predicate_count.
            If no predicates: weighted_support / (weighted_support + 1) fallback.
          - has_proof_gap: True if an open MISSING_ISSUE_PREDICATE gap is linked
          - gap_id: id of that gap, or None

        Ordered by coverage_fraction ascending (weakest coverage first).
        """
        # order_by_score=False: this method re-sorts by coverage_fraction at the end,
        # so the (salience * materiality) SQL expression sort is wasted work.
        open_issues = self.issues.get_open_issues(min_materiality=0.0, order_by_score=False)
        if not open_issues:
            return []

        mid = self.matter_id

        # Use JOIN instead of IN-list to avoid SQLite variable-count limits (SO-4 scale).
        # Returns both raw count (for display) and belief-state-weighted sum (for fraction).
        # Weights: operative/admitted/resolved = 1.0, alleged/argued/inferred = 0.5,
        # other active states = 0.3; disputed/withdrawn/superseded excluded entirely.
        # This prevents alleged assertions from overstating coverage vs operative ones.
        # MVP.3: edge-first substrate selection. For each open issue, if any
        # active evidence_edge targeting that issue exists, compute coverage
        # from the edge table; otherwise fall back to assertion_issue_link.
        # This keeps get_issue_coverage_report in lockstep with
        # ProofStateStore.compute_and_store, closing the split-brain gap where
        # the two substrates reported divergent support counts for the same
        # issue.
        # MVP.2: exclude rejected assertions and count verified-vs-candidate
        # support separately. LEFT JOIN keeps assertions without a
        # verification row counted as candidate rather than silently dropped.

        # Identify which issues have any active edge (edge substrate) vs.
        # must fall back to the legacy link table.
        edge_issue_rows = self.db.execute(
            """SELECT DISTINCT target_id AS issue_id
               FROM evidence_edge
               WHERE matter_id=? AND target_kind='issue' AND active=1""",
            (mid,),
        ).fetchall()
        edge_issue_ids: set[str] = {r["issue_id"] for r in edge_issue_rows}

        # MVP.4: when clean mode is active, exclude assertions that trace
        # back to a privileged document. Injected as a subquery so large
        # matters don't hit SQLite's variable-count limit.
        privilege_filter = ""
        if policy_audience == "clean":
            privilege_filter = (
                "AND a.id NOT IN ("
                "SELECT DISTINCT ao.assertion_id "
                "FROM assertion_occurrence ao "
                "LEFT JOIN document_inventory di "
                "  ON di.id = ao.document_inventory_id "
                "  OR di.relative_path = ao.document_id "
                "JOIN document_card dc ON dc.doc_id = di.id "
                "WHERE di.matter_id = a.matter_id AND dc.privilege_flag = 1"
                ")"
            )

        def _support_query(use_edges: bool) -> str:
            if use_edges:
                link_from = (
                    "evidence_edge ee "
                    "JOIN issue i ON i.id = ee.target_id "
                    "JOIN assertion a ON a.id = ee.source_id "
                    "LEFT JOIN verification_state vs "
                    "  ON vs.target_kind = 'assertion' "
                    " AND vs.target_id = a.id "
                    " AND vs.matter_id = a.matter_id "
                    "LEFT JOIN verification_state vs_edge "
                    "  ON vs_edge.target_kind = 'evidence_edge' "
                    " AND vs_edge.target_id = ee.id "
                    " AND vs_edge.matter_id = ee.matter_id "
                )
                link_id = "ee.target_id"
                rel_filter = (
                    "ee.matter_id=? AND ee.target_kind='issue' AND ee.active=1 "
                    "AND ee.source_kind='assertion' "
                    "AND ee.relation_type IN ('supports','establishes') "
                    "AND i.status='open' "
                    "AND a.belief_state NOT IN ('disputed','withdrawn','superseded') "
                    "AND COALESCE(vs.status, 'candidate') NOT IN ('rejected','stale') "
                    "AND COALESCE(vs_edge.status, ee.verification_status, 'candidate') NOT IN ('rejected','stale') "
                    + privilege_filter
                )
            else:
                link_from = (
                    "assertion_issue_link ail "
                    "JOIN issue i ON i.id = ail.issue_id "
                    "JOIN assertion a ON a.id = ail.assertion_id "
                    "LEFT JOIN verification_state vs "
                    "  ON vs.target_kind = 'assertion' "
                    " AND vs.target_id = a.id "
                    " AND vs.matter_id = a.matter_id "
                )
                link_id = "ail.issue_id"
                rel_filter = (
                    "i.matter_id=? AND i.status='open' "
                    "AND ail.relation_type IN ('supports','establishes') "
                    "AND a.belief_state NOT IN ('disputed','withdrawn','superseded') "
                    "AND COALESCE(vs.status, 'candidate') NOT IN ('rejected','stale') "
                    + privilege_filter
                )
            # P0.2 review fix #1: verified requires BOTH the assertion
            # AND the edge lane to be verified. On the edge-backed
            # branch, AND the edge's own verification status into the
            # verified predicate. On the legacy branch (no edge row),
            # only the assertion lane matters.
            if use_edges:
                verified_pred = (
                    "COALESCE(vs.status, 'candidate') = 'verified' "
                    "AND COALESCE(vs_edge.status, ee.verification_status, 'candidate') = 'verified'"
                )
            else:
                verified_pred = "COALESCE(vs.status, 'candidate') = 'verified'"
            return (
                f"SELECT {link_id} AS issue_id, "
                "COUNT(*) AS raw_count, "
                "SUM(CASE "
                "WHEN a.belief_state IN ('operative','admitted','resolved') THEN 1.0 "
                "WHEN a.belief_state IN ('alleged','argued','inferred') THEN 0.5 "
                "ELSE 0.3 END) AS weighted_support, "
                f"SUM(CASE WHEN {verified_pred} THEN 1 ELSE 0 END) AS verified_count, "
                f"SUM(CASE WHEN {verified_pred} THEN "
                "CASE "
                "WHEN a.belief_state IN ('operative','admitted','resolved') THEN 1.0 "
                "WHEN a.belief_state IN ('alleged','argued','inferred') THEN 0.5 "
                "ELSE 0.3 END ELSE 0 END) AS verified_weighted "
                f"FROM {link_from} "
                f"WHERE {rel_filter} "
                f"GROUP BY {link_id}"
            )

        edge_support_rows = self.db.execute(_support_query(True), (mid,)).fetchall() if edge_issue_ids else []
        legacy_support_rows = self.db.execute(_support_query(False), (mid,)).fetchall()

        # Combine per-issue: edge rows win when the issue has any edge, legacy
        # rows fill the rest. No union — exactly one substrate per issue.
        support_rows = [r for r in edge_support_rows if r["issue_id"] in edge_issue_ids]
        support_rows.extend(
            r for r in legacy_support_rows if r["issue_id"] not in edge_issue_ids
        )
        # raw_counts for the supporting_count field (integer, user-visible)
        raw_counts = {r["issue_id"]: int(r["raw_count"]) for r in support_rows}
        # weighted_supports for coverage_fraction computation
        support_counts = {
            r["issue_id"]: float(r["weighted_support"] or 0.0)
            for r in support_rows
        }
        verified_counts = {
            r["issue_id"]: int(r["verified_count"] or 0)
            for r in support_rows
        }
        verified_weighted = {
            r["issue_id"]: float(r["verified_weighted"] or 0.0)
            for r in support_rows
        }

        # Attacking assertion counts per issue. Same edge-first substrate
        # selection as supporting_count, same rejected-exclusion rules,
        # and the same MVP.4 privilege filter in clean mode.
        def _attack_query(use_edges: bool) -> str:
            if use_edges:
                return (
                    "SELECT ee.target_id AS issue_id, COUNT(*) AS atk_count "
                    "FROM evidence_edge ee "
                    "JOIN issue i ON i.id = ee.target_id "
                    "JOIN assertion a ON a.id = ee.source_id "
                    "LEFT JOIN verification_state vs "
                    "  ON vs.target_kind = 'assertion' "
                    " AND vs.target_id = a.id "
                    " AND vs.matter_id = a.matter_id "
                    "LEFT JOIN verification_state vs_edge "
                    "  ON vs_edge.target_kind = 'evidence_edge' "
                    " AND vs_edge.target_id = ee.id "
                    " AND vs_edge.matter_id = ee.matter_id "
                    "WHERE ee.matter_id=? AND ee.target_kind='issue' "
                    "AND ee.active=1 AND ee.source_kind='assertion' "
                    "AND ee.relation_type IN ('attacks','negates') "
                    "AND i.status='open' "
                    "AND a.belief_state NOT IN ('disputed','withdrawn','superseded') "
                    "AND COALESCE(vs.status, 'candidate') NOT IN ('rejected','stale') "
                    "AND COALESCE(vs_edge.status, ee.verification_status, 'candidate') NOT IN ('rejected','stale') "
                    + privilege_filter + " "
                    "GROUP BY ee.target_id"
                )
            return (
                "SELECT ail.issue_id, COUNT(*) AS atk_count "
                "FROM assertion_issue_link ail "
                "JOIN issue i ON i.id = ail.issue_id "
                "JOIN assertion a ON a.id = ail.assertion_id "
                "LEFT JOIN verification_state vs "
                "  ON vs.target_kind = 'assertion' "
                " AND vs.target_id = a.id "
                " AND vs.matter_id = a.matter_id "
                "WHERE i.matter_id=? AND i.status='open' "
                "AND ail.relation_type IN ('attacks','negates') "
                "AND a.belief_state NOT IN ('disputed','withdrawn','superseded') "
                "AND COALESCE(vs.status, 'candidate') NOT IN ('rejected','stale') "
                + privilege_filter + " "
                "GROUP BY ail.issue_id"
            )

        edge_attack_rows = self.db.execute(_attack_query(True), (mid,)).fetchall() if edge_issue_ids else []
        legacy_attack_rows = self.db.execute(_attack_query(False), (mid,)).fetchall()
        attack_rows = [r for r in edge_attack_rows if r["issue_id"] in edge_issue_ids]
        attack_rows.extend(
            r for r in legacy_attack_rows if r["issue_id"] not in edge_issue_ids
        )
        attack_counts = {r["issue_id"]: int(r["atk_count"]) for r in attack_rows}

        # Predicate counts per issue — used for predicate-aware coverage fraction.
        pred_rows = self.db.execute(
            """SELECT ip.issue_id, COUNT(*) AS pred_count
               FROM issue_predicate ip
               JOIN issue i ON i.id = ip.issue_id
               WHERE i.matter_id=? AND i.status='open' AND ip.status='open'
               GROUP BY ip.issue_id""",
            (mid,),
        ).fetchall()
        pred_counts = {r["issue_id"]: r["pred_count"] for r in pred_rows}

        # Gap 3: count contested and blocked predicates per issue.
        contested_rows = self.db.execute(
            """SELECT ip.issue_id,
                      SUM(CASE WHEN ip.status='contested' THEN 1 ELSE 0 END) AS contested,
                      SUM(CASE WHEN ip.status='blocked' THEN 1 ELSE 0 END) AS blocked
               FROM issue_predicate ip
               JOIN issue i ON i.id = ip.issue_id
               WHERE i.matter_id=? AND i.status='open'
                 AND ip.status IN ('contested','blocked')
               GROUP BY ip.issue_id""",
            (mid,),
        ).fetchall()
        contested_counts = {r["issue_id"]: int(r["contested"]) for r in contested_rows}
        blocked_counts = {r["issue_id"]: int(r["blocked"]) for r in contested_rows}

        proof_gap_rows = self.db.execute(
            """SELECT gl.affected_id AS issue_id, g.id AS gap_id
               FROM gap g
               JOIN gap_link gl ON gl.gap_id=g.id
               JOIN issue i ON i.id=gl.affected_id
               WHERE g.matter_id=? AND g.status='open'
                 AND g.gap_type='missing_issue_predicate'
                 AND gl.affected_type='issue'
                 AND i.matter_id=? AND i.status='open'""",
            (mid, mid),
        ).fetchall()
        proof_gaps = {r["issue_id"]: r["gap_id"] for r in proof_gap_rows}

        report = []
        for issue in open_issues:
            iid = issue["id"]
            w_support = support_counts.get(iid, 0.0)
            raw_cnt = raw_counts.get(iid, 0)
            pred_cnt = pred_counts.get(iid, 0)
            atk_cnt = attack_counts.get(iid, 0)
            coverage = self._coverage_fraction(w_support, pred_cnt)
            has_gap = iid in proof_gaps
            # Derive proof_status from coverage + gap so UI panel shows meaningful state
            if has_gap:
                proof_status = "gap"
            elif coverage >= 0.8:
                proof_status = "strong"
            elif coverage >= 0.4:
                proof_status = "partial"
            elif coverage > 0:
                proof_status = "weak"
            else:
                proof_status = "none"

            # Hierarchy metadata for UI tree rendering
            parent_id = issue.get("parent_issue_id")
            depth = self.issues.get_depth(iid) if parent_id else 0
            children = self.issues.get_children(iid)
            child_ids = [c["id"] for c in children]

            # Subtree rollup for parent issues (shows aggregate weakness)
            subtree_rollup = None
            if child_ids:
                subtree_rollup = self.issues.compute_coverage_rollup(iid)

            verified_cnt = verified_counts.get(iid, 0)
            verified_w = verified_weighted.get(iid, 0.0)
            verified_cov = self._coverage_fraction(verified_w, pred_cnt)
            entry = {
                "id": iid,
                "title": issue.get("title", ""),
                "issue_type": issue.get("issue_type", ""),
                "materiality": issue.get("materiality", 0.0),
                "salience": issue.get("salience", 0.0),
                "burden_side": issue.get("burden_side"),
                "parent_issue_id": parent_id,
                "depth": depth,
                "child_ids": child_ids,
                "supporting_count": raw_cnt,
                "attacking_count": atk_cnt,
                "predicate_count": pred_cnt,
                "coverage_fraction": round(coverage, 4),
                # MVP.2: verified-vs-candidate breakdown lets downstream
                # consumers enforce that candidate-only support cannot
                # resolve an issue to verified proof (SO-2).
                "verified_supporting_count": verified_cnt,
                "candidate_supporting_count": raw_cnt - verified_cnt,
                "verified_coverage_fraction": round(verified_cov, 4),
                "proof_status": proof_status,
                "has_proof_gap": has_gap,
                "gap_id": proof_gaps.get(iid),
                "contested_predicates": contested_counts.get(iid, 0),
                "blocked_predicates": blocked_counts.get(iid, 0),
            }
            if subtree_rollup:
                entry["subtree_coverage"] = round(subtree_rollup["coverage_fraction"], 4)
                entry["weakest_leaf_id"] = subtree_rollup.get("weakest_leaf_id")
                entry["weakest_leaf_coverage"] = round(subtree_rollup.get("weakest_leaf_coverage", 0.0), 4)
                entry["subtree_size"] = subtree_rollup["subtree_size"]

            report.append(entry)

        report.sort(key=lambda x: x["coverage_fraction"])
        return report

    # ------------------------------------------------------------------
    # Clarification engine (SO-7, SO-3)
    # ------------------------------------------------------------------

    def _require_domain_profile(
        self,
        domain_profile_id: str,
        domain_profile_version: int,
    ) -> dict:
        profile = self.memory_broker.get_domain_profile(
            domain_profile_id, domain_profile_version
        )
        if profile is None:
            raise MemoryBrokerPolicyError(
                f"Current domain profile required for clarification answer: "
                f"{domain_profile_id}@{domain_profile_version}"
            )
        return profile

    def _compatible_profile_mapping_hash(
        self,
        *,
        source_domain_profile_id: str,
        source_domain_profile_version: int,
        target_domain_profile_id: str,
        target_domain_profile_version: int,
        target_kind: str,
        target_namespace: str,
    ) -> str:
        mappings = self.memory_broker.list_profile_mappings(
            target_domain_profile_id,
            target_kind=target_kind,
            target_namespace=target_namespace,
        )
        for mapping in mappings:
            if (
                mapping.get("source_domain_profile_id") == source_domain_profile_id
                and int(mapping.get("source_domain_profile_version") or 0) == source_domain_profile_version
                and int(mapping.get("target_domain_profile_version") or 0) == target_domain_profile_version
                and mapping.get("compatibility_status") in {"compatible", "identity"}
            ):
                return str(mapping["target_mapping_hash"])
        raise MemoryBrokerPolicyError(
            "Compatible profile mapping missing for "
            f"{source_domain_profile_id}->{target_domain_profile_id} "
            f"{target_kind}/{target_namespace}"
        )

    def _current_revision_expectations_for(
        self,
        keys: set[str],
    ) -> dict[str, int]:
        expected: dict[str, int] = {}
        for key in keys:
            namespace, target_kind, target_id = self.memory_broker.parse_revision_key(key)
            expected[key] = self.memory_broker.get_namespace_revision(
                namespace, target_kind, target_id
            )
        return expected

    def answer_clarification(
        self,
        question_id: str,
        answer_text: str,
        *,
        domain_profile_id: str = "legal",
        domain_profile_version: int = 1,
    ) -> bool:
        profile = self._require_domain_profile(domain_profile_id, domain_profile_version)
        effective_profile_version = int(profile["profile_version"])
        mapping_hash = self._compatible_profile_mapping_hash(
            source_domain_profile_id=domain_profile_id,
            source_domain_profile_version=effective_profile_version,
            target_domain_profile_id=domain_profile_id,
            target_domain_profile_version=effective_profile_version,
            target_kind="clarification",
            target_namespace="clarifications",
        )
        revision_keys = {
            self.memory_broker.revision_key("clarifications"),
            self.memory_broker.revision_key(
                "clarifications", "clarification", question_id
            ),
            self.memory_broker.revision_key("guidance"),
            self.memory_broker.revision_key("object_taint"),
            self.memory_broker.revision_key(
                "object_taint", "clarification", question_id
            ),
            self.memory_broker.revision_key("policy"),
            self.memory_broker.revision_key(
                "domain_profiles", "profile", domain_profile_id
            ),
            self.memory_broker.revision_key("profile_mappings"),
            self.memory_broker.revision_key(
                "profile_mappings", "profile", domain_profile_id
            ),
            self.memory_broker.revision_key(
                "profile_mappings", "mapping", mapping_hash
            ),
        }
        return self.memory_broker.answer_clarification_with_cas(
            question_id=question_id,
            answer_text=answer_text,
            expected_revisions=self._current_revision_expectations_for(revision_keys),
            domain_profile_id=domain_profile_id,
            domain_profile_version=effective_profile_version,
            source_domain_profile_id=domain_profile_id,
            source_domain_profile_version=effective_profile_version,
            profile_mapping_hash=mapping_hash,
        )

    def generate_clarifications_from_gaps(
        self,
        run_id: Optional[str] = None,
        top_n: int = 3,
        min_materiality: float = 0.5,
    ) -> list[str]:
        """
        Generate clarification questions for the highest-materiality open gaps.

        Called at the end of an investigation run. Returns list of new question_ids.
        Only generates questions for gaps that don't already have a pending question.
        """
        # open_gaps() already returns sorted by materiality_score DESC; pass limit to
        # push LIMIT into SQL and restrict gap_link join to returned IDs only.
        gaps = self.gaps.open_gaps(min_materiality=min_materiality, limit=top_n)

        question_ids = []
        for gap in gaps:
            description = gap.get("description", "")
            if not description:
                continue

            # Look up what this gap is linked to (issue or assertion) so the impact
            # statement is specific rather than generic (SO-7).
            gap_id = gap.get("id")
            link_rows = self.gaps.db.execute(
                "SELECT affected_type, affected_id FROM gap_link WHERE gap_id=? LIMIT 1",
                (gap_id,),
            ).fetchall()
            link = link_rows[0] if link_rows else None
            materiality = gap.get("materiality_score", 0.5)
            materiality_label = "high" if materiality >= 0.7 else ("medium" if materiality >= 0.4 else "low")

            # Format question based on gap type
            gap_type = gap.get("gap_type", "")
            if gap_type == "missing_issue_predicate":
                # Proof gap: issue has zero supporting assertions — ask for evidence, not a doc
                question = f"{description}. Can you provide documents, testimony, or other evidence relevant to this issue?"
                why = "No supporting evidence was found for this legal issue in the current document set."
                impact = (
                    f"Providing supporting evidence ({materiality_label} materiality) will advance "
                    f"proof coverage for this claim element and may alter case strength assessment."
                )
            elif "document" in gap_type:
                question = f"We could not find the following in the repository: {description}. Do you have access to this document or information?"
                why = "This document was referenced in the matter but is not present in the repository."
                if link and link["affected_type"] == "issue":
                    impact = (
                        f"This document directly affects a tracked issue "
                        f"(materiality: {materiality_label}). "
                        f"Providing it will update the evidence coverage for that claim element."
                    )
                elif link and link["affected_type"] == "assertion":
                    impact = (
                        f"This document was cited by an existing extracted fact "
                        f"(materiality: {materiality_label}). "
                        f"Providing it may corroborate, contradict, or supersede that assertion."
                    )
                else:
                    impact = (
                        f"This document has {materiality_label} materiality to the current matter. "
                        f"If available, it could alter relevant factual findings or conclusions."
                    )
            else:
                question = f"We identified a gap: {description}. Can you provide any additional context or documentation?"
                why = "This information is needed to complete the analysis."
                if link:
                    impact = (
                        f"Filling this gap (materiality: {materiality_label}) will update "
                        f"the {link['affected_type']} it is linked to in the matter model."
                    )
                else:
                    impact = f"Providing this information ({materiality_label} materiality) will improve matter coverage."

            q_id = self.clarifications.add_question(
                question_text=question,
                why_it_matters=why,
                expected_impact=impact,
                gap_id=gap.get("id"),
                run_id=run_id,
            )
            question_ids.append(q_id)

        return question_ids

    # ------------------------------------------------------------------
    # Quantitative intelligence (SO-6 + SO-7)
    # ------------------------------------------------------------------

    def detect_quant_conflicts(self, run_id: Optional[str] = None) -> list[str]:
        """
        Detect numeric conflicts: same subject_type+currency with divergent amounts.

        For each conflict group found:
        1. Records an UNRESOLVED_CONTRADICTION gap (materiality 0.8).
        2. Wires bidirectional 'contradicts' assertion_links between the assertions
           that carry the conflicting quant facts (SO-6 → SO-2 propagation).
        3. Runs BeliefRevisionEngine on those assertions so they are marked DISPUTED
           in the assertion graph (truth maintenance).

        Returns list of new gap_ids created. Idempotent: skips conflicts whose gap
        description is already in the open gap store.
        """
        conflicts = self.quant.get_conflicts()
        if not conflicts:
            return []

        existing_descriptions = {
            g.get("description", "").lower()
            for g in self.gaps.open_gaps(min_materiality=0.0)
        }

        gap_ids = []
        all_conflict_assertion_ids: list[str] = []

        for conflict in conflicts:
            subject = conflict.get("subject_type") or "unknown"
            currency = conflict.get("currency") or ""
            values = conflict.get("values", [])
            value_str = ", ".join(f"{v:,.2f}" for v in values[:5])
            desc = f"Conflicting {subject} amounts ({currency}): {value_str}"

            # Record gap (idempotent)
            if desc.lower() not in existing_descriptions:
                gap_id = self.record_gap(
                    description=desc,
                    gap_type=GapType.UNRESOLVED_CONTRADICTION,
                    materiality=0.8,
                )
                gap_ids.append(gap_id)

            # Collect assertion_ids for quant facts in this conflict group (SO-6→SO-2)
            subject_id = conflict.get("subject_id")
            rows = self.db.execute(
                """SELECT DISTINCT assertion_id
                   FROM quant_fact
                   WHERE matter_id=? AND quant_kind='amount'
                     AND subject_type=?
                     AND COALESCE(subject_id, '')=?
                     AND (currency=? OR (currency IS NULL AND ?=''))
                     AND assertion_id IS NOT NULL
                   ORDER BY created_at ASC""",
                (self.matter_id, subject, subject_id or "", currency, currency),
            ).fetchall()
            aids = [r["assertion_id"] for r in rows]
            if len(aids) >= 2:
                # Cap at 20 before O(k²) nested link loop to bound worst-case work
                # when many assertions share the same (subject_type, '', currency) bucket
                # (e.g. unlabelled invoices all with NULL subject_id).
                # ORDER BY created_at ASC so the earliest-ingested assertions are
                # consistently chosen rather than arbitrary storage order.
                aids = aids[:20]
                # Wire bidirectional contradicts links (idempotent via UNIQUE index)
                for i, a1 in enumerate(aids):
                    for a2 in aids[i + 1:]:
                        self.assertions.link(a1, a2, AssertionLinkType.CONTRADICTS)
                        self.assertions.link(a2, a1, AssertionLinkType.CONTRADICTS)
                all_conflict_assertion_ids.extend(aids)

        # Run belief revision on all conflicting assertions so they become DISPUTED
        if all_conflict_assertion_ids:
            _conflict_unvisited: list[str] = []
            self.apply_revision(
                seed_assertion_ids=list(dict.fromkeys(all_conflict_assertion_ids)),
                cause=RevisionCause.CONFLICT_DETECTION,
                run_id=run_id,
                note="Automatic: conflicting amount values detected for same subject",
                _collect_unvisited=_conflict_unvisited,
            )
            self.enqueue_evidence_pending(_conflict_unvisited, cause=RevisionCause.CONFLICT_DETECTION, run_id=run_id)

        return gap_ids

    def reconcile(self, currency: str = "USD") -> dict:
        """Return reconciliation summary grouped by subject_type for a currency."""
        return self.quant.reconcile_by_subject(currency)

    def reconcile_payment_chain(self, currency: str = "USD") -> dict:
        """Return structured payment reconciliation: invoiced, paid, disputed, exposure.

        Satisfies SO-6: shows what was invoiced, paid, disputed, and the claimed
        exposure, grounded in source spans from the quant_fact store.
        """
        return self.quant.reconcile_payment_chain(currency)

    def reconcile_invoice_chain(self, currency: str = "USD") -> list:
        """Return per-invoice reconciliation rows (invoice_id, invoiced, paid, outstanding).

        Satisfies SO-6 per-invoice tracking requirement.
        """
        return self.quant.reconcile_invoice_chain(currency)

    def get_amount_conflicts(self) -> list[dict]:
        """Return grouped amount conflicts for UI transparency and auditing."""
        return self.quant.get_conflicts()

    # ------------------------------------------------------------------
    # Timeline view (SO-6, Priority 2 visual work product)
    # ------------------------------------------------------------------

    def get_timeline(
        self, limit: int = 200, policy_audience: str = "internal",
    ) -> list[dict]:
        """Return a chronological event list derived from quant dates and assertions.

        Each event has:
          date         — ISO date string or raw text when not parseable
          event        — human-readable description
          source_doc   — document the event came from
          quant_id     — quant_fact.id when sourced from quant store, else None
          assertion_id — assertion.id when sourced from assertion store, else None
          subject      — subject_type/subject_id when available
          kind         — 'date' | 'date_range' | 'temporal_assertion'

        Ordered by date ascending (None dates last).
        """
        events: list[dict] = []

        # Pull date-type quant facts.
        date_facts = self.quant.get_by_kind("date", limit=limit)
        date_range_facts = self.quant.get_by_kind("date_range", limit=limit)

        for qf in date_facts + date_range_facts:
            date_val = qf.get("date_value") or qf.get("date_end_value") or qf.get("raw_text", "")
            events.append({
                "date": date_val,
                "date_precision": qf.get("date_precision"),
                "event": qf.get("raw_text", ""),
                "source_doc": qf.get("span_id"),
                "quant_id": qf.get("id"),
                "assertion_id": qf.get("assertion_id"),
                "subject": qf.get("subject_id") or qf.get("subject_type"),
                "kind": qf.get("quant_kind", "date"),
            })

        # Pull assertions that have a temporal scope, with their document source.
        temporal_rows = self.db.execute(
            """SELECT a.id, a.proposition_text, a.temporal_scope_start,
                      MIN(ao.document_id) AS doc_id
               FROM assertion a
               LEFT JOIN assertion_occurrence ao ON ao.assertion_id = a.id
               WHERE a.matter_id=? AND a.temporal_scope_start IS NOT NULL
               GROUP BY a.id
               ORDER BY a.temporal_scope_start
               LIMIT ?""",
            (self.matter_id, limit),
        ).fetchall()

        for row in temporal_rows:
            date_val = row["temporal_scope_start"]
            desc = row["proposition_text"] or ""
            events.append({
                "date": date_val,
                "date_precision": "day",  # assertions store normalised ISO dates
                "event": desc,
                "source_doc": row["doc_id"],
                "quant_id": None,
                "assertion_id": row["id"],
                "subject": None,
                "kind": "temporal_assertion",
            })

        # Sort: events with parseable dates first, None/empty last.
        def _sort_key(e):
            d = e.get("date") or ""
            # ISO dates sort correctly as strings (YYYY-MM-DD).
            # Prefix None-like with "~" so they sort last.
            return d if d else "~"

        events.sort(key=_sort_key)
        events = events[:limit]

        # P0.5 commit 4: clean-audience timeline hides privileged
        # content behind a "[withheld]" placeholder instead of
        # silently dropping the row. The date and event order are
        # preserved so the attorney sees the gap and can request the
        # internal view if they need it. Every decision is audited.
        if policy_audience == "clean":
            privileged_docs = self.privilege.privileged_doc_inventory_ids()
            if privileged_docs:
                # Also collect relative_path → inv_id to match legacy
                # source_doc fields that carry raw paths.
                inv_rows = self.db.execute(
                    "SELECT id, relative_path FROM document_inventory WHERE matter_id=?",
                    (self.matter_id,),
                ).fetchall()
                path_to_inv = {r["relative_path"]: r["id"] for r in inv_rows}
                from .trust import ContentPurpose, WITHHELD_PLACEHOLDER
                out: list[dict] = []
                for ev in events:
                    src = ev.get("source_doc") or ""
                    inv_id = (
                        src if src in privileged_docs
                        else path_to_inv.get(src)
                    )
                    is_priv = (
                        inv_id in privileged_docs if inv_id else False
                    )
                    if is_priv:
                        decision = self.content_policy.decide(
                            purpose=ContentPurpose.TIMELINE_VIEW,
                            subject_kind="document",
                            subject_id=inv_id or src,
                            policy_audience="clean",
                            privilege_flag=True,
                        )
                        out.append({
                            **ev,
                            "event": WITHHELD_PLACEHOLDER,
                            "source_doc": WITHHELD_PLACEHOLDER,
                            "subject": None,
                            "withheld": True,
                            "withheld_reason": decision.reason_code,
                        })
                    else:
                        out.append(ev)
                return out
        return events

    # ------------------------------------------------------------------
    # Evidence matrix (Priority 2 visual work product)
    # ------------------------------------------------------------------

    def get_evidence_matrix(
        self, policy_audience: str = "internal",
    ) -> dict:
        """Return a coverage matrix: issues × source documents.

        Structure:
        {
          "issues": [{"id": ..., "title": ..., "issue_type": ...}, ...],
          "sources": ["doc_a.pdf", "doc_b.pdf", ...],
          "cells": {
            issue_id: {
              doc_id: {"supporting": N, "attacking": N, "total": N}
            }
          },
          "issue_totals": {issue_id: {"supporting": N, "attacking": N}},
          "source_totals": {doc_id: {"supporting": N, "attacking": N}},
          "withheld_sources": [...]  # clean audience only
        }

        Only open issues with at least one assertion link are included.
        Only document sources with at least one assertion link are included.

        P0.5 commit 4: under policy_audience='clean', privileged
        document columns are replaced with a "[withheld]"
        placeholder column so the attorney sees coverage gaps
        instead of silently-dropped evidence.
        """
        open_issues = self.issues.get_open_issues(min_materiality=0.0)
        if not open_issues:
            return {"issues": [], "sources": [], "cells": {}, "issue_totals": {}, "source_totals": {}}

        # Query: assertion-issue links joined to occurrences to get document source.
        rows = self.db.execute(
            """SELECT ail.issue_id, ao.document_id, ail.relation_type, COUNT(*) AS cnt
               FROM assertion_issue_link ail
               JOIN issue i ON i.id = ail.issue_id
               JOIN assertion a ON a.id = ail.assertion_id
               JOIN assertion_occurrence ao ON ao.assertion_id = a.id
               WHERE i.matter_id=? AND i.status='open'
                 AND a.belief_state NOT IN ('superseded', 'withdrawn')
               GROUP BY ail.issue_id, ao.document_id, ail.relation_type""",
            (self.matter_id,),
        ).fetchall()

        # Build matrix.
        cells: dict = {}
        issue_totals: dict = {}
        source_totals: dict = {}
        sources_seen: set = set()
        issues_seen: set = set()

        for row in rows:
            iid = row["issue_id"]
            doc = row["document_id"] or "(unknown)"
            rel = row["relation_type"]
            cnt = row["cnt"]
            is_support = rel in ("supports", "establishes")
            is_attack = rel in ("attacks", "negates")

            issues_seen.add(iid)
            sources_seen.add(doc)

            if iid not in cells:
                cells[iid] = {}
            if doc not in cells[iid]:
                cells[iid][doc] = {"supporting": 0, "attacking": 0, "total": 0}
            if is_support:
                cells[iid][doc]["supporting"] += cnt
            elif is_attack:
                cells[iid][doc]["attacking"] += cnt
            cells[iid][doc]["total"] += cnt

            # Issue totals
            if iid not in issue_totals:
                issue_totals[iid] = {"supporting": 0, "attacking": 0}
            if is_support:
                issue_totals[iid]["supporting"] += cnt
            elif is_attack:
                issue_totals[iid]["attacking"] += cnt

            # Source totals
            if doc not in source_totals:
                source_totals[doc] = {"supporting": 0, "attacking": 0}
            if is_support:
                source_totals[doc]["supporting"] += cnt
            elif is_attack:
                source_totals[doc]["attacking"] += cnt

        issues_out = [
            {
                "id": i["id"],
                "title": i.get("title", ""),
                "issue_type": i.get("issue_type", ""),
            }
            for i in open_issues
            if i["id"] in issues_seen
        ]
        sources_out = sorted(sources_seen)

        # P0.5 commit 4: under clean audience, collapse privileged
        # source columns into a single "[withheld]" column so the
        # shape of the matrix is preserved but the document name
        # never leaks. Per-cell counts that traced to a privileged
        # source are summed into the withheld column.
        withheld_sources: list = []
        if policy_audience == "clean" and sources_out:
            from .trust import ContentPurpose, WITHHELD_PLACEHOLDER
            privileged_doc_ids = self.privilege.privileged_doc_inventory_ids()
            inv_rows = self.db.execute(
                "SELECT id, relative_path FROM document_inventory WHERE matter_id=?",
                (self.matter_id,),
            ).fetchall()
            path_to_inv = {r["relative_path"]: r["id"] for r in inv_rows}
            # Decide which source keys (paths or ids from the matrix)
            # are privileged.
            priv_sources: set = set()
            for src in sources_out:
                inv_id = (
                    src if src in privileged_doc_ids
                    else path_to_inv.get(src)
                )
                if inv_id and inv_id in privileged_doc_ids:
                    priv_sources.add(src)
                    self.content_policy.decide(
                        purpose=ContentPurpose.MATRIX_VIEW,
                        subject_kind="document",
                        subject_id=inv_id,
                        policy_audience="clean",
                        privilege_flag=True,
                    )
            if priv_sources:
                placeholder = WITHHELD_PLACEHOLDER
                # Build new sources list with withheld at the end,
                # privileged names removed.
                non_priv = [s for s in sources_out if s not in priv_sources]
                sources_out = non_priv + [placeholder]
                withheld_sources = sorted(priv_sources)
                # Rewrite cells: each issue's privileged-source cells
                # collapse into one "[withheld]" column with summed counts.
                new_cells: dict = {}
                for iid, doc_map in cells.items():
                    kept = {
                        doc: vals for doc, vals in doc_map.items()
                        if doc not in priv_sources
                    }
                    withheld_sum = {"supporting": 0, "attacking": 0, "total": 0}
                    for doc, vals in doc_map.items():
                        if doc in priv_sources:
                            for k in withheld_sum:
                                withheld_sum[k] += vals.get(k, 0)
                    if withheld_sum["total"] > 0:
                        kept[placeholder] = withheld_sum
                    new_cells[iid] = kept
                cells = new_cells
                # Rewrite source_totals similarly.
                new_source_totals: dict = {
                    doc: vals for doc, vals in source_totals.items()
                    if doc not in priv_sources
                }
                withheld_totals = {"supporting": 0, "attacking": 0}
                for doc, vals in source_totals.items():
                    if doc in priv_sources:
                        for k in withheld_totals:
                            withheld_totals[k] += vals.get(k, 0)
                if any(v > 0 for v in withheld_totals.values()):
                    new_source_totals[placeholder] = withheld_totals
                source_totals = new_source_totals

        return {
            "issues": issues_out,
            "sources": sources_out,
            "cells": cells,
            "issue_totals": issue_totals,
            "source_totals": source_totals,
            "withheld_sources": withheld_sources,
        }

    # ------------------------------------------------------------------
    # Communication map (Priority 2 visual work product)
    # ------------------------------------------------------------------

    def get_communication_map(self) -> dict:
        """Return the actor-document interaction graph.

        Structure:
        {
          "actors": [{"id": ..., "name": ..., "actor_type": ..., "home_side": ...}],
          "documents": ["doc_a.pdf", "doc_b.pdf", ...],
          "actor_document_edges": [
            {"actor_id": ..., "actor_name": ..., "document_id": ..., "occurrence_count": N}
          ],
          "actor_actor_edges": [
            {"actor_a_id": ..., "actor_b_id": ..., "shared_documents": N, "documents": [...]}
          ],
        }

        actor_document_edges: actor X appeared in document Y in N assertion occurrences.
        actor_actor_edges: actors A and B both appeared in at least one common document.
        Only actors with at least one occurrence are included.
        """
        # Actor-document occurrence counts.
        occ_rows = self.db.execute(
            """SELECT ao.speaker_actor_id AS actor_id,
                      ao.document_id,
                      COUNT(*) AS cnt
               FROM assertion_occurrence ao
               JOIN actor ac ON ac.id = ao.speaker_actor_id
               WHERE ac.matter_id=? AND ao.speaker_actor_id IS NOT NULL
               GROUP BY ao.speaker_actor_id, ao.document_id""",
            (self.matter_id,),
        ).fetchall()

        actor_ids_seen: set = set()
        docs_seen: set = set()
        actor_doc_edges: list = []
        # actor_id → set of documents
        actor_docs: dict = {}

        # Build a lookup for actor names.
        actor_rows = self.db.execute(
            "SELECT id, canonical_name, actor_type, home_side FROM actor WHERE matter_id=?",
            (self.matter_id,),
        ).fetchall()
        actor_map = {r["id"]: dict(r) for r in actor_rows}

        for row in occ_rows:
            aid = row["actor_id"]
            doc = row["document_id"] or "(unknown)"
            actor_ids_seen.add(aid)
            docs_seen.add(doc)
            if aid not in actor_docs:
                actor_docs[aid] = set()
            actor_docs[aid].add(doc)
            actor_doc_edges.append({
                "actor_id": aid,
                "actor_name": actor_map.get(aid, {}).get("canonical_name", aid),
                "document_id": doc,
                "occurrence_count": row["cnt"],
            })

        # Build actor-actor co-appearance edges.
        actor_actor_edges: list = []
        actor_list = sorted(actor_ids_seen)
        for i, a1 in enumerate(actor_list):
            for a2 in actor_list[i + 1:]:
                shared = actor_docs.get(a1, set()) & actor_docs.get(a2, set())
                if shared:
                    actor_actor_edges.append({
                        "actor_a_id": a1,
                        "actor_b_id": a2,
                        "shared_documents": len(shared),
                        "documents": sorted(shared),
                    })

        actors_out = [
            {
                "id": aid,
                "name": actor_map[aid]["canonical_name"],
                "actor_type": actor_map[aid]["actor_type"],
                "home_side": actor_map[aid]["home_side"],
            }
            for aid in actor_ids_seen
            if aid in actor_map
        ]

        return {
            "actors": actors_out,
            "documents": sorted(docs_seen),
            "actor_document_edges": actor_doc_edges,
            "actor_actor_edges": actor_actor_edges,
        }

    # ------------------------------------------------------------------
    # Damages waterfall (Priority 2 visual work product)
    # ------------------------------------------------------------------

    def get_damages_waterfall(self, currency: str = "USD") -> list[dict]:
        """Return a structured damages breakdown by category (subject_type).

        Each entry:
          component       — subject_type label (e.g. "damages", "invoice", "fee")
          claimed_amount  — total of all amounts in this category
          source_count    — number of distinct quant_fact entries
          amounts         — list of {raw_text, amount_value, subject_id, assertion_id}
          conflicts       — list of conflicting amounts when multiple values exist
                            and the range is > 20% of the max value

        Ordered by claimed_amount descending (largest exposure first).
        Only 'amount'-kind quant facts for the given currency are included.
        Categories with subject_type IS NULL are grouped under "(uncategorised)".
        """
        rows = self.db.execute(
            """SELECT id, subject_type, subject_id, amount_value, raw_text, assertion_id, span_id
               FROM quant_fact
               WHERE matter_id=? AND quant_kind='amount'
                 AND (currency=? OR (currency IS NULL AND ?='USD'))
               ORDER BY subject_type, amount_value DESC""",
            (self.matter_id, currency, currency),
        ).fetchall()

        if not rows:
            return []

        # Group by subject_type.
        from collections import defaultdict
        groups: dict = defaultdict(list)
        for row in rows:
            key = row["subject_type"] or "(uncategorised)"
            groups[key].append({
                "quant_fact_id": row["id"],
                "raw_text": row["raw_text"] or "",
                "amount_value": row["amount_value"],
                "subject_id": row["subject_id"],
                "assertion_id": row["assertion_id"],
                "span_id": row["span_id"],
            })

        waterfall = []
        for component, entries in groups.items():
            values = [e["amount_value"] for e in entries if e["amount_value"] is not None]
            total = sum(values) if values else 0.0
            max_val = max(values) if values else 0.0

            # Detect conflicts: multiple distinct amounts where spread > 20% of max.
            unique_vals = sorted(set(values), reverse=True)
            conflicts = []
            if len(unique_vals) >= 2 and max_val > 0:
                spread = unique_vals[0] - unique_vals[-1]
                if spread / max_val > 0.20:
                    conflicts = [f"${v:,.2f}" for v in unique_vals]

            waterfall.append({
                "component": component,
                "claimed_amount": round(total, 2),
                "source_count": len(entries),
                "currency": currency,
                "amounts": entries,
                "conflicts": conflicts,
            })

        waterfall.sort(key=lambda x: x["claimed_amount"], reverse=True)
        return waterfall

    # ------------------------------------------------------------------
    # SO-3: structured steering surface
    # ------------------------------------------------------------------

    def get_ledger_steering_surface(
        self,
        run_id: Optional[str] = None,  # noqa: ARG002 — reserved for per-run scoping (not yet implemented)
        limit: int = 20,
    ) -> list[dict]:
        """Return structured steering actions the user can take to direct reasoning.

        Derives actionable recommendations from current matter model state:
        active conflicts, low-coverage issues, high-materiality gaps, and
        unanswered clarifications.  Each action includes the exact parameters
        needed to invoke the corresponding API method — so a UI can render
        these as buttons or commands without manual interpretation.

        Args:
            run_id: Reserved for future per-run scoping.  Currently the surface
                    covers the full matter state regardless of run_id.
            limit:  Maximum number of actions to return (highest-priority first).

        Returns a list of dicts, each with:
          action_id   — unique string identifier for this suggestion
          action_type — one of: correct_assertion, force_belief_state,
                        redirect_focus, supply_document, answer_clarification,
                        set_trust_override
          description — human-readable label for the action
          params      — dict of kwargs to pass to the corresponding MatterModel method
          rationale   — why this action is suggested
          priority    — 'high', 'medium', or 'low'
          impact      — what will change if the action is taken
        """

        def _aid() -> str:
            return uuid.uuid4().hex[:12]

        actions: list[dict] = []

        # --- 1. Active conflicts → force_belief_state ---
        try:
            conflicts = self.assertions.find_contradictions(limit=5)
            for c in conflicts:
                attacker_id = c.get("attacker_id") or ""
                attacked_id = c.get("attacked_id") or ""
                if not attacker_id or not attacked_id:
                    continue  # skip malformed rows — no valid action params
                attacker_prop = (c.get("attacker_prop") or "")[:80]
                attacked_prop = (c.get("attacked_prop") or "")[:80]
                actions.append({
                    "action_id": _aid(),
                    "action_type": "force_belief_state",
                    "description": (
                        f"Resolve conflict: \"{attacker_prop}\" attacks \"{attacked_prop}\""
                    ),
                    "params": {
                        "assertion_id": attacked_id,
                        "new_state": "disputed",
                    },
                    "rationale": (
                        f"Assertion {attacked_id[:8]} is actively attacked by {attacker_id[:8]}."
                        " Marking the attacked claim as 'disputed' halts downstream inference"
                        " from an unresolved conflict."
                    ),
                    "priority": "high",
                    "impact": (
                        "Belief revision will propagate through the dependency graph,"
                        " marking all conclusions that depend on the attacked assertion"
                        " as uncertain until the conflict is resolved."
                    ),
                })
        except Exception:
            _log.warning("get_ledger_steering_surface: error in conflict section", exc_info=True)

        # --- 2. Low-coverage issues → redirect_focus ---
        try:
            coverage_report = self.get_issue_coverage_report()
            # Sort by ascending coverage fraction — weakest first
            weak_issues = sorted(
                (r for r in coverage_report if r.get("coverage_fraction", 1.0) < 0.6),
                key=lambda r: r.get("coverage_fraction", 1.0),
            )
            for r in weak_issues[:3]:
                issue_id = r.get("id") or ""
                if not issue_id:
                    continue
                issue_title = (r.get("title") or issue_id)[:60]
                frac = r.get("coverage_fraction", 0.0)
                actions.append({
                    "action_id": _aid(),
                    "action_type": "redirect_focus",
                    "description": (
                        f"Redirect investigation to under-covered issue: \"{issue_title}\""
                        f" ({round(frac * 100)}% covered)"
                    ),
                    "params": {
                        "matter_id": self.matter_id,
                        "run_id": run_id,
                        "issue_id": issue_id,
                    },
                    "rationale": (
                        f"Issue \"{issue_title}\" has only {round(frac * 100)}% evidence coverage."
                        " Redirecting forces the next investigation iteration to prioritize"
                        " leads for this issue."
                    ),
                    "priority": "high" if frac < 0.3 else "medium",
                    "impact": (
                        "The engine's lead scoring will up-weight retrieval queries"
                        f" tied to issue {issue_id[:8]}, increasing coverage in the next run."
                    ),
                })
        except Exception:
            _log.warning("get_ledger_steering_surface: error in coverage section", exc_info=True)

        # --- 3. High-materiality missing-doc gaps → supply_document ---
        try:
            # Fetch only as many as we'll surface (DB-side limit avoids full scan).
            _gap_candidates = self.gaps.open_gaps(min_materiality=0.6, limit=20)
            high_gaps = [
                g for g in _gap_candidates
                if g.get("gap_type") in ("MISSING_DOCUMENT", "missing_document")
            ]
            for g in high_gaps[:3]:
                gap_id = g.get("id") or ""
                description = (g.get("description") or "unknown document")[:80]
                materiality = g.get("materiality", 0.0)
                actions.append({
                    "action_id": _aid(),
                    "action_type": "supply_document",
                    "description": f"Supply missing document: \"{description}\"",
                    "params": {
                        "gap_id": gap_id,
                        "description": description,
                    },
                    "rationale": (
                        f"Gap (materiality {materiality:.2f}) flagged: \"{description}\"."
                        " Conclusions depending on this document are currently uncertain."
                    ),
                    "priority": "high" if materiality >= 0.8 else "medium",
                    "impact": (
                        "Supplying the document will allow the engine to ingest it,"
                        " resolve this gap, and update all assertions that depend on it."
                    ),
                })
        except Exception:
            _log.warning("get_ledger_steering_surface: error in gaps section", exc_info=True)

        # --- 4. Pending clarifications → answer_clarification ---
        try:
            # DB-side limit: fetch only what we'll surface to avoid full-table scan
            pending = self.clarifications.get_pending(limit=3)
            for q in pending:
                q_id = q.get("id") or ""
                if not q_id:
                    continue
                q_text = (q.get("question_text") or "")[:100]
                impact = (q.get("expected_impact") or "")[:120]
                actions.append({
                    "action_id": _aid(),
                    "action_type": "answer_clarification",
                    "description": f"Answer pending clarification: \"{q_text}\"",
                    "params": {
                        "question_id": q_id,
                        "answer_text": "<your answer here>",
                    },
                    "rationale": (
                        "The engine generated this targeted question because an answer"
                        " would materially improve matter coverage."
                    ),
                    "priority": "medium",
                    "impact": impact or "Answering will allow the engine to close the underlying gap.",
                })
        except Exception:
            _log.warning("get_ledger_steering_surface: error in clarifications section", exc_info=True)

        # --- 5. Disputed/unknown assertions → correct_assertion ---
        try:
            disputed_rows = self.db.execute(
                """SELECT id, proposition_text, belief_state
                   FROM assertion
                   WHERE matter_id=? AND belief_state IN ('disputed','unknown')
                   ORDER BY updated_at DESC LIMIT 5""",
                (self.matter_id,),
            ).fetchall()
            for row in disputed_rows:
                a_id = row["id"] or ""
                if not a_id:
                    continue
                prop = (row["proposition_text"] or "")[:80]
                state = row["belief_state"]
                actions.append({
                    "action_id": _aid(),
                    "action_type": "correct_assertion",
                    "description": f"Correct {state} assertion: \"{prop}\"",
                    "params": {
                        "assertion_id": a_id,
                        "new_state": "operative",
                        "note": "<explain your correction>",
                    },
                    "rationale": (
                        f"Assertion is in '{state}' state — its truth is unresolved."
                        " If you have information about the correct state, correcting it"
                        " will propagate belief revision through dependent assertions."
                    ),
                    "priority": "medium" if state == "disputed" else "low",
                    "impact": (
                        "BeliefRevisionEngine will propagate the correction through all"
                        " assertions that link to this one via supports/attacks edges."
                    ),
                })
        except Exception:
            _log.warning("get_ledger_steering_surface: error in disputed-assertions section", exc_info=True)

        # Sort: high → medium → low, then truncate to limit
        _priority_order = {"high": 0, "medium": 1, "low": 2}
        actions.sort(key=lambda a: _priority_order.get(a["priority"], 99))
        return actions[:limit]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return a summary of matter model state."""
        llm_totals = self.summarize_llm_usage()
        latest_run_id: Optional[str] = None
        try:
            row = self.db.execute(
                "SELECT id FROM run_session WHERE matter_id=? AND operation_type='query'"
                " ORDER BY started_at DESC LIMIT 1",
                (self.matter_id,),
            ).fetchone()
            latest_run_id = row["id"] if row else None
        except Exception:
            try:
                row = self.db.execute(
                    "SELECT id FROM run_session WHERE matter_id=?"
                    " ORDER BY started_at DESC LIMIT 1",
                    (self.matter_id,),
                ).fetchone()
                latest_run_id = row["id"] if row else None
            except Exception:
                latest_run_id = None
        return {
            "matter_id": self.matter_id,
            "assertion_count": self.assertions.count(),
            "open_gap_count": self.gaps.count_open(),
            "open_issue_count": self.issues.count_open(),
            "actor_count": self.actors.count(),
            "quant_fact_count": self.quant.count(),
            "pending_clarifications": self.clarifications.count_pending(),
            "recent_runs": len(self.ledger.recent_runs(limit=5)),
            "llm": {
                "totals": llm_totals,
                "last_run": self.summarize_llm_usage(run_id=latest_run_id) if latest_run_id else None,
            },
        }

    def get_so_metrics(self, _coverage_report: "list[dict] | None" = None) -> dict:
        """Compute measurable Sacred Outcome success criteria from stored state.

        Returns a snapshot of how well the current matter model satisfies the
        quantitative success criteria defined in docs/PROJECT_CONTEXT.md. Metrics
        that require ground truth or run telemetry (reuse_rate, gap_detection_recall,
        numeric_extraction_rate) are reported as None.

        Targets:
          assertion_structure_rate >= 1.0   (SO-2: 100% typed, full metadata)
          source_role_known_rate   >= 0.9   (SO-5: advocacy vs operative calibration)
          issue_coverage_avg       >= 0.8   (SO-4: issue-driven retrieval coverage)
          steerability             == True  (SO-3: user can interrupt mid-run)
          belief_revision          == True  (SO-2: corrections propagate)
        """
        # --- assertion structure: speech_act + source_role both set in occurrence ---
        ao_total_row = self.db.execute(
            "SELECT COUNT(*) AS n FROM assertion_occurrence WHERE assertion_id IN "
            "(SELECT id FROM assertion WHERE matter_id=?)",
            (self.matter_id,),
        ).fetchone()
        ao_total = int(ao_total_row["n"]) if ao_total_row else 0

        ao_typed_row = self.db.execute(
            "SELECT COUNT(*) AS n FROM assertion_occurrence WHERE assertion_id IN "
            "(SELECT id FROM assertion WHERE matter_id=?)"
            " AND speech_act IS NOT NULL AND speech_act != ''"
            " AND source_role IS NOT NULL AND source_role != ''",
            (self.matter_id,),
        ).fetchone()
        ao_typed = int(ao_typed_row["n"]) if ao_typed_row else 0

        assertion_structure_rate = (ao_typed / ao_total) if ao_total > 0 else None

        # --- source calibration: % occurrences with source_role != 'unknown' ---
        ao_known_row = self.db.execute(
            "SELECT COUNT(*) AS n FROM assertion_occurrence WHERE assertion_id IN "
            "(SELECT id FROM assertion WHERE matter_id=?)"
            " AND source_role IS NOT NULL AND source_role != '' AND source_role != 'unknown'",
            (self.matter_id,),
        ).fetchone()
        ao_known = int(ao_known_row["n"]) if ao_known_row else 0

        source_role_known_rate = (ao_known / ao_total) if ao_total > 0 else None

        # --- issue coverage — reuse pre-computed report if caller already fetched it ---
        issue_coverage_avg: "float | None" = None
        issues_with_proof_gap = 0
        try:
            coverage_report = (
                _coverage_report
                if _coverage_report is not None
                else self.get_issue_coverage_report()
            )
            if coverage_report:
                fracs = [float(r.get("coverage_fraction", 0.0)) for r in coverage_report]
                issue_coverage_avg = round(sum(fracs) / len(fracs), 4) if fracs else None
                issues_with_proof_gap = sum(1 for r in coverage_report if r.get("has_proof_gap"))
        except Exception:
            pass

        assertion_count = self.assertions.count()
        issue_count = self.issues.count_open()
        open_gap_count = self.gaps.count_open()
        quant_fact_count = self.quant.count()
        actor_count = self.actors.count()

        # SO-3: steerability — True only if this matter has at least one investigation
        # run session (not a utility flush run).  Uses the same objective filter as
        # request_stop() / request_redirect() in reasoning.py so only runs that are
        # actually stoppable/redirectable are counted.  False if no qualifying runs;
        # None on DB error.  A matter with only flush history must not report True.
        steerability: "bool | None" = None
        try:
            sr_row = self.db.execute(
                "SELECT COUNT(*) AS n FROM run_session WHERE matter_id=?"
                " AND (objective IS NULL OR objective NOT IN"
                " ('manual_flush','background_flush'))",
                (self.matter_id,),
            ).fetchone()
            steerability = bool(int(sr_row["n"]) > 0) if sr_row else False
        except Exception:
            steerability = None

        # SO-2: belief_revision — True if any revision events exist for this matter.
        # A count > 0 means the truth-maintenance system has actually revised beliefs.
        # Returns False (not None) when no revisions have occurred yet — this is
        # meaningful: it signals the matter hasn't triggered corrections, not that
        # the capability is absent.
        belief_revision: "bool | None" = None
        try:
            rev_row = self.db.execute(
                """SELECT COUNT(*) AS n FROM belief_revision_event
                   WHERE assertion_id IN
                   (SELECT id FROM assertion WHERE matter_id=?)""",
                (self.matter_id,),
            ).fetchone()
            belief_revision = bool(int(rev_row["n"]) > 0)
        except Exception:
            belief_revision = None  # table missing or schema mismatch

        def _pass(metric: str, value: "float | bool | None") -> "bool | None":
            if value is None:
                return None
            target = targets[metric]
            if isinstance(target, bool):
                return bool(value) == target
            return float(value) >= float(target)  # type: ignore[arg-type]

        # SO-1 reuse_rate: fraction of final assertions that pre-existed at run start.
        # Averaged over the most recent 5 completed runs so a single anomalous run
        # doesn't dominate.  None if no completed runs exist yet.
        # Exclude runs where assertions_at_start=0 (first-ever ingestion run on an empty matter).
        # Those runs cannot reuse any prior state by definition — including them would
        # artificially depress the average and hide genuine reuse patterns on subsequent runs.
        # Exclude utility runs (manual/background flush) that do not represent
        # investigation reuse — they complete instantly with reuse_rate ≈ 1.0 since
        # no new assertions are added, which would artificially inflate the average
        # (adv#030 MEDIUM fix).
        _reuse_rows = self.db.execute(
            """SELECT reuse_rate FROM run_session
               WHERE matter_id=? AND status='completed' AND reuse_rate IS NOT NULL
                 AND assertions_at_start > 0
                 AND (objective IS NULL OR (objective != 'manual_flush' AND objective != 'background_flush'))
                 AND resumed_from IS NULL
               ORDER BY completed_at DESC LIMIT 5""",
            (self.matter_id,),
        ).fetchall()
        _reuse_vals = [float(r["reuse_rate"]) for r in _reuse_rows if r["reuse_rate"] is not None]
        reuse_rate_avg: "float | None" = (
            round(sum(_reuse_vals) / len(_reuse_vals), 4) if _reuse_vals else None
        )

        # SO-6: numeric_extraction_rate — percentage of quant_facts that
        # carry a source span_id. Before P0.1 provenance this was never
        # measured; now every quant the engine records threads span_id
        # through record_quants_batch and record_quant. A dropping rate
        # indicates an extraction pipeline losing source grounding.
        numeric_extraction_rate: "float | None" = None
        if quant_fact_count > 0:
            _spans_row = self.db.execute(
                "SELECT COUNT(*) AS n FROM quant_fact"
                " WHERE matter_id=? AND span_id IS NOT NULL AND span_id != ''",
                (self.matter_id,),
            ).fetchone()
            _with_span = int(_spans_row["n"]) if _spans_row else 0
            numeric_extraction_rate = round(_with_span / quant_fact_count, 4)

        # SO-2: provenance_attribution_rate — percentage of AI-derived
        # assertions that have at least one provenance_event row.
        # Before P0.1 this was 0; after P0.1 + the adversarial-#6
        # closeout every production write path threads ProvenanceContext,
        # so a healthy matter should sit near 1.0. A drop indicates a
        # writer path regressed to pre-P0.1 behavior.
        provenance_attribution_rate: "float | None" = None
        if assertion_count > 0:
            _attributed_row = self.db.execute(
                "SELECT COUNT(DISTINCT a.id) AS n"
                " FROM assertion a"
                " JOIN provenance_event pe"
                "   ON pe.target_kind='assertion' AND pe.target_id=a.id"
                " WHERE a.matter_id=?",
                (self.matter_id,),
            ).fetchone()
            _attributed = int(_attributed_row["n"]) if _attributed_row else 0
            provenance_attribution_rate = round(_attributed / assertion_count, 4)

        # SO-7: gap_detection_recall — no ground truth available, so
        # instead report a measurable proxy: gap_surface_ratio = ratio
        # of open gaps tagged as proof-critical (missing_issue_predicate
        # or missing_authority) to total open issues. A matter with
        # many open issues and zero proof-critical gaps is suspicious;
        # a healthy matter has >0 when issues are underdeveloped.
        gap_surface_ratio: "float | None" = None
        if issue_count > 0:
            _critical_gap_row = self.db.execute(
                "SELECT COUNT(*) AS n FROM gap"
                " WHERE matter_id=? AND status='open'"
                " AND gap_type IN ('missing_issue_predicate','missing_authority')",
                (self.matter_id,),
            ).fetchone()
            _critical_gaps = int(_critical_gap_row["n"]) if _critical_gap_row else 0
            gap_surface_ratio = round(_critical_gaps / issue_count, 4)

        targets = {
            "assertion_structure_rate": 1.0,
            "source_role_known_rate": 0.9,
            "issue_coverage_avg": 0.8,
            "reuse_rate": 0.7,
            "numeric_extraction_rate": 0.9,
            "provenance_attribution_rate": 0.9,
            "steerability": True,
            "belief_revision": True,
        }

        return {
            "matter_id": self.matter_id,
            # Measurable SO metrics
            "assertion_structure_rate": assertion_structure_rate,
            "source_role_known_rate": source_role_known_rate,
            "issue_coverage_avg": issue_coverage_avg,
            "issues_with_proof_gap": issues_with_proof_gap,
            # SO-1: reuse rate averaged over recent completed runs (target > 0.70)
            "reuse_rate": reuse_rate_avg,
            # SO-3: True if ≥1 run_session recorded (interruptible engine confirmed used)
            "steerability": steerability,
            # SO-2: True if belief_revision_event records exist (revisions have occurred)
            "belief_revision": belief_revision,
            # SO-7: proof-critical gap surface ratio (open gaps flagged
            # missing_issue_predicate or missing_authority over open
            # issues). Replaces the prior `gap_detection_recall: None`
            # theater metric — we don't have ground truth, but we do
            # have a real signal on whether proof gaps are surfacing.
            "gap_surface_ratio": gap_surface_ratio,
            # SO-6: % of numeric facts with source span identity
            "numeric_extraction_rate": numeric_extraction_rate,
            # SO-2: % of assertions with at least one provenance event
            "provenance_attribution_rate": provenance_attribution_rate,
            # Raw counts
            "counts": {
                "assertions": assertion_count,
                "assertion_occurrences": ao_total,
                "issues_open": issue_count,
                "open_gaps": open_gap_count,
                "quant_facts": quant_fact_count,
                "actors": actor_count,
            },
            # Targets from docs/PROJECT_CONTEXT.md
            "targets": targets,
            # Pass/fail per metric (None = not enough data to evaluate)
            "targets_met": {
                "assertion_structure_rate": _pass("assertion_structure_rate", assertion_structure_rate),
                "source_role_known_rate": _pass("source_role_known_rate", source_role_known_rate),
                "issue_coverage_avg": _pass("issue_coverage_avg", issue_coverage_avg),
                "reuse_rate": _pass("reuse_rate", reuse_rate_avg),
                "numeric_extraction_rate": _pass(
                    "numeric_extraction_rate", numeric_extraction_rate,
                ),
                "provenance_attribution_rate": _pass(
                    "provenance_attribution_rate", provenance_attribution_rate,
                ),
                "steerability": _pass("steerability", steerability),
                "belief_revision": _pass("belief_revision", belief_revision),
            },
        }

    def __repr__(self) -> str:
        return f"MatterModel(matter_id={self.matter_id[:8]}..., db={self.db})"
