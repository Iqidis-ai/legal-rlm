"""MatterModel — the central facade for all matter intelligence stores.

Usage:
    model = MatterModel.open("path/to/repository", matter_name="Acme v TechServices")
    run_id = model.start_run("What are the key obligations?")
    assertion_id, is_new = model.record_assertion(candidate)
    model.ledger.append_event(run_id, LedgerEventType.ASSERTION_ADDED, ...)
    model.complete_run(run_id)
"""

import json
import logging
import math
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
    AssertionStore, GapStore, ActorStore, IssueStore, ClarificationStore, QuantStore, MetricAliasStore, KnowledgeSeedStore,
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
        self.metric_aliases = MetricAliasStore(db, matter_id)
        self.knowledge_seeds = KnowledgeSeedStore(db, matter_id)
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

    _VERIFICATION_CAS_NAMESPACES = (
        "verification_state", "cache_records",
    )

    _VERIFICATION_TARGET_TABLES: dict[str, str] = {
        "assertion": "assertion",
        "evidence_edge": "evidence_edge",
        "quant_fact": "quant_fact",
    }

    def verification_revision_keys(self, target_kind: str, target_id: str) -> dict[str, int]:
        """Snapshot namespace revisions for CAS-protected verify/reject.

        Raises ValueError when target_kind maps to a concrete table and
        the target row does not exist.
        """
        table = self._VERIFICATION_TARGET_TABLES.get(target_kind)
        if table is not None:
            row = self.db.execute(
                f"SELECT 1 FROM {table} WHERE id=? AND matter_id=?",
                (target_id, self.matter_id),
            ).fetchone()
            if row is None:
                raise ValueError(
                    f"{target_kind} {target_id} not found"
                )
        broker = self.memory_broker
        revisions: dict[str, int] = {}
        for ns in self._VERIFICATION_CAS_NAMESPACES:
            key = broker.revision_key(ns)
            revisions[key] = broker.get_namespace_revision(ns)
        key = broker.revision_key("verification_state", target_kind, target_id)
        revisions[key] = broker.get_namespace_revision(
            "verification_state", target_kind, target_id,
        )
        return revisions

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
        expected_revisions: Optional[dict[str, int]] = None,
    ) -> str:
        """P0.3: promote a target to verified, append ledger audit
        event, recompute proof state for any open issues it supports.

        When expected_revisions is provided, validates namespace revisions
        before writing (CAS protection against stale-view verifications).
        """
        if expected_revisions is not None:
            return self._verify_target_brokered(
                target_kind, target_id,
                reviewed_by_kind=reviewed_by_kind,
                reviewed_by_id=reviewed_by_id,
                review_note=review_note,
                review_scope=review_scope,
                run_id=run_id,
                expected_revisions=expected_revisions,
            )
        return self._verify_target_inner(
            target_kind, target_id,
            reviewed_by_kind=reviewed_by_kind,
            reviewed_by_id=reviewed_by_id,
            review_note=review_note,
            review_scope=review_scope,
            run_id=run_id,
        )

    def _verify_target_brokered(
        self,
        target_kind: str,
        target_id: str,
        *,
        reviewed_by_kind: str,
        reviewed_by_id: Optional[str],
        review_note: Optional[str],
        review_scope: str,
        run_id: Optional[str],
        expected_revisions: dict[str, int],
    ) -> str:
        broker = self.memory_broker
        required_keys = set()
        for ns in self._VERIFICATION_CAS_NAMESPACES:
            required_keys.add(broker.revision_key(ns))
        required_keys.add(
            broker.revision_key("verification_state", target_kind, target_id)
        )
        now = _now()
        with self.db.write_transaction():
            broker._assert_expected_revisions(expected_revisions, required_keys)
            vid = self._verify_target_inner(
                target_kind, target_id,
                reviewed_by_kind=reviewed_by_kind,
                reviewed_by_id=reviewed_by_id,
                review_note=review_note,
                review_scope=review_scope,
                run_id=run_id,
            )
            for ns in self._VERIFICATION_CAS_NAMESPACES:
                broker._bump_namespace_revision_in_tx(ns, now=now)
            broker._bump_namespace_revision_in_tx(
                "verification_state", target_kind, target_id, now=now,
            )
            broker._bump_namespace_revision_in_tx(
                "verification_state", now=now,
            )
        return vid

    def _verify_target_inner(
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
        vid = self.verification.verify(
            target_kind, target_id,
            reviewed_by_kind=reviewed_by_kind,
            reviewed_by_id=reviewed_by_id,
            review_note=review_note,
            review_scope=review_scope,
            run_id=run_id,
        )
        if target_kind == "assertion":
            self._verify_companion_edges(
                assertion_id=target_id,
                reviewed_by_kind=reviewed_by_kind,
                reviewed_by_id=reviewed_by_id,
                review_note=review_note,
                review_scope=review_scope,
                run_id=run_id,
            )
        if run_id is not None:
            self.ledger.append_event(
                run_id=run_id,
                event_type=LedgerEventType.ASSERTION_REVISED,
                summary=f"Verified {target_kind}:{target_id}",
                changed_object_type=target_kind,
                changed_object_id=target_id,
            )
        import sqlite3 as _sqlite3
        _any_recomputed = False
        for iid in self._issues_affected_by_target(target_kind, target_id):
            try:
                self.proof_state.compute_and_store(iid, policy_audience="internal")
                _any_recomputed = True
            except _sqlite3.Error as _exc:
                _log.warning(
                    "verify_target: proof recompute failed for issue %s: %s",
                    iid, _exc,
                )
        if _any_recomputed:
            self.memory_broker.bump_namespace_revision("proof_state")
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
        expected_revisions: Optional[dict[str, int]] = None,
    ) -> str:
        """P0.3: reject a target, append ledger audit event, and
        recompute proof state for any open issues it supported so
        their coverage drops accordingly.

        When expected_revisions is provided, validates namespace revisions
        before writing (CAS protection against stale-view rejections).
        """
        if expected_revisions is not None:
            return self._reject_target_brokered(
                target_kind, target_id,
                reviewed_by_kind=reviewed_by_kind,
                rejection_reason=rejection_reason,
                reviewed_by_id=reviewed_by_id,
                review_note=review_note,
                review_scope=review_scope,
                run_id=run_id,
                expected_revisions=expected_revisions,
            )
        return self._reject_target_inner(
            target_kind, target_id,
            reviewed_by_kind=reviewed_by_kind,
            rejection_reason=rejection_reason,
            reviewed_by_id=reviewed_by_id,
            review_note=review_note,
            review_scope=review_scope,
            run_id=run_id,
        )

    def _reject_target_brokered(
        self,
        target_kind: str,
        target_id: str,
        *,
        reviewed_by_kind: str,
        rejection_reason: str,
        reviewed_by_id: Optional[str],
        review_note: Optional[str],
        review_scope: str,
        run_id: Optional[str],
        expected_revisions: dict[str, int],
    ) -> str:
        broker = self.memory_broker
        required_keys = set()
        for ns in self._VERIFICATION_CAS_NAMESPACES:
            required_keys.add(broker.revision_key(ns))
        required_keys.add(
            broker.revision_key("verification_state", target_kind, target_id)
        )
        now = _now()
        with self.db.write_transaction():
            broker._assert_expected_revisions(expected_revisions, required_keys)
            vid = self._reject_target_inner(
                target_kind, target_id,
                reviewed_by_kind=reviewed_by_kind,
                rejection_reason=rejection_reason,
                reviewed_by_id=reviewed_by_id,
                review_note=review_note,
                review_scope=review_scope,
                run_id=run_id,
            )
            for ns in self._VERIFICATION_CAS_NAMESPACES:
                broker._bump_namespace_revision_in_tx(ns, now=now)
            broker._bump_namespace_revision_in_tx(
                "verification_state", target_kind, target_id, now=now,
            )
            broker._bump_namespace_revision_in_tx(
                "verification_state", now=now,
            )
        return vid

    def _reject_target_inner(
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
        self._stale_rejection_dependents(target_kind, target_id)
        import sqlite3 as _sqlite3
        _any_recomputed = False
        for iid in self._issues_affected_by_target(target_kind, target_id):
            try:
                self.proof_state.compute_and_store(iid, policy_audience="internal")
                _any_recomputed = True
            except _sqlite3.Error as _exc:
                _log.warning(
                    "reject_target: proof recompute failed for issue %s: %s",
                    iid, _exc,
                )
        if _any_recomputed:
            self.memory_broker.bump_namespace_revision("proof_state")
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
            """SELECT id, relative_path FROM document_inventory
               WHERE matter_id=? AND (id=? OR relative_path=?)""",
            (self.matter_id, doc_id, doc_id),
        ).fetchone()
        if inv is None:
            return 0
        inv_id = inv["id"]
        rel_path = inv["relative_path"]
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
        staled = self._apply_invalidation(
            scope,
            reason=f"privilege_reclassified:{old_flag}->{1 if new_flag else 0}",
        )
        # Propagate privilege taint to the document and all dependents
        # so the clean output pipeline filters them.
        self._propagate_privilege_taint(inv_id, rel_path, scope, new_flag)
        return staled

    def _propagate_privilege_taint(
        self,
        doc_id: str,
        relative_path: str,
        scope: dict[str, set[str]],
        privileged: bool,
    ) -> None:
        """Write or remove privilege_restricted taint for a document and its dependents."""
        taint_class = "privilege_restricted"
        kind_map = {
            "assertion_ids": "assertion",
            "occurrence_ids": "assertion_occurrence",
            "edge_ids": "evidence_edge",
            "quant_ids": "quant_fact",
            "authority_ids": "authority",
        }
        broker = self.memory_broker
        # Taint both the inventory ID and the relative path so
        # build_query_context filters on either identity.
        targets: list[tuple[str, str]] = [("artifact", doc_id)]
        if relative_path and relative_path != doc_id:
            targets.append(("artifact", relative_path))
        for scope_key, target_kind in kind_map.items():
            for tid in scope.get(scope_key, set()):
                targets.append((target_kind, tid))
        if privileged:
            for target_kind, target_id in targets:
                broker.record_object_taint(
                    target_kind=target_kind,
                    target_id=target_id,
                    taint_class=taint_class,
                    derivation_reason="privilege_reclassified",
                )
        else:
            for target_kind, target_id in targets:
                broker.remove_object_taint_by_class(
                    target_kind, target_id, taint_class,
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
        """Load composed trust weights into the belief engine, refreshing on facet changes."""
        broker = self.memory_broker
        current_rev = broker.get_namespace_revision("domain_facets", "*", "*")
        last = getattr(self, "_belief_tw_facet_rev", -1)
        if self.belief.trust_weights is not None and current_rev == last:
            return
        _, tw, _ = self._read_matter_domain_composition()
        if tw:
            self.belief.trust_weights = tw
        self._belief_tw_facet_rev = current_rev

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

    _CORRECT_ASSERTION_CAS_NAMESPACES = (
        "assertions", "claim_occurrences",
        "cache_records", "object_taint",
    )

    def correct_assertion(
        self,
        assertion_id: str,
        new_state: BeliefState,
        run_id: Optional[str] = None,
        note: Optional[str] = None,
        confidence: Optional[float] = None,
        expected_revisions: Optional[dict[str, int]] = None,
    ) -> RevisionResult:
        """Apply a user correction to an assertion and propagate.

        After belief revision propagates through the assertion graph, this
        triggers a targeted proof_state recompute for all issues linked to
        the corrected assertion — so issue-level prioritization in the loop
        reflects the correction, not a stale pre-correction state (SO-2).

        When expected_revisions is provided, the write is protected by CAS:
        namespace revisions are validated before the correction proceeds,
        preventing stale-view writes from the API.

        run_id is validated: if provided but not a running session for this matter,
        it is silently cleared to None so stale IDs cannot misattribute audit rows
        regardless of the calling path (REST, in-process, or engine).
        """
        if expected_revisions is not None:
            return self._correct_assertion_brokered(
                assertion_id, new_state, run_id, note, confidence,
                expected_revisions,
            )
        self._ensure_belief_trust_weights()
        return self._correct_assertion_inner(
            assertion_id, new_state, run_id, note, confidence,
        )

    def _correct_assertion_brokered(
        self,
        assertion_id: str,
        new_state: BeliefState,
        run_id: Optional[str],
        note: Optional[str],
        confidence: Optional[float],
        expected_revisions: dict[str, int],
    ) -> RevisionResult:
        """CAS-protected correction: validates namespace revisions before writing."""
        from .graph import MemoryBrokerCASMismatch

        broker = self.memory_broker
        required_keys = set()
        for ns in self._CORRECT_ASSERTION_CAS_NAMESPACES:
            required_keys.add(broker.revision_key(ns))
        required_keys.add(
            broker.revision_key("assertions", "assertion", assertion_id)
        )

        now = _now()
        with self.db.write_transaction():
            broker._assert_expected_revisions(expected_revisions, required_keys)
            self._ensure_belief_trust_weights()

            result = self._correct_assertion_inner(
                assertion_id, new_state, run_id, note, confidence,
            )

            for ns in self._CORRECT_ASSERTION_CAS_NAMESPACES:
                broker._bump_namespace_revision_in_tx(ns, now=now)
            broker._bump_namespace_revision_in_tx(
                "assertions", "assertion", assertion_id, now=now,
            )

        return result

    def _correct_assertion_inner(
        self,
        assertion_id: str,
        new_state: BeliefState,
        run_id: Optional[str],
        note: Optional[str],
        confidence: Optional[float],
    ) -> RevisionResult:
        """Core correction logic shared by legacy and brokered paths."""
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

        _pending: list[str] = list(result.truncation_pending)
        result.truncation_pending = []
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

        result.propagation_truncated = bool(_pending)
        if _pending:
            result.truncation_pending = _pending
            self.enqueue_correction_pending(_pending, run_id=run_id)

        try:
            affected = list(dict.fromkeys([assertion_id] + (result.propagated_to or [])))
            if affected:
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
                    _override_rows = self.db.execute(
                        """SELECT document_pattern, trust_level FROM document_trust_override
                           WHERE matter_id=? AND trust_level != 'normal'
                           ORDER BY LENGTH(document_pattern) DESC""",
                        (self.matter_id,),
                    ).fetchall()
                    _overrides = [
                        (r["document_pattern"], r["trust_level"]) for r in _override_rows
                    ]
                    for _iid in issue_ids_to_recompute:
                        self.proof_state.compute_and_store(
                            _iid, _preloaded_overrides=_overrides
                        )
            self.memory_broker.bump_namespace_revision("proof_state")
        except Exception as exc:
            _log.warning(
                "proof_state recompute after correct_assertion failed for %r: %s",
                assertion_id, exc,
            )

        try:
            self.cache.bump_trust_revision()
        except sqlite3.Error as _exc:
            _log.warning("correct_assertion: trust_revision bump failed: %s", _exc)

        return result

    def correct_assertion_revision_keys(self, assertion_id: str) -> dict[str, int]:
        """Snapshot current namespace revisions needed for a brokered correction.

        Callers (e.g. the REST API) call this before presenting the correction
        form, then pass the result as expected_revisions to correct_assertion().
        Raises ValueError if the assertion does not exist.
        """
        row = self.db.execute(
            "SELECT 1 FROM assertion WHERE id=? AND matter_id=?",
            (assertion_id, self.matter_id),
        ).fetchone()
        if row is None:
            raise ValueError(f"Assertion {assertion_id} not found")
        broker = self.memory_broker
        revisions: dict[str, int] = {}
        for ns in self._CORRECT_ASSERTION_CAS_NAMESPACES:
            key = broker.revision_key(ns)
            revisions[key] = broker.get_namespace_revision(ns)
        key = broker.revision_key("assertions", "assertion", assertion_id)
        revisions[key] = broker.get_namespace_revision(
            "assertions", "assertion", assertion_id,
        )
        return revisions

    _TRUST_OVERRIDE_CAS_NAMESPACES = (
        "trust_overrides", "cache_records",
    )

    def trust_override_revision_keys(self, document_pattern: str) -> dict[str, int]:
        """Snapshot namespace revisions for CAS-protected set/delete trust override.

        Callers (e.g. the REST API) call this before presenting the trust
        override form, then pass the result as expected_revisions to
        set_trust_override() or delete_trust_override().
        """
        broker = self.memory_broker
        revisions: dict[str, int] = {}
        for ns in self._TRUST_OVERRIDE_CAS_NAMESPACES:
            key = broker.revision_key(ns)
            revisions[key] = broker.get_namespace_revision(ns)
        key = broker.revision_key("trust_overrides", "document", document_pattern)
        revisions[key] = broker.get_namespace_revision(
            "trust_overrides", "document", document_pattern,
        )
        return revisions

    def set_trust_override(
        self,
        document_pattern: str,
        trust_level: str,
        note: Optional[str] = None,
        run_id: Optional[str] = None,
        expected_revisions: Optional[dict[str, int]] = None,
    ) -> str:
        """Set a document trust override and trigger belief revision on affected assertions.

        This is the high-level entry point for SO-3 trust steering.  It:
        1. Persists the trust override to document_trust_override (SO-3).
        2. Finds all assertions whose primary document matches the pattern.
        3. Triggers belief revision (RevisionCause.TRUST_OVERRIDE) on those assertions
           so the new effective source_role weight flows through to stored belief states (SO-2).

        Belief revision failure does not block the override — the override is persisted
        regardless of whether propagation succeeds.

        When expected_revisions is provided, the write is protected by CAS:
        namespace revisions are validated before the mutation proceeds.

        Returns the override_id.
        """
        if expected_revisions is not None:
            return self._set_trust_override_brokered(
                document_pattern, trust_level, note, run_id,
                expected_revisions,
            )
        return self._set_trust_override_inner(
            document_pattern, trust_level, note, run_id,
        )

    def _set_trust_override_brokered(
        self,
        document_pattern: str,
        trust_level: str,
        note: Optional[str],
        run_id: Optional[str],
        expected_revisions: dict[str, int],
    ) -> str:
        """CAS-protected trust override: validates namespace revisions before writing."""
        from .graph import MemoryBrokerCASMismatch

        broker = self.memory_broker
        required_keys = set()
        for ns in self._TRUST_OVERRIDE_CAS_NAMESPACES:
            required_keys.add(broker.revision_key(ns))
        required_keys.add(
            broker.revision_key("trust_overrides", "document", document_pattern)
        )

        now = _now()
        with self.db.write_transaction():
            broker._assert_expected_revisions(expected_revisions, required_keys)

            override_id = self._set_trust_override_inner(
                document_pattern, trust_level, note, run_id,
            )

            for ns in self._TRUST_OVERRIDE_CAS_NAMESPACES:
                broker._bump_namespace_revision_in_tx(ns, now=now)
            broker._bump_namespace_revision_in_tx(
                "trust_overrides", "document", document_pattern, now=now,
            )
            broker._bump_namespace_revision_in_tx(
                "trust_overrides", now=now,
            )
            broker._bump_namespace_revision_in_tx("assertions", now=now)

        return override_id

    def _set_trust_override_inner(
        self,
        document_pattern: str,
        trust_level: str,
        note: Optional[str],
        run_id: Optional[str],
    ) -> str:
        """Core trust override logic shared by legacy and brokered paths."""
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

        try:
            self.cache.bump_trust_revision()
        except sqlite3.Error as _exc:
            _log.warning("set_trust_override: trust_revision bump failed: %s", _exc)

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
        self.enqueue_evidence_pending(_trust_unvisited, cause=RevisionCause.TRUST_OVERRIDE, run_id=run_id)

        import sqlite3 as _sqlite3
        _any_recomputed = False
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
                            try:
                                self.proof_state.compute_and_store(
                                    _iid, _preloaded_overrides=_preloaded
                                )
                                _any_recomputed = True
                            except _sqlite3.Error as _exc:
                                _log.warning(
                                    "set_trust_override: proof recompute failed for issue %s: %s",
                                    _iid, _exc,
                                )
        except (sqlite3.Error, ValueError, RuntimeError) as exc:
            _log.warning("Trust override proof state refresh failed for %r: %s", document_pattern, exc)
        if _any_recomputed:
            self.memory_broker.bump_namespace_revision("proof_state")

        return override_id

    def delete_trust_override(
        self,
        document_pattern: str,
        run_id: Optional[str] = None,
        expected_revisions: Optional[dict[str, int]] = None,
    ) -> None:
        """Delete a document trust override and re-propagate belief revision.

        Mirrors set_trust_override: deletes the row first, then re-runs belief
        revision on affected assertions so beliefs revert to auto-inferred trust.
        Proof state is recomputed for affected issues.

        When expected_revisions is provided, the write is protected by CAS.
        """
        if expected_revisions is not None:
            return self._delete_trust_override_brokered(
                document_pattern, run_id, expected_revisions,
            )
        return self._delete_trust_override_inner(document_pattern, run_id)

    def _delete_trust_override_brokered(
        self,
        document_pattern: str,
        run_id: Optional[str],
        expected_revisions: dict[str, int],
    ) -> None:
        """CAS-protected trust override delete: validates namespace revisions before writing."""
        from .graph import MemoryBrokerCASMismatch

        broker = self.memory_broker
        required_keys = set()
        for ns in self._TRUST_OVERRIDE_CAS_NAMESPACES:
            required_keys.add(broker.revision_key(ns))
        required_keys.add(
            broker.revision_key("trust_overrides", "document", document_pattern)
        )

        now = _now()
        with self.db.write_transaction():
            broker._assert_expected_revisions(expected_revisions, required_keys)

            self._delete_trust_override_inner(document_pattern, run_id)

            for ns in self._TRUST_OVERRIDE_CAS_NAMESPACES:
                broker._bump_namespace_revision_in_tx(ns, now=now)
            broker._bump_namespace_revision_in_tx(
                "trust_overrides", "document", document_pattern, now=now,
            )
            broker._bump_namespace_revision_in_tx(
                "trust_overrides", now=now,
            )
            broker._bump_namespace_revision_in_tx("assertions", now=now)

    def _delete_trust_override_inner(
        self,
        document_pattern: str,
        run_id: Optional[str],
    ) -> None:
        """Core trust override delete logic shared by legacy and brokered paths."""
        deleted = self.trust_overrides.delete(document_pattern)
        if not deleted:
            return

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

        import sqlite3 as _sqlite3
        _any_recomputed = False
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
                            try:
                                self.proof_state.compute_and_store(
                                    _iid, _preloaded_overrides=_preloaded
                                )
                                _any_recomputed = True
                            except _sqlite3.Error as _exc:
                                _log.warning(
                                    "delete_trust_override: proof recompute failed for issue %s: %s",
                                    _iid, _exc,
                                )
        except (sqlite3.Error, ValueError, RuntimeError) as exc:
            _log.warning("Trust override delete proof state refresh failed for %r: %s", document_pattern, exc)
        if _any_recomputed:
            self.memory_broker.bump_namespace_revision("proof_state")

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

    def list_version_families(self) -> list[dict]:
        """Return document version families with operative HEAD marked."""
        return self.inventory.list_version_families()

    def list_assertion_history(self, assertion_id: str, limit: int = 20) -> dict:
        """Return field-level revision history for an assertion (SO-2)."""
        import json as _json
        limit = max(1, min(limit, 500))
        row = self.db.execute(
            "SELECT id FROM assertion WHERE id=? AND matter_id=?",
            (assertion_id, self.matter_id),
        ).fetchone()
        if row is None:
            return {"assertion_id": assertion_id, "history": [], "count": 0}
        rev_rows = self.db.execute(
            """SELECT ar.id, ar.batch_id, ar.changed_field,
                      ar.old_value_json, ar.new_value_json,
                      ar.actor_kind, ar.actor_ref, ar.cause,
                      ar.run_id, ar.note, ar.created_at
               FROM assertion_revision ar
               WHERE ar.assertion_id=?
               ORDER BY ar.created_at DESC, ar.batch_id DESC, ar.id DESC
               LIMIT ?""",
            (assertion_id, limit),
        ).fetchall()

        def _decode(raw):
            if not raw:
                return None
            try:
                return _json.loads(raw)
            except (_json.JSONDecodeError, TypeError):
                return raw

        history = [
            {
                "id": r["id"],
                "batch_id": r["batch_id"],
                "changed_field": r["changed_field"],
                "old_value": _decode(r["old_value_json"]),
                "new_value": _decode(r["new_value_json"]),
                "actor_kind": r["actor_kind"],
                "actor_ref": r["actor_ref"],
                "cause": r["cause"],
                "run_id": r["run_id"],
                "note": r["note"],
                "created_at": r["created_at"],
            }
            for r in rev_rows
        ]
        return {
            "assertion_id": assertion_id,
            "history": history,
            "count": len(history),
            "history_note": "Revision history since schema v34.",
        }

    def list_content_policy_decisions(self, limit: int = 50) -> list[dict]:
        """Return recent content policy audit decisions (SO-5)."""
        return self.content_policy.list_decisions(limit=max(1, min(limit, 500)))

    def get_assertion_health(self, assertion_id: str) -> dict:
        """Return assertion health: oscillation, neighbors, provenance, linked issues (SO-2 + SO-5)."""
        record = self.assertions.get(assertion_id)
        if record is None:
            return {"error": "assertion_not_found"}
        neighbors = self.assertions.get_neighbor_belief_states(assertion_id)
        linked_issues = self.issues.get_issues_for_assertion(assertion_id)
        return {
            "assertion_id": assertion_id,
            "proposition_text": record.proposition_text,
            "belief_state": record.belief_state,
            "confidence": record.confidence,
            "oscillating": self.assertions.detect_oscillation(assertion_id),
            "support_count": len(neighbors.get("support_states", [])),
            "attack_count": len(neighbors.get("attack_states", [])),
            "has_superseding": neighbors.get("has_superseding", False),
            "support_source_roles": neighbors.get("support_source_roles", []),
            "attack_source_roles": neighbors.get("attack_source_roles", []),
            "provenance": self.get_provenance("assertion", assertion_id, limit=10),
            "linked_issues": linked_issues,
        }

    def get_source_agreement_for_issue(self, issue_id: str) -> list[dict]:
        """Per-document support/attack breakdown for an issue (SO-5).

        Groups assertions linked to the issue by their source document,
        counting how many support vs attack the issue from each source.
        """
        rows = self.db.execute(
            """SELECT di.id AS doc_id,
                      COALESCE(di.relative_path, di.original_filename, di.id) AS doc_label,
                      di.source_role,
                      ail.relation_type,
                      COUNT(*) AS cnt
               FROM assertion_issue_link ail
               JOIN assertion_occurrence ao ON ao.assertion_id = ail.assertion_id
               JOIN document_inventory di ON di.id = ao.document_inventory_id
               JOIN assertion a ON a.id = ail.assertion_id
               WHERE ail.issue_id = ?
                 AND a.belief_state NOT IN ('superseded', 'withdrawn')
               GROUP BY di.id, ail.relation_type
               ORDER BY di.source_role, di.id""",
            (issue_id,),
        ).fetchall()
        doc_map: dict[str, dict] = {}
        for r in rows:
            did = r["doc_id"]
            if did not in doc_map:
                doc_map[did] = {
                    "doc_id": did,
                    "doc_label": r["doc_label"],
                    "source_role": r["source_role"] or "unknown",
                    "supports": 0,
                    "attacks": 0,
                }
            rel = r["relation_type"] or ""
            if rel in ("supports", "establishes"):
                doc_map[did]["supports"] += r["cnt"]
            elif rel in ("attacks", "negates"):
                doc_map[did]["attacks"] += r["cnt"]
        return sorted(doc_map.values(), key=lambda d: d["supports"] + d["attacks"], reverse=True)

    def get_assertion_graph_for_issue(self, issue_id: str) -> dict:
        """Return assertion nodes + inter-assertion edges for graph visualization (SO-2)."""
        return self.issues.get_assertion_graph_for_issue(issue_id)

    def get_system_health(self) -> dict:
        """Return system health diagnostics for the truth maintenance panel (SO-2)."""
        assertion_count = self.assertions.count()
        disputed_row = self.db.execute(
            "SELECT COUNT(*) AS n FROM assertion WHERE matter_id=? AND belief_state='disputed'",
            (self.matter_id,),
        ).fetchone()
        disputed_count = int(disputed_row["n"]) if disputed_row else 0

        revision_row = self.db.execute(
            "SELECT COUNT(*) AS n FROM belief_revision_event"
            " WHERE assertion_id IN (SELECT id FROM assertion WHERE matter_id=?)",
            (self.matter_id,),
        ).fetchone()
        revision_count = int(revision_row["n"]) if revision_row else 0

        gap_count = self.gaps.count_open()
        contradiction_count = self.assertions.count_contradictions()
        version_chain_count = len(self.list_version_families())

        oscillating_count = 0
        if assertion_count <= 200:
            aids = self.db.execute(
                "SELECT id FROM assertion WHERE matter_id=? LIMIT 200",
                (self.matter_id,),
            ).fetchall()
            oscillating_count = sum(
                1 for row in aids if self.assertions.detect_oscillation(row["id"])
            )

        return {
            "assertion_count": assertion_count,
            "disputed_count": disputed_count,
            "disputed_fraction": round(disputed_count / assertion_count, 4) if assertion_count else 0,
            "revision_count": revision_count,
            "open_gap_count": gap_count,
            "contradiction_count": contradiction_count,
            "version_chain_count": version_chain_count,
            "oscillating_count": oscillating_count,
            "health_score": "good" if (
                oscillating_count == 0 and (
                    disputed_count / assertion_count < 0.3 if assertion_count else True
                )
            ) else "attention_needed",
        }

    def compute_quant_thresholds(
        self,
        currency: str = "USD",
        *,
        exposure_high: float = 10_000.0,
        disputed_fraction_min: float = 0.10,
    ) -> list[dict]:
        """Detect quantitative threshold violations and record them as gaps (SO-6).

        Checks positive exposure, high disputed fraction, and numeric conflicts.
        Records violations as gaps in the gap store for downstream synthesis.
        Returns list of violation dicts with threshold, level, description, amount.
        """
        return self.quant.compute_thresholds(
            self.gaps, currency=currency,
            exposure_high=exposure_high, disputed_fraction_min=disputed_fraction_min,
        )

    # ------------------------------------------------------------------
    # Decision leverage map (SO-2, SO-3, SO-4, SO-5, SO-7)
    # ------------------------------------------------------------------

    def get_decision_leverage_map(self, top_n: int = 15) -> dict:
        """Ranked leverage points: things a professional should review next
        to maximally shift objective coverage or confidence."""
        items: list[dict] = []

        def _safe_float(v, default: float = 0.5) -> float:
            if isinstance(v, (int, float)) and math.isfinite(v):
                return max(0.0, min(1.0, float(v)))
            return default

        coverage_wb = self.get_objective_coverage_workbench()
        for obj in coverage_wb.get("objectives", []):
            if not isinstance(obj, dict):
                continue
            badge = obj.get("coverage_badge", "missing")
            if badge in ("missing", "blocked", "contradicted", "thin"):
                mat = _safe_float(obj.get("materiality"), 0.5)
                cov = _safe_float(obj.get("coverage_fraction"), 0.0)
                impact = mat * (1.0 - cov)
                items.append({
                    "kind": "weak_objective",
                    "id": obj.get("id", ""),
                    "title": obj.get("title", ""),
                    "blocker": badge,
                    "impact": round(impact, 3),
                    "detail": f"{obj.get('predicate_blocked', 0)} blocked, {len(obj.get('gaps', []))} gaps",
                    "action": "Strengthen evidence or resolve blocked criteria",
                })

        try:
            assumption_wb = self.get_assumption_review_workbench()
            for a in assumption_wb.get("provisional", []):
                if not isinstance(a, dict):
                    continue
                linked = a.get("linked_target_count", 0)
                if not isinstance(linked, (int, float)) or not math.isfinite(linked):
                    linked = 0
                if linked > 0:
                    items.append({
                        "kind": "unreviewed_assumption",
                        "id": a.get("id", ""),
                        "title": (a.get("statement") or "")[:100],
                        "blocker": f"{int(linked)} linked target(s)",
                        "impact": round(min(1.0, int(linked) * 0.15), 3),
                        "detail": f"Provisional assumption with {int(linked)} dependent predicate(s)/assertion(s)",
                        "action": "Confirm or invalidate this assumption",
                    })
        except Exception as exc:
            _log.warning("leverage_map: assumption load failed: %s", exc)

        try:
            taint = self.summarize_taint(limit=10)
            taint_total = taint.get("total", 0) if isinstance(taint, dict) else 0
            if isinstance(taint_total, (int, float)) and math.isfinite(taint_total) and taint_total > 0:
                items.append({
                    "kind": "tainted_evidence",
                    "id": "",
                    "title": f"{int(taint_total)} tainted evidence record(s)",
                    "blocker": "Evidence integrity risk",
                    "impact": round(min(1.0, int(taint_total) * 0.1), 3),
                    "detail": "Tainted sources may undermine dependent assertions",
                    "action": "Review taint summary and assess affected assertions",
                })
        except Exception as exc:
            _log.warning("leverage_map: taint load failed: %s", exc)

        try:
            conflicts = self.quant.get_conflicts()
            if conflicts:
                items.append({
                    "kind": "quant_conflict",
                    "id": "",
                    "title": f"{len(conflicts)} numeric conflict group(s)",
                    "blocker": "Conflicting amounts",
                    "impact": round(min(1.0, len(conflicts) * 0.12), 3),
                    "detail": "Different values for the same entity undermine quantitative claims",
                    "action": "Resolve conflicting amounts in the Quant Fact Review panel",
                })
        except Exception as exc:
            _log.warning("leverage_map: quant conflict load failed: %s", exc)

        try:
            top_gaps = self.gaps.open_gaps(limit=10)
            for gap in top_gaps:
                if not isinstance(gap, dict):
                    continue
                mat = _safe_float(gap.get("materiality_score"), 0.5)
                items.append({
                    "kind": "open_gap",
                    "id": gap.get("id", ""),
                    "title": (gap.get("description") or "")[:100],
                    "blocker": gap.get("gap_type", "unknown"),
                    "impact": round(mat * 0.8, 3),
                    "detail": f"Gap type: {gap.get('gap_type', 'unknown')}",
                    "action": "Resolve or escalate this gap",
                })
        except Exception as exc:
            _log.warning("leverage_map: gaps load failed: %s", exc)

        try:
            review_counts = self.count_review_queue()
            if isinstance(review_counts, dict):
                pending = review_counts.get("candidate", 0)
                if isinstance(pending, (int, float)) and math.isfinite(pending) and pending > 0:
                    items.append({
                        "kind": "pending_review",
                        "id": "",
                        "title": f"{int(pending)} assertion(s) awaiting review",
                        "blocker": "Unverified extractions",
                        "impact": round(min(1.0, int(pending) * 0.02), 3),
                        "detail": "Unreviewed assertions reduce confidence in dependent objectives",
                        "action": "Verify or reject assertions in the Review Queue",
                    })
        except Exception as exc:
            _log.warning("leverage_map: review queue load failed: %s", exc)

        items.sort(key=lambda x: x.get("impact", 0), reverse=True)
        items = items[:top_n]

        return {
            "total": len(items),
            "items": items,
        }

    # ------------------------------------------------------------------
    # Quant fact review workbench (SO-6)
    # ------------------------------------------------------------------

    def get_quant_fact_workbench(self, limit: int = 200) -> dict:
        """Professional quant fact review surface: extracted numbers grouped
        by kind, with source doc, conflict status, and assertion links."""
        all_facts = self.quant.list_all(limit=limit)
        conflicts = self.quant.get_conflicts()
        conflict_subjects: set[tuple] = set()
        for c in conflicts:
            if isinstance(c, dict):
                conflict_subjects.add((
                    c.get("subject_type", ""),
                    c.get("subject_id") or "",
                    c.get("currency") or "",
                ))

        by_kind: dict[str, list[dict]] = {}
        for fact in all_facts:
            if not isinstance(fact, dict):
                continue
            kind = fact.get("quant_kind", "unknown")
            subj_key = (
                fact.get("subject_type", ""),
                fact.get("subject_id") or "",
                fact.get("currency") or "",
            )
            fact["has_conflict"] = subj_key in conflict_subjects
            by_kind.setdefault(kind, []).append(fact)

        kind_summaries = []
        for kind in ("amount", "date", "date_range", "rate", "balance", "count"):
            items = by_kind.get(kind, [])
            conflicted = sum(1 for f in items if f.get("has_conflict"))
            kind_summaries.append({
                "kind": kind,
                "count": len(items),
                "conflicted": conflicted,
                "facts": items,
            })
        other_kinds = [k for k in by_kind if k not in ("amount", "date", "date_range", "rate", "balance", "count")]
        for kind in other_kinds:
            items = by_kind[kind]
            kind_summaries.append({
                "kind": kind,
                "count": len(items),
                "conflicted": sum(1 for f in items if f.get("has_conflict")),
                "facts": items,
            })

        total = sum(ks["count"] for ks in kind_summaries)
        total_conflicted = sum(ks["conflicted"] for ks in kind_summaries)

        return {
            "total": total,
            "total_conflicted": total_conflicted,
            "conflict_groups": len(conflicts),
            "by_kind": kind_summaries,
        }

    # ------------------------------------------------------------------
    # Quantitative ontology workbench (SO-6, SO-1, SO-3)
    # ------------------------------------------------------------------

    def get_quant_ontology_workbench(self) -> dict:
        _, _, primary_profile = self._read_matter_domain_composition()
        domain = primary_profile or "legal"

        quant_rows = self.quant.list_all(limit=200)
        by_metric: dict[str, list[dict]] = {}
        for qf in quant_rows:
            if not isinstance(qf, dict):
                continue
            st = qf.get("subject_type") or "other"
            by_metric.setdefault(st, []).append(qf)

        aliases = self.metric_aliases.get_all(domain_profile_id=domain)
        alias_map = {a["raw_label"]: a for a in aliases if isinstance(a, dict)}

        metric_groups: list[dict] = []
        for metric_type, facts in sorted(by_metric.items()):
            alias = alias_map.get(metric_type)
            metric_groups.append({
                "metric_type": metric_type,
                "canonical_metric": alias["canonical_metric"] if alias else None,
                "approved": bool(alias["approved_by_user"]) if alias else False,
                "unit": alias.get("unit") if alias else None,
                "fact_count": len(facts),
                "sample_facts": facts[:5],
                "total_value": sum(
                    float(f.get("amount_value") or 0) for f in facts
                    if f.get("amount_value") is not None
                ),
            })

        approved_count = self.metric_aliases.count_approved()
        total_types = len(by_metric)

        return {
            "matter_id": self.matter_id,
            "domain": domain,
            "metric_groups": metric_groups,
            "approved_count": approved_count,
            "total_metric_types": total_types,
            "coverage_fraction": approved_count / total_types if total_types > 0 else 0,
            "aliases": aliases,
        }

    def approve_metric_alias(
        self,
        raw_label: str,
        canonical_metric: str,
        domain_profile_id: str | None = None,
        unit: str | None = None,
    ) -> bool:
        if domain_profile_id is None:
            _, _, primary = self._read_matter_domain_composition()
            domain_profile_id = primary or "legal"
        self.metric_aliases.upsert(
            domain_profile_id=domain_profile_id,
            raw_label=raw_label,
            canonical_metric=canonical_metric,
            unit=unit,
            approved_by_user=True,
        )
        return True

    # ------------------------------------------------------------------
    # Answer audit workbench (SO-1, SO-2, SO-3, SO-5, SO-7)
    # ------------------------------------------------------------------

    def get_answer_audit_workbench(self, manifest_hash: str | None = None) -> dict:
        _, _, primary_profile = self._read_matter_domain_composition()
        domain = primary_profile or "legal"

        manifests = self.memory_broker.list_recent_manifests(limit=50)
        if not manifests:
            return {
                "matter_id": self.matter_id,
                "domain": domain,
                "audits": [],
                "total_manifests": 0,
            }

        if manifest_hash:
            targets = [m for m in manifests if m.get("manifest_hash") == manifest_hash]
            if not targets:
                return {
                    "matter_id": self.matter_id,
                    "domain": domain,
                    "audits": [],
                    "total_manifests": len(manifests),
                    "error": "Manifest not found",
                }
        else:
            targets = manifests[:10]

        audits: list[dict] = []
        for m in targets:
            if not isinstance(m, dict):
                continue
            mh = m.get("manifest_hash", "")
            try:
                validation = self.memory_broker.validate_dependency_manifest(mh)
                validation_dict = validation.to_canonical_dict() if validation else {}
            except Exception as exc:
                _log.warning("validate_dependency_manifest failed for %s: %s", mh, exc)
                validation_dict = {"valid": False, "status": "error", "stale_reasons": [str(exc)]}

            full_manifest = None
            obj_groups: dict[str, list[dict]] = {}
            neg_deps: list[dict] = []
            try:
                full_manifest = self.memory_broker.get_dependency_manifest(mh)
                if full_manifest:
                    for od in full_manifest.object_dependencies:
                        kind = od.target_kind or "unknown"
                        obj_groups.setdefault(kind, []).append({
                            "target_id": od.target_id,
                            "target_kind": od.target_kind,
                            "digest": getattr(od, "digest", ""),
                        })
                    for nd in full_manifest.negative_dependencies:
                        neg_deps.append({
                            "namespace": nd.namespace,
                            "query_predicate": nd.query_predicate,
                            "revision": nd.revision,
                        })
            except Exception as exc:
                _log.warning("get_dependency_manifest failed for %s: %s", mh, exc)

            valid = validation_dict.get("valid", False)
            stale_reasons = validation_dict.get("stale_reasons", [])

            if valid:
                status_badge = "fresh"
            elif "not_found" in validation_dict.get("status", ""):
                status_badge = "unknown"
            elif any(isinstance(r, str) and "policy" in r for r in stale_reasons):
                status_badge = "policy_limited"
            else:
                status_badge = "stale"

            audits.append({
                "manifest_hash": mh,
                "purpose": m.get("purpose", ""),
                "created_at": m.get("created_at", ""),
                "domain_profile_id": m.get("domain_profile_id", ""),
                "domain_profile_version": m.get("domain_profile_version", 0),
                "policy_audience": m.get("policy_audience", ""),
                "taint_class": m.get("taint_class", ""),
                "broker_version": m.get("broker_version", ""),
                "profile_mapping_hash": m.get("profile_mapping_hash", ""),
                "object_dependency_count": m.get("object_dependency_count", 0),
                "negative_dependency_count": m.get("negative_dependency_count", 0),
                "status_badge": status_badge,
                "valid": valid,
                "stale_reasons": stale_reasons,
                "object_groups": {k: v[:5] for k, v in obj_groups.items()},
                "negative_dependencies": neg_deps[:10],
            })

        return {
            "matter_id": self.matter_id,
            "domain": domain,
            "audits": audits,
            "total_manifests": len(manifests),
        }

    # ------------------------------------------------------------------
    # Dependency Manifest Inspector (SO-1, SO-2, SO-5)
    # ------------------------------------------------------------------

    def get_dependency_manifest_inspector(
        self,
        *,
        manifest_hash: str | None = None,
        run_id: str | None = None,
        limit: int = 25,
        policy_audience: str = "clean",
    ) -> dict:
        """Inspect dependency manifests with namespace-level staleness drilldown."""
        _, _, primary = self._read_matter_domain_composition()
        domain = primary or "legal"
        limit = max(1, min(limit, 100))

        all_manifests = self.memory_broker.list_recent_manifests(limit=200)
        if not all_manifests:
            return {
                "matter_id": self.matter_id,
                "domain": domain,
                "manifests": [],
                "total_count": 0,
                "fresh_count": 0,
                "stale_count": 0,
            }

        targets = all_manifests
        if manifest_hash:
            targets = [m for m in targets if isinstance(m, dict) and m.get("manifest_hash") == manifest_hash]
        if run_id:
            run_linked = set()
            try:
                rows = self.db.execute(
                    "SELECT DISTINCT dependency_manifest_hash FROM reasoning_cache "
                    "WHERE matter_id=? AND run_id=?",
                    (self.matter_id, run_id),
                ).fetchall()
                for r in rows:
                    h = r["dependency_manifest_hash"]
                    if h:
                        run_linked.add(h)
            except Exception as exc:
                _log.warning("manifest_inspector: run_id filter failed: %s", exc)
            if run_linked:
                targets = [m for m in targets if isinstance(m, dict) and m.get("manifest_hash") in run_linked]

        if policy_audience and policy_audience != "all":
            targets = [m for m in targets if isinstance(m, dict) and m.get("policy_audience", "") == policy_audience]

        targets = targets[:limit]

        manifests_out: list[dict] = []
        fresh_count = 0
        stale_count = 0

        for m in targets:
            if not isinstance(m, dict):
                continue
            mh = m.get("manifest_hash", "")
            try:
                validation = self.memory_broker.validate_dependency_manifest(mh)
                v_dict = validation.to_canonical_dict() if validation else {}
            except Exception as exc:
                _log.warning("manifest_inspector: validate failed for %s: %s", mh, exc)
                v_dict = {"valid": False, "status": "error", "stale_reasons": [str(exc)]}

            valid = v_dict.get("valid", False)
            stale_reasons = v_dict.get("stale_reasons", [])
            current_revisions = v_dict.get("current_revisions", {})

            if valid:
                status = "fresh"
                fresh_count += 1
            elif "not_found" in v_dict.get("status", ""):
                status = "unknown"
                stale_count += 1
            elif any(isinstance(r, str) and "taint" in r.lower() for r in stale_reasons):
                status = "taint_blocked"
                stale_count += 1
            elif any(isinstance(r, str) and "policy" in r.lower() for r in stale_reasons):
                status = "policy_limited"
                stale_count += 1
            else:
                status = "stale"
                stale_count += 1

            stale_namespaces: list[dict] = []
            for reason in stale_reasons:
                if not isinstance(reason, str):
                    continue
                if "namespace" in reason and "expected" in reason and "current" in reason:
                    stale_namespaces.append({"reason": reason[:300]})

            full_manifest = None
            obj_counts: dict[str, int] = {}
            try:
                full_manifest = self.memory_broker.get_dependency_manifest(mh)
                if full_manifest:
                    for od in full_manifest.object_dependencies:
                        kind = od.target_kind or "unknown"
                        obj_counts[kind] = obj_counts.get(kind, 0) + 1
            except Exception as exc:
                _log.warning("manifest_inspector: get_manifest failed for %s: %s", mh, exc)

            manifests_out.append({
                "manifest_hash": mh,
                "purpose": m.get("purpose", ""),
                "created_at": m.get("created_at", ""),
                "domain_profile_id": m.get("domain_profile_id", ""),
                "policy_audience": m.get("policy_audience", ""),
                "taint_class": m.get("taint_class", ""),
                "broker_version": m.get("broker_version", ""),
                "object_dependency_count": int(m.get("object_dependency_count", 0) or 0),
                "negative_dependency_count": int(m.get("negative_dependency_count", 0) or 0),
                "status": status,
                "valid": valid,
                "stale_reasons": stale_reasons[:10],
                "stale_namespaces": stale_namespaces[:10],
                "consumed_objects_by_kind": obj_counts,
                "current_revisions": {k: v for k, v in list(current_revisions.items())[:20]},
            })

        return {
            "matter_id": self.matter_id,
            "domain": domain,
            "manifests": manifests_out,
            "total_count": len(all_manifests),
            "fresh_count": fresh_count,
            "stale_count": stale_count,
        }

    # ------------------------------------------------------------------
    # Contradiction resolution workflow (SO-1, SO-2, SO-3, SO-7)
    # ------------------------------------------------------------------

    def resolve_contradiction(
        self,
        attacker_id: str,
        attacked_id: str,
        decision: str,
        rationale: str,
        run_id: str | None = None,
    ) -> dict:
        """Resolve a contradiction pair by applying user decision.

        decision must be one of:
          prefer_attacker — mark attacked as superseded
          prefer_attacked — mark attacker as superseded
          mark_both_disputed — mark both as disputed
          request_evidence — keep both, record gap for more evidence
        """
        valid_decisions = {
            "prefer_attacker", "prefer_attacked",
            "mark_both_disputed", "request_evidence",
        }
        if decision not in valid_decisions:
            return {"error": f"Invalid decision. Must be one of: {', '.join(sorted(valid_decisions))}"}

        attacker = self.assertions.get(attacker_id)
        attacked = self.assertions.get(attacked_id)
        if not attacker:
            return {"error": f"Attacker assertion {attacker_id} not found"}
        if not attacked:
            return {"error": f"Attacked assertion {attacked_id} not found"}

        self._ensure_belief_trust_weights()
        results: list[str] = []

        note_prefix = f"Resolution: {decision}. {rationale}"

        if decision == "prefer_attacker":
            r = self._correct_assertion_inner(
                attacked_id, BeliefState.SUPERSEDED, run_id,
                note=f"{note_prefix} Superseded by {attacker_id[:12]}",
                confidence=0.1,
            )
            results.append(f"Marked {attacked_id[:12]} as SUPERSEDED")
        elif decision == "prefer_attacked":
            r = self._correct_assertion_inner(
                attacker_id, BeliefState.SUPERSEDED, run_id,
                note=f"{note_prefix} Superseded by {attacked_id[:12]}",
                confidence=0.1,
            )
            results.append(f"Marked {attacker_id[:12]} as SUPERSEDED")
        elif decision == "mark_both_disputed":
            for aid in (attacker_id, attacked_id):
                r = self._correct_assertion_inner(
                    aid, BeliefState.DISPUTED, run_id,
                    note=note_prefix, confidence=0.3,
                )
                results.append(f"Marked {aid[:12]} as DISPUTED")
        elif decision == "request_evidence":
            self.gaps.record(
                gap_type=GapType.MISSING_DOCUMENT,
                description=f"More evidence needed to resolve: '{rationale[:120]}'",
                materiality=0.8,
                affected_type="assertion",
                affected_id=attacked_id,
            )
            results.append("Recorded gap for additional evidence")

        # Close related UNRESOLVED_CONTRADICTION gaps
        if decision != "request_evidence":
            closed = 0
            for gid in self._find_contradiction_gaps(attacker_id, attacked_id):
                if self.gaps.resolve_gap(gid, resolution_note=note_prefix[:200]):
                    closed += 1
            if closed:
                results.append(f"Closed {closed} contradiction gap(s)")

        return {
            "success": True,
            "decision": decision,
            "actions": results,
            "attacker_id": attacker_id,
            "attacked_id": attacked_id,
        }

    def _find_contradiction_gaps(self, attacker_id: str, attacked_id: str) -> list[str]:
        rows = self.db.execute(
            """SELECT g.id FROM gap g
               JOIN gap_link gl ON gl.gap_id = g.id
               WHERE g.matter_id = ? AND g.status = 'open'
                 AND g.gap_type = 'unresolved_contradiction'
                 AND gl.affected_id IN (?, ?)""",
            (self.matter_id, attacker_id, attacked_id),
        ).fetchall()
        return [r["id"] for r in rows]

    # ------------------------------------------------------------------
    # Cross-matter knowledge reuse (SO-1)
    # ------------------------------------------------------------------

    def get_knowledge_seed_workbench(self) -> dict:
        """Dashboard data for the knowledge seed reuse review panel."""
        all_seeds = self.knowledge_seeds.list_all()
        counts = self.knowledge_seeds.count_by_status()
        promotable = [s for s in all_seeds if s.get("promotion_status") == "promotable"]
        accepted = [s for s in all_seeds if s.get("promotion_status") == "matter_local"]
        rejected = [s for s in all_seeds if s.get("promotion_status") == "rejected"]
        return {
            "total": len(all_seeds),
            "counts": counts,
            "promotable": promotable,
            "accepted": accepted,
            "rejected": rejected,
        }

    def review_knowledge_seed(
        self,
        seed_id: str,
        decision: str,
        promoted_by: str | None = None,
        review_note: str | None = None,
    ) -> dict:
        """Review a knowledge seed: approve (matter_local), reject, or keep promotable."""
        valid = ("promotable", "matter_local", "rejected")
        if decision not in valid:
            return {"error": f"Invalid decision. Must be one of: {', '.join(valid)}"}
        seed = self.knowledge_seeds.get(seed_id)
        if not seed:
            return {"error": f"Seed {seed_id} not found"}
        ok = self.knowledge_seeds.review(seed_id, decision, promoted_by, review_note)
        if not ok:
            return {"error": "Failed to update seed"}
        return {
            "success": True,
            "seed_id": seed_id,
            "decision": decision,
            "seed_kind": seed.get("seed_kind"),
        }

    def promote_knowledge_seed(
        self,
        seed_kind: str,
        domain_profile_id: str,
        payload_json: str,
        source_matter_id: str | None = None,
    ) -> dict:
        """Create a new knowledge seed from the current matter for cross-matter reuse."""
        seed_id = self.knowledge_seeds.upsert(
            seed_kind=seed_kind,
            domain_profile_id=domain_profile_id,
            payload_json=payload_json,
            source_matter_id=source_matter_id,
        )
        return {"success": True, "seed_id": seed_id, "seed_kind": seed_kind}

    # ------------------------------------------------------------------
    # Objective coverage workbench (SO-4)
    # ------------------------------------------------------------------

    def get_objective_coverage_workbench(self) -> dict:
        """Professional objective coverage dashboard: per-objective criteria,
        support/attack counts, coverage badges, and open gaps."""
        coverage_report = self.get_issue_coverage_report()
        coverage_map = {r["id"]: r for r in coverage_report if isinstance(r, dict) and "id" in r}

        open_issues = self.issues.get_open_issues(min_materiality=0.0)
        objectives = []
        for issue in open_issues:
            if not isinstance(issue, dict):
                continue
            iid = issue.get("id")
            if not iid:
                continue
            cov = coverage_map.get(iid, {})

            predicates = self.issues.get_predicates_by_status(
                iid, statuses=("open", "resolved", "contested", "blocked"),
            )
            pred_total = len(predicates)
            pred_satisfied = sum(1 for p in predicates if isinstance(p, dict) and p.get("status") == "resolved")
            pred_blocked = sum(1 for p in predicates if isinstance(p, dict) and p.get("status") == "blocked")
            pred_contested = sum(1 for p in predicates if isinstance(p, dict) and p.get("status") == "contested")

            gap_rows = self.db.execute(
                """SELECT g.id, g.gap_type, g.description, g.materiality_score
                   FROM gap g JOIN gap_link gl ON gl.gap_id = g.id
                   WHERE g.matter_id=? AND g.status='open'
                     AND gl.affected_type='issue' AND gl.affected_id=?
                   ORDER BY g.materiality_score DESC LIMIT 10""",
                (self.matter_id, iid),
            ).fetchall()
            gaps = [dict(r) for r in gap_rows]

            coverage_frac = float(cov.get("coverage_fraction", 0.0))
            supporting = int(cov.get("supporting_count", 0))
            has_proof_gap = bool(cov.get("has_proof_gap", False))

            if coverage_frac >= 0.8 and not has_proof_gap and pred_blocked == 0:
                badge = "covered"
            elif pred_blocked > 0:
                badge = "blocked"
            elif has_proof_gap or len(gaps) > 0:
                badge = "missing"
            elif pred_contested > 0:
                badge = "contradicted"
            elif coverage_frac >= 0.3:
                badge = "thin"
            else:
                badge = "missing"

            objectives.append({
                "id": iid,
                "title": issue.get("title", ""),
                "issue_type": issue.get("issue_type", ""),
                "materiality": float(issue.get("materiality", 0.5)),
                "salience": float(issue.get("salience", 0.5)),
                "burden_side": issue.get("burden_side"),
                "parent_issue_id": issue.get("parent_issue_id"),
                "coverage_fraction": coverage_frac,
                "coverage_badge": badge,
                "supporting_count": supporting,
                "predicate_total": pred_total,
                "predicate_satisfied": pred_satisfied,
                "predicate_blocked": pred_blocked,
                "predicate_contested": pred_contested,
                "predicates": predicates[:20],
                "gaps": gaps,
                "has_proof_gap": has_proof_gap,
            })

        covered = sum(1 for o in objectives if o["coverage_badge"] == "covered")
        thin = sum(1 for o in objectives if o["coverage_badge"] == "thin")
        blocked = sum(1 for o in objectives if o["coverage_badge"] == "blocked")
        missing = sum(1 for o in objectives if o["coverage_badge"] == "missing")
        contradicted = sum(1 for o in objectives if o["coverage_badge"] == "contradicted")

        return {
            "total": len(objectives),
            "objectives": objectives,
            "summary": {
                "covered": covered,
                "thin": thin,
                "blocked": blocked,
                "missing": missing,
                "contradicted": contradicted,
            },
        }

    # ------------------------------------------------------------------
    # Assumption lifecycle review (SO-3, SO-7)
    # ------------------------------------------------------------------

    def get_assumption_review_workbench(self) -> dict:
        """Dashboard data for assumption lifecycle review: active, confirmed,
        invalidated assumptions with linked target counts."""
        all_assumptions = self.assumptions.get_all()
        provisional = []
        confirmed = []
        invalidated = []
        for a in all_assumptions:
            if not isinstance(a, dict):
                continue
            aid = a.get("id")
            if not aid:
                continue
            targets = self.assumptions.get_linked_targets(aid)
            a["linked_target_count"] = len(targets)
            a["linked_targets"] = targets[:5]
            status = a.get("status", "provisional")
            if status == "provisional":
                provisional.append(a)
            elif status == "confirmed":
                confirmed.append(a)
            elif status == "invalidated":
                invalidated.append(a)
        return {
            "total": len(all_assumptions),
            "provisional": provisional,
            "confirmed": confirmed,
            "invalidated": invalidated,
            "counts": {
                "provisional": len(provisional),
                "confirmed": len(confirmed),
                "invalidated": len(invalidated),
            },
        }

    def review_assumption(
        self,
        assumption_id: str,
        decision: str,
        reason: str | None = None,
    ) -> dict:
        """Review an assumption: confirm, invalidate, or revert to provisional.

        When invalidated, linked predicates are blocked and a gap is recorded.
        """
        valid = ("provisional", "confirmed", "invalidated")
        if decision not in valid:
            return {"error": f"Invalid decision. Must be one of: {', '.join(valid)}"}
        assumption = None
        for a in self.assumptions.get_all():
            if isinstance(a, dict) and a.get("id") == assumption_id:
                assumption = a
                break
        if not assumption:
            return {"error": f"Assumption {assumption_id} not found"}

        ok = self.assumptions.set_status(assumption_id, decision, reason)
        if not ok:
            return {"error": "Failed to update assumption"}

        actions: list[str] = [f"Status → {decision}"]

        if decision == "invalidated":
            targets = self.assumptions.get_linked_targets(assumption_id)
            blocked_count = 0
            for t in targets:
                if not isinstance(t, dict):
                    continue
                if t.get("target_type") == "predicate":
                    try:
                        self.issues.set_predicate_status(
                            t["target_id"], "blocked",
                            reason=f"Assumption invalidated: {reason or 'no reason'}",
                        )
                        blocked_count += 1
                    except Exception as exc:
                        _log.warning("Failed to block predicate %s: %s", t.get("target_id"), exc)
            if blocked_count:
                actions.append(f"Blocked {blocked_count} predicate(s)")
            stmt = assumption.get("statement", "")[:120]
            self.gaps.record(
                gap_type=GapType.MISSING_DOCUMENT,
                description=f"Assumption invalidated: '{stmt}' — {reason or 'no reason'}",
                materiality=0.7,
                affected_type="assumption",
                affected_id=assumption_id,
            )
            actions.append("Recorded gap for invalidated assumption")

        return {
            "success": True,
            "assumption_id": assumption_id,
            "decision": decision,
            "actions": actions,
        }

    # ------------------------------------------------------------------
    # Assertion trace with impact (SO-2, SO-3, SO-5)
    # ------------------------------------------------------------------

    def get_assertion_trace(self, assertion_id: str) -> dict:
        """Full impact trace for a single assertion: source documents,
        affected issues, dependent assertions, verification status,
        and belief revision history."""
        record = self.assertions.get(assertion_id)
        if not record:
            return {"error": "Assertion not found", "assertion_id": assertion_id}

        a = record.__dict__ if hasattr(record, "__dict__") else dict(record)

        occurrences = self.assertions.get_occurrences(assertion_id)
        source_docs: list[dict] = []
        for occ in occurrences:
            if not isinstance(occ, dict):
                continue
            doc_id = occ.get("document_inventory_id") or occ.get("document_id")
            span_id = occ.get("span_id")
            doc_label = doc_id
            section_label = span_id
            if occ.get("document_inventory_id"):
                try:
                    inv = self.inventory.get_by_id(occ["document_inventory_id"])
                    if inv:
                        doc_label = inv.get("relative_path") or doc_id
                except Exception as exc:
                    _log.warning("assertion_trace: inventory lookup failed for %s: %s", doc_id, exc)
            if span_id:
                try:
                    span_row = self.db.execute(
                        "SELECT section_ref, clause_ref FROM span WHERE id=?",
                        (span_id,),
                    ).fetchone()
                    if span_row:
                        section_label = span_row["section_ref"] or span_row["clause_ref"] or span_id
                except Exception as exc:
                    _log.warning("assertion_trace: span lookup failed for %s: %s", span_id, exc)
            source_docs.append({
                "document_id": doc_id,
                "document_label": doc_label,
                "span_id": span_id,
                "section_label": section_label,
            })

        affected_issue_ids = self._issues_affected_by_target("assertion", assertion_id)
        affected_issues: list[dict] = []
        for iid in affected_issue_ids:
            try:
                iss = self.issues.get(iid)
                if iss:
                    affected_issues.append({
                        "id": iid,
                        "title": iss.get("title", "") if hasattr(iss, "get") else getattr(iss, "title", ""),
                        "materiality": iss.get("materiality", 0) if hasattr(iss, "get") else getattr(iss, "materiality", 0),
                    })
            except Exception:
                affected_issues.append({"id": iid, "title": "", "materiality": 0})

        dependent_ids = self.assertions.get_dependents(assertion_id)
        dependents: list[dict] = []
        for did in dependent_ids[:10]:
            dep = self.assertions.get(did)
            if dep:
                d = dep.__dict__ if hasattr(dep, "__dict__") else dict(dep)
                dependents.append({
                    "id": did,
                    "proposition_text": d.get("proposition_text", ""),
                    "belief_state": d.get("belief_state", ""),
                })

        verification = {}
        try:
            vs_row = self.verification.get_status(assertion_id, "assertion")
            if vs_row:
                verification = dict(vs_row) if hasattr(vs_row, "keys") else {}
        except Exception as exc:
            _log.warning("assertion_trace: verification lookup failed for %s: %s", assertion_id, exc)

        revisions: list[dict] = []
        try:
            rev_rows = self.db.execute(
                """SELECT old_state, new_state, cause, explanation, created_at
                   FROM belief_revision_event
                   WHERE assertion_id=?
                   ORDER BY created_at DESC LIMIT 10""",
                (assertion_id,),
            ).fetchall()
            revisions = [dict(r) for r in rev_rows]
        except Exception as exc:
            _log.warning("assertion_trace: revision history failed for %s: %s", assertion_id, exc)

        return {
            "assertion_id": assertion_id,
            "proposition_text": a.get("proposition_text", ""),
            "belief_state": a.get("belief_state", ""),
            "confidence": a.get("confidence", 0),
            "speech_act": a.get("speech_act", ""),
            "source_documents": source_docs,
            "affected_issues": affected_issues,
            "dependent_assertions": dependents,
            "verification": verification,
            "revision_history": revisions,
            "impact_summary": {
                "issues_affected": len(affected_issues),
                "dependents_count": len(dependent_ids),
                "source_doc_count": len(source_docs),
                "revision_count": len(revisions),
            },
        }

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

    def escalate_gap(self, gap_id: str) -> bool:
        return self.gaps.escalate_gap(gap_id)

    def get_gap_workbench(
        self,
        min_materiality: float = 0.0,
        limit: int = 50,
    ) -> dict:
        return {
            "matter_id": self.matter_id,
            "items": self.gaps.workbench(
                min_materiality=min_materiality,
                limit=limit,
            ),
        }

    # ------------------------------------------------------------------
    # Document intelligence (cards + spans)
    def get_issue_closure_workbench(self, issue_id: str) -> dict:
        """Consolidated issue closure surface: proof state + coverage +
        gaps + verification status + source agreement (SO-2, SO-3, SO-4, SO-7).

        Returns everything a professional needs to decide if an issue is
        ready to rely on and what must happen next."""
        issue_row = self.issues.get(issue_id)
        if not issue_row:
            return {"error": "Issue not found", "issue_id": issue_id}

        issue = dict(issue_row)

        coverage_rows = self.get_issue_coverage_report()
        coverage = next(
            (r for r in coverage_rows if isinstance(r, dict) and r.get("id") == issue_id),
            {},
        )

        proof = {}
        try:
            proof_row = self.proof_state.get(issue_id)
            if proof_row:
                proof = dict(proof_row) if hasattr(proof_row, "keys") else {}
        except Exception as exc:
            _log.warning("issue_closure: proof_state lookup failed for %s: %s", issue_id, exc)

        issue_gaps = []
        try:
            issue_gaps = self.gaps.gaps_for_issue(issue_id)
        except Exception as exc:
            _log.warning("issue_closure: gap query failed for %s: %s", issue_id, exc)

        pending_count = 0
        verified_count = 0
        try:
            assertions = self.issues.get_assertions_for_issue(issue_id)
            for a in assertions:
                if not isinstance(a, dict):
                    continue
                vs = (a.get("verification_status") or "candidate").lower()
                if vs == "verified":
                    verified_count += 1
                elif vs == "candidate":
                    pending_count += 1
        except Exception as exc:
            _log.warning("issue_closure: assertion query failed for %s: %s", issue_id, exc)

        source_agreement = []
        try:
            source_agreement = self.get_source_agreement_for_issue(issue_id)
        except Exception as exc:
            _log.warning("issue_closure: source_agreement failed for %s: %s", issue_id, exc)

        blockers: list[str] = []
        coverage_frac = float(coverage.get("coverage_fraction", 0)) if coverage else 0
        if coverage_frac < 0.5:
            blockers.append("Low evidence coverage")
        if issue_gaps:
            blockers.append(f"{len(issue_gaps)} open gap(s)")
        if pending_count > 0 and verified_count == 0:
            blockers.append("No verified supporting facts")
        if coverage.get("has_proof_gap"):
            blockers.append("Missing element of proof")

        readiness = "ready" if not blockers else "blocked"

        return {
            "issue_id": issue_id,
            "title": issue.get("title", ""),
            "status": issue.get("status", ""),
            "materiality": issue.get("materiality", 0),
            "salience": issue.get("salience", 0),
            "coverage_fraction": coverage_frac,
            "supporting_count": coverage.get("supporting_count", 0),
            "predicate_count": coverage.get("predicate_count", 0),
            "verified_count": verified_count,
            "pending_count": pending_count,
            "proof_state": proof,
            "gaps": issue_gaps,
            "source_agreement": source_agreement,
            "blockers": blockers,
            "readiness": readiness,
        }

    def get_investigation_readiness(self, run_id: Optional[str] = None) -> dict:
        """Matter-wide readiness assessment: what blocks confident reliance
        on this investigation (SO-3, SO-7).

        Aggregates: coverage report, proof state, open gaps, unresolved
        contradictions, pending clarifications, and review queue depth into
        a single readiness surface with explicit blocker list."""
        coverage_rows = self.get_issue_coverage_report()
        proof_summary = self.proof_state.get_summary()

        high_mat_low_coverage: list[dict] = []
        candidate_only_critical: list[dict] = []
        proof_gap_issues: list[dict] = []
        for row in coverage_rows:
            if not isinstance(row, dict):
                continue
            mat = float(row.get("materiality", 0))
            cov = float(row.get("coverage_fraction", 0))
            if mat >= 0.5 and cov < 0.5:
                high_mat_low_coverage.append({
                    "issue_id": row.get("id", ""),
                    "title": row.get("title", ""),
                    "materiality": mat,
                    "coverage_fraction": cov,
                })
            if row.get("has_proof_gap"):
                proof_gap_issues.append({
                    "issue_id": row.get("id", ""),
                    "title": row.get("title", ""),
                    "gap_id": row.get("gap_id"),
                })
            verified = int(row.get("verified_count", 0))
            pending = int(row.get("candidate_count", row.get("supporting_count", 0)))
            if mat >= 0.7 and verified == 0 and pending > 0:
                candidate_only_critical.append({
                    "issue_id": row.get("id", ""),
                    "title": row.get("title", ""),
                    "pending_count": pending,
                })

        open_gap_count = self.gaps.count_open()
        high_mat_gaps = self.gaps.open_gaps(min_materiality=0.7, limit=20)

        contradiction_count = self.assertions.count_contradictions()
        pending_clarifications = self.clarifications.count_pending()

        review_counts = self.count_review_queue()
        pending_review = review_counts.get("total", 0) if isinstance(review_counts, dict) else 0

        blockers: list[dict] = []
        if high_mat_low_coverage:
            blockers.append({
                "type": "low_coverage",
                "severity": "high",
                "label": f"{len(high_mat_low_coverage)} high-materiality issue(s) below 50% coverage",
                "items": high_mat_low_coverage,
            })
        if proof_gap_issues:
            blockers.append({
                "type": "proof_gap",
                "severity": "high",
                "label": f"{len(proof_gap_issues)} issue(s) with missing element of proof",
                "items": proof_gap_issues,
            })
        if candidate_only_critical:
            blockers.append({
                "type": "unverified_critical",
                "severity": "medium",
                "label": f"{len(candidate_only_critical)} critical issue(s) with no verified facts",
                "items": candidate_only_critical,
            })
        if contradiction_count > 0:
            blockers.append({
                "type": "contradictions",
                "severity": "medium" if contradiction_count <= 3 else "high",
                "label": f"{contradiction_count} unresolved contradiction(s)",
                "items": [],
            })
        if pending_clarifications > 0:
            blockers.append({
                "type": "pending_clarifications",
                "severity": "low",
                "label": f"{pending_clarifications} pending clarification(s)",
                "items": [],
            })
        if len(high_mat_gaps) > 0:
            blockers.append({
                "type": "high_materiality_gaps",
                "severity": "high",
                "label": f"{len(high_mat_gaps)} high-materiality gap(s) remain open",
                "items": [{"gap_id": g["id"], "description": g.get("description", "")}
                          for g in high_mat_gaps if isinstance(g, dict)],
            })

        readiness = "ready" if not blockers else "blocked"
        if readiness == "blocked" and all(b["severity"] == "low" for b in blockers):
            readiness = "caution"

        return {
            "matter_id": self.matter_id,
            "readiness": readiness,
            "blocker_count": len(blockers),
            "blockers": blockers,
            "summary": {
                "issue_count": len(coverage_rows),
                "avg_coverage": round(
                    sum(float(r.get("coverage_fraction", 0)) for r in coverage_rows if isinstance(r, dict))
                    / max(len(coverage_rows), 1), 3
                ),
                "proof_summary": proof_summary,
                "open_gap_count": open_gap_count,
                "contradiction_count": contradiction_count,
                "pending_clarifications": pending_clarifications,
                "pending_review": pending_review,
            },
        }

    # ------------------------------------------------------------------
    # SO-3/SO-4: output quality contract workbench
    # ------------------------------------------------------------------

    def get_output_quality_workbench(self, run_id: Optional[str] = None) -> dict:
        """Aggregate output quality signals into a professional review surface.

        Combines: latest run session metadata, ledger completion events,
        readiness blockers, manifest freshness, and LLM efficiency metrics
        into a single workbench for domain professionals to assess whether
        an investigation's output is reliance-ready.
        """
        runs = self.ledger.recent_runs(limit=5)
        run_summaries: list[dict] = []
        for r in runs:
            if not isinstance(r, dict):
                continue
            rid = r.get("id", "")
            status = r.get("status", "unknown")
            started = r.get("started_at", "")
            completed = r.get("completed_at", "")
            query = (r.get("query") or "")[:100]
            op_type = r.get("operation_type", "query")
            research_mode = r.get("research_mode", "deep")

            completion_summary = ""
            event_count = 0
            try:
                events = self.ledger.get_events(rid)
                event_count = len(events)
                for e in reversed(events):
                    if not isinstance(e, dict):
                        continue
                    if e.get("event_type") in ("run_completed", "run_failed"):
                        completion_summary = (e.get("summary") or "")[:200]
                        break
            except Exception:
                pass

            llm_avoided = r.get("llm_calls_avoided") or 0
            llm_required = r.get("llm_calls_required") or 0
            llm_total = llm_avoided + llm_required
            reuse_rate = round(llm_avoided / max(llm_total, 1), 3)

            run_summaries.append({
                "run_id": rid,
                "status": status,
                "started_at": started,
                "completed_at": completed,
                "query": query,
                "operation_type": op_type,
                "research_mode": research_mode,
                "event_count": event_count,
                "completion_summary": completion_summary,
                "llm_calls_avoided": llm_avoided,
                "llm_calls_required": llm_required,
                "cache_reuse_rate": reuse_rate,
            })

        readiness = self.get_investigation_readiness(run_id=run_id)

        manifest_fresh = False
        manifest_count = 0
        stale_count = 0
        try:
            manifests = self.memory_broker.list_recent_manifests(limit=10)
            manifest_count = len(manifests)
            for m in manifests:
                if not isinstance(m, dict):
                    continue
                mh = m.get("manifest_hash", "")
                try:
                    validation = self.memory_broker.validate_dependency_manifest(mh)
                    if validation and getattr(validation, "valid", False):
                        manifest_fresh = True
                    else:
                        stale_count += 1
                except Exception:
                    stale_count += 1
        except Exception:
            pass

        obligations: list[dict] = []
        blocker_types = set()
        for b in readiness.get("blockers", []):
            if not isinstance(b, dict):
                continue
            btype = b.get("type", "unknown")
            blocker_types.add(btype)
            obligations.append({
                "name": b.get("label", btype),
                "satisfied": False,
                "severity": b.get("severity", "medium"),
                "item_count": len(b.get("items", [])),
            })

        standard_checks = [
            ("no_contradictions", "No unresolved contradictions", "contradictions"),
            ("no_proof_gaps", "No missing elements of proof", "proof_gap"),
            ("coverage_adequate", "All issues above 50% coverage", "low_coverage"),
            ("no_high_mat_gaps", "No high-materiality gaps open", "high_materiality_gaps"),
            ("clarifications_answered", "All clarifications answered", "pending_clarifications"),
            ("critical_facts_verified", "Critical facts have verified evidence", "unverified_critical"),
        ]
        for check_id, label, btype in standard_checks:
            if btype not in blocker_types:
                obligations.append({
                    "name": label,
                    "satisfied": True,
                    "severity": "passed",
                    "item_count": 0,
                })

        return {
            "matter_id": self.matter_id,
            "readiness": readiness.get("readiness", "unknown"),
            "blocker_count": readiness.get("blocker_count", 0),
            "runs": run_summaries,
            "obligations": obligations,
            "manifest_count": manifest_count,
            "manifest_fresh": manifest_fresh,
            "stale_manifest_count": stale_count,
            "summary": readiness.get("summary", {}),
        }

    # ------------------------------------------------------------------
    # SO-3/SO-4: deliverable preparation workbench
    # ------------------------------------------------------------------

    def get_deliverable_workbench(self, issue_ids: list[str] | None = None) -> dict:
        """Assemble inputs for a professional deliverable: verified issues,
        supporting assertions, source citations, and a reliance gate.

        Returns the building blocks a UI needs to let a domain professional
        preview, scope, and export a memo/letter/outline from the matter model.
        """
        readiness = self.get_investigation_readiness()
        ready = readiness.get("readiness", "unknown")
        blockers = readiness.get("blockers", [])

        all_issues = self.issues.get_open_issues(min_materiality=0.0)
        if issue_ids:
            scope_set = set(issue_ids)
            scoped = [i for i in all_issues if isinstance(i, dict) and i.get("id") in scope_set]
        else:
            scoped = [i for i in all_issues if isinstance(i, dict)]

        issue_sections: list[dict] = []
        for iss in scoped:
            if not isinstance(iss, dict):
                continue
            iid = iss.get("id", "")
            title = (iss.get("title") or "")[:120]
            materiality = iss.get("materiality", 0.0)

            verified_assertions: list[dict] = []
            try:
                edges = self.evidence.list_edges_for_target("issue", iid)
                for edge in edges:
                    if not isinstance(edge, dict):
                        continue
                    aid = edge.get("source_id", "")
                    if not aid:
                        continue
                    arow = self.db.execute(
                        "SELECT proposition_text, belief_state FROM assertion WHERE id=?",
                        (aid,),
                    ).fetchone()
                    if not arow:
                        continue
                    vrow = self.db.execute(
                        "SELECT status FROM verification_state WHERE target_id=? AND target_kind='assertion' ORDER BY updated_at DESC LIMIT 1",
                        (aid,),
                    ).fetchone()
                    v_status = vrow["status"] if vrow else "candidate"
                    if v_status != "verified":
                        continue
                    verified_assertions.append({
                        "assertion_id": aid,
                        "proposition": (arow["proposition_text"] or "")[:200],
                        "belief_state": arow["belief_state"],
                    })
            except Exception as exc:
                _log.warning("deliverable: evidence lookup failed for issue %s: %s", iid, exc)

            source_docs: list[str] = []
            try:
                for va in verified_assertions[:20]:
                    occs = self.db.execute(
                        "SELECT DISTINCT d.relative_path FROM assertion_occurrence ao "
                        "JOIN document_inventory d ON d.id = ao.document_id "
                        "WHERE ao.assertion_id=? LIMIT 5",
                        (va["assertion_id"],),
                    ).fetchall()
                    for occ in occs:
                        path = occ["relative_path"]
                        if path and path not in source_docs:
                            source_docs.append(path)
            except Exception as exc:
                _log.warning("deliverable: source doc lookup failed for issue %s: %s", iid, exc)

            issue_sections.append({
                "issue_id": iid,
                "title": title,
                "materiality": materiality,
                "verified_assertion_count": len(verified_assertions),
                "verified_assertions": verified_assertions[:10],
                "source_documents": source_docs[:10],
            })

        total_verified = sum(s.get("verified_assertion_count", 0) for s in issue_sections)
        total_sources = len({
            doc for s in issue_sections for doc in s.get("source_documents", [])
        })

        return {
            "matter_id": self.matter_id,
            "reliance_gate": ready,
            "blocker_count": len(blockers),
            "blockers": blockers[:5],
            "issue_count": len(issue_sections),
            "issues": issue_sections,
            "total_verified_assertions": total_verified,
            "total_source_documents": total_sources,
        }

    def compile_issue_brief(
        self,
        *,
        issue_ids: list[str] | None = None,
        policy_audience: str = "clean",
        include_gaps: bool = True,
        include_contradictions: bool = True,
        include_quant: bool = True,
    ) -> dict:
        """Compile a structured issue brief from matter model state.

        Gathers assertions (all belief states, not just verified), evidence
        edges, proof gaps, contradictions, quant facts, and source citations
        per issue.  Returns structured sections with provenance links suitable
        for rendering as a professional deliverable.
        """
        _, _, primary_profile = self._read_matter_domain_composition()
        domain = primary_profile or "legal"

        readiness = self.get_investigation_readiness()
        reliance_gate = readiness.get("readiness", "unknown")

        all_issues = self.issues.get_open_issues(min_materiality=0.0)
        if issue_ids:
            scope_set = set(issue_ids)
            scoped = [i for i in all_issues if isinstance(i, dict) and i.get("id") in scope_set]
        else:
            scoped = [i for i in all_issues if isinstance(i, dict)]

        sections: list[dict] = []
        total_assertions = 0
        total_gaps = 0
        total_contradictions = 0
        all_source_docs: set[str] = set()

        for iss in scoped:
            if not isinstance(iss, dict):
                continue
            iid = iss.get("id", "")
            title = (iss.get("title") or "")[:200]
            materiality = iss.get("materiality", 0.0)

            assertions: list[dict] = []
            source_docs: list[str] = []
            try:
                edges = self.evidence.list_edges_for_target("issue", iid)
                for edge in edges:
                    if not isinstance(edge, dict):
                        continue
                    aid = edge.get("source_id", "")
                    if not aid:
                        continue
                    arow = self.db.execute(
                        "SELECT id, proposition_text, belief_state, confidence"
                        " FROM assertion WHERE id=?",
                        (aid,),
                    ).fetchone()
                    if not arow:
                        continue
                    if arow["belief_state"] in ("superseded", "withdrawn"):
                        continue

                    occ_rows = self.db.execute(
                        "SELECT ao.source_role, d.relative_path"
                        " FROM assertion_occurrence ao"
                        " LEFT JOIN document_inventory d ON d.id = ao.document_id"
                        " WHERE ao.assertion_id=? LIMIT 3",
                        (aid,),
                    ).fetchall()
                    roles = set()
                    for occ in occ_rows:
                        if occ["relative_path"]:
                            p = occ["relative_path"]
                            source_docs.append(p)
                            all_source_docs.add(p)
                        if occ["source_role"]:
                            roles.add(occ["source_role"])

                    assertions.append({
                        "assertion_id": aid,
                        "proposition": (arow["proposition_text"] or "")[:300],
                        "belief_state": arow["belief_state"],
                        "confidence": arow["confidence"],
                        "source_roles": sorted(roles),
                        "edge_type": edge.get("edge_type", "supports"),
                    })
            except Exception as exc:
                _log.warning("compile_issue_brief: edge lookup failed for %s: %s", iid, exc)

            total_assertions += len(assertions)

            gaps: list[dict] = []
            if include_gaps:
                try:
                    issue_gaps = self.gaps.gaps_for_issue(iid)
                    for g in issue_gaps:
                        if not isinstance(g, dict):
                            continue
                        gaps.append({
                            "gap_id": g.get("id", ""),
                            "gap_type": g.get("gap_type", ""),
                            "description": (g.get("description") or "")[:200],
                            "materiality_score": g.get("materiality_score", 0.0),
                        })
                    total_gaps += len(gaps)
                except Exception as exc:
                    _log.warning("compile_issue_brief: gap lookup failed for %s: %s", iid, exc)

            contradictions: list[dict] = []
            if include_contradictions:
                try:
                    aid_set = {a["assertion_id"] for a in assertions}
                    if aid_set:
                        all_contradictions = self.assertions.find_contradictions(limit=50)
                        for c in all_contradictions:
                            if not isinstance(c, dict):
                                continue
                            if c.get("attacker_id") in aid_set or c.get("attacked_id") in aid_set:
                                contradictions.append({
                                    "attacker_id": c.get("attacker_id", ""),
                                    "attacked_id": c.get("attacked_id", ""),
                                    "attacker_prop": (c.get("attacker_prop") or "")[:150],
                                    "attacked_prop": (c.get("attacked_prop") or "")[:150],
                                })
                    total_contradictions += len(contradictions)
                except Exception as exc:
                    _log.warning("compile_issue_brief: contradiction lookup for %s: %s", iid, exc)

            supporting = [a for a in assertions if a.get("edge_type") == "supports"]
            attacking = [a for a in assertions if a.get("edge_type") in ("attacks", "contradicts")]

            sections.append({
                "issue_id": iid,
                "title": title,
                "materiality": materiality,
                "assertion_count": len(assertions),
                "supporting_count": len(supporting),
                "attacking_count": len(attacking),
                "assertions": assertions[:20],
                "source_documents": sorted(set(source_docs))[:15],
                "gaps": gaps,
                "contradictions": contradictions,
            })

        return {
            "matter_id": self.matter_id,
            "domain": domain,
            "reliance_gate": reliance_gate,
            "section_count": len(sections),
            "total_assertions": total_assertions,
            "total_gaps": total_gaps,
            "total_contradictions": total_contradictions,
            "total_source_documents": len(all_source_docs),
            "sections": sections,
        }

    # ------------------------------------------------------------------
    # Scenario branches (SO-1, SO-3)
    # ------------------------------------------------------------------

    def create_scenario_branch(
        self,
        *,
        name: str,
        assumptions: list[dict],
        objective_ids: list[str] | None = None,
        source_branch_id: str | None = None,
        notes: str = "",
    ) -> dict:
        """Create a persistent counterfactual branch over this matter."""
        branch_id = str(uuid.uuid4())
        now = _now()
        assumptions_json = json.dumps(assumptions, default=str)
        obj_ids_json = json.dumps(objective_ids or [])
        self.db.execute(
            """INSERT INTO scenario_branch
               (id, matter_id, name, status, assumptions_json, objective_ids_json,
                source_branch_id, notes, created_at, updated_at)
               VALUES (?, ?, ?, 'active', ?, ?, ?, ?, ?, ?)""",
            (branch_id, self.matter_id, name, assumptions_json, obj_ids_json,
             source_branch_id, notes, now, now),
        )
        self.db.conn.commit()
        return {
            "branch_id": branch_id,
            "name": name,
            "status": "active",
            "assumptions": assumptions,
            "objective_ids": objective_ids or [],
            "source_branch_id": source_branch_id,
            "notes": notes,
            "created_at": now,
        }

    def list_scenario_branches(self) -> list[dict]:
        """Return all scenario branches for this matter."""
        rows = self.db.execute(
            """SELECT * FROM scenario_branch
               WHERE matter_id=? ORDER BY created_at DESC""",
            (self.matter_id,),
        ).fetchall()
        result = []
        for row in rows:
            if not isinstance(row, (dict, sqlite3.Row)):
                continue
            d = dict(row)
            try:
                d["assumptions"] = json.loads(d.pop("assumptions_json", "[]"))
            except (TypeError, ValueError) as exc:
                _log.warning("list_scenario_branches: malformed assumptions_json for %s: %s", d.get("id", "?"), exc)
                d["assumptions"] = []
            try:
                d["objective_ids"] = json.loads(d.pop("objective_ids_json", "[]"))
            except (TypeError, ValueError) as exc:
                _log.warning("list_scenario_branches: malformed objective_ids_json for %s: %s", d.get("id", "?"), exc)
                d["objective_ids"] = []
            result.append(d)
        return result

    def get_scenario_branch(self, branch_id: str) -> dict | None:
        """Return a single scenario branch by ID."""
        row = self.db.execute(
            "SELECT * FROM scenario_branch WHERE id=? AND matter_id=?",
            (branch_id, self.matter_id),
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        try:
            d["assumptions"] = json.loads(d.pop("assumptions_json", "[]"))
        except (TypeError, ValueError):
            d["assumptions"] = []
        try:
            d["objective_ids"] = json.loads(d.pop("objective_ids_json", "[]"))
        except (TypeError, ValueError):
            d["objective_ids"] = []
        return d

    def archive_scenario_branch(self, branch_id: str) -> bool:
        """Archive (soft-delete) a scenario branch. Returns True if found."""
        cursor = self.db.execute(
            "UPDATE scenario_branch SET status='archived', updated_at=? WHERE id=? AND matter_id=?",
            (_now(), branch_id, self.matter_id),
        )
        self.db.conn.commit()
        return cursor.rowcount > 0

    def get_scenario_workbench(self) -> dict:
        """Assemble scenario branch summary for UI display."""
        branches = self.list_scenario_branches()
        active = [b for b in branches if isinstance(b, dict) and b.get("status") == "active"]
        archived = [b for b in branches if isinstance(b, dict) and b.get("status") == "archived"]
        return {
            "matter_id": self.matter_id,
            "total_branches": len(branches),
            "active_count": len(active),
            "archived_count": len(archived),
            "branches": active,
        }

    # ------------------------------------------------------------------
    # Durable Scenario Graphs (SO-1, SO-3)
    # ------------------------------------------------------------------

    _DELTA_OPERATIONS = frozenset({
        "override_belief", "suppress", "add_gap", "resolve_gap",
        "add_assertion", "assume",
    })
    _DELTA_KINDS = frozenset({
        "assertion", "issue", "predicate", "quant_fact", "authority", "gap",
    })

    def apply_scenario_delta(
        self,
        branch_id: str,
        target_kind: str,
        target_id: str,
        operation: str,
        payload: dict | None = None,
    ) -> dict:
        """Apply a delta to a scenario branch without mutating baseline state."""
        branch = self.get_scenario_branch(branch_id)
        if not branch:
            return {"error": f"Branch {branch_id} not found"}
        if branch.get("status") != "active":
            return {"error": f"Branch {branch_id} is not active"}
        if target_kind not in self._DELTA_KINDS:
            return {"error": f"Invalid target_kind. Must be one of: {', '.join(sorted(self._DELTA_KINDS))}"}
        if operation not in self._DELTA_OPERATIONS:
            return {"error": f"Invalid operation. Must be one of: {', '.join(sorted(self._DELTA_OPERATIONS))}"}

        delta_id = str(uuid.uuid4())
        now = _now()
        self.db.execute(
            """INSERT INTO scenario_branch_delta
               (id, branch_id, target_kind, target_id, operation, payload_json,
                created_at, created_by)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'user')""",
            (delta_id, branch_id, target_kind, target_id, operation,
             json.dumps(payload or {}, default=str), now),
        )

        delta_counts = self.db.execute(
            "SELECT COUNT(*) AS n FROM scenario_branch_delta WHERE branch_id=?",
            (branch_id,),
        ).fetchone()
        dc = int(delta_counts["n"]) if delta_counts else 1

        kind_counts = {}
        for kind in ("assertion", "gap", "quant_fact"):
            row = self.db.execute(
                "SELECT COUNT(*) AS n FROM scenario_branch_delta"
                " WHERE branch_id=? AND target_kind=?",
                (branch_id, kind),
            ).fetchone()
            kind_counts[kind] = int(row["n"]) if row else 0

        self.db.execute(
            "UPDATE scenario_branch SET assertion_delta=?, gap_delta=?, quant_delta=?,"
            " updated_at=? WHERE id=?",
            (kind_counts.get("assertion", 0), kind_counts.get("gap", 0),
             kind_counts.get("quant_fact", 0), now, branch_id),
        )
        self.db.conn.commit()

        return {
            "delta_id": delta_id,
            "branch_id": branch_id,
            "operation": operation,
            "target_kind": target_kind,
            "target_id": target_id,
            "total_deltas": dc,
        }

    def list_scenario_deltas(self, branch_id: str) -> list[dict]:
        """Return all deltas for a scenario branch."""
        rows = self.db.execute(
            """SELECT id, branch_id, target_kind, target_id, operation,
                      payload_json, created_at, created_by
               FROM scenario_branch_delta
               WHERE branch_id=?
               ORDER BY created_at""",
            (branch_id,),
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            try:
                d["payload"] = json.loads(d.pop("payload_json", "{}"))
            except (json.JSONDecodeError, TypeError) as exc:
                _log.warning(
                    "list_scenario_deltas: malformed payload_json for delta %s on branch %s: %s",
                    d.get("id", "?"), branch_id, exc,
                )
                d["payload"] = {}
            result.append(d)
        return result

    def compute_scenario_snapshot(self, branch_id: str) -> dict:
        """Compute a snapshot comparing branch state against baseline.

        Evaluates each delta to project changes to issue coverage, gaps, and
        SO metrics without mutating the actual matter state. Returns a snapshot
        record with baseline vs branch comparison data.
        """
        branch = self.get_scenario_branch(branch_id)
        if not branch:
            return {"error": f"Branch {branch_id} not found"}

        deltas = self.list_scenario_deltas(branch_id)

        warnings: list[str] = []

        baseline_coverage = {}
        try:
            cov = self.get_objective_coverage_workbench()
            baseline_coverage = {
                "total_issues": cov.get("total_issues", 0),
                "covered_issues": cov.get("fully_covered", 0),
                "partial_issues": cov.get("partially_covered", 0),
            }
        except Exception as exc:
            _log.warning("compute_scenario_snapshot: coverage failed: %s", exc)
            warnings.append(f"coverage unavailable: {exc}")

        baseline_gaps = {}
        try:
            gap_count = self.gaps.count_open()
            baseline_gaps = {"open_gaps": gap_count}
        except Exception as exc:
            _log.warning("compute_scenario_snapshot: gap count failed: %s", exc)
            warnings.append(f"gap count unavailable: {exc}")

        baseline_so = {}
        try:
            so = self.get_so_metrics()
            if isinstance(so, dict):
                baseline_so = {k: v for k, v in so.items() if isinstance(v, (int, float, str, bool))}
        except Exception as exc:
            _log.warning("compute_scenario_snapshot: SO metrics failed: %s", exc)
            warnings.append(f"SO metrics unavailable: {exc}")

        branch_coverage = dict(baseline_coverage)
        branch_gaps = dict(baseline_gaps)
        overridden_beliefs: dict[str, str] = {}
        suppressed_ids: set[str] = set()
        new_gaps: list[str] = []
        resolved_gaps: list[str] = []
        new_assertions: list[str] = []

        for delta in deltas:
            if not isinstance(delta, dict):
                continue
            op = delta.get("operation", "")
            tkind = delta.get("target_kind", "")
            tid = delta.get("target_id", "")
            payload = delta.get("payload", {})

            if op == "override_belief" and tkind == "assertion":
                overridden_beliefs[tid] = str(payload.get("new_belief", ""))
            elif op == "suppress":
                suppressed_ids.add(tid)
            elif op == "add_gap":
                new_gaps.append(tid)
                branch_gaps["open_gaps"] = branch_gaps.get("open_gaps", 0) + 1
            elif op == "resolve_gap" and tkind == "gap":
                resolved_gaps.append(tid)
                branch_gaps["open_gaps"] = max(0, branch_gaps.get("open_gaps", 0) - 1)
            elif op == "add_assertion":
                new_assertions.append(tid)

        snapshot_id = str(uuid.uuid4())
        now = _now()

        coverage_json = json.dumps({
            "baseline": baseline_coverage,
            "branch": branch_coverage,
        }, default=str)
        gap_json = json.dumps({
            "baseline": baseline_gaps,
            "branch": branch_gaps,
            "new_gaps": new_gaps,
            "resolved_gaps": resolved_gaps,
        }, default=str)
        so_json = json.dumps({"baseline": baseline_so}, default=str)

        self.db.execute(
            """INSERT INTO scenario_branch_snapshot
               (id, branch_id, baseline_manifest, branch_manifest,
                coverage_json, gap_json, so_metrics_json, delta_count, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (snapshot_id, branch_id, "", "", coverage_json, gap_json,
             so_json, len(deltas), now),
        )
        self.db.conn.commit()

        return {
            "snapshot_id": snapshot_id,
            "branch_id": branch_id,
            "branch_name": branch.get("name", ""),
            "delta_count": len(deltas),
            "overridden_beliefs": overridden_beliefs,
            "suppressed_ids": list(suppressed_ids),
            "new_gaps": new_gaps,
            "resolved_gaps": resolved_gaps,
            "new_assertions": new_assertions,
            "coverage": {"baseline": baseline_coverage, "branch": branch_coverage},
            "gaps": {"baseline": baseline_gaps, "branch": branch_gaps},
            "so_metrics": {"baseline": baseline_so},
            "warnings": warnings,
        }

    def list_scenario_snapshots(self, branch_id: str, limit: int = 10) -> list[dict]:
        """Return snapshot history for a scenario branch from scenario_branch_snapshot."""
        rows = self.db.execute(
            """SELECT id, branch_id, delta_count, created_at
               FROM scenario_branch_snapshot
               WHERE branch_id=?
               ORDER BY created_at DESC
               LIMIT ?""",
            (branch_id, max(1, min(limit, 100))),
        ).fetchall()
        result = []
        for r in rows:
            if not isinstance(r, (dict, sqlite3.Row)):
                _log.warning("list_scenario_snapshots: unexpected row type %s", type(r).__name__)
                continue
            result.append(dict(r))
        return result

    def compare_scenario_to_baseline(self, branch_id: str) -> dict:
        """Side-by-side comparison of a scenario branch vs baseline state.

        Returns structured diff showing what changed and what the impact is,
        suitable for professional decision-making.
        """
        branch = self.get_scenario_branch(branch_id)
        if not branch:
            return {"error": f"Branch {branch_id} not found"}

        deltas = self.list_scenario_deltas(branch_id)

        belief_changes: list[dict] = []
        suppressions: list[dict] = []
        new_gaps: list[dict] = []
        resolved_gaps_list: list[dict] = []
        new_assertions_list: list[dict] = []

        for delta in deltas:
            if not isinstance(delta, dict):
                continue
            op = delta.get("operation", "")
            tkind = delta.get("target_kind", "")
            tid = delta.get("target_id", "")
            payload = delta.get("payload", {})

            if op == "override_belief" and tkind == "assertion":
                current = self.assertions.get(tid)
                belief_changes.append({
                    "assertion_id": tid,
                    "proposition": (current.proposition_text if current else "")[:200],
                    "baseline_belief": current.belief_state if current else "unknown",
                    "branch_belief": str(payload.get("new_belief", "")),
                })
            elif op == "suppress":
                label = ""
                if tkind == "assertion":
                    rec = self.assertions.get(tid)
                    label = (rec.proposition_text if rec else "")[:200]
                suppressions.append({
                    "target_kind": tkind,
                    "target_id": tid,
                    "label": label,
                })
            elif op == "add_gap":
                new_gaps.append({
                    "target_id": tid,
                    "description": str(payload.get("description", ""))[:200],
                    "gap_type": str(payload.get("gap_type", "")),
                })
            elif op == "resolve_gap":
                resolved_gaps_list.append({
                    "gap_id": tid,
                    "reason": str(payload.get("reason", ""))[:200],
                })
            elif op == "add_assertion":
                new_assertions_list.append({
                    "assertion_id": tid,
                    "proposition": str(payload.get("proposition", ""))[:200],
                    "belief_state": str(payload.get("belief_state", "provisional")),
                })

        raw = branch.get("assumptions", [])
        assumptions = [a for a in raw if isinstance(a, dict)] if isinstance(raw, list) else []

        return {
            "branch_id": branch_id,
            "branch_name": branch.get("name", ""),
            "branch_status": branch.get("status", ""),
            "delta_count": len(deltas),
            "assumptions": assumptions,
            "belief_changes": belief_changes,
            "suppressions": suppressions,
            "new_gaps": new_gaps,
            "resolved_gaps": resolved_gaps_list,
            "new_assertions": new_assertions_list,
        }

    # ------------------------------------------------------------------
    # Alternative Theory Portfolio (SO-2, SO-3, SO-4, SO-5, SO-7)
    # ------------------------------------------------------------------

    def get_alternative_theory_portfolio(
        self,
        *,
        objective_id: str | None = None,
        max_theories: int = 5,
        include_discriminators: bool = True,
    ) -> dict:
        """Derive competing interpretations from the matter graph.

        Returns theories ranked by support/attack/missingness that represent
        the strongest alternative explanations for the facts in this matter.
        """
        facets, composed_weights, primary = self._read_matter_domain_composition()
        domain_profile_id = primary or "legal"

        max_theories = max(1, min(max_theories, 10))

        issues = self.issues.get_open_issues(min_materiality=0.0)
        if objective_id:
            issues = [i for i in issues if isinstance(i, dict) and i.get("id") == objective_id]

        all_assertions = self.assertions.list_recent(limit=500)
        if objective_id and issues:
            linked_aids: set[str] = set()
            for issue in issues:
                if not isinstance(issue, dict):
                    continue
                iid = issue.get("id", "")
                if iid:
                    linked = self.issues.get_assertions_for_issue(iid)
                    for la in linked:
                        if isinstance(la, dict):
                            linked_aids.add(la.get("assertion_id", "") or la.get("id", ""))
            if linked_aids:
                all_assertions = [a for a in all_assertions if isinstance(a, dict) and a.get("id") in linked_aids]

        all_gaps = self.gaps.workbench(min_materiality=0.0, limit=200)
        all_assumptions = self.assumptions.get_all(max_rows=200)

        active_beliefs = {
            "operative", "admitted", "resolved", "performed",
        }
        contested_beliefs = {
            "disputed", "alleged", "argued", "inferred",
        }
        negative_beliefs = {
            "not_performed", "superseded", "withdrawn",
        }

        supporting = []
        attacking = []
        uncertain = []
        negative = []

        for a in all_assertions:
            if not isinstance(a, dict):
                continue
            bs = str(a.get("belief_state", "unknown")).lower()
            if bs in active_beliefs:
                supporting.append(a)
            elif bs in negative_beliefs:
                negative.append(a)
            elif bs in contested_beliefs:
                uncertain.append(a)
            else:
                uncertain.append(a)

        for a in all_assertions:
            if not isinstance(a, dict):
                continue
            attackers = self.assertions.get_attackers(a.get("id", ""))
            if attackers:
                attacking.append(a)

        def _source_role_mix(assertions: list[dict]) -> dict[str, int]:
            mix: dict[str, int] = {}
            for a in assertions:
                if not isinstance(a, dict):
                    continue
                roles = a.get("source_roles", [])
                if isinstance(roles, str):
                    roles = [r.strip() for r in roles.split(",") if r.strip()]
                elif not isinstance(roles, list):
                    roles = []
                for role in roles:
                    mix[str(role)] = mix.get(str(role), 0) + 1
            return mix

        def _taint_count(assertions: list[dict]) -> int:
            count = 0
            for a in assertions:
                if not isinstance(a, dict):
                    continue
                aid = a.get("id", "")
                if aid:
                    try:
                        row = self.db.execute(
                            "SELECT COUNT(*) AS c FROM object_taint WHERE matter_id=? AND target_kind='assertion' AND target_id=?",
                            (self.matter_id, aid),
                        ).fetchone()
                        if row and int(row["c"]) > 0:
                            count += 1
                    except Exception:
                        _log.warning("_taint_count: error checking taint for %s", aid, exc_info=True)
            return count

        def _confidence_range(assertions: list[dict]) -> list[float]:
            confs = []
            for a in assertions:
                if not isinstance(a, dict):
                    continue
                c = a.get("confidence")
                if c is not None:
                    try:
                        fv = float(c)
                        if math.isfinite(fv):
                            confs.append(fv)
                    except (TypeError, ValueError):
                        _log.debug("_confidence_range: non-numeric confidence %r", c)
            if not confs:
                return [0.0, 0.0]
            return [round(min(confs), 3), round(max(confs), 3)]

        open_gap_ids = [g.get("id", "") for g in all_gaps if isinstance(g, dict) and g.get("status") == "open"]
        provisional_assumptions = [a for a in all_assumptions if isinstance(a, dict) and a.get("status") == "provisional"]

        def _disc_questions(theory_assertions: list[dict], other_assertions: list[dict]) -> list[str]:
            if not include_discriminators:
                return []
            questions: list[str] = []
            theory_ids = {a.get("id") for a in theory_assertions if isinstance(a, dict)}
            for g in all_gaps:
                if not isinstance(g, dict):
                    continue
                deps = g.get("dependencies", [])
                for dep in deps:
                    if not isinstance(dep, dict):
                        continue
                    aid = dep.get("affected_id", "")
                    if aid in theory_ids:
                        desc = str(g.get("description", ""))[:200]
                        if desc and desc not in questions:
                            questions.append(desc)
                        break
                if len(questions) >= 5:
                    break
            if len(questions) < 3:
                for a_other in other_assertions[:5]:
                    if not isinstance(a_other, dict):
                        continue
                    prop = str(a_other.get("proposition_text", ""))[:200]
                    if prop:
                        q = f"Is it true that: {prop}?"
                        if q not in questions:
                            questions.append(q)
                    if len(questions) >= 5:
                        break
            return questions[:5]

        theories: list[dict] = []
        theory_idx = 0

        if supporting:
            theory_idx += 1
            theories.append({
                "id": f"theory-baseline-{theory_idx}",
                "label": "Baseline Theory (strongest supported interpretation)",
                "domain_profile_id": domain_profile_id,
                "stance": "supporting",
                "supporting_assertions": len(supporting),
                "attacking_assertions": len([a for a in supporting if a in attacking]),
                "assumptions": len([a for a in provisional_assumptions
                                    if isinstance(a, dict)]),
                "open_gaps": len(open_gap_ids),
                "discriminator_questions": _disc_questions(supporting, uncertain + negative),
                "confidence_range": _confidence_range(supporting),
                "source_role_mix": _source_role_mix(supporting),
                "taint_summary": {"tainted_assertion_count": _taint_count(supporting)},
            })

        if negative:
            theory_idx += 1
            theories.append({
                "id": f"theory-opposition-{theory_idx}",
                "label": "Opposition Theory (strongest contrary interpretation)",
                "domain_profile_id": domain_profile_id,
                "stance": "opposing",
                "supporting_assertions": len(negative),
                "attacking_assertions": len([a for a in negative if a in attacking]),
                "assumptions": len(provisional_assumptions),
                "open_gaps": len(open_gap_ids),
                "discriminator_questions": _disc_questions(negative, supporting),
                "confidence_range": _confidence_range(negative),
                "source_role_mix": _source_role_mix(negative),
                "taint_summary": {"tainted_assertion_count": _taint_count(negative)},
            })

        if uncertain:
            theory_idx += 1
            theories.append({
                "id": f"theory-uncertainty-{theory_idx}",
                "label": "Uncertainty Theory (contested / under-determined facts)",
                "domain_profile_id": domain_profile_id,
                "stance": "uncertain",
                "supporting_assertions": len(uncertain),
                "attacking_assertions": len([a for a in uncertain if a in attacking]),
                "assumptions": len(provisional_assumptions),
                "open_gaps": len(open_gap_ids),
                "discriminator_questions": _disc_questions(uncertain, supporting + negative),
                "confidence_range": _confidence_range(uncertain),
                "source_role_mix": _source_role_mix(uncertain),
                "taint_summary": {"tainted_assertion_count": _taint_count(uncertain)},
            })

        if all_gaps:
            theory_idx += 1
            theories.append({
                "id": f"theory-missing-{theory_idx}",
                "label": "Missing-Evidence Theory (what we don't know yet)",
                "domain_profile_id": domain_profile_id,
                "stance": "missing",
                "supporting_assertions": 0,
                "attacking_assertions": 0,
                "assumptions": len(provisional_assumptions),
                "open_gaps": len(open_gap_ids),
                "discriminator_questions": [
                    str(g.get("description", ""))[:200]
                    for g in all_gaps[:5]
                    if isinstance(g, dict) and g.get("description")
                ],
                "confidence_range": [0.0, 0.0],
                "source_role_mix": {},
                "taint_summary": {"tainted_assertion_count": 0},
            })

        theories = theories[:max_theories]

        return {
            "matter_id": self.matter_id,
            "domain_profile_id": domain_profile_id,
            "objective_id": objective_id,
            "theory_count": len(theories),
            "theories": theories,
        }

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

    def get_document_console(self, document_ref: str) -> dict:
        """Consolidated per-document review surface (SO-3, SO-5).

        Returns card metadata, candidate/verified fact counts, the facts
        themselves, linked issues, and actor roles — everything a reviewer
        needs to assess and act on a single source document."""
        ref_norm = (document_ref or "").replace("\\", "/")

        card: dict = {}
        try:
            inv_row = self.inventory.get_by_path(ref_norm)
            if inv_row:
                doc_id = inv_row["id"] if hasattr(inv_row, "__getitem__") else getattr(inv_row, "id", None)
                card = self.get_document_card(doc_id=doc_id) or {}
                if hasattr(card, "keys"):
                    card = dict(card)
            if not card:
                card = self.get_document_card(relative_path=ref_norm) or {}
                if hasattr(card, "keys"):
                    card = dict(card)
        except Exception as exc:
            _log.warning("document_console: card lookup failed for %s: %s", ref_norm, exc)

        candidates = self.list_candidate_assertions_for_document(ref_norm)
        candidate_count = len(candidates)

        verified_count = 0
        rejected_count = 0
        all_rows = self.db.execute(
            """SELECT COUNT(*) AS n,
                      SUM(CASE WHEN vs.status='verified' THEN 1 ELSE 0 END) AS verified,
                      SUM(CASE WHEN vs.status='rejected' THEN 1 ELSE 0 END) AS rejected
               FROM assertion_occurrence ao
               JOIN assertion a ON a.id=ao.assertion_id
               LEFT JOIN document_inventory di ON di.id = ao.document_inventory_id
               LEFT JOIN verification_state vs
                 ON vs.target_kind='assertion' AND vs.target_id=a.id AND vs.matter_id=a.matter_id
               WHERE a.matter_id=?
                 AND (ao.document_id=? OR ao.document_id=? OR di.relative_path=?)""",
            (self.matter_id, ref_norm, ref_norm.rsplit("/", 1)[-1] if "/" in ref_norm else ref_norm, ref_norm),
        ).fetchone()
        if all_rows:
            verified_count = int(all_rows["verified"] or 0)
            rejected_count = int(all_rows["rejected"] or 0)

        linked_issues: list[dict] = []
        try:
            issue_rows = self.db.execute(
                """SELECT DISTINCT i.id, i.title, i.status, i.materiality
                   FROM assertion_occurrence ao
                   JOIN assertion_issue_link ail ON ail.assertion_id = ao.assertion_id
                   JOIN issue i ON i.id = ail.issue_id
                   LEFT JOIN document_inventory di ON di.id = ao.document_inventory_id
                   WHERE i.matter_id=?
                     AND (ao.document_id=? OR di.relative_path=?)
                   ORDER BY i.materiality DESC""",
                (self.matter_id, ref_norm, ref_norm),
            ).fetchall()
            linked_issues = [dict(r) for r in issue_rows]
        except Exception as exc:
            _log.warning("document_console: linked_issues query failed for %s: %s", ref_norm, exc)

        actor_roles: list[dict] = []
        try:
            role_rows = self.db.execute(
                """SELECT dar.actor_id, act.name AS actor_name,
                          dar.role, dar.confidence
                   FROM document_actor_role dar
                   JOIN actor act ON act.id = dar.actor_id
                   LEFT JOIN document_inventory di ON di.id = dar.doc_id
                   WHERE act.matter_id=?
                     AND (dar.doc_id=? OR di.relative_path=?)
                   ORDER BY dar.confidence DESC""",
                (self.matter_id, ref_norm, ref_norm),
            ).fetchall()
            actor_roles = [dict(r) for r in role_rows]
        except Exception as exc:
            _log.warning("document_console: actor_roles query failed for %s: %s", ref_norm, exc)

        return {
            "document_ref": document_ref,
            "card": card,
            "candidate_count": candidate_count,
            "verified_count": verified_count,
            "rejected_count": rejected_count,
            "candidates": candidates[:50],
            "linked_issues": linked_issues,
            "actor_roles": actor_roles,
        }

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

    def reclassify_document_card_fields(
        self,
        doc_id: str,
        *,
        doc_type: Optional[str] = None,
        source_role: Optional[str] = None,
        privilege_flag: Optional[bool] = None,
        operative_status: Optional[str] = None,
        unresolved_flags: Optional[list] = None,
        reviewed_by_kind: str = "user",
        reviewed_by_id: Optional[str] = None,
    ) -> dict:
        """Correct classification fields on a document card (SO-3, SO-5).

        Partial update: only non-None fields are written. Returns
        {changed_fields, staled_count, card_id}. Downstream staling
        runs for doc_type/source_role/operative_status/privilege_flag
        changes but not for flags-only edits.
        """
        inv = self.db.execute(
            """SELECT id FROM document_inventory
               WHERE matter_id=? AND (id=? OR relative_path=?)""",
            (self.matter_id, doc_id, doc_id),
        ).fetchone()
        if inv is None:
            return {"error": "Document not found", "doc_id": doc_id}
        inv_id = inv["id"]
        card = self.document_cards.get_by_doc_id(inv_id)
        old = dict(card) if card else {}

        changed: list[str] = []
        upsert_kwargs: dict = {}

        if doc_type is not None and doc_type != old.get("doc_type"):
            upsert_kwargs["doc_type"] = doc_type
            changed.append("doc_type")
        if source_role is not None and source_role != old.get("source_role"):
            upsert_kwargs["source_role"] = source_role
            changed.append("source_role")
        if operative_status is not None and operative_status != old.get("operative_status"):
            upsert_kwargs["operative_status"] = operative_status
            changed.append("operative_status")
        if privilege_flag is not None:
            old_pf = old.get("privilege_flag")
            new_pf_int = 1 if privilege_flag else 0
            if old_pf != new_pf_int:
                upsert_kwargs["privilege_flag"] = privilege_flag
                changed.append("privilege_flag")
        if unresolved_flags is not None:
            upsert_kwargs["unresolved_flags"] = unresolved_flags
            changed.append("unresolved_flags")

        if not changed:
            return {"changed_fields": [], "staled_count": 0, "card_id": old.get("id", "")}

        card_id = self.document_cards.upsert(doc_id=inv_id, **upsert_kwargs)

        try:
            self.verify_target(
                "document_card", card_id,
                reviewed_by_kind=reviewed_by_kind,
                reviewed_by_id=reviewed_by_id,
                review_scope="document_card_classification",
            )
        except (ValueError, Exception) as _exc:
            _log.warning(
                "reclassify_document_card_fields: verify blocked: %s", _exc,
            )
            return {"error": f"Card updated but verification failed: {_exc}",
                    "changed_fields": changed, "card_id": card_id}

        staled = 0
        staling_fields = {"doc_type", "source_role", "operative_status", "privilege_flag"}
        if changed and staling_fields.intersection(changed):
            scope = self._collect_document_invalidation_scope(inv_id)
            scope["document_card_ids"] = set()
            reason_parts = [f"{f}:{old.get(f)}->{upsert_kwargs.get(f)}" for f in changed if f in staling_fields]
            staled = self._apply_invalidation(
                scope,
                reason=f"document_card_reclassified:{','.join(reason_parts)}",
            )
            if "privilege_flag" in changed:
                inv_row = self.db.execute(
                    "SELECT relative_path FROM document_inventory WHERE id=?",
                    (inv_id,),
                ).fetchone()
                rel_path = inv_row["relative_path"] if inv_row else ""
                self._propagate_privilege_taint(
                    inv_id, rel_path, scope, bool(privilege_flag),
                )

        return {"changed_fields": changed, "staled_count": staled, "card_id": card_id}

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
            for key in ("key_facts", "quotes", "entities", "numeric_facts"):
                val = analysis.get(key)
                if isinstance(val, list):
                    for item in val:
                        if isinstance(item, str):
                            text_parts.append(item)
                        elif isinstance(item, dict):
                            text_parts.extend(
                                str(v) for v in item.values()
                                if isinstance(v, str)
                            )
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
        "domain_facets:*", "domain_detection:*", "domain_profiles:*",
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
        _, _, primary_profile = self._read_matter_domain_composition()
        dpid = primary_profile or "legal"
        ns_deps = broker.namespace_dependencies_for_keys(
            self._SEMANTIC_CACHE_NAMESPACES
        )
        mapping_hash = broker.current_profile_mapping_hash(
            domain_profile_id=dpid,
            domain_profile_version=1,
            target_kind="clarification",
            target_namespace="clarifications",
        )
        manifest = DependencyManifest(
            matter_id=self.matter_id,
            namespace_dependencies=ns_deps,
            domain_profile_id=dpid,
            domain_profile_version=1,
            profile_mapping_hash=mapping_hash,
            purpose=purpose,
            policy_audience=policy_audience,
            taint_class=taint_class,
        )
        broker.record_dependency_manifest(manifest)
        return manifest.manifest_hash()

    def build_output_dependency_manifest(
        self,
        *,
        purpose: str,
        policy_audience: str = "clean",
        taint_class: str = "public_clean",
        object_refs: "Iterable[tuple[str, str]]" = (),
        negative_dependencies: "Iterable" = (),
        namespace_keys: "Iterable[str]" = (),
    ) -> str:
        """Build and record a per-output DependencyManifest (SO-1, SO-5).

        Unlike build_semantic_cache_manifest() which captures broad namespace
        revisions, this records the specific objects an output consumed so
        invalidation is precise: a changed assertion only invalidates outputs
        that consumed it.

        Returns the manifest hash.
        """
        from .memory_contracts import DependencyManifest, ObjectDependency, NamespaceDependency

        broker = self.memory_broker
        _, _, primary_profile = self._read_matter_domain_composition()
        dpid = primary_profile or "legal"

        obj_deps = []
        for kind, obj_id in object_refs:
            dep = ObjectDependency(target_kind=kind, target_id=obj_id)
            if kind in ("assertion", "assertions", "claim"):
                try:
                    row = self.db.execute(
                        "SELECT belief_state, verification_state FROM assertion "
                        "WHERE id=? AND matter_id=?",
                        (obj_id, self.matter_id),
                    ).fetchone()
                    if row:
                        dep = ObjectDependency(
                            target_kind=kind,
                            target_id=obj_id,
                            belief_state=row["belief_state"],
                            verification_state=row["verification_state"],
                        )
                except Exception:
                    pass
            obj_deps.append(dep)

        ns_deps = []
        ns_keys = set(namespace_keys) if namespace_keys else set()
        if ns_keys:
            ns_deps = list(broker.namespace_dependencies_for_keys(tuple(ns_keys)))

        mapping_hash = broker.current_profile_mapping_hash(
            domain_profile_id=dpid,
            domain_profile_version=1,
            target_kind="clarification",
            target_namespace="clarifications",
        )

        neg_deps = tuple(negative_dependencies) if negative_dependencies else ()

        manifest = DependencyManifest(
            matter_id=self.matter_id,
            namespace_dependencies=tuple(ns_deps),
            object_dependencies=tuple(obj_deps),
            negative_dependencies=neg_deps,
            domain_profile_id=dpid,
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
            _log.warning(
                "Domain composition unavailable for matter %s — falling back to legal trust weights",
                self.matter_id,
            )
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

    def get_domain_profile_summary(
        self,
        profile_id: str | None = None,
    ) -> dict:
        """Return a comprehensive summary of the active domain profile.

        If *profile_id* is None, uses the primary detected profile.
        """
        broker = self.memory_broker
        facets, composed_weights, primary = self._read_matter_domain_composition()
        is_fallback = primary is None and profile_id is None
        pid = profile_id or primary or "legal"

        profile = broker.get_domain_profile(pid)
        vocab = broker.get_profile_vocabulary(pid) if profile else None

        source_roles = list(vocab.get("source_roles", [])) if vocab else []
        belief_states = list(vocab.get("belief_states", [])) if vocab else []
        taint_classes = list(vocab.get("taint_classes", [])) if vocab else []
        speech_acts = list(vocab.get("speech_acts", [])) if vocab else []
        neutral_kernel = dict(vocab.get("neutral_kernel", {})) if vocab else {}

        return {
            "profile_id": pid,
            "profile_version": int(profile["profile_version"]) if profile else 1,
            "profile_kind": profile.get("profile_kind", pid) if profile else pid,
            "status": profile.get("status", "unknown") if profile else "not_found",
            "is_primary": pid == primary,
            "is_fallback": is_fallback,
            "facets": facets,
            "composed_trust_weights": composed_weights,
            "neutral_kernel": neutral_kernel,
            "source_roles": source_roles,
            "belief_states": belief_states,
            "taint_classes": taint_classes,
            "speech_acts": speech_acts,
        }

    def summarize_taint(self, limit: int = 100) -> dict:
        """Return a summary of taint records grouped by class and kind."""
        try:
            by_class = self.db.execute(
                """SELECT taint_class, COUNT(*) AS cnt
                   FROM object_taint WHERE matter_id=?
                   GROUP BY taint_class ORDER BY cnt DESC""",
                (self.matter_id,),
            ).fetchall()
            by_kind = self.db.execute(
                """SELECT target_kind, COUNT(*) AS cnt
                   FROM object_taint WHERE matter_id=?
                   GROUP BY target_kind ORDER BY cnt DESC""",
                (self.matter_id,),
            ).fetchall()
            recent = self.db.execute(
                """SELECT id, target_kind, target_id, taint_class,
                          derivation_reason, created_at
                   FROM object_taint WHERE matter_id=?
                   ORDER BY created_at DESC LIMIT ?""",
                (self.matter_id, max(1, min(limit, 500))),
            ).fetchall()
        except Exception:
            return {"by_class": [], "by_kind": [], "recent": [], "total": 0}

        return {
            "by_class": [
                {"taint_class": r["taint_class"], "count": int(r["cnt"])}
                for r in by_class
            ],
            "by_kind": [
                {"target_kind": r["target_kind"], "count": int(r["cnt"])}
                for r in by_kind
            ],
            "recent": [dict(r) for r in recent],
            "total": sum(int(r["cnt"]) for r in by_class),
        }

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

    _PREVIEW_ACTION_TYPES = frozenset({
        "resolve_gap", "correct_assertion", "resolve_contradiction",
        "approve_metric_alias", "escalate_gap",
    })

    def get_steering_impact_preview(
        self,
        action_type: str,
        payload: dict,
        *,
        domain_profile_id: str | None = None,
        policy_audience: str = "clean",
    ) -> dict:
        """Project the impact of a steering action without mutating state (SO-3).

        Reads current system health / SO metrics, computes what would change
        for the given action_type + payload, and returns a structured
        before/after/delta dict.  No database writes occur.
        """
        if action_type not in self._PREVIEW_ACTION_TYPES:
            return {
                "action_type": action_type,
                "valid": False,
                "warnings": [f"Unknown action_type: {action_type}"],
                "before": {},
                "after": {},
                "deltas": {},
                "recommended_followups": [],
            }

        warnings: list[str] = []
        affected_objectives: list[str] = []
        affected_assertions: list[str] = []

        health = self.get_system_health()
        coverage_report = self.get_issue_coverage_report(policy_audience)
        coverage_avg = 0.0
        if coverage_report:
            fracs = [float(r.get("coverage_fraction", 0.0)) for r in coverage_report]
            coverage_avg = round(sum(fracs) / len(fracs), 4) if fracs else 0.0

        before = {
            "issue_coverage_avg": coverage_avg,
            "open_gap_count": health.get("open_gap_count", 0),
            "contradiction_count": health.get("contradiction_count", 0),
            "disputed_count": health.get("disputed_count", 0),
            "readiness": health.get("health_score", "unknown"),
        }

        gaps_closed = 0
        gaps_opened = 0
        contradictions_resolved = 0
        coverage_delta = 0.0
        disputed_delta = 0

        if action_type == "resolve_gap":
            gap_id = payload.get("gap_id", "")
            if not gap_id:
                warnings.append("Missing required field: gap_id")
            else:
                gap_row = self.db.execute(
                    "SELECT id, gap_type, status FROM gap WHERE id=? AND matter_id=?",
                    (gap_id, self.matter_id),
                ).fetchone()
                if not gap_row:
                    warnings.append(f"Gap {gap_id} not found")
                elif gap_row["status"] != "open":
                    warnings.append(f"Gap {gap_id} is already {gap_row['status']}")
                else:
                    gaps_closed = 1
                    link_rows = self.db.execute(
                        "SELECT affected_type, affected_id FROM gap_link WHERE gap_id=?",
                        (gap_id,),
                    ).fetchall()
                    for lr in link_rows:
                        if lr["affected_type"] == "issue":
                            affected_objectives.append(lr["affected_id"])
                    if affected_objectives and coverage_report:
                        n_issues = len(coverage_report)
                        if n_issues > 0:
                            coverage_delta = round(0.05 / n_issues, 4)

        elif action_type == "correct_assertion":
            assertion_id = payload.get("assertion_id", "")
            new_state = payload.get("new_state", "")
            if not assertion_id:
                warnings.append("Missing required field: assertion_id")
            elif not new_state:
                warnings.append("Missing required field: new_state")
            else:
                rec = self.assertions.get(assertion_id)
                if not rec:
                    warnings.append(f"Assertion {assertion_id} not found")
                else:
                    affected_assertions.append(assertion_id)
                    dependents = self.assertions.get_dependents(assertion_id)
                    affected_assertions.extend(dependents)

                    old_is_disputed = rec.belief_state == "disputed"
                    new_is_disputed = new_state == "disputed"
                    if old_is_disputed and not new_is_disputed:
                        disputed_delta = -1
                    elif not old_is_disputed and new_is_disputed:
                        disputed_delta = 1

                    old_inactive = rec.belief_state in ("superseded", "withdrawn", "resolved")
                    new_inactive = new_state in ("superseded", "withdrawn", "resolved")
                    if old_inactive and not new_inactive and coverage_report:
                        n_issues = len(coverage_report)
                        if n_issues > 0:
                            coverage_delta = round(0.03 / n_issues, 4)
                    elif not old_inactive and new_inactive and coverage_report:
                        n_issues = len(coverage_report)
                        if n_issues > 0:
                            coverage_delta = round(-0.03 / n_issues, 4)

        elif action_type == "resolve_contradiction":
            attacker_id = payload.get("attacker_id", "")
            attacked_id = payload.get("attacked_id", "")
            decision = payload.get("decision", "")
            if not attacker_id or not attacked_id:
                warnings.append("Missing required fields: attacker_id, attacked_id")
            elif decision not in ("prefer_attacker", "prefer_attacked",
                                  "mark_both_disputed", "request_evidence"):
                warnings.append(f"Invalid decision: {decision}")
            else:
                contradictions_resolved = 1
                affected_assertions.extend([attacker_id, attacked_id])
                if decision == "mark_both_disputed":
                    disputed_delta = 2
                elif decision in ("prefer_attacker", "prefer_attacked"):
                    disputed_delta = 0
                if decision == "request_evidence":
                    gaps_opened = 1
                    contradictions_resolved = 0
                else:
                    gaps_closed_rows = self.db.execute(
                        "SELECT COUNT(*) AS n FROM gap WHERE matter_id=? AND status='open'"
                        " AND gap_type='unresolved_contradiction'",
                        (self.matter_id,),
                    ).fetchone()
                    if gaps_closed_rows and int(gaps_closed_rows["n"]) > 0:
                        gaps_closed = 1

        elif action_type == "approve_metric_alias":
            raw_label = payload.get("raw_label", "")
            canonical_metric = payload.get("canonical_metric", "")
            if not raw_label or not canonical_metric:
                warnings.append("Missing required fields: raw_label, canonical_metric")

        elif action_type == "escalate_gap":
            gap_id = payload.get("gap_id", "")
            if not gap_id:
                warnings.append("Missing required field: gap_id")
            else:
                gap_row = self.db.execute(
                    "SELECT id, status, blocker_score FROM gap WHERE id=? AND matter_id=?",
                    (gap_id, self.matter_id),
                ).fetchone()
                if not gap_row:
                    warnings.append(f"Gap {gap_id} not found")
                elif gap_row["status"] != "open":
                    warnings.append(f"Gap {gap_id} is already {gap_row['status']}")
                else:
                    link_rows = self.db.execute(
                        "SELECT affected_type, affected_id FROM gap_link WHERE gap_id=?",
                        (gap_id,),
                    ).fetchall()
                    for lr in link_rows:
                        if lr["affected_type"] == "issue":
                            affected_objectives.append(lr["affected_id"])

        after_gap_count = max(0, before["open_gap_count"] - gaps_closed + gaps_opened)
        after_contradiction_count = max(0, before["contradiction_count"] - contradictions_resolved)
        after_disputed = max(0, before["disputed_count"] + disputed_delta)
        after_coverage = round(min(1.0, max(0.0, coverage_avg + coverage_delta)), 4)

        after_health = before["readiness"]
        if after_gap_count < before["open_gap_count"] or after_contradiction_count < before["contradiction_count"]:
            if before["readiness"] == "attention_needed" and after_disputed == 0:
                after_health = "good"

        after = {
            "issue_coverage_avg": after_coverage,
            "open_gap_count": after_gap_count,
            "contradiction_count": after_contradiction_count,
            "disputed_count": after_disputed,
            "readiness": after_health,
        }

        deltas = {
            "coverage_delta": coverage_delta,
            "gaps_closed": gaps_closed,
            "gaps_opened": gaps_opened,
            "contradictions_resolved": contradictions_resolved,
            "disputed_delta": disputed_delta,
            "affected_objectives": affected_objectives,
            "affected_assertions": affected_assertions,
        }

        followups: list[str] = []
        if gaps_closed > 0 and after_gap_count > 0:
            followups.append("Review remaining open gaps")
        if contradictions_resolved > 0 and after_contradiction_count > 0:
            followups.append("Review remaining contradictions")
        if disputed_delta > 0:
            followups.append("Review newly disputed assertions for evidence")
        if affected_assertions and len(affected_assertions) > 1:
            followups.append(f"Review {len(affected_assertions) - 1} dependent assertions affected by propagation")

        return {
            "action_type": action_type,
            "valid": len(warnings) == 0,
            "warnings": warnings,
            "before": before,
            "after": after,
            "deltas": deltas,
            "recommended_followups": followups,
        }

    _ALL_PROFILES = ("legal", "finance", "coding", "academic_research", "biomedical")

    def evaluate_domain_investigation_readiness(
        self,
        *,
        profile_ids: list[str] | None = None,
        include_repair_recommendations: bool = True,
        policy_mode: str = "clean",
    ) -> dict:
        """Evaluate cross-domain readiness of the matter model (all SOs).

        For each requested profile, checks: assertion quality, source role
        calibration, objective coverage, quantitative coverage, gap modeling,
        steering readiness, and deliverable readiness. Identifies cross-domain
        blockers.
        """
        target_profiles = list(profile_ids) if profile_ids else list(self._ALL_PROFILES)
        target_profiles = [p for p in target_profiles if p in self._ALL_PROFILES]
        if not target_profiles:
            target_profiles = list(self._ALL_PROFILES)

        broker = self.memory_broker
        facets, _, primary_profile = self._read_matter_domain_composition()
        so_metrics = self.get_so_metrics()
        health = self.get_system_health()
        coverage_report = self.get_issue_coverage_report(policy_mode)

        assertion_count = so_metrics.get("assertion_count", 0) or 0
        issue_count = so_metrics.get("issue_count", 0) or 0
        open_gap_count = so_metrics.get("open_gap_count", 0) or 0
        quant_fact_count = so_metrics.get("quant_fact_count", 0) or 0
        coverage_avg = so_metrics.get("issue_coverage_avg")

        structure_rate = so_metrics.get("assertion_structure_rate")
        source_known_rate = so_metrics.get("source_role_known_rate")
        steerability = so_metrics.get("steerability", False)
        belief_revision = so_metrics.get("belief_revision", False)

        profiles_result: list[dict] = []
        statuses: list[str] = []

        for pid in target_profiles:
            profile_data = broker.get_domain_profile(pid)
            vocab = broker.get_profile_vocabulary(pid) if profile_data else None
            has_profile = profile_data is not None
            source_roles = list(vocab.get("source_roles", [])) if vocab else []
            has_vocab = bool(source_roles)

            primary_failures: list[str] = []

            detection_conf = 0.0
            for f in facets:
                if isinstance(f, dict) and f.get("domain_profile_id") == pid:
                    detection_conf = float(f.get("confidence", 0.0))
                    break

            if not has_profile:
                primary_failures.append(f"Profile {pid} not installed in broker")
            if not has_vocab:
                primary_failures.append(f"No vocabulary for profile {pid}")

            sr_calibration = {
                "source_role_known_rate": source_known_rate,
                "defined_roles": len(source_roles),
                "pass": (source_known_rate or 0) >= 0.5,
            }

            assertion_quality = {
                "assertion_count": assertion_count,
                "structure_rate": structure_rate,
                "pass": (structure_rate or 0) >= 0.8,
            }

            obj_coverage = {
                "issue_count": issue_count,
                "coverage_avg": coverage_avg,
                "issues_with_proof_gap": so_metrics.get("issues_with_proof_gap", 0),
                "pass": (coverage_avg or 0) >= 0.3,
            }

            quant_coverage = {
                "quant_fact_count": quant_fact_count,
                "pass": quant_fact_count > 0 or pid in ("coding", "academic_research"),
            }

            gap_modeling = {
                "open_gap_count": open_gap_count,
                "pass": True,
            }

            steering_ready = {
                "steerability": steerability,
                "belief_revision": belief_revision,
                "pass": bool(steerability),
            }

            deliverable_ready = {
                "pass": assertion_count > 0 and issue_count > 0,
            }

            checks = [
                sr_calibration["pass"],
                assertion_quality["pass"],
                obj_coverage["pass"],
                gap_modeling["pass"],
            ]
            if all(checks):
                status = "ready"
            elif any(checks):
                status = "partial"
            else:
                status = "blocked"

            if primary_failures:
                status = "blocked"

            repairs: list[str] = []
            if include_repair_recommendations:
                if not assertion_quality["pass"]:
                    repairs.append("Run investigation to populate assertions with typed metadata")
                if not sr_calibration["pass"]:
                    repairs.append(f"Review source roles — {pid} profile expects {len(source_roles)} roles")
                if not obj_coverage["pass"]:
                    repairs.append("Run investigation to improve issue coverage")
                if not steering_ready["pass"]:
                    repairs.append("Run at least one investigation to enable steering")

            profiles_result.append({
                "profile_id": pid,
                "status": status,
                "primary_failures": primary_failures,
                "domain_detection": {
                    "confidence": detection_conf,
                    "is_primary": pid == primary_profile,
                },
                "source_role_calibration": sr_calibration,
                "assertion_quality": assertion_quality,
                "objective_coverage": obj_coverage,
                "quantitative_coverage": quant_coverage,
                "gap_modeling": gap_modeling,
                "steering_readiness": steering_ready,
                "deliverable_readiness": deliverable_ready,
                "recommended_repairs": repairs,
            })
            statuses.append(status)

        cross_domain: list[dict] = []
        active_pids = {f.get("domain_profile_id") for f in facets if isinstance(f, dict)}
        for pid in target_profiles:
            if pid not in active_pids and pid != "legal":
                cross_domain.append({
                    "kind": "mapping_gap",
                    "profiles": [pid],
                    "severity": "medium",
                    "message": f"Profile {pid} has no detection signal in current matter",
                })

        if health.get("contradiction_count", 0) > 0 and len(active_pids) > 1:
            cross_domain.append({
                "kind": "source_role_drift",
                "profiles": sorted(active_pids),
                "severity": "high",
                "message": (
                    f"{health['contradiction_count']} contradictions across "
                    f"{len(active_pids)} active profiles — source role calibration may drift"
                ),
            })

        if all(s == "ready" for s in statuses):
            overall = "ready"
        elif all(s == "blocked" for s in statuses):
            overall = "blocked"
        else:
            overall = "partial"

        return {
            "matter_id": self.matter_id,
            "overall_status": overall,
            "profiles": profiles_result,
            "cross_domain_findings": cross_domain,
        }

    def list_belief_revisions(self, limit: int = 100) -> list[dict]:
        """Return recent belief revision events with assertion context.

        Each row includes the assertion text, old/new belief states, cause,
        and timestamp — the full transparency trail for SO-2 truth maintenance.
        """
        rows = self.db.execute(
            """SELECT bre.id, bre.assertion_id, bre.run_id, bre.cause,
                      bre.old_belief_state, bre.new_belief_state,
                      bre.old_confidence, bre.new_confidence,
                      bre.note, bre.created_at,
                      a.proposition_text
               FROM belief_revision_event bre
               JOIN assertion a ON a.id = bre.assertion_id
               WHERE a.matter_id = ?
               ORDER BY bre.created_at DESC
               LIMIT ?""",
            (self.matter_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

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

    def export_matter_summary(self) -> dict:
        """Assemble a structured summary for report export.

        Returns a dict with sections for issues, key assertions, gaps,
        contradictions, timeline, financials, and SO metrics — everything
        a professional needs for a snapshot report without opening the UI.
        """
        coverage = self.get_issue_coverage_report()
        assertions = self.assertions.list_recent(limit=100)
        gaps = self.gaps.open_gaps(limit=50)
        contradictions = self.assertions.find_contradictions(limit=30)
        timeline = self.get_timeline(limit=50)
        so = self.get_so_metrics(_coverage_report=coverage)
        stats = self.stats()

        runs = self.ledger.recent_runs(limit=5)
        run_summaries = []
        for r in runs:
            if isinstance(r, dict):
                run_summaries.append({
                    "id": r.get("id"),
                    "status": r.get("status"),
                    "objective": r.get("objective"),
                    "started_at": r.get("started_at"),
                    "completed_at": r.get("completed_at"),
                })
            else:
                run_summaries.append({
                    "id": r.id,
                    "status": r.status,
                    "objective": getattr(r, "objective", None),
                    "started_at": r.started_at,
                    "completed_at": getattr(r, "completed_at", None),
                })

        return {
            "matter_id": self.matter_id,
            "generated_at": _now(),
            "stats": stats,
            "issues": coverage,
            "assertions": assertions[:50],
            "gaps": gaps,
            "contradictions": contradictions,
            "timeline": timeline[:30],
            "so_metrics": so,
            "recent_runs": run_summaries,
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

    def get_freshness_report(self) -> dict:
        rows = self.db.execute(
            """SELECT namespace, revision, updated_at
               FROM namespace_revision
               WHERE matter_id=? AND target_kind='*' AND target_id='*'
               ORDER BY namespace""",
            (self.matter_id,),
        ).fetchall()

        namespaces: list[dict] = []
        for row in rows:
            ns = row["namespace"]
            rev = int(row["revision"] or 0)
            namespaces.append({
                "namespace": ns,
                "revision": rev,
                "state": "fresh" if rev > 0 else "missing",
                "updated_at": row["updated_at"],
            })

        stale_list = [ns["namespace"] for ns in namespaces if ns["state"] != "fresh"]

        run_row = self.db.execute(
            "SELECT COUNT(*) AS cnt FROM run_session WHERE matter_id=? AND status='running'",
            (self.matter_id,),
        ).fetchone()
        active_runs = int(run_row["cnt"]) if run_row else 0

        last_update_row = self.db.execute(
            """SELECT MAX(updated_at) AS last_up FROM namespace_revision
               WHERE matter_id=?""",
            (self.matter_id,),
        ).fetchone()
        last_update = last_update_row["last_up"] if last_update_row else None

        return {
            "matter_id": self.matter_id,
            "namespace_count": len(namespaces),
            "stale_namespaces": stale_list,
            "is_hot_answerable": len(namespaces) > 0 and len(stale_list) == 0 and active_runs == 0,
            "active_run_count": active_runs,
            "last_update_at": last_update,
            "namespaces": namespaces,
        }

    def get_reasoning_cache_stats(self) -> dict:
        rows = self.db.execute(
            """SELECT stage,
                      COUNT(*) AS total,
                      SUM(CASE WHEN last_hit_at IS NOT NULL THEN 1 ELSE 0 END) AS hit_count
               FROM reasoning_cache
               WHERE matter_id=?
               GROUP BY stage
               ORDER BY stage""",
            (self.matter_id,),
        ).fetchall()

        trust_rev = self.reasoning_cache.current_trust_revision()
        stages: list[dict] = []
        total_entries = 0
        total_hits = 0
        for row in rows:
            t = int(row["total"])
            h = int(row["hit_count"])
            total_entries += t
            total_hits += h
            stages.append({
                "stage": row["stage"],
                "total_entries": t,
                "hit_count": h,
                "hit_rate": round(h / t, 3) if t > 0 else 0.0,
            })

        return {
            "matter_id": self.matter_id,
            "trust_revision": trust_rev,
            "total_entries": total_entries,
            "total_hits": total_hits,
            "overall_hit_rate": round(total_hits / total_entries, 3) if total_entries > 0 else 0.0,
            "stages": stages,
        }

    def __repr__(self) -> str:
        return f"MatterModel(matter_id={self.matter_id[:8]}..., db={self.db})"
