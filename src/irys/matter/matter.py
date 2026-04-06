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
from typing import Optional

_log = logging.getLogger(__name__)

from .db import SQLiteMatterDB
from .graph import (
    AssertionStore, GapStore, ActorStore, IssueStore, ClarificationStore, QuantStore,
    DocumentInventoryStore, ReasoningCacheStore, TrustOverrideStore, DocumentAnnotationStore,
    DecisionContextStore, AuthorityStore, ProofStateStore,
)
from .reasoning import ReasoningLedgerStore
from .belief_revision import BeliefRevisionEngine
from .enums import (
    BeliefState, AssertionLinkType, RevisionCause,
    LedgerEventType, GapType,
)
from .models import (
    AssertionCandidate, AssertionRecord, RevisionResult,
    QueryMatterContext, RunSessionRecord,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id() -> str:
    return uuid.uuid4().hex


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
        self.cache = ReasoningCacheStore(db, matter_id)
        self.trust_overrides = TrustOverrideStore(db, matter_id)
        self.annotations = DocumentAnnotationStore(db, matter_id)
        self.decision_context = DecisionContextStore(db, matter_id)
        self.authority = AuthorityStore(db, matter_id)
        self.proof_state = ProofStateStore(db, matter_id)
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
        # New-evidence assertion IDs dropped by BFS truncation during flush_revisions().
        # Drained at the start of the next flush_revisions() new-evidence pass (r29 fix).
        self._evidence_pending_ids: set[str] = set()
        self._evidence_pending_lock = threading.Lock()

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

    def start_run(self, query: str, objective: Optional[str] = None) -> str:
        """Start a new investigation run. Returns run_id.

        Snapshots the current assertion count so that ``complete_run`` can
        compute a measurable reuse_rate (SO-1 success criterion: > 0.70 on
        repeated queries over a stable matter).
        """
        assertions_at_start = self.assertions.count()
        run_id = self.ledger.start_run(query, objective, assertions_at_start)
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

    # ------------------------------------------------------------------
    # Assertion management
    # ------------------------------------------------------------------

    def record_assertion(
        self, candidate: AssertionCandidate
    ) -> tuple[str, bool]:
        """
        Upsert a canonical assertion and record an occurrence.

        Returns (assertion_id, is_new_assertion).
        """
        return self.assertions.upsert_occurrence(candidate)

    def link_assertions(
        self,
        src_id: str,
        dst_id: str,
        link_type: AssertionLinkType,
        weight: float = 1.0,
    ) -> str:
        return self.assertions.link(src_id, dst_id, link_type, weight)

    def apply_revision(
        self,
        seed_assertion_ids: list[str],
        cause: RevisionCause,
        run_id: Optional[str] = None,
        note: Optional[str] = None,
        _collect_unvisited: "list[str] | None" = None,
    ) -> list[RevisionResult]:
        """Trigger belief revision from seed assertions."""
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
        """
        if not ids:
            return
        with self._correction_pending_lock:
            for aid in ids:
                if aid not in self._correction_pending:
                    self._correction_pending[aid] = run_id

    def drain_correction_pending(self) -> "dict[str, str | None]":
        """Return and clear the durable correction retry queue (thread-safe).

        Returns assertion_id → originating_run_id. Called by RuntimeModel.flush_revisions()
        so correction-truncated nodes are included in the next BFS sweep (adv#028 HIGH fix).
        Lock protects dict clear/copy atomicity across concurrent REST requests (r25 fix).
        """
        with self._correction_pending_lock:
            result = dict(self._correction_pending)
            self._correction_pending.clear()
        return result

    def enqueue_evidence_pending(self, ids: list[str]) -> None:
        """Add new-evidence assertion IDs dropped by BFS truncation to the durable queue.

        Drained at the start of the next flush_revisions() new-evidence pass so work
        dropped by a budget-constrained flush is not silently lost (r29 MEDIUM fix).
        """
        if not ids:
            return
        with self._evidence_pending_lock:
            self._evidence_pending_ids.update(ids)

    def drain_evidence_pending(self) -> list[str]:
        """Return and clear truncated new-evidence assertion IDs (thread-safe)."""
        with self._evidence_pending_lock:
            result = list(self._evidence_pending_ids)
            self._evidence_pending_ids.clear()
        return result

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
        """
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

        return result

    def set_trust_override(
        self,
        document_pattern: str,
        trust_level: str,
        note: Optional[str] = None,
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
        override_id = self.trust_overrides.set(document_pattern, trust_level, note)

        # Trigger belief revision on all assertions from the affected document.
        # Uses indexed doc_basename column (schema v27) for exact basename lookup —
        # replaces leading-wildcard LIKE ('%/basename') which was not sargable.
        affected_ids: list[str] = []
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
                (self.matter_id, document_pattern, basename),
            ).fetchall()
            for row in occurrence_rows:
                doc = (row["document_id"] or "").replace("\\\\", "/").replace("\\", "/")
                doc_basename = Path(doc).name
                if pat_norm == doc or pat_norm == doc_basename:
                    affected_ids.append(row["assertion_id"])
            if affected_ids:
                self.belief.apply(
                    affected_ids,
                    cause=RevisionCause.TRUST_OVERRIDE,
                    note=f"Document trust override set to '{trust_level}' for {document_pattern!r}",
                )
        except (sqlite3.Error, ValueError, RuntimeError) as exc:
            _log.warning("Trust override belief revision failed for %r: %s", document_pattern, exc)

        # Targeted proof state recompute: only recompute issues linked to affected assertions.
        # Falls back to compute_all() when affected_ids is empty (pattern matched nothing).
        try:
            if affected_ids:
                issue_rows = self.db.execute(
                    "SELECT DISTINCT issue_id FROM assertion_issue_link WHERE assertion_id IN ({})".format(
                        ",".join("?" * len(affected_ids))
                    ),
                    affected_ids,
                ).fetchall()
                if issue_rows:
                    # Pre-fetch overrides once for all targeted issue recomputes.
                    _ov_rows = self.db.execute(
                        """SELECT document_pattern, trust_level FROM document_trust_override
                           WHERE matter_id=? AND trust_level != 'normal'
                           ORDER BY LENGTH(document_pattern) DESC""",
                        (self.matter_id,),
                    ).fetchall()
                    _preloaded = [
                        (r["document_pattern"], r["trust_level"]) for r in _ov_rows
                    ]
                    for row in issue_rows:
                        self.proof_state.compute_and_store(
                            row["issue_id"], _preloaded_overrides=_preloaded
                        )
            else:
                # No assertions matched — still run full recompute in case the override
                # pattern will match future assertions (eager proof state refresh).
                self.proof_state.compute_all()
        except (sqlite3.Error, ValueError, RuntimeError) as exc:
            _log.warning("Trust override proof state refresh failed for %r: %s", document_pattern, exc)

        return override_id

    def mine_contradictions(self) -> list[dict]:
        """
        Run the contradiction mining pass over all assertions in this matter.

        Finds explicit attacks/contradicts links between active assertions,
        marks attacked assertions as DISPUTED when the attacker is high-trust,
        and records UNRESOLVED_CONTRADICTION gaps for each open conflict.

        Returns the list of contradiction dicts found.
        """
        return self.assertions.mine_and_mark_contradictions(
            gap_store=self.gaps,
            belief_engine=self.belief,
        )

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
    # Query context (read at start of each run)
    # ------------------------------------------------------------------

    def build_query_context(self) -> QueryMatterContext:
        """
        Build a QueryMatterContext from current matter state.

        Called at the start of each investigation run to give the engine
        a snapshot of what is already known, enabling targeted retrieval.
        """
        row = self.db.execute(
            "SELECT id, name FROM matter WHERE id=?", (self.matter_id,)
        ).fetchone()
        matter_name = row["name"] if row else "unknown"

        assertion_count = self.assertions.count()
        # Limit to 10: engine context only uses count + first 3 descriptions.
        open_gaps = self.gaps.open_gaps(min_materiality=0.3, limit=10)
        open_issues = self.issues.get_open_issues(min_materiality=0.3)
        actor_count = self.actors.count()

        # Top actors by canonical name (limit 10 to keep context brief)
        known_actors = [
            a["canonical_name"]
            for a in self.actors.list_actors(limit=10)
        ]

        # Documents already indexed in the assertion store
        rows = self.db.execute(
            """SELECT DISTINCT document_id FROM assertion_occurrence
               WHERE assertion_id IN (SELECT id FROM assertion WHERE matter_id=?)
               ORDER BY document_id LIMIT 20""",
            (self.matter_id,),
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
            support_rows = self.db.execute(
                """SELECT ail.issue_id,
                          SUM(CASE
                                WHEN a.belief_state IN ('operative','admitted','resolved') THEN 1.0
                                WHEN a.belief_state IN ('alleged','argued','inferred') THEN 0.5
                                ELSE 0.3
                              END) AS weighted_support
                   FROM assertion_issue_link ail
                   JOIN issue i ON i.id=ail.issue_id
                   JOIN assertion a ON a.id=ail.assertion_id
                   WHERE i.matter_id=? AND i.status='open'
                     AND ail.relation_type IN ('supports','establishes')
                     AND a.belief_state NOT IN ('disputed','withdrawn','superseded')
                   GROUP BY ail.issue_id""",
                (self.matter_id,),
            ).fetchall()
            support_counts = {
                r["issue_id"]: float(r["weighted_support"] or 0.0)
                for r in support_rows
            }

            pred_rows_ctx = self.db.execute(
                """SELECT ip.issue_id, COUNT(*) AS pred_count
                   FROM issue_predicate ip
                   JOIN issue i ON i.id = ip.issue_id
                   WHERE i.matter_id=? AND i.status='open' AND ip.status='open'
                   GROUP BY ip.issue_id""",
                (self.matter_id,),
            ).fetchall()
            pred_counts_ctx = {r["issue_id"]: r["pred_count"] for r in pred_rows_ctx}

            def _weakness(issue: dict) -> tuple:
                w_support = support_counts.get(issue["id"], 0.0)
                pred_cnt = pred_counts_ctx.get(issue["id"], 0)
                coverage = self._coverage_fraction(w_support, pred_cnt)
                priority = issue["materiality"] * issue["salience"] * (1.0 - coverage)
                # Higher priority = higher weakness; negate for min()
                return (-priority, issue["id"])

            weakest = min(open_issues, key=_weakness)
            weakest_issue_id = weakest["id"]

        # Answered clarifications: inject user context into orientation (limit to 3 most recent)
        answered_clarifications = self.clarifications.get_answered(limit=3)

        # Document annotations: strategic notes from user (SO-3 annotation)
        document_annotations = self.annotations.list_recent(limit=10)

        # SO-2: top predicate_key values from the typed assertion graph so orientation
        # can generate SPO-aware search leads targeting known relationship types.
        pred_rows = self.db.execute(
            """SELECT predicate_key, COUNT(*) AS cnt
               FROM assertion
               WHERE matter_id=? AND predicate_key IS NOT NULL
               GROUP BY predicate_key
               ORDER BY cnt DESC
               LIMIT 10""",
            (self.matter_id,),
        ).fetchall()
        key_predicates = [r["predicate_key"] for r in pred_rows]

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
        )

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

    def get_issue_coverage_report(self) -> list[dict]:
        """Return per-issue evidence coverage for all open issues (SO-4).

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
        support_rows = self.db.execute(
            """SELECT ail.issue_id,
                      COUNT(*) AS raw_count,
                      SUM(CASE
                            WHEN a.belief_state IN ('operative','admitted','resolved') THEN 1.0
                            WHEN a.belief_state IN ('alleged','argued','inferred') THEN 0.5
                            ELSE 0.3
                          END) AS weighted_support
               FROM assertion_issue_link ail
               JOIN issue i ON i.id = ail.issue_id
               JOIN assertion a ON a.id = ail.assertion_id
               WHERE i.matter_id=? AND i.status='open'
                 AND ail.relation_type IN ('supports','establishes')
                 AND a.belief_state NOT IN ('disputed','withdrawn','superseded')
               GROUP BY ail.issue_id""",
            (mid,),
        ).fetchall()
        # raw_counts for the supporting_count field (integer, user-visible)
        raw_counts = {r["issue_id"]: int(r["raw_count"]) for r in support_rows}
        # weighted_supports for coverage_fraction computation
        support_counts = {
            r["issue_id"]: float(r["weighted_support"] or 0.0)
            for r in support_rows
        }

        # Attacking assertion counts per issue (for UI display — SO-4).
        attack_rows = self.db.execute(
            """SELECT ail.issue_id, COUNT(*) AS atk_count
               FROM assertion_issue_link ail
               JOIN issue i ON i.id = ail.issue_id
               JOIN assertion a ON a.id = ail.assertion_id
               WHERE i.matter_id=? AND i.status='open'
                 AND ail.relation_type IN ('attacks','negates')
                 AND a.belief_state NOT IN ('disputed','withdrawn','superseded')
               GROUP BY ail.issue_id""",
            (mid,),
        ).fetchall()
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
            w_support = support_counts.get(issue["id"], 0.0)
            raw_cnt = raw_counts.get(issue["id"], 0)
            pred_cnt = pred_counts.get(issue["id"], 0)
            atk_cnt = attack_counts.get(issue["id"], 0)
            coverage = self._coverage_fraction(w_support, pred_cnt)
            has_gap = issue["id"] in proof_gaps
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
            report.append({
                "id": issue["id"],
                "title": issue.get("title", ""),
                "issue_type": issue.get("issue_type", ""),
                "materiality": issue.get("materiality", 0.0),
                "salience": issue.get("salience", 0.0),
                "supporting_count": raw_cnt,
                "attacking_count": atk_cnt,
                "predicate_count": pred_cnt,
                "coverage_fraction": round(coverage, 4),
                "proof_status": proof_status,
                "has_proof_gap": has_gap,
                "gap_id": proof_gaps.get(issue["id"]),
            })

        report.sort(key=lambda x: x["coverage_fraction"])
        return report

    # ------------------------------------------------------------------
    # Clarification engine (SO-7, SO-3)
    # ------------------------------------------------------------------

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

    def detect_quant_conflicts(self) -> list[str]:
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
                     AND assertion_id IS NOT NULL""",
                (self.matter_id, subject, subject_id or "", currency, currency),
            ).fetchall()
            aids = [r["assertion_id"] for r in rows]
            if len(aids) >= 2:
                # Wire bidirectional contradicts links (idempotent via UNIQUE index)
                for i, a1 in enumerate(aids):
                    for a2 in aids[i + 1:]:
                        self.assertions.link(a1, a2, AssertionLinkType.CONTRADICTS)
                        self.assertions.link(a2, a1, AssertionLinkType.CONTRADICTS)
                all_conflict_assertion_ids.extend(aids)

        # Run belief revision on all conflicting assertions so they become DISPUTED
        if all_conflict_assertion_ids:
            self.belief.apply(
                seed_assertion_ids=list(dict.fromkeys(all_conflict_assertion_ids)),
                cause=RevisionCause.CONFLICT_DETECTION,
                note="Automatic: conflicting amount values detected for same subject",
            )

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

    # ------------------------------------------------------------------
    # Timeline view (SO-6, Priority 2 visual work product)
    # ------------------------------------------------------------------

    def get_timeline(self, limit: int = 200) -> list[dict]:
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
            date_val = qf.get("date_value") or qf.get("date_end_value") or qf.get("raw_text", "")[:60]
            events.append({
                "date": date_val,
                "event": qf.get("raw_text", "")[:200],
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
            desc = (row["proposition_text"] or "")[:200]
            events.append({
                "date": date_val,
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
        return events[:limit]

    # ------------------------------------------------------------------
    # Evidence matrix (Priority 2 visual work product)
    # ------------------------------------------------------------------

    def get_evidence_matrix(self) -> dict:
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
        }

        Only open issues with at least one assertion link are included.
        Only document sources with at least one assertion link are included.
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

        return {
            "issues": issues_out,
            "sources": sources_out,
            "cells": cells,
            "issue_totals": issue_totals,
            "source_totals": source_totals,
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
            """SELECT id, subject_type, subject_id, amount_value, raw_text, assertion_id
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
                "raw_text": (row["raw_text"] or "")[:200],
                "amount_value": row["amount_value"],
                "subject_id": row["subject_id"],
                "assertion_id": row["assertion_id"],
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
                    conflicts = [f"${v:,.2f}" for v in unique_vals[:5]]

            waterfall.append({
                "component": component,
                "claimed_amount": round(total, 2),
                "source_count": len(entries),
                "currency": currency,
                "amounts": entries[:20],  # cap for readability
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
        return {
            "matter_id": self.matter_id,
            "assertion_count": self.assertions.count(),
            "open_gap_count": self.gaps.count_open(),
            "open_issue_count": self.issues.count_open(),
            "actor_count": self.actors.count(),
            "quant_fact_count": self.quant.count(),
            "pending_clarifications": self.clarifications.count_pending(),
            "recent_runs": len(self.ledger.recent_runs(limit=5)),
        }

    def get_so_metrics(self, _coverage_report: "list[dict] | None" = None) -> dict:
        """Compute measurable Sacred Outcome success criteria from stored state.

        Returns a snapshot of how well the current matter model satisfies the
        quantitative success criteria defined in the project CLAUDE.md.  Metrics
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

        # SO-3: steerability is a capability flag — the infrastructure is always
        # wired (engine checks is_stop_requested() throughout the run loop).
        # This is True unconditionally; it reflects presence of the mechanism,
        # not runtime activity which cannot be measured from stored state.
        steerability: "bool | None" = True

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
        _reuse_rows = self.db.execute(
            """SELECT reuse_rate FROM run_session
               WHERE matter_id=? AND status='completed' AND reuse_rate IS NOT NULL
                 AND assertions_at_start > 0
               ORDER BY completed_at DESC LIMIT 5""",
            (self.matter_id,),
        ).fetchall()
        _reuse_vals = [float(r["reuse_rate"]) for r in _reuse_rows if r["reuse_rate"] is not None]
        reuse_rate_avg: "float | None" = (
            round(sum(_reuse_vals) / len(_reuse_vals), 4) if _reuse_vals else None
        )

        targets = {
            "assertion_structure_rate": 1.0,
            "source_role_known_rate": 0.9,
            "issue_coverage_avg": 0.8,
            "reuse_rate": 0.7,
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
            # SO-3: steerability is a capability flag (infrastructure always wired)
            "steerability": steerability,
            # SO-2: True if belief_revision_event records exist (revisions have occurred)
            "belief_revision": belief_revision,
            "gap_detection_recall": None,
            "numeric_extraction_rate": None,
            # Raw counts
            "counts": {
                "assertions": assertion_count,
                "assertion_occurrences": ao_total,
                "issues_open": issue_count,
                "open_gaps": open_gap_count,
                "quant_facts": quant_fact_count,
                "actors": actor_count,
            },
            # Targets from CLAUDE.md
            "targets": targets,
            # Pass/fail per metric (None = not enough data to evaluate)
            "targets_met": {
                "assertion_structure_rate": _pass("assertion_structure_rate", assertion_structure_rate),
                "source_role_known_rate": _pass("source_role_known_rate", source_role_known_rate),
                "issue_coverage_avg": _pass("issue_coverage_avg", issue_coverage_avg),
                "reuse_rate": _pass("reuse_rate", reuse_rate_avg),
                "steerability": _pass("steerability", steerability),
                "belief_revision": _pass("belief_revision", belief_revision),
            },
        }

    def __repr__(self) -> str:
        return f"MatterModel(matter_id={self.matter_id[:8]}..., db={self.db})"
