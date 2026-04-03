"""AssertionStore, GapStore, ActorStore, IssueStore, ClarificationStore, QuantStore,
DocumentInventoryStore, TrustOverrideStore, DocumentAnnotationStore, ReasoningCacheStore.

The assertion store is the heart of the intelligence layer. It maintains
typed assertions with speech-act classification, support/attack links,
and revisable belief states.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from .db import SQLiteMatterDB
from .enums import (
    BeliefState, SpeechAct, SourceRole, ModelLayer, AssertionKind,
    AssertionLinkType, OriginKind, GapType, IssueType,
)
from .models import AssertionCandidate, AssertionRecord, RevisionResult


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id() -> str:
    return uuid.uuid4().hex


def _initial_belief_state(speech_act: SpeechAct) -> tuple[BeliefState, float]:
    """Derive initial belief state and confidence from the assertion's speech act.

    Avoids the "every assertion starts UNKNOWN" problem where operative facts
    require a separate flush_revisions() cycle before they become useful for
    downstream belief revision. Starting with a speech-act-derived state means
    the assertion graph is immediately populated with meaningful initial values.

    SpeechAct members: ALLEGED, ARGUED, DENIED, ADMITTED, ORDERED, PERFORMED,
    PAID, REQUESTED, THREATENED, PROMISED, ESTIMATED, CALCULATED, OBSERVED,
    TESTIFIED, STIPULATED, AMENDED, WAIVED, TERMINATED, INFERRED, OPERATIVE,
    EXTRACTED.  Note: DISPUTED/SUPERSEDED/WITHDRAWN are BeliefState members, not SpeechAct.
    """
    if speech_act == SpeechAct.OPERATIVE:
        return BeliefState.OPERATIVE, 0.8
    if speech_act in (SpeechAct.ADMITTED, SpeechAct.STIPULATED):
        # Admissions/stipulations are distinct from operative text — they carry legal
        # significance but their truth is asserted by the admitting party, not by
        # the document itself.
        return BeliefState.ADMITTED, 0.8
    if speech_act in (SpeechAct.PERFORMED, SpeechAct.PAID):
        return BeliefState.PERFORMED, 0.8
    if speech_act == SpeechAct.INFERRED:
        return BeliefState.INFERRED, 0.6
    if speech_act == SpeechAct.ALLEGED:
        # Alleged = claimed without independent proof → ALLEGED belief state, lower confidence
        return BeliefState.ALLEGED, 0.3
    if speech_act == SpeechAct.ARGUED:
        return BeliefState.ARGUED, 0.3
    if speech_act in (SpeechAct.WAIVED, SpeechAct.TERMINATED, SpeechAct.AMENDED):
        return BeliefState.OPERATIVE, 0.7
    # EXTRACTED, DENIED, ORDERED, REQUESTED, etc. — generic unknown until revision
    return BeliefState.UNKNOWN, 0.5


class AssertionStore:
    """
    Manages canonical assertions and their occurrences.

    Deduplication invariant: the same proposition from two documents
    yields ONE assertion row and TWO assertion_occurrence rows.
    Speech act is occurrence-level; belief state is canonical-assertion-level.
    """

    def __init__(self, db: SQLiteMatterDB, matter_id: str):
        self.db = db
        self.matter_id = matter_id

    def upsert_occurrence(self, candidate: AssertionCandidate) -> tuple[str, bool]:
        """
        Upsert a canonical assertion and record one occurrence.

        Returns (assertion_id, is_new_assertion).
        If the proposition already exists, only the occurrence is inserted.
        """
        prop_key = candidate.proposition_key()
        now = _now()
        _init_state, _init_conf = _initial_belief_state(candidate.speech_act)
        _candidate_id = _id()

        with self.db.transaction():
            # INSERT OR IGNORE avoids a SELECT-then-INSERT race: two concurrent callers
            # both attempting INSERT on the same proposition_key would previously cause
            # the second to hit an IntegrityError. INSERT OR IGNORE lets both proceed
            # safely — one inserts, the other is silently ignored.
            # unique key: (matter_id, model_layer, proposition_key) — see ux_assertion_prop.
            _assert_cur = self.db.execute(
                """INSERT OR IGNORE INTO assertion
                   (id, matter_id, proposition_key, proposition_text,
                    model_layer, assertion_kind,
                    subject_ref_type, subject_ref_id, predicate_key, object_json,
                    temporal_scope_start, temporal_scope_end,
                    belief_state, confidence, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    _candidate_id, self.matter_id, prop_key,
                    candidate.proposition_text,
                    candidate.model_layer.value,
                    candidate.assertion_kind.value,
                    candidate.subject_ref_type,
                    candidate.subject_ref_id,
                    candidate.predicate_key,
                    candidate.object_json,
                    candidate.temporal_scope_start,
                    candidate.temporal_scope_end,
                    _init_state.value,
                    _init_conf,
                    now, now,
                ),
            )
            is_new = _assert_cur.rowcount > 0

            if is_new:
                assertion_id = _candidate_id
                row = None  # No existing row — upgrade logic does not apply
            else:
                # The INSERT was ignored: re-SELECT to get the actual stored ID and state.
                # Must match matter, layer, AND prop key (same filters as the unique index).
                row = self.db.execute(
                    "SELECT id, belief_state, confidence FROM assertion"
                    " WHERE matter_id=? AND model_layer=? AND proposition_key=?",
                    (self.matter_id, candidate.model_layer.value, prop_key),
                ).fetchone()
                assertion_id = row["id"]

            # Always attempt to insert an occurrence (even for known assertions from new docs).
            # Capture the cursor so we can detect whether the row was actually inserted
            # (rowcount=1) or silently ignored due to the UNIQUE index (rowcount=0).
            occ_id = _id()
            _occ_cur = self.db.execute(
                """INSERT OR IGNORE INTO assertion_occurrence
                   (id, assertion_id, document_id, span_id,
                    speaker_actor_id, source_role, source_side,
                    speech_act, origin_kind, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    occ_id, assertion_id,
                    candidate.document_id,
                    candidate.span_id,
                    candidate.speaker_actor_id,
                    candidate.source_role.value,
                    candidate.source_side,
                    candidate.speech_act.value,
                    candidate.origin_kind.value,
                    now,
                ),
            )

            # Upgrade canonical belief_state only when:
            #   (a) the occurrence was actually new (not a duplicate re-ingest), AND
            #   (b) the assertion is not in a terminal state (SUPERSEDED/WITHDRAWN).
            # Running the upgrade before the INSERT OR IGNORE would overwrite user-corrected
            # or graph-derived states on duplicate ingestion with no new evidence.
            if not is_new and _occ_cur.rowcount > 0:
                _new_state, _new_conf = _initial_belief_state(candidate.speech_act)
                _current_conf = row["confidence"] if row["confidence"] is not None else 0.5
                _TERMINAL = (BeliefState.SUPERSEDED.value, BeliefState.WITHDRAWN.value)
                if _new_conf > _current_conf and row["belief_state"] not in _TERMINAL:
                    self.db.execute(
                        "UPDATE assertion SET belief_state=?, confidence=?, updated_at=? WHERE id=?",
                        (_new_state.value, _new_conf, now, assertion_id),
                    )

        return assertion_id, is_new

    def link(
        self,
        src_id: str,
        dst_id: str,
        link_type: AssertionLinkType,
        weight: float = 1.0,
    ) -> str:
        """Create a directed link between two assertions. Idempotent."""
        link_id = _id()
        now = _now()
        with self.db.transaction():
            self.db.execute(
                """INSERT OR IGNORE INTO assertion_link
                   (id, src_assertion_id, dst_assertion_id, link_type, weight, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (link_id, src_id, dst_id, link_type.value, weight, now),
            )
        # Return actual id (may have been ignored due to UNIQUE)
        row = self.db.execute(
            "SELECT id FROM assertion_link WHERE src_assertion_id=? AND dst_assertion_id=? AND link_type=?",
            (src_id, dst_id, link_type.value),
        ).fetchone()
        return row["id"] if row else link_id

    def set_belief_state(
        self,
        assertion_id: str,
        belief_state: BeliefState,
        confidence: Optional[float] = None,
    ) -> None:
        """Directly set the belief state on a canonical assertion."""
        now = _now()
        if confidence is not None:
            self.db.execute(
                "UPDATE assertion SET belief_state=?, confidence=?, updated_at=? WHERE id=?",
                (belief_state.value, confidence, now, assertion_id),
            )
        else:
            self.db.execute(
                "UPDATE assertion SET belief_state=?, updated_at=? WHERE id=?",
                (belief_state.value, now, assertion_id),
            )

    def get(self, assertion_id: str) -> Optional[AssertionRecord]:
        """Fetch a canonical assertion by ID."""
        row = self.db.execute(
            "SELECT * FROM assertion WHERE id=?", (assertion_id,)
        ).fetchone()
        if row is None:
            return None
        return AssertionRecord(**dict(row))

    def get_occurrences(self, assertion_id: str) -> list[dict]:
        """Fetch all occurrences for a canonical assertion."""
        rows = self.db.execute(
            "SELECT * FROM assertion_occurrence WHERE assertion_id=? ORDER BY created_at",
            (assertion_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_dependents(self, assertion_id: str) -> list[str]:
        """
        Return assertion IDs that depend on this assertion and must be re-evaluated
        when it changes.

        Two propagation paths:
        - supports: A --supports--> B — B depends on A; if A weakens, B may weaken.
        - attacks/negates: A --attacks--> B — B is attacked by A; if A is withdrawn or
          disputed, B's status may recover and must be re-evaluated.

        Note: depends_on links are intentionally excluded — propagation would go in
        the wrong direction (toward the prerequisite, not toward the dependent).
        """
        # corroborates: A --corroborates--> B — B gets additional support; changes in A
        #   affect B's confidence, so B must be re-evaluated.
        # supersedes: A --supersedes--> B — B is the superseded node; if A weakens, B may recover.
        #   We traverse from A to B so the superseded assertion gets re-evaluated when
        #   the superseding one changes.
        # 'negates' removed: not a valid AssertionLinkType in the current enum.
        rows = self.db.execute(
            """SELECT dst_assertion_id FROM assertion_link
               WHERE src_assertion_id=? AND link_type IN (
                 'supports', 'attacks', 'contradicts', 'corroborates', 'supersedes'
               )""",
            (assertion_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_attackers(self, assertion_id: str) -> list[str]:
        """Return assertion IDs that attack this assertion.
        Link direction: attacker --ATTACKS--> assertion_id
        """
        rows = self.db.execute(
            """SELECT src_assertion_id FROM assertion_link
               WHERE dst_assertion_id=? AND link_type IN ('attacks','contradicts')""",
            (assertion_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_supports(self, assertion_id: str) -> list[str]:
        """Return assertion IDs that support or corroborate this assertion.
        Link direction: supporter --SUPPORTS/CORROBORATES--> assertion_id

        corroborates = independently confirms; treated as support for belief state
        computation so convergent evidence from multiple sources raises confidence.
        """
        rows = self.db.execute(
            """SELECT src_assertion_id FROM assertion_link
               WHERE dst_assertion_id=? AND link_type IN ('supports','corroborates')""",
            (assertion_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_superseding(self, assertion_id: str) -> list[str]:
        """Return assertion IDs that supersede this assertion.
        Link direction: newer --SUPERSEDES--> assertion_id
        If any superseding assertion exists, this assertion should be SUPERSEDED.
        """
        rows = self.db.execute(
            """SELECT src_assertion_id FROM assertion_link
               WHERE dst_assertion_id=? AND link_type='supersedes'""",
            (assertion_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_neighbor_belief_states(self, assertion_id: str) -> dict:
        """Batch-load support/attack/supersedes neighbor belief states in one query.

        Returns dict with keys 'support_states', 'attack_states', 'has_superseding'.
        Avoids N+1 pattern in belief revision (3 queries total, not 3 + N + M).
        """
        rows = self.db.execute(
            """SELECT al.link_type, a.belief_state
               FROM assertion_link al
               JOIN assertion a ON a.id = al.src_assertion_id
               WHERE al.dst_assertion_id=?
                 AND al.link_type IN ('supports','corroborates','attacks','contradicts','supersedes')""",
            (assertion_id,),
        ).fetchall()
        support_states = []
        attack_states = []
        has_superseding = False
        for row in rows:
            lt = row["link_type"]
            bs = BeliefState(row["belief_state"])
            if lt in ("supports", "corroborates"):
                support_states.append(bs)
            elif lt in ("attacks", "contradicts"):
                attack_states.append(bs)
            elif lt == "supersedes":
                has_superseding = True
        return {
            "support_states": support_states,
            "attack_states": attack_states,
            "has_superseding": has_superseding,
        }

    def count(self) -> int:
        row = self.db.execute(
            "SELECT COUNT(*) FROM assertion WHERE matter_id=?", (self.matter_id,)
        ).fetchone()
        return row[0]

    def list_recent(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """Return recent assertions with all occurrence source metadata aggregated.

        Each row includes:
        - occurrence_count: total number of times this proposition was observed
        - source_roles: JSON-encoded list of distinct source roles seen
        - documents: JSON-encoded list of document_ids that contain this assertion
        - primary_document_id / primary_source_role / primary_speech_act: earliest
          occurrence values (backward compat)

        This closes the SO-5 single-occurrence collapse where an assertion appearing
        in both a complaint (ADVOCACY) and a contract (OPERATIVE) would show only the
        first-seen source role.
        """
        # primary_* fields come from the earliest occurrence (chronological, not lexicographic).
        # We use a correlated subquery per field to avoid MIN() on non-sortable text columns.
        rows = self.db.execute(
            """SELECT a.id, a.proposition_text, a.model_layer, a.assertion_kind,
                      a.belief_state, a.confidence, a.created_at,
                      COUNT(ao.id) AS occurrence_count,
                      GROUP_CONCAT(DISTINCT ao.source_role) AS source_roles_csv,
                      GROUP_CONCAT(DISTINCT ao.speech_act) AS speech_acts_csv,
                      GROUP_CONCAT(DISTINCT ao.document_id) AS documents_csv,
                      (SELECT ao2.document_id FROM assertion_occurrence ao2
                       WHERE ao2.assertion_id = a.id
                       ORDER BY ao2.created_at ASC, ao2.id ASC LIMIT 1) AS primary_document_id,
                      (SELECT ao2.source_role FROM assertion_occurrence ao2
                       WHERE ao2.assertion_id = a.id
                       ORDER BY ao2.created_at ASC, ao2.id ASC LIMIT 1) AS primary_source_role,
                      (SELECT ao2.speech_act FROM assertion_occurrence ao2
                       WHERE ao2.assertion_id = a.id
                       ORDER BY ao2.created_at ASC, ao2.id ASC LIMIT 1) AS primary_speech_act
               FROM assertion a
               LEFT JOIN assertion_occurrence ao ON ao.assertion_id = a.id
               WHERE a.matter_id=?
               GROUP BY a.id
               ORDER BY a.created_at DESC
               LIMIT ? OFFSET ?""",
            (self.matter_id, limit, offset),
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            # Expand CSV aggregates into proper lists
            d["source_roles"] = [
                s for s in (d.pop("source_roles_csv") or "").split(",") if s
            ]
            d["speech_acts"] = [
                s for s in (d.pop("speech_acts_csv") or "").split(",") if s
            ]
            d["documents"] = [
                s for s in (d.pop("documents_csv") or "").split(",") if s
            ]
            result.append(d)
        return result

    def list_recent_for_hydration(self, limit: int = 200) -> list[dict]:
        """Lightweight query returning only the fields needed by engine hydration.

        Avoids the three correlated subqueries in list_recent() and the GROUP_CONCAT
        aggregates — ~3-5× faster for large assertion stores because we only need
        proposition_text, belief_state, and a single source_role per assertion.

        Filters out inactive belief states (disputed/withdrawn/superseded) at the
        DB level so the LIMIT budget is not wasted on assertions that will be skipped
        during hydration. This ensures the 200-slot window contains only active facts
        even after many user corrections. (SO-2 budget efficiency)

        Returns: [{id, proposition_text, belief_state, source_role}]
        """
        rows = self.db.execute(
            """SELECT a.id, a.proposition_text, a.belief_state,
                      (SELECT ao.source_role FROM assertion_occurrence ao
                       WHERE ao.assertion_id = a.id
                       ORDER BY ao.created_at ASC, ao.id ASC LIMIT 1) AS source_role
               FROM assertion a
               WHERE a.matter_id=?
                 AND a.belief_state NOT IN ('disputed','withdrawn','superseded')
               ORDER BY a.created_at DESC
               LIMIT ?""",
            (self.matter_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_by_proposition(self, proposition_text: str) -> Optional[AssertionRecord]:
        """Look up an assertion by normalized proposition text."""
        import hashlib
        normalized = " ".join(proposition_text.lower().split())
        prop_key = hashlib.sha256(normalized.encode()).hexdigest()[:32]
        row = self.db.execute(
            "SELECT * FROM assertion WHERE matter_id=? AND proposition_key=?",
            (self.matter_id, prop_key),
        ).fetchone()
        if row is None:
            return None
        return AssertionRecord(**dict(row))


class GapStore:
    """Tracks structured missingness — documents, predicates, authorities, etc."""

    def __init__(self, db: SQLiteMatterDB, matter_id: str):
        self.db = db
        self.matter_id = matter_id

    def record(
        self,
        gap_type: GapType,
        description: str,
        expected_artifact: Optional[str] = None,
        materiality: float = 0.5,
        blocker_score: float = 0.0,
        affected_type: Optional[str] = None,
        affected_id: Optional[str] = None,
    ) -> str:
        """Record a gap. Returns gap_id."""
        gap_id = _id()
        now = _now()
        with self.db.transaction():
            self.db.execute(
                """INSERT INTO gap
                   (id, matter_id, gap_type, description, expected_artifact,
                    materiality_score, blocker_score, status, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (gap_id, self.matter_id, gap_type.value, description,
                 expected_artifact, materiality, blocker_score, "open", now, now),
            )
            if affected_type and affected_id:
                self.db.execute(
                    "INSERT INTO gap_link (id, gap_id, affected_type, affected_id, created_at) VALUES (?,?,?,?,?)",
                    (_id(), gap_id, affected_type, affected_id, now),
                )
        return gap_id

    def record_many(self, specs: list[dict]) -> list[str]:
        """Bulk-insert multiple gaps in a single transaction.

        Each spec is a dict with the same keys as `record()`:
          gap_type, description, expected_artifact, materiality,
          blocker_score, affected_type, affected_id (all optional except
          gap_type and description).

        Returns list of gap IDs in insertion order.
        """
        now = _now()
        gap_rows = []
        link_rows = []
        gap_ids = []
        for spec in specs:
            gap_id = _id()
            gap_ids.append(gap_id)
            gap_rows.append((
                gap_id,
                self.matter_id,
                spec["gap_type"].value if hasattr(spec["gap_type"], "value") else spec["gap_type"],
                spec["description"],
                spec.get("expected_artifact"),
                spec.get("materiality", 0.5),
                spec.get("blocker_score", 0.0),
                "open", now, now,
            ))
            if spec.get("affected_type") and spec.get("affected_id"):
                link_rows.append((_id(), gap_id, spec["affected_type"], spec["affected_id"], now))

        with self.db.transaction():
            self.db.executemany(
                """INSERT INTO gap
                   (id, matter_id, gap_type, description, expected_artifact,
                    materiality_score, blocker_score, status, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                gap_rows,
            )
            if link_rows:
                self.db.executemany(
                    "INSERT INTO gap_link (id, gap_id, affected_type, affected_id, created_at) VALUES (?,?,?,?,?)",
                    link_rows,
                )
        return gap_ids

    def open_gaps(self, min_materiality: float = 0.0) -> list[dict]:
        """Return open gaps above a materiality threshold.

        Each gap dict includes a 'dependencies' key: list of
        {affected_type, affected_id} dicts from gap_link so callers
        can see what the gap is linked to without a second query (SO-7).
        """
        rows = self.db.execute(
            """SELECT * FROM gap WHERE matter_id=? AND status='open'
               AND materiality_score >= ? ORDER BY materiality_score DESC""",
            (self.matter_id, min_materiality),
        ).fetchall()
        if not rows:
            return []

        # Fetch gap_links for all matching open gaps in one query via subquery
        # (avoids variable-count IN-list limit on large matters).
        link_rows = self.db.execute(
            """SELECT gl.gap_id, gl.affected_type, gl.affected_id
               FROM gap_link gl
               WHERE gl.gap_id IN (
                   SELECT id FROM gap
                   WHERE matter_id=? AND status='open' AND materiality_score >= ?
               )""",
            (self.matter_id, min_materiality),
        ).fetchall()
        links_by_gap: dict = {}
        for lr in link_rows:
            links_by_gap.setdefault(lr["gap_id"], []).append({
                "affected_type": lr["affected_type"],
                "affected_id": lr["affected_id"],
            })

        result = []
        for r in rows:
            d = dict(r)
            d["dependencies"] = links_by_gap.get(r["id"], [])
            result.append(d)
        return result

    def count_open(self) -> int:
        row = self.db.execute(
            "SELECT COUNT(*) FROM gap WHERE matter_id=? AND status='open'",
            (self.matter_id,),
        ).fetchone()
        return row[0]


class ActorStore:
    """
    Manages actors (parties, counsel, witnesses, entities) and their aliases.

    Deduplication invariant: the same real-world actor may appear under many
    name variations. One actor row + N actor_alias rows.
    Alias lookup is the primary entry point: get_by_alias() is the dedup gate.
    """

    def __init__(self, db: SQLiteMatterDB, matter_id: str):
        self.db = db
        self.matter_id = matter_id

    @staticmethod
    def _normalize(name: str) -> str:
        """Normalize for deduplication: lowercase, collapse whitespace."""
        return " ".join(name.lower().split())

    def upsert_actor(
        self,
        canonical_name: str,
        actor_type: str = "person",
        home_side: Optional[str] = None,
        agenda_notes: Optional[str] = None,
    ) -> tuple[str, bool]:
        """
        Create or retrieve an actor by canonical name.
        Also registers the canonical name as an alias.
        Returns (actor_id, is_new).
        """
        normalized = self._normalize(canonical_name)
        now = _now()

        with self.db.transaction():
            row = self.db.execute(
                "SELECT id FROM actor WHERE matter_id=? AND normalized_name=?",
                (self.matter_id, normalized),
            ).fetchone()

            if row is not None:
                return row["id"], False

            actor_id = _id()
            self.db.execute(
                """INSERT INTO actor
                   (id, matter_id, canonical_name, normalized_name, actor_type,
                    home_side, agenda_notes, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (actor_id, self.matter_id, canonical_name, normalized,
                 actor_type, home_side, agenda_notes, now, now),
            )
            # Register canonical name as primary alias
            self.db.execute(
                """INSERT OR IGNORE INTO actor_alias
                   (id, actor_id, alias_text, alias_type, created_at)
                   VALUES (?,?,?,?,?)""",
                (_id(), actor_id, normalized, "canonical", now),
            )

        return actor_id, True

    def add_alias(self, actor_id: str, alias_text: str, alias_type: str = "name") -> str:
        """Register an alias for an actor. Idempotent. Returns alias_id."""
        normalized = self._normalize(alias_text)
        now = _now()
        alias_id = _id()
        with self.db.transaction():
            self.db.execute(
                """INSERT OR IGNORE INTO actor_alias
                   (id, actor_id, alias_text, alias_type, created_at)
                   VALUES (?,?,?,?,?)""",
                (alias_id, actor_id, normalized, alias_type, now),
            )
        row = self.db.execute(
            "SELECT id FROM actor_alias WHERE actor_id=? AND alias_text=?",
            (actor_id, normalized),
        ).fetchone()
        return row["id"] if row else alias_id

    def get_by_alias(self, alias_text: str) -> Optional[str]:
        """
        Look up an actor_id by any alias (including canonical name).
        Returns actor_id or None if not found.
        """
        normalized = self._normalize(alias_text)
        row = self.db.execute(
            """SELECT a.id FROM actor a
               JOIN actor_alias aa ON aa.actor_id = a.id
               WHERE a.matter_id=? AND aa.alias_text=?""",
            (self.matter_id, normalized),
        ).fetchone()
        return row["id"] if row else None

    def list_actors(self) -> list[dict]:
        """Return all actors for this matter."""
        rows = self.db.execute(
            "SELECT * FROM actor WHERE matter_id=? ORDER BY canonical_name",
            (self.matter_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_aliases(self, actor_id: str) -> list[str]:
        """Return all alias texts for an actor."""
        rows = self.db.execute(
            "SELECT alias_text FROM actor_alias WHERE actor_id=? ORDER BY alias_type, alias_text",
            (actor_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def count(self) -> int:
        row = self.db.execute(
            "SELECT COUNT(*) FROM actor WHERE matter_id=?", (self.matter_id,)
        ).fetchone()
        return row[0]


class IssueStore:
    """
    Manages the legal issue tree for a matter.

    Issues can be hierarchical (claim → sub-claim → predicate).
    Assertions link to issues via assertion_issue_link.
    This drives issue-targeted retrieval: instead of asking
    "what do the documents say?", the engine asks "what supports/attacks
    each open issue predicate?".
    """

    def __init__(self, db: SQLiteMatterDB, matter_id: str):
        self.db = db
        self.matter_id = matter_id

    def upsert_issue(
        self,
        title: str,
        issue_type: IssueType,
        parent_issue_id: Optional[str] = None,
        burden_side: Optional[str] = None,
        materiality: float = 0.5,
        salience: float = 0.5,
        sort_order: int = 0,
    ) -> tuple[str, bool]:
        """
        Create or retrieve an issue by normalized title.
        Returns (issue_id, is_new).
        """
        normalized_title = " ".join(title.lower().split())
        now = _now()

        with self.db.transaction():
            row = self.db.execute(
                """SELECT id FROM issue
                   WHERE matter_id=? AND LOWER(title)=?""",
                (self.matter_id, normalized_title),
            ).fetchone()

            if row is not None:
                return row["id"], False

            issue_id = _id()
            self.db.execute(
                """INSERT INTO issue
                   (id, matter_id, parent_issue_id, title, issue_type,
                    burden_side, materiality, salience, status, sort_order,
                    created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (issue_id, self.matter_id, parent_issue_id, title,
                 issue_type.value, burden_side, materiality, salience,
                 "open", sort_order, now, now),
            )

        return issue_id, True

    def add_predicate(
        self,
        issue_id: str,
        description: str,
        burden_side: Optional[str] = None,
    ) -> str:
        """
        Add a testable predicate to an issue. Idempotent: if a predicate with the
        same (issue_id, description) already exists, the existing row is returned.
        Raises ValueError for blank descriptions.
        Returns predicate_id.
        """
        description = description.strip()
        if not description:
            raise ValueError("Predicate description must not be blank")
        pred_id = _id()
        now = _now()
        with self.db.transaction():
            self.db.execute(
                """INSERT OR IGNORE INTO issue_predicate
                   (id, issue_id, description, burden_side, status, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (pred_id, issue_id, description, burden_side, "open", now),
            )
        row = self.db.execute(
            "SELECT id FROM issue_predicate WHERE issue_id=? AND description=?",
            (issue_id, description),
        ).fetchone()
        return row["id"] if row else pred_id

    def add_predicates_batch(
        self,
        issue_id: str,
        descriptions: list[str],
        burden_side: Optional[str] = None,
    ) -> list[str]:
        """
        Add multiple predicates for an issue in a single transaction. Idempotent.
        Deduplicates inputs; duplicate descriptions are treated as one predicate.
        Returns unordered list of predicate IDs (existing or newly created).
        """
        now = _now()
        # Deduplicate while preserving first occurrence order; filter blanks.
        seen: set[str] = set()
        descs: list[str] = []
        for d in descriptions:
            if not isinstance(d, str):
                continue
            norm = d.strip()[:300]
            if norm and norm not in seen:
                seen.add(norm)
                descs.append(norm)
        if not descs:
            return []
        # SQLite hard-limits bind params to ~999; cap to avoid runtime errors.
        # At the current _orient() call site descs is capped to 4, so this is
        # a safety net for any future callers with larger lists.
        _SQL_PARAM_LIMIT = 900
        descs = descs[:_SQL_PARAM_LIMIT - 1]  # -1 for the issue_id param in SELECT
        with self.db.transaction():
            for d in descs:
                self.db.execute(
                    """INSERT OR IGNORE INTO issue_predicate
                       (id, issue_id, description, burden_side, status, created_at)
                       VALUES (?,?,?,?,?,?)""",
                    (_id(), issue_id, d, burden_side, "open", now),
                )
        rows = self.db.execute(
            f"SELECT id FROM issue_predicate"
            f" WHERE issue_id=? AND description IN ({','.join('?' * len(descs))})"
            f" ORDER BY created_at",
            [issue_id] + descs,
        ).fetchall()
        return [r["id"] for r in rows]

    def link_assertion(
        self,
        assertion_id: str,
        issue_id: str,
        relation_type: str = "supports",
    ) -> str:
        """
        Link an assertion to an issue. Idempotent.
        relation_type: 'supports', 'attacks', 'establishes', 'negates'
        Returns link_id.
        """
        link_id = _id()
        now = _now()
        with self.db.transaction():
            self.db.execute(
                """INSERT OR IGNORE INTO assertion_issue_link
                   (id, assertion_id, issue_id, relation_type, created_at)
                   VALUES (?,?,?,?,?)""",
                (link_id, assertion_id, issue_id, relation_type, now),
            )
        row = self.db.execute(
            "SELECT id FROM assertion_issue_link WHERE assertion_id=? AND issue_id=? AND relation_type=?",
            (assertion_id, issue_id, relation_type),
        ).fetchone()
        return row["id"] if row else link_id

    def get_open_issues(self, min_materiality: float = 0.0) -> list[dict]:
        """Return open issues ordered by salience × materiality descending."""
        rows = self.db.execute(
            """SELECT * FROM issue
               WHERE matter_id=? AND status='open' AND materiality >= ?
               ORDER BY (salience * materiality) DESC, id ASC""",
            (self.matter_id, min_materiality),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_assertions_for_issue(self, issue_id: str) -> list[dict]:
        """Return all assertions linked to an issue with their relation types."""
        rows = self.db.execute(
            """SELECT a.*, ail.relation_type
               FROM assertion a
               JOIN assertion_issue_link ail ON ail.assertion_id = a.id
               WHERE ail.issue_id=?
               ORDER BY ail.relation_type, a.belief_state""",
            (issue_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_predicates(self, issue_id: str, limit: Optional[int] = None) -> list[dict]:
        """Return open predicates for an issue, ordered by creation.

        limit: when provided, returns at most that many rows (DB-level bound).
        """
        if limit is not None:
            rows = self.db.execute(
                "SELECT id, issue_id, description, burden_side, status, created_at"
                " FROM issue_predicate WHERE issue_id=? AND status='open'"
                " ORDER BY created_at LIMIT ?",
                (issue_id, limit),
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT * FROM issue_predicate WHERE issue_id=? AND status='open'"
                " ORDER BY created_at",
                (issue_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_issue(self, issue_id: str) -> Optional[dict]:
        """Fetch a single issue by ID. Returns dict or None."""
        row = self.db.execute(
            "SELECT * FROM issue WHERE id=? AND matter_id=?",
            (issue_id, self.matter_id),
        ).fetchone()
        return dict(row) if row else None

    def count_open(self) -> int:
        row = self.db.execute(
            "SELECT COUNT(*) FROM issue WHERE matter_id=? AND status='open'",
            (self.matter_id,),
        ).fetchone()
        return row[0]


class ClarificationStore:
    """
    Manages clarification questions and their answers (SO-7, SO-3).

    Clarification questions are generated from high-materiality gaps.
    They are surfaced to the user between investigation runs.
    Answered questions are injected into the next orientation context.
    """

    def __init__(self, db: SQLiteMatterDB, matter_id: str):
        self.db = db
        self.matter_id = matter_id

    def add_question(
        self,
        question_text: str,
        why_it_matters: Optional[str] = None,
        expected_impact: Optional[str] = None,
        gap_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> str:
        """
        Record a clarification question. Returns question_id.
        Idempotent on (matter_id, question_text) — will not create duplicates.
        """
        now = _now()
        # Dedup: check if same question text already exists (pending OR answered).
        # Answered questions must not be re-issued on subsequent runs — the answer
        # is already captured and will be injected into the next orientation context.
        row = self.db.execute(
            "SELECT id FROM clarification_question "
            "WHERE matter_id=? AND question_text=?",
            (self.matter_id, question_text),
        ).fetchone()
        if row is not None:
            return row["id"]

        q_id = _id()
        with self.db.transaction():
            self.db.execute(
                """INSERT INTO clarification_question
                   (id, matter_id, gap_id, run_id, question_text, why_it_matters,
                    expected_impact, status, created_at)
                   VALUES (?,?,?,?,?,?,?,'pending',?)""",
                (q_id, self.matter_id, gap_id, run_id,
                 question_text, why_it_matters, expected_impact, now),
            )
        return q_id

    def answer_question(self, question_id: str, answer_text: str) -> None:
        """Record the user's answer to a clarification question."""
        now = _now()
        self.db.execute(
            """UPDATE clarification_question
               SET answer_text=?, answered_at=?, status='answered'
               WHERE id=?""",
            (answer_text, now, question_id),
        )

    def get_pending(self) -> list[dict]:
        """Return unanswered clarification questions, newest first."""
        rows = self.db.execute(
            """SELECT * FROM clarification_question
               WHERE matter_id=? AND status='pending'
               ORDER BY created_at DESC""",
            (self.matter_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_answered(self) -> list[dict]:
        """Return answered questions — for injection into orientation context."""
        rows = self.db.execute(
            """SELECT * FROM clarification_question
               WHERE matter_id=? AND status='answered'
               ORDER BY answered_at DESC""",
            (self.matter_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_answered_since(self, since_iso: str) -> list[dict]:
        """Return questions answered after since_iso — for mid-run active steering (SO-3)."""
        rows = self.db.execute(
            """SELECT * FROM clarification_question
               WHERE matter_id=? AND status='answered' AND answered_at > ?
               ORDER BY answered_at ASC""",
            (self.matter_id, since_iso),
        ).fetchall()
        return [dict(r) for r in rows]

    def count_pending(self) -> int:
        row = self.db.execute(
            "SELECT COUNT(*) FROM clarification_question "
            "WHERE matter_id=? AND status='pending'",
            (self.matter_id,),
        ).fetchone()
        return row[0]


class QuantStore:
    """
    Stores structured numeric facts extracted from documents (SO-6).

    quant_kind values: "amount", "date", "date_range", "rate", "balance", "count"
    """

    def __init__(self, db: SQLiteMatterDB, matter_id: str):
        self.db = db
        self.matter_id = matter_id

    def record(
        self,
        quant_kind: str,
        raw_text: str,
        amount_value: Optional[float] = None,
        currency: Optional[str] = None,
        date_value: Optional[str] = None,
        date_end_value: Optional[str] = None,
        rate_value: Optional[float] = None,
        unit: Optional[str] = None,
        subject_type: Optional[str] = None,
        subject_id: Optional[str] = None,
        assertion_id: Optional[str] = None,
        span_id: Optional[str] = None,
    ) -> str:
        """Persist a structured numeric fact. Returns quant_fact_id.

        Idempotent on (matter_id, quant_kind, raw_text[:500]) via INSERT OR IGNORE
        backed by ux_quant_fact_key unique index (added in migration v13).
        """
        qf_id = _id()
        now = _now()
        raw_key = raw_text[:500]
        with self.db.transaction():
            cur = self.db.execute(
                """INSERT OR IGNORE INTO quant_fact
                   (id, matter_id, quant_kind, amount_value, date_value, date_end_value,
                    rate_value, currency, unit, raw_text, subject_type, subject_id,
                    span_id, assertion_id, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (qf_id, self.matter_id, quant_kind, amount_value, date_value, date_end_value,
                 rate_value, currency, unit, raw_key, subject_type, subject_id,
                 span_id, assertion_id, now),
            )
            if cur.rowcount == 0:
                # Already exists — return the existing ID
                row = self.db.execute(
                    "SELECT id FROM quant_fact WHERE matter_id=? AND quant_kind=? AND raw_text=?",
                    (self.matter_id, quant_kind, raw_key),
                ).fetchone()
                return row["id"]
        return qf_id

    def record_many(self, specs: list[dict]) -> list[str]:
        """Bulk-insert multiple quant facts in a single transaction.

        Each spec is a dict with the same keys as record() (quant_kind and
        raw_text required; all others optional). Uses executemany + INSERT OR
        IGNORE so duplicate raw_text entries are silently skipped.

        Returns list of IDs (newly inserted or existing) in insertion order.
        """
        if not specs:
            return []
        now = _now()
        rows = []
        ids = []
        for spec in specs:
            qf_id = _id()
            ids.append(qf_id)
            rows.append((
                qf_id, self.matter_id,
                spec["quant_kind"],
                spec.get("amount_value"),
                spec.get("date_value"),
                spec.get("date_end_value"),
                spec.get("rate_value"),
                spec.get("currency"),
                spec.get("unit"),
                spec["raw_text"][:500],
                spec.get("subject_type"),
                spec.get("subject_id"),
                spec.get("span_id"),
                spec.get("assertion_id"),
                now,
            ))

        with self.db.transaction():
            self.db.executemany(
                """INSERT OR IGNORE INTO quant_fact
                   (id, matter_id, quant_kind, amount_value, date_value, date_end_value,
                    rate_value, currency, unit, raw_text, subject_type, subject_id,
                    span_id, assertion_id, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
        # Return candidate IDs; IDs for duplicate rows (IGNORED) are the candidate
        # UUIDs which won't match the stored row — callers that need exact IDs must
        # use record() individually. Engine call sites do not use the return value.
        return ids

    def get_by_kind(self, quant_kind: str, limit: Optional[int] = None) -> list[dict]:
        """Return quant facts of a given kind, sorted by date_value then created_at.

        limit: when provided, returns at most that many rows (DB-level bound).
        """
        if limit is not None:
            rows = self.db.execute(
                """SELECT * FROM quant_fact
                   WHERE matter_id=? AND quant_kind=?
                   ORDER BY date_value, created_at LIMIT ?""",
                (self.matter_id, quant_kind, limit),
            ).fetchall()
        else:
            rows = self.db.execute(
                """SELECT * FROM quant_fact
                   WHERE matter_id=? AND quant_kind=?
                   ORDER BY date_value, created_at""",
                (self.matter_id, quant_kind),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_amounts(self, min_value: Optional[float] = None) -> list[dict]:
        """Return all monetary amounts, optionally filtered by minimum value."""
        if min_value is not None:
            rows = self.db.execute(
                """SELECT * FROM quant_fact
                   WHERE matter_id=? AND quant_kind='amount' AND amount_value >= ?
                   ORDER BY amount_value DESC""",
                (self.matter_id, min_value),
            ).fetchall()
        else:
            rows = self.db.execute(
                """SELECT * FROM quant_fact
                   WHERE matter_id=? AND quant_kind='amount'
                   ORDER BY amount_value DESC NULLS LAST""",
                (self.matter_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        row = self.db.execute(
            "SELECT COUNT(*) FROM quant_fact WHERE matter_id=?",
            (self.matter_id,),
        ).fetchone()
        return row[0]

    def get_conflicts(self) -> list[dict]:
        """Return amount fact groups that have conflicting values for the same entity.

        Groups by (subject_type, COALESCE(subject_id, ''), currency) so that:
        - Two entries for Invoice #1042 with different amounts → conflict
        - Invoice #1042 ($50k) vs Invoice #2017 ($75k) → NOT a conflict (different subject_id)
        - Items without subject_id still group by subject_type (original coarse behavior)

        Returns one dict per conflict group with keys:
          subject_type, subject_id (or None), currency, values, raw_texts.
        """
        rows = self.db.execute(
            """SELECT subject_type, subject_id, currency,
                      COUNT(DISTINCT ROUND(amount_value, 2)) AS distinct_values,
                      GROUP_CONCAT(ROUND(amount_value, 2)) AS value_list,
                      GROUP_CONCAT(raw_text, ' || ') AS texts
               FROM quant_fact
               WHERE matter_id=? AND quant_kind='amount'
                 AND subject_type IS NOT NULL AND amount_value IS NOT NULL
               GROUP BY subject_type, COALESCE(subject_id, ''), currency
               HAVING distinct_values > 1
               ORDER BY distinct_values DESC""",
            (self.matter_id,),
        ).fetchall()
        conflicts = []
        for r in rows:
            r = dict(r)
            r["values"] = [float(v) for v in (r.pop("value_list") or "").split(",") if v]
            r.pop("distinct_values", None)
            conflicts.append(r)
        return conflicts

    def reconcile_by_subject(self, currency: str = "USD") -> dict:
        """Summarise amount facts grouped by subject_type for a given currency.

        Returns a dict mapping subject_type → {"total": float, "count": int, "facts": list}.
        Useful for payment reconciliation: compare 'invoice' totals vs 'payment' totals.
        """
        rows = self.db.execute(
            """SELECT subject_type,
                      SUM(amount_value) AS total,
                      COUNT(*) AS cnt
               FROM quant_fact
               WHERE matter_id=? AND quant_kind='amount'
                 AND (currency=? OR (currency IS NULL AND ?='USD'))
                 AND amount_value IS NOT NULL
               GROUP BY subject_type
               ORDER BY total DESC""",
            (self.matter_id, currency, currency),
        ).fetchall()
        result: dict[str, dict] = {}
        for r in rows:
            subject = r["subject_type"] or "unknown"
            result[subject] = {
                "total": round(r["total"], 2),
                "count": r["cnt"],
            }
        return result


class DocumentInventoryStore:
    """
    Tracks which documents have been ingested into the matter model.

    Uses the document_inventory table (defined in schema.py).  The primary purpose
    is to gate the cold/hot split in _deep_read_document(): a document whose
    ingest_status == 'complete' has already been fully analysed and stored in the
    assertion graph; the engine can skip the expensive LLM pass on re-runs.

    Dedup key: (matter_id, relative_path).  sha256 is also stored so a future
    "content changed" check can detect when a cold re-ingest is required.
    """

    def __init__(self, db: SQLiteMatterDB, matter_id: str):
        self.db = db
        self.matter_id = matter_id

    def upsert(
        self,
        relative_path: str,
        sha256: str,
        size_bytes: int = 0,
        file_type: Optional[str] = None,
    ) -> tuple[str, bool]:
        """Insert or locate a document_inventory row.

        Returns (doc_id, is_new).  is_new=True means this path has not been seen
        before in this matter; is_new=False means it already exists.

        Each (matter_id, relative_path) pair has exactly one row — same-content
        files at different paths are tracked independently.  Content-change detection
        is per-path via sha256 comparison; the old ux_inventory_hash cross-path dedup
        was removed in v6 because it caused is_ingested(path) to be inconsistent.
        """
        now = _now()
        doc_id = _id()
        # Keep the SELECT inside the same transaction as the INSERT so there is no
        # TOCTOU window where a concurrent coroutine could read a stale (or absent)
        # row between the two operations.
        with self.db.transaction():
            self.db.execute(
                """INSERT OR IGNORE INTO document_inventory
                   (id, matter_id, relative_path, sha256, size_bytes, file_type,
                    discovered_at, ingest_status, parse_status)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (doc_id, self.matter_id, relative_path, sha256,
                 size_bytes, file_type, now, "pending", "pending"),
            )
            # Fetch the row for this path (INSERT may have been a no-op if path already exists)
            row = self.db.execute(
                "SELECT id, ingest_status, sha256 FROM document_inventory WHERE matter_id=? AND relative_path=?",
                (self.matter_id, relative_path),
            ).fetchone()
            if row is not None and row["sha256"] != sha256:
                # Content changed at same path — reset to pending to force cold re-ingest
                self.db.execute(
                    "UPDATE document_inventory SET sha256=?, size_bytes=?, ingest_status='pending', last_read_at=? WHERE id=?",
                    (sha256, size_bytes, now, row["id"]),
                )
        actual_id = row["id"] if row else doc_id
        is_new = actual_id == doc_id  # True only if INSERT succeeded (no prior row for this path)
        return actual_id, is_new

    def mark_ingested(self, doc_id: str) -> None:
        """Set ingest_status='complete' and record last_read_at."""
        now = _now()
        self.db.execute(
            "UPDATE document_inventory SET ingest_status='complete', last_read_at=? WHERE id=?",
            (now, doc_id),
        )

    def is_ingested(self, relative_path: str) -> bool:
        """Return True if this document has already been fully ingested."""
        row = self.db.execute(
            "SELECT ingest_status FROM document_inventory WHERE matter_id=? AND relative_path=?",
            (self.matter_id, relative_path),
        ).fetchone()
        return row is not None and row["ingest_status"] == "complete"

    def get_ingested_paths(self) -> list[str]:
        """Return relative_paths of all fully-ingested documents."""
        rows = self.db.execute(
            "SELECT relative_path FROM document_inventory WHERE matter_id=? AND ingest_status='complete'",
            (self.matter_id,),
        ).fetchall()
        return [r["relative_path"] for r in rows]

    def count(self) -> int:
        row = self.db.execute(
            "SELECT COUNT(*) FROM document_inventory WHERE matter_id=?",
            (self.matter_id,),
        ).fetchone()
        return row[0]


class ReasoningCacheStore:
    """
    Cache for expensive LLM reasoning steps (e.g. orientation planning).

    Keyed on (matter_id, stage, cache_key) where cache_key is a
    sha256 hash of the query and repo-state proxy computed by the caller.
    On warm runs the engine checks this store before calling the LLM,
    skipping the FLASH model call when the query and repo structure are
    unchanged (SO-1 hot path).
    """

    def __init__(self, db: SQLiteMatterDB, matter_id: str):
        self.db = db
        self.matter_id = matter_id

    def get(self, stage: str, cache_key: str) -> Optional[dict]:
        """Return cached plan dict or None on cache miss or DB error.

        Wraps all DB access in try/except so a corrupt or missing cache table
        never prevents the calling code (engine._orient) from falling through
        to the LLM call.
        """
        import json
        try:
            row = self.db.execute(
                "SELECT id, plan_json FROM reasoning_cache"
                " WHERE matter_id=? AND stage=? AND cache_key=?",
                (self.matter_id, stage, cache_key),
            ).fetchone()
            if row is None:
                return None
            try:
                self.db.execute(
                    "UPDATE reasoning_cache SET last_hit_at=? WHERE id=?",
                    (_now(), row["id"]),
                )
            except Exception:
                pass  # last_hit_at update is non-critical
            return json.loads(row["plan_json"])
        except Exception:
            return None

    def put(self, stage: str, cache_key: str, plan: dict) -> None:
        """Upsert a cache entry (insert or overwrite on key collision).

        Silently ignores errors — cache failures must never break the calling path.
        """
        import json
        try:
            now = _now()
            self.db.execute(
                """INSERT INTO reasoning_cache
                   (id, matter_id, stage, cache_key, plan_json, created_at, last_hit_at)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(matter_id, stage, cache_key)
                   DO UPDATE SET plan_json=excluded.plan_json,
                                 last_hit_at=excluded.last_hit_at""",
                (_id(), self.matter_id, stage, cache_key, json.dumps(plan), now, now),
            )
        except Exception:
            pass  # non-critical; next run will populate from LLM


class TrustOverrideStore:
    """User-set trust overrides for specific documents (SO-3 trust steering, SO-5 calibration).

    A trust override lets a user mark a document pattern as 'low' or 'high' trust,
    which overrides the automatic source-role inference in record_fact().

    trust_level values:
      'low'    — force speech_act to ALLEGED regardless of source role
      'normal' — use automatic inference (default, no override)
      'high'   — promote ALLEGED → OPERATIVE (user asserts this document is authoritative)

    document_pattern is matched against:
      1. The full relative document_id (e.g. "pleadings/complaint.pdf")
      2. The basename only (e.g. "complaint.pdf")
    First match wins; patterns are checked by length descending (more specific wins).
    """

    def __init__(self, db: SQLiteMatterDB, matter_id: str):
        self.db = db
        self.matter_id = matter_id

    def set(
        self,
        document_pattern: str,
        trust_level: str,
        note: Optional[str] = None,
    ) -> str:
        """Upsert a trust override. Returns override id."""
        if trust_level not in ("low", "normal", "high"):
            raise ValueError(f"trust_level must be 'low', 'normal', or 'high', got {trust_level!r}")
        # Normalize to forward slashes so Windows backslash paths match API/user-supplied patterns
        document_pattern = document_pattern.replace("\\\\", "/").replace("\\", "/")
        _new_id = _id()
        now = _now()
        self.db.execute(
            """INSERT INTO document_trust_override
               (id, matter_id, document_pattern, trust_level, note, created_at)
               VALUES (?,?,?,?,?,?)
               ON CONFLICT(matter_id, document_pattern)
               DO UPDATE SET trust_level=excluded.trust_level,
                             note=excluded.note""",
            (_new_id, self.matter_id, document_pattern, trust_level, note, now),
        )
        # Return the actually stored ID — on ON CONFLICT path, _new_id is discarded
        row = self.db.execute(
            "SELECT id FROM document_trust_override WHERE matter_id=? AND document_pattern=?",
            (self.matter_id, document_pattern),
        ).fetchone()
        return row["id"] if row else _new_id

    def get(self, document_id: str) -> Optional[str]:
        """Return trust_level for a document_id, or None if no override set.

        Matches against full document_id path first, then basename only.
        Among multiple matching patterns, the longest (most specific) wins.
        Path separators are normalized to forward slashes to handle Windows paths.
        """
        from pathlib import Path
        # Normalize both sides to forward slashes before comparison
        _doc_normalized = document_id.replace("\\\\", "/").replace("\\", "/")
        basename = Path(_doc_normalized).name
        rows = self.db.execute(
            """SELECT document_pattern, trust_level FROM document_trust_override
               WHERE matter_id=? AND trust_level != 'normal'
               ORDER BY LENGTH(document_pattern) DESC""",
            (self.matter_id,),
        ).fetchall()
        for row in rows:
            pattern = row["document_pattern"].replace("\\\\", "/").replace("\\", "/")
            if pattern == _doc_normalized or pattern == basename:
                return row["trust_level"]
        return None

    def list_all(self) -> list[dict]:
        """Return all trust overrides for this matter."""
        rows = self.db.execute(
            """SELECT id, document_pattern, trust_level, note, created_at
               FROM document_trust_override WHERE matter_id=?
               ORDER BY created_at DESC""",
            (self.matter_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete(self, document_pattern: str) -> bool:
        """Remove a trust override. Returns True if a row was deleted."""
        self.db.execute(
            "DELETE FROM document_trust_override WHERE matter_id=? AND document_pattern=?",
            (self.matter_id, document_pattern),
        )
        return True


class DocumentAnnotationStore:
    """User-authored strategic notes attached to document patterns (SO-3 annotation).

    Annotations are injected into the orientation prompt so the engine can
    use the user's domain knowledge when building its investigation plan.

    annotation_type values:
      'strategic' — broad strategic context (e.g., "this report overstates damages")
      'reliability' — reliability note (e.g., "chain of custody issues")
      'scope' — scope/relevance note (e.g., "irrelevant to core claim, skip")
    """

    def __init__(self, db: SQLiteMatterDB, matter_id: str):
        self.db = db
        self.matter_id = matter_id

    def add(
        self,
        document_pattern: str,
        annotation_text: str,
        annotation_type: str = "strategic",
    ) -> str:
        """Add an annotation. Returns annotation_id."""
        ann_id = _id()
        now = _now()
        self.db.execute(
            """INSERT INTO document_annotation
               (id, matter_id, document_pattern, annotation_text, annotation_type, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?)""",
            (ann_id, self.matter_id, document_pattern, annotation_text, annotation_type, now, now),
        )
        return ann_id

    def get_for_document(self, document_id: str) -> list[dict]:
        """Return all annotations matching a document_id (by full path or basename)."""
        from pathlib import Path
        basename = Path(document_id).name
        rows = self.db.execute(
            """SELECT id, document_pattern, annotation_text, annotation_type, created_at
               FROM document_annotation
               WHERE matter_id=? AND (document_pattern=? OR document_pattern=?)
               ORDER BY created_at ASC""",
            (self.matter_id, document_id, basename),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_recent(self, limit: int = 20) -> list[dict]:
        """Return recent annotations across all documents."""
        rows = self.db.execute(
            """SELECT id, document_pattern, annotation_text, annotation_type, created_at
               FROM document_annotation
               WHERE matter_id=?
               ORDER BY created_at DESC LIMIT ?""",
            (self.matter_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete(self, annotation_id: str) -> bool:
        """Remove an annotation by id."""
        self.db.execute(
            "DELETE FROM document_annotation WHERE matter_id=? AND id=?",
            (self.matter_id, annotation_id),
        )
        return True
