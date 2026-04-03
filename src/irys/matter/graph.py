"""AssertionStore, GapStore, ActorStore, IssueStore, ClarificationStore, QuantStore,
DocumentInventoryStore, TrustOverrideStore, DocumentAnnotationStore, ReasoningCacheStore.

The assertion store is the heart of the intelligence layer. It maintains
typed assertions with speech-act classification, support/attack links,
and revisable belief states.
"""

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Optional

from .db import SQLiteMatterDB
import pathlib
from .enums import (
    BeliefState, SpeechAct, SourceRole, ModelLayer, AssertionKind,
    AssertionLinkType, OriginKind, GapType, IssueType, SOURCE_TRUST_WEIGHTS,
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
            _doc_norm = (candidate.document_id or "").replace("\\\\", "/").replace("\\", "/")
            _doc_basename = pathlib.Path(_doc_norm).name if _doc_norm else None
            _occ_cur = self.db.execute(
                """INSERT OR IGNORE INTO assertion_occurrence
                   (id, assertion_id, document_id, doc_basename, span_id,
                    speaker_actor_id, source_role, source_side,
                    speech_act, origin_kind, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    occ_id, assertion_id,
                    candidate.document_id,
                    _doc_basename,
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
            cursor = self.db.execute(
                """INSERT OR IGNORE INTO assertion_link
                   (id, src_assertion_id, dst_assertion_id, link_type, weight, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (link_id, src_id, dst_id, link_type.value, weight, now),
            )
            if cursor.rowcount == 1:
                # New row inserted — return the id we just wrote.
                return link_id
        # Row already existed (UNIQUE conflict) — look up the existing id.
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

    def get_neighbor_belief_states(
        self,
        assertion_id: str,
        _override_cache: "list[tuple[str, str]] | None" = None,
    ) -> dict:
        """Batch-load support/attack/supersedes neighbor belief states in one query.

        Returns dict with keys:
          'support_states', 'attack_states', 'has_superseding' — as before,
          'support_source_roles', 'attack_source_roles' — effective source_role per
            neighboring assertion (SO-5 trust-gated belief revision).

        Effective source_role is derived in two steps:
        1. Best stored source_role from assertion_occurrence (highest-trust across occurrences).
        2. If a user-set document_trust_override matches the assertion's primary document,
           the override replaces the stored role: 'low' → 'advocacy', 'high' → 'operative'.
           This ensures user trust steering (SO-3) flows through belief revision (SO-2).

        Falls back to 'unknown' when no occurrences exist.

        _override_cache: pre-fetched list of (document_pattern, trust_level) tuples.
            When provided, skips the per-call DB query (use from BeliefRevisionEngine.apply()
            to avoid N override queries during BFS). When None, fetches from DB as before.
        """
        # CTE-based query: precomputes best source_role and primary_document_id for
        # all neighbor assertions in a single assertion_occurrence scan, replacing
        # 2×N correlated scalar subqueries with one set-based pass (SO-5).
        rows = self.db.execute(
            """WITH linked AS (
                   SELECT al.link_type, al.src_assertion_id
                   FROM assertion_link al
                   WHERE al.dst_assertion_id = ?
                     AND al.link_type IN ('supports','corroborates','attacks','contradicts','supersedes')
               ),
               occ_ranked AS (
                   SELECT ao.assertion_id,
                          ao.source_role,
                          ao.document_id,
                          ROW_NUMBER() OVER (
                              PARTITION BY ao.assertion_id
                              ORDER BY CASE ao.source_role
                                  WHEN 'authoritative' THEN 6
                                  WHEN 'operative'     THEN 5
                                  WHEN 'procedural'    THEN 4
                                  WHEN 'post_hoc'      THEN 3
                                  WHEN 'informal'      THEN 2
                                  WHEN 'draft'         THEN 1
                                  WHEN 'unknown'       THEN 1
                                  WHEN 'advocacy'      THEN 0
                                  ELSE 1 END DESC
                          ) AS role_rn,
                          ROW_NUMBER() OVER (
                              PARTITION BY ao.assertion_id
                              ORDER BY ao.created_at ASC, ao.id ASC
                          ) AS doc_rn
                   FROM assertion_occurrence ao
                   WHERE ao.assertion_id IN (SELECT src_assertion_id FROM linked)
               )
               SELECT l.link_type,
                      a.belief_state,
                      COALESCE(MAX(CASE WHEN o.role_rn = 1 THEN o.source_role END), 'unknown')
                          AS source_role,
                      MAX(CASE WHEN o.doc_rn = 1 THEN o.document_id END)
                          AS primary_document_id
               FROM linked l
               JOIN assertion a ON a.id = l.src_assertion_id
               LEFT JOIN occ_ranked o ON o.assertion_id = l.src_assertion_id
               GROUP BY l.src_assertion_id, l.link_type, a.belief_state""",
            (assertion_id,),
        ).fetchall()

        # Use pre-fetched override cache when available (avoids N DB queries in BFS).
        if _override_cache is not None:
            overrides = _override_cache
        else:
            override_rows = self.db.execute(
                """SELECT document_pattern, trust_level FROM document_trust_override
                   WHERE matter_id=? AND trust_level != 'normal'
                   ORDER BY LENGTH(document_pattern) DESC""",
                (self.matter_id,),
            ).fetchall()
            overrides = [(r["document_pattern"], r["trust_level"]) for r in override_rows]

        def _effective_role(source_role: str, document_id: "str | None") -> str:
            """Apply trust override if any pattern matches the document. Otherwise keep role."""
            if not overrides or not document_id:
                return source_role
            doc_norm = document_id.replace("\\\\", "/").replace("\\", "/")
            basename = pathlib.Path(doc_norm).name
            for pattern, level in overrides:
                pat_norm = pattern.replace("\\\\", "/").replace("\\", "/")
                if pat_norm == doc_norm or pat_norm == basename:
                    return "advocacy" if level == "low" else "operative"
            return source_role

        support_states: list = []
        attack_states: list = []
        support_source_roles: list = []
        attack_source_roles: list = []
        has_superseding = False
        for row in rows:
            lt = row["link_type"]
            bs = BeliefState(row["belief_state"])
            role = _effective_role(row["source_role"], row["primary_document_id"])
            if lt in ("supports", "corroborates"):
                support_states.append(bs)
                support_source_roles.append(role)
            elif lt in ("attacks", "contradicts"):
                attack_states.append(bs)
                attack_source_roles.append(role)
            elif lt == "supersedes":
                has_superseding = True
        return {
            "support_states": support_states,
            "attack_states": attack_states,
            "has_superseding": has_superseding,
            "support_source_roles": support_source_roles,
            "attack_source_roles": attack_source_roles,
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
                      a.subject_ref_type, a.subject_ref_id,
                      a.predicate_key, a.object_json,
                      (SELECT ao.source_role FROM assertion_occurrence ao
                       WHERE ao.assertion_id = a.id
                       ORDER BY CASE ao.source_role
                         WHEN 'authoritative' THEN 6
                         WHEN 'operative'     THEN 5
                         WHEN 'procedural'    THEN 4
                         WHEN 'post_hoc'      THEN 3
                         WHEN 'informal'      THEN 2
                         WHEN 'draft'         THEN 1
                         WHEN 'unknown'       THEN 1
                         WHEN 'advocacy'      THEN 0
                         ELSE 1 END DESC, ao.created_at ASC LIMIT 1) AS source_role
               FROM assertion a
               WHERE a.matter_id=?
                 AND a.belief_state NOT IN ('disputed','withdrawn','superseded')
               ORDER BY a.created_at DESC
               LIMIT ?""",
            (self.matter_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_by_proposition(
        self,
        proposition_text: str,
        model_layer: Optional[str] = None,
    ) -> Optional["AssertionRecord"]:
        """Look up an assertion by normalized proposition text.

        If *model_layer* is supplied the lookup is restricted to that layer,
        which is the correct behaviour when the caller knows which reasoning
        layer the assertion lives in (record / reality / proof / legal /
        decision_context).  Without an explicit layer the query can return an
        assertion from any layer — callers should supply the layer whenever
        possible to avoid cross-layer leakage.
        """
        import hashlib
        normalized = " ".join(proposition_text.lower().split())
        prop_key = hashlib.sha256(normalized.encode()).hexdigest()[:32]
        if model_layer is not None:
            row = self.db.execute(
                "SELECT * FROM assertion"
                " WHERE matter_id=? AND model_layer=? AND proposition_key=?",
                (self.matter_id, model_layer, prop_key),
            ).fetchone()
        else:
            # No layer specified: return the earliest-created assertion with this
            # proposition key across all layers.  ORDER BY + LIMIT 1 ensures
            # deterministic output even when the same text exists in multiple layers.
            # Callers should supply model_layer to avoid cross-layer leakage.
            row = self.db.execute(
                "SELECT * FROM assertion"
                " WHERE matter_id=? AND proposition_key=?"
                " ORDER BY created_at ASC LIMIT 1",
                (self.matter_id, prop_key),
            ).fetchone()
        if row is None:
            return None
        return AssertionRecord(**dict(row))

    # ------------------------------------------------------------------
    # Contradiction mining (background maintenance — spec §26)
    # ------------------------------------------------------------------

    _ACTIVE_STATES = ("alleged", "argued", "admitted", "operative", "inferred", "partial")
    _INACTIVE_STATES = ("superseded", "withdrawn", "resolved")

    def find_contradictions(self) -> list[dict]:
        """
        Return all pairs of assertions in this matter that are in active conflict.

        Detection strategy: explicit assertion_link rows where link_type IN
        ('attacks', 'contradicts') and BOTH the src and dst assertions have
        an active belief_state (not superseded/withdrawn/resolved).

        Link direction convention:
          src_assertion_id --ATTACKS/CONTRADICTS--> dst_assertion_id
          src is the attacker; dst is the assertion being challenged.

        Returns list of dicts:
          {
            'attacker_id':     str,
            'attacked_id':     str,
            'link_type':       'attacks' | 'contradicts',
            'attacker_belief': str,
            'attacked_belief': str,
            'attacker_prop':   str,
            'attacked_prop':   str,
          }
        """
        rows = self.db.execute(
            """SELECT al.src_assertion_id AS attacker_id,
                      al.dst_assertion_id AS attacked_id,
                      al.link_type,
                      a_src.belief_state AS attacker_belief,
                      a_dst.belief_state AS attacked_belief,
                      a_src.proposition_text AS attacker_prop,
                      a_dst.proposition_text AS attacked_prop
               FROM assertion_link al
               JOIN assertion a_src ON a_src.id = al.src_assertion_id
               JOIN assertion a_dst ON a_dst.id = al.dst_assertion_id
               WHERE al.link_type IN ('attacks', 'contradicts')
                 AND a_src.matter_id = ?
                 AND a_dst.matter_id = ?
                 AND a_src.belief_state NOT IN ('superseded','withdrawn','resolved')
                 AND a_dst.belief_state NOT IN ('superseded','withdrawn','resolved')""",
            (self.matter_id, self.matter_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def detect_heuristic_contradictions(self) -> int:
        """
        Heuristically detect and link contradicting assertion pairs (SO-2).

        For each open issue, compare active assertion pairs on the same issue using:
        1. Word overlap (Jaccard ≥ 0.4 on significant tokens) — same topic.
        2. Negation asymmetry (exactly one contains negation markers) — opposite polarity.

        When both conditions hold, creates a 'contradicts' link (negated assertion
        points at the positive one). This allows mine_and_mark_contradictions() to
        propagate belief state for autonomously detected conflicts, not just pre-linked ones.

        Caps at 50 assertions per issue to prevent O(n²) explosion on large graphs.
        Returns the number of new contradiction links created.
        """
        # Conservative negation markers only: unambiguous grammatical negation.
        # Removed: 'failed', 'denied', 'refused', 'rejected', 'absent', 'missing',
        # 'void' — these carry independent semantic content and trigger false positives
        # (e.g. 'failed to deliver' vs 'failed to pay' are not contradictions).
        _NEGATION_WORDS = frozenset(
            ["not", "never", "cannot", "can't", "didn't", "wasn't", "hasn't",
             "haven't", "don't", "doesn't", "isn't", "aren't"]
        )
        _STOP_WORDS = frozenset(
            ["the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
             "with", "by", "from", "as", "this", "that", "these", "those"]
        )
        _SIMILARITY_THRESHOLD = 0.4
        _MIN_OVERLAP_COUNT = 2   # require ≥2 shared significant words, not just ratio
        _MAX_PER_ISSUE = 50

        def _tokenize(text: str) -> frozenset:
            return frozenset(
                w.strip(".,;:()\"'!?")
                for w in text.lower().split()
                if len(w.strip(".,;:()\"'!?")) > 3
                and w.strip(".,;:()\"'!?") not in _STOP_WORDS
            )

        def _has_negation(text: str) -> bool:
            return any(
                w.strip(".,;:()\"'!?") in _NEGATION_WORDS
                for w in text.lower().split()
            )

        issue_rows = self.db.execute(
            "SELECT id FROM issue WHERE matter_id=? AND status='open'",
            (self.matter_id,),
        ).fetchall()

        existing_links: set = set()
        for r in self.db.execute(
            "SELECT src_assertion_id, dst_assertion_id FROM assertion_link "
            "WHERE link_type IN ('attacks','contradicts')"
        ).fetchall():
            existing_links.add((r["src_assertion_id"], r["dst_assertion_id"]))
            existing_links.add((r["dst_assertion_id"], r["src_assertion_id"]))

        created = 0
        now = _now()
        for issue_row in issue_rows:
            issue_id = issue_row["id"]
            rows = self.db.execute(
                """SELECT a.id, a.proposition_text
                   FROM assertion a
                   JOIN assertion_issue_link ail ON ail.assertion_id = a.id
                   WHERE ail.issue_id=? AND a.matter_id=?
                     AND a.belief_state NOT IN ('superseded','withdrawn','resolved')
                   ORDER BY a.created_at
                   LIMIT ?""",
                (issue_id, self.matter_id, _MAX_PER_ISSUE),
            ).fetchall()
            assertions = [dict(r) for r in rows]
            if len(assertions) < 2:
                continue

            for i in range(len(assertions)):
                for j in range(i + 1, len(assertions)):
                    a, b = assertions[i], assertions[j]
                    if (a["id"], b["id"]) in existing_links:
                        continue
                    ta = _tokenize(a["proposition_text"])
                    tb = _tokenize(b["proposition_text"])
                    if not ta or not tb:
                        continue
                    union = ta | tb
                    if not union:
                        continue
                    intersection = ta & tb
                    # Require both ratio threshold AND minimum absolute overlap count
                    # to reduce false positives from short assertions with few tokens.
                    if len(intersection) < _MIN_OVERLAP_COUNT:
                        continue
                    similarity = len(intersection) / len(union)
                    if similarity < _SIMILARITY_THRESHOLD:
                        continue
                    neg_a = _has_negation(a["proposition_text"])
                    neg_b = _has_negation(b["proposition_text"])
                    if neg_a == neg_b:
                        continue  # both same polarity → not a contradiction
                    src_id = a["id"] if neg_a else b["id"]
                    dst_id = b["id"] if neg_a else a["id"]
                    try:
                        self.db.execute(
                            """INSERT OR IGNORE INTO assertion_link
                               (id, src_assertion_id, dst_assertion_id, link_type,
                                weight, created_at)
                               VALUES (?,?,?,?,?,?)""",
                            (_id(), src_id, dst_id, "contradicts", 0.7, now),
                        )
                        existing_links.add((src_id, dst_id))
                        existing_links.add((dst_id, src_id))
                        created += 1
                    except Exception:
                        pass
        return created

    def mine_and_mark_contradictions(
        self,
        gap_store: "GapStore",
        belief_engine: "BeliefRevisionEngine",
    ) -> list[dict]:
        """
        Run contradiction mining and enforce belief states.

        Step 1: auto-detect heuristic contradictions (SO-2) — creates 'contradicts'
        links for assertion pairs that share topic context but have opposite polarity,
        so mining is not limited to pre-existing manually-created links.

        Step 2: for each conflict pair found by find_contradictions():
        - If the attacker has OPERATIVE or ADMITTED belief_state AND the attacked
          assertion is still OPERATIVE or ALLEGED: mark the attacked assertion as
          DISPUTED via the belief revision engine (cause=CONFLICT_DETECTION).
        - Always record an UNRESOLVED_CONTRADICTION gap for open conflicts.
        - Skip pairs where the attacked assertion is already DISPUTED.

        Returns the list of contradiction dicts (same shape as find_contradictions).
        """
        from .enums import RevisionCause, BeliefState, GapType
        # SO-2: auto-discover heuristic contradictions before propagating
        self.detect_heuristic_contradictions()
        conflicts = self.find_contradictions()
        _HIGH_TRUST = {"operative", "admitted"}
        _DISPUTABLE = {"operative", "alleged", "argued", "inferred"}

        for conflict in conflicts:
            attacked_id = conflict["attacked_id"]
            attacker_belief = conflict["attacker_belief"]
            attacked_belief = conflict["attacked_belief"]

            # Keep each operation isolated so force_state failure does not also
            # suppress gap recording (SO-7 signal must not be lost when belief
            # revision fails on the same pair).
            try:
                # Mark attacked assertion as DISPUTED when attacker is high-trust.
                if (
                    attacker_belief in _HIGH_TRUST
                    and attacked_belief in _DISPUTABLE
                ):
                    belief_engine.force_state(
                        assertion_id=attacked_id,
                        new_state=BeliefState.DISPUTED,
                        new_confidence=0.3,
                        cause=RevisionCause.CONFLICT_DETECTION,
                        note=(
                            f"Marked disputed by {conflict['link_type']} link from "
                            f"assertion {conflict['attacker_id']}"
                        ),
                    )
            except Exception:
                pass  # belief revision failure does not abort gap recording

            try:
                # Record gap for any open conflict that isn't already resolved.
                if attacked_belief not in ("superseded", "withdrawn", "resolved"):
                    gap_store.record(
                        gap_type=GapType.UNRESOLVED_CONTRADICTION,
                        description=(
                            f"Contradiction: '{conflict['attacker_prop'][:80]}' "
                            f"{conflict['link_type']} '{conflict['attacked_prop'][:80]}'"
                        ),
                        materiality=0.7,
                        affected_type="assertion",
                        affected_id=attacked_id,
                    )
            except Exception:
                pass  # gap recording failure does not abort the mining pass

        return conflicts


class GapStore:
    """Tracks structured missingness — documents, predicates, authorities, etc."""

    def __init__(self, db: SQLiteMatterDB, matter_id: str):
        self.db = db
        self.matter_id = matter_id

    @staticmethod
    def _gap_key(gap_type_value: str, description: str) -> str:
        """Compute dedup key: sha256(gap_type:normalized_description)[:32]."""
        normalized = description.lower().strip()
        return hashlib.sha256(f"{gap_type_value}:{normalized}".encode()).hexdigest()[:32]

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
        """Record a gap. Returns gap_id (existing, reopened, or new).

        Idempotent with three cases (SO-7 dedup):
        - Open gap exists with same description → return existing id (no-op)
        - Closed gap exists with same description → reopen it, return its id
        - No matching gap → insert new, return new id

        This prevents repeated runs from duplicating the open-gap list while
        still allowing gaps that were resolved and then re-detected to resurface.
        """
        gap_id = _id()
        now = _now()
        desc_key = self._gap_key(gap_type.value, description)
        with self.db.transaction():
            existing = self.db.execute(
                """SELECT id, status FROM gap
                   WHERE matter_id=? AND gap_type=? AND description_key=?
                   ORDER BY created_at DESC LIMIT 1""",
                (self.matter_id, gap_type.value, desc_key),
            ).fetchone()
            if existing:
                gap_id = existing["id"]
                if existing["status"] != "open":
                    # Reopen closed/resolved gap that is still relevant
                    self.db.execute(
                        "UPDATE gap SET status='open', updated_at=? WHERE id=?",
                        (now, gap_id),
                    )
            else:
                self.db.execute(
                    """INSERT INTO gap
                       (id, matter_id, gap_type, description, description_key,
                        expected_artifact, materiality_score, blocker_score,
                        status, created_at, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (gap_id, self.matter_id, gap_type.value, description, desc_key,
                     expected_artifact, materiality, blocker_score, "open", now, now),
                )
            if affected_type and affected_id:
                existing_link = self.db.execute(
                    "SELECT id FROM gap_link WHERE gap_id=? AND affected_type=? AND affected_id=?",
                    (gap_id, affected_type, affected_id),
                ).fetchone()
                if not existing_link:
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

        Idempotent: duplicate descriptions (same matter/type/description) return
        existing gap_ids; closed gaps matching a spec are reopened. (SO-7)

        Uses batch SELECT + executemany INSERT for performance — avoids the
        N×(SELECT+INSERT) per-row overhead of calling record() sequentially.

        Returns list of gap IDs in insertion order.
        """
        if not specs:
            return []
        now = _now()
        # Compute (gt, desc_key) for every spec in original order (for return mapping).
        # Deduplicate keys for DB operations — keep first-occurrence spec data per key.
        # SQLite bind-parameter limit: 1 + 2*N params per chunk; cap chunk at 200.
        _BATCH_CHUNK = 200
        spec_keys: list[tuple] = []       # parallel to specs, for final return mapping
        deduped: dict[tuple, dict] = {}   # key → spec data (first occurrence wins)
        for spec in specs:
            gt_val = spec["gap_type"].value if hasattr(spec["gap_type"], "value") else spec["gap_type"]
            desc_key = self._gap_key(gt_val, spec["description"])
            key = (gt_val, desc_key)
            spec_keys.append(key)
            if key not in deduped:
                deduped[key] = {
                    "id": _id(),
                    "gt": gt_val,
                    "description": spec["description"],
                    "desc_key": desc_key,
                    "expected_artifact": spec.get("expected_artifact"),
                    "materiality": spec.get("materiality", 0.5),
                    "blocker_score": spec.get("blocker_score", 0.0),
                    "affected_type": spec.get("affected_type"),
                    "affected_id": spec.get("affected_id"),
                }

        unique_items = list(deduped.values())
        all_keys = [(n["gt"], n["desc_key"]) for n in unique_items]

        with self.db.transaction():
            # Batch lookup in chunks to stay within SQLite param limits
            existing_map: dict[tuple, dict] = {}
            for chunk_start in range(0, len(all_keys), _BATCH_CHUNK):
                chunk = all_keys[chunk_start:chunk_start + _BATCH_CHUNK]
                placeholders = ",".join("(?,?)" for _ in chunk)
                flat_params = [v for pair in chunk for v in pair]
                rows = self.db.execute(
                    f"""SELECT id, gap_type, description_key, status
                        FROM gap
                        WHERE matter_id=? AND (gap_type, description_key) IN ({placeholders})
                        ORDER BY created_at DESC""",
                    [self.matter_id] + flat_params,
                ).fetchall()
                for r in rows:
                    k = (r["gap_type"], r["description_key"])
                    if k not in existing_map:  # keep most recent per key
                        existing_map[k] = dict(r)

            # Resolve each unique item to its final gap_id
            key_to_id: dict[tuple, str] = {}
            insert_rows = []
            reopen_ids = []
            link_checks = []

            for n in unique_items:
                key = (n["gt"], n["desc_key"])
                ex = existing_map.get(key)
                if ex:
                    gap_id = ex["id"]
                    if ex["status"] != "open":
                        reopen_ids.append((now, gap_id))
                else:
                    gap_id = n["id"]
                    insert_rows.append((
                        gap_id, self.matter_id, n["gt"], n["description"], n["desc_key"],
                        n["expected_artifact"], n["materiality"], n["blocker_score"],
                        "open", now, now,
                    ))
                key_to_id[key] = gap_id

            # Collect gap_links from ALL original specs (not just deduped first-occurrence)
            # so intra-batch duplicates with different affected_type/affected_id targets
            # all get their dependency links recorded. (SO-7 dependency lineage)
            link_checks = []
            for spec, key in zip(specs, spec_keys):
                gap_id = key_to_id.get(key)
                if gap_id and spec.get("affected_type") and spec.get("affected_id"):
                    link_checks.append((gap_id, spec["affected_type"], spec["affected_id"]))

            if insert_rows:
                self.db.executemany(
                    """INSERT INTO gap
                       (id, matter_id, gap_type, description, description_key,
                        expected_artifact, materiality_score, blocker_score,
                        status, created_at, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    insert_rows,
                )
            if reopen_ids:
                self.db.executemany(
                    "UPDATE gap SET status='open', updated_at=? WHERE id=?",
                    reopen_ids,
                )
            for gap_id, at, ai in link_checks:
                existing_link = self.db.execute(
                    "SELECT id FROM gap_link WHERE gap_id=? AND affected_type=? AND affected_id=?",
                    (gap_id, at, ai),
                ).fetchone()
                if not existing_link:
                    self.db.execute(
                        "INSERT INTO gap_link (id, gap_id, affected_type, affected_id, created_at) VALUES (?,?,?,?,?)",
                        (_id(), gap_id, at, ai, now),
                    )
        # Return in original spec order; intra-batch duplicates share the same id
        return [key_to_id[k] for k in spec_keys]

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

    def resolve_by_name(self, name: str) -> Optional[str]:
        """Resolve an actor_id from a name string.

        Resolution order:
        1. Exact alias match (get_by_alias)
        2. Substring containment: canonical name contains the normalized name,
           or normalized name contains the canonical name (prefix overlap ≥ 5 chars)

        Returns actor_id or None if no match found.
        """
        normalized = self._normalize(name)

        # Step 1: exact alias match.
        actor_id = self.get_by_alias(name)
        if actor_id:
            return actor_id

        # Step 2: substring containment.
        if len(normalized) < 5:
            return None  # too short for fuzzy matching — risk of false positives

        rows = self.db.execute(
            "SELECT id, normalized_name FROM actor WHERE matter_id=?",
            (self.matter_id,),
        ).fetchall()
        for row in rows:
            cname = row["normalized_name"] or ""
            if normalized in cname or cname in normalized:
                return row["id"]

        return None

    def find_possible_duplicates(
        self, min_prefix_len: int = 6
    ) -> list[dict]:
        """Return pairs of actors whose normalized names share a common prefix.

        Each entry: {actor_a: {...}, actor_b: {...}, shared_prefix: str}
        Only pairs where both actors have different ids are returned.
        Ordered by shared_prefix length descending (most similar first).
        """
        rows = self.db.execute(
            "SELECT id, canonical_name, normalized_name, actor_type FROM actor WHERE matter_id=? ORDER BY normalized_name",
            (self.matter_id,),
        ).fetchall()

        actors = [dict(r) for r in rows]
        pairs = []
        seen_pairs: set = set()

        for i, a in enumerate(actors):
            for b in actors[i + 1:]:
                n_a = a.get("normalized_name") or ""
                n_b = b.get("normalized_name") or ""
                # Find common prefix length.
                prefix_len = 0
                for ca, cb in zip(n_a, n_b):
                    if ca == cb:
                        prefix_len += 1
                    else:
                        break
                if prefix_len >= min_prefix_len:
                    pair_key = (min(a["id"], b["id"]), max(a["id"], b["id"]))
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        pairs.append({
                            "actor_a": a,
                            "actor_b": b,
                            "shared_prefix": n_a[:prefix_len],
                        })

        pairs.sort(key=lambda x: len(x["shared_prefix"]), reverse=True)
        return pairs

    def merge_actors(self, keep_id: str, merge_id: str) -> None:
        """Merge merge_id into keep_id, then delete merge_id.

        Moves:
        - actor_alias rows (INSERT OR IGNORE to avoid duplicate conflicts)
        - assertion_occurrence.speaker_actor_id references

        The keep_id actor is not modified (canonical name stays as-is).
        The merge_id actor row is deleted after migration.

        Raises ValueError if either id is not found or if keep_id == merge_id.
        """
        if keep_id == merge_id:
            raise ValueError("Cannot merge an actor with itself")

        # Verify both actors exist in this matter.
        for aid in (keep_id, merge_id):
            row = self.db.execute(
                "SELECT id FROM actor WHERE id=? AND matter_id=?",
                (aid, self.matter_id),
            ).fetchone()
            if row is None:
                raise ValueError(f"Actor {aid} not found in matter {self.matter_id}")

        now = _now()
        with self.db.transaction():
            # Move aliases.
            merge_aliases = self.db.execute(
                "SELECT alias_text, alias_type FROM actor_alias WHERE actor_id=?",
                (merge_id,),
            ).fetchall()
            for alias_row in merge_aliases:
                self.db.execute(
                    """INSERT OR IGNORE INTO actor_alias
                       (id, actor_id, alias_text, alias_type, created_at)
                       VALUES (?,?,?,?,?)""",
                    (_id(), keep_id, alias_row["alias_text"], alias_row["alias_type"], now),
                )

            # Redirect assertion occurrences.
            self.db.execute(
                "UPDATE assertion_occurrence SET speaker_actor_id=? WHERE speaker_actor_id=?",
                (keep_id, merge_id),
            )

            # Remove old alias rows for merge_id (already copied to keep_id above).
            self.db.execute("DELETE FROM actor_alias WHERE actor_id=?", (merge_id,))

            # Remove affiliation rows referencing merge_id.
            self.db.execute(
                "DELETE FROM actor_affiliation WHERE actor_id=? OR org_actor_id=?",
                (merge_id, merge_id),
            )

            # Delete the merged actor.
            self.db.execute(
                "DELETE FROM actor WHERE id=?",
                (merge_id,),
            )


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

    def resolve_predicate(self, predicate_id: str) -> bool:
        """Mark an issue predicate as resolved (SO-4 predicate-aware coverage).

        Returns True if the predicate existed and was updated; False if not found.
        Called when a supporting assertion is confirmed to satisfy a claim element.
        """
        with self.db.transaction():
            cursor = self.db.execute(
                "UPDATE issue_predicate SET status='resolved' WHERE id=?",
                (predicate_id,),
            )
        return cursor.rowcount > 0

    def resolve_predicate_by_description(self, issue_id: str, description: str) -> bool:
        """Mark the first open predicate matching description as resolved.

        Returns True if a predicate was found and resolved; False otherwise.
        Useful when the engine knows a predicate description was satisfied but does not
        have the predicate_id.
        """
        row = self.db.execute(
            "SELECT id FROM issue_predicate WHERE issue_id=? AND description=? AND status='open'",
            (issue_id, description.strip()[:300]),
        ).fetchone()
        if row is None:
            return False
        return self.resolve_predicate(row["id"])

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

    @staticmethod
    def _quant_key(quant_kind: str, subject_id: Optional[str], raw_text: str) -> str:
        """Content-addressed dedup key for a quant fact (SO-6).

        Incorporates subject_id so two different invoices with identical raw_text
        (e.g. "Invoice Amount: $50,000") are treated as distinct facts when their
        subject identifiers differ.

        sha256(quant_kind + ':' + (subject_id or '') + ':' + raw_text[:200])[:32]
        """
        payload = f"{quant_kind}:{subject_id or ''}:{raw_text[:200]}"
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

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

        Idempotent on (matter_id, quant_dedup_key) via INSERT OR IGNORE backed by
        ux_quant_fact_key unique index (rebuilt in migration v19 to include subject_id
        in the dedup key, preventing collision between different invoices with the
        same raw_text).
        """
        qf_id = _id()
        now = _now()
        raw_key = raw_text[:500]
        dedup_key = self._quant_key(quant_kind, subject_id, raw_text)
        with self.db.transaction():
            cur = self.db.execute(
                """INSERT OR IGNORE INTO quant_fact
                   (id, matter_id, quant_kind, amount_value, date_value, date_end_value,
                    rate_value, currency, unit, raw_text, subject_type, subject_id,
                    span_id, assertion_id, quant_dedup_key, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (qf_id, self.matter_id, quant_kind, amount_value, date_value, date_end_value,
                 rate_value, currency, unit, raw_key, subject_type, subject_id,
                 span_id, assertion_id, dedup_key, now),
            )
            if cur.rowcount == 0:
                # Already exists — return the existing ID
                row = self.db.execute(
                    "SELECT id FROM quant_fact WHERE matter_id=? AND quant_dedup_key=?",
                    (self.matter_id, dedup_key),
                ).fetchone()
                return row["id"]
        return qf_id

    def record_many(self, specs: list[dict]) -> list[str]:
        """Bulk-insert multiple quant facts in a single transaction.

        Each spec is a dict with the same keys as record() (quant_kind and
        raw_text required; all others optional). Uses executemany + INSERT OR
        IGNORE so duplicate entries (same quant_dedup_key) are silently skipped.

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
            raw_text = spec["raw_text"]
            subject_id = spec.get("subject_id")
            quant_kind = spec["quant_kind"]
            dedup_key = self._quant_key(quant_kind, subject_id, raw_text)
            rows.append((
                qf_id, self.matter_id,
                quant_kind,
                spec.get("amount_value"),
                spec.get("date_value"),
                spec.get("date_end_value"),
                spec.get("rate_value"),
                spec.get("currency"),
                spec.get("unit"),
                raw_text[:500],
                spec.get("subject_type"),
                subject_id,
                spec.get("span_id"),
                spec.get("assertion_id"),
                dedup_key,
                now,
            ))

        with self.db.transaction():
            self.db.executemany(
                """INSERT OR IGNORE INTO quant_fact
                   (id, matter_id, quant_kind, amount_value, date_value, date_end_value,
                    rate_value, currency, unit, raw_text, subject_type, subject_id,
                    span_id, assertion_id, quant_dedup_key, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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

    def reconcile_payment_chain(self, currency: str = "USD") -> dict:
        """Return structured payment reconciliation: invoiced, paid, disputed, exposure.

        Satisfies SO-6: produces a reconciliation showing what was invoiced, what
        was paid, what is disputed, and what the claimed exposure is, grounded in
        source spans (span_id on each quant_fact).

        'Disputed' = quant_facts linked (via assertion_id) to an assertion whose
        belief_state is 'disputed' — captures facts that truth-maintenance has flagged
        as contested. Falls back to conflict-detected amounts when no belief-revision
        has occurred (heuristic: conflicting amounts for the same subject_id are treated
        as disputed until resolved).

        Returns dict with keys:
          invoiced   — total invoiced amounts (subject_type='invoice')
          paid       — total payment amounts (subject_type='payment')
          disputed   — total amounts linked to disputed assertions
          exposure   — invoiced − paid (claimed outstanding balance)
          currency   — the currency used
          by_category — raw reconcile_by_subject() output for all categories
          source_spans — list of {quant_fact_id, subject_type, amount, span_id}
                         for top 20 facts with a span link (SO-6 grounding)
        """
        by_cat = self.reconcile_by_subject(currency)
        invoiced = by_cat.get("invoice", {}).get("total", 0.0)
        paid = by_cat.get("payment", {}).get("total", 0.0)
        exposure = round(invoiced - paid, 2)

        # Disputed: quant_facts joined to assertion with belief_state='disputed'
        disputed_row = self.db.execute(
            """SELECT COALESCE(SUM(qf.amount_value), 0.0) AS total
               FROM quant_fact qf
               JOIN assertion a ON a.id = qf.assertion_id
               WHERE qf.matter_id=? AND qf.quant_kind='amount'
                 AND (qf.currency=? OR (qf.currency IS NULL AND ?='USD'))
                 AND qf.amount_value IS NOT NULL
                 AND a.belief_state = 'disputed'""",
            (self.matter_id, currency, currency),
        ).fetchone()
        disputed = round(float(disputed_row["total"]) if disputed_row else 0.0, 2)
        # Note: disputed reflects only facts whose linked assertion.belief_state = 'disputed'.
        # Numeric conflicts (same subject_id, different amounts) are surfaced separately via
        # get_conflicts() — they indicate potential disputes that have not yet been resolved
        # through belief revision. Do not conflate conflicts with confirmed disputed amounts.

        # Source spans: top 20 quant_facts with a span_id for SO-6 grounding
        span_rows = self.db.execute(
            """SELECT id, subject_type, subject_id, amount_value, span_id
               FROM quant_fact
               WHERE matter_id=? AND quant_kind='amount'
                 AND span_id IS NOT NULL
                 AND amount_value IS NOT NULL
               ORDER BY amount_value DESC
               LIMIT 20""",
            (self.matter_id,),
        ).fetchall()
        source_spans = [
            {
                "quant_fact_id": r["id"],
                "subject_type": r["subject_type"],
                "subject_id": r["subject_id"],
                "amount": round(float(r["amount_value"]), 2),
                "span_id": r["span_id"],
            }
            for r in span_rows
        ]

        return {
            "invoiced": round(invoiced, 2),
            "paid": round(paid, 2),
            "disputed": disputed,
            "exposure": exposure,
            "currency": currency,
            "by_category": by_cat,
            "source_spans": source_spans,
        }

    # Maximum number of distinct named invoice IDs processed in one reconciliation pass.
    # SQLite has a default SQLITE_MAX_VARIABLE_NUMBER of 999; we stay well below it
    # and bound memory for span_map construction.  In practice legal matters rarely
    # exceed a few hundred invoices.
    _MAX_INVOICE_IDS = 500

    def reconcile_invoice_chain(self, currency: str = "USD") -> list:
        """Return per-invoice reconciliation rows.

        Each entry: {invoice_id, invoiced, paid, outstanding, currency, source_spans}.

        Payments are matched to invoices by subject_id equality — a payment
        quant_fact whose subject_id equals an invoice quant_fact's subject_id
        is treated as a payment against that invoice.  NULL-subject_id invoice
        and payment facts are bucketed together under invoice_id="".

        Named invoices are capped at _MAX_INVOICE_IDS (sorted by descending invoiced
        amount).  The NULL bucket is fetched via a separate query so it is never
        excluded by the LIMIT applied to named invoices.

        Returns an empty list when no invoice quant_facts exist.
        """
        _ccy_filter = "AND (currency=? OR (currency IS NULL AND ?='USD'))"

        # Query 1a: top named (non-NULL subject_id) invoice amounts, capped.
        named_rows = self.db.execute(
            f"""SELECT subject_id, SUM(amount_value) AS invoiced
                FROM quant_fact
                WHERE matter_id=? AND subject_type='invoice' AND quant_kind='amount'
                  AND subject_id IS NOT NULL AND amount_value IS NOT NULL
                  {_ccy_filter}
                GROUP BY subject_id
                ORDER BY invoiced DESC
                LIMIT ?""",
            (self.matter_id, currency, currency, self._MAX_INVOICE_IDS),
        ).fetchall()

        # Query 1b: NULL-bucket invoice total (separate query — never excluded by cap).
        # SUM returns NULL (not 0.0) when no rows match, so null_row["invoiced"] IS NULL
        # means no NULL-subject invoice rows exist; 0.0 means rows exist but sum to zero.
        null_row = self.db.execute(
            f"""SELECT SUM(amount_value) AS invoiced
                FROM quant_fact
                WHERE matter_id=? AND subject_type='invoice' AND quant_kind='amount'
                  AND subject_id IS NULL AND amount_value IS NOT NULL
                  {_ccy_filter}""",
            (self.matter_id, currency, currency),
        ).fetchone()
        has_null_bucket = null_row is not None and null_row["invoiced"] is not None
        null_invoiced = float(null_row["invoiced"]) if has_null_bucket else 0.0

        if not named_rows and not has_null_bucket:
            return []

        invoice_ids = [r["subject_id"] for r in named_rows]

        # Query 2a: payment amounts matched to named invoices via IN list.
        pay_map: "dict[str, float]" = {}
        if invoice_ids:
            placeholders = ",".join("?" * len(invoice_ids))
            pay_rows = self.db.execute(
                f"""SELECT subject_id, SUM(amount_value) AS paid
                    FROM quant_fact
                    WHERE matter_id=? AND subject_type='payment' AND quant_kind='amount'
                      AND amount_value IS NOT NULL {_ccy_filter}
                      AND subject_id IN ({placeholders})
                    GROUP BY subject_id""",
                (self.matter_id, currency, currency, *invoice_ids),
            ).fetchall()
            pay_map = {r["subject_id"]: float(r["paid"]) for r in pay_rows}

        # Query 2b: payments with NULL subject_id matched to NULL-bucket invoices.
        null_paid = 0.0
        if has_null_bucket:
            null_pay_row = self.db.execute(
                f"""SELECT COALESCE(SUM(amount_value), 0.0) AS paid
                    FROM quant_fact
                    WHERE matter_id=? AND subject_type='payment' AND quant_kind='amount'
                      AND subject_id IS NULL AND amount_value IS NOT NULL {_ccy_filter}""",
                (self.matter_id, currency, currency),
            ).fetchone()
            null_paid = float(null_pay_row["paid"]) if null_pay_row else 0.0

        # Query 3: up to 3 source spans per named invoice using a window function.
        # ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY amount_value DESC) gives
        # each invoice its own independent rank so no single invoice can starve others.
        # Requires SQLite >= 3.25 (available as of Python 3.13 / SQLite 3.50).
        span_map: "dict[str, list]" = {inv_id: [] for inv_id in invoice_ids}
        if invoice_ids:
            placeholders = ",".join("?" * len(invoice_ids))
            span_rows = self.db.execute(
                f"""SELECT id, subject_id, amount_value, span_id
                    FROM (
                        SELECT id, subject_id, amount_value, span_id,
                               ROW_NUMBER() OVER (
                                   PARTITION BY subject_id
                                   ORDER BY amount_value DESC
                               ) AS rn
                        FROM quant_fact
                        WHERE matter_id=? AND subject_type='invoice' AND quant_kind='amount'
                          AND span_id IS NOT NULL AND amount_value IS NOT NULL
                          AND subject_id IN ({placeholders})
                    )
                    WHERE rn <= 3""",
                (self.matter_id, *invoice_ids),
            ).fetchall()
            for sr in span_rows:
                sid = sr["subject_id"]
                if sid in span_map:
                    span_map[sid].append(
                        {
                            "quant_fact_id": sr["id"],
                            "amount": round(float(sr["amount_value"]), 2),
                            "span_id": sr["span_id"],
                        }
                    )

        result = []
        for r in named_rows:
            sid = r["subject_id"]
            invoiced = round(float(r["invoiced"]), 2)
            paid = round(pay_map.get(sid, 0.0), 2)
            result.append(
                {
                    "invoice_id": sid,
                    "invoiced": invoiced,
                    "paid": paid,
                    "outstanding": round(invoiced - paid, 2),
                    "currency": currency,
                    "source_spans": span_map.get(sid, []),
                }
            )

        if has_null_bucket:
            invoiced = round(null_invoiced, 2)
            paid = round(null_paid, 2)
            result.append(
                {
                    "invoice_id": "",
                    "invoiced": invoiced,
                    "paid": paid,
                    "outstanding": round(invoiced - paid, 2),
                    "currency": currency,
                    "source_spans": [],
                }
            )

        result.sort(key=lambda x: x["invoiced"], reverse=True)
        return result

    def compute_thresholds(self, gap_store: "GapStore", currency: str = "USD") -> list[dict]:
        """Detect quantitative threshold violations and record them as gaps (SO-6).

        Thresholds checked:
        1. Positive exposure (invoiced > paid) → MISSING_DOCUMENT gap so synthesis
           must address the outstanding balance with hard specificity.
        2. High disputed fraction (>10% of invoiced) → UNRESOLVED_CONTRADICTION gap.
        3. Numeric conflicts → UNRESOLVED_CONTRADICTION gap per conflict group.

        Returns list of violation dicts:
          {threshold: str, level: 'HIGH'|'MED'|'LOW', description: str, amount: float|None}

        Idempotent: underlying gap_store.record() deduplicates by description.
        """
        violations = []
        try:
            chain = self.reconcile_payment_chain(currency)
        except Exception:
            return violations

        exposure = chain.get("exposure", 0.0)
        invoiced = chain.get("invoiced", 0.0)
        disputed = chain.get("disputed", 0.0)

        # Threshold 1: positive financial exposure
        if exposure > 0:
            level = "HIGH" if exposure >= 10_000 else "MED"
            desc = (
                f"Claimed financial exposure: {currency} {exposure:,.2f} "
                f"(invoiced {currency} {invoiced:,.2f} − paid {currency} {chain.get('paid', 0):,.2f})"
            )
            try:
                gap_store.record(
                    gap_type=GapType.MISSING_DOCUMENT,
                    description=desc,
                    materiality=0.9 if exposure >= 10_000 else 0.6,
                    affected_type="quant",
                    affected_id="exposure",
                )
            except Exception:
                pass
            violations.append({"threshold": "positive_exposure", "level": level,
                                "description": desc, "amount": exposure})

        # Threshold 2: high disputed fraction
        if invoiced > 0 and disputed > 0:
            frac = disputed / invoiced
            if frac >= 0.10:
                level = "HIGH" if frac >= 0.30 else "MED"
                desc = (
                    f"Disputed amounts ({currency} {disputed:,.2f}) represent "
                    f"{frac:.0%} of total invoiced — significant contested balance"
                )
                try:
                    gap_store.record(
                        gap_type=GapType.UNRESOLVED_CONTRADICTION,
                        description=desc,
                        materiality=0.8 if frac >= 0.30 else 0.5,
                        affected_type="quant",
                        affected_id="disputed_fraction",
                    )
                except Exception:
                    pass
                violations.append({"threshold": "disputed_fraction", "level": level,
                                    "description": desc, "amount": disputed})

        # Threshold 3: numeric conflicts
        try:
            conflicts = self.get_conflicts()
        except Exception:
            conflicts = []
        for conflict in conflicts[:5]:  # cap at 5 to bound gap store growth
            sid = conflict.get("subject_id") or conflict.get("subject_type", "unknown")
            vals = conflict.get("values", [])
            desc = (
                f"Numeric conflict for '{sid}': "
                f"multiple sources report different values — {vals}"
            )
            try:
                gap_store.record(
                    gap_type=GapType.UNRESOLVED_CONTRADICTION,
                    description=desc[:200],
                    materiality=0.75,
                    affected_type="quant",
                    affected_id=str(sid)[:64],
                )
            except Exception:
                pass
            violations.append({"threshold": "numeric_conflict", "level": "HIGH",
                                "description": desc, "amount": None})

        return violations


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

    def get_doc_row(self, doc_id: str) -> Optional[dict]:
        """Return the document_inventory row for a given doc_id, or None."""
        row = self.db.execute(
            "SELECT id, relative_path, ingest_status, sha256 FROM document_inventory WHERE id=?",
            (doc_id,),
        ).fetchone()
        return dict(row) if row else None

    def count(self) -> int:
        row = self.db.execute(
            "SELECT COUNT(*) FROM document_inventory WHERE matter_id=?",
            (self.matter_id,),
        ).fetchone()
        return row[0]

    # ------------------------------------------------------------------
    # Document relations and version-chain detection (spec §14, §26)
    # ------------------------------------------------------------------

    #: Version-indicator patterns, in ascending priority order.
    #: Each entry is (regex_pattern, sort_key_extractor).
    _VERSION_SUFFIXES = [
        # v1, v2, v3, ...  or  _v1, _v2
        (r"[_\-\s]*v(\d+)$", lambda m: int(m.group(1))),
        # _1, _2, _3 (trailing digits only)
        (r"[_\-\s]+(\d+)$", lambda m: int(m.group(1))),
        # draft < redline < revised < final < executed/signed
        (r"[_\-\s]*(draft|redline|revised|final|executed|signed)$",
         lambda m: {"draft": 0, "redline": 1, "revised": 2, "final": 3,
                    "executed": 4, "signed": 4}.get(m.group(1).lower(), 0)),
        # amendment_1, amendment_2, amend_3 → base="amendment", order by number
        # (handled by the v\d+ pattern above when applied after stripping)
    ]

    @staticmethod
    def _normalize_stem(filename: str) -> tuple[str, int]:
        """
        Strip a version suffix from a filename stem and return (base_stem, sort_key).

        Examples:
          "contract_v2.pdf" → ("contract", 2)
          "agreement_final.pdf" → ("agreement", 3)
          "exhibit_a.pdf" → ("exhibit_a", -1)   # no version → -1

        filename should be the basename without directory prefix; the extension
        will be stripped internally.
        """
        import re
        import os
        stem = os.path.splitext(os.path.basename(filename))[0].lower().strip()
        for pattern, extractor in DocumentInventoryStore._VERSION_SUFFIXES:
            m = re.search(pattern, stem, re.IGNORECASE)
            if m:
                base = stem[: m.start()].rstrip("_- ")
                return base, extractor(m)
        # No version pattern found — treat as base document, sort key -1
        return stem, -1

    def link_documents(
        self,
        source_doc_id: str,
        target_doc_id: str,
        relation_type: str,
        confidence: float = 1.0,
    ) -> str:
        """
        Record a directed document relation.

        Convention:
          source_doc_id --relation_type--> target_doc_id

        Common relation_type values:
          'version_of'  — source is a later version of target
          'amends'      — source amends target
          'supersedes'  — source replaces target
          'attachment_to' — source is an attachment to target

        Idempotent: a duplicate (source, target, relation_type) triple is ignored.
        Returns the relation_id (existing or new).
        """
        rel_id = _id()
        now = _now()
        with self.db.transaction():
            self.db.execute(
                """INSERT OR IGNORE INTO document_relation
                   (id, source_doc_id, target_doc_id, relation_type, confidence, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (rel_id, source_doc_id, target_doc_id, relation_type, confidence, now),
            )
            existing = self.db.execute(
                """SELECT id FROM document_relation
                   WHERE source_doc_id=? AND target_doc_id=? AND relation_type=?""",
                (source_doc_id, target_doc_id, relation_type),
            ).fetchone()
        return existing["id"] if existing else rel_id

    def get_version_family(self, doc_id: str) -> list[dict]:
        """
        Return all documents in the version chain containing doc_id.

        Traverses both directions of 'version_of' relations (predecessors and
        successors) up to 20 hops to prevent cycles.

        Returns list of dicts: {id, relative_path, relation_type, direction}
        where direction is 'predecessor' | 'successor' | 'self'.
        """
        visited = {doc_id}
        result = [{"id": doc_id, "relative_path": None, "relation_type": None, "direction": "self"}]
        queue = [(doc_id, 0)]
        while queue:
            current_id, depth = queue.pop(0)
            if depth >= 20:
                continue
            # Successors: current --version_of--> current means current is later;
            # here we look for docs that declare themselves version_of current (successors).
            succ_rows = self.db.execute(
                """SELECT dr.source_doc_id AS other_id, di.relative_path
                   FROM document_relation dr
                   JOIN document_inventory di ON di.id = dr.source_doc_id
                   WHERE dr.target_doc_id=? AND dr.relation_type='version_of'""",
                (current_id,),
            ).fetchall()
            for r in succ_rows:
                if r["other_id"] not in visited:
                    visited.add(r["other_id"])
                    result.append({
                        "id": r["other_id"],
                        "relative_path": r["relative_path"],
                        "relation_type": "version_of",
                        "direction": "successor",
                    })
                    queue.append((r["other_id"], depth + 1))
            # Predecessors: current --version_of--> predecessor
            pred_rows = self.db.execute(
                """SELECT dr.target_doc_id AS other_id, di.relative_path
                   FROM document_relation dr
                   JOIN document_inventory di ON di.id = dr.target_doc_id
                   WHERE dr.source_doc_id=? AND dr.relation_type='version_of'""",
                (current_id,),
            ).fetchall()
            for r in pred_rows:
                if r["other_id"] not in visited:
                    visited.add(r["other_id"])
                    result.append({
                        "id": r["other_id"],
                        "relative_path": r["relative_path"],
                        "relation_type": "version_of",
                        "direction": "predecessor",
                    })
                    queue.append((r["other_id"], depth + 1))

        # Fill in relative_path for the seed doc
        seed_row = self.db.execute(
            "SELECT relative_path FROM document_inventory WHERE id=?", (doc_id,)
        ).fetchone()
        if seed_row:
            result[0]["relative_path"] = seed_row["relative_path"]

        return result

    def get_operative_version(self, doc_id: str) -> str:
        """
        Return the ID of the operative (latest) version in the chain containing doc_id.

        The operative version is the HEAD: a document that is not itself the
        source of any 'version_of' link (nothing is 'a later version of' it).

        Returns doc_id unchanged if it is already the HEAD, or if no version
        relations exist for it.

        If multiple HEADs exist (branched chain), returns the first found and
        logs ambiguity via a silent tie-break — callers should check for
        ambiguity with get_version_family() if precision is needed.
        """
        # Build the full family
        family = self.get_version_family(doc_id)
        family_ids = {m["id"] for m in family}

        # Find nodes that are NOT the source (i.e., nothing points TO them as a later version)
        # In a well-formed chain: HEADs have no successors
        heads = []
        for mid in family_ids:
            successor_row = self.db.execute(
                """SELECT 1 FROM document_relation
                   WHERE target_doc_id=? AND relation_type='version_of'
                   LIMIT 1""",
                (mid,),
            ).fetchone()
            if successor_row is None:
                heads.append(mid)

        if not heads:
            return doc_id  # degenerate: no family found
        if len(heads) == 1:
            return heads[0]
        # Multiple HEADs (branched chain): return the one that is not the seed
        non_seed = [h for h in heads if h != doc_id]
        return non_seed[0] if non_seed else heads[0]

    def detect_version_chains(self, gap_store: "GapStore") -> list[dict]:
        """
        Heuristically detect document version chains from filename patterns.

        Algorithm:
        1. Fetch all document_inventory rows for this matter.
        2. Normalize each filename stem using _normalize_stem() to extract
           (base_stem, sort_key) pairs.
        3. Group documents by base_stem.
        4. For groups with 2+ documents, sort by sort_key and create
           'version_of' links: each document points to its immediate predecessor.
        5. Record a MISSING_DOCUMENT gap when a chain exists but the base
           document (sort_key == -1 or lowest) is absent from the group.

        Returns list of dicts describing the links created:
          {source_doc_id, target_doc_id, source_path, target_path, sort_key}
        """
        from .enums import GapType
        rows = self.db.execute(
            "SELECT id, relative_path FROM document_inventory WHERE matter_id=?",
            (self.matter_id,),
        ).fetchall()

        # Group by normalized base stem
        from collections import defaultdict
        groups: dict = defaultdict(list)
        for r in rows:
            base, sort_key = self._normalize_stem(r["relative_path"])
            groups[base].append((sort_key, r["id"], r["relative_path"]))

        created = []
        for base, members in groups.items():
            if len(members) < 2:
                continue
            # Sort by sort_key ascending (-1 = no version → treat as oldest)
            members_sorted = sorted(members, key=lambda t: t[0])
            # Link each to its predecessor
            for i in range(1, len(members_sorted)):
                pred_key, pred_id, pred_path = members_sorted[i - 1]
                succ_key, succ_id, succ_path = members_sorted[i]
                if pred_id == succ_id:
                    continue
                self.link_documents(succ_id, pred_id, "version_of", confidence=0.8)
                created.append({
                    "source_doc_id": succ_id,
                    "target_doc_id": pred_id,
                    "source_path": succ_path,
                    "target_path": pred_path,
                    "sort_key": succ_key,
                })

            # If there is no sort_key==-1 member (no unversioned base), log a gap
            has_base = any(sk == -1 for sk, _, _ in members_sorted)
            if not has_base:
                gap_store.record(
                    gap_type=GapType.MISSING_DOCUMENT,
                    description=(
                        f"Version chain detected for '{base}' but base (unversioned) "
                        f"document is absent from repository"
                    ),
                    materiality=0.4,
                )

        return created


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
        # Normalize both sides to forward slashes before comparison
        _doc_normalized = document_id.replace("\\\\", "/").replace("\\", "/")
        basename = pathlib.Path(_doc_normalized).name
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
        basename = pathlib.Path(document_id).name
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


class DecisionContextStore:
    """Persists the decision-context overlay for a matter.

    One row per matter.  Stores context that influences synthesis framing
    (prioritization, output ranking, recommendation framing) WITHOUT
    rewriting the canonical record model or assertion graph.

    Valid decision_maker_type values:
        judge, partner, client, mediator, arbitrator, regulator, unknown

    Valid objective values:
        motion_practice, settlement, diligence, audit, advisory, trial_prep,
        regulatory_response, transactional, unknown
    """

    VALID_MAKER_TYPES = frozenset({
        "judge", "partner", "client", "mediator", "arbitrator",
        "regulator", "unknown",
    })
    VALID_OBJECTIVES = frozenset({
        "motion_practice", "settlement", "diligence", "audit", "advisory",
        "trial_prep", "regulatory_response", "transactional", "unknown",
    })

    def __init__(self, db: "SQLiteMatterDB", matter_id: str) -> None:
        self.db = db
        self.matter_id = matter_id

    def set(
        self,
        decision_maker_type: Optional[str] = None,
        decision_maker_name: Optional[str] = None,
        objective: Optional[str] = None,
        strategic_notes: Optional[str] = None,
        scope_narrow: bool = False,
    ) -> str:
        """Upsert the decision context for this matter.  Returns the row id.

        Unknown/invalid decision_maker_type or objective values are coerced
        to 'unknown' to keep the constraint surface small and avoid silent
        data corruption.
        """
        if decision_maker_type and decision_maker_type not in self.VALID_MAKER_TYPES:
            decision_maker_type = "unknown"
        if objective and objective not in self.VALID_OBJECTIVES:
            objective = "unknown"

        now = _now()
        existing = self.db.execute(
            "SELECT id FROM decision_context WHERE matter_id=?",
            (self.matter_id,),
        ).fetchone()

        if existing:
            ctx_id = existing["id"]
            self.db.execute(
                """UPDATE decision_context
                   SET decision_maker_type=?, decision_maker_name=?,
                       objective=?, strategic_notes=?,
                       scope_narrow=?, updated_at=?
                   WHERE id=?""",
                (decision_maker_type, decision_maker_name, objective,
                 strategic_notes, int(scope_narrow), now, ctx_id),
            )
            return ctx_id
        else:
            ctx_id = _id()
            self.db.execute(
                """INSERT INTO decision_context
                   (id, matter_id, decision_maker_type, decision_maker_name,
                    objective, strategic_notes, scope_narrow, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (ctx_id, self.matter_id, decision_maker_type, decision_maker_name,
                 objective, strategic_notes, int(scope_narrow), now, now),
            )
            return ctx_id

    def get(self) -> Optional[dict]:
        """Return the decision context dict, or None if not yet set."""
        row = self.db.execute(
            "SELECT * FROM decision_context WHERE matter_id=?",
            (self.matter_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "matter_id": row["matter_id"],
            "decision_maker_type": row["decision_maker_type"],
            "decision_maker_name": row["decision_maker_name"],
            "objective": row["objective"],
            "strategic_notes": row["strategic_notes"],
            "scope_narrow": bool(row["scope_narrow"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def clear(self) -> None:
        """Remove the decision context for this matter."""
        self.db.execute(
            "DELETE FROM decision_context WHERE matter_id=?",
            (self.matter_id,),
        )


class AuthorityStore:
    """Stores legal authorities as structured objects (SO-4 legal research layer).

    Authorities are cases, statutes, regulations, rules, and secondary sources
    cited in the matter analysis.  Each authority is uniquely identified by its
    citation within a matter.

    Valid authority_type values:
        case, statute, regulation, rule, secondary, unknown

    Valid weight values:
        binding, persuasive, neutral, unknown

    Valid relevance values (for issue links):
        supporting, attacking, neutral
    """

    VALID_TYPES = frozenset({
        "case", "statute", "regulation", "rule", "secondary", "unknown",
    })
    VALID_WEIGHTS = frozenset({
        "binding", "persuasive", "neutral", "unknown",
    })
    VALID_RELEVANCE = frozenset({
        "supporting", "attacking", "neutral",
    })

    def __init__(self, db: "SQLiteMatterDB", matter_id: str) -> None:
        self.db = db
        self.matter_id = matter_id

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(
        self,
        citation: str,
        authority_type: str = "case",
        name: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        decided_at: Optional[str] = None,
        holdings: Optional[list] = None,
        key_rules: Optional[list] = None,
        weight: str = "persuasive",
        applicability: Optional[str] = None,
        source_doc_id: Optional[str] = None,
        source_span_id: Optional[str] = None,
    ) -> tuple[str, bool]:
        """Create or update an authority by citation.

        Returns (authority_id, is_new).
        holdings and key_rules are stored as JSON arrays.
        """
        import json as _json

        citation = citation.strip()
        if not citation:
            raise ValueError("citation must not be blank")
        if authority_type not in self.VALID_TYPES:
            authority_type = "unknown"
        if weight not in self.VALID_WEIGHTS:
            weight = "unknown"

        holdings_json = _json.dumps(holdings) if holdings is not None else None
        key_rules_json = _json.dumps(key_rules) if key_rules is not None else None
        now = _now()

        with self.db.transaction():
            row = self.db.execute(
                "SELECT id FROM authority WHERE matter_id=? AND citation=?",
                (self.matter_id, citation),
            ).fetchone()

            if row is not None:
                auth_id = row["id"]
                self.db.execute(
                    """UPDATE authority
                       SET authority_type=?, name=?, jurisdiction=?, decided_at=?,
                           holdings=?, key_rules=?, weight=?, applicability=?,
                           source_doc_id=?, source_span_id=?, updated_at=?
                       WHERE id=?""",
                    (authority_type, name, jurisdiction, decided_at,
                     holdings_json, key_rules_json, weight, applicability,
                     source_doc_id, source_span_id, now, auth_id),
                )
                return auth_id, False

            auth_id = _id()
            self.db.execute(
                """INSERT INTO authority
                   (id, matter_id, authority_type, citation, name, jurisdiction,
                    decided_at, holdings, key_rules, weight, applicability,
                    source_doc_id, source_span_id, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (auth_id, self.matter_id, authority_type, citation, name,
                 jurisdiction, decided_at, holdings_json, key_rules_json,
                 weight, applicability, source_doc_id, source_span_id, now, now),
            )
        return auth_id, True

    def link_to_issue(
        self,
        authority_id: str,
        issue_id: str,
        relevance: str = "supporting",
    ) -> None:
        """Link an authority to an issue.  Idempotent."""
        if relevance not in self.VALID_RELEVANCE:
            relevance = "neutral"
        now = _now()
        self.db.execute(
            """INSERT OR REPLACE INTO authority_issue_link
               (authority_id, issue_id, relevance, created_at)
               VALUES (?,?,?,?)""",
            (authority_id, issue_id, relevance, now),
        )

    def unlink_from_issue(self, authority_id: str, issue_id: str) -> None:
        """Remove an authority-issue link."""
        self.db.execute(
            "DELETE FROM authority_issue_link WHERE authority_id=? AND issue_id=?",
            (authority_id, issue_id),
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, authority_id: str) -> Optional[dict]:
        """Return a single authority by id, or None."""
        row = self.db.execute(
            "SELECT * FROM authority WHERE id=?",
            (authority_id,),
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_by_citation(self, citation: str) -> Optional[dict]:
        """Return an authority by citation string, or None."""
        row = self.db.execute(
            "SELECT * FROM authority WHERE matter_id=? AND citation=?",
            (self.matter_id, citation.strip()),
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def list_all(
        self,
        authority_type: Optional[str] = None,
        weight: Optional[str] = None,
        limit: int = 200,
    ) -> list[dict]:
        """Return authorities, optionally filtered by type or weight."""
        params: list = [self.matter_id]
        clauses: list[str] = ["matter_id=?"]

        if authority_type:
            clauses.append("authority_type=?")
            params.append(authority_type)
        if weight:
            clauses.append("weight=?")
            params.append(weight)

        params.append(limit)
        rows = self.db.execute(
            "SELECT * FROM authority WHERE "
            + " AND ".join(clauses)
            + " ORDER BY weight DESC, citation ASC LIMIT ?",
            params,
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def list_for_issue(self, issue_id: str) -> list[dict]:
        """Return all authorities linked to an issue, with their relevance."""
        rows = self.db.execute(
            """SELECT a.*, l.relevance AS link_relevance
               FROM authority a
               JOIN authority_issue_link l ON l.authority_id = a.id
               WHERE l.issue_id=?
               ORDER BY a.weight DESC, a.citation ASC""",
            (issue_id,),
        ).fetchall()
        results = []
        for r in rows:
            d = self._row_to_dict(r)
            d["link_relevance"] = r["link_relevance"]
            results.append(d)
        return results

    def count(self) -> int:
        """Total authority records for this matter."""
        row = self.db.execute(
            "SELECT COUNT(*) AS n FROM authority WHERE matter_id=?",
            (self.matter_id,),
        ).fetchone()
        return row["n"] if row else 0

    def search(self, query: str, limit: int = 20) -> list[dict]:
        """Substring search across citation and name fields."""
        pattern = f"%{query}%"
        rows = self.db.execute(
            """SELECT * FROM authority
               WHERE matter_id=?
                 AND (citation LIKE ? OR name LIKE ?)
               ORDER BY weight DESC, citation ASC
               LIMIT ?""",
            (self.matter_id, pattern, pattern, limit),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row) -> dict:
        import json as _json
        d = dict(row)
        for field in ("holdings", "key_rules"):
            raw = d.get(field)
            if raw is not None:
                try:
                    d[field] = _json.loads(raw)
                except (TypeError, ValueError):
                    d[field] = []
            else:
                d[field] = []
        return d


class ProofStateStore:
    """Tracks proof coverage per issue (SO-4/SO-5 proof-aware reasoning).

    For each issue the store maintains a computed snapshot of:
    - sufficiency: float 0..1 — how well-proved the issue is
    - supporting/attacking assertion counts
    - predicate satisfaction ratio
    - categorical proof_status
    - trust_weighted_support: source-role-weighted evidence score (SO-5)
    - advocacy_only: True when all supporting assertions come from advocacy sources

    Call compute_and_store(issue_id) after adding assertions or resolving
    predicates to refresh the snapshot.  Reads are always from the stored
    snapshot — never recomputed on read — so hot-path reads are cheap.

    proof_status values:
        insufficient — sufficiency < 0.25, or zero supporting assertions
        partial      — sufficiency 0.25..0.75
        sufficient   — sufficiency > 0.75
        contested    — attacking_count >= supporting_count > 0

    Source trust weights (SO-5 enforcement):
        operative / authoritative: 1.0  — signed orders, contracts, statutes
        procedural: 0.7                 — court filings, process documents
        informal: 0.5                   — emails, notes
        unknown: 0.5                    — no role assigned
        draft: 0.4                      — not final, can be revised
        advocacy: 0.3                   — pleadings, briefs, demand letters
        post_hoc: 0.3                   — memos written to explain past events
    """

    SUFFICIENT_THRESHOLD = 0.75
    PARTIAL_THRESHOLD = 0.25

    # Source role trust weights (SO-5). Canonical in enums.SOURCE_TRUST_WEIGHTS.
    # This class attribute preserves the existing API (ProofStateStore.SOURCE_TRUST[...]).
    SOURCE_TRUST: dict[str, float] = SOURCE_TRUST_WEIGHTS
    # Trust threshold below which an assertion is considered "low-trust" for
    # the advocacy_only flag.
    ADVOCACY_TRUST_THRESHOLD = 0.35

    def __init__(self, db: "SQLiteMatterDB", matter_id: str) -> None:
        self.db = db
        self.matter_id = matter_id

    # ------------------------------------------------------------------
    # Compute + store
    # ------------------------------------------------------------------

    def compute_and_store(
        self,
        issue_id: str,
        _preloaded_overrides: "list[tuple[str, str]] | None" = None,
    ) -> dict:
        """Recompute proof state for an issue and persist it.

        Derives counts from live assertion_issue_link and issue_predicate rows,
        so it always reflects the current model state.

        _preloaded_overrides — optional pre-fetched list of (document_pattern, trust_level)
            for non-normal overrides.  When provided, the per-call DB query is skipped.
            Callers like compute_all() batch this once to avoid N fetches for N issues.

        Returns the newly computed proof state dict.
        """
        now = _now()

        # Count supporting and attacking assertions linked to this issue,
        # and gather source roles for trust weighting (SO-5).
        #
        # Use a correlated subquery to pick ONE source_role per assertion —
        # the highest-trust role across all its occurrences.  A plain
        # LEFT JOIN returns one row per occurrence, inflating counts and
        # trust sums when an assertion appears in multiple documents.
        # Trust overrides: use caller-supplied list when available (batch path from
        # compute_all) to avoid one DB query per issue.  Otherwise fetch here.
        if _preloaded_overrides is not None:
            _overrides = _preloaded_overrides
        else:
            _override_rows = self.db.execute(
                """SELECT document_pattern, trust_level FROM document_trust_override
                   WHERE matter_id=? AND trust_level != 'normal'
                   ORDER BY LENGTH(document_pattern) DESC""",
                (self.matter_id,),
            ).fetchall()
            _overrides = [(r["document_pattern"], r["trust_level"]) for r in _override_rows]

        def _effective_trust(source_role: str, primary_doc_id: "str | None") -> float:
            """Apply document trust override if set; otherwise use stored source_role weight."""
            if _overrides and primary_doc_id:
                doc_norm = primary_doc_id.replace("\\\\", "/").replace("\\", "/")
                basename = pathlib.Path(doc_norm).name
                for pat, lvl in _overrides:
                    pat_norm = pat.replace("\\\\", "/").replace("\\", "/")
                    if pat_norm == doc_norm or pat_norm == basename:
                        return self.SOURCE_TRUST.get(
                            "advocacy" if lvl == "low" else "operative", 0.5
                        )
            return self.SOURCE_TRUST.get(source_role, 0.5)

        # Single CTE pass: precompute best source_role and primary_document_id for
        # every assertion linked to this issue, then split into sup/atk buckets.
        # Replaces 2 queries × N correlated subqueries with one set-based scan (SO-5).
        _linked_rows = self.db.execute(
            """WITH occ_ranked AS (
                   SELECT ao.assertion_id,
                          ao.source_role,
                          ao.document_id,
                          ROW_NUMBER() OVER (
                              PARTITION BY ao.assertion_id
                              ORDER BY CASE ao.source_role
                                  WHEN 'authoritative' THEN 6
                                  WHEN 'operative'     THEN 5
                                  WHEN 'procedural'    THEN 4
                                  WHEN 'post_hoc'      THEN 3
                                  WHEN 'informal'      THEN 2
                                  WHEN 'draft'         THEN 1
                                  WHEN 'unknown'       THEN 1
                                  WHEN 'advocacy'      THEN 0
                                  ELSE 1 END DESC
                          ) AS role_rn,
                          ROW_NUMBER() OVER (
                              PARTITION BY ao.assertion_id
                              ORDER BY ao.created_at ASC, ao.id ASC
                          ) AS doc_rn
                   FROM assertion_occurrence ao
                   WHERE ao.assertion_id IN (
                       SELECT assertion_id FROM assertion_issue_link
                       WHERE issue_id = ?
                       AND relation_type IN ('supports','establishes','attacks','negates')
                   )
               )
               SELECT ail.relation_type,
                      COALESCE(MAX(CASE WHEN o.role_rn = 1 THEN o.source_role END), 'unknown')
                          AS source_role,
                      MAX(CASE WHEN o.doc_rn = 1 THEN o.document_id END)
                          AS primary_doc_id
               FROM assertion_issue_link ail
               JOIN assertion a ON a.id = ail.assertion_id
               LEFT JOIN occ_ranked o ON o.assertion_id = ail.assertion_id
               WHERE ail.issue_id = ?
                 AND ail.relation_type IN ('supports','establishes','attacks','negates')
                 AND a.belief_state NOT IN ('superseded','withdrawn')
               GROUP BY ail.assertion_id, ail.relation_type""",
            (issue_id, issue_id),
        ).fetchall()
        sup_rows = [r for r in _linked_rows if r["relation_type"] in ("supports", "establishes")]
        atk_rows = [r for r in _linked_rows if r["relation_type"] in ("attacks", "negates")]
        supporting = len(sup_rows)
        attacking = len(atk_rows)

        # Compute trust-weighted scores for SO-5 source role enforcement.
        # _effective_trust() applies document trust overrides when set.
        def _tw(rows: list) -> float:
            return sum(
                _effective_trust(r["source_role"], r["primary_doc_id"])
                for r in rows
            )

        trust_weighted_support = round(_tw(sup_rows), 4)
        trust_weighted_attack = round(_tw(atk_rows), 4)

        # advocacy_only: True when every supporting assertion has effective trust <= threshold
        # (i.e., evidence is exclusively from advocacy/post_hoc sources after applying overrides).
        if sup_rows:
            advocacy_only = all(
                _effective_trust(r["source_role"], r["primary_doc_id"])
                <= self.ADVOCACY_TRUST_THRESHOLD
                for r in sup_rows
            )
        else:
            advocacy_only = False

        # Count total and satisfied predicates.
        total_pred_row = self.db.execute(
            "SELECT COUNT(*) AS n FROM issue_predicate WHERE issue_id=?",
            (issue_id,),
        ).fetchone()
        sat_pred_row = self.db.execute(
            "SELECT COUNT(*) AS n FROM issue_predicate WHERE issue_id=? AND status='resolved'",
            (issue_id,),
        ).fetchone()
        total_predicates = total_pred_row["n"] if total_pred_row else 0
        satisfied_predicates = sat_pred_row["n"] if sat_pred_row else 0

        # Compute sufficiency score.
        sufficiency = self._score(
            supporting, attacking, total_predicates, satisfied_predicates
        )

        # Categorise.
        if attacking >= supporting > 0:
            proof_status = "contested"
        elif sufficiency >= self.SUFFICIENT_THRESHOLD:
            proof_status = "sufficient"
        elif sufficiency >= self.PARTIAL_THRESHOLD:
            proof_status = "partial"
        else:
            proof_status = "insufficient"

        # Upsert.
        existing = self.db.execute(
            "SELECT id FROM proof_state WHERE matter_id=? AND issue_id=?",
            (self.matter_id, issue_id),
        ).fetchone()

        _advocacy_only_int = 1 if advocacy_only else 0

        if existing:
            ps_id = existing["id"]
            self.db.execute(
                """UPDATE proof_state
                   SET sufficiency=?, supporting_count=?, attacking_count=?,
                       total_predicate_count=?, satisfied_predicate_count=?,
                       proof_status=?,
                       trust_weighted_support=?, trust_weighted_attack=?,
                       advocacy_only=?, computed_at=?
                   WHERE id=?""",
                (sufficiency, supporting, attacking,
                 total_predicates, satisfied_predicates,
                 proof_status,
                 trust_weighted_support, trust_weighted_attack,
                 _advocacy_only_int, now, ps_id),
            )
        else:
            ps_id = _id()
            self.db.execute(
                """INSERT INTO proof_state
                   (id, matter_id, issue_id, sufficiency, supporting_count,
                    attacking_count, total_predicate_count, satisfied_predicate_count,
                    proof_status, trust_weighted_support, trust_weighted_attack,
                    advocacy_only, computed_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (ps_id, self.matter_id, issue_id, sufficiency,
                 supporting, attacking, total_predicates, satisfied_predicates,
                 proof_status,
                 trust_weighted_support, trust_weighted_attack,
                 _advocacy_only_int, now),
            )

        return {
            "id": ps_id,
            "matter_id": self.matter_id,
            "issue_id": issue_id,
            "sufficiency": sufficiency,
            "supporting_count": supporting,
            "attacking_count": attacking,
            "total_predicate_count": total_predicates,
            "satisfied_predicate_count": satisfied_predicates,
            "proof_status": proof_status,
            "trust_weighted_support": trust_weighted_support,
            "trust_weighted_attack": trust_weighted_attack,
            "advocacy_only": advocacy_only,  # bool in returned dict
            "computed_at": now,
        }

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row) -> dict:
        """Convert a proof_state DB row to a dict, normalising advocacy_only to bool."""
        d = dict(row)
        # advocacy_only is stored as INTEGER (0/1) in SQLite; expose as Python bool.
        d["advocacy_only"] = bool(d.get("advocacy_only", 0))
        # trust fields always present from schema v24 columns; ensure floats.
        d.setdefault("trust_weighted_support", 0.0)
        d.setdefault("trust_weighted_attack", 0.0)
        return d

    def get(self, issue_id: str) -> Optional[dict]:
        """Return the stored proof state for an issue, or None if never computed."""
        row = self.db.execute(
            "SELECT * FROM proof_state WHERE matter_id=? AND issue_id=?",
            (self.matter_id, issue_id),
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_all(self, min_sufficiency: float = 0.0) -> list[dict]:
        """Return proof states for all issues, filtered by min sufficiency.

        Ordered by sufficiency ascending (weakest proof first — drives attention).
        """
        rows = self.db.execute(
            """SELECT * FROM proof_state
               WHERE matter_id=? AND sufficiency >= ?
               ORDER BY sufficiency ASC, proof_status""",
            (self.matter_id, min_sufficiency),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_by_status(self, proof_status: str) -> list[dict]:
        """Return all proof states with a given proof_status."""
        rows = self.db.execute(
            """SELECT * FROM proof_state
               WHERE matter_id=? AND proof_status=?
               ORDER BY sufficiency ASC""",
            (self.matter_id, proof_status),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_advocacy_only(self) -> list[dict]:
        """Return proof states where advocacy_only=1, using ix_proof_state_advocacy.

        Used by the synthesis gate to find issues whose support comes exclusively
        from advocacy/post_hoc sources. The dedicated index makes this efficient
        at scale rather than full-scan + Python filter.
        """
        rows = self.db.execute(
            """SELECT * FROM proof_state
               WHERE matter_id=? AND advocacy_only=1
               ORDER BY sufficiency ASC, proof_status""",
            (self.matter_id,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_gaps(self, threshold: float = PARTIAL_THRESHOLD) -> list[dict]:
        """Return proof states with sufficiency below threshold — the weakest issues.

        These are the issues most in need of additional evidence.
        """
        rows = self.db.execute(
            """SELECT * FROM proof_state
               WHERE matter_id=? AND sufficiency < ?
               ORDER BY sufficiency ASC""",
            (self.matter_id, threshold),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_summary(self) -> dict:
        """Return aggregate proof coverage statistics for the matter."""
        rows = self.db.execute(
            """SELECT proof_status, COUNT(*) AS n, AVG(sufficiency) AS avg_suf
               FROM proof_state
               WHERE matter_id=?
               GROUP BY proof_status""",
            (self.matter_id,),
        ).fetchall()

        total_row = self.db.execute(
            "SELECT COUNT(*) AS n, AVG(sufficiency) AS avg_suf FROM proof_state WHERE matter_id=?",
            (self.matter_id,),
        ).fetchone()

        by_status = {r["proof_status"]: r["n"] for r in rows}
        return {
            "total_issues_tracked": total_row["n"] if total_row else 0,
            "avg_sufficiency": round(total_row["avg_suf"] or 0.0, 3) if total_row else 0.0,
            "by_status": by_status,
            "gap_count": by_status.get("insufficient", 0) + by_status.get("partial", 0),
        }

    def compute_all(self) -> list[dict]:
        """Recompute proof state for every open issue in the matter.

        Fetches trust overrides once and passes them into each per-issue call
        to avoid N separate override queries for N issues.

        Returns the list of updated proof state dicts.
        """
        # Batch fetch overrides once for all issues in this invocation.
        override_rows = self.db.execute(
            """SELECT document_pattern, trust_level FROM document_trust_override
               WHERE matter_id=? AND trust_level != 'normal'
               ORDER BY LENGTH(document_pattern) DESC""",
            (self.matter_id,),
        ).fetchall()
        preloaded = [(r["document_pattern"], r["trust_level"]) for r in override_rows]

        rows = self.db.execute(
            "SELECT id FROM issue WHERE matter_id=? AND status='open'",
            (self.matter_id,),
        ).fetchall()
        return [self.compute_and_store(r["id"], _preloaded_overrides=preloaded) for r in rows]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _score(
        supporting: int,
        attacking: int,
        total_predicates: int,
        satisfied_predicates: int,
    ) -> float:
        """Compute a 0..1 sufficiency score.

        Formula:
        - If predicates exist: predicate_ratio × assertion_ratio
        - If no predicates: assertion_ratio alone (softer signal)

        assertion_ratio = supporting / (supporting + attacking + 1)
            The +1 prevents division-by-zero and penalises zero supporting.
        """
        assertion_ratio = supporting / (supporting + attacking + 1)

        if total_predicates > 0 and satisfied_predicates > 0:
            # Both predicates defined AND some resolved: multiplicative boost.
            predicate_ratio = satisfied_predicates / total_predicates
            score = predicate_ratio * assertion_ratio
        else:
            # No predicates defined, OR predicates defined but none resolved yet
            # (no production writer has called resolve_predicate() yet).
            # Fall back to assertion_ratio alone so issues with predicates are not
            # unfairly scored 0 before the predicate resolver is wired in (SO-4).
            score = assertion_ratio

        return round(min(max(score, 0.0), 1.0), 4)
