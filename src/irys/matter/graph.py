"""AssertionStore, GapStore, ActorStore, IssueStore.

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

        with self.db.transaction():
            # Check for existing canonical assertion — must match matter, layer, AND prop key
            # so the same text proposition can coexist in separate reasoning layers (SO-2/arch §2)
            row = self.db.execute(
                "SELECT id, belief_state FROM assertion WHERE matter_id=? AND model_layer=? AND proposition_key=?",
                (self.matter_id, candidate.model_layer.value, prop_key),
            ).fetchone()

            if row is None:
                # New canonical assertion
                assertion_id = _id()
                self.db.execute(
                    """INSERT INTO assertion
                       (id, matter_id, proposition_key, proposition_text,
                        model_layer, assertion_kind,
                        subject_ref_type, subject_ref_id, predicate_key, object_json,
                        temporal_scope_start, temporal_scope_end,
                        belief_state, confidence, created_at, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        assertion_id, self.matter_id, prop_key,
                        candidate.proposition_text,
                        candidate.model_layer.value,
                        candidate.assertion_kind.value,
                        candidate.subject_ref_type,
                        candidate.subject_ref_id,
                        candidate.predicate_key,
                        candidate.object_json,
                        candidate.temporal_scope_start,
                        candidate.temporal_scope_end,
                        BeliefState.UNKNOWN.value,
                        0.5,
                        now, now,
                    ),
                )
                is_new = True
            else:
                assertion_id = row["id"]
                is_new = False

            # Always insert an occurrence (even for known assertions from new docs)
            occ_id = _id()
            self.db.execute(
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
        Return assertion IDs that this assertion supports (they depend on it).

        When assertion_id changes, all dependents must be re-evaluated.
        Link direction: assertion_id --SUPPORTS--> dependent

        Note: depends_on links are stored but intentionally excluded from
        propagation — the direction semantics (src depends on dst, so dst's
        changes should re-evaluate src) require a separate reversed query that
        is not implemented yet. Including depends_on here propagates in the
        wrong direction (toward the prerequisite, not toward the dependent).
        """
        rows = self.db.execute(
            """SELECT dst_assertion_id FROM assertion_link
               WHERE src_assertion_id=? AND link_type='supports'""",
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
        """Return assertion IDs that support this assertion.
        Link direction: supporter --SUPPORTS--> assertion_id
        """
        rows = self.db.execute(
            """SELECT src_assertion_id FROM assertion_link
               WHERE dst_assertion_id=? AND link_type='supports'""",
            (assertion_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def count(self) -> int:
        row = self.db.execute(
            "SELECT COUNT(*) FROM assertion WHERE matter_id=?", (self.matter_id,)
        ).fetchone()
        return row[0]

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

    def open_gaps(self, min_materiality: float = 0.0) -> list[dict]:
        """Return open gaps above a materiality threshold."""
        rows = self.db.execute(
            """SELECT * FROM gap WHERE matter_id=? AND status='open'
               AND materiality_score >= ? ORDER BY materiality_score DESC""",
            (self.matter_id, min_materiality),
        ).fetchall()
        return [dict(r) for r in rows]

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
        Add a testable predicate to an issue.
        Returns predicate_id.
        """
        pred_id = _id()
        now = _now()
        with self.db.transaction():
            self.db.execute(
                """INSERT INTO issue_predicate
                   (id, issue_id, description, burden_side, status, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (pred_id, issue_id, description, burden_side, "open", now),
            )
        return pred_id

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

    def get_predicates(self, issue_id: str) -> list[dict]:
        """Return all predicates for an issue."""
        rows = self.db.execute(
            "SELECT * FROM issue_predicate WHERE issue_id=? AND status='open' ORDER BY created_at",
            (issue_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def count_open(self) -> int:
        row = self.db.execute(
            "SELECT COUNT(*) FROM issue WHERE matter_id=? AND status='open'",
            (self.matter_id,),
        ).fetchone()
        return row[0]
