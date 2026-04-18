"""AssertionStore, GapStore, ActorStore, IssueStore, ClarificationStore, QuantStore,
DocumentInventoryStore, DocumentCardStore, SpanStore, DocumentActorRoleStore,
TrustOverrideStore, DocumentAnnotationStore, ReasoningCacheStore, AssumptionStore.

The assertion store is the heart of the intelligence layer. It maintains
typed assertions with speech-act classification, support/attack links,
and revisable belief states.
"""

import hashlib
import json as _json_mod
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

_log = logging.getLogger(__name__)

from .db import SQLiteMatterDB
import pathlib
from .enums import (
    BeliefState, SpeechAct, SourceRole, ModelLayer, AssertionKind,
    AssertionLinkType, OriginKind, GapType, IssueType, SOURCE_TRUST_WEIGHTS,
    VerificationStatus, VerificationTargetKind, ReviewScope, ReviewedByKind,
    EvidenceRelationType, EvidenceOriginKind,
)
from .models import AssertionCandidate, AssertionRecord, RevisionResult, ClaimIdentity, ProvenanceContext


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

    @staticmethod
    def _pick_occurrence_value(existing_value, candidate_value, promote: bool):
        if existing_value is None:
            return candidate_value
        if promote and candidate_value is not None:
            return candidate_value
        return existing_value

    def resolve_claim_key(self, candidate: AssertionCandidate) -> ClaimIdentity:
        return candidate.resolve_claim_identity()

    def _resolve_document_inventory_id(self, candidate: AssertionCandidate) -> Optional[str]:
        if candidate.document_inventory_id:
            return candidate.document_inventory_id
        doc_norm = (candidate.document_id or "").replace("\\", "/")
        if not doc_norm:
            return None
        row = self.db.execute(
            "SELECT id FROM document_inventory WHERE matter_id=? AND relative_path=?",
            (self.matter_id, doc_norm),
        ).fetchone()
        return row["id"] if row else None

    def _find_occurrence(
        self,
        document_inventory_id: Optional[str],
        document_id: str,
        speech_act: str,
        span_id: Optional[str],
        claim_key: Optional[str] = None,
    ):
        """Find a prior occurrence for the same (doc, speech_act, span) position.

        When claim_key is provided, first tries an exact match (same claim or its
        legacy predecessor). Falls back to a broader search only when span_id is
        non-null (positional re-extraction). This prevents genuinely different facts
        from the same document colliding when both have span_id=None.
        """
        # Exact match: same claim_key_candidate (re-extraction of the same fact)
        if claim_key:
            row = self.db.execute(
                """SELECT ao.*
                   FROM assertion_occurrence ao
                   JOIN assertion a ON a.id = ao.assertion_id
                   WHERE a.matter_id=?
                     AND COALESCE(ao.document_inventory_id, ao.document_id)=COALESCE(?, ?)
                     AND ao.speech_act=?
                     AND COALESCE(ao.span_id, '')=COALESCE(?, '')
                     AND ao.claim_key_candidate=?
                   ORDER BY ao.created_at ASC, ao.id ASC
                   LIMIT 1""",
                (self.matter_id, document_inventory_id, document_id, speech_act, span_id, claim_key),
            ).fetchone()
            if row:
                return row

        # Broader match only when span_id is set (positional dedup).
        # Without a span_id, different facts from the same doc would collide.
        if span_id:
            return self.db.execute(
                """SELECT ao.*
                   FROM assertion_occurrence ao
                   JOIN assertion a ON a.id = ao.assertion_id
                   WHERE a.matter_id=?
                     AND COALESCE(ao.document_inventory_id, ao.document_id)=COALESCE(?, ?)
                     AND ao.speech_act=?
                     AND ao.span_id=?
                   ORDER BY ao.created_at ASC, ao.id ASC
                   LIMIT 1""",
                (self.matter_id, document_inventory_id, document_id, speech_act, span_id),
            ).fetchone()

        return None

    def upsert_occurrence(
        self,
        candidate: AssertionCandidate,
        run_id: Optional[str] = None,
        *,
        provenance: "Optional[ProvenanceContext]" = None,
    ) -> tuple[str, bool]:
        """
        Upsert a canonical assertion keyed by claim_key and record one occurrence.

        proposition_key remains legacy compatibility only and is never used as the
        canonical uniqueness key after claim identity v2.

        Returns (assertion_id, is_new_assertion).
        """
        identity = self.resolve_claim_key(candidate)
        prop_key = identity.legacy_proposition_key
        now = _now()
        _init_state, _init_conf = _initial_belief_state(candidate.speech_act)
        _candidate_id = _id()
        _doc_norm = (candidate.document_id or "").replace("\\\\", "/").replace("\\", "/")
        _doc_basename = pathlib.Path(_doc_norm).name if _doc_norm else None
        document_inventory_id = self._resolve_document_inventory_id(candidate)
        raw_text = candidate.raw_text or candidate.proposition_text

        with self.db.transaction():
            # Ambiguity detection for non-Tier-A resolutions: warn if the same
            # proposition_key already maps to a different claim_key.
            if identity.resolution_strategy != "tier_a_structured":
                competing_rows = self.db.execute(
                    """SELECT id, claim_key
                       FROM assertion
                       WHERE matter_id=? AND model_layer=? AND proposition_key=?
                         AND claim_key IS NOT NULL
                         AND claim_key != ?
                       ORDER BY canonicalization_confidence DESC, created_at ASC""",
                    (
                        self.matter_id,
                        candidate.model_layer.value,
                        prop_key,
                        identity.claim_key,
                    ),
                ).fetchall()
                if competing_rows:
                    _log.warning(
                        "claim_identity_ambiguous matter=%s prop_key=%s candidate=%s competing=%s",
                        self.matter_id,
                        prop_key,
                        identity.claim_key,
                        [r["id"] for r in competing_rows],
                    )

            # INSERT OR IGNORE keyed on ux_assertion_claim_key (matter_id, model_layer, claim_key).
            _assert_cur = self.db.execute(
                """INSERT OR IGNORE INTO assertion
                   (id, matter_id, proposition_key, claim_key, identity_version,
                    proposition_text, model_layer, assertion_kind, polarity,
                    canonical_subject_key, subject_ref_type, subject_ref_id, predicate_key,
                    canonical_object_key, object_json, temporal_scope_start, temporal_scope_end,
                    temporal_identity_key, speaker_scope_key, canonicalization_confidence,
                    belief_state, confidence, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    _candidate_id,
                    self.matter_id,
                    prop_key,
                    identity.claim_key,
                    identity.identity_version,
                    candidate.proposition_text,
                    candidate.model_layer.value,
                    candidate.assertion_kind.value,
                    identity.polarity,
                    identity.canonical_subject_key,
                    candidate.subject_ref_type,
                    candidate.subject_ref_id,
                    candidate.predicate_key,
                    identity.canonical_object_key,
                    candidate.object_json,
                    candidate.temporal_scope_start,
                    candidate.temporal_scope_end,
                    identity.temporal_identity_key,
                    identity.speaker_scope_key,
                    identity.canonicalization_confidence,
                    _init_state.value,
                    _init_conf,
                    now,
                    now,
                ),
            )
            is_new = _assert_cur.rowcount > 0

            if is_new:
                assertion_id = _candidate_id
                row = None
            else:
                row = self.db.execute(
                    """SELECT id, belief_state, confidence,
                              predicate_key, subject_ref_type, subject_ref_id,
                              object_json, temporal_scope_start, temporal_scope_end,
                              claim_key, identity_version, polarity,
                              canonical_subject_key, canonical_object_key,
                              temporal_identity_key, speaker_scope_key,
                              canonicalization_confidence
                       FROM assertion
                       WHERE matter_id=? AND model_layer=? AND claim_key=?""",
                    (
                        self.matter_id,
                        candidate.model_layer.value,
                        identity.claim_key,
                    ),
                ).fetchone()
                assertion_id = row["id"]

            # Find existing occurrence for this doc+speech_act+span+claim
            existing_occurrence = self._find_occurrence(
                document_inventory_id,
                candidate.document_id,
                candidate.speech_act.value,
                candidate.span_id,
                claim_key=identity.claim_key,
            )
            occurrence_added_to_assertion = (
                existing_occurrence is None
                or existing_occurrence["assertion_id"] != assertion_id
            )

            new_occurrence_id: Optional[str] = None
            if existing_occurrence is None:
                new_occurrence_id = _id()
                self.db.execute(
                    """INSERT INTO assertion_occurrence
                       (id, assertion_id, document_id, document_inventory_id, doc_basename,
                        raw_text, span_id, speaker_actor_id, source_role, source_side,
                        speech_act, origin_kind, subject_ref_type, subject_ref_id,
                        predicate_key, object_json, temporal_scope_start, temporal_scope_end,
                        polarity, speaker_scope_key, claim_key_candidate, resolution_strategy,
                        extraction_confidence, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        new_occurrence_id,
                        assertion_id,
                        candidate.document_id,
                        document_inventory_id,
                        _doc_basename,
                        raw_text,
                        candidate.span_id,
                        candidate.speaker_actor_id,
                        candidate.source_role.value,
                        candidate.source_side,
                        candidate.speech_act.value,
                        candidate.origin_kind.value,
                        candidate.subject_ref_type,
                        candidate.subject_ref_id,
                        candidate.predicate_key,
                        candidate.object_json,
                        candidate.temporal_scope_start,
                        candidate.temporal_scope_end,
                        identity.polarity,
                        identity.speaker_scope_key,
                        identity.claim_key,
                        identity.resolution_strategy,
                        identity.canonicalization_confidence,
                        now,
                    ),
                )
            else:
                existing_conf = (
                    existing_occurrence["extraction_confidence"]
                    if existing_occurrence["extraction_confidence"] is not None
                    else 0.0
                )
                promote = identity.canonicalization_confidence >= existing_conf
                self.db.execute(
                    """UPDATE assertion_occurrence
                       SET assertion_id=?,
                           document_id=?,
                           document_inventory_id=COALESCE(document_inventory_id, ?),
                           doc_basename=COALESCE(doc_basename, ?),
                           raw_text=COALESCE(raw_text, ?),
                           subject_ref_type=?,
                           subject_ref_id=?,
                           predicate_key=?,
                           object_json=?,
                           temporal_scope_start=?,
                           temporal_scope_end=?,
                           polarity=?,
                           speaker_scope_key=?,
                           claim_key_candidate=?,
                           resolution_strategy=?,
                           extraction_confidence=?
                       WHERE id=?""",
                    (
                        assertion_id,
                        candidate.document_id,
                        document_inventory_id,
                        _doc_basename,
                        raw_text,
                        self._pick_occurrence_value(existing_occurrence["subject_ref_type"], candidate.subject_ref_type, promote),
                        self._pick_occurrence_value(existing_occurrence["subject_ref_id"], candidate.subject_ref_id, promote),
                        self._pick_occurrence_value(existing_occurrence["predicate_key"], candidate.predicate_key, promote),
                        self._pick_occurrence_value(existing_occurrence["object_json"], candidate.object_json, promote),
                        self._pick_occurrence_value(existing_occurrence["temporal_scope_start"], candidate.temporal_scope_start, promote),
                        self._pick_occurrence_value(existing_occurrence["temporal_scope_end"], candidate.temporal_scope_end, promote),
                        identity.polarity if promote else (existing_occurrence["polarity"] or identity.polarity),
                        self._pick_occurrence_value(existing_occurrence["speaker_scope_key"], identity.speaker_scope_key, promote),
                        identity.claim_key if promote or existing_occurrence["claim_key_candidate"] is None else existing_occurrence["claim_key_candidate"],
                        identity.resolution_strategy if promote or existing_occurrence["resolution_strategy"] is None else existing_occurrence["resolution_strategy"],
                        max(existing_conf, identity.canonicalization_confidence),
                        existing_occurrence["id"],
                    ),
                )

            # Upgrade canonical belief_state only when:
            #   (a) the occurrence was actually new for this assertion, AND
            #   (b) the assertion is not in a terminal state (SUPERSEDED/WITHDRAWN).
            if not is_new and occurrence_added_to_assertion:
                _new_state, _new_conf = _initial_belief_state(candidate.speech_act)
                _current_conf = row["confidence"] if row["confidence"] is not None else 0.5
                _TERMINAL = (BeliefState.SUPERSEDED.value, BeliefState.WITHDRAWN.value)
                if _new_conf > _current_conf and row["belief_state"] not in _TERMINAL:
                    _up_rev_rows: list[tuple[str, str, str]] = []
                    if row["belief_state"] != _new_state.value:
                        _up_rev_rows.append((
                            "belief_state",
                            _json_mod.dumps(row["belief_state"]),
                            _json_mod.dumps(_new_state.value),
                        ))
                    _up_rev_rows.append((
                        "confidence",
                        _json_mod.dumps(_current_conf),
                        _json_mod.dumps(_new_conf),
                    ))
                    self.write_revision_rows(
                        assertion_id, _up_rev_rows, _id(),
                        "occurrence_upgrade", "system",
                        run_id=run_id,
                    )
                    self.db.execute(
                        "UPDATE assertion SET belief_state=?, confidence=?, updated_at=? WHERE id=?",
                        (_new_state.value, _new_conf, now, assertion_id),
                    )

            # Upgrade identity + SPO fields on existing assertions (NULL→non-NULL backfill).
            _identity_field_names = (
                "predicate_key", "subject_ref_type", "subject_ref_id",
                "object_json", "temporal_scope_start", "temporal_scope_end",
                "claim_key", "identity_version", "polarity",
                "canonical_subject_key", "canonical_object_key",
                "temporal_identity_key", "speaker_scope_key",
            )
            if (
                not is_new
                and row is not None
                and candidate.predicate_key is not None
                and any(row[f] is None for f in _identity_field_names)
            ):
                _identity_fields_with_old = [
                    ("predicate_key", row["predicate_key"], candidate.predicate_key),
                    ("subject_ref_type", row["subject_ref_type"], candidate.subject_ref_type),
                    ("subject_ref_id", row["subject_ref_id"], candidate.subject_ref_id),
                    ("object_json", row["object_json"], candidate.object_json),
                    ("temporal_scope_start", row["temporal_scope_start"], candidate.temporal_scope_start),
                    ("temporal_scope_end", row["temporal_scope_end"], candidate.temporal_scope_end),
                    ("claim_key", row["claim_key"], identity.claim_key),
                    ("identity_version", row["identity_version"], identity.identity_version if row["identity_version"] == "legacy_text_v1" else row["identity_version"]),
                    ("polarity", row["polarity"], identity.polarity),
                    ("canonical_subject_key", row["canonical_subject_key"], identity.canonical_subject_key),
                    ("canonical_object_key", row["canonical_object_key"], identity.canonical_object_key),
                    ("temporal_identity_key", row["temporal_identity_key"], identity.temporal_identity_key if row["temporal_identity_key"] == "atemporal" else row["temporal_identity_key"]),
                    ("speaker_scope_key", row["speaker_scope_key"], identity.speaker_scope_key),
                ]
                _identity_rev_rows = [
                    (field, _json_mod.dumps(None), _json_mod.dumps(new_val))
                    for field, old_val, new_val in _identity_fields_with_old
                    if old_val is None and new_val is not None
                ]
                if _identity_rev_rows:
                    self.write_revision_rows(
                        assertion_id, _identity_rev_rows, _id(),
                        "occurrence_upgrade", "system",
                        run_id=run_id,
                    )
                    self.db.execute(
                    """UPDATE assertion
                       SET subject_ref_type=COALESCE(subject_ref_type, ?),
                           subject_ref_id=COALESCE(subject_ref_id, ?),
                           predicate_key=COALESCE(predicate_key, ?),
                           object_json=COALESCE(object_json, ?),
                           temporal_scope_start=COALESCE(temporal_scope_start, ?),
                           temporal_scope_end=COALESCE(temporal_scope_end, ?),
                           claim_key=COALESCE(claim_key, ?),
                           identity_version=CASE
                               WHEN identity_version='legacy_text_v1' THEN ?
                               ELSE identity_version
                           END,
                           polarity=?,
                           canonical_subject_key=COALESCE(canonical_subject_key, ?),
                           canonical_object_key=COALESCE(canonical_object_key, ?),
                           temporal_identity_key=CASE
                               WHEN temporal_identity_key='atemporal' AND ? != 'atemporal' THEN ?
                               ELSE temporal_identity_key
                           END,
                           speaker_scope_key=COALESCE(speaker_scope_key, ?),
                           canonicalization_confidence=MAX(canonicalization_confidence, ?),
                           updated_at=?
                       WHERE id=?""",
                    (
                        candidate.subject_ref_type,
                        candidate.subject_ref_id,
                        candidate.predicate_key,
                        candidate.object_json,
                        candidate.temporal_scope_start,
                        candidate.temporal_scope_end,
                        identity.claim_key,
                        identity.identity_version,
                        identity.polarity,
                        identity.canonical_subject_key,
                        identity.canonical_object_key,
                        identity.temporal_identity_key,
                        identity.temporal_identity_key,
                        identity.speaker_scope_key,
                        identity.canonicalization_confidence,
                        now,
                        assertion_id,
                    ),
                )

            # MVP.2 + P0.4: every AI write touches verification_state.
            # touch_ai_target() seeds candidate on first write, revives
            # stale → candidate on re-extraction (so a re-ingested
            # document doesn't leave targets stale forever), and leaves
            # candidate/verified/rejected alone.
            _vs_store = VerificationStateStore(self.db, self.matter_id)
            _vs_store.touch_ai_target(
                VerificationTargetKind.ASSERTION,
                assertion_id,
                ai_confidence=_init_conf,
                cause="assertion_upsert",
                run_id=run_id,
            )
            # P0.4 review fix: revive both NEW (INSERT) and EXISTING
            # (UPDATE) occurrence rows. The UPDATE branch runs when
            # the same (doc, span, speech_act) slot gets a re-written
            # occurrence — without this, a stale occurrence stays
            # stale forever after re-extraction.
            _occ_revive_id = (
                new_occurrence_id if new_occurrence_id is not None
                else (existing_occurrence["id"] if existing_occurrence else None)
            )
            if _occ_revive_id is not None:
                _vs_store.touch_ai_target(
                    VerificationTargetKind.ASSERTION_OCCURRENCE,
                    _occ_revive_id,
                    ai_confidence=_init_conf,
                    cause="assertion_upsert",
                    run_id=run_id,
                )
            if is_new:
                # P0.1 provenance: emit rows for both the canonical
                # assertion and the occurrence that produced it so an
                # audit can trace either back to the originating AI
                # call. Occurrence rows can later be looked up by
                # target_kind='assertion_occurrence'. We only record
                # these on NEW assertions — subsequent occurrences of
                # the same assertion go through a different path.
                if provenance is not None:
                    prov_store = ProvenanceStore(self.db, self.matter_id)
                    prov_store.record(
                        target_kind="assertion",
                        target_id=assertion_id,
                        context=provenance,
                    )
                    # Adversarial audit #6: the occurrence row was
                    # claimed attributable but never had a provenance
                    # event written for it. Record one now so an audit
                    # can trace each raw utterance back to its call.
                    if new_occurrence_id is not None:
                        prov_store.record(
                            target_kind="assertion_occurrence",
                            target_id=new_occurrence_id,
                            context=provenance,
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

    def write_revision_rows(
        self,
        assertion_id: str,
        rows: "list[tuple[str, str, str]]",
        batch_id: str,
        cause: str,
        actor_kind: str,
        run_id: Optional[str] = None,
        note: Optional[str] = None,
        actor_ref: Optional[str] = None,
    ) -> None:
        """Write immutable field-diff rows to assertion_revision (Q4 HIGH, SO-2).

        Each entry in rows is (changed_field, old_value_json, new_value_json) where
        values are pre-serialized JSON strings so null, numbers, and nested objects
        round-trip cleanly without ambiguity.

        All rows share the same batch_id so they can be grouped by correction call.
        Must be called inside an active transaction when multiple mutations are batched.
        """
        now = _now()
        for changed_field, old_val_json, new_val_json in rows:
            self.db.execute(
                """INSERT INTO assertion_revision
                   (id, batch_id, assertion_id, changed_field,
                    old_value_json, new_value_json,
                    actor_kind, actor_ref, cause, run_id, note, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    _id(), batch_id, assertion_id, changed_field,
                    old_val_json, new_val_json,
                    actor_kind, actor_ref, cause, run_id, note, now,
                ),
            )

    def detect_oscillation(self, assertion_id: str, window: int = 6) -> bool:
        """Return True if the assertion's belief_state has cycled (A→B→A) in recent history.

        Reads the last `window` belief_state changes from assertion_revision, ordered by
        creation time.  Detects an oscillation pattern where any non-adjacent pair shares
        the same value (e.g., [operative, disputed, operative] over 3 batches).

        An oscillating assertion indicates a contradictory or unstable justification
        network: contradictory evidence, missing temporal/supersession semantics, or an
        over-eager auto-upgrade loop.  Callers should emit a SYSTEM_WARNING and mark
        the affected issue as contested.

        Returns False if there is insufficient history or the matter model is unavailable.
        """
        rows = self.db.execute(
            """SELECT new_value_json FROM assertion_revision
               WHERE assertion_id=? AND changed_field='belief_state'
               ORDER BY created_at DESC LIMIT ?""",
            (assertion_id, window),
        ).fetchall()
        if len(rows) < 3:
            return False
        # Extract belief state values from JSON (values are stored as JSON-encoded strings)
        import json as _json_osc
        try:
            values = [_json_osc.loads(r["new_value_json"]) for r in rows]
        except Exception:
            return False
        # Oscillation: any state appears more than once with a different state between
        seen: set[str] = set()
        for i, val in enumerate(values):
            if val in seen:
                # Found a repeated value — check if anything different appeared between
                # the first occurrence and this one.
                first_idx = values.index(val)
                if any(v != val for v in values[first_idx:i]):
                    return True
            seen.add(val)
        return False

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
        superseding_states: list = []
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
                # Collect belief state of every superseding assertion so
                # _compute_belief_state can distinguish "active superseder" (→ SUPERSEDED)
                # from "superseder was itself withdrawn/superseded" (→ allow recovery).
                # (HIGH #1 supersession-recovery fix)
                superseding_states.append(bs)
        # Derive legacy bool for callers that use has_superseding directly.
        _INERT_SUPERSEDER = (BeliefState.WITHDRAWN, BeliefState.SUPERSEDED)
        has_superseding = any(s not in _INERT_SUPERSEDER for s in superseding_states)
        return {
            "support_states": support_states,
            "attack_states": attack_states,
            "has_superseding": has_superseding,
            "superseding_states": superseding_states,
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
        # OPT-4: the three correlated subqueries that pulled
        # primary_document_id/primary_source_role/primary_speech_act per
        # row are replaced with a single ROW_NUMBER()-windowed CTE over
        # assertion_occurrence so only the first occurrence per
        # assertion materialises. For the aggregates we still GROUP BY
        # on the bounded id set.
        rows = self.db.execute(
            """WITH recent_ids AS (
                   SELECT id, proposition_text, model_layer, assertion_kind,
                          belief_state, confidence, created_at
                   FROM assertion
                   WHERE matter_id=?
                   ORDER BY created_at DESC
                   LIMIT ? OFFSET ?
               ),
               first_occurrence AS (
                   SELECT ao.assertion_id,
                          ao.document_id  AS primary_document_id,
                          ao.source_role  AS primary_source_role,
                          ao.speech_act   AS primary_speech_act,
                          ROW_NUMBER() OVER (
                              PARTITION BY ao.assertion_id
                              ORDER BY ao.created_at ASC, ao.id ASC
                          ) AS rn
                   FROM assertion_occurrence ao
                   WHERE ao.assertion_id IN (SELECT id FROM recent_ids)
               )
               SELECT a.id, a.proposition_text, a.model_layer, a.assertion_kind,
                      a.belief_state, a.confidence, a.created_at,
                      COUNT(ao.id) AS occurrence_count,
                      GROUP_CONCAT(DISTINCT ao.source_role) AS source_roles_csv,
                      GROUP_CONCAT(DISTINCT ao.speech_act) AS speech_acts_csv,
                      GROUP_CONCAT(DISTINCT ao.document_id) AS documents_csv,
                      fo.primary_document_id,
                      fo.primary_source_role,
                      fo.primary_speech_act
               FROM recent_ids a
               LEFT JOIN assertion_occurrence ao ON ao.assertion_id = a.id
               LEFT JOIN first_occurrence fo ON fo.assertion_id = a.id AND fo.rn = 1
               GROUP BY a.id
               ORDER BY a.created_at DESC""",
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

        P0.2: also returns verification_status so callers can classify
        each row through TrustPolicy and partition into
        verified/candidate/stale/excluded buckets. The hydration read
        is the one place that deliberately wants to SEE stale and
        rejected rows — the engine renders them in separate buckets
        rather than dropping them silently — so this method returns
        them and lets TrustPolicy decide eligibility downstream.

        Returns: [{id, proposition_text, belief_state, source_role,
                   verification_status}]
        """
        # Pre-aggregate source roles for the filtered set in a single CTE pass
        # instead of two correlated subqueries per row (resolves Tier 1 perf MEDIUM).
        # filtered_ids: the LIMIT-200 candidate set.
        # src_agg: GROUP_CONCAT and best-rank computation done once over those IDs.
        # P0.2: oversample so verified/candidate rows are not starved
        # by a flood of stale/rejected rows that share the head of the
        # chronological window. Caller's limit is enforced by
        # _hydrate_from_matter_model after TrustPolicy classification.
        oversample = max(limit, limit * 3)
        rows = self.db.execute(
            """WITH filtered_ids AS (
                   SELECT id FROM assertion
                   WHERE matter_id=?
                     AND belief_state NOT IN ('disputed','withdrawn','superseded')
                   ORDER BY created_at DESC
                   LIMIT ?
               ),
               src_agg AS (
                   SELECT ao.assertion_id,
                          GROUP_CONCAT(DISTINCT ao.source_role) AS source_roles_csv,
                          (SELECT ao2.source_role FROM assertion_occurrence ao2
                           WHERE ao2.assertion_id = ao.assertion_id
                           ORDER BY CASE ao2.source_role
                             WHEN 'authoritative' THEN 6
                             WHEN 'operative'     THEN 5
                             WHEN 'procedural'    THEN 4
                             WHEN 'post_hoc'      THEN 3
                             WHEN 'informal'      THEN 2
                             WHEN 'draft'         THEN 1
                             WHEN 'unknown'       THEN 1
                             WHEN 'advocacy'      THEN 0
                             ELSE 1 END DESC, ao2.created_at ASC LIMIT 1
                          ) AS source_role
                   FROM assertion_occurrence ao
                   WHERE ao.assertion_id IN (SELECT id FROM filtered_ids)
                   GROUP BY ao.assertion_id
               )
               SELECT a.id, a.proposition_text, a.belief_state,
                      a.subject_ref_type, a.subject_ref_id,
                      a.predicate_key, a.object_json,
                      sa.source_role, sa.source_roles_csv,
                      vs.status AS verification_status
               FROM filtered_ids fi
               JOIN assertion a ON a.id = fi.id
               LEFT JOIN src_agg sa ON sa.assertion_id = a.id
               LEFT JOIN verification_state vs
                      ON vs.matter_id = a.matter_id
                     AND vs.target_kind = 'assertion'
                     AND vs.target_id = a.id
               ORDER BY a.created_at DESC""",
            (self.matter_id, oversample),
        ).fetchall()
        return [dict(r) for r in rows]

    def search(
        self,
        queries: "list[str]",
        issue_id: "str | None" = None,
        limit: int = 20,
    ) -> list[dict]:
        """Search active assertions for hot-path retrieval.

        Matches against canonical proposition text plus occurrence raw_text and
        document identifiers so filename-targeted leads can still resolve to the
        already-ingested evidence they point to.

        P0.2: drops stale/rejected rows at the DB layer (not eligible
        under TrustPurpose.CACHED_SEARCH) and returns trust_bucket on
        every row so callers can label candidate hits as leads rather
        than source-text equivalents.
        """
        terms: list[str] = []
        seen: set[str] = set()
        for query in queries or []:
            cleaned = " ".join((query or "").strip().split())
            if len(cleaned) < 2:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            terms.append(cleaned)
        if not terms:
            return []

        term_score_parts: list[str] = []
        term_params: list[str] = []
        for term in terms:
            pattern = f"%{term.lower()}%"
            term_score_parts.append(
                """MAX(CASE
                       WHEN LOWER(COALESCE(a.proposition_text, '')) LIKE ?
                         OR LOWER(COALESCE(ao.raw_text, '')) LIKE ?
                         OR LOWER(COALESCE(ao.document_id, '')) LIKE ?
                         OR LOWER(COALESCE(ao.doc_basename, '')) LIKE ?
                       THEN 1 ELSE 0 END)"""
            )
            term_params.extend([pattern, pattern, pattern, pattern])

        term_matches_sql = " + ".join(term_score_parts)
        issue_match_sql = (
            "MAX(CASE WHEN ail.issue_id = ? THEN 1 ELSE 0 END)"
            if issue_id else "0"
        )

        params: list = list(term_params)
        if issue_id:
            params.append(issue_id)
        params.extend([self.matter_id, limit])

        rows = self.db.execute(
            f"""WITH matched AS (
                   SELECT a.id,
                          a.proposition_text,
                          a.belief_state,
                          a.created_at,
                          {term_matches_sql} AS term_matches,
                          {issue_match_sql} AS issue_match,
                          vs.status AS verification_status,
                          (SELECT ao2.document_id FROM assertion_occurrence ao2
                           WHERE ao2.assertion_id = a.id
                           ORDER BY ao2.created_at ASC, ao2.id ASC
                           LIMIT 1) AS primary_document_id,
                          (SELECT ao2.raw_text FROM assertion_occurrence ao2
                           WHERE ao2.assertion_id = a.id
                           ORDER BY ao2.created_at ASC, ao2.id ASC
                           LIMIT 1) AS primary_raw_text,
                          (SELECT ao2.source_role FROM assertion_occurrence ao2
                           WHERE ao2.assertion_id = a.id
                           ORDER BY CASE ao2.source_role
                             WHEN 'authoritative' THEN 6
                             WHEN 'operative'     THEN 5
                             WHEN 'procedural'    THEN 4
                             WHEN 'post_hoc'      THEN 3
                             WHEN 'informal'      THEN 2
                             WHEN 'draft'         THEN 1
                             WHEN 'unknown'       THEN 1
                             WHEN 'advocacy'      THEN 0
                             ELSE 1 END DESC,
                                    ao2.created_at ASC,
                                    ao2.id ASC
                           LIMIT 1) AS source_role,
                          (SELECT GROUP_CONCAT(DISTINCT ao2.source_role)
                           FROM assertion_occurrence ao2
                           WHERE ao2.assertion_id = a.id) AS source_roles_csv
                   FROM assertion a
                   LEFT JOIN assertion_occurrence ao ON ao.assertion_id = a.id
                   LEFT JOIN assertion_issue_link ail ON ail.assertion_id = a.id
                   LEFT JOIN verification_state vs
                          ON vs.matter_id = a.matter_id
                         AND vs.target_kind = 'assertion'
                         AND vs.target_id = a.id
                   WHERE a.matter_id=?
                     AND a.belief_state NOT IN ('disputed','withdrawn','superseded')
                   GROUP BY a.id
               )
               SELECT *
               FROM matched
               WHERE term_matches > 0
                 -- P0.2: TrustPurpose.CACHED_SEARCH only accepts
                 -- verified and candidate. Stale/rejected drop here.
                 AND (verification_status IS NULL
                      OR verification_status IN ('verified','candidate'))
               ORDER BY issue_match DESC, term_matches DESC, created_at DESC
               LIMIT ?""",
            params,
        ).fetchall()
        # P0.2: classify each row into a trust_bucket label so
        # consumers can tag candidate hits as leads rather than
        # source-text equivalents.
        out: list[dict] = []
        for r in rows:
            d = dict(r)
            d["trust_bucket"] = (
                "verified" if d.get("verification_status") == "verified"
                else "candidate"
            )
            out.append(d)
        return out

    def get_by_proposition(
        self,
        proposition_text: str,
        model_layer: Optional[str] = None,
    ) -> Optional["AssertionRecord"]:
        """Legacy compatibility lookup by normalized proposition text.

        proposition_key is no longer unique inside a model layer once claim identity v2
        lands, so this method must be deterministic and clearly treated as fallback-only.
        """
        import hashlib
        normalized = " ".join(proposition_text.lower().split())
        prop_key = hashlib.sha256(normalized.encode()).hexdigest()[:32]
        if model_layer is not None:
            row = self.db.execute(
                "SELECT * FROM assertion"
                " WHERE matter_id=? AND model_layer=? AND proposition_key=?"
                " ORDER BY canonicalization_confidence DESC, created_at ASC LIMIT 1",
                (self.matter_id, model_layer, prop_key),
            ).fetchone()
        else:
            row = self.db.execute(
                "SELECT * FROM assertion"
                " WHERE matter_id=? AND proposition_key=?"
                " ORDER BY canonicalization_confidence DESC, created_at ASC LIMIT 1",
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

    def find_contradictions(self, limit: Optional[int] = None) -> list[dict]:
        """
        Return pairs of assertions in this matter that are in active conflict.

        Detection strategy: explicit assertion_link rows where link_type IN
        ('attacks', 'contradicts') and BOTH the src and dst assertions have
        an active belief_state (not superseded/withdrawn/resolved).

        Link direction convention:
          src_assertion_id --ATTACKS/CONTRADICTS--> dst_assertion_id
          src is the attacker; dst is the assertion being challenged.

        Args:
            limit: Optional DB-side limit.  Callers that only need the top N
                   (e.g. steering surface) should pass limit to avoid loading
                   the full conflict set on large matters.

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
        limit_clause = f" LIMIT {int(limit)}" if limit is not None else ""
        rows = self.db.execute(
            f"""SELECT al.src_assertion_id AS attacker_id,
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
                 AND a_dst.belief_state NOT IN ('superseded','withdrawn','resolved')
               {limit_clause}""",
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
            "SELECT al.src_assertion_id, al.dst_assertion_id FROM assertion_link al"
            " JOIN assertion a ON a.id = al.src_assertion_id"
            " WHERE al.link_type IN ('attacks','contradicts') AND a.matter_id=?",
            (self.matter_id,),
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
                    except Exception as _e:
                        _log.warning("detect_heuristic_contradictions: link insert failed (%s→%s): %s", src_id, dst_id, _e)
        return created

    def mine_and_mark_contradictions(
        self,
        gap_store: "GapStore",
        belief_engine: "BeliefRevisionEngine",
        run_id: "str | None" = None,
        _truncated_nodes: "list[str] | None" = None,
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
                    _fs_result = belief_engine.force_state(
                        assertion_id=attacked_id,
                        new_state=BeliefState.DISPUTED,
                        new_confidence=0.3,
                        cause=RevisionCause.CONFLICT_DETECTION,
                        run_id=run_id,
                        note=(
                            f"Marked disputed by {conflict['link_type']} link from "
                            f"assertion {conflict['attacker_id']}"
                        ),
                    )
                    if _fs_result.propagation_truncated:
                        _log.warning(
                            "detect_conflicts: force_state truncated propagation for %s"
                            " — downstream belief states may be stale.",
                            attacked_id,
                        )
                        if _truncated_nodes is not None:
                            _truncated_nodes.extend(_fs_result.truncation_pending)
            except Exception as _e:
                _log.warning("mine_and_mark_contradictions: force_state failed for %s: %s", attacked_id, _e)

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
            except Exception as _e:
                _log.warning("mine_and_mark_contradictions: gap record failed for %s: %s", attacked_id, _e)

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

    def open_gaps(self, min_materiality: float = 0.0, limit: "int | None" = None) -> list[dict]:
        """Return open gaps above a materiality threshold.

        Each gap dict includes a 'dependencies' key: list of
        {affected_type, affected_id} dicts from gap_link so callers
        can see what the gap is linked to without a second query (SO-7).

        limit: if set, return at most this many gaps and restrict the
        gap_link fetch to only those rows (avoids full-table scan for
        small-limit callers like the overview panel).
        """
        limit_sql = f" LIMIT {int(limit)}" if limit is not None else ""
        rows = self.db.execute(
            f"""SELECT * FROM gap WHERE matter_id=? AND status='open'
               AND materiality_score >= ? ORDER BY materiality_score DESC{limit_sql}""",
            (self.matter_id, min_materiality),
        ).fetchall()
        if not rows:
            return []

        # Fetch gap_links only for the returned gap IDs.
        # When limit is small this avoids scanning all gap_link rows.
        gap_ids = [r["id"] for r in rows]
        placeholders = ",".join("?" * len(gap_ids))
        link_rows = self.db.execute(
            f"""SELECT gl.gap_id, gl.affected_type, gl.affected_id
               FROM gap_link gl WHERE gl.gap_id IN ({placeholders})""",
            gap_ids,
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

    def list_actors(self, limit: "int | None" = None) -> list[dict]:
        """Return actors for this matter, ordered by canonical name."""
        limit_sql = f" LIMIT {int(limit)}" if limit is not None else ""
        rows = self.db.execute(
            f"SELECT * FROM actor WHERE matter_id=? ORDER BY canonical_name{limit_sql}",
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
        self, min_prefix_len: int = 6, limit: int = 100
    ) -> list[dict]:
        """Return pairs of actors whose normalized names share a common prefix.

        Each entry: {actor_a: {...}, actor_b: {...}, shared_prefix: str}
        Only pairs where both actors have different ids are returned.
        Ordered by shared_prefix length descending (most similar first).

        Complexity: O(N log N) for the sort (done by SQLite) plus O(N + P·k)
        for the scan, where P is the number of matching pairs and k is the
        average cluster size.  Exploits the sorted order so pairs with matching
        min_prefix are always adjacent — no full cross-product needed.
        """
        rows = self.db.execute(
            "SELECT id, canonical_name, normalized_name, actor_type FROM actor WHERE matter_id=? ORDER BY normalized_name",
            (self.matter_id,),
        ).fetchall()

        actors = [dict(r) for r in rows]
        pairs: list[dict] = []
        seen_pairs: set = set()

        for i, a in enumerate(actors):
            n_a = a.get("normalized_name") or ""
            if len(n_a) < min_prefix_len:
                continue
            a_prefix = n_a[:min_prefix_len]
            # Actors are sorted by normalized_name so all potential matches are
            # adjacent.  Stop scanning forward as soon as the shared min-prefix
            # is no longer satisfied.
            for b in actors[i + 1:]:
                n_b = b.get("normalized_name") or ""
                if len(n_b) < min_prefix_len or n_b[:min_prefix_len] != a_prefix:
                    break  # sorted order guarantees no further matches
                # Compute full common prefix length.
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
        return pairs[:limit]

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
        Create or retrieve an issue by normalized title scoped to parent + type.
        Returns (issue_id, is_new).
        """
        normalized_title = " ".join(title.lower().split())
        now = _now()

        with self.db.transaction():
            if parent_issue_id:
                row = self.db.execute(
                    """SELECT id FROM issue
                       WHERE matter_id=? AND LOWER(title)=?
                       AND parent_issue_id=? AND issue_type=?""",
                    (self.matter_id, normalized_title,
                     parent_issue_id, issue_type.value),
                ).fetchone()
            else:
                row = self.db.execute(
                    """SELECT id FROM issue
                       WHERE matter_id=? AND LOWER(title)=?
                       AND parent_issue_id IS NULL AND issue_type=?""",
                    (self.matter_id, normalized_title, issue_type.value),
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
            cur = self.db.execute(
                """INSERT OR IGNORE INTO issue_predicate
                   (id, issue_id, description, burden_side, status, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (pred_id, issue_id, description, burden_side, "open", now),
            )
            is_new = cur.rowcount > 0
        row = self.db.execute(
            "SELECT id FROM issue_predicate WHERE issue_id=? AND description=?",
            (issue_id, description),
        ).fetchone()
        resolved_id = row["id"] if row else pred_id
        # MVP.2: predicates are AI-derived until reviewed. candidate() is
        # idempotent, so batch/legacy callers that already hold a row are
        # safe too.
        if is_new:
            VerificationStateStore(self.db, self.matter_id).candidate(
                VerificationTargetKind.ISSUE_PREDICATE,
                resolved_id,
                cause="predicate_add",
            )
        return resolved_id

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

    def apply_template(
        self,
        issue_id: str,
        template_id: str,
        *,
        registry=None,
    ) -> list[str]:
        """MVP.5: materialize template elements as issue_predicate rows.

        Idempotent on (issue_id, template_id, element_key) via the
        unique index added in migration v54, so re-applying the same
        template twice does not duplicate predicates.

        Returns the list of issue_predicate.id values for the template's
        elements, in element_order. Every newly-inserted predicate seeds
        a candidate verification_state row per MVP.2 contract so the
        LLM-cannot-resolve-elements invariant holds from inception.
        """
        from .templates import default_registry

        reg = registry if registry is not None else default_registry()
        template = reg.require(template_id)

        # Validate that the issue belongs to this matter (matter-scoped
        # writes are a project-wide discipline).
        issue_row = self.db.execute(
            "SELECT id FROM issue WHERE id=? AND matter_id=?",
            (issue_id, self.matter_id),
        ).fetchone()
        if issue_row is None:
            raise ValueError(
                f"issue {issue_id!r} does not exist in matter {self.matter_id!r}"
            )
        # MVP.5: single-template issues. Applying two different templates
        # to the same issue creates silent contamination because
        # propose_element_mappings reads only the first template it
        # finds. Reject the second apply explicitly; callers that need
        # a different template should remove the old predicates first.
        existing = self.db.execute(
            """SELECT DISTINCT template_id FROM issue_predicate
               WHERE issue_id=? AND template_id IS NOT NULL
                 AND template_id != ?""",
            (issue_id, template.id),
        ).fetchone()
        if existing is not None:
            raise ValueError(
                f"issue {issue_id!r} already carries template "
                f"{existing['template_id']!r}; refuse to apply a second "
                f"template ({template.id!r}) on the same issue"
            )

        predicate_ids: list[str] = []
        with self.db.transaction():
            for el in template.elements:
                pred_id = _id()
                now = _now()
                # Insert via the natural unique index (issue_id,
                # template_id, element_key). If a row already exists for
                # this template element, INSERT OR IGNORE leaves it alone.
                cur = self.db.execute(
                    """INSERT OR IGNORE INTO issue_predicate
                        (id, issue_id, description, burden_side, status,
                         created_at, template_id, template_version,
                         element_key, element_order)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (
                        pred_id, issue_id, el.description, el.burden_side,
                        "open", now, template.id, template.version,
                        el.key, el.order,
                    ),
                )
                is_new = cur.rowcount > 0
                # Resolve the actual id: on conflict the row already exists.
                row = self.db.execute(
                    """SELECT id FROM issue_predicate
                       WHERE issue_id=? AND template_id=? AND element_key=?""",
                    (issue_id, template.id, el.key),
                ).fetchone()
                resolved_id = row["id"] if row else pred_id
                predicate_ids.append(resolved_id)
                if is_new:
                    VerificationStateStore(self.db, self.matter_id).candidate(
                        VerificationTargetKind.ISSUE_PREDICATE,
                        resolved_id,
                        cause="template_apply",
                    )
        return predicate_ids

    def link_assertion_to_predicate(
        self,
        assertion_id: str,
        predicate_id: str,
        relation_type: str = "supports",
    ) -> str:
        """MVP.5: write a predicate-granularity evidence_edge row and
        mirror the support up to the issue level so the canonical
        substrate surfaces (get_issue_coverage_report,
        ProofStateStore.compute_and_store, _detect_proof_gaps) cannot
        disagree with the predicate-level view.

        Adversarial audit #5 reproduced a split-brain where verified
        predicate proof read True from get_predicates_with_proof while
        the issue-level substrate still reported supporting_count=0 and
        opened a missing_issue_predicate gap. The mirror closes that
        gap: every element-level mapping also becomes an issue-level
        evidence_edge via link_assertion (which idempotently writes the
        legacy assertion_issue_link AND the issue-level evidence_edge).

        Returns the predicate-level edge id. Idempotent across both
        surfaces via the natural key.
        """
        with self.db.transaction():
            edge_id, _ = EvidenceStore(self.db, self.matter_id).upsert_edge(
                source_kind="assertion",
                source_id=assertion_id,
                target_kind="issue_predicate",
                target_id=predicate_id,
                relation_type=relation_type,
                proof_weight=0.5,
                origin_kind=EvidenceOriginKind.SYSTEM_INFERRED,
            )
            # Mirror up to the issue. Resolve the issue id from the
            # predicate row; both writes end up inside this one
            # transaction so a mid-sequence failure cannot split state.
            pred_row = self.db.execute(
                "SELECT issue_id FROM issue_predicate WHERE id=?",
                (predicate_id,),
            ).fetchone()
            if pred_row is not None and pred_row["issue_id"]:
                self.link_assertion(
                    assertion_id, pred_row["issue_id"], relation_type
                )
        return edge_id

    def propose_element_mappings(
        self,
        assertion_id: str,
        issue_id: str,
        *,
        min_score: float = 0.1,
    ) -> list[tuple[str, float]]:
        """MVP.5: for an assertion linked to a template-driven issue,
        return [(predicate_id, score)] ranked by text overlap against
        each template element's mapping hints. Does NOT write edges —
        callers decide whether to materialize via
        link_assertion_to_predicate. Returns empty list when the issue
        has no template or the assertion has no resolvable text.
        """
        from .templates import default_registry

        # Find the template on this issue by inspecting any predicate
        # that carries template metadata. MVP.5 issues are single-template.
        pred_rows = self.db.execute(
            """SELECT id, template_id, element_key FROM issue_predicate
               WHERE issue_id=? AND template_id IS NOT NULL
               ORDER BY element_order""",
            (issue_id,),
        ).fetchall()
        if not pred_rows:
            return []
        template_id = pred_rows[0]["template_id"]
        reg = default_registry()
        template = reg.get(template_id)
        if template is None:
            return []

        # Resolve assertion text.
        a_row = self.db.execute(
            "SELECT proposition_text FROM assertion WHERE id=? AND matter_id=?",
            (assertion_id, self.matter_id),
        ).fetchone()
        if a_row is None:
            return []
        scored = reg.score_assertion_to_elements(a_row["proposition_text"], template)
        # Map element_key -> predicate_id for this specific issue
        key_to_pid = {r["element_key"]: r["id"] for r in pred_rows}
        result: list[tuple[str, float]] = []
        for key, score in scored:
            if score < min_score:
                continue
            pid = key_to_pid.get(key)
            if pid is None:
                continue
            result.append((pid, score))
        return result

    def link_assertion(
        self,
        assertion_id: str,
        issue_id: str,
        relation_type: str = "supports",
        *,
        provenance: "Optional[ProvenanceContext]" = None,
    ) -> str:
        """
        Link an assertion to an issue. Idempotent.
        relation_type: 'supports', 'attacks', 'establishes', 'negates'
        Returns link_id.

        P0.1: when provenance is supplied, the companion evidence_edge
        row carries a provenance_event so an audit can trace how the
        edge was inferred. Adversarial audit #6 regression — the
        implicit SYSTEM_INFERRED edge was the most common production
        edge write and previously had no provenance.
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
            # MVP.3: every assertion→issue link gets a companion
            # evidence_edge row so the proof-edge substrate stays in sync
            # with the legacy link table. upsert_edge is idempotent on the
            # same natural key the link uses, so re-linking the same
            # assertion+issue+relation does not inflate edge counts.
            if relation_type in (
                EvidenceRelationType.SUPPORTS.value,
                EvidenceRelationType.ESTABLISHES.value,
                EvidenceRelationType.ATTACKS.value,
                EvidenceRelationType.NEGATES.value,
            ):
                EvidenceStore(self.db, self.matter_id).upsert_edge(
                    source_kind="assertion",
                    source_id=assertion_id,
                    target_kind="issue",
                    target_id=issue_id,
                    relation_type=relation_type,
                    proof_weight=0.5,
                    origin_kind=EvidenceOriginKind.SYSTEM_INFERRED,
                    provenance=provenance,
                )
        row = self.db.execute(
            "SELECT id FROM assertion_issue_link WHERE assertion_id=? AND issue_id=? AND relation_type=?",
            (assertion_id, issue_id, relation_type),
        ).fetchone()
        return row["id"] if row else link_id

    def get_open_issues(self, min_materiality: float = 0.0, order_by_score: bool = True) -> list[dict]:
        """Return open issues, optionally ordered by salience × materiality descending.

        order_by_score=False skips the expression sort for callers that will re-sort
        the result themselves (e.g. get_issue_coverage_report() sorts by coverage_fraction).
        The (salience * materiality) expression cannot use a simple column index so
        SQLite must compute and sort the full result set; skipping it when unnecessary
        avoids this O(n log n) work on every overview/issues panel refresh.
        """
        order_clause = "ORDER BY (salience * materiality) DESC, id ASC" if order_by_score else "ORDER BY id ASC"
        rows = self.db.execute(
            f"""SELECT * FROM issue
               WHERE matter_id=? AND status='open' AND materiality >= ?
               {order_clause}""",
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

    def get_predicates_with_proof(
        self,
        issue_id: str,
        *,
        policy_audience: str = "clean",
    ) -> list[dict]:
        """MVP.5: return template predicates with per-element proof lanes.

        Each row includes template_id, element_key, element_order, plus:
        - supporting_count / verified_supporting_count
        - attacking_count / verified_attacking_count
        - candidate_sufficiency / verified_sufficiency (ratio in [0,1])
        - resolvable: True only when verified sufficiency passes the
          threshold — this is the gate that prevents MVP.5 AC #5
          (unverified mapping cannot fully resolve a predicate).

        Sufficiency formula (MVP.5): verified_supporting_count /
        (verified_supporting_count + verified_attacking_count + 1).
        Candidate lane uses the non-rejected remainder. Sources are
        evidence_edge rows with target_kind='issue_predicate'.
        """
        priv_filter = ""
        if policy_audience == "clean":
            priv_filter = (
                " AND a.id NOT IN ("
                " SELECT DISTINCT ao.assertion_id FROM assertion_occurrence ao"
                " LEFT JOIN document_inventory di"
                "   ON di.id = ao.document_inventory_id"
                "   OR di.relative_path = ao.document_id"
                " JOIN document_card dc ON dc.doc_id = di.id"
                " WHERE di.matter_id = a.matter_id AND dc.privilege_flag = 1"
                ")"
            )

        pred_rows = self.db.execute(
            """SELECT id, issue_id, description, burden_side, status,
                      template_id, template_version, element_key, element_order
               FROM issue_predicate
               WHERE issue_id=? AND status='open'
               ORDER BY element_order, created_at""",
            (issue_id,),
        ).fetchall()
        if not pred_rows:
            return []

        # One bulk query gathers edge stats per predicate so we avoid N+1.
        edge_query = f"""
            SELECT ee.target_id AS predicate_id,
                   ee.relation_type,
                   COUNT(*) AS raw_count,
                   SUM(CASE WHEN COALESCE(vs.status, 'candidate') = 'verified'
                        AND COALESCE(vs_edge.status, ee.verification_status, 'candidate') = 'verified'
                        THEN 1 ELSE 0 END) AS verified_count
            FROM evidence_edge ee
            JOIN assertion a ON a.id = ee.source_id
            LEFT JOIN verification_state vs
              ON vs.target_kind = 'assertion'
             AND vs.target_id = a.id
             AND vs.matter_id = a.matter_id
            LEFT JOIN verification_state vs_edge
              ON vs_edge.target_kind = 'evidence_edge'
             AND vs_edge.target_id = ee.id
             AND vs_edge.matter_id = ee.matter_id
            WHERE ee.matter_id=? AND ee.target_kind='issue_predicate'
              AND ee.active=1 AND ee.source_kind='assertion'
              AND ee.relation_type IN ('supports','establishes','attacks','negates')
              AND a.belief_state NOT IN ('disputed','withdrawn','superseded')
              -- P0.2: TrustPurpose.PROOF_CANDIDATE — stale drops alongside rejected.
              AND COALESCE(vs.status, 'candidate') NOT IN ('rejected','stale')
              AND COALESCE(vs_edge.status, ee.verification_status, 'candidate') NOT IN ('rejected','stale')
              {priv_filter}
            GROUP BY ee.target_id, ee.relation_type
        """
        edge_rows = self.db.execute(edge_query, (self.matter_id,)).fetchall()
        support: dict[str, dict] = {}
        attack: dict[str, dict] = {}
        for r in edge_rows:
            bucket = support if r["relation_type"] in ("supports", "establishes") else attack
            prev = bucket.setdefault(r["predicate_id"], {"raw": 0, "verified": 0})
            prev["raw"] += int(r["raw_count"])
            prev["verified"] += int(r["verified_count"] or 0)

        result: list[dict] = []
        for p in pred_rows:
            pid = p["id"]
            sup = support.get(pid, {"raw": 0, "verified": 0})
            atk = attack.get(pid, {"raw": 0, "verified": 0})
            verified_suff = (
                sup["verified"] / (sup["verified"] + atk["verified"] + 1)
                if sup["verified"] or atk["verified"]
                else 0.0
            )
            candidate_suff = (
                sup["raw"] / (sup["raw"] + atk["raw"] + 1)
                if sup["raw"] or atk["raw"]
                else 0.0
            )
            result.append({
                **dict(p),
                "supporting_count": sup["raw"],
                "verified_supporting_count": sup["verified"],
                "attacking_count": atk["raw"],
                "verified_attacking_count": atk["verified"],
                "candidate_sufficiency": round(candidate_suff, 4),
                "verified_sufficiency": round(verified_suff, 4),
                # Only verified sufficiency can resolve; candidate is advisory
                # (MVP.5 AC #5). Threshold 0.5 matches the existing proof_state
                # sufficient/partial threshold.
                "resolvable": verified_suff >= 0.5,
            })
        return result

    def resolve_predicate(self, predicate_id: str) -> bool:
        """Mark an issue predicate as resolved (SO-4 predicate-aware coverage).

        Returns True if the predicate existed and was updated; False if not found.
        Matter-scoped: only resolves predicates belonging to this matter.
        Called when a supporting assertion is confirmed to satisfy a claim element.
        """
        with self.db.transaction():
            cursor = self.db.execute(
                "UPDATE issue_predicate SET status='resolved'"
                " WHERE id=? AND issue_id IN (SELECT id FROM issue WHERE matter_id=?)",
                (predicate_id, self.matter_id),
            )
        return cursor.rowcount > 0

    def set_predicate_status(
        self, predicate_id: str, status: str, reason: str | None = None
    ) -> bool:
        """Set predicate status to any valid value (Gap 3: conditional logic).

        Valid statuses: open, resolved, contested, blocked.
        Returns True if updated.
        """
        valid = ("open", "resolved", "contested", "blocked")
        if status not in valid:
            raise ValueError(f"Invalid predicate status '{status}'; must be one of {valid}")
        with self.db.transaction():
            cursor = self.db.execute(
                "UPDATE issue_predicate SET status=?"
                " WHERE id=? AND issue_id IN (SELECT id FROM issue WHERE matter_id=?)",
                (status, predicate_id, self.matter_id),
            )
        return cursor.rowcount > 0

    def get_predicates_by_status(
        self,
        issue_id: str,
        statuses: tuple[str, ...] = ("open", "contested", "blocked"),
    ) -> list[dict]:
        """Return predicates for an issue filtered by status(es).

        Extends get_predicates() to support the full status lifecycle
        (open/resolved/contested/blocked) introduced in Gap 3.
        """
        placeholders = ",".join("?" for _ in statuses)
        rows = self.db.execute(
            f"SELECT id, issue_id, description, burden_side, status, created_at"
            f" FROM issue_predicate WHERE issue_id=? AND status IN ({placeholders})"
            f" ORDER BY created_at",
            (issue_id, *statuses),
        ).fetchall()
        return [dict(r) for r in rows]

    def resolve_predicate_by_description(self, issue_id: str, description: str) -> bool:
        """Mark the first open predicate matching description as resolved.

        Returns True if a predicate was found and resolved; False otherwise.
        Atomic (single UPDATE with subquery) and matter-scoped.
        Useful when the engine knows a predicate description was satisfied but does not
        have the predicate_id.
        """
        if not description:
            return False
        desc = description.strip()
        with self.db.transaction():
            cursor = self.db.execute(
                "UPDATE issue_predicate SET status='resolved'"
                " WHERE id = ("
                "   SELECT ip.id FROM issue_predicate ip"
                "   JOIN issue i ON i.id = ip.issue_id"
                "   WHERE ip.issue_id=? AND ip.description=? AND ip.status='open'"
                "     AND i.matter_id=?"
                "   ORDER BY ip.created_at LIMIT 1"
                ")",
                (issue_id, desc, self.matter_id),
            )
        return cursor.rowcount > 0

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

    # ------------------------------------------------------------------ #
    # Tree traversal (Gap 1: hierarchical issue model)                    #
    # ------------------------------------------------------------------ #

    def get_children(self, issue_id: str, include_closed: bool = False) -> list[dict]:
        """Return direct children of an issue, ordered by sort_order."""
        status_clause = "" if include_closed else " AND status='open'"
        rows = self.db.execute(
            f"SELECT * FROM issue WHERE matter_id=? AND parent_issue_id=?"
            f"{status_clause} ORDER BY sort_order, id",
            (self.matter_id, issue_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_ancestors(self, issue_id: str, include_self: bool = False) -> list[dict]:
        """Return ancestors from immediate parent up to root. Cycle-safe.

        If include_self is True, the issue itself is the first element.
        Result order: [self?], parent, grandparent, ..., root.
        """
        ancestors: list[dict] = []
        visited: set[str] = set()
        current_id: Optional[str] = issue_id

        if include_self:
            issue = self.get_issue(issue_id)
            if issue:
                ancestors.append(issue)
                visited.add(issue_id)
                current_id = issue.get("parent_issue_id")
            else:
                return []
        else:
            issue = self.get_issue(issue_id)
            if not issue:
                return []
            visited.add(issue_id)
            current_id = issue.get("parent_issue_id")

        while current_id and current_id not in visited:
            visited.add(current_id)
            parent = self.get_issue(current_id)
            if not parent:
                break
            ancestors.append(parent)
            current_id = parent.get("parent_issue_id")

        return ancestors

    def get_depth(self, issue_id: str) -> int:
        """Return the depth of an issue in the tree (root = 0). Cycle-safe."""
        return len(self.get_ancestors(issue_id))

    def get_subtree(
        self,
        issue_id: str,
        max_depth: Optional[int] = None,
        include_self: bool = True,
        include_closed: bool = False,
    ) -> list[dict]:
        """Return all descendants via BFS. Cycle-safe, bounded depth.

        Result is BFS order (parent before children). Each dict gets an extra
        '_depth' key indicating depth relative to the root issue (root = 0).
        """
        root = self.get_issue(issue_id)
        if not root:
            return []

        result: list[dict] = []
        visited: set[str] = {issue_id}

        if include_self:
            root["_depth"] = 0
            result.append(root)

        # BFS queue: (issue_id, depth)
        queue: list[tuple[str, int]] = [(issue_id, 0)]

        while queue:
            parent_id, depth = queue.pop(0)
            if max_depth is not None and depth >= max_depth:
                continue
            children = self.get_children(parent_id, include_closed=include_closed)
            for child in children:
                cid = child["id"]
                if cid in visited:
                    continue
                visited.add(cid)
                child["_depth"] = depth + 1
                result.append(child)
                queue.append((cid, depth + 1))

        return result

    def get_root_issues(self, include_closed: bool = False) -> list[dict]:
        """Return top-level issues (no parent), ordered by salience * materiality."""
        status_clause = "" if include_closed else " AND status='open'"
        rows = self.db.execute(
            f"SELECT * FROM issue WHERE matter_id=? AND parent_issue_id IS NULL"
            f"{status_clause} ORDER BY (salience * materiality) DESC, id ASC",
            (self.matter_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def compute_coverage_rollup(self, issue_id: str, include_closed: bool = False) -> dict:
        """Compute aggregate coverage for an issue and all its descendants.

        For leaf issues (no children), coverage = its own assertion support ratio.
        For parent issues, coverage = weighted average of children's coverage,
        weighted by (materiality * salience).

        Returns dict with: coverage_fraction, supporting_count, attacking_count,
        predicate_total, predicate_resolved, subtree_size, leaf_issue_ids,
        weakest_leaf_id, weakest_leaf_coverage.
        """
        subtree = self.get_subtree(issue_id, include_self=True, include_closed=include_closed)
        if not subtree:
            return {
                "coverage_fraction": 0.0, "supporting_count": 0,
                "attacking_count": 0, "predicate_total": 0,
                "predicate_resolved": 0, "subtree_size": 0,
                "leaf_issue_ids": [], "weakest_leaf_id": None,
                "weakest_leaf_coverage": 0.0,
            }

        # Find leaf nodes (issues with no children in the subtree)
        parent_ids = {i.get("parent_issue_id") for i in subtree}
        all_ids = {i["id"] for i in subtree}
        leaf_ids = all_ids - parent_ids

        # Compute per-issue coverage from assertions and predicates
        coverage_map: dict[str, float] = {}
        sup_map: dict[str, int] = {}
        atk_map: dict[str, int] = {}
        pred_total = 0
        pred_resolved = 0

        # P0.2 review fix #2 (round 2): the previous version only
        # filtered assertion-lane verification — edge-stale rows
        # still leaked into the rollup. Use an edge-first / legacy-
        # fallback pattern that mirrors
        # ProofStateStore._query_issue_linked_assertions so the
        # rollup matches canonical coverage. For each issue, prefer
        # evidence_edge rows (which carry both assertion and edge
        # verification); fall back to assertion_issue_link with
        # assertion-only filtering when the issue has no edges yet
        # (matters pre-MVP.3 migration).
        for issue in subtree:
            iid = issue["id"]
            # P0.2 re-review: decide substrate by presence of ANY
            # evidence_edge for this issue, not by whether the
            # eligibility-filtered query happened to return rows.
            # Otherwise a stale-all-edges issue falls through to the
            # legacy link-table read, which cannot see edge
            # verification at all and re-inflates support.
            has_any_edge = bool(self.db.execute(
                """SELECT 1 FROM evidence_edge
                   WHERE matter_id=? AND target_kind='issue' AND target_id=?
                     AND active=1 AND source_kind='assertion' LIMIT 1""",
                (self.matter_id, iid),
            ).fetchone())
            if has_any_edge:
                rows = self.db.execute(
                    """SELECT ee.relation_type, COUNT(*) as cnt
                       FROM evidence_edge ee
                       JOIN assertion a ON a.id=ee.source_id
                       LEFT JOIN verification_state vs
                         ON vs.target_kind='assertion'
                        AND vs.target_id=a.id
                        AND vs.matter_id=a.matter_id
                       LEFT JOIN verification_state vs_edge
                         ON vs_edge.target_kind='evidence_edge'
                        AND vs_edge.target_id=ee.id
                        AND vs_edge.matter_id=ee.matter_id
                       WHERE ee.matter_id=? AND ee.target_kind='issue'
                         AND ee.target_id=? AND ee.active=1
                         AND ee.source_kind='assertion'
                         AND ee.relation_type IN ('supports','establishes','attacks','negates')
                         AND a.belief_state NOT IN ('disputed','withdrawn','superseded')
                         AND COALESCE(vs.status, 'candidate') NOT IN ('rejected','stale')
                         AND COALESCE(vs_edge.status, ee.verification_status, 'candidate')
                             NOT IN ('rejected','stale')
                       GROUP BY ee.relation_type""",
                    (self.matter_id, iid),
                ).fetchall()
            else:
                rows = self.db.execute(
                    """SELECT ail.relation_type, COUNT(*) as cnt
                       FROM assertion_issue_link ail
                       JOIN assertion a ON a.id = ail.assertion_id
                       LEFT JOIN verification_state vs
                         ON vs.target_kind='assertion'
                        AND vs.target_id=a.id
                        AND vs.matter_id=a.matter_id
                       WHERE ail.issue_id=?
                         AND a.belief_state NOT IN ('disputed','withdrawn','superseded')
                         AND COALESCE(vs.status, 'candidate') NOT IN ('rejected','stale')
                       GROUP BY ail.relation_type""",
                    (iid,),
                ).fetchall()
            sup = sum(r["cnt"] for r in rows if r["relation_type"] in ("supports", "establishes"))
            atk = sum(r["cnt"] for r in rows if r["relation_type"] in ("attacks", "negates"))
            sup_map[iid] = sup
            atk_map[iid] = atk

            # Predicate coverage for this issue
            preds = self.db.execute(
                "SELECT status FROM issue_predicate WHERE issue_id=?",
                (iid,),
            ).fetchall()
            p_total = len(preds)
            p_resolved = sum(1 for p in preds if p["status"] == "resolved")
            pred_total += p_total
            pred_resolved += p_resolved

            # Leaf coverage: predicate-based if predicates exist, else assertion ratio
            if iid in leaf_ids:
                if p_total > 0:
                    coverage_map[iid] = p_resolved / p_total
                elif sup + atk > 0:
                    coverage_map[iid] = sup / (sup + atk)
                else:
                    coverage_map[iid] = 0.0

        # Bottom-up rollup: parent coverage = weighted avg of children
        # Process in reverse BFS order (deepest first)
        for issue in reversed(subtree):
            iid = issue["id"]
            if iid in coverage_map:
                continue  # leaf — already computed
            children = [i for i in subtree if i.get("parent_issue_id") == iid]
            if not children:
                coverage_map[iid] = 0.0
                continue
            total_weight = 0.0
            weighted_sum = 0.0
            for child in children:
                cid = child["id"]
                w = (child.get("materiality") or 0.5) * (child.get("salience") or 0.5)
                weighted_sum += coverage_map.get(cid, 0.0) * w
                total_weight += w
            coverage_map[iid] = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Find weakest leaf
        weakest_id = None
        weakest_cov = 1.0
        for lid in leaf_ids:
            cov = coverage_map.get(lid, 0.0)
            if cov < weakest_cov:
                weakest_cov = cov
                weakest_id = lid

        return {
            "coverage_fraction": coverage_map.get(issue_id, 0.0),
            "supporting_count": sum(sup_map.values()),
            "attacking_count": sum(atk_map.values()),
            "predicate_total": pred_total,
            "predicate_resolved": pred_resolved,
            "subtree_size": len(subtree),
            "leaf_issue_ids": list(leaf_ids),
            "weakest_leaf_id": weakest_id,
            "weakest_leaf_coverage": weakest_cov,
        }


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

    def answer_question(self, question_id: str, answer_text: str) -> bool:
        """Record the user's answer to a clarification question.

        Returns True if the question was found and updated, False if it does not
        exist in this matter (caller should return 404).
        """
        now = _now()
        cur = self.db.execute(
            """UPDATE clarification_question
               SET answer_text=?, answered_at=?, status='answered'
               WHERE id=? AND matter_id=?""",
            (answer_text, now, question_id, self.matter_id),
        )
        return cur.rowcount > 0

    def get_pending(self, limit: "int | None" = None) -> list[dict]:
        """Return unanswered clarification questions, newest first."""
        limit_sql = f" LIMIT {int(limit)}" if limit is not None else ""
        rows = self.db.execute(
            f"""SELECT * FROM clarification_question
               WHERE matter_id=? AND status='pending'
               ORDER BY created_at DESC{limit_sql}""",
            (self.matter_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_answered(self, limit: "int | None" = None) -> list[dict]:
        """Return answered questions — for injection into orientation context."""
        limit_sql = f" LIMIT {int(limit)}" if limit is not None else ""
        rows = self.db.execute(
            f"""SELECT * FROM clarification_question
               WHERE matter_id=? AND status='answered'
               ORDER BY answered_at DESC{limit_sql}""",
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
        date_precision: Optional[str] = None,
        provenance: "Optional[ProvenanceContext]" = None,
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
                    span_id, assertion_id, date_precision, quant_dedup_key, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (qf_id, self.matter_id, quant_kind, amount_value, date_value, date_end_value,
                 rate_value, currency, unit, raw_key, subject_type, subject_id,
                 span_id, assertion_id, date_precision, dedup_key, now),
            )
            if cur.rowcount == 0:
                # Already exists — return the existing ID. P0.4 review
                # fix: still touch verification_state so a stale
                # quant revives to candidate on re-extraction.
                row = self.db.execute(
                    "SELECT id FROM quant_fact WHERE matter_id=? AND quant_dedup_key=?",
                    (self.matter_id, dedup_key),
                ).fetchone()
                VerificationStateStore(self.db, self.matter_id).touch_ai_target(
                    VerificationTargetKind.QUANT_FACT,
                    row["id"],
                    cause="quant_record_reextraction",
                )
                return row["id"]
            # MVP.2 + P0.4: touch_ai_target seeds on first insert and
            # revives stale on re-extraction (document re-ingest).
            VerificationStateStore(self.db, self.matter_id).touch_ai_target(
                VerificationTargetKind.QUANT_FACT,
                qf_id,
                cause="quant_record",
            )
            # P0.1 provenance (new inserts only).
            if provenance is not None:
                ProvenanceStore(self.db, self.matter_id).record(
                    target_kind="quant_fact",
                    target_id=qf_id,
                    context=provenance,
                )
        return qf_id

    def record_many(
        self,
        specs: list[dict],
        *,
        provenance: "Optional[ProvenanceContext]" = None,
    ) -> list[str]:
        """Bulk-insert multiple quant facts in a single transaction.

        Each spec is a dict with the same keys as record() (quant_kind and
        raw_text required; all others optional). Uses executemany + INSERT OR
        IGNORE so duplicate entries (same quant_dedup_key) are silently skipped.

        MVP.2 + P0.1: every newly-inserted quant_fact also seeds a
        candidate verification_state row and, when a provenance context
        is supplied, appends a provenance_event row. Adversarial audit
        #6 reproduced the prior hole: record_quants_batch → record_many
        skipped both substrates.

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
                spec.get("date_precision"),
                dedup_key,
                now,
            ))

        with self.db.transaction():
            self.db.executemany(
                """INSERT OR IGNORE INTO quant_fact
                   (id, matter_id, quant_kind, amount_value, date_value, date_end_value,
                    rate_value, currency, unit, raw_text, subject_type, subject_id,
                    span_id, assertion_id, date_precision, quant_dedup_key, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
            # MVP.2 + P0.1: look up the actual ids for the rows that were
            # inserted (not duplicates) by re-querying with dedup_key,
            # then seed verification_state and provenance_event for each.
            _ver = VerificationStateStore(self.db, self.matter_id)
            _prov = ProvenanceStore(self.db, self.matter_id) if provenance else None
            for candidate_id, spec in zip(ids, specs):
                dedup_key = self._quant_key(
                    spec["quant_kind"], spec.get("subject_id"), spec["raw_text"],
                )
                row = self.db.execute(
                    "SELECT id FROM quant_fact WHERE matter_id=? AND quant_dedup_key=?",
                    (self.matter_id, dedup_key),
                ).fetchone()
                if row is None:
                    continue  # shouldn't happen, but skip defensively
                actual_id = row["id"]
                # P0.4 review fix: touch ALWAYS — seed on first insert
                # AND revive stale on re-extraction. The previous
                # `continue` when actual_id != candidate_id silently
                # left stale batched quants stale after re-ingest.
                _ver.touch_ai_target(
                    VerificationTargetKind.QUANT_FACT,
                    actual_id,
                    cause="quant_record_batch",
                )
                # Only new-insert rows get provenance (one attribution
                # event per real AI call, not per duplicate touch).
                if actual_id != candidate_id:
                    continue
                if _prov is not None:
                    _prov.record(
                        target_kind="quant_fact",
                        target_id=actual_id,
                        context=provenance,
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
            r["raw_texts"] = [t for t in (r.pop("texts") or "").split(" || ") if t]
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

    def update_hash(
        self, doc_id: str, sha256: str, size_bytes: int = 0,
    ) -> tuple[bool, Optional[str]]:
        """P0.4: record a content hash for a known inventory row
        after the real bytes have been read. Returns
        (hash_changed, old_sha256).

        Guards against clobbering a real hash with the "pending"
        placeholder: if the caller passes sha256="pending" and a
        real hash already exists, we leave the row alone and report
        hash_changed=False. hash_changed is True only when both the
        old and new hashes are real and different.
        """
        row = self.db.execute(
            "SELECT sha256 FROM document_inventory WHERE id=? AND matter_id=?",
            (doc_id, self.matter_id),
        ).fetchone()
        if row is None:
            return (False, None)
        old_sha = row["sha256"]
        # Never downgrade a real hash with the placeholder.
        if sha256 == "pending" and old_sha and old_sha != "pending":
            return (False, old_sha)
        if old_sha == sha256:
            return (False, old_sha)
        now = _now()
        self.db.execute(
            "UPDATE document_inventory SET sha256=?, size_bytes=?, last_read_at=? WHERE id=?",
            (sha256, size_bytes, now, doc_id),
        )
        # Only flag a change when the OLD value was also a real hash.
        # First real hash after "pending" is not an invalidation —
        # it's just the placeholder being replaced.
        changed = bool(old_sha and old_sha != "pending" and sha256 != "pending")
        return (changed, old_sha)

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
    # Cold-path maintenance scheduling (Priority 1)
    # ------------------------------------------------------------------

    def list_needing_profile(self, limit: int = 200) -> list[dict]:
        """Return docs that have not yet been profiled (maintenance_status='pending').

        Ordered by salience descending so the most-important documents get
        profiled first. Used by the query-agnostic maintenance scheduler.
        """
        rows = self.db.execute(
            """SELECT id, relative_path, file_type, sha256, size_bytes,
                      salience_score, ingest_status
               FROM document_inventory
               WHERE matter_id = ? AND maintenance_status = 'pending'
               ORDER BY salience_score DESC
               LIMIT ?""",
            (self.matter_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_profile_started(self, doc_id: str) -> None:
        """Set maintenance_status='profiling' to prevent duplicate work."""
        self.db.execute(
            "UPDATE document_inventory SET maintenance_status='profiling' WHERE id=?",
            (doc_id,),
        )

    def mark_profile_complete(self, doc_id: str) -> None:
        """Set maintenance_status='profiled' and record profiled_at timestamp."""
        now = _now()
        self.db.execute(
            """UPDATE document_inventory
               SET maintenance_status='profiled', profiled_at=?, last_maintained_at=?
               WHERE id=?""",
            (now, now, doc_id),
        )

    def mark_profile_failed(self, doc_id: str) -> None:
        """Set maintenance_status='failed' — distinct from profiled (success)
        and pending (never attempted). Clears profiled_at to avoid
        misclassifying failed docs as successfully profiled."""
        now = _now()
        self.db.execute(
            """UPDATE document_inventory
               SET maintenance_status='failed', profiled_at=NULL, last_maintained_at=?
               WHERE id=?""",
            (now, doc_id),
        )

    def set_family_membership(
        self,
        doc_ids: list[str],
        family_id: str,
        version_chain_id: Optional[str] = None,
    ) -> None:
        """Assign a batch of documents to a family/version chain."""
        for doc_id in doc_ids:
            self.db.execute(
                """UPDATE document_inventory
                   SET family_id=?, version_chain_id=?
                   WHERE id=?""",
                (family_id, version_chain_id, doc_id),
            )

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

    def set_salience(self, doc_id: str, salience_score: float) -> None:
        """Update the salience score for a document inventory row."""
        salience_score = max(0.0, min(1.0, salience_score))
        self.db.execute(
            "UPDATE document_inventory SET salience_score = ? WHERE id = ?",
            (salience_score, doc_id),
        )

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


class PrivilegeGate:
    """MVP.4 document-level privilege helper (SO-5).

    Read-only gate over document_card.privilege_flag that lets downstream
    consumers filter privileged material out of clean-mode proof,
    coverage, gaps, and context packets. This is deliberately minimal:
    no span-level taint, no privilege_classification enum, no
    review_task — those land in later phases.

    Privileged docs are identified by document_card.privilege_flag=1
    joined to document_inventory. Assertions, evidence_edges, and
    quant_facts are considered "privileged-sourced" if ANY occurrence
    they reference traces back to a privileged doc inventory id. This is
    fail-closed by design: partial privilege is treated as full
    privilege for clean output.
    """

    def __init__(self, db: SQLiteMatterDB, matter_id: str):
        self.db = db
        self.matter_id = matter_id

    def privileged_doc_inventory_ids(self) -> set[str]:
        """Return the set of document_inventory.id values flagged as
        privileged on their document_card row for this matter. Fetched
        once per call; callers that need to test many targets should
        cache the result."""
        rows = self.db.execute(
            """SELECT di.id
               FROM document_card dc
               JOIN document_inventory di ON di.id = dc.doc_id
               WHERE di.matter_id=? AND dc.privilege_flag=1""",
            (self.matter_id,),
        ).fetchall()
        return {r["id"] for r in rows}

    def is_document_privileged(self, doc_ref: str) -> bool:
        """Accepts either a document_inventory.id or a relative_path."""
        row = self.db.execute(
            """SELECT dc.privilege_flag
               FROM document_card dc
               JOIN document_inventory di ON di.id = dc.doc_id
               WHERE di.matter_id=? AND (di.id=? OR di.relative_path=?)
               LIMIT 1""",
            (self.matter_id, doc_ref, doc_ref),
        ).fetchone()
        return bool(row and row["privilege_flag"])

    def is_assertion_from_privileged_source(self, assertion_id: str) -> bool:
        """An assertion is considered privileged if any of its occurrences
        resolve to a privileged document inventory row. assertion_occurrence
        rows carry both document_inventory_id (newer) and document_id
        (relative path, legacy); we check both."""
        row = self.db.execute(
            """SELECT 1 FROM assertion_occurrence ao
               LEFT JOIN document_inventory di
                 ON di.id = ao.document_inventory_id
                 OR di.relative_path = ao.document_id
               JOIN document_card dc ON dc.doc_id = di.id
               WHERE ao.assertion_id=? AND di.matter_id=? AND dc.privilege_flag=1
               LIMIT 1""",
            (assertion_id, self.matter_id),
        ).fetchone()
        return row is not None

    def privileged_assertion_ids(self) -> set[str]:
        """Return every assertion id in this matter whose source is
        privileged. Single set-based query so proof consumers can
        filter in bulk without a per-assertion round-trip."""
        rows = self.db.execute(
            """SELECT DISTINCT ao.assertion_id
               FROM assertion_occurrence ao
               LEFT JOIN document_inventory di
                 ON di.id = ao.document_inventory_id
                 OR di.relative_path = ao.document_id
               JOIN document_card dc ON dc.doc_id = di.id
               WHERE di.matter_id=? AND dc.privilege_flag=1""",
            (self.matter_id,),
        ).fetchall()
        return {r["assertion_id"] for r in rows}


class ContentPolicyGuard:
    """P0.5 Content Policy MVI unified guard (SO-5).

    Composes ContentPolicy.decide() with the runtime stores so every
    user-facing surface (profile, deep_read, search_snippets_to_llm,
    hydration, synthesis_context, timeline_view, matrix_view,
    chat_response, export) routes through one place before
    privileged/unknown/low-trust material can enter clean output.

    Every decision is persisted to content_policy_audit so a later
    audit can answer "for this target, every time a clean-mode
    surface asked the guard, what did it decide and why?".
    """

    def __init__(self, db: "SQLiteMatterDB", matter_id: str) -> None:
        self.db = db
        self.matter_id = matter_id

    def decide(
        self,
        *,
        purpose: "ContentPurpose | str",
        subject_kind: str,
        subject_id: str,
        policy_audience: str = "clean",
        assertion_verification_status: Optional[str] = None,
        edge_verification_status: Optional[str] = None,
        belief_state: Optional[str] = None,
        privilege_flag: Optional[bool] = None,
        note: Optional[str] = None,
        record: bool = True,
    ) -> "ContentPolicyDecision":
        """Evaluate a single content-policy request and (by default)
        persist the decision to content_policy_audit. Callers that
        iterate over large sets and want to skip per-row audit writes
        can pass record=False.
        """
        # Local imports to keep graph.py's import section minimal and
        # avoid a circular-import risk.
        from .trust import ContentPolicy, ContentPurpose as _CP
        purpose_enum = (
            purpose if isinstance(purpose, _CP) else _CP(purpose)
        )
        decision = ContentPolicy.decide(
            purpose=purpose_enum,
            policy_audience=policy_audience,
            assertion_verification_status=assertion_verification_status,
            edge_verification_status=edge_verification_status,
            belief_state=belief_state,
            privilege_flag=privilege_flag,
        )
        if record:
            try:
                self._append_audit(
                    purpose=purpose_enum.value,
                    policy_audience=policy_audience,
                    subject_kind=subject_kind,
                    subject_id=subject_id,
                    action=decision.action.value,
                    reason_code=decision.reason_code,
                    trust_bucket=decision.trust_bucket.value,
                    privilege_flag=privilege_flag,
                    note=note,
                )
            except Exception:
                # Audit write must never break the calling path. A
                # failed audit is worse than no audit but not worse
                # than a failed read for an attorney waiting on the UI.
                pass
        return decision

    def _append_audit(
        self,
        *,
        purpose: str,
        policy_audience: str,
        subject_kind: str,
        subject_id: str,
        action: str,
        reason_code: str,
        trust_bucket: str,
        privilege_flag: Optional[bool],
        note: Optional[str],
    ) -> str:
        """Append one row to content_policy_audit. Called internally;
        direct use from stores is fine when a non-standard audit
        shape is needed."""
        row_id = _id()
        priv_int: Optional[int] = None
        if privilege_flag is True:
            priv_int = 1
        elif privilege_flag is False:
            priv_int = 0
        self.db.execute(
            """INSERT INTO content_policy_audit
                (id, matter_id, purpose, policy_audience,
                 target_kind, target_id, action, reason_code,
                 trust_bucket, privilege_flag, note, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (row_id, self.matter_id, purpose, policy_audience,
             subject_kind, subject_id, action, reason_code,
             trust_bucket, priv_int, note, _now()),
        )
        return row_id

    def list_decisions(
        self,
        target_kind: Optional[str] = None,
        target_id: Optional[str] = None,
        purpose: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """Read the audit trail. Callers filter by any combination of
        target, target_id, and purpose — or omit filters for the
        newest N decisions matter-wide."""
        clauses = ["matter_id=?"]
        params: list = [self.matter_id]
        if target_kind is not None:
            clauses.append("target_kind=?")
            params.append(target_kind)
        if target_id is not None:
            clauses.append("target_id=?")
            params.append(target_id)
        if purpose is not None:
            clauses.append("purpose=?")
            params.append(purpose)
        where = " AND ".join(clauses)
        params.append(int(limit))
        rows = self.db.execute(
            f"""SELECT * FROM content_policy_audit
                WHERE {where}
                ORDER BY created_at DESC LIMIT ?""",
            params,
        ).fetchall()
        return [dict(r) for r in rows]


class DocumentCardStore:
    """Per-document intelligence card store.

    Wraps the ``document_card`` table — each row is a structured summary of
    what a document IS: its type, source side, author, rhetorical posture,
    operative status, and unresolved flags.  Cards are written during
    ``_deep_read_document`` cold-path and read during retrieval planning
    to bias search toward high-value or under-explored documents.
    """

    def __init__(self, db: SQLiteMatterDB, matter_id: str):
        self.db = db
        self.matter_id = matter_id

    def upsert(
        self,
        doc_id: str,
        *,
        title: Optional[str] = None,
        doc_type: Optional[str] = None,
        doc_subtype: Optional[str] = None,
        source_side: Optional[str] = None,
        author: Optional[str] = None,
        sender: Optional[str] = None,
        recipient: Optional[str] = None,
        creation_date: Optional[str] = None,
        sent_date: Optional[str] = None,
        effective_date: Optional[str] = None,
        discovery_date: Optional[str] = None,
        purpose: Optional[str] = None,
        rhetorical_posture: Optional[str] = None,
        reliability_posture: Optional[str] = None,
        operative_status: str = "unknown",
        privilege_flag: Optional[bool] = None,
        unresolved_flags: Optional[list] = None,
        source_role: Optional[str] = None,
        signatories_json: Optional[str] = None,
        provenance: "Optional[ProvenanceContext]" = None,
    ) -> str:
        """Insert or update a document card for the given inventory doc_id.

        MVP.4 SO-5 privilege discipline: privilege_flag is now Optional and
        defaults to None. A None value means "don't change the existing
        flag" on conflict, so a profile call that doesn't know about
        privilege cannot silently clear a previously-privileged card.
        On insert without an existing row, None is coerced to False.
        """
        now = _now()
        # MVP.4: resolve privilege_flag to a concrete int for SQLite.
        # None means preserve the existing value (fail-closed — a refresh
        # cannot clear a prior privileged card). Absent an existing row,
        # None defaults to 0 on initial insert.
        if privilege_flag is None:
            existing = self.db.execute(
                "SELECT privilege_flag FROM document_card WHERE doc_id=?",
                (doc_id,),
            ).fetchone()
            resolved_privilege = int(existing["privilege_flag"]) if existing else 0
        else:
            resolved_privilege = int(bool(privilege_flag))
        card_id = str(uuid.uuid4())
        flags_json = _json_mod.dumps(unresolved_flags) if unresolved_flags else None
        self.db.execute(
            """INSERT INTO document_card
               (id, doc_id, title, doc_type, doc_subtype, source_side,
                author, sender, recipient, creation_date, sent_date,
                effective_date, discovery_date, purpose,
                rhetorical_posture, reliability_posture,
                operative_status, privilege_flag, unresolved_flags,
                source_role, signatories_json,
                created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(doc_id) DO UPDATE SET
                 title = COALESCE(excluded.title, document_card.title),
                 doc_type = COALESCE(excluded.doc_type, document_card.doc_type),
                 doc_subtype = COALESCE(excluded.doc_subtype, document_card.doc_subtype),
                 source_side = COALESCE(excluded.source_side, document_card.source_side),
                 author = COALESCE(excluded.author, document_card.author),
                 sender = COALESCE(excluded.sender, document_card.sender),
                 recipient = COALESCE(excluded.recipient, document_card.recipient),
                 creation_date = COALESCE(excluded.creation_date, document_card.creation_date),
                 sent_date = COALESCE(excluded.sent_date, document_card.sent_date),
                 effective_date = COALESCE(excluded.effective_date, document_card.effective_date),
                 discovery_date = COALESCE(excluded.discovery_date, document_card.discovery_date),
                 purpose = COALESCE(excluded.purpose, document_card.purpose),
                 rhetorical_posture = COALESCE(excluded.rhetorical_posture, document_card.rhetorical_posture),
                 reliability_posture = COALESCE(excluded.reliability_posture, document_card.reliability_posture),
                 operative_status = excluded.operative_status,
                 -- MVP.4: privilege_flag is already fail-closed in Python
                 -- (None callers get the existing row's value via the
                 -- lookup above), so the ON CONFLICT can just apply.
                 privilege_flag = excluded.privilege_flag,
                 unresolved_flags = COALESCE(excluded.unresolved_flags, document_card.unresolved_flags),
                 source_role = COALESCE(excluded.source_role, document_card.source_role),
                 signatories_json = COALESCE(excluded.signatories_json, document_card.signatories_json),
                 updated_at = excluded.updated_at
            """,
            (card_id, doc_id, title, doc_type, doc_subtype, source_side,
             author, sender, recipient, creation_date, sent_date,
             effective_date, discovery_date, purpose,
             rhetorical_posture, reliability_posture,
             operative_status, resolved_privilege, flags_json,
             source_role, signatories_json,
             now, now),
        )
        # Return the actual card id (might be existing row on conflict)
        row = self.db.execute(
            "SELECT id FROM document_card WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        actual_id = row["id"] if row else card_id
        # MVP.2 + P0.4: touch_ai_target seeds + revives stale cards on
        # re-profile (document re-ingest).
        VerificationStateStore(self.db, self.matter_id).touch_ai_target(
            VerificationTargetKind.DOCUMENT_CARD,
            actual_id,
            cause="document_card_upsert",
        )
        # P0.1 provenance: every profile write gets an event even on
        # update — the profile is a continuously-refreshed AI output, so
        # the history matters.
        if provenance is not None:
            ProvenanceStore(self.db, self.matter_id).record(
                target_kind="document_card",
                target_id=actual_id,
                context=provenance,
            )
        return actual_id

    def get_by_doc_id(self, doc_id: str) -> Optional[dict]:
        """Return card dict for an inventory doc_id, or None."""
        row = self.db.execute(
            "SELECT * FROM document_card WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        if d.get("unresolved_flags"):
            try:
                d["unresolved_flags"] = _json_mod.loads(d["unresolved_flags"])
            except (ValueError, TypeError):
                d["unresolved_flags"] = []
        return d

    def get_by_path(self, relative_path: str) -> Optional[dict]:
        """Return card dict by joining inventory on relative_path."""
        row = self.db.execute(
            """SELECT dc.* FROM document_card dc
               JOIN document_inventory di ON dc.doc_id = di.id
               WHERE di.matter_id = ? AND di.relative_path = ?""",
            (self.matter_id, relative_path),
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        if d.get("unresolved_flags"):
            try:
                d["unresolved_flags"] = _json_mod.loads(d["unresolved_flags"])
            except (ValueError, TypeError):
                d["unresolved_flags"] = []
        return d

    def list_candidates(
        self,
        doc_types: Optional[list] = None,
        unresolved_only: bool = False,
        issue_id: Optional[str] = None,
        limit: int = 200,
    ) -> list[dict]:
        """Return cards ranked by salience/recency for retrieval planning.

        Joins document_inventory to include salience_score and path.
        """
        sql = """SELECT dc.*, di.relative_path, di.salience_score, di.last_read_at
                 FROM document_card dc
                 JOIN document_inventory di ON dc.doc_id = di.id
                 WHERE di.matter_id = ?"""
        params: list = [self.matter_id]
        if doc_types:
            placeholders = ",".join("?" * len(doc_types))
            sql += f" AND dc.doc_type IN ({placeholders})"
            params.extend(doc_types)
        if unresolved_only:
            sql += " AND dc.unresolved_flags IS NOT NULL AND dc.unresolved_flags != '[]'"
        sql += " ORDER BY di.salience_score DESC, di.last_read_at ASC NULLS FIRST"
        sql += " LIMIT ?"
        params.append(limit)
        rows = self.db.execute(sql, params).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            if d.get("unresolved_flags"):
                try:
                    d["unresolved_flags"] = _json_mod.loads(d["unresolved_flags"])
                except (ValueError, TypeError):
                    d["unresolved_flags"] = []
            results.append(d)
        return results

    def count(self) -> int:
        """Return total number of document cards for this matter."""
        row = self.db.execute(
            """SELECT COUNT(*) as cnt FROM document_card dc
               JOIN document_inventory di ON dc.doc_id = di.id
               WHERE di.matter_id = ?""",
            (self.matter_id,),
        ).fetchone()
        return row["cnt"] if row else 0


class SpanStore:
    """Section-level read memory for documents.

    Wraps the ``span`` table — each row represents a located section, clause,
    quote, or evidence anchor within a document.  Spans enable section-level
    re-read avoidance: the engine can check which parts of a document have
    already been analyzed and target only untouched sections.
    """

    def __init__(self, db: SQLiteMatterDB, matter_id: str):
        self.db = db
        self.matter_id = matter_id

    def upsert(
        self,
        document_id: str,
        span_type: str,
        span_text: str,
        *,
        page_start: Optional[int] = None,
        page_end: Optional[int] = None,
        line_start: Optional[int] = None,
        line_end: Optional[int] = None,
        char_start: Optional[int] = None,
        char_end: Optional[int] = None,
        section_ref: Optional[str] = None,
        clause_ref: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        ordinal_in_doc: Optional[int] = None,
        text_hash: Optional[str] = None,
    ) -> str:
        """Insert or locate a span row. Deduplicates on (document_id, text_hash, span_type, page_start, char_start)."""
        if text_hash is None:
            text_hash = hashlib.sha256(span_text.encode()).hexdigest()
        now = _now()
        span_id = str(uuid.uuid4())
        self.db.execute(
            """INSERT INTO span
               (id, document_id, span_type, span_text, text_hash,
                page_start, page_end, line_start, line_end,
                char_start, char_end, section_ref, clause_ref,
                parent_span_id, ordinal_in_doc, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(document_id, text_hash, span_type, page_start, char_start)
               DO UPDATE SET span_text = excluded.span_text
            """,
            (span_id, document_id, span_type, span_text, text_hash,
             page_start, page_end, line_start, line_end,
             char_start, char_end, section_ref, clause_ref,
             parent_span_id, ordinal_in_doc, now),
        )
        # Return actual span id (may be existing row on conflict)
        row = self.db.execute(
            "SELECT id FROM span WHERE document_id = ? AND text_hash = ? AND span_type = ?",
            (document_id, text_hash, span_type),
        ).fetchone()
        return row["id"] if row else span_id

    def list_by_document(
        self,
        document_id: str,
        span_type: Optional[str] = None,
        limit: int = 200,
    ) -> list[dict]:
        """Return spans for a document, optionally filtered by type."""
        if span_type:
            rows = self.db.execute(
                """SELECT * FROM span
                   WHERE document_id = ? AND span_type = ?
                   ORDER BY ordinal_in_doc ASC NULLS LAST, page_start ASC NULLS LAST
                   LIMIT ?""",
                (document_id, span_type, limit),
            ).fetchall()
        else:
            rows = self.db.execute(
                """SELECT * FROM span
                   WHERE document_id = ?
                   ORDER BY ordinal_in_doc ASC NULLS LAST, page_start ASC NULLS LAST
                   LIMIT ?""",
                (document_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def count_by_document(self, document_id: str) -> int:
        """Return total span count for a document."""
        row = self.db.execute(
            "SELECT COUNT(*) as cnt FROM span WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        return row["cnt"] if row else 0

    def upsert_many(self, document_id: str, specs: list[dict]) -> list[str]:
        """Batch-insert multiple spans for a document.

        Each spec dict should contain keys matching upsert() parameters:
        span_type, span_text, and optional page_start, page_end, etc.
        Returns list of span IDs.
        """
        ids = []
        for spec in specs:
            sid = self.upsert(
                document_id=document_id,
                span_type=spec.get("span_type", "quote"),
                span_text=spec.get("span_text", ""),
                page_start=spec.get("page_start"),
                page_end=spec.get("page_end"),
                line_start=spec.get("line_start"),
                line_end=spec.get("line_end"),
                char_start=spec.get("char_start"),
                char_end=spec.get("char_end"),
                section_ref=spec.get("section_ref"),
                clause_ref=spec.get("clause_ref"),
                ordinal_in_doc=spec.get("ordinal_in_doc"),
            )
            ids.append(sid)
        return ids


class DocumentActorRoleStore:
    """Links actors to documents with role types.

    Wraps the ``document_actor_role`` table — each row records that a specific
    actor plays a specific role in a specific document (e.g., "author",
    "signatory", "recipient", "named_party", "witness").
    """

    def __init__(self, db: SQLiteMatterDB, matter_id: str):
        self.db = db
        self.matter_id = matter_id

    def upsert(
        self,
        doc_id: str,
        actor_id: str,
        role_type: str,
        *,
        raw_name: Optional[str] = None,
        confidence: float = 1.0,
    ) -> str:
        """Insert or update a document-actor role link."""
        now = _now()
        role_id = _id()
        self.db.execute(
            """INSERT INTO document_actor_role
               (id, doc_id, actor_id, role_type, raw_name, confidence, created_at)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(doc_id, actor_id, role_type)
               DO UPDATE SET raw_name = COALESCE(excluded.raw_name, document_actor_role.raw_name),
                             confidence = excluded.confidence""",
            (role_id, doc_id, actor_id, role_type, raw_name, confidence, now),
        )
        row = self.db.execute(
            "SELECT id FROM document_actor_role WHERE doc_id=? AND actor_id=? AND role_type=?",
            (doc_id, actor_id, role_type),
        ).fetchone()
        return row["id"] if row else role_id

    def list_by_document(self, doc_id: str) -> list[dict]:
        """Return all actor roles for a document."""
        rows = self.db.execute(
            """SELECT dar.*, a.canonical_name, a.actor_type
               FROM document_actor_role dar
               JOIN actor a ON dar.actor_id = a.id
               WHERE dar.doc_id = ?
               ORDER BY dar.role_type, a.canonical_name""",
            (doc_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_by_actor(self, actor_id: str) -> list[dict]:
        """Return all document roles for an actor."""
        rows = self.db.execute(
            """SELECT dar.*, di.relative_path
               FROM document_actor_role dar
               JOIN document_inventory di ON dar.doc_id = di.id
               WHERE dar.actor_id = ?
               ORDER BY dar.role_type""",
            (actor_id,),
        ).fetchall()
        return [dict(r) for r in rows]


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

    # ----- P0.4 trust revision helpers --------------------------------
    def current_trust_revision(self) -> int:
        """Return matter.trust_revision, the invalidation fingerprint
        every cache key is prefixed with. Legacy matters default to 0
        via the schema default."""
        try:
            row = self.db.execute(
                "SELECT trust_revision FROM matter WHERE id=?",
                (self.matter_id,),
            ).fetchone()
            if row is None:
                return 0
            return int(row["trust_revision"] or 0)
        except Exception:
            return 0

    def bump_trust_revision(self) -> int:
        """Increment matter.trust_revision by 1, returning the new value.
        P0.4 invalidation triggers call this after any write that
        marks downstream targets stale so cached reasoning plans keyed
        on the old revision become unreachable."""
        try:
            now = _now()
            self.db.execute(
                "UPDATE matter SET trust_revision=trust_revision+1, updated_at=? WHERE id=?",
                (now, self.matter_id),
            )
        except Exception:
            return 0
        return self.current_trust_revision()

    def _scoped_key(self, cache_key: str) -> str:
        """Prefix the caller's cache_key with the current trust
        revision so a revision bump invalidates every prior entry."""
        return f"tr{self.current_trust_revision()}:{cache_key}"

    def get(self, stage: str, cache_key: str) -> Optional[dict]:
        """Return cached plan dict or None on cache miss or DB error.

        Wraps all DB access in try/except so a corrupt or missing cache table
        never prevents the calling code (engine._orient) from falling through
        to the LLM call.

        P0.4: cache_key is prefixed with the current matter.trust_revision
        so a bump after an invalidation trigger silently misses every
        prior cache row without deleting anything.
        """
        import json
        scoped = self._scoped_key(cache_key)
        try:
            row = self.db.execute(
                "SELECT id, plan_json FROM reasoning_cache"
                " WHERE matter_id=? AND stage=? AND cache_key=?",
                (self.matter_id, stage, scoped),
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
        scoped = self._scoped_key(cache_key)
        try:
            now = _now()
            self.db.execute(
                """INSERT INTO reasoning_cache
                   (id, matter_id, stage, cache_key, plan_json, created_at, last_hit_at)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(matter_id, stage, cache_key)
                   DO UPDATE SET plan_json=excluded.plan_json,
                                 last_hit_at=excluded.last_hit_at""",
                (_id(), self.matter_id, stage, scoped, json.dumps(plan), now, now),
            )
        except Exception:
            pass  # non-critical; next run will populate from LLM

    def gc_stale_revisions(self, keep_last: int = 10) -> int:
        """OPT-7: delete reasoning_cache rows whose `tr{N}:` prefix
        encodes a trust_revision older than current - keep_last. Those
        rows are already UNREACHABLE — _scoped_key only queries under
        the current revision — so they are strictly dead space. Rows
        with no `tr` prefix (legacy rows from before P0.4) are left
        alone; a dedicated migration can retire them.

        Returns the number of rows deleted. Defaults keep 10 prior
        revisions so an operator can rollback-inspect recent plans.
        """
        current = self.current_trust_revision()
        min_keep = max(0, current - max(0, int(keep_last)))
        try:
            rows = self.db.execute(
                "SELECT id, cache_key FROM reasoning_cache WHERE matter_id=?",
                (self.matter_id,),
            ).fetchall()
        except Exception:
            return 0
        to_delete: list[str] = []
        for row in rows:
            key = row["cache_key"] or ""
            if not key.startswith("tr"):
                continue
            colon = key.find(":")
            if colon <= 2:
                continue
            try:
                rev = int(key[2:colon])
            except ValueError:
                continue
            if rev < min_keep:
                to_delete.append(row["id"])
        if not to_delete:
            return 0
        _BATCH = 900
        deleted = 0
        try:
            with self.db.transaction():
                for _i in range(0, len(to_delete), _BATCH):
                    batch = to_delete[_i : _i + _BATCH]
                    self.db.execute(
                        "DELETE FROM reasoning_cache WHERE id IN ({})".format(
                            ",".join("?" * len(batch))
                        ),
                        batch,
                    )
                    deleted += len(batch)
        except Exception:
            return 0
        return deleted


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
        """Remove a trust override. Returns True if a row was deleted.

        Normalizes backslashes to forward slashes before matching, consistent
        with set() which normalizes on insert.
        """
        document_pattern = document_pattern.replace("\\\\", "/").replace("\\", "/")
        cur = self.db.execute(
            "DELETE FROM document_trust_override WHERE matter_id=? AND document_pattern=?",
            (self.matter_id, document_pattern),
        )
        return cur.rowcount > 0


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
        provenance: "Optional[ProvenanceContext]" = None,
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
                # P0.4 review fix: revive stale authorities on
                # re-citation so the re-ingest of a doc that cites
                # the same case doesn't leave authority stale forever.
                VerificationStateStore(self.db, self.matter_id).touch_ai_target(
                    VerificationTargetKind.AUTHORITY,
                    auth_id,
                    cause="authority_reupsert",
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
            # MVP.2 + P0.4: touch_ai_target seeds on new insert and
            # revives stale on re-citation.
            VerificationStateStore(self.db, self.matter_id).touch_ai_target(
                VerificationTargetKind.AUTHORITY,
                auth_id,
                cause="authority_upsert",
            )
            # P0.1 provenance (new inserts only).
            if provenance is not None:
                ProvenanceStore(self.db, self.matter_id).record(
                    target_kind="authority",
                    target_id=auth_id,
                    context=provenance,
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
            + " ORDER BY precedential_rank DESC, weight DESC, citation ASC LIMIT ?",
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
               ORDER BY a.precedential_rank DESC, a.weight DESC, a.citation ASC""",
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


class EvidenceStore:
    """MVP.3: canonical evidence_edge substrate (SO-4).

    Proof edges connect a source (assertion, work_product, authority) to
    a target (issue, issue_predicate, other_assertion). MVP.3 writes
    assertion→issue edges synchronously when IssueStore.link_assertion
    fires, backfills existing assertion_issue_link rows at migration v53,
    and exposes a read path for ProofStateStore to prefer evidence_edge
    over the legacy link table.

    Occurrence/span identity is deliberately deferred: the current link
    APIs do not preserve it, so backfilled rows carry
    source_identity_status='missing_occurrence_span'.
    """

    _NATURAL_KEY_COLUMNS = (
        "matter_id", "source_kind", "source_id",
        "target_kind", "target_id", "relation_type",
    )

    def __init__(self, db: SQLiteMatterDB, matter_id: str):
        self.db = db
        self.matter_id = matter_id

    # ----- Read ---------------------------------------------------------

    def get(self, edge_id: str) -> Optional[dict]:
        row = self.db.execute(
            "SELECT * FROM evidence_edge WHERE id=? AND matter_id=?",
            (edge_id, self.matter_id),
        ).fetchone()
        return dict(row) if row else None

    def list_edges_for_target(
        self,
        target_kind: str,
        target_id: str,
    ) -> list[dict]:
        """Return active evidence edges whose target matches. Ordered by
        relation_type, then effective_weight DESC for deterministic
        consumer iteration."""
        rows = self.db.execute(
            """SELECT * FROM evidence_edge
               WHERE matter_id=? AND target_kind=? AND target_id=? AND active=1
               ORDER BY relation_type, effective_weight DESC, id""",
            (self.matter_id, target_kind, target_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def target_has_edges(self, target_kind: str, target_id: str) -> bool:
        row = self.db.execute(
            """SELECT 1 FROM evidence_edge
               WHERE matter_id=? AND target_kind=? AND target_id=? AND active=1
               LIMIT 1""",
            (self.matter_id, target_kind, target_id),
        ).fetchone()
        return row is not None

    # ----- Write --------------------------------------------------------

    def upsert_edge(
        self,
        *,
        source_kind: str,
        source_id: str,
        target_kind: str,
        target_id: str,
        relation_type: "EvidenceRelationType | str",
        proof_weight: float = 0.5,
        origin_kind: "EvidenceOriginKind | str" = EvidenceOriginKind.SYSTEM_INFERRED,
        backfill_source: Optional[str] = None,
        source_document_inventory_id: Optional[str] = None,
        source_span_id: Optional[str] = None,
        source_occurrence_id: Optional[str] = None,
        source_confidence: float = 1.0,
        independence_factor: float = 1.0,
        provenance: "Optional[ProvenanceContext]" = None,
    ) -> tuple[str, bool]:
        """Insert or return the existing edge matching the natural key.

        Returns (edge_id, is_new). When the edge already exists, origin_kind
        and backfill_source are preserved, verification_status is not
        downgraded, and proof_weight/effective_weight are refreshed only for
        unreviewed system-inferred edges.
        """
        rel_val = _evidence_relation_value(relation_type)
        origin_val = _evidence_origin_value(origin_kind)
        # MVP.3 issue-level writes do not carry occurrence/span identity.
        identity_status = (
            "present" if (source_occurrence_id or source_span_id)
            else "missing_occurrence_span"
        )
        now = _now()
        effective_weight = proof_weight

        existing = self.db.execute(
            """SELECT id, verification_status, origin_kind FROM evidence_edge
               WHERE matter_id=? AND source_kind=? AND source_id=?
                 AND target_kind=? AND target_id=? AND relation_type=?""",
            (self.matter_id, source_kind, source_id, target_kind, target_id, rel_val),
        ).fetchone()
        if existing is not None:
            edge_id = existing["id"]
            # Only refresh weights for unreviewed system-inferred edges.
            if (
                existing["verification_status"] == "candidate"
                and existing["origin_kind"] in ("system_inferred", "ai_extracted")
            ):
                self.db.execute(
                    """UPDATE evidence_edge
                       SET proof_weight=?, effective_weight=?, updated_at=?
                       WHERE id=?""",
                    (proof_weight, effective_weight, now, edge_id),
                )
            # P0.4 review fix: revive a stale edge to candidate when
            # the same assertion→issue relation is re-extracted.
            # Previously this path returned without touching
            # verification, leaving stale edges stale forever after
            # a re-ingest.
            VerificationStateStore(self.db, self.matter_id).touch_ai_target(
                VerificationTargetKind.EVIDENCE_EDGE,
                edge_id,
                cause="evidence_edge_reextraction",
            )
            return edge_id, False

        edge_id = _id()
        self.db.execute(
            """INSERT INTO evidence_edge
                (id, matter_id,
                 source_kind, source_id,
                 source_document_inventory_id, source_span_id, source_occurrence_id,
                 target_kind, target_id, relation_type,
                 proof_weight, source_confidence, admissibility_status,
                 vulnerability_json, note,
                 verification_status, independence_factor, backfill_source,
                 source_identity_status, origin_kind, active,
                 effective_weight, independence_cluster_id,
                 created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL,
                       'candidate', ?, ?, ?, ?, 1, ?, NULL, ?, ?)""",
            (
                edge_id, self.matter_id,
                source_kind, source_id,
                source_document_inventory_id, source_span_id, source_occurrence_id,
                target_kind, target_id, rel_val,
                proof_weight, source_confidence,
                independence_factor, backfill_source, identity_status, origin_val,
                effective_weight, now, now,
            ),
        )
        # MVP.2 + P0.4: seed candidate on first write, revive stale on
        # re-extraction via touch_ai_target.
        VerificationStateStore(self.db, self.matter_id).touch_ai_target(
            VerificationTargetKind.EVIDENCE_EDGE,
            edge_id,
            cause="evidence_edge_upsert",
        )
        # P0.1: record provenance if the caller supplied context.
        if provenance is not None:
            ProvenanceStore(self.db, self.matter_id).record(
                target_kind="evidence_edge",
                target_id=edge_id,
                context=provenance,
            )
        return edge_id, True

    def backfill_from_legacy_links(self) -> int:
        """Idempotent backfill helper. Mirrors _migration_v53 for this
        matter only and returns the number of newly inserted edges.

        Used by tests and by any future runtime re-sync path. Idempotence
        is guaranteed by the natural-key unique index on evidence_edge.
        """
        before = self.db.execute(
            "SELECT COUNT(*) FROM evidence_edge WHERE matter_id=?",
            (self.matter_id,),
        ).fetchone()[0]
        self.db.execute(
            """INSERT OR IGNORE INTO evidence_edge
                (id, matter_id,
                 source_kind, source_id,
                 source_document_inventory_id, source_span_id, source_occurrence_id,
                 target_kind, target_id, relation_type,
                 proof_weight, source_confidence, admissibility_status,
                 vulnerability_json, note,
                 verification_status, independence_factor, backfill_source,
                 source_identity_status, origin_kind, active,
                 effective_weight, independence_cluster_id,
                 created_at, updated_at)
               SELECT
                   lower(hex(randomblob(16))), i.matter_id,
                   'assertion', ail.assertion_id,
                   NULL, NULL, NULL,
                   'issue', ail.issue_id, ail.relation_type,
                   0.5, COALESCE(a.confidence, 0.5), NULL,
                   NULL,
                   'Backfilled from assertion_issue_link; occurrence/span identity unavailable in legacy link.',
                   'candidate', 1.0, 'assertion_issue_link',
                   'missing_occurrence_span', 'legacy_backfill', 1,
                   0.5, NULL,
                   COALESCE(ail.created_at, datetime('now')), datetime('now')
               FROM assertion_issue_link ail
               JOIN assertion a ON a.id = ail.assertion_id
               JOIN issue i ON i.id = ail.issue_id
               WHERE i.matter_id=?
                 AND ail.relation_type IN ('supports','establishes','attacks','negates')""",
            (self.matter_id,),
        )
        # Seed verification_state candidate rows for all edges in this matter.
        self.db.execute(
            """INSERT OR IGNORE INTO verification_state
                (id, matter_id, target_kind, target_id, status,
                 review_scope, version, created_at, updated_at)
               SELECT lower(hex(randomblob(16))), matter_id, 'evidence_edge', id,
                      'candidate', 'inference', 1, datetime('now'), datetime('now')
               FROM evidence_edge WHERE matter_id=?""",
            (self.matter_id,),
        )
        after = self.db.execute(
            "SELECT COUNT(*) FROM evidence_edge WHERE matter_id=?",
            (self.matter_id,),
        ).fetchone()[0]
        return int(after) - int(before)


def _evidence_relation_value(rel: "EvidenceRelationType | str") -> str:
    return rel.value if isinstance(rel, EvidenceRelationType) else str(rel)


def _evidence_origin_value(origin: "EvidenceOriginKind | str") -> str:
    return origin.value if isinstance(origin, EvidenceOriginKind) else str(origin)


class ProvenanceStore:
    """P0.1 append-only provenance event store (SO-2).

    Writers do not call this directly. Instead they receive an
    Optional[ProvenanceContext], and when present, call
    ProvenanceStore.record(target_kind, target_id, context) to append
    an audit row. The table is query-only via list_for_target.
    """

    def __init__(self, db: SQLiteMatterDB, matter_id: str):
        self.db = db
        self.matter_id = matter_id

    def record(
        self,
        *,
        target_kind: str,
        target_id: str,
        context: "ProvenanceContext",
    ) -> str:
        """Append one provenance_event row. Returns event id."""
        event_id = _id()
        self.db.execute(
            """INSERT INTO provenance_event
                (id, matter_id, target_kind, target_id, event_kind,
                 writer_name, run_id, model_id, model_tier,
                 prompt_version, extractor_version, llm_call_id,
                 prompt_hash, response_hash,
                 source_document_ref, source_document_inventory_id,
                 source_span_id, source_span_status, note, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                event_id, self.matter_id, target_kind, target_id,
                context.event_kind, context.writer_name,
                context.run_id, context.model_id, context.model_tier,
                context.prompt_version, context.extractor_version,
                context.llm_call_id,
                context.prompt_hash, context.response_hash,
                context.source_document_ref,
                context.source_document_inventory_id,
                context.source_span_id, context.source_span_status,
                context.note, _now(),
            ),
        )
        return event_id

    def list_for_target(
        self, target_kind: str, target_id: str, limit: int = 50,
    ) -> list[dict]:
        rows = self.db.execute(
            """SELECT * FROM provenance_event
               WHERE matter_id=? AND target_kind=? AND target_id=?
               ORDER BY created_at DESC LIMIT ?""",
            (self.matter_id, target_kind, target_id, int(limit)),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_for_llm_call(self, llm_call_id: str) -> list[dict]:
        rows = self.db.execute(
            """SELECT * FROM provenance_event
               WHERE matter_id=? AND llm_call_id=?
               ORDER BY created_at DESC""",
            (self.matter_id, llm_call_id),
        ).fetchall()
        return [dict(r) for r in rows]


class VerificationStateStore:
    """MVP.2: canonical verification state for AI-derived intelligence (SO-2).

    Tracks whether a human has reviewed an AI-derived object. Candidate is
    the default for every AI extraction path. Only 'user' and 'attorney'
    reviewers may promote to 'verified'; 'system' and 'import' are
    automation markers that raise ValueError if passed to verify().

    Verification is independent of belief_state: a verified assertion can
    still be disputed, and a candidate assertion can still be useful as a
    lead. Rejected targets are excluded from proof and clean synthesis by
    downstream consumers (PR/MVP downstream of MVP.2).
    """

    # Human reviewers who may promote to verified.
    _HUMAN_REVIEWERS: frozenset[str] = frozenset({"user", "attorney"})

    def __init__(self, db: SQLiteMatterDB, matter_id: str):
        self.db = db
        self.matter_id = matter_id

    # ----- Read API -----------------------------------------------------

    def get(
        self,
        target_kind: "VerificationTargetKind | str",
        target_id: str,
    ) -> Optional[dict]:
        """Return the verification_state row for a target, or None if none exists."""
        kind = _verification_kind_value(target_kind)
        row = self.db.execute(
            """SELECT * FROM verification_state
               WHERE matter_id=? AND target_kind=? AND target_id=?""",
            (self.matter_id, kind, target_id),
        ).fetchone()
        return dict(row) if row else None

    def list_by_status(
        self,
        status: "VerificationStatus | str",
        target_kind: "VerificationTargetKind | str | None" = None,
    ) -> list[dict]:
        """Return verification rows filtered by status and optional target_kind.

        Ordered by updated_at DESC so the review queue surfaces the newest
        candidates first.
        """
        status_val = _verification_status_value(status)
        if target_kind is None:
            rows = self.db.execute(
                """SELECT * FROM verification_state
                   WHERE matter_id=? AND status=?
                   ORDER BY updated_at DESC""",
                (self.matter_id, status_val),
            ).fetchall()
        else:
            rows = self.db.execute(
                """SELECT * FROM verification_state
                   WHERE matter_id=? AND status=? AND target_kind=?
                   ORDER BY updated_at DESC""",
                (self.matter_id, status_val, _verification_kind_value(target_kind)),
            ).fetchall()
        return [dict(r) for r in rows]

    def review_queue(
        self,
        limit: int = 50,
        offset: int = 0,
        target_kind: "VerificationTargetKind | str | None" = None,
    ) -> list[dict]:
        """P0.3: prioritized review queue of candidate verification
        targets (SO-3). Attorneys pull this to decide what to promote
        or reject next.

        Priority order:
          1. Assertions + edges linked to open issues with a proof
             gap, ordered by materiality × salience descending.
          2. Assertions + edges linked to open issues without a
             proof gap, ordered by materiality × salience descending.
          3. Quant facts flagged as threshold-triggering (high
             materiality numeric intelligence).
          4. Authorities (cited law) — fewer gates but still
             reviewable.
          5. Other candidate targets (predicates, document cards,
             etc.), ordered by recency.

        Each row includes: verification_state fields, target_kind,
        target_id, plus purpose-dependent context (issue title /
        materiality for assertions + edges, raw_text for quants,
        citation for authorities, proposition_text for assertions).
        """
        kind_filter_sql = ""
        params: list = [self.matter_id]
        if target_kind is not None:
            kind_filter_sql = " AND vs.target_kind = ?"
            params.append(_verification_kind_value(target_kind))
        # Priority uses a CASE on target_kind and joins for the
        # strongest signal per kind. Buckets (codex P0.3 review fix):
        #   0 = issue-linked assertion/edge with a proof gap
        #   1 = contradicted assertion (attacks/contradicts link)
        #   2 = issue-linked assertion/edge without a gap
        #   3 = issue_predicate (candidate predicate, proof-critical)
        #   4 = quant_fact
        #   5 = authority
        #   6 = everything else
        rows = self.db.execute(
            f"""WITH assertion_issue AS (
                   SELECT ee.source_id AS target_id,
                          MAX(i.materiality * i.salience) AS max_priority,
                          MAX(CASE WHEN g.id IS NOT NULL THEN 1 ELSE 0 END) AS has_gap
                   FROM evidence_edge ee
                   JOIN issue i ON i.id = ee.target_id
                   LEFT JOIN gap_link gl ON gl.affected_type='issue' AND gl.affected_id=ee.target_id
                   LEFT JOIN gap g ON g.id = gl.gap_id AND g.status='open'
                                   AND g.gap_type='missing_issue_predicate'
                   WHERE ee.matter_id=? AND ee.source_kind='assertion'
                     AND ee.target_kind='issue' AND ee.active=1
                     AND i.status='open'
                     AND ee.relation_type IN ('supports','establishes','attacks','negates')
                   GROUP BY ee.source_id
               ),
               edge_issue AS (
                   SELECT ee.id AS target_id,
                          i.materiality * i.salience AS priority,
                          CASE WHEN g.id IS NOT NULL THEN 1 ELSE 0 END AS has_gap
                   FROM evidence_edge ee
                   JOIN issue i ON i.id = ee.target_id
                   LEFT JOIN gap_link gl ON gl.affected_type='issue' AND gl.affected_id=ee.target_id
                   LEFT JOIN gap g ON g.id = gl.gap_id AND g.status='open'
                                   AND g.gap_type='missing_issue_predicate'
                   WHERE ee.matter_id=? AND ee.target_kind='issue' AND ee.active=1
                     AND i.status='open'
                     AND ee.relation_type IN ('supports','establishes','attacks','negates')
               ),
               contradicted AS (
                   -- Codex P0.3 review fix #1: flag assertions that
                   -- appear on either side of an attacks/contradicts
                   -- link so reviewers see live conflicts near the
                   -- top of the queue, not buried at bucket 6.
                   SELECT DISTINCT a.id AS target_id
                   FROM assertion a
                   JOIN assertion_link al
                     ON (al.src_assertion_id=a.id OR al.dst_assertion_id=a.id)
                    AND al.link_type IN ('attacks','contradicts')
                   WHERE a.matter_id=?
               ),
               predicate_issue AS (
                   -- Codex P0.3 review fix #1: issue_predicate is a
                   -- first-class review target. Surface the parent
                   -- issue's materiality × salience so proof-critical
                   -- predicates rank above generic predicates.
                   SELECT ip.id AS target_id,
                          i.materiality * i.salience AS priority
                   FROM issue_predicate ip
                   JOIN issue i ON i.id=ip.issue_id
                   WHERE i.matter_id=? AND i.status='open'
                     AND ip.status='open'
               )
               SELECT vs.id AS verification_id, vs.status, vs.target_kind, vs.target_id,
                      vs.ai_confidence, vs.created_at, vs.updated_at,
                      vs.reviewed_by_kind, vs.reviewed_by_id, vs.reviewed_at,
                      CASE
                          WHEN vs.target_kind='assertion' AND ai.has_gap=1 THEN 0
                          WHEN vs.target_kind='evidence_edge' AND ei.has_gap=1 THEN 0
                          WHEN vs.target_kind='assertion' AND ct.target_id IS NOT NULL THEN 1
                          WHEN vs.target_kind='assertion' AND ai.max_priority IS NOT NULL THEN 2
                          WHEN vs.target_kind='evidence_edge' AND ei.priority IS NOT NULL THEN 2
                          WHEN vs.target_kind='issue_predicate' THEN 3
                          WHEN vs.target_kind='quant_fact' THEN 4
                          WHEN vs.target_kind='authority' THEN 5
                          ELSE 6
                      END AS priority_bucket,
                      COALESCE(ai.max_priority, ei.priority, pi.priority, 0.0) AS priority_score,
                      -- Target context columns: at most one of these
                      -- is non-null per row thanks to the target_kind
                      -- guards on each join. Replaces four correlated
                      -- subqueries with four indexed LEFT JOINs on
                      -- primary keys (OPT-4).
                      a.proposition_text  AS proposition_text,
                      q.raw_text          AS quant_raw_text,
                      au.citation         AS authority_citation,
                      ipd.description     AS predicate_description
               FROM verification_state vs
               LEFT JOIN assertion_issue ai ON ai.target_id = vs.target_id
                                            AND vs.target_kind='assertion'
               LEFT JOIN edge_issue ei ON ei.target_id = vs.target_id
                                       AND vs.target_kind='evidence_edge'
               LEFT JOIN contradicted ct ON ct.target_id = vs.target_id
                                         AND vs.target_kind='assertion'
               LEFT JOIN predicate_issue pi ON pi.target_id = vs.target_id
                                            AND vs.target_kind='issue_predicate'
               LEFT JOIN assertion a       ON vs.target_kind='assertion'
                                           AND a.id = vs.target_id
               LEFT JOIN quant_fact q      ON vs.target_kind='quant_fact'
                                           AND q.id = vs.target_id
               LEFT JOIN authority au      ON vs.target_kind='authority'
                                           AND au.id = vs.target_id
               LEFT JOIN issue_predicate ipd ON vs.target_kind='issue_predicate'
                                             AND ipd.id = vs.target_id
               WHERE vs.matter_id=? AND vs.status='candidate'
                 {kind_filter_sql}
               ORDER BY priority_bucket ASC, priority_score DESC, vs.created_at DESC
               LIMIT ? OFFSET ?""",
            (
                self.matter_id, self.matter_id,  # assertion_issue, edge_issue
                self.matter_id,                   # contradicted
                self.matter_id,                   # predicate_issue
                *params, int(limit), int(offset),
            ),
        ).fetchall()
        return [dict(r) for r in rows]

    def count_review_queue(self) -> dict:
        """Lightweight count for the badge and for verify/reject toast
        sizing. Returns {'total': int, 'by_bucket': {bucket: count}}.

        The full `review_queue` runs ~4 correlated subqueries per row for
        display text (proposition_text, quant_raw_text, etc.). The badge
        and the pre/post verify snapshots only need integers, so this
        variant drops the display subqueries and the ORDER BY, keeping
        only the CTEs that drive the priority_bucket CASE. One query,
        no per-row work — a ~10× cut on 500-row queues.
        """
        rows = self.db.execute(
            """WITH assertion_issue AS (
                   SELECT ee.source_id AS target_id,
                          MAX(CASE WHEN g.id IS NOT NULL THEN 1 ELSE 0 END) AS has_gap,
                          MAX(i.materiality * i.salience) AS max_priority
                   FROM evidence_edge ee
                   JOIN issue i ON i.id = ee.target_id
                   LEFT JOIN gap_link gl ON gl.affected_type='issue' AND gl.affected_id=ee.target_id
                   LEFT JOIN gap g ON g.id = gl.gap_id AND g.status='open'
                                   AND g.gap_type='missing_issue_predicate'
                   WHERE ee.matter_id=? AND ee.source_kind='assertion'
                     AND ee.target_kind='issue' AND ee.active=1
                     AND i.status='open'
                     AND ee.relation_type IN ('supports','establishes','attacks','negates')
                   GROUP BY ee.source_id
               ),
               edge_issue AS (
                   SELECT ee.id AS target_id,
                          CASE WHEN g.id IS NOT NULL THEN 1 ELSE 0 END AS has_gap,
                          i.materiality * i.salience AS priority
                   FROM evidence_edge ee
                   JOIN issue i ON i.id = ee.target_id
                   LEFT JOIN gap_link gl ON gl.affected_type='issue' AND gl.affected_id=ee.target_id
                   LEFT JOIN gap g ON g.id = gl.gap_id AND g.status='open'
                                   AND g.gap_type='missing_issue_predicate'
                   WHERE ee.matter_id=? AND ee.target_kind='issue' AND ee.active=1
                     AND i.status='open'
                     AND ee.relation_type IN ('supports','establishes','attacks','negates')
               ),
               contradicted AS (
                   SELECT DISTINCT a.id AS target_id
                   FROM assertion a
                   JOIN assertion_link al
                     ON (al.src_assertion_id=a.id OR al.dst_assertion_id=a.id)
                    AND al.link_type IN ('attacks','contradicts')
                   WHERE a.matter_id=?
               )
               SELECT
                   CASE
                       WHEN vs.target_kind='assertion' AND ai.has_gap=1 THEN 0
                       WHEN vs.target_kind='evidence_edge' AND ei.has_gap=1 THEN 0
                       WHEN vs.target_kind='assertion' AND ct.target_id IS NOT NULL THEN 1
                       WHEN vs.target_kind='assertion' AND ai.max_priority IS NOT NULL THEN 2
                       WHEN vs.target_kind='evidence_edge' AND ei.priority IS NOT NULL THEN 2
                       WHEN vs.target_kind='issue_predicate' THEN 3
                       WHEN vs.target_kind='quant_fact' THEN 4
                       WHEN vs.target_kind='authority' THEN 5
                       ELSE 6
                   END AS priority_bucket,
                   COUNT(*) AS n
               FROM verification_state vs
               LEFT JOIN assertion_issue ai ON ai.target_id = vs.target_id
                                            AND vs.target_kind='assertion'
               LEFT JOIN edge_issue ei ON ei.target_id = vs.target_id
                                       AND vs.target_kind='evidence_edge'
               LEFT JOIN contradicted ct ON ct.target_id = vs.target_id
                                         AND vs.target_kind='assertion'
               WHERE vs.matter_id=? AND vs.status='candidate'
               GROUP BY priority_bucket""",
            (
                self.matter_id,  # assertion_issue
                self.matter_id,  # edge_issue
                self.matter_id,  # contradicted
                self.matter_id,  # outer
            ),
        ).fetchall()
        by_bucket = {int(r["priority_bucket"]): int(r["n"]) for r in rows}
        return {"total": sum(by_bucket.values()), "by_bucket": by_bucket}

    def bulk_set_status(
        self,
        specs: list[dict],
        *,
        new_status: "VerificationStatus | str",
        reviewed_by_kind: "ReviewedByKind | str",
        reviewed_by_id: Optional[str] = None,
        review_scope: "ReviewScope | str" = ReviewScope.EXTRACTION_CORRECT,
        review_note: Optional[str] = None,
        rejection_reason: Optional[str] = None,
        cause: str = "human_bulk_review",
        run_id: Optional[str] = None,
    ) -> list[str]:
        """P0.3: transition multiple targets to the same status in a
        single logical operation (SO-3). Each spec is {target_kind,
        target_id}. verify/reject rules still apply — automation
        cannot bulk-verify or bulk-reject.

        Returns the list of verification_state ids touched.
        """
        status_val = _verification_status_value(new_status)
        kind_val = _reviewed_by_value(reviewed_by_kind)
        if status_val in {"verified", "rejected"} and kind_val not in self._HUMAN_REVIEWERS:
            raise ValueError(
                f"reviewed_by_kind {kind_val!r} cannot bulk-set status={status_val}; "
                "only 'user' or 'attorney' may verify or reject"
            )
        if status_val == "rejected" and (not rejection_reason or not rejection_reason.strip()):
            raise ValueError("rejection_reason required for bulk rejection")
        ids: list[str] = []
        for spec in specs:
            vid = self._set_status(
                target_kind=spec["target_kind"],
                target_id=spec["target_id"],
                new_status=status_val,
                reviewed_by_kind=kind_val,
                reviewed_by_id=reviewed_by_id,
                review_scope=_review_scope_value(review_scope),
                review_scope_json=None,
                review_note=review_note,
                rejection_reason=(
                    rejection_reason.strip() if rejection_reason else None
                ),
                stale_reason=None,
                ai_confidence=None,
                cause=cause,
                run_id=run_id,
            )
            ids.append(vid)
        return ids

    def list_events(
        self,
        target_kind: "VerificationTargetKind | str | None" = None,
        target_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """Return verification_event rows for audit surfaces.

        Unfiltered call returns the newest N events across the matter.
        target_kind + target_id narrow to one object's history.
        """
        if target_kind is not None and target_id is not None:
            rows = self.db.execute(
                """SELECT * FROM verification_event
                   WHERE matter_id=? AND target_kind=? AND target_id=?
                   ORDER BY created_at DESC LIMIT ?""",
                (self.matter_id, _verification_kind_value(target_kind), target_id, int(limit)),
            ).fetchall()
        else:
            rows = self.db.execute(
                """SELECT * FROM verification_event
                   WHERE matter_id=? ORDER BY created_at DESC LIMIT ?""",
                (self.matter_id, int(limit)),
            ).fetchall()
        return [dict(r) for r in rows]

    # ----- Write API ----------------------------------------------------

    def candidate(
        self,
        target_kind: "VerificationTargetKind | str",
        target_id: str,
        *,
        ai_confidence: Optional[float] = None,
        cause: str = "ai_extraction",
        run_id: Optional[str] = None,
    ) -> str:
        """Idempotently ensure a candidate row exists for a target.

        Never downgrades an already-verified or rejected row. Returns the
        verification_state.id.
        """
        existing = self.get(target_kind, target_id)
        if existing is not None:
            return existing["id"]
        kind = _verification_kind_value(target_kind)
        vid = uuid.uuid4().hex
        now = _now()
        self.db.execute(
            """INSERT INTO verification_state
                (id, matter_id, target_kind, target_id, status, ai_confidence,
                 review_scope, version, created_at, updated_at)
               VALUES (?, ?, ?, ?, 'candidate', ?, 'extraction_correct', 1, ?, ?)""",
            (vid, self.matter_id, kind, target_id, ai_confidence, now, now),
        )
        self._append_event(
            verification_id=vid,
            target_kind=kind,
            target_id=target_id,
            old_status=None,
            new_status="candidate",
            reviewed_by_kind="system",
            reviewed_by_id=None,
            review_scope="extraction_correct",
            rejection_reason=None,
            run_id=run_id,
            cause=cause,
            note=None,
            old_version=None,
            new_version=1,
        )
        return vid

    def verify(
        self,
        target_kind: "VerificationTargetKind | str",
        target_id: str,
        *,
        reviewed_by_kind: "ReviewedByKind | str",
        reviewed_by_id: Optional[str] = None,
        review_scope: "ReviewScope | str" = ReviewScope.EXTRACTION_CORRECT,
        review_scope_json: Optional[str] = None,
        review_note: Optional[str] = None,
        ai_confidence: Optional[float] = None,
        cause: str = "human_verification",
        run_id: Optional[str] = None,
    ) -> str:
        """Promote a target to verified. Raises ValueError if the reviewer
        is automation (system/import) — only humans can verify.
        """
        kind_val = _reviewed_by_value(reviewed_by_kind)
        if kind_val not in self._HUMAN_REVIEWERS:
            raise ValueError(
                f"reviewed_by_kind {kind_val!r} cannot set status=verified; "
                "only 'user' or 'attorney' may promote to verified"
            )
        return self._set_status(
            target_kind=target_kind,
            target_id=target_id,
            new_status="verified",
            reviewed_by_kind=kind_val,
            reviewed_by_id=reviewed_by_id,
            review_scope=_review_scope_value(review_scope),
            review_scope_json=review_scope_json,
            review_note=review_note,
            rejection_reason=None,
            stale_reason=None,
            ai_confidence=ai_confidence,
            cause=cause,
            run_id=run_id,
        )

    def reject(
        self,
        target_kind: "VerificationTargetKind | str",
        target_id: str,
        *,
        reviewed_by_kind: "ReviewedByKind | str",
        rejection_reason: str,
        reviewed_by_id: Optional[str] = None,
        review_scope: "ReviewScope | str" = ReviewScope.EXTRACTION_CORRECT,
        review_note: Optional[str] = None,
        cause: str = "human_rejection",
        run_id: Optional[str] = None,
    ) -> str:
        """Promote a target to rejected. Rejection also requires a human
        reviewer — automation cannot implicitly reject live intelligence.
        """
        kind_val = _reviewed_by_value(reviewed_by_kind)
        if kind_val not in self._HUMAN_REVIEWERS:
            raise ValueError(
                f"reviewed_by_kind {kind_val!r} cannot set status=rejected; "
                "only 'user' or 'attorney' may reject"
            )
        if not rejection_reason or not rejection_reason.strip():
            raise ValueError("rejection_reason is required when rejecting a target")
        return self._set_status(
            target_kind=target_kind,
            target_id=target_id,
            new_status="rejected",
            reviewed_by_kind=kind_val,
            reviewed_by_id=reviewed_by_id,
            review_scope=_review_scope_value(review_scope),
            review_scope_json=None,
            review_note=review_note,
            rejection_reason=rejection_reason.strip(),
            stale_reason=None,
            ai_confidence=None,
            cause=cause,
            run_id=run_id,
        )

    def mark_stale(
        self,
        target_kind: "VerificationTargetKind | str",
        target_id: str,
        *,
        stale_reason: str,
        cause: str = "trust_invalidation",
        run_id: Optional[str] = None,
    ) -> Optional[str]:
        """P0.4: mark a target stale because upstream support changed
        (document hash flip, span replacement, privilege flip). Never
        downgrades a rejected row — rejection is a stronger human
        opinion than stale. Verified and candidate rows move to stale.

        Returns the verification_state id touched, or None if the row
        was rejected and left alone.
        """
        kind = _verification_kind_value(target_kind)
        existing = self.get(kind, target_id)
        if existing is not None and existing["status"] == "rejected":
            # Rejection outranks stale — a human has already removed
            # this target from all consumers.
            return None
        # Use review_note to carry the stale_reason since the generic
        # set_status path doesn't pass stale_reason through otherwise.
        return self._set_status(
            target_kind=kind,
            target_id=target_id,
            new_status="stale",
            reviewed_by_kind="system",
            reviewed_by_id=None,
            review_scope="extraction_correct",
            review_scope_json=None,
            review_note=stale_reason,
            rejection_reason=None,
            stale_reason=stale_reason,
            ai_confidence=None,
            cause=cause,
            run_id=run_id,
        )

    def bulk_mark_stale(
        self,
        specs: list[dict],
        *,
        stale_reason: str,
        cause: str = "trust_invalidation",
        run_id: Optional[str] = None,
    ) -> list[str]:
        """P0.4: mark multiple targets stale in one sweep. Each spec
        is {target_kind, target_id}. Skips rejected rows (never
        downgraded). Returns the list of ids actually touched."""
        touched: list[str] = []
        for spec in specs:
            vid = self.mark_stale(
                spec["target_kind"], spec["target_id"],
                stale_reason=stale_reason, cause=cause, run_id=run_id,
            )
            if vid:
                touched.append(vid)
        return touched

    def touch_ai_target(
        self,
        target_kind: "VerificationTargetKind | str",
        target_id: str,
        *,
        ai_confidence: Optional[float] = None,
        cause: str = "ai_rewrite",
        run_id: Optional[str] = None,
    ) -> str:
        """P0.4: called from every AI writer path so a fresh write
        revives a stale target back to candidate. Rules:
          - missing row → seed as candidate
          - stale row → promote back to candidate
          - candidate/verified/rejected → leave unchanged

        This closes the "stale forever" trap that would otherwise
        occur after a document is re-ingested: mark_document_stale
        marks downstream targets stale, then the next extraction pass
        rewrites the same targets, and touch_ai_target revives them.
        """
        kind = _verification_kind_value(target_kind)
        existing = self.get(kind, target_id)
        if existing is None:
            return self.candidate(
                kind, target_id, ai_confidence=ai_confidence,
                cause=cause, run_id=run_id,
            )
        if existing["status"] == "stale":
            return self._set_status(
                target_kind=kind,
                target_id=target_id,
                new_status="candidate",
                reviewed_by_kind="system",
                reviewed_by_id=None,
                review_scope="extraction_correct",
                review_scope_json=None,
                review_note="ai_rewrite_revival",
                rejection_reason=None,
                stale_reason=None,
                ai_confidence=ai_confidence,
                cause=cause,
                run_id=run_id,
            )
        return existing["id"]

    def set_status(
        self,
        target_kind: "VerificationTargetKind | str",
        target_id: str,
        status: "VerificationStatus | str",
        *,
        reviewed_by_kind: "ReviewedByKind | str",
        reviewed_by_id: Optional[str] = None,
        review_scope: "ReviewScope | str" = ReviewScope.EXTRACTION_CORRECT,
        review_scope_json: Optional[str] = None,
        review_note: Optional[str] = None,
        rejection_reason: Optional[str] = None,
        stale_reason: Optional[str] = None,
        ai_confidence: Optional[float] = None,
        cause: str = "manual_set_status",
        run_id: Optional[str] = None,
    ) -> str:
        """Generic status setter. verify() and reject() are the preferred APIs;
        set_status exists for stale transitions that automation may emit.
        """
        status_val = _verification_status_value(status)
        kind_val = _reviewed_by_value(reviewed_by_kind)
        if status_val in {"verified", "rejected"} and kind_val not in self._HUMAN_REVIEWERS:
            raise ValueError(
                f"reviewed_by_kind {kind_val!r} cannot set status={status_val}; "
                "only 'user' or 'attorney' may verify or reject"
            )
        return self._set_status(
            target_kind=target_kind,
            target_id=target_id,
            new_status=status_val,
            reviewed_by_kind=kind_val,
            reviewed_by_id=reviewed_by_id,
            review_scope=_review_scope_value(review_scope),
            review_scope_json=review_scope_json,
            review_note=review_note,
            rejection_reason=rejection_reason,
            stale_reason=stale_reason,
            ai_confidence=ai_confidence,
            cause=cause,
            run_id=run_id,
        )

    # ----- Internals ----------------------------------------------------

    def _set_status(
        self,
        *,
        target_kind: "VerificationTargetKind | str",
        target_id: str,
        new_status: str,
        reviewed_by_kind: str,
        reviewed_by_id: Optional[str],
        review_scope: str,
        review_scope_json: Optional[str],
        review_note: Optional[str],
        rejection_reason: Optional[str],
        stale_reason: Optional[str],
        ai_confidence: Optional[float],
        cause: str,
        run_id: Optional[str],
    ) -> str:
        """Upsert-and-transition. Ensures the row exists as candidate if it
        doesn't yet, then transitions to new_status and appends the event.
        """
        kind = _verification_kind_value(target_kind)
        existing = self.get(kind, target_id)
        if existing is None:
            # Seed as candidate before applying the intended transition.
            self.candidate(kind, target_id, ai_confidence=ai_confidence, run_id=run_id)
            existing = self.get(kind, target_id)

        vid = existing["id"]
        old_status = existing["status"]
        old_version = int(existing["version"])
        new_version = old_version + 1
        now = _now()

        # Wrap the three writes (verification_state update, optional
        # evidence_edge mirror, verification_event append) in one
        # transaction so a mid-sequence failure cannot leave split state.
        with self.db.transaction():
            self.db.execute(
                """UPDATE verification_state
                   SET status=?, reviewed_by_kind=?, reviewed_by_id=?, reviewed_at=?,
                       review_scope=?, review_scope_json=?, review_note=?,
                       rejection_reason=?, stale_reason=?, ai_confidence=COALESCE(?, ai_confidence),
                       version=?, updated_at=?
                   WHERE id=?""",
                (
                    new_status, reviewed_by_kind, reviewed_by_id, now,
                    review_scope, review_scope_json, review_note,
                    rejection_reason, stale_reason, ai_confidence,
                    new_version, now, vid,
                ),
            )
            # MVP.3: mirror the canonical verification_state to
            # evidence_edge.verification_status when the target is an edge.
            if kind == VerificationTargetKind.EVIDENCE_EDGE.value:
                self.db.execute(
                    "UPDATE evidence_edge SET verification_status=?, updated_at=? WHERE id=? AND matter_id=?",
                    (new_status, now, target_id, self.matter_id),
                )
            self._append_event(
                verification_id=vid,
                target_kind=kind,
                target_id=target_id,
                old_status=old_status,
                new_status=new_status,
                reviewed_by_kind=reviewed_by_kind,
                reviewed_by_id=reviewed_by_id,
                review_scope=review_scope,
                rejection_reason=rejection_reason,
                run_id=run_id,
                cause=cause,
                note=review_note,
                old_version=old_version,
                new_version=new_version,
            )
        return vid

    def _append_event(
        self,
        *,
        verification_id: str,
        target_kind: str,
        target_id: str,
        old_status: Optional[str],
        new_status: str,
        reviewed_by_kind: str,
        reviewed_by_id: Optional[str],
        review_scope: str,
        rejection_reason: Optional[str],
        run_id: Optional[str],
        cause: str,
        note: Optional[str],
        old_version: Optional[int],
        new_version: int,
    ) -> None:
        self.db.execute(
            """INSERT INTO verification_event
                (id, matter_id, verification_id, target_kind, target_id,
                 old_status, new_status, reviewed_by_kind, reviewed_by_id,
                 review_scope, rejection_reason, run_id, cause, note,
                 old_version, new_version, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                uuid.uuid4().hex, self.matter_id, verification_id, target_kind, target_id,
                old_status, new_status, reviewed_by_kind, reviewed_by_id,
                review_scope, rejection_reason, run_id, cause, note,
                old_version, new_version, _now(),
            ),
        )


def _verification_kind_value(kind: "VerificationTargetKind | str") -> str:
    return kind.value if isinstance(kind, VerificationTargetKind) else str(kind)


def _verification_status_value(status: "VerificationStatus | str") -> str:
    return status.value if isinstance(status, VerificationStatus) else str(status)


def _review_scope_value(scope: "ReviewScope | str") -> str:
    return scope.value if isinstance(scope, ReviewScope) else str(scope)


def _reviewed_by_value(kind: "ReviewedByKind | str") -> str:
    return kind.value if isinstance(kind, ReviewedByKind) else str(kind)


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
    # Substrate selection
    # ------------------------------------------------------------------

    def _query_issue_linked_assertions(
        self, issue_id: str, use_edges: bool, policy_audience: str = "clean"
    ) -> list:
        """Return one row per assertion linked to the issue, carrying
        relation_type, best source_role, and primary document id.

        use_edges=True reads from evidence_edge (MVP.3 canonical substrate).
        use_edges=False falls back to assertion_issue_link so matters
        mid-migration or with no edges yet still compute correctly.

        policy_audience="clean" excludes assertions sourced from a
        privileged document (MVP.4). "internal" lifts the filter.

        The two branches return rows with the same shape so downstream
        proof math is identical. Rejected assertions and rejected edges
        are excluded at the substrate level.
        """
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
        # Shared occurrence-ranking CTE: one row per assertion with the
        # highest-trust source_role and the primary document_id.
        occ_ranked_cte = """
            WITH occ_ranked AS (
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
                WHERE ao.assertion_id IN ({inner_ids})
            )
        """
        if use_edges:
            inner_ids_sql = (
                "SELECT source_id FROM evidence_edge "
                "WHERE matter_id=? AND target_kind='issue' AND target_id=? "
                "AND active=1 AND source_kind='assertion' "
                "AND relation_type IN ('supports','establishes','attacks','negates')"
            )
            query = occ_ranked_cte.format(inner_ids=inner_ids_sql) + """
                SELECT ee.relation_type,
                       COALESCE(MAX(CASE WHEN o.role_rn = 1 THEN o.source_role END), 'unknown')
                           AS source_role,
                       MAX(CASE WHEN o.doc_rn = 1 THEN o.document_id END)
                           AS primary_doc_id
                FROM evidence_edge ee
                JOIN assertion a ON a.id = ee.source_id
                LEFT JOIN occ_ranked o ON o.assertion_id = ee.source_id
                LEFT JOIN verification_state vs
                  ON vs.target_kind = 'assertion'
                 AND vs.target_id = a.id
                 AND vs.matter_id = a.matter_id
                LEFT JOIN verification_state vs_edge
                  ON vs_edge.target_kind = 'evidence_edge'
                 AND vs_edge.target_id = ee.id
                 AND vs_edge.matter_id = ee.matter_id
                WHERE ee.matter_id = ?
                  AND ee.target_kind = 'issue' AND ee.target_id = ?
                  AND ee.active = 1
                  AND ee.source_kind = 'assertion'
                  AND ee.relation_type IN ('supports','establishes','attacks','negates')
                  AND a.belief_state NOT IN ('superseded','withdrawn','disputed')
                  -- P0.2: TrustPurpose.PROOF_CANDIDATE drops stale and
                  -- rejected on both assertion and edge lanes.
                  AND COALESCE(vs.status, 'candidate') NOT IN ('rejected','stale')
                  AND COALESCE(vs_edge.status, ee.verification_status, 'candidate') NOT IN ('rejected','stale')
                  """ + privilege_filter + """
                GROUP BY ee.source_id, ee.relation_type
            """
            rows = self.db.execute(
                query,
                (self.matter_id, issue_id, self.matter_id, issue_id),
            ).fetchall()
        else:
            inner_ids_sql = (
                "SELECT assertion_id FROM assertion_issue_link "
                "WHERE issue_id = ? "
                "AND relation_type IN ('supports','establishes','attacks','negates')"
            )
            query = occ_ranked_cte.format(inner_ids=inner_ids_sql) + """
                SELECT ail.relation_type,
                       COALESCE(MAX(CASE WHEN o.role_rn = 1 THEN o.source_role END), 'unknown')
                           AS source_role,
                       MAX(CASE WHEN o.doc_rn = 1 THEN o.document_id END)
                           AS primary_doc_id
                FROM assertion_issue_link ail
                JOIN assertion a ON a.id = ail.assertion_id
                LEFT JOIN occ_ranked o ON o.assertion_id = ail.assertion_id
                LEFT JOIN verification_state vs
                  ON vs.target_kind = 'assertion'
                 AND vs.target_id = a.id
                 AND vs.matter_id = a.matter_id
                WHERE ail.issue_id = ?
                  AND ail.relation_type IN ('supports','establishes','attacks','negates')
                  AND a.belief_state NOT IN ('superseded','withdrawn','disputed')
                  -- P0.2: stale drops out alongside rejected.
                  AND COALESCE(vs.status, 'candidate') NOT IN ('rejected','stale')
                  """ + privilege_filter + """
                GROUP BY ail.assertion_id, ail.relation_type
            """
            rows = self.db.execute(query, (issue_id, issue_id)).fetchall()
        return list(rows)

    # ------------------------------------------------------------------
    # Compute + store
    # ------------------------------------------------------------------

    def compute_and_store(
        self,
        issue_id: str,
        _preloaded_overrides: "list[tuple[str, str]] | None" = None,
        policy_audience: str = "clean",
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

        # MVP.3: prefer evidence_edge as the canonical proof substrate when
        # the issue has any active edges. Legacy assertion_issue_link is
        # only used as a migration fallback when no edges exist. Proof
        # math is unchanged — only the upstream substrate selection is.
        has_edges = EvidenceStore(self.db, self.matter_id).target_has_edges(
            "issue", issue_id
        )
        _linked_rows = self._query_issue_linked_assertions(
            issue_id, has_edges, policy_audience=policy_audience
        )
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
        - If predicates defined AND at least one resolved:
              predicate_ratio × assertion_ratio
              where predicate_ratio = satisfied / total
        - Otherwise (no predicates, or predicates defined but none yet resolved):
              assertion_ratio alone — fallback prevents unfair 0-scoring before
              resolve_predicate() is wired in (SO-4).

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


class AssumptionStore:
    """CRUD for the assumption / assumption_link tables (Gap 3: conditional logic).

    Assumptions are provisional beliefs that gate predicate resolution.
    Each assumption can be linked to issues, predicates, or assertions via
    assumption_link. When an assumption is invalidated, any predicate it
    guards should be marked 'blocked' rather than resolved.

    Status lifecycle: provisional → confirmed | invalidated
    """

    VALID_STATUSES = ("provisional", "confirmed", "invalidated")

    def __init__(self, db: SQLiteMatterDB, matter_id: str):
        self.db = db
        self.matter_id = matter_id

    def upsert(
        self,
        statement: str,
        rationale: str | None = None,
        invalidation_condition: str | None = None,
        source_kind: str = "system",
        status: str = "provisional",
    ) -> str:
        """Insert or update an assumption. Returns assumption ID.

        Deduplicates on (matter_id, statement) — if an assumption with the same
        statement already exists, updates rationale/condition/status and returns
        the existing ID.
        """
        now = _now()
        existing = self.db.execute(
            "SELECT id FROM assumption WHERE matter_id=? AND statement=?",
            (self.matter_id, statement),
        ).fetchone()
        if existing:
            aid = existing["id"]
            self.db.execute(
                "UPDATE assumption SET rationale=?, invalidation_condition=?,"
                " source_kind=?, status=?, updated_at=? WHERE id=?",
                (rationale, invalidation_condition, source_kind, status, now, aid),
            )
            return aid
        aid = _id()
        self.db.execute(
            "INSERT INTO assumption (id, matter_id, statement, rationale,"
            " invalidation_condition, source_kind, status, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (aid, self.matter_id, statement, rationale,
             invalidation_condition, source_kind, status, now, now),
        )
        return aid

    def link(self, assumption_id: str, target_type: str, target_id: str) -> str:
        """Link an assumption to a target (issue, predicate, assertion).

        Returns link ID. Idempotent — duplicate links are ignored.
        """
        existing = self.db.execute(
            "SELECT id FROM assumption_link WHERE assumption_id=?"
            " AND target_type=? AND target_id=?",
            (assumption_id, target_type, target_id),
        ).fetchone()
        if existing:
            return existing["id"]
        lid = _id()
        self.db.execute(
            "INSERT INTO assumption_link (id, assumption_id, target_type,"
            " target_id, created_at) VALUES (?,?,?,?,?)",
            (lid, assumption_id, target_type, target_id, _now()),
        )
        return lid

    def get_for_target(
        self, target_type: str, target_id: str, max_rows: int = 50
    ) -> list[dict]:
        """Return assumptions linked to a specific target."""
        rows = self.db.execute(
            "SELECT a.* FROM assumption a"
            " JOIN assumption_link al ON al.assumption_id = a.id"
            " WHERE al.target_type=? AND al.target_id=? AND a.matter_id=?"
            " ORDER BY a.created_at DESC LIMIT ?",
            (target_type, target_id, self.matter_id, max_rows),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_active(self, max_rows: int = 50) -> list[dict]:
        """Return all provisional assumptions for this matter."""
        rows = self.db.execute(
            "SELECT * FROM assumption WHERE matter_id=? AND status='provisional'"
            " ORDER BY created_at DESC LIMIT ?",
            (self.matter_id, max_rows),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all(self, max_rows: int = 100) -> list[dict]:
        """Return all assumptions for this matter regardless of status."""
        rows = self.db.execute(
            "SELECT * FROM assumption WHERE matter_id=?"
            " ORDER BY created_at DESC LIMIT ?",
            (self.matter_id, max_rows),
        ).fetchall()
        return [dict(r) for r in rows]

    def set_status(self, assumption_id: str, status: str, reason: str | None = None) -> bool:
        """Set assumption status. Returns True if updated."""
        if status not in self.VALID_STATUSES:
            raise ValueError(f"Invalid assumption status: {status}")
        now = _now()
        cursor = self.db.execute(
            "UPDATE assumption SET status=?, rationale=COALESCE(?, rationale),"
            " updated_at=? WHERE id=? AND matter_id=?",
            (status, reason, now, assumption_id, self.matter_id),
        )
        return cursor.rowcount > 0

    def confirm(self, assumption_id: str) -> bool:
        """Mark assumption as confirmed."""
        return self.set_status(assumption_id, "confirmed")

    def invalidate(self, assumption_id: str, reason: str | None = None) -> bool:
        """Mark assumption as invalidated."""
        return self.set_status(assumption_id, "invalidated", reason)

    def count(self) -> int:
        row = self.db.execute(
            "SELECT COUNT(*) FROM assumption WHERE matter_id=?",
            (self.matter_id,),
        ).fetchone()
        return row[0]

    def has_blocking_assumptions(self, target_type: str, target_id: str) -> bool:
        """Check if any linked assumptions are invalidated (blocking the target)."""
        row = self.db.execute(
            "SELECT COUNT(*) FROM assumption a"
            " JOIN assumption_link al ON al.assumption_id = a.id"
            " WHERE al.target_type=? AND al.target_id=? AND a.matter_id=?"
            " AND a.status='invalidated'",
            (target_type, target_id, self.matter_id),
        ).fetchone()
        return row[0] > 0

    def has_unresolved_assumptions(self, target_type: str, target_id: str) -> bool:
        """Check if any linked assumptions are still provisional (unresolved)."""
        row = self.db.execute(
            "SELECT COUNT(*) FROM assumption a"
            " JOIN assumption_link al ON al.assumption_id = a.id"
            " WHERE al.target_type=? AND al.target_id=? AND a.matter_id=?"
            " AND a.status='provisional'",
            (target_type, target_id, self.matter_id),
        ).fetchone()
        return row[0] > 0
