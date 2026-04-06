"""MatterRuntimeAdapter — bridges engine.py to the matter model.

Keeps engine.py on one canonical path while the matter model is
optional (enable_matter_model=False preserves all existing behavior).

When enabled:
- engine reads QueryMatterContext at run start
- extracted facts are converted to AssertionCandidates and upserted
- ledger events are appended in real time
- conflicts trigger BeliefRevision after each write batch
"""

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from .matter import MatterModel
from .enums import (
    LedgerEventType, SpeechAct, SourceRole, ModelLayer,
    AssertionKind, OriginKind, RevisionCause, AssertionLinkType, GapType,
)
from .models import AssertionCandidate, QueryMatterContext


# ---------------------------------------------------------------------------
# Source-role inference from document filename/path
# ---------------------------------------------------------------------------
# Maps filename keyword patterns → SourceRole.
# Listed in priority order — first match wins.

_SOURCE_ROLE_PATTERNS: list[tuple[re.Pattern, SourceRole]] = [
    # Draft: check FIRST — a draft contract is not yet operative
    (re.compile(
        r"(draft|redline|redlined|markup|track.?change)",
        re.IGNORECASE,
    ), SourceRole.DRAFT),
    # Advocacy: party-authored pleadings and argument documents (check BEFORE procedural
    # so that "complaint" and "answer" get advocacy weight, not procedural weight).
    # Use \b word boundaries for terms that are substrings of procedural words
    # (e.g. "position" is a substring of "deposition").
    (re.compile(
        r"(complaint|answer|brief|demand|letter|memo\b|memorandum|\bposition\b|argument|appeal)",
        re.IGNORECASE,
    ), SourceRole.ADVOCACY),
    # Procedural: court filings, discovery documents, neutral court records
    (re.compile(
        r"(motion|petition|pleading|filing|discovery|"
        r"subpoena|deposition|interrogator|exhibit|affidavit)",
        re.IGNORECASE,
    ), SourceRole.PROCEDURAL),
    # Authoritative: statutes, regulations, court orders, judicial decisions.
    # Check BEFORE operative so that "order_approving_settlement_agreement" is
    # classified AUTHORITATIVE (it is a court order) rather than OPERATIVE (settlement).
    # Bare "order" is NOT included because "purchase_order", "change_order", and
    # "work_order" would falsely match. Court orders are identified by compound patterns
    # (court_order, order_approv*, final_order, etc.) or by unambiguous terms
    # (statute, ruling, injunction, etc.). A standalone "order.pdf" resolves as UNKNOWN.
    (re.compile(
        r"(statute|regulation|rule|code|opinion|decision|judgment|judgement|"
        r"mandate|injunction|ruling|decree|"
        r"order[._-]approv\w*|order[._-]grant\w*|order[._-]enter\w*|"
        r"order[._-]den[yi]\w*|order[._-]enjoin\w*|order[._-]dismiss\w*|"
        r"court[._-]order|consent[._-]order|final[._-]order|interim[._-]order|"
        r"preliminary[._-]order|protective[._-]order|restraining[._-]order|"
        r"show[._-]cause)",
        re.IGNORECASE,
    ), SourceRole.AUTHORITATIVE),
    # Operative: contracts, agreements, amendments, leases. Checked after authoritative
    # so that court orders that approve settlements are not misclassified as operative.
    (re.compile(
        r"(contract|agreement|msa|sow|nda|lease|license|amendment|addendum|"
        r"settlement|deed|covenant|warrant|indenture|resolution)",
        re.IGNORECASE,
    ), SourceRole.OPERATIVE),
    # Informal: emails, messages, notes, chats, texts
    (re.compile(
        r"(email|mail|message|note|chat|text|sms|slack|teams|whatsapp|"
        r"thread|correspondence)",
        re.IGNORECASE,
    ), SourceRole.INFORMAL),
    # Post-hoc: explanatory memos, expert reports, declarations written after events
    (re.compile(
        r"(report|expert|declaration|analysis|assessment|audit|review|evaluation)",
        re.IGNORECASE,
    ), SourceRole.POST_HOC_EXPLANATORY),
]


def infer_source_role(document_id: str) -> SourceRole:
    """
    Infer SourceRole from document filename/path keywords.
    Returns SourceRole.UNKNOWN if no pattern matches.
    """
    stem = Path(document_id).name  # filename only, not full path
    for pattern, role in _SOURCE_ROLE_PATTERNS:
        if pattern.search(stem):
            return role
    return SourceRole.UNKNOWN


_SOURCE_SIDE_PLAINTIFF_PATTERN = re.compile(
    r"(?<![a-zA-Z])(plaintiff|plaintif|petitioner|claimant|complainant|prosecution|relator)s?(?![a-zA-Z])",
    re.IGNORECASE,
)
_SOURCE_SIDE_DEFENDANT_PATTERN = re.compile(
    r"(?<![a-zA-Z])(defendant|respondent|defense|defence|accused)s?(?![a-zA-Z])",
    re.IGNORECASE,
)


def infer_source_side(document_id: str) -> Optional[str]:
    """Infer litigation side from document filename keywords.

    Returns "plaintiff", "defendant", or None (neutral/unknown).
    Neutral documents (court orders, third-party records) return None.

    Strategy:
    1. Examine the basename first. If it has a clear, unambiguous side signal,
       use it. Word boundaries prevent "counterclaimant" matching "claimant".
    2. If the basename carries no signal, fall back to the immediate parent
       directory name only (not the full path). This handles common layouts
       like "defendant/answer.pdf" without the broader false-positive risk
       of scanning the entire path (e.g. "plaintiff_exhibits/defendant_answer.pdf"
       correctly returns "defendant" from the basename in step 1).
    3. If both patterns match at any level, return None (ambiguous).
    """
    import os as _os
    parts = document_id.replace("\\", "/").split("/")
    basename = parts[-1].lower()
    parent = parts[-2].lower() if len(parts) >= 2 else ""

    b_plt = bool(_SOURCE_SIDE_PLAINTIFF_PATTERN.search(basename))
    b_def = bool(_SOURCE_SIDE_DEFENDANT_PATTERN.search(basename))

    if b_plt and b_def:
        return None  # Ambiguous basename
    if b_plt:
        return "plaintiff"
    if b_def:
        return "defendant"

    # Basename has no signal — check immediate parent directory as fallback
    p_plt = bool(_SOURCE_SIDE_PLAINTIFF_PATTERN.search(parent))
    p_def = bool(_SOURCE_SIDE_DEFENDANT_PATTERN.search(parent))
    if p_plt and p_def:
        return None
    if p_plt:
        return "plaintiff"
    if p_def:
        return "defendant"
    return None


class MatterRuntimeAdapter:
    """
    Thin adapter used by RLMEngine when enable_matter_model=True.

    The engine creates one adapter per investigation run and calls it
    at key points in the pipeline. When not enabled, all methods are
    no-ops and the engine behaves exactly as before.
    """

    def __init__(
        self,
        matter_model: MatterModel,
        run_id: str,
    ):
        self.model = matter_model
        self.run_id = run_id
        self._pending_assertion_ids: list[str] = []
        # Record run start time for mid-run clarification injection (SO-3)
        from datetime import datetime, timezone
        self._run_started_at: str = datetime.now(timezone.utc).isoformat()
        # Track which clarification IDs have already been injected this run
        self._injected_clarification_ids: set[str] = set()
        # In-memory stop flag: once set True it stays True, avoiding repeated DB reads.
        # Checked on every lead/doc boundary — same-process stops are O(1); cross-process
        # stops (API call from another thread) require a DB read on every check to maintain
        # the "detects stop between every pair of consecutive check points" contract.
        self._stop_flag: bool = False

    # ------------------------------------------------------------------
    # Called from engine._orient()
    # ------------------------------------------------------------------

    def get_context(self) -> QueryMatterContext:
        """Return current matter context for injection into orientation prompt."""
        return self.model.build_query_context()

    def log_objective(self, objective: str) -> None:
        self.model.ledger.append_event(
            run_id=self.run_id,
            event_type=LedgerEventType.OBJECTIVE_SET,
            summary=f"Objective: {objective[:200]}",
        )

    # ------------------------------------------------------------------
    # Called from engine._analyze_search_results() / _deep_read_document()
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_model_layer(
        assertion_kind: "AssertionKind",
        origin_kind: "OriginKind",
        explicit_layer: "ModelLayer",
    ) -> "ModelLayer":
        """Auto-select model layer when the caller left the default RECORD.

        Enforcement rules (spec §24 — 5 layers must remain distinct):
          NORMATIVE assertions (obligations/rights/duties) belong to the LEGAL
            layer — they describe what the law says, not what the record says.
          INFERRED-origin assertions belong to the REALITY layer — they represent
            conclusions the system derived from record evidence, not direct quotes.
          All other assertions stay in the RECORD layer (default).

        Callers that explicitly pass a non-RECORD layer are never overridden.
        """
        if explicit_layer != ModelLayer.RECORD:
            return explicit_layer
        if assertion_kind == AssertionKind.NORMATIVE:
            return ModelLayer.LEGAL
        if origin_kind == OriginKind.INFERRED:
            return ModelLayer.REALITY
        return ModelLayer.RECORD

    def record_fact(
        self,
        proposition_text: str,
        document_id: str,
        source_role: SourceRole = SourceRole.UNKNOWN,
        speech_act: SpeechAct = SpeechAct.EXTRACTED,
        model_layer: ModelLayer = ModelLayer.RECORD,
        assertion_kind: AssertionKind = AssertionKind.FACTUAL,
        origin_kind: OriginKind = OriginKind.EXTRACTED,
        span_id: Optional[str] = None,
        issue_id: Optional[str] = None,
        issue_link_type: str = "supports",
        temporal_scope_start: Optional[str] = None,
        subject_ref_type: Optional[str] = None,
        subject_ref_id: Optional[str] = None,
        predicate_key: Optional[str] = None,
        object_json: Optional[str] = None,
        temporal_scope_end: Optional[str] = None,
    ) -> str:
        """
        Convert an extracted fact string into a typed assertion.
        Auto-infers source_role from document_id if not explicitly provided.
        Also auto-infers speech_act from source_role when speech_act is EXTRACTED
        (advocacy → alleged, operative → operative, authoritative → operative).
        Auto-infers model_layer: NORMATIVE→LEGAL, INFERRED origin→REALITY, else RECORD.
        If issue_id is provided, links the assertion to that issue with issue_link_type.
        Optional subject_ref_type/subject_ref_id/predicate_key/object_json populate
        the typed SPO fields when the LLM extracts structured triples (SO-2).
        Returns assertion_id.
        """
        if source_role == SourceRole.UNKNOWN:
            source_role = infer_source_role(document_id)

        # Auto-elevate speech_act when the caller left it as the generic EXTRACTED default
        if speech_act == SpeechAct.EXTRACTED:
            if source_role == SourceRole.ADVOCACY:
                speech_act = SpeechAct.ALLEGED
            elif source_role in (SourceRole.OPERATIVE, SourceRole.AUTHORITATIVE):
                speech_act = SpeechAct.OPERATIVE

        # Apply user trust override (SO-3 trust steering, SO-5 calibration).
        # 'low'  → force ALLEGED regardless of inferred role
        # 'high' → promote ALLEGED → OPERATIVE (user asserts this document is authoritative)
        trust = self.model.trust_overrides.get(document_id)
        if trust == "low":
            speech_act = SpeechAct.ALLEGED
        elif trust == "high" and speech_act in (SpeechAct.ALLEGED, SpeechAct.EXTRACTED):
            # Promote both ALLEGED and EXTRACTED sources when user marks document as high-trust
            speech_act = SpeechAct.OPERATIVE

        # Auto-assign model_layer based on assertion_kind and origin_kind (spec §24,
        # 5-layer enforcement).  NORMATIVE→LEGAL, INFERRED origin→REALITY, else RECORD.
        model_layer = self._infer_model_layer(assertion_kind, origin_kind, model_layer)

        candidate = AssertionCandidate(
            proposition_text=proposition_text,
            model_layer=model_layer,
            assertion_kind=assertion_kind,
            document_id=document_id,
            span_id=span_id,
            source_role=source_role,
            source_side=infer_source_side(document_id),
            speech_act=speech_act,
            origin_kind=origin_kind,
            temporal_scope_start=temporal_scope_start,
            subject_ref_type=subject_ref_type,
            subject_ref_id=subject_ref_id,
            predicate_key=predicate_key,
            object_json=object_json,
            temporal_scope_end=temporal_scope_end,
        )
        assertion_id, is_new = self.model.record_assertion(candidate, run_id=self.run_id)
        self._pending_assertion_ids.append(assertion_id)

        if is_new:
            self.model.ledger.append_event(
                run_id=self.run_id,
                event_type=LedgerEventType.ASSERTION_ADDED,
                summary=f"New assertion: {proposition_text[:120]}",
                changed_object_type="assertion",
                changed_object_id=assertion_id,
            )

        # Link to issue if a focus issue was specified for this lead (SO-4).
        # 'neutral' facts are recorded but intentionally not linked — they provide
        # context without claiming to support or attack the issue predicate.
        if issue_id is not None and issue_link_type != "neutral":
            self.model.issues.link_assertion(assertion_id, issue_id, issue_link_type)

        return assertion_id

    def record_facts_batch(
        self,
        facts: list,
        issue_id: Optional[str] = None,
        default_source_role: SourceRole = SourceRole.UNKNOWN,
    ) -> list[str]:
        """Record multiple facts in a single outer transaction to reduce per-fact commit overhead.

        Each element of `facts` is either a tuple or a dict:

        Tuple forms:
          - (proposition_text, document_id)                               — defaults
          - (proposition_text, document_id, issue_relation)               — explicit relation
          - (proposition_text, document_id, issue_relation, temporal_scope_start) — + ISO date

        Dict form (supports typed SPO fields for SO-2):
          {
            "proposition_text": str,
            "document_id": str,
            "issue_link_type": str,           # default "supports"
            "temporal_scope_start": str|None,
            "subject_ref_type": str|None,     # SO-2 typed triple
            "subject_ref_id": str|None,
            "predicate_key": str|None,
            "object_json": str|None,
            "temporal_scope_end": str|None,
          }

        default_source_role: when provided (not UNKNOWN), used as the source_role for all
        facts in the batch instead of filename-based heuristic inference (SO-5 content fix).
        Individual facts may still be overridden via record_fact(source_role=...) if needed.

        Processed inside one outer ``with self.model.db.transaction()`` so that inner
        per-fact transactions become savepoints instead of full BEGIN/COMMITs,
        collapsing N disk syncs into 1.

        Returns list of assertion IDs in the same order as `facts`.
        """
        if not facts:
            return []
        assertion_ids = []
        with self.model.db.transaction():
            for item in facts:
                if isinstance(item, dict):
                    proposition_text = item.get("proposition_text") or ""
                    document_id = item.get("document_id") or ""
                    if not proposition_text:
                        # Keep alignment with caller's positional indexing (e.g.
                        # fact_relationships from_idx/to_idx) by appending a sentinel
                        # empty string rather than shrinking the list.
                        assertion_ids.append("")
                        continue
                    issue_link_type = item.get("issue_link_type", "supports")
                    temporal_scope_start = item.get("temporal_scope_start")
                    subject_ref_type = item.get("subject_ref_type")
                    subject_ref_id = item.get("subject_ref_id")
                    predicate_key = item.get("predicate_key")
                    object_json = item.get("object_json")
                    temporal_scope_end = item.get("temporal_scope_end")
                else:
                    proposition_text, document_id = item[0], item[1]
                    issue_link_type = item[2] if len(item) > 2 else "supports"
                    temporal_scope_start = item[3] if len(item) > 3 else None
                    subject_ref_type = subject_ref_id = predicate_key = object_json = temporal_scope_end = None
                aid = self.record_fact(
                    proposition_text,
                    document_id=document_id,
                    source_role=default_source_role,
                    issue_id=issue_id,
                    issue_link_type=issue_link_type,
                    temporal_scope_start=temporal_scope_start,
                    subject_ref_type=subject_ref_type,
                    subject_ref_id=subject_ref_id,
                    predicate_key=predicate_key,
                    object_json=object_json,
                    temporal_scope_end=temporal_scope_end,
                )
                assertion_ids.append(aid)
        return assertion_ids

    def record_assertion_link(
        self,
        src_assertion_id: str,
        dst_assertion_id: str,
        link_type: str,
    ) -> None:
        """Create a directed edge between two assertions in the dependency graph (SO-2).

        Called after deep-read relationship extraction to populate the assertion graph
        from LLM-identified logical relationships within a document.
        """
        try:
            lt = AssertionLinkType(link_type)
        except ValueError:
            _warn_msg = (
                f"record_assertion_link: unknown link_type '{link_type}' — edge dropped "
                f"({src_assertion_id[:8]}→{dst_assertion_id[:8]})"
            )
            try:
                self.log_warning(_warn_msg)
            except Exception as _warn_exc:
                # Ledger write failed — fall back to Python logger so the failure
                # is always observable (SO-3: no silent swallowing of diagnostic events).
                logger.warning("%s (ledger write failed: %s)", _warn_msg, _warn_exc)
            return
        try:
            self.model.assertions.link(src_assertion_id, dst_assertion_id, lt)
        except Exception as exc:
            # Link write must not block already-recorded facts, but the failure is
            # observable via the ledger so the dependency graph gap is diagnosed.
            _link_warn = (
                f"record_assertion_link: write failed ({src_assertion_id[:8]}→"
                f"{dst_assertion_id[:8]}, {link_type}): {str(exc)[:120]}"
            )
            try:
                self.log_warning(_link_warn)
            except Exception as _warn_exc:
                # Ledger write failed — fall back to Python logger (SO-3: no silent swallowing).
                logger.warning("%s (ledger write failed: %s)", _link_warn, _warn_exc)

    def flush_revisions(self) -> int:
        """
        Trigger belief revision for all assertions added since last flush.
        Returns count of revised assertions.

        Seeds are processed in batches sized to stay within the BFS work budget
        so that every seed receives at least one revision pass.  Without batching,
        a large flush (>2000 seeds) would silently skip high-index seeds because
        the BFS frontier budget is exhausted before reaching them.

        Acquires model._flush_lock to serialize concurrent callers (one per MatterModel
        instance). This prevents reload_pending_from_db() + drain from racing between
        two simultaneous flush_revisions() calls (adv#029 SO-1 fix r7).
        """
        with self.model._flush_lock:
            return self._flush_revisions_locked()

    def _flush_revisions_locked(self) -> int:
        """Body of flush_revisions(); called with model._flush_lock held."""
        # Drain nodes left unvisited by truncated correct_assertion() calls (adv#028 HIGH fix).
        # Processed separately from new-evidence pending so USER_CORRECTION provenance is
        # preserved in revision rows — mixing them would replay corrections as NEW_EVIDENCE
        # and corrupt the audit trail (r25 MEDIUM fix).
        # Audit attribution: assertion_revision rows (written by BFS) always use self.run_id.
        # Rather than remapping per-assertion in ASSERTION_REVISED events (which creates a
        # DB/ledger inconsistency), we emit one USER_CORRECTION ledger event naming the
        # originating runs before processing, so the full deferred correction batch is
        # traceable (r30 MEDIUM provenance fix). Second-level truncation re-enqueues for
        # the next flush_revisions() call (r28 MEDIUM fix). Count tracks unique assertion
        # IDs to handle fixpoint re-visits (r28 MEDIUM fix).
        # Recover any rows stranded in DB by a previous failed delete (adv#029 SO-1 r6).
        # This is a no-op on the normal path (all DB rows are in-memory from enqueue).
        self.model.reload_pending_from_db()
        _seed_batch = max(1, self.model.belief.MAX_WORK // 2)
        _revised_ids: set[str] = set()
        _correction_map, _correction_db_ids = self.model.drain_correction_pending()
        _correction_ids = list(_correction_map)
        if _correction_ids:
            # Emit one attribution event so auditors can trace this flush back to the
            # corrections that produced the deferred seeds, without fragmenting run_id
            # across individual ASSERTION_REVISED rows.
            _orig_runs = sorted({r for r in _correction_map.values() if r})
            self.model.ledger.append_event(
                run_id=self.run_id,
                event_type=LedgerEventType.USER_CORRECTION,
                summary=(
                    f"Deferred correction replay: {len(_correction_ids)} assertion(s) "
                    f"from {len(_orig_runs)} originating run(s)"
                    + (f": {', '.join(_orig_runs)}" if _orig_runs else "")
                ),
            )
        for i in range(0, len(_correction_ids), _seed_batch):
            _batch = _correction_ids[i : i + _seed_batch]
            _unvisited: list[str] = []
            _cr_results = self.model.apply_revision(
                seed_assertion_ids=_batch,
                cause=RevisionCause.USER_CORRECTION,
                run_id=self.run_id,
                note="deferred correction retry",
                _collect_unvisited=_unvisited,
            )
            if _unvisited:
                # Re-enqueue work dropped by a second truncation so the next
                # flush_revisions() call can continue where this one left off.
                # Use self.run_id so second-level nodes are traceable to this flush run
                # rather than being completely unattributed (r31 MEDIUM fix).
                self.model.enqueue_correction_pending(_unvisited, run_id=self.run_id)
            for result in _cr_results:
                if result.old_belief_state != result.new_belief_state:
                    _revised_ids.add(result.assertion_id)
                    self.model.ledger.append_event(
                        run_id=self.run_id,
                        event_type=LedgerEventType.ASSERTION_REVISED,
                        summary=(
                            f"Belief revised: {result.old_belief_state.value} → "
                            f"{result.new_belief_state.value}"
                        ),
                        changed_object_type="assertion",
                        changed_object_id=result.assertion_id,
                    )
        # Delete the originally-drained correction rows by primary key (not assertion_id).
        # INSERT OR REPLACE in enqueue_* ensures re-enqueued same-ID nodes get fresh
        # primary keys, so delete by old IDs is a no-op for those rows (adv#029 r4 fix).
        self.model.delete_pending_propagation_db(_correction_db_ids)

        # Merge durable evidence-pending (nodes truncated by a prior flush, keyed by
        # originating (cause, run_id)) with current-request NEW_EVIDENCE seeds (r29–r33 fix).
        # Group by cause so each replay batch uses the correct RevisionCause (r31 fix).
        # Batch attribution events per cause name originating run_ids (r33 fix).
        _durable_evidence_map, _evidence_db_ids = self.model.drain_evidence_pending()
        _evidence_carry: "dict[str, tuple[RevisionCause, str | None]]" = dict(_durable_evidence_map)
        for aid in self._pending_assertion_ids:
            if aid not in _evidence_carry:
                _evidence_carry[aid] = (RevisionCause.NEW_EVIDENCE, self.run_id)
        self._pending_assertion_ids.clear()
        if _evidence_carry:
            # Group by cause; collect originating run_ids per cause for batch attribution event.
            _by_cause: "dict[RevisionCause, list[str]]" = {}
            _orig_runs_by_cause: "dict[RevisionCause, set[str]]" = {}
            for aid, (cause, orig_run) in _evidence_carry.items():
                _by_cause.setdefault(cause, []).append(aid)
                if orig_run:
                    _orig_runs_by_cause.setdefault(cause, set()).add(orig_run)
            for cause, ids in _by_cause.items():
                _orig = sorted(_orig_runs_by_cause.get(cause, set()))
                if _orig:
                    self.model.ledger.append_event(
                        run_id=self.run_id,
                        event_type=LedgerEventType.ASSERTION_REVISED,
                        summary=(
                            f"Deferred {cause.value} replay: {len(ids)} assertion(s) "
                            f"from originating run(s): {', '.join(_orig)}"
                        ),
                    )
                for i in range(0, len(ids), _seed_batch):
                    batch = ids[i : i + _seed_batch]
                    _ev_unvisited: list[str] = []
                    results = self.model.apply_revision(
                        seed_assertion_ids=batch,
                        cause=cause,
                        run_id=self.run_id,
                        _collect_unvisited=_ev_unvisited,
                    )
                    if _ev_unvisited:
                        self.model.enqueue_evidence_pending(_ev_unvisited, cause=cause, run_id=self.run_id)
                    for result in results:
                        if result.old_belief_state != result.new_belief_state:
                            _revised_ids.add(result.assertion_id)
                            self.model.ledger.append_event(
                                run_id=self.run_id,
                                event_type=LedgerEventType.ASSERTION_REVISED,
                                summary=(
                                    f"Belief revised: {result.old_belief_state.value} → "
                                    f"{result.new_belief_state.value}"
                                ),
                                changed_object_type="assertion",
                                changed_object_id=result.assertion_id,
                            )
        # Delete the originally-drained evidence rows by primary key — same pattern as
        # correction delete above (adv#029 SO-1 r4 fix).
        self.model.delete_pending_propagation_db(_evidence_db_ids)

        # Proof_state recompute for all assertions revised in this flush — covers issues
        # linked to any corrected or re-evaluated assertion so that issue-level consumers
        # (engine reweighting, contested/advocacy-only flags) see the current state
        # immediately after flush rather than on the next compute_all() (adv#029 SO-4 fix).
        if _revised_ids:
            try:
                _SQL_PARAM_LIMIT = 900
                _flush_affected = list(_revised_ids)
                _issue_ids: set[str] = set()
                for _bs in range(0, len(_flush_affected), _SQL_PARAM_LIMIT):
                    _batch = _flush_affected[_bs : _bs + _SQL_PARAM_LIMIT]
                    _rows = self.model.db.execute(
                        "SELECT DISTINCT issue_id FROM assertion_issue_link"
                        " WHERE assertion_id IN ({})".format(",".join("?" * len(_batch))),
                        _batch,
                    ).fetchall()
                    _issue_ids.update(r["issue_id"] for r in _rows)
                if _issue_ids:
                    _ov_rows = self.model.db.execute(
                        """SELECT document_pattern, trust_level FROM document_trust_override
                           WHERE matter_id=? AND trust_level != 'normal'
                           ORDER BY LENGTH(document_pattern) DESC""",
                        (self.model.matter_id,),
                    ).fetchall()
                    _overrides = [(r["document_pattern"], r["trust_level"]) for r in _ov_rows]
                    with self.model.db.transaction():
                        for _iid in _issue_ids:
                            self.model.proof_state.compute_and_store(
                                _iid, _preloaded_overrides=_overrides
                            )
            except Exception as exc:
                logger.warning("proof_state recompute after flush_revisions failed: %s", exc)

        return len(_revised_ids)

    # ------------------------------------------------------------------
    # Called from engine._emit_step() / general progress
    # ------------------------------------------------------------------

    def log_step(self, summary: str, why: Optional[str] = None) -> None:
        """Append a generic progress note to the reasoning ledger."""
        self.model.ledger.append_event(
            run_id=self.run_id,
            event_type=LedgerEventType.PROGRESS_NOTE,
            summary=summary[:500],
            why=why,
        )

    def log_conflict(self, summary: str) -> None:
        self.model.ledger.append_event(
            run_id=self.run_id,
            event_type=LedgerEventType.CONFLICT_DETECTED,
            summary=summary[:500],
        )

    def log_warning(self, summary: str) -> None:
        """Append a system warning to the reasoning ledger (non-fatal errors, dropped data)."""
        self.model.ledger.append_event(
            run_id=self.run_id,
            event_type=LedgerEventType.SYSTEM_WARNING,
            summary=summary[:500],
        )

    def log_gap(self, summary: str, gap_id: Optional[str] = None) -> None:
        self.model.ledger.append_event(
            run_id=self.run_id,
            event_type=LedgerEventType.GAP_IDENTIFIED,
            summary=summary[:500],
            changed_object_type="gap" if gap_id else None,
            changed_object_id=gap_id,
        )

    def record_gap(
        self,
        description: str,
        gap_type: GapType = GapType.MISSING_DOCUMENT,
        expected_artifact: Optional[str] = None,
        materiality: float = 0.5,
        affected_type: Optional[str] = None,
        affected_id: Optional[str] = None,
    ) -> str:
        """
        Persist a structured gap in the gap store and write a ledger event.
        Returns gap_id.
        """
        gap_id = self.model.gaps.record(
            gap_type=gap_type,
            description=description,
            expected_artifact=expected_artifact,
            materiality=materiality,
            affected_type=affected_type,
            affected_id=affected_id,
        )
        self.log_gap(description, gap_id=gap_id)
        return gap_id

    # ------------------------------------------------------------------
    # User steering (called from API layer)
    # ------------------------------------------------------------------

    def request_redirect(self, issue_id: str) -> None:
        """Signal the engine to redirect focus to the given issue on the next iteration."""
        self.model.ledger.request_redirect(self.run_id, issue_id)
        self.model.ledger.append_event(
            run_id=self.run_id,
            event_type=LedgerEventType.BRANCH_SELECTED,
            summary=f"User requested redirect to issue: {issue_id[:60]}",
            branch_issue_id=issue_id,
        )

    def is_redirect_requested(self) -> bool:
        return self.model.ledger.is_redirect_requested(self.run_id)

    def get_redirect_issue_id(self) -> Optional[str]:
        return self.model.ledger.get_redirect_issue_id(self.run_id)

    def clear_redirect(self) -> None:
        self.model.ledger.clear_redirect(self.run_id)

    def get_new_answered_clarifications(self) -> list[dict]:
        """Return clarification answers that arrived after this run started and
        have not yet been injected into active leads (SO-3 mid-run steering).

        De-duplicated by question ID so repeated loop iterations don't re-inject
        the same answer as multiple leads.
        """
        all_new = self.model.clarifications.get_answered_since(self._run_started_at)
        fresh = [
            c for c in all_new
            if c.get("id") not in self._injected_clarification_ids
        ]
        for c in fresh:
            self._injected_clarification_ids.add(c["id"])
        return fresh

    def request_stop(self) -> None:
        """Signal the engine to stop after the current iteration."""
        self._stop_flag = True
        self.model.ledger.request_stop(self.run_id)
        self.model.ledger.append_event(
            run_id=self.run_id,
            event_type=LedgerEventType.USER_INTERRUPTED,
            summary="User requested stop",
        )

    def is_stop_requested(self) -> bool:
        # Fast path: in-memory flag — O(1), never stale for same-process stops.
        # The flag is set in request_stop() before the DB write, so it is never stale
        # within the same adapter instance (one adapter = one run = one process).
        if self._stop_flag:
            return True
        # DB check: necessary for cross-process stops (e.g. API call on another thread).
        # SQLite single-row SELECT is sub-millisecond; negligible vs. LLM call latency.
        result = self.model.ledger.is_stop_requested(self.run_id)
        if result:
            self._stop_flag = True  # cache for all future same-process calls
        return result

    def record_actor(
        self,
        canonical_name: str,
        actor_type: str = "person",
    ) -> str:
        """
        Persist an actor (person or organization) to the durable actor store.
        Idempotent: same normalized name returns the existing actor_id.
        Returns actor_id.
        """
        actor_id, _ = self.model.actors.upsert_actor(
            canonical_name=canonical_name,
            actor_type=actor_type,
        )
        return actor_id

    def record_quant(
        self,
        quant_kind: str,
        raw_text: str,
        amount_value: Optional[float] = None,
        currency: Optional[str] = None,
        date_value: Optional[str] = None,
        rate_value: Optional[float] = None,
        unit: Optional[str] = None,
        subject_type: Optional[str] = None,
        subject_id: Optional[str] = None,
        assertion_id: Optional[str] = None,
        span_id: Optional[str] = None,
    ) -> str:
        """
        Persist a structured numeric fact to the quant store (SO-6).
        subject_id: specific identifier (e.g. "Invoice #1042") for fine-grained
                    reconciliation — prevents two distinct invoices from appearing
                    as a conflict just because they share the same subject_type.
        Returns quant_fact_id.
        """
        return self.model.quant.record(
            quant_kind=quant_kind,
            raw_text=raw_text,
            amount_value=amount_value,
            currency=currency,
            date_value=date_value,
            rate_value=rate_value,
            unit=unit,
            subject_type=subject_type,
            subject_id=subject_id,
            assertion_id=assertion_id,
            span_id=span_id,
        )

    def record_quants_batch(self, specs: list[dict]) -> None:
        """Bulk-insert multiple quant facts in a single transaction.

        Each spec is a dict with keys matching record_quant() parameters
        (quant_kind and raw_text required; all others optional). Wraps
        QuantStore.record_many() — uses INSERT OR IGNORE with executemany
        so N numeric facts → 1 outer transaction commit.
        """
        self.model.quant.record_many(specs)

    def set_trust_override(
        self,
        document_pattern: str,
        trust_level: str,
        note: Optional[str] = None,
    ) -> str:
        """Set a trust override for a document pattern (SO-3 trust steering).

        trust_level: 'low' | 'normal' | 'high'
        Returns override_id.
        """
        return self.model.set_trust_override(document_pattern, trust_level, note, run_id=self.run_id)

    def list_trust_overrides(self) -> list[dict]:
        """Return all trust overrides for this matter."""
        return self.model.trust_overrides.list_all()

    def annotate_document(
        self,
        document_pattern: str,
        annotation_text: str,
        annotation_type: str = "strategic",
    ) -> str:
        """Attach a strategic note to a document pattern (SO-3 annotation).

        Returns annotation_id.
        """
        return self.model.annotations.add(document_pattern, annotation_text, annotation_type)

    def list_annotations(self, document_id: Optional[str] = None) -> list[dict]:
        """Return annotations for a specific document, or all recent annotations."""
        if document_id is not None:
            return self.model.annotations.get_for_document(document_id)
        return self.model.annotations.list_recent()


class NullMatterAdapter:
    """
    No-op adapter used when enable_matter_model=False.

    All methods return safe defaults so engine.py can call the adapter
    unconditionally regardless of whether the matter model is enabled.
    """

    def get_context(self) -> Optional[QueryMatterContext]:
        return None

    def log_objective(self, objective: str) -> None:
        pass

    def record_fact(self, proposition_text: str, document_id: str, **kwargs) -> str:
        return ""

    def record_facts_batch(self, facts: list, **kwargs) -> list:
        return [""] * len(facts)

    def flush_revisions(self) -> int:
        return 0

    def log_step(self, summary: str, why: Optional[str] = None) -> None:
        pass

    def log_warning(self, summary: str) -> None:
        pass

    def log_conflict(self, summary: str) -> None:
        pass

    def log_gap(self, summary: str, gap_id: Optional[str] = None) -> None:
        pass

    def record_gap(self, description: str, **kwargs) -> str:
        return ""

    def request_redirect(self, issue_id: str) -> None:
        pass

    def is_redirect_requested(self) -> bool:
        return False

    def get_redirect_issue_id(self) -> Optional[str]:
        return None

    def clear_redirect(self) -> None:
        pass

    def request_stop(self) -> None:
        pass

    def is_stop_requested(self) -> bool:
        return False

    def record_actor(self, canonical_name: str, actor_type: str = "person") -> str:
        return ""

    def record_quant(self, quant_kind: str, raw_text: str, **kwargs) -> str:
        return ""

    def record_quants_batch(self, specs: list) -> None:
        pass

    def record_assertion_link(
        self,
        src_assertion_id: str,
        dst_assertion_id: str,
        link_type: str,
    ) -> None:
        pass

    def get_new_answered_clarifications(self) -> list[dict]:
        return []

    def set_trust_override(self, document_pattern: str, trust_level: str, **kwargs) -> str:
        return ""

    def list_trust_overrides(self) -> list[dict]:
        return []

    def annotate_document(self, document_pattern: str, annotation_text: str, **kwargs) -> str:
        return ""

    def list_annotations(self, document_id: Optional[str] = None) -> list[dict]:
        return []
