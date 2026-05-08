"""Domain-neutral slot-profile registry for the Irys reasoning engine.

Slot profiles encapsulate the domain-specific extraction expectations the
engine consults at investigation time:

  * which queries should trigger profiling at all
  * which corpus cues estimate expected counts (scout)
  * what JSON shape deep-read should return (prompt addendum + parser)
  * how to render filled rows into the synthesis context
  * how a row maps onto issue-tree predicates (SO-4 link)

The substrate (extraction_slot, extraction_slot_evidence, etc.) lives in
the matter model. This module is the engine-side dispatcher. New profiles
extend the registry without touching engine code.

Profiles are registered by built-in module import or programmatically; the
default registry is populated by `default_registry()`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence


# ---------------------------------------------------------------------------
# Shared dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IssueLinkSpec:
    """How a slot row connects to the issue tree (SO-4)."""
    issue_title: str
    predicate_description: str
    relation: str = "supports"
    issue_type: str = "diligence_red_flag"


@dataclass(frozen=True)
class SlotProfileContext:
    """Read-only context for profile match/scout/parse decisions.

    Carries only data that profiles legitimately depend on. Domain
    composition data is opaque to profiles other than the hash, which is
    used as a dependency input.
    """
    matter_id: str
    query: str
    task_spec: Mapping[str, Any]
    domain_composition_hash: str
    domain_facets: tuple[Mapping[str, Any], ...]
    corpus_signals: Mapping[str, Any]

    @classmethod
    def empty(cls, matter_id: str = "", query: str = "") -> "SlotProfileContext":
        return cls(
            matter_id=matter_id,
            query=query,
            task_spec={},
            domain_composition_hash="",
            domain_facets=(),
            corpus_signals={},
        )


@dataclass(frozen=True)
class ProfileMatch:
    """Result of a profile match decision."""
    profile_id: str
    score: float
    reasons: tuple[str, ...] = ()
    corpus_targets: tuple[str, ...] = ()


@dataclass(frozen=True)
class SlotRegistration:
    """A scout's expected-row finding, ready for ExtractionSlotStore.register()."""
    slot_kind: str
    slot_key: str
    expected_count: Optional[int]
    expected_count_confidence: float
    schema_ref: str
    profile_id: str
    artifact_family_id: Optional[str] = None
    issue_link: Optional[IssueLinkSpec] = None
    scout_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class TypedEvidenceWrite:
    """A parsed deep-read row, ready for typed_evidence.upsert + slot link."""
    record_kind: str
    record_key: str
    payload: Mapping[str, Any]
    label: str
    schema_ref: str
    profile_id: str
    document_id: Optional[str]
    span_id: Optional[str] = None
    confidence: float = 0.0
    slot_key: Optional[str] = None
    issue_link: Optional[IssueLinkSpec] = None


# ---------------------------------------------------------------------------
# Profile interface
# ---------------------------------------------------------------------------


class SlotProfile(Protocol):
    profile_id: str
    profile_version: int
    domain_profile_ids: tuple[str, ...]
    schema_ref: str
    record_kind: str
    slot_kind: str
    priority: int
    exclusive_group: Optional[str]
    supports_mixed_corpus: bool
    enabled: bool

    def match(self, context: SlotProfileContext) -> Optional[ProfileMatch]: ...
    def scout(
        self,
        context: SlotProfileContext,
        runtime: Any,
    ) -> Sequence[SlotRegistration]: ...
    def prompt_addendum(self, context: SlotProfileContext) -> str: ...
    def parse_evidence(
        self,
        *,
        analysis: Mapping[str, Any],
        document: Any,
        context: SlotProfileContext,
    ) -> Sequence[TypedEvidenceWrite]: ...
    def render_answer_rows(self, rows: Sequence[Mapping[str, Any]]) -> str: ...


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlotProfileDispatch:
    selected: tuple[SlotProfile, ...]
    prompt_profiles: tuple[SlotProfile, ...]
    candidates: tuple[ProfileMatch, ...]
    suppressed: tuple[Mapping[str, Any], ...]
    prompt_cap_drops: tuple[str, ...]


class SlotProfileRegistry:
    """Mutable registry of installed profiles + dispatcher.

    Profile lookup and dispatch are O(N) over registered profiles per query.
    For sub-millisecond cost on small N (3-15 profiles in practice).
    """

    def __init__(self, profiles: Sequence[SlotProfile] = ()) -> None:
        self._profiles: dict[str, SlotProfile] = {}
        for p in profiles:
            self.register(p)

    def register(self, profile: SlotProfile) -> None:
        if profile.profile_id in self._profiles:
            raise ValueError(f"duplicate slot profile: {profile.profile_id}")
        self._profiles[profile.profile_id] = profile

    def replace(self, profile: SlotProfile) -> None:
        """Swap an existing profile (test convenience)."""
        self._profiles[profile.profile_id] = profile

    def get(self, profile_id: str) -> Optional[SlotProfile]:
        return self._profiles.get(profile_id)

    def all(self) -> tuple[SlotProfile, ...]:
        return tuple(self._profiles.values())

    def dispatch(
        self,
        context: SlotProfileContext,
        *,
        prompt_profile_cap: int = 2,
    ) -> SlotProfileDispatch:
        scored: list[tuple[SlotProfile, ProfileMatch]] = []
        for profile in self._profiles.values():
            if not getattr(profile, "enabled", True):
                continue
            try:
                match = profile.match(context)
            except Exception:
                # A broken match() must not break dispatch
                continue
            if match is None:
                continue
            scored.append((profile, match))

        def rank(item: tuple[SlotProfile, ProfileMatch]) -> tuple[float, int, int, str]:
            profile, match = item
            return (
                match.score,
                profile.priority,
                profile.profile_version,
                profile.profile_id,
            )

        scored.sort(key=rank, reverse=True)

        selected: list[SlotProfile] = []
        suppressed: list[dict[str, Any]] = []
        occupied_groups: dict[str, str] = {}
        for profile, match in scored:
            group = profile.exclusive_group
            if group and group in occupied_groups:
                suppressed.append({
                    "profile_id": profile.profile_id,
                    "reason": "exclusive_group_lower_rank",
                    "winner": occupied_groups[group],
                    "score": match.score,
                })
                continue
            selected.append(profile)
            if group:
                occupied_groups[group] = profile.profile_id

        prompt_profiles = tuple(selected[:prompt_profile_cap])
        prompt_cap_drops = tuple(p.profile_id for p in selected[prompt_profile_cap:])
        return SlotProfileDispatch(
            selected=tuple(selected),
            prompt_profiles=prompt_profiles,
            candidates=tuple(m for _, m in scored),
            suppressed=tuple(suppressed),
            prompt_cap_drops=prompt_cap_drops,
        )


# ---------------------------------------------------------------------------
# Built-in profile: legal.market_row.v1
# ---------------------------------------------------------------------------


_REGULATORY_QUERY_TERMS = (
    "antitrust", "hsr", "merger review", "merger remed",
    "market share", "market shares", "hhi", "competitive effects",
    "regulatory", "leniency",
)

_PATH_CUES = (
    "market", "msa", "hhi", "share", "competitive", "competition",
    "overlap", "divestiture", "remedy", "ftc", "doj", "hsr",
    "antitrust", "concentration",
)

_HHI_SIGNAL_RE = re.compile(
    r"\bHHI\b|\bdelta\b|market share|post-merger|pre-merger",
    re.IGNORECASE,
)
_MSA_RE = re.compile(
    r"\b([A-Z][A-Za-z\.\-]+(?:[\-\s][A-Z][A-Za-z\.\-]+){0,3})\s+MSA\b"
)


def _normalize_market_name(name: str) -> str:
    s = (name or "").strip()
    s = re.sub(
        r"\s+(?:msa|metropolitan\s+statistical\s+area|metro\s+area|market)$",
        "", s, flags=re.IGNORECASE,
    )
    s = s.lower()
    s = re.sub(r"[^\w\s\-]", " ", s)
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


_MARKET_ROW_PROMPT = """
ROW-ATOMIC EXTRACTION (legal.market_row.v1):

For every MSA/geographic market mentioned in the documents with HHI or
market-share data, emit ONE structured "market_row" entry. Do NOT skip a
market because data is partial — emit it with whichever fields are
available and leave others null.

"market_rows": [
    {"market_name": "Greenville-Spartanburg MSA",
     "product_market": "bulk industrial gas",
     "geographic_market": "Greenville-Spartanburg MSA",
     "acquirer_share": "28%",
     "target_share": "18%",
     "other_shares": [{"name": "Competitor A", "share": "22%"}],
     "pre_merger_hhi": 2234,
     "post_merger_hhi": 3224,
     "delta_hhi": 990,
     "structural_presumption": true,
     "risk_rating": "high",
     "source_detail": "p.7 / Market Concentration table",
     "source_quote": "short verbatim quote or table row text"}
]

CRITICAL: emit one market_row per market mentioned. Missing one MSA when
others are extracted is a critical failure — incomplete competitive analysis.
""".strip()


@dataclass
class LegalMarketRowProfile:
    """Built-in: regulatory market-share / HHI extraction."""
    profile_id: str = "legal.market_row.v1"
    profile_version: int = 1
    domain_profile_ids: tuple[str, ...] = ("legal:1",)
    schema_ref: str = "legal.market_row.v1"
    record_kind: str = "market_row"
    slot_kind: str = "collection_item"
    priority: int = 100
    exclusive_group: Optional[str] = "regulatory_concentration"
    supports_mixed_corpus: bool = True
    enabled: bool = True
    prompt_token_budget: int = 450
    min_match_score: float = 0.55

    def match(self, context: SlotProfileContext) -> Optional[ProfileMatch]:
        q = (context.query or "").lower().strip()
        if not q:
            return None
        # Skip cheap factual lookups (handled by callers who set task_spec)
        family = (context.task_spec.get("family") or "").lower()
        if family in {"inventory_lookup", "metadata_lookup"}:
            return None
        score = 0.0
        reasons: list[str] = []
        for term in _REGULATORY_QUERY_TERMS:
            if term in q:
                # First term hits the floor by itself; additional terms
                # boost confidence but cap at 1.0.
                score = max(score, 0.65) + 0.1
                reasons.append(f"q_term:{term}")
                if score >= 1.0:
                    break
        score = min(score, 1.0)
        if score < self.min_match_score:
            return None
        return ProfileMatch(
            profile_id=self.profile_id,
            score=score,
            reasons=tuple(reasons),
        )

    def scout(
        self,
        context: SlotProfileContext,
        runtime: Any,
    ) -> Sequence[SlotRegistration]:
        """Scout MSA mentions across candidate documents.

        runtime is expected to provide:
          - list_files() -> Sequence[FileInfo]
          - read_excerpt(path, max_chars) -> str (or None on failure)
          - inventory_get_by_path(path) -> Mapping or None
          - card_get_by_doc_id(doc_id) -> Mapping or None
          - path_score_extra(path_lower) -> int (caller-side cues)
        """
        files = list(runtime.list_files() or [])
        if not files:
            return ()

        candidates: list[tuple[float, Any]] = []
        for f in files:
            path = str(getattr(f, "relative_path", "") or f)
            lc = path.lower()
            score = sum(1 for cue in _PATH_CUES if cue in lc)
            try:
                inv = runtime.inventory_get_by_path(path)
                card = runtime.card_get_by_doc_id(inv["id"]) if inv else None
            except Exception:
                card = None
            if card:
                doc_type = (card.get("doc_type") or "").lower()
                doc_subtype = (card.get("doc_subtype") or "").lower()
                purpose = (card.get("purpose") or "").lower()
                if doc_type in {"report", "memo", "presentation", "exhibit", "filing"}:
                    score += 1
                if any(cue in doc_subtype for cue in (
                    "market", "competition", "antitrust", "regulatory", "hhi", "overlap",
                )):
                    score += 2
                if any(cue in purpose for cue in (
                    "market", "competition", "antitrust", "regulatory", "hhi", "overlap",
                )):
                    score += 1
            candidates.append((score, f))

        candidates.sort(key=lambda t: -t[0])
        # Include all candidates; score is for ordering only. Filename cues
        # are noisy — MSAs often live in board decks/memos with neutral names.
        top = candidates[:12]

        market_candidates: dict[str, dict] = {}
        for score, f in top:
            path = str(getattr(f, "relative_path", "") or f)
            try:
                text = runtime.read_excerpt(path, 40000) or ""
            except Exception:
                continue
            if not text:
                continue
            for m in _MSA_RE.finditer(text):
                name = m.group(1).strip()
                norm = _normalize_market_name(name)
                if not norm:
                    continue
                cand = market_candidates.setdefault(norm, {
                    "display": f"{name} MSA",
                    "table_evidence": False,
                    "heading_evidence": False,
                    "filename_evidence": False,
                    "card_evidence": False,
                    "occurrences": 0,
                    "source_paths": set(),
                })
                cand["occurrences"] += 1
                cand["source_paths"].add(path)
                ctx_start = max(0, m.start() - 200)
                ctx_end = min(len(text), m.end() + 200)
                ctx = text[ctx_start:ctx_end]
                if (_HHI_SIGNAL_RE.search(ctx)
                    and ("|" in ctx or "\t" in ctx
                         or re.search(r"\d{3,}", ctx))):
                    cand["table_evidence"] = True
                else:
                    cand["heading_evidence"] = True

        # Filename-only nudge
        for norm, cand in market_candidates.items():
            for path in cand["source_paths"]:
                lc = path.lower()
                if any(cue in lc for cue in ("market", "msa", "hhi", "share")):
                    cand["filename_evidence"] = True
                    break

        registrations: list[SlotRegistration] = []
        for norm, cand in market_candidates.items():
            if cand["table_evidence"]:
                conf = 0.85
            elif cand["heading_evidence"] and (cand["card_evidence"] or cand["occurrences"] >= 2):
                conf = 0.75
            elif cand["heading_evidence"] and cand["filename_evidence"]:
                conf = 0.60
            else:
                continue  # below registration threshold

            slot_key = f"collection_item:{self.schema_ref}:msa:{norm}"
            registrations.append(SlotRegistration(
                slot_kind=self.slot_kind,
                slot_key=slot_key,
                expected_count=1,
                expected_count_confidence=conf,
                schema_ref=self.schema_ref,
                profile_id=self.profile_id,
                scout_refs=tuple(sorted(cand["source_paths"]))[:5],
            ))
        return tuple(registrations)

    def prompt_addendum(self, context: SlotProfileContext) -> str:
        return _MARKET_ROW_PROMPT

    def parse_evidence(
        self,
        *,
        analysis: Mapping[str, Any],
        document: Any,
        context: SlotProfileContext,
    ) -> Sequence[TypedEvidenceWrite]:
        rows = analysis.get("market_rows")
        if not isinstance(rows, list):
            return ()
        doc_id = getattr(document, "filename", None) or "unknown"
        out: list[TypedEvidenceWrite] = []
        import hashlib as _hashlib
        for r in rows[:100]:
            if not isinstance(r, dict):
                continue
            market = r.get("market_name") or r.get("geographic_market") or ""
            if not market:
                continue
            # Need at least one substantive HHI/share signal
            if not any(r.get(k) for k in (
                "pre_merger_hhi", "post_merger_hhi", "delta_hhi",
                "acquirer_share", "target_share", "other_shares",
                "risk_rating", "structural_presumption",
            )):
                continue
            norm = _normalize_market_name(market)
            if not norm:
                continue
            hash_input = (
                f"{norm}|{r.get('post_merger_hhi') or ''}|"
                f"{r.get('delta_hhi') or ''}|{r.get('acquirer_share') or ''}"
            )
            row_hash = _hashlib.md5(hash_input.encode()).hexdigest()[:8]
            payload = {
                "schema_ref": self.schema_ref,
                "market_name": market,
                "product_market": r.get("product_market"),
                "geographic_market": r.get("geographic_market") or market,
                "acquirer_share": r.get("acquirer_share"),
                "target_share": r.get("target_share"),
                "other_shares": r.get("other_shares") or [],
                "pre_merger_hhi": r.get("pre_merger_hhi"),
                "post_merger_hhi": r.get("post_merger_hhi"),
                "delta_hhi": r.get("delta_hhi"),
                "structural_presumption": bool(r.get("structural_presumption")),
                "risk_rating": r.get("risk_rating"),
                "source_detail": r.get("source_detail"),
                "source_quote": r.get("source_quote"),
                "source_document": doc_id,
            }
            out.append(TypedEvidenceWrite(
                record_kind=self.record_kind,
                record_key=f"market:{norm}:{doc_id}:{row_hash}",
                payload=payload,
                label=(
                    f"{market}: HHI {r.get('post_merger_hhi') or 'unknown'} / "
                    f"delta {r.get('delta_hhi') or 'unknown'}"
                ),
                schema_ref=self.schema_ref,
                profile_id=self.profile_id,
                document_id=doc_id,
                confidence=0.9,
                slot_key=f"collection_item:{self.schema_ref}:msa:{norm}",
            ))
        return tuple(out)

    def render_answer_rows(self, rows: Sequence[Mapping[str, Any]]) -> str:
        if not rows:
            return ""

        def _key(r: Mapping[str, Any]):
            payload = r.get("payload") or {}
            sp = 1 if payload.get("structural_presumption") else 0
            try:
                dh = int(payload.get("delta_hhi") or 0)
            except (TypeError, ValueError):
                dh = 0
            try:
                ph = int(payload.get("post_merger_hhi") or 0)
            except (TypeError, ValueError):
                ph = 0
            return (-sp, -dh, -ph, str(payload.get("market_name") or ""))

        rows_sorted = sorted(rows, key=_key)[:50]

        def _fmt(v):
            if v is None or v == "":
                return "—"
            return str(v)

        out = [
            "MARKET ROW SUMMARY (structured extracted rows; one row per market):",
            "",
            "| Market | Product | Acquirer Share | Target Share | "
            "Post-Merger HHI | Delta HHI | Presumption | Risk | Source |",
            "|---|---|---:|---:|---:|---:|---|---|---|",
        ]
        for r in rows_sorted:
            p = r.get("payload") or {}
            out.append(
                "| "
                + " | ".join([
                    _fmt(p.get("market_name")),
                    _fmt(p.get("product_market")),
                    _fmt(p.get("acquirer_share")),
                    _fmt(p.get("target_share")),
                    _fmt(p.get("post_merger_hhi")),
                    _fmt(p.get("delta_hhi")),
                    "Yes" if p.get("structural_presumption") else "No",
                    _fmt(p.get("risk_rating")),
                    _fmt(p.get("source_detail")),
                ])
                + " |"
            )
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Built-in profile: legal.cp_gap.v1
# ---------------------------------------------------------------------------


_CP_QUERY_TERMS = (
    "conditions precedent", "closing condition", "closing checklist",
    "compare closing", "borrower disclos", "compliance certificate",
    "covenant compliance", "missing cp", "missing condition",
    "closing deliverable", "credit agreement vs", "term sheet vs",
    "credit agreement against", "term sheet against",
    "compare credit agreement", "restructuring condition",
    "due diligence findings",
)


_CP_PROMPT = """
ROW-ATOMIC EXTRACTION (legal.cp_gap.v1):

For every condition precedent / closing deliverable / covenant requirement
in the source agreement, emit ONE structured "cp_gap" row indicating
whether the requirement is met, partial, or missing in the closing set or
counterparty disclosure. Do NOT collapse requirements into narrative —
each requirement is its own row.

"cp_gaps": [
    {"cp_id": "4.01(a)",
     "cp_section": "Credit Agreement Section 4.01(a)",
     "requirement_text": "Borrower must deliver executed secretary certificate.",
     "observed_evidence": "Closing set contains officer certificate but no secretary certificate.",
     "status": "met|missing|partial",
     "severity": "critical|high|medium|low",
     "required_by_document": "Credit Agreement.pdf",
     "evidence_document": "Closing Checklist.xlsx",
     "source_detail": "Section 4.01(a); checklist item 7",
     "source_quote": "optional short verbatim excerpt"}
]

CRITICAL: emit one cp_gap per requirement, even when status=met. Missing
requirements (status=missing) are the highest-value rows; never omit them.
""".strip()


def _normalize_cp_id(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\.\(\)\-]", " ", s)
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "unknown"


def _normalize_doc_name(s: str) -> str:
    s = (s or "").strip().lower()
    # Strip extension
    s = re.sub(r"\.(pdf|docx|doc|xlsx|xls|txt)$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[^\w\-]", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "unknown"


@dataclass
class LegalCpGapProfile:
    """Built-in: condition-precedent / closing-condition gap analysis."""
    profile_id: str = "legal.cp_gap.v1"
    profile_version: int = 1
    domain_profile_ids: tuple[str, ...] = ("legal:1",)
    schema_ref: str = "legal.cp_gap.v1"
    record_kind: str = "cp_gap"
    slot_kind: str = "obligation"
    priority: int = 100
    exclusive_group: Optional[str] = "obligation_compare"
    supports_mixed_corpus: bool = True
    enabled: bool = True
    prompt_token_budget: int = 450
    min_match_score: float = 0.55

    def match(self, context: SlotProfileContext) -> Optional[ProfileMatch]:
        q = (context.query or "").lower().strip()
        if not q:
            return None
        family = (context.task_spec.get("family") or "").lower()
        if family in {"inventory_lookup", "metadata_lookup"}:
            return None
        score = 0.0
        reasons: list[str] = []
        for term in _CP_QUERY_TERMS:
            if term in q:
                score = max(score, 0.65) + 0.1
                reasons.append(f"q_term:{term}")
                if score >= 1.0:
                    break
        score = min(score, 1.0)
        if score < self.min_match_score:
            return None
        return ProfileMatch(
            profile_id=self.profile_id,
            score=score,
            reasons=tuple(reasons),
        )

    def scout(
        self, context: SlotProfileContext, runtime: Any,
    ) -> Sequence[SlotRegistration]:
        """CP scouting requires a section/numbering parse — costly. For
        v1, defer to lazy-registration during deep-read parse."""
        return ()

    def prompt_addendum(self, context: SlotProfileContext) -> str:
        return _CP_PROMPT

    def parse_evidence(
        self,
        *,
        analysis: Mapping[str, Any],
        document: Any,
        context: SlotProfileContext,
    ) -> Sequence[TypedEvidenceWrite]:
        rows = analysis.get("cp_gaps")
        if not isinstance(rows, list):
            return ()
        doc_id = getattr(document, "filename", None) or "unknown"
        out: list[TypedEvidenceWrite] = []
        import hashlib as _hashlib
        for r in rows[:200]:
            if not isinstance(r, dict):
                continue
            cp_id = (r.get("cp_id") or r.get("cp_section") or "").strip()
            requirement = (r.get("requirement_text") or "").strip()
            if not cp_id and not requirement:
                continue
            required_doc = r.get("required_by_document") or doc_id
            requirement_id = cp_id or requirement[:60]
            req_norm = _normalize_cp_id(requirement_id)
            doc_norm = _normalize_doc_name(required_doc)
            slot_key = f"obligation:{self.schema_ref}:cp:{doc_norm}:{req_norm}"
            # Typed evidence key includes a generation hash so multiple
            # revisions per requirement (e.g., status flip from missing to
            # met after a closing-set update) don't collide.
            gen_input = (
                f"{r.get('status') or ''}|{r.get('observed_evidence') or ''}|"
                f"{r.get('evidence_document') or ''}|{doc_id}"
            )
            gen_id = _hashlib.md5(gen_input.encode()).hexdigest()[:10]
            evidence_key = f"cp_gap_revision:{doc_norm}:{req_norm}:{gen_id}"
            payload = {
                "schema_ref": self.schema_ref,
                "cp_id": cp_id,
                "cp_section": r.get("cp_section"),
                "requirement_text": requirement,
                "observed_evidence": r.get("observed_evidence"),
                "status": (r.get("status") or "unknown").lower(),
                "severity": (r.get("severity") or "medium").lower(),
                "required_by_document": required_doc,
                "evidence_document": r.get("evidence_document"),
                "source_document": doc_id,
                "source_detail": r.get("source_detail"),
                "source_quote": r.get("source_quote"),
            }
            label_status = payload["status"]
            short_req = (requirement or cp_id)[:80]
            out.append(TypedEvidenceWrite(
                record_kind=self.record_kind,
                record_key=evidence_key,
                payload=payload,
                label=f"[{label_status.upper()}] {cp_id or requirement[:30]}: {short_req}",
                schema_ref=self.schema_ref,
                profile_id=self.profile_id,
                document_id=doc_id,
                confidence=0.9,
                slot_key=slot_key,
                issue_link=IssueLinkSpec(
                    issue_title="Closing condition coverage",
                    predicate_description=(
                        f"{cp_id or 'requirement'}: {label_status}"
                    ),
                    relation="supports" if label_status == "met" else "attacks",
                    issue_type="diligence_red_flag",
                ),
            ))
        return tuple(out)

    def render_answer_rows(self, rows: Sequence[Mapping[str, Any]]) -> str:
        if not rows:
            return ""

        # Sort: missing first (highest severity), then partial, then met.
        # Within each, sort by severity descending.
        severity_rank = {
            "critical": 0, "high": 1, "medium": 2, "low": 3,
        }
        status_rank = {"missing": 0, "partial": 1, "met": 2, "unknown": 3}

        def _key(r: Mapping[str, Any]):
            p = r.get("payload") or {}
            return (
                status_rank.get(str(p.get("status") or "unknown").lower(), 3),
                severity_rank.get(str(p.get("severity") or "medium").lower(), 2),
                str(p.get("cp_id") or ""),
            )

        rows_sorted = sorted(rows, key=_key)[:80]

        def _fmt(v):
            if v is None or v == "":
                return "—"
            return str(v)

        out = [
            "CP GAP SUMMARY (one row per condition precedent / requirement):",
            "",
            "| CP | Requirement | Observed Evidence | Status | Severity | Required Doc | Evidence Doc |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in rows_sorted:
            p = r.get("payload") or {}
            req = (p.get("requirement_text") or "")[:120]
            obs = (p.get("observed_evidence") or "")[:120]
            out.append(
                "| " + " | ".join([
                    _fmt(p.get("cp_id") or p.get("cp_section")),
                    _fmt(req) if req else "—",
                    _fmt(obs) if obs else "—",
                    _fmt(p.get("status")),
                    _fmt(p.get("severity")),
                    _fmt(p.get("required_by_document")),
                    _fmt(p.get("evidence_document")),
                ]) + " |"
            )
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Built-in profile: legal.qoe_line_item.v1
# ---------------------------------------------------------------------------


_QOE_QUERY_TERMS = (
    "qoe", "quality of earnings", "ebitda bridge",
    "working capital reconciliation", "nwc reconciliation",
    "ppa reconciliation", "ppa allocation", "purchase price allocation",
    "qoe reconciliation", "ebitda reconciliation",
)

_QOE_PROMPT = """
ROW-ATOMIC EXTRACTION (legal.qoe_line_item.v1):

For every financial adjustment / reconciliation line item in the QoE
documents, emit ONE structured "qoe_line_items" row. Preserve dollar
amounts exactly. Do NOT summarize bridges into prose; each line item is
its own row.

"qoe_line_items": [
    {"category": "EBITDA bridge",
     "line_item_label": "Owner compensation normalization",
     "period": "FY2024",
     "schedule": "EBITDA Bridge Reconciliation",
     "currency": "USD",
     "seller_value": "$0.8M",
     "buyer_value": "$1.3M",
     "delta": "-$0.5M",
     "recommended_value": "$1.3M",
     "unit": "USD millions",
     "source_detail": "EBITDA bridge schedule, row 12",
     "source_quote": "optional short verbatim excerpt"}
]

CRITICAL: every adjustment is a row. Owner comp, legal settlement,
inventory write-down, rent normalization, etc. — each as its own line.
""".strip()


def _normalize_qoe_token(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "unknown"


@dataclass
class LegalQoeLineItemProfile:
    """Built-in: QoE / EBITDA bridge / reconciliation line items."""
    profile_id: str = "legal.qoe_line_item.v1"
    profile_version: int = 1
    domain_profile_ids: tuple[str, ...] = ("legal:1", "finance:1")
    schema_ref: str = "legal.qoe_line_item.v1"
    record_kind: str = "qoe_line_item"
    slot_kind: str = "measurement"
    priority: int = 100
    exclusive_group: Optional[str] = "financial_reconciliation"
    supports_mixed_corpus: bool = True
    enabled: bool = True
    prompt_token_budget: int = 450
    min_match_score: float = 0.55

    def match(self, context: SlotProfileContext) -> Optional[ProfileMatch]:
        q = (context.query or "").lower().strip()
        if not q:
            return None
        family = (context.task_spec.get("family") or "").lower()
        if family in {"inventory_lookup", "metadata_lookup"}:
            return None
        score = 0.0
        reasons: list[str] = []
        for term in _QOE_QUERY_TERMS:
            if term in q:
                score = max(score, 0.65) + 0.1
                reasons.append(f"q_term:{term}")
                if score >= 1.0:
                    break
        score = min(score, 1.0)
        if score < self.min_match_score:
            return None
        return ProfileMatch(
            profile_id=self.profile_id,
            score=score,
            reasons=tuple(reasons),
        )

    def scout(
        self, context: SlotProfileContext, runtime: Any,
    ) -> Sequence[SlotRegistration]:
        # QoE expected counts come from the workbook tab structure and
        # adjustments-list layout. v1 defers to lazy-registration; the
        # scout layer can be added when we have per-tab parsing.
        return ()

    def prompt_addendum(self, context: SlotProfileContext) -> str:
        return _QOE_PROMPT

    def parse_evidence(
        self,
        *,
        analysis: Mapping[str, Any],
        document: Any,
        context: SlotProfileContext,
    ) -> Sequence[TypedEvidenceWrite]:
        rows = analysis.get("qoe_line_items")
        if not isinstance(rows, list):
            return ()
        doc_id = getattr(document, "filename", None) or "unknown"
        artifact_family = getattr(document, "family_id", None) or doc_id
        out: list[TypedEvidenceWrite] = []
        import hashlib as _hashlib
        for r in rows[:200]:
            if not isinstance(r, dict):
                continue
            label = (r.get("line_item_label") or "").strip()
            category = (r.get("category") or "").strip()
            period = (r.get("period") or "").strip()
            if not label and not category:
                continue
            schedule = (r.get("schedule") or category or "").strip()
            currency = (r.get("currency") or r.get("unit") or "USD").strip()
            af_norm = _normalize_qoe_token(artifact_family)
            sched_norm = _normalize_qoe_token(schedule)
            curr_norm = _normalize_qoe_token(currency)
            cat_norm = _normalize_qoe_token(category)
            period_norm = _normalize_qoe_token(period)
            label_norm = _normalize_qoe_token(label)
            slot_key = (
                f"measurement:{self.schema_ref}:"
                f"{af_norm}:{sched_norm}:{curr_norm}:"
                f"{cat_norm}:{period_norm}:{label_norm}"
            )
            row_hash_input = (
                f"{r.get('seller_value') or ''}|{r.get('buyer_value') or ''}|"
                f"{r.get('recommended_value') or ''}|{r.get('delta') or ''}"
            )
            row_hash = _hashlib.md5(row_hash_input.encode()).hexdigest()[:8]
            evidence_key = (
                f"qoe_line:{af_norm}:{sched_norm}:{curr_norm}:"
                f"{cat_norm}:{period_norm}:{label_norm}:{doc_id}:{row_hash}"
            )
            payload = {
                "schema_ref": self.schema_ref,
                "category": category,
                "line_item_label": label,
                "period": period,
                "schedule": schedule,
                "currency": currency,
                "seller_value": r.get("seller_value"),
                "buyer_value": r.get("buyer_value"),
                "delta": r.get("delta"),
                "recommended_value": r.get("recommended_value"),
                "unit": r.get("unit") or currency,
                "source_document": doc_id,
                "source_detail": r.get("source_detail"),
                "source_quote": r.get("source_quote"),
            }
            out.append(TypedEvidenceWrite(
                record_kind=self.record_kind,
                record_key=evidence_key,
                payload=payload,
                label=f"{category} | {period} | {label}: delta={r.get('delta') or '—'}",
                schema_ref=self.schema_ref,
                profile_id=self.profile_id,
                document_id=doc_id,
                confidence=0.9,
                slot_key=slot_key,
                issue_link=IssueLinkSpec(
                    issue_title="Purchase price reconciliation",
                    predicate_description=f"{category} {period}: {label}",
                    relation="supports",
                    issue_type="financial_reconciliation",
                ),
            ))
        return tuple(out)

    def render_answer_rows(self, rows: Sequence[Mapping[str, Any]]) -> str:
        if not rows:
            return ""

        def _key(r: Mapping[str, Any]):
            p = r.get("payload") or {}
            return (
                str(p.get("category") or ""),
                str(p.get("period") or ""),
                str(p.get("line_item_label") or ""),
            )

        rows_sorted = sorted(rows, key=_key)[:120]

        def _fmt(v):
            if v is None or v == "":
                return "—"
            return str(v)

        out = [
            "QOE LINE ITEM SUMMARY (one row per adjustment):",
            "",
            "| Category | Period | Line Item | Seller | Buyer | Delta | Recommended | Source |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in rows_sorted:
            p = r.get("payload") or {}
            out.append(
                "| " + " | ".join([
                    _fmt(p.get("category")),
                    _fmt(p.get("period")),
                    _fmt(p.get("line_item_label")),
                    _fmt(p.get("seller_value")),
                    _fmt(p.get("buyer_value")),
                    _fmt(p.get("delta")),
                    _fmt(p.get("recommended_value")),
                    _fmt(p.get("source_detail") or p.get("source_document")),
                ]) + " |"
            )
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Default registry factory
# ---------------------------------------------------------------------------


def default_registry() -> SlotProfileRegistry:
    """Registry with built-in profiles installed."""
    return SlotProfileRegistry(profiles=(
        LegalMarketRowProfile(),
        LegalCpGapProfile(),
        LegalQoeLineItemProfile(),
    ))
