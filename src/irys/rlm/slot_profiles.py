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
# Default registry factory
# ---------------------------------------------------------------------------


def default_registry() -> SlotProfileRegistry:
    """Registry with built-in profiles installed."""
    return SlotProfileRegistry(profiles=(LegalMarketRowProfile(),))
