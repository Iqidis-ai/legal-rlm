"""Structured Term Grid Extractor — pre-synthesis source-grounded operator.

Operator Substrate Thesis: Sub-agents are bounded operators, not mini
chatbots. This one turns relevant document sections into typed
`term_grid.v1` rows and `conditional_rule_tree.v1` artifacts so the
synthesis LLM receives a clean, normalized view of obligations, rights,
triggers, conditions, exceptions, thresholds, and consequences — instead
of having to re-extract them from raw prose every time.

Anti-Gaming Gate: this operator works on a fuzzy user prompt with no
benchmark rubric attached. Inputs are user query + TaskSpec + section
maps + source spans. NEVER reads benchmark task IDs, scoring keys, or
rubric text. LAB-supplied criteria (when present) may help narrow
section selection as a fast path, but are NOT required.

Minimize-Post-Synthesis: this is a PRE-SYNTHESIS INGREDIENT operator.
It improves the input synthesis sees, so the first synthesis call is
better — instead of running synthesis, then verifying output, then
re-running. Cost: 0 LLM calls when no candidate sections; otherwise
bounded JSON-mode calls (capped per matter).

Cross-domain: legal covenants/CPs/exceptions; finance covenants/waterfalls;
coding policy rules/acceptance criteria; research inclusion/exclusion
criteria; biomedical eligibility/dose/endpoint conditions. All share
the underlying conditional structure (actor + trigger + condition +
consequence).
"""

from __future__ import annotations

import asyncio as _asyncio
import hashlib as _hashlib
import json as _json
import logging as _logging
import re as _re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from .contracts import (
    AgentArtifact,
    AgentInvocation,
    AgentInvocationResult,
    AgentMatch,
    AgentRequirement,
)

_logger = _logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — locked schema vocabulary
# ---------------------------------------------------------------------------


SCHEMA_REF_TERM_GRID = "term_grid.v1"
SCHEMA_REF_CONDITIONAL_RULE_TREE = "conditional_rule_tree.v1"
ARTIFACT_KIND_TERM_GRID = "term_grid.v1"
ARTIFACT_KIND_CONDITIONAL_RULE_TREE = "conditional_rule_tree.v1"

# Match() keyword triggers — phrases users actually type when they want
# typed clause/term/rule extraction. Tight phrase match (anti-noise per
# v9 lesson on bare keywords).
_TERM_INTENT_PHRASES = (
    "extract", "extract the", "list the", "summarize the",
    "pull out", "pull the", "identify the", "compare the",
    "review the", "analyze the", "verify the",
)
_TERM_OBJECT_KEYWORDS = (
    "covenant", "covenants", "provision", "provisions", "clause", "clauses",
    "term", "terms", "condition", "conditions", "exception", "exceptions",
    "obligation", "obligations", "right", "rights", "trigger", "triggers",
    "deadline", "deadlines", "requirement", "requirements", "rule", "rules",
    "policy", "policies", "criteria", "criterion", "restriction", "restrictions",
    "covenant package", "compliance", "waterfall", "standstill",
    "indemnity", "indemnities", "warranty", "warranties", "rep", "reps",
    "termination", "amendment", "consent", "approval", "veto",
)


# Confidence threshold below which rows are dropped
_MIN_ROW_CONFIDENCE = 0.5

# Section selection caps
_MAX_SECTIONS_PER_BATCH = 8
_MAX_BATCHES_PER_INVOCATION = 3
_MAX_SECTION_CHARS = 6000  # per section span sent to LLM


# ---------------------------------------------------------------------------
# Helpers — IDs, normalization, JSON parsing
# ---------------------------------------------------------------------------


def _canonical_json(obj: Any) -> str:
    return _json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _sha24(s: str) -> str:
    return _hashlib.sha256(s.encode("utf-8")).hexdigest()[:24]


def _normalize_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def make_term_row_id(
    *,
    document_id: str,
    section_ref: str,
    topic: str,
    actor: str,
    obligation_or_right: str,
) -> str:
    """Deterministic row id = hash of source identity.

    Excludes any benchmark task IDs or criterion IDs (anti-gaming).
    Stable across reruns of the same source.
    """
    canonical = _canonical_json({
        "schema_ref": SCHEMA_REF_TERM_GRID,
        "document_id": document_id,
        "section_ref": section_ref,
        "topic": _normalize_text(topic)[:80],
        "actor": _normalize_text(actor)[:80],
        "obligation_or_right": _normalize_text(obligation_or_right)[:120],
    })
    return f"termrow_v1:{_sha24(canonical)}"


def make_rule_id(
    *,
    document_id: str,
    section_ref: str,
    if_clauses: Sequence[str],
    then_clauses: Sequence[str],
) -> str:
    canonical = _canonical_json({
        "schema_ref": SCHEMA_REF_CONDITIONAL_RULE_TREE,
        "document_id": document_id,
        "section_ref": section_ref,
        "if": [_normalize_text(c)[:80] for c in if_clauses][:5],
        "then": [_normalize_text(c)[:80] for c in then_clauses][:5],
    })
    return f"rule_v1:{_sha24(canonical)}"


def _strip_json_fences(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    return s


def _parse_extraction_json(raw_text: str) -> Optional[dict]:
    s = _strip_json_fences(raw_text or "")
    try:
        obj = _json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = _re.search(r"\{[\s\S]*\}", s)
    if m:
        try:
            obj = _json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Section selection
# ---------------------------------------------------------------------------


_TERM_SECTION_HEADER_KEYWORDS = (
    "covenant", "obligation", "condition", "right", "duty",
    "termination", "default", "remedy", "indemnif", "warranty",
    "representation", "exception", "waterfall", "standstill",
    "priority", "intercreditor", "subordination", "amendment",
    "consent", "approval", "veto", "cure", "notice",
    "distribution", "trustee", "fiduciary", "compliance",
    "restrict", "limitation", "prohibition", "permit",
    "schedule", "exhibit", "trigger", "deadline",
)


def _section_relevance_score(section: dict, query_keywords: set[str]) -> float:
    """Score a section for term-grid relevance.

    0.0-1.0 score based on:
    - Header text matching term-grid keywords (clauses, covenants, etc.)
    - Header text matching the user query keywords
    - Section length (very short headers/sections deprioritized)
    """
    title = str(section.get("title") or section.get("label") or "").lower()
    if not title:
        return 0.0
    score = 0.0
    for kw in _TERM_SECTION_HEADER_KEYWORDS:
        if kw in title:
            score += 0.20
    for q in query_keywords:
        if q in title and len(q) >= 4:
            score += 0.30
    return min(score, 1.0)


def _query_keywords(query: str) -> set[str]:
    text = (query or "").lower()
    # Extract content words longer than 3 chars; drop common stop terms
    stop = {
        "the", "and", "for", "with", "this", "that", "have", "what", "when",
        "where", "which", "from", "into", "your", "their", "between",
        "should", "would", "could", "above", "below", "about", "draft",
        "summarize", "extract", "review", "analyze", "compare", "list",
    }
    out: set[str] = set()
    for tok in _re.findall(r"[a-zA-Z]{4,}", text):
        if tok not in stop:
            out.add(tok)
    return out


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------


_TERM_GRID_PROMPT_TEMPLATE = """You are the structured term grid extractor inside a bounded operator. Your only job is to read the source sections below and emit normalized clause/rule rows. You are NOT a chatbot, NOT a synthesis model, and NOT a rubric reader.

For each obligation, right, trigger, condition, exception, threshold, deadline, or consequence in the source sections, emit ONE row in `term_grid.rows`. For each conditional structure (if/then/unless/timing/threshold), emit ONE rule in `rules` with the source span IDs that justify it.

NEVER:
- Invent facts not present in the source.
- Reference scoring rubrics, benchmark IDs, criterion IDs, or expected answers (none exist; this is a real user request).
- Emit prose explanations or commentary.
- Include rows you cannot ground in a specific section.

Severity / role notes are out of scope here — this operator extracts source-grounded facts, not coverage judgment.

Return ONLY a single JSON object matching this schema (no commentary, no markdown fences):
{{
  "schema_ref": "term_grid.v1",
  "rows": [
    {{
      "document_id": "string",
      "section_ref": "string (e.g. 'Section 5.3' or '## Heading')",
      "topic": "short slug, e.g. 'standstill_period'",
      "actor": "the party / entity bound or empowered",
      "obligation_or_right": "what the actor must / may do",
      "trigger": "what event or condition causes this",
      "condition": "additional preconditions",
      "exception": "carve-outs or exclusions",
      "threshold": "numeric or qualitative threshold (or null)",
      "amount": "amount with units (or null)",
      "date_or_period": "duration / deadline / period (or null)",
      "consequence": "what happens when triggered",
      "source_refs": ["span ids or section refs"],
      "confidence": 0.0
    }}
  ],
  "rules": [
    {{
      "section_ref": "string",
      "if": ["clause"],
      "then": ["consequence"],
      "unless": ["exception"],
      "timing": "string (or null)",
      "thresholds": ["string"],
      "parties": ["actor"],
      "source_span_ids": ["span ids"],
      "confidence": 0.0
    }}
  ]
}}

Granularity rules:
- Prefer 0-12 rows per call. Quality over quantity.
- One row per (actor, obligation/right) pair. Combine identical conditions/exceptions; do NOT duplicate.
- A row's `confidence` should reflect how clearly the source supports it (>=0.7 for explicit clauses, 0.5-0.7 for implied, <0.5 means drop the row).
- Rules are for explicit if/then patterns. Plain narrative obligations belong in `rows`, not `rules`.

USER QUERY (for relevance, not for criteria invention):
{user_query}

DOMAIN: {domain}

SOURCE SECTIONS:
{sections_json}

Return ONLY the JSON object."""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


@dataclass
class StructuredTermGridExtractor:
    """Pre-synthesis source-grounded clause / rule extractor.

    Cross-domain: emits typed obligation/right/trigger/condition/exception
    rows that synthesis can render directly without re-extracting from
    raw prose. Bounded-LLM (one JSON-mode call per section batch, capped).
    """

    agent_id: str = "structured_term_grid_extractor.v1"
    version: int = 1
    enabled: bool = True
    priority: int = 80
    capability_tags: tuple[str, ...] = (
        "clause.extract",
        "term_grid.build",
        "rule.conditional",
        "requirement.extract",
        "threshold.normalize",
        "source_answer_ingredient",
    )
    supported_domain_profiles: tuple[str, ...] = (
        "legal:1", "finance:1", "coding:1",
        "academic_research:1", "biomedical:1",
    )
    phases: tuple[str, ...] = ("pre_synthesis",)
    exclusive_group: Optional[str] = "source_semantic_extraction"
    deterministic: bool = False  # bounded LLM step

    # Tunable knobs (kept as instance attributes for test override)
    max_sections_per_batch: int = _MAX_SECTIONS_PER_BATCH
    max_batches_per_invocation: int = _MAX_BATCHES_PER_INVOCATION
    max_section_chars: int = _MAX_SECTION_CHARS
    min_row_confidence: float = _MIN_ROW_CONFIDENCE

    # ------------------------------------------------------------------
    # match
    # ------------------------------------------------------------------

    def match(self, invocation: AgentInvocation) -> Optional[AgentMatch]:
        family = (invocation.execution_family or "").lower()
        # Only relevant for tasks that consume documents — read/query/
        # clarify with no docs are out.
        if family in {"clarify", "scenario", "steer"}:
            return None

        wp = invocation.work_profile or {}
        n_section_maps = int(wp.get("document_section_map_count", -1))
        n_obligation_rows = int(wp.get("obligation_row_count", -1))
        n_term_grid_oblig = int(wp.get("term_grid_obligation_count", -1))
        n_contract_provisions = int(wp.get("contract_provision_count", -1))

        # Strong signal: an obligation row explicitly expects a term_grid
        # artifact, OR there are existing contract_provision typed evidence.
        if n_term_grid_oblig > 0 or n_contract_provisions > 0:
            return AgentMatch(
                agent_id=self.agent_id, score=0.95,
                reasons=(
                    f"term_grid_obligations:{n_term_grid_oblig},"
                    f"contract_provisions:{n_contract_provisions}",
                ),
                requirement=AgentRequirement.REQUIRED,
                phase="pre_synthesis",
            )

        # Medium signal: section maps exist AND user query matches
        # term-extraction intent.
        intent_match = self._query_indicates_term_intent(invocation)
        if n_section_maps > 0 and intent_match:
            return AgentMatch(
                agent_id=self.agent_id, score=0.85,
                reasons=(f"section_maps:{n_section_maps},intent_match",),
                requirement=AgentRequirement.REQUIRED,
                phase="pre_synthesis",
            )

        # Weak signal: section maps exist + obligation rows expect
        # something (could be related). Run as optional.
        if n_section_maps > 0 and n_obligation_rows > 0:
            return AgentMatch(
                agent_id=self.agent_id, score=0.55,
                reasons=(f"section_maps:{n_section_maps},obligations_present",),
                requirement=AgentRequirement.OPTIONAL,
                phase="pre_synthesis",
            )

        # No signal — skip
        return None

    @staticmethod
    def _query_indicates_term_intent(invocation: AgentInvocation) -> bool:
        wp = invocation.work_profile or {}
        # Engine populates `query_text_normalized` for keyword scanning;
        # fall back to invocation.input_hash as a degraded source.
        q = str(wp.get("query_text_normalized") or "")
        if not q:
            return False
        # Match phrases like "extract the X" where X is a term-shaped object.
        has_intent_verb = any(p in q for p in _TERM_INTENT_PHRASES)
        has_object = any(k in q for k in _TERM_OBJECT_KEYWORDS)
        return has_intent_verb and has_object

    # ------------------------------------------------------------------
    # invoke
    # ------------------------------------------------------------------

    async def invoke(
        self,
        invocation: AgentInvocation,
        runtime: Any,
    ) -> AgentInvocationResult:
        import time as _time
        t0 = _time.perf_counter()
        warnings: list[str] = []

        llm_client = getattr(runtime, "llm_client", None)
        if llm_client is None:
            warnings.append("no_llm_client")
            return AgentInvocationResult(
                status="success",
                elapsed_ms=int((_time.perf_counter() - t0) * 1000),
                warnings=tuple(warnings),
            )

        # Load section maps + source documents
        try:
            sections = await _asyncio.to_thread(
                self._load_candidate_sections, runtime, invocation, warnings,
            )
        except Exception as exc:
            return AgentInvocationResult(
                status="error",
                error_class=type(exc).__name__,
                error=str(exc)[:300],
                elapsed_ms=int((_time.perf_counter() - t0) * 1000),
            )

        if not sections:
            warnings.append("no_candidate_sections")
            return AgentInvocationResult(
                status="success",
                elapsed_ms=int((_time.perf_counter() - t0) * 1000),
                warnings=tuple(warnings),
            )

        # Score + rank sections
        query_kw = _query_keywords(self._user_query(invocation, runtime))
        scored = [
            (s, _section_relevance_score(s, query_kw))
            for s in sections
        ]
        scored = [(s, sc) for s, sc in scored if sc > 0.0]
        scored.sort(key=lambda x: -x[1])
        if not scored:
            warnings.append("no_relevant_sections")
            return AgentInvocationResult(
                status="success",
                elapsed_ms=int((_time.perf_counter() - t0) * 1000),
                warnings=tuple(warnings),
            )

        # Batch into LLM calls
        batches = self._make_batches(
            [s for s, _ in scored],
            max_batches=self.max_batches_per_invocation,
            max_per_batch=self.max_sections_per_batch,
        )

        domain = self._domain_label(invocation, runtime)
        all_rows: list[dict] = []
        all_rules: list[dict] = []
        llm_calls = 0
        token_estimate = 0
        for batch in batches:
            try:
                batch_result, used_tokens = await self._extract_batch(
                    llm_client=llm_client, user_query=self._user_query(invocation, runtime),
                    domain=domain, sections=batch,
                )
            except Exception as exc:
                warnings.append(f"batch_extract_failed:{type(exc).__name__}")
                llm_calls += 1
                continue
            llm_calls += 1
            token_estimate += used_tokens
            if batch_result is None:
                warnings.append("batch_unparseable")
                continue
            for r in batch_result.get("rows") or []:
                if isinstance(r, dict):
                    all_rows.append(r)
            for r in batch_result.get("rules") or []:
                if isinstance(r, dict):
                    all_rules.append(r)

        # Normalize, dedupe, filter by confidence
        rows = self._normalize_rows(all_rows)
        rules = self._normalize_rules(all_rules)

        if not rows and not rules:
            warnings.append("no_extractable_terms")
            return AgentInvocationResult(
                status="success",
                elapsed_ms=int((_time.perf_counter() - t0) * 1000),
                warnings=tuple(warnings),
                llm_calls=llm_calls,
                token_estimate=token_estimate,
            )

        # Build artifacts
        artifacts: list[AgentArtifact] = []
        if rows:
            artifacts.append(self._build_term_grid_artifact(
                rows=rows,
                run_id=invocation.run_id,
            ))
        if rules:
            artifacts.append(self._build_rule_tree_artifact(
                rules=rules,
                run_id=invocation.run_id,
            ))

        return AgentInvocationResult(
            status="success",
            artifacts=tuple(artifacts),
            warnings=tuple(warnings),
            elapsed_ms=int((_time.perf_counter() - t0) * 1000),
            llm_calls=llm_calls,
            token_estimate=token_estimate,
        )

    # ------------------------------------------------------------------
    # verify_output
    # ------------------------------------------------------------------

    def verify_output(
        self,
        invocation: AgentInvocation,
        result: AgentInvocationResult,
    ) -> AgentInvocationResult:
        if result.status != "success":
            return result
        for a in result.artifacts:
            p = a.payload or {}
            if a.artifact_kind == ARTIFACT_KIND_TERM_GRID:
                if "rows" not in p:
                    return AgentInvocationResult(
                        status="invalid",
                        error_class="MalformedTermGrid",
                        error=f"artifact {a.artifact_key} missing rows",
                        elapsed_ms=result.elapsed_ms,
                        warnings=result.warnings,
                        artifacts=result.artifacts,
                        llm_calls=result.llm_calls,
                        token_estimate=result.token_estimate,
                    )
            elif a.artifact_kind == ARTIFACT_KIND_CONDITIONAL_RULE_TREE:
                if "rules" not in p:
                    return AgentInvocationResult(
                        status="invalid",
                        error_class="MalformedRuleTree",
                        error=f"artifact {a.artifact_key} missing rules",
                        elapsed_ms=result.elapsed_ms,
                        warnings=result.warnings,
                        artifacts=result.artifacts,
                        llm_calls=result.llm_calls,
                        token_estimate=result.token_estimate,
                    )
        return result

    # ------------------------------------------------------------------
    # Internals — section loading + scoring
    # ------------------------------------------------------------------

    def _user_query(self, invocation: AgentInvocation, runtime: Any) -> str:
        try:
            state = getattr(runtime, "state", None)
            if state is not None:
                q = getattr(state, "query", None)
                if q:
                    return str(q)
        except Exception:
            pass
        return str(invocation.input_hash or "")

    def _domain_label(self, invocation: AgentInvocation, runtime: Any) -> str:
        pid = invocation.domain_profile_id or ""
        for prefix, lbl in (
            ("legal:", "legal"),
            ("finance:", "finance"),
            ("coding:", "coding"),
            ("academic_research:", "academic_research"),
            ("biomedical:", "biomedical"),
        ):
            if pid.startswith(prefix):
                return lbl
        return "legal"

    def _load_candidate_sections(
        self, runtime: Any, invocation: AgentInvocation, warnings: list[str],
    ) -> list[dict]:
        """Pull section_map artifacts produced by DocumentFileReader."""
        mm = getattr(runtime, "matter_model", None)
        if mm is None:
            return []
        try:
            rows = mm.db.execute(
                "SELECT artifact_key, payload_json FROM agent_artifact "
                "WHERE matter_id=? AND artifact_kind=?",
                (mm.matter_id, "document.section_map"),
            ).fetchall()
        except Exception as exc:
            warnings.append(f"section_map_load_failed:{type(exc).__name__}")
            return []
        sections: list[dict] = []
        for r in rows:
            try:
                payload = _json.loads(r["payload_json"] or "{}")
            except Exception:
                continue
            doc_id = payload.get("document_id") or r["artifact_key"]
            for s in payload.get("sections") or []:
                if not isinstance(s, dict):
                    continue
                s2 = dict(s)
                s2.setdefault("document_id", doc_id)
                sections.append(s2)
        return sections

    @staticmethod
    def _make_batches(
        items: Sequence[dict], *,
        max_batches: int, max_per_batch: int,
    ) -> list[list[dict]]:
        out: list[list[dict]] = []
        cur: list[dict] = []
        for it in items:
            cur.append(it)
            if len(cur) >= max_per_batch:
                out.append(cur)
                cur = []
                if len(out) >= max_batches:
                    return out
        if cur and len(out) < max_batches:
            out.append(cur)
        return out

    # ------------------------------------------------------------------
    # LLM extraction
    # ------------------------------------------------------------------

    async def _extract_batch(
        self, *,
        llm_client: Any,
        user_query: str,
        domain: str,
        sections: list[dict],
    ) -> tuple[Optional[dict], int]:
        """Call the LLM for one section batch. Returns (parsed, tokens)."""
        sections_payload = []
        for s in sections:
            sections_payload.append({
                "document_id": s.get("document_id", ""),
                "section_ref": s.get("title") or s.get("label") or s.get("ref") or "",
                "text": (s.get("text") or s.get("body") or "")[:self.max_section_chars],
            })
        prompt = _TERM_GRID_PROMPT_TEMPLATE.format(
            user_query=(user_query or "")[:2000],
            domain=domain,
            sections_json=_json.dumps(sections_payload, indent=2),
        )
        try:
            from ...core.models import ModelTier as _ModelTier
            response = await llm_client.complete(
                prompt=prompt,
                tier=_ModelTier.FLASH,
                timeout=45.0,
                usage_label="term_grid_extraction",
                json_mode=True,
            )
        except TypeError:
            response = await llm_client.complete(
                prompt=prompt,
                tier=_ModelTier.FLASH,
                timeout=45.0,
                usage_label="term_grid_extraction",
            )
        parsed = _parse_extraction_json(response or "")
        return parsed, len(response or "") // 4

    # ------------------------------------------------------------------
    # Normalize / dedupe
    # ------------------------------------------------------------------

    def _normalize_rows(self, raw_rows: list[dict]) -> list[dict]:
        seen: dict[str, dict] = {}
        for r in raw_rows:
            try:
                conf = float(r.get("confidence") or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
            if conf < self.min_row_confidence:
                continue
            doc_id = str(r.get("document_id") or "")
            section_ref = str(r.get("section_ref") or "")
            topic = str(r.get("topic") or "")
            actor = str(r.get("actor") or "")
            obligation = str(r.get("obligation_or_right") or "")
            if not (section_ref and (topic or obligation)):
                continue
            row_id = make_term_row_id(
                document_id=doc_id, section_ref=section_ref,
                topic=topic, actor=actor,
                obligation_or_right=obligation,
            )
            if row_id in seen:
                # Keep highest confidence
                if conf > float(seen[row_id].get("confidence") or 0.0):
                    seen[row_id] = self._row_payload(r, row_id, conf)
                continue
            seen[row_id] = self._row_payload(r, row_id, conf)
        return list(seen.values())

    @staticmethod
    def _row_payload(r: dict, row_id: str, conf: float) -> dict:
        return {
            "row_id": row_id,
            "document_id": str(r.get("document_id") or ""),
            "section_ref": str(r.get("section_ref") or ""),
            "topic": str(r.get("topic") or "")[:80],
            "actor": str(r.get("actor") or "")[:160],
            "obligation_or_right": str(r.get("obligation_or_right") or "")[:400],
            "trigger": str(r.get("trigger") or "")[:300],
            "condition": str(r.get("condition") or "")[:300],
            "exception": str(r.get("exception") or "")[:300],
            "threshold": r.get("threshold") if r.get("threshold") not in (None, "") else None,
            "amount": r.get("amount") if r.get("amount") not in (None, "") else None,
            "date_or_period": r.get("date_or_period") if r.get("date_or_period") not in (None, "") else None,
            "consequence": str(r.get("consequence") or "")[:300],
            "source_refs": list(r.get("source_refs") or [])[:8],
            "confidence": conf,
        }

    def _normalize_rules(self, raw_rules: list[dict]) -> list[dict]:
        seen: dict[str, dict] = {}
        for r in raw_rules:
            try:
                conf = float(r.get("confidence") or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
            if conf < self.min_row_confidence:
                continue
            doc_id = str(r.get("document_id") or "")
            section_ref = str(r.get("section_ref") or "")
            if_clauses = list(r.get("if") or [])
            then_clauses = list(r.get("then") or [])
            if not section_ref or not (if_clauses and then_clauses):
                continue
            rule_id = make_rule_id(
                document_id=doc_id, section_ref=section_ref,
                if_clauses=if_clauses, then_clauses=then_clauses,
            )
            if rule_id in seen:
                if conf > float(seen[rule_id].get("confidence") or 0.0):
                    seen[rule_id] = self._rule_payload(r, rule_id, conf)
                continue
            seen[rule_id] = self._rule_payload(r, rule_id, conf)
        return list(seen.values())

    @staticmethod
    def _rule_payload(r: dict, rule_id: str, conf: float) -> dict:
        return {
            "rule_id": rule_id,
            "document_id": str(r.get("document_id") or ""),
            "section_ref": str(r.get("section_ref") or ""),
            "if": list(r.get("if") or [])[:5],
            "then": list(r.get("then") or [])[:5],
            "unless": list(r.get("unless") or [])[:5],
            "timing": r.get("timing"),
            "thresholds": list(r.get("thresholds") or [])[:5],
            "parties": list(r.get("parties") or [])[:5],
            "source_span_ids": list(r.get("source_span_ids") or [])[:8],
            "confidence": conf,
        }

    # ------------------------------------------------------------------
    # Artifact builders
    # ------------------------------------------------------------------

    def _build_term_grid_artifact(
        self, rows: list[dict], run_id: str,
    ) -> AgentArtifact:
        # Group rows by topic for a stable artifact key
        topics = sorted({r.get("topic", "") for r in rows if r.get("topic")})
        topic_summary = ",".join(topics[:4]) or "term_grid"
        artifact_key = f"term_grid:{_sha24(topic_summary)}:{run_id}"
        payload = {
            "schema_ref": SCHEMA_REF_TERM_GRID,
            "run_id": run_id,
            "render_policy": "source_answer_ingredient_only",
            "n_rows": len(rows),
            "topics": topics,
            "rows": rows,
        }
        label = f"Term grid: {len(rows)} rows across {len(topics)} topics"
        return AgentArtifact(
            artifact_kind=ARTIFACT_KIND_TERM_GRID,
            artifact_key=artifact_key,
            payload=payload,
            label=label,
            synthesis_visibility="answer_ingredient",
            confidence=min(
                1.0,
                sum(float(r.get("confidence") or 0.0) for r in rows) / max(1, len(rows)),
            ),
            verification_state="verified",
        )

    def _build_rule_tree_artifact(
        self, rules: list[dict], run_id: str,
    ) -> AgentArtifact:
        artifact_key = f"rule_tree:{_sha24(str(len(rules)))}:{run_id}"
        payload = {
            "schema_ref": SCHEMA_REF_CONDITIONAL_RULE_TREE,
            "run_id": run_id,
            "render_policy": "source_answer_ingredient_only",
            "n_rules": len(rules),
            "rules": rules,
        }
        label = f"Conditional rule tree: {len(rules)} rules"
        return AgentArtifact(
            artifact_kind=ARTIFACT_KIND_CONDITIONAL_RULE_TREE,
            artifact_key=artifact_key,
            payload=payload,
            label=label,
            synthesis_visibility="answer_ingredient",
            confidence=min(
                1.0,
                sum(float(r.get("confidence") or 0.0) for r in rules) / max(1, len(rules)),
            ),
            verification_state="verified",
        )
