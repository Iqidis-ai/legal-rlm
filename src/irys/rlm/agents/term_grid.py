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


def _coerce_str_list(value: Any, *, max_len: int = 16, max_str: int = 200) -> list[str]:
    """Defensively coerce arbitrary LLM output into a clean str list.

    Reviewer non-blocker fix: previously `list(value)` on a string would
    yield a character array. This helper handles strings, sequences,
    and noise gracefully.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value[:max_str]] if value.strip() else []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value[:max_len]:
            if isinstance(item, str) and item.strip():
                out.append(item[:max_str])
            elif item is not None:
                s = str(item)[:max_str]
                if s.strip():
                    out.append(s)
        return out
    return []


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

    Reviewer round 1 blocker B4: previously only scored title text. Now
    scans body text too so a section titled bare "Section 5.3" with a
    body about restrictions/deadlines/exceptions is not dropped.

    Scoring sources:
    - Title matches (heavy weight per match — title hits are reliable)
    - Body matches on term-grid keywords AND query keywords
    - Source-kind boost: schedule_index / table_index / contract_provision
      sections are pre-filtered for relevance and start with a positive prior
    """
    title = str(section.get("title") or section.get("label") or "").lower()
    body = str(section.get("text") or section.get("body") or "").lower()
    if not (title or body):
        return 0.0
    score = 0.0
    # Title matches: high weight per hit
    for kw in _TERM_SECTION_HEADER_KEYWORDS:
        if kw in title:
            score += 0.20
    for q in query_keywords:
        if q in title and len(q) >= 4:
            score += 0.30

    # Body matches: lower per-hit weight but cumulative; prefer sections
    # whose body contains *multiple* term-grid signals.
    if body:
        body_kw_hits = sum(1 for kw in _TERM_SECTION_HEADER_KEYWORDS if kw in body)
        score += min(0.4, 0.05 * body_kw_hits)
        body_q_hits = sum(1 for q in query_keywords if len(q) >= 4 and q in body)
        score += min(0.35, 0.07 * body_q_hits)

    # Source-kind prior: typed evidence + structure artifacts are
    # higher-value than raw section maps.
    src = section.get("_source_kind")
    if src in ("contract_provision", "schedule_entry"):
        score += 0.30
    elif src in ("schedule_index", "table_index"):
        score += 0.15

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
      "source_refs": ["span ids or section refs — REQUIRED, do not omit; rows without source_refs WILL BE DROPPED"],
      "confidence": 0.0
    }}
  ],
  "rules": [
    {{
      "document_id": "string (which doc this rule comes from)",
      "section_ref": "string",
      "if": ["clause"],
      "then": ["consequence"],
      "unless": ["exception"],
      "timing": "string (or null)",
      "thresholds": ["string"],
      "parties": ["actor"],
      "source_span_ids": ["span ids — REQUIRED, do not omit"],
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
        n_schedule_index = int(wp.get("schedule_index_count", -1))
        n_table_index = int(wp.get("table_index_count", -1))
        n_schedule_entries = int(wp.get("schedule_entry_count", -1))
        n_obligation_rows = int(wp.get("obligation_row_count", -1))
        n_term_grid_oblig = int(wp.get("term_grid_obligation_count", -1))
        n_contract_provisions = int(wp.get("contract_provision_count", -1))

        # ANY source-kind that loadable now contributes signal. Reviewer
        # r2 blocker: a matter with only `schedule_entry` /
        # `document.schedule_index` / `document.table_index` could load
        # but never be selected. Now any of them gates the operator.
        n_any_source = max(
            n_section_maps, n_schedule_index, n_table_index,
            n_schedule_entries, n_contract_provisions,
        )

        # Strong signal: an obligation row explicitly expects a term_grid
        # artifact, OR contract_provision typed evidence already exists,
        # OR schedule_entry typed evidence is present (high-value source).
        if (
            n_term_grid_oblig > 0
            or n_contract_provisions > 0
            or n_schedule_entries > 0
        ):
            return AgentMatch(
                agent_id=self.agent_id, score=0.95,
                reasons=(
                    f"term_grid_obligations:{n_term_grid_oblig},"
                    f"contract_provisions:{n_contract_provisions},"
                    f"schedule_entries:{n_schedule_entries}",
                ),
                requirement=AgentRequirement.REQUIRED,
                phase="pre_synthesis",
            )

        # Medium signal: any source kind exists AND user query matches
        # term-extraction intent.
        intent_match = self._query_indicates_term_intent(invocation)
        if n_any_source > 0 and intent_match:
            return AgentMatch(
                agent_id=self.agent_id, score=0.85,
                reasons=(
                    f"sources:section_map:{n_section_maps},"
                    f"schedule_index:{n_schedule_index},"
                    f"table_index:{n_table_index},intent_match",
                ),
                requirement=AgentRequirement.REQUIRED,
                phase="pre_synthesis",
            )

        # Weak signal: section maps OR schedule_index OR table_index
        # exist + obligation rows expect something. Run as optional.
        has_any_structure = (
            n_section_maps > 0
            or n_schedule_index > 0
            or n_table_index > 0
        )
        if has_any_structure and n_obligation_rows > 0:
            return AgentMatch(
                agent_id=self.agent_id, score=0.55,
                reasons=(
                    f"structure_sources:{has_any_structure},"
                    f"obligations:{n_obligation_rows}",
                ),
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
            # Reviewer round 1 blocker B2: differentiate "ran empty"
            # (no candidate inputs) from "upstream required evidence
            # missing" (the matter expects term grid work but the
            # source structure isn't there yet).
            wp = invocation.work_profile or {}
            had_strong_signal = (
                int(wp.get("term_grid_obligation_count", 0) or 0) > 0
                or int(wp.get("contract_provision_count", 0) or 0) > 0
            )
            warnings.append(
                "upstream_required_evidence_missing"
                if had_strong_signal
                else "ran_empty"
            )
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
        """Aggregate every plausible source-section input.

        Reviewer round 1 blocker B1: previously only consumed
        `document.section_map`. Now pulls section maps + schedule_index +
        table_index + contract_provision + schedule_entry typed evidence.
        """
        mm = getattr(runtime, "matter_model", None)
        if mm is None:
            return []
        sections: list[dict] = []

        # Source 1: document.section_map artifacts
        try:
            rows = mm.db.execute(
                "SELECT artifact_key, payload_json FROM agent_artifact "
                "WHERE matter_id=? AND artifact_kind=?",
                (mm.matter_id, "document.section_map"),
            ).fetchall()
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
                    s2.setdefault("_source_kind", "section_map")
                    sections.append(s2)
        except Exception as exc:
            warnings.append(f"section_map_load_failed:{type(exc).__name__}")

        # Source 2: document.schedule_index artifacts (named schedules,
        # exhibits, annexes — high-value for term-grid extraction).
        try:
            rows = mm.db.execute(
                "SELECT artifact_key, payload_json FROM agent_artifact "
                "WHERE matter_id=? AND artifact_kind=?",
                (mm.matter_id, "document.schedule_index"),
            ).fetchall()
            for r in rows:
                try:
                    payload = _json.loads(r["payload_json"] or "{}")
                except Exception:
                    continue
                doc_id = payload.get("document_id") or r["artifact_key"]
                for sched in payload.get("schedules") or payload.get("entries") or []:
                    if not isinstance(sched, dict):
                        continue
                    sections.append({
                        "document_id": doc_id,
                        "title": str(sched.get("label") or sched.get("kind") or "schedule"),
                        "text": str(sched.get("text") or sched.get("body") or "")[:self.max_section_chars],
                        "_source_kind": "schedule_index",
                    })
        except Exception as exc:
            warnings.append(f"schedule_index_load_failed:{type(exc).__name__}")

        # Source 3: document.table_index artifacts (typed rows / tables
        # are sometimes the cleanest source for thresholds/amounts).
        try:
            rows = mm.db.execute(
                "SELECT artifact_key, payload_json FROM agent_artifact "
                "WHERE matter_id=? AND artifact_kind=?",
                (mm.matter_id, "document.table_index"),
            ).fetchall()
            for r in rows:
                try:
                    payload = _json.loads(r["payload_json"] or "{}")
                except Exception:
                    continue
                doc_id = payload.get("document_id") or r["artifact_key"]
                for tab in payload.get("tables") or []:
                    if not isinstance(tab, dict):
                        continue
                    sections.append({
                        "document_id": doc_id,
                        "title": str(tab.get("title") or tab.get("caption") or "table"),
                        "text": str(tab.get("text") or tab.get("rendered") or "")[:self.max_section_chars],
                        "_source_kind": "table_index",
                    })
        except Exception as exc:
            warnings.append(f"table_index_load_failed:{type(exc).__name__}")

        # Source 4: contract_provision typed evidence (already-extracted
        # provisions from prior runs / slot wedges).
        try:
            rows = mm.db.execute(
                "SELECT record_key, payload_json, document_id FROM typed_evidence_record "
                "WHERE matter_id=? AND record_kind=?",
                (mm.matter_id, "contract_provision"),
            ).fetchall()
            for r in rows:
                try:
                    payload = _json.loads(r["payload_json"] or "{}")
                except Exception:
                    continue
                doc_id = r["document_id"] or payload.get("document_id") or ""
                title = str(payload.get("section_ref") or payload.get("title") or "provision")
                text = str(payload.get("text") or payload.get("body") or payload.get("provision_text") or "")
                if text:
                    sections.append({
                        "document_id": doc_id, "title": title,
                        "text": text[:self.max_section_chars],
                        "_source_kind": "contract_provision",
                        "record_key": r["record_key"],
                    })
        except Exception as exc:
            warnings.append(f"contract_provision_load_failed:{type(exc).__name__}")

        # Source 5: schedule_entry typed evidence
        try:
            rows = mm.db.execute(
                "SELECT record_key, payload_json, document_id FROM typed_evidence_record "
                "WHERE matter_id=? AND record_kind=?",
                (mm.matter_id, "schedule_entry"),
            ).fetchall()
            for r in rows:
                try:
                    payload = _json.loads(r["payload_json"] or "{}")
                except Exception:
                    continue
                doc_id = r["document_id"] or payload.get("document_id") or ""
                title = str(payload.get("schedule_ref") or "schedule_entry")
                text = str(payload.get("text") or payload.get("description") or "")
                if text:
                    sections.append({
                        "document_id": doc_id, "title": title,
                        "text": text[:self.max_section_chars],
                        "_source_kind": "schedule_entry",
                        "record_key": r["record_key"],
                    })
        except Exception as exc:
            warnings.append(f"schedule_entry_load_failed:{type(exc).__name__}")

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
            section_payload = {
                "document_id": s.get("document_id", ""),
                "section_ref": s.get("title") or s.get("label") or s.get("ref") or "",
                "text": (s.get("text") or s.get("body") or "")[:self.max_section_chars],
            }
            # Reviewer r2 fix: preserve typed-evidence record_key /
            # span_id into the prompt so the LLM can return stable
            # source ids (`record_key:..` / `span:..`) rather than
            # falling back to section refs alone.
            if s.get("record_key"):
                section_payload["source_id"] = f"record:{s['record_key']}"
            elif s.get("span_id"):
                section_payload["source_id"] = f"span:{s['span_id']}"
            else:
                section_payload["source_id"] = (
                    f"{section_payload['document_id']}:{section_payload['section_ref']}"
                    if section_payload["section_ref"] else section_payload["document_id"]
                )
            sections_payload.append(section_payload)
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
            # Reviewer round 1 blocker B3: every term-grid row must be
            # source-grounded. Drop rows the LLM emits without any
            # source_refs — those are ungrounded inferences, not facts.
            source_refs = _coerce_str_list(r.get("source_refs"))
            if not source_refs:
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
            "source_refs": _coerce_str_list(r.get("source_refs"), max_len=8),
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
            if_clauses = _coerce_str_list(r.get("if"), max_len=5)
            then_clauses = _coerce_str_list(r.get("then"), max_len=5)
            if not section_ref or not (if_clauses and then_clauses):
                continue
            # Reviewer non-blocker: rules also require source_span_ids,
            # otherwise the rule is ungrounded inference, not extraction.
            source_spans = _coerce_str_list(r.get("source_span_ids"), max_len=8)
            if not source_spans:
                continue
            rule_id = make_rule_id(
                document_id=doc_id, section_ref=section_ref,
                if_clauses=if_clauses, then_clauses=then_clauses,
            )
            if rule_id in seen:
                if conf > float(seen[rule_id].get("confidence") or 0.0):
                    seen[rule_id] = self._rule_payload(r, rule_id, conf, if_clauses, then_clauses, source_spans)
                continue
            seen[rule_id] = self._rule_payload(r, rule_id, conf, if_clauses, then_clauses, source_spans)
        return list(seen.values())

    @staticmethod
    def _rule_payload(
        r: dict, rule_id: str, conf: float,
        if_clauses: list[str], then_clauses: list[str],
        source_spans: list[str],
    ) -> dict:
        return {
            "rule_id": rule_id,
            "document_id": str(r.get("document_id") or ""),
            "section_ref": str(r.get("section_ref") or ""),
            "if": if_clauses,
            "then": then_clauses,
            "unless": _coerce_str_list(r.get("unless"), max_len=5),
            "timing": r.get("timing") if isinstance(r.get("timing"), str) else None,
            "thresholds": _coerce_str_list(r.get("thresholds"), max_len=5),
            "parties": _coerce_str_list(r.get("parties"), max_len=5),
            "source_span_ids": source_spans,
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
