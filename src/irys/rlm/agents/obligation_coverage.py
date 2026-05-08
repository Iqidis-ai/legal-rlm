"""Obligation Coverage Matrix — first cross-domain coverage operator.

Operator Substrate Thesis: Sub-agents are bounded operators, not mini
chatbots. This operator turns task criteria + deliverable specs into an
auditable matrix of required-slot coverage rows BEFORE synthesis can drift.

Inputs (Codex Item 1 D5, tiered preference):
  1. typed_evidence_record(record_kind='task_criteria', schema_ref='task.criteria.v1')
  2. typed_evidence_record(record_kind='task_deliverable_spec', schema_ref='task.deliverable_spec.v1')
  3. ExecutionContract.output_contract.criteria + .deliverable_spec
  4. RunObjective.success_criteria — DISABLED (post-smoke-v6, Codex r3
     SHIP). The engine populates success_criteria via
     `_workflow_success_criteria(contract)` with internal scaffolding
     ("ground the answer in evidence", "satisfy task X"), not user
     intent. Letting it source obligation criteria short-circuits Tier 5
     and produces noisy matrices that hurt synthesis.
  5. Bounded LLM inference (Tier 5, prompt-derived). Codex T5 design
     locked this as the production workhorse for fuzzy user prompts.

Outputs:
  - agent_artifact(artifact_kind='obligation.coverage_matrix.v1') — immutable
    per-run envelope with row IDs, criteria/deliverable hashes, drift summary.
  - typed_evidence_record(record_kind='obligation_row',
      schema_ref='task.obligation_row.v1') — one active row per obligation,
    upserted by deterministic obligation_id, marked active=False when
    superseded by a newer matrix.

Severity (Codex D3, locked):
  critical=0 < required=1 < recommended=2 < optional=3
Row status (D4): met, partial, missing, unknown, repair_required,
                 not_applicable, upstream_required_evidence_missing
Matrix status (D4): success, ran_empty, upstream_required_evidence_missing,
                    invalid_input

Failure policy (D8): blocking_validator by default for external/deployable
final work product (deliverable family + drafting workflow). v1
implementation: detect-and-halt (return status='invalid' from
verify_output when critical/required obligations are unmet, which the
dispatcher treats as a hard fail). Repair pass orchestration is deferred
to a follow-up cycle alongside the deliverable materializer.
"""

from __future__ import annotations

import asyncio as _asyncio
import hashlib as _hashlib
import json as _json
import logging as _logging
import re as _re
from dataclasses import dataclass
from typing import Any, Mapping, Optional

_logger = _logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tier 5 — prompt-inferred criteria
# ---------------------------------------------------------------------------


TIER5_INFERENCE_VERSION = "t5_criteria_inference_v1"

# Confidence thresholds (Codex T5-D4)
TIER5_CONFIDENCE_NORMAL = 0.65
TIER5_CONFIDENCE_LOW = 0.40

# Family + answer_shape gating (Codex T5-D7)
_TIER5_STRUCTURED_ANSWER_SHAPES = {
    "memo", "report", "table", "validation_matrix",
    "multi_document_memo", "scoped_quant_table", "grounded_inventory",
}
_TIER5_DRAFTING_WORKFLOWS = {"drafting", "analysis_deliverable"}


def _normalize_query_for_cache(query: str) -> str:
    """Stable cache key normalization — lowercase, collapse whitespace."""
    return " ".join((query or "").lower().split())


def _tier5_inference_cache_key(
    *,
    user_query: str,
    task_spec: Mapping[str, Any],
    execution_family: str,
    workflow_kind: str,
    domain_profile_id: str,
    domain_profile_version: int,
) -> str:
    """Codex T5-D8 cache key — by prompt + spec, not benchmark task."""
    canonical = _json.dumps({
        "inference_version": TIER5_INFERENCE_VERSION,
        "normalized_user_query": _normalize_query_for_cache(user_query),
        "task_spec": dict(task_spec),
        "execution_family": execution_family or "",
        "workflow_kind": workflow_kind or "",
        "domain_profile_id": domain_profile_id or "",
        "domain_profile_version": int(domain_profile_version or 0),
    }, sort_keys=True, separators=(",", ":"))
    return _hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


_TIER5_PROMPT_TEMPLATE = """You are the criteria-inference step inside a bounded operator, not a chatbot.

Infer the minimum useful deliverable criteria that a competent professional would apply to the user's requested work product. Use only the user request, task spec, execution family, workflow kind, domain, and general domain profile. Do not invent facts. Do not optimize for any benchmark. Do not include criteria that require information not implied by the task type or ordinary professional norms.

Return ONLY a single JSON object matching this schema (no commentary, no markdown fences):
{{
  "schema_ref": "task.criteria_inference.v1",
  "inference_version": "t5_criteria_inference_v1",
  "confidence": 0.0,
  "should_materialize_matrix": true,
  "vagueness_reason": "",
  "criteria": [
    {{
      "criterion_id": "inferred_slug",
      "title": "short criterion title",
      "description": "observable deliverable obligation",
      "severity": "critical|required|recommended|optional",
      "expected_deliverable_shape": "section|table|artifact|narrative|code_patch|citation_set",
      "source": "inferred"
    }}
  ],
  "deliverables": [
    {{
      "deliverable_key": "stable_slug",
      "filename": "",
      "format": "markdown|json|patch|other",
      "required_sections": [],
      "required_tables": [],
      "required_artifact_kinds": []
    }}
  ]
}}

Severity rules:
- critical: the deliverable is unsafe or invalid without it.
- required: the requested work product is materially incomplete without it.
- recommended: improves quality but should not block ordinary synthesis.
- optional: only useful if the user asked for completeness.

Granularity rules:
- Prefer 3-7 criteria.
- Criteria must be checkable against output, artifacts, or evidence.
- Do not create one criterion per possible subtopic.
- If the user asks only a narrow factual lookup, return should_materialize_matrix=false unless a structured deliverable is requested.

Anti-gaming rules (these are absolute):
- You have NOT been given a scoring rubric or benchmark task ID. None exists. Do not pretend to have one.
- Use only general professional norms for the active domain.
- Do not output benchmark-specific criterion IDs, task names, or fixture identifiers.

INPUT:
{input_json}

Return ONLY the JSON object."""


def _strip_json_fences(text: str) -> str:
    """LLMs sometimes emit ```json ... ```; strip them defensively."""
    s = (text or "").strip()
    if s.startswith("```"):
        # remove first fence line
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    return s


def _parse_tier5_response(raw_text: str) -> Optional[dict]:
    """Best-effort JSON parse, accepting fenced or unfenced output."""
    s = _strip_json_fences(raw_text)
    try:
        obj = _json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # last-resort: locate first {...} balanced block
    m = _re.search(r"\{[\s\S]*\}", s)
    if m:
        try:
            obj = _json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def _normalize_inferred_criteria(parsed: dict) -> tuple[list[dict], list[dict], float, bool, str]:
    """Coerce LLM JSON into the locked criteria/deliverable shape.

    Returns (criteria, deliverables, confidence, should_materialize, vagueness_reason).
    Aggressively defensive — bad / partial output produces empty lists rather than crashing.
    """
    criteria_raw = parsed.get("criteria") or []
    deliverables_raw = parsed.get("deliverables") or []
    try:
        confidence = float(parsed.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0))
    should_materialize = bool(parsed.get("should_materialize_matrix", True))
    vagueness_reason = str(parsed.get("vagueness_reason") or "")[:200]

    criteria: list[dict] = []
    for i, c in enumerate(criteria_raw if isinstance(criteria_raw, list) else []):
        if not isinstance(c, dict):
            continue
        title = str(c.get("title") or "").strip()
        if not title:
            continue
        sev = str(c.get("severity") or "required").lower()
        if sev not in SEVERITY_RANK:
            sev = "required"
        criteria.append({
            "criterion_id": str(c.get("criterion_id") or f"inferred_{i}"),
            "title": title[:160],
            "description": str(c.get("description") or "")[:600],
            "severity": sev,
            "expected_deliverable_shape": str(
                c.get("expected_deliverable_shape") or "section"
            ),
            "source": "inferred",
        })

    deliverables: list[dict] = []
    for d in deliverables_raw if isinstance(deliverables_raw, list) else []:
        if not isinstance(d, dict):
            continue
        key = str(d.get("deliverable_key") or "").strip()
        if not key:
            continue
        deliverables.append({
            "deliverable_key": key,
            "filename": str(d.get("filename") or ""),
            "format": str(d.get("format") or "markdown"),
            "required_sections": list(d.get("required_sections") or []),
            "required_tables": list(d.get("required_tables") or []),
            "required_artifact_kinds": list(d.get("required_artifact_kinds") or []),
        })

    return criteria, deliverables, confidence, should_materialize, vagueness_reason


def _apply_confidence_policy(
    criteria: list[dict], confidence: float, execution_family: str,
) -> tuple[list[dict], bool]:
    """Codex T5-D4 confidence-gated severity policy.

    Returns (filtered_criteria, should_materialize_override).
    """
    # Below low threshold: only deliverable family materializes a tiny matrix;
    # everything else returns ran_empty.
    if confidence < TIER5_CONFIDENCE_LOW:
        if (execution_family or "").lower() != "deliverable":
            return [], False
        # For deliverable family: keep critical/required only, downgrade rest
        kept: list[dict] = []
        for c in criteria:
            if c["severity"] in {"critical", "required"}:
                kept.append(c)
        return kept, bool(kept)
    # Mid-confidence: keep all but downgrade non-critical to recommended,
    # except evidence-grounding-shaped criteria stay required.
    if confidence < TIER5_CONFIDENCE_NORMAL:
        out: list[dict] = []
        for c in criteria:
            if c["severity"] == "critical":
                out.append(c)
            elif c["severity"] == "required":
                # Keep required for evidence/citation/grounding patterns;
                # downgrade other required to recommended.
                title_lc = (c.get("title") or "").lower()
                if any(k in title_lc for k in (
                    "cite", "ground", "source", "evidence", "missing",
                )):
                    out.append(c)
                else:
                    out.append({**c, "severity": "recommended"})
            else:
                out.append(c)
        return out, True
    # Normal confidence: pass through
    return list(criteria), True


def _should_run_tier5(
    *,
    execution_family: str,
    answer_shape: str,
    workflow_kind: str,
    user_asked_completeness: bool,
) -> bool:
    """Codex T5-D7 family + work-profile gating."""
    fam = (execution_family or "").lower()
    if fam == "deliverable":
        return True
    if fam == "compare":
        return answer_shape in _TIER5_STRUCTURED_ANSWER_SHAPES or user_asked_completeness
    if fam == "investigate":
        return (
            answer_shape in _TIER5_STRUCTURED_ANSWER_SHAPES
            or workflow_kind in _TIER5_DRAFTING_WORKFLOWS
        )
    if fam == "trace":
        return user_asked_completeness
    if fam in {"read", "query"}:
        return user_asked_completeness
    return False  # clarify, scenario, steer: no by default

from .contracts import (
    AgentArtifact,
    AgentInvocation,
    AgentInvocationResult,
    AgentMatch,
    AgentRequirement,
)


# ---------------------------------------------------------------------------
# Constants — locked schema vocabulary (Codex Item 1 design)
# ---------------------------------------------------------------------------


SCHEMA_REF_OBLIGATION_ROW = "task.obligation_row.v1"
SCHEMA_REF_TASK_CRITERIA = "task.criteria.v1"
SCHEMA_REF_TASK_DELIVERABLE_SPEC = "task.deliverable_spec.v1"

ARTIFACT_KIND_MATRIX = "obligation.coverage_matrix.v1"
ARTIFACT_KIND_VALIDATOR_FAILURE = "validator_failure.obligation_coverage.v1"

RECORD_KIND_OBLIGATION_ROW = "obligation_row"
RECORD_KIND_TASK_CRITERIA = "task_criteria"
RECORD_KIND_TASK_DELIVERABLE_SPEC = "task_deliverable_spec"

SEVERITY_RANK = {"critical": 0, "required": 1, "recommended": 2, "optional": 3}
ROW_STATUSES = (
    "met", "partial", "missing", "unknown",
    "repair_required", "not_applicable", "upstream_required_evidence_missing",
)
MATRIX_STATUSES = (
    "success", "ran_empty",
    "upstream_required_evidence_missing", "invalid_input",
)
BLOCKING_SEVERITIES = ("critical", "required")
ACCEPTABLE_STATUSES = ("met", "not_applicable")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical_json(obj: Any) -> str:
    return _json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _normalize_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def _sha24(s: str) -> str:
    return _hashlib.sha256(s.encode("utf-8")).hexdigest()[:24]


def make_obligation_id(
    *,
    task_fingerprint: str,
    source_family: str,
    source_criterion_key_or_text: str,
    deliverable_key: str,
    required_slot_key: str,
    expected_artifact_kind: str,
) -> str:
    """Deterministic semantic ID per Codex D1.

    Stable across runs of the same task; includes deliverable + slot so
    one criterion can yield multiple rows. Excludes matter_id and run_id
    so the ID is portable across reruns and the memory broker.
    """
    canonical = _canonical_json({
        "schema_ref": SCHEMA_REF_OBLIGATION_ROW,
        "task_fingerprint": task_fingerprint,
        "source_family": source_family,
        "source_criterion_key_or_normalized_text": source_criterion_key_or_text,
        "deliverable_key": deliverable_key,
        "required_slot_key": required_slot_key,
        "expected_artifact_kind": expected_artifact_kind,
    })
    return f"obl_v1:{_sha24(canonical)}"


def _hash_payload(obj: Any) -> str:
    return _sha24(_canonical_json(obj))


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


@dataclass
class ObligationCoverageMatrix:
    """First cross-domain coverage operator.

    Reads task criteria + deliverable spec from typed evidence (preferred)
    or ExecutionContract (transitional). Emits a matrix artifact + per-row
    typed evidence records. Halts under blocking_validator when critical or
    required obligations are not met / not_applicable.
    """

    agent_id: str = "obligation_coverage_matrix"
    version: int = 1
    enabled: bool = True
    priority: int = 90  # higher than HHI/Recon — this gates them
    capability_tags: tuple[str, ...] = (
        "coverage.obligation",
        "validate.required_slots",
        "artifact.coverage",
        "deliverable.validator",
    )
    supported_domain_profiles: tuple[str, ...] = (
        "legal:1", "finance:1", "coding:1", "academic_research:1", "biomedical:1",
    )
    phases: tuple[str, ...] = ("pre_synthesis",)
    exclusive_group: Optional[str] = None
    deterministic: bool = False  # bounded LLM step is allowed for criteria parsing

    # ------------------------------------------------------------------
    # match — work-aware, bias toward deliverable family
    # ------------------------------------------------------------------

    def match(self, invocation: AgentInvocation) -> Optional[AgentMatch]:
        family = (invocation.execution_family or "").lower()
        wp = invocation.work_profile or {}
        n_criteria = int(wp.get("task_criteria_count", -1))
        n_deliverable = int(wp.get("task_deliverable_spec_count", -1))
        n_existing = int(wp.get("obligation_row_count", -1))
        answer_shape = (invocation.task.answer_shape or "").lower()
        workflow_kind = (invocation.workflow_kind or "").lower()
        # Codex Phase-2 r1 (Tier 5) blocker fix: read completeness flag
        # from work_profile so read/query "what gaps?" prompts can route
        # here. Engine populates this from the user query.
        user_asked_completeness = bool(wp.get("query_asks_completeness", 0))

        # Tier 5 family gating (Codex T5-D7): the operator can fire even
        # without pre-supplied criteria, as long as the family/work-profile
        # combination expects a structured deliverable.
        eligible = _should_run_tier5(
            execution_family=family,
            answer_shape=answer_shape,
            workflow_kind=workflow_kind,
            user_asked_completeness=user_asked_completeness,
        ) or (n_criteria > 0 or n_deliverable > 0 or n_existing > 0)

        # Hard skip: pure read/query/clarify/scenario/steer with no signals.
        if not eligible:
            return None

        # Strong signal: criteria or deliverable specs already exist.
        if n_criteria > 0 or n_deliverable > 0:
            return AgentMatch(
                agent_id=self.agent_id,
                score=0.97,
                reasons=(f"task_criteria:{n_criteria},deliverable_specs:{n_deliverable}",),
                # blocking_validator on deliverable family; required elsewhere
                requirement=(
                    AgentRequirement.BLOCKING_VALIDATOR
                    if family == "deliverable"
                    else AgentRequirement.REQUIRED
                ),
                phase="pre_synthesis",
            )
        # Re-run case: stale matrix from prior run is still useful to refresh.
        if n_existing > 0:
            return AgentMatch(
                agent_id=self.agent_id, score=0.6,
                reasons=(f"prior_obligation_rows:{n_existing}",),
                requirement=AgentRequirement.OPTIONAL,
                phase="pre_synthesis",
            )
        # Deliverable family with no criteria evidence yet — still important
        # because the operator can fall back to ExecutionContract surfaces
        # OR Tier 5 inference. Codex Phase-2 r1 (Tier 5) blocker fix:
        # external/deployable deliverables MUST escalate to
        # BLOCKING_VALIDATOR so inferred-missing-required rows actually
        # halt the synthesis. Internal preview/draft modes stay REQUIRED.
        if family == "deliverable":
            external_deliverable = bool(wp.get("external_deliverable", 0)) or (
                workflow_kind in {"drafting", "analysis_deliverable"}
            )
            return AgentMatch(
                agent_id=self.agent_id, score=0.7,
                reasons=("deliverable_family_no_criteria_yet",),
                requirement=(
                    AgentRequirement.BLOCKING_VALIDATOR
                    if external_deliverable
                    else AgentRequirement.REQUIRED
                ),
                phase="pre_synthesis",
            )
        # Investigate / compare with structured deliverable shape: REQUIRED.
        if (
            family in {"investigate", "compare"}
            and (
                answer_shape in _TIER5_STRUCTURED_ANSWER_SHAPES
                or workflow_kind in _TIER5_DRAFTING_WORKFLOWS
            )
        ):
            return AgentMatch(
                agent_id=self.agent_id, score=0.65,
                reasons=(
                    f"structured_{family}:{answer_shape}|{workflow_kind}",
                ),
                requirement=AgentRequirement.REQUIRED,
                phase="pre_synthesis",
            )
        # Read/query/trace with explicit completeness ask: OPTIONAL audit.
        if user_asked_completeness:
            return AgentMatch(
                agent_id=self.agent_id, score=0.55,
                reasons=("user_asks_completeness",),
                requirement=AgentRequirement.OPTIONAL,
                phase="pre_synthesis",
            )
        return None

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

        # Resolve criteria + deliverable spec sources (D5 tiered preference)
        # Tiers 1-4 (typed evidence, ExecutionContract, RunObjective).
        try:
            criteria, criteria_source = await _asyncio.to_thread(
                self._load_criteria, runtime, invocation, warnings,
            )
            deliverables, deliverable_source = await _asyncio.to_thread(
                self._load_deliverables, runtime, invocation, warnings,
            )
        except Exception as exc:
            return AgentInvocationResult(
                status="error",
                error_class=type(exc).__name__,
                error=str(exc)[:300],
                elapsed_ms=int((_time.perf_counter() - t0) * 1000),
            )

        # Tier 5 — bounded LLM inference (Codex T5-D1).
        # Anti-gaming foundation: this is the path that makes the operator
        # work for fuzzy user prompts with no benchmark metadata. We only
        # invoke it when Tiers 1-4 produced nothing AND family/work-profile
        # gating says a coverage matrix is meaningful here.
        tier5_llm_calls = 0
        tier5_tokens = 0
        if (not criteria and not deliverables):
            user_query = self._user_query(invocation, runtime)
            should_run = _should_run_tier5(
                execution_family=invocation.execution_family,
                answer_shape=invocation.task.answer_shape,
                workflow_kind=invocation.workflow_kind,
                user_asked_completeness=any(
                    k in (user_query or "").lower()
                    for k in ("missing", "complete", "coverage", "checklist", "validation", "gap")
                ),
            )
            if should_run:
                try:
                    (criteria, deliverables, criteria_source,
                     deliverable_source, tier5_llm_calls, tier5_tokens
                    ) = await self._tier5_infer(
                        runtime=runtime, invocation=invocation,
                        user_query=user_query, warnings=warnings,
                    )
                except Exception as exc:
                    warnings.append(f"tier5_inference_failed:{type(exc).__name__}:{str(exc)[:120]}")

        # No criteria AND no deliverables → ran_empty matrix; no row writes.
        if not criteria and not deliverables:
            return AgentInvocationResult(
                status="success",
                elapsed_ms=int((_time.perf_counter() - t0) * 1000),
                warnings=tuple(warnings) + ("no_criteria_no_deliverables",),
                llm_calls=tier5_llm_calls,
                token_estimate=tier5_tokens,
            )

        task_fingerprint = self._resolve_task_fingerprint(
            invocation, runtime,
            criteria=criteria, deliverables=deliverables,
        )
        run_id = invocation.run_id or ""

        # Build obligation rows (deterministic; one row per
        # criterion×deliverable_slot combination, with a fallback when no
        # deliverable is specified).
        rows = self._materialize_rows(
            criteria=criteria,
            deliverables=deliverables,
            task_fingerprint=task_fingerprint,
            criteria_source=criteria_source,
            deliverable_source=deliverable_source,
            warnings=warnings,
        )

        # Score each row's status against currently-available evidence + artifacts.
        rows = await _asyncio.to_thread(self._score_rows, rows, runtime, warnings)

        # Drift summary vs prior matrix (if any).
        drift = await _asyncio.to_thread(
            self._compute_drift, runtime, rows, task_fingerprint,
        )

        # Mark stale prior rows. Track failures explicitly so the matrix
        # artifact does not falsely look clean when persistence is broken.
        # Codex Phase-2 r2: differentiate "could not query prior rows" from
        # "had nothing to mark" — both must surface to the caller.
        stale_failed_count, stale_load_failed = await _asyncio.to_thread(
            self._stale_prior_rows, runtime, rows, task_fingerprint, run_id,
        )
        if stale_load_failed:
            warnings.append("stale_prior_load_failed")
        if stale_failed_count:
            warnings.append(f"stale_marking_failures:{stale_failed_count}")

        # Write current obligation_row typed evidence (active, upsert by id).
        # Track per-row failure count so a corrupt typed-evidence layer
        # surfaces as a result-level concern instead of silent success.
        upsert_failed_count = await _asyncio.to_thread(
            self._upsert_obligation_rows, runtime, rows, run_id,
        )
        if upsert_failed_count:
            warnings.append(f"row_upsert_failures:{upsert_failed_count}")
            _logger.error(
                "obligation_coverage: %d/%d row upserts failed",
                upsert_failed_count, len(rows),
            )

        # If row persistence failed for ALL rows, the typed-evidence
        # contract (D2) is broken — surface as error so downstream
        # consumers don't trust the matrix.
        if upsert_failed_count and upsert_failed_count == len(rows):
            return AgentInvocationResult(
                status="error",
                error_class="ObligationRowPersistenceFailure",
                error="all obligation_row upserts failed",
                warnings=tuple(warnings),
                elapsed_ms=int((_time.perf_counter() - t0) * 1000),
                llm_calls=tier5_llm_calls,
                token_estimate=tier5_tokens,
            )

        # Build matrix envelope artifact
        matrix_artifact = self._build_matrix_artifact(
            rows=rows,
            criteria=criteria,
            deliverables=deliverables,
            criteria_source=criteria_source,
            deliverable_source=deliverable_source,
            task_fingerprint=task_fingerprint,
            run_id=run_id,
            drift=drift,
            invocation_requirement=invocation.requirement.value,
        )
        artifacts: tuple[AgentArtifact, ...] = (matrix_artifact,)

        # If blocking and any critical/required obligation is unmet,
        # add a validator_failure artifact so the audit trail records why.
        unmet_blocking = [
            r for r in rows
            if r["severity"] in BLOCKING_SEVERITIES
            and r["status"] not in ACCEPTABLE_STATUSES
        ]
        if unmet_blocking and invocation.requirement == AgentRequirement.BLOCKING_VALIDATOR:
            artifacts = artifacts + (
                self._build_validator_failure_artifact(
                    rows=unmet_blocking,
                    matrix_key=matrix_artifact.artifact_key,
                    run_id=run_id,
                ),
            )

        return AgentInvocationResult(
            status="success",
            artifacts=artifacts,
            warnings=tuple(warnings),
            elapsed_ms=int((_time.perf_counter() - t0) * 1000),
            llm_calls=tier5_llm_calls,
            token_estimate=tier5_tokens,
        )

    # ------------------------------------------------------------------
    # verify_output — D8 halt mechanism
    # ------------------------------------------------------------------

    def verify_output(
        self,
        invocation: AgentInvocation,
        result: AgentInvocationResult,
    ) -> AgentInvocationResult:
        if result.status != "success":
            return result
        if invocation.requirement != AgentRequirement.BLOCKING_VALIDATOR:
            return result
        # Find matrix artifact and check for blocking unmet obligations.
        matrix = next(
            (a for a in result.artifacts if a.artifact_kind == ARTIFACT_KIND_MATRIX),
            None,
        )
        if matrix is None:
            return result
        items = (matrix.payload or {}).get("rows", [])
        # Codex lifecycle decision E: pre-synthesis Tier-5 inferred rows
        # are placeholders until output exists. Pending statuses
        # (`unknown`, `upstream_required_evidence_missing`) must NOT halt
        # because they reflect "we haven't written the output yet", not
        # "the output is wrong." Halt only on CONCRETE gaps verifiable
        # pre-synthesis (e.g. a required artifact_kind is absent from
        # the matter). Post-synthesis verification of pending rows
        # belongs to a future cycle.
        _CONCRETE_GAPS = ("missing", "partial", "repair_required")
        unmet_blocking = [
            it for it in items
            if it.get("severity") in BLOCKING_SEVERITIES
            and it.get("status") in _CONCRETE_GAPS
        ]
        if not unmet_blocking:
            return result
        # Per Codex D8: halt. The dispatcher's blocking_validator handler
        # will surface this status as a RuntimeError to the engine.
        return AgentInvocationResult(
            status="invalid",
            error_class="ObligationCoverageBlocking",
            error=(
                f"{len(unmet_blocking)} blocking obligation(s) unmet: "
                + ", ".join(
                    f"{it.get('source_criterion',{}).get('title','?')[:40]} "
                    f"({it.get('severity')}/{it.get('status')})"
                    for it in unmet_blocking[:3]
                )
            ),
            elapsed_ms=result.elapsed_ms,
            warnings=result.warnings + ("blocking_validator_halt",),
            artifacts=result.artifacts,  # keep artifacts so renderer can surface them
        )

    # ------------------------------------------------------------------
    # Internals — loading sources (D5 tiered)
    # ------------------------------------------------------------------

    def _load_criteria(
        self, runtime: Any, invocation: AgentInvocation, warnings: list[str],
    ) -> tuple[list[dict], str]:
        """Return (criteria_list, source_label). Tiered per D5."""
        mm = getattr(runtime, "matter_model", None)
        # Tier 1: typed_evidence_record
        if mm is not None:
            try:
                rows = mm.typed_evidence.list_by_kind(
                    RECORD_KIND_TASK_CRITERIA, limit=50,
                )
                # Codex Phase-2 r1 (Tier 5): EXCLUDE inferred rows from
                # Tier 1. Inferred rows are persisted with `source='inferred'`
                # AND record_key prefix `inferred:` so Tier 5 alone manages
                # them via cache lookup keyed by current prompt. If we
                # accept inferred rows here, a stale antitrust inference
                # leaks into the next prompt in the same matter.
                explicit_rows = [
                    r for r in (rows or [])
                    if not str(r.get("record_key", "")).startswith("inferred:")
                ]
                items = self._extract_criteria_from_rows(explicit_rows)
                if items:
                    return items, "typed_evidence_record"
            except Exception as exc:
                warnings.append(f"criteria_typed_evidence_load_failed:{exc}")

        # Tier 3: ExecutionContract.output_contract.criteria
        ec_criteria = self._criteria_from_execution_contract(runtime)
        if ec_criteria:
            return ec_criteria, "execution_contract"

        # Tier 4 — RunObjective.success_criteria — DISABLED.
        # Smoke v6 confirmed RunObjective.success_criteria is engine-internal
        # boilerplate ("satisfy task X", "ground the answer in evidence",
        # "do not rely on cached summaries") populated by
        # `_workflow_success_criteria(contract)` in engine.py:2639. It is
        # NOT user intent and short-circuits Tier 5 inference, producing
        # noisy obligation matrices that hurt synthesis quality without
        # representing real deliverable requirements. Anti-gaming-adjacent:
        # it's engine-supplied scaffolding masquerading as criteria.
        # Tier 5 must handle the no-criteria case.

        # Tier 5: bounded LLM inference (handled in invoke() after this returns).
        return [], "none"

    def _load_deliverables(
        self, runtime: Any, invocation: AgentInvocation, warnings: list[str],
    ) -> tuple[list[dict], str]:
        mm = getattr(runtime, "matter_model", None)
        if mm is not None:
            try:
                rows = mm.typed_evidence.list_by_kind(
                    RECORD_KIND_TASK_DELIVERABLE_SPEC, limit=20,
                )
                # Same Tier-5-isolation rule as criteria: skip inferred rows.
                explicit_rows = [
                    r for r in (rows or [])
                    if not str(r.get("record_key", "")).startswith("inferred:")
                ]
                items = self._extract_deliverables_from_rows(explicit_rows)
                if items:
                    return items, "typed_evidence_record"
            except Exception as exc:
                warnings.append(f"deliverable_typed_evidence_load_failed:{exc}")
        ec_deliverables = self._deliverables_from_execution_contract(runtime)
        if ec_deliverables:
            return ec_deliverables, "execution_contract"
        return [], "none"

    @staticmethod
    def _extract_criteria_from_rows(rows: list[dict]) -> list[dict]:
        out: list[dict] = []
        for r in rows or []:
            payload_raw = r.get("payload_json") or "{}"
            try:
                payload = (
                    _json.loads(payload_raw) if isinstance(payload_raw, str)
                    else dict(payload_raw)
                )
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("schema_ref") != SCHEMA_REF_TASK_CRITERIA:
                continue
            for c in payload.get("criteria") or []:
                if isinstance(c, dict) and c.get("title"):
                    out.append({
                        "criterion_id": str(c.get("criterion_id") or ""),
                        "title": str(c.get("title") or ""),
                        "description": str(c.get("description") or ""),
                        "severity": str(c.get("severity") or "required").lower(),
                        "source": str(c.get("source") or "explicit_task_contract"),
                        "task_id": str(payload.get("task_id") or ""),
                    })
        return out

    @staticmethod
    def _extract_deliverables_from_rows(rows: list[dict]) -> list[dict]:
        out: list[dict] = []
        for r in rows or []:
            payload_raw = r.get("payload_json") or "{}"
            try:
                payload = (
                    _json.loads(payload_raw) if isinstance(payload_raw, str)
                    else dict(payload_raw)
                )
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("schema_ref") != SCHEMA_REF_TASK_DELIVERABLE_SPEC:
                continue
            for d in payload.get("deliverables") or []:
                if isinstance(d, dict) and d.get("deliverable_key"):
                    out.append({
                        "deliverable_key": str(d.get("deliverable_key")),
                        "filename": str(d.get("filename") or ""),
                        "format": str(d.get("format") or "markdown"),
                        "required_sections": list(d.get("required_sections") or []),
                        "required_tables": list(d.get("required_tables") or []),
                        "required_artifact_kinds": list(d.get("required_artifact_kinds") or []),
                        "task_id": str(payload.get("task_id") or ""),
                    })
        return out

    @staticmethod
    def _criteria_from_execution_contract(runtime: Any) -> list[dict]:
        contract = (
            getattr(getattr(runtime, "state", None), "execution_contract", None)
            if hasattr(runtime, "state") else None
        )
        if contract is None:
            return []
        oc = getattr(contract, "output_contract", {}) or {}
        raw = oc.get("criteria") or []
        out: list[dict] = []
        for c in raw if isinstance(raw, list) else []:
            if isinstance(c, dict) and c.get("title"):
                out.append({
                    "criterion_id": str(c.get("criterion_id") or ""),
                    "title": str(c.get("title")),
                    "description": str(c.get("description") or ""),
                    "severity": str(c.get("severity") or "required").lower(),
                    "source": "contract",
                    "task_id": str(oc.get("task_id") or ""),
                })
        return out

    @staticmethod
    def _deliverables_from_execution_contract(runtime: Any) -> list[dict]:
        contract = (
            getattr(getattr(runtime, "state", None), "execution_contract", None)
            if hasattr(runtime, "state") else None
        )
        if contract is None:
            return []
        oc = getattr(contract, "output_contract", {}) or {}
        raw = oc.get("deliverable_spec") or []
        if isinstance(raw, dict):
            raw = [raw]
        out: list[dict] = []
        for d in raw if isinstance(raw, list) else []:
            if isinstance(d, dict) and d.get("deliverable_key"):
                out.append({
                    "deliverable_key": str(d.get("deliverable_key")),
                    "filename": str(d.get("filename") or ""),
                    "format": str(d.get("format") or "markdown"),
                    "required_sections": list(d.get("required_sections") or []),
                    "required_tables": list(d.get("required_tables") or []),
                    "required_artifact_kinds": list(d.get("required_artifact_kinds") or []),
                    "task_id": str(oc.get("task_id") or ""),
                })
        return out

    # Note: an earlier `_criteria_from_run_objective()` helper sourced
    # criteria from `state.run_objective.success_criteria`. Smoke v6
    # demonstrated that source is engine-internal scaffolding from
    # `engine._workflow_success_criteria()`, not user intent — using it
    # short-circuited Tier 5 and produced noisy matrices. The helper has
    # been removed (Codex holistic review action item 6). If a future
    # caller needs RunObjective-shaped criteria, design a new typed
    # surface for it; do NOT reactivate the engine-scaffolding path.

    # ------------------------------------------------------------------
    # Tier 5 — prompt-inferred criteria
    # ------------------------------------------------------------------

    def _user_query(
        self, invocation: AgentInvocation, runtime: Any,
    ) -> str:
        """Pull the original user query from runtime state for Tier 5."""
        try:
            state = getattr(runtime, "state", None)
            if state is not None:
                q = getattr(state, "query", None)
                if q:
                    return str(q)
        except Exception:
            pass
        # Fall back to a tiny derivation from invocation if state isn't there
        return str(invocation.input_hash or invocation.task.task_type or "")

    def _domain_profile_for_invocation(
        self, invocation: AgentInvocation, runtime: Any,
    ) -> tuple[str, int, str]:
        """Resolve (profile_id, profile_version, domain_label) for prompt input.

        Domain label maps the profile id to one of the 5 target domains
        for the Tier 5 prompt. Falls back to 'legal' if not derivable.
        """
        pid = invocation.domain_profile_id or ""
        pver = int(invocation.domain_profile_version or 0)
        if not pid:
            try:
                pid = runtime.domain_profile_id() or ""
            except Exception:
                pid = ""
        domain = "legal"
        for prefix, lbl in (
            ("legal:", "legal"),
            ("finance:", "finance"),
            ("coding:", "coding"),
            ("academic_research:", "academic_research"),
            ("biomedical:", "biomedical"),
        ):
            if pid.startswith(prefix):
                domain = lbl
                break
        return pid, pver, domain

    async def _tier5_infer(
        self,
        *,
        runtime: Any,
        invocation: AgentInvocation,
        user_query: str,
        warnings: list[str],
    ) -> tuple[list[dict], list[dict], str, str, int, int]:
        """Run a single bounded LLM call to infer criteria + deliverable spec.

        Returns: (criteria, deliverables, criteria_source, deliverable_source,
                  llm_calls, token_estimate). All zero/empty if inference is
                  skipped or fails.

        Anti-gaming: takes only user_query + task_spec + family + domain
        profile. NEVER reads benchmark identifiers, scoring rubrics, or
        evaluator metadata.
        """
        llm_client = getattr(runtime, "llm_client", None)
        if llm_client is None:
            warnings.append("tier5_no_llm_client")
            return [], [], "none", "none", 0, 0

        profile_id, profile_version, domain = self._domain_profile_for_invocation(
            invocation, runtime,
        )

        cache_key = _tier5_inference_cache_key(
            user_query=user_query,
            task_spec=invocation.task.__dict__ if hasattr(invocation.task, "__dict__")
                     else dict(getattr(invocation.task, "_asdict", lambda: {})()),
            execution_family=invocation.execution_family,
            workflow_kind=invocation.workflow_kind,
            domain_profile_id=profile_id,
            domain_profile_version=profile_version,
        )

        # Cache hit: prior inferred typed evidence row by stable cache key.
        cached = await _asyncio.to_thread(
            self._tier5_load_cached, runtime, cache_key,
        )
        if cached is not None:
            return cached + (0, 0)

        # Build the bounded inference prompt.
        input_payload = {
            "user_query": user_query[:8000],
            "task_spec": {
                "task_type": invocation.task.task_type,
                "operation": invocation.task.operation,
                "answer_shape": invocation.task.answer_shape,
                "required_evidence": list(invocation.task.required_evidence or ()),
            },
            "execution_family": invocation.execution_family,
            "workflow_kind": invocation.workflow_kind,
            "domain": domain,
            "domain_profile": {
                "profile_id": profile_id,
                "profile_version": profile_version,
                "general_deliverable_patterns": [],
                "general_quality_norms": [],
            },
        }
        prompt = _TIER5_PROMPT_TEMPLATE.format(
            input_json=_json.dumps(input_payload, indent=2)
        )

        try:
            from ...core.models import ModelTier as _ModelTier
            response_text = await llm_client.complete(
                prompt=prompt,
                tier=_ModelTier.FLASH,
                timeout=30.0,
                usage_label="obligation_criteria_inference",
                json_mode=True,
            )
        except TypeError:
            # Fake clients in tests may not accept json_mode kwarg —
            # retry without it (production GeminiClient always accepts it).
            try:
                response_text = await llm_client.complete(
                    prompt=prompt,
                    tier=_ModelTier.FLASH,
                    timeout=30.0,
                    usage_label="obligation_criteria_inference",
                )
            except Exception as exc:
                warnings.append(f"tier5_llm_call_failed:{type(exc).__name__}")
                # Codex Phase-2 r1 non-blocker: count attempted call.
                return [], [], "none", "none", 1, 0
        except Exception as exc:
            warnings.append(f"tier5_llm_call_failed:{type(exc).__name__}")
            return [], [], "none", "none", 1, 0

        parsed = _parse_tier5_response(response_text or "")
        if parsed is None:
            warnings.append("tier5_unparseable_response")
            return [], [], "none", "none", 1, len(response_text or "") // 4

        criteria, deliverables, confidence, should_materialize, vagueness = (
            _normalize_inferred_criteria(parsed)
        )
        if not should_materialize:
            warnings.append(f"tier5_should_not_materialize:{vagueness[:60]}")
            return [], [], "none", "none", 1, len(response_text or "") // 4

        # Confidence policy
        criteria, materialize_after_policy = _apply_confidence_policy(
            criteria, confidence, invocation.execution_family,
        )
        if not materialize_after_policy or not criteria:
            warnings.append(f"tier5_low_confidence:{confidence:.2f}")
            return [], [], "none", "none", 1, len(response_text or "") // 4

        # Persist as task_criteria + task_deliverable_spec typed evidence
        # so reruns hit cache and downstream operators can consume rows.
        try:
            await _asyncio.to_thread(
                self._tier5_persist, runtime, cache_key, profile_id,
                profile_version, criteria, deliverables, confidence,
                user_query,
            )
        except Exception as exc:
            warnings.append(f"tier5_persist_failed:{type(exc).__name__}")

        # Tag source on the criteria/deliverables so the rest of the
        # pipeline knows where they came from.
        for c in criteria:
            c["task_id"] = f"inferred:{cache_key}"
            c["source"] = "inferred"
        for d in deliverables:
            d["task_id"] = f"inferred:{cache_key}"

        return (
            criteria, deliverables,
            "llm_inferred", "llm_inferred",
            1, len(response_text or "") // 4,
        )

    def _tier5_load_cached(
        self, runtime: Any, cache_key: str,
    ) -> Optional[tuple[list[dict], list[dict], str, str]]:
        """Look up a previously-persisted Tier 5 inference."""
        mm = getattr(runtime, "matter_model", None)
        if mm is None:
            return None
        try:
            row = mm.db.execute(
                "SELECT payload_json FROM typed_evidence_record "
                "WHERE matter_id=? AND record_kind=? AND record_key=?",
                (mm.matter_id, RECORD_KIND_TASK_CRITERIA,
                 f"inferred:{cache_key}"),
            ).fetchone()
        except Exception:
            return None
        if row is None:
            return None
        try:
            payload = _json.loads(row["payload_json"] or "{}")
        except Exception:
            return None
        criteria = payload.get("criteria") or []
        if not criteria:
            return None
        # Also load deliverable spec
        try:
            d_row = mm.db.execute(
                "SELECT payload_json FROM typed_evidence_record "
                "WHERE matter_id=? AND record_kind=? AND record_key=?",
                (mm.matter_id, RECORD_KIND_TASK_DELIVERABLE_SPEC,
                 f"inferred:{cache_key}"),
            ).fetchone()
            if d_row is not None:
                d_payload = _json.loads(d_row["payload_json"] or "{}")
                deliverables = d_payload.get("deliverables") or []
            else:
                deliverables = []
        except Exception:
            deliverables = []
        # Inject task_id on each row for fingerprint resolution
        for c in criteria:
            c.setdefault("task_id", f"inferred:{cache_key}")
        for d in deliverables:
            d.setdefault("task_id", f"inferred:{cache_key}")
        return criteria, deliverables, "llm_inferred_cached", "llm_inferred_cached"

    def _tier5_persist(
        self, runtime: Any, cache_key: str, profile_id: str,
        profile_version: int, criteria: list[dict], deliverables: list[dict],
        confidence: float, user_query: str,
    ) -> None:
        """Persist inferred criteria + deliverables as typed evidence so
        re-runs hit cache and downstream operators consume them uniformly.
        """
        mm = getattr(runtime, "matter_model", None)
        if mm is None:
            return
        prompt_hash = _sha24(user_query or "")
        criteria_payload = {
            "schema_ref": SCHEMA_REF_TASK_CRITERIA,
            "task_id": f"inferred:{cache_key}",
            "source": "inferred",
            "inference_version": TIER5_INFERENCE_VERSION,
            "domain_profile_id": profile_id,
            "domain_profile_version": profile_version,
            "prompt_hash": prompt_hash,
            "confidence": confidence,
            "criteria": criteria,
        }
        try:
            mm.typed_evidence.upsert(
                RECORD_KIND_TASK_CRITERIA,
                f"inferred:{cache_key}",
                payload=criteria_payload,
                document_id=None,
                confidence=confidence,
            )
        except Exception as exc:
            _logger.error("tier5 criteria persist failed: %s", exc)
            raise

        if deliverables:
            deliverable_payload = {
                "schema_ref": SCHEMA_REF_TASK_DELIVERABLE_SPEC,
                "task_id": f"inferred:{cache_key}",
                "source": "inferred",
                "inference_version": TIER5_INFERENCE_VERSION,
                "domain_profile_id": profile_id,
                "domain_profile_version": profile_version,
                "prompt_hash": prompt_hash,
                "confidence": confidence,
                "deliverables": deliverables,
            }
            try:
                mm.typed_evidence.upsert(
                    RECORD_KIND_TASK_DELIVERABLE_SPEC,
                    f"inferred:{cache_key}",
                    payload=deliverable_payload,
                    document_id=None,
                    confidence=confidence,
                )
            except Exception as exc:
                _logger.error("tier5 deliverable persist failed: %s", exc)

    # ------------------------------------------------------------------
    # Materialize rows
    # ------------------------------------------------------------------

    def _resolve_task_fingerprint(
        self,
        invocation: AgentInvocation,
        runtime: Any,
        *,
        criteria: list[dict],
        deliverables: list[dict],
    ) -> str:
        """Stable fingerprint for obligation_id key derivation.

        Preference order (Codex Phase-2 r1 fix — `input_hash` is just
        `state.query[:64]` which collides for same-prefix tasks):
          1. task_id present on any criterion/deliverable payload
          2. ExecutionContract.output_contract.task_id
          3. invocation.task.task_type + invocation.input_hash (last resort)
        """
        # Tier 1: criteria/deliverable carry an external task_id (e.g. caller pre-supplied via task contract)
        for src in criteria + deliverables:
            tid = src.get("task_id")
            if tid:
                return f"task:{tid}"
        # Tier 2: ExecutionContract.output_contract.task_id
        try:
            contract = (
                getattr(getattr(runtime, "state", None), "execution_contract", None)
                if hasattr(runtime, "state") else None
            )
            if contract is not None:
                oc = getattr(contract, "output_contract", {}) or {}
                tid = oc.get("task_id")
                if tid:
                    return f"task:{tid}"
        except Exception:
            pass
        # Tier 3: fallback — keep last 64 query chars but tag clearly so
        # downstream consumers can see it's not a stable external id.
        return f"query_hash:{invocation.task.task_type}:{invocation.input_hash}"

    def _materialize_rows(
        self,
        *,
        criteria: list[dict],
        deliverables: list[dict],
        task_fingerprint: str,
        criteria_source: str,
        deliverable_source: str,
        warnings: list[str],
    ) -> list[dict]:
        rows: list[dict] = []
        # If we have deliverables, generate one row per criterion×slot. If not,
        # generate one row per criterion against a synthetic single deliverable.
        deliv_iter = deliverables or [{
            "deliverable_key": "__default__",
            "filename": "",
            "format": "markdown",
            "required_sections": [],
            "required_tables": [],
            "required_artifact_kinds": [],
        }]

        # If criteria empty but deliverables present, generate one row per
        # required slot (section / table / artifact_kind).
        if not criteria and deliverables:
            for d in deliverables:
                slots = self._enumerate_slots(d)
                for slot_kind, slot_key, label in slots:
                    rows.append(self._make_row(
                        task_fingerprint=task_fingerprint,
                        criterion={
                            "criterion_id": f"deliverable_slot:{slot_key}",
                            "title": f"Required {slot_kind}: {label}",
                            "description": "",
                            "severity": "required",
                            "source": deliverable_source,
                            "task_id": d.get("task_id", ""),
                        },
                        deliverable=d,
                        slot_kind=slot_kind,
                        slot_key=slot_key,
                        slot_label=label,
                    ))
            return rows

        # Criteria present (with or without deliverables).
        for c in criteria:
            for d in deliv_iter:
                slots = self._enumerate_slots(d)
                if not slots:
                    # Single row at deliverable-level
                    rows.append(self._make_row(
                        task_fingerprint=task_fingerprint,
                        criterion=c,
                        deliverable=d,
                        slot_kind="deliverable",
                        slot_key=d["deliverable_key"],
                        slot_label=d.get("filename") or d["deliverable_key"],
                    ))
                else:
                    for slot_kind, slot_key, label in slots:
                        rows.append(self._make_row(
                            task_fingerprint=task_fingerprint,
                            criterion=c,
                            deliverable=d,
                            slot_kind=slot_kind,
                            slot_key=slot_key,
                            slot_label=label,
                        ))
        return rows

    @staticmethod
    def _enumerate_slots(deliverable: Mapping[str, Any]) -> list[tuple[str, str, str]]:
        out: list[tuple[str, str, str]] = []
        for s in deliverable.get("required_sections") or []:
            label = s if isinstance(s, str) else (s.get("label") or s.get("key") or "")
            key = s if isinstance(s, str) else (s.get("key") or label)
            if key:
                out.append(("section", str(key), str(label)))
        for t in deliverable.get("required_tables") or []:
            label = t if isinstance(t, str) else (t.get("label") or t.get("key") or "")
            key = t if isinstance(t, str) else (t.get("key") or label)
            if key:
                out.append(("table", str(key), str(label)))
        for a in deliverable.get("required_artifact_kinds") or []:
            out.append(("artifact_kind", str(a), str(a)))
        return out

    def _make_row(
        self,
        *,
        task_fingerprint: str,
        criterion: Mapping[str, Any],
        deliverable: Mapping[str, Any],
        slot_kind: str,
        slot_key: str,
        slot_label: str,
    ) -> dict:
        sev = (criterion.get("severity") or "required").lower()
        if sev not in SEVERITY_RANK:
            sev = "required"
        criterion_key = (
            criterion.get("criterion_id")
            or _normalize_text(criterion.get("title") or "")[:80]
        )
        expected_artifact_kind = (
            "draft_document.v1" if slot_kind != "artifact_kind"
            else slot_key
        )
        ob_id = make_obligation_id(
            task_fingerprint=task_fingerprint,
            source_family=str(criterion.get("source") or "explicit_task_contract"),
            source_criterion_key_or_text=str(criterion_key),
            deliverable_key=str(deliverable.get("deliverable_key", "")),
            required_slot_key=f"{slot_kind}:{slot_key}",
            expected_artifact_kind=expected_artifact_kind,
        )
        return {
            "schema_ref": SCHEMA_REF_OBLIGATION_ROW,
            "obligation_id": ob_id,
            "active": True,
            "task_fingerprint": task_fingerprint,
            "source_criterion": {
                "criterion_id": str(criterion.get("criterion_id") or ""),
                "title": str(criterion.get("title") or ""),
                "normalized_text_hash": _sha24(
                    _normalize_text(criterion.get("description") or criterion.get("title") or "")
                ),
            },
            "deliverable": {
                "deliverable_key": str(deliverable.get("deliverable_key", "")),
                "filename": str(deliverable.get("filename") or ""),
                "format": str(deliverable.get("format") or "markdown"),
            },
            "required_slot": {
                "slot_key": slot_key,
                "label": slot_label,
                "section": slot_label if slot_kind == "section" else None,
                "table": slot_label if slot_kind == "table" else None,
                "kind": slot_kind,
            },
            "expected_artifact_kind": expected_artifact_kind,
            "evidence_refs": [],
            "artifact_refs": [],
            "output_refs": [],
            "status": "missing",  # default — _score_rows will update
            "status_reason": "",
            "severity": sev,
            "severity_rank": SEVERITY_RANK[sev],
            "repair_instruction": "",
        }

    # ------------------------------------------------------------------
    # Scoring rows against current evidence
    # ------------------------------------------------------------------

    def _score_rows(
        self, rows: list[dict], runtime: Any, warnings: list[str],
    ) -> list[dict]:
        mm = getattr(runtime, "matter_model", None)
        if mm is None:
            return rows
        # Pull recent agent_artifacts and typed_evidence kinds once for matching.
        try:
            artifact_kinds_present = {
                r["artifact_kind"]
                for r in mm.db.execute(
                    "SELECT DISTINCT artifact_kind FROM agent_artifact "
                    "WHERE matter_id=?",
                    (mm.matter_id,),
                ).fetchall()
            }
        except Exception:
            artifact_kinds_present = set()

        try:
            recent_te_kinds = {
                r["record_kind"]
                for r in mm.db.execute(
                    "SELECT DISTINCT record_kind FROM typed_evidence_record "
                    "WHERE matter_id=?",
                    (mm.matter_id,),
                ).fetchall()
            }
        except Exception:
            recent_te_kinds = set()

        for r in rows:
            slot = r["required_slot"]
            kind = slot.get("kind")
            expected_ak = r["expected_artifact_kind"]

            if kind == "artifact_kind":
                if expected_ak in artifact_kinds_present:
                    r["status"] = "met"
                    r["status_reason"] = f"artifact_kind '{expected_ak}' present"
                else:
                    r["status"] = "missing"
                    r["status_reason"] = f"artifact_kind '{expected_ak}' absent"
            elif kind in ("section", "table"):
                # v1: cannot directly verify section/table coverage in the
                # output (output is built post-synthesis). Mark as
                # upstream_required_evidence_missing if no relevant artifacts
                # exist; otherwise mark unknown so synthesis attempts and a
                # future re-score can confirm.
                relevant = (
                    "document.section_map" in artifact_kinds_present
                    or "draft_document.v1" in artifact_kinds_present
                )
                if not relevant:
                    r["status"] = "upstream_required_evidence_missing"
                    r["status_reason"] = (
                        "no document.section_map or draft_document.v1 "
                        "available to confirm section/table coverage"
                    )
                else:
                    r["status"] = "unknown"
                    r["status_reason"] = (
                        "section/table coverage cannot be verified "
                        "pre-synthesis; will be re-checked post-output"
                    )
            else:  # deliverable-level
                r["status"] = "unknown"
                r["status_reason"] = "deliverable-level row; needs post-synthesis check"
        return rows

    # ------------------------------------------------------------------
    # Drift + persistence
    # ------------------------------------------------------------------

    def _compute_drift(
        self, runtime: Any, current_rows: list[dict], task_fingerprint: str,
    ) -> dict:
        mm = getattr(runtime, "matter_model", None)
        if mm is None:
            return {"added": [], "removed": [], "changed_status": [], "changed_severity": []}
        try:
            prior = mm.db.execute(
                "SELECT record_key, payload_json FROM typed_evidence_record "
                "WHERE matter_id=? AND record_kind=?",
                (mm.matter_id, RECORD_KIND_OBLIGATION_ROW),
            ).fetchall()
        except Exception:
            return {"added": [], "removed": [], "changed_status": [], "changed_severity": []}

        prior_by_id: dict[str, dict] = {}
        for r in prior:
            try:
                payload = _json.loads(r["payload_json"] or "{}")
            except Exception:
                continue
            if (payload.get("task_fingerprint") or "") != task_fingerprint:
                continue
            ob_id = r["record_key"]
            prior_by_id[ob_id] = payload

        current_ids = {r["obligation_id"] for r in current_rows}
        prior_ids = set(prior_by_id.keys())
        added = sorted(current_ids - prior_ids)
        removed = sorted(prior_ids - current_ids)
        changed_status: list[dict] = []
        changed_severity: list[dict] = []
        cur_by_id = {r["obligation_id"]: r for r in current_rows}
        for cid in current_ids & prior_ids:
            old, new = prior_by_id[cid], cur_by_id[cid]
            if old.get("status") != new.get("status"):
                changed_status.append(
                    {"obligation_id": cid, "from": old.get("status"),
                     "to": new.get("status")}
                )
            if old.get("severity") != new.get("severity"):
                changed_severity.append(
                    {"obligation_id": cid, "from": old.get("severity"),
                     "to": new.get("severity")}
                )
        return {
            "added": added, "removed": removed,
            "changed_status": changed_status,
            "changed_severity": changed_severity,
        }

    def _stale_prior_rows(
        self, runtime: Any, current_rows: list[dict],
        task_fingerprint: str, run_id: str,
    ) -> tuple[int, bool]:
        """Return (n_failed_upserts, load_failed_flag).

        Codex Phase-2 r2: differentiate "no rows to mark" from "SELECT
        prior rows broke" — D6 stale-marking contract depends on it.
        """
        mm = getattr(runtime, "matter_model", None)
        if mm is None:
            return 0, False
        current_ids = {r["obligation_id"] for r in current_rows}
        try:
            prior = mm.db.execute(
                "SELECT id, record_key, payload_json FROM typed_evidence_record "
                "WHERE matter_id=? AND record_kind=?",
                (mm.matter_id, RECORD_KIND_OBLIGATION_ROW),
            ).fetchall()
        except Exception as exc:
            _logger.error("obligation_coverage stale prior load failed: %s", exc)
            return 0, True
        n_failed = 0
        for r in prior:
            ob_id = r["record_key"]
            try:
                payload = _json.loads(r["payload_json"] or "{}")
            except Exception:
                continue
            if (payload.get("task_fingerprint") or "") != task_fingerprint:
                continue
            if ob_id in current_ids:
                continue
            if not payload.get("active", True):
                continue
            payload["active"] = False
            payload["stale_reason"] = "not_in_current_matrix"
            payload["superseded_by_run_id"] = run_id
            try:
                mm.typed_evidence.upsert(
                    RECORD_KIND_OBLIGATION_ROW,
                    ob_id,
                    payload=payload,
                    confidence=0.5,
                )
            except Exception as exc:
                n_failed += 1
                _logger.error(
                    "obligation_coverage stale upsert failed for %s: %s",
                    ob_id, exc,
                )
        return n_failed, False

    def _upsert_obligation_rows(
        self, runtime: Any, rows: list[dict], run_id: str,
    ) -> int:
        """Return number of upsert failures (Codex Phase-2 r1: surface, don't swallow)."""
        mm = getattr(runtime, "matter_model", None)
        if mm is None:
            return 0
        n_failed = 0
        for r in rows:
            payload = dict(r)
            payload["run_id"] = run_id
            try:
                mm.typed_evidence.upsert(
                    RECORD_KIND_OBLIGATION_ROW,
                    r["obligation_id"],
                    payload=payload,
                    document_id=None,
                    confidence=1.0 if r["status"] == "met" else 0.7,
                )
            except Exception as exc:
                n_failed += 1
                _logger.error(
                    "obligation_coverage row upsert failed for %s: %s",
                    r["obligation_id"], exc,
                )
        return n_failed

    # ------------------------------------------------------------------
    # Artifact builders
    # ------------------------------------------------------------------

    def _build_matrix_artifact(
        self,
        *,
        rows: list[dict],
        criteria: list[dict],
        deliverables: list[dict],
        criteria_source: str,
        deliverable_source: str,
        task_fingerprint: str,
        run_id: str,
        drift: dict,
        invocation_requirement: str,
    ) -> AgentArtifact:
        n_total = len(rows)
        n_met = sum(1 for r in rows if r["status"] == "met")
        n_missing = sum(1 for r in rows if r["status"] == "missing")
        n_partial = sum(1 for r in rows if r["status"] == "partial")
        n_unknown = sum(1 for r in rows if r["status"] == "unknown")
        n_upstream_missing = sum(
            1 for r in rows
            if r["status"] == "upstream_required_evidence_missing"
        )
        n_critical_missing = sum(
            1 for r in rows
            if r["severity"] == "critical" and r["status"] not in ACCEPTABLE_STATUSES
        )
        n_required_missing = sum(
            1 for r in rows
            if r["severity"] == "required" and r["status"] not in ACCEPTABLE_STATUSES
        )
        all_required_filled = all(
            r["status"] in ACCEPTABLE_STATUSES
            for r in rows
            if r["severity"] in BLOCKING_SEVERITIES
        )

        # Codex Phase-2 r1 fix: matrix_status reflects what actually happened.
        if not rows:
            matrix_status = "ran_empty"
        elif n_upstream_missing == n_total:
            matrix_status = "upstream_required_evidence_missing"
        else:
            matrix_status = "success"

        # Codex lifecycle decision E (B-now, D-later): explicit pre-vs-post
        # synthesis observability. Tier 5 produces *expected output
        # requirements* before the answer exists — most rows will be
        # pending until post-synthesis verification lands. Distinguish
        # rows that have CONCRETE pre-synthesis-verifiable status
        # (artifact_kind already present/absent) from rows that are
        # placeholders awaiting output.
        n_pending_output = sum(
            1 for r in rows
            if r["status"] in ("unknown", "upstream_required_evidence_missing")
        )
        n_verifiable_gaps = sum(
            1 for r in rows
            if r["status"] in ("missing", "partial", "repair_required")
        )

        criteria_hash = _hash_payload(criteria)
        deliverable_spec_hash = _hash_payload(deliverables)
        artifact_key = f"obligation_matrix:{task_fingerprint}:{run_id}"
        payload = {
            "schema_ref": ARTIFACT_KIND_MATRIX,
            "task_fingerprint": task_fingerprint,
            "run_id": run_id,
            "invocation_requirement": invocation_requirement,
            "criteria_source": criteria_source,
            "deliverable_source": deliverable_source,
            "criteria_hash": criteria_hash,
            "deliverable_spec_hash": deliverable_spec_hash,
            "matrix_status": matrix_status,
            # Codex lifecycle decision E observability fields:
            "matrix_phase": "pre_synthesis_inference",
            "render_policy": "concrete_gaps_only_until_post_synthesis",
            "n_pending_output": n_pending_output,
            "n_verifiable_gaps": n_verifiable_gaps,
            "n_total": n_total,
            "n_met": n_met,
            "n_missing": n_missing,
            "n_partial": n_partial,
            "n_unknown": n_unknown,
            "n_upstream_missing": n_upstream_missing,
            "n_critical_missing": n_critical_missing,
            "n_required_missing": n_required_missing,
            "all_required_filled": all_required_filled,
            "drift": drift,
            "rows": rows,
        }
        label = (
            f"Obligations: {n_met}/{n_total} met "
            f"({n_critical_missing} critical, {n_required_missing} required missing)"
        )
        verification = "verified" if all_required_filled else "candidate"
        # Codex Phase-2 r2: keep typed_evidence_refs aligned with the
        # established precedent (typed_evidence_record.id, not record_key).
        # Items 2/9 should look up obligation rows by matter_id+record_kind
        # OR via the explicit `obligation_ids` field added to the payload.
        payload["obligation_ids"] = tuple(r["obligation_id"] for r in rows)

        return AgentArtifact(
            artifact_kind=ARTIFACT_KIND_MATRIX,
            artifact_key=artifact_key,
            payload=payload,
            label=label,
            synthesis_visibility="answer_ingredient",
            confidence=1.0 if all_required_filled else 0.7,
            verification_state=verification,
        )

    def _build_validator_failure_artifact(
        self, *, rows: list[dict], matrix_key: str, run_id: str,
    ) -> AgentArtifact:
        return AgentArtifact(
            artifact_kind=ARTIFACT_KIND_VALIDATOR_FAILURE,
            artifact_key=f"validator_failure:{matrix_key}",
            payload={
                "schema_ref": ARTIFACT_KIND_VALIDATOR_FAILURE,
                "matrix_artifact_key": matrix_key,
                "run_id": run_id,
                "blocking_rows": [
                    {
                        "obligation_id": r["obligation_id"],
                        "severity": r["severity"],
                        "status": r["status"],
                        "title": r["source_criterion"]["title"],
                        "deliverable_key": r["deliverable"]["deliverable_key"],
                        "slot_label": r["required_slot"]["label"],
                        "status_reason": r.get("status_reason", ""),
                    }
                    for r in rows
                ],
            },
            label=f"Blocking obligations unmet ({len(rows)})",
            synthesis_visibility="answer_ingredient",
            confidence=1.0,
            verification_state="verified",
        )
