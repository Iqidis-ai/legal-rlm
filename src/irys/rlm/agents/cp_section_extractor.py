"""CP (Conditions Precedent) Section Extractor — third deterministic operator.

Operator Substrate Thesis: Sub-agents are bounded operators, not mini
chatbots. This agent does NOT prompt the LLM. It consumes cp_gap typed
evidence rows already produced by the slot wedge (legal.cp_gap.v1) and
emits a structured CP coverage report grouped by required document.

Inputs: cp_gap typed_evidence rows (status: met / missing / partial).
Outputs: cp.coverage_report artifact per required_document, listing
each requirement with current best status, severity, and evidence
document. Plus aggregate counts.

Why a separate operator (not just the slot wedge): the slot wedge
extracts row data; the operator deterministically *organizes* and
*verifies* — multiple revisions per requirement collapse to "current
best", duplicates are flagged, and the structured table flows into
synthesis as a verified answer ingredient instead of as prose recall.

Targets the banking compare-CP / closing-checklist family where the
LLM kept producing inconsistent narrative summaries even when the
underlying rows were correctly extracted.
"""

from __future__ import annotations

import json as _json
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .contracts import (
    AgentArtifact,
    AgentInvocation,
    AgentInvocationResult,
    AgentMatch,
    AgentRequirement,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_STATUS_PRIORITY = {"missing": 0, "partial": 1, "met": 2, "unknown": 3}
_SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def _better_revision(a: Mapping[str, Any], b: Mapping[str, Any]) -> Mapping[str, Any]:
    """Pick the higher-priority revision for the same requirement.

    Preference (lower-rank-number wins):
      1. Status: missing > partial > met > unknown — i.e., when there's
         a missing finding, that's the more important revision to surface.
         For CP coverage we want to highlight problems, not declare
         everything fine because the latest revision said met.
      2. Severity: critical > high > medium > low
      3. Otherwise keep the existing one (stable)
    """
    a_status = (a.get("status") or "unknown").lower()
    b_status = (b.get("status") or "unknown").lower()
    a_rank = _STATUS_PRIORITY.get(a_status, 3)
    b_rank = _STATUS_PRIORITY.get(b_status, 3)
    if a_rank != b_rank:
        return a if a_rank < b_rank else b
    a_sev = (a.get("severity") or "medium").lower()
    b_sev = (b.get("severity") or "medium").lower()
    a_sev_rank = _SEVERITY_RANK.get(a_sev, 2)
    b_sev_rank = _SEVERITY_RANK.get(b_sev, 2)
    if a_sev_rank != b_sev_rank:
        return a if a_sev_rank < b_sev_rank else b
    return a


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


@dataclass
class CpSectionExtractorAgent:
    """Deterministic CP coverage operator.

    Consumes cp_gap typed evidence and produces a structured per-required-
    document coverage report. Multiple revisions per requirement collapse
    to a current-best row using status + severity priority.
    """

    agent_id: str = "banking.cp_section_extractor.v1"
    version: int = 1
    enabled: bool = True
    priority: int = 100
    capability_tags: tuple[str, ...] = (
        "extract.section", "compute.numerical", "verify.extraction",
        "compare.coverage",
    )
    supported_domain_profiles: tuple[str, ...] = ("legal:1",)
    phases: tuple[str, ...] = ("pre_synthesis",)
    exclusive_group: Optional[str] = None
    deterministic: bool = True

    # ------------------------------------------------------------------
    # match
    # ------------------------------------------------------------------

    def match(self, invocation: AgentInvocation) -> Optional[AgentMatch]:
        family = (invocation.execution_family or "").lower()
        if family and family not in {"investigate", "extract", "compare"}:
            return None
        wp = invocation.work_profile or {}
        n_cp = int(wp.get("cp_gap_count", -1))
        if n_cp > 0:
            return AgentMatch(
                agent_id=self.agent_id, score=0.95,
                reasons=(f"cp_gaps:{n_cp}",),
                requirement=AgentRequirement.OPTIONAL,
                phase="pre_synthesis",
            )
        if n_cp == 0:
            return AgentMatch(
                agent_id=self.agent_id, score=0.20,
                reasons=("no_cp_gaps",),
                requirement=AgentRequirement.OPTIONAL,
                phase="pre_synthesis",
            )
        return AgentMatch(
            agent_id=self.agent_id, score=0.85,
            reasons=("no_work_profile",),
            requirement=AgentRequirement.OPTIONAL,
            phase="pre_synthesis",
        )

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
        try:
            rows = self._load_cp_gaps(runtime)
        except Exception as exc:
            return AgentInvocationResult(
                status="error",
                error_class=type(exc).__name__,
                error=str(exc)[:300],
                elapsed_ms=int((_time.perf_counter() - t0) * 1000),
            )
        if not rows:
            return AgentInvocationResult(
                status="success",
                elapsed_ms=int((_time.perf_counter() - t0) * 1000),
                warnings=("no_cp_gap_evidence",),
            )

        # Organize by (required_document, requirement_id), collapse revisions
        by_doc: dict[str, dict[str, dict]] = {}
        revision_counts: dict[tuple[str, str], int] = {}
        for r in rows:
            req_doc = (r.get("required_by_document") or "").strip() or "(unknown agreement)"
            req_id = (r.get("cp_id") or r.get("cp_section")
                       or (r.get("requirement_text") or "")[:60]).strip()
            if not req_id:
                continue
            key = (req_doc, req_id)
            revision_counts[key] = revision_counts.get(key, 0) + 1
            current = by_doc.setdefault(req_doc, {}).get(req_id)
            if current is None:
                by_doc[req_doc][req_id] = dict(r)
            else:
                by_doc[req_doc][req_id] = dict(_better_revision(current, r))

        artifacts: list[AgentArtifact] = []
        for req_doc, requirements in by_doc.items():
            try:
                art = self._build_doc_artifact(
                    req_doc, requirements, revision_counts, warnings,
                )
            except Exception as exc:
                warnings.append(f"doc_artifact_error:{req_doc}:{exc}")
                continue
            if art is not None:
                artifacts.append(art)

        return AgentInvocationResult(
            status="success",
            artifacts=tuple(artifacts),
            warnings=tuple(warnings),
            elapsed_ms=int((_time.perf_counter() - t0) * 1000),
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
            if (a.artifact_kind == "cp.coverage_report"
                    and "n_total" not in p):
                return AgentInvocationResult(
                    status="invalid",
                    error_class="MalformedArtifact",
                    error=f"artifact {a.artifact_key} missing n_total",
                    elapsed_ms=result.elapsed_ms,
                    warnings=result.warnings,
                )
        return result

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _load_cp_gaps(self, runtime: Any) -> list[dict]:
        mm = runtime.matter_model
        if mm is None:
            return []
        try:
            rows = mm.typed_evidence.list_by_kind("cp_gap", limit=400)
        except Exception:
            return []
        out: list[dict] = []
        for r in rows:
            payload_raw = r.get("payload_json") or "{}"
            try:
                payload = (
                    _json.loads(payload_raw)
                    if isinstance(payload_raw, str) else dict(payload_raw)
                )
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("schema_ref") != "legal.cp_gap.v1":
                continue
            payload["_typed_evidence_id"] = r.get("id")
            out.append(payload)
        return out

    def _build_doc_artifact(
        self,
        req_doc: str,
        requirements: dict[str, dict],
        revision_counts: dict[tuple[str, str], int],
        warnings: list[str],
    ) -> Optional[AgentArtifact]:
        if not requirements:
            return None

        items: list[dict] = []
        n_missing = n_partial = n_met = n_unknown = 0
        n_critical_missing = 0

        for req_id, row in sorted(requirements.items()):
            status = (row.get("status") or "unknown").lower()
            severity = (row.get("severity") or "medium").lower()
            if status == "missing":
                n_missing += 1
                if severity == "critical":
                    n_critical_missing += 1
            elif status == "partial":
                n_partial += 1
            elif status == "met":
                n_met += 1
            else:
                n_unknown += 1
            n_revisions = revision_counts.get((req_doc, req_id), 1)
            if n_revisions > 1:
                warnings.append(
                    f"cp_revisions:{req_doc}:{req_id}:n={n_revisions}"
                )
            items.append({
                "cp_id": row.get("cp_id"),
                "cp_section": row.get("cp_section"),
                "requirement_text": row.get("requirement_text"),
                "observed_evidence": row.get("observed_evidence"),
                "status": status,
                "severity": severity,
                "evidence_document": row.get("evidence_document"),
                "source_document": row.get("source_document"),
                "n_revisions": n_revisions,
                "typed_evidence_id": row.get("_typed_evidence_id"),
            })

        # Sort: missing first, then partial, then met; within each by severity
        items.sort(key=lambda d: (
            _STATUS_PRIORITY.get(d["status"], 3),
            _SEVERITY_RANK.get(d["severity"], 2),
            str(d.get("cp_id") or ""),
        ))

        n_total = len(items)
        coverage_pct = (
            round(100.0 * n_met / n_total, 1) if n_total else 0.0
        )

        artifact_key = (
            "cp_coverage:" +
            req_doc.replace(" ", "_").replace("/", "_").lower()
        )
        payload = {
            "schema_ref": "agent.cp_coverage_report.v1",
            "required_document": req_doc,
            "n_total": n_total,
            "n_missing": n_missing,
            "n_partial": n_partial,
            "n_met": n_met,
            "n_unknown": n_unknown,
            "n_critical_missing": n_critical_missing,
            "coverage_pct": coverage_pct,
            "items": items,
        }
        label = (
            f"CP coverage [{req_doc}]: {n_met}/{n_total} met "
            f"({n_missing} missing, {n_partial} partial)"
        )
        typed_refs = tuple(
            i["typed_evidence_id"] for i in items
            if i.get("typed_evidence_id")
        )
        # Verification state: any critical missing → "candidate" (urgent
        # human review); otherwise verified.
        verification = "candidate" if n_critical_missing > 0 else "verified"
        return AgentArtifact(
            artifact_kind="cp.coverage_report",
            artifact_key=artifact_key,
            payload=payload,
            label=label,
            synthesis_visibility="answer_ingredient",
            confidence=1.0 if n_critical_missing == 0 else 0.85,
            typed_evidence_refs=typed_refs,
            verification_state=verification,
        )
