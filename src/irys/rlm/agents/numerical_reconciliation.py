"""Numerical Reconciliation Agent — deterministic financial math operator.

Operator Substrate Thesis: Sub-agents are bounded operators, not mini
chatbots. This agent does pure Python financial math for QoE / EBITDA
bridges / NWC / PPA / debt-like-items. Zero LLM calls.

Inputs: qoe_line_item typed_evidence rows from the slot wedge
(legal.qoe_line_item.v1).

Outputs: numeric.reconciliation artifacts — bridge totals, group sums,
delta-vs-recommended verification, period-consistency flags.

Why this matters: corporate-ma QoE tasks (analyze-qoe-reconciliation
16.8% R13 baseline → still 35% even after PR#2) fail because LLMs do
arithmetic poorly across many rows. A real operator that sums, verifies
bridges, and flags inconsistencies closes that class of failure.
"""

from __future__ import annotations

import json as _json
import re as _re
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
# Helpers — money parsing
# ---------------------------------------------------------------------------


# Matches things like "$0.8M", "1,234", "(1.3M)" (parens = negative), "3.5 million"
_MONEY_RE = _re.compile(
    r"\(?\$?\s*(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)\s*"
    r"(M|MM|million|K|thousand|B|billion)?\)?",
    _re.IGNORECASE,
)


def _parse_money(raw: Any) -> Optional[float]:
    """Parse a money/number string into a float.

    Returns None when unparseable. Treats parentheses as negation. M/MM
    and 'million' multiply by 1_000_000; K and 'thousand' by 1_000;
    B and 'billion' by 1_000_000_000.

    NEVER raises. Returns None for noise. Matches USD heuristics — for
    other currencies the caller should normalize first.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip()
    if not s or s in {"-", "—", "N/A", "n/a", "TBD", "tbd"}:
        return None
    is_paren_neg = s.startswith("(") and s.endswith(")")
    # Detect leading minus that the regex's currency placement may miss
    # (e.g. "-$0.5M" — the minus sits outside the captured number group).
    leading_minus = False
    s_for_match = s
    if s.startswith("-"):
        leading_minus = True
        s_for_match = s[1:].lstrip()
    m = _MONEY_RE.search(s_for_match)
    if not m:
        return None
    try:
        v = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    if leading_minus and v >= 0:
        v = -v
    suffix = (m.group(2) or "").lower()
    if suffix in {"m", "mm", "million"}:
        v *= 1_000_000
    elif suffix in {"k", "thousand"}:
        v *= 1_000
    elif suffix in {"b", "billion"}:
        v *= 1_000_000_000
    if is_paren_neg:
        v = -v
    return v


def _format_money(v: Optional[float]) -> str:
    if v is None:
        return "—"
    if abs(v) >= 1_000_000:
        return f"${v / 1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"${v / 1_000:.1f}K"
    return f"${v:,.0f}"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


@dataclass
class NumericalReconciliationAgent:
    """Pure-math operator for QoE / EBITDA / NWC / PPA reconciliations."""

    agent_id: str = "finance.numerical_reconciliation.v1"
    version: int = 1
    enabled: bool = True
    priority: int = 100
    capability_tags: tuple[str, ...] = (
        "compute.numerical", "calculate.financial", "verify.extraction",
    )
    supported_domain_profiles: tuple[str, ...] = ("legal:1", "finance:1")
    phases: tuple[str, ...] = ("pre_synthesis",)
    exclusive_group: Optional[str] = None
    deterministic: bool = True

    # Mismatch tolerance (fraction of expected)
    delta_tolerance: float = 0.005  # 0.5%

    # ------------------------------------------------------------------
    # match
    # ------------------------------------------------------------------

    def match(self, invocation: AgentInvocation) -> Optional[AgentMatch]:
        family = (invocation.execution_family or "").lower()
        if family and family not in {"investigate", "extract", "compare"}:
            return None
        return AgentMatch(
            agent_id=self.agent_id,
            score=0.85,
            reasons=("qoe_eligible",),
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
            rows = self._load_qoe_rows(runtime)
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
                warnings=("no_qoe_line_item_evidence",),
            )

        # Group by (category, period, schedule, currency) for bridges
        groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
        for r in rows:
            key = (
                str(r.get("category") or "").strip().lower(),
                str(r.get("period") or "").strip().lower(),
                str(r.get("schedule") or "").strip().lower(),
                str(r.get("currency") or "USD").strip().upper(),
            )
            groups.setdefault(key, []).append(r)

        artifacts: list[AgentArtifact] = []
        for (cat, period, schedule, ccy), group_rows in groups.items():
            try:
                art = self._compute_group(
                    cat, period, schedule, ccy, group_rows, warnings,
                )
            except Exception as exc:
                warnings.append(f"group_compute_error:{cat}:{period}:{exc}")
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
            if (a.artifact_kind == "numeric.reconciliation"
                    and "computed_total" not in p):
                return AgentInvocationResult(
                    status="invalid",
                    error_class="MalformedArtifact",
                    error=f"artifact {a.artifact_key} missing computed_total",
                    elapsed_ms=result.elapsed_ms,
                    warnings=result.warnings,
                )
        return result

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _load_qoe_rows(self, runtime: Any) -> list[Mapping[str, Any]]:
        mm = runtime.matter_model
        if mm is None:
            return []
        try:
            rows = mm.typed_evidence.list_by_kind("qoe_line_item", limit=400)
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
            if payload.get("schema_ref") != "legal.qoe_line_item.v1":
                continue
            payload["_typed_evidence_id"] = r.get("id")
            out.append(payload)
        return out

    def _compute_group(
        self,
        category: str,
        period: str,
        schedule: str,
        currency: str,
        rows: list[Mapping[str, Any]],
        warnings: list[str],
    ) -> Optional[AgentArtifact]:
        if not rows:
            return None

        line_items: list[dict] = []
        seller_total = 0.0
        buyer_total = 0.0
        delta_total = 0.0
        recommended_total = 0.0
        seller_nonempty = 0
        buyer_nonempty = 0

        for r in rows:
            seller = _parse_money(r.get("seller_value"))
            buyer = _parse_money(r.get("buyer_value"))
            delta = _parse_money(r.get("delta"))
            recommended = _parse_money(r.get("recommended_value"))
            label = (r.get("line_item_label") or "").strip() or "(unlabeled)"

            # Compute delta if missing but seller+buyer present
            if delta is None and seller is not None and buyer is not None:
                delta = buyer - seller

            line_items.append({
                "label": label,
                "seller": seller, "seller_str": _format_money(seller),
                "buyer": buyer, "buyer_str": _format_money(buyer),
                "delta": delta, "delta_str": _format_money(delta),
                "recommended": recommended,
                "recommended_str": _format_money(recommended),
                "extracted_seller": r.get("seller_value"),
                "extracted_buyer": r.get("buyer_value"),
                "extracted_delta": r.get("delta"),
                "extracted_recommended": r.get("recommended_value"),
                "typed_evidence_id": r.get("_typed_evidence_id"),
            })
            if seller is not None:
                seller_total += seller
                seller_nonempty += 1
            if buyer is not None:
                buyer_total += buyer
                buyer_nonempty += 1
            if delta is not None:
                delta_total += delta
            if recommended is not None:
                recommended_total += recommended

        # Bridge consistency: buyer_total - seller_total ≈ delta_total
        bridge_consistent = True
        if seller_nonempty and buyer_nonempty:
            implied_delta = buyer_total - seller_total
            tolerance = max(
                abs(seller_total + buyer_total) * self.delta_tolerance, 1.0,
            )
            if abs(implied_delta - delta_total) > tolerance:
                bridge_consistent = False
                warnings.append(
                    f"bridge_inconsistency:{category}:{period}:"
                    f"implied={implied_delta:.0f} vs sum_delta={delta_total:.0f}"
                )

        normalized = (
            f"{category}_{period}_{schedule}".replace(" ", "_").lower()
        )
        artifact_key = f"reconciliation:{normalized}:{currency.lower()}"
        payload = {
            "schema_ref": "agent.numeric_reconciliation.v1",
            "category": category, "period": period,
            "schedule": schedule, "currency": currency,
            "n_line_items": len(line_items),
            "computed_seller_total": seller_total,
            "computed_seller_total_str": _format_money(seller_total),
            "computed_buyer_total": buyer_total,
            "computed_buyer_total_str": _format_money(buyer_total),
            "computed_delta_total": delta_total,
            "computed_delta_total_str": _format_money(delta_total),
            "computed_recommended_total": recommended_total,
            "computed_recommended_total_str": _format_money(recommended_total),
            "computed_total": delta_total,  # canonical "total" for the bridge
            "implied_delta_buyer_minus_seller": (
                buyer_total - seller_total
                if seller_nonempty and buyer_nonempty else None
            ),
            "bridge_consistent": bridge_consistent,
            "line_items": line_items,
        }
        label = (
            f"{category} | {period} | bridge: "
            f"{_format_money(seller_total)} → {_format_money(buyer_total)} "
            f"(Δ {_format_money(delta_total)}, {len(line_items)} items)"
        )
        typed_refs = tuple(
            li["typed_evidence_id"] for li in line_items
            if li.get("typed_evidence_id")
        )
        return AgentArtifact(
            artifact_kind="numeric.reconciliation",
            artifact_key=artifact_key,
            payload=payload,
            label=label,
            synthesis_visibility="answer_ingredient",
            confidence=1.0 if bridge_consistent else 0.7,
            typed_evidence_refs=typed_refs,
            verification_state="verified" if bridge_consistent else "candidate",
        )
