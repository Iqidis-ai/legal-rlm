"""HHI / Market-Share Calculator — first concrete sub-agent (operator).

Operator Substrate Thesis: Sub-agents are bounded operators, not mini
chatbots. This agent does pure deterministic math — sum shares, square
them, sum the squares, verify against extracted HHI values. Zero LLM
calls.

Inputs: market_row typed evidence rows already extracted by the slot
wedge (legal.market_row.v1).
Outputs: hhi.calculation artifacts + verification flags for any extracted
values that disagree with computed values.

Why this is the strongest first agent (Codex 8/10): the math is trivial
but the LLM gets it wrong often enough that "compute it ourselves and
flag discrepancies" is high signal. Hits the antitrust HHI 1-of-9 task
directly.
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
# Helpers — share parsing
# ---------------------------------------------------------------------------


_SHARE_NUMBER_RE = _re.compile(r"-?\d+(?:\.\d+)?")


def _parse_share(raw: Any) -> Optional[float]:
    """Parse a share string like '28%' or '0.28' or 28 into a fractional float.

    Returns None if unparseable. Treats values > 1 as percent (returns
    fraction); values <= 1 as already-fractional. NEVER raises — returns
    None for noise.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        v = float(raw)
        return v / 100.0 if v > 1 else v
    s = str(raw).strip().lower()
    if not s:
        return None
    is_percent = "%" in s
    m = _SHARE_NUMBER_RE.search(s)
    if not m:
        return None
    try:
        v = float(m.group(0))
    except ValueError:
        return None
    if is_percent or v > 1:
        return v / 100.0
    return v


def _hhi_from_shares(shares: list[float]) -> int:
    """HHI = sum of squared shares, expressed in 0-10000 basis points."""
    total = sum(s * s * 10000 for s in shares if s is not None)
    return int(round(total))


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


@dataclass
class HhiMarketShareCalculator:
    """Deterministic HHI math operator + extracted-value verifier."""

    agent_id: str = "antitrust.hhi_market_share.v1"
    version: int = 1
    enabled: bool = True
    priority: int = 100
    capability_tags: tuple[str, ...] = (
        "calculate.hhi", "compute.numerical", "verify.extraction",
    )
    supported_domain_profiles: tuple[str, ...] = ("legal:1",)
    phases: tuple[str, ...] = ("pre_synthesis",)
    exclusive_group: Optional[str] = None
    deterministic: bool = True

    # Structural-presumption thresholds (2023 Merger Guidelines)
    presumption_post_hhi: int = 1800
    presumption_delta: int = 100

    # ------------------------------------------------------------------
    # match
    # ------------------------------------------------------------------

    def match(self, invocation: AgentInvocation) -> Optional[AgentMatch]:
        """Match when the task family signals HHI/market-share work."""
        # We rely on capability-tag policy + the registry's phase filter.
        # An additional defensive gate: agent only runs when the
        # invocation's task touches antitrust/regulatory work.
        family = (invocation.execution_family or "").lower()
        workflow = (invocation.workflow_kind or "").lower()
        ok_family = family in {"investigate", "extract", "compare"} or not family
        if not ok_family:
            return None
        # In phase 3 we don't gate by a dedicated antitrust signal — the
        # agent reads market_row typed_evidence; if there are none, the
        # invoke() returns success with zero artifacts. That's intentional:
        # cheap to run, valuable when relevant.
        return AgentMatch(
            agent_id=self.agent_id,
            score=0.85,
            reasons=("market_row_eligible",),
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
            rows = self._load_market_rows(runtime)
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
                warnings=("no_market_row_evidence",),
            )

        artifacts: list[AgentArtifact] = []
        for row in rows:
            try:
                art = self._compute_for_row(row, warnings)
            except Exception as exc:
                warnings.append(f"row_compute_error:{exc}")
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
        """Sanity-check: every artifact must carry computed HHI fields."""
        if result.status != "success":
            return result
        for a in result.artifacts:
            if a.artifact_kind != "hhi.calculation":
                continue
            payload = a.payload or {}
            if "computed_post_hhi" not in payload:
                return AgentInvocationResult(
                    status="invalid",
                    error_class="MalformedArtifact",
                    error=f"artifact {a.artifact_key} missing computed_post_hhi",
                    elapsed_ms=result.elapsed_ms,
                    warnings=result.warnings,
                )
        return result

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _load_market_rows(self, runtime: Any) -> list[Mapping[str, Any]]:
        """Pull market_row typed_evidence from the matter model."""
        mm = runtime.matter_model
        if mm is None:
            return []
        try:
            rows = mm.typed_evidence.list_by_kind("market_row", limit=200)
        except Exception:
            return []
        out: list[dict] = []
        for r in rows:
            payload_raw = r.get("payload_json") or "{}"
            try:
                payload = _json.loads(payload_raw) if isinstance(payload_raw, str) else dict(payload_raw)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("schema_ref") != "legal.market_row.v1":
                continue
            payload["_typed_evidence_id"] = r.get("id")
            out.append(payload)
        return out

    def _compute_for_row(
        self,
        row: Mapping[str, Any],
        warnings: list[str],
    ) -> Optional[AgentArtifact]:
        market = row.get("market_name") or row.get("geographic_market") or ""
        if not market:
            return None

        acq = _parse_share(row.get("acquirer_share"))
        tgt = _parse_share(row.get("target_share"))
        others_raw = row.get("other_shares") or []
        others: list[tuple[str, float]] = []
        for o in others_raw if isinstance(others_raw, list) else []:
            if not isinstance(o, dict):
                continue
            v = _parse_share(o.get("share"))
            if v is not None:
                others.append((str(o.get("name") or ""), v))

        all_shares: list[float] = []
        if acq is not None:
            all_shares.append(acq)
        if tgt is not None:
            all_shares.append(tgt)
        all_shares.extend(s for _, s in others)

        # Pre-merger HHI: include acq + tgt + others (each as separate firms)
        # Post-merger HHI: combined acq+tgt as one firm + others
        if not all_shares:
            return None
        pre_hhi = _hhi_from_shares(all_shares)
        if acq is not None and tgt is not None:
            combined = acq + tgt
            post_shares = [combined] + [s for _, s in others]
            post_hhi = _hhi_from_shares(post_shares)
            delta = post_hhi - pre_hhi
        else:
            post_hhi = None
            delta = None

        share_sum = sum(all_shares)
        if share_sum > 1.05:
            warnings.append(f"share_sum_over_100:{market}:{share_sum:.3f}")

        # Compare against extracted values
        extracted_pre = _safe_int(row.get("pre_merger_hhi"))
        extracted_post = _safe_int(row.get("post_merger_hhi"))
        extracted_delta = _safe_int(row.get("delta_hhi"))
        discrepancies: dict[str, dict] = {}
        if extracted_pre is not None and abs(extracted_pre - pre_hhi) > 50:
            discrepancies["pre_merger_hhi"] = {
                "extracted": extracted_pre, "computed": pre_hhi,
            }
        if (post_hhi is not None and extracted_post is not None
                and abs(extracted_post - post_hhi) > 50):
            discrepancies["post_merger_hhi"] = {
                "extracted": extracted_post, "computed": post_hhi,
            }
        if (delta is not None and extracted_delta is not None
                and abs(extracted_delta - delta) > 50):
            discrepancies["delta_hhi"] = {
                "extracted": extracted_delta, "computed": delta,
            }

        presumption_triggered = (
            post_hhi is not None and post_hhi >= self.presumption_post_hhi
            and delta is not None and delta >= self.presumption_delta
        )

        normalized_market = (
            market.replace(" ", "_").replace(",", "").replace(".", "").lower()
        )
        artifact_key = f"hhi:{normalized_market}"
        payload = {
            "schema_ref": "agent.hhi_calculation.v1",
            "market_name": market,
            "acquirer_share_fraction": acq,
            "target_share_fraction": tgt,
            "other_shares_fraction": [
                {"name": n, "share": s} for n, s in others
            ],
            "computed_pre_hhi": pre_hhi,
            "computed_post_hhi": post_hhi,
            "computed_delta_hhi": delta,
            "extracted_pre_hhi": extracted_pre,
            "extracted_post_hhi": extracted_post,
            "extracted_delta_hhi": extracted_delta,
            "structural_presumption": bool(presumption_triggered),
            "share_sum_check": round(share_sum, 4),
            "discrepancies": discrepancies,
        }
        label = (
            f"HHI {market}: pre={pre_hhi} post={post_hhi} "
            f"Δ={delta if delta is not None else '—'}"
        )
        return AgentArtifact(
            artifact_kind="hhi.calculation",
            artifact_key=artifact_key,
            payload=payload,
            label=label,
            synthesis_visibility="answer_ingredient",
            confidence=1.0 if not discrepancies else 0.7,
            typed_evidence_refs=(row.get("_typed_evidence_id"),)
                                if row.get("_typed_evidence_id") else (),
            verification_state="verified" if not discrepancies else "candidate",
        )


def _safe_int(v: Any) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None
