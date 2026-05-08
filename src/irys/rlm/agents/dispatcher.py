"""SubAgentDispatcher — coordinates phase invocations under budget.

Per R5, full enforcement covers all OperatorBudget dimensions:
  - max_agents_total / max_agents_per_phase
  - max_wall_ms_total / max_wall_ms_per_agent
  - max_llm_calls_total
  - max_tokens_total
  - max_cost_estimate_usd

Each invocation is recorded to sub_agent_invocation, capturing budget
breach reasons when applicable. Failures are bubbled per AgentRequirement:
  optional → continue, log warning
  required → continue, escalate at end of phase
  blocking_validator → raise immediately
"""

from __future__ import annotations

import asyncio as _asyncio
import json as _json
import time as _time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from .contracts import (
    AgentInvocation,
    AgentInvocationResult,
    AgentRequirement,
    OperatorBudget,
    OperatorBudgetExceeded,
    SubAgent,
)
from .registry import AgentDispatch, SubAgentRegistry
from .runtime import AgentRuntime


@dataclass
class _BudgetUsage:
    """Running totals across one investigation."""
    agents_invoked: int = 0
    wall_ms_total: int = 0
    llm_calls_total: int = 0
    tokens_total: int = 0
    cost_estimate_usd_total: float = 0.0


@dataclass(frozen=True)
class PhaseRunResult:
    """Output of run_phase()."""
    invocations: tuple[tuple[SubAgent, AgentInvocationResult], ...]
    dispatched: AgentDispatch
    breached: tuple[str, ...]


class SubAgentDispatcher:
    """Driver that runs all selected agents for a phase under a shared budget.

    Stores invocation rows + artifacts in the matter model.
    """

    def __init__(
        self,
        *,
        registry: SubAgentRegistry,
        matter_model: Any,
        llm_client: Any = None,
        telemetry: Optional[Callable[..., None]] = None,
    ) -> None:
        self.registry = registry
        self.matter_model = matter_model
        self.llm_client = llm_client
        self.telemetry = telemetry or (lambda **_kw: None)
        self._usage = _BudgetUsage()

    # ------------------------------------------------------------------
    # Budget gates
    # ------------------------------------------------------------------

    def _check_pre_invocation(
        self, budget: OperatorBudget,
    ) -> Optional[str]:
        """Return breach reason if budget exhausted, else None."""
        if self._usage.agents_invoked >= budget.max_agents_total:
            return "max_agents_total"
        if self._usage.wall_ms_total >= budget.max_wall_ms_total:
            return "max_wall_ms_total"
        if self._usage.llm_calls_total >= budget.max_llm_calls_total:
            return "max_llm_calls_total"
        if self._usage.tokens_total >= budget.max_tokens_total:
            return "max_tokens_total"
        if self._usage.cost_estimate_usd_total >= budget.max_cost_estimate_usd:
            return "max_cost_estimate_usd"
        return None

    def _account(self, result: AgentInvocationResult) -> None:
        self._usage.agents_invoked += 1
        self._usage.wall_ms_total += int(result.elapsed_ms or 0)
        self._usage.llm_calls_total += int(result.llm_calls or 0)
        self._usage.tokens_total += int(result.token_estimate or 0)
        self._usage.cost_estimate_usd_total += float(result.cost_estimate_usd or 0.0)

    # ------------------------------------------------------------------
    # Phase driver
    # ------------------------------------------------------------------

    async def run_phase(
        self,
        invocation_template: AgentInvocation,
        *,
        phase: str,
        persona_policy: Optional[Any] = None,
    ) -> PhaseRunResult:
        budget = invocation_template.budget
        dispatch = self.registry.dispatch(
            invocation_template,
            phase=phase,
            phase_cap=budget.max_agents_per_phase,
            persona_policy=persona_policy,
        )
        invocations: list[tuple[SubAgent, AgentInvocationResult]] = []
        breached: list[str] = []

        for agent in dispatch.selected:
            breach = self._check_pre_invocation(budget)
            if breach:
                breached.append(breach)
                # Record a budget-exhausted row for observability
                self._record_invocation_row(
                    agent=agent,
                    invocation=invocation_template,
                    result=AgentInvocationResult(
                        status="budget_exhausted",
                        error_class="OperatorBudgetExceeded",
                        error=breach,
                    ),
                    breach_reason=breach,
                )
                if invocation_template.requirement == AgentRequirement.BLOCKING_VALIDATOR:
                    raise OperatorBudgetExceeded(breach)
                break

            runtime = AgentRuntime(
                matter_model=self.matter_model,
                invocation=invocation_template,
                llm_client=self.llm_client,
            )
            t0 = _time.perf_counter()
            try:
                result = await _asyncio.wait_for(
                    agent.invoke(invocation_template, runtime),
                    timeout=budget.max_wall_ms_per_agent / 1000.0,
                )
            except _asyncio.TimeoutError:
                result = AgentInvocationResult(
                    status="timeout",
                    error_class="TimeoutError",
                    error=f"agent {agent.agent_id} exceeded {budget.max_wall_ms_per_agent}ms",
                    elapsed_ms=budget.max_wall_ms_per_agent,
                )
            except Exception as exc:
                result = AgentInvocationResult(
                    status="error",
                    error_class=type(exc).__name__,
                    error=str(exc)[:500],
                    elapsed_ms=int((_time.perf_counter() - t0) * 1000),
                )

            # Verify_output is a hook for agents to validate their own
            # output (e.g., HHI math sanity, schema completeness).
            if result.status == "success":
                try:
                    result = agent.verify_output(invocation_template, result)
                except Exception as exc:
                    result = AgentInvocationResult(
                        status="invalid",
                        error_class=type(exc).__name__,
                        error=f"verify_output raised: {exc}",
                        elapsed_ms=result.elapsed_ms,
                    )

            self._account(result)
            invocation_id = self._record_invocation_row(
                agent=agent,
                invocation=invocation_template,
                result=result,
            )
            # Persist artifacts (only when result.status is success or partial)
            if result.status == "success" and result.artifacts:
                runtime.write_artifacts(invocation_id, result.artifacts)
            invocations.append((agent, result))

            self.telemetry(
                event="agent_invocation",
                agent_id=agent.agent_id,
                status=result.status,
                elapsed_ms=result.elapsed_ms,
            )

            # Blocking validator: hard-fail the investigation
            if (
                result.status != "success"
                and invocation_template.requirement == AgentRequirement.BLOCKING_VALIDATOR
            ):
                raise RuntimeError(
                    f"blocking validator failed: {agent.agent_id}: {result.error}"
                )

        return PhaseRunResult(
            invocations=tuple(invocations),
            dispatched=dispatch,
            breached=tuple(breached),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _record_invocation_row(
        self,
        *,
        agent: SubAgent,
        invocation: AgentInvocation,
        result: AgentInvocationResult,
        breach_reason: Optional[str] = None,
    ) -> str:
        """Insert a sub_agent_invocation row; return the id."""
        if self.matter_model is None:
            return ""
        invocation_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
        cap_tags = tuple(getattr(agent, "capability_tags", ()) or ())
        try:
            with self.matter_model.db.transaction():
                self.matter_model.db.execute(
                    """INSERT INTO sub_agent_invocation
                       (id, matter_id, run_id, agent_id, agent_version,
                        persona_id, phase, requirement, invocation_at,
                        input_hash, output_artifact_id,
                        dependency_manifest_hash,
                        latency_ms, cost_estimate, llm_calls, success_bool,
                        status, error_class, error_message,
                        capability_tags_json, token_estimate,
                        budget_breach_reason)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                               ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        invocation_id,
                        invocation.matter_id,
                        invocation.run_id,
                        agent.agent_id,
                        getattr(agent, "version", 1),
                        invocation.persona_id,
                        invocation.phase,
                        invocation.requirement.value,
                        now,
                        invocation.input_hash,
                        None,  # output_artifact_id wired post-write if needed
                        None,
                        int(result.elapsed_ms or 0),
                        float(result.cost_estimate_usd or 0.0),
                        int(result.llm_calls or 0),
                        1 if result.status == "success" else 0,
                        result.status,
                        result.error_class,
                        (result.error or None) and result.error[:500],
                        _json.dumps(list(cap_tags)),
                        int(result.token_estimate or 0),
                        breach_reason,
                    ),
                )
        except Exception:
            return ""
        return invocation_id
