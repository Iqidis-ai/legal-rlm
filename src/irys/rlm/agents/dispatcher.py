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
        runtime_extras: Optional[dict] = None,
    ) -> None:
        self.registry = registry
        self.matter_model = matter_model
        self.llm_client = llm_client
        self.telemetry = telemetry or (lambda **_kw: None)
        # runtime_extras: attributes set on each AgentRuntime so agents can
        # access non-default things (e.g. _repo for DocumentFileReader,
        # _test_doc_text for tests).
        self.runtime_extras = dict(runtime_extras or {})
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

        # Use selected_with_match if available so each agent gets its own
        # invocation with the match's requirement (Codex Phase 2 round-1
        # blocker: AgentMatch.requirement was being ignored, meaning an
        # agent could not escalate itself to blocking_validator at runtime).
        if dispatch.selected_with_match:
            agent_match_pairs = list(dispatch.selected_with_match)
        else:
            agent_match_pairs = [(a, None) for a in dispatch.selected]

        # Strictness ranking — an agent's match can ESCALATE its own
        # requirement (e.g. obligation_coverage flagging itself as a
        # blocking_validator on deliverable family) but cannot DOWNGRADE
        # below the template's expectation for the phase.
        _STRICTNESS_RANK = {
            AgentRequirement.OPTIONAL: 0,
            AgentRequirement.REQUIRED: 1,
            AgentRequirement.BLOCKING_VALIDATOR: 2,
        }

        for agent, match in agent_match_pairs:
            from dataclasses import replace as _dc_replace
            template_req = invocation_template.requirement
            match_req = match.requirement if match is not None else template_req
            effective_req = (
                match_req
                if _STRICTNESS_RANK[match_req] > _STRICTNESS_RANK[template_req]
                else template_req
            )
            per_invocation = _dc_replace(
                invocation_template,
                requirement=effective_req,
                agent_id=agent.agent_id,
            )

            breach = self._check_pre_invocation(budget)
            if breach:
                breached.append(breach)
                # Record a budget-exhausted row for observability
                self._record_invocation_row(
                    agent=agent,
                    invocation=per_invocation,
                    result=AgentInvocationResult(
                        status="budget_exhausted",
                        error_class="OperatorBudgetExceeded",
                        error=breach,
                    ),
                    breach_reason=breach,
                )
                if per_invocation.requirement == AgentRequirement.BLOCKING_VALIDATOR:
                    raise OperatorBudgetExceeded(breach)
                break

            runtime = AgentRuntime(
                matter_model=self.matter_model,
                invocation=per_invocation,
                llm_client=self.llm_client,
            )
            for _attr, _val in self.runtime_extras.items():
                setattr(runtime, _attr, _val)
            t0 = _time.perf_counter()
            try:
                result = await _asyncio.wait_for(
                    agent.invoke(per_invocation, runtime),
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
            # CRITICAL: keep result.artifacts even when verify_output
            # downgrades status to 'invalid' (Codex Phase 2 r1 blocker:
            # validator_failure artifact must be persisted for audit).
            if result.status == "success":
                try:
                    result = agent.verify_output(per_invocation, result)
                except Exception as exc:
                    result = AgentInvocationResult(
                        status="invalid",
                        error_class=type(exc).__name__,
                        error=f"verify_output raised: {exc}",
                        elapsed_ms=result.elapsed_ms,
                        artifacts=result.artifacts,
                    )

            self._account(result)
            try:
                invocation_id = self._record_invocation_row(
                    agent=agent,
                    invocation=per_invocation,
                    result=result,
                )
            except Exception as _persist_exc:
                # Codex HOLD-4 fix: this failure is loud now. The current
                # agent's run is recorded as a failure result; we do NOT
                # then call write_artifacts() with no FK.
                import logging as _logging
                _logging.getLogger(__name__).error(
                    "agent %s recorded with status=success but persistence failed: %s",
                    agent.agent_id, _persist_exc,
                )
                invocation_id = ""

            # Persist artifacts whenever the operator produced any. Codex
            # Phase 2 r1 blocker: previously we required status='success',
            # which dropped validator_failure artifacts even though they
            # are explicitly the audit trail for blocking failures.
            artifact_persistence_failed = False
            if invocation_id and result.artifacts:
                try:
                    runtime.write_artifacts(invocation_id, result.artifacts)
                except Exception as _wa_exc:
                    artifact_persistence_failed = True
                    import logging as _logging
                    _logging.getLogger(__name__).error(
                        "agent %s write_artifacts failed: %s",
                        agent.agent_id, _wa_exc,
                    )
            invocations.append((agent, result))

            self.telemetry(
                event="agent_invocation",
                agent_id=agent.agent_id,
                status=result.status,
                elapsed_ms=result.elapsed_ms,
            )

            # Blocking validator: hard-fail the investigation. Read from
            # per_invocation (post-match override), not the template.
            if (
                result.status != "success"
                and per_invocation.requirement == AgentRequirement.BLOCKING_VALIDATOR
            ):
                # Codex Phase-2 r2: if write_artifacts failed, the audit
                # trail is missing. Surface it in the halt message so the
                # outer engine + observability layer can record it.
                halt_msg = f"blocking validator failed: {agent.agent_id}: {result.error}"
                if artifact_persistence_failed:
                    halt_msg += (
                        " | AUDIT-PERSISTENCE-FAILED: validator_failure artifact"
                        " could not be written"
                    )
                raise RuntimeError(halt_msg)
            # If a blocking validator SUCCEEDED but its audit artifact
            # write failed, that's also a contract violation — surface as
            # halt so we don't silently lose the run-level audit trail.
            if (
                result.status == "success"
                and per_invocation.requirement == AgentRequirement.BLOCKING_VALIDATOR
                and artifact_persistence_failed
                and result.artifacts
            ):
                raise RuntimeError(
                    f"blocking validator audit failed: {agent.agent_id}: "
                    "matrix artifact persistence error"
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
        """Insert a sub_agent_invocation row; return the id.

        Codex HOLD-4 fix: persistence failures are NOT swallowed silently.
        On exception, log loudly via logger.error and re-raise so the
        dispatcher's caller sees the failure (otherwise write_artifacts()
        would later get an empty FK and fail with a confusing
        IntegrityError).
        """
        import logging as _logging
        _log = _logging.getLogger(__name__)
        if self.matter_model is None:
            _log.error(
                "sub_agent_invocation NOT recorded for %s: matter_model is None",
                agent.agent_id,
            )
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
        except Exception as _exc:
            _log.error(
                "sub_agent_invocation INSERT failed for %s/%s: %s",
                agent.agent_id, invocation_id, _exc,
            )
            # Re-raise: the dispatcher's caller MUST see this so we don't
            # silently call write_artifacts() with an empty/invalid FK.
            raise
        return invocation_id
