"""SubAgentRegistry + dispatch.

Capability-tag dispatch (R5 patch): personas allow/deny capabilities, not
specific agent IDs. Adding a new agent with a known tag does not require
editing every persona.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from .contracts import (
    AgentInvocation,
    AgentMatch,
    SubAgent,
)


@dataclass(frozen=True)
class AgentDispatch:
    selected: tuple[SubAgent, ...]
    candidates: tuple[AgentMatch, ...]
    suppressed: tuple[Mapping[str, object], ...]
    # Per-agent (agent, match) pairs in the same order as `selected`.
    # Dispatcher uses match.requirement to build per-agent invocations
    # so an agent's blocking_validator escalation is honored end-to-end.
    selected_with_match: tuple[tuple[SubAgent, AgentMatch], ...] = ()


class SubAgentRegistry:
    """In-memory registry of installed sub-agents."""

    def __init__(self, agents: Sequence[SubAgent] = ()) -> None:
        self._agents: dict[str, SubAgent] = {}
        for agent in agents:
            self.register(agent)

    def register(self, agent: SubAgent) -> None:
        if agent.agent_id in self._agents:
            raise ValueError(f"duplicate sub-agent: {agent.agent_id}")
        self._agents[agent.agent_id] = agent

    def replace(self, agent: SubAgent) -> None:
        """Test convenience — replace an existing agent registration."""
        self._agents[agent.agent_id] = agent

    def get(self, agent_id: str) -> Optional[SubAgent]:
        return self._agents.get(agent_id)

    def list(self) -> tuple[SubAgent, ...]:
        return tuple(self._agents.values())

    def dispatch(
        self,
        invocation: AgentInvocation,
        *,
        phase: str,
        phase_cap: int,
        persona_policy: "Optional[PersonaPolicy]" = None,
    ) -> AgentDispatch:
        """Score-rank agents that match the invocation; apply policy + caps.

        Filtering order:
          1. enabled
          2. phase compatibility
          3. capability-tag policy (if persona policy given)
          4. agent.match() returns non-None
          5. exclusive group dedup
          6. phase cap
        """
        scored: list[tuple[SubAgent, AgentMatch]] = []
        suppressed: list[dict[str, object]] = []

        for agent in self._agents.values():
            if not getattr(agent, "enabled", True):
                suppressed.append({"agent_id": agent.agent_id, "reason": "disabled"})
                continue
            if phase not in getattr(agent, "phases", ()):
                suppressed.append({
                    "agent_id": agent.agent_id, "reason": "phase_mismatch",
                    "expected_phase": phase,
                })
                continue
            if persona_policy is not None:
                policy_decision = persona_policy.evaluate(agent)
                if not policy_decision.allowed:
                    suppressed.append({
                        "agent_id": agent.agent_id,
                        "reason": "persona_policy_denied",
                        "policy_reason": policy_decision.reason,
                    })
                    continue
            # Domain profile filtering — agents declare which domains
            # they support. If the invocation's domain doesn't match,
            # the agent is suppressed. Empty `supported_domain_profiles`
            # is treated as wildcard (cross-domain operator).
            invocation_domain = invocation.domain_profile_id or ""
            supported = tuple(getattr(agent, "supported_domain_profiles", ()) or ())
            if supported and invocation_domain and invocation_domain not in supported:
                suppressed.append({
                    "agent_id": agent.agent_id,
                    "reason": "domain_profile_mismatch",
                    "expected_one_of": list(supported),
                    "got": invocation_domain,
                })
                continue
            try:
                match = agent.match(invocation)
            except Exception as exc:
                suppressed.append({
                    "agent_id": agent.agent_id,
                    "reason": "match_exception",
                    "error_class": type(exc).__name__,
                })
                continue
            if match is not None:
                scored.append((agent, match))

        scored.sort(
            key=lambda item: (
                -item[1].score,
                -getattr(item[0], "priority", 0),
                item[0].agent_id,
            )
        )

        selected: list[SubAgent] = []
        selected_pairs: list[tuple[SubAgent, AgentMatch]] = []
        occupied: dict[str, str] = {}
        for agent, match in scored:
            group = match.exclusive_group or getattr(agent, "exclusive_group", None)
            if group and group in occupied:
                suppressed.append({
                    "agent_id": agent.agent_id,
                    "reason": "exclusive_group_lower_rank",
                    "winner": occupied[group],
                })
                continue
            if len(selected) >= phase_cap:
                suppressed.append({
                    "agent_id": agent.agent_id, "reason": "phase_cap",
                    "phase_cap": phase_cap,
                })
                continue
            selected.append(agent)
            selected_pairs.append((agent, match))
            if group:
                occupied[group] = agent.agent_id

        return AgentDispatch(
            selected=tuple(selected),
            candidates=tuple(m for _, m in scored),
            suppressed=tuple(suppressed),
            selected_with_match=tuple(selected_pairs),
        )


# Late import for type hint resolution
from .personas import PersonaPolicy  # noqa: E402
