"""Persona registry + capability-tag policy.

Personas govern professional policy and voice. They do NOT restrict which
sub-agents can run by default — every persona can access every sub-agent
based on whatever the task needs. Otherwise we'd keep missing details by
artificially scoping which operators are available per persona.

The capability-tag policy is an OPT-IN safety mechanism for high-risk
operators only. Default behavior:
  - Empty allowed_tags + empty denied_tags + no overrides = allow ALL agents
  - Set denied_tags only for actually-dangerous capabilities (e.g.,
    irreversible writes, costly external API calls)
  - Set explicit_agent_overrides[agent_id]=False for narrow per-agent denials
  - Wildcard "*" in allowed_tags is the same as the empty-default

The richer policy machinery is preserved for the rare case where a
persona must be sandboxed (e.g., a "Read-Only Reviewer" persona that
must not invoke any write/compute operators), not as the default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str = ""


@dataclass(frozen=True)
class PersonaPolicy:
    """Capability-tag allowlist/denylist + explicit overrides.

    Evaluation order:
      1. Explicit override (allow/deny by agent_id) — wins everything
      2. Denied tags — if any agent tag matches, deny
      3. Allowed tags — if no agent tag matches, deny
      4. Otherwise — allow

    Wildcards: "*" in allowed_tags is "allow all (subject to deny rules)".
    """
    allowed_tags: tuple[str, ...] = ()
    denied_tags: tuple[str, ...] = ()
    explicit_agent_overrides: Mapping[str, bool] = field(default_factory=dict)

    def evaluate(self, agent: Any) -> PolicyDecision:
        agent_id = getattr(agent, "agent_id", "")
        if agent_id in self.explicit_agent_overrides:
            allowed = bool(self.explicit_agent_overrides[agent_id])
            return PolicyDecision(
                allowed=allowed,
                reason=f"explicit_override:{'allow' if allowed else 'deny'}",
            )
        agent_tags = set(getattr(agent, "capability_tags", ()) or ())
        denied = set(self.denied_tags)
        if agent_tags & denied:
            return PolicyDecision(
                allowed=False,
                reason=f"denied_tag:{sorted(agent_tags & denied)[0]}",
            )
        allowed_set = set(self.allowed_tags)
        if "*" in allowed_set:
            return PolicyDecision(allowed=True, reason="wildcard_allow")
        if not (agent_tags & allowed_set):
            if not allowed_set:
                # No allowlist defined → permissive default
                return PolicyDecision(allowed=True, reason="no_policy_default_allow")
            return PolicyDecision(
                allowed=False,
                reason="no_allowed_tag_match",
            )
        return PolicyDecision(allowed=True, reason="allowed_tag_match")


@dataclass(frozen=True)
class Persona:
    """A bounded professional persona.

    Personas have:
      - persona_id (unique)
      - voice / system prompt scope
      - capability-tag policy (which agents may run)
      - selection criteria (when this persona applies)
    """
    persona_id: str
    version: int = 1
    voice_summary: str = ""
    domain_profile_ids: tuple[str, ...] = ()
    task_types: tuple[str, ...] = ()
    workflow_kinds: tuple[str, ...] = ()
    policy: PersonaPolicy = field(default_factory=PersonaPolicy)
    enabled: bool = True
    priority: int = 100


@dataclass(frozen=True)
class PersonaSelection:
    persona_id: str
    persona_version: int
    score: float
    reasons: tuple[str, ...]
    policy: PersonaPolicy


class PersonaRegistry:
    def __init__(self, personas: Sequence[Persona] = ()) -> None:
        self._personas: dict[str, Persona] = {}
        for p in personas:
            self.register(p)

    def register(self, persona: Persona) -> None:
        if persona.persona_id in self._personas:
            raise ValueError(f"duplicate persona: {persona.persona_id}")
        self._personas[persona.persona_id] = persona

    def get(self, persona_id: str) -> Optional[Persona]:
        return self._personas.get(persona_id)

    def list(self) -> tuple[Persona, ...]:
        return tuple(self._personas.values())

    def select(
        self,
        *,
        task_type: str = "",
        workflow_kind: str = "",
        domain_profile_id: str = "",
    ) -> Optional[PersonaSelection]:
        """Pick the highest-scoring enabled persona for the given task.

        Score = number of matching dimensions (task_type, workflow_kind,
        domain). Ties broken by priority. Returns None when no personas
        match (caller falls back to default behavior).
        """
        best: Optional[tuple[float, Persona, list[str]]] = None
        for p in self._personas.values():
            if not p.enabled:
                continue
            score = 0.0
            reasons: list[str] = []
            if task_type and (not p.task_types or task_type in p.task_types):
                if p.task_types:
                    score += 1.0
                    reasons.append(f"task:{task_type}")
            if workflow_kind and (not p.workflow_kinds
                                  or workflow_kind in p.workflow_kinds):
                if p.workflow_kinds:
                    score += 1.0
                    reasons.append(f"workflow:{workflow_kind}")
            if domain_profile_id and (not p.domain_profile_ids
                                       or domain_profile_id in p.domain_profile_ids):
                if p.domain_profile_ids:
                    score += 1.0
                    reasons.append(f"domain:{domain_profile_id}")
            if score == 0 and not (p.task_types or p.workflow_kinds
                                    or p.domain_profile_ids):
                # Fully generic persona — eligible at lowest score
                score = 0.5
                reasons.append("generic_default")
            if score == 0:
                continue
            if best is None or (
                score > best[0] or (score == best[0] and p.priority > best[1].priority)
            ):
                best = (score, p, reasons)

        if best is None:
            return None
        score, persona, reasons = best
        return PersonaSelection(
            persona_id=persona.persona_id,
            persona_version=persona.version,
            score=score,
            reasons=tuple(reasons),
            policy=persona.policy,
        )
