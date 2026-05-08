"""Sub-agent contract dataclasses + Protocol.

Operator Substrate Thesis: Sub-agents are bounded operators, not mini
chatbots. Every agent has a typed contract (inputs → outputs), a
deterministic-where-possible behavior, a failure policy, and a
budget allowance.

These dataclasses are immutable (frozen) so they can flow through async
boundaries and serialize to telemetry without aliasing surprises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Protocol, Sequence


class AgentRequirement(str, Enum):
    """How a failed agent invocation affects the outer investigation."""
    OPTIONAL = "optional"
    REQUIRED = "required"
    BLOCKING_VALIDATOR = "blocking_validator"


class OperatorBudgetExceeded(RuntimeError):
    """Raised when an enforced budget dimension is breached."""
    pass


# ---------------------------------------------------------------------------
# Operator budget — full enforcement coverage per R5 patch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OperatorBudget:
    """Per-investigation budget for agent invocations.

    All dimensions are enforced by the dispatcher; pre-invocation checks
    fail fast, post-invocation accounting maintains running totals.
    """
    max_agents_total: int = 6
    max_agents_per_phase: int = 3
    max_wall_ms_total: int = 45_000
    max_wall_ms_per_agent: int = 15_000
    max_llm_calls_total: int = 4
    max_tokens_total: int = 40_000
    max_cost_estimate_usd: float = 0.75

    @classmethod
    def from_execution_contract(cls, contract: Any | None) -> "OperatorBudget":
        """Build a budget from ExecutionContract.output_contract.operator_budget.

        Falls back to defaults if the field is absent. Unknown keys are
        ignored to avoid version-skew breakage when contract evolves.
        """
        if contract is None:
            return cls()
        output_contract = getattr(contract, "output_contract", None) or {}
        raw = output_contract.get("operator_budget") or {}
        valid = {k: raw[k] for k in raw if k in cls.__dataclass_fields__}
        return cls(**valid)

    @classmethod
    def for_mode(cls, research_mode: str) -> "OperatorBudget":
        """Per-mode default budgets.

        Default operators are deterministic + cheap (no LLM calls), so the
        per-phase cap intentionally allows all built-in operators to run
        even in simple mode. The wall-time budget bounds total cost.
        """
        mode = (research_mode or "simple").lower()
        if mode == "simple":
            return cls(
                max_agents_total=8,
                max_agents_per_phase=8,
                max_wall_ms_total=45_000,
                max_wall_ms_per_agent=10_000,
                max_llm_calls_total=3,
                max_tokens_total=8_000,
                max_cost_estimate_usd=0.10,
            )
        return cls()  # deep / sebih_special use wider defaults


# ---------------------------------------------------------------------------
# Task view — passed into agent.match()
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentTaskView:
    """Sanitized view of the current task spec for agent matching.

    Agents see only what they need. Domain composition, matter id, and
    investigation state are passed via AgentInvocation, not here.
    """
    task_type: str = ""
    operation: str = ""
    required_evidence: tuple[str, ...] = ()
    fresh_extraction_required: bool = False
    cached_state_allowed: bool = True
    external_tool_required: bool = False
    answer_shape: str = "narrative_answer"

    @classmethod
    def normalize(cls, task_spec: Any) -> "AgentTaskView":
        if task_spec is None:
            return cls()
        if hasattr(task_spec, "to_dict"):
            data = task_spec.to_dict()
        elif hasattr(task_spec, "__dict__") and not isinstance(task_spec, dict):
            data = dict(task_spec.__dict__)
        else:
            data = dict(task_spec or {})
        return cls(
            task_type=str(data.get("task_type") or ""),
            operation=str(data.get("operation") or ""),
            required_evidence=tuple(data.get("required_evidence") or ()),
            fresh_extraction_required=bool(data.get("fresh_extraction_required", False)),
            cached_state_allowed=bool(data.get("cached_state_allowed", True)),
            external_tool_required=bool(data.get("external_tool_required", False)),
            answer_shape=str(data.get("answer_shape") or "narrative_answer"),
        )


# ---------------------------------------------------------------------------
# Inputs / outputs / invocation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentInputRef:
    """Reference to a piece of state an agent reads (for provenance tracking)."""
    kind: str
    id: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    dependency_manifest_hash: Optional[str] = None


@dataclass(frozen=True)
class AgentInvocation:
    """One call to a sub-agent — fully self-contained context."""
    matter_id: str
    run_id: str
    agent_id: str
    phase: str
    persona_id: Optional[str]
    requirement: AgentRequirement
    task: AgentTaskView
    execution_family: str
    workflow_kind: str
    budget: OperatorBudget
    input_refs: tuple[AgentInputRef, ...]
    input_hash: str
    capability_tags: tuple[str, ...] = ()
    domain_profile_id: str = ""
    domain_profile_version: int = 0
    work_profile: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentArtifact:
    """A durable agent output — persisted to agent_artifact table."""
    artifact_kind: str
    artifact_key: str
    payload: Mapping[str, Any]
    label: str = ""
    synthesis_visibility: str = "answer_ingredient"  # none|answer_ingredient|audit_only
    confidence: float = 0.0
    source_refs: tuple[str, ...] = ()
    typed_evidence_refs: tuple[str, ...] = ()
    dependency_manifest_hash: Optional[str] = None
    memory_packet_id: Optional[str] = None
    verification_state: str = "candidate"


@dataclass(frozen=True)
class AgentInvocationResult:
    """Outcome of a single agent.invoke() call."""
    status: str  # success|skipped|invalid|timeout|error|budget_exhausted
    artifacts: tuple[AgentArtifact, ...] = ()
    warnings: tuple[str, ...] = ()
    error_class: Optional[str] = None
    error: Optional[str] = None
    elapsed_ms: int = 0
    cost_estimate_usd: float = 0.0
    llm_calls: int = 0
    token_estimate: int = 0


@dataclass(frozen=True)
class AgentMatch:
    """Result of agent.match() — does this agent want to run for this invocation?"""
    agent_id: str
    score: float
    reasons: tuple[str, ...] = ()
    requirement: AgentRequirement = AgentRequirement.OPTIONAL
    phase: str = "pre_synthesis"
    exclusive_group: Optional[str] = None


# ---------------------------------------------------------------------------
# SubAgent Protocol
# ---------------------------------------------------------------------------


class SubAgent(Protocol):
    """A bounded operator. Not a chatbot.

    Required attributes:
      agent_id, version, enabled, priority,
      capability_tags, supported_domain_profiles, phases,
      exclusive_group, deterministic
    """
    agent_id: str
    version: int
    enabled: bool
    priority: int
    capability_tags: tuple[str, ...]
    supported_domain_profiles: tuple[str, ...]
    phases: tuple[str, ...]
    exclusive_group: Optional[str]
    deterministic: bool

    def match(self, invocation: AgentInvocation) -> Optional[AgentMatch]: ...

    async def invoke(
        self,
        invocation: AgentInvocation,
        runtime: Any,
    ) -> AgentInvocationResult: ...

    def verify_output(
        self,
        invocation: AgentInvocation,
        result: AgentInvocationResult,
    ) -> AgentInvocationResult: ...
