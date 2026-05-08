"""Sub-agent operator substrate (PR#3).

Architectural North Star (Operator Substrate Thesis):
1. Irys should stop treating benchmark failures as prompt-shape problems
   and treat them as missing operators.
2. Sub-agents are bounded operators, not mini chatbots.
3. Personas govern professional policy and voice.
4. Scoped prompts inside those contracts are allowed; prompt-only changes
   are not an acceptable fix.

This package exposes:
  - SubAgent Protocol + supporting dataclasses
  - SubAgentRegistry with capability-tag dispatch + budget enforcement
  - SubAgentDispatcher coordinating phase invocations
  - AgentRuntime (broker integration + LLM client wrapper)
  - PersonaPolicy with capability-tag allowlist/denylist + explicit overrides
"""

from .contracts import (
    AgentArtifact,
    AgentInputRef,
    AgentInvocation,
    AgentInvocationResult,
    AgentMatch,
    AgentRequirement,
    AgentTaskView,
    OperatorBudget,
    OperatorBudgetExceeded,
    SubAgent,
)
from .registry import AgentDispatch, SubAgentRegistry
from .dispatcher import SubAgentDispatcher
from .runtime import AgentRuntime
from .personas import Persona, PersonaPolicy, PersonaRegistry, PersonaSelection, PolicyDecision
from .hhi_calculator import HhiMarketShareCalculator
from .numerical_reconciliation import NumericalReconciliationAgent

__all__ = (
    "AgentArtifact",
    "AgentDispatch",
    "HhiMarketShareCalculator",
    "NumericalReconciliationAgent",
    "AgentInputRef",
    "AgentInvocation",
    "AgentInvocationResult",
    "AgentMatch",
    "AgentRequirement",
    "AgentRuntime",
    "AgentTaskView",
    "OperatorBudget",
    "OperatorBudgetExceeded",
    "Persona",
    "PersonaPolicy",
    "PersonaRegistry",
    "PersonaSelection",
    "PolicyDecision",
    "SubAgent",
    "SubAgentDispatcher",
    "SubAgentRegistry",
)
