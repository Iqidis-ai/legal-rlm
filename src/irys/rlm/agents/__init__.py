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
from .cp_section_extractor import CpSectionExtractorAgent
from .file_reader import DocumentFileReader
from .cross_doc_linker import CrossDocLinker


def default_registry() -> SubAgentRegistry:
    """Registry pre-populated with the deterministic built-in operators."""
    return SubAgentRegistry(agents=(
        DocumentFileReader(),
        CrossDocLinker(),
        HhiMarketShareCalculator(),
        NumericalReconciliationAgent(),
        CpSectionExtractorAgent(),
    ))


def default_persona_registry() -> PersonaRegistry:
    """Registry of default personas. By user directive: every persona has
    open access to every sub-agent (PersonaPolicy() = no_policy_default_allow).
    Restrictions are explicit opt-in for high-risk operators only.
    """
    return PersonaRegistry(personas=(
        Persona(
            persona_id="senior_ma_attorney",
            voice_summary="Senior M&A attorney; numbers-driven; cite-or-don't-claim",
            domain_profile_ids=("legal:1",),
            task_types=("corporate-ma",),
            workflow_kinds=("investigate",),
        ),
        Persona(
            persona_id="antitrust_economist",
            voice_summary="Antitrust economist; HHI-driven; structural presumption analysis",
            domain_profile_ids=("legal:1",),
            task_types=("antitrust-competition",),
            workflow_kinds=("investigate",),
        ),
        Persona(
            persona_id="banking_finance_attorney",
            voice_summary="Banking attorney; CP-coverage-driven; closing-set focused",
            domain_profile_ids=("legal:1",),
            task_types=("banking-finance",),
            workflow_kinds=("investigate",),
        ),
        # Generic fallback persona — used when no domain-specific persona matches
        Persona(persona_id="generalist", voice_summary="Generalist analyst"),
    ))

__all__ = (
    "AgentArtifact",
    "AgentDispatch",
    "default_persona_registry",
    "default_registry",
    "CpSectionExtractorAgent",
    "CrossDocLinker",
    "DocumentFileReader",
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
