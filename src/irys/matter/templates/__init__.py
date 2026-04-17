"""MVP.5 legal template registry (SO-4).

Templates are config-backed Python dataclasses — no YAML, no DB rows.
Each IssueTemplate names a list of TemplateElement entries (claim
elements, each with a predicate description and deterministic mapping
hints). IssueStore.apply_template() materializes template elements as
issue_predicate rows so the proof substrate can track per-element
candidate vs verified sufficiency.

MVP.5 ships two templates: contract_breach and negligence. Templates
are intentionally conservative for MVP — broader surface lands in
later phases.
"""

from .registry import (
    IssueTemplate,
    TemplateElement,
    TemplateRegistry,
    default_registry,
)

__all__ = [
    "IssueTemplate",
    "TemplateElement",
    "TemplateRegistry",
    "default_registry",
]
