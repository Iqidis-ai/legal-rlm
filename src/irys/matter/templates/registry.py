"""Template registry — pure Python config, no DB access.

Consumers pass registry entries to IssueStore.apply_template(), which
writes issue_predicate rows with template_id/element_key metadata so
the proof substrate can compute per-element sufficiency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class TemplateElement:
    """One claim element within an IssueTemplate.

    mapping_hints is a tuple of exact lower-case tokens the assertion
    mapper should prefer when proposing which element a given assertion
    supports. Matching is token-level and deterministic (Jaccard-style)
    to keep MVP.5 from needing an LLM. More sophisticated mapping is a
    later phase.
    """

    key: str
    label: str
    description: str
    order: int = 0
    burden_side: Optional[str] = None  # "plaintiff", "defendant", or None
    mapping_hints: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class IssueTemplate:
    """A legal claim template — e.g. contract_breach or negligence.

    version is a simple semver string. When a template changes
    materially, bump the version so existing issues that applied the
    old template don't silently get new elements on the next re-apply.
    """

    id: str
    version: str
    issue_type: str
    burden_side: Optional[str]
    elements: tuple[TemplateElement, ...]
    description: str = ""


class TemplateRegistry:
    """Immutable registry of built-in templates."""

    def __init__(self, templates: "list[IssueTemplate] | tuple[IssueTemplate, ...]"):
        by_id: dict[str, IssueTemplate] = {}
        for tpl in templates:
            if tpl.id in by_id:
                raise ValueError(f"duplicate template id {tpl.id!r}")
            by_id[tpl.id] = tpl
        self._by_id = by_id

    def list(self) -> list[IssueTemplate]:
        return list(self._by_id.values())

    def get(self, template_id: str) -> Optional[IssueTemplate]:
        return self._by_id.get(template_id)

    def require(self, template_id: str) -> IssueTemplate:
        tpl = self._by_id.get(template_id)
        if tpl is None:
            raise KeyError(
                f"template {template_id!r} not registered; "
                f"available: {sorted(self._by_id.keys())}"
            )
        return tpl

    def score_assertion_to_elements(
        self,
        assertion_text: str,
        template: IssueTemplate,
    ) -> list[tuple[str, float]]:
        """Deterministic token-overlap scoring of an assertion against every
        element in the template. Returns [(element_key, score)] ordered by
        score desc; 0.0 when no tokens match.

        Matching is lower-case token intersection normalized by
        max(|assertion_tokens|, |hints|). MVP.5 uses this to propose
        edges from IssueStore.propose_element_mappings; non-zero scores
        are advisory only — they never resolve a predicate on their own
        because all edges seed as candidate (MVP.3 + MVP.2 contract).
        """
        text_tokens = {
            t.strip(".,;:()\"'").lower()
            for t in (assertion_text or "").split()
            if len(t.strip(".,;:()\"'")) >= 4
        }
        scored: list[tuple[str, float]] = []
        for el in template.elements:
            hint_tokens = set(el.mapping_hints)
            hint_tokens.update(
                t.lower() for t in el.label.split() if len(t) >= 4
            )
            if not hint_tokens or not text_tokens:
                scored.append((el.key, 0.0))
                continue
            overlap = len(hint_tokens & text_tokens)
            norm = max(len(hint_tokens), len(text_tokens), 1)
            scored.append((el.key, overlap / norm))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored


# ------------------------------------------------------------------
# Built-in templates
# ------------------------------------------------------------------

_CONTRACT_BREACH = IssueTemplate(
    id="contract_breach",
    version="v1",
    issue_type="claim",
    burden_side="plaintiff",
    description=(
        "Standard common-law breach-of-contract claim. Plaintiff must "
        "prove formation, own performance or excuse, defendant's breach, "
        "causation of loss, and quantifiable damages."
    ),
    elements=(
        TemplateElement(
            key="formation",
            label="Contract formation",
            description=(
                "A valid, enforceable contract was formed between the "
                "parties (offer, acceptance, consideration, definite terms)."
            ),
            order=0,
            burden_side="plaintiff",
            mapping_hints=("contract", "agreement", "offer", "acceptance",
                           "consideration", "signed", "executed", "formation"),
        ),
        TemplateElement(
            key="performance_or_excuse",
            label="Plaintiff's performance or legal excuse",
            description=(
                "Plaintiff performed its obligations under the contract, "
                "or was legally excused from performance."
            ),
            order=1,
            burden_side="plaintiff",
            mapping_hints=("performance", "performed", "delivered", "tender",
                           "excuse", "waiver", "prevention", "hindrance"),
        ),
        TemplateElement(
            key="breach",
            label="Defendant's breach",
            description=(
                "Defendant failed to perform a material obligation under "
                "the contract."
            ),
            order=2,
            burden_side="plaintiff",
            mapping_hints=("breach", "failed", "nonpayment", "unpaid", "default",
                           "repudiation", "rejected", "refused", "violated"),
        ),
        TemplateElement(
            key="causation",
            label="Causation of loss",
            description=(
                "Defendant's breach caused plaintiff's damages — the loss "
                "would not have occurred but for the breach."
            ),
            order=3,
            burden_side="plaintiff",
            mapping_hints=("caused", "resulted", "because", "consequence",
                           "foreseeable", "proximate"),
        ),
        TemplateElement(
            key="damages",
            label="Damages",
            description=(
                "Plaintiff suffered quantifiable damages recoverable under "
                "the applicable measure of damages."
            ),
            order=4,
            burden_side="plaintiff",
            mapping_hints=("damages", "loss", "amount", "invoice", "balance",
                           "unpaid", "expenses", "costs"),
        ),
    ),
)


_NEGLIGENCE = IssueTemplate(
    id="negligence",
    version="v1",
    issue_type="claim",
    burden_side="plaintiff",
    description=(
        "Basic common-law negligence. Plaintiff must prove duty, breach, "
        "causation, and damages. MVP.5 combines actual and proximate "
        "causation into a single element to keep scope narrow."
    ),
    elements=(
        TemplateElement(
            key="duty",
            label="Duty of care",
            description=(
                "Defendant owed plaintiff a legally cognizable duty of care."
            ),
            order=0,
            burden_side="plaintiff",
            mapping_hints=("duty", "obligation", "standard", "reasonable",
                           "care", "relationship"),
        ),
        TemplateElement(
            key="breach",
            label="Breach of duty",
            description=(
                "Defendant's conduct fell below the applicable standard of "
                "care."
            ),
            order=1,
            burden_side="plaintiff",
            mapping_hints=("breach", "failed", "negligent", "unreasonable",
                           "substandard", "reckless"),
        ),
        TemplateElement(
            key="causation",
            label="Causation",
            description=(
                "Defendant's breach actually and proximately caused the "
                "plaintiff's injury."
            ),
            order=2,
            burden_side="plaintiff",
            mapping_hints=("caused", "proximate", "foreseeable", "because",
                           "consequence", "resulted", "but_for"),
        ),
        TemplateElement(
            key="damages",
            label="Damages",
            description=(
                "Plaintiff suffered legally cognizable damages (bodily, "
                "property, or economic as applicable)."
            ),
            order=3,
            burden_side="plaintiff",
            mapping_hints=("damages", "injury", "harm", "loss", "expenses",
                           "medical", "treatment"),
        ),
    ),
)


def default_registry() -> TemplateRegistry:
    """Registry of built-in MVP.5 templates.

    Expanded in later phases. External callers should import and call
    this rather than instantiating TemplateRegistry directly so the
    built-in set stays authoritative.
    """
    return TemplateRegistry([_CONTRACT_BREACH, _NEGLIGENCE])
