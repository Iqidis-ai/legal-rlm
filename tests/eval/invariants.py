"""Invariant registry for the eval harness.

Each invariant is a named callable with a required-capability tag set. Tests
run only the invariants whose capabilities are implemented today — everything
else skips with a clear reason. Violations raise InvariantViolation so
failure messages stay specific and avoid snapshot-diff noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .harness import HarnessResult
from .schema import InvariantSpec


# Capabilities implemented as of MVP.1 (PR.1 + PR.2 + PR.3 landed).
# Invariants that require a capability NOT in this set skip with a clear
# reason instead of failing. When a future MVP step ships the missing
# capability, add the tag here and every gated invariant activates.
IMPLEMENTED_CAPABILITIES: set[str] = {
    "schema_v49",  # PR.1 schema discipline gate
    "proof_state_partial",  # ProofStateStore writes trust_weighted_* and proof_status
    "coverage_report",  # get_issue_coverage_report is canonical
    "mandatory_context_packet",  # PR.3 capped coverage + gap sections
    "proof_gap_detection",  # _detect_proof_gaps opens missing_issue_predicate gaps
}

# Capabilities that are intentionally NOT yet implemented and gate future
# fixture invariants. Documenting them here keeps the capability map honest
# and lets fixtures reference stable tags before the production work lands.
PLANNED_CAPABILITIES: set[str] = {
    "evidence_edge_backfill",  # MVP.3 EvidenceEdgeStore + legacy-link backfill
    "verification_state",  # MVP.2 candidate/verified separation
    "privilege_containment",  # MVP.4 clean-output gating on privileged support
}


class InvariantViolation(AssertionError):
    """Raised when a fixture's declared invariant does not hold.

    The str(exc) must identify fixture + mode + invariant name and include a
    one-line explanation of what failed and why.
    """


@dataclass(frozen=True)
class Invariant:
    name: str
    group: str
    requires: tuple[str, ...]
    check: Callable[[HarnessResult, dict[str, Any]], None]

    def is_activatable(self) -> bool:
        return all(r in IMPLEMENTED_CAPABILITIES for r in self.requires)


# ---------------------------------------------------------------------------
# Concrete invariants
# ---------------------------------------------------------------------------


def _require_alias(result: HarnessResult, key: str) -> str:
    if key not in result.alias_to_id:
        raise InvariantViolation(
            f"invariant precondition — fixture alias {key!r} missing in "
            f"seeded ids; did the fixture forget to declare it?"
        )
    return result.alias_to_id[key]


def _issue_gap_created(result: HarnessResult, params: dict[str, Any]) -> None:
    issue_alias = params["issue_alias"]
    issue_id = _require_alias(result, f"issue:{issue_alias}")
    all_gaps = result.model.gaps.open_gaps(min_materiality=0.0)
    found = [
        g for g in all_gaps
        if g.get("gap_type") == "missing_issue_predicate"
        and any(
            d.get("affected_type") == "issue" and d.get("affected_id") == issue_id
            for d in (g.get("dependencies") or [])
        )
    ]
    if not found:
        raise InvariantViolation(
            f"expected >=1 open missing_issue_predicate gap linked to issue "
            f"alias {issue_alias!r}; found 0 across "
            f"{len(all_gaps)} open gap(s)"
        )


def _coverage_partial_and_nonzero(result: HarnessResult, params: dict[str, Any]) -> None:
    issue_alias = params["issue_alias"]
    issue_id = _require_alias(result, f"issue:{issue_alias}")
    report = result.model.get_issue_coverage_report()
    row = next((r for r in report if r["id"] == issue_id), None)
    if row is None:
        raise InvariantViolation(
            f"issue alias {issue_alias!r} missing from coverage report; "
            f"coverage_partial_and_nonzero cannot evaluate"
        )
    frac = float(row.get("coverage_fraction") or 0.0)
    if not (0.0 < frac < 1.0):
        raise InvariantViolation(
            f"coverage_fraction for issue {issue_alias!r} must be in (0,1); "
            f"got {frac!r}"
        )


def _mandatory_gap_section_present(result: HarnessResult, params: dict[str, Any]) -> None:
    """After PR.3, _build_capped_gap_section must render a nonempty section
    whenever at least one open gap at materiality >= 0.4 exists."""
    if result.engine is None:
        raise InvariantViolation(
            "mandatory_gap_section_present requires engine-stub mode"
        )
    query = params.get("query", "")
    section = result.engine._build_capped_gap_section(query, None)
    if not section or "Known Gaps" not in section:
        raise InvariantViolation(
            f"gap section missing from packet helper output; got {section!r}"
        )


def _planned_capability_placeholder(
    result: HarnessResult, params: dict[str, Any]
) -> None:
    """Placeholder for invariants whose capability has not shipped.

    Never actually runs because its capability tag is not in
    IMPLEMENTED_CAPABILITIES. If somehow it did run, raise so the failure
    is impossible to miss during future MVP work.
    """
    raise InvariantViolation(
        "planned-capability invariant executed but capability is not marked "
        "implemented; either activate the capability or remove this check"
    )


# Registry: invariant name -> definition.
_INVARIANTS: dict[str, Invariant] = {
    "issue_gap_created": Invariant(
        name="issue_gap_created",
        group="proof_gap",
        requires=("proof_gap_detection",),
        check=_issue_gap_created,
    ),
    "coverage_partial_and_nonzero": Invariant(
        name="coverage_partial_and_nonzero",
        group="proof_gap",
        requires=("coverage_report",),
        check=_coverage_partial_and_nonzero,
    ),
    "mandatory_gap_section_present": Invariant(
        name="mandatory_gap_section_present",
        group="context_packet",
        requires=("mandatory_context_packet",),
        check=_mandatory_gap_section_present,
    ),
    # MVP.3 — activates once EvidenceEdgeStore lands.
    "evidence_edge_backfill_idempotent": Invariant(
        name="evidence_edge_backfill_idempotent",
        group="evidence_edge",
        requires=("evidence_edge_backfill",),
        check=_planned_capability_placeholder,
    ),
    # MVP.2 — activates once verification_state substrate lands.
    "candidate_support_not_verified": Invariant(
        name="candidate_support_not_verified",
        group="verification",
        requires=("verification_state",),
        check=_planned_capability_placeholder,
    ),
    # MVP.4 — activates once privilege containment lands.
    "no_privileged_doc_in_clean_context": Invariant(
        name="no_privileged_doc_in_clean_context",
        group="privilege",
        requires=("privilege_containment",),
        check=_planned_capability_placeholder,
    ),
}


def resolve_invariant(spec: InvariantSpec) -> Invariant | None:
    """Return the registered Invariant for a spec, or None if it is gated
    behind a not-yet-implemented capability. Missing names raise so a
    fixture cannot silently reference a nonexistent invariant."""
    if spec.name not in _INVARIANTS:
        raise InvariantViolation(
            f"fixture references unknown invariant {spec.name!r}; "
            f"register it in tests/eval/invariants.py"
        )
    inv = _INVARIANTS[spec.name]
    if not inv.is_activatable():
        return None
    return inv


# Invariant groups that require an RLMEngine wired with a scripted client.
# In store mode these skip rather than fail, because store mode deliberately
# does not construct an engine.
_ENGINE_ONLY_GROUPS: set[str] = {"context_packet"}


def run_invariants(
    fixture_name: str, mode: str, result: HarnessResult
) -> list[tuple[str, str]]:
    """Run all activatable invariants declared on the fixture.

    Returns a list of (invariant_name, reason) tuples for every skipped
    invariant, where reason names the missing capability or the mode
    mismatch that caused the skip. Violated invariants raise
    InvariantViolation with fixture + mode + invariant name + message.
    """
    skipped: list[tuple[str, str]] = []
    for spec in result.fixture.invariants:
        if spec.name not in _INVARIANTS:
            raise InvariantViolation(
                f"fixture references unknown invariant {spec.name!r}; "
                f"register it in tests/eval/invariants.py"
            )
        inv = _INVARIANTS[spec.name]
        if not inv.is_activatable():
            missing = sorted(set(inv.requires) - IMPLEMENTED_CAPABILITIES)
            skipped.append(
                (spec.name, f"capability not implemented: {', '.join(missing)}")
            )
            continue
        if mode == "store" and inv.group in _ENGINE_ONLY_GROUPS:
            skipped.append(
                (spec.name, f"group {inv.group!r} requires engine-stub mode")
            )
            continue
        try:
            inv.check(result, spec.params)
        except InvariantViolation as exc:
            raise InvariantViolation(
                f"fixture={fixture_name} mode={mode} invariant={spec.name}: {exc}"
            ) from exc
    return skipped
