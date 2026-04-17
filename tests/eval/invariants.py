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
IMPLEMENTED_CAPABILITIES: set[str] = {
    "schema_v49",
    "proof_state_partial",  # compute_and_store writes trust_weighted_*, proof_status
    "coverage_report",  # get_issue_coverage_report is canonical
    "mandatory_context_packet",  # PR.3 capped coverage + gap sections
    "proof_gap_detection",  # _detect_proof_gaps maps to missing_issue_predicate
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
) -> list[str]:
    """Run all activatable invariants declared on the fixture.

    Returns a list of invariant names that were skipped because either the
    capability is not yet implemented or the invariant's group needs an
    engine that the current mode does not provide.
    """
    skipped: list[str] = []
    for spec in result.fixture.invariants:
        inv = resolve_invariant(spec)
        if inv is None:
            skipped.append(spec.name)
            continue
        if mode == "store" and inv.group in _ENGINE_ONLY_GROUPS:
            skipped.append(spec.name)
            continue
        try:
            inv.check(result, spec.params)
        except InvariantViolation as exc:
            raise InvariantViolation(
                f"fixture={fixture_name} mode={mode} invariant={spec.name}: {exc}"
            ) from exc
    return skipped
