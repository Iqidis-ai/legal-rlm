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
    "verification_state",  # MVP.2 VerificationStateStore + candidate/verified columns
    "evidence_edge_backfill",  # MVP.3 EvidenceStore + v53 backfill
    "privilege_containment",  # MVP.4 document-level privilege filter in clean mode
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


def _evidence_edge_backfill_idempotent(
    result: HarnessResult, params: dict[str, Any]
) -> None:
    """MVP.3: EvidenceStore.backfill_from_legacy_links must create exactly
    one evidence_edge per legacy assertion_issue_link, preserve proof-state
    counts, and remain idempotent on rerun. Backfilled rows carry
    source_identity_status='missing_occurrence_span' because current link
    APIs don't preserve occurrence/span identity.
    """
    issue_alias = params["issue_alias"]
    expected = int(params.get("expected_edge_count", 0))
    issue_id = _require_alias(result, f"issue:{issue_alias}")
    model = result.model

    # Legacy link count for the issue.
    legacy_count = model.db.execute(
        """SELECT COUNT(*) FROM assertion_issue_link
           WHERE issue_id=?
             AND relation_type IN ('supports','establishes','attacks','negates')""",
        (issue_id,),
    ).fetchone()[0]
    if legacy_count != expected:
        raise InvariantViolation(
            f"fixture precondition — expected {expected} legacy links for issue "
            f"{issue_alias!r}; found {legacy_count}"
        )

    # Baseline: proof state as computed from the legacy substrate.
    model.proof_state.compute_and_store(issue_id)
    baseline = model.proof_state.get(issue_id)

    # First backfill must insert exactly the expected number of edges.
    new_edges_first = model.evidence.backfill_from_legacy_links()
    if new_edges_first != expected:
        raise InvariantViolation(
            f"first backfill must insert exactly {expected} new edges for "
            f"issue {issue_alias!r}; got {new_edges_first}"
        )
    edges_for_issue = model.evidence.list_edges_for_target("issue", issue_id)
    if len(edges_for_issue) != expected:
        raise InvariantViolation(
            f"expected {expected} evidence_edge rows after first backfill for "
            f"issue {issue_alias!r}; got {len(edges_for_issue)}"
        )

    # All backfilled rows must carry missing-identity markers.
    for e in edges_for_issue:
        if (
            e.get("source_occurrence_id") is not None
            or e.get("source_span_id") is not None
            or e.get("source_identity_status") != "missing_occurrence_span"
        ):
            raise InvariantViolation(
                "backfilled edge must have null occurrence/span identity "
                "and source_identity_status='missing_occurrence_span'; got "
                f"{e!r}"
            )

    # Second backfill must be idempotent — zero new edges.
    new_edges_second = model.evidence.backfill_from_legacy_links()
    edges_for_issue_2 = model.evidence.list_edges_for_target("issue", issue_id)
    if new_edges_second != 0 or len(edges_for_issue_2) != expected:
        raise InvariantViolation(
            f"backfill must be idempotent — second call produced "
            f"{new_edges_second} new edges, total {len(edges_for_issue_2)}"
        )

    # Proof counts must not shift — the substrate switch preserves math.
    model.proof_state.compute_and_store(issue_id)
    after = model.proof_state.get(issue_id)
    for field in ("supporting_count", "attacking_count"):
        if baseline.get(field) != after.get(field):
            raise InvariantViolation(
                f"proof_state.{field} diverged after backfill: "
                f"before={baseline.get(field)} after={after.get(field)}"
            )


def _no_privileged_doc_in_clean_context(
    result: HarnessResult, params: dict[str, Any]
) -> None:
    """MVP.4: under clean policy, privileged assertions/documents do not
    contribute to coverage/proof, and the privileged-only support case
    surfaces as a missing_issue_predicate gap. Also verifies that the
    privileged document is not visible in the coverage report via any
    supporting_count attribution."""
    from irys.rlm.engine import RLMEngine

    issue_alias = params["issue_alias"]
    issue_id = _require_alias(result, f"issue:{issue_alias}")
    model = result.model

    # Coverage report in clean mode must report zero supporting_count and
    # verified_supporting_count for the privileged-only issue.
    clean_row = next(
        (r for r in model.get_issue_coverage_report("clean") if r["id"] == issue_id),
        None,
    )
    if clean_row is None:
        raise InvariantViolation(
            f"issue alias {issue_alias!r} missing from clean coverage report"
        )
    if clean_row["supporting_count"] != 0:
        raise InvariantViolation(
            f"clean mode must report zero supporting_count for a "
            f"privileged-only issue; got {clean_row['supporting_count']}"
        )
    if clean_row["verified_supporting_count"] != 0:
        raise InvariantViolation(
            f"clean mode must report zero verified_supporting_count; "
            f"got {clean_row['verified_supporting_count']}"
        )

    # Internal mode must still see the privileged support.
    internal_row = next(
        r for r in model.get_issue_coverage_report("internal") if r["id"] == issue_id
    )
    if internal_row["supporting_count"] == 0:
        raise InvariantViolation(
            "fixture precondition — internal mode must see at least one "
            "privileged support for the issue"
        )

    # Proof-state substrate under clean policy must also see zero support.
    model.proof_state.compute_and_store(issue_id, policy_audience="clean")
    ps = model.proof_state.get(issue_id)
    if ps and ps["supporting_count"] != 0:
        raise InvariantViolation(
            f"proof_state in clean mode must report zero supporting_count "
            f"for a privileged-only issue; got {ps['supporting_count']}"
        )

    # _detect_proof_gaps must open a missing_issue_predicate gap for the
    # privileged-only issue because clean support is zero.
    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = model
    engine._detect_proof_gaps(policy_audience="clean")
    gaps = model.gaps.open_gaps(min_materiality=0.0)
    linked = [
        g for g in gaps
        if g.get("gap_type") == "missing_issue_predicate"
        and any(
            d.get("affected_type") == "issue" and d.get("affected_id") == issue_id
            for d in (g.get("dependencies") or [])
        )
    ]
    if not linked:
        raise InvariantViolation(
            f"clean-mode gap detection must open a missing_issue_predicate "
            f"gap for issue {issue_alias!r} when only privileged support exists"
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


def _candidate_support_not_verified(
    result: HarnessResult, params: dict[str, Any]
) -> None:
    """MVP.2 SO-2: candidate-only support cannot resolve an issue to verified
    proof. Fixture declares one verified and one candidate assertion both
    linked as 'supports' to the same issue. The coverage report must expose
    the verified/candidate split, and verified coverage must be strictly
    less than total coverage — otherwise the candidate support is being
    treated as verified.
    """
    issue_alias = params["issue_alias"]
    candidate_alias = params["candidate_assertion_alias"]
    issue_id = _require_alias(result, f"issue:{issue_alias}")
    candidate_aid = _require_alias(result, f"assertion:{candidate_alias}")

    report = result.model.get_issue_coverage_report()
    row = next((r for r in report if r["id"] == issue_id), None)
    if row is None:
        raise InvariantViolation(
            f"issue alias {issue_alias!r} missing from coverage report"
        )

    verified_count = int(row.get("verified_supporting_count") or 0)
    candidate_count = int(row.get("candidate_supporting_count") or 0)
    verified_cov = float(row.get("verified_coverage_fraction") or 0.0)
    total_cov = float(row.get("coverage_fraction") or 0.0)

    if candidate_count <= 0:
        raise InvariantViolation(
            f"issue {issue_alias!r} must report at least one candidate "
            f"supporting assertion; got candidate_supporting_count={candidate_count}"
        )
    if verified_cov >= total_cov:
        raise InvariantViolation(
            f"verified_coverage_fraction ({verified_cov}) must be strictly "
            f"less than coverage_fraction ({total_cov}) when candidate "
            f"support exists on issue {issue_alias!r}"
        )
    # Candidate assertion must have status='candidate' in verification_state.
    from irys.matter.enums import VerificationTargetKind
    vs = result.model.verification.get(VerificationTargetKind.ASSERTION, candidate_aid)
    if vs is None or vs.get("status") != "candidate":
        raise InvariantViolation(
            f"candidate assertion alias {candidate_alias!r} must have "
            f"verification_state.status='candidate'; got "
            f"{(vs or {}).get('status')!r}"
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
    # MVP.3 — EvidenceEdgeStore + v53 backfill landed.
    "evidence_edge_backfill_idempotent": Invariant(
        name="evidence_edge_backfill_idempotent",
        group="evidence_edge",
        requires=("evidence_edge_backfill",),
        check=_evidence_edge_backfill_idempotent,
    ),
    # MVP.2 — verification_state substrate landed.
    "candidate_support_not_verified": Invariant(
        name="candidate_support_not_verified",
        group="verification",
        requires=("verification_state",),
        check=_candidate_support_not_verified,
    ),
    # MVP.4 — document-level privilege containment landed.
    "no_privileged_doc_in_clean_context": Invariant(
        name="no_privileged_doc_in_clean_context",
        group="privilege",
        requires=("privilege_containment",),
        check=_no_privileged_doc_in_clean_context,
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
