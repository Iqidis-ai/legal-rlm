"""Unit tests for MVP.5 template registry + apply + element mapping."""

import pytest

from irys.matter import (
    AssertionKind,
    MatterModel,
    ModelLayer,
    SourceRole,
    SpeechAct,
)
from irys.matter.enums import (
    IssueType,
    OriginKind,
    ReviewedByKind,
    VerificationTargetKind,
)
from irys.matter.models import AssertionCandidate
from irys.matter.templates import (
    IssueTemplate,
    TemplateElement,
    TemplateRegistry,
    default_registry,
)


@pytest.fixture
def model():
    return MatterModel.open_in_memory()


def _add(model, text, doc="d.pdf"):
    cand = AssertionCandidate(
        proposition_text=text,
        speech_act=SpeechAct.OPERATIVE,
        source_role=SourceRole.OPERATIVE,
        assertion_kind=AssertionKind.FACTUAL,
        model_layer=ModelLayer.RECORD,
        document_id=doc,
        origin_kind=OriginKind.EXTRACTED,
    )
    aid, _ = model.assertions.upsert_occurrence(cand)
    return aid


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_default_registry_has_contract_breach_and_negligence():
    reg = default_registry()
    ids = {t.id for t in reg.list()}
    assert "contract_breach" in ids
    assert "negligence" in ids
    cb = reg.require("contract_breach")
    assert cb.version == "v1"
    assert [e.key for e in cb.elements] == [
        "formation", "performance_or_excuse", "breach", "causation", "damages",
    ]


def test_registry_rejects_duplicate_ids():
    dup = IssueTemplate(
        id="x", version="v1", issue_type="claim", burden_side="plaintiff",
        elements=(TemplateElement(key="a", label="A", description="A", order=0),),
    )
    dup2 = IssueTemplate(
        id="x", version="v2", issue_type="claim", burden_side="plaintiff",
        elements=(TemplateElement(key="b", label="B", description="B", order=0),),
    )
    with pytest.raises(ValueError, match="duplicate template id"):
        TemplateRegistry([dup, dup2])


def test_score_assertion_to_elements_ranks_by_overlap():
    reg = default_registry()
    cb = reg.require("contract_breach")
    scored = reg.score_assertion_to_elements(
        "Defendant failed to pay the invoice when due", cb
    )
    top_key, top_score = scored[0]
    assert top_score > 0.0
    assert top_key in {"breach", "damages"}


# ---------------------------------------------------------------------------
# apply_template
# ---------------------------------------------------------------------------

def test_apply_template_creates_predicates_with_metadata(model):
    iid, _ = model.issues.upsert_issue("Breach", IssueType.CLAIM)
    pred_ids = model.issues.apply_template(iid, "contract_breach")
    assert len(pred_ids) == 5
    preds = model.issues.get_predicates(iid)
    by_key = {p["element_key"]: p for p in preds}
    assert set(by_key.keys()) == {
        "formation", "performance_or_excuse", "breach", "causation", "damages",
    }
    assert by_key["formation"]["template_id"] == "contract_breach"
    assert by_key["formation"]["template_version"] == "v1"
    assert by_key["formation"]["element_order"] == 0
    assert by_key["damages"]["element_order"] == 4


def test_apply_template_is_idempotent(model):
    iid, _ = model.issues.upsert_issue("Breach", IssueType.CLAIM)
    first = model.issues.apply_template(iid, "contract_breach")
    second = model.issues.apply_template(iid, "contract_breach")
    assert first == second
    # Still only five rows for this template.
    rows = model.db.execute(
        "SELECT COUNT(*) FROM issue_predicate WHERE issue_id=? AND template_id=?",
        (iid, "contract_breach"),
    ).fetchone()[0]
    assert rows == 5


def test_apply_template_seeds_candidate_verification(model):
    iid, _ = model.issues.upsert_issue("Breach", IssueType.CLAIM)
    pred_ids = model.issues.apply_template(iid, "contract_breach")
    for pid in pred_ids:
        vs = model.verification.get(VerificationTargetKind.ISSUE_PREDICATE, pid)
        assert vs is not None
        assert vs["status"] == "candidate"


def test_apply_template_raises_on_unknown_template(model):
    iid, _ = model.issues.upsert_issue("Breach", IssueType.CLAIM)
    with pytest.raises(KeyError, match="not registered"):
        model.issues.apply_template(iid, "does_not_exist")


def test_apply_template_rejects_cross_matter_issue(model):
    iid, _ = model.issues.upsert_issue("Breach", IssueType.CLAIM)
    # Second matter, different issue store.
    other = MatterModel.open_in_memory()
    with pytest.raises(ValueError, match="does not exist"):
        other.issues.apply_template(iid, "contract_breach")


# ---------------------------------------------------------------------------
# link_assertion_to_predicate + propose_element_mappings
# ---------------------------------------------------------------------------

def test_link_assertion_to_predicate_creates_edge(model):
    iid, _ = model.issues.upsert_issue("Breach", IssueType.CLAIM)
    pred_ids = model.issues.apply_template(iid, "contract_breach")
    aid = _add(model, "Defendant failed to pay")
    breach_pid = pred_ids[2]
    edge_id = model.issues.link_assertion_to_predicate(aid, breach_pid, "supports")
    edges = model.evidence.list_edges_for_target("issue_predicate", breach_pid)
    assert len(edges) == 1
    assert edges[0]["id"] == edge_id
    assert edges[0]["source_id"] == aid
    assert edges[0]["target_kind"] == "issue_predicate"
    # Re-link must stay idempotent.
    edge_id2 = model.issues.link_assertion_to_predicate(aid, breach_pid, "supports")
    assert edge_id2 == edge_id


def test_propose_element_mappings_returns_ranked_predicate_ids(model):
    iid, _ = model.issues.upsert_issue("Breach", IssueType.CLAIM)
    pred_ids = model.issues.apply_template(iid, "contract_breach")
    aid = _add(model, "Defendant failed to pay the invoice on the due date")
    proposals = model.issues.propose_element_mappings(aid, iid)
    assert proposals, "expected at least one proposal for a clearly-breach fact"
    top_pid = proposals[0][0]
    top_pred = next(p for p in model.issues.get_predicates(iid) if p["id"] == top_pid)
    assert top_pred["element_key"] in {"breach", "damages"}


def test_propose_element_mappings_empty_for_issue_without_template(model):
    iid, _ = model.issues.upsert_issue("Untemplated", IssueType.CLAIM)
    model.issues.add_predicate(iid, "Custom predicate")
    aid = _add(model, "Any fact")
    assert model.issues.propose_element_mappings(aid, iid) == []


# ---------------------------------------------------------------------------
# get_predicates_with_proof — per-element candidate vs verified sufficiency
# ---------------------------------------------------------------------------

def test_candidate_mapping_cannot_resolve_predicate(model):
    """MVP.5 AC #5: a candidate-only mapping must not make an element
    fully resolved, no matter how many candidate supports exist."""
    iid, _ = model.issues.upsert_issue("Breach", IssueType.CLAIM)
    pred_ids = model.issues.apply_template(iid, "contract_breach")
    breach_pid = pred_ids[2]
    for i in range(5):
        aid = _add(model, f"Candidate breach fact {i}", doc=f"d{i}.pdf")
        model.issues.link_assertion_to_predicate(aid, breach_pid, "supports")

    preds = model.issues.get_predicates_with_proof(iid)
    breach_p = next(p for p in preds if p["element_key"] == "breach")
    assert breach_p["supporting_count"] == 5
    assert breach_p["verified_supporting_count"] == 0
    assert breach_p["verified_sufficiency"] == 0.0
    assert breach_p["resolvable"] is False


def test_verified_mapping_can_resolve_predicate(model):
    iid, _ = model.issues.upsert_issue("Breach", IssueType.CLAIM)
    pred_ids = model.issues.apply_template(iid, "contract_breach")
    breach_pid = pred_ids[2]
    aid = _add(model, "Defendant failed to pay invoice")
    model.issues.link_assertion_to_predicate(aid, breach_pid, "supports")
    # Verify both the assertion AND the edge — both are required for the
    # verified lane per MVP.5 contract.
    model.verification.verify(
        VerificationTargetKind.ASSERTION, aid,
        reviewed_by_kind=ReviewedByKind.ATTORNEY,
    )
    edge = model.evidence.list_edges_for_target("issue_predicate", breach_pid)[0]
    model.verification.verify(
        VerificationTargetKind.EVIDENCE_EDGE, edge["id"],
        reviewed_by_kind=ReviewedByKind.ATTORNEY,
    )
    preds = model.issues.get_predicates_with_proof(iid)
    breach_p = next(p for p in preds if p["element_key"] == "breach")
    assert breach_p["verified_sufficiency"] > 0.0
    assert breach_p["resolvable"] is True


def test_verified_assertion_candidate_edge_does_not_resolve(model):
    """Both the assertion AND the mapping edge must be verified. A
    verified assertion attached via a still-candidate mapping edge must
    not reach the verified lane."""
    iid, _ = model.issues.upsert_issue("Breach", IssueType.CLAIM)
    pred_ids = model.issues.apply_template(iid, "contract_breach")
    breach_pid = pred_ids[2]
    aid = _add(model, "Defendant failed to pay")
    model.issues.link_assertion_to_predicate(aid, breach_pid, "supports")
    model.verification.verify(
        VerificationTargetKind.ASSERTION, aid,
        reviewed_by_kind=ReviewedByKind.ATTORNEY,
    )
    preds = model.issues.get_predicates_with_proof(iid)
    breach_p = next(p for p in preds if p["element_key"] == "breach")
    assert breach_p["verified_sufficiency"] == 0.0
    assert breach_p["resolvable"] is False
