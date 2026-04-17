"""Unit tests for document-level privilege containment (MVP.4, SO-5).

Covers PrivilegeGate helpers and the clean-mode filter applied to
get_issue_coverage_report, ProofStateStore.compute_and_store, and
RLMEngine._detect_proof_gaps.
"""

import pytest

from irys.matter import (
    AssertionKind,
    MatterModel,
    ModelLayer,
    SourceRole,
    SpeechAct,
)
from irys.matter.enums import IssueType, OriginKind
from irys.matter.models import AssertionCandidate


@pytest.fixture
def model():
    return MatterModel.open_in_memory()


def _seed_doc(model, path, privilege_flag=False):
    inv_id, _ = model.inventory.upsert(path, f"sha_{path}".ljust(64, "0")[:64], size_bytes=100)
    model.document_cards.upsert(doc_id=inv_id, title=path, privilege_flag=privilege_flag)
    return inv_id


def _seed_assertion(model, text, doc_path):
    cand = AssertionCandidate(
        proposition_text=text,
        speech_act=SpeechAct.OPERATIVE,
        source_role=SourceRole.OPERATIVE,
        assertion_kind=AssertionKind.FACTUAL,
        model_layer=ModelLayer.RECORD,
        document_id=doc_path,
        origin_kind=OriginKind.EXTRACTED,
    )
    aid, _ = model.assertions.upsert_occurrence(cand)
    return aid


def test_privilege_gate_detects_document_by_path(model):
    _seed_doc(model, "internal/memo.docx", privilege_flag=True)
    _seed_doc(model, "contracts/msa.pdf", privilege_flag=False)
    assert model.privilege.is_document_privileged("internal/memo.docx") is True
    assert model.privilege.is_document_privileged("contracts/msa.pdf") is False
    assert model.privilege.is_document_privileged("does/not/exist.pdf") is False


def test_privilege_gate_detects_assertion_via_occurrence(model):
    _seed_doc(model, "internal/memo.docx", privilege_flag=True)
    _seed_doc(model, "contracts/msa.pdf", privilege_flag=False)
    priv_aid = _seed_assertion(model, "privileged fact", "internal/memo.docx")
    clean_aid = _seed_assertion(model, "clean fact", "contracts/msa.pdf")
    assert model.privilege.is_assertion_from_privileged_source(priv_aid) is True
    assert model.privilege.is_assertion_from_privileged_source(clean_aid) is False
    assert model.privilege.privileged_assertion_ids() == {priv_aid}


def test_clean_coverage_excludes_privileged_support(model):
    _seed_doc(model, "priv.docx", privilege_flag=True)
    _seed_doc(model, "clean.pdf", privilege_flag=False)
    priv_aid = _seed_assertion(model, "p", "priv.docx")
    clean_aid = _seed_assertion(model, "c", "clean.pdf")
    iid, _ = model.issues.upsert_issue("Claim", IssueType.CLAIM, materiality=0.7)
    model.issues.link_assertion(priv_aid, iid, "supports")
    model.issues.link_assertion(clean_aid, iid, "supports")

    clean = next(r for r in model.get_issue_coverage_report("clean") if r["id"] == iid)
    internal = next(r for r in model.get_issue_coverage_report("internal") if r["id"] == iid)
    assert clean["supporting_count"] == 1
    assert internal["supporting_count"] == 2


def test_clean_coverage_excludes_privileged_attacks(model):
    _seed_doc(model, "priv.docx", privilege_flag=True)
    priv_aid = _seed_assertion(model, "privileged attack", "priv.docx")
    iid, _ = model.issues.upsert_issue("Claim", IssueType.CLAIM, materiality=0.7)
    model.issues.link_assertion(priv_aid, iid, "attacks")

    clean = next(r for r in model.get_issue_coverage_report("clean") if r["id"] == iid)
    internal = next(r for r in model.get_issue_coverage_report("internal") if r["id"] == iid)
    assert clean["attacking_count"] == 0
    assert internal["attacking_count"] == 1


def test_proof_state_clean_mode_drops_privileged_support(model):
    _seed_doc(model, "priv.docx", privilege_flag=True)
    priv_aid = _seed_assertion(model, "priv", "priv.docx")
    iid, _ = model.issues.upsert_issue("Claim", IssueType.CLAIM, materiality=0.7)
    model.issues.link_assertion(priv_aid, iid, "supports")

    model.proof_state.compute_and_store(iid, policy_audience="clean")
    clean_ps = model.proof_state.get(iid)
    assert clean_ps["supporting_count"] == 0

    model.proof_state.compute_and_store(iid, policy_audience="internal")
    internal_ps = model.proof_state.get(iid)
    assert internal_ps["supporting_count"] == 1


def test_detect_proof_gaps_opens_gap_when_support_is_privileged_only(model):
    """Clean-mode gap detection must open a missing_issue_predicate gap on
    an issue whose only support traces back to a privileged document."""
    from irys.rlm.engine import RLMEngine

    _seed_doc(model, "internal/memo.docx", privilege_flag=True)
    priv_aid = _seed_assertion(model, "privileged support", "internal/memo.docx")
    iid, _ = model.issues.upsert_issue(
        "High-materiality claim", IssueType.CLAIM, materiality=0.9,
    )
    model.issues.link_assertion(priv_aid, iid, "supports")

    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = model
    engine._detect_proof_gaps(policy_audience="clean")

    gaps = model.gaps.open_gaps(min_materiality=0.0)
    linked = [
        g for g in gaps
        if g.get("gap_type") == "missing_issue_predicate"
        and any(
            d.get("affected_type") == "issue" and d.get("affected_id") == iid
            for d in (g.get("dependencies") or [])
        )
    ]
    assert linked, (
        "clean-mode gap detection must open a missing_issue_predicate gap "
        "when only privileged support exists"
    )


def test_privilege_flag_preserved_across_profile_refresh(model):
    """MVP.4 AC #5: trust overrides (or any profile refresh) cannot clear
    privilege. DocumentCardStore.upsert without an explicit privilege_flag
    must leave a previously-privileged card privileged."""
    inv_id, _ = model.inventory.upsert("internal/memo.docx", "sha".ljust(64, "0"), size_bytes=1)
    # First upsert marks the card privileged.
    model.document_cards.upsert(doc_id=inv_id, title="memo", privilege_flag=True)
    # Second upsert comes from a profile refresh that doesn't know about
    # privilege (e.g. a source-role reclassification). MVP.4 requires the
    # existing privilege flag to survive.
    model.document_cards.upsert(doc_id=inv_id, title="memo (refreshed)", source_role="operative")
    row = model.db.execute(
        "SELECT privilege_flag, title, source_role FROM document_card WHERE doc_id=?",
        (inv_id,),
    ).fetchone()
    assert row["privilege_flag"] == 1, (
        "privilege_flag must not be cleared by a profile refresh that "
        "omits the privilege_flag kwarg"
    )
    assert row["title"] == "memo (refreshed)"
    assert row["source_role"] == "operative"

    # An explicit privilege_flag=False IS allowed to clear the flag —
    # that path is a deliberate attorney-level declassification.
    model.document_cards.upsert(doc_id=inv_id, privilege_flag=False)
    row = model.db.execute(
        "SELECT privilege_flag FROM document_card WHERE doc_id=?", (inv_id,),
    ).fetchone()
    assert row["privilege_flag"] == 0


def test_edge_substrate_respects_privilege_filter(model):
    """The edge-first proof substrate branch must also drop privileged
    edges. Adversarial audit flagged that filters have to land on every
    substrate — edge or legacy."""
    from irys.matter.enums import EvidenceRelationType

    _seed_doc(model, "priv.docx", privilege_flag=True)
    _seed_doc(model, "clean.pdf", privilege_flag=False)
    priv_aid = _seed_assertion(model, "p", "priv.docx")
    clean_aid = _seed_assertion(model, "c", "clean.pdf")
    iid, _ = model.issues.upsert_issue("Claim", IssueType.CLAIM, materiality=0.7)
    # Use link_assertion so evidence_edge is written (forces edge substrate).
    model.issues.link_assertion(priv_aid, iid, "supports")
    model.issues.link_assertion(clean_aid, iid, "supports")

    model.proof_state.compute_and_store(iid, policy_audience="clean")
    ps = model.proof_state.get(iid)
    assert ps["supporting_count"] == 1, (
        "edge substrate must drop privileged edges under clean policy"
    )
