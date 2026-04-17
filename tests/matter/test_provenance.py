"""P0.1 Provenance Lite tests (SO-2).

Covers:
- LLMCallRecord carries call_id, prompt_hash, response_hash (client
  responsibility; here we verify the matter-side persistence works).
- ProvenanceStore.record + list_for_target round-trip.
- Five AI-derived writer paths attach provenance:
  assertion, quant_fact, authority, document_card, evidence_edge.
- source_span_status explicitly records 'missing' when span identity
  is absent per P0.1 AC #4.
- get_provenance queryable by (target_kind, target_id).
"""

import pytest

from irys.core.models import LLMCallRecord
from irys.matter import (
    AssertionKind,
    MatterModel,
    ModelLayer,
    SourceRole,
    SpeechAct,
)
from irys.matter.enums import (
    EvidenceRelationType,
    IssueType,
    OriginKind,
)
from irys.matter.models import AssertionCandidate, ProvenanceContext


@pytest.fixture
def model():
    return MatterModel.open_in_memory()


def _prov(**overrides) -> ProvenanceContext:
    # run_id defaults to None so tests don't need to bootstrap a
    # run_session row to exercise provenance plumbing. Tests that
    # specifically verify run-linked provenance should start a real
    # run and override run_id.
    base = dict(
        event_kind="assertion_extraction",
        writer_name="AssertionStore.upsert_occurrence",
        run_id=None,
        model_id="gemini-test",
        model_tier="FLASH",
        prompt_version="AX.DOC_ASSERTION.v1",
        extractor_version="2026-04-17.p01.v1",
        llm_call_id="call_abc",
        prompt_hash="p" * 64,
        response_hash="r" * 64,
        source_document_ref="contracts/msa.pdf",
        source_span_id="span_1",
        source_span_status="present",
    )
    base.update(overrides)
    return ProvenanceContext(**base)


# ---------------------------------------------------------------------------
# llm_call persistence
# ---------------------------------------------------------------------------

def test_record_llm_call_persists_call_id_and_hashes(model):
    model.record_llm_call(LLMCallRecord(
        model_tier="FLASH",
        model_id="gemini-test",
        input_tokens=100,
        cache_read_tokens=0,
        output_tokens=50,
        total_prompt_tokens=100,
        estimated_cost_usd=0.001,
        latency_ms=500,
        success=True,
        call_id="call_unique_id",
        prompt_hash="p" * 64,
        response_hash="r" * 64,
    ))
    row = model.db.execute(
        "SELECT id, prompt_hash, response_hash FROM llm_call WHERE id=?",
        ("call_unique_id",),
    ).fetchone()
    assert row is not None
    assert row["prompt_hash"] == "p" * 64
    assert row["response_hash"] == "r" * 64


# ---------------------------------------------------------------------------
# ProvenanceStore round-trip
# ---------------------------------------------------------------------------

def test_provenance_store_record_and_list(model):
    ctx = _prov()
    eid = model.provenance.record(
        target_kind="assertion", target_id="a1", context=ctx,
    )
    assert eid
    rows = model.provenance.list_for_target("assertion", "a1")
    assert len(rows) == 1
    assert rows[0]["event_kind"] == "assertion_extraction"
    assert rows[0]["writer_name"] == "AssertionStore.upsert_occurrence"
    assert rows[0]["llm_call_id"] == "call_abc"
    assert rows[0]["prompt_hash"] == "p" * 64
    assert rows[0]["response_hash"] == "r" * 64
    assert rows[0]["source_span_status"] == "present"


def test_provenance_query_by_llm_call(model):
    ctx = _prov(llm_call_id="shared_call")
    model.provenance.record(target_kind="assertion", target_id="a1", context=ctx)
    model.provenance.record(target_kind="evidence_edge", target_id="e1", context=ctx)
    rows = model.provenance.list_for_llm_call("shared_call")
    assert len(rows) == 2
    target_kinds = {r["target_kind"] for r in rows}
    assert target_kinds == {"assertion", "evidence_edge"}


# ---------------------------------------------------------------------------
# Writer integrations (five paths)
# ---------------------------------------------------------------------------

def test_assertion_upsert_records_provenance_when_provided(model):
    cand = AssertionCandidate(
        proposition_text="Defendant owed payment",
        speech_act=SpeechAct.OPERATIVE,
        source_role=SourceRole.OPERATIVE,
        assertion_kind=AssertionKind.FACTUAL,
        model_layer=ModelLayer.RECORD,
        document_id="contracts/msa.pdf",
        origin_kind=OriginKind.EXTRACTED,
    )
    aid, is_new = model.assertions.upsert_occurrence(cand, provenance=_prov())
    assert is_new
    events = model.get_provenance("assertion", aid)
    assert len(events) >= 1
    assert events[0]["event_kind"] == "assertion_extraction"


def test_quant_record_records_provenance_when_provided(model):
    qid = model.quant.record(
        quant_kind="amount",
        raw_text="Invoice $100",
        amount_value=100.0,
        currency="USD",
        provenance=_prov(
            event_kind="quant_record",
            writer_name="QuantStore.record",
            prompt_version="QX.NUMERIC_FACT.v1",
        ),
    )
    events = model.get_provenance("quant_fact", qid)
    assert len(events) == 1
    assert events[0]["event_kind"] == "quant_record"


def test_authority_upsert_records_provenance_when_provided(model):
    auth_id, is_new = model.authority.upsert(
        "Smith v. Jones, 1 F.3d 100 (9th Cir. 2020)",
        provenance=_prov(
            event_kind="authority_upsert",
            writer_name="AuthorityStore.upsert",
            prompt_version="SPEC.AUTHORITY_TREATMENT.v1",
        ),
    )
    assert is_new
    events = model.get_provenance("authority", auth_id)
    assert len(events) == 1
    assert events[0]["writer_name"] == "AuthorityStore.upsert"


def test_document_card_upsert_records_provenance_when_provided(model):
    inv_id, _ = model.inventory.upsert(
        "contracts/msa.pdf", "a" * 64, size_bytes=1
    )
    card_id = model.document_cards.upsert(
        doc_id=inv_id,
        title="MSA",
        doc_type="contract",
        provenance=_prov(
            event_kind="card_profile",
            writer_name="DocumentCardStore.upsert",
            prompt_version="DI.DOC_TYPE.v1",
            source_span_status="not_applicable",
        ),
    )
    events = model.get_provenance("document_card", card_id)
    assert len(events) == 1
    assert events[0]["source_span_status"] == "not_applicable"


def test_evidence_edge_upsert_records_provenance_when_provided(model):
    iid, _ = model.issues.upsert_issue("Test", IssueType.CLAIM)
    cand = AssertionCandidate(
        proposition_text="fact",
        speech_act=SpeechAct.OPERATIVE,
        source_role=SourceRole.OPERATIVE,
        assertion_kind=AssertionKind.FACTUAL,
        model_layer=ModelLayer.RECORD,
        document_id="d.pdf",
        origin_kind=OriginKind.EXTRACTED,
    )
    aid, _ = model.assertions.upsert_occurrence(cand)
    edge_id, is_new = model.evidence.upsert_edge(
        source_kind="assertion",
        source_id=aid,
        target_kind="issue",
        target_id=iid,
        relation_type=EvidenceRelationType.SUPPORTS,
        provenance=_prov(
            event_kind="edge_write",
            writer_name="EvidenceStore.upsert_edge",
            prompt_version="EL.EDGE_EXTRACT.v1",
        ),
    )
    assert is_new
    events = model.get_provenance("evidence_edge", edge_id)
    assert len(events) == 1
    assert events[0]["event_kind"] == "edge_write"


# ---------------------------------------------------------------------------
# P0.1 AC #4 — missing source span identity is explicit
# ---------------------------------------------------------------------------

def test_missing_source_span_is_explicitly_recorded(model):
    ctx = _prov(source_span_id=None, source_span_status="missing")
    eid = model.provenance.record(
        target_kind="assertion", target_id="a1", context=ctx,
    )
    rows = model.provenance.list_for_target("assertion", "a1")
    assert rows[0]["source_span_id"] is None
    assert rows[0]["source_span_status"] == "missing"


def test_provenance_default_span_status_is_unknown(model):
    # P0.1 AC #4: explicit 'missing' is the contract. 'unknown' is the
    # default for callers that haven't decided yet — the schema's CHECK
    # accepts it, but writers SHOULD set one of the three concrete
    # values when they know.
    ctx = ProvenanceContext(
        event_kind="e", writer_name="w",
    )
    eid = model.provenance.record(
        target_kind="assertion", target_id="a_default", context=ctx,
    )
    rows = model.provenance.list_for_target("assertion", "a_default")
    assert rows[0]["source_span_status"] == "unknown"


def test_provenance_rejects_invalid_span_status(model):
    """Schema CHECK constraint must reject unrecognized span_status values."""
    import sqlite3

    ctx = ProvenanceContext(
        event_kind="e", writer_name="w", source_span_status="bogus_value",
    )
    with pytest.raises(sqlite3.IntegrityError):
        model.provenance.record(
            target_kind="assertion", target_id="a1", context=ctx,
        )


# ---------------------------------------------------------------------------
# Query surface
# ---------------------------------------------------------------------------

def test_get_provenance_orders_by_created_at_desc(model):
    ctx1 = _prov(event_kind="first", llm_call_id="c1")
    ctx2 = _prov(event_kind="second", llm_call_id="c2")
    model.provenance.record(target_kind="assertion", target_id="a", context=ctx1)
    model.provenance.record(target_kind="assertion", target_id="a", context=ctx2)
    rows = model.provenance.list_for_target("assertion", "a")
    assert [r["event_kind"] for r in rows[:2]] == ["second", "first"]


def test_get_provenance_empty_for_unknown_target(model):
    assert model.get_provenance("assertion", "does_not_exist") == []


# ---------------------------------------------------------------------------
# End-to-end: production wire-up through MatterRuntimeAdapter.record_fact
# ---------------------------------------------------------------------------

def test_matter_runtime_record_fact_auto_attaches_provenance(model):
    """MatterRuntimeAdapter.record_fact must auto-build provenance when
    callers don't supply one. This exercises the production extraction
    path end-to-end: engine → runtime.record_fact → MatterModel.record_assertion
    → AssertionStore.upsert_occurrence → ProvenanceStore.record."""
    from irys.matter.runtime import MatterRuntimeAdapter

    run_id = model.start_run("provenance smoke")
    adapter = MatterRuntimeAdapter(model, run_id=run_id)
    aid = adapter.record_fact(
        proposition_text="Defendant owed payment",
        document_id="contracts/msa.pdf",
        span_id="span_42",
    )
    rows = model.get_provenance("assertion", aid)
    assert len(rows) >= 1
    row = rows[0]
    assert row["run_id"] == run_id
    assert row["extractor_version"] == adapter.DEFAULT_EXTRACTOR_VERSION
    assert row["source_document_ref"] == "contracts/msa.pdf"
    assert row["source_span_id"] == "span_42"
    assert row["source_span_status"] == "present"
    assert row["writer_name"] == "AssertionStore.upsert_occurrence"
    model.complete_run(run_id)


def test_record_fact_with_issue_link_writes_edge_provenance(model):
    """Adversarial audit #6 finding #4: link_assertion's implicit
    evidence_edge write had no provenance. The runtime adapter now
    builds an edge_write ProvenanceContext when the issue link is
    created via record_fact, so the resulting edge is attributable."""
    from irys.matter.runtime import MatterRuntimeAdapter

    run_id = model.start_run("edge provenance")
    adapter = MatterRuntimeAdapter(model, run_id=run_id)
    iid, _ = model.issues.upsert_issue("Test issue", IssueType.CLAIM)
    aid = adapter.record_fact(
        proposition_text="Defendant breached.",
        document_id="contracts/msa.pdf",
        span_id="span_1",
        issue_id=iid,
        issue_link_type="supports",
    )
    edge_row = model.db.execute(
        """SELECT id FROM evidence_edge
           WHERE matter_id=? AND source_kind='assertion' AND source_id=?
             AND target_kind='issue' AND target_id=?""",
        (model.matter_id, aid, iid),
    ).fetchone()
    assert edge_row is not None, "evidence_edge must exist for linked assertion"
    events = model.get_provenance("evidence_edge", edge_row["id"])
    assert len(events) == 1
    assert events[0]["event_kind"] == "edge_write"
    assert events[0]["writer_name"] == "EvidenceStore.upsert_edge"
    assert events[0]["run_id"] == run_id
    model.complete_run(run_id)


def test_active_llm_call_bridges_into_provenance(model):
    """Adversarial audit #6 finding #6: GeminiClient mints a call_id
    but nothing upstream read it, so every production provenance row
    had llm_call_id=NULL. The ACTIVE_LLM_CALL ContextVar now bridges
    the minted identity into MatterRuntimeAdapter._auto_provenance."""
    from irys.core.models import ACTIVE_LLM_CALL
    from irys.matter.runtime import MatterRuntimeAdapter

    run_id = model.start_run("active call bridge")
    adapter = MatterRuntimeAdapter(model, run_id=run_id)
    token = ACTIVE_LLM_CALL.set({
        "call_id": "call_bridge_test",
        "model_id": "gemini-2.5-flash-lite",
        "model_tier": "LITE",
        "prompt_hash": "h" * 64,
    })
    try:
        aid = adapter.record_fact(
            proposition_text="Bridged claim",
            document_id="doc.pdf",
            span_id="span_1",
        )
    finally:
        ACTIVE_LLM_CALL.reset(token)
    rows = model.get_provenance("assertion", aid)
    assert rows[0]["llm_call_id"] == "call_bridge_test"
    assert rows[0]["model_id"] == "gemini-2.5-flash-lite"
    assert rows[0]["model_tier"] == "LITE"
    assert rows[0]["prompt_hash"] == "h" * 64
    model.complete_run(run_id)


def test_record_fact_writes_provenance_for_occurrence(model):
    """Adversarial audit #6: the comment on AssertionStore.upsert_occurrence
    promised occurrence-level provenance but never wrote it. Regression
    test to make sure every new occurrence row gets a
    target_kind='assertion_occurrence' provenance event."""
    from irys.matter.runtime import MatterRuntimeAdapter

    run_id = model.start_run("provenance occurrence")
    adapter = MatterRuntimeAdapter(model, run_id=run_id)
    aid = adapter.record_fact(
        proposition_text="An occurrence-level claim",
        document_id="contracts/msa.pdf",
        span_id="span_occ_1",
    )
    # Assertion-level provenance still exists.
    assert len(model.get_provenance("assertion", aid)) >= 1
    # Occurrence-level provenance is now written too. We don't know the
    # occurrence id directly, so fetch it from assertion_occurrence and
    # verify a provenance row exists for it.
    occ_rows = model.db.execute(
        "SELECT id FROM assertion_occurrence WHERE assertion_id=?", (aid,),
    ).fetchall()
    assert len(occ_rows) == 1
    occ_id = occ_rows[0]["id"]
    occ_events = model.get_provenance("assertion_occurrence", occ_id)
    assert len(occ_events) == 1, (
        "occurrence row must have a target_kind='assertion_occurrence' "
        "provenance event — adversarial audit #6 regression"
    )
    assert occ_events[0]["run_id"] == run_id
    model.complete_run(run_id)


def test_matter_runtime_record_fact_records_missing_span_status(model):
    """AC #4 via the production path: when span_id is None, the
    provenance row records 'missing' explicitly."""
    from irys.matter.runtime import MatterRuntimeAdapter

    run_id = model.start_run("provenance span-missing")
    adapter = MatterRuntimeAdapter(model, run_id=run_id)
    aid = adapter.record_fact(
        proposition_text="Claim without a known span",
        document_id="contracts/msa.pdf",
        span_id=None,
    )
    rows = model.get_provenance("assertion", aid)
    assert rows[0]["source_span_id"] is None
    assert rows[0]["source_span_status"] == "missing"
    model.complete_run(run_id)
