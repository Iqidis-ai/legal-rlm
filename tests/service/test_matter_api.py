"""Tests for matter model REST endpoints.

Uses FastAPI TestClient with an in-memory matter model injected into
the service's _active_matter_models registry — no real investigation run needed.
"""

import pytest
from fastapi.testclient import TestClient

from irys.service.api import app, _active_matter_models
from irys.matter import MatterModel, AssertionCandidate, SpeechAct, SourceRole
from irys.matter import ModelLayer, AssertionKind, BeliefState
from irys.matter.enums import OriginKind, IssueType, GapType
from irys.matter.reasoning import ReasoningLedgerStore


MATTER_ID = "test_matter_id_abc123"


@pytest.fixture(autouse=True)
def register_model():
    """Create an in-memory matter model and register it in the service registry."""
    model = MatterModel.open_in_memory()
    # Override matter_id to the predictable test value
    # (open_in_memory uses a random hex ID, so we patch the registry key instead)
    _active_matter_models[MATTER_ID] = model
    yield model
    del _active_matter_models[MATTER_ID]


@pytest.fixture
def client():
    return TestClient(app)


def _add_assertion(model, text="Payment was due January 15.", doc="contract.pdf"):
    c = AssertionCandidate(
        proposition_text=text,
        model_layer=ModelLayer.RECORD,
        assertion_kind=AssertionKind.FACTUAL,
        document_id=doc,
        speech_act=SpeechAct.EXTRACTED,
        source_role=SourceRole.UNKNOWN,
        origin_kind=OriginKind.EXTRACTED,
    )
    aid, _ = model.assertions.upsert_occurrence(c)
    return aid


# ---------------------------------------------------------------------------
# GET /matter/{matter_id} — stats
# ---------------------------------------------------------------------------

def test_get_matter_stats_200(client, register_model):
    model = register_model
    _add_assertion(model)
    resp = client.get(f"/matter/{MATTER_ID}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["matter_id"] == model.matter_id
    assert data["assertion_count"] >= 1


def test_get_matter_stats_404(client):
    resp = client.get("/matter/nonexistent_matter_id")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /matter/{matter_id}/runs
# ---------------------------------------------------------------------------

def test_get_matter_runs_empty(client, register_model):
    resp = client.get(f"/matter/{MATTER_ID}/runs")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_get_matter_runs_after_run_starts(client, register_model):
    model = register_model
    run_id = model.start_run("What are the key obligations?")
    model.complete_run(run_id)

    resp = client.get(f"/matter/{MATTER_ID}/runs?limit=5")
    assert resp.status_code == 200
    runs = resp.json()
    assert len(runs) == 1
    assert runs[0]["id"] == run_id
    assert runs[0]["status"] == "completed"


# ---------------------------------------------------------------------------
# GET /matter/{matter_id}/runs/{run_id}/events
# ---------------------------------------------------------------------------

def test_get_run_events(client, register_model):
    model = register_model
    run_id = model.start_run("Event test")
    model.complete_run(run_id)

    resp = client.get(f"/matter/{MATTER_ID}/runs/{run_id}/events")
    assert resp.status_code == 200
    events = resp.json()
    assert len(events) >= 1
    assert events[0]["event_type"] == "run_started"


# ---------------------------------------------------------------------------
# POST /matter/{matter_id}/stop
# ---------------------------------------------------------------------------

def test_stop_returns_409_when_no_running_run(client, register_model):
    resp = client.post(f"/matter/{MATTER_ID}/stop", json={})
    assert resp.status_code == 409


def test_stop_sets_stop_requested(client, register_model):
    model = register_model
    run_id = model.start_run("Stoppable investigation")

    resp = client.post(f"/matter/{MATTER_ID}/stop", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "stop_requested"
    assert data["run_id"] == run_id
    assert model.ledger.is_stop_requested(run_id)


# ---------------------------------------------------------------------------
# POST /matter/{matter_id}/runs/{run_id}/redirect
# ---------------------------------------------------------------------------

def test_redirect_404_for_unknown_run(client, register_model):
    resp = client.post(
        f"/matter/{MATTER_ID}/runs/nonexistent_run/redirect",
        json={"issue_id": "anything"},
    )
    assert resp.status_code == 404


def test_redirect_409_for_completed_run(client, register_model):
    model = register_model
    issue_id, _ = model.issues.upsert_issue("Payment obligation", IssueType.CLAIM)
    run_id = model.start_run("Completed run")
    model.complete_run(run_id)

    resp = client.post(
        f"/matter/{MATTER_ID}/runs/{run_id}/redirect",
        json={"issue_id": issue_id},
    )
    assert resp.status_code == 409


def test_redirect_404_for_unknown_issue(client, register_model):
    model = register_model
    run_id = model.start_run("Running for redirect test")

    resp = client.post(
        f"/matter/{MATTER_ID}/runs/{run_id}/redirect",
        json={"issue_id": "nonexistent_issue"},
    )
    assert resp.status_code == 404


def test_redirect_sets_flag(client, register_model):
    model = register_model
    issue_id, _ = model.issues.upsert_issue("Breach of contract", IssueType.CLAIM)
    run_id = model.start_run("Redirectable run")

    resp = client.post(
        f"/matter/{MATTER_ID}/runs/{run_id}/redirect",
        json={"issue_id": issue_id},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "redirect_requested"
    assert model.ledger.is_redirect_requested(run_id)


# ---------------------------------------------------------------------------
# GET /matter/{matter_id}/clarifications
# ---------------------------------------------------------------------------

def test_get_clarifications_empty(client, register_model):
    resp = client.get(f"/matter/{MATTER_ID}/clarifications")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_clarifications_returns_pending(client, register_model):
    model = register_model
    model.clarifications.add_question(
        question_text="Is the signed amendment available?",
        why_it_matters="Referenced in main contract",
        expected_impact="Could change damages calculation",
    )

    resp = client.get(f"/matter/{MATTER_ID}/clarifications")
    assert resp.status_code == 200
    questions = resp.json()
    assert len(questions) == 1
    assert "amendment" in questions[0]["question_text"]


def test_answer_clarification(client, register_model):
    model = register_model
    q_id = model.clarifications.add_question(
        question_text="Is the signed amendment available?",
    )

    resp = client.post(
        f"/matter/{MATTER_ID}/clarifications/{q_id}/answer",
        json={"answer_text": "Yes, it is attached as Exhibit B."},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "answered"

    # Should no longer appear in pending
    pending = model.clarifications.get_pending()
    assert not any(q["id"] == q_id for q in pending)


# ---------------------------------------------------------------------------
# GET /matter/{matter_id}/issues
# ---------------------------------------------------------------------------

def test_get_issues_with_counts(client, register_model):
    model = register_model
    issue_id, _ = model.issues.upsert_issue("Payment obligation", IssueType.CLAIM)
    a_id = _add_assertion(model, "Payment was not made.")
    model.issues.link_assertion(a_id, issue_id, "supports")

    resp = client.get(f"/matter/{MATTER_ID}/issues")
    assert resp.status_code == 200
    issues = resp.json()
    assert len(issues) == 1
    # API now returns get_issue_coverage_report() fields (SO-4)
    assert issues[0]["supporting_count"] == 1
    assert "coverage_fraction" in issues[0]
    assert "has_proof_gap" in issues[0]


# ---------------------------------------------------------------------------
# GET /matter/{matter_id}/gaps
# ---------------------------------------------------------------------------

def test_get_gaps_empty(client, register_model):
    resp = client.get(f"/matter/{MATTER_ID}/gaps")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_gaps_returns_recorded_gaps(client, register_model):
    model = register_model
    model.record_gap(
        description="Signed amendment not found",
        gap_type=GapType.MISSING_DOCUMENT,
        materiality=0.8,
    )

    resp = client.get(f"/matter/{MATTER_ID}/gaps")
    assert resp.status_code == 200
    gaps = resp.json()
    assert len(gaps) == 1
    assert "amendment" in gaps[0]["description"]


def test_get_gaps_materiality_filter(client, register_model):
    model = register_model
    model.record_gap(description="Low materiality gap", gap_type=GapType.MISSING_DOCUMENT, materiality=0.2)
    model.record_gap(description="High materiality gap", gap_type=GapType.MISSING_DOCUMENT, materiality=0.9)

    resp = client.get(f"/matter/{MATTER_ID}/gaps?min_materiality=0.5")
    assert resp.status_code == 200
    gaps = resp.json()
    assert len(gaps) == 1
    assert "High" in gaps[0]["description"]


# ---------------------------------------------------------------------------
# GET /matter/{matter_id}/reconciliation
# ---------------------------------------------------------------------------

def test_reconciliation_returns_structure_when_no_quant(client, register_model):
    resp = client.get(f"/matter/{MATTER_ID}/reconciliation")
    assert resp.status_code == 200
    data = resp.json()
    assert data["currency"] == "USD"
    assert data["invoiced"] == 0.0
    assert data["paid"] == 0.0


def test_reconciliation_shows_invoiced_and_paid(client, register_model):
    model = register_model
    model.quant.record(quant_kind="amount", raw_text="inv1", amount_value=50_000.0,
                       currency="USD", subject_type="invoice")
    model.quant.record(quant_kind="amount", raw_text="pmt1", amount_value=40_000.0,
                       currency="USD", subject_type="payment")

    resp = client.get(f"/matter/{MATTER_ID}/reconciliation")
    assert resp.status_code == 200
    data = resp.json()
    assert data["invoiced"] == 50_000.0
    assert data["paid"] == 40_000.0


def test_reconciliation_conflicts_returns_raw_texts(client, register_model):
    model = register_model
    model.quant.record(
        quant_kind="amount",
        raw_text="Invoice 1042 says $50,000",
        amount_value=50_000.0,
        currency="USD",
        subject_type="invoice",
        subject_id="inv_1042",
    )
    model.quant.record(
        quant_kind="amount",
        raw_text="Revised invoice 1042 says $65,000",
        amount_value=65_000.0,
        currency="USD",
        subject_type="invoice",
        subject_id="inv_1042",
    )

    resp = client.get(f"/matter/{MATTER_ID}/reconciliation/conflicts")
    assert resp.status_code == 200
    conflicts = resp.json()
    assert len(conflicts) == 1
    assert conflicts[0]["subject_type"] == "invoice"
    assert conflicts[0]["subject_id"] == "inv_1042"
    assert conflicts[0]["values"] == [50000.0, 65000.0]
    assert conflicts[0]["raw_texts"] == [
        "Invoice 1042 says $50,000",
        "Revised invoice 1042 says $65,000",
    ]


# ---------------------------------------------------------------------------
# POST /matter/{matter_id}/assertions/{assertion_id}/correct
# ---------------------------------------------------------------------------

def test_correct_assertion_200(client, register_model):
    model = register_model
    a_id = _add_assertion(model, "Payment received on time.")
    model.assertions.set_belief_state(a_id, BeliefState.ALLEGED, 0.5)

    resp = client.post(
        f"/matter/{MATTER_ID}/assertions/{a_id}/correct",
        json={
            "new_belief_state": "disputed",
            "confidence": 0.9,
            "note": "Contradicted by bank records",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["new_belief_state"] == "disputed"
    assert data["old_belief_state"] == "alleged"
    assert data["assertion_id"] == a_id


def test_correct_assertion_400_invalid_state(client, register_model):
    model = register_model
    a_id = _add_assertion(model)

    resp = client.post(
        f"/matter/{MATTER_ID}/assertions/{a_id}/correct",
        json={"new_belief_state": "totally_invalid", "confidence": 0.5, "note": "test"},
    )
    assert resp.status_code == 400


def test_correct_assertion_404_unknown_assertion(client, register_model):
    resp = client.post(
        f"/matter/{MATTER_ID}/assertions/nonexistent_id/correct",
        json={"new_belief_state": "disputed", "confidence": 0.5, "note": "test"},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST/GET /matter/{matter_id}/trust-overrides
# ---------------------------------------------------------------------------

def test_set_trust_override_persists(client, register_model):
    resp = client.post(
        f"/matter/{MATTER_ID}/trust-overrides",
        json={"document_pattern": "complaint.pdf", "trust_level": "low", "note": "advocacy doc"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "set"
    assert "override_id" in data


def test_set_trust_override_invalid_level(client, register_model):
    resp = client.post(
        f"/matter/{MATTER_ID}/trust-overrides",
        json={"document_pattern": "doc.pdf", "trust_level": "INVALID"},
    )
    assert resp.status_code == 400


def test_list_trust_overrides(client, register_model):
    client.post(
        f"/matter/{MATTER_ID}/trust-overrides",
        json={"document_pattern": "contract.pdf", "trust_level": "high"},
    )
    resp = client.get(f"/matter/{MATTER_ID}/trust-overrides")
    assert resp.status_code == 200
    overrides = resp.json()["overrides"]
    assert any(o["document_pattern"] == "contract.pdf" for o in overrides)


def test_trust_override_404_unknown_matter(client):
    resp = client.post(
        "/matter/no-such-matter/trust-overrides",
        json={"document_pattern": "doc.pdf", "trust_level": "low"},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST/GET /matter/{matter_id}/annotations
# ---------------------------------------------------------------------------

def test_add_annotation_persists(client, register_model):
    resp = client.post(
        f"/matter/{MATTER_ID}/annotations",
        json={
            "document_pattern": "deposition.pdf",
            "annotation_text": "Witness contradicts prior statement on page 12.",
            "annotation_type": "strategic_note",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "added"
    assert "annotation_id" in data


def test_list_annotations_all(client, register_model):
    client.post(
        f"/matter/{MATTER_ID}/annotations",
        json={"document_pattern": "motion.pdf", "annotation_text": "Key exhibit.", "annotation_type": "note"},
    )
    resp = client.get(f"/matter/{MATTER_ID}/annotations")
    assert resp.status_code == 200
    annotations = resp.json()["annotations"]
    assert len(annotations) >= 1


def test_list_annotations_filtered_by_document(client, register_model):
    client.post(
        f"/matter/{MATTER_ID}/annotations",
        json={"document_pattern": "specific_doc.pdf", "annotation_text": "Important.", "annotation_type": "note"},
    )
    resp = client.get(f"/matter/{MATTER_ID}/annotations?document=specific_doc.pdf")
    assert resp.status_code == 200
    annotations = resp.json()["annotations"]
    assert all("specific_doc" in a["document_pattern"] for a in annotations)


def test_annotation_404_unknown_matter(client):
    resp = client.post(
        "/matter/no-such-matter/annotations",
        json={"document_pattern": "doc.pdf", "annotation_text": "note", "annotation_type": "note"},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# _compute_corpus_key() stability
# ---------------------------------------------------------------------------

def test_compute_corpus_key_stable():
    """Same input always produces the same 16-char hex key."""
    from irys.service.api import _compute_corpus_key
    k1 = _compute_corpus_key("s3://my-bucket/matters/acme-v-techco")
    k2 = _compute_corpus_key("s3://my-bucket/matters/acme-v-techco")
    assert k1 == k2
    assert len(k1) == 16
    assert all(c in "0123456789abcdef" for c in k1)


def test_compute_corpus_key_different_inputs():
    """Different descriptors produce different keys."""
    from irys.service.api import _compute_corpus_key
    k1 = _compute_corpus_key("s3://bucket/prefix-a")
    k2 = _compute_corpus_key("s3://bucket/prefix-b")
    assert k1 != k2


def test_compute_corpus_key_url_order_independent():
    """URL corpus_key is stable regardless of URL list order."""
    from irys.service.api import _compute_corpus_key
    urls_a = ["https://s3.us-east-1.amazonaws.com/b/doc1.pdf",
              "https://s3.us-east-1.amazonaws.com/b/doc2.pdf"]
    urls_b = list(reversed(urls_a))
    k1 = _compute_corpus_key(",".join(sorted(urls_a)))
    k2 = _compute_corpus_key(",".join(sorted(urls_b)))
    assert k1 == k2


def test_url_to_str_normalizes_plain_string():
    """_url_to_str must return a plain string unchanged."""
    from irys.service.api import _url_to_str
    assert _url_to_str("https://example.com/contract.pdf") == "https://example.com/contract.pdf"


def test_url_to_str_normalizes_url_with_metadata():
    """_url_to_str must extract .url from UrlWithMetadata objects."""
    from irys.service.api import _url_to_str
    from irys.service.models import UrlWithMetadata
    u = UrlWithMetadata(url="https://example.com/contract.pdf", name="contract.pdf")
    assert _url_to_str(u) == "https://example.com/contract.pdf"


def test_corpus_key_stable_with_url_with_metadata():
    """Corpus key must be stable when URL list contains UrlWithMetadata objects."""
    from irys.service.api import _compute_corpus_key, _url_to_str
    from irys.service.models import UrlWithMetadata
    urls = [
        UrlWithMetadata(url="https://s3.example.com/doc1.pdf", name="doc1.pdf"),
        "https://s3.example.com/doc2.pdf",
    ]
    # Should not raise TypeError (UrlWithMetadata has no __lt__)
    key = _compute_corpus_key(",".join(sorted(_url_to_str(u) for u in urls)))
    assert len(key) == 16
    assert all(c in "0123456789abcdef" for c in key)


def test_sync_investigate_response_includes_open_gaps_field():
    """SyncInvestigateResponse must include open_gaps field defaulting to []."""
    from irys.service.models import SyncInvestigateResponse
    resp = SyncInvestigateResponse(
        query="test",
        analysis="analysis",
        documents_processed=1,
        duration_seconds=1.0,
    )
    assert hasattr(resp, "open_gaps")
    assert resp.open_gaps == []


def test_sync_investigate_response_open_gaps_roundtrips():
    """open_gaps must survive JSON round-trip."""
    from irys.service.models import SyncInvestigateResponse
    gap = {"description": "Missing signed amendment", "gap_type": "missing_document", "materiality_score": 0.8}
    resp = SyncInvestigateResponse(
        query="test",
        analysis="analysis",
        documents_processed=1,
        duration_seconds=1.0,
        open_gaps=[gap],
    )
    d = resp.model_dump()
    assert d["open_gaps"] == [gap]


# ---------------------------------------------------------------------------
# GET /matter/{matter_id}/assertions/{assertion_id}/history
# ---------------------------------------------------------------------------

def test_assertion_history_empty_when_no_revisions(client, register_model):
    model = register_model
    a_id = _add_assertion(model)

    resp = client.get(f"/matter/{MATTER_ID}/assertions/{a_id}/history")
    assert resp.status_code == 200
    data = resp.json()
    assert data["assertion_id"] == a_id
    assert data["count"] == 0
    assert data["truncated"] is False
    assert data["next_offset"] is None
    assert data["history"] == []
    assert "history_note" in data


def test_assertion_history_returns_revision_after_correction(client, register_model):
    model = register_model
    a_id = _add_assertion(model, "Payment was late.")
    model.assertions.set_belief_state(a_id, BeliefState.ALLEGED, 0.5)

    # Perform a correction to generate revision rows
    client.post(
        f"/matter/{MATTER_ID}/assertions/{a_id}/correct",
        json={"new_belief_state": "disputed", "confidence": 0.8, "note": "Corrected"},
    )

    resp = client.get(f"/matter/{MATTER_ID}/assertions/{a_id}/history")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] >= 1
    row = data["history"][0]
    assert row["changed_field"] in ("belief_state", "confidence")
    assert row["actor_kind"] is not None
    assert "created_at" in row
    assert "batch_id" in row


def test_assertion_history_404_unknown_assertion(client, register_model):
    resp = client.get(f"/matter/{MATTER_ID}/assertions/nonexistent_assertion_id/history")
    assert resp.status_code == 404


def test_assertion_history_404_wrong_matter(client, register_model):
    model = register_model
    a_id = _add_assertion(model)

    # Request under a nonexistent matter — should 404 at the matter level
    resp = client.get(f"/matter/wrong_matter_id/assertions/{a_id}/history")
    assert resp.status_code == 404


def test_assertion_history_limit_and_offset(client, register_model):
    model = register_model
    a_id = _add_assertion(model, "Amount in dispute.")
    model.assertions.set_belief_state(a_id, BeliefState.ALLEGED, 0.5)

    # Generate multiple revisions via multiple corrections
    for i in range(3):
        client.post(
            f"/matter/{MATTER_ID}/assertions/{a_id}/correct",
            json={"new_belief_state": "disputed", "confidence": 0.6 + i * 0.1, "note": f"Correction {i}"},
        )

    # First page
    resp = client.get(f"/matter/{MATTER_ID}/assertions/{a_id}/history?limit=2&offset=0")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    assert data["truncated"] is True
    assert data["next_offset"] == 2

    # Second page
    resp2 = client.get(f"/matter/{MATTER_ID}/assertions/{a_id}/history?limit=2&offset=2")
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["count"] >= 1  # at least one more row


# ---------------------------------------------------------------------------
# Matter isolation: run-scoped endpoints must reject foreign matter's run_id
# ---------------------------------------------------------------------------

def test_get_run_events_404_cross_matter(client, register_model):
    model = register_model
    run_id = model.start_run("isolation test")
    model.complete_run(run_id)

    # Register a second matter but try to read run from the first matter via the second
    model2 = MatterModel.open_in_memory()
    _active_matter_models["other_matter"] = model2
    try:
        resp = client.get(f"/matter/other_matter/runs/{run_id}/events")
        assert resp.status_code == 404
    finally:
        del _active_matter_models["other_matter"]


def test_stop_run_404_cross_matter(client, register_model):
    model = register_model
    run_id = model.start_run("isolation stop test")

    model2 = MatterModel.open_in_memory()
    _active_matter_models["other_matter_stop"] = model2
    try:
        resp = client.post(f"/matter/other_matter_stop/runs/{run_id}/stop")
        assert resp.status_code == 404
    finally:
        del _active_matter_models["other_matter_stop"]
        # clean up the running run
        try:
            model.fail_run(run_id)
        except Exception:
            pass
