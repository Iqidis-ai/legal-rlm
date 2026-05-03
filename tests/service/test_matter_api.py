"""Tests for matter model REST endpoints.

Uses FastAPI TestClient with an in-memory matter model injected into
the service's _active_matter_models registry — no real investigation run needed.
"""

import pytest
from fastapi.testclient import TestClient

from irys.service.api import app, _active_matter_models
from irys.matter import MatterModel, AssertionCandidate, SpeechAct, SourceRole
from irys.matter import ModelLayer, AssertionKind, BeliefState
from irys.matter.enums import OriginKind, IssueType, GapType, RevisionCause, AssertionLinkType
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


# ---------------------------------------------------------------------------
# MVP.2: assertions list carries verification_status
# ---------------------------------------------------------------------------

def test_get_matter_assertions_labels_candidate_vs_verified(client, register_model):
    """MVP.2 AC #5: the read model must surface verification status so clients
    can label candidate facts as candidate rather than established truth."""
    from irys.matter.enums import ReviewedByKind, VerificationTargetKind

    model = register_model
    candidate_aid = _add_assertion(model, text="Candidate proposition", doc="c.pdf")
    verified_aid = _add_assertion(model, text="Verified proposition", doc="v.pdf")
    model.verification.verify(
        VerificationTargetKind.ASSERTION, verified_aid,
        reviewed_by_kind=ReviewedByKind.ATTORNEY,
    )

    resp = client.get(f"/matter/{MATTER_ID}/assertions?limit=50")
    assert resp.status_code == 200
    body = resp.json()
    by_id = {a["id"]: a for a in body["assertions"]}

    # Every assertion row must carry verification_status.
    assert "verification_status" in by_id[candidate_aid]
    assert by_id[candidate_aid]["verification_status"] == "candidate"
    assert by_id[verified_aid]["verification_status"] == "verified"
    assert by_id[verified_aid]["reviewed_by_kind"] == "attorney"
    assert by_id[verified_aid]["reviewed_at"] is not None


# ---------------------------------------------------------------------------
# P0.3: GET /matter/{id}/review-queue  |  POST /verify  |  POST /verify/bulk-by-document
# ---------------------------------------------------------------------------

def test_review_queue_endpoint_returns_candidate_rows(client, register_model):
    """GET /review-queue returns the prioritized candidate list with
    target_kind, target_id, status='candidate', and target context
    (proposition_text for assertions)."""
    from irys.matter.runtime import MatterRuntimeAdapter

    model = register_model
    run_id = model.start_run("review-queue endpoint")
    adapter = MatterRuntimeAdapter(model, run_id)
    aid = adapter.record_fact("Candidate claim", "doc.pdf")

    resp = client.get(f"/matter/{MATTER_ID}/review-queue?limit=10")
    assert resp.status_code == 200
    body = resp.json()
    assert body["matter_id"] == MATTER_ID
    assert body["count"] >= 1
    assert any(r["target_id"] == aid for r in body["queue"])
    assert all(r["status"] == "candidate" for r in body["queue"])


def test_review_queue_endpoint_target_kind_filter(client, register_model):
    """?target_kind=quant_fact narrows the response to only quants."""
    from irys.matter.runtime import MatterRuntimeAdapter

    model = register_model
    run_id = model.start_run("review-queue filter")
    adapter = MatterRuntimeAdapter(model, run_id)
    adapter.record_fact("An assertion", "doc.pdf")
    adapter.record_quant(quant_kind="amount", raw_text="$7", amount_value=7.0)

    resp = client.get(
        f"/matter/{MATTER_ID}/review-queue?target_kind=quant_fact"
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["target_kind_filter"] == "quant_fact"
    assert all(r["target_kind"] == "quant_fact" for r in body["queue"])


def test_verify_endpoint_promotes_and_returns_id(client, register_model):
    from irys.matter.runtime import MatterRuntimeAdapter

    model = register_model
    run_id = model.start_run("verify endpoint")
    adapter = MatterRuntimeAdapter(model, run_id)
    aid = adapter.record_fact("Claim to verify", "doc.pdf")
    resp = client.post(
        f"/matter/{MATTER_ID}/verify",
        json={
            "target_kind": "assertion",
            "target_id": aid,
            "status": "verified",
            "reviewed_by_kind": "attorney",
            "reviewed_by_id": "atty1",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "verified"
    assert body["verification_id"]
    vs = model.verification.get("assertion", aid)
    assert vs["status"] == "verified"


def test_verify_endpoint_rejection_requires_reason(client, register_model):
    from irys.matter.runtime import MatterRuntimeAdapter

    model = register_model
    run_id = model.start_run("reject endpoint")
    adapter = MatterRuntimeAdapter(model, run_id)
    aid = adapter.record_fact("Claim to reject", "doc.pdf")
    # Missing rejection_reason → 400.
    resp = client.post(
        f"/matter/{MATTER_ID}/verify",
        json={
            "target_kind": "assertion",
            "target_id": aid,
            "status": "rejected",
            "reviewed_by_kind": "user",
            "reviewed_by_id": "u1",
        },
    )
    assert resp.status_code == 400
    assert "rejection_reason" in resp.text.lower()
    # With reason → 200.
    resp2 = client.post(
        f"/matter/{MATTER_ID}/verify",
        json={
            "target_kind": "assertion",
            "target_id": aid,
            "status": "rejected",
            "reviewed_by_kind": "user",
            "reviewed_by_id": "u1",
            "rejection_reason": "fabricated",
        },
    )
    assert resp2.status_code == 200


def test_verify_endpoint_blocks_automation_reviewer(client, register_model):
    """Human-only gate: reviewed_by_kind='system' cannot promote."""
    from irys.matter.runtime import MatterRuntimeAdapter

    model = register_model
    run_id = model.start_run("block automation")
    adapter = MatterRuntimeAdapter(model, run_id)
    aid = adapter.record_fact("Claim", "doc.pdf")
    resp = client.post(
        f"/matter/{MATTER_ID}/verify",
        json={
            "target_kind": "assertion",
            "target_id": aid,
            "status": "verified",
            "reviewed_by_kind": "system",
        },
    )
    assert resp.status_code == 400


def test_bulk_verify_by_document_endpoint(client, register_model):
    from irys.matter.runtime import MatterRuntimeAdapter

    model = register_model
    run_id = model.start_run("bulk endpoint")
    adapter = MatterRuntimeAdapter(model, run_id)
    adapter.record_fact("fact A", "docX.pdf")
    adapter.record_fact("fact B", "docX.pdf")
    adapter.record_fact("fact C", "docY.pdf")
    resp = client.post(
        f"/matter/{MATTER_ID}/verify/bulk-by-document",
        json={
            "document_ref": "docX.pdf",
            "reviewed_by_kind": "user",
            "reviewed_by_id": "u1",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["document_ref"] == "docX.pdf"
    assert body["verified_count"] == 2
    assert len(body["verification_ids"]) == 2


def test_bulk_verify_endpoint_404_for_unknown_matter(client):
    resp = client.post(
        "/matter/unknown_matter/verify/bulk-by-document",
        json={
            "document_ref": "any.pdf",
            "reviewed_by_kind": "user",
        },
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /matter/{matter_id}/proof-state — proof state summary
# ---------------------------------------------------------------------------

def test_get_proof_state_summary_empty(client, register_model):
    resp = client.get(f"/matter/{MATTER_ID}/proof-state")
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert "issues" in data
    assert data["matter_id"] == MATTER_ID


def test_get_proof_state_summary_with_issue(client, register_model):
    model = register_model
    issue_id, _ = model.issues.upsert_issue("Breach of contract", IssueType.CLAIM)
    aid = _add_assertion(model)
    model.issues.link_assertion(aid, issue_id, "supports")
    model.proof_state.compute_and_store(issue_id)
    resp = client.get(f"/matter/{MATTER_ID}/proof-state")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["issues"]) >= 1
    ps = data["issues"][0]
    assert ps["issue_id"] == issue_id
    assert "issue_title" in ps
    assert ps["issue_title"] == "Breach of contract"


def test_get_proof_state_compute(client, register_model):
    model = register_model
    issue_id, _ = model.issues.upsert_issue("Payment due", IssueType.CLAIM)
    resp = client.post(f"/matter/{MATTER_ID}/proof-state/compute")
    assert resp.status_code == 200


def test_get_issue_proof_state(client, register_model):
    model = register_model
    issue_id, _ = model.issues.upsert_issue("Damages claim", IssueType.CLAIM)
    model.proof_state.compute_and_store(issue_id)
    resp = client.get(f"/matter/{MATTER_ID}/issues/{issue_id}/proof-state")
    assert resp.status_code == 200


def test_proof_state_404_for_unknown_matter(client):
    resp = client.get("/matter/unknown/proof-state")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /matter/{matter_id}/authority-network — aggregate authority + issue links
# ---------------------------------------------------------------------------

def test_authority_network_empty(client, register_model):
    resp = client.get(f"/matter/{MATTER_ID}/authority-network")
    assert resp.status_code == 200
    data = resp.json()
    assert data["authorities"] == []
    assert data["issue_links"] == {}


def test_authority_network_with_linked_issue(client, register_model):
    model = register_model
    auth_id, _ = model.authority.upsert(
        citation="Twombly, 550 U.S. 544",
        authority_type="case",
        weight="binding",
    )
    issue_id, _ = model.issues.upsert_issue("Plausibility standard", IssueType.CLAIM)
    model.authority.link_to_issue(auth_id, issue_id, relevance="supporting")
    resp = client.get(f"/matter/{MATTER_ID}/authority-network")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["authorities"]) == 1
    assert data["authorities"][0]["citation"] == "Twombly, 550 U.S. 544"
    assert auth_id in data["issue_links"]
    links = data["issue_links"][auth_id]
    assert len(links) == 1
    assert links[0]["issue_id"] == issue_id
    assert links[0]["relevance"] == "supporting"


def test_authority_network_404_for_unknown_matter(client):
    resp = client.get("/matter/unknown/authority-network")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /matter/{matter_id}/documents/cards — document intelligence
# ---------------------------------------------------------------------------

def test_document_cards_empty(client, register_model):
    resp = client.get(f"/matter/{MATTER_ID}/documents/cards")
    assert resp.status_code == 200
    data = resp.json()
    assert data["cards"] == []
    assert data["total_inventory"] == 0
    assert data["ingested_count"] == 0


def test_document_cards_with_inventory(client, register_model):
    model = register_model
    inv_id, _ = model.inventory.upsert("contract.pdf", "a" * 64, size_bytes=1024)
    model.inventory.mark_ingested(inv_id)
    model.document_cards.upsert(
        doc_id=inv_id, title="contract.pdf", doc_type="contract",
    )
    resp = client.get(f"/matter/{MATTER_ID}/documents/cards")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["cards"]) == 1
    assert data["total_inventory"] >= 1
    assert data["ingested_count"] >= 1
    assert data["cards"][0]["title"] == "contract.pdf"


def test_document_cards_404_for_unknown_matter(client):
    resp = client.get("/matter/unknown/documents/cards")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /matter/{matter_id}/timeline — timeline events
# ---------------------------------------------------------------------------

def test_timeline_empty(client, register_model):
    resp = client.get(f"/matter/{MATTER_ID}/timeline")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


def test_timeline_404_for_unknown_matter(client):
    resp = client.get("/matter/unknown/timeline")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /matter/{matter_id}/evidence-matrix — issue × document matrix
# ---------------------------------------------------------------------------

def test_evidence_matrix_empty(client, register_model):
    resp = client.get(f"/matter/{MATTER_ID}/evidence-matrix")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)


def test_evidence_matrix_404_for_unknown_matter(client):
    resp = client.get("/matter/unknown/evidence-matrix")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /matter/{matter_id}/communication-map — actor interaction graph
# ---------------------------------------------------------------------------

def test_communication_map_empty(client, register_model):
    resp = client.get(f"/matter/{MATTER_ID}/communication-map")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)


def test_communication_map_404_for_unknown_matter(client):
    resp = client.get("/matter/unknown/communication-map")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /matter/{matter_id}/damages-waterfall — quantitative breakdown
# ---------------------------------------------------------------------------

def test_damages_waterfall_empty(client, register_model):
    resp = client.get(f"/matter/{MATTER_ID}/damages-waterfall")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


def test_damages_waterfall_404_for_unknown_matter(client):
    resp = client.get("/matter/unknown/damages-waterfall")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /matter/{matter_id}/steering-surface — actionable steering actions
# ---------------------------------------------------------------------------

def test_steering_surface_empty(client, register_model):
    resp = client.get(f"/matter/{MATTER_ID}/steering-surface")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


def test_steering_surface_404_for_unknown_matter(client):
    resp = client.get("/matter/unknown/steering-surface")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /matter/{matter_id}/overview — overview with domain composition
# ---------------------------------------------------------------------------

def test_overview_includes_domain_composition(client, register_model):
    resp = client.get(f"/matter/{MATTER_ID}/overview")
    assert resp.status_code == 200
    data = resp.json()
    assert "domain_composition" in data
    dc = data["domain_composition"]
    assert "primary_domain_profile_id" in dc


# ---------------------------------------------------------------------------
# GET /matter/{matter_id}/belief-revisions — belief revision audit trail
# ---------------------------------------------------------------------------

def test_belief_revisions_empty(client, register_model):
    resp = client.get(f"/matter/{MATTER_ID}/belief-revisions")
    assert resp.status_code == 200
    assert resp.json() == []


def test_belief_revisions_with_data(client, register_model):
    model = register_model
    aid, _ = model.assertions.upsert_occurrence(AssertionCandidate(
        proposition_text="Delivery was on time.",
        model_layer=ModelLayer.RECORD,
        assertion_kind=AssertionKind.FACTUAL,
        document_id="doc.pdf",
        speech_act=SpeechAct.EXTRACTED,
        source_role=SourceRole.UNKNOWN,
    ))
    model.assertions.set_belief_state(aid, BeliefState.OPERATIVE, 0.9)
    model.belief.force_state(
        assertion_id=aid,
        new_state=BeliefState.DISPUTED,
        new_confidence=0.3,
        cause=RevisionCause.CONFLICT_DETECTION,
    )
    resp = client.get(f"/matter/{MATTER_ID}/belief-revisions")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) >= 1
    assert "proposition_text" in rows[0]
    assert rows[0]["new_belief_state"] == "disputed"


def test_belief_revisions_404_for_unknown_matter(client):
    resp = client.get("/matter/unknown/belief-revisions")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /matter/{matter_id}/contradictions — contradiction analysis
# ---------------------------------------------------------------------------

def test_contradictions_empty(client, register_model):
    resp = client.get(f"/matter/{MATTER_ID}/contradictions")
    assert resp.status_code == 200
    assert resp.json() == []


def test_contradictions_with_data(client, register_model):
    model = register_model
    a1, _ = model.assertions.upsert_occurrence(AssertionCandidate(
        proposition_text="Payment was received on March 1.",
        model_layer=ModelLayer.RECORD,
        assertion_kind=AssertionKind.FACTUAL,
        document_id="doc1.pdf",
        speech_act=SpeechAct.EXTRACTED,
        source_role=SourceRole.UNKNOWN,
    ))
    a2, _ = model.assertions.upsert_occurrence(AssertionCandidate(
        proposition_text="Payment was never received.",
        model_layer=ModelLayer.RECORD,
        assertion_kind=AssertionKind.FACTUAL,
        document_id="doc2.pdf",
        speech_act=SpeechAct.EXTRACTED,
        source_role=SourceRole.UNKNOWN,
    ))
    model.assertions.set_belief_state(a1, BeliefState.OPERATIVE, 0.9)
    model.assertions.set_belief_state(a2, BeliefState.OPERATIVE, 0.8)
    model.assertions.link(a2, a1, AssertionLinkType.CONTRADICTS)
    resp = client.get(f"/matter/{MATTER_ID}/contradictions")
    assert resp.status_code == 200
    conflicts = resp.json()
    assert len(conflicts) >= 1
    c = conflicts[0]
    assert "attacker_prop" in c
    assert "attacked_prop" in c
    assert c["link_type"] == "contradicts"


def test_contradictions_404_for_unknown_matter(client):
    resp = client.get("/matter/unknown/contradictions")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /matter/{matter_id}/document-versions — version chain families
# ---------------------------------------------------------------------------

def test_document_versions_empty(client, register_model):
    resp = client.get(f"/matter/{MATTER_ID}/document-versions")
    assert resp.status_code == 200
    assert resp.json() == []


def test_document_versions_with_data(client, register_model):
    model = register_model
    d1_id, _ = model.inventory.upsert("contract_v1.pdf", sha256="aaa", size_bytes=100)
    d2_id, _ = model.inventory.upsert("contract_v2.pdf", sha256="bbb", size_bytes=200)
    model.inventory.link_documents(d2_id, d1_id, "version_of", confidence=0.9)
    model.inventory.set_family_membership([d1_id, d2_id], family_id=d1_id)
    resp = client.get(f"/matter/{MATTER_ID}/document-versions")
    assert resp.status_code == 200
    families = resp.json()
    assert len(families) == 1
    fam = families[0]
    assert fam["family_id"] == d1_id
    assert len(fam["members"]) == 2
    operative = [m for m in fam["members"] if m["is_operative"]]
    assert len(operative) == 1
    assert operative[0]["relative_path"] == "contract_v2.pdf"


def test_document_versions_via_refresh_families(client, register_model):
    """Integration: refresh_document_families detects chains and the endpoint returns them."""
    model = register_model
    model.inventory.upsert("agreement_v1.pdf", sha256="aaa", size_bytes=100)
    model.inventory.upsert("agreement_v2.pdf", sha256="bbb", size_bytes=200)
    model.refresh_document_families()
    resp = client.get(f"/matter/{MATTER_ID}/document-versions")
    assert resp.status_code == 200
    families = resp.json()
    assert len(families) >= 1
    fam = families[0]
    paths = {m["relative_path"] for m in fam["members"]}
    assert "agreement_v1.pdf" in paths
    assert "agreement_v2.pdf" in paths
    operative = [m for m in fam["members"] if m["is_operative"]]
    assert len(operative) == 1
    assert operative[0]["relative_path"] == "agreement_v2.pdf"


def test_document_versions_404_for_unknown_matter(client):
    resp = client.get("/matter/unknown/document-versions")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /matter/{matter_id}/mine-contradictions — on-demand mining
# ---------------------------------------------------------------------------

def test_mine_contradictions_empty(client, register_model):
    resp = client.post(f"/matter/{MATTER_ID}/mine-contradictions")
    assert resp.status_code == 200
    assert resp.json() == []


def test_mine_contradictions_404_for_unknown_matter(client):
    resp = client.post("/matter/unknown/mine-contradictions")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /matter/{matter_id}/provenance/{target_kind}/{target_id}
# ---------------------------------------------------------------------------

def test_provenance_empty(client, register_model):
    resp = client.get(f"/matter/{MATTER_ID}/provenance/assertion/nonexistent")
    assert resp.status_code == 200
    assert resp.json() == []


def test_provenance_404_for_unknown_matter(client):
    resp = client.get("/matter/unknown/provenance/assertion/test-id")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /matter/{matter_id}/assertion/{assertion_id}/health
# ---------------------------------------------------------------------------

def test_assertion_health_returns_data(client, register_model):
    model = register_model
    aid = _add_assertion(model)
    resp = client.get(f"/matter/{MATTER_ID}/assertion/{aid}/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["assertion_id"] == aid
    assert "oscillating" in data
    assert "support_count" in data
    assert "attack_count" in data
    assert isinstance(data["provenance"], list)


def test_assertion_health_not_found(client, register_model):
    resp = client.get(f"/matter/{MATTER_ID}/assertion/nonexistent/health")
    assert resp.status_code == 404


def test_assertion_health_404_for_unknown_matter(client):
    resp = client.get("/matter/unknown/assertion/a1/health")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /matter/{matter_id}/quant-thresholds
# ---------------------------------------------------------------------------

def test_quant_thresholds_empty(client, register_model):
    resp = client.get(f"/matter/{MATTER_ID}/quant-thresholds")
    assert resp.status_code == 200
    assert resp.json() == []


def test_quant_thresholds_404_for_unknown_matter(client):
    resp = client.get("/matter/unknown/quant-thresholds")
    assert resp.status_code == 404


def test_system_health_returns_data(client, register_model):
    resp = client.get(f"/matter/{MATTER_ID}/system-health")
    assert resp.status_code == 200
    data = resp.json()
    assert "assertion_count" in data
    assert "disputed_count" in data
    assert "disputed_fraction" in data
    assert "revision_count" in data
    assert "open_gap_count" in data
    assert "contradiction_count" in data
    assert "version_chain_count" in data
    assert "oscillating_count" in data
    assert data["health_score"] in ("good", "attention_needed")


def test_system_health_404_for_unknown_matter(client):
    resp = client.get("/matter/unknown/system-health")
    assert resp.status_code == 404


def test_so_scorecard_returns_data(client, register_model):
    resp = client.get(f"/matter/{MATTER_ID}/so-scorecard")
    assert resp.status_code == 200
    data = resp.json()
    assert "targets" in data
    assert "targets_met" in data
    assert "counts" in data


def test_so_scorecard_404_for_unknown_matter(client):
    resp = client.get("/matter/unknown/so-scorecard")
    assert resp.status_code == 404
