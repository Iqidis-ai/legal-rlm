"""Tests for get_communication_map() — visual work product Priority 2.

Verifies:
1. Empty model returns empty map structure
2. Actor with occurrence in a document creates actor_document_edge
3. Occurrence count matches assertion_occurrence rows
4. Two actors sharing a document creates actor_actor_edge
5. Actors sharing no document produce no actor_actor_edge
6. Multiple documents per actor appear in edges
7. Actors with no occurrences are excluded
8. actor_actor_edges include shared_documents count and document list
9. API endpoint returns correct communication map
"""

import pytest
from irys.matter import MatterModel, SpeechAct, SourceRole, AssertionKind
from irys.matter.models import AssertionCandidate


@pytest.fixture
def model():
    return MatterModel.open_in_memory()


def _actor(model, name, actor_type="person"):
    aid, _ = model.actors.upsert_actor(canonical_name=name, actor_type=actor_type)
    return aid


def _assert_by(model, actor_id, doc_id):
    """Create an assertion occurrence by actor_id in doc_id."""
    cand = AssertionCandidate(
        proposition_text=f"Statement {_assert_by._ctr}",
        speech_act=SpeechAct.ALLEGED,
        source_role=SourceRole.ADVOCACY,
        assertion_kind=AssertionKind.FACTUAL,
        document_id=doc_id,
        speaker_actor_id=actor_id,
    )
    _assert_by._ctr += 1
    aid, _ = model.assertions.upsert_occurrence(cand)
    return aid


_assert_by._ctr = 0


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------

def test_empty_model_returns_empty_map(model):
    m = model.get_communication_map()
    assert m["actors"] == []
    assert m["documents"] == []
    assert m["actor_document_edges"] == []
    assert m["actor_actor_edges"] == []


def test_actor_document_edge_created(model):
    aid = _actor(model, "Alice")
    _assert_by(model, aid, "complaint.pdf")

    m = model.get_communication_map()
    assert len(m["actors"]) == 1
    assert "complaint.pdf" in m["documents"]
    assert len(m["actor_document_edges"]) == 1
    edge = m["actor_document_edges"][0]
    assert edge["actor_id"] == aid
    assert edge["document_id"] == "complaint.pdf"
    assert edge["occurrence_count"] == 1


def test_occurrence_count_reflects_multiple_assertions(model):
    aid = _actor(model, "Bob")
    _assert_by(model, aid, "contract.pdf")
    _assert_by(model, aid, "contract.pdf")
    _assert_by(model, aid, "contract.pdf")

    m = model.get_communication_map()
    edge = m["actor_document_edges"][0]
    assert edge["occurrence_count"] == 3


def test_two_actors_sharing_document_creates_actor_actor_edge(model):
    a1 = _actor(model, "Alice")
    a2 = _actor(model, "Bob")
    _assert_by(model, a1, "deposition.pdf")
    _assert_by(model, a2, "deposition.pdf")

    m = model.get_communication_map()
    assert len(m["actor_actor_edges"]) == 1
    edge = m["actor_actor_edges"][0]
    assert set([edge["actor_a_id"], edge["actor_b_id"]]) == {a1, a2}
    assert edge["shared_documents"] == 1
    assert "deposition.pdf" in edge["documents"]


def test_actors_no_shared_document_no_edge(model):
    a1 = _actor(model, "Alice")
    a2 = _actor(model, "Bob")
    _assert_by(model, a1, "doc_a.pdf")
    _assert_by(model, a2, "doc_b.pdf")

    m = model.get_communication_map()
    assert m["actor_actor_edges"] == []


def test_multiple_documents_per_actor(model):
    aid = _actor(model, "Alice")
    _assert_by(model, aid, "doc_a.pdf")
    _assert_by(model, aid, "doc_b.pdf")

    m = model.get_communication_map()
    docs_for_actor = {e["document_id"] for e in m["actor_document_edges"]}
    assert "doc_a.pdf" in docs_for_actor
    assert "doc_b.pdf" in docs_for_actor


def test_actor_without_occurrences_excluded(model):
    a_active = _actor(model, "Active")
    a_silent = _actor(model, "Silent")  # never speaks
    _assert_by(model, a_active, "doc.pdf")

    m = model.get_communication_map()
    actor_ids = {a["id"] for a in m["actors"]}
    assert a_active in actor_ids
    assert a_silent not in actor_ids


def test_shared_documents_count_in_actor_actor_edge(model):
    a1 = _actor(model, "Alice")
    a2 = _actor(model, "Bob")
    _assert_by(model, a1, "doc_a.pdf")
    _assert_by(model, a2, "doc_a.pdf")
    _assert_by(model, a1, "doc_b.pdf")
    _assert_by(model, a2, "doc_b.pdf")

    m = model.get_communication_map()
    edge = m["actor_actor_edges"][0]
    assert edge["shared_documents"] == 2
    assert set(edge["documents"]) == {"doc_a.pdf", "doc_b.pdf"}


def test_actor_name_in_edge(model):
    aid = _actor(model, "Jane Smith")
    _assert_by(model, aid, "letter.pdf")

    m = model.get_communication_map()
    edge = m["actor_document_edges"][0]
    assert edge["actor_name"] == "Jane Smith"


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------

def _reg(active, m):
    active[m.matter_id] = m
    return m.matter_id


def test_api_communication_map_empty(api_client):
    client, active = api_client
    m = MatterModel.open_in_memory()
    mid = _reg(active, m)
    resp = client.get(f"/matter/{mid}/communication-map")
    assert resp.status_code == 200
    body = resp.json()
    assert body["actors"] == []
    assert body["documents"] == []


def test_api_communication_map_with_actors(api_client):
    client, active = api_client
    m = MatterModel.open_in_memory()
    mid = _reg(active, m)

    a1 = _actor(m, "Alice")
    a2 = _actor(m, "Bob")
    _assert_by(m, a1, "contract.pdf")
    _assert_by(m, a2, "contract.pdf")

    resp = client.get(f"/matter/{mid}/communication-map")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["actors"]) == 2
    assert "contract.pdf" in body["documents"]
    assert len(body["actor_actor_edges"]) == 1
