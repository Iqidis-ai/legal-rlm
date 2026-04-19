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


# ---------------------------------------------------------------------------
# Bridge: runtime.record_fact auto-populates speaker_actor_id from doc_card
# ---------------------------------------------------------------------------


def test_record_fact_resolves_speaker_from_doc_card_sender(model):
    """Subagent (adv#13 cycle) identified: the communication graph
    was LIVE-BUT-STARVED — schema + query + UI all wired, but
    speaker_actor_id was never populated during extraction, so every
    real matter's get_communication_map() returned empty. Fix:
    runtime.record_fact now resolves document_card.sender to an
    actor and stamps speaker_actor_id on the occurrence.

    This regression asserts the whole bridge end-to-end:
     - inventory + card seeded with a sender
     - record_fact called with only document_id (no explicit actor)
     - occurrence row shows the resolved speaker_actor_id
     - get_communication_map now returns a real actor_document_edge
    """
    from irys.matter.runtime import MatterRuntimeAdapter

    inv_id, _ = model.inventory.upsert(
        "emails/smith_to_jones.eml", "s" * 64, size_bytes=100, file_type="eml",
    )
    model.document_cards.upsert(
        doc_id=inv_id, title="smith_to_jones.eml", doc_type="email",
        source_side="plaintiff", source_role="informal",
        sender="John Smith", recipient="Jane Jones",
    )
    run_id = model.start_run("bridge test")
    adapter = MatterRuntimeAdapter(model, run_id=run_id)

    aid = adapter.record_fact(
        "Smith confirmed the April 15 deadline",
        "emails/smith_to_jones.eml",
    )
    row = model.db.execute(
        "SELECT speaker_actor_id FROM assertion_occurrence WHERE assertion_id=?",
        (aid,),
    ).fetchone()
    assert row["speaker_actor_id"] is not None
    cm = model.get_communication_map()
    assert len(cm["actors"]) == 1
    assert len(cm["actor_document_edges"]) == 1
    # Sender name round-trips.
    assert cm["actors"][0]["name"] == "John Smith"


def test_speaker_actor_resolution_is_cached_per_adapter(model):
    """adv#14 Finding #3: the bridge was doing one doc_card+actor
    lookup per assertion — a 10k-assertion import from 100 docs
    fired 10k redundant queries. Now per-adapter cache keyed on
    document_id collapses repeat lookups."""
    from irys.matter.runtime import MatterRuntimeAdapter

    inv_id, _ = model.inventory.upsert(
        "emails/e.eml", "s" * 64, size_bytes=1, file_type="eml",
    )
    model.document_cards.upsert(
        doc_id=inv_id, title="e.eml", doc_type="email",
        source_side="plaintiff", source_role="informal",
        sender="Cache Subject",
    )
    run_id = model.start_run("cache test")
    adapter = MatterRuntimeAdapter(model, run_id=run_id)

    # First call populates the cache. Subsequent calls with the
    # same doc_id must return the same actor id WITHOUT hitting
    # the DB — verified by spying on model.db.execute.
    a1 = adapter._resolve_speaker_actor_for_document("emails/e.eml")
    assert a1 is not None

    calls_after_first = 0
    orig_execute = model.db.execute

    def _counting(sql, params=()):
        nonlocal calls_after_first
        calls_after_first += 1
        return orig_execute(sql, params)

    model.db.execute = _counting  # type: ignore[assignment]
    try:
        for _ in range(50):
            a2 = adapter._resolve_speaker_actor_for_document("emails/e.eml")
            assert a2 == a1
    finally:
        model.db.execute = orig_execute  # type: ignore[assignment]
    assert calls_after_first == 0, (
        f"expected 0 DB calls for repeat lookups, got {calls_after_first}"
    )


def test_speaker_bridge_resolves_to_existing_alias(model):
    """adv#14 Finding #4: the bridge used to call upsert_actor
    directly, which bypasses ActorStore's alias resolution. A
    doc_card with sender 'John Smith' would create a distinct actor
    even when an existing 'Jonathan Smith' has 'John Smith'
    registered as an alias. Now the bridge tries resolve_by_name
    first."""
    from irys.matter.runtime import MatterRuntimeAdapter

    # Pre-seed an actor with an alias.
    jon_id, _ = model.actors.upsert_actor("Jonathan Smith")
    model.actors.add_alias(jon_id, "John Smith")

    inv_id, _ = model.inventory.upsert(
        "letter.pdf", "s" * 64, size_bytes=1, file_type="pdf",
    )
    model.document_cards.upsert(
        doc_id=inv_id, title="letter.pdf", doc_type="letter",
        source_side="plaintiff", source_role="informal",
        sender="John Smith",  # alias for jon_id
    )
    run_id = model.start_run("alias test")
    adapter = MatterRuntimeAdapter(model, run_id=run_id)

    resolved = adapter._resolve_speaker_actor_for_document("letter.pdf")
    assert resolved == jon_id, (
        "expected speaker to resolve to existing Jonathan Smith via alias"
    )
    # Actor count stays at 1 — no duplicate was forked.
    row = model.db.execute(
        "SELECT COUNT(*) AS n FROM actor WHERE matter_id = ?",
        (model.matter_id,),
    ).fetchone()
    assert row["n"] == 1


def test_record_fact_without_doc_card_sender_leaves_speaker_null(model):
    """Guard: when the document has no card OR the card has no
    sender, the bridge returns None. record_fact still succeeds;
    the occurrence just has a NULL speaker_actor_id."""
    from irys.matter.runtime import MatterRuntimeAdapter

    run_id = model.start_run("no-card")
    adapter = MatterRuntimeAdapter(model, run_id=run_id)
    aid = adapter.record_fact(
        "A fact from a doc with no card",
        "random/doc.pdf",
    )
    row = model.db.execute(
        "SELECT speaker_actor_id FROM assertion_occurrence WHERE assertion_id=?",
        (aid,),
    ).fetchone()
    assert row["speaker_actor_id"] is None
    cm = model.get_communication_map()
    assert cm["actor_document_edges"] == []
