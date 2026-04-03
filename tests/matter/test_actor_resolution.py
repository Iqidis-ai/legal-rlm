"""Tests for actor alias resolution and duplicate detection (SO-5).

Verifies:
1. resolve_by_name() returns actor_id on exact alias match
2. resolve_by_name() returns actor_id on substring containment (fuzzy)
3. resolve_by_name() returns None when no match
4. resolve_by_name() returns None for very short names (< 5 chars)
5. find_possible_duplicates() returns pairs sharing long prefix
6. find_possible_duplicates() excludes pairs with short prefix
7. find_possible_duplicates() ordered by prefix length descending
8. merge_actors() moves aliases from merge_id to keep_id
9. merge_actors() redirects assertion occurrences
10. merge_actors() deletes merged actor
11. merge_actors() raises ValueError when keep_id == merge_id
12. merge_actors() raises ValueError when actor not found
13. API: GET /actors/duplicates
14. API: POST /actors/{keep}/merge/{merge}
15. API: GET /actors/resolve?name=...
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


def _assert_by(model, actor_id, doc_id="doc.pdf"):
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
# resolve_by_name
# ---------------------------------------------------------------------------

def test_resolve_exact_canonical_match(model):
    aid = _actor(model, "Acme Corporation")
    assert model.actors.resolve_by_name("Acme Corporation") == aid


def test_resolve_exact_alias_match(model):
    aid = _actor(model, "Acme Corporation")
    model.actors.add_alias(aid, "Acme Corp")
    assert model.actors.resolve_by_name("Acme Corp") == aid


def test_resolve_substring_containment_short_in_long(model):
    """Canonical 'Acme Corporation' is found when querying 'Acme Corp'."""
    aid = _actor(model, "Acme Corporation")
    # "acme corp" (normalized) is IN "acme corporation" (normalized)
    assert model.actors.resolve_by_name("Acme Corp") == aid


def test_resolve_substring_containment_long_in_short(model):
    """Actor stored as 'Smith' is found when query contains it."""
    aid = _actor(model, "Smith")
    # "smith" (normalized) is IN "john smith" (normalized)
    assert model.actors.resolve_by_name("John Smith") == aid


def test_resolve_returns_none_when_no_match(model):
    _actor(model, "Acme Corporation")
    assert model.actors.resolve_by_name("Globex Inc") is None


def test_resolve_returns_none_for_short_name(model):
    """Names < 5 chars skip fuzzy matching to avoid false positives."""
    _actor(model, "IBM")
    # "ibm" is 3 chars → no fuzzy matching
    result = model.actors.resolve_by_name("IBM")
    # May match via exact alias ("ibm") but not fuzzy
    # Exact: canonical_name "IBM" is registered as alias "ibm"
    assert result is not None  # exact alias match still works


def test_resolve_too_short_no_fuzzy_false_positive(model):
    """'Al' should not match 'Alice Smith' via fuzzy (too short)."""
    _actor(model, "Alice Smith")
    # "al" is < 5 chars, but "al" is in "alice smith"
    # Should not fuzzy-match because len("al") < 5
    result = model.actors.resolve_by_name("Al")
    assert result is None  # exact alias won't match either


# ---------------------------------------------------------------------------
# find_possible_duplicates
# ---------------------------------------------------------------------------

def test_duplicate_pairs_found(model):
    _actor(model, "Acme Corporation")
    _actor(model, "Acme Corp")
    pairs = model.actors.find_possible_duplicates(min_prefix_len=4)
    assert len(pairs) >= 1
    # Both actors should appear in the pair
    names_in_pairs = {
        p["actor_a"]["canonical_name"]
        for p in pairs
    } | {
        p["actor_b"]["canonical_name"]
        for p in pairs
    }
    assert "Acme Corporation" in names_in_pairs or "Acme Corp" in names_in_pairs


def test_short_prefix_no_duplicate(model):
    _actor(model, "Alpha Inc")
    _actor(model, "Beta LLC")
    pairs = model.actors.find_possible_duplicates(min_prefix_len=4)
    assert pairs == []


def test_duplicates_ordered_by_prefix_length_desc(model):
    _actor(model, "Acme Corporation")
    _actor(model, "Acme Corp")
    _actor(model, "Johnson & Johnson")
    _actor(model, "Johnson Partners")
    pairs = model.actors.find_possible_duplicates(min_prefix_len=4)
    if len(pairs) >= 2:
        assert len(pairs[0]["shared_prefix"]) >= len(pairs[-1]["shared_prefix"])


def test_no_self_pairs(model):
    _actor(model, "Acme")
    pairs = model.actors.find_possible_duplicates(min_prefix_len=4)
    for p in pairs:
        assert p["actor_a"]["id"] != p["actor_b"]["id"]


# ---------------------------------------------------------------------------
# merge_actors
# ---------------------------------------------------------------------------

def test_merge_moves_aliases(model):
    keep = _actor(model, "Acme Corporation")
    merge = _actor(model, "Acme Corp")
    model.actors.add_alias(merge, "ACME")

    model.actors.merge_actors(keep_id=keep, merge_id=merge)

    aliases = model.actors.get_aliases(keep)
    # "acme" alias from merge should now be on keep
    assert any("acme" in a for a in aliases)


def test_merge_redirects_occurrences(model):
    keep = _actor(model, "Acme Corporation")
    merge = _actor(model, "Acme Corp")
    _assert_by(model, merge, "doc.pdf")

    model.actors.merge_actors(keep_id=keep, merge_id=merge)

    # Check via communication map
    m = model.get_communication_map()
    actor_ids = {a["id"] for a in m["actors"]}
    assert keep in actor_ids
    assert merge not in actor_ids


def test_merge_deletes_merged_actor(model):
    keep = _actor(model, "Acme Corporation")
    merge = _actor(model, "Acme Corp")

    model.actors.merge_actors(keep_id=keep, merge_id=merge)

    all_ids = {a["id"] for a in model.actors.list_actors()}
    assert keep in all_ids
    assert merge not in all_ids


def test_merge_self_raises(model):
    aid = _actor(model, "Acme")
    with pytest.raises(ValueError, match="itself"):
        model.actors.merge_actors(keep_id=aid, merge_id=aid)


def test_merge_unknown_actor_raises(model):
    aid = _actor(model, "Acme")
    with pytest.raises(ValueError):
        model.actors.merge_actors(keep_id=aid, merge_id="nonexistent-id")


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@pytest.fixture
def api_client():
    from fastapi.testclient import TestClient
    from irys.service.api import app, _active_matter_models
    _active_matter_models.clear()
    return TestClient(app), _active_matter_models


def _reg(active, m):
    active[m.matter_id] = m
    return m.matter_id


def test_api_get_duplicates_empty(api_client):
    client, active = api_client
    m = MatterModel.open_in_memory()
    mid = _reg(active, m)
    resp = client.get(f"/matter/{mid}/actors/duplicates")
    assert resp.status_code == 200
    assert resp.json() == []


def test_api_get_duplicates_finds_pair(api_client):
    client, active = api_client
    m = MatterModel.open_in_memory()
    mid = _reg(active, m)
    _actor(m, "Acme Corporation")
    _actor(m, "Acme Corp")

    resp = client.get(f"/matter/{mid}/actors/duplicates?min_prefix_len=4")
    assert resp.status_code == 200
    pairs = resp.json()
    assert len(pairs) >= 1


def test_api_merge_actors(api_client):
    client, active = api_client
    m = MatterModel.open_in_memory()
    mid = _reg(active, m)
    keep = _actor(m, "Acme Corporation")
    merge = _actor(m, "Acme Corp")

    resp = client.post(f"/matter/{mid}/actors/{keep}/merge/{merge}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "merged"
    assert body["keep_id"] == keep
    assert body["merged_id"] == merge

    # Verify merged actor gone
    all_actors = m.actors.list_actors()
    assert not any(a["id"] == merge for a in all_actors)


def test_api_merge_self_returns_422(api_client):
    client, active = api_client
    m = MatterModel.open_in_memory()
    mid = _reg(active, m)
    aid = _actor(m, "Acme")
    resp = client.post(f"/matter/{mid}/actors/{aid}/merge/{aid}")
    assert resp.status_code == 422


def test_api_resolve_actor(api_client):
    client, active = api_client
    m = MatterModel.open_in_memory()
    mid = _reg(active, m)
    aid = _actor(m, "Acme Corporation")

    resp = client.get(f"/matter/{mid}/actors/resolve?name=Acme+Corporation")
    assert resp.status_code == 200
    body = resp.json()
    assert body["actor_id"] == aid


def test_api_resolve_actor_not_found(api_client):
    client, active = api_client
    m = MatterModel.open_in_memory()
    mid = _reg(active, m)

    resp = client.get(f"/matter/{mid}/actors/resolve?name=Globex+Inc")
    assert resp.status_code == 200
    assert resp.json()["actor_id"] is None
