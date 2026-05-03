"""Tests for AuthorityStore — legal research layer (SO-4).

Verifies:
1. upsert() creates a new authority and returns (id, True)
2. upsert() updates an existing authority and returns (id, False)
3. Blank citation raises ValueError
4. Invalid authority_type is coerced to 'unknown'
5. Invalid weight is coerced to 'unknown'
6. holdings and key_rules round-trip as lists
7. get() returns the record; get() for unknown id returns None
8. get_by_citation() returns the record or None
9. list_all() returns all; filtered by type/weight
10. search() matches citation and name substrings
11. link_to_issue() / unlink_from_issue() / list_for_issue()
12. Invalid relevance is coerced to 'neutral'
13. count() reflects stored records
14. Multi-matter isolation
15. API endpoints: POST, GET list, GET single, link, unlink, GET by issue
16. _extract_and_store_authorities() populates case/statute/regulation from text
"""

import pytest
from irys.matter import MatterModel, AuthorityStore, IssueType


@pytest.fixture
def model():
    return MatterModel.open_in_memory()


@pytest.fixture
def issue_id(model):
    iid, _ = model.issues.upsert_issue("Breach of contract", IssueType.CLAIM)
    return iid


# ---------------------------------------------------------------------------
# Basic upsert
# ---------------------------------------------------------------------------

def test_upsert_creates_new(model):
    auth_id, is_new = model.authority.upsert(
        citation="Bell Atl. Corp. v. Twombly, 550 U.S. 544",
        authority_type="case",
        weight="binding",
    )
    assert is_new is True
    assert auth_id and len(auth_id) > 0


def test_upsert_returns_false_on_duplicate_citation(model):
    model.authority.upsert(citation="Twombly, 550 U.S. 544")
    _, is_new = model.authority.upsert(citation="Twombly, 550 U.S. 544")
    assert is_new is False


def test_upsert_updates_fields_on_duplicate(model):
    model.authority.upsert(citation="Iqbal, 556 U.S. 662", name=None, jurisdiction=None)
    auth_id, _ = model.authority.upsert(
        citation="Iqbal, 556 U.S. 662",
        name="Ashcroft v. Iqbal",
        jurisdiction="U.S. Supreme Court",
        weight="binding",
    )
    auth = model.authority.get(auth_id)
    assert auth["name"] == "Ashcroft v. Iqbal"
    assert auth["jurisdiction"] == "U.S. Supreme Court"
    assert auth["weight"] == "binding"


def test_blank_citation_raises(model):
    with pytest.raises(ValueError, match="citation must not be blank"):
        model.authority.upsert(citation="   ")


# ---------------------------------------------------------------------------
# Validation / coercion
# ---------------------------------------------------------------------------

def test_invalid_type_coerced_to_unknown(model):
    auth_id, _ = model.authority.upsert(citation="Test v. Test", authority_type="alien")
    assert model.authority.get(auth_id)["authority_type"] == "unknown"


def test_invalid_weight_coerced_to_unknown(model):
    auth_id, _ = model.authority.upsert(citation="Test v. Test", weight="super_binding")
    assert model.authority.get(auth_id)["weight"] == "unknown"


def test_all_valid_types_accepted(model):
    for i, atype in enumerate(AuthorityStore.VALID_TYPES):
        auth_id, _ = model.authority.upsert(
            citation=f"Cite_{i}", authority_type=atype
        )
        assert model.authority.get(auth_id)["authority_type"] == atype


def test_all_valid_weights_accepted(model):
    for i, w in enumerate(AuthorityStore.VALID_WEIGHTS):
        auth_id, _ = model.authority.upsert(citation=f"WeightCite_{i}", weight=w)
        assert model.authority.get(auth_id)["weight"] == w


# ---------------------------------------------------------------------------
# holdings and key_rules round-trip
# ---------------------------------------------------------------------------

def test_holdings_list_roundtrip(model):
    holdings = ["Pleading standard requires plausibility", "Rule 12(b)(6) motion"]
    auth_id, _ = model.authority.upsert(citation="Twombly", holdings=holdings)
    result = model.authority.get(auth_id)
    assert result["holdings"] == holdings


def test_key_rules_list_roundtrip(model):
    rules = ["Notice pleading replaced by plausibility", "Labels insufficient"]
    auth_id, _ = model.authority.upsert(citation="Iqbal", key_rules=rules)
    assert model.authority.get(auth_id)["key_rules"] == rules


def test_null_holdings_returns_empty_list(model):
    auth_id, _ = model.authority.upsert(citation="NullHoldings", holdings=None)
    assert model.authority.get(auth_id)["holdings"] == []


# ---------------------------------------------------------------------------
# get / get_by_citation
# ---------------------------------------------------------------------------

def test_get_returns_none_for_unknown_id(model):
    assert model.authority.get("nonexistent-id") is None


def test_get_by_citation_returns_record(model):
    model.authority.upsert(citation="Smith v. Jones, 42 F.3d 1", name="Smith")
    result = model.authority.get_by_citation("Smith v. Jones, 42 F.3d 1")
    assert result is not None
    assert result["name"] == "Smith"


def test_get_by_citation_returns_none_for_unknown(model):
    assert model.authority.get_by_citation("Nonexistent v. Citation") is None


# ---------------------------------------------------------------------------
# list_all / search
# ---------------------------------------------------------------------------

def test_list_all_returns_all(model):
    model.authority.upsert(citation="Case A", authority_type="case")
    model.authority.upsert(citation="42 U.S.C. § 1983", authority_type="statute")
    assert len(model.authority.list_all()) == 2


def test_list_all_filter_by_type(model):
    model.authority.upsert(citation="Case A", authority_type="case")
    model.authority.upsert(citation="Statute B", authority_type="statute")
    cases = model.authority.list_all(authority_type="case")
    assert len(cases) == 1
    assert cases[0]["authority_type"] == "case"


def test_list_all_filter_by_weight(model):
    model.authority.upsert(citation="Binding A", weight="binding")
    model.authority.upsert(citation="Persuasive B", weight="persuasive")
    binding = model.authority.list_all(weight="binding")
    assert len(binding) == 1


def test_search_matches_citation(model):
    model.authority.upsert(citation="Twombly Plausibility Case, 550 U.S. 544")
    results = model.authority.search("Twombly")
    assert len(results) == 1


def test_search_matches_name(model):
    model.authority.upsert(citation="550 U.S. 544", name="Bell Atlantic v. Twombly")
    results = model.authority.search("Bell Atlantic")
    assert len(results) == 1


def test_search_returns_empty_for_no_match(model):
    model.authority.upsert(citation="Smith v. Jones")
    assert model.authority.search("Iqbal") == []


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------

def test_count_empty(model):
    assert model.authority.count() == 0


def test_count_increments(model):
    model.authority.upsert(citation="A")
    model.authority.upsert(citation="B")
    assert model.authority.count() == 2


# ---------------------------------------------------------------------------
# Issue links
# ---------------------------------------------------------------------------

def test_link_to_issue(model, issue_id):
    auth_id, _ = model.authority.upsert(citation="Twombly, 550 U.S. 544")
    model.authority.link_to_issue(auth_id, issue_id, relevance="supporting")
    linked = model.authority.list_for_issue(issue_id)
    assert len(linked) == 1
    assert linked[0]["link_relevance"] == "supporting"


def test_link_idempotent(model, issue_id):
    auth_id, _ = model.authority.upsert(citation="Twombly")
    model.authority.link_to_issue(auth_id, issue_id, relevance="supporting")
    model.authority.link_to_issue(auth_id, issue_id, relevance="supporting")  # second call
    assert len(model.authority.list_for_issue(issue_id)) == 1


def test_link_relevance_update_via_replace(model, issue_id):
    """INSERT OR REPLACE changes relevance on second call."""
    auth_id, _ = model.authority.upsert(citation="Twombly")
    model.authority.link_to_issue(auth_id, issue_id, relevance="supporting")
    model.authority.link_to_issue(auth_id, issue_id, relevance="attacking")
    linked = model.authority.list_for_issue(issue_id)
    assert linked[0]["link_relevance"] == "attacking"


def test_invalid_relevance_coerced_to_neutral(model, issue_id):
    auth_id, _ = model.authority.upsert(citation="Test")
    model.authority.link_to_issue(auth_id, issue_id, relevance="extremely_supporting")
    assert model.authority.list_for_issue(issue_id)[0]["link_relevance"] == "neutral"


def test_unlink_from_issue(model, issue_id):
    auth_id, _ = model.authority.upsert(citation="Twombly")
    model.authority.link_to_issue(auth_id, issue_id)
    model.authority.unlink_from_issue(auth_id, issue_id)
    assert model.authority.list_for_issue(issue_id) == []


def test_list_for_issue_empty_when_none_linked(model, issue_id):
    model.authority.upsert(citation="Twombly")  # not linked
    assert model.authority.list_for_issue(issue_id) == []


# ---------------------------------------------------------------------------
# Multi-matter isolation
# ---------------------------------------------------------------------------

def test_authority_isolated_per_matter(model):
    m_b = MatterModel.open_in_memory()
    model.authority.upsert(citation="Twombly")
    assert m_b.authority.count() == 0


# ---------------------------------------------------------------------------
# _extract_and_store_authorities (engine integration)
# ---------------------------------------------------------------------------

@pytest.fixture
def engine_with_model(model):
    """Return a minimal RLMEngine with _matter_model wired."""
    import unittest.mock as mock
    from irys.rlm.engine import RLMEngine, ServiceConfig
    config = ServiceConfig()
    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = model
    engine._logger = logging.getLogger("test_engine")
    return engine


import logging


def test_extract_case_citation(model):
    """Case citations matching 'P v. D, Vol Reporter Page' are stored."""
    from irys.rlm.engine import RLMEngine
    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = model

    text = (
        "The court applied Bell Atl. Corp. v. Twombly, 550 U.S. 544 (2007), "
        "holding that plausibility is required."
    )
    engine._extract_and_store_authorities(text)
    assert model.authority.count() >= 1
    results = model.authority.search("Twombly")
    assert len(results) == 1
    assert results[0]["authority_type"] == "case"


def test_extract_federal_statute(model):
    """42 U.S.C. § 1983 is stored as statute with binding weight."""
    from irys.rlm.engine import RLMEngine
    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = model

    text = "The plaintiff brings claims under 42 U.S.C. § 1983 for deprivation of rights."
    engine._extract_and_store_authorities(text)
    statutes = model.authority.list_all(authority_type="statute")
    assert len(statutes) >= 1
    assert statutes[0]["weight"] == "binding"


def test_extract_cfr_regulation(model):
    """29 C.F.R. § 825.100 is stored as regulation."""
    from irys.rlm.engine import RLMEngine
    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = model

    text = "The FMLA regulations at 29 C.F.R. § 825.100 apply."
    engine._extract_and_store_authorities(text)
    regs = model.authority.list_all(authority_type="regulation")
    assert len(regs) >= 1


def test_extract_no_duplicates_from_repeated_citations(model):
    """Same citation appearing twice produces only one record."""
    from irys.rlm.engine import RLMEngine
    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = model

    text = (
        "See Twombly, 550 U.S. 544.  The Twombly, 550 U.S. 544 standard requires plausibility."
    )
    engine._extract_and_store_authorities(text)
    # Should not double-store; upsert handles it
    assert model.authority.count() <= 1  # <=1 because regex might not match truncated form


def test_extract_no_false_positives_from_plain_text(model):
    """Plain text without citations produces no authority records."""
    from irys.rlm.engine import RLMEngine
    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = model

    text = "The parties disagree about the contract terms. No legal citations here."
    engine._extract_and_store_authorities(text)
    assert model.authority.count() == 0


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

def _reg(active, m):
    active[m.matter_id] = m
    return m.matter_id


def test_api_post_authority(api_client):
    client, active = api_client
    m = MatterModel.open_in_memory()
    mid = _reg(active, m)

    resp = client.post(f"/matter/{mid}/authorities", json={
        "citation": "Bell Atl. Corp. v. Twombly, 550 U.S. 544",
        "authority_type": "case",
        "weight": "binding",
        "holdings": ["Plausibility pleading standard"],
    })
    assert resp.status_code == 201
    body = resp.json()
    assert body["is_new"] is True
    assert body["id"]


def test_api_post_authority_missing_citation(api_client):
    client, active = api_client
    m = MatterModel.open_in_memory()
    mid = _reg(active, m)
    resp = client.post(f"/matter/{mid}/authorities", json={"authority_type": "case"})
    assert resp.status_code == 422


def test_api_list_authorities(api_client):
    client, active = api_client
    m = MatterModel.open_in_memory()
    mid = _reg(active, m)

    client.post(f"/matter/{mid}/authorities", json={"citation": "A v. B", "authority_type": "case"})
    client.post(f"/matter/{mid}/authorities", json={"citation": "42 U.S.C. § 1983", "authority_type": "statute"})

    resp = client.get(f"/matter/{mid}/authorities")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_api_list_authorities_filtered(api_client):
    client, active = api_client
    m = MatterModel.open_in_memory()
    mid = _reg(active, m)

    client.post(f"/matter/{mid}/authorities", json={"citation": "A v. B", "authority_type": "case"})
    client.post(f"/matter/{mid}/authorities", json={"citation": "42 U.S.C. § 1983", "authority_type": "statute"})

    resp = client.get(f"/matter/{mid}/authorities?authority_type=case")
    assert len(resp.json()) == 1


def test_api_get_authority_by_id(api_client):
    client, active = api_client
    m = MatterModel.open_in_memory()
    mid = _reg(active, m)

    post_resp = client.post(f"/matter/{mid}/authorities", json={"citation": "Twombly"})
    auth_id = post_resp.json()["id"]

    resp = client.get(f"/matter/{mid}/authorities/{auth_id}")
    assert resp.status_code == 200
    assert resp.json()["citation"] == "Twombly"


def test_api_get_authority_not_found(api_client):
    client, active = api_client
    m = MatterModel.open_in_memory()
    mid = _reg(active, m)
    resp = client.get(f"/matter/{mid}/authorities/nonexistent-id")
    assert resp.status_code == 404


def test_api_link_and_unlink_authority_to_issue(api_client):
    client, active = api_client
    m = MatterModel.open_in_memory()
    mid = _reg(active, m)

    issue_id, _ = m.issues.upsert_issue("Pleading standard", IssueType.CLAIM)
    post_resp = client.post(f"/matter/{mid}/authorities", json={"citation": "Twombly"})
    auth_id = post_resp.json()["id"]

    # Link
    link_resp = client.post(
        f"/matter/{mid}/authorities/{auth_id}/issues/{issue_id}",
        params={"relevance": "supporting"},
    )
    assert link_resp.status_code == 200
    assert link_resp.json()["status"] == "linked"

    # Verify via issue authorities endpoint
    issue_resp = client.get(f"/matter/{mid}/issues/{issue_id}/authorities")
    assert issue_resp.status_code == 200
    assert len(issue_resp.json()) == 1

    # Unlink
    unlink_resp = client.delete(f"/matter/{mid}/authorities/{auth_id}/issues/{issue_id}")
    assert unlink_resp.json()["status"] == "unlinked"
    assert client.get(f"/matter/{mid}/issues/{issue_id}/authorities").json() == []


def test_api_authority_search(api_client):
    client, active = api_client
    m = MatterModel.open_in_memory()
    mid = _reg(active, m)

    client.post(f"/matter/{mid}/authorities", json={"citation": "Twombly, 550 U.S. 544"})
    client.post(f"/matter/{mid}/authorities", json={"citation": "Iqbal, 556 U.S. 662"})

    resp = client.get(f"/matter/{mid}/authorities?search=Twombly")
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) == 1
    assert "Twombly" in results[0]["citation"]


def test_get_network_empty():
    m = MatterModel.open_in_memory()
    net = m.authority.get_network()
    assert net["authorities"] == []
    assert net["issue_links"] == {}


def test_get_network_with_links():
    m = MatterModel.open_in_memory()
    aid, _ = m.authority.upsert("Twombly, 550 U.S. 544", weight="binding")
    iid, _ = m.issues.upsert_issue("Plausibility", IssueType.CLAIM)
    m.authority.link_to_issue(aid, iid, relevance="supporting")
    net = m.authority.get_network(issue_titles={iid: "Plausibility"})
    assert len(net["authorities"]) == 1
    assert aid in net["issue_links"]
    link = net["issue_links"][aid][0]
    assert link["issue_id"] == iid
    assert link["issue_title"] == "Plausibility"
    assert link["relevance"] == "supporting"


def test_get_network_multiple_authorities_and_issues():
    m = MatterModel.open_in_memory()
    a1, _ = m.authority.upsert("Case A", weight="binding")
    a2, _ = m.authority.upsert("Statute B", authority_type="statute", weight="persuasive")
    i1, _ = m.issues.upsert_issue("Issue 1", IssueType.CLAIM)
    i2, _ = m.issues.upsert_issue("Issue 2", IssueType.CLAIM)
    m.authority.link_to_issue(a1, i1, relevance="supporting")
    m.authority.link_to_issue(a1, i2, relevance="attacking")
    m.authority.link_to_issue(a2, i1, relevance="neutral")
    net = m.authority.get_network()
    assert len(net["authorities"]) == 2
    assert len(net["issue_links"][a1]) == 2
    assert len(net["issue_links"][a2]) == 1


def test_api_authority_network(api_client):
    client, active = api_client
    m = MatterModel.open_in_memory()
    mid = _reg(active, m)
    client.post(f"/matter/{mid}/authorities", json={"citation": "Twombly"})
    resp = client.get(f"/matter/{mid}/authority-network")
    assert resp.status_code == 200
    data = resp.json()
    assert "authorities" in data
    assert "issue_links" in data
    assert len(data["authorities"]) == 1
