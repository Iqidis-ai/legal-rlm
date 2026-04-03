"""Tests for DecisionContextStore and decision-context API endpoints.

Verifies:
1. DecisionContextStore.set() persists all fields correctly
2. DecisionContextStore.get() returns None when unset, dict when set
3. Upsert semantics: second set() updates the same row (no duplicates)
4. Invalid decision_maker_type is coerced to 'unknown'
5. Invalid objective is coerced to 'unknown'
6. scope_narrow=True / False round-trips correctly
7. clear() removes the row; subsequent get() returns None
8. Isolated per-matter: context from matter A does not bleed into matter B
9. API GET /matter/{id}/decision-context returns null when unset
10. API PUT /matter/{id}/decision-context stores and returns context
11. API DELETE /matter/{id}/decision-context clears and confirms
"""

import pytest
from irys.matter import MatterModel, DecisionContextStore


@pytest.fixture
def model():
    return MatterModel.open_in_memory()


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------

def test_get_returns_none_when_unset(model):
    assert model.decision_context.get() is None


def test_set_persists_fields(model):
    model.decision_context.set(
        decision_maker_type="judge",
        decision_maker_name="Hon. Smith",
        objective="motion_practice",
        strategic_notes="Focus on contract formation",
        scope_narrow=True,
    )
    ctx = model.decision_context.get()
    assert ctx is not None
    assert ctx["decision_maker_type"] == "judge"
    assert ctx["decision_maker_name"] == "Hon. Smith"
    assert ctx["objective"] == "motion_practice"
    assert ctx["strategic_notes"] == "Focus on contract formation"
    assert ctx["scope_narrow"] is True


def test_set_returns_id(model):
    ctx_id = model.decision_context.set(decision_maker_type="partner")
    assert ctx_id and len(ctx_id) > 0


def test_get_includes_matter_id(model):
    model.decision_context.set(decision_maker_type="client")
    ctx = model.decision_context.get()
    assert ctx["matter_id"] == model.matter_id


# ---------------------------------------------------------------------------
# Upsert semantics
# ---------------------------------------------------------------------------

def test_second_set_updates_not_inserts(model):
    model.decision_context.set(decision_maker_type="partner", objective="settlement")
    id1 = model.decision_context.get()["id"]

    model.decision_context.set(decision_maker_type="judge", objective="motion_practice")
    ctx = model.decision_context.get()
    assert ctx["id"] == id1  # same row
    assert ctx["decision_maker_type"] == "judge"
    assert ctx["objective"] == "motion_practice"


def test_upsert_updates_all_fields(model):
    model.decision_context.set(
        decision_maker_type="partner",
        strategic_notes="Old notes",
        scope_narrow=False,
    )
    model.decision_context.set(
        decision_maker_type="client",
        strategic_notes="New notes",
        scope_narrow=True,
    )
    ctx = model.decision_context.get()
    assert ctx["decision_maker_type"] == "client"
    assert ctx["strategic_notes"] == "New notes"
    assert ctx["scope_narrow"] is True


# ---------------------------------------------------------------------------
# Validation / coercion
# ---------------------------------------------------------------------------

def test_invalid_maker_type_coerced_to_unknown(model):
    model.decision_context.set(decision_maker_type="alien_overlord")
    ctx = model.decision_context.get()
    assert ctx["decision_maker_type"] == "unknown"


def test_invalid_objective_coerced_to_unknown(model):
    model.decision_context.set(objective="win_at_all_costs")
    ctx = model.decision_context.get()
    assert ctx["objective"] == "unknown"


def test_valid_maker_types_accepted(model):
    for maker_type in DecisionContextStore.VALID_MAKER_TYPES:
        m = MatterModel.open_in_memory()
        m.decision_context.set(decision_maker_type=maker_type)
        assert m.decision_context.get()["decision_maker_type"] == maker_type


def test_valid_objectives_accepted(model):
    for obj in DecisionContextStore.VALID_OBJECTIVES:
        m = MatterModel.open_in_memory()
        m.decision_context.set(objective=obj)
        assert m.decision_context.get()["objective"] == obj


def test_none_maker_type_stored_as_none(model):
    model.decision_context.set(decision_maker_type=None)
    ctx = model.decision_context.get()
    assert ctx["decision_maker_type"] is None


# ---------------------------------------------------------------------------
# scope_narrow flag
# ---------------------------------------------------------------------------

def test_scope_narrow_false_roundtrip(model):
    model.decision_context.set(scope_narrow=False)
    assert model.decision_context.get()["scope_narrow"] is False


def test_scope_narrow_true_roundtrip(model):
    model.decision_context.set(scope_narrow=True)
    assert model.decision_context.get()["scope_narrow"] is True


# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------

def test_clear_removes_row(model):
    model.decision_context.set(decision_maker_type="judge")
    model.decision_context.clear()
    assert model.decision_context.get() is None


def test_clear_when_unset_is_noop(model):
    # Should not raise
    model.decision_context.clear()
    assert model.decision_context.get() is None


def test_set_after_clear_works(model):
    model.decision_context.set(decision_maker_type="partner")
    model.decision_context.clear()
    model.decision_context.set(decision_maker_type="client")
    ctx = model.decision_context.get()
    assert ctx["decision_maker_type"] == "client"


# ---------------------------------------------------------------------------
# Multi-matter isolation
# ---------------------------------------------------------------------------

def test_isolation_between_matters(model):
    """Context stored for matter A must not bleed into matter B."""
    model_b = MatterModel.open_in_memory()

    model.decision_context.set(decision_maker_type="judge", objective="trial_prep")
    # matter B has no context set
    assert model_b.decision_context.get() is None


def test_separate_matters_independent_contexts():
    """Two matters with separate contexts don't overwrite each other."""
    m_a = MatterModel.open_in_memory()
    m_b = MatterModel.open_in_memory()

    m_a.decision_context.set(decision_maker_type="judge")
    m_b.decision_context.set(decision_maker_type="client")

    assert m_a.decision_context.get()["decision_maker_type"] == "judge"
    assert m_b.decision_context.get()["decision_maker_type"] == "client"


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@pytest.fixture
def api_client():
    from fastapi.testclient import TestClient
    from irys.service.api import app, _active_matter_models
    _active_matter_models.clear()
    return TestClient(app), _active_matter_models


def _register_model(active_models, model):
    active_models[model.matter_id] = model
    return model.matter_id


def test_api_get_returns_null_when_unset(api_client):
    client, active = api_client
    m = MatterModel.open_in_memory()
    mid = _register_model(active, m)

    resp = client.get(f"/matter/{mid}/decision-context")
    assert resp.status_code == 200
    assert resp.json() is None


def test_api_put_stores_context(api_client):
    client, active = api_client
    m = MatterModel.open_in_memory()
    mid = _register_model(active, m)

    payload = {
        "decision_maker_type": "judge",
        "decision_maker_name": "Hon. Rivera",
        "objective": "motion_practice",
        "strategic_notes": "Focus on venue",
        "scope_narrow": True,
    }
    resp = client.put(f"/matter/{mid}/decision-context", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["matter_id"] == mid
    assert body["id"]


def test_api_put_then_get_round_trip(api_client):
    client, active = api_client
    m = MatterModel.open_in_memory()
    mid = _register_model(active, m)

    client.put(f"/matter/{mid}/decision-context", json={
        "decision_maker_type": "partner",
        "objective": "settlement",
    })

    resp = client.get(f"/matter/{mid}/decision-context")
    assert resp.status_code == 200
    ctx = resp.json()
    assert ctx["decision_maker_type"] == "partner"
    assert ctx["objective"] == "settlement"


def test_api_delete_clears_context(api_client):
    client, active = api_client
    m = MatterModel.open_in_memory()
    mid = _register_model(active, m)

    client.put(f"/matter/{mid}/decision-context", json={"decision_maker_type": "client"})
    resp = client.delete(f"/matter/{mid}/decision-context")
    assert resp.status_code == 200
    assert resp.json()["status"] == "cleared"

    # Verify it's gone
    resp2 = client.get(f"/matter/{mid}/decision-context")
    assert resp2.json() is None


def test_api_get_404_for_unknown_matter(api_client):
    client, _ = api_client
    resp = client.get("/matter/nonexistent-matter/decision-context")
    assert resp.status_code == 404


def test_api_put_invalid_type_coerced(api_client):
    client, active = api_client
    m = MatterModel.open_in_memory()
    mid = _register_model(active, m)

    client.put(f"/matter/{mid}/decision-context", json={"decision_maker_type": "supervillain"})
    ctx = client.get(f"/matter/{mid}/decision-context").json()
    assert ctx["decision_maker_type"] == "unknown"
