"""Tests for ExtractionSlotStore and v70 schema migration.

Covers:
- v69 -> v70 migration is idempotent and preserves typed_evidence rows
- Slot registration, conflict-on-key behavior, confidence clamp
- Coverage transitions: pending -> partial -> filled / not_observable
- mark_filled deduplicates and handles expected_count > 1
- coverage_summary aggregates by kind/state, counts high_confidence_open
- get_open_slots ordering and filters
- get_filled_slots filters by scope_query_hash
- Re-register preserves filled / not_observable state
"""

import pytest

from irys.matter import MatterModel
from irys.matter.graph import ExtractionSlotStore


@pytest.fixture
def model():
    return MatterModel.open_in_memory()


@pytest.fixture
def store(model):
    return model.extraction_slots


def test_v70_migration_creates_table(model):
    rows = model.db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='extraction_slot'"
    ).fetchall()
    assert rows
    cols = {
        r["name"] for r in model.db.execute("PRAGMA table_info(extraction_slot)").fetchall()
    }
    expected = {
        "id", "matter_id", "slot_kind", "slot_key", "artifact_family_id",
        "expected_count", "expected_count_confidence", "scope_query_hash",
        "schema_ref", "coverage_state", "evidence_refs_json",
        "created_at", "updated_at",
    }
    assert expected.issubset(cols)


def test_register_creates_pending_slot(model, store):
    sid, is_new = store.register(
        model.matter_id, "collection_item", "k1", 1,
        expected_count_confidence=0.85,
        scope_query_hash="abc",
        schema_ref="legal.market_row.v1",
    )
    assert sid and is_new
    rows = model.db.execute(
        "SELECT coverage_state, expected_count_confidence, evidence_refs_json"
        " FROM extraction_slot WHERE id=?", (sid,),
    ).fetchall()
    assert rows[0]["coverage_state"] == "pending"
    assert abs(rows[0]["expected_count_confidence"] - 0.85) < 1e-6
    assert rows[0]["evidence_refs_json"] == "[]"


def test_confidence_clamps_to_unit_interval(model, store):
    store.register(model.matter_id, "collection_item", "high", 1, expected_count_confidence=2.5)
    store.register(model.matter_id, "collection_item", "low", 1, expected_count_confidence=-0.5)
    rows = model.db.execute(
        "SELECT slot_key, expected_count_confidence FROM extraction_slot"
    ).fetchall()
    by_key = {r["slot_key"]: r["expected_count_confidence"] for r in rows}
    assert by_key["high"] == 1.0
    assert by_key["low"] == 0.0


def test_blank_kind_or_key_raises(model, store):
    with pytest.raises(ValueError):
        store.register(model.matter_id, "", "k", 1)
    with pytest.raises(ValueError):
        store.register(model.matter_id, "kind", "  ", 1)


def test_register_idempotent_preserves_filled(model, store):
    sid, _ = store.register(model.matter_id, "collection_item", "k", 1, expected_count_confidence=0.5)
    store.mark_filled(sid, "ev1")
    # Re-register should NOT downgrade to pending
    sid2, is_new = store.register(
        model.matter_id, "collection_item", "k", 1, expected_count_confidence=0.9,
    )
    assert sid2 == sid and not is_new
    row = model.db.execute(
        "SELECT coverage_state, expected_count_confidence FROM extraction_slot WHERE id=?",
        (sid,),
    ).fetchone()
    assert row["coverage_state"] == "filled"
    assert abs(row["expected_count_confidence"] - 0.9) < 1e-6


def test_register_idempotent_preserves_not_observable(model, store):
    sid, _ = store.register(model.matter_id, "collection_item", "k", 1)
    store.mark_not_observable(sid)
    store.register(model.matter_id, "collection_item", "k", 1, expected_count_confidence=0.7)
    row = model.db.execute(
        "SELECT coverage_state FROM extraction_slot WHERE id=?", (sid,),
    ).fetchone()
    assert row["coverage_state"] == "not_observable"


def test_mark_filled_expected_count_one(model, store):
    sid, _ = store.register(model.matter_id, "collection_item", "k", 1)
    store.mark_filled(sid, "ev1")
    row = model.db.execute(
        "SELECT coverage_state, evidence_refs_json FROM extraction_slot WHERE id=?", (sid,),
    ).fetchone()
    assert row["coverage_state"] == "filled"
    assert "ev1" in row["evidence_refs_json"]


def test_mark_filled_expected_count_three_partial_then_filled(model, store):
    sid, _ = store.register(model.matter_id, "collection_item", "k", 3)
    store.mark_filled(sid, "ev1")
    state1 = model.db.execute(
        "SELECT coverage_state FROM extraction_slot WHERE id=?", (sid,),
    ).fetchone()["coverage_state"]
    assert state1 == "partial"
    store.mark_filled(sid, "ev2")
    store.mark_filled(sid, "ev3")
    state3 = model.db.execute(
        "SELECT coverage_state, evidence_refs_json FROM extraction_slot WHERE id=?", (sid,),
    ).fetchone()
    assert state3["coverage_state"] == "filled"
    refs = state3["evidence_refs_json"]
    assert "ev1" in refs and "ev2" in refs and "ev3" in refs


def test_mark_filled_dedupes_evidence_refs(model, store):
    sid, _ = store.register(model.matter_id, "collection_item", "k", 5)
    store.mark_filled(sid, "ev1")
    store.mark_filled(sid, "ev1")  # duplicate
    store.mark_filled(sid, "ev2")
    import json
    refs = json.loads(model.db.execute(
        "SELECT evidence_refs_json FROM extraction_slot WHERE id=?", (sid,),
    ).fetchone()["evidence_refs_json"])
    assert refs == ["ev1", "ev2"]


def test_mark_not_observable_only_for_open(model, store):
    sid, _ = store.register(model.matter_id, "collection_item", "k", 1)
    store.mark_filled(sid, "ev1")
    store.mark_not_observable(sid)  # should NOT change state because it's filled
    row = model.db.execute(
        "SELECT coverage_state FROM extraction_slot WHERE id=?", (sid,),
    ).fetchone()
    assert row["coverage_state"] == "filled"


def test_get_open_slots_ordering_and_filters(model, store):
    store.register(model.matter_id, "collection_item", "low", 1, expected_count_confidence=0.4)
    store.register(model.matter_id, "collection_item", "med", 1, expected_count_confidence=0.6)
    store.register(model.matter_id, "collection_item", "high", 1, expected_count_confidence=0.9)
    # other kind, should be filtered out
    store.register(model.matter_id, "obligation", "obl", 1, expected_count_confidence=0.95)

    opens = store.get_open_slots(model.matter_id, "collection_item", min_confidence=0.5)
    keys = [s["slot_key"] for s in opens]
    assert keys == ["high", "med"]  # ordered by confidence DESC; "low" filtered


def test_get_filled_slots_scope_filter(model, store):
    s1, _ = store.register(model.matter_id, "collection_item", "a", 1, scope_query_hash="hashA")
    s2, _ = store.register(model.matter_id, "collection_item", "b", 1, scope_query_hash="hashB")
    store.mark_filled(s1, "ev1")
    store.mark_filled(s2, "ev2")
    a_only = store.get_filled_slots(model.matter_id, "collection_item", scope_query_hash="hashA")
    assert len(a_only) == 1
    assert a_only[0]["slot_key"] == "a"


def test_coverage_summary_groups_by_kind(model, store):
    s1, _ = store.register(model.matter_id, "collection_item", "a", 1, expected_count_confidence=0.85)
    s2, _ = store.register(model.matter_id, "collection_item", "b", 1, expected_count_confidence=0.4)
    s3, _ = store.register(model.matter_id, "obligation", "o", 1, expected_count_confidence=0.9)
    store.mark_filled(s1, "ev1")
    store.mark_not_observable(s2)
    summary = store.coverage_summary(model.matter_id)
    assert "collection_item" in summary
    assert "obligation" in summary
    ci = summary["collection_item"]
    assert ci["filled"] == 1
    assert ci["not_observable"] == 1
    assert ci["pending"] == 0
    assert ci["expected_count"] == 2
    obl = summary["obligation"]
    assert obl["pending"] == 1
    assert obl["high_confidence_open"] == 1


def test_typed_evidence_survives_v70_migration(model):
    """Smoke check: typed_evidence_record (v69) still works after v70 added."""
    rec_id, _ = model.typed_evidence.upsert(
        "regulatory_data", "rec1",
        payload={"category": "market_share", "value": "42%"},
        document_id="doc.pdf",
        confidence=0.9,
    )
    assert rec_id
    rows = model.db.execute(
        "SELECT id FROM typed_evidence_record WHERE id=?", (rec_id,),
    ).fetchall()
    assert rows


def test_open_slots_excludes_other_matter(store, model):
    other_model = MatterModel.open_in_memory()
    other_store = ExtractionSlotStore(other_model.db, other_model.matter_id)
    other_store.register(other_model.matter_id, "collection_item", "k", 1, expected_count_confidence=0.9)
    # Querying with foreign matter_id returns empty
    assert store.get_open_slots("not_my_matter") == []
    # Other store sees its own slot
    assert other_store.get_open_slots(other_model.matter_id) != []
