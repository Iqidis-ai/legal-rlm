"""Tests for ActorStore and source-role inference.

Test 4 (Codex): Actor alias resolution — the same real-world actor under
multiple name variations maps to ONE actor row.
"""

import pytest
from irys.matter import MatterModel
from irys.matter.enums import SourceRole
from irys.matter.runtime import infer_source_role


@pytest.fixture
def model():
    return MatterModel.open_in_memory()


# ---------------------------------------------------------------------------
# ActorStore: dedup invariant
# ---------------------------------------------------------------------------

def test_same_actor_one_row(model):
    """Test 4 (Codex): same actor, created twice → 1 actor row."""
    id1, is_new1 = model.actors.upsert_actor("Acme Corporation", actor_type="company")
    id2, is_new2 = model.actors.upsert_actor("Acme Corporation", actor_type="company")

    assert id1 == id2
    assert is_new1 is True
    assert is_new2 is False
    assert model.actors.count() == 1


def test_different_actors_different_rows(model):
    id1, _ = model.actors.upsert_actor("John Smith", actor_type="person")
    id2, _ = model.actors.upsert_actor("Jane Doe", actor_type="person")
    assert id1 != id2
    assert model.actors.count() == 2


def test_whitespace_normalization_deduplication(model):
    """Whitespace-variant names of the same actor should deduplicate."""
    id1, is_new1 = model.actors.upsert_actor("Acme Corp", actor_type="company")
    id2, is_new2 = model.actors.upsert_actor("  Acme  Corp  ", actor_type="company")
    assert id1 == id2
    assert model.actors.count() == 1


def test_alias_resolution(model):
    """Test 4 (Codex): alias lookup returns the canonical actor."""
    actor_id, _ = model.actors.upsert_actor("TechServices Inc.", actor_type="company")
    model.actors.add_alias(actor_id, "TechServices")
    model.actors.add_alias(actor_id, "TSI")
    model.actors.add_alias(actor_id, "Tech Services Incorporated")

    assert model.actors.get_by_alias("TechServices") == actor_id
    assert model.actors.get_by_alias("TSI") == actor_id
    assert model.actors.get_by_alias("Tech Services Incorporated") == actor_id
    assert model.actors.get_by_alias("TechServices Inc.") == actor_id  # canonical name = alias


def test_alias_case_insensitive(model):
    actor_id, _ = model.actors.upsert_actor("John Smith", actor_type="person")
    model.actors.add_alias(actor_id, "Johnny")

    assert model.actors.get_by_alias("johnny") == actor_id
    assert model.actors.get_by_alias("JOHNNY") == actor_id
    assert model.actors.get_by_alias("Johnny") == actor_id


def test_alias_idempotent(model):
    """Adding the same alias twice must not create duplicate rows."""
    actor_id, _ = model.actors.upsert_actor("Acme Corp", actor_type="company")
    alias_id1 = model.actors.add_alias(actor_id, "Acme")
    alias_id2 = model.actors.add_alias(actor_id, "Acme")
    assert alias_id1 == alias_id2


def test_get_by_alias_not_found(model):
    assert model.actors.get_by_alias("Nonexistent Party") is None


def test_list_actors(model):
    model.actors.upsert_actor("Alice", actor_type="person")
    model.actors.upsert_actor("Bob", actor_type="person")
    model.actors.upsert_actor("Acme Corp", actor_type="company")

    actors = model.actors.list_actors()
    assert len(actors) == 3
    names = {a["canonical_name"] for a in actors}
    assert "Alice" in names
    assert "Bob" in names
    assert "Acme Corp" in names


def test_get_aliases_returns_all(model):
    actor_id, _ = model.actors.upsert_actor("Jane Smith", actor_type="person")
    model.actors.add_alias(actor_id, "Jane")
    model.actors.add_alias(actor_id, "J. Smith")

    aliases = model.actors.get_aliases(actor_id)
    # Should include canonical alias + added aliases
    assert "jane smith" in aliases  # normalized canonical
    assert "jane" in aliases
    assert "j. smith" in aliases


def test_actor_home_side(model):
    actor_id, _ = model.actors.upsert_actor(
        "Plaintiff Corp",
        actor_type="company",
        home_side="plaintiff",
    )
    actors = model.actors.list_actors()
    assert actors[0]["home_side"] == "plaintiff"


# ---------------------------------------------------------------------------
# Source-role inference from filename
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename,expected_role", [
    # Operative
    ("Service_Agreement_v3.pdf", SourceRole.OPERATIVE),
    ("MSA_2023.pdf", SourceRole.OPERATIVE),
    ("NDA_executed.docx", SourceRole.OPERATIVE),
    ("Amendment_1_to_Contract.pdf", SourceRole.OPERATIVE),
    # Procedural (motions, discovery, depositions — neutral court filings)
    ("Motion_to_Dismiss.pdf", SourceRole.PROCEDURAL),
    ("Deposition_Smith_2024.pdf", SourceRole.PROCEDURAL),
    # Advocacy (party-authored: complaints, answers, briefs, demands)
    ("Complaint_filed_2024.pdf", SourceRole.ADVOCACY),
    ("Plaintiff_Brief.pdf", SourceRole.ADVOCACY),
    ("Demand_Letter_Jan15.pdf", SourceRole.ADVOCACY),
    # Informal
    ("Email_thread_March.eml", SourceRole.INFORMAL),
    ("Slack_messages_export.json", SourceRole.INFORMAL),
    # Draft
    ("Contract_Draft_v2.docx", SourceRole.DRAFT),
    ("Redlined_Agreement.docx", SourceRole.DRAFT),
    # Post-hoc explanatory
    ("Expert_Report_damages.pdf", SourceRole.POST_HOC_EXPLANATORY),
    ("Audit_findings.pdf", SourceRole.POST_HOC_EXPLANATORY),
    # Unknown (no match)
    ("document_001.pdf", SourceRole.UNKNOWN),
    ("scan_0042.tiff", SourceRole.UNKNOWN),
])
def test_infer_source_role(filename, expected_role):
    assert infer_source_role(filename) == expected_role


def test_infer_source_role_path_uses_filename_only():
    """Full paths should resolve the role from the filename component."""
    role = infer_source_role("/matters/acme_v_techservices/contracts/MSA_2023.pdf")
    assert role == SourceRole.OPERATIVE


def test_adapter_auto_infers_source_role():
    """record_fact with SourceRole.UNKNOWN auto-infers from document_id."""
    from irys.matter.runtime import MatterRuntimeAdapter

    model = MatterModel.open_in_memory()
    run_id = model.start_run("Source role test")
    adapter = MatterRuntimeAdapter(model, run_id)

    fact_id = adapter.record_fact(
        "The contract requires 30 days written notice.",
        document_id="Service_Agreement_signed.pdf",
        # source_role not provided — should auto-infer OPERATIVE
    )

    occ = model.assertions.get_occurrences(fact_id)
    assert len(occ) == 1
    assert occ[0]["source_role"] == SourceRole.OPERATIVE.value


def test_so5_speech_act_elevation_complaint_vs_contract():
    """SO-5 core test: complaint → ALLEGED; contract → OPERATIVE.

    The same proposition ("Defendant owes $500,000") asserted in a complaint
    and in a signed contract must be stored with different speech acts.
    Advocacy documents auto-elevate to ALLEGED; operative documents auto-elevate
    to OPERATIVE. This distinction must survive in the assertion occurrence rows.
    """
    from irys.matter.runtime import MatterRuntimeAdapter
    from irys.matter.enums import SpeechAct

    model = MatterModel.open_in_memory()
    run_id = model.start_run("SO-5 test")
    adapter = MatterRuntimeAdapter(model, run_id)

    # Same proposition text, different document sources
    proposition = "Defendant owes $500,000 in unpaid invoices"

    aid_complaint = adapter.record_fact(
        proposition,
        document_id="plaintiff_complaint.pdf",  # ADVOCACY → ALLEGED
    )
    aid_contract = adapter.record_fact(
        proposition,
        document_id="Master_Service_Agreement.pdf",  # OPERATIVE → OPERATIVE
    )

    # These are the SAME canonical assertion (same proposition_text hash)
    assert aid_complaint == aid_contract, (
        "Same proposition should map to one canonical assertion regardless of source"
    )

    # But the OCCURRENCES must differ in speech_act
    occs = model.assertions.get_occurrences(aid_complaint)
    assert len(occs) == 2, f"Expected 2 occurrences, got {len(occs)}"

    speech_acts = {occ["speech_act"] for occ in occs}
    assert SpeechAct.ALLEGED.value in speech_acts, (
        f"Complaint occurrence should be ALLEGED; got speech_acts={speech_acts}"
    )
    assert SpeechAct.OPERATIVE.value in speech_acts, (
        f"Contract occurrence should be OPERATIVE; got speech_acts={speech_acts}"
    )
