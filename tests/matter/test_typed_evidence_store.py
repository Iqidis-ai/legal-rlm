from irys.matter import (
    AbsenceStatus,
    AbsenceStatusRecord,
    CrossReferenceRecord,
    DefinedTermRecord,
    EvidenceSpanRef,
    MatterModel,
    SignatureBlockRecord,
)
from irys.matter.enums import VerificationTargetKind


def test_typed_evidence_store_persists_defined_terms_with_verification():
    model = MatterModel.open_in_memory()
    span = EvidenceSpanRef(
        document_id="lion.pdf",
        span_id="span-1",
        section_ref="Article 1",
        excerpt="Three Month LIBOR means...",
    )
    record = DefinedTermRecord(
        document_id="lion.pdf",
        term="Three Month LIBOR",
        definition_text="The rate determined by reference to LIBOR.",
        first_defined_in="Article 1",
        source_span=span,
        confidence=0.91,
    )

    record_id, is_new = model.typed_evidence.upsert_record(record)

    assert is_new is True
    stored = model.typed_evidence.get("defined_term", record.identity_key())
    assert stored["id"] == record_id
    assert stored["record_kind"] == "defined_term"
    assert stored["label"] == "Three Month LIBOR"
    assert stored["document_id"] == "lion.pdf"
    assert stored["span_id"] == "span-1"
    assert stored["payload"]["term"] == "Three Month LIBOR"
    assert stored["payload"]["target_kind"] == "defined_term"
    assert model.typed_evidence.count_by_kind("defined_term") == 1

    verification = model.verification.get(VerificationTargetKind.DEFINED_TERM, record_id)
    assert verification is not None
    assert verification["status"] == "candidate"


def test_typed_evidence_store_updates_same_key_without_new_row():
    model = MatterModel.open_in_memory()
    record = DefinedTermRecord(
        document_id="arks.pdf",
        term="Trade Sheet",
        definition_text="Initial wording.",
        first_defined_in="Schedule Q",
        confidence=0.5,
    )
    record_id, is_new = model.typed_evidence.upsert_record(record)
    assert is_new is True

    updated = DefinedTermRecord(
        document_id="arks.pdf",
        term="Trade Sheet",
        definition_text="Updated wording.",
        first_defined_in="Schedule Q",
        confidence=0.75,
    )
    updated_id, updated_is_new = model.typed_evidence.upsert_record(updated)

    assert updated_id == record_id
    assert updated_is_new is False
    stored = model.typed_evidence.get_by_id(record_id)
    assert stored["payload"]["definition_text"] == "Updated wording."
    assert stored["confidence"] == 0.75
    assert model.typed_evidence.count_by_kind("defined_term") == 1


def test_typed_evidence_store_handles_signatures_cross_refs_and_absence():
    model = MatterModel.open_in_memory()
    sig = SignatureBlockRecord(
        document_id="bsr.pdf",
        entity_name="Delek Marketing & Supply, LLC",
        signatory_name="Frederec Green",
        title="Executive Vice President",
        confidence=0.88,
    )
    sig_id, _ = model.typed_evidence.upsert_record(sig)

    ref = CrossReferenceRecord(
        source_document_id="lion.pdf",
        target_label="Step-Out Inventory Sales Agreement",
        reference_text="The Step-Out Inventory Sales Agreement remains in effect.",
        normalized_target="step_out_inventory_sales_agreement",
        confidence=0.84,
    )
    ref_id, _ = model.typed_evidence.upsert_record(ref)

    absence = AbsenceStatusRecord(
        target="Section 19.7 ESG covenants",
        status=AbsenceStatus.FALSE_PREMISE_LIKELY,
        searched_documents=3,
        searched_terms=("Section 19.7", "ESG"),
        confidence=0.8,
    )
    absence_id, _ = model.typed_evidence.upsert_record(
        absence,
        record_kind=VerificationTargetKind.ABSENCE_STATUS,
        record_key="lion:section-19.7:esg",
    )

    assert model.typed_evidence.get_by_id(sig_id)["payload"]["signatory_name"] == "Frederec Green"
    assert model.typed_evidence.get_by_id(ref_id)["payload"]["normalized_target"] == "step_out_inventory_sales_agreement"
    assert model.typed_evidence.get_by_id(absence_id)["payload"]["status"] == "false_premise_likely"
    assert model.verification.get(VerificationTargetKind.SIGNATURE_BLOCK, sig_id)
    assert model.verification.get(VerificationTargetKind.CROSS_REFERENCE, ref_id)
    assert model.verification.get(VerificationTargetKind.ABSENCE_STATUS, absence_id)


def test_typed_evidence_unknown_kind_still_gets_artifact_verification():
    model = MatterModel.open_in_memory()

    record_id, is_new = model.typed_evidence.upsert(
        "domain_specific_object",
        "custom-key",
        {"field": "value"},
        label="Custom object",
        confidence=0.4,
    )

    assert is_new is True
    assert model.typed_evidence.get("domain_specific_object", "custom-key")["payload"] == {"field": "value"}
    assert model.verification.get(VerificationTargetKind.ARTIFACT, record_id)
