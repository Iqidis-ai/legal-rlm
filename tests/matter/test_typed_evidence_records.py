from irys.matter import (
    CrossReferenceRecord,
    DefinedTermRecord,
    EvidenceSpanRef,
    QuantFactRecord,
    SignatureBlockRecord,
)
from irys.matter.enums import VerificationTargetKind


def test_defined_term_record_requires_term_definition_and_section():
    span = EvidenceSpanRef(
        document_id="Lion.pdf",
        span_id="span-1",
        section_ref="Article 1",
        excerpt='"Third Party Supplier" means...',
    )
    record = DefinedTermRecord(
        document_id="Lion.pdf",
        term="Third Party Supplier",
        definition_text="A third-party seller of product.",
        first_defined_in="Section 5.3(b)",
        source_span=span,
        confidence=0.91,
    )

    assert record.target_kind == VerificationTargetKind.DEFINED_TERM
    assert record.has_required_fields is True
    assert record.identity_key() == record.identity_key()
    prompt_line = record.to_prompt_line()
    assert "defined_term=Third Party Supplier" in prompt_line
    assert "first_defined_in=Section 5.3(b)" in prompt_line
    assert "source=Lion.pdf | Article 1" in prompt_line


def test_signature_block_record_requires_name_title_and_entity():
    incomplete = SignatureBlockRecord(
        document_id="BSR.pdf",
        entity_name="J. Aron",
        signatory_name="",
        title="unknown",
    )
    complete = SignatureBlockRecord(
        document_id="BSR.pdf",
        entity_name="J. Aron",
        signatory_name="Simon Collier",
        capacity="Attorney-in-fact",
        source_span=EvidenceSpanRef(
            document_id="BSR.pdf",
            page_start=86,
            excerpt="By: Simon Collier, Attorney-in-fact",
        ),
    )

    assert incomplete.has_name_title_entity is False
    assert complete.target_kind == VerificationTargetKind.SIGNATURE_BLOCK
    assert complete.has_name_title_entity is True
    assert complete.to_prompt_row()["name"] == "Simon Collier"
    assert "p. 86" in complete.to_prompt_row()["source"]


def test_cross_reference_record_requires_source_span():
    without_span = CrossReferenceRecord(
        source_document_id="ARKS.pdf",
        target_label="Step-Out Inventory Sales Agreement",
        reference_text="Step-Out Inventory Sales Agreement",
    )
    with_span = CrossReferenceRecord(
        source_document_id="ARKS.pdf",
        target_label="Step-Out Inventory Sales Agreement",
        normalized_target="step-out inventory sales agreement",
        reference_text="as set forth in the Step-Out Inventory Sales Agreement",
        source_span=EvidenceSpanRef(
            document_id="ARKS.pdf",
            section_ref="Schedule HH",
            excerpt="Step-Out Inventory Sales Agreement",
        ),
    )

    assert without_span.has_reference_span is False
    assert with_span.target_kind == VerificationTargetKind.CROSS_REFERENCE
    assert with_span.has_reference_span is True
    assert "reference_target=Step-Out Inventory Sales Agreement" in with_span.to_prompt_line()


def test_quant_fact_record_compares_only_same_subject_metric_and_period():
    purchase_price = QuantFactRecord(
        subject_key="whitfield_contract",
        metric_key="purchase_price",
        value=72500,
        currency="USD",
        document_id="Contract.pdf",
    )
    repeated_purchase_price = QuantFactRecord(
        subject_key="whitfield_contract",
        metric_key="purchase_price",
        value=72500,
        currency="USD",
        document_id="Plaintiffs MSJ.pdf",
    )
    paid_amount = QuantFactRecord(
        subject_key="whitfield_contract",
        metric_key="paid_amount",
        value=48000,
        currency="USD",
        document_id="Payment ledger.pdf",
    )
    different_subject_payment = QuantFactRecord(
        subject_key="improvement_claim",
        metric_key="paid_amount",
        value=57842.16,
        currency="USD",
        document_id="Damages brief.pdf",
    )

    assert purchase_price.target_kind == VerificationTargetKind.QUANT_FACT
    assert purchase_price.comparable_to(repeated_purchase_price) is True
    assert purchase_price.comparable_to(paid_amount) is False
    assert paid_amount.comparable_to(different_subject_payment) is False
    assert "subject=whitfield_contract" in paid_amount.to_prompt_line()
