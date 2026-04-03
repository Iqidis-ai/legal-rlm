"""Tests for model-layer enforcement in record_fact() — spec §24.

Verifies that RuntimeMatterAdapter._infer_model_layer() and record_fact()
correctly route assertions to the appropriate reasoning layer:
  NORMATIVE assertion_kind → LEGAL layer
  INFERRED origin_kind     → REALITY layer
  Explicit layer param     → always honoured (never overridden)
  All other assertions     → RECORD layer (default)

This enforces the 5-layer separation the spec requires so cross-layer
policy filters operate on correctly typed, smaller subsets.
"""

import pytest
from irys.matter import (
    MatterModel, SpeechAct, SourceRole, AssertionKind, ModelLayer,
)
from irys.matter.enums import OriginKind
from irys.matter.runtime import MatterRuntimeAdapter as RuntimeMatterAdapter


@pytest.fixture
def model():
    return MatterModel.open_in_memory()


@pytest.fixture
def adapter(model):
    from irys.matter.runtime import MatterRuntimeAdapter
    run_id = model.start_run("test query")
    return MatterRuntimeAdapter(model, run_id)


# ---------------------------------------------------------------------------
# _infer_model_layer() static method
# ---------------------------------------------------------------------------

def test_infer_layer_normative_to_legal():
    layer = RuntimeMatterAdapter._infer_model_layer(
        AssertionKind.NORMATIVE, OriginKind.EXTRACTED, ModelLayer.RECORD
    )
    assert layer == ModelLayer.LEGAL


def test_infer_layer_inferred_to_reality():
    layer = RuntimeMatterAdapter._infer_model_layer(
        AssertionKind.FACTUAL, OriginKind.INFERRED, ModelLayer.RECORD
    )
    assert layer == ModelLayer.REALITY


def test_infer_layer_factual_extracted_stays_record():
    layer = RuntimeMatterAdapter._infer_model_layer(
        AssertionKind.FACTUAL, OriginKind.EXTRACTED, ModelLayer.RECORD
    )
    assert layer == ModelLayer.RECORD


def test_infer_layer_quantitative_stays_record():
    layer = RuntimeMatterAdapter._infer_model_layer(
        AssertionKind.QUANTITATIVE, OriginKind.EXTRACTED, ModelLayer.RECORD
    )
    assert layer == ModelLayer.RECORD


def test_infer_layer_explicit_never_overridden():
    """Caller-specified layers are never touched by auto-inference."""
    layer = RuntimeMatterAdapter._infer_model_layer(
        AssertionKind.NORMATIVE, OriginKind.EXTRACTED, ModelLayer.PROOF
    )
    assert layer == ModelLayer.PROOF  # explicit PROOF, not auto-upgraded to LEGAL


def test_infer_layer_explicit_record_with_normative_upgrades():
    """Explicit RECORD + NORMATIVE kind → upgrades to LEGAL (caller used default)."""
    layer = RuntimeMatterAdapter._infer_model_layer(
        AssertionKind.NORMATIVE, OriginKind.EXTRACTED, ModelLayer.RECORD
    )
    assert layer == ModelLayer.LEGAL


# ---------------------------------------------------------------------------
# record_fact() integration
# ---------------------------------------------------------------------------

def test_record_fact_normative_stored_as_legal_layer(model, adapter):
    aid = adapter.record_fact(
        "Party A owes a duty of care",
        document_id="statute.pdf",
        assertion_kind=AssertionKind.NORMATIVE,
    )
    row = model.assertions.get(aid)
    assert row.model_layer == ModelLayer.LEGAL.value


def test_record_fact_inferred_stored_as_reality_layer(model, adapter):
    aid = adapter.record_fact(
        "Party A likely breached the contract",
        document_id="doc.pdf",
        origin_kind=OriginKind.INFERRED,
    )
    row = model.assertions.get(aid)
    assert row.model_layer == ModelLayer.REALITY.value


def test_record_fact_factual_extracted_stays_record(model, adapter):
    aid = adapter.record_fact(
        "Invoice dated June 1 in the amount of $10,000",
        document_id="invoice.pdf",
        assertion_kind=AssertionKind.FACTUAL,
    )
    row = model.assertions.get(aid)
    assert row.model_layer == ModelLayer.RECORD.value


def test_record_fact_explicit_layer_not_overridden(model, adapter):
    """Caller explicitly passes ModelLayer.PROOF — must not be overridden."""
    aid = adapter.record_fact(
        "Plaintiff can prove breach beyond doubt",
        document_id="doc.pdf",
        model_layer=ModelLayer.PROOF,
    )
    row = model.assertions.get(aid)
    assert row.model_layer == ModelLayer.PROOF.value


def test_record_fact_origin_kind_stored_in_occurrence(model, adapter):
    """origin_kind=INFERRED is persisted in assertion_occurrence."""
    aid = adapter.record_fact(
        "Contract was repudiated by conduct",
        document_id="doc.pdf",
        origin_kind=OriginKind.INFERRED,
    )
    occurrences = model.assertions.get_occurrences(aid)
    assert len(occurrences) >= 1
    assert any(o.get("origin_kind") == OriginKind.INFERRED.value for o in occurrences)
