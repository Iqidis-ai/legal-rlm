"""Critical correctness tests for the assertion store.

Test 1 (Codex): Same proposition from two documents yields ONE assertion
row and TWO assertion_occurrence rows with different speech acts/source roles.

Test 4 (Codex): Actor alias resolution — handled separately, but dedup
behavior is tested here as the analogous invariant for assertions.
"""

import pytest
from irys.matter import (
    MatterModel, AssertionCandidate, SpeechAct, SourceRole,
    ModelLayer, AssertionKind, AssertionLinkType, BeliefState,
)


@pytest.fixture
def model():
    return MatterModel.open_in_memory()


def make_candidate(
    text: str,
    doc_id: str = "doc1",
    speech_act: SpeechAct = SpeechAct.ALLEGED,
    source_role: SourceRole = SourceRole.ADVOCACY,
) -> AssertionCandidate:
    return AssertionCandidate(
        proposition_text=text,
        model_layer=ModelLayer.RECORD,
        assertion_kind=AssertionKind.FACTUAL,
        document_id=doc_id,
        speech_act=speech_act,
        source_role=source_role,
        origin_kind=__import__("irys.matter.enums", fromlist=["OriginKind"]).OriginKind.EXTRACTED,
    )


# ---------------------------------------------------------------------------
# Test 1: Same proposition, two documents → 1 assertion, 2 occurrences
# ---------------------------------------------------------------------------

def test_same_proposition_two_docs_one_assertion_two_occurrences(model):
    text = "The contract requires payment of $50,000 by January 15, 2024."

    c1 = make_candidate(text, doc_id="complaint.pdf", speech_act=SpeechAct.ALLEGED, source_role=SourceRole.ADVOCACY)
    c2 = make_candidate(text, doc_id="contract.pdf", speech_act=SpeechAct.OPERATIVE, source_role=SourceRole.OPERATIVE)

    id1, is_new1 = model.assertions.upsert_occurrence(c1)
    id2, is_new2 = model.assertions.upsert_occurrence(c2)

    assert id1 == id2, "Same proposition must map to same assertion_id"
    assert is_new1 is True
    assert is_new2 is False

    occurrences = model.assertions.get_occurrences(id1)
    assert len(occurrences) == 2

    speech_acts = {occ["speech_act"] for occ in occurrences}
    assert SpeechAct.ALLEGED.value in speech_acts
    assert SpeechAct.OPERATIVE.value in speech_acts

    source_roles = {occ["source_role"] for occ in occurrences}
    assert SourceRole.ADVOCACY.value in source_roles
    assert SourceRole.OPERATIVE.value in source_roles


def test_different_propositions_two_assertions(model):
    c1 = make_candidate("Plaintiff alleges breach of contract.", doc_id="complaint.pdf")
    c2 = make_candidate("Defendant denies all allegations.", doc_id="answer.pdf")

    id1, _ = model.assertions.upsert_occurrence(c1)
    id2, _ = model.assertions.upsert_occurrence(c2)

    assert id1 != id2
    assert model.assertions.count() == 2


def test_normalized_dedup_whitespace(model):
    """Whitespace variations of the same proposition should dedup."""
    c1 = make_candidate("Payment was made on time.", doc_id="doc1")
    c2 = make_candidate("  Payment  was   made on time.  ", doc_id="doc2")

    id1, is_new1 = model.assertions.upsert_occurrence(c1)
    id2, is_new2 = model.assertions.upsert_occurrence(c2)

    assert id1 == id2
    assert model.assertions.count() == 1
    assert len(model.assertions.get_occurrences(id1)) == 2


def test_get_by_proposition(model):
    text = "The effective date is January 15, 2024."
    c = make_candidate(text)
    assertion_id, _ = model.assertions.upsert_occurrence(c)

    result = model.assertions.get_by_proposition(text)
    assert result is not None
    assert result.id == assertion_id
    assert result.proposition_text == text


def test_assertion_link_creates_graph(model):
    c1 = make_candidate("Payment was due on January 15.", doc_id="contract.pdf")
    c2 = make_candidate("No payment was received by January 15.", doc_id="email.pdf",
                        speech_act=SpeechAct.ALLEGED)

    id1, _ = model.assertions.upsert_occurrence(c1)
    id2, _ = model.assertions.upsert_occurrence(c2)

    # c2 attacks c1
    model.assertions.link(id2, id1, AssertionLinkType.ATTACKS)

    attackers = model.assertions.get_attackers(id1)
    assert id2 in attackers

    dependents = model.assertions.get_dependents(id2)
    # id2 attacks id1 — id1 IS a dependent of id2 because when the attacker
    # changes state (e.g. gets withdrawn), the attacked assertion must be
    # re-evaluated (it may recover).  get_dependents includes attacks/negates edges.
    assert id1 in dependents


def test_assertion_link_idempotent(model):
    c1 = make_candidate("Claim A.", doc_id="doc1")
    c2 = make_candidate("Claim B.", doc_id="doc2")
    id1, _ = model.assertions.upsert_occurrence(c1)
    id2, _ = model.assertions.upsert_occurrence(c2)

    link1 = model.assertions.link(id1, id2, AssertionLinkType.SUPPORTS)
    link2 = model.assertions.link(id1, id2, AssertionLinkType.SUPPORTS)

    assert link1 == link2  # Same link, not duplicated


def test_same_assertion_same_doc_no_duplicate_occurrence(model):
    """INSERT OR IGNORE deduplicates same (assertion_id, document_id) via UNIQUE INDEX.

    This covers the MEDIUM finding: concurrent runs ingesting the same document must not
    produce duplicate assertion_occurrence rows. The UNIQUE INDEX on
    (assertion_id, document_id) makes the second INSERT silently a no-op.
    """
    text = "The defendant failed to deliver by the deadline."
    c = make_candidate(text, doc_id="complaint.pdf")

    id1, is_new1 = model.assertions.upsert_occurrence(c)
    # Simulate concurrent / duplicate ingestion of the exact same doc
    id2, is_new2 = model.assertions.upsert_occurrence(c)

    assert id1 == id2
    assert is_new1 is True
    assert is_new2 is False
    # Second INSERT OR IGNORE must be silently dropped — only ONE occurrence row
    occurrences = model.assertions.get_occurrences(id1)
    assert len(occurrences) == 1, (
        "duplicate ingestion of same assertion from same doc must not create two occurrence rows"
    )


def test_belief_state_default_unknown(model):
    c = make_candidate("Something happened.", doc_id="doc1")
    assertion_id, _ = model.assertions.upsert_occurrence(c)
    record = model.assertions.get(assertion_id)
    assert record.belief_state == BeliefState.UNKNOWN.value


def test_set_belief_state(model):
    c = make_candidate("Payment was made.", doc_id="contract.pdf",
                       speech_act=SpeechAct.OPERATIVE, source_role=SourceRole.OPERATIVE)
    assertion_id, _ = model.assertions.upsert_occurrence(c)
    model.assertions.set_belief_state(assertion_id, BeliefState.OPERATIVE, 0.95)

    record = model.assertions.get(assertion_id)
    assert record.belief_state == BeliefState.OPERATIVE.value
    assert record.confidence == pytest.approx(0.95)
