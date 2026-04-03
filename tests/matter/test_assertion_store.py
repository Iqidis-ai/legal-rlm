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


def test_same_assertion_same_doc_same_speech_act_no_duplicate_occurrence(model):
    """INSERT OR IGNORE deduplicates same (assertion_id, document_id, speech_act) via UNIQUE INDEX.

    Concurrent runs ingesting the same document with the same speech-act classification
    must not produce duplicate assertion_occurrence rows. The UNIQUE INDEX on
    (assertion_id, document_id, speech_act) makes the second INSERT a no-op.
    """
    text = "The defendant failed to deliver by the deadline."
    c = make_candidate(text, doc_id="complaint.pdf", speech_act=SpeechAct.ALLEGED)

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


def test_same_assertion_same_doc_different_speech_acts_both_recorded(model):
    """Same assertion with different speech acts in the same doc must produce two occurrence rows.

    This catches the INTRODUCED_NEW_BUG finding: ix_occurrence_unique_doc on
    (assertion_id, document_id) was too coarse and silently dropped legitimate
    second occurrences with a different speech_act. The fix includes speech_act
    in the uniqueness key.
    """
    text = "The payment was due on January 15, 2024."
    c_alleged = make_candidate(text, doc_id="complaint.pdf", speech_act=SpeechAct.ALLEGED)
    c_admitted = make_candidate(text, doc_id="complaint.pdf", speech_act=SpeechAct.ADMITTED)

    id1, is_new1 = model.assertions.upsert_occurrence(c_alleged)
    id2, is_new2 = model.assertions.upsert_occurrence(c_admitted)

    # Same proposition → same assertion_id
    assert id1 == id2
    # Both occurrences must survive — different speech acts are semantically distinct
    occurrences = model.assertions.get_occurrences(id1)
    speech_acts_recorded = {occ["speech_act"] for occ in occurrences}
    assert SpeechAct.ALLEGED.value in speech_acts_recorded, "ALLEGED occurrence must be recorded"
    assert SpeechAct.ADMITTED.value in speech_acts_recorded, (
        "ADMITTED occurrence must not be dropped — different speech act from same doc"
    )
    assert len(occurrences) == 2, "two distinct speech acts must produce two occurrence rows"


def test_belief_state_seeded_from_speech_act(model):
    """New assertions are seeded with belief state derived from speech act (not hardcoded UNKNOWN).

    ALLEGED → BeliefState.ALLEGED; OPERATIVE → BeliefState.OPERATIVE; EXTRACTED → UNKNOWN.
    This avoids the cold-start problem where every assertion starts UNKNOWN and
    requires a separate flush_revisions() pass to acquire meaningful belief state.
    """
    # ALLEGED (default in make_candidate) → BeliefState.ALLEGED
    c_alleged = make_candidate("Something happened.", doc_id="doc1")
    aid, _ = model.assertions.upsert_occurrence(c_alleged)
    record = model.assertions.get(aid)
    assert record.belief_state == BeliefState.ALLEGED.value

    # OPERATIVE → BeliefState.OPERATIVE
    c_op = make_candidate("Payment was due.", doc_id="contract.pdf",
                           speech_act=SpeechAct.OPERATIVE, source_role=SourceRole.OPERATIVE)
    aid_op, _ = model.assertions.upsert_occurrence(c_op)
    assert model.assertions.get(aid_op).belief_state == BeliefState.OPERATIVE.value


def test_set_belief_state(model):
    c = make_candidate("Payment was made.", doc_id="contract.pdf",
                       speech_act=SpeechAct.OPERATIVE, source_role=SourceRole.OPERATIVE)
    assertion_id, _ = model.assertions.upsert_occurrence(c)
    model.assertions.set_belief_state(assertion_id, BeliefState.OPERATIVE, 0.95)

    record = model.assertions.get(assertion_id)
    assert record.belief_state == BeliefState.OPERATIVE.value


# ---------------------------------------------------------------------------
# SO-2: Truth maintenance via corroborates/supersedes traversal
# ---------------------------------------------------------------------------

def test_get_dependents_includes_corroborates(model):
    """get_dependents() must traverse corroborates edges so corroborated assertions
    are re-evaluated when the source assertion changes."""
    from irys.matter.enums import AssertionLinkType
    a1, _ = model.assertions.upsert_occurrence(
        make_candidate("Fact A.", doc_id="doc1", speech_act=SpeechAct.OPERATIVE))
    a2, _ = model.assertions.upsert_occurrence(
        make_candidate("Fact B corroborates A.", doc_id="doc2", speech_act=SpeechAct.OPERATIVE))
    # a1 --corroborates--> a2 (a1 independently confirms a2)
    model.assertions.link(a1, a2, AssertionLinkType.CORROBORATES)

    dependents = model.assertions.get_dependents(a1)
    assert a2 in dependents, "corroborates target must be in dependents for propagation"


def test_get_dependents_includes_supersedes(model):
    """get_dependents() must traverse supersedes edges so the superseded assertion
    gets re-evaluated if the superseding one is withdrawn."""
    from irys.matter.enums import AssertionLinkType
    old_a, _ = model.assertions.upsert_occurrence(
        make_candidate("Old fact.", doc_id="doc1", speech_act=SpeechAct.OPERATIVE))
    new_a, _ = model.assertions.upsert_occurrence(
        make_candidate("New fact supersedes old.", doc_id="doc2", speech_act=SpeechAct.OPERATIVE))
    # new_a --supersedes--> old_a
    model.assertions.link(new_a, old_a, AssertionLinkType.SUPERSEDES)

    # When new_a changes, old_a must be re-evaluated (maybe it can recover)
    dependents = model.assertions.get_dependents(new_a)
    assert old_a in dependents, "supersedes target must be in dependents for propagation"


def test_superseded_assertion_marked_via_belief_revision(model):
    """When a superseding assertion exists, the superseded node becomes SUPERSEDED via belief revision."""
    from irys.matter.enums import AssertionLinkType, RevisionCause
    old_a, _ = model.assertions.upsert_occurrence(
        make_candidate("Original contract date: Jan 15.", doc_id="contract.pdf",
                       speech_act=SpeechAct.OPERATIVE))
    new_a, _ = model.assertions.upsert_occurrence(
        make_candidate("Amended contract date: Feb 1.", doc_id="amendment.pdf",
                       speech_act=SpeechAct.OPERATIVE))
    model.assertions.link(new_a, old_a, AssertionLinkType.SUPERSEDES)

    results = model.apply_revision([new_a], RevisionCause.NEW_EVIDENCE)
    # old_a should now be SUPERSEDED
    old_record = model.assertions.get(old_a)
    assert old_record.belief_state == BeliefState.SUPERSEDED.value, (
        f"Superseded assertion must transition to SUPERSEDED, got {old_record.belief_state}"
    )
