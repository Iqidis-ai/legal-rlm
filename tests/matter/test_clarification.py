"""Tests for the clarification engine (SO-7, SO-3).

Verifies:
1. ClarificationStore.add_question() persists and deduplicates questions
2. MatterModel.answer_clarification() updates status through the broker
3. get_pending() / get_answered() filter by status
4. MatterModel.generate_clarifications_from_gaps() creates questions from high-materiality gaps
5. Answered clarifications are included in QueryMatterContext
6. Gap-to-issue link produces targeted impact statement (SO-7 "identifies which conclusions depend on it")
"""

import pytest
from irys.matter import MatterModel
from irys.matter.enums import GapType, IssueType


@pytest.fixture
def model():
    return MatterModel.open_in_memory()


# ---------------------------------------------------------------------------
# ClarificationStore basic contract
# ---------------------------------------------------------------------------

def test_add_question_persists(model):
    q_id = model.clarifications.add_question(
        question_text="Do you have access to the signed amendment?",
        why_it_matters="The amendment changes the payment terms.",
        expected_impact="If available, it resolves the payment dispute.",
    )
    assert q_id
    pending = model.clarifications.get_pending()
    assert len(pending) == 1
    assert pending[0]["id"] == q_id
    assert "signed amendment" in pending[0]["question_text"]


def test_add_question_is_idempotent(model):
    """Same question text → returns existing id, no duplicate."""
    q1 = model.clarifications.add_question("Same question?")
    q2 = model.clarifications.add_question("Same question?")
    assert q1 == q2
    assert model.clarifications.count_pending() == 1


def test_answer_clarification_updates_status(model):
    q_id = model.clarifications.add_question("Do you have the wire transfer record?")
    assert model.clarifications.count_pending() == 1

    model.answer_clarification(q_id, "Yes, we have a wire transfer record dated Jan 20.")

    assert model.clarifications.count_pending() == 0
    answered = model.clarifications.get_answered()
    assert len(answered) == 1
    assert answered[0]["id"] == q_id
    assert "Jan 20" in answered[0]["answer_text"]
    assert answered[0]["answered_at"] is not None


def test_direct_clarification_answer_writer_is_disabled(model):
    q_id = model.clarifications.add_question("Do you have the wire transfer record?")
    with pytest.raises(RuntimeError, match="memory broker"):
        model.clarifications.answer_question(q_id, "Bypass attempt.")


def test_get_pending_excludes_answered(model):
    q1 = model.clarifications.add_question("Question A?")
    q2 = model.clarifications.add_question("Question B?")
    model.answer_clarification(q1, "Answer A")

    pending = model.clarifications.get_pending()
    assert len(pending) == 1
    assert pending[0]["id"] == q2

    answered = model.clarifications.get_answered()
    assert len(answered) == 1
    assert answered[0]["id"] == q1


# ---------------------------------------------------------------------------
# generate_clarifications_from_gaps()
# ---------------------------------------------------------------------------

def test_generate_clarifications_creates_questions_from_high_materiality_gaps(model):
    """generate_clarifications_from_gaps() must create questions for materiality >= 0.5."""
    run_id = model.start_run("Payment dispute analysis")

    # Record a high-materiality gap
    model.gaps.record(
        gap_type=GapType.MISSING_DOCUMENT,
        description="Signed amendment No. 2 referenced but not found",
        expected_artifact="Amendment No. 2",
        materiality=0.8,
    )
    # Low-materiality gap — should be filtered out
    model.gaps.record(
        gap_type=GapType.MISSING_DOCUMENT,
        description="CC email chain not found",
        materiality=0.2,
    )

    question_ids = model.generate_clarifications_from_gaps(run_id=run_id, min_materiality=0.5)

    assert len(question_ids) == 1
    pending = model.clarifications.get_pending()
    assert len(pending) == 1
    assert "amendment" in pending[0]["question_text"].lower() or "found" in pending[0]["question_text"].lower()


def test_generate_clarifications_does_not_duplicate(model):
    """Calling generate_clarifications twice must not duplicate questions."""
    run_id = model.start_run("Test run")
    model.gaps.record(
        gap_type=GapType.MISSING_DOCUMENT,
        description="Signed contract attachment missing",
        materiality=0.7,
    )

    model.generate_clarifications_from_gaps(run_id=run_id)
    model.generate_clarifications_from_gaps(run_id=run_id)  # second call

    assert model.clarifications.count_pending() == 1


def test_generate_clarifications_top_n_limits_questions(model):
    """generate_clarifications_from_gaps(top_n=N) must create at most N questions (SO-7).

    The engine should not spam the user with clarifications for every gap.
    top_n controls how many questions are generated per run — the highest-materiality
    gaps are selected first.  Gaps beyond top_n are deferred.
    """
    run_id = model.start_run("Top-N test")

    # Record 5 gaps with different materiality scores
    for i, mat in enumerate([0.9, 0.85, 0.8, 0.75, 0.7]):
        model.gaps.record(
            gap_type=GapType.MISSING_DOCUMENT,
            description=f"Missing document priority {i+1}",
            materiality=mat,
        )

    # Generate at most 3 clarification questions
    question_ids = model.generate_clarifications_from_gaps(run_id=run_id, top_n=3, min_materiality=0.5)

    assert len(question_ids) == 3, (
        f"generate_clarifications_from_gaps(top_n=3) must create exactly 3 questions, got {len(question_ids)}"
    )
    assert model.clarifications.count_pending() == 3, (
        "Only top-3 questions must be pending — gaps beyond top_n deferred (SO-7 focused surfacing)"
    )


# ---------------------------------------------------------------------------
# QueryMatterContext includes answered clarifications
# ---------------------------------------------------------------------------

def test_query_context_includes_answered_clarifications(model):
    """Answered clarifications must appear in QueryMatterContext."""
    q_id = model.clarifications.add_question(
        question_text="Is the payment record in the repository?",
    )
    model.answer_clarification(q_id, "No, we need to request it from the client.")

    ctx = model.build_query_context()
    assert len(ctx.answered_clarifications) == 1
    assert "payment record" in ctx.answered_clarifications[0]["question_text"].lower()


def test_query_context_excludes_profile_stale_answered_clarifications(model):
    q_id = model.clarifications.add_question(
        question_text="Is the payment record in the repository?",
    )
    model.answer_clarification(q_id, "No, request it from the client.")
    assert len(model.build_query_context().answered_clarifications) == 1

    default_hash = model.memory_broker.default_legal_profile_hash()
    model.memory_broker.record_profile_mapping(
        source_domain_profile_id="legal",
        source_domain_profile_version=1,
        target_domain_profile_id="legal",
        target_domain_profile_version=1,
        source_mapping_hash=default_hash,
        target_mapping_hash="sha256:rotated-legal-profile",
        target_kind="clarification",
        target_namespace="clarifications",
        compatibility_status="identity",
    )

    ctx = model.build_query_context()

    assert ctx.answered_clarifications == []


def test_query_context_includes_current_nonlegal_profile_clarification(model):
    model.memory_broker.upsert_domain_profile(
        profile_id="finance",
        profile_version=1,
        profile_kind="finance",
        profile_json='{"metric":"revenue"}',
        mapping_hash="sha256:finance",
    )
    model.memory_broker.record_profile_mapping(
        source_domain_profile_id="finance",
        source_domain_profile_version=1,
        target_domain_profile_id="finance",
        target_domain_profile_version=1,
        source_mapping_hash="sha256:finance",
        target_mapping_hash="sha256:finance",
        target_kind="clarification",
        target_namespace="clarifications",
        compatibility_status="identity",
    )
    q_id = model.clarifications.add_question(
        question_text="Is revenue recognized ratably?",
    )
    model.answer_clarification(
        q_id,
        "Yes.",
        domain_profile_id="finance",
        domain_profile_version=1,
    )

    ctx = model.build_query_context()

    assert len(ctx.answered_clarifications) == 1
    assert "revenue" in ctx.answered_clarifications[0]["question_text"].lower()


def test_query_context_excludes_tainted_answered_clarifications(model):
    """Broker taint policy quarantines dirty context rows."""
    q_id = model.clarifications.add_question(
        question_text="Is the payment record in the repository?",
    )
    model.answer_clarification(q_id, "No, request it from the client.")
    model.memory_broker.record_object_taint(
        target_kind="clarification",
        target_id=q_id,
        taint_class="unknown_taint",
        derivation_reason="test quarantine",
    )

    ctx = model.build_query_context()

    assert ctx.answered_clarifications == []


def test_query_context_excludes_pending_clarifications(model):
    """Pending (unanswered) questions must NOT appear in QueryMatterContext."""
    model.clarifications.add_question("Is the contract signed?")

    ctx = model.build_query_context()
    assert len(ctx.answered_clarifications) == 0


# ---------------------------------------------------------------------------
# SO-7: Gap-to-issue link produces targeted impact statement
# ---------------------------------------------------------------------------

def test_gap_linked_to_issue_produces_targeted_impact_statement(model):
    """When a gap is linked to an issue, the generated clarification must include
    a 'tracked issue' impact statement — proving the system identified which
    conclusions depend on the missing document (SO-7 'identifies which conclusions').
    """
    run_id = model.start_run("SO-7 test")
    issue_id, _ = model.issues.upsert_issue("Payment obligation breach", IssueType.CLAIM, materiality=0.9)

    # Record a gap linked to the issue
    model.gaps.record(
        gap_type=GapType.MISSING_DOCUMENT,
        description="Signed Amendment No. 2 referenced in §4.2 but absent from repository",
        expected_artifact="Amendment No. 2",
        materiality=0.9,
        affected_type="issue",
        affected_id=issue_id,
    )

    question_ids = model.generate_clarifications_from_gaps(run_id=run_id, min_materiality=0.5)
    assert len(question_ids) == 1

    pending = model.clarifications.get_pending()
    assert len(pending) == 1
    q = pending[0]

    # The expected_impact must mention the tracked issue — this proves the link is used
    assert q.get("expected_impact"), "expected_impact must be non-empty"
    impact = q["expected_impact"].lower()
    assert "issue" in impact, (
        f"expected_impact must reference the linked issue. Got: {q['expected_impact']!r}"
    )


def test_gap_without_link_produces_generic_impact_statement(model):
    """A gap with no linked issue/assertion should still get a generic (but present) impact."""
    run_id = model.start_run("Generic gap test")
    model.gaps.record(
        gap_type=GapType.MISSING_DOCUMENT,
        description="Email attachment referenced but not provided",
        materiality=0.6,
    )
    question_ids = model.generate_clarifications_from_gaps(run_id=run_id, min_materiality=0.5)
    assert len(question_ids) == 1

    q = model.clarifications.get_pending()[0]
    assert q.get("expected_impact"), "expected_impact must always be non-empty"
    # Generic impact should mention materiality level
    assert "materiality" in q["expected_impact"].lower() or "medium" in q["expected_impact"].lower()


def test_proof_gap_clarification_asks_for_evidence_not_document(model):
    """A MISSING_ISSUE_PREDICATE gap must generate an evidence-request question (not document-request).

    Proof gaps arise when an issue has zero supporting assertions.
    The clarification should ask for evidence/testimony, not specifically a document.
    This verifies that the gap_type routing in generate_clarifications_from_gaps()
    produces the right question form (SO-7).
    """
    run_id = model.start_run("Proof gap test")
    model.gaps.record(
        gap_type=GapType.MISSING_ISSUE_PREDICATE,
        description="Causation: no evidence linking the breach to the claimed damages",
        materiality=0.8,
    )

    question_ids = model.generate_clarifications_from_gaps(run_id=run_id, min_materiality=0.5)
    assert len(question_ids) == 1

    q = model.clarifications.get_pending()[0]
    text = q["question_text"].lower()
    # Proof gap question should ask for evidence broadly, not specifically a document
    assert "evidence" in text or "testimony" in text or "documents" in text, (
        f"Proof gap question must ask for evidence broadly; got: {q['question_text']!r}"
    )
    # Must not ask "we could not find the following in the repository" (that's for missing docs)
    assert "we could not find" not in text, (
        "Proof gap question must not use the missing-document template"
    )


# ---------------------------------------------------------------------------
# SO-3: get_new_answered_clarifications() mid-run steering (de-dup + time filter)
# ---------------------------------------------------------------------------

def test_get_new_answered_clarifications_returns_post_run_answers(model):
    """get_new_answered_clarifications() must return questions answered AFTER run started (SO-3).

    The adapter tracks _run_started_at at construction time.  Only answers recorded
    after that timestamp should appear — so the engine can pick up user responses
    that arrived mid-run without re-processing answers from earlier sessions.
    """
    from irys.matter.runtime import MatterRuntimeAdapter

    # Create the run/adapter FIRST — this sets _run_started_at to now
    run_id = model.start_run("mid-run steering test")
    adapter = MatterRuntimeAdapter(model, run_id)

    # Add and answer a question AFTER adapter creation — must appear
    q_id = model.clarifications.add_question("Do you have the signed amendment?")
    model.answer_clarification(q_id, "Yes, it's in the contract folder.")

    new_answers = adapter.get_new_answered_clarifications()
    assert len(new_answers) == 1
    assert new_answers[0]["id"] == q_id


def test_get_new_answered_clarifications_deduplicates_across_calls(model):
    """get_new_answered_clarifications() must return each answer only once (SO-3).

    The adapter tracks _injected_clarification_ids to de-dup across multiple loop
    iterations.  A second call must not re-return the same clarification as a new lead.
    """
    from irys.matter.runtime import MatterRuntimeAdapter

    run_id = model.start_run("dedup test")
    adapter = MatterRuntimeAdapter(model, run_id)

    q_id = model.clarifications.add_question("Is the payment record in the repository?")
    model.answer_clarification(q_id, "No, we need to request it.")

    # First call: should return the answer
    first = adapter.get_new_answered_clarifications()
    assert len(first) == 1

    # Second call: must be empty — already injected
    second = adapter.get_new_answered_clarifications()
    assert len(second) == 0, (
        "get_new_answered_clarifications() must de-dup: same answer must not be returned twice"
    )


def test_get_new_answered_clarifications_pre_run_answers_excluded(model):
    """Answers recorded BEFORE the run started must not appear in get_new_answered_clarifications.

    The _run_started_at filter ensures answers from prior sessions or before this
    run began are not re-injected as 'new' steering inputs.  Simulated by setting
    _run_started_at to a future time so all answers appear to be 'before' the run.
    """
    from irys.matter.runtime import MatterRuntimeAdapter

    # Answer a question first
    q_id = model.clarifications.add_question("Is Exhibit B signed?")
    model.answer_clarification(q_id, "Yes.")

    # Now start run
    run_id = model.start_run("pre-run filter test")
    adapter = MatterRuntimeAdapter(model, run_id)

    # Override _run_started_at to simulate "run started just now, after the answer"
    # (which is the normal case — answer was before, run started after)
    # To make the filter clear, set it to a far-future time so the existing answer
    # appears to have been answered "in the past" relative to the run start.
    import datetime
    future_iso = "2099-01-01T00:00:00+00:00"
    adapter._run_started_at = future_iso

    answers = adapter.get_new_answered_clarifications()
    assert len(answers) == 0, (
        "Answers recorded before _run_started_at must not appear in get_new_answered_clarifications"
    )


def test_gap_linked_to_assertion_produces_assertion_impact_statement(model):
    """A gap linked to an assertion must mention the assertion in the impact (SO-7).

    When an absent document is cited by an existing assertion, the clarification
    must say it 'may corroborate, contradict, or supersede that assertion' —
    not a generic impact.
    """
    from irys.matter import AssertionCandidate, SpeechAct, SourceRole, ModelLayer, AssertionKind
    from irys.matter.enums import OriginKind

    run_id = model.start_run("Assertion gap test")

    # Record an assertion that cites a missing document
    cand = AssertionCandidate(
        proposition_text="The payment was made on January 15, per wire transfer records.",
        model_layer=ModelLayer.RECORD,
        assertion_kind=AssertionKind.FACTUAL,
        document_id="email.pdf",
        speech_act=SpeechAct.ALLEGED,
        source_role=SourceRole.INFORMAL,
        origin_kind=OriginKind.EXTRACTED,
    )
    assertion_id, _ = model.assertions.upsert_occurrence(cand)

    # Record a gap linked to that assertion (the wire transfer was referenced but absent)
    model.gaps.record(
        gap_type=GapType.MISSING_DOCUMENT,
        description="Wire transfer confirmation referenced in email but not in repository",
        expected_artifact="Wire transfer Jan 15",
        materiality=0.75,
        affected_type="assertion",
        affected_id=assertion_id,
    )

    question_ids = model.generate_clarifications_from_gaps(run_id=run_id, min_materiality=0.5)
    assert len(question_ids) == 1

    q = model.clarifications.get_pending()[0]
    impact = q["expected_impact"].lower()
    # The impact for assertion-linked gaps mentions the assertion
    assert "assertion" in impact or "fact" in impact or "corroborate" in impact or "contradict" in impact, (
        f"Impact for assertion-linked gap must mention the assertion. Got: {q['expected_impact']!r}"
    )
