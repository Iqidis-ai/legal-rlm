"""Tests for the clarification engine (SO-7, SO-3).

Verifies:
1. ClarificationStore.add_question() persists and deduplicates questions
2. ClarificationStore.answer_question() updates status and records answer
3. get_pending() / get_answered() filter by status
4. MatterModel.generate_clarifications_from_gaps() creates questions from high-materiality gaps
5. Answered clarifications are included in QueryMatterContext
"""

import pytest
from irys.matter import MatterModel, ClarificationStore
from irys.matter.enums import GapType


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


def test_answer_question_updates_status(model):
    q_id = model.clarifications.add_question("Do you have the wire transfer record?")
    assert model.clarifications.count_pending() == 1

    model.clarifications.answer_question(q_id, "Yes, we have a wire transfer record dated Jan 20.")

    assert model.clarifications.count_pending() == 0
    answered = model.clarifications.get_answered()
    assert len(answered) == 1
    assert answered[0]["id"] == q_id
    assert "Jan 20" in answered[0]["answer_text"]
    assert answered[0]["answered_at"] is not None


def test_get_pending_excludes_answered(model):
    q1 = model.clarifications.add_question("Question A?")
    q2 = model.clarifications.add_question("Question B?")
    model.clarifications.answer_question(q1, "Answer A")

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


# ---------------------------------------------------------------------------
# QueryMatterContext includes answered clarifications
# ---------------------------------------------------------------------------

def test_query_context_includes_answered_clarifications(model):
    """Answered clarifications must appear in QueryMatterContext."""
    q_id = model.clarifications.add_question(
        question_text="Is the payment record in the repository?",
    )
    model.clarifications.answer_question(q_id, "No, we need to request it from the client.")

    ctx = model.build_query_context()
    assert len(ctx.answered_clarifications) == 1
    assert "payment record" in ctx.answered_clarifications[0]["question_text"].lower()


def test_query_context_excludes_pending_clarifications(model):
    """Pending (unanswered) questions must NOT appear in QueryMatterContext."""
    model.clarifications.add_question("Is the contract signed?")

    ctx = model.build_query_context()
    assert len(ctx.answered_clarifications) == 0
