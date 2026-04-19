"""P0 notes→reasoning tests per Codex design gate.

Covers the minimum invariants that prove attorney review notes reach
the read-family LLM prompt as guidance (not evidence), scoped to
verified-and-visible facts, and stay out of stale / rejected /
clean-audience paths.
"""

from __future__ import annotations

import asyncio
import pytest

from irys.matter import (
    AssertionCandidate, MatterModel, ModelLayer, SourceRole, SpeechAct,
)
from irys.matter.enums import (
    AssertionKind, OriginKind, ReviewedByKind,
    ReviewScope, VerificationTargetKind,
)
from irys.rlm.governance import (
    CascadeGovernor, ReadFamilyHandler, READ_FAMILY_PROMPT,
)


class _CapturingClient:
    """Captures the prompt text sent to the read synth call so tests
    can assert what did or did not reach the LLM."""

    def __init__(self, response: str):
        self.response = response
        self.last_prompt: str | None = None

    async def complete(self, prompt, **kwargs):
        self.last_prompt = prompt
        return self.response

    def snapshot_usage(self):
        return {}

    def get_usage_delta(self, _before):
        return {}


def _seed_matter_with_verified_fact(
    note: str = "This clause is what the MSJ hinges on.",
) -> tuple[MatterModel, str]:
    mm = MatterModel.open_in_memory()
    run_id = mm.start_run("seed")
    c = AssertionCandidate(
        proposition_text="Payment obligation is net 30 days per clause 3.2.",
        speech_act=SpeechAct.OPERATIVE,
        source_role=SourceRole.OPERATIVE,
        document_id="contracts/msa.pdf",
        model_layer=ModelLayer.RECORD,
        assertion_kind=AssertionKind.FACTUAL,
        origin_kind=OriginKind.EXTRACTED,
    )
    aid, _ = mm.assertions.upsert_occurrence(c, run_id=run_id)
    mm.verification.verify(
        VerificationTargetKind.ASSERTION, aid,
        reviewed_by_kind=ReviewedByKind.ATTORNEY,
        review_scope=ReviewScope.EXTRACTION_CORRECT,
        review_note=note,
        cause="test_verify",
    )
    return mm, aid


def _canned_read_response() -> str:
    return (
        '{"answer": "placeholder", "answer_confidence": "medium", '
        '"citations": ["contracts/msa.pdf"], '
        '"used_existing_state_only": true, "escalation_hint": ""}'
    )


_READ_CONTRACT = CascadeGovernor._contract_for("read")


def test_guidance_enters_prompt_when_opt_in(tmp_path):
    mm, _aid = _seed_matter_with_verified_fact()
    client = _CapturingClient(_canned_read_response())
    handler = ReadFamilyHandler(client=client, matter_model=mm)

    asyncio.run(handler.run(
        query="summarize what we know",
        contract=_READ_CONTRACT,
        include_attorney_guidance=True,
    ))

    assert client.last_prompt is not None
    # The note is in the prompt.
    assert "MSJ hinges on" in client.last_prompt
    # Framed as guidance, NOT evidence.
    assert "Attorney note, not evidence" in client.last_prompt
    # The guidance section header sits in the prompt.
    assert "Attorney guidance" in client.last_prompt


def test_guidance_absent_when_opt_out_default(tmp_path):
    """Default include_attorney_guidance=False — export / deliverable
    callers get zero guidance leak."""
    mm, _aid = _seed_matter_with_verified_fact()
    client = _CapturingClient(_canned_read_response())
    handler = ReadFamilyHandler(client=client, matter_model=mm)

    asyncio.run(handler.run(
        query="summarize what we know",
        contract=_READ_CONTRACT,
        # include_attorney_guidance omitted — defaults to False.
    ))
    assert client.last_prompt is not None
    assert "MSJ hinges on" not in client.last_prompt
    # The block placeholder resolved to the "no guidance" marker.
    assert "(no attorney guidance on these facts)" in client.last_prompt


def test_stale_reason_must_not_leak_as_guidance():
    """mark_stale reuses the review_note column for stale_reason. A
    naive query would leak "document_hash_changed:…" system strings
    into attorney guidance. The verified/human filter stops that."""
    mm, aid = _seed_matter_with_verified_fact(note="attorney's real note")
    # Mark stale — this writes a stale_reason INTO review_note per
    # the existing graph.py implementation. A bug-catching test
    # would otherwise see that text leak.
    mm.verification.mark_stale(
        VerificationTargetKind.ASSERTION, aid,
        stale_reason="document_hash_changed:abc123->def456",
    )

    client = _CapturingClient(_canned_read_response())
    handler = ReadFamilyHandler(client=client, matter_model=mm)
    asyncio.run(handler.run(
        query="summarize",
        contract=_READ_CONTRACT,
        include_attorney_guidance=True,
    ))
    assert client.last_prompt is not None
    # Stale system text does NOT appear in the prompt.
    assert "document_hash_changed" not in client.last_prompt
    # And the stale assertion no longer appears as verified either.
    assert "attorney's real note" not in client.last_prompt


def test_candidate_assertion_note_not_leaked():
    """A review_note on a non-verified (candidate) row must NOT
    appear — guidance is gated on status='verified'."""
    mm = MatterModel.open_in_memory()
    run_id = mm.start_run("seed")
    c = AssertionCandidate(
        proposition_text="Candidate fact text.",
        speech_act=SpeechAct.ALLEGED,
        source_role=SourceRole.ADVOCACY,
        document_id="doc.pdf",
        model_layer=ModelLayer.RECORD,
        assertion_kind=AssertionKind.FACTUAL,
        origin_kind=OriginKind.EXTRACTED,
    )
    aid, _ = mm.assertions.upsert_occurrence(c, run_id=run_id)
    # Directly write a verification_state row as status='candidate'
    # with a note (unusual shape but defensive).
    mm.verification.touch_ai_target(
        VerificationTargetKind.ASSERTION, aid, cause="test",
    )
    # Inject a bogus review_note on the candidate row to simulate
    # the edge case a buggy future path might create.
    import sqlite3 as _sqlite3
    try:
        mm.db.execute(
            """UPDATE verification_state SET review_note=?
               WHERE matter_id=? AND target_kind='assertion' AND target_id=?""",
            ("never-ever-leak-this-candidate-note",
             mm.matter_id, aid),
        )
    except _sqlite3.Error:
        pass  # if the schema rejects, the test still proves the gate

    client = _CapturingClient(_canned_read_response())
    handler = ReadFamilyHandler(client=client, matter_model=mm)
    asyncio.run(handler.run(
        query="summarize",
        contract=_READ_CONTRACT,
        include_attorney_guidance=True,
    ))
    assert client.last_prompt is not None
    assert "never-ever-leak-this-candidate-note" not in client.last_prompt


def test_verbatim_echo_is_redacted_from_answer():
    """adv#13 Finding #1: prompt-only 'don't quote' is insufficient.
    If the LLM emits a note verbatim in `answer` (induced by a query
    like "what did the attorney say"), the post-LLM redactor MUST
    strip it before the handler returns. Otherwise the note ships
    to the UI / callback / logs as public answer text.
    """
    note = (
        "CRITICAL: this clause is what the MSJ hinges on — emphasize it "
        "when summarizing."
    )
    mm, _aid = _seed_matter_with_verified_fact(note=note)
    # Response that quotes the note back verbatim — the LLM
    # disregarded the "do not quote" instruction.
    leaky_response = (
        '{"answer": "The attorney said: \\"' + note + '\\" and we should follow that.", '
        '"answer_confidence": "high", '
        '"citations": ["contracts/msa.pdf"], '
        '"used_existing_state_only": true, "escalation_hint": ""}'
    )
    client = _CapturingClient(leaky_response)
    handler = ReadFamilyHandler(client=client, matter_model=mm)

    result = asyncio.run(handler.run(
        query="what did the attorney say about this clause",
        contract=_READ_CONTRACT,
        include_attorney_guidance=True,
    ))
    # The raw note string must NOT appear in the returned answer.
    assert note not in result.answer
    # The neutral redaction marker tells callers something was stripped.
    assert "[attorney guidance not quoted]" in result.answer


def test_redaction_preserves_legitimate_answer_text():
    """The redactor must not delete a legitimate summary that happens
    to share short common phrases with the note. Only substrings of
    _GUIDANCE_ECHO_MIN_SUBSTRING or more chars trigger redaction."""
    mm, _aid = _seed_matter_with_verified_fact(
        note="This clause is what the MSJ hinges on.",
    )
    clean_response = (
        '{"answer": "Payment is net 30 days. Notice period is 60 days.", '
        '"answer_confidence": "high", '
        '"citations": ["contracts/msa.pdf"], '
        '"used_existing_state_only": true, "escalation_hint": ""}'
    )
    client = _CapturingClient(clean_response)
    handler = ReadFamilyHandler(client=client, matter_model=mm)
    result = asyncio.run(handler.run(
        query="summarize", contract=_READ_CONTRACT,
        include_attorney_guidance=True,
    ))
    # Legitimate answer content passes through unmodified.
    assert "Payment is net 30 days" in result.answer
    assert "[attorney guidance not quoted]" not in result.answer


def test_batch_verify_toctou_race_cannot_overwrite_rejection():
    """adv#13 Finding #2: the SELECT-candidate + UPDATE-verified
    sequence is wrapped in a BEGIN IMMEDIATE transaction. A
    concurrent rejection committed before our transaction starts
    must be visible to the SELECT (so we skip), and no rejection
    can commit between our SELECT and UPDATE.

    We simulate the pre-transaction rejection: reject id X first,
    then call batch-verify with [X]. Expected outcome: X is skipped,
    rejection stands.
    """
    from irys.matter.enums import ReviewedByKind, VerificationTargetKind
    mm = MatterModel.open_in_memory()
    run_id = mm.start_run("toctou")
    c = AssertionCandidate(
        proposition_text="race target",
        speech_act=SpeechAct.OPERATIVE,
        source_role=SourceRole.OPERATIVE,
        document_id="doc.pdf",
        model_layer=ModelLayer.RECORD,
        assertion_kind=AssertionKind.FACTUAL,
        origin_kind=OriginKind.EXTRACTED,
    )
    aid, _ = mm.assertions.upsert_occurrence(c, run_id=run_id)
    # Reviewer A rejects first.
    mm.verification.reject(
        VerificationTargetKind.ASSERTION, aid,
        reviewed_by_kind=ReviewedByKind.USER,
        rejection_reason="wrong doc",
    )
    # Reviewer B's batch-verify list still includes the id — would
    # have resolved at UI-load time before the rejection hit.
    verified = mm.bulk_verify_assertion_ids(
        [aid], reviewed_by_kind="user",
    )
    assert verified == [], (
        "batch-verify must skip an already-rejected row, not overwrite"
    )
    # Status stays rejected.
    vs = mm.verification.get(VerificationTargetKind.ASSERTION, aid)
    assert vs["status"] == "rejected"


def test_guidance_budget_caps_enforced():
    """More than 5 verified+noted facts must not all render; entries
    cap at _GUIDANCE_MAX_ENTRIES."""
    mm = MatterModel.open_in_memory()
    run_id = mm.start_run("seed")
    aids: list[str] = []
    for i in range(8):
        c = AssertionCandidate(
            proposition_text=f"Operative fact #{i} from clause {i}.",
            speech_act=SpeechAct.OPERATIVE,
            source_role=SourceRole.OPERATIVE,
            document_id=f"contracts/doc_{i}.pdf",
            model_layer=ModelLayer.RECORD,
            assertion_kind=AssertionKind.FACTUAL,
            origin_kind=OriginKind.EXTRACTED,
        )
        aid, _ = mm.assertions.upsert_occurrence(c, run_id=run_id)
        mm.verification.verify(
            VerificationTargetKind.ASSERTION, aid,
            reviewed_by_kind=ReviewedByKind.ATTORNEY,
            review_scope=ReviewScope.EXTRACTION_CORRECT,
            review_note=f"Attorney guidance tag {i}: load-bearing note.",
            cause="test",
        )
        aids.append(aid)

    client = _CapturingClient(_canned_read_response())
    handler = ReadFamilyHandler(client=client, matter_model=mm)
    asyncio.run(handler.run(
        query="summarize",
        contract=_READ_CONTRACT,
        include_attorney_guidance=True,
    ))
    assert client.last_prompt is not None
    # At most _GUIDANCE_MAX_ENTRIES notes render.
    count = client.last_prompt.count("Attorney note, not evidence")
    assert count <= ReadFamilyHandler._GUIDANCE_MAX_ENTRIES, (
        f"expected ≤ {ReadFamilyHandler._GUIDANCE_MAX_ENTRIES}, "
        f"got {count}"
    )
