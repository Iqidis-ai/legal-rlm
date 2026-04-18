"""MVI-1 cascade governance tests.

Covers:
  - cold-start hard-route (no matter model → investigate)
  - snapshot building on a warm matter
  - classifier contract defaults per family
  - decision cache key stability
  - ReadFamilyHandler escalation behavior when matter is empty
  - ReadFamilyHandler contract-floor escalation

Does NOT hit a real Gemini API — the classifier/read handler is
called against a fake client that returns canned JSON.
"""

from __future__ import annotations

import asyncio
import pytest

from irys.matter import MatterModel, AssertionCandidate, SpeechAct, SourceRole
from irys.rlm.governance import (
    CLASSIFIER_SCHEMA_VERSION,
    AnswerabilitySnapshot,
    CascadeDecision,
    CascadeGovernor,
    CompareFamilyHandler,
    DeliverableFamilyHandler,
    ExecutionContract,
    QueryFamilyHandler,
    ReadFamilyHandler,
    ScenarioFamilyHandler,
    SteerFamilyHandler,
    TraceFamilyHandler,
    decision_cache_key,
)


# ---------------------------------------------------------------------------
# Fake client — enough to exercise the async complete() surface
# ---------------------------------------------------------------------------


class _FakeClient:
    """Canned-response GeminiClient stub. Records every complete() call
    so tests can assert which tier / label was used."""

    def __init__(self, responses_by_label: dict[str, str]):
        self.responses = responses_by_label
        self.calls: list[dict] = []

    async def complete(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        label = kwargs.get("usage_label") or ""
        return self.responses.get(label, '{}')


# ---------------------------------------------------------------------------
# CascadeGovernor
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_matter():
    return MatterModel.open_in_memory()


@pytest.fixture
def warm_matter():
    m = MatterModel.open_in_memory()
    run_id = m.start_run("seed")
    for i in range(3):
        m.record_assertion(
            AssertionCandidate(
                proposition_text=f"Fact {i}",
                speech_act=SpeechAct.ALLEGED,
                source_role=SourceRole.ADVOCACY,
                document_id="doc.pdf",
            ),
            run_id=run_id,
        )
    m.complete_run(run_id)
    return m


def test_cold_start_hard_routes_to_investigate(empty_matter):
    """Fresh matter (no facts) must short-circuit to investigate
    without calling the classifier — read has nothing to read from."""
    client = _FakeClient({})
    gov = CascadeGovernor(client=client, matter_model=empty_matter)
    decision = asyncio.run(gov.decide(query="summarize", conversation_history=None))
    assert decision.family == "investigate"
    assert decision.confidence == 1.0
    assert "cold-start" in decision.rationale
    # Classifier must NOT have been called on cold start.
    assert len(client.calls) == 0


def test_warm_matter_classifier_returns_read(warm_matter):
    """When classifier picks `read` on a warm matter, the governor
    returns a read-family decision with the read contract."""
    client = _FakeClient({
        "intent_classifier": '{"family": "read", "confidence": 0.9, "rationale": "summary over existing facts"}',
    })
    gov = CascadeGovernor(client=client, matter_model=warm_matter)
    decision = asyncio.run(gov.decide(
        query="Summarize what we know so far",
        conversation_history=[{"query": "prior", "answer": "answer"}],
    ))
    assert decision.family == "read"
    assert decision.contract.family == "read"
    assert decision.contract.max_iter == 1
    assert decision.contract.escalation_allowed is True
    # Classifier WAS called on warm matter.
    assert len(client.calls) == 1
    assert client.calls[0]["usage_label"] == "intent_classifier"


def test_read_route_on_empty_matter_overrides_to_investigate(empty_matter):
    """Belt-and-suspenders: even if the classifier somehow returns
    `read` on an empty matter, the governor overrides. The cold-start
    shortcut catches this first, but the override in _classify is the
    last line of defense."""
    # Cold-start shortcut catches the empty case before the classifier
    # fires — this confirms the guard.
    client = _FakeClient({
        "intent_classifier": '{"family": "read", "confidence": 0.95, "rationale": "bad call"}',
    })
    gov = CascadeGovernor(client=client, matter_model=empty_matter)
    decision = asyncio.run(gov.decide(query="summarize"))
    assert decision.family == "investigate"


def test_classifier_parse_failure_falls_back_to_investigate(warm_matter):
    """Malformed JSON from the classifier must not abort the run — we
    default to investigate (safe over silent cheap-wrong)."""
    client = _FakeClient({
        "intent_classifier": 'not valid json {{{',
    })
    gov = CascadeGovernor(client=client, matter_model=warm_matter)
    decision = asyncio.run(gov.decide(query="anything"))
    assert decision.family == "investigate"
    assert decision.confidence == 0.0
    assert "parse error" in decision.rationale


def test_contract_for_each_family():
    """Codex master plan: each family has a distinct ExecutionContract.
    Investigate has a floor, read/clarify have 0 min_iter."""
    investigate = CascadeGovernor._contract_for("investigate")
    read = CascadeGovernor._contract_for("read")
    clarify = CascadeGovernor._contract_for("clarify")
    assert investigate.min_iter >= 1
    assert read.min_iter == 0
    assert read.max_iter == 1
    assert clarify.max_iter == 0
    assert clarify.escalation_allowed is False


def test_snapshot_fields_reflect_matter_state(warm_matter):
    """Snapshot must surface has_any_facts + trust_revision so the
    classifier has state-aware signal."""
    client = _FakeClient({})
    gov = CascadeGovernor(client=client, matter_model=warm_matter)
    snap = gov._build_snapshot(conversation_history=None)
    assert snap.matter_id == warm_matter.matter_id
    assert snap.has_any_facts is True
    assert snap.assertion_count >= 3


# ---------------------------------------------------------------------------
# decision_cache_key
# ---------------------------------------------------------------------------


def _snap(**overrides):
    base = dict(
        matter_id="m1",
        assertion_count=10,
        verified_assertion_count=2,
        open_issue_count=3,
        open_gap_count=1,
        actor_count=2,
        has_any_facts=True,
        has_any_verified=True,
        trust_revision=5,
        policy_audience="clean",
        recent_turn_count=0,
        last_turn_summary=None,
    )
    base.update(overrides)
    return AnswerabilitySnapshot(**base)


def test_cache_key_stable_across_unrelated_changes():
    """Changing conversation turn count must NOT change the cache key
    (per Codex: don't hash raw turns — crushes hit rate)."""
    k1 = decision_cache_key("What's the timeline?", _snap(recent_turn_count=0))
    k2 = decision_cache_key("What's the timeline?", _snap(recent_turn_count=5))
    assert k1 == k2


def test_cache_key_changes_on_trust_revision():
    """Trust revision is the scoping fingerprint — must invalidate."""
    k1 = decision_cache_key("anything", _snap(trust_revision=1))
    k2 = decision_cache_key("anything", _snap(trust_revision=2))
    assert k1 != k2


def test_cache_key_changes_on_classifier_version():
    """Schema version must invalidate across prompt upgrades."""
    k1 = decision_cache_key("q", _snap(), classifier_version="mvi1.0")
    k2 = decision_cache_key("q", _snap(), classifier_version="mvi1.1")
    assert k1 != k2


def test_cache_key_case_insensitive_on_query():
    """Query is normalized lowercased + stripped — same intent hits
    the same route."""
    k1 = decision_cache_key("  Summarize This  ", _snap())
    k2 = decision_cache_key("summarize this", _snap())
    assert k1 == k2


# ---------------------------------------------------------------------------
# ReadFamilyHandler
# ---------------------------------------------------------------------------


def test_read_handler_escalates_without_matter_model():
    """Handler must return an escalation rather than crash when no
    matter model is wired."""
    handler = ReadFamilyHandler(client=_FakeClient({}), matter_model=None)
    result = asyncio.run(handler.run(
        query="summarize",
        contract=CascadeGovernor._contract_for("read"),
    ))
    assert result.escalation_needed is True
    assert "no matter model" in (result.escalation_reason or "")


def test_read_handler_low_confidence_escalates(warm_matter):
    """Contract has answer_confidence_floor=0.5. If the LLM returns
    `low`, handler should escalate to investigate."""
    client = _FakeClient({
        "read_synth": (
            '{"answer": "not enough info", "answer_confidence": "low", '
            '"citations": [], "used_existing_state_only": true, '
            '"escalation_hint": "need to read the MSA"}'
        ),
    })
    handler = ReadFamilyHandler(client=client, matter_model=warm_matter)
    result = asyncio.run(handler.run(
        query="what does the MSA say about termination?",
        contract=CascadeGovernor._contract_for("read"),
    ))
    assert result.confidence_label == "low"
    assert result.confidence_score < 0.5
    assert result.escalation_needed is True
    assert "MSA" in (result.escalation_reason or "")


def test_read_handler_high_confidence_does_not_escalate(warm_matter):
    """`high` confidence + eligible contract should NOT escalate."""
    client = _FakeClient({
        "read_synth": (
            '{"answer": "The notice period is 30 days.", '
            '"answer_confidence": "high", "citations": ["msa.pdf"], '
            '"used_existing_state_only": true, "escalation_hint": ""}'
        ),
    })
    handler = ReadFamilyHandler(client=client, matter_model=warm_matter)
    result = asyncio.run(handler.run(
        query="what's the notice period?",
        contract=CascadeGovernor._contract_for("read"),
    ))
    assert result.confidence_label == "high"
    assert result.confidence_score >= 0.5
    assert result.escalation_needed is False
    assert result.citations == ["msa.pdf"]
    assert "30 days" in result.answer


def test_read_handler_malformed_json_escalates(warm_matter):
    """A non-JSON response must surface as a low-confidence
    escalation, not crash."""
    client = _FakeClient({"read_synth": "garbage output"})
    handler = ReadFamilyHandler(client=client, matter_model=warm_matter)
    result = asyncio.run(handler.run(
        query="summarize",
        contract=CascadeGovernor._contract_for("read"),
    ))
    assert result.escalation_needed is True
    assert result.confidence_label == "low"


# ---------------------------------------------------------------------------
# QueryFamilyHandler
# ---------------------------------------------------------------------------


def test_query_handler_keyword_fastpath(warm_matter):
    """Unambiguous keyword match resolves sub-intent without NANO."""
    # Client not used on fast path — pass an empty fake.
    handler = QueryFamilyHandler(matter_model=warm_matter, client=_FakeClient({}))
    result = asyncio.run(handler.run(
        query="show me all the actors",
        contract=CascadeGovernor._contract_for("query"),
    ))
    assert result.intent == "list_actors"
    assert result.escalation_needed is False


def test_query_handler_falls_back_to_nano(warm_matter):
    """Ambiguous / no-keyword query routes through NANO sub-intent."""
    client = _FakeClient({"query_sub_intent": '{"intent": "list_gaps"}'})
    handler = QueryFamilyHandler(matter_model=warm_matter, client=client)
    # "What's still unaddressed" doesn't match any fast-path keyword.
    result = asyncio.run(handler.run(
        query="what's still unaddressed in this matter",
        contract=CascadeGovernor._contract_for("query"),
    ))
    assert result.intent == "list_gaps"
    # Confirm NANO was consulted.
    assert any(
        c.get("usage_label") == "query_sub_intent"
        for c in client.calls
    )


def test_query_handler_nano_says_none_escalates(warm_matter):
    """When NANO returns 'none', handler escalates."""
    client = _FakeClient({"query_sub_intent": '{"intent": "none"}'})
    handler = QueryFamilyHandler(matter_model=warm_matter, client=client)
    result = asyncio.run(handler.run(
        query="give me a narrative analysis of the matter",
        contract=CascadeGovernor._contract_for("query"),
    ))
    assert result.intent == ""
    assert result.escalation_needed is True


# ---------------------------------------------------------------------------
# TraceFamilyHandler
# ---------------------------------------------------------------------------


def test_trace_handler_no_prior_runs(empty_matter):
    """A matter with no runs renders a clean empty response, not
    an error."""
    handler = TraceFamilyHandler(matter_model=empty_matter)
    result = handler.run(
        query="why did you say that",
        contract=CascadeGovernor._contract_for("trace"),
    )
    assert result.escalation_needed is False
    assert "No prior runs" in result.rendered_answer


def test_trace_handler_points_at_most_recent_run(warm_matter):
    """Trace defaults to the most recent completed run."""
    handler = TraceFamilyHandler(matter_model=warm_matter)
    result = handler.run(
        query="why did you say that",
        contract=CascadeGovernor._contract_for("trace"),
    )
    assert result.target_kind == "run"
    assert result.target_id is not None
    assert "Trace" in result.rendered_answer


# ---------------------------------------------------------------------------
# SteerFamilyHandler
# ---------------------------------------------------------------------------


def test_steer_handler_correct_assertion_preview(warm_matter):
    """NANO parses the correction; handler finds candidate assertions
    matching the target_hint; renders a preview but does NOT apply."""
    client = _FakeClient({
        "steer_parse": (
            '{"action": "correct_assertion", "target_hint": "Fact 1", '
            '"old_value": "March", "new_value": "April", '
            '"rationale": "date correction"}'
        ),
    })
    handler = SteerFamilyHandler(matter_model=warm_matter, client=client)
    result = asyncio.run(handler.run(
        query="Actually the date was April, not March",
        contract=CascadeGovernor._contract_for("steer"),
    ))
    assert result.action == "correct_assertion"
    assert result.new_value == "April"
    assert "Proposed change" in result.rendered_answer
    assert result.escalation_needed is False
    # Crucially — this is a PREVIEW, not an application. The matter
    # model's assertion count should be unchanged.
    assert warm_matter.assertions.count() == 3


def test_steer_handler_parse_other_escalates(warm_matter):
    """When NANO returns 'other' (intent unclear), escalate rather
    than silently proposing something."""
    client = _FakeClient({
        "steer_parse": '{"action": "other", "target_hint": "", "rationale": "unclear"}',
    })
    handler = SteerFamilyHandler(matter_model=warm_matter, client=client)
    result = asyncio.run(handler.run(
        query="huh?",
        contract=CascadeGovernor._contract_for("steer"),
    ))
    assert result.action == "other"
    assert result.escalation_needed is True


def test_steer_handler_no_matter_model_escalates():
    client = _FakeClient({})
    handler = SteerFamilyHandler(matter_model=None, client=client)
    result = asyncio.run(handler.run(
        query="correct that",
        contract=CascadeGovernor._contract_for("steer"),
    ))
    assert result.escalation_needed is True


# ---------------------------------------------------------------------------
# CompareFamilyHandler
# ---------------------------------------------------------------------------


def test_compare_handler_no_prior_runs(empty_matter):
    """Empty matter with no runs renders a baseline-empty comparison,
    not an error."""
    handler = CompareFamilyHandler(matter_model=empty_matter)
    result = handler.run(
        query="what changed",
        contract=CascadeGovernor._contract_for("compare"),
    )
    assert result.escalation_needed is False
    assert "Baseline" in result.rendered_answer


def test_compare_handler_reports_delta(warm_matter):
    """Compare surfaces current vs baseline assertion count."""
    handler = CompareFamilyHandler(matter_model=warm_matter)
    result = handler.run(
        query="what changed since last run",
        contract=CascadeGovernor._contract_for("compare"),
    )
    assert result.current_assertion_count >= 3
    assert "What changed" in result.rendered_answer


# ---------------------------------------------------------------------------
# ScenarioFamilyHandler
# ---------------------------------------------------------------------------


def test_scenario_handler_parses_and_answers(warm_matter):
    """NANO parses assumption; read handler answers under the override
    without mutating matter state."""
    client = _FakeClient({
        "scenario_parse": (
            '{"assumption": "The contract is void", '
            '"core_question": "What are our damages?"}'
        ),
        "read_synth": (
            '{"answer": "Under that assumption, damages would be zero.", '
            '"answer_confidence": "medium", "citations": [], '
            '"used_existing_state_only": true, "escalation_hint": ""}'
        ),
    })
    handler = ScenarioFamilyHandler(client=client, matter_model=warm_matter)
    before_count = warm_matter.assertions.count()
    result = asyncio.run(handler.run(
        query="what if the contract is void — what are our damages?",
        contract=CascadeGovernor._contract_for("scenario"),
    ))
    assert result.assumption == "The contract is void"
    assert "damages would be zero" in result.answer
    # Crucially: no state mutation from a scenario turn.
    assert warm_matter.assertions.count() == before_count


# ---------------------------------------------------------------------------
# DeliverableFamilyHandler — privilege log renderer (MVI-7)
# ---------------------------------------------------------------------------


def _seed_privileged_doc(m, path: str, flag: int, is_tbd: bool = False):
    """Directly seed a document_inventory + document_card row so the
    privilege-log renderer has something to read. flag=1 → privileged,
    is_tbd=True → populate unresolved_flags so the renderer marks TBD."""
    import uuid as _uuid
    import time as _time
    import json as _json
    inv_id = _uuid.uuid4().hex
    now = _time.strftime("%Y-%m-%dT%H:%M:%S")
    m.db.execute(
        """INSERT INTO document_inventory
             (id, matter_id, relative_path, size_bytes, sha256,
              salience_score, discovered_at)
           VALUES (?, ?, ?, 0, '', 0.5, ?)""",
        (inv_id, m.matter_id, path, now),
    )
    unresolved = _json.dumps(["privilege_classification"]) if is_tbd else None
    m.db.execute(
        """INSERT INTO document_card
             (id, doc_id, doc_type, doc_subtype, title, author,
              sender, recipient, creation_date, effective_date,
              privilege_flag, unresolved_flags, purpose,
              operative_status, rhetorical_posture,
              created_at, updated_at)
           VALUES (?, ?, 'memo', 'legal_memo', ?, ?, ?, ?,
                   '2026-03-15', '2026-03-15', ?, ?, ?,
                   'operative', 'neutral', ?, ?)""",
        (
            _uuid.uuid4().hex, inv_id,
            f"Title of {path}", "Attorney A", "Attorney A", "Client B",
            flag, unresolved, "privileged communication",
            now, now,
        ),
    )


def test_deliverable_handler_privilege_log_no_rows(warm_matter):
    """No privileged docs → renderer returns an empty-state message,
    does not crash or silently produce an empty table."""
    client = _FakeClient({
        "deliverable_sub_intent": '{"intent": "privilege_log"}',
    })
    handler = DeliverableFamilyHandler(
        matter_model=warm_matter, client=client,
    )
    result = asyncio.run(handler.run(
        query="generate a privilege log",
        contract=CascadeGovernor._contract_for("deliverable"),
    ))
    assert result.intent == "privilege_log"
    assert result.row_count == 0
    assert "No documents currently classified as privileged" in result.rendered_answer


def test_deliverable_handler_privilege_log_renders(warm_matter):
    """Seeded privileged + tbd rows render in the log with correct
    TBD handling and a footer summary."""
    _seed_privileged_doc(warm_matter, "memo1.pdf", flag=1)
    _seed_privileged_doc(warm_matter, "memo2.pdf", flag=1, is_tbd=True)
    _seed_privileged_doc(warm_matter, "memo3.pdf", flag=1)
    client = _FakeClient({
        "deliverable_sub_intent": '{"intent": "privilege_log"}',
    })
    handler = DeliverableFamilyHandler(
        matter_model=warm_matter, client=client,
    )
    result = asyncio.run(handler.run(
        query="generate a privilege log",
        contract=CascadeGovernor._contract_for("deliverable"),
    ))
    assert result.row_count == 3
    assert "Privilege log" in result.rendered_answer
    assert "Attorney A" in result.rendered_answer
    # One TBD row seeded via unresolved_flags.
    assert "TBD" in result.rendered_answer
    assert "privileged: 2" in result.rendered_answer
    assert "TBD / needs review: 1" in result.rendered_answer


def test_deliverable_handler_unsupported_intent_escalates(warm_matter):
    """Sub-intents other than privilege_log escalate in MVI-7."""
    client = _FakeClient({
        "deliverable_sub_intent": '{"intent": "dep_outline"}',
    })
    handler = DeliverableFamilyHandler(
        matter_model=warm_matter, client=client,
    )
    result = asyncio.run(handler.run(
        query="outline my deposition of smith",
        contract=CascadeGovernor._contract_for("deliverable"),
    ))
    assert result.intent == "dep_outline"
    assert result.escalation_needed is True
    assert "not yet implemented" in (result.escalation_reason or "")


def test_scenario_handler_unparseable_escalates(warm_matter):
    """When NANO can't extract a usable assumption, escalate."""
    client = _FakeClient({
        "scenario_parse": '{"assumption": "", "core_question": ""}',
    })
    handler = ScenarioFamilyHandler(client=client, matter_model=warm_matter)
    result = asyncio.run(handler.run(
        query="what if",
        contract=CascadeGovernor._contract_for("scenario"),
    ))
    assert result.escalation_needed is True


def test_steer_handler_reject_target_with_candidates(warm_matter):
    """`reject_target` action looks up assertions by substring."""
    client = _FakeClient({
        "steer_parse": (
            '{"action": "reject_target", "target_hint": "Fact 2", '
            '"old_value": null, "new_value": null, '
            '"rationale": "should be ignored"}'
        ),
    })
    handler = SteerFamilyHandler(matter_model=warm_matter, client=client)
    result = asyncio.run(handler.run(
        query="ignore fact 2, it's wrong",
        contract=CascadeGovernor._contract_for("steer"),
    ))
    assert result.action == "reject_target"
    # warm_matter seeded Fact 0/1/2 so Fact 2 should match.
    assert any("Fact 2" in str(c.get("proposition_text", "")) for c in result.candidates)
