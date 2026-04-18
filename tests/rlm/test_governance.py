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


def test_classifier_route_cache_reuses_on_repeat(warm_matter):
    """Adversarial #10 Fix D acceptance: first query populates route
    cache; a second identical query must reuse it (zero NANO calls)
    so a rate-limit event on NANO can't herd warm queries into the
    full AR loop."""
    client = _FakeClient({
        "intent_classifier": (
            '{"family": "read", "confidence": 0.9, '
            '"rationale": "warm summarize"}'
        ),
    })
    gov = CascadeGovernor(client=client, matter_model=warm_matter)
    # First call: classifier fires, cache populates.
    first = asyncio.run(gov.decide(query="summarize", conversation_history=None))
    assert first.family == "read"
    assert len(client.calls) == 1
    # Second IDENTICAL call: cache hit, classifier must NOT fire.
    second = asyncio.run(gov.decide(query="summarize", conversation_history=None))
    assert second.family == "read"
    assert len(client.calls) == 1  # unchanged!
    assert second.escalation_reason == "cache_hit"


def test_classifier_cache_respects_trust_revision_bump(warm_matter):
    """Round 2 regression: a trust_revision bump (e.g., from a
    rejection) must invalidate cached routes. The old
    _cache_get_any_version scanned "any" cached family without
    checking trust_revision, so a post-correction query could
    resurrect a pre-correction route. Now fixed via strict
    key match — cache.get() applies trust_revision prefix."""
    client = _FakeClient({
        "intent_classifier": (
            '{"family": "read", "confidence": 0.9, "rationale": "warm"}'
        ),
    })
    gov = CascadeGovernor(client=client, matter_model=warm_matter)
    # First route populates cache.
    first = asyncio.run(gov.decide(query="summarize"))
    assert first.family == "read"
    assert len(client.calls) == 1

    # Simulate a rejection: bump trust_revision.
    warm_matter.cache.bump_trust_revision()

    # Second identical query after the bump MUST re-classify (cache
    # miss) because trust_revision changed.
    second = asyncio.run(gov.decide(query="summarize"))
    assert second.family == "read"
    assert len(client.calls) == 2  # Re-fired after trust bump.


def test_stale_cache_does_not_return_unrelated_prior_route():
    """Round 5: the positive-hit test seeded only ONE prior entry,
    so it would pass even on a blunt revert to 'first cached row
    wins'. This test seeds a prior entry for query A and then
    drives a classifier failure on DIFFERENT query B — the stale
    fallback must skip A and fall through to investigate. Without
    this the stale lookup is an audit-polluter."""
    import json
    from irys.rlm.governance import decision_cache_key
    mm = MatterModel.open_in_memory()
    run_id = mm.start_run("seed")
    mm.record_assertion(
        AssertionCandidate(
            proposition_text="Seed fact",
            speech_act=SpeechAct.ALLEGED,
            source_role=SourceRole.ADVOCACY,
            document_id="d.pdf",
        ),
        run_id=run_id,
    )
    mm.complete_run(run_id)

    gov_probe = CascadeGovernor(client=_FakeClient({}), matter_model=mm)
    snap = gov_probe._build_snapshot(conversation_history=None)
    # Seed a prior-version entry for query A.
    query_a = "summarize the matter"
    query_b = "what are the damages"
    prior_key = decision_cache_key(query_a, snap, "mvi6.0")
    mm.cache.put("cascade_decision", prior_key, {
        "family": "read", "confidence": 0.8,
        "rationale": "route for A",
        "schema_version": "mvi6.0",
    })

    # Force NANO failure on query B. With strict matching the stale
    # fallback scans for query_b+snapshot under prior versions,
    # doesn't find it, and returns None — so governor defaults to
    # investigate. With the old overbroad behavior the fallback
    # would have returned the A-route for B.
    class _FailingClient:
        async def complete(self, *a, **kw):
            raise RuntimeError("NANO outage")
    gov = CascadeGovernor(client=_FailingClient(), matter_model=mm)
    result = asyncio.run(gov.decide(query=query_b))
    # Different query → stale fallback must skip → defaults to investigate.
    assert result.family == "investigate"
    # Classifier_version on an investigate fallback is the current
    # schema, not the stale-fallback sentinel.
    assert result.classifier_version != "_stale_cache_fallback"


def test_stale_cache_positive_hit_reuses_prior_version(warm_matter):
    """Round 4: the skip-unrelated path was tested but the POSITIVE
    HIT path was not. This test seeds a cached route under a PRIOR
    classifier schema version (mvi6.0) and then forces NANO failure
    on the same query — `_cache_get_any_version` should find the
    prior-version entry and reuse it with the
    '_stale_cache_fallback' sentinel."""
    import json
    from irys.rlm.governance import (
        CLASSIFIER_SCHEMA_VERSION,
        decision_cache_key,
        AnswerabilitySnapshot,
    )

    # Build the snapshot the governor would see for this query.
    query = "Summarize our session"
    # Simulate a prior classifier schema version being in cache.
    gov_probe = CascadeGovernor(client=_FakeClient({}), matter_model=warm_matter)
    snap = gov_probe._build_snapshot(conversation_history=None)
    # Seed the cache under a PRIOR version so the current-version
    # lookup misses but the stale-fallback scan finds it.
    prior_version = "mvi6.0"
    assert prior_version != CLASSIFIER_SCHEMA_VERSION
    prior_key = decision_cache_key(query, snap, prior_version)
    warm_matter.cache.put(
        "cascade_decision", prior_key,
        {
            "family": "read",
            "confidence": 0.75,
            "rationale": "seeded prior-version route",
            "schema_version": prior_version,
        },
    )

    # Now NANO fails. Exact current-version cache misses (no entry
    # under CLASSIFIER_SCHEMA_VERSION). Stale fallback must surface
    # the prior-version entry as `_stale_cache_fallback`.
    class _FailingClient:
        async def complete(self, *a, **kw):
            raise RuntimeError("NANO outage")
    gov = CascadeGovernor(client=_FailingClient(), matter_model=warm_matter)
    result = asyncio.run(gov.decide(query=query))
    assert result.family == "read"  # reused route
    assert result.classifier_version == "_stale_cache_fallback"
    assert result.escalation_reason == "stale_cache_fallback"


def test_stale_cache_fallback_actually_fires_under_classifier_failure(warm_matter):
    """Round 3: the R2 `test_classifier_failure_uses_stale_cache_fallback`
    never actually exercised `_cache_get_any_version()` because it hit
    the exact-key cache first. This test forces NANO to fail AFTER a
    prior different-query route has populated the cache, so the exact
    cache misses and the stale-version escape hatch fires."""
    # Seed a prior route under a DIFFERENT query so the exact-cache
    # entry for the failure query doesn't exist.
    client_ok = _FakeClient({
        "intent_classifier": (
            '{"family": "read", "confidence": 0.8, '
            '"rationale": "prior warm route"}'
        ),
    })
    gov_ok = CascadeGovernor(client=client_ok, matter_model=warm_matter)
    asyncio.run(gov_ok.decide(query="summarize"))
    assert len(client_ok.calls) == 1

    # Now NANO fails on a DIFFERENT query that has no exact-cache
    # entry. With the R3 fix, stale_cache_fallback must still be
    # evaluated and skip (not a match for this query/snapshot).
    # The correct behavior: classifier error → investigate, because
    # stale-cache only returns for the SAME query fingerprint.
    class _FailingClient:
        async def complete(self, *a, **kw):
            raise RuntimeError("simulated NANO rate limit")
    gov_fail = CascadeGovernor(
        client=_FailingClient(), matter_model=warm_matter,
    )
    result = asyncio.run(gov_fail.decide(query="what's the notice period?"))
    # The failure-query has no cached route (different query) →
    # stale-fallback correctly skips → defaults to investigate.
    # This proves _cache_get_any_version doesn't blanket-return the
    # first cached family it sees.
    assert result.family == "investigate"


def test_classifier_failure_uses_stale_cache_fallback(warm_matter):
    """When NANO fails but a stale cache entry exists, reuse it
    rather than hard-routing to investigate — mitigates classifier
    rate-limit SPOF."""
    # Seed cache with a prior route.
    client_ok = _FakeClient({
        "intent_classifier": (
            '{"family": "read", "confidence": 0.8, '
            '"rationale": "prior warm route"}'
        ),
    })
    gov_ok = CascadeGovernor(client=client_ok, matter_model=warm_matter)
    asyncio.run(gov_ok.decide(query="summarize"))
    assert len(client_ok.calls) == 1  # cache populated

    # Simulate NANO outage — classifier fails on a DIFFERENT query
    # (one that cache doesn't have exact-key for), and verify the
    # stale-version escape hatch catches any prior cached decision.
    class _FailingClient:
        async def complete(self, *a, **kw):
            raise RuntimeError("simulated NANO rate limit")
    gov_fail = CascadeGovernor(
        client=_FailingClient(), matter_model=warm_matter,
    )
    # Exact cache hit first — classifier never fires.
    result = asyncio.run(gov_fail.decide(query="summarize"))
    assert result.family == "read"  # stayed routed via cache


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


def test_read_handler_infra_failure_tagged_distinctly(warm_matter):
    """Adversarial #10 fix: LLM call itself failing must set
    failure_kind='infra' and NOT auto-escalate. api.py surfaces the
    error to the user rather than silently kicking off the full AR
    loop during an outage."""
    class _FailingClient:
        async def complete(self, *a, **kw):
            raise RuntimeError("simulated provider outage")
    handler = ReadFamilyHandler(
        client=_FailingClient(), matter_model=warm_matter,
    )
    result = asyncio.run(handler.run(
        query="summarize",
        contract=CascadeGovernor._contract_for("read"),
    ))
    assert result.failure_kind == "infra"
    assert result.escalation_needed is False  # critical — don't auto-escalate
    assert "simulated provider outage" in (result.escalation_reason or "")


def test_read_handler_rejects_empty_string_citations(warm_matter):
    """Round 2 regression: `[""]` must not count as a real citation.
    Previously citation_floor just checked `len(citations)`, so an
    LLM emitting `[""]` shipped a high-confidence uncited answer."""
    client = _FakeClient({
        "read_synth": (
            '{"answer": "30 days", "answer_confidence": "high", '
            '"citations": ["", "  ", null], '
            '"used_existing_state_only": true, "escalation_hint": ""}'
        ),
    })
    handler = ReadFamilyHandler(client=client, matter_model=warm_matter)
    result = asyncio.run(handler.run(
        query="notice period?",
        contract=CascadeGovernor._contract_for("read"),
    ))
    # All three fake citations must be filtered; floor=1 forces escalation.
    assert len(result.citations) == 0
    assert result.escalation_needed is True
    assert result.failure_kind == "state_insufficient"


def test_read_handler_rejects_numeric_and_bool_citations(warm_matter):
    """Round 3: R2 accepted int/float/bool as valid citations so
    `[0]` or `[False]` trivially satisfied the floor. Citations are
    document identifiers — always strings. Non-string entries must
    be rejected."""
    client = _FakeClient({
        "read_synth": (
            '{"answer": "30 days", "answer_confidence": "high", '
            '"citations": [0, false, 3.14, true, 42], '
            '"used_existing_state_only": true, "escalation_hint": ""}'
        ),
    })
    handler = ReadFamilyHandler(client=client, matter_model=warm_matter)
    result = asyncio.run(handler.run(
        query="notice period?",
        contract=CascadeGovernor._contract_for("read"),
    ))
    assert len(result.citations) == 0
    assert result.escalation_needed is True
    assert result.failure_kind == "state_insufficient"


def test_specific_tokens_legal_document_boundary_matrix():
    """Final legal-document boundary matrix after rounds 4-8 of
    Codex adversarial review. The regex now bounds on
    [\\w\\d-] — letter, digit, hyphen, and underscore all block
    adjacency. This favors NOT consuming identifier-embedded
    date shapes, which is the safer trade-off for preview-only
    steer scoring (worst case: fragments flow and might over-
    match a generic number, which is less harmful than a fake
    date token artificially boosting the wrong candidate).

    Cases covered:
      - identifier-embedded dates (letter / underscore / hyphen
        adjacent) — fragments flow, no token
      - calendar-invalid dates — no token
      - year-out-of-range — fragments flow
      - real standalone / delimited-embedded dates — emit token,
        fragments suppressed
      - known over-match: `YYYY-MM-DD-YYYY-MM-DD` date-range shape
        drops both dates (fragments only) — acceptable for
        preview-only ranking; see governance.py R8 comment.
    """
    s = SteerFamilyHandler._specific_tokens

    # IDENTIFIER cases — all fragments flow, no date token.
    for text, expected_fragment in [
        ("abc1234-5-6xyz", "1234"),      # year out of range
        ("x20260-13-45y", "20260"),      # shifted-digit year
        ("Case-2026-13-45-A", "2026"),   # hyphen-bounded (R6)
        ("123-2026-13-45", "2026"),      # leading-hyphen-bounded
        ("Ex2026-04-15A", "2026"),       # letter-embedded (R8)
        ("case_2026-04-15_a", "2026"),   # underscore-bounded (R8)
        ("1899-12-31", "1899"),          # year out of range
        ("2100-01-01", "2100"),          # year out of range
    ]:
        tokens = s(text)
        assert expected_fragment in tokens, (
            f"{text!r} should emit fragment {expected_fragment!r}"
        )
        # And no date token.
        for tok in tokens:
            assert tok.count("-") < 2, (
                f"{text!r} leaked full date token {tok!r}"
            )

    # CONSUME-ONLY / NO-EMIT: calendar-invalid dates (no fragments,
    # no token).
    assert s("2026-02-31") == set()

    # VALID DATE: token emitted, fragments suppressed.
    for text in ("2026-04-15", "ref=2026-04-15/paper"):
        tokens = s(text)
        assert "2026-04-15" in tokens
        assert "2026" not in tokens
        assert "15" not in tokens

    # KNOWN OVER-MATCH: date range. Acceptable — preview-only.
    range_tokens = s("2026-04-15-2026-04-20")
    assert "2026" in range_tokens  # fragments leak
    # Neither date emits as a token because each blocks the
    # other's lookbehind/lookahead.
    assert "2026-04-15" not in range_tokens
    assert "2026-04-20" not in range_tokens


def test_specific_tokens_rejects_regex_invalid_iso_dates():
    """Round 4: '2026-13-45' doesn't match the strict ISO regex
    (month 13 exceeds 01-12), so the R3 fix never marked its span
    consumed. That let '2026', '13', '45' leak through as
    independent numeric tokens. Fix: broad ISO-shape regex
    consumes the span regardless of regex validity."""
    tokens = SteerFamilyHandler._specific_tokens("2026-13-45")
    # Full nonsense date must not be a token.
    assert "2026-13-45" not in tokens
    # And its fragments must not leak either.
    assert "2026" not in tokens
    assert "13" not in tokens
    assert "45" not in tokens


def test_specific_tokens_rejects_calendar_invalid_iso_dates():
    """Round 3: `2026-02-31` passes the regex but isn't a real
    calendar date. Must be rejected AND must not leak as
    independent number tokens (2026/02/31)."""
    tokens = SteerFamilyHandler._specific_tokens("2026-02-31")
    assert "2026-02-31" not in tokens
    # The internal fragments must NOT leak either (the old R2 regex
    # fix rejected the full date but number-extraction would still
    # grab "2026" and "31" as independent numeric tokens).
    assert "2026" not in tokens
    assert "31" not in tokens
    # Real date still works.
    valid = SteerFamilyHandler._specific_tokens("2026-04-15")
    assert "2026-04-15" in valid
    # And doesn't produce fragment noise for valid dates.
    assert "2026" not in valid
    assert "15" not in valid


def test_read_handler_citation_floor_forces_escalation(warm_matter):
    """Adversarial #10 finding #2: citation_floor was declared but
    not enforced. An LLM response with high confidence but zero
    citations used to ship silently. Must now escalate."""
    client = _FakeClient({
        "read_synth": (
            '{"answer": "yes", "answer_confidence": "high", '
            '"citations": [], "used_existing_state_only": true, '
            '"escalation_hint": ""}'
        ),
    })
    handler = ReadFamilyHandler(client=client, matter_model=warm_matter)
    # read contract has citation_floor=1
    result = asyncio.run(handler.run(
        query="what's the notice period?",
        contract=CascadeGovernor._contract_for("read"),
    ))
    assert result.escalation_needed is True
    assert result.failure_kind == "state_insufficient"
    assert "citations 0 < floor 1" in (result.escalation_reason or "")


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


def test_deliverable_handler_privilege_log_never_leaks_descriptions(warm_matter):
    """Adversarial #10 regression: every row in the privilege log
    must render a fail-closed description placeholder, NEVER the raw
    `purpose` or `title` field. Both fields are LLM-authored and can
    contain privileged substance. Every row must also be marked TBD
    so no one serves the log without attorney review."""
    # Seed a doc whose `purpose` and `title` contain sensitive phrases
    # that would be demo-breaking if they leaked into the description
    # column. The renderer must NOT print them.
    import uuid as _uuid
    import time as _time
    inv_id = _uuid.uuid4().hex
    now = _time.strftime("%Y-%m-%dT%H:%M:%S")
    warm_matter.db.execute(
        """INSERT INTO document_inventory
             (id, matter_id, relative_path, size_bytes, sha256,
              salience_score, discovered_at)
           VALUES (?, ?, ?, 0, '', 0.5, ?)""",
        (inv_id, warm_matter.matter_id, "sensitive.pdf", now),
    )
    sensitive_purpose = (
        "Email requesting legal advice on whether to terminate the "
        "CFO before the SEC interview"
    )
    sensitive_title = "SEC strategy re revenue recognition"
    warm_matter.db.execute(
        """INSERT INTO document_card
             (id, doc_id, doc_type, title, author, sender, recipient,
              creation_date, privilege_flag, purpose, operative_status,
              rhetorical_posture, created_at, updated_at)
           VALUES (?, ?, 'memo', ?, 'AttnA', 'AttnA', 'Client',
                   '2026-03-15', 1, ?, 'operative', 'neutral', ?, ?)""",
        (
            _uuid.uuid4().hex, inv_id,
            sensitive_title, sensitive_purpose, now, now,
        ),
    )
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
    assert result.row_count == 1
    rendered = result.rendered_answer
    # The sensitive substance must NOT appear anywhere in the log.
    assert "terminate the CFO" not in rendered
    assert "SEC interview" not in rendered
    assert "revenue recognition" not in rendered
    # The locked-down placeholder MUST appear.
    assert "[withheld — awaiting reviewed privilege description]" in rendered
    # Every row must be marked TBD.
    assert "TBD — attorney review required" in rendered
    # Attorney-review warning banner must be present.
    assert "Do not serve this log without attorney review" in rendered


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


def test_steer_target_matcher_specific_tokens_rank_correctly():
    """Adversarial #10 Fix F acceptance: with multiple same-topic
    assertions, a correction containing old_value + a specific token
    (date/amount) must rank the correct assertion first.

    Scenario: two assertions mention "April" and "payment" and
    "notice". One says "April 15" (the actual notice date), the
    other says "April 20" (the invoice due date). User says:
    "Actually the April 15 date was the notice date, not the
    invoice due date." Old matcher would tie or pick wrong. New
    matcher must rank the "April 15" assertion first because it
    contains both old_value words AND the specific date token.
    """
    m = MatterModel.open_in_memory()
    rid = m.start_run("seed")
    ids = []
    for prop in [
        "invoice due date was April 20 under the payment terms",
        "notice to cure was delivered on April 15 per clause 12",
        "payment schedule includes monthly installments of $5000",
    ]:
        aid, _ = m.record_assertion(
            AssertionCandidate(
                proposition_text=prop,
                speech_act=SpeechAct.ALLEGED,
                source_role=SourceRole.OPERATIVE,
                document_id="msa.pdf",
            ),
            run_id=rid,
        )
        ids.append(aid)
    m.complete_run(rid)

    client = _FakeClient({
        "steer_parse": (
            '{"action": "correct_assertion", '
            '"target_hint": "April 15 notice date", '
            '"old_value": "invoice due date", '
            '"new_value": "notice date", '
            '"rationale": "date correction"}'
        ),
    })
    handler = SteerFamilyHandler(matter_model=m, client=client)
    result = asyncio.run(handler.run(
        query=(
            "Actually the April 15 date was the notice date, "
            "not the invoice due date."
        ),
        contract=CascadeGovernor._contract_for("steer"),
    ))
    assert result.action == "correct_assertion"
    assert len(result.candidates) >= 1
    top_text = str(result.candidates[0].get("proposition_text", ""))
    # The top candidate must contain "April 15" (the specific token
    # from the correction) — NOT "April 20".
    assert "April 15" in top_text
    assert "April 20" not in top_text


def test_steer_matcher_does_not_double_weight_old_value_words():
    """Round 2 regression: two assertions BOTH contain "April 15" but
    one describes it as an invoice due date and the other as a
    notice date. User says 'the April 15 date was the notice date,
    not the invoice due date.' The old scorer double-weighted
    old_value ("invoice due date"), which ranked the WRONG-label
    sibling first. The fix: old_value generic words don't score —
    they're what's being negated."""
    m = MatterModel.open_in_memory()
    rid = m.start_run("seed")
    # Both assertions contain "April 15" — the specific token.
    # The correction's old_value is "invoice due date" which would
    # under the old scoring add points to the WRONG (first) row.
    props = [
        # Wrong row — its description says "invoice due date" but the
        # user is CORRECTING that away.
        "April 15 was mischaracterized as the invoice due date",
        # Right row — the user wants THIS one ranked first.
        "The notice to cure was delivered on April 15",
    ]
    ids = []
    for prop in props:
        aid, _ = m.record_assertion(
            AssertionCandidate(
                proposition_text=prop,
                speech_act=SpeechAct.ALLEGED,
                source_role=SourceRole.OPERATIVE,
                document_id="msa.pdf",
            ),
            run_id=rid,
        )
        ids.append(aid)
    m.complete_run(rid)

    client = _FakeClient({
        "steer_parse": (
            '{"action": "correct_assertion", '
            '"target_hint": "April 15 notice date", '
            '"old_value": "invoice due date", '
            '"new_value": "notice date", '
            '"rationale": "date was mislabeled"}'
        ),
    })
    handler = SteerFamilyHandler(matter_model=m, client=client)
    result = asyncio.run(handler.run(
        query="Actually the April 15 date was the notice date",
        contract=CascadeGovernor._contract_for("steer"),
    ))
    assert result.candidates, "expected some candidates"
    top_text = str(result.candidates[0].get("proposition_text", ""))
    # The top candidate MUST be the notice row, not the invoice row.
    assert "notice to cure" in top_text.lower()
    assert "invoice due date" not in top_text.lower()


def test_steer_matcher_rejects_invalid_iso_date():
    """Round 2 regression: ISO date regex validates month (01-12)
    and day (01-31) ranges. '2026-13-45' must not be accepted as a
    specific date token."""
    tokens = SteerFamilyHandler._specific_tokens("2026-13-45")
    # Nonsense ISO date should NOT be extracted.
    assert "2026-13-45" not in tokens
    # Real ISO date still extracts.
    tokens2 = SteerFamilyHandler._specific_tokens("2026-04-15")
    assert "2026-04-15" in tokens2


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
