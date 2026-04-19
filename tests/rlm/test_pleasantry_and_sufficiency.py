"""Pleasantry fast-path + interim sufficiency probe regression tests."""

from __future__ import annotations

import asyncio

import pytest

from irys.rlm.engine import RLMEngine, RLMConfig
from irys.rlm.governance import _is_pleasantry
from irys.rlm.state import InvestigationState


# ---------------------------------------------------------------------------
# Pleasantry detector
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query, expected", [
    ("hi", True),
    ("Hi!", True),
    ("Hello.", True),
    ("HELLO", True),
    ("thanks", True),
    ("thank you", True),
    ("how are you", True),
    ("how are you?", True),
    ("ok", True),
    ("great!", True),
    ("good morning", True),
    ("bye", True),
    # NOT pleasantries — they have real substance or length.
    ("hi, what's the damages exposure", False),
    ("summarize what we know", False),
    ("how are we doing on the MSA analysis today", False),
    ("", False),
    ("   ", False),
])
def test_is_pleasantry(query, expected):
    assert _is_pleasantry(query) is expected


# ---------------------------------------------------------------------------
# Sufficiency probe
# ---------------------------------------------------------------------------


class _ProbeClient:
    """Captures the probe prompt + returns a scripted response."""

    def __init__(self, response: str):
        self.response = response
        self.probe_prompts: list[str] = []

    async def complete(self, prompt, **kwargs):
        if (kwargs.get("usage_label") or "") == "sufficiency_probe":
            self.probe_prompts.append(prompt)
        return self.response

    def snapshot_usage(self):
        return {}

    def get_usage_delta(self, _before):
        return {}


@pytest.fixture
def engine():
    return RLMEngine(gemini_client=_ProbeClient('{}'), config=RLMConfig())


def _state_with_facts(query="summarize", n_facts=3, n_cits=1) -> InvestigationState:
    from irys.rlm.state import Citation
    from datetime import datetime
    s = InvestigationState.create(query, "/tmp/repo", research_mode="deep")
    s.findings["accumulated_facts"] = [
        {"text": f"seed fact {i}"} for i in range(n_facts)
    ]
    for i in range(n_cits):
        s.citations.append(Citation(
            id=str(i),
            document=f"contracts/doc_{i}.pdf",
            page=None,
            text=f"cite {i}",
            context="",
            relevance="supporting",
            timestamp=datetime.now(),
        ))
    return s


def test_probe_can_answer_stamps_early_terminate(engine):
    """Probe says yes at ≥medium confidence with ≥1 citation —
    engine writes final_output + early_terminate_reason."""
    client = _ProbeClient(
        '{"can_answer": true, "answer": "Payment is net 30 per the MSA.", '
        '"confidence": "medium", "citations": ["contracts/msa.pdf"], '
        '"reason_not_yet": ""}'
    )
    engine.client = client
    s = _state_with_facts()
    result = asyncio.run(engine._run_sufficiency_probe(s))
    assert result is True
    assert s.early_terminate_reason is not None
    assert "medium" in s.early_terminate_reason
    assert s.findings["final_output"].startswith("Payment is net 30")


def test_probe_insufficient_answer_does_not_terminate(engine):
    """Probe claims can_answer=true but with low confidence / zero
    citations — engine distrusts the eager YES and keeps running."""
    client = _ProbeClient(
        '{"can_answer": true, "answer": "Probably 30 days.", '
        '"confidence": "low", "citations": [], '
        '"reason_not_yet": ""}'
    )
    engine.client = client
    s = _state_with_facts()
    assert asyncio.run(engine._run_sufficiency_probe(s)) is False
    assert s.early_terminate_reason is None


def test_probe_can_answer_false_keeps_loop_running(engine):
    """Probe says no — loop should keep running."""
    client = _ProbeClient(
        '{"can_answer": false, "answer": "", "confidence": "low", '
        '"citations": [], '
        '"reason_not_yet": "need to read the latest production emails"}'
    )
    engine.client = client
    s = _state_with_facts()
    assert asyncio.run(engine._run_sufficiency_probe(s)) is False
    assert s.early_terminate_reason is None


def test_probe_respects_max_per_run(engine):
    """The per-run probe cap (_SUFFICIENCY_PROBE_MAX_PER_RUN=3)
    must prevent a runaway investigation from paying 20 LITE calls
    to a chatty probe."""
    client = _ProbeClient('{"can_answer": false, "answer": "", "confidence": "low", "citations": [], "reason_not_yet": "x"}')
    engine.client = client
    s = _state_with_facts()
    # Call probe 5 times. First 3 hit the LLM; last 2 short-circuit
    # at the cap.
    for _ in range(5):
        asyncio.run(engine._run_sufficiency_probe(s))
    assert len(client.probe_prompts) == engine._SUFFICIENCY_PROBE_MAX_PER_RUN


# ---------------------------------------------------------------------------
# Min-iter override via early_terminate_reason
# ---------------------------------------------------------------------------


def test_early_terminate_reason_overrides_min_iter(engine):
    """When the sufficiency probe has stamped early_terminate_reason,
    the termination controller must return (False, ...) IMMEDIATELY —
    before it even checks the contract's min_iter floor. Research
    modes otherwise mandate more iters than needed."""
    from irys.rlm.governance import ExecutionContract
    s = _state_with_facts()
    # Force a min_iter=5 contract and state at iter 0. Normally the
    # loop would be forced to keep going for 5 iters.
    s.execution_contract = ExecutionContract(
        family="investigate", min_iter=5, max_iter=10,
    )
    s.max_depth_reached = 0
    # But the probe already decided we can stop.
    s.early_terminate_reason = "test: matter already sufficient"
    should_continue, reason = engine._should_continue_investigation(s)
    assert should_continue is False
    assert "Sufficiency probe" in reason


def test_min_iter_still_enforced_without_probe_signal(engine):
    """Guard: removing the early-terminate flag must NOT accidentally
    suppress the min_iter gate. Contract-mandated min iters still
    hold when the probe hasn't signaled early stop."""
    from irys.rlm.governance import ExecutionContract
    s = _state_with_facts()
    s.execution_contract = ExecutionContract(
        family="investigate", min_iter=5, max_iter=10,
    )
    s.max_depth_reached = 0
    s.early_terminate_reason = None  # no probe signal
    should_continue, reason = engine._should_continue_investigation(s)
    assert should_continue is True
    assert "minimum evidence base" in reason.lower()


def test_early_terminate_survives_checkpoint_roundtrip():
    """The early_terminate_reason field must persist through
    to_dict/from_dict so a resume doesn't re-enter the loop after
    we already decided to stop."""
    s = _state_with_facts()
    s.early_terminate_reason = "sufficiency probe at iter 4"
    restored = InvestigationState.from_dict(s.to_dict())
    assert restored.early_terminate_reason == "sufficiency probe at iter 4"
