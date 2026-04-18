"""API-level adversarial regression gate for the cascade.

Codex adversarial #10 finding #3: the existing cascade tests are
handler/unit coverage. They do NOT drive `Irys.investigate()`
through the real bad paths. Without that, fixes A–F can "land"
unit-green and still fail end-to-end.

This suite drives the actual Irys API through each adversarial
scenario Codex named. It runs against in-memory matters and a
stubbed engine so it doesn't pay real-LLM / real-repo cost.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from irys.api import Irys, IrysConfig, InvestigationResult
from irys.matter import MatterModel, AssertionCandidate, SpeechAct, SourceRole
from irys.rlm.state import InvestigationState


@pytest.fixture
def repo_path(tmp_path):
    """A real existing dir that passes path-validation but is empty.
    Uses pytest's tmp_path fixture — pytest handles teardown with
    platform-aware cleanup (Windows file-lock races killed the old
    tempfile.TemporaryDirectory approach)."""
    return str(tmp_path.resolve())


class _FakeClient:
    """Canned-response GeminiClient stub. `raise_next_label` forces
    one specific `usage_label` to raise — simulates provider outage."""

    def __init__(self, responses_by_label):
        self.responses = responses_by_label
        self.calls = []
        self.raise_next_label = None

    async def complete(self, prompt, **kwargs):
        self.calls.append(kwargs)
        label = kwargs.get("usage_label") or ""
        if self.raise_next_label == label:
            self.raise_next_label = None
            raise RuntimeError(f"simulated outage on {label}")
        return self.responses.get(label, "{}")

    def snapshot_usage(self):
        return {}

    def get_usage_delta(self, _before):
        return {"request_count": 0, "estimated_cost_usd": 0.0}


def _warm_in_memory_matter():
    mm = MatterModel.open_in_memory()
    run_id = mm.start_run("seed")
    for i in range(3):
        mm.record_assertion(
            AssertionCandidate(
                proposition_text=f"Seed fact {i}",
                speech_act=SpeechAct.ALLEGED,
                source_role=SourceRole.ADVOCACY,
                document_id="doc.pdf",
            ),
            run_id=run_id,
        )
    mm.complete_run(run_id)
    return mm


def _make_irys(fake_client, matter_model):
    """Construct an Irys with a fake client and a pre-wired in-memory
    matter model so `investigate()` doesn't need a real repo / real
    Gemini / real disk."""
    cfg = IrysConfig(api_key="test", enable_matter_model=False)
    irys = Irys(config=cfg)
    irys._client = fake_client
    irys._ensure_initialized()
    irys._engine.client = fake_client
    irys._engine._matter_model = matter_model
    # Disable engine-side matter-model wiring in investigate() so we
    # don't try to open a real MatterModel from disk. The hot path in
    # api.py will still read self._engine._matter_model which we set
    # above.
    irys.config.enable_matter_model = False
    # Stub the engine's full investigate so the tests don't actually
    # drive the AR loop — we only care whether it's CALLED.
    irys._engine.investigate = AsyncMock(return_value=InvestigationState.create(
        "stub", "/tmp/repo", research_mode="deep",
    ))
    return irys


# ---------------------------------------------------------------------------
# Fix B acceptance — read infra failure + citation floor
# ---------------------------------------------------------------------------


def test_api_read_infra_failure_does_not_silently_run_investigate(repo_path):
    """Adversarial #10 acceptance B: when the read_synth LLM call
    itself fails, the API MUST surface the outage to the user and
    MUST NOT silently invoke the full investigate loop."""
    fake = _FakeClient({
        "intent_classifier": (
            '{"family": "read", "confidence": 0.9, '
            '"rationale": "warm matter summary"}'
        ),
    })
    fake.raise_next_label = "read_synth"
    mm = _warm_in_memory_matter()
    irys = _make_irys(fake, mm)

    result = asyncio.run(irys.investigate(
        query="Summarize what we know",
        repository=repo_path,
    ))

    # Critical: infra-failure notice in output.
    assert "couldn't reach the LLM service" in result.output
    # Critical: the full investigate loop must NOT have been invoked.
    irys._engine.investigate.assert_not_called()


def test_api_read_zero_citations_forces_escalation(repo_path):
    """Adversarial #10 acceptance B: a high-confidence read with
    zero citations must not ship under citation_floor=1 — it must
    escalate to investigate."""
    fake = _FakeClient({
        "intent_classifier": (
            '{"family": "read", "confidence": 0.9, '
            '"rationale": "warm matter"}'
        ),
        "read_synth": (
            '{"answer": "it is X", "answer_confidence": "high", '
            '"citations": [], "used_existing_state_only": true, '
            '"escalation_hint": ""}'
        ),
    })
    mm = _warm_in_memory_matter()
    irys = _make_irys(fake, mm)

    asyncio.run(irys.investigate(
        query="What's the notice period?",
        repository=repo_path,
    ))

    # Read fired first.
    labels = [c.get("usage_label") for c in fake.calls]
    assert "read_synth" in labels
    # Then escalated — engine.investigate was called. This is the
    # citation_floor gate working: even at high confidence, zero
    # citations forced escalation rather than shipping "it is X".
    irys._engine.investigate.assert_called_once()


def test_api_route_audit_preserves_classifier_and_terminal_family(repo_path):
    """Adversarial #10 Fix C acceptance: when classifier routes to
    `deliverable` but the handler escalates to `read` (MVI-7 ships
    only privilege_log), the persisted audit record must show BOTH
    the classifier's original call AND the terminal family — the
    old code mutated decision.family in place and destroyed that
    signal."""
    import json
    fake = _FakeClient({
        "intent_classifier": (
            '{"family": "deliverable", "confidence": 0.8, '
            '"rationale": "deposition outline asked"}'
        ),
        # dep_outline escalates in MVI-7.
        "deliverable_sub_intent": '{"intent": "dep_outline"}',
        # Then read synth answers (no citations, zero confidence) so
        # the path reaches a terminal family that differs from the
        # classifier family.
        "read_synth": (
            '{"answer": "placeholder", "answer_confidence": "medium", '
            '"citations": ["x.pdf"], "used_existing_state_only": true, '
            '"escalation_hint": ""}'
        ),
    })
    mm = _warm_in_memory_matter()
    # Use a real on-matter path so ledger writes land; the
    # helper keeps everything in-memory.
    irys = _make_irys(fake, mm)

    asyncio.run(irys.investigate(
        query="draft a dep outline for smith",
        repository=repo_path,
    ))

    # Look up the most recent ledger event and parse the audit JSON.
    rows = mm.db.execute(
        """SELECT snapshot_json FROM ledger_event
           WHERE event_type='route_decision'
           ORDER BY created_at DESC LIMIT 1"""
    ).fetchall()
    assert rows, "expected a ROUTE_DECISION ledger event"
    payload = json.loads(rows[0]["snapshot_json"])
    # Classifier family stays the original call.
    assert payload["classifier_family"] == "deliverable"
    # Terminal family reflects the escalation target.
    assert payload["terminal_family"] == "read"
    # The `family` audit field from decision.to_audit_dict is ALSO
    # the classifier's original call — verifying no in-place mutation.
    assert payload["family"] == "deliverable"


def test_api_read_ships_when_contract_met(repo_path):
    """Sanity: high-confidence read WITH citations ships without
    escalation. Guards against over-correction in Fix B."""
    fake = _FakeClient({
        "intent_classifier": (
            '{"family": "read", "confidence": 0.9, "rationale": "warm"}'
        ),
        "read_synth": (
            '{"answer": "30 days notice", "answer_confidence": "high", '
            '"citations": ["msa.pdf"], "used_existing_state_only": true, '
            '"escalation_hint": ""}'
        ),
    })
    mm = _warm_in_memory_matter()
    irys = _make_irys(fake, mm)

    result = asyncio.run(irys.investigate(
        query="Notice period?",
        repository=repo_path,
    ))
    assert "30 days notice" in result.output
    irys._engine.investigate.assert_not_called()
