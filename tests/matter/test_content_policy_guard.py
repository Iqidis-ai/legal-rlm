"""P0.5 commit 2 tests: ContentPolicyGuard + content_policy_audit.

Verifies the guard composes ContentPolicy.decide() with the DB-backed
audit log so every clean-mode surface can check and the decision is
persisted for audit.
"""

import pytest

from irys.matter import MatterModel
from irys.matter.trust import (
    ContentAction, ContentPurpose, TrustBucket,
    REASON_ALLOWED_VERIFIED, REASON_ALLOWED_INTERNAL,
    REASON_PRIVILEGE_CLEAN_MODE, REASON_VERIFICATION_REJECTED,
    WITHHELD_PLACEHOLDER,
)


@pytest.fixture
def model():
    return MatterModel.open_in_memory()


# ---------------------------------------------------------------------------
# Decision contract (composes with trust.py)
# ---------------------------------------------------------------------------

def test_guard_decide_returns_content_policy_decision(model):
    """The guard wraps ContentPolicy.decide() — same decision, plus
    the audit side effect."""
    d = model.content_policy.decide(
        purpose=ContentPurpose.SYNTHESIS_CONTEXT,
        subject_kind="assertion",
        subject_id="a1",
        policy_audience="clean",
        privilege_flag=False,
        assertion_verification_status="verified",
    )
    assert d.action is ContentAction.ALLOW
    assert d.reason_code == REASON_ALLOWED_VERIFIED
    assert d.trust_bucket is TrustBucket.VERIFIED


def test_guard_accepts_string_purpose(model):
    """Callers coming from HTTP/JSON land pass strings — the guard
    coerces to ContentPurpose without crashing."""
    d = model.content_policy.decide(
        purpose="chat_response",
        subject_kind="assertion",
        subject_id="a2",
        policy_audience="clean",
        privilege_flag=True,
    )
    assert d.action is ContentAction.BLOCK
    assert d.reason_code == REASON_PRIVILEGE_CLEAN_MODE


# ---------------------------------------------------------------------------
# Audit persistence
# ---------------------------------------------------------------------------

def test_every_decision_writes_one_audit_row(model):
    """One call to decide() → one row in content_policy_audit."""
    model.content_policy.decide(
        purpose=ContentPurpose.TIMELINE_VIEW,
        subject_kind="assertion",
        subject_id="a1",
        policy_audience="clean",
        privilege_flag=True,
    )
    rows = model.content_policy.list_decisions(
        target_kind="assertion", target_id="a1",
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["purpose"] == "timeline_view"
    assert row["policy_audience"] == "clean"
    assert row["action"] == "withhold"
    assert row["reason_code"] == REASON_PRIVILEGE_CLEAN_MODE
    assert row["privilege_flag"] == 1
    assert row["trust_bucket"] == "excluded"


def test_audit_preserves_chronology(model):
    """Multiple decisions for the same target stack in time; list
    returns newest first."""
    for _ in range(3):
        model.content_policy.decide(
            purpose=ContentPurpose.SYNTHESIS_CONTEXT,
            subject_kind="assertion",
            subject_id="a_chronology",
            policy_audience="clean",
            privilege_flag=False,
            assertion_verification_status="candidate",
        )
    rows = model.content_policy.list_decisions(
        target_kind="assertion", target_id="a_chronology",
    )
    assert len(rows) == 3
    # Each row has a distinct id.
    assert len({r["id"] for r in rows}) == 3


def test_audit_filters_by_purpose_and_target(model):
    """list_decisions filters work — callers can narrow to one
    purpose, one target, or both."""
    model.content_policy.decide(
        purpose=ContentPurpose.TIMELINE_VIEW,
        subject_kind="assertion", subject_id="alpha",
        privilege_flag=True,
    )
    model.content_policy.decide(
        purpose=ContentPurpose.EXPORT,
        subject_kind="assertion", subject_id="alpha",
        privilege_flag=False,
        assertion_verification_status="verified",
    )
    model.content_policy.decide(
        purpose=ContentPurpose.TIMELINE_VIEW,
        subject_kind="authority", subject_id="beta",
        privilege_flag=False,
    )
    all_alpha = model.content_policy.list_decisions(
        target_kind="assertion", target_id="alpha",
    )
    assert len(all_alpha) == 2
    only_timeline = model.content_policy.list_decisions(
        target_kind="assertion", target_id="alpha",
        purpose="timeline_view",
    )
    assert len(only_timeline) == 1
    only_authority = model.content_policy.list_decisions(
        target_kind="authority",
    )
    assert len(only_authority) == 1


def test_audit_skipped_when_record_false(model):
    """Callers iterating over big sets can skip per-row audit writes."""
    model.content_policy.decide(
        purpose=ContentPurpose.HYDRATION,
        subject_kind="assertion", subject_id="skip",
        policy_audience="clean",
        privilege_flag=False,
        assertion_verification_status="candidate",
        record=False,
    )
    rows = model.content_policy.list_decisions(
        target_kind="assertion", target_id="skip",
    )
    assert rows == []


def test_internal_audience_still_audits(model):
    """Internal audience allows everything but still records the
    decision so the audit reflects the read."""
    d = model.content_policy.decide(
        purpose=ContentPurpose.CHAT_RESPONSE,
        subject_kind="assertion", subject_id="internal_one",
        policy_audience="internal",
        privilege_flag=True,
    )
    assert d.is_allowed
    assert d.reason_code == REASON_ALLOWED_INTERNAL
    rows = model.content_policy.list_decisions(
        target_kind="assertion", target_id="internal_one",
    )
    assert len(rows) == 1
    assert rows[0]["action"] == "allow"
    assert rows[0]["policy_audience"] == "internal"


def test_rejected_row_recorded_with_reason_code(model):
    d = model.content_policy.decide(
        purpose=ContentPurpose.TIMELINE_VIEW,
        subject_kind="assertion", subject_id="rej",
        policy_audience="clean",
        privilege_flag=False,
        assertion_verification_status="rejected",
    )
    assert d.is_withheld
    assert d.placeholder == WITHHELD_PLACEHOLDER
    row = model.content_policy.list_decisions(
        target_kind="assertion", target_id="rej",
    )[0]
    assert row["reason_code"] == REASON_VERIFICATION_REJECTED
    assert row["action"] == "withhold"


def test_hydration_writes_content_policy_audit_rows(model):
    """P0.5 commit 3: engine hydration now routes through the guard
    for audit. Every hydrated assertion produces a
    ContentPurpose.HYDRATION decision row so the audit trail shows
    what entered the LLM's context."""
    from irys.rlm.engine import RLMEngine, RLMConfig
    from irys.rlm.state import InvestigationState
    from irys.matter.runtime import MatterRuntimeAdapter
    from unittest.mock import MagicMock

    run_id = model.start_run("hydration audit")
    adapter = MatterRuntimeAdapter(model, run_id)
    aid = adapter.record_fact("Some fact to hydrate", "doc.pdf")

    engine = RLMEngine(gemini_client=MagicMock(), config=RLMConfig(), matter_model=model)
    state = InvestigationState(id="h-audit", query="test", repository_path="/tmp/h")
    engine._hydrate_from_matter_model(state)

    # Exactly one audit row for the hydrated assertion, purpose=hydration.
    rows = model.content_policy.list_decisions(
        target_kind="assertion", target_id=aid, purpose="hydration",
    )
    assert len(rows) == 1
    assert rows[0]["action"] == "allow"  # candidate is eligible for hydration


def test_audit_write_failure_doesnt_break_decision(model):
    """The guard's audit append is best-effort — a failing audit
    must not propagate to the caller and break the read."""
    # Monkeypatch the db execute so the INSERT raises.
    real_execute = model.db.execute

    def fake_execute(sql, *args, **kwargs):
        if "INSERT INTO content_policy_audit" in sql:
            raise RuntimeError("simulated audit failure")
        return real_execute(sql, *args, **kwargs)

    model.db.execute = fake_execute
    try:
        d = model.content_policy.decide(
            purpose=ContentPurpose.CHAT_RESPONSE,
            subject_kind="assertion", subject_id="a_fail",
            policy_audience="clean",
            privilege_flag=False,
            assertion_verification_status="verified",
        )
    finally:
        model.db.execute = real_execute
    assert d.is_allowed
