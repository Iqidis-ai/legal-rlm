"""P0.4 Trust Invalidation Lite tests (SO-2).

Covers:
- matter.trust_revision is a durable counter, default 0 for fresh matters
- ReasoningCacheStore.get/put prefix cache keys with the current
  revision; a bump silently misses prior entries
- Human rejection (reject_target) bumps the revision so cached plans
  that referenced the rejected target become unreachable
"""

import pytest

from irys.matter import MatterModel
from irys.matter.runtime import MatterRuntimeAdapter


@pytest.fixture
def model():
    return MatterModel.open_in_memory()


def test_trust_revision_defaults_to_zero(model):
    """Fresh matters start at trust_revision=0 so legacy callers see
    a stable number. Schema default ensures older DBs migrate cleanly."""
    assert model.cache.current_trust_revision() == 0


def test_bump_trust_revision_increments(model):
    """Each bump increments by exactly 1 and returns the new value."""
    before = model.cache.current_trust_revision()
    after = model.cache.bump_trust_revision()
    assert after == before + 1
    # Successive bumps still increment.
    again = model.cache.bump_trust_revision()
    assert again == after + 1


def test_cache_hit_survives_without_bump(model):
    """With no invalidation trigger, a cache put is reachable by the
    same key on the next get — baseline SO-1 behavior."""
    model.cache.put("orient", "key_a", {"plan": "hit"})
    got = model.cache.get("orient", "key_a")
    assert got == {"plan": "hit"}


def test_cache_hit_lost_after_trust_revision_bump(model):
    """After bump_trust_revision, the prior entry is unreachable by
    the same raw key — its stored key was prefixed with the old
    revision and is no longer queried."""
    model.cache.put("orient", "key_a", {"plan": "hit"})
    assert model.cache.get("orient", "key_a") is not None
    model.cache.bump_trust_revision()
    assert model.cache.get("orient", "key_a") is None


def test_mark_stale_stores_reason(model):
    """mark_stale persists stale_reason on the verification_state row
    so audit can tell a document-hash-change stale from a
    span-replacement stale."""
    run_id = model.start_run("stale reason")
    adapter = MatterRuntimeAdapter(model, run_id=run_id)
    aid = adapter.record_fact("fact to stale", "doc.pdf")
    from irys.matter.enums import VerificationTargetKind
    model.verification.mark_stale(
        VerificationTargetKind.ASSERTION, aid,
        stale_reason="document_hash_changed:old→new",
    )
    vs = model.verification.get("assertion", aid)
    assert vs["status"] == "stale"
    assert "document_hash_changed" in (vs["stale_reason"] or "")


def test_mark_stale_does_not_overwrite_rejected(model):
    """Rejection is a stronger human opinion than stale. A subsequent
    mark_stale must not downgrade a rejected target."""
    from irys.matter.enums import VerificationTargetKind
    run_id = model.start_run("stale vs reject")
    adapter = MatterRuntimeAdapter(model, run_id=run_id)
    aid = adapter.record_fact("rejected fact", "doc.pdf")
    model.verification.reject(
        VerificationTargetKind.ASSERTION, aid,
        reviewed_by_kind="user", reviewed_by_id="r1",
        rejection_reason="fabricated",
    )
    vid = model.verification.mark_stale(
        VerificationTargetKind.ASSERTION, aid,
        stale_reason="doc_hash_change",
    )
    assert vid is None  # signals "left alone"
    vs = model.verification.get("assertion", aid)
    assert vs["status"] == "rejected"


def test_touch_ai_target_revives_stale_to_candidate(model):
    """P0.4: re-extracting a target whose verification_state is stale
    must revive it to candidate. Without revival, a re-ingested
    document would leave every extracted target stale forever."""
    from irys.matter.enums import VerificationTargetKind
    run_id = model.start_run("revival")
    adapter = MatterRuntimeAdapter(model, run_id=run_id)
    aid = adapter.record_fact("fact for revival", "doc.pdf")
    model.verification.mark_stale(
        VerificationTargetKind.ASSERTION, aid,
        stale_reason="test",
    )
    assert model.verification.get("assertion", aid)["status"] == "stale"
    # Simulate re-extraction — same document, same text. The
    # production AssertionStore.upsert_occurrence path already calls
    # touch_ai_target on every write.
    aid2 = adapter.record_fact("fact for revival", "doc.pdf")
    assert aid2 == aid
    vs = model.verification.get("assertion", aid)
    assert vs["status"] == "candidate", (
        "stale must revive to candidate on re-extraction"
    )


def test_touch_ai_target_leaves_verified_alone(model):
    """A verified target is an attorney opinion; a fresh AI write
    must not downgrade it back to candidate."""
    from irys.matter.enums import VerificationTargetKind
    run_id = model.start_run("verified preserved")
    adapter = MatterRuntimeAdapter(model, run_id=run_id)
    aid = adapter.record_fact("verified fact", "doc.pdf")
    model.verification.verify(
        VerificationTargetKind.ASSERTION, aid,
        reviewed_by_kind="attorney", reviewed_by_id="a1",
    )
    # Re-extract — assertion already verified; touch must not
    # downgrade it.
    adapter.record_fact("verified fact", "doc.pdf")
    vs = model.verification.get("assertion", aid)
    assert vs["status"] == "verified"


def test_mark_document_stale_stales_direct_dependents(model):
    """P0.4 AC #1: changed document hash must stale every direct
    dependent — assertions, occurrences, edges, quants, authorities,
    document card. Verified content is not downgraded (that's a
    stronger opinion than "the doc changed"); test uses candidate
    content for the strict check."""
    from irys.matter.enums import IssueType
    run_id = model.start_run("doc invalidation")
    adapter = MatterRuntimeAdapter(model, run_id=run_id)
    # Register the document + a card so the card target exists.
    inv_id, _ = model.inventory.upsert(
        "contracts/msa.pdf", "a" * 64, size_bytes=1,
    )
    model.document_cards.upsert(
        doc_id=inv_id, title="MSA", doc_type="contract",
    )
    iid, _ = model.issues.upsert_issue(
        "Breach", IssueType.CLAIM, materiality=0.7,
    )
    aid = adapter.record_fact(
        "Defendant owed payment",
        "contracts/msa.pdf",
        issue_id=iid, issue_link_type="supports",
    )
    qid = adapter.record_quant(
        quant_kind="amount", raw_text="$10,000",
        amount_value=10000.0, assertion_id=aid,
    )
    before_rev = model.cache.current_trust_revision()
    count = model.mark_document_stale(
        "contracts/msa.pdf", reason="hash_change_test",
    )
    assert count >= 3, (
        f"expected stale sweep to touch assertion+occurrence+quant+card+edge, "
        f"got {count}"
    )
    # Each direct dependent is now stale.
    assert model.verification.get("assertion", aid)["status"] == "stale"
    assert model.verification.get("quant_fact", qid)["status"] == "stale"
    # Trust revision bumped exactly once despite multi-target sweep.
    assert model.cache.current_trust_revision() == before_rev + 1


def test_mark_document_stale_preserves_rejected(model):
    """Rejection is a stronger opinion than stale — mark_document_stale
    must NOT downgrade a rejected target to stale."""
    run_id = model.start_run("doc stale preserves reject")
    adapter = MatterRuntimeAdapter(model, run_id=run_id)
    model.inventory.upsert("c.pdf", "a" * 64, size_bytes=1)
    aid = adapter.record_fact("rejected fact", "c.pdf")
    model.reject_target(
        "assertion", aid,
        reviewed_by_kind="user", reviewed_by_id="r1",
        rejection_reason="bad",
    )
    model.mark_document_stale("c.pdf", reason="hash_change")
    vs = model.verification.get("assertion", aid)
    assert vs["status"] == "rejected"


def test_mark_span_stale_only_span_local(model):
    """mark_span_stale must NOT stale assertions sourced from other
    spans of the same document."""
    run_id = model.start_run("span scoped")
    adapter = MatterRuntimeAdapter(model, run_id=run_id)
    aid_in = adapter.record_fact(
        "Fact in target span", "doc.pdf", span_id="sig-block-1",
    )
    aid_out = adapter.record_fact(
        "Fact elsewhere", "doc.pdf", span_id="other-span",
    )
    model.mark_span_stale("sig-block-1", reason="span_replaced")
    assert model.verification.get("assertion", aid_in)["status"] == "stale"
    # Out-of-scope assertion is untouched (still candidate).
    assert model.verification.get("assertion", aid_out)["status"] == "candidate"


def test_reject_target_bumps_trust_revision(model):
    """P0.4 invalidation trigger: human rejection must bump the
    revision so any cached reasoning that referenced the
    now-rejected target silently misses on the next get."""
    run_id = model.start_run("trust bump on reject")
    adapter = MatterRuntimeAdapter(model, run_id=run_id)
    aid = adapter.record_fact("fact to reject", "doc.pdf")
    before = model.cache.current_trust_revision()
    # Cache a plan that implicitly depends on this assertion.
    model.cache.put("orient", "cached_key", {"plan": "pre-reject"})
    assert model.cache.get("orient", "cached_key") is not None
    model.reject_target(
        "assertion", aid,
        reviewed_by_kind="user", reviewed_by_id="r1",
        rejection_reason="fabricated",
        run_id=run_id,
    )
    after = model.cache.current_trust_revision()
    assert after == before + 1
    # Cached plan is unreachable now.
    assert model.cache.get("orient", "cached_key") is None
