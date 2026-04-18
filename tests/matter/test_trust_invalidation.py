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


def test_set_trust_override_bumps_trust_revision(model):
    """Adv#11 Fix 1: setting a trust override must bump trust_revision
    so reasoning_cache / cascade-decision entries keyed on the prior
    revision become unreachable. Without this, the governance router
    could serve a stale route decision after a source was downgraded."""
    before = model.cache.current_trust_revision()
    model.set_trust_override("example.pdf", "low", note="audit")
    assert model.cache.current_trust_revision() == before + 1


def test_delete_trust_override_bumps_trust_revision(model):
    """Adv#11 Fix 1: deleting an existing trust override also bumps
    trust_revision so the posture reversal invalidates cached plans
    made while the override was in force. A delete of a missing
    override stays a no-op — no spurious bump."""
    model.set_trust_override("example.pdf", "low", note="audit")
    after_set = model.cache.current_trust_revision()
    model.delete_trust_override("example.pdf")
    assert model.cache.current_trust_revision() == after_set + 1
    # Deleting a missing override must not bump.
    before_noop = model.cache.current_trust_revision()
    model.delete_trust_override("not_an_override.pdf")
    assert model.cache.current_trust_revision() == before_noop


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


def test_update_hash_detects_real_change(model):
    """P0.4: update_hash returns (True, old_sha) when an actual
    content hash replaces a prior real hash — the signal engine
    code uses to trigger mark_document_stale."""
    inv_id, _ = model.inventory.upsert("doc.pdf", "a" * 64, size_bytes=100)
    changed, old = model.inventory.update_hash(inv_id, "b" * 64, size_bytes=200)
    assert changed is True
    assert old == "a" * 64


def test_update_hash_does_not_clobber_real_with_pending(model):
    """Guard: a real hash must never be overwritten with the
    'pending' placeholder. Returns (False, real_old)."""
    inv_id, _ = model.inventory.upsert("doc.pdf", "a" * 64, size_bytes=100)
    changed, old = model.inventory.update_hash(inv_id, "pending", size_bytes=0)
    assert changed is False
    assert old == "a" * 64


def test_update_hash_first_real_hash_after_pending_is_not_a_change(model):
    """Replacing 'pending' with a first real hash is just placeholder
    replacement, not a content change."""
    inv_id, _ = model.inventory.upsert("doc.pdf", "pending", size_bytes=0)
    changed, old = model.inventory.update_hash(inv_id, "a" * 64, size_bytes=100)
    assert changed is False
    assert old == "pending"


def test_reclassify_privilege_stales_downstream_on_flag_flip(model):
    """Attorney flips privilege — downstream direct dependents are
    staled and trust_revision bumps so clean synthesis re-evaluates."""
    run_id = model.start_run("priv flip")
    adapter = MatterRuntimeAdapter(model, run_id=run_id)
    inv_id, _ = model.inventory.upsert("internal.docx", "a" * 64, size_bytes=1)
    model.document_cards.upsert(
        doc_id=inv_id, title="memo", doc_type="internal", privilege_flag=False,
    )
    aid = adapter.record_fact("Internal fact", "internal.docx")
    before_rev = model.cache.current_trust_revision()
    touched = model.reclassify_privilege(
        "internal.docx", new_flag=True,
        reviewed_by_kind="attorney", reviewed_by_id="a1",
    )
    assert touched >= 1
    assert model.verification.get("assertion", aid)["status"] == "stale"
    assert model.cache.current_trust_revision() == before_rev + 1


def test_reclassify_privilege_no_op_when_flag_unchanged(model):
    """Re-asserting the same classification must not stale downstream
    or bump trust_revision."""
    run_id = model.start_run("priv no-op")
    adapter = MatterRuntimeAdapter(model, run_id=run_id)
    inv_id, _ = model.inventory.upsert("d.pdf", "a" * 64, size_bytes=1)
    model.document_cards.upsert(
        doc_id=inv_id, title="doc", doc_type="contract", privilege_flag=False,
    )
    aid = adapter.record_fact("A fact", "d.pdf")
    before_rev = model.cache.current_trust_revision()
    touched = model.reclassify_privilege(
        "d.pdf", new_flag=False,
        reviewed_by_kind="attorney", reviewed_by_id="a1",
    )
    assert touched == 0
    assert model.verification.get("assertion", aid)["status"] == "candidate"
    assert model.cache.current_trust_revision() == before_rev


def test_re_extraction_revives_stale_occurrence(model):
    """P0.4 review fix #1: re-extracting an existing (doc, span)
    occurrence slot must revive its stale verification to candidate.
    Without this, a document re-ingest leaves every occurrence stale
    forever even though the same text is extracted again."""
    from irys.matter.enums import VerificationTargetKind

    run_id = model.start_run("occurrence revival")
    adapter = MatterRuntimeAdapter(model, run_id=run_id)
    aid = adapter.record_fact(
        "Same fact text", "doc.pdf", span_id="para-1",
    )
    occ = model.db.execute(
        "SELECT id FROM assertion_occurrence WHERE assertion_id=? LIMIT 1",
        (aid,),
    ).fetchone()
    # Mark the occurrence stale.
    model.verification.mark_stale(
        VerificationTargetKind.ASSERTION_OCCURRENCE, occ["id"],
        stale_reason="test",
    )
    assert model.verification.get("assertion_occurrence", occ["id"])["status"] == "stale"
    # Re-extract: same doc + same span + same text → UPDATE path.
    adapter.record_fact("Same fact text", "doc.pdf", span_id="para-1")
    assert model.verification.get("assertion_occurrence", occ["id"])["status"] == "candidate", (
        "existing-occurrence UPDATE path must revive stale to candidate"
    )


def test_re_extraction_revives_stale_edge(model):
    """P0.4 review fix #1: re-extracting the same assertion→issue
    link must revive a stale edge to candidate."""
    from irys.matter.enums import IssueType, VerificationTargetKind

    iid, _ = model.issues.upsert_issue("Claim", IssueType.CLAIM, materiality=0.7)
    run_id = model.start_run("edge revival")
    adapter = MatterRuntimeAdapter(model, run_id=run_id)
    aid = adapter.record_fact(
        "supporting fact", "doc.pdf",
        issue_id=iid, issue_link_type="supports",
    )
    edge = model.db.execute(
        "SELECT id FROM evidence_edge WHERE source_id=? AND target_id=?",
        (aid, iid),
    ).fetchone()
    # Stale the edge.
    model.verification.mark_stale(
        VerificationTargetKind.EVIDENCE_EDGE, edge["id"],
        stale_reason="test",
    )
    assert model.verification.get("evidence_edge", edge["id"])["status"] == "stale"
    # Re-extract: same assertion, same issue, same relation → existing-edge path.
    adapter.record_fact(
        "supporting fact", "doc.pdf",
        issue_id=iid, issue_link_type="supports",
    )
    assert model.verification.get("evidence_edge", edge["id"])["status"] == "candidate", (
        "existing-edge path must revive stale to candidate"
    )


def test_re_extraction_revives_stale_quant(model):
    """P0.4 review fix #1: re-recording the same numeric fact must
    revive its stale verification to candidate."""
    from irys.matter.enums import VerificationTargetKind

    run_id = model.start_run("quant revival")
    adapter = MatterRuntimeAdapter(model, run_id=run_id)
    qid = adapter.record_quant(
        quant_kind="amount", raw_text="$500", amount_value=500.0,
    )
    model.verification.mark_stale(
        VerificationTargetKind.QUANT_FACT, qid, stale_reason="test",
    )
    assert model.verification.get("quant_fact", qid)["status"] == "stale"
    # Same quant — dedup hits existing row.
    qid2 = adapter.record_quant(
        quant_kind="amount", raw_text="$500", amount_value=500.0,
    )
    assert qid2 == qid  # same row
    assert model.verification.get("quant_fact", qid)["status"] == "candidate"


def test_re_upsert_revives_stale_authority(model):
    """P0.4 review fix #1: re-citing the same case must revive a
    stale authority to candidate."""
    from irys.matter.enums import VerificationTargetKind

    aid, is_new = model.authority.upsert("Smith v. Jones, 1 F.3d 100")
    assert is_new
    model.verification.mark_stale(
        VerificationTargetKind.AUTHORITY, aid, stale_reason="test",
    )
    assert model.verification.get("authority", aid)["status"] == "stale"
    # Same citation → UPDATE path.
    aid2, is_new2 = model.authority.upsert("Smith v. Jones, 1 F.3d 100")
    assert aid2 == aid and not is_new2
    assert model.verification.get("authority", aid)["status"] == "candidate"


def test_mark_document_stale_catches_bare_span_quants(model):
    """P0.4 review fix #2: quants pinned to a bare span_id string
    (no `span` table row) must still be staled by
    mark_document_stale when the span belongs to a doc occurrence."""
    run_id = model.start_run("bare span quant")
    adapter = MatterRuntimeAdapter(model, run_id=run_id)
    model.inventory.upsert("contract.pdf", "a" * 64, size_bytes=1)
    aid = adapter.record_fact(
        "fact with span", "contract.pdf", span_id="para-3",
    )
    # Runtime records a quant pinned to the same bare span_id but
    # never writes a `span` table row — only assertion_occurrence
    # knows that span_id exists.
    qid = model.quant.record(
        quant_kind="amount", raw_text="$500",
        amount_value=500.0, span_id="para-3",
    )
    model.mark_document_stale("contract.pdf", reason="hash_change")
    assert model.verification.get("assertion", aid)["status"] == "stale"
    assert model.verification.get("quant_fact", qid)["status"] == "stale", (
        "bare-span quant must be swept by document invalidation"
    )


def test_reject_assertion_stales_its_edges_and_quants(model):
    """Adversarial #7 finding #4 (SO-2 killer): rejecting a
    supporting assertion must stale its dependent edges and quants.
    Previously proof was recomputed but the companion edge stayed
    candidate and silently re-entered consumers that read from the
    edge substrate without checking the assertion's verification."""
    from irys.matter.enums import IssueType, VerificationTargetKind

    iid, _ = model.issues.upsert_issue("Claim", IssueType.CLAIM, materiality=0.7)
    run_id = model.start_run("reject fanout")
    adapter = MatterRuntimeAdapter(model, run_id=run_id)
    aid = adapter.record_fact(
        "Support fact", "doc.pdf",
        issue_id=iid, issue_link_type="supports",
    )
    qid = adapter.record_quant(
        quant_kind="amount", raw_text="$100",
        amount_value=100.0, assertion_id=aid,
    )
    edge = model.db.execute(
        "SELECT id FROM evidence_edge WHERE source_id=? AND target_id=?",
        (aid, iid),
    ).fetchone()
    model.reject_target(
        "assertion", aid,
        reviewed_by_kind="user", reviewed_by_id="r1",
        rejection_reason="fabricated",
        run_id=run_id,
    )
    # Dependent edge and quant must now be stale.
    assert model.verification.get("evidence_edge", edge["id"])["status"] == "stale"
    assert model.verification.get("quant_fact", qid)["status"] == "stale"


def test_card_provenance_carries_llm_fields_via_active_context(model):
    """Adversarial #7 finding #2: document_card provenance rows must
    carry llm_call_id + model_id + prompt_hash when ACTIVE_LLM_CALL
    is set, matching what assertion/edge/quant/authority rows already
    do."""
    from irys.core.models import ACTIVE_LLM_CALL

    inv_id, _ = model.inventory.upsert(
        "doc.pdf", "a" * 64, size_bytes=1,
    )
    token = ACTIVE_LLM_CALL.set({
        "call_id": "call_card",
        "model_id": "gemini-flash",
        "model_tier": "FLASH",
        "prompt_hash": "h" * 64,
    })
    try:
        run_id = model.start_run("card provenance")
        model.upsert_document_profile(
            "doc.pdf",
            analysis={"doc_type": "contract", "doc_source_role": "operative"},
            run_id=run_id,
        )
    finally:
        ACTIVE_LLM_CALL.reset(token)
    card_row = model.db.execute(
        "SELECT id FROM document_card WHERE doc_id=?", (inv_id,),
    ).fetchone()
    events = model.get_provenance("document_card", card_row["id"])
    assert events, "card must have at least one provenance event"
    ev = events[0]
    assert ev["llm_call_id"] == "call_card"
    assert ev["model_id"] == "gemini-flash"
    assert ev["prompt_hash"] == "h" * 64


def test_verify_assertion_also_verifies_companion_edges(model):
    """Adversarial #8 fix: the attorney clicks Verify on a fact in
    the Review Inbox. For the verified-coverage bar to move, the
    companion evidence_edge(s) must also promote. Previously only
    the assertion lane moved, TrustPolicy required both, and the
    UI appeared to lie about success."""
    from irys.matter.enums import IssueType, VerificationTargetKind

    iid, _ = model.issues.upsert_issue("Claim", IssueType.CLAIM, materiality=0.7)
    run_id = model.start_run("verify fan-out to edge")
    adapter = MatterRuntimeAdapter(model, run_id=run_id)
    aid = adapter.record_fact(
        "Support fact", "doc.pdf",
        issue_id=iid, issue_link_type="supports",
    )
    edge = model.db.execute(
        "SELECT id FROM evidence_edge WHERE source_id=? AND target_id=?",
        (aid, iid),
    ).fetchone()
    model.verify_target(
        "assertion", aid,
        reviewed_by_kind="user", reviewed_by_id="r1",
    )
    # BOTH lanes must be verified after a single Verify click on the
    # assertion target.
    assert model.verification.get("assertion", aid)["status"] == "verified"
    assert model.verification.get("evidence_edge", edge["id"])["status"] == "verified"
    # And the coverage report reflects the change: verified lane > 0.
    report = model.get_issue_coverage_report(policy_audience="internal")
    entry = next(r for r in report if r["id"] == iid)
    assert entry["verified_supporting_count"] == 1


def test_verify_assertion_does_not_promote_user_rejected_edge(model):
    """A user who explicitly rejected a system-inferred edge should
    not have that rejection quietly reversed when they later verify
    the parent assertion."""
    from irys.matter.enums import (
        IssueType, VerificationTargetKind,
    )

    iid, _ = model.issues.upsert_issue("Claim", IssueType.CLAIM, materiality=0.7)
    run_id = model.start_run("respect rejected edge")
    adapter = MatterRuntimeAdapter(model, run_id=run_id)
    aid = adapter.record_fact(
        "Support fact", "doc.pdf",
        issue_id=iid, issue_link_type="supports",
    )
    edge = model.db.execute(
        "SELECT id FROM evidence_edge WHERE source_id=? AND target_id=?",
        (aid, iid),
    ).fetchone()
    # User rejects the edge first.
    model.verification.reject(
        VerificationTargetKind.EVIDENCE_EDGE, edge["id"],
        reviewed_by_kind="user", reviewed_by_id="r1",
        rejection_reason="edge misinterprets the fact",
    )
    # Then verifies the parent assertion.
    model.verify_target(
        "assertion", aid,
        reviewed_by_kind="user", reviewed_by_id="r1",
    )
    # The rejected edge must stay rejected.
    assert model.verification.get("evidence_edge", edge["id"])["status"] == "rejected"


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
