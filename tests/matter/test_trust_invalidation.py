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
