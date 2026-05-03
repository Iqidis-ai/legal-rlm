"""Tests for multi-domain profile registration, hashes, and mappings."""

import json

import pytest

from irys.matter import MatterModel
from irys.matter.domain_profiles import (
    DOMAIN_PROFILE_IDS,
    DOMAIN_PROFILE_VERSION,
    MAPPING_TARGET_PAIRS,
    _CANONICAL_KERNEL_KEYS,
    _CROSS_DOMAIN_PAIRS,
    canonical_profile_hash,
    canonical_profile_json,
    iter_builtin_domain_profiles,
    iter_builtin_profile_mappings,
)


def _open_model():
    return MatterModel.open_in_memory()


def test_matter_open_installs_five_profiles():
    model = _open_model()
    broker = model.memory_broker
    for pid in DOMAIN_PROFILE_IDS:
        profile = broker.get_domain_profile(pid, DOMAIN_PROFILE_VERSION)
        assert profile is not None, f"Profile {pid}:{DOMAIN_PROFILE_VERSION} not installed"


def test_legal_profile_unchanged():
    model = _open_model()
    broker = model.memory_broker
    profile = broker.get_domain_profile("legal", 1)
    assert profile is not None
    expected_json = broker.default_legal_profile_json()
    expected_hash = broker.default_legal_profile_hash()
    assert profile["profile_json"] == expected_json
    assert profile["mapping_hash"] == expected_hash


def test_non_legal_profiles_have_required_top_level_keys():
    required_keys = {
        "broker_protocol", "profile_id", "profile_kind", "profile_version",
        "neutral_kernel", "source_roles", "belief_states",
        "trust_weights", "taint_classes", "speech_acts",
    }
    for pid in DOMAIN_PROFILE_IDS:
        if pid == "legal":
            continue
        pjson = canonical_profile_json(pid)
        data = json.loads(pjson)
        missing = required_keys - set(data.keys())
        assert not missing, f"Profile {pid} missing keys: {missing}"


def test_all_profiles_have_canonical_neutral_kernel_keys():
    # Legal profile has a different kernel shape (v14 legacy) — check non-legal only
    for pid in DOMAIN_PROFILE_IDS:
        if pid == "legal":
            model = _open_model()
            data = json.loads(model.memory_broker.default_legal_profile_json())
            kernel = data["neutral_kernel"]
            shared_keys = set(kernel.keys()) & _CANONICAL_KERNEL_KEYS
            assert len(shared_keys) >= 6, (
                f"Legal profile should share most canonical keys, has: {shared_keys}"
            )
            continue
        data = json.loads(canonical_profile_json(pid))
        kernel = data["neutral_kernel"]
        assert set(kernel.keys()) == _CANONICAL_KERNEL_KEYS, (
            f"Profile {pid} neutral_kernel keys mismatch: {set(kernel.keys())} != {_CANONICAL_KERNEL_KEYS}"
        )


def test_identity_mappings_exist_for_all_profiles():
    model = _open_model()
    broker = model.memory_broker
    for pid in DOMAIN_PROFILE_IDS:
        for target_kind, target_namespace in MAPPING_TARGET_PAIRS:
            mappings = broker.list_profile_mappings(
                pid, target_kind=target_kind, target_namespace=target_namespace
            )
            identity = [
                m for m in mappings
                if m["source_domain_profile_id"] == pid
                and m["target_domain_profile_id"] == pid
                and m["compatibility_status"] == "identity"
            ]
            assert len(identity) >= 1, (
                f"Missing identity mapping for {pid} -> {target_kind}/{target_namespace}"
            )


def test_cross_domain_compatible_mappings_exist():
    model = _open_model()
    broker = model.memory_broker
    for src, tgt in _CROSS_DOMAIN_PAIRS:
        for target_kind, target_namespace in MAPPING_TARGET_PAIRS:
            fwd = broker.list_profile_mappings(
                tgt, target_kind=target_kind, target_namespace=target_namespace
            )
            fwd_compat = [
                m for m in fwd
                if m["source_domain_profile_id"] == src
                and m["compatibility_status"] == "compatible"
            ]
            assert len(fwd_compat) >= 1, (
                f"Missing compatible mapping {src}->{tgt} for {target_kind}/{target_namespace}"
            )
            rev = broker.list_profile_mappings(
                src, target_kind=target_kind, target_namespace=target_namespace
            )
            rev_compat = [
                m for m in rev
                if m["source_domain_profile_id"] == tgt
                and m["compatibility_status"] == "compatible"
            ]
            assert len(rev_compat) >= 1, (
                f"Missing compatible mapping {tgt}->{src} for {target_kind}/{target_namespace}"
            )


def test_answer_clarification_works_with_legal():
    model = _open_model()
    q_id = model.clarifications.add_question("Is there a contract?")
    model.answer_clarification(q_id, "Yes.")
    answered = model.clarifications.get_answered()
    assert any(a["id"] == q_id for a in answered)


def test_synthetic_clarification_with_finance():
    model = _open_model()
    q_id = model.clarifications.add_question("Is revenue recognized ratably?")
    model.answer_clarification(
        q_id,
        "Yes, per ASC 606.",
        domain_profile_id="finance",
        domain_profile_version=1,
    )
    answered = model.clarifications.get_answered()
    assert any(a["id"] == q_id for a in answered)


def test_missing_mapping_fails_closed():
    from irys.matter.graph import MemoryBrokerPolicyError

    model = _open_model()
    broker = model.memory_broker
    broker.upsert_domain_profile(
        profile_id="alien",
        profile_version=1,
        profile_kind="alien",
        profile_json='{"test":true}',
        mapping_hash="sha256:alien1",
    )
    q_id = model.clarifications.add_question("Does it work?")
    with pytest.raises(MemoryBrokerPolicyError):
        model.answer_clarification(
            q_id,
            "No.",
            domain_profile_id="alien",
            domain_profile_version=1,
        )


def test_namespace_coverage_checker_passes():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "tools/check_memory_namespace_coverage.py"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Namespace coverage check failed:\n{result.stdout}\n{result.stderr}"


def test_profile_hashes_are_deterministic():
    for pid in DOMAIN_PROFILE_IDS:
        if pid == "legal":
            continue
        h1 = canonical_profile_hash(pid)
        h2 = canonical_profile_hash(pid)
        assert h1 == h2
        assert h1.startswith("sha256:")


def test_legal_profile_json_raises_for_canonical():
    with pytest.raises(ValueError, match="Legal profile JSON is owned"):
        canonical_profile_json("legal")


def test_vocabulary_reader_finance():
    model = _open_model()
    broker = model.memory_broker
    tw = broker.get_profile_trust_weights("finance")
    assert "auditor" in tw
    assert tw["auditor"] == 0.9
    assert tw["investor"] == 0.35

    roles = broker.get_profile_source_roles("finance")
    assert "auditor" in roles
    assert "regulator" in roles
    assert len(roles) == 10

    states = broker.get_profile_belief_states("finance")
    assert "audited" in states
    assert "unsupported" in states

    taints = broker.get_profile_taint_classes("finance")
    assert "material_nonpublic" in taints
    assert "unknown_taint" in taints

    acts = broker.get_profile_speech_acts("finance")
    assert "audit_opinion" in acts


def test_vocabulary_reader_coding():
    model = _open_model()
    broker = model.memory_broker
    tw = broker.get_profile_trust_weights("coding")
    assert tw["automated_test"] == 0.85
    assert tw["issue_reporter"] == 0.38

    roles = broker.get_profile_source_roles("coding")
    assert "ci_system" in roles
    assert "security_scanner" in roles


def test_vocabulary_reader_returns_empty_for_missing():
    model = _open_model()
    broker = model.memory_broker
    assert broker.get_profile_trust_weights("nonexistent") == {}
    assert broker.get_profile_source_roles("nonexistent") == []
    assert broker.get_profile_belief_states("nonexistent") == []
    assert broker.get_profile_taint_classes("nonexistent") == []
    assert broker.get_profile_speech_acts("nonexistent") == []


def test_vocabulary_reader_all_profiles():
    model = _open_model()
    broker = model.memory_broker
    for pid in DOMAIN_PROFILE_IDS:
        vocab = broker.get_profile_vocabulary(pid)
        assert vocab is not None, f"Profile {pid} vocabulary should exist"
        assert "neutral_kernel" in vocab
        if pid != "legal":
            assert "trust_weights" in vocab
            assert "source_roles" in vocab
            assert len(vocab["trust_weights"]) >= 5
            assert len(vocab["source_roles"]) >= 5
