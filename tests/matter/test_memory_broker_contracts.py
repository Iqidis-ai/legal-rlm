"""Integration tests for broker contract persistence and validation."""

import pytest

from irys.matter import MatterModel
from irys.matter.graph import MemoryBrokerPolicyError
from irys.matter.memory_contracts import (
    DependencyManifest,
    MemoryPacket,
    MemoryPacketSection,
    NamespaceDependency,
    ObjectDependency,
    OmittedSection,
)


def _setup_broker():
    model = MatterModel.open_in_memory()
    broker = model.memory_broker
    matter_id = model.matter_id
    return model.db, broker, matter_id


def _make_manifest(broker, matter_id, *, ns_keys=("claims:*", "artifacts:*")):
    ns_deps = broker.namespace_dependencies_for_keys(ns_keys)
    profile = broker.get_domain_profile("legal", 1)
    mapping_hash = broker.current_profile_mapping_hash(
        domain_profile_id="legal",
        domain_profile_version=1,
        target_kind="clarification",
        target_namespace="clarifications",
    )
    return DependencyManifest(
        matter_id=matter_id,
        purpose="test_synthesis",
        policy_audience="internal",
        taint_class="clean",
        domain_profile_id="legal",
        domain_profile_version=1,
        profile_mapping_hash=mapping_hash or profile["mapping_hash"],
        namespace_dependencies=tuple(ns_deps),
    )


def test_broker_records_and_validates_dependency_manifest():
    db, broker, matter_id = _setup_broker()
    manifest = _make_manifest(broker, matter_id)
    row_id = broker.record_dependency_manifest(manifest)
    assert row_id

    manifest_hash = manifest.manifest_hash()
    loaded = broker.get_dependency_manifest(manifest_hash)
    assert loaded is not None
    assert loaded.manifest_hash() == manifest_hash
    assert loaded.purpose == "test_synthesis"
    assert loaded.domain_profile_id == "legal"

    result = broker.validate_dependency_manifest(manifest_hash)
    assert result.valid is True
    assert result.status == "valid"


def test_dependency_manifest_validation_fails_after_namespace_bump():
    db, broker, matter_id = _setup_broker()
    manifest = _make_manifest(broker, matter_id)
    broker.record_dependency_manifest(manifest)
    manifest_hash = manifest.manifest_hash()

    result_before = broker.validate_dependency_manifest(manifest_hash)
    assert result_before.valid is True

    broker.bump_namespace_revision("claims")

    result_after = broker.validate_dependency_manifest(manifest_hash)
    assert result_after.valid is False
    assert result_after.status == "stale"
    assert any("claims" in r for r in result_after.stale_reasons)


def test_dependency_manifest_validation_rejects_unknown_taint():
    db, broker, matter_id = _setup_broker()
    ns_deps = broker.namespace_dependencies_for_keys(("claims:*",))
    profile = broker.get_domain_profile("legal", 1)
    manifest = DependencyManifest(
        matter_id=matter_id,
        purpose="test",
        policy_audience="internal",
        taint_class="unknown_taint",
        domain_profile_id="legal",
        domain_profile_version=1,
        profile_mapping_hash=profile["mapping_hash"],
        namespace_dependencies=tuple(ns_deps),
    )
    broker.record_dependency_manifest(manifest)
    result = broker.validate_dependency_manifest(manifest.manifest_hash())
    assert result.valid is False
    assert any("unknown_taint" in r for r in result.stale_reasons)


def test_dependency_manifest_not_found():
    db, broker, matter_id = _setup_broker()
    result = broker.validate_dependency_manifest("sha256:nonexistent")
    assert result.valid is False
    assert result.status == "not_found"


def test_dependency_manifest_taint_class_filter():
    db, broker, matter_id = _setup_broker()
    manifest = _make_manifest(broker, matter_id)
    broker.record_dependency_manifest(manifest)
    manifest_hash = manifest.manifest_hash()

    result = broker.validate_dependency_manifest(
        manifest_hash, allowed_taint_classes={"public_clean"}
    )
    assert result.valid is False
    assert any("taint_class" in r for r in result.stale_reasons)

    result2 = broker.validate_dependency_manifest(
        manifest_hash, allowed_taint_classes={"clean", "public_clean"}
    )
    assert result2.valid is True


def test_dependency_manifest_policy_audience_filter():
    db, broker, matter_id = _setup_broker()
    manifest = _make_manifest(broker, matter_id)
    broker.record_dependency_manifest(manifest)
    manifest_hash = manifest.manifest_hash()

    result = broker.validate_dependency_manifest(
        manifest_hash, required_policy_audience="external"
    )
    assert result.valid is False
    assert any("policy_audience" in r for r in result.stale_reasons)


def test_memory_packet_event_requires_existing_manifest():
    db, broker, matter_id = _setup_broker()
    packet = MemoryPacket(
        packet_id="pk-1",
        matter_id=matter_id,
        request_hash="sha256:req1",
        purpose="test",
        policy_audience="internal",
        taint_class="clean",
        domain_profile_id="legal",
        domain_profile_version=1,
        profile_mapping_hash="sha256:abc",
        dependency_manifest_hash="sha256:no_such_manifest",
    )
    with pytest.raises(MemoryBrokerPolicyError, match="persisted dependency manifest"):
        broker.record_memory_packet_event(packet)


def test_memory_packet_event_round_trip():
    db, broker, matter_id = _setup_broker()
    manifest = _make_manifest(broker, matter_id)
    broker.record_dependency_manifest(manifest)
    manifest_hash = manifest.manifest_hash()

    sec = MemoryPacketSection(
        section_id="s1",
        section_kind="assertions",
        text="test content",
        text_hash="sha256:content",
        token_estimate=5,
        materiality="high",
        policy_status="clean",
        selector_reason="relevant",
        object_refs=("a1",),
    )
    omit = OmittedSection(
        section_kind="privileged",
        reason="taint_filtered",
        count=2,
    )
    profile = broker.get_domain_profile("legal", 1)
    mapping_hash = broker.current_profile_mapping_hash(
        domain_profile_id="legal",
        domain_profile_version=1,
        target_kind="clarification",
        target_namespace="clarifications",
    )
    packet = MemoryPacket(
        packet_id="pk-1",
        matter_id=matter_id,
        request_hash="sha256:req1",
        purpose="synthesis",
        policy_audience="internal",
        taint_class="clean",
        domain_profile_id="legal",
        domain_profile_version=1,
        profile_mapping_hash=mapping_hash or profile["mapping_hash"],
        dependency_manifest_hash=manifest_hash,
        sections=(sec,),
        omitted_sections=(omit,),
        answerability_state="answerable",
        run_id="run-1",
        model_call_id="mc-1",
    )
    row_id = broker.record_memory_packet_event(packet)
    assert row_id

    loaded = broker.get_memory_packet_event(packet_id="pk-1")
    assert loaded is not None
    assert loaded.packet_hash() == packet.packet_hash()
    assert loaded.dependency_manifest_hash == manifest_hash
    assert len(loaded.sections) == 1
    assert loaded.sections[0].section_kind == "assertions"

    loaded_by_hash = broker.get_memory_packet_event(
        packet_hash=packet.packet_hash()
    )
    assert loaded_by_hash is not None


def test_namespace_dependencies_for_keys():
    db, broker, matter_id = _setup_broker()
    broker.bump_namespace_revision("claims")
    broker.bump_namespace_revision("claims")
    broker.bump_namespace_revision("artifacts")

    deps = broker.namespace_dependencies_for_keys(("claims:*", "artifacts:*"))
    by_ns = {d.namespace: d for d in deps}
    assert by_ns["claims"].revision == 2
    assert by_ns["artifacts"].revision == 1


def test_manifest_idempotent_insert():
    db, broker, matter_id = _setup_broker()
    manifest = _make_manifest(broker, matter_id)
    id1 = broker.record_dependency_manifest(manifest)
    id2 = broker.record_dependency_manifest(manifest)
    assert id1 == id2


def test_manifest_idempotent_insert_does_not_bump_namespace():
    db, broker, matter_id = _setup_broker()
    manifest = _make_manifest(broker, matter_id)
    broker.record_dependency_manifest(manifest)
    rev_after_first = broker.get_namespace_revision("dependency_manifests")
    broker.record_dependency_manifest(manifest)
    rev_after_second = broker.get_namespace_revision("dependency_manifests")
    assert rev_after_first == rev_after_second


def test_packet_idempotent_by_hash_returns_existing():
    db, broker, matter_id = _setup_broker()
    manifest = _make_manifest(broker, matter_id)
    broker.record_dependency_manifest(manifest)
    manifest_hash = manifest.manifest_hash()
    profile = broker.get_domain_profile("legal", 1)
    mapping_hash = broker.current_profile_mapping_hash(
        domain_profile_id="legal",
        domain_profile_version=1,
        target_kind="clarification",
        target_namespace="clarifications",
    )

    kwargs = dict(
        matter_id=matter_id,
        request_hash="sha256:req1",
        purpose="synthesis",
        policy_audience="internal",
        taint_class="clean",
        domain_profile_id="legal",
        domain_profile_version=1,
        profile_mapping_hash=mapping_hash or profile["mapping_hash"],
        dependency_manifest_hash=manifest_hash,
    )
    p1 = MemoryPacket(packet_id="pk-1", **kwargs)
    id1 = broker.record_memory_packet_event(p1)
    p2 = MemoryPacket(packet_id="pk-2", **kwargs)
    id2 = broker.record_memory_packet_event(p2)
    assert id1 == id2


def test_packet_idempotent_insert_does_not_bump_namespace():
    db, broker, matter_id = _setup_broker()
    manifest = _make_manifest(broker, matter_id)
    broker.record_dependency_manifest(manifest)
    manifest_hash = manifest.manifest_hash()
    profile = broker.get_domain_profile("legal", 1)
    mapping_hash = broker.current_profile_mapping_hash(
        domain_profile_id="legal",
        domain_profile_version=1,
        target_kind="clarification",
        target_namespace="clarifications",
    )

    p = MemoryPacket(
        packet_id="pk-1",
        matter_id=matter_id,
        request_hash="sha256:req1",
        purpose="synthesis",
        policy_audience="internal",
        taint_class="clean",
        domain_profile_id="legal",
        domain_profile_version=1,
        profile_mapping_hash=mapping_hash or profile["mapping_hash"],
        dependency_manifest_hash=manifest_hash,
    )
    broker.record_memory_packet_event(p)
    rev_after_first = broker.get_namespace_revision("memory_packets")
    broker.record_memory_packet_event(p)
    rev_after_second = broker.get_namespace_revision("memory_packets")
    assert rev_after_first == rev_after_second


def test_packet_event_persists_section_text():
    db, broker, matter_id = _setup_broker()
    manifest = _make_manifest(broker, matter_id)
    broker.record_dependency_manifest(manifest)
    manifest_hash = manifest.manifest_hash()
    profile = broker.get_domain_profile("legal", 1)
    mapping_hash = broker.current_profile_mapping_hash(
        domain_profile_id="legal",
        domain_profile_version=1,
        target_kind="clarification",
        target_namespace="clarifications",
    )

    sec = MemoryPacketSection(
        section_id="s1",
        section_kind="assertions",
        text="important legal text that must survive persistence",
        text_hash="sha256:content",
        token_estimate=10,
        materiality="high",
        policy_status="clean",
        selector_reason="relevant",
    )
    p = MemoryPacket(
        packet_id="pk-1",
        matter_id=matter_id,
        request_hash="sha256:req1",
        purpose="synthesis",
        policy_audience="internal",
        taint_class="clean",
        domain_profile_id="legal",
        domain_profile_version=1,
        profile_mapping_hash=mapping_hash or profile["mapping_hash"],
        dependency_manifest_hash=manifest_hash,
        sections=(sec,),
    )
    broker.record_memory_packet_event(p)
    loaded = broker.get_memory_packet_event(packet_id="pk-1")
    assert loaded is not None
    assert loaded.sections[0].text == "important legal text that must survive persistence"


def test_validation_checks_profile_existence():
    db, broker, matter_id = _setup_broker()
    ns_deps = broker.namespace_dependencies_for_keys(("claims:*",))
    manifest = DependencyManifest(
        matter_id=matter_id,
        purpose="test",
        policy_audience="internal",
        taint_class="clean",
        domain_profile_id="nonexistent_domain",
        domain_profile_version=99,
        profile_mapping_hash="sha256:fake",
        namespace_dependencies=tuple(ns_deps),
    )
    broker.record_dependency_manifest(manifest)
    result = broker.validate_dependency_manifest(manifest.manifest_hash())
    assert result.valid is False
    assert any("domain_profile" in r for r in result.stale_reasons)


def test_validation_checks_mapping_hash():
    db, broker, matter_id = _setup_broker()
    ns_deps = broker.namespace_dependencies_for_keys(("claims:*",))
    manifest = DependencyManifest(
        matter_id=matter_id,
        purpose="test",
        policy_audience="internal",
        taint_class="clean",
        domain_profile_id="legal",
        domain_profile_version=1,
        profile_mapping_hash="sha256:nonexistent_mapping",
        namespace_dependencies=tuple(ns_deps),
    )
    broker.record_dependency_manifest(manifest)
    result = broker.validate_dependency_manifest(manifest.manifest_hash())
    assert result.valid is False
    assert any("profile_mapping_hash" in r for r in result.stale_reasons)
