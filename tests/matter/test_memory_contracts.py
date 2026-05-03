"""Tests for broker contract types: deterministic hashes, round-trip, profile binding."""

import json

from irys.matter.memory_contracts import (
    BROKER_VERSION,
    DependencyManifest,
    DependencyValidationResult,
    MemoryPacket,
    MemoryPacketSection,
    NamespaceDependency,
    NegativeDependency,
    ObjectDependency,
    OmittedSection,
)


def _make_manifest(
    *,
    matter_id="m1",
    purpose="test",
    policy_audience="internal",
    taint_class="clean",
    profile_id="legal",
    profile_version=1,
    mapping_hash="sha256:abc",
    ns_deps=(),
    obj_deps=(),
    neg_deps=(),
):
    return DependencyManifest(
        matter_id=matter_id,
        purpose=purpose,
        policy_audience=policy_audience,
        taint_class=taint_class,
        domain_profile_id=profile_id,
        domain_profile_version=profile_version,
        profile_mapping_hash=mapping_hash,
        namespace_dependencies=tuple(ns_deps),
        object_dependencies=tuple(obj_deps),
        negative_dependencies=tuple(neg_deps),
    )


def test_dependency_manifest_hash_is_canonical_and_profile_bound():
    ns1 = NamespaceDependency("claims", "*", "*", 3)
    ns2 = NamespaceDependency("artifacts", "*", "*", 1)

    m1 = _make_manifest(ns_deps=[ns1, ns2])
    m2 = _make_manifest(ns_deps=[ns2, ns1])
    assert m1.manifest_hash() == m2.manifest_hash(), "Order should not affect hash"

    m3 = _make_manifest(ns_deps=[ns1, ns2], profile_id="finance")
    assert m1.manifest_hash() != m3.manifest_hash(), "Profile must affect identity"

    m4 = _make_manifest(ns_deps=[ns1, ns2], profile_version=2)
    assert m1.manifest_hash() != m4.manifest_hash(), "Profile version must affect identity"

    m5 = _make_manifest(ns_deps=[ns1, ns2], mapping_hash="sha256:xyz")
    assert m1.manifest_hash() != m5.manifest_hash(), "Mapping hash must affect identity"


def test_dependency_manifest_round_trip():
    ns = NamespaceDependency("claims", "assertion", "a1", 5)
    obj = ObjectDependency("assertion", "a1", row_digest="sha256:row1", belief_state="operative")
    neg = NegativeDependency("gaps", "kind=factual", 0)
    m = _make_manifest(ns_deps=[ns], obj_deps=[obj], neg_deps=[neg])

    d = m.to_canonical_dict()
    j = m.to_json()
    parsed = json.loads(j)
    assert parsed == d

    m2 = DependencyManifest.from_dict(parsed)
    assert m2.manifest_hash() == m.manifest_hash()
    assert m2.namespace_dependencies[0].revision == 5
    assert m2.object_dependencies[0].belief_state == "operative"
    assert m2.negative_dependencies[0].query_predicate == "kind=factual"


def test_dependency_manifest_canonical_json_is_deterministic():
    m = _make_manifest(
        ns_deps=[
            NamespaceDependency("z_ns", "*", "*", 1),
            NamespaceDependency("a_ns", "*", "*", 2),
        ]
    )
    j1 = m.to_json()
    j2 = m.to_json()
    assert j1 == j2
    parsed = json.loads(j1)
    assert parsed["namespace_dependencies"][0]["namespace"] == "a_ns"
    assert parsed["namespace_dependencies"][1]["namespace"] == "z_ns"


def test_memory_packet_hash_binds_manifest_sections_and_taint():
    sec = MemoryPacketSection(
        section_id="s1",
        section_kind="assertions",
        text="some text",
        text_hash="sha256:text1",
        token_estimate=10,
        materiality="high",
        policy_status="clean",
        selector_reason="test",
        object_refs=("a1", "a2"),
    )
    omit = OmittedSection(
        section_kind="privileged",
        reason="taint_filtered",
        count=3,
    )

    p1 = MemoryPacket(
        packet_id="p1",
        matter_id="m1",
        request_hash="sha256:req1",
        purpose="synthesis",
        policy_audience="internal",
        taint_class="clean",
        domain_profile_id="legal",
        domain_profile_version=1,
        profile_mapping_hash="sha256:abc",
        dependency_manifest_hash="sha256:manifest1",
        sections=(sec,),
        omitted_sections=(omit,),
    )

    p2 = MemoryPacket(
        packet_id="p2",
        matter_id="m1",
        request_hash="sha256:req1",
        purpose="synthesis",
        policy_audience="internal",
        taint_class="internal_work_product",
        domain_profile_id="legal",
        domain_profile_version=1,
        profile_mapping_hash="sha256:abc",
        dependency_manifest_hash="sha256:manifest1",
        sections=(sec,),
        omitted_sections=(omit,),
    )

    assert p1.packet_hash() != p2.packet_hash(), "Taint class must affect packet hash"

    p3 = MemoryPacket(
        packet_id="p3",
        matter_id="m1",
        request_hash="sha256:req1",
        purpose="synthesis",
        policy_audience="internal",
        taint_class="clean",
        domain_profile_id="finance",
        domain_profile_version=1,
        profile_mapping_hash="sha256:abc",
        dependency_manifest_hash="sha256:manifest1",
        sections=(sec,),
        omitted_sections=(omit,),
    )
    assert p1.packet_hash() != p3.packet_hash(), "Profile must affect packet hash"

    sec2 = MemoryPacketSection(
        section_id="s1",
        section_kind="assertions",
        text="different text",
        text_hash="sha256:text2",
        token_estimate=10,
        materiality="high",
        policy_status="clean",
        selector_reason="test",
        object_refs=("a1", "a2"),
    )
    p4 = MemoryPacket(
        packet_id="p4",
        matter_id="m1",
        request_hash="sha256:req1",
        purpose="synthesis",
        policy_audience="internal",
        taint_class="clean",
        domain_profile_id="legal",
        domain_profile_version=1,
        profile_mapping_hash="sha256:abc",
        dependency_manifest_hash="sha256:manifest1",
        sections=(sec2,),
        omitted_sections=(omit,),
    )
    assert p1.packet_hash() != p4.packet_hash(), "Section content must affect packet hash"


def test_memory_packet_round_trip():
    sec = MemoryPacketSection(
        section_id="s1",
        section_kind="assertions",
        text="hello",
        text_hash="sha256:hello",
        token_estimate=5,
        materiality="high",
        policy_status="clean",
        selector_reason="relevant",
        object_refs=("a1",),
    )
    p = MemoryPacket(
        packet_id="pk1",
        matter_id="m1",
        request_hash="sha256:req",
        purpose="orient",
        policy_audience="external",
        taint_class="public_clean",
        domain_profile_id="legal",
        domain_profile_version=1,
        profile_mapping_hash="sha256:map",
        dependency_manifest_hash="sha256:dm",
        sections=(sec,),
        omitted_sections=(),
        answerability_state="answerable",
        allowed_citation_objects=("a1",),
        forbidden_internal_guidance_objects=("g1",),
        run_id="run1",
        model_call_id="mc1",
    )
    j = p.to_json()
    d = json.loads(j)
    p2 = MemoryPacket.from_dict({**d, "packet_id": "pk1", "run_id": "run1", "model_call_id": "mc1"})
    assert p2.packet_hash() == p.packet_hash()
    assert p2.answerability_state == "answerable"
    assert p2.run_id == "run1"


def test_validation_result_round_trip():
    vr = DependencyValidationResult(
        manifest_hash="sha256:test",
        valid=False,
        status="stale",
        stale_reasons=("namespace claims:*: expected 1, current 2",),
        current_revisions={"claims:*": 2},
    )
    d = vr.to_canonical_dict()
    j = vr.to_json()
    vr2 = DependencyValidationResult.from_dict(json.loads(j))
    assert vr2.valid is False
    assert vr2.stale_reasons == vr.stale_reasons
    assert vr2.current_revisions == {"claims:*": 2}


def test_packet_id_excluded_from_hash():
    """packet_id is a generated id — two packets with different ids but same content must hash equal."""
    kwargs = dict(
        matter_id="m1",
        request_hash="sha256:req",
        purpose="test",
        policy_audience="internal",
        taint_class="clean",
        domain_profile_id="legal",
        domain_profile_version=1,
        profile_mapping_hash="sha256:map",
        dependency_manifest_hash="sha256:dm",
    )
    p1 = MemoryPacket(packet_id="id-aaa", **kwargs)
    p2 = MemoryPacket(packet_id="id-bbb", **kwargs)
    assert p1.packet_hash() == p2.packet_hash()


def test_namespace_dependency_revision_key():
    d1 = NamespaceDependency("claims", "*", "*", 1)
    assert d1.revision_key() == "claims:*"
    d2 = NamespaceDependency("claims", "assertion", "a1", 2)
    assert d2.revision_key() == "claims:assertion:a1"
