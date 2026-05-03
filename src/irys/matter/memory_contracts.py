"""Broker contract types for memory coordination.

These data classes define the structural contracts that bind dependency
manifests, memory packets, and validation results to the broker's freshness,
taint, and profile substrate.
"""

from __future__ import annotations

import hashlib
import json as _json_mod
from dataclasses import dataclass, field
from typing import Any

BROKER_VERSION = "memory_coordination_v15"


def _canonical_json(obj: Any) -> str:
    return _json_mod.dumps(obj, sort_keys=True, separators=(",", ":"))


def _sha256(payload: str) -> str:
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NamespaceDependency:
    namespace: str
    target_kind: str
    target_id: str
    revision: int

    def to_canonical_dict(self) -> dict:
        return {
            "namespace": self.namespace,
            "target_kind": self.target_kind,
            "target_id": self.target_id,
            "revision": self.revision,
        }

    def revision_key(self) -> str:
        if self.target_kind == "*" and self.target_id == "*":
            return f"{self.namespace}:*"
        return f"{self.namespace}:{self.target_kind}:{self.target_id}"

    @classmethod
    def from_dict(cls, d: dict) -> NamespaceDependency:
        return cls(
            namespace=d["namespace"],
            target_kind=d["target_kind"],
            target_id=d["target_id"],
            revision=int(d["revision"]),
        )


@dataclass(frozen=True)
class NegativeDependency:
    namespace: str
    query_predicate: str
    revision: int

    def to_canonical_dict(self) -> dict:
        return {
            "namespace": self.namespace,
            "query_predicate": self.query_predicate,
            "revision": self.revision,
        }

    @classmethod
    def from_dict(cls, d: dict) -> NegativeDependency:
        return cls(
            namespace=d["namespace"],
            query_predicate=d["query_predicate"],
            revision=int(d["revision"]),
        )


@dataclass(frozen=True)
class ObjectDependency:
    target_kind: str
    target_id: str
    row_digest: str | None = None
    row_version: int | None = None
    belief_state: str | None = None
    verification_state: str | None = None
    policy_state: str | None = None

    def to_canonical_dict(self) -> dict:
        d: dict[str, Any] = {
            "target_kind": self.target_kind,
            "target_id": self.target_id,
        }
        if self.row_digest is not None:
            d["row_digest"] = self.row_digest
        if self.row_version is not None:
            d["row_version"] = self.row_version
        if self.belief_state is not None:
            d["belief_state"] = self.belief_state
        if self.verification_state is not None:
            d["verification_state"] = self.verification_state
        if self.policy_state is not None:
            d["policy_state"] = self.policy_state
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ObjectDependency:
        return cls(
            target_kind=d["target_kind"],
            target_id=d["target_id"],
            row_digest=d.get("row_digest"),
            row_version=int(d["row_version"]) if d.get("row_version") is not None else None,
            belief_state=d.get("belief_state"),
            verification_state=d.get("verification_state"),
            policy_state=d.get("policy_state"),
        )


@dataclass(frozen=True)
class DependencyManifest:
    matter_id: str
    purpose: str
    policy_audience: str
    taint_class: str
    domain_profile_id: str
    domain_profile_version: int
    profile_mapping_hash: str
    namespace_dependencies: tuple[NamespaceDependency, ...] = ()
    object_dependencies: tuple[ObjectDependency, ...] = ()
    negative_dependencies: tuple[NegativeDependency, ...] = ()

    def to_canonical_dict(self) -> dict:
        ns_deps = sorted(
            [d.to_canonical_dict() for d in self.namespace_dependencies],
            key=lambda x: (x["namespace"], x["target_kind"], x["target_id"]),
        )
        obj_deps = sorted(
            [d.to_canonical_dict() for d in self.object_dependencies],
            key=lambda x: (x["target_kind"], x["target_id"]),
        )
        neg_deps = sorted(
            [d.to_canonical_dict() for d in self.negative_dependencies],
            key=lambda x: (x["namespace"], x["query_predicate"]),
        )
        return {
            "broker_version": BROKER_VERSION,
            "domain_profile_id": self.domain_profile_id,
            "domain_profile_version": self.domain_profile_version,
            "matter_id": self.matter_id,
            "namespace_dependencies": ns_deps,
            "negative_dependencies": neg_deps,
            "object_dependencies": obj_deps,
            "policy_audience": self.policy_audience,
            "profile_mapping_hash": self.profile_mapping_hash,
            "purpose": self.purpose,
            "taint_class": self.taint_class,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_canonical_dict())

    def manifest_hash(self) -> str:
        return _sha256(self.to_json())

    def namespace_fingerprint_json(self) -> str:
        ns_deps = sorted(
            [d.to_canonical_dict() for d in self.namespace_dependencies],
            key=lambda x: (x["namespace"], x["target_kind"], x["target_id"]),
        )
        return _canonical_json(ns_deps)

    @classmethod
    def from_dict(cls, d: dict) -> DependencyManifest:
        return cls(
            matter_id=d["matter_id"],
            purpose=d["purpose"],
            policy_audience=d["policy_audience"],
            taint_class=d["taint_class"],
            domain_profile_id=d["domain_profile_id"],
            domain_profile_version=int(d["domain_profile_version"]),
            profile_mapping_hash=d["profile_mapping_hash"],
            namespace_dependencies=tuple(
                NamespaceDependency.from_dict(nd)
                for nd in d.get("namespace_dependencies", ())
            ),
            object_dependencies=tuple(
                ObjectDependency.from_dict(od)
                for od in d.get("object_dependencies", ())
            ),
            negative_dependencies=tuple(
                NegativeDependency.from_dict(nd)
                for nd in d.get("negative_dependencies", ())
            ),
        )


@dataclass(frozen=True)
class MemoryPacketSection:
    section_id: str
    section_kind: str
    text: str
    text_hash: str
    token_estimate: int
    materiality: str
    policy_status: str
    selector_reason: str
    object_refs: tuple[str, ...] = ()

    def to_canonical_dict(self) -> dict:
        return {
            "materiality": self.materiality,
            "object_refs": sorted(self.object_refs),
            "policy_status": self.policy_status,
            "section_id": self.section_id,
            "section_kind": self.section_kind,
            "selector_reason": self.selector_reason,
            "text_hash": self.text_hash,
            "token_estimate": self.token_estimate,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_canonical_dict())

    @classmethod
    def from_dict(cls, d: dict) -> MemoryPacketSection:
        return cls(
            section_id=d["section_id"],
            section_kind=d["section_kind"],
            text=d.get("text", ""),
            text_hash=d["text_hash"],
            token_estimate=int(d["token_estimate"]),
            materiality=d["materiality"],
            policy_status=d["policy_status"],
            selector_reason=d["selector_reason"],
            object_refs=tuple(d.get("object_refs", ())),
        )


@dataclass(frozen=True)
class OmittedSection:
    section_kind: str
    reason: str
    count: int = 0
    detail: str | None = None

    def to_canonical_dict(self) -> dict:
        d: dict[str, Any] = {
            "count": self.count,
            "reason": self.reason,
            "section_kind": self.section_kind,
        }
        if self.detail is not None:
            d["detail"] = self.detail
        return d

    @classmethod
    def from_dict(cls, d: dict) -> OmittedSection:
        return cls(
            section_kind=d["section_kind"],
            reason=d["reason"],
            count=int(d.get("count", 0)),
            detail=d.get("detail"),
        )


@dataclass(frozen=True)
class MemoryPacket:
    packet_id: str
    matter_id: str
    request_hash: str
    purpose: str
    policy_audience: str
    taint_class: str
    domain_profile_id: str
    domain_profile_version: int
    profile_mapping_hash: str
    dependency_manifest_hash: str
    sections: tuple[MemoryPacketSection, ...] = ()
    omitted_sections: tuple[OmittedSection, ...] = ()
    answerability_state: str | None = None
    allowed_citation_objects: tuple[str, ...] = ()
    forbidden_internal_guidance_objects: tuple[str, ...] = ()
    run_id: str | None = None
    model_call_id: str | None = None

    def to_canonical_dict(self) -> dict:
        sections = sorted(
            [s.to_canonical_dict() for s in self.sections],
            key=lambda x: x["section_id"],
        )
        omitted = sorted(
            [o.to_canonical_dict() for o in self.omitted_sections],
            key=lambda x: x["section_kind"],
        )
        d: dict[str, Any] = {
            "allowed_citation_objects": sorted(self.allowed_citation_objects),
            "dependency_manifest_hash": self.dependency_manifest_hash,
            "domain_profile_id": self.domain_profile_id,
            "domain_profile_version": self.domain_profile_version,
            "forbidden_internal_guidance_objects": sorted(
                self.forbidden_internal_guidance_objects
            ),
            "matter_id": self.matter_id,
            "omitted_sections": omitted,
            "policy_audience": self.policy_audience,
            "profile_mapping_hash": self.profile_mapping_hash,
            "purpose": self.purpose,
            "request_hash": self.request_hash,
            "sections": sections,
            "taint_class": self.taint_class,
        }
        if self.answerability_state is not None:
            d["answerability_state"] = self.answerability_state
        return d

    def to_json(self) -> str:
        return _canonical_json(self.to_canonical_dict())

    def packet_hash(self) -> str:
        return _sha256(self.to_json())

    @classmethod
    def from_dict(cls, d: dict) -> MemoryPacket:
        return cls(
            packet_id=d["packet_id"],
            matter_id=d["matter_id"],
            request_hash=d["request_hash"],
            purpose=d["purpose"],
            policy_audience=d["policy_audience"],
            taint_class=d["taint_class"],
            domain_profile_id=d["domain_profile_id"],
            domain_profile_version=int(d["domain_profile_version"]),
            profile_mapping_hash=d["profile_mapping_hash"],
            dependency_manifest_hash=d["dependency_manifest_hash"],
            sections=tuple(
                MemoryPacketSection.from_dict(s) for s in d.get("sections", ())
            ),
            omitted_sections=tuple(
                OmittedSection.from_dict(o) for o in d.get("omitted_sections", ())
            ),
            answerability_state=d.get("answerability_state"),
            allowed_citation_objects=tuple(d.get("allowed_citation_objects", ())),
            forbidden_internal_guidance_objects=tuple(
                d.get("forbidden_internal_guidance_objects", ())
            ),
            run_id=d.get("run_id"),
            model_call_id=d.get("model_call_id"),
        )


@dataclass(frozen=True)
class DependencyValidationResult:
    manifest_hash: str
    valid: bool
    status: str
    stale_reasons: tuple[str, ...] = ()
    current_revisions: dict[str, int] = field(default_factory=dict)

    def to_canonical_dict(self) -> dict:
        return {
            "current_revisions": dict(sorted(self.current_revisions.items())),
            "manifest_hash": self.manifest_hash,
            "stale_reasons": list(self.stale_reasons),
            "status": self.status,
            "valid": self.valid,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_canonical_dict())

    @classmethod
    def from_dict(cls, d: dict) -> DependencyValidationResult:
        return cls(
            manifest_hash=d["manifest_hash"],
            valid=bool(d["valid"]),
            status=d["status"],
            stale_reasons=tuple(d.get("stale_reasons", ())),
            current_revisions=dict(d.get("current_revisions", {})),
        )
