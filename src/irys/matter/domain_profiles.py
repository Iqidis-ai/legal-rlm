"""Data-driven domain profile templates for multi-domain matter models.

The legal profile is NOT defined here — it lives in
MemoryBrokerStore.default_legal_profile_json() and must not change
(doing so would alter the legal:1 hash).
"""

from __future__ import annotations

import hashlib
import json as _json_mod
from typing import Iterator

DOMAIN_PROFILE_VERSION = 1
DOMAIN_PROFILE_IDS = ("legal", "finance", "coding", "academic_research", "biomedical")

MAPPING_TARGET_PAIRS = (
    ("claim", "assertions"),
    ("objective_node", "objective_nodes"),
    ("entity", "actors"),
    ("artifact", "documents"),
    ("support_edge", "evidence_edges"),
    ("criterion", "issue_predicates"),
    ("gap", "gaps"),
    ("clarification", "clarifications"),
)

_FINANCE_PROFILE = {
    "broker_protocol": "memory_coordination_v15",
    "profile_id": "finance",
    "profile_kind": "finance",
    "profile_version": DOMAIN_PROFILE_VERSION,
    "neutral_kernel": {
        "claim": "financial_claim",
        "objective_node": "investment_or_compliance_question",
        "entity": "market_participant",
        "artifact": "financial_document",
        "support_edge": "financial_evidence_edge",
        "criterion": "metric_or_disclosure_criterion",
        "gap": "missing_financial_evidence",
    },
    "source_roles": [
        "issuer_management",
        "auditor",
        "board",
        "regulator",
        "analyst",
        "rating_agency",
        "market_data_provider",
        "investor",
        "journalist",
        "counterparty",
    ],
    "belief_states": [
        "reported",
        "audited",
        "management_guidance",
        "analyst_estimate",
        "market_implied",
        "regulatory_position",
        "contested",
        "revised",
        "unsupported",
    ],
    "trust_weights": {
        "auditor": 0.9,
        "regulator": 0.9,
        "issuer_management": 0.72,
        "board": 0.72,
        "market_data_provider": 0.68,
        "rating_agency": 0.62,
        "analyst": 0.55,
        "counterparty": 0.5,
        "journalist": 0.4,
        "investor": 0.35,
    },
    "taint_classes": [
        "public_clean",
        "clean_with_withheld",
        "internal_work_product",
        "material_nonpublic",
        "confidential_counterparty",
        "sealed_privileged",
        "unknown_taint",
    ],
    "speech_acts": [
        "reported_fact",
        "forward_guidance",
        "risk_disclosure",
        "audit_opinion",
        "analyst_opinion",
        "regulatory_finding",
        "market_quote",
        "restatement",
        "denial",
    ],
}

_CODING_PROFILE = {
    "broker_protocol": "memory_coordination_v15",
    "profile_id": "coding",
    "profile_kind": "coding",
    "profile_version": DOMAIN_PROFILE_VERSION,
    "neutral_kernel": {
        "claim": "behavior_or_design_claim",
        "objective_node": "requirement_or_bug",
        "entity": "component_or_actor",
        "artifact": "code_artifact",
        "support_edge": "test_or_trace_edge",
        "criterion": "acceptance_or_invariant",
        "gap": "missing_repro_or_spec",
    },
    "source_roles": [
        "author",
        "reviewer",
        "maintainer",
        "automated_test",
        "ci_system",
        "runtime_trace",
        "issue_reporter",
        "security_scanner",
        "dependency_metadata",
        "documentation",
    ],
    "belief_states": [
        "implemented",
        "tested",
        "reviewed",
        "failing",
        "reproduced",
        "suspected",
        "deprecated",
        "contradicted",
        "unsupported",
    ],
    "trust_weights": {
        "automated_test": 0.85,
        "ci_system": 0.82,
        "runtime_trace": 0.8,
        "maintainer": 0.72,
        "reviewer": 0.68,
        "security_scanner": 0.65,
        "author": 0.6,
        "dependency_metadata": 0.58,
        "documentation": 0.45,
        "issue_reporter": 0.38,
    },
    "taint_classes": [
        "public_clean",
        "clean_with_withheld",
        "internal_work_product",
        "security_sensitive",
        "secret_or_credential",
        "license_restricted",
        "unknown_taint",
    ],
    "speech_acts": [
        "implementation",
        "test_result",
        "review_comment",
        "bug_report",
        "design_rationale",
        "api_contract",
        "deprecation_notice",
        "security_finding",
        "build_log",
    ],
}

_ACADEMIC_RESEARCH_PROFILE = {
    "broker_protocol": "memory_coordination_v15",
    "profile_id": "academic_research",
    "profile_kind": "academic_research",
    "profile_version": DOMAIN_PROFILE_VERSION,
    "neutral_kernel": {
        "claim": "research_claim",
        "objective_node": "research_question",
        "entity": "research_actor_or_object",
        "artifact": "research_artifact",
        "support_edge": "citation_or_method_edge",
        "criterion": "methodological_criterion",
        "gap": "missing_evidence_or_replication",
    },
    "source_roles": [
        "peer_reviewed_paper",
        "preprint",
        "replication_study",
        "dataset",
        "method_author",
        "review_article",
        "editorial",
        "funding_disclosure",
        "institution",
        "conference_talk",
    ],
    "belief_states": [
        "hypothesized",
        "reported_finding",
        "peer_reviewed",
        "replicated",
        "failed_replication",
        "method_limited",
        "retracted",
        "contested",
        "unsupported",
    ],
    "trust_weights": {
        "replication_study": 0.9,
        "peer_reviewed_paper": 0.78,
        "dataset": 0.72,
        "review_article": 0.68,
        "institution": 0.58,
        "method_author": 0.55,
        "preprint": 0.5,
        "funding_disclosure": 0.45,
        "conference_talk": 0.38,
        "editorial": 0.3,
    },
    "taint_classes": [
        "public_clean",
        "clean_with_withheld",
        "internal_work_product",
        "embargoed_research",
        "human_subjects_sensitive",
        "proprietary_dataset",
        "unknown_taint",
    ],
    "speech_acts": [
        "hypothesis",
        "finding",
        "method_description",
        "statistical_result",
        "limitation",
        "citation",
        "replication_result",
        "retraction",
        "funding_disclosure",
    ],
}

_BIOMEDICAL_PROFILE = {
    "broker_protocol": "memory_coordination_v15",
    "profile_id": "biomedical",
    "profile_kind": "biomedical",
    "profile_version": DOMAIN_PROFILE_VERSION,
    "neutral_kernel": {
        "claim": "clinical_or_biological_claim",
        "objective_node": "clinical_or_mechanistic_question",
        "entity": "biomedical_entity",
        "artifact": "biomedical_artifact",
        "support_edge": "clinical_or_lab_evidence_edge",
        "criterion": "endpoint_or_biological_criterion",
        "gap": "missing_clinical_or_lab_evidence",
    },
    "source_roles": [
        "phase_iii_trial",
        "phase_ii_trial",
        "case_report",
        "in_vitro_study",
        "animal_study",
        "regulator",
        "clinical_guideline",
        "lab_result",
        "patient_record",
        "manufacturer",
    ],
    "belief_states": [
        "observed",
        "clinically_validated",
        "mechanistic_hypothesis",
        "statistically_significant",
        "not_significant",
        "adverse_signal",
        "regulatory_accepted",
        "contraindicated",
        "unsupported",
    ],
    "trust_weights": {
        "phase_iii_trial": 0.92,
        "regulator": 0.9,
        "clinical_guideline": 0.82,
        "phase_ii_trial": 0.72,
        "lab_result": 0.68,
        "manufacturer": 0.55,
        "animal_study": 0.48,
        "in_vitro_study": 0.42,
        "case_report": 0.35,
        "patient_record": 0.34,
    },
    "taint_classes": [
        "public_clean",
        "clean_with_withheld",
        "internal_work_product",
        "phi",
        "clinical_trial_confidential",
        "regulatory_confidential",
        "unknown_taint",
    ],
    "speech_acts": [
        "clinical_finding",
        "endpoint_result",
        "safety_signal",
        "mechanism_claim",
        "dosage_statement",
        "contraindication",
        "regulatory_labeling",
        "case_observation",
        "lab_measurement",
    ],
}

_PROFILES_BY_ID: dict[str, dict] = {
    "finance": _FINANCE_PROFILE,
    "coding": _CODING_PROFILE,
    "academic_research": _ACADEMIC_RESEARCH_PROFILE,
    "biomedical": _BIOMEDICAL_PROFILE,
}

_CANONICAL_KERNEL_KEYS = frozenset(
    ("claim", "objective_node", "entity", "artifact", "support_edge", "criterion", "gap")
)

_CROSS_DOMAIN_PAIRS: tuple[tuple[str, str], ...] = (
    ("legal", "finance"),
    ("legal", "coding"),
    ("legal", "academic_research"),
    ("legal", "biomedical"),
    ("finance", "coding"),
    ("finance", "academic_research"),
    ("finance", "biomedical"),
    ("coding", "academic_research"),
    ("coding", "biomedical"),
    ("academic_research", "biomedical"),
)


def canonical_profile_json(profile_id: str) -> str:
    if profile_id == "legal":
        raise ValueError("Legal profile JSON is owned by MemoryBrokerStore.default_legal_profile_json()")
    profile = _PROFILES_BY_ID.get(profile_id)
    if profile is None:
        raise ValueError(f"Unknown builtin profile: {profile_id}")
    return _json_mod.dumps(profile, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_profile_hash(profile_id: str) -> str:
    payload = canonical_profile_json(profile_id)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def iter_builtin_domain_profiles() -> Iterator[tuple[str, int, str, str, str]]:
    """Yield (profile_id, version, profile_kind, profile_json, mapping_hash) for non-legal profiles."""
    for pid in DOMAIN_PROFILE_IDS:
        if pid == "legal":
            continue
        pjson = canonical_profile_json(pid)
        phash = "sha256:" + hashlib.sha256(pjson.encode("utf-8")).hexdigest()
        kind = _PROFILES_BY_ID[pid]["profile_kind"]
        yield pid, DOMAIN_PROFILE_VERSION, kind, pjson, phash


def iter_builtin_profile_mappings(
    legal_hash: str,
) -> Iterator[dict]:
    """Yield mapping row dicts for identity and cross-domain compatible mappings.

    legal_hash must be the current legal:1 mapping_hash from the broker.
    """
    profile_hashes: dict[str, str] = {"legal": legal_hash}
    for pid in DOMAIN_PROFILE_IDS:
        if pid == "legal":
            continue
        profile_hashes[pid] = canonical_profile_hash(pid)

    # Identity mappings: each profile maps to itself for all 8 target pairs
    for pid in DOMAIN_PROFILE_IDS:
        phash = profile_hashes[pid]
        for target_kind, target_namespace in MAPPING_TARGET_PAIRS:
            yield {
                "source_domain_profile_id": pid,
                "source_domain_profile_version": DOMAIN_PROFILE_VERSION,
                "target_domain_profile_id": pid,
                "target_domain_profile_version": DOMAIN_PROFILE_VERSION,
                "source_mapping_hash": phash,
                "target_mapping_hash": phash,
                "target_kind": target_kind,
                "target_namespace": target_namespace,
                "compatibility_status": "identity",
            }

    # Cross-domain compatible mappings for neutral-kernel types + clarifications
    for src, tgt in _CROSS_DOMAIN_PAIRS:
        src_hash = profile_hashes[src]
        tgt_hash = profile_hashes[tgt]
        for target_kind, target_namespace in MAPPING_TARGET_PAIRS:
            yield {
                "source_domain_profile_id": src,
                "source_domain_profile_version": DOMAIN_PROFILE_VERSION,
                "target_domain_profile_id": tgt,
                "target_domain_profile_version": DOMAIN_PROFILE_VERSION,
                "source_mapping_hash": src_hash,
                "target_mapping_hash": tgt_hash,
                "target_kind": target_kind,
                "target_namespace": target_namespace,
                "compatibility_status": "compatible",
            }
            yield {
                "source_domain_profile_id": tgt,
                "source_domain_profile_version": DOMAIN_PROFILE_VERSION,
                "target_domain_profile_id": src,
                "target_domain_profile_version": DOMAIN_PROFILE_VERSION,
                "source_mapping_hash": tgt_hash,
                "target_mapping_hash": src_hash,
                "target_kind": target_kind,
                "target_namespace": target_namespace,
                "compatibility_status": "compatible",
            }
