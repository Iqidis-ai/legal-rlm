"""Table coverage manifest for the PR.1 schema discipline gate.

Every non-legacy table in the matter schema must either have an explicit
runtime writer/reader in production code, or an explicit deferred status
saying when it is expected to get one. The goal is that a new table cannot
enter the schema without someone taking responsibility for reads and writes
or declaring why those reads and writes are not yet in place.

LEGACY_TABLES: tables that predate this manifest. They have long-standing
reader/writer surfaces that are not itemized here; new tables must not be
added to this set.

TABLE_COVERAGE_MANIFEST: every non-legacy table, either covered (with
writers and readers) or deferred (with deferred_until and reason).
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TableCoverageSpec:
    """Coverage declaration for a single non-legacy table.

    A covered table names at least one writer and one reader, each as a
    `module:qualname` string. A deferred table names no writers or readers
    and instead provides a milestone + reason.
    """

    writers: tuple[str, ...] = field(default=())
    readers: tuple[str, ...] = field(default=())
    deferred_until: str | None = None
    reason: str | None = None


LEGACY_TABLES: frozenset[str] = frozenset({
    "actor",
    "actor_affiliation",
    "actor_alias",
    "assertion",
    "assertion_issue_link",
    "assertion_link",
    "assertion_occurrence",
    "assertion_revision",
    "assumption",
    "assumption_link",
    "belief_revision_event",
    "clarification_question",
    "document_card",
    "document_inventory",
    "document_relation",
    "gap",
    "gap_link",
    "issue",
    "issue_predicate",
    "ledger_event",
    "matter",
    "quant_fact",
    "run_session",
    "schema_version",
    "span",
})


TABLE_COVERAGE_MANIFEST: dict[str, TableCoverageSpec] = {
    "reasoning_cache": TableCoverageSpec(
        writers=("irys.matter.graph:ReasoningCacheStore.put",),
        readers=("irys.matter.graph:ReasoningCacheStore.get",),
    ),
    "document_trust_override": TableCoverageSpec(
        writers=("irys.matter.graph:TrustOverrideStore.set",),
        readers=("irys.matter.graph:TrustOverrideStore.get",),
    ),
    "document_annotation": TableCoverageSpec(
        writers=("irys.matter.graph:DocumentAnnotationStore.add",),
        readers=("irys.matter.graph:DocumentAnnotationStore.get_for_document",),
    ),
    "decision_context": TableCoverageSpec(
        writers=("irys.matter.graph:DecisionContextStore.set",),
        readers=("irys.matter.graph:DecisionContextStore.get",),
    ),
    "authority": TableCoverageSpec(
        writers=("irys.matter.graph:AuthorityStore.upsert",),
        readers=("irys.matter.graph:AuthorityStore.get",),
    ),
    "authority_issue_link": TableCoverageSpec(
        writers=("irys.matter.graph:AuthorityStore.link_to_issue",),
        readers=("irys.matter.graph:AuthorityStore.list_for_issue",),
    ),
    "proof_state": TableCoverageSpec(
        writers=("irys.matter.graph:ProofStateStore.compute_and_store",),
        readers=("irys.matter.graph:ProofStateStore.get",),
    ),
    "document_actor_role": TableCoverageSpec(
        writers=("irys.matter.graph:DocumentActorRoleStore.upsert",),
        readers=("irys.matter.graph:DocumentActorRoleStore.list_by_document",),
    ),
    "pending_propagation": TableCoverageSpec(
        writers=(
            "irys.matter.matter:MatterModel.enqueue_correction_pending",
            "irys.matter.matter:MatterModel.enqueue_evidence_pending",
        ),
        readers=(
            "irys.matter.matter:MatterModel._load_pending_propagation",
            "irys.matter.matter:MatterModel.reload_pending_from_db",
        ),
    ),
    "llm_call": TableCoverageSpec(
        writers=("irys.matter.matter:MatterModel.record_llm_call",),
        readers=("irys.matter.matter:MatterModel.list_llm_calls",),
    ),
    "schema_migration": TableCoverageSpec(
        writers=("irys.matter.schema:_record_schema_migration",),
        readers=("irys.matter.schema:get_schema_ledger_versions",),
    ),
    "evidence_edge": TableCoverageSpec(
        writers=("irys.matter.graph:EvidenceStore.upsert_edge",),
        readers=(
            "irys.matter.graph:EvidenceStore.list_edges_for_target",
            "irys.matter.graph:ProofStateStore._query_issue_linked_assertions",
        ),
    ),
    "evidence_link": TableCoverageSpec(
        deferred_until="remove_or_replace",
        reason=(
            "legacy proof table superseded by evidence_edge (MVP.3); no "
            "runtime writer/reader and slated for removal. Must not gain "
            "new runtime surfaces."
        ),
    ),
    "migration_backfill_job": TableCoverageSpec(
        deferred_until="migration_cli",
        reason=(
            "schema-only queue table introduced by migration v49; no runtime "
            "surface until the multi-matter migration CLI is built"
        ),
    ),
    "verification_state": TableCoverageSpec(
        writers=(
            "irys.matter.graph:VerificationStateStore.candidate",
            "irys.matter.graph:VerificationStateStore._set_status",
        ),
        readers=(
            "irys.matter.graph:VerificationStateStore.get",
            "irys.matter.graph:VerificationStateStore.list_by_status",
        ),
    ),
    "verification_event": TableCoverageSpec(
        writers=("irys.matter.graph:VerificationStateStore._append_event",),
        readers=("irys.matter.graph:VerificationStateStore.list_events",),
    ),
    "provenance_event": TableCoverageSpec(
        writers=("irys.matter.graph:ProvenanceStore.record",),
        readers=(
            "irys.matter.graph:ProvenanceStore.list_for_target",
            "irys.matter.graph:ProvenanceStore.list_for_llm_call",
        ),
    ),
    "content_policy_audit": TableCoverageSpec(
        writers=("irys.matter.graph:ContentPolicyGuard._append_audit",),
        readers=("irys.matter.graph:ContentPolicyGuard.list_decisions",),
    ),
    "namespace_revision": TableCoverageSpec(
        writers=("irys.matter.graph:MemoryBrokerStore.bump_namespace_revision",),
        readers=("irys.matter.graph:MemoryBrokerStore.get_namespace_revision",),
    ),
    "object_taint": TableCoverageSpec(
        writers=("irys.matter.graph:MemoryBrokerStore.record_object_taint",),
        readers=("irys.matter.graph:MemoryBrokerStore.list_object_taint",),
    ),
    "domain_profile": TableCoverageSpec(
        writers=("irys.matter.graph:MemoryBrokerStore.upsert_domain_profile",),
        readers=("irys.matter.graph:MemoryBrokerStore.get_domain_profile",),
    ),
    "profile_mapping": TableCoverageSpec(
        writers=("irys.matter.graph:MemoryBrokerStore.record_profile_mapping",),
        readers=("irys.matter.graph:MemoryBrokerStore.list_profile_mappings",),
    ),
    "dependency_manifest": TableCoverageSpec(
        writers=("irys.matter.graph:MemoryBrokerStore.record_dependency_manifest",),
        readers=(
            "irys.matter.graph:MemoryBrokerStore.get_dependency_manifest",
            "irys.matter.graph:MemoryBrokerStore.validate_dependency_manifest",
        ),
    ),
    "memory_packet_event": TableCoverageSpec(
        writers=("irys.matter.graph:MemoryBrokerStore.record_memory_packet_event",),
        readers=("irys.matter.graph:MemoryBrokerStore.get_memory_packet_event",),
    ),
    "domain_detection_event": TableCoverageSpec(
        writers=("irys.matter.graph:MemoryBrokerStore.record_domain_detection_event",),
        readers=("irys.matter.graph:MemoryBrokerStore.record_domain_detection_event",),
    ),
    "object_domain_facet": TableCoverageSpec(
        writers=("irys.matter.graph:MemoryBrokerStore.upsert_object_domain_facet",),
        readers=("irys.matter.graph:MemoryBrokerStore.get_object_domain_facets",),
    ),
    "domain_composition": TableCoverageSpec(
        writers=("irys.matter.graph:MemoryBrokerStore.record_domain_composition",),
        readers=("irys.matter.graph:MemoryBrokerStore.get_domain_composition",),
    ),
    "unknown_domain_candidate": TableCoverageSpec(
        writers=("irys.matter.graph:MemoryBrokerStore.record_unknown_domain_candidate",),
        readers=("irys.matter.graph:MemoryBrokerStore.record_unknown_domain_candidate",),
    ),
}
