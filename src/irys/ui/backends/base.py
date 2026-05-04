"""Abstract UI backend interface.

The UI never calls MatterModel or RLMEngine directly. It calls the backend,
which translates to either an HTTP call to the FastAPI service (canonical) or
an in-process call (dev fallback only).
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Optional


class UIBackend(ABC):
    """Abstract interface for UI ↔ backend communication."""

    # ------------------------------------------------------------------ #
    # Investigation control                                                #
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def start_investigation(
        self,
        repo_path: str,
        query: str,
        matter_id: Optional[str] = None,
        research_mode: Optional[str] = None,
        conversation_history: Optional[list[dict[str, str]]] = None,
    ) -> dict:
        """Start an investigation. Returns {matter_id, run_id, job_id}."""
        ...

    @abstractmethod
    async def stop_run(self, matter_id: str, run_id: str) -> dict:
        """Stop a specific run."""
        ...

    # ------------------------------------------------------------------ #
    # Overview / dashboard                                                  #
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def get_overview(self, matter_id: str) -> dict:
        """Aggregated landing-page payload: stats, weakest issues, gaps, SO metrics."""
        ...

    # ------------------------------------------------------------------ #
    # Live ledger streaming                                                 #
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def stream_run_events(
        self, matter_id: str, run_id: str, after_seq: int = -1
    ) -> AsyncIterator[dict]:
        """Async iterator of ledger events for a run (SSE via HTTP or DB polling)."""
        ...

    # ------------------------------------------------------------------ #
    # Matter model data                                                    #
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def list_runs(self, matter_id: str, limit: int = 10) -> list[dict]:
        ...

    @abstractmethod
    async def get_run_events(self, matter_id: str, run_id: str) -> list[dict]:
        ...

    @abstractmethod
    async def list_issues(self, matter_id: str) -> list[dict]:
        ...

    @abstractmethod
    async def list_assertions(
        self, matter_id: str, limit: int = 50, offset: int = 0
    ) -> list[dict]:
        ...

    @abstractmethod
    async def get_issue_assertions(self, matter_id: str, issue_id: str) -> list[dict]:
        """Return assertions linked to a specific issue with relation types (SO-4)."""
        ...

    @abstractmethod
    async def get_source_agreement(self, matter_id: str, issue_id: str) -> list[dict]:
        """Per-document support/attack breakdown for an issue (SO-5)."""
        ...

    @abstractmethod
    async def get_assertion_graph(self, matter_id: str, issue_id: str) -> dict:
        """Return assertion nodes + edges for graph visualization (SO-2)."""
        ...

    @abstractmethod
    async def get_issue_closure_workbench(self, matter_id: str, issue_id: str) -> dict:
        """Consolidated issue closure surface (SO-2, SO-3, SO-4, SO-7)."""
        ...

    @abstractmethod
    async def get_issue_authorities(self, matter_id: str, issue_id: str) -> list[dict]:
        """Return authorities linked to a specific issue with relevance (SO-4)."""
        ...

    @abstractmethod
    async def list_gaps(self, matter_id: str, limit: int = 50) -> list[dict]:
        ...

    @abstractmethod
    async def get_gap_workbench(self, matter_id: str, limit: int = 50, min_materiality: float = 0.0) -> dict:
        """Consolidated gap-to-action workbench (SO-7, SO-3)."""
        ...

    @abstractmethod
    async def get_investigation_readiness(self, matter_id: str) -> dict:
        """Matter-wide investigation readiness assessment (SO-3, SO-7)."""
        ...

    @abstractmethod
    async def get_assertion_trace(self, matter_id: str, assertion_id: str) -> dict:
        """Full impact trace for a single assertion (SO-2, SO-3, SO-5)."""
        ...

    @abstractmethod
    async def resolve_gap(self, matter_id: str, gap_id: str, resolution_note: str = "") -> bool:
        """Mark a gap as resolved with an optional note (SO-7)."""
        ...

    @abstractmethod
    async def escalate_gap(self, matter_id: str, gap_id: str) -> bool:
        """Escalate a gap to maximum blocker priority (SO-7)."""
        ...

    @abstractmethod
    async def list_clarifications(self, matter_id: str, limit: int = 20) -> list[dict]:
        ...

    # ------------------------------------------------------------------ #
    # User steering                                                        #
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def correct_assertion(
        self,
        matter_id: str,
        assertion_id: str,
        new_state: str,
        reason: str,
        run_id: "str | None" = None,
    ) -> dict:
        ...

    @abstractmethod
    async def redirect_run(
        self, matter_id: str, run_id: str, issue_id: str
    ) -> dict:
        ...

    @abstractmethod
    async def resume_run(
        self,
        matter_id: str,
        run_id: str,
        follow_up_query: Optional[str] = None,
        research_mode: Optional[str] = None,
        conversation_history: Optional[list[dict[str, str]]] = None,
    ) -> dict:
        """Resume an interrupted run from its checkpoint. Returns {new_run_id, status}."""
        ...

    # ------------------------------------------------------------------ #
    # SO-3 / SO-6 supplemental surfaces                                   #
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def get_steering_surface(
        self, matter_id: str, run_id: Optional[str] = None
    ) -> list[dict]:
        """Return structured steering actions from get_ledger_steering_surface().

        run_id is embedded in redirect_focus action params so callers can invoke
        the redirect directly without a separate lookup.
        """
        ...

    @abstractmethod
    async def get_quant_summary(self, matter_id: str) -> dict:
        """Return quant summary payloads for the SO-6 panel."""
        ...

    @abstractmethod
    async def list_assumptions(self, matter_id: str, limit: int = 30) -> list[dict]:
        """Return active/all assumptions for the matter (Gap 3)."""
        ...

    @abstractmethod
    async def update_assumption_status(self, matter_id: str, assumption_id: str, status: str, reason: str = "") -> bool:
        """Set assumption status: provisional, confirmed, or invalidated (SO-3)."""
        ...

    @abstractmethod
    async def get_quant_facts(self, matter_id: str, limit: int = 200) -> dict:
        """Quant fact review workbench: extracted numbers by kind with conflict status (SO-6)."""
        ...

    @abstractmethod
    async def get_decision_leverage(self, matter_id: str, top_n: int = 15) -> dict:
        """Ranked leverage map: what to review next to shift outcomes (SO-2 through SO-7)."""
        ...

    @abstractmethod
    async def get_output_quality(self, matter_id: str, run_id: str | None = None) -> dict:
        """Output quality contract workbench: obligations, run summaries, manifest freshness."""
        ...

    @abstractmethod
    async def get_deliverable_workbench(self, matter_id: str) -> dict:
        """Deliverable preparation: verified issues, cited assertions, reliance gate."""
        ...

    @abstractmethod
    async def get_scenario_workbench(self, matter_id: str) -> dict:
        ...

    @abstractmethod
    async def create_scenario_branch(self, matter_id: str, payload: dict) -> dict:
        ...

    @abstractmethod
    async def archive_scenario_branch(self, matter_id: str, branch_id: str) -> dict:
        ...

    @abstractmethod
    async def get_objective_coverage(self, matter_id: str) -> dict:
        """Objective coverage workbench: per-objective criteria, support, gaps (SO-4)."""
        ...

    @abstractmethod
    async def set_criterion_status(
        self, matter_id: str, predicate_id: str, status: str, reason: str = "",
    ) -> dict:
        """Set criterion/predicate status: open, resolved, contested, or blocked (SO-4)."""
        ...

    @abstractmethod
    async def add_criterion(
        self, matter_id: str, objective_id: str, description: str, burden_side: str = "",
    ) -> dict:
        """Add a criterion/predicate to an objective (SO-4)."""
        ...

    @abstractmethod
    async def get_assumption_review(self, matter_id: str) -> dict:
        """Assumption review workbench: lifecycle groups with impact data (SO-3, SO-7)."""
        ...

    @abstractmethod
    async def review_assumption(
        self, matter_id: str, assumption_id: str, decision: str, reason: str = "",
    ) -> dict:
        """Review an assumption: confirm, invalidate, or revert (SO-3, SO-7)."""
        ...

    @abstractmethod
    async def set_issue_priority(self, matter_id: str, issue_id: str, priority: str) -> bool:
        """Set issue priority: critical, high, medium, low (SO-3)."""
        ...

    @abstractmethod
    async def get_timeline(self, matter_id: str, limit: int = 80, policy_audience: str = "clean") -> list[dict]:
        """Return timeline events for the matter."""
        ...

    @abstractmethod
    async def get_evidence_matrix(self, matter_id: str, policy_audience: str = "clean") -> dict:
        """Return issue x source evidence coverage data."""
        ...

    @abstractmethod
    async def get_communication_map(self, matter_id: str) -> dict:
        """Return actor/document communication graph data."""
        ...

    @abstractmethod
    async def list_llm_calls(
        self,
        matter_id: str,
        run_id: Optional[str] = None,
        limit: int = 120,
    ) -> list[dict]:
        """Return recent persisted LLM call rows for analytics."""
        ...

    @abstractmethod
    async def get_proof_state_summary(self, matter_id: str) -> dict:
        """Return proof state summary and per-issue proof states."""
        ...

    @abstractmethod
    async def get_authority_network(self, matter_id: str) -> dict:
        """Return authorities with issue links for the authority panel."""
        ...

    @abstractmethod
    async def upsert_authority(
        self,
        matter_id: str,
        citation: str,
        *,
        authority_type: str = "case",
        name: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        weight: str = "persuasive",
    ) -> dict:
        """Create or update an authority. Returns {authority_id, is_new}."""
        ...

    @abstractmethod
    async def link_authority_to_issue(
        self,
        matter_id: str,
        authority_id: str,
        issue_id: str,
        relevance: str = "supporting",
    ) -> dict:
        """Link an authority to an issue. Returns {status}."""
        ...

    @abstractmethod
    async def unlink_authority_from_issue(
        self, matter_id: str, authority_id: str, issue_id: str
    ) -> dict:
        """Unlink an authority from an issue. Returns {status}."""
        ...

    @abstractmethod
    async def search_authorities(
        self, matter_id: str, query: str, limit: int = 20
    ) -> list[dict]:
        """Search authorities by citation/name."""
        ...

    @abstractmethod
    async def get_document_intelligence(self, matter_id: str) -> dict:
        """Return document cards with inventory metadata for the document panel."""
        ...

    @abstractmethod
    async def list_belief_revisions(self, matter_id: str, limit: int = 100) -> list[dict]:
        """Return belief revision events for the transparency panel."""
        ...

    @abstractmethod
    async def get_contradictions(self, matter_id: str, limit: int = 100) -> list[dict]:
        """Return active contradiction pairs in the assertion graph."""
        ...

    @abstractmethod
    async def get_document_versions(self, matter_id: str) -> list[dict]:
        """Return document version families with operative HEAD marked."""
        ...

    @abstractmethod
    async def mine_contradictions(self, matter_id: str) -> list[dict]:
        """Trigger on-demand contradiction mining pass."""
        ...

    @abstractmethod
    async def get_provenance(self, matter_id: str, target_kind: str, target_id: str, limit: int = 50) -> list[dict]:
        """Return provenance trail for a specific object."""
        ...

    @abstractmethod
    async def refresh_document_families(self, matter_id: str) -> list[dict]:
        """Trigger version chain detection and persist family membership."""
        ...

    @abstractmethod
    async def get_assertion_health(self, matter_id: str, assertion_id: str) -> dict:
        """Return health diagnostics for a single assertion."""
        ...

    @abstractmethod
    async def get_assertion_history(self, matter_id: str, assertion_id: str, limit: int = 20) -> dict:
        """Return field-level revision history for an assertion."""
        ...

    @abstractmethod
    async def list_content_policy_decisions(self, matter_id: str, limit: int = 50) -> list[dict]:
        """Return recent content policy audit decisions."""
        ...

    @abstractmethod
    async def get_quant_thresholds(
        self, matter_id: str, currency: str = "USD",
        *, exposure_high: float = 10_000.0, disputed_fraction_min: float = 0.10,
    ) -> list[dict]:
        """Return quantitative threshold violations."""
        ...

    @abstractmethod
    async def get_amount_conflicts(self, matter_id: str) -> list[dict]:
        """Return grouped amount conflicts for SO-6 transparency."""
        ...

    @abstractmethod
    async def detect_quant_conflicts(self, matter_id: str) -> list[str]:
        """Trigger conflict detection. Returns new gap_ids created."""
        ...

    @abstractmethod
    async def get_reconciliation(self, matter_id: str, currency: str = "USD") -> dict:
        """Return payment chain reconciliation (SO-6)."""
        ...

    @abstractmethod
    async def get_invoice_chain(self, matter_id: str, currency: str = "USD") -> list:
        """Return per-invoice reconciliation rows (SO-6)."""
        ...

    @abstractmethod
    async def get_damages_waterfall(self, matter_id: str, currency: str = "USD") -> list[dict]:
        """Return structured damages breakdown by category with conflict detection (SO-6)."""
        ...

    @abstractmethod
    async def get_system_health(self, matter_id: str) -> dict:
        """Return system health diagnostics."""
        ...

    @abstractmethod
    async def compute_proof_state(self, matter_id: str) -> dict:
        """Recompute proof state for all open issues. Returns updated count."""
        ...

    @abstractmethod
    async def compute_issue_proof_state(self, matter_id: str, issue_id: str) -> dict:
        """Recompute proof state for a single issue. Returns the proof state."""
        ...

    @abstractmethod
    async def flush_pending(self, matter_id: str) -> dict:
        """Drain pending propagation queues for belief revision convergence."""
        ...

    @abstractmethod
    async def get_so_scorecard(self, matter_id: str) -> dict:
        """Return Sacred Outcome metrics with targets and pass/fail status."""
        ...

    @abstractmethod
    async def get_domain_profile_summary(
        self, matter_id: str, profile_id: str | None = None
    ) -> dict:
        """Return comprehensive domain profile configuration summary."""
        ...

    @abstractmethod
    async def get_domain_composition(self, matter_id: str) -> dict:
        """Return domain facets, composed trust weights, and detection events."""
        ...

    @abstractmethod
    async def list_documents_needing_profile(
        self, matter_id: str, limit: int = 50
    ) -> list[dict]:
        """Return documents not yet fully profiled, ordered by salience."""
        ...

    @abstractmethod
    async def get_taint_summary(
        self, matter_id: str, limit: int = 50
    ) -> dict:
        """Return aggregated taint records: counts by class/kind + recent entries."""
        ...

    @abstractmethod
    async def answer_clarification(
        self, matter_id: str, question_id: str, answer_text: str
    ) -> bool:
        """Answer a pending clarification question. Returns True if found."""
        ...

    @abstractmethod
    async def list_trust_overrides(self, matter_id: str) -> list[dict]:
        """Return all document trust overrides."""
        ...

    @abstractmethod
    async def set_trust_override(
        self, matter_id: str, document_pattern: str, trust_level: str, note: str = ""
    ) -> str:
        """Set a trust override. Returns the override_id."""
        ...

    @abstractmethod
    async def delete_trust_override(self, matter_id: str, document_pattern: str) -> bool:
        """Delete a trust override. Returns True if deleted."""
        ...

    @abstractmethod
    async def generate_clarifications(self, matter_id: str, top_n: int = 3) -> list[str]:
        """Generate clarification questions from gaps. Returns new question_ids."""
        ...

    @abstractmethod
    async def search_assertions(self, matter_id: str, query: str, limit: int = 20) -> list[dict]:
        """Search assertions by text."""
        ...

    # ------------------------------------------------------------------ #
    # Actor resolution (SO-5)                                              #
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def find_duplicate_actors(self, matter_id: str, min_prefix_len: int = 6) -> list[dict]:
        """Return potential duplicate actor pairs with shared prefix."""
        ...

    @abstractmethod
    async def merge_actors(self, matter_id: str, keep_id: str, merge_id: str) -> dict:
        """Merge merge_id into keep_id. Returns merge result."""
        ...

    @abstractmethod
    async def resolve_actor(self, matter_id: str, name: str) -> dict:
        """Resolve an actor by name/alias. Returns {actor_id, actor} or {actor_id: None}."""
        ...

    # ------------------------------------------------------------------ #
    # Decision context (SO-3)                                              #
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def get_decision_context(self, matter_id: str) -> "dict | None":
        """Return the decision context overlay, or None if not set."""
        ...

    @abstractmethod
    async def set_decision_context(
        self, matter_id: str,
        decision_maker_type: Optional[str] = None,
        decision_maker_name: Optional[str] = None,
        objective: Optional[str] = None,
        strategic_notes: Optional[str] = None,
        scope_narrow: bool = False,
    ) -> str:
        """Set the decision context. Returns context_id."""
        ...

    @abstractmethod
    async def clear_decision_context(self, matter_id: str) -> bool:
        """Clear the decision context overlay."""
        ...

    # ------------------------------------------------------------------ #
    # Document annotations (SO-3)                                          #
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def list_annotations(self, matter_id: str, document: Optional[str] = None) -> list[dict]:
        """Return annotations, optionally filtered by document."""
        ...

    @abstractmethod
    async def add_annotation(
        self, matter_id: str, document_pattern: str,
        annotation_text: str, annotation_type: str = "strategic",
    ) -> str:
        """Add a document annotation. Returns annotation_id."""
        ...

    @abstractmethod
    async def delete_annotation(self, matter_id: str, annotation_id: str) -> bool:
        """Delete an annotation. Returns True if deleted."""
        ...

    # ------------------------------------------------------------------ #
    # Report export                                                        #
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def export_matter_summary(self, matter_id: str) -> dict:
        """Return structured summary for report export."""
        ...

    # ------------------------------------------------------------------ #
    # Cost analytics                                                       #
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def get_cost_breakdown(
        self, matter_id: str, run_id: Optional[str] = None
    ) -> dict:
        """Return cost breakdown with percentiles, cache rate, trend, and burn projection."""
        ...

    @abstractmethod
    async def get_cost_anomalies(
        self, matter_id: str, limit: int = 10, run_id: Optional[str] = None
    ) -> list[dict]:
        """Return outlier LLM calls flagged by z-score."""
        ...

    # ------------------------------------------------------------------ #
    # Review queue (SO-3)                                                  #
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def get_review_queue(
        self, matter_id: str, limit: int = 50, offset: int = 0,
        target_kind: Optional[str] = None,
    ) -> list[dict]:
        """Return prioritized review queue items."""
        ...

    @abstractmethod
    async def count_review_queue(self, matter_id: str) -> dict:
        """Return review queue counts by verification bucket."""
        ...

    @abstractmethod
    async def verify_target(
        self, matter_id: str, target_kind: str, target_id: str,
        *, reviewed_by_kind: str = "user",
        reviewed_by_id: Optional[str] = None,
        review_note: Optional[str] = None,
        review_scope: str = "extraction_correct",
    ) -> str:
        """Verify a review target. Returns verification_id."""
        ...

    @abstractmethod
    async def reject_target(
        self, matter_id: str, target_kind: str, target_id: str,
        *, rejection_reason: str,
        reviewed_by_kind: str = "user",
        reviewed_by_id: Optional[str] = None,
        review_note: Optional[str] = None,
    ) -> str:
        """Reject a review target. Returns verification_id."""
        ...

    @abstractmethod
    async def bulk_verify_by_document(
        self, matter_id: str, document_ref: str,
        *, reviewed_by_kind: str = "user",
        reviewed_by_id: Optional[str] = None,
    ) -> list[str]:
        """Bulk-verify all candidate assertions for a document."""
        ...

    @abstractmethod
    async def bulk_verify_by_span(
        self, matter_id: str, span_id: str,
        *, reviewed_by_kind: str = "user",
        reviewed_by_id: Optional[str] = None,
        review_note: Optional[str] = None,
        review_scope: str = "extraction_correct",
    ) -> list[str]:
        """Bulk-verify all candidate assertions sourced from a span (SO-3)."""
        ...

    @abstractmethod
    async def bulk_verify_assertion_ids(
        self, matter_id: str, assertion_ids: list[str],
        *, reviewed_by_kind: str = "user",
        reviewed_by_id: Optional[str] = None,
        review_note: Optional[str] = None,
    ) -> list[str]:
        """Bulk-verify specific assertion IDs."""
        ...

    @abstractmethod
    async def list_candidate_assertions_for_document(
        self, matter_id: str, document_ref: str,
    ) -> list[dict]:
        """Return candidate assertions for a document review flow."""
        ...

    @abstractmethod
    async def get_document_console(self, matter_id: str, document_ref: str) -> dict:
        """Consolidated per-document review surface (SO-3, SO-5)."""
        ...

    @abstractmethod
    async def list_reviewable_documents(self, matter_id: str) -> list[dict]:
        """Return documents with pending/verified counts for the review picker."""
        ...

    @abstractmethod
    async def get_quant_ontology(self, matter_id: str) -> dict:
        """Quantitative ontology workbench: metric groups with domain classification (SO-6)."""
        ...

    @abstractmethod
    async def approve_metric_alias(
        self, matter_id: str, raw_label: str, canonical_metric: str, unit: str | None = None,
    ) -> bool:
        """Approve a metric alias in the quantitative ontology (SO-6)."""
        ...

    @abstractmethod
    async def get_answer_audits(
        self, matter_id: str, manifest_hash: str | None = None,
    ) -> dict:
        """Answer audit workbench: freshness, sources, policy for recent answers (SO-1)."""
        ...

    @abstractmethod
    async def resolve_contradiction(
        self, matter_id: str, attacker_id: str, attacked_id: str,
        decision: str, rationale: str = "",
    ) -> dict:
        """Resolve a contradiction pair: prefer one assertion, dispute both, or request evidence (SO-2)."""
        ...

    @abstractmethod
    async def get_knowledge_seeds(self, matter_id: str) -> dict:
        """Knowledge seed workbench: cross-matter reuse review data (SO-1)."""
        ...

    @abstractmethod
    async def review_knowledge_seed(
        self, matter_id: str, seed_id: str, decision: str,
        review_note: str = "",
    ) -> dict:
        """Review a knowledge seed: approve, reject, or keep promotable (SO-1)."""
        ...

    @abstractmethod
    async def promote_knowledge_seed(
        self, matter_id: str, seed_kind: str, domain_profile_id: str,
        payload_json: str, source_matter_id: str | None = None,
    ) -> dict:
        """Promote intelligence as a reusable knowledge seed (SO-1)."""
        ...

    @abstractmethod
    async def get_verification_events(
        self, matter_id: str, target_kind: Optional[str] = None,
        target_id: Optional[str] = None, limit: int = 50,
    ) -> list[dict]:
        """Return verification event history."""
        ...
