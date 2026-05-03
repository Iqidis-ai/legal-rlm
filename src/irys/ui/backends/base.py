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
    async def list_gaps(self, matter_id: str, limit: int = 50) -> list[dict]:
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
    async def get_timeline(self, matter_id: str, limit: int = 80) -> list[dict]:
        """Return timeline events for the matter."""
        ...

    @abstractmethod
    async def get_evidence_matrix(self, matter_id: str) -> dict:
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
    async def get_quant_thresholds(self, matter_id: str, currency: str = "USD") -> list[dict]:
        """Return quantitative threshold violations."""
        ...

    @abstractmethod
    async def get_system_health(self, matter_id: str) -> dict:
        """Return system health diagnostics."""
        ...

    @abstractmethod
    async def get_so_scorecard(self, matter_id: str) -> dict:
        """Return Sacred Outcome metrics with targets and pass/fail status."""
        ...

    @abstractmethod
    async def answer_clarification(
        self, matter_id: str, question_id: str, answer_text: str
    ) -> bool:
        """Answer a pending clarification question. Returns True if found."""
        ...
