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
    ) -> dict:
        ...

    @abstractmethod
    async def redirect_run(
        self, matter_id: str, run_id: str, issue_id: str
    ) -> dict:
        ...

    # ------------------------------------------------------------------ #
    # SO-3 / SO-6 supplemental surfaces                                   #
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def get_steering_surface(self, matter_id: str) -> list[dict]:
        """Return structured steering actions from get_ledger_steering_surface()."""
        ...

    @abstractmethod
    async def get_quant_summary(self, matter_id: str) -> dict:
        """Return {payment_reconciliation, damages_waterfall} for SO-6 panel."""
        ...
