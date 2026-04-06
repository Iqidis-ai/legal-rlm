"""HTTP backend — communicates with the FastAPI service.

Canonical implementation. The UI talks to the service; the service talks to
the matter model. This gives deployment isolation and lets the UI survive
engine restarts cleanly.
"""

import json
from typing import Any, AsyncIterator, Optional

import httpx

from .base import UIBackend


class HttpBackend(UIBackend):
    """Communicates with the Irys FastAPI service over HTTP."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
        )

    async def _get(self, path: str, params: dict = None) -> Any:
        r = await self._client.get(path, params=params or {})
        r.raise_for_status()
        return r.json()

    async def _post(self, path: str, body: dict = None) -> Any:
        r = await self._client.post(path, json=body or {})
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------ #
    # Investigation control                                                #
    # ------------------------------------------------------------------ #

    async def start_investigation(
        self,
        repo_path: str,
        query: str,
        matter_id: Optional[str] = None,
    ) -> dict:
        """Start an investigation via the service.

        NOTE: The HTTP backend only supports S3-backed investigations — the service
        /investigate endpoint requires an s3_prefix, not a local repo_path.  For
        local-path investigations (dev/testing), use InProcessBackend instead.
        This raises RuntimeError so the UI surfaces a clear message rather than
        sending a malformed request that fails silently.
        """
        raise RuntimeError(
            "HttpBackend does not support local-path investigations. "
            "The service /investigate endpoint requires an S3 prefix. "
            "Use InProcessBackend (default dev mode) for local repositories."
        )

    async def stop_run(self, matter_id: str, run_id: str) -> dict:
        return await self._post(f"/matter/{matter_id}/runs/{run_id}/stop")

    # ------------------------------------------------------------------ #
    # Overview / dashboard                                                  #
    # ------------------------------------------------------------------ #

    async def get_overview(self, matter_id: str) -> dict:
        return await self._get(f"/matter/{matter_id}/overview")

    # ------------------------------------------------------------------ #
    # Live ledger streaming                                                 #
    # ------------------------------------------------------------------ #

    async def stream_run_events(
        self, matter_id: str, run_id: str, after_seq: int = -1
    ) -> AsyncIterator[dict]:
        """Stream ledger events via SSE."""
        url = f"{self.base_url}/matter/{matter_id}/runs/{run_id}/events/stream"
        params = {"after_seq": after_seq}
        async with self._client.stream("GET", url, params=params, timeout=None) as r:
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    try:
                        yield json.loads(line[6:])
                    except json.JSONDecodeError:
                        pass

    # ------------------------------------------------------------------ #
    # Matter model data                                                    #
    # ------------------------------------------------------------------ #

    async def list_runs(self, matter_id: str, limit: int = 10) -> list[dict]:
        return await self._get(f"/matter/{matter_id}/runs", {"limit": limit})

    async def get_run_events(self, matter_id: str, run_id: str) -> list[dict]:
        return await self._get(f"/matter/{matter_id}/runs/{run_id}/events")

    async def list_issues(self, matter_id: str) -> list[dict]:
        return await self._get(f"/matter/{matter_id}/issues")

    async def list_assertions(
        self, matter_id: str, limit: int = 50, offset: int = 0,
        issue_id: Optional[str] = None
    ) -> list[dict]:
        params = {"limit": limit, "offset": offset}
        if issue_id:
            params["issue_id"] = issue_id
        result = await self._get(f"/matter/{matter_id}/assertions", params)
        # Service returns paginated envelope: {total, limit, offset, assertions}
        if isinstance(result, dict):
            return result.get("assertions", [])
        return result if isinstance(result, list) else []

    async def list_gaps(self, matter_id: str, limit: int = 50) -> list[dict]:
        return await self._get(f"/matter/{matter_id}/gaps", {"limit": limit})

    async def list_clarifications(self, matter_id: str, limit: int = 20) -> list[dict]:
        return await self._get(f"/matter/{matter_id}/clarifications", {"limit": limit})

    # ------------------------------------------------------------------ #
    # User steering                                                        #
    # ------------------------------------------------------------------ #

    async def correct_assertion(
        self,
        matter_id: str,
        assertion_id: str,
        new_state: str,
        reason: str,
    ) -> dict:
        return await self._post(
            f"/matter/{matter_id}/assertions/{assertion_id}/correct",
            {"new_belief_state": new_state, "note": reason},
        )

    async def redirect_run(
        self, matter_id: str, run_id: str, issue_id: str
    ) -> dict:
        return await self._post(
            f"/matter/{matter_id}/runs/{run_id}/redirect",
            {"issue_id": issue_id},
        )
