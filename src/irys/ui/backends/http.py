"""HTTP backend — communicates with the FastAPI service.

Canonical implementation. The UI talks to the service; the service talks to
the matter model. This gives deployment isolation and lets the UI survive
engine restarts cleanly.
"""

import json
from typing import Any, AsyncIterator, Optional
from urllib.parse import quote as _url_quote

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
        research_mode: Optional[str] = None,
        conversation_history: Optional[list[dict[str, str]]] = None,
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
        self, matter_id: str, limit: int = 50, offset: int = 0
    ) -> list[dict]:
        result = await self._get(f"/matter/{matter_id}/assertions", {"limit": limit, "offset": offset})
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
        run_id: "str | None" = None,
    ) -> dict:
        payload: dict = {"new_belief_state": new_state, "note": reason}
        if run_id is not None:
            payload["run_id"] = run_id
        return await self._post(
            f"/matter/{matter_id}/assertions/{assertion_id}/correct",
            payload,
        )

    async def redirect_run(
        self, matter_id: str, run_id: str, issue_id: str
    ) -> dict:
        return await self._post(
            f"/matter/{matter_id}/runs/{run_id}/redirect",
            {"issue_id": issue_id},
        )

    async def resume_run(
        self,
        matter_id: str,
        run_id: str,
        follow_up_query: Optional[str] = None,
        research_mode: Optional[str] = None,
        conversation_history: Optional[list[dict[str, str]]] = None,
    ) -> dict:
        body: dict[str, Any] = {}
        if follow_up_query:
            body["follow_up_query"] = follow_up_query
        if research_mode:
            body["research_mode"] = research_mode
        if conversation_history:
            body["conversation_history"] = conversation_history
        return await self._post(f"/matter/{matter_id}/runs/{run_id}/resume", body)

    # ------------------------------------------------------------------ #
    # SO-3 / SO-6 supplemental surfaces                                   #
    # ------------------------------------------------------------------ #

    async def get_steering_surface(
        self, matter_id: str, run_id: Optional[str] = None
    ) -> list[dict]:
        params = {"run_id": run_id} if run_id else {}
        result = await self._get(f"/matter/{matter_id}/steering-surface", params)
        return result if isinstance(result, list) else []

    async def get_quant_summary(self, matter_id: str) -> dict:
        """Fetch quant data from service endpoints and combine."""
        recon = await self._get(f"/matter/{matter_id}/reconciliation")
        invoice_chain = await self._get(f"/matter/{matter_id}/reconciliation/invoices")
        amount_conflicts = await self._get(f"/matter/{matter_id}/reconciliation/conflicts")
        damages = await self._get(f"/matter/{matter_id}/damages-waterfall")
        return {
            "payment_reconciliation": recon,
            "invoice_reconciliation": invoice_chain,
            "amount_conflicts": amount_conflicts,
            "damages_waterfall": damages,
        }

    async def list_assumptions(self, matter_id: str, limit: int = 30) -> list[dict]:
        return await self._get(f"/matter/{matter_id}/assumptions?limit={limit}")

    async def get_timeline(self, matter_id: str, limit: int = 80) -> list[dict]:
        return await self._get(f"/matter/{matter_id}/timeline", {"limit": limit})

    async def get_evidence_matrix(self, matter_id: str) -> dict:
        return await self._get(f"/matter/{matter_id}/evidence-matrix")

    async def get_communication_map(self, matter_id: str) -> dict:
        return await self._get(f"/matter/{matter_id}/communication-map")

    async def list_belief_revisions(self, matter_id: str, limit: int = 100) -> list[dict]:
        result = await self._get(f"/matter/{matter_id}/belief-revisions", {"limit": limit})
        return result if isinstance(result, list) else []

    async def get_contradictions(self, matter_id: str, limit: int = 100) -> list[dict]:
        result = await self._get(f"/matter/{matter_id}/contradictions", {"limit": limit})
        return result if isinstance(result, list) else []

    async def get_document_versions(self, matter_id: str) -> list[dict]:
        result = await self._get(f"/matter/{matter_id}/document-versions")
        return result if isinstance(result, list) else []

    async def mine_contradictions(self, matter_id: str) -> list[dict]:
        result = await self._post(f"/matter/{matter_id}/mine-contradictions")
        return result if isinstance(result, list) else []

    async def get_provenance(self, matter_id: str, target_kind: str, target_id: str, limit: int = 50) -> list[dict]:
        kind_enc = _url_quote(target_kind, safe="")
        id_enc = _url_quote(target_id, safe="")
        result = await self._get(f"/matter/{matter_id}/provenance/{kind_enc}/{id_enc}", {"limit": limit})
        return result if isinstance(result, list) else []

    async def refresh_document_families(self, matter_id: str) -> list[dict]:
        result = await self._post(f"/matter/{matter_id}/refresh-document-families")
        return result if isinstance(result, list) else []

    async def get_assertion_health(self, matter_id: str, assertion_id: str) -> dict:
        aid_enc = _url_quote(assertion_id, safe="")
        result = await self._get(f"/matter/{matter_id}/assertion/{aid_enc}/health")
        return result if isinstance(result, dict) else {}

    async def get_quant_thresholds(self, matter_id: str, currency: str = "USD") -> list[dict]:
        result = await self._get(f"/matter/{matter_id}/quant-thresholds", {"currency": currency})
        return result if isinstance(result, list) else []

    async def list_llm_calls(
        self,
        matter_id: str,
        run_id: Optional[str] = None,
        limit: int = 120,
    ) -> list[dict]:
        params = {"limit": limit}
        if run_id:
            params["run_id"] = run_id
        result = await self._get(f"/matter/{matter_id}/llm-calls", params)
        return result if isinstance(result, list) else []

    async def get_document_intelligence(self, matter_id: str) -> dict:
        cards = await self._get(f"/matter/{matter_id}/documents/cards")
        return cards if isinstance(cards, dict) else {"cards": [], "total_inventory": 0, "ingested_count": 0}

    async def get_proof_state_summary(self, matter_id: str) -> dict:
        return await self._get(f"/matter/{matter_id}/proof-state")

    async def get_authority_network(self, matter_id: str) -> dict:
        result = await self._get(f"/matter/{matter_id}/authority-network")
        if isinstance(result, dict):
            return result
        return {"authorities": [], "issue_links": {}}
