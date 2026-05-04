"""HTTP backend — communicates with the FastAPI service.

Canonical implementation. The UI talks to the service; the service talks to
the matter model. This gives deployment isolation and lets the UI survive
engine restarts cleanly.
"""

import json
import logging
from typing import Any, AsyncIterator, Optional
from urllib.parse import quote as _url_quote

import httpx

from .base import UIBackend

_log = logging.getLogger(__name__)


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

    async def _patch(self, path: str, body: dict = None) -> Any:
        r = await self._client.patch(path, json=body or {})
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
        result = await self._post(f"/matter/{matter_id}/runs/{run_id}/stop")
        return result if isinstance(result, dict) else {}

    # ------------------------------------------------------------------ #
    # Overview / dashboard                                                  #
    # ------------------------------------------------------------------ #

    async def get_overview(self, matter_id: str) -> dict:
        result = await self._get(f"/matter/{matter_id}/overview")
        if not isinstance(result, dict):
            _log.warning("get_overview: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

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
        result = await self._get(f"/matter/{matter_id}/runs", {"limit": limit})
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return result.get("runs", [])
        _log.warning("list_runs: expected list, got %s", type(result).__name__)
        return []

    async def get_run_events(self, matter_id: str, run_id: str) -> list[dict]:
        result = await self._get(f"/matter/{matter_id}/runs/{run_id}/events")
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return result.get("events", [])
        _log.warning("get_run_events: expected list, got %s", type(result).__name__)
        return []

    async def list_issues(self, matter_id: str) -> list[dict]:
        result = await self._get(f"/matter/{matter_id}/issues")
        if isinstance(result, dict):
            return result.get("issues", [])
        if not isinstance(result, list):
            _log.warning("list_issues: expected list, got %s", type(result).__name__)
            return []
        return result

    async def list_assertions(
        self, matter_id: str, limit: int = 50, offset: int = 0
    ) -> list[dict]:
        result = await self._get(f"/matter/{matter_id}/assertions", {"limit": limit, "offset": offset})
        # Service returns paginated envelope: {total, limit, offset, assertions}
        if isinstance(result, dict):
            return result.get("assertions", [])
        return result if isinstance(result, list) else []

    async def get_issue_assertions(self, matter_id: str, issue_id: str) -> list[dict]:
        result = await self._get(f"/matter/{matter_id}/issues/{issue_id}/assertions")
        return result if isinstance(result, list) else []

    async def get_source_agreement(self, matter_id: str, issue_id: str) -> list[dict]:
        result = await self._get(f"/matter/{matter_id}/issues/{issue_id}/source-agreement")
        return result if isinstance(result, list) else []

    async def get_assertion_graph(self, matter_id: str, issue_id: str) -> dict:
        result = await self._get(f"/matter/{matter_id}/issues/{issue_id}/assertion-graph")
        return result if isinstance(result, dict) else {"nodes": [], "edges": []}

    async def get_issue_closure_workbench(self, matter_id: str, issue_id: str) -> dict:
        result = await self._get(f"/matter/{matter_id}/issues/{issue_id}/closure-workbench")
        return result if isinstance(result, dict) else {"error": "Request failed"}

    async def get_issue_authorities(self, matter_id: str, issue_id: str) -> list[dict]:
        result = await self._get(f"/matter/{matter_id}/issues/{issue_id}/authorities")
        return result if isinstance(result, list) else []

    async def list_gaps(self, matter_id: str, limit: int = 50) -> list[dict]:
        result = await self._get(f"/matter/{matter_id}/gaps", {"limit": limit})
        if isinstance(result, dict):
            return result.get("gaps", [])
        if not isinstance(result, list):
            _log.warning("list_gaps: expected list, got %s", type(result).__name__)
            return []
        return result

    async def get_gap_workbench(self, matter_id: str, limit: int = 50, min_materiality: float = 0.0) -> dict:
        params: dict = {"limit": limit}
        if min_materiality > 0:
            params["min_materiality"] = min_materiality
        result = await self._get(f"/matter/{matter_id}/gap-workbench", params)
        return result if isinstance(result, dict) else {"items": []}

    async def get_investigation_readiness(self, matter_id: str) -> dict:
        result = await self._get(f"/matter/{matter_id}/readiness")
        return result if isinstance(result, dict) else {"readiness": "unknown", "blockers": []}

    async def get_assertion_trace(self, matter_id: str, assertion_id: str) -> dict:
        result = await self._get(f"/matter/{matter_id}/assertions/{assertion_id}/trace")
        return result if isinstance(result, dict) else {"error": "Request failed"}

    async def resolve_gap(self, matter_id: str, gap_id: str, resolution_note: str = "") -> bool:
        result = await self._post(f"/matter/{matter_id}/gaps/{gap_id}/resolve", {"resolution_note": resolution_note})
        return bool(result.get("resolved")) if isinstance(result, dict) else False

    async def escalate_gap(self, matter_id: str, gap_id: str) -> bool:
        result = await self._post(f"/matter/{matter_id}/gaps/{gap_id}/escalate", {})
        return bool(result.get("escalated")) if isinstance(result, dict) else False

    async def list_clarifications(self, matter_id: str, limit: int = 20) -> list[dict]:
        result = await self._get(f"/matter/{matter_id}/clarifications", {"limit": limit})
        if isinstance(result, dict):
            return result.get("clarifications", [])
        if not isinstance(result, list):
            _log.warning("list_clarifications: expected list, got %s", type(result).__name__)
            return []
        return result

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
        result = await self._get(f"/matter/{matter_id}/assumptions", {"limit": limit})
        if isinstance(result, dict):
            return result.get("assumptions", [])
        if not isinstance(result, list):
            _log.warning("list_assumptions: expected list, got %s", type(result).__name__)
            return []
        return result

    async def update_assumption_status(self, matter_id: str, assumption_id: str, status: str, reason: str = "") -> bool:
        result = await self._post(
            f"/matter/{matter_id}/assumptions/{assumption_id}/status",
            {"status": status, "reason": reason},
        )
        return bool(result.get("updated")) if isinstance(result, dict) else False

    async def get_quant_facts(self, matter_id: str, limit: int = 200) -> dict:
        result = await self._get(f"/matter/{matter_id}/quant-facts", {"limit": limit})
        if not isinstance(result, dict):
            _log.warning("get_quant_facts: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def get_decision_leverage(self, matter_id: str, top_n: int = 15) -> dict:
        result = await self._get(f"/matter/{matter_id}/decision-leverage", {"top_n": top_n})
        if not isinstance(result, dict):
            _log.warning("get_decision_leverage: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def get_output_quality(self, matter_id: str, run_id: str | None = None) -> dict:
        params = {"run_id": run_id} if run_id else {}
        result = await self._get(f"/matter/{matter_id}/output-quality", params)
        if not isinstance(result, dict):
            _log.warning("get_output_quality: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def get_deliverable_workbench(self, matter_id: str) -> dict:
        result = await self._get(f"/matter/{matter_id}/deliverable-workbench")
        if not isinstance(result, dict):
            _log.warning("get_deliverable_workbench: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def get_scenario_workbench(self, matter_id: str) -> dict:
        result = await self._get(f"/matter/{matter_id}/scenario-branches")
        if not isinstance(result, dict):
            _log.warning("get_scenario_workbench: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def create_scenario_branch(self, matter_id: str, payload: dict) -> dict:
        result = await self._post(f"/matter/{matter_id}/scenario-branches", payload)
        if not isinstance(result, dict):
            _log.warning("create_scenario_branch: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def archive_scenario_branch(self, matter_id: str, branch_id: str) -> dict:
        result = await self._post(f"/matter/{matter_id}/scenario-branches/{branch_id}/archive")
        if not isinstance(result, dict):
            _log.warning("archive_scenario_branch: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def apply_scenario_delta(self, matter_id: str, branch_id: str,
                                   target_kind: str, target_id: str,
                                   operation: str, payload: dict) -> dict:
        result = await self._post(
            f"/matter/{matter_id}/scenario-branches/{branch_id}/deltas",
            json={"target_kind": target_kind, "target_id": target_id,
                  "operation": operation, "payload": payload},
        )
        if not isinstance(result, dict):
            _log.warning("apply_scenario_delta: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def list_scenario_deltas(self, matter_id: str, branch_id: str) -> list[dict]:
        result = await self._get(f"/matter/{matter_id}/scenario-branches/{branch_id}/deltas")
        if isinstance(result, dict):
            return result.get("deltas", [])
        if isinstance(result, list):
            _log.warning("list_scenario_deltas: expected dict envelope, got bare list")
            return result
        _log.warning("list_scenario_deltas: unexpected type %s", type(result).__name__)
        return []

    async def compute_scenario_snapshot(self, matter_id: str, branch_id: str) -> dict:
        result = await self._post(f"/matter/{matter_id}/scenario-branches/{branch_id}/snapshot")
        if not isinstance(result, dict):
            _log.warning("compute_scenario_snapshot: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def compare_scenario_to_baseline(self, matter_id: str, branch_id: str) -> dict:
        result = await self._get(f"/matter/{matter_id}/scenario-branches/{branch_id}/compare")
        if not isinstance(result, dict):
            _log.warning("compare_scenario_to_baseline: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def list_scenario_snapshots(
        self, matter_id: str, branch_id: str, limit: int = 10,
    ) -> dict:
        result = await self._get(
            f"/matter/{matter_id}/scenario-branches/{branch_id}/snapshots",
            {"limit": limit},
        )
        if not isinstance(result, dict):
            _log.warning("list_scenario_snapshots: expected dict, got %s", type(result).__name__)
            return {"branch_id": branch_id, "snapshots": [], "count": 0}
        return result

    async def get_alternative_theory_portfolio(self, matter_id: str, objective_id: str | None = None) -> dict:
        params = f"?objective_id={_url_quote(objective_id, safe='')}" if objective_id else ""
        result = await self._get(f"/matter/{matter_id}/alternative-theories{params}")
        if not isinstance(result, dict):
            _log.warning("get_alternative_theory_portfolio: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def get_dependency_manifest_inspector(self, matter_id: str) -> dict:
        result = await self._get(f"/matter/{matter_id}/dependency-manifests")
        if not isinstance(result, dict):
            _log.warning("get_dependency_manifest_inspector: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def get_steering_impact_preview(self, matter_id: str, action_type: str, payload: dict) -> dict:
        result = await self._post(
            f"/matter/{matter_id}/steering-impact-preview",
            body={"action_type": action_type, "payload": payload},
        )
        if not isinstance(result, dict):
            _log.warning("get_steering_impact_preview: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def get_domain_investigation_readiness(self, matter_id: str) -> dict:
        result = await self._get(f"/matter/{matter_id}/domain-investigation-readiness")
        if not isinstance(result, dict):
            _log.warning("get_domain_investigation_readiness: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def compile_issue_brief(self, matter_id: str) -> dict:
        result = await self._get(f"/matter/{matter_id}/issue-brief")
        if not isinstance(result, dict):
            _log.warning("compile_issue_brief: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def get_objective_coverage(self, matter_id: str) -> dict:
        result = await self._get(f"/matter/{matter_id}/objective-coverage")
        if not isinstance(result, dict):
            _log.warning("get_objective_coverage: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def set_criterion_status(
        self, matter_id: str, predicate_id: str, status: str, reason: str = "",
    ) -> dict:
        pid_enc = _url_quote(predicate_id, safe="")
        result = await self._post(
            f"/matter/{matter_id}/criteria/{pid_enc}/status",
            {"status": status, "reason": reason},
        )
        if not isinstance(result, dict):
            _log.warning("set_criterion_status: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def add_criterion(
        self, matter_id: str, objective_id: str, description: str, burden_side: str = "",
    ) -> dict:
        oid_enc = _url_quote(objective_id, safe="")
        result = await self._post(
            f"/matter/{matter_id}/objectives/{oid_enc}/criteria",
            {"description": description, "burden_side": burden_side},
        )
        if not isinstance(result, dict):
            _log.warning("add_criterion: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def get_assumption_review(self, matter_id: str) -> dict:
        result = await self._get(f"/matter/{matter_id}/assumption-review")
        if not isinstance(result, dict):
            _log.warning("get_assumption_review: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def review_assumption(
        self, matter_id: str, assumption_id: str, decision: str, reason: str = "",
    ) -> dict:
        params = {"assumption_id": assumption_id, "decision": decision, "reason": reason}
        result = await self._post(f"/matter/{matter_id}/assumption-review", params=params)
        if not isinstance(result, dict):
            _log.warning("review_assumption: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def set_issue_priority(self, matter_id: str, issue_id: str, priority: str) -> bool:
        result = await self._post(
            f"/matter/{matter_id}/issues/{issue_id}/priority",
            {"priority": priority},
        )
        return bool(result.get("updated")) if isinstance(result, dict) else False

    async def get_timeline(self, matter_id: str, limit: int = 80, policy_audience: str = "clean") -> list[dict]:
        result = await self._get(f"/matter/{matter_id}/timeline", {"limit": limit, "policy_audience": policy_audience})
        if isinstance(result, dict):
            return result.get("events", [])
        if not isinstance(result, list):
            _log.warning("get_timeline: expected list, got %s", type(result).__name__)
            return []
        return result

    async def get_evidence_matrix(self, matter_id: str, policy_audience: str = "clean") -> dict:
        result = await self._get(f"/matter/{matter_id}/evidence-matrix", {"policy_audience": policy_audience})
        if not isinstance(result, dict):
            _log.warning("get_evidence_matrix: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def get_communication_map(self, matter_id: str) -> dict:
        result = await self._get(f"/matter/{matter_id}/communication-map")
        if not isinstance(result, dict):
            _log.warning("get_communication_map: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def list_belief_revisions(self, matter_id: str, limit: int = 100) -> list[dict]:
        result = await self._get(f"/matter/{matter_id}/belief-revisions", {"limit": limit})
        return result if isinstance(result, list) else []

    async def get_contradictions(self, matter_id: str, limit: int = 100) -> list[dict]:
        result = await self._get(f"/matter/{matter_id}/contradictions", {"limit": limit})
        return result if isinstance(result, list) else []

    async def get_document_versions(self, matter_id: str) -> list[dict]:
        result = await self._get(f"/matter/{matter_id}/document-versions")
        return result if isinstance(result, list) else []

    async def get_operative_document_version(self, matter_id: str, doc_id: str) -> dict:
        result = await self._get(
            f"/matter/{matter_id}/documents/{doc_id}/operative-version",
        )
        if not isinstance(result, dict):
            _log.warning("get_operative_document_version: expected dict, got %s", type(result).__name__)
            return {"doc_id": doc_id, "operative_doc_id": doc_id, "is_operative": True}
        return result

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

    async def get_assertion_history(self, matter_id: str, assertion_id: str, limit: int = 20) -> dict:
        aid_enc = _url_quote(assertion_id, safe="")
        result = await self._get(
            f"/matter/{matter_id}/assertions/{aid_enc}/history",
            {"limit": str(limit)},
        )
        if not isinstance(result, dict):
            _log.warning("get_assertion_history: unexpected response type %s", type(result).__name__)
            return {"assertion_id": assertion_id, "history": [], "count": 0}
        return result

    async def list_content_policy_decisions(self, matter_id: str, limit: int = 50) -> list[dict]:
        result = await self._get(
            f"/matter/{matter_id}/content-policy-audit",
            {"limit": str(limit)},
        )
        if isinstance(result, dict):
            decisions = result.get("decisions", [])
            return decisions if isinstance(decisions, list) else []
        if not isinstance(result, list):
            _log.warning("list_content_policy_decisions: unexpected response type %s", type(result).__name__)
            return []
        return result

    async def get_quant_thresholds(
        self, matter_id: str, currency: str = "USD",
        *, exposure_high: float = 10_000.0, disputed_fraction_min: float = 0.10,
    ) -> list[dict]:
        result = await self._get(f"/matter/{matter_id}/quant-thresholds", {
            "currency": currency,
            "exposure_high": exposure_high,
            "disputed_fraction_min": disputed_fraction_min,
        })
        return result if isinstance(result, list) else []

    async def get_amount_conflicts(self, matter_id: str) -> list[dict]:
        result = await self._get(f"/matter/{matter_id}/reconciliation/conflicts")
        if not isinstance(result, list):
            _log.warning("get_amount_conflicts: unexpected response type %s", type(result).__name__)
            return []
        return result

    async def detect_quant_conflicts(self, matter_id: str) -> list[str]:
        result = await self._post(f"/matter/{matter_id}/detect-quant-conflicts", {})
        if isinstance(result, dict):
            return result.get("gap_ids", [])
        if not isinstance(result, list):
            _log.warning("detect_quant_conflicts: unexpected response type %s", type(result).__name__)
            return []
        return result

    async def get_reconciliation(self, matter_id: str, currency: str = "USD") -> dict:
        result = await self._get(f"/matter/{matter_id}/reconciliation", {"currency": currency})
        if not isinstance(result, dict):
            _log.warning("get_reconciliation: unexpected response type %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def get_invoice_chain(self, matter_id: str, currency: str = "USD") -> list:
        result = await self._get(f"/matter/{matter_id}/reconciliation/invoices", {"currency": currency})
        if not isinstance(result, list):
            _log.warning("get_invoice_chain: unexpected response type %s", type(result).__name__)
            return []
        return result

    async def get_damages_waterfall(self, matter_id: str, currency: str = "USD") -> list[dict]:
        result = await self._get(f"/matter/{matter_id}/damages-waterfall", {"currency": currency})
        return result if isinstance(result, list) else []

    async def get_system_health(self, matter_id: str) -> dict:
        result = await self._get(f"/matter/{matter_id}/system-health")
        return result if isinstance(result, dict) else {}

    async def compute_proof_state(self, matter_id: str) -> dict:
        result = await self._post(f"/matter/{matter_id}/proof-state/compute")
        if not isinstance(result, dict):
            _log.warning("compute_proof_state: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def compute_issue_proof_state(self, matter_id: str, issue_id: str) -> dict:
        result = await self._post(f"/matter/{matter_id}/issues/{issue_id}/proof-state/compute")
        return result if isinstance(result, dict) else {}

    async def flush_pending(self, matter_id: str) -> dict:
        result = await self._post(f"/matter/{matter_id}/flush-pending")
        if not isinstance(result, dict):
            _log.warning("flush_pending: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def get_so_scorecard(self, matter_id: str) -> dict:
        result = await self._get(f"/matter/{matter_id}/so-scorecard")
        return result if isinstance(result, dict) else {}

    async def get_domain_profile_summary(
        self, matter_id: str, profile_id: str | None = None
    ) -> dict:
        params = {}
        if profile_id is not None:
            params["profile_id"] = profile_id
        result = await self._get(f"/matter/{matter_id}/domain-profile", params)
        if not isinstance(result, dict):
            _log.warning("get_domain_profile_summary: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def get_domain_composition(self, matter_id: str) -> dict:
        result = await self._get(f"/matter/{matter_id}/domain-composition")
        return result if isinstance(result, dict) else {}

    async def list_documents_needing_profile(
        self, matter_id: str, limit: int = 50
    ) -> list[dict]:
        result = await self._get(
            f"/matter/{matter_id}/document-triage", {"limit": limit}
        )
        if isinstance(result, dict):
            docs = result.get("documents", [])
            return docs if isinstance(docs, list) else []
        if isinstance(result, list):
            return result
        _log.warning("list_documents_needing_profile: unexpected type %s", type(result).__name__)
        return []

    async def get_document_card(
        self,
        matter_id: str,
        *,
        relative_path: str | None = None,
        doc_id: str | None = None,
    ) -> dict:
        params: dict[str, str] = {}
        if relative_path:
            params["relative_path"] = relative_path
        if doc_id:
            params["doc_id"] = doc_id
        result = await self._get(
            f"/matter/{matter_id}/documents/card", params
        )
        if not isinstance(result, dict):
            _log.warning("get_document_card: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def patch_document_card(
        self,
        matter_id: str,
        doc_id: str,
        fields: dict,
    ) -> dict:
        from urllib.parse import quote as _url_quote_local
        did = _url_quote_local(doc_id, safe="")
        result = await self._patch(
            f"/matter/{matter_id}/documents/{did}/card", fields
        )
        if not isinstance(result, dict):
            _log.warning("patch_document_card: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def get_taint_summary(
        self, matter_id: str, limit: int = 50
    ) -> dict:
        result = await self._get(
            f"/matter/{matter_id}/taint-summary", {"limit": limit}
        )
        if not isinstance(result, dict):
            _log.warning("get_taint_summary: expected dict, got %s", type(result).__name__)
            return {"by_class": [], "by_kind": [], "recent": [], "total": 0}
        return result

    async def answer_clarification(
        self, matter_id: str, question_id: str, answer_text: str
    ) -> bool:
        qid_enc = _url_quote(question_id, safe="")
        result = await self._post(
            f"/matter/{matter_id}/clarifications/{qid_enc}/answer",
            {"answer_text": answer_text},
        )
        return bool(result.get("found", False)) if isinstance(result, dict) else False

    async def list_trust_overrides(self, matter_id: str) -> list[dict]:
        result = await self._get(f"/matter/{matter_id}/trust-overrides")
        if isinstance(result, dict):
            return result.get("overrides", [])
        return result if isinstance(result, list) else []

    async def set_trust_override(
        self, matter_id: str, document_pattern: str, trust_level: str, note: str = ""
    ) -> str:
        result = await self._post(
            f"/matter/{matter_id}/trust-overrides",
            {"document_pattern": document_pattern, "trust_level": trust_level, "note": note or None},
        )
        return result.get("override_id", "") if isinstance(result, dict) else ""

    async def delete_trust_override(self, matter_id: str, document_pattern: str) -> bool:
        pattern_enc = _url_quote(document_pattern, safe="")
        try:
            r = await self._client.delete(f"/matter/{matter_id}/trust-overrides/{pattern_enc}")
            r.raise_for_status()
            return True
        except Exception as exc:
            _log.warning("Failed to delete trust override %r for %s: %s", document_pattern, matter_id, exc)
            return False

    async def generate_clarifications(self, matter_id: str, top_n: int = 3) -> list[str]:
        result = await self._post(
            f"/matter/{matter_id}/generate-clarifications",
            {"top_n": top_n},
        )
        return result.get("question_ids", []) if isinstance(result, dict) else []

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

    async def search_assertions(self, matter_id: str, query: str, limit: int = 20) -> list[dict]:
        result = await self._get(f"/matter/{matter_id}/assertions/search", {"q": query, "limit": limit})
        return result if isinstance(result, list) else []

    async def find_duplicate_actors(self, matter_id: str, min_prefix_len: int = 6) -> list[dict]:
        result = await self._get(
            f"/matter/{matter_id}/actors/duplicates",
            {"min_prefix_len": min_prefix_len},
        )
        if not isinstance(result, list):
            _log.warning("find_duplicate_actors: unexpected response type %s", type(result).__name__)
            return []
        return result

    async def merge_actors(self, matter_id: str, keep_id: str, merge_id: str) -> dict:
        result = await self._post(
            f"/matter/{matter_id}/actors/{_url_quote(keep_id, safe='')}/merge/{_url_quote(merge_id, safe='')}",
            {},
        )
        if not isinstance(result, dict):
            _log.warning("merge_actors: unexpected response type %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def resolve_actor(self, matter_id: str, name: str) -> dict:
        result = await self._get(f"/matter/{matter_id}/actors/resolve", {"name": name})
        return result if isinstance(result, dict) else {"actor_id": None}

    async def get_decision_context(self, matter_id: str) -> "dict | None":
        result = await self._get(f"/matter/{matter_id}/decision-context")
        return result if isinstance(result, dict) else None

    async def set_decision_context(
        self, matter_id: str,
        decision_maker_type: Optional[str] = None,
        decision_maker_name: Optional[str] = None,
        objective: Optional[str] = None,
        strategic_notes: Optional[str] = None,
        scope_narrow: bool = False,
    ) -> str:
        body: dict = {"scope_narrow": scope_narrow}
        if decision_maker_type:
            body["decision_maker_type"] = decision_maker_type
        if decision_maker_name:
            body["decision_maker_name"] = decision_maker_name
        if objective:
            body["objective"] = objective
        if strategic_notes:
            body["strategic_notes"] = strategic_notes
        result = await self._client.put(
            f"/matter/{matter_id}/decision-context",
            json=body,
        )
        result.raise_for_status()
        data = result.json()
        return data.get("id", "") if isinstance(data, dict) else ""

    async def clear_decision_context(self, matter_id: str) -> bool:
        r = await self._client.delete(f"/matter/{matter_id}/decision-context")
        r.raise_for_status()
        return True

    async def list_annotations(self, matter_id: str, document: Optional[str] = None) -> list[dict]:
        params = {}
        if document:
            params["document"] = document
        result = await self._get(f"/matter/{matter_id}/annotations", params)
        if isinstance(result, dict):
            return result.get("annotations", [])
        return result if isinstance(result, list) else []

    async def add_annotation(
        self, matter_id: str, document_pattern: str,
        annotation_text: str, annotation_type: str = "strategic",
    ) -> str:
        result = await self._post(
            f"/matter/{matter_id}/annotations",
            {"document_pattern": document_pattern, "annotation_text": annotation_text, "annotation_type": annotation_type},
        )
        return result.get("annotation_id", "") if isinstance(result, dict) else ""

    async def delete_annotation(self, matter_id: str, annotation_id: str) -> bool:
        r = await self._client.delete(f"/matter/{matter_id}/annotations/{_url_quote(annotation_id, safe='')}")
        r.raise_for_status()
        data = r.json()
        return data.get("status") == "deleted" if isinstance(data, dict) else False

    async def export_matter_summary(self, matter_id: str) -> dict:
        result = await self._get(f"/matter/{matter_id}/export-summary")
        return result if isinstance(result, dict) else {}

    async def get_document_intelligence(self, matter_id: str) -> dict:
        cards = await self._get(f"/matter/{matter_id}/documents/cards")
        return cards if isinstance(cards, dict) else {"cards": [], "total_inventory": 0, "ingested_count": 0}

    async def get_proof_state_summary(self, matter_id: str) -> dict:
        result = await self._get(f"/matter/{matter_id}/proof-state")
        if not isinstance(result, dict):
            _log.warning("get_proof_state_summary: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def get_authority_network(self, matter_id: str) -> dict:
        result = await self._get(f"/matter/{matter_id}/authority-network")
        if isinstance(result, dict):
            return result
        return {"authorities": [], "issue_links": {}}

    async def upsert_authority(
        self,
        matter_id: str,
        citation: str,
        *,
        authority_type: str = "case",
        name: str | None = None,
        jurisdiction: str | None = None,
        weight: str = "persuasive",
    ) -> dict:
        body: dict = {"citation": citation, "authority_type": authority_type, "weight": weight}
        if name:
            body["name"] = name
        if jurisdiction:
            body["jurisdiction"] = jurisdiction
        result = await self._post(f"/matter/{matter_id}/authorities", body)
        if not isinstance(result, dict):
            return {}
        if "authority_id" not in result and "id" in result:
            result["authority_id"] = result["id"]
        return result

    async def link_authority_to_issue(
        self,
        matter_id: str,
        authority_id: str,
        issue_id: str,
        relevance: str = "supporting",
    ) -> dict:
        r = await self._client.post(
            f"/matter/{matter_id}/authorities/{authority_id}/issues/{issue_id}",
            params={"relevance": relevance},
        )
        r.raise_for_status()
        result = r.json()
        return result if isinstance(result, dict) else {"status": "linked"}

    async def unlink_authority_from_issue(
        self, matter_id: str, authority_id: str, issue_id: str
    ) -> dict:
        r = await self._client.delete(
            f"/matter/{matter_id}/authorities/{authority_id}/issues/{issue_id}"
        )
        r.raise_for_status()
        return {"status": "unlinked"}

    async def search_authorities(
        self, matter_id: str, query: str, limit: int = 20
    ) -> list[dict]:
        result = await self._get(
            f"/matter/{matter_id}/authorities",
            {"search": query, "limit": limit},
        )
        return result if isinstance(result, list) else []

    # ------------------------------------------------------------------ #
    # Cost analytics                                                       #
    # ------------------------------------------------------------------ #

    async def get_cost_breakdown(
        self, matter_id: str, run_id: Optional[str] = None
    ) -> dict:
        params = {}
        if run_id:
            params["run_id"] = run_id
        result = await self._get(f"/matter/{matter_id}/cost-breakdown", params)
        return result if isinstance(result, dict) else {}

    async def get_cost_anomalies(
        self, matter_id: str, limit: int = 10, run_id: Optional[str] = None
    ) -> list[dict]:
        params: dict = {"limit": limit}
        if run_id:
            params["run_id"] = run_id
        result = await self._get(f"/matter/{matter_id}/cost-anomalies", params)
        return result if isinstance(result, list) else []

    # ------------------------------------------------------------------ #
    # Review queue (SO-3)                                                  #
    # ------------------------------------------------------------------ #

    async def get_review_queue(
        self, matter_id: str, limit: int = 50, offset: int = 0,
        target_kind: Optional[str] = None,
    ) -> list[dict]:
        params: dict = {"limit": limit, "offset": offset}
        if target_kind:
            params["target_kind"] = target_kind
        result = await self._get(f"/matter/{matter_id}/review-queue", params)
        if isinstance(result, dict):
            return result.get("queue", [])
        return result if isinstance(result, list) else []

    async def count_review_queue(self, matter_id: str) -> dict:
        result = await self._get(f"/matter/{matter_id}/review-queue/count")
        return result if isinstance(result, dict) else {}

    async def verify_target(
        self, matter_id: str, target_kind: str, target_id: str,
        *, reviewed_by_kind: str = "user",
        reviewed_by_id: Optional[str] = None,
        review_note: Optional[str] = None,
        review_scope: str = "extraction_correct",
    ) -> str:
        body: dict = {
            "target_kind": target_kind,
            "target_id": target_id,
            "status": "verified",
            "reviewed_by_kind": reviewed_by_kind,
            "review_scope": review_scope,
        }
        if reviewed_by_id:
            body["reviewed_by_id"] = reviewed_by_id
        if review_note:
            body["review_note"] = review_note
        result = await self._post(f"/matter/{matter_id}/verify", body)
        return result.get("verification_id", "") if isinstance(result, dict) else ""

    async def reject_target(
        self, matter_id: str, target_kind: str, target_id: str,
        *, rejection_reason: str,
        reviewed_by_kind: str = "user",
        reviewed_by_id: Optional[str] = None,
        review_note: Optional[str] = None,
    ) -> str:
        body: dict = {
            "target_kind": target_kind,
            "target_id": target_id,
            "status": "rejected",
            "reviewed_by_kind": reviewed_by_kind,
            "rejection_reason": rejection_reason,
        }
        if reviewed_by_id:
            body["reviewed_by_id"] = reviewed_by_id
        if review_note:
            body["review_note"] = review_note
        result = await self._post(f"/matter/{matter_id}/verify", body)
        return result.get("verification_id", "") if isinstance(result, dict) else ""

    async def bulk_verify_by_document(
        self, matter_id: str, document_ref: str,
        *, reviewed_by_kind: str = "user",
        reviewed_by_id: Optional[str] = None,
    ) -> list[str]:
        body: dict = {
            "document_ref": document_ref,
            "reviewed_by_kind": reviewed_by_kind,
        }
        if reviewed_by_id:
            body["reviewed_by_id"] = reviewed_by_id
        result = await self._post(f"/matter/{matter_id}/verify/bulk-by-document", body)
        return result.get("verification_ids", []) if isinstance(result, dict) else []

    async def bulk_verify_by_span(
        self, matter_id: str, span_id: str,
        *, reviewed_by_kind: str = "user",
        reviewed_by_id: Optional[str] = None,
        review_note: Optional[str] = None,
        review_scope: str = "extraction_correct",
    ) -> list[str]:
        body: dict = {
            "span_id": span_id,
            "reviewed_by_kind": reviewed_by_kind,
            "review_scope": review_scope,
        }
        if reviewed_by_id:
            body["reviewed_by_id"] = reviewed_by_id
        if review_note:
            body["review_note"] = review_note
        result = await self._post(f"/matter/{matter_id}/verify/bulk-by-span", body)
        return result.get("verification_ids", []) if isinstance(result, dict) else []

    async def bulk_verify_assertion_ids(
        self, matter_id: str, assertion_ids: list[str],
        *, reviewed_by_kind: str = "user",
        reviewed_by_id: Optional[str] = None,
        review_note: Optional[str] = None,
    ) -> list[str]:
        body: dict = {
            "assertion_ids": assertion_ids,
            "reviewed_by_kind": reviewed_by_kind,
        }
        if reviewed_by_id:
            body["reviewed_by_id"] = reviewed_by_id
        if review_note:
            body["review_note"] = review_note
        result = await self._post(f"/matter/{matter_id}/verify/bulk-by-ids", body)
        return result.get("verification_ids", []) if isinstance(result, dict) else []

    async def list_candidate_assertions_for_document(
        self, matter_id: str, document_ref: str,
    ) -> list[dict]:
        result = await self._get(
            f"/matter/{matter_id}/review-queue/by-document",
            {"document_ref": document_ref},
        )
        if isinstance(result, dict):
            return result.get("assertions", [])
        return result if isinstance(result, list) else []

    async def get_document_console(self, matter_id: str, document_ref: str) -> dict:
        result = await self._get(
            f"/matter/{matter_id}/document-console",
            {"document_ref": document_ref},
        )
        return result if isinstance(result, dict) else {}

    async def get_quant_ontology(self, matter_id: str) -> dict:
        result = await self._get(f"/matter/{matter_id}/quant-ontology")
        return result if isinstance(result, dict) else {}

    async def approve_metric_alias(
        self, matter_id: str, raw_label: str, canonical_metric: str, unit: str | None = None,
    ) -> bool:
        params = {"raw_label": raw_label, "canonical_metric": canonical_metric}
        if unit:
            params["unit"] = unit
        result = await self._post(f"/matter/{matter_id}/quant-ontology/approve", params=params)
        return isinstance(result, dict) and result.get("success", False)

    async def get_answer_audits(
        self, matter_id: str, manifest_hash: str | None = None,
    ) -> dict:
        params = {}
        if manifest_hash:
            params["manifest_hash"] = manifest_hash
        result = await self._get(f"/matter/{matter_id}/answer-audits", params or None)
        return result if isinstance(result, dict) else {}

    async def resolve_contradiction(
        self, matter_id: str, attacker_id: str, attacked_id: str,
        decision: str, rationale: str = "",
    ) -> dict:
        params = {
            "attacker_id": attacker_id,
            "attacked_id": attacked_id,
            "decision": decision,
            "rationale": rationale,
        }
        result = await self._post(f"/matter/{matter_id}/resolve-contradiction", params=params)
        return result if isinstance(result, dict) else {}

    async def get_knowledge_seeds(self, matter_id: str) -> dict:
        result = await self._get(f"/matter/{matter_id}/knowledge-seeds")
        if not isinstance(result, dict):
            _log.warning("get_knowledge_seeds: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def review_knowledge_seed(
        self, matter_id: str, seed_id: str, decision: str,
        review_note: str = "",
    ) -> dict:
        params = {"seed_id": seed_id, "decision": decision, "review_note": review_note}
        result = await self._post(f"/matter/{matter_id}/knowledge-seeds/review", params=params)
        if not isinstance(result, dict):
            _log.warning("review_knowledge_seed: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def promote_knowledge_seed(
        self, matter_id: str, seed_kind: str, domain_profile_id: str,
        payload_json: str, source_matter_id: str | None = None,
    ) -> dict:
        params = {
            "seed_kind": seed_kind,
            "domain_profile_id": domain_profile_id,
            "payload_json": payload_json,
        }
        if source_matter_id:
            params["source_matter_id"] = source_matter_id
        result = await self._post(f"/matter/{matter_id}/knowledge-seeds/promote", params=params)
        if not isinstance(result, dict):
            _log.warning("promote_knowledge_seed: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def list_reviewable_documents(self, matter_id: str) -> list[dict]:
        result = await self._get(
            f"/matter/{matter_id}/review-queue/documents",
        )
        if isinstance(result, dict):
            return result.get("documents", [])
        return result if isinstance(result, list) else []

    async def reclassify_document_sensitivity(
        self, matter_id: str, doc_id: str, privilege_flag: bool,
        reviewed_by_kind: str = "user", reviewed_by_id: str | None = None,
    ) -> dict:
        body = {"privilege_flag": privilege_flag, "reviewed_by_kind": reviewed_by_kind}
        if reviewed_by_id:
            body["reviewed_by_id"] = reviewed_by_id
        result = await self._post(
            f"/matter/{matter_id}/documents/{doc_id}/reclassify-sensitivity", body,
        )
        if not isinstance(result, dict):
            _log.warning("reclassify_document_sensitivity: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def mark_document_stale(
        self, matter_id: str, doc_id: str, reason: str = "manual_stale",
    ) -> dict:
        result = await self._post(
            f"/matter/{matter_id}/documents/{doc_id}/mark-stale", {"reason": reason},
        )
        if not isinstance(result, dict):
            _log.warning("mark_document_stale: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def mark_span_stale(
        self, matter_id: str, span_id: str, reason: str = "manual_span_stale",
    ) -> dict:
        result = await self._post(
            f"/matter/{matter_id}/spans/{span_id}/mark-stale", {"reason": reason},
        )
        if not isinstance(result, dict):
            _log.warning("mark_span_stale: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def get_verification_events(
        self, matter_id: str, target_kind: Optional[str] = None,
        target_id: Optional[str] = None, limit: int = 50,
    ) -> list[dict]:
        params: dict = {"limit": limit}
        if target_kind:
            params["target_kind"] = target_kind
        if target_id:
            params["target_id"] = target_id
        result = await self._get(f"/matter/{matter_id}/verification-events", params)
        if isinstance(result, dict):
            return result.get("events", [])
        return result if isinstance(result, list) else []

    async def get_query_context(self, matter_id: str) -> dict:
        result = await self._get(f"/matter/{matter_id}/query-context")
        if not isinstance(result, dict):
            _log.warning("get_query_context: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result

    async def get_source_calibration(self, matter_id: str) -> dict:
        result = await self._get(f"/matter/{matter_id}/source-calibration")
        if not isinstance(result, dict):
            _log.warning("get_source_calibration: expected dict, got %s", type(result).__name__)
            return {"error": f"unexpected response type: {type(result).__name__}"}
        return result
