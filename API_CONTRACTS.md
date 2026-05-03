# Irys RLM API Contract v1

Status: **design contract / aspirational roadmap**. The `/api/v1/*` paths
specified later in this document are a target shape, not the runtime.
`src/irys/service/api.py` ships **70 endpoints at un-prefixed paths** such as
`/matter/{matter_id}/...`. Those are the real production surface — read
section "Current Implementation Status" below for what clients can actually
call today.
Date: 2026-04-15. Updated 2026-05-03 (CAS revision endpoints for
correct/verify/reject, domain-composition endpoint, verify/revisions
target validation).

## Current Implementation Status

The live FastAPI app exposes these 70 endpoints. No `/api/v1` prefix today;
no `api_version` response wrapper; no `Idempotency-Key` or `If-Match` on
mutations; no OIDC/OAuth2 auth gate; no permission strings; no cursor
pagination (list endpoints use `limit`/`offset`). Span-level privilege
taint, deep-graph reads, and operation_job async envelopes are NOT
implemented. When a new endpoint lands, update this section AND add its
spec row below so the two stay in sync.

**System + Investigation**
- `GET /`, `GET /health`
- `POST /investigate`, `GET /investigate/{job_id}`, `GET /jobs`
- `POST /search`
- `POST /upload/investigate`, `/upload/search`, `/upload/investigate/sync`
- `POST /investigate/urls`, `/search/urls`, `/investigate/urls/sync`

**Matter model — facts, issues, gaps, overview**
- `GET /matter/{matter_id}` (stats), `/overview`
- `GET /matter/{matter_id}/issues`
- `GET /matter/{matter_id}/assertions`, `/assertions/{assertion_id}/history`
- `POST /matter/{matter_id}/assertions/{assertion_id}/correct` —
  CAS-protected when `expected_revisions` provided (409 on stale view)
- `GET /matter/{matter_id}/assertions/{assertion_id}/correct/revisions`
  — snapshot namespace revisions for CAS-protected correction
- `GET /matter/{matter_id}/gaps`
- `GET /matter/{matter_id}/clarifications`,
  `POST /clarifications/{question_id}/answer`
- `GET /matter/{matter_id}/metrics` — real SO KPIs (task #7)
- `GET /matter/{matter_id}/decision-context` (+ `PUT`, `DELETE`)

**P0.3 Review Queue (shipped)**
- `GET /matter/{matter_id}/review-queue`
- `GET /matter/{matter_id}/review-queue/count` — cheap unread count for
  the sidebar badge (OPT-2a)
- `POST /matter/{matter_id}/verify` — CAS-protected when
  `expected_revisions` provided (409 on stale view)
- `GET /matter/{matter_id}/verify/revisions` — snapshot namespace
  revisions for CAS-protected verify/reject (validates target existence
  for concrete kinds: assertion, evidence_edge, quant_fact)
- `POST /matter/{matter_id}/verify/bulk-by-document`
- `POST /matter/{matter_id}/verify/bulk-by-span`
- `GET /matter/{matter_id}/verification-events`

**Cost visibility (shipped)**
- `GET /matter/{matter_id}/cost-breakdown` — per-model / per-run cost
  rollup
- `GET /matter/{matter_id}/cost-anomalies` — runs whose spend-per-fact
  flags outside the rolling median band

**SO-3 steering (shipped)**
- `POST /matter/{matter_id}/stop`
- `POST /matter/{matter_id}/runs/{run_id}/redirect`
- `POST /matter/{matter_id}/runs/{run_id}/stop`
- `POST /matter/{matter_id}/runs/{run_id}/resume`
- `GET /matter/{matter_id}/runs`, `/runs/{run_id}/events`,
  `/runs/{run_id}/events/stream`
- `GET /matter/{matter_id}/steering-surface`
- `POST /matter/{matter_id}/trust-overrides` (+ `GET`, `DELETE`)
- `POST /matter/{matter_id}/annotations` (+ `GET`, `DELETE`)

**Domain composition (SO-5)**
- `GET /matter/{matter_id}/domain-composition` — primary domain,
  facets with confidence, composed trust weights, detection events

**SO-6 quantitative**
- `GET /matter/{matter_id}/reconciliation`, `/reconciliation/invoices`,
  `/reconciliation/conflicts`
- `GET /matter/{matter_id}/damages-waterfall`

**Authorities (SO-4)**
- `POST /matter/{matter_id}/authorities`, `GET`, `GET /{authority_id}`
- `POST /authorities/{authority_id}/issues/{issue_id}` (link), `DELETE`
- `GET /matter/{matter_id}/issues/{issue_id}/authorities`

**Proof + visual (Priority 2)**
- `POST /matter/{matter_id}/proof-state/compute`,
  `/issues/{issue_id}/proof-state/compute`
- `GET /matter/{matter_id}/proof-state`, `/issues/{issue_id}/proof-state`,
  `/proof-state/gaps`
- `GET /matter/{matter_id}/timeline` — `policy_audience='clean'` default
  (P0.5 commit 4)
- `GET /matter/{matter_id}/evidence-matrix` — `policy_audience='clean'`
  default
- `GET /matter/{matter_id}/communication-map`
- `GET /matter/{matter_id}/llm-calls`

**Actor resolution**
- `GET /matter/{matter_id}/actors/duplicates`,
  `POST /actors/{keep_id}/merge/{merge_id}`, `GET /actors/resolve`

**Maintenance**
- `POST /matter/{matter_id}/flush-pending`

### Cascade response surface (adv#11 Fix 3)

`POST /upload/investigate/sync` (and async `GET /investigate/{job_id}`
polling) now expose two new fields so clients can render family-
specific UI without parsing prose:

- `route` — audit dict with `classifier_family` (NANO's initial read),
  `terminal_family` (what actually ran after any escalation; one of
  `read`, `query`, `trace`, `steer`, `compare`, `scenario`,
  `deliverable`, `clarify`, `investigate`, `read_infra_failure`),
  plus `rationale` and `escalation_reason`.
- `family_payload` — family-specific structured fields. Present keys
  depend on `terminal_family`:
  - `query`: `query_intent`, `query_row_count`
  - `steer`: `steer_action`, `steer_target_hint`, `steer_candidates`
  - `trace`: `trace_target_kind`, `trace_target_id`
  - `deliverable`: `deliverable_intent`, `deliverable_row_count`
  - `read_infra_failure`: `read_infra_failure: true`

Both fields are `null` for legacy/non-cascade callers.

Everything below this line uses `/api/v1/*` and describes the **target**
shape. Treat it as the post-Phase-0 design reference. For what actually
ships today, read the list above or `src/irys/service/api.py`.

---

This contract standardizes the full matter-intelligence API surface distilled
from the prior planning artifacts now consolidated in `docs/PROJECT_CONTEXT.md`.
Current implementation code lives under
`src/irys/service`, `src/irys/matter`, and `src/irys/rlm`; existing singular
paths such as `/matter/{matter_id}` should remain compatibility aliases while
new clients target the canonical v1 paths below.

## Versioning

Canonical prefix: `/api/v1`.

- `v1` is stable at the HTTP path level. Backward-compatible response field
  additions are allowed.
- Removing fields, changing enum values, changing output-policy defaults, or
  changing verification semantics requires `/api/v2`.
- Every response includes `api_version: "v1"` unless it is wrapped by a generic
  operation or page envelope.
- Compatibility aliases must be marked deprecated in OpenAPI and return
  `Deprecation` plus `Link` headers after `/api/v1` exists.
- Mutations should accept `Idempotency-Key`. Reviewed/legal mutations should
  support `If-Match` or request body `expected_version`.

## Authentication And Authorization

Production API requires OIDC/OAuth2 bearer tokens. The current prototype has no
auth guard; this is a deployment blocker.

Required token claims: `sub`, `tenant_id`, `roles`, and either
`matter_permissions` or a server-resolvable authorization reference.

Permission strings:

- `matter:read`: read matter views allowed by output policy.
- `matter:query`: run hot/cold queries.
- `matter:steer`: pause, resume, cancel, redirect, trust, correct, and context.
- `matter:review`: verify/reject candidate intelligence.
- `matter:attorney_verify`: promote legal facts to `verified`; must be a human
  user token, not a service account.
- `matter:privilege`: view and override privilege classifications.
- `matter:admin`: maintenance and destructive issue/gap actions.

## Rate Limits

Use token buckets keyed by `(tenant_id, sub, matter_id, bucket)`.

- `read`: 300/min/user, 3000/min/tenant.
- `dashboard`: 120/min/user.
- `mutation`: 60/min/user.
- `verification_batch`: 10/min/user, 50/hour/tenant.
- `hot_query`: 30/min/user, 300/hour/tenant.
- `cold_query`: 5/hour/user, 20/hour/tenant, max 1 active cold query per matter
  unless queued by the operation service.
- `maintenance`: 3/hour/matter.
- `websocket`: 5 concurrent matter sockets per user, 50 per tenant.

## Pagination

Canonical pagination is cursor-based.

- Query params: `limit: int = 50` with min `1`, max `200`; `cursor: str | None`;
  endpoint-specific `sort`.
- Page response: `items`, `next_cursor`, `has_more`, and optional
  `total_estimate`.
- Cursors encode a stable ordering tuple such as `(created_at, id)`, not a
  mutable offset.

## Error Format

All errors use:

```json
{
  "error": {
    "code": "assertion_not_found",
    "message": "Assertion was not found in this matter.",
    "details": {"assertion_id": "as_123"},
    "request_id": "req_01H",
    "retryable": false
  }
}
```

Common errors for all endpoints: `400 bad_request`, `401 unauthenticated`,
`403 forbidden`, `404 matter_not_found`, `409 conflict`,
`422 validation_error`, `429 rate_limited`, `500 internal_error`,
`503 unavailable`.

Endpoint-specific errors are listed in the matrix.

## Async Operation Pattern

Cold queries, large batch verification, and maintenance triggers can exceed HTTP
latency budgets and return `202 OperationAccepted`.

Polling endpoint:

`GET /api/v1/operations/{operation_id}`

Auth: same matter permission as the operation target. Rate bucket: `read`.
Reads: `operation_job`, `run_session`, `ledger_event`. Errors: common plus
`404 operation_not_found`.

| Method and path | Query params | Body | Response | Auth | Rate | Mode | Reads | Writes | Realtime | Specific errors |
|---|---|---|---|---|---|---|---|---|---|---|
| `GET /api/v1/operations/{operation_id}` | `include_result=true` | None | `OperationStatusResponse` | Same permission as operation target | `read` | Sync | `operation_job`, `run_session`, `ledger_event` | None | `operation.updated` | `404 operation_not_found`, `403 operation_scope_forbidden` |

## Pydantic Schemas

These schemas are normative. They can be split into modules, but OpenAPI should
preserve names and field semantics.

```python
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class QueryMode(str, Enum):
    AUTO = "auto"
    HOT = "hot"
    COLD = "cold"


class OperationStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class VerificationStatus(str, Enum):
    CANDIDATE = "candidate"
    VERIFIED = "verified"
    REJECTED = "rejected"
    STALE = "stale"


class VerificationTargetKind(str, Enum):
    ASSERTION = "assertion"
    ISSUE_PREDICATE = "issue_predicate"
    EVIDENCE_EDGE = "evidence_edge"
    QUANT_FACT = "quant_fact"
    AUTHORITY = "authority"
    DOCUMENT_CARD = "document_card"
    PRIVILEGE_CLASSIFICATION = "privilege_classification"


class BeliefState(str, Enum):
    ALLEGED = "alleged"
    ARGUED = "argued"
    ADMITTED = "admitted"
    OPERATIVE = "operative"
    PERFORMED = "performed"
    NOT_PERFORMED = "not_performed"
    DISPUTED = "disputed"
    SUPERSEDED = "superseded"
    WITHDRAWN = "withdrawn"
    INFERRED = "inferred"
    RESOLVED = "resolved"
    UNKNOWN = "unknown"


class IssueKind(str, Enum):
    ISSUE = "issue"
    CLAIM = "claim"
    DEFENSE = "defense"
    ELEMENT = "element"
    CONDITION = "condition"
    DAMAGES_COMPONENT = "damages_component"
    PROCEDURAL_BARRIER = "procedural_barrier"
    EVIDENTIARY_BOTTLENECK = "evidentiary_bottleneck"
    DILIGENCE_RED_FLAG = "diligence_red_flag"
    COMPLIANCE_FAILURE = "compliance_failure"


class EvidenceRelation(str, Enum):
    SUPPORTS = "supports"
    ATTACKS = "attacks"
    ESTABLISHES = "establishes"
    NEGATES = "negates"
    CONTEXT = "context"


class PolicyAudience(str, Enum):
    INTERNAL_PRIVILEGED = "internal_privileged"
    INTERNAL_CLEAN = "internal_clean"
    CLIENT_SHARE = "client_share"
    COURT = "court"
    OPPOSING_PARTY = "opposing_party"
    PUBLIC = "public"
    EXTERNAL_AI = "external_ai"


class PolicyMode(str, Enum):
    INTERNAL = "internal"
    CLEAN = "clean"


class PrivilegeClass(str, Enum):
    UNKNOWN = "unknown"
    NOT_PRIVILEGED = "not_privileged"
    CONFIDENTIAL = "confidential"
    ATTORNEY_CLIENT = "attorney_client"
    WORK_PRODUCT = "work_product"
    COMMON_INTEREST = "common_interest"
    PARTIALLY_PRIVILEGED = "partially_privileged"
    PRIVILEGED = "privileged"
    WITHHELD = "withheld"


class TrustLevel(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXCLUDED = "excluded"


class GapStatus(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"
    STALE = "stale"


class GapType(str, Enum):
    MISSING_DOCUMENT = "missing_document"
    MISSING_METADATA = "missing_metadata"
    MISSING_ISSUE_PREDICATE = "missing_issue_predicate"
    MISSING_AUTHORITY = "missing_authority"
    MISSING_USER_CONTEXT = "missing_user_context"
    MISSING_QUANTITATIVE_INPUT = "missing_quantitative_input"
    UNRESOLVED_CONTRADICTION = "unresolved_contradiction"
    EXPECTED_ABSENT_ATTACHMENT = "expected_absent_attachment"
    EXPECTED_ABSENT_NOTICE = "expected_absent_notice"


class ErrorDetail(APIModel):
    code: str = Field(..., min_length=1, max_length=80)
    message: str = Field(..., min_length=1, max_length=500)
    details: dict[str, Any] | None = None
    request_id: str = Field(..., min_length=1, max_length=80)
    retryable: bool = False


class ErrorResponse(APIModel):
    error: ErrorDetail


class PageMeta(APIModel):
    next_cursor: str | None = None
    has_more: bool = False
    total_estimate: int | None = Field(None, ge=0)


class OperationAccepted(APIModel):
    api_version: Literal["v1"] = "v1"
    operation_id: str
    status: OperationStatus = OperationStatus.QUEUED
    matter_id: str | None = None
    run_id: str | None = None
    status_url: str
    websocket_url: str | None = None
    estimated_seconds: int | None = Field(None, ge=0)


class OperationStatusResponse(OperationAccepted):
    status: OperationStatus
    progress_fraction: float | None = Field(None, ge=0.0, le=1.0)
    result: dict[str, Any] | None = None
    error: ErrorDetail | None = None
    created_at: datetime
    updated_at: datetime


class SourceSpan(APIModel):
    id: str
    document_id: str
    page_start: int | None = Field(None, ge=1)
    page_end: int | None = Field(None, ge=1)
    line_start: int | None = Field(None, ge=1)
    line_end: int | None = Field(None, ge=1)
    char_start: int | None = Field(None, ge=0)
    char_end: int | None = Field(None, ge=0)
    section_ref: str | None = Field(None, max_length=200)
    clause_ref: str | None = Field(None, max_length=200)
    span_type: str = Field(..., min_length=1, max_length=80)
    text: str | None = Field(None, max_length=5000)
    text_hash: str | None = Field(None, max_length=128)
    policy_withheld: bool = False


class AssertionSummary(APIModel):
    id: str
    proposition_text: str
    belief_state: BeliefState
    confidence: float = Field(..., ge=0.0, le=1.0)
    verification_status: VerificationStatus = VerificationStatus.CANDIDATE
    document_ids: list[str] = Field(default_factory=list, max_length=100)
    issue_ids: list[str] = Field(default_factory=list, max_length=100)
    source_spans: list[SourceSpan] = Field(default_factory=list, max_length=20)
    updated_at: datetime


class VerificationStateModel(APIModel):
    id: str
    matter_id: str
    target_kind: VerificationTargetKind
    target_id: str
    status: VerificationStatus
    reviewer_id: str | None = None
    review_note: str | None = Field(None, max_length=2000)
    rejection_reason: str | None = Field(None, max_length=500)
    scope: str = Field("target_only", max_length=80)
    version: int = Field(..., ge=1)
    created_at: datetime
    updated_at: datetime


class ReviewQueueItem(APIModel):
    queue_id: str
    target_kind: VerificationTargetKind
    target_id: str
    priority: Literal["low", "medium", "high", "critical"]
    confidence: float | None = Field(None, ge=0.0, le=1.0)
    reason: str = Field(..., max_length=500)
    issue_ids: list[str] = Field(default_factory=list, max_length=50)
    document_ids: list[str] = Field(default_factory=list, max_length=50)
    assertion: AssertionSummary | None = None
    verification: VerificationStateModel
    lineage: list[dict[str, Any]] = Field(default_factory=list, max_length=50)
    created_at: datetime


class ReviewQueuePage(APIModel):
    api_version: Literal["v1"] = "v1"
    matter_id: str
    items: list[ReviewQueueItem]
    page: PageMeta


class VerifyAssertionRequest(APIModel):
    note: str | None = Field(None, max_length=2000)
    scope: Literal["target_only", "same_claim", "same_document_span"] = "target_only"
    expected_version: int | None = Field(None, ge=1)
    run_id: str | None = None


class RejectAssertionRequest(APIModel):
    reason: Literal[
        "unsupported",
        "incorrect_extraction",
        "privileged",
        "duplicative",
        "irrelevant",
        "superseded",
        "other",
    ]
    note: str = Field(..., min_length=1, max_length=2000)
    mark_dependents_stale: bool = True
    expected_version: int | None = Field(None, ge=1)
    run_id: str | None = None


class VerificationTransitionResponse(APIModel):
    api_version: Literal["v1"] = "v1"
    matter_id: str
    target_kind: VerificationTargetKind
    target_id: str
    old_status: VerificationStatus
    new_status: VerificationStatus
    verification: VerificationStateModel
    dependent_updates: list[dict[str, Any]] = Field(default_factory=list, max_length=500)
    audit_event_id: str


class BatchVerifyAssertionsRequest(APIModel):
    assertion_ids: list[str] | None = Field(None, min_length=1, max_length=1000)
    document_id: str | None = None
    issue_id: str | None = None
    confidence_min: float = Field(0.85, ge=0.0, le=1.0)
    require_source_spans: bool = True
    max_items: int = Field(100, ge=1, le=1000)
    dry_run: bool = False
    note: str | None = Field(None, max_length=2000)
    run_async: bool = False


class BatchVerificationResponse(APIModel):
    api_version: Literal["v1"] = "v1"
    matter_id: str
    dry_run: bool
    matched_count: int = Field(..., ge=0)
    verified_count: int = Field(..., ge=0)
    skipped: list[dict[str, Any]] = Field(default_factory=list, max_length=1000)
    verification_ids: list[str] = Field(default_factory=list, max_length=1000)
    audit_event_ids: list[str] = Field(default_factory=list, max_length=1000)


class VerificationStatsResponse(APIModel):
    api_version: Literal["v1"] = "v1"
    matter_id: str
    by_status: dict[VerificationStatus, int]
    by_target_kind: dict[VerificationTargetKind, dict[VerificationStatus, int]]
    stale_count: int = Field(..., ge=0)
    rejected_count: int = Field(..., ge=0)
    review_queue_count: int = Field(..., ge=0)
    updated_at: datetime


class QueryRequest(APIModel):
    query: str = Field(..., min_length=1, max_length=12000)
    query_mode: QueryMode = QueryMode.AUTO
    research_mode: Literal["simple", "deep", "sebih_special"] = "deep"
    output_policy_id: str | None = None
    policy_audience: PolicyAudience = PolicyAudience.INTERNAL_CLEAN
    conversation_history: list[dict[str, str]] = Field(default_factory=list, max_length=50)
    issue_ids: list[str] = Field(default_factory=list, max_length=50)
    require_verified_support: bool = False
    stream: bool = False
    wait_timeout_ms: int = Field(0, ge=0, le=30000)
    callback_url: HttpUrl | None = None


class CitationModel(APIModel):
    id: str
    document_id: str | None = None
    document: str
    page: int | None = Field(None, ge=1)
    text: str = Field(..., max_length=4000)
    context: str | None = Field(None, max_length=4000)
    relevance: str | None = Field(None, max_length=500)
    verified: bool | None = None
    verification_note: str | None = Field(None, max_length=1000)


class QueryResponse(APIModel):
    api_version: Literal["v1"] = "v1"
    matter_id: str
    run_id: str
    query_mode_used: QueryMode
    hot_path: bool
    answer: str = Field(..., max_length=200000)
    citations: list[CitationModel] = Field(default_factory=list, max_length=500)
    open_gaps: list[dict[str, Any]] = Field(default_factory=list, max_length=100)
    verification_warnings: list[str] = Field(default_factory=list, max_length=50)
    policy_manifest: dict[str, Any] = Field(default_factory=dict)
    freshness: dict[str, Any] = Field(default_factory=dict)
    llm_usage: dict[str, Any] = Field(default_factory=dict)


class FreshnessNamespace(APIModel):
    namespace: str
    revision: int = Field(..., ge=0)
    state: Literal["fresh", "stale", "missing", "recomputing"]
    last_updated_at: datetime | None = None
    stale_reason: str | None = Field(None, max_length=500)


class MatterFreshnessResponse(APIModel):
    api_version: Literal["v1"] = "v1"
    matter_id: str
    repository_revision: str | None = None
    matter_revision: int = Field(..., ge=0)
    last_update_at: datetime | None = None
    is_hot_answerable: bool
    stale_namespaces: list[str] = Field(default_factory=list)
    namespaces: list[FreshnessNamespace] = Field(default_factory=list)


class CancelQueryRequest(APIModel):
    run_id: str
    reason: str | None = Field(None, max_length=500)


class RunCommandResponse(APIModel):
    api_version: Literal["v1"] = "v1"
    matter_id: str
    run_id: str
    status: RunStatus
    command: str
    accepted: bool
    ledger_event_id: str | None = None


class RedirectSteeringRequest(APIModel):
    issue_id: str
    run_id: str | None = None
    reason: str | None = Field(None, max_length=1000)


class TrustSteeringRequest(APIModel):
    source_kind: Literal["document", "document_pattern", "actor", "authority", "repository"]
    source_id: str | None = None
    document_pattern: str | None = Field(None, max_length=500)
    trust_level: TrustLevel
    note: str | None = Field(None, max_length=2000)
    run_id: str | None = None
    mark_dependents_stale: bool = True


class TrustSteeringResponse(APIModel):
    api_version: Literal["v1"] = "v1"
    matter_id: str
    override_id: str
    affected_assertion_count: int = Field(..., ge=0)
    recomputed_issue_count: int = Field(..., ge=0)
    ledger_event_id: str | None = None


class CorrectionSteeringRequest(APIModel):
    assertion_id: str | None = None
    corrected_statement: str | None = Field(None, max_length=12000)
    new_belief_state: BeliefState
    confidence: float = Field(0.8, ge=0.0, le=1.0)
    note: str = Field(..., min_length=1, max_length=2000)
    run_id: str | None = None


class CorrectionResponse(APIModel):
    api_version: Literal["v1"] = "v1"
    matter_id: str
    assertion_id: str
    old_belief_state: BeliefState
    new_belief_state: BeliefState
    propagated_to: list[str] = Field(default_factory=list, max_length=5000)
    propagation_truncated: bool = False
    proof_recompute_status: Literal["not_needed", "completed", "queued", "failed"]
    ledger_event_id: str | None = None


class StrategicContextRequest(APIModel):
    context_type: Literal["strategy", "decision_context", "scope", "client_goal", "court_posture"]
    text: str = Field(..., min_length=1, max_length=12000)
    issue_ids: list[str] = Field(default_factory=list, max_length=50)
    expires_at: datetime | None = None
    run_id: str | None = None


class StrategicContextResponse(APIModel):
    api_version: Literal["v1"] = "v1"
    matter_id: str
    context_id: str
    ledger_event_id: str | None = None


class LedgerEventModel(APIModel):
    id: str
    run_id: str
    seq_no: int = Field(..., ge=0)
    event_type: str
    summary: str
    why: str | None = None
    branch_issue_id: str | None = None
    changed_object_type: str | None = None
    changed_object_id: str | None = None
    snapshot: dict[str, Any] | None = None
    created_at: datetime


class LedgerPage(APIModel):
    api_version: Literal["v1"] = "v1"
    matter_id: str
    items: list[LedgerEventModel]
    page: PageMeta


class PauseRunRequest(APIModel):
    run_id: str
    reason: str | None = Field(None, max_length=500)


class ResumeInvestigationRequest(APIModel):
    run_id: str
    follow_up_query: str | None = Field(None, max_length=12000)
    research_mode: Literal["simple", "deep", "sebih_special"] | None = None


class ResumeRunResponse(APIModel):
    api_version: Literal["v1"] = "v1"
    matter_id: str
    original_run_id: str
    new_run_id: str
    status: RunStatus


class PrivilegeClassificationModel(APIModel):
    id: str
    document_id: str
    span_id: str | None = None
    classification: PrivilegeClass
    confidence: float = Field(..., ge=0.0, le=1.0)
    basis: str | None = Field(None, max_length=2000)
    reviewer_id: str | None = None
    verification_status: VerificationStatus = VerificationStatus.CANDIDATE
    taint_state: Literal["clean", "tainted", "withheld", "unknown"]
    updated_at: datetime


class PrivilegeClassificationPage(APIModel):
    api_version: Literal["v1"] = "v1"
    matter_id: str
    items: list[PrivilegeClassificationModel]
    page: PageMeta


class PrivilegeOverrideRequest(APIModel):
    classification: PrivilegeClass
    span_ids: list[str] = Field(default_factory=list, max_length=1000)
    basis: str = Field(..., min_length=1, max_length=2000)
    verification_status: VerificationStatus = VerificationStatus.VERIFIED
    apply_to_derived_assertions: bool = True
    quarantine_prior_outputs: bool = True
    expected_version: int | None = Field(None, ge=1)


class PrivilegeClassificationResponse(APIModel):
    api_version: Literal["v1"] = "v1"
    matter_id: str
    classification: PrivilegeClassificationModel
    tainted_assertion_count: int = Field(..., ge=0)
    invalidated_cache_count: int = Field(..., ge=0)
    audit_event_id: str


class TaintNode(APIModel):
    target_kind: str
    target_id: str
    privilege_classification: PrivilegeClass | None = None
    taint_state: str


class TaintEdge(APIModel):
    source_kind: str
    source_id: str
    target_kind: str
    target_id: str
    relation: str


class PrivilegeTaintGraphResponse(APIModel):
    api_version: Literal["v1"] = "v1"
    matter_id: str
    root_target_kind: str | None = None
    root_target_id: str | None = None
    nodes: list[TaintNode] = Field(default_factory=list, max_length=5000)
    edges: list[TaintEdge] = Field(default_factory=list, max_length=10000)
    truncated: bool = False


class OutputPolicyRequest(APIModel):
    audience: PolicyAudience
    name: str | None = Field(None, max_length=160)
    default_policy: bool = False
    allow_privileged: bool = False
    allow_confidential: bool = False
    allow_candidate_facts: bool = False
    allow_stale_facts: bool = False
    require_verified_support: bool = True
    redaction_style: Literal["withhold", "placeholder", "summary_only"] = "withhold"


class OutputPolicyResponse(APIModel):
    api_version: Literal["v1"] = "v1"
    matter_id: str
    policy_id: str
    audience: PolicyAudience
    created_at: datetime
    updated_at: datetime


class PrivilegeLogEntry(APIModel):
    id: str
    document_id: str
    document_title: str | None = None
    date: date | None = None
    author: str | None = None
    recipients: list[str] = Field(default_factory=list, max_length=100)
    privilege_claim: PrivilegeClass
    description: str = Field(..., max_length=2000)
    withheld_basis: str | None = Field(None, max_length=2000)
    verification_status: VerificationStatus


class PrivilegeLogPage(APIModel):
    api_version: Literal["v1"] = "v1"
    matter_id: str
    policy_id: str | None = None
    items: list[PrivilegeLogEntry]
    page: PageMeta


class IssueNode(APIModel):
    id: str
    parent_issue_id: str | None = None
    title: str
    issue_type: IssueKind
    burden_side: str | None = Field(None, max_length=80)
    materiality: float = Field(..., ge=0.0, le=1.0)
    salience: float = Field(..., ge=0.0, le=1.0)
    status: Literal["open", "closed", "removed"] = "open"
    sort_order: int = 0
    predicates: list[dict[str, Any]] = Field(default_factory=list)
    coverage: dict[str, Any] | None = None
    children: list["IssueNode"] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class IssueTreeResponse(APIModel):
    api_version: Literal["v1"] = "v1"
    matter_id: str
    roots: list[IssueNode]


class CreateIssueRequest(APIModel):
    title: str = Field(..., min_length=1, max_length=300)
    issue_type: IssueKind
    parent_issue_id: str | None = None
    burden_side: str | None = Field(None, max_length=80)
    materiality: float = Field(0.5, ge=0.0, le=1.0)
    salience: float = Field(0.5, ge=0.0, le=1.0)
    sort_order: int = 0
    predicates: list[str] = Field(default_factory=list, max_length=100)


class UpdateIssueRequest(APIModel):
    title: str | None = Field(None, min_length=1, max_length=300)
    parent_issue_id: str | None = None
    issue_type: IssueKind | None = None
    burden_side: str | None = Field(None, max_length=80)
    materiality: float | None = Field(None, ge=0.0, le=1.0)
    salience: float | None = Field(None, ge=0.0, le=1.0)
    status: Literal["open", "closed"] | None = None
    sort_order: int | None = None
    predicates_add: list[str] = Field(default_factory=list, max_length=100)
    predicates_remove: list[str] = Field(default_factory=list, max_length=100)
    expected_version: int | None = Field(None, ge=1)


class IssueResponse(APIModel):
    api_version: Literal["v1"] = "v1"
    matter_id: str
    issue: IssueNode
    audit_event_id: str | None = None


class DeleteIssueResponse(APIModel):
    api_version: Literal["v1"] = "v1"
    matter_id: str
    issue_id: str
    deleted: bool
    mode: Literal["soft", "hard"]
    affected_children: int = Field(0, ge=0)
    affected_links: int = Field(0, ge=0)


class IssueCoverageResponse(APIModel):
    api_version: Literal["v1"] = "v1"
    matter_id: str
    issue_id: str
    coverage_fraction: float = Field(..., ge=0.0, le=1.0)
    supporting_count: int = Field(..., ge=0)
    attacking_count: int = Field(..., ge=0)
    predicate_total: int = Field(..., ge=0)
    predicate_resolved: int = Field(..., ge=0)
    verified_sufficiency: float = Field(..., ge=0.0, le=1.0)
    candidate_sufficiency: float = Field(..., ge=0.0, le=1.0)
    has_proof_gap: bool
    gap_ids: list[str] = Field(default_factory=list)
    assertions: list[AssertionSummary] = Field(default_factory=list)


class LinkAssertionToIssueRequest(APIModel):
    assertion_id: str
    predicate_id: str | None = None
    relation_type: EvidenceRelation = EvidenceRelation.SUPPORTS
    proof_weight: float = Field(0.5, ge=0.0, le=1.0)
    source_span_id: str | None = None
    source_occurrence_id: str | None = None
    note: str | None = Field(None, max_length=1000)


class IssueLinkResponse(APIModel):
    api_version: Literal["v1"] = "v1"
    matter_id: str
    link_id: str
    evidence_edge_id: str
    issue_id: str
    assertion_id: str
    predicate_id: str | None = None


class DocumentCard(APIModel):
    id: str
    document_id: str
    relative_path: str
    title: str | None = None
    doc_type: str | None = None
    doc_subtype: str | None = None
    source_side: str | None = None
    source_role: str | None = None
    author: str | None = None
    sender: str | None = None
    recipient: str | None = None
    creation_date: date | None = None
    sent_date: date | None = None
    effective_date: date | None = None
    purpose: str | None = None
    rhetorical_posture: str | None = None
    reliability_posture: str | None = None
    operative_status: str = "unknown"
    privilege: PrivilegeClassificationModel | None = None
    unresolved_flags: list[str] = Field(default_factory=list)
    salience_score: float = Field(0.5, ge=0.0, le=1.0)
    updated_at: datetime


class DocumentCardResponse(APIModel):
    api_version: Literal["v1"] = "v1"
    matter_id: str
    card: DocumentCard
    assertions: list[AssertionSummary] = Field(default_factory=list)
    spans: list[SourceSpan] = Field(default_factory=list)
    policy_manifest: dict[str, Any] = Field(default_factory=dict)


class SourceSpanPage(APIModel):
    api_version: Literal["v1"] = "v1"
    matter_id: str
    document_id: str
    items: list[SourceSpan]
    page: PageMeta


class MissingDocumentItem(APIModel):
    gap_id: str
    expected_artifact: str | None = None
    description: str
    materiality_score: float = Field(..., ge=0.0, le=1.0)
    blocker_score: float = Field(..., ge=0.0, le=1.0)
    affected_issues: list[str] = Field(default_factory=list)
    affected_assertions: list[str] = Field(default_factory=list)
    suggested_request: str | None = Field(None, max_length=1000)


class MissingDocumentsPage(APIModel):
    api_version: Literal["v1"] = "v1"
    matter_id: str
    items: list[MissingDocumentItem]
    page: PageMeta


class DocumentAnnotationCreateRequest(APIModel):
    annotation_text: str = Field(..., min_length=1, max_length=12000)
    annotation_type: Literal["strategic", "reliability", "scope", "privilege", "fact_note"] = "strategic"
    span_id: str | None = None
    issue_ids: list[str] = Field(default_factory=list, max_length=50)
    visibility: Literal["internal", "clean"] = "internal"


class DocumentAnnotationResponse(APIModel):
    api_version: Literal["v1"] = "v1"
    matter_id: str
    document_id: str
    annotation_id: str
    created_at: datetime


class TimelineEvent(APIModel):
    id: str
    event_date: date | None = None
    date_precision: Literal["day", "month", "year", "range", "unknown"] = "unknown"
    event_type: str
    description: str
    issue_ids: list[str] = Field(default_factory=list)
    actor_ids: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    source_spans: list[SourceSpan] = Field(default_factory=list, max_length=20)
    verification_status: VerificationStatus = VerificationStatus.CANDIDATE
    policy_withheld: bool = False


class TimelinePage(APIModel):
    api_version: Literal["v1"] = "v1"
    matter_id: str
    items: list[TimelineEvent]
    page: PageMeta


class DamagesComponent(APIModel):
    id: str
    label: str
    amount: float | None = None
    currency: str = Field("USD", min_length=3, max_length=3)
    formula: str | None = Field(None, max_length=1000)
    assumptions: list[str] = Field(default_factory=list)
    source_quant_fact_ids: list[str] = Field(default_factory=list)
    verification_status: VerificationStatus = VerificationStatus.CANDIDATE
    disputed: bool = False


class DamagesModelResponse(APIModel):
    api_version: Literal["v1"] = "v1"
    matter_id: str
    currency: str
    components: list[DamagesComponent]
    total_claimed: float | None = None
    total_verified: float | None = None
    total_disputed: float | None = None
    conflicts: list[dict[str, Any]] = Field(default_factory=list)


class ReconciliationResponse(APIModel):
    api_version: Literal["v1"] = "v1"
    matter_id: str
    currency: str
    invoiced_total: float = 0.0
    paid_total: float = 0.0
    disputed_total: float = 0.0
    claimed_exposure: float = 0.0
    rows: list[dict[str, Any]] = Field(default_factory=list)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)


class QuantConflictPage(APIModel):
    api_version: Literal["v1"] = "v1"
    matter_id: str
    items: list[dict[str, Any]]
    page: PageMeta


class MatterHealthResponse(APIModel):
    api_version: Literal["v1"] = "v1"
    matter_id: str
    coverage: dict[str, Any]
    gaps: dict[str, Any]
    verification: VerificationStatsResponse
    privilege: dict[str, Any]
    freshness: MatterFreshnessResponse
    active_run_count: int = Field(..., ge=0)
    warnings: list[str] = Field(default_factory=list)


class ActivityEvent(APIModel):
    id: str
    activity_type: str
    actor_id: str | None = None
    run_id: str | None = None
    target_kind: str | None = None
    target_id: str | None = None
    summary: str
    created_at: datetime


class ActivityPage(APIModel):
    api_version: Literal["v1"] = "v1"
    matter_id: str
    items: list[ActivityEvent]
    page: PageMeta


class RunSummary(APIModel):
    id: str
    matter_id: str
    query: str
    status: RunStatus
    query_mode: QueryMode | None = None
    operation_type: Literal["query", "revise", "maintenance"] = "query"
    active_branch_issue_id: str | None = None
    stop_requested: bool = False
    redirect_requested: bool = False
    started_at: datetime
    completed_at: datetime | None = None
    resumed_from: str | None = None
    reuse_rate: float | None = Field(None, ge=0.0, le=1.0)
    llm_usage: dict[str, Any] = Field(default_factory=dict)


class RunPage(APIModel):
    api_version: Literal["v1"] = "v1"
    matter_id: str
    items: list[RunSummary]
    page: PageMeta


class GapModel(APIModel):
    id: str
    gap_type: GapType
    description: str
    expected_artifact: str | None = None
    materiality_score: float = Field(..., ge=0.0, le=1.0)
    blocker_score: float = Field(..., ge=0.0, le=1.0)
    status: GapStatus
    dependencies: list[dict[str, str]] = Field(default_factory=list)
    resolution_note: str | None = None
    created_at: datetime
    updated_at: datetime


class GapPage(APIModel):
    api_version: Literal["v1"] = "v1"
    matter_id: str
    items: list[GapModel]
    page: PageMeta


class ResolveGapRequest(APIModel):
    resolution_note: str = Field(..., min_length=1, max_length=2000)
    evidence_target_kind: str | None = Field(None, max_length=80)
    evidence_target_id: str | None = None
    mark_dependents_fresh: bool = True


class DismissGapRequest(APIModel):
    reason: Literal["not_material", "duplicate", "incorrect", "outside_scope", "other"]
    note: str = Field(..., min_length=1, max_length=2000)


class GapActionResponse(APIModel):
    api_version: Literal["v1"] = "v1"
    matter_id: str
    gap: GapModel
    audit_event_id: str


class MaintenanceTriggerRequest(APIModel):
    namespaces: list[
        Literal[
            "freshness",
            "verification",
            "proof_state",
            "privilege_taint",
            "hot_cache",
            "quant",
            "gaps",
            "all",
        ]
    ] = Field(default_factory=lambda: ["all"], max_length=20)
    reason: str | None = Field(None, max_length=500)
    force: bool = False
    budget_seconds: int = Field(300, ge=1, le=3600)


class MaintenanceTaskSummary(APIModel):
    id: str
    namespace: str
    status: OperationStatus
    priority: int = Field(..., ge=0, le=100)
    attempts: int = Field(..., ge=0)
    last_error: str | None = None
    leased_until: datetime | None = None
    updated_at: datetime


class MaintenanceStatusResponse(APIModel):
    api_version: Literal["v1"] = "v1"
    matter_id: str
    queue_depth: int = Field(..., ge=0)
    running_count: int = Field(..., ge=0)
    failed_count: int = Field(..., ge=0)
    tasks: list[MaintenanceTaskSummary]
```

## Endpoint Matrix

Common errors for all endpoints: `401 unauthenticated`, `403 forbidden`,
`404 matter_not_found`, `422 validation_error`, `429 rate_limited`,
`500 internal_error`. Endpoint-specific errors are additive.

### Verification Endpoints

| Method and path | Query params | Body | Response | Auth | Rate | Mode | Reads | Writes | Realtime | Specific errors |
|---|---|---|---|---|---|---|---|---|---|---|
| `POST /api/v1/matters/{matter_id}/assertions/{assertion_id}/verify` | None | `VerifyAssertionRequest` | `VerificationTransitionResponse` | `matter:attorney_verify` | `mutation` | Sync | `matter`, `assertion`, `assertion_occurrence`, `verification_state`, `review_task` | `verification_state`, `verification_event`, `review_task`, `assertion_revision`, `ledger_event`, `proof_state`, `activity_event` | `verification.updated`, `review_queue.updated` | `404 assertion_not_found`, `409 invalid_transition`, `409 stale_version`, `409 privileged_policy_block` |
| `GET /api/v1/matters/{matter_id}/review-queue` | `cursor`, `limit=50`, `target_kind`, `status=candidate`, `confidence_min`, `confidence_max`, `issue_id`, `document_id`, `priority`, `assigned_to`, `sort=priority|confidence|created_at`, `include_lineage=false` | None | `ReviewQueuePage` | `matter:review` | `read` | Sync | `review_task`, `verification_state`, `assertion`, `assertion_occurrence`, `issue`, `assertion_issue_link`, `document_inventory`, `quant_fact`, `authority`, `privilege_classification` | None | `review_queue.updated` | `400 invalid_filter`, `403 privilege_scope_required` |
| `POST /api/v1/matters/{matter_id}/assertions/{assertion_id}/reject` | None | `RejectAssertionRequest` | `VerificationTransitionResponse` | `matter:review` | `mutation` | Sync, with background stale propagation if large | `assertion`, `verification_state`, `assertion_link`, `assertion_issue_link`, `evidence_edge`, `proof_state` | `verification_state`, `verification_event`, `review_task`, `assertion_revision`, `pending_propagation`, `proof_state`, `ledger_event`, `activity_event` | `verification.updated`, `proof_state.updated` | `404 assertion_not_found`, `409 invalid_transition`, `409 stale_version` |
| `POST /api/v1/matters/{matter_id}/assertions/batch-verify` | None | `BatchVerifyAssertionsRequest` | `BatchVerificationResponse` or `202 OperationAccepted` | `matter:attorney_verify` | `verification_batch` | Sync to 500 items; async when requested or estimated over 2s | `assertion`, `assertion_occurrence`, `verification_state`, `review_task`, `issue`, `document_inventory` | `verification_state`, `verification_event`, `review_task`, `ledger_event`, `activity_event`, optional `operation_job` | `verification.batch_progress`, `review_queue.updated` | `400 empty_selector`, `409 batch_too_large_without_async`, `409 privileged_policy_block` |
| `GET /api/v1/matters/{matter_id}/verification-stats` | `target_kind`, `issue_id`, `document_id`, `include_stale=true` | None | `VerificationStatsResponse` | `matter:read` | `dashboard` | Sync | `verification_state`, `review_task`, `assertion_issue_link`, `assertion_occurrence`, `document_inventory` | None | `verification.updated` | `400 invalid_filter` |

### Query Mode Endpoints

| Method and path | Query params | Body | Response | Auth | Rate | Mode | Reads | Writes | Realtime | Specific errors |
|---|---|---|---|---|---|---|---|---|---|---|
| `POST /api/v1/matters/{matter_id}/query` | None | `QueryRequest` | `QueryResponse` or `202 OperationAccepted` | `matter:query`; `matter:privilege` if privileged policy requested | `hot_query` or `cold_query` by decision | Hot sync when fresh; cold async by default; `wait_timeout_ms` may hold to 30s | `matter`, `matter_revision`, `namespace_revision`, `run_session`, `document_inventory`, `document_card`, `assertion`, `verification_state`, `issue`, `proof_state`, `gap`, `quant_fact`, `reasoning_cache`, `hot_context_cache`, `answer_cache`, `output_policy`, `privilege_taint_edge` | `run_session`, `ledger_event`, `llm_call`, `reasoning_cache`, `answer_cache`, `operation_job`; cold path writes derived stores | `run.progress`, `ledger.event`, `query.completed` | `400 cold_required_for_hot_mode`, `409 active_cold_query_exists`, `409 policy_blocked`, `503 llm_provider_unavailable` |
| `GET /api/v1/matters/{matter_id}/freshness` | `namespace` repeated, `include_namespaces=true`, `include_manifest_diff=false` | None | `MatterFreshnessResponse` | `matter:read` | `dashboard` | Sync | `matter`, `matter_revision`, `namespace_revision`, `repo_manifest_entry`, `document_inventory`, `proof_state`, `verification_state`, `maintenance_task` | None | `freshness.updated` | `400 invalid_namespace` |
| `POST /api/v1/matters/{matter_id}/query/cancel` | None | `CancelQueryRequest` | `RunCommandResponse` | `matter:steer` | `mutation` | Sync command; cancellation applies at checkpoint/provider boundary | `run_session`, `operation_job` | `run_session`, `ledger_event`, `operation_job`, `activity_event` | `run.canceled`, `ledger.event` | `404 run_not_found`, `409 run_not_cancelable`, `409 utility_run_not_cancelable` |

### Steering Endpoints

| Method and path | Query params | Body | Response | Auth | Rate | Mode | Reads | Writes | Realtime | Specific errors |
|---|---|---|---|---|---|---|---|---|---|---|
| `POST /api/v1/matters/{matter_id}/steering/redirect` | None | `RedirectSteeringRequest` | `RunCommandResponse` | `matter:steer` | `mutation` | Sync command; engine applies next iteration | `run_session`, `issue` | `run_session`, `steering_command`, `ledger_event`, `activity_event` | `steering.applied`, `run.redirect_requested` | `404 run_not_found`, `404 issue_not_found`, `409 run_not_active` |
| `POST /api/v1/matters/{matter_id}/steering/trust` | None | `TrustSteeringRequest` | `TrustSteeringResponse` | `matter:steer` | `mutation` | Sync persistence; propagation may queue maintenance | `document_inventory`, `actor`, `authority`, `assertion_occurrence`, `assertion_issue_link`, `proof_state` | `source_trust_override` or current `document_trust_override`, `belief_revision_event`, `pending_propagation`, `proof_state`, `ledger_event`, `activity_event` | `trust.updated`, `proof_state.updated` | `400 invalid_source_target`, `404 source_not_found`, `409 privilege_override_not_allowed` |
| `POST /api/v1/matters/{matter_id}/steering/correct` | None | `CorrectionSteeringRequest` | `CorrectionResponse` | `matter:steer` and `matter:review` | `mutation` | Sync bounded propagation; background flush if truncated | `assertion`, `assertion_link`, `assertion_issue_link`, `verification_state`, `proof_state` | `assertion`, `belief_revision_event`, `assertion_revision`, `pending_propagation`, `verification_state`, `proof_state`, `ledger_event`, `activity_event` | `assertion.corrected`, `belief_revision.progress`, `proof_state.updated` | `404 assertion_not_found`, `409 verified_assertion_contradicted`, `409 propagation_in_progress` |
| `POST /api/v1/matters/{matter_id}/steering/context` | None | `StrategicContextRequest` | `StrategicContextResponse` | `matter:steer` | `mutation` | Sync | `issue`, `decision_context` | `decision_context`, `document_annotation` when scoped to docs, `ledger_event`, `activity_event` | `context.updated` | `400 invalid_context_type`, `404 issue_not_found` |
| `GET /api/v1/matters/{matter_id}/reasoning-ledger` | `run_id`, `cursor`, `limit=100`, `event_type`, `changed_object_type`, `changed_object_id`, `include_snapshots=false`, `sort=seq|created_at` | None | `LedgerPage` | `matter:read` | `read` | Sync | `run_session`, `ledger_event` | None | `ledger.event` | `404 run_not_found`, `400 invalid_cursor` |
| `POST /api/v1/matters/{matter_id}/steering/pause` | None | `PauseRunRequest` | `RunCommandResponse` | `matter:steer` | `mutation` | Sync command; engine pauses at checkpoint | `run_session` | `run_session`, `ledger_event`, `activity_event` | `run.paused` | `404 run_not_found`, `409 run_not_running`, `409 utility_run_not_pauseable` |
| `POST /api/v1/matters/{matter_id}/steering/resume` | None | `ResumeInvestigationRequest` | `ResumeRunResponse` or `202 OperationAccepted` | `matter:steer`, `matter:query` | `cold_query` | Async by default | `run_session`, checkpoint metadata, `operation_job` | `run_session`, `ledger_event`, `operation_job`, `activity_event` | `run.resumed`, `run.progress` | `404 run_not_found`, `409 run_not_paused_or_interrupted`, `400 checkpoint_missing`, `409 active_run_exists` |

### Privilege Endpoints

| Method and path | Query params | Body | Response | Auth | Rate | Mode | Reads | Writes | Realtime | Specific errors |
|---|---|---|---|---|---|---|---|---|---|---|
| `GET /api/v1/matters/{matter_id}/privilege/classifications` | `cursor`, `limit=50`, `document_id`, `classification`, `verification_status`, `taint_state`, `include_clean=false`, `sort=updated_at|classification|confidence` | None | `PrivilegeClassificationPage` | `matter:privilege` | `read` | Sync | `privilege_classification`, `document_inventory`, `span`, `verification_state`, `document_card` | None | `privilege.updated` | `400 invalid_filter` |
| `POST /api/v1/matters/{matter_id}/documents/{document_id}/privilege` | None | `PrivilegeOverrideRequest` | `PrivilegeClassificationResponse` | `matter:privilege`; `matter:attorney_verify` when verified | `mutation` | Sync doc-level; span-level propagation may queue maintenance | `document_inventory`, `span`, `assertion_occurrence`, `assertion`, `evidence_edge`, `verification_state`, `output_policy` | `privilege_classification`, `verification_state`, `verification_event`, `privilege_taint_edge`, `policy_decision_log`, `pending_propagation`, `namespace_revision`, `ledger_event`, `activity_event` | `privilege.updated`, `taint.updated` | `404 document_not_found`, `409 stale_version`, `409 cannot_clear_taint_without_basis` |
| `GET /api/v1/matters/{matter_id}/privilege/taint` | `target_kind`, `target_id`, `depth=3`, `include_clean=false`, `limit=5000`, `policy_id` | None | `PrivilegeTaintGraphResponse` | `matter:privilege` | `read` | Sync, truncated at limit | `privilege_taint_edge`, `privilege_classification`, `assertion`, `evidence_edge`, `quant_fact`, `authority`, `reasoning_cache`, `output_policy` | None | `taint.updated` | `400 target_required_for_deep_graph`, `400 depth_too_large` |
| `POST /api/v1/matters/{matter_id}/privilege/output-policy` | None | `OutputPolicyRequest` | `OutputPolicyResponse` | `matter:privilege` | `mutation` | Sync | `output_policy` | `output_policy`, `policy_decision_log`, `ledger_event`, `activity_event` | `policy.updated` | `409 invalid_policy_for_audience`, `409 cannot_default_external_ai_privileged` |
| `GET /api/v1/matters/{matter_id}/privilege/log` | `cursor`, `limit=100`, `policy_id`, `classification`, `include_non_privileged=false`, `format=json`, `sort=date|document` | None | `PrivilegeLogPage` for JSON; file stream for CSV | `matter:privilege` | `read` | Sync paginated JSON; stream for CSV | `privilege_classification`, `document_inventory`, `document_card`, `span`, `document_actor_role`, `actor`, `verification_state`, `output_policy` | `policy_decision_log` for export access | `privilege_log.generated` | `400 unsupported_format`, `409 unresolved_unknown_classifications` |

### Issue Model Endpoints

| Method and path | Query params | Body | Response | Auth | Rate | Mode | Reads | Writes | Realtime | Specific errors |
|---|---|---|---|---|---|---|---|---|---|---|
| `GET /api/v1/matters/{matter_id}/issues` | `root_id`, `include_closed=false`, `include_predicates=true`, `include_coverage=true`, `max_depth=8`, `policy_mode=clean|internal` | None | `IssueTreeResponse` | `matter:read` | `read` | Sync | `issue`, `issue_predicate`, `assertion_issue_link`, `evidence_edge`, `proof_state`, `verification_state`, `privilege_taint_edge` | None | `issue.updated`, `proof_state.updated` | `404 root_issue_not_found`, `400 max_depth_too_large` |
| `POST /api/v1/matters/{matter_id}/issues` | None | `CreateIssueRequest` | `IssueResponse` | `matter:steer` | `mutation` | Sync | `issue` | `issue`, `issue_predicate`, `ledger_event`, `activity_event`, `namespace_revision` | `issue.created` | `404 parent_issue_not_found`, `409 duplicate_issue`, `422 invalid_issue_tree` |
| `PUT /api/v1/matters/{matter_id}/issues/{issue_id}` | None | `UpdateIssueRequest` | `IssueResponse` | `matter:steer` | `mutation` | Sync | `issue`, `issue_predicate` | `issue`, `issue_predicate`, `proof_state` stale marker, `ledger_event`, `activity_event`, `namespace_revision` | `issue.updated` | `404 issue_not_found`, `409 stale_version`, `409 cycle_detected` |
| `DELETE /api/v1/matters/{matter_id}/issues/{issue_id}` | `mode=soft|hard` default `soft`, `cascade=false` | None | `DeleteIssueResponse` | `matter:admin` for hard delete; `matter:steer` for soft delete | `mutation` | Sync | `issue`, `issue_predicate`, `assertion_issue_link`, `evidence_edge` | Soft: `issue.status`; hard: `issue`, `issue_predicate`, `assertion_issue_link`, `evidence_edge`, `proof_state`; always `ledger_event`, `activity_event`, `namespace_revision` | `issue.deleted` | `404 issue_not_found`, `409 has_children_without_cascade`, `409 linked_evidence_without_soft_delete` |
| `GET /api/v1/matters/{matter_id}/issues/{issue_id}/coverage` | `include_children=true`, `include_assertions=true`, `policy_mode=clean`, `verification_floor=candidate|verified`, `refresh=false` | None | `IssueCoverageResponse` | `matter:read` | `read` | Sync; `refresh=true` may recompute one issue | `issue`, `issue_predicate`, `assertion_issue_link`, `evidence_edge`, `assertion`, `verification_state`, `proof_state`, `gap`, `privilege_taint_edge` | Optional `proof_state` when refreshed | `proof_state.updated` | `404 issue_not_found`, `409 policy_blocked` |
| `POST /api/v1/matters/{matter_id}/issues/{issue_id}/link` | None | `LinkAssertionToIssueRequest` | `IssueLinkResponse` | `matter:review` | `mutation` | Sync | `issue`, `issue_predicate`, `assertion`, `assertion_occurrence`, `verification_state` | `assertion_issue_link`, `evidence_edge`, `proof_state`, `ledger_event`, `activity_event`, `namespace_revision` | `issue.linked`, `proof_state.updated` | `404 issue_not_found`, `404 assertion_not_found`, `404 predicate_not_found`, `409 rejected_assertion_not_linkable` |

### Document Endpoints

| Method and path | Query params | Body | Response | Auth | Rate | Mode | Reads | Writes | Realtime | Specific errors |
|---|---|---|---|---|---|---|---|---|---|---|
| `GET /api/v1/matters/{matter_id}/documents/{document_id}/card` | `include_spans=false`, `include_assertions=false`, `policy_mode=clean`, `span_limit=50` | None | `DocumentCardResponse` | `matter:read`; `matter:privilege` for internal policy | `read` | Sync | `document_inventory`, `document_card`, `span`, `assertion_occurrence`, `assertion`, `verification_state`, `privilege_classification`, `privilege_taint_edge`, `document_actor_role`, `actor` | `policy_decision_log` when fields hidden | `document.updated`, `privilege.updated` | `404 document_not_found`, `403 privileged_scope_required` |
| `GET /api/v1/matters/{matter_id}/documents/{document_id}/spans` | `cursor`, `limit=100`, `span_type`, `referenced_only=true`, `assertion_id`, `include_text=true`, `policy_mode=clean` | None | `SourceSpanPage` | `matter:read`; `matter:privilege` for internal text | `read` | Sync | `document_inventory`, `span`, `assertion_occurrence`, `assertion`, `privilege_classification`, `privilege_taint_edge` | `policy_decision_log` when redacted | `document.spans_updated` | `404 document_not_found`, `404 assertion_not_found`, `403 privileged_scope_required` |
| `GET /api/v1/matters/{matter_id}/documents/missing` | `cursor`, `limit=50`, `issue_id`, `min_materiality=0.0`, `status=open`, `gap_type=missing_document|expected_absent_attachment` | None | `MissingDocumentsPage` | `matter:read` | `read` | Sync | `gap`, `gap_link`, `issue`, `assertion`, `document_relation` | None | `gap.updated` | `400 invalid_gap_type`, `404 issue_not_found` |
| `POST /api/v1/matters/{matter_id}/documents/{document_id}/annotate` | None | `DocumentAnnotationCreateRequest` | `DocumentAnnotationResponse` | `matter:steer` | `mutation` | Sync | `document_inventory`, `span`, `issue` | `document_annotation`, `ledger_event`, `activity_event`, `namespace_revision` | `document.annotation_created`, `context.updated` | `404 document_not_found`, `404 span_not_found`, `404 issue_not_found` |

### Quantitative Endpoints

| Method and path | Query params | Body | Response | Auth | Rate | Mode | Reads | Writes | Realtime | Specific errors |
|---|---|---|---|---|---|---|---|---|---|---|
| `GET /api/v1/matters/{matter_id}/quant/timeline` | `cursor`, `limit=100`, `date_from`, `date_to`, `issue_id`, `actor_id`, `document_id`, `verification_status`, `policy_mode=clean`, `sort=date_asc|date_desc` | None | `TimelinePage` | `matter:read`; `matter:privilege` for internal policy | `read` | Sync | `quant_fact`, `assertion`, `assertion_occurrence`, `span`, `document_inventory`, `issue`, `assertion_issue_link`, `verification_state`, `privilege_taint_edge` | `policy_decision_log` when redacted | `quant.updated` | `400 invalid_date_range`, `404 issue_not_found`, `403 privileged_scope_required` |
| `GET /api/v1/matters/{matter_id}/quant/damages` | `currency=USD`, `issue_id`, `as_of`, `include_candidates=true`, `policy_mode=clean` | None | `DamagesModelResponse` | `matter:read` | `read` | Sync | `quant_fact`, `assertion`, `assertion_issue_link`, `issue`, `verification_state`, `gap`, `privilege_taint_edge` | None | `quant.updated` | `400 invalid_currency`, `404 issue_not_found` |
| `GET /api/v1/matters/{matter_id}/quant/reconciliation` | `currency=USD`, `subject_type`, `subject_id`, `include_conflicts=true`, `policy_mode=clean` | None | `ReconciliationResponse` | `matter:read` | `read` | Sync | `quant_fact`, `assertion`, `span`, `document_inventory`, `verification_state`, `privilege_taint_edge` | None | `quant.updated` | `400 invalid_currency`, `404 subject_not_found` |
| `GET /api/v1/matters/{matter_id}/quant/conflicts` | `cursor`, `limit=50`, `quant_kind=amount`, `subject_type`, `min_delta`, `verification_status` | None | `QuantConflictPage` | `matter:read` | `read` | Sync | `quant_fact`, `assertion`, `verification_state`, `gap` | None | `quant.conflict_detected` | `400 invalid_quant_kind` |

### Matter Dashboard Endpoints

| Method and path | Query params | Body | Response | Auth | Rate | Mode | Reads | Writes | Realtime | Specific errors |
|---|---|---|---|---|---|---|---|---|---|---|
| `GET /api/v1/matters/{matter_id}/health` | `policy_mode=clean`, `include_warnings=true` | None | `MatterHealthResponse` | `matter:read` | `dashboard` | Sync | `matter`, `issue`, `proof_state`, `gap`, `verification_state`, `review_task`, `privilege_classification`, `namespace_revision`, `run_session`, `llm_call` | None | `matter.health_updated` | `409 health_stale` when caller requires fresh |
| `GET /api/v1/matters/{matter_id}/activity` | `cursor`, `limit=100`, `activity_type`, `actor_id`, `run_id`, `target_kind`, `target_id`, `since` | None | `ActivityPage` | `matter:read` | `dashboard` | Sync | `activity_event`, `ledger_event`, `verification_event`, `assertion_revision`, `policy_decision_log`, `run_session` | None | `activity.created` | `400 invalid_cursor`, `404 run_not_found` |
| `GET /api/v1/matters/{matter_id}/runs` | `cursor`, `limit=50`, `status`, `operation_type`, `active_only=false`, `started_after`, `include_usage=true` | None | `RunPage` | `matter:read` | `dashboard` | Sync | `run_session`, `ledger_event`, `llm_call`, `operation_job` | None | `run.started`, `run.completed`, `run.failed` | `400 invalid_status` |

### Gap Store Endpoints

| Method and path | Query params | Body | Response | Auth | Rate | Mode | Reads | Writes | Realtime | Specific errors |
|---|---|---|---|---|---|---|---|---|---|---|
| `GET /api/v1/matters/{matter_id}/gaps` | `cursor`, `limit=50`, `gap_type`, `status=open`, `min_materiality=0.0`, `affected_type`, `affected_id`, `issue_id`, `sort=materiality|updated_at|blocker` | None | `GapPage` | `matter:read` | `read` | Sync | `gap`, `gap_link`, `issue`, `assertion_issue_link` | None | `gap.updated` | `400 invalid_filter`, `404 issue_not_found` |
| `POST /api/v1/matters/{matter_id}/gaps/{gap_id}/resolve` | None | `ResolveGapRequest` | `GapActionResponse` | `matter:review` | `mutation` | Sync; dependent freshness recompute may queue maintenance | `gap`, `gap_link`, referenced evidence target | `gap`, `verification_event`, `ledger_event`, `activity_event`, `namespace_revision`, optional `maintenance_task` | `gap.resolved`, `freshness.updated` | `404 gap_not_found`, `409 gap_already_closed`, `404 evidence_target_not_found` |
| `POST /api/v1/matters/{matter_id}/gaps/{gap_id}/dismiss` | None | `DismissGapRequest` | `GapActionResponse` | `matter:review` | `mutation` | Sync | `gap`, `gap_link` | `gap`, `ledger_event`, `activity_event`, `namespace_revision` | `gap.dismissed` | `404 gap_not_found`, `409 gap_already_closed` |

### Maintenance Endpoints

| Method and path | Query params | Body | Response | Auth | Rate | Mode | Reads | Writes | Realtime | Specific errors |
|---|---|---|---|---|---|---|---|---|---|---|
| `POST /api/v1/matters/{matter_id}/maintenance/trigger` | None | `MaintenanceTriggerRequest` | `202 OperationAccepted` | `matter:admin` | `maintenance` | Async | `matter`, `namespace_revision`, `maintenance_task`, `run_session` | `maintenance_task`, `operation_job`, `run_session`, `ledger_event`, `activity_event` | `maintenance.started`, `maintenance.progress`, `maintenance.completed` | `409 maintenance_already_running`, `400 invalid_namespace`, `503 worker_unavailable` |
| `GET /api/v1/matters/{matter_id}/maintenance/status` | `namespace`, `include_completed=false`, `limit=100` | None | `MaintenanceStatusResponse` | `matter:admin` | `dashboard` | Sync | `maintenance_task`, `operation_job`, `namespace_revision`, `run_session` | None | `maintenance.progress` | `400 invalid_namespace` |

## Store Requirements

Existing stores/tables used by this contract:

- Matter and run state: `matter`, `run_session`, `ledger_event`, `llm_call`.
- Assertions and truth maintenance: `assertion`, `assertion_occurrence`,
  `assertion_link`, `belief_revision_event`, `assertion_revision`,
  `pending_propagation`.
- Issues and proof: `issue`, `issue_predicate`, `assertion_issue_link`,
  `evidence_edge`, `proof_state`.
- Documents: `document_inventory`, `document_relation`, `document_card`, `span`,
  `document_actor_role`, `document_annotation`.
- Actors and authority: `actor`, `actor_alias`, `actor_affiliation`, `authority`,
  `authority_issue_link`.
- Gaps and quant: `gap`, `gap_link`, `quant_fact`, `clarification_question`.
- Current steering/trust: `document_trust_override`, `decision_context`.

New stores/tables required:

- `verification_state`: unique `(matter_id, target_kind, target_id)` with
  `candidate|verified|rejected|stale`, reviewer metadata, version, timestamps.
- `verification_event`: append-only audit log for verification transitions.
- `review_task`: candidate/stale/rejected queue materialized from proof-critical
  targets. This covers assertions, issue predicates, evidence edges, quant facts,
  authorities, document cards, and privilege classifications.
- `source_trust_override`: generalized trust overrides for documents, actors,
  authorities, patterns, and repositories. Current `document_trust_override` can
  remain as a compatibility specialization.
- `steering_command`: durable requested/applied steering commands linked to
  `run_session` and `ledger_event`.
- `privilege_classification`: document/span classification state. Review status
  still lives in `verification_state` with target kind `privilege_classification`;
  do not create a second privilege review state machine.
- `privilege_taint_edge`: propagation graph from privileged documents/spans into
  assertions, evidence edges, quant facts, caches, and outputs.
- `output_policy`: audience policy profiles.
- `policy_decision_log`: append-only allow/withhold/redact decisions.
- `activity_event`: normalized matter activity feed derived from ledger,
  verification, privilege, issue, gap, and maintenance events.
- `matter_revision`, `namespace_revision`, `repo_manifest_entry`: freshness and
  hot-path invalidation.
- `hot_context_cache`, `citation_verification_cache`, `answer_cache`: explicit
  hot query artifacts.
- `maintenance_task`: background task queue with namespace, priority, lease,
  attempts, retry time, status, and error.
- `operation_job`: generic async operation status for polling.

## WebSocket Contract

Canonical WebSocket:

`WS /api/v1/ws/matters/{matter_id}?topics=run,ledger,verification,privilege,maintenance&run_id=<optional>&last_event_id=<optional>`

Auth: bearer token with the same permissions as subscribed topics. A socket
requesting privileged events requires `matter:privilege`.

Fallback SSE may be preserved for run-only progress:

`GET /api/v1/matters/{matter_id}/runs/{run_id}/events/stream`

```python
class WSEvent(APIModel):
    event_id: str
    event_type: str
    matter_id: str
    run_id: str | None = None
    seq_no: int | None = Field(None, ge=0)
    occurred_at: datetime
    payload: dict[str, Any]


class RunProgressPayload(APIModel):
    status: RunStatus
    phase: str | None = None
    progress_fraction: float | None = Field(None, ge=0.0, le=1.0)
    active_issue_id: str | None = None
    message: str | None = Field(None, max_length=500)
    llm_usage_delta: dict[str, Any] = Field(default_factory=dict)


class SteeringPayload(APIModel):
    command_id: str
    command_type: Literal["redirect", "pause", "resume", "cancel", "trust", "correct", "context"]
    status: Literal["requested", "applied", "rejected", "failed"]
    target: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = Field(None, max_length=1000)


class VerificationPayload(APIModel):
    target_kind: VerificationTargetKind
    target_id: str
    old_status: VerificationStatus | None = None
    new_status: VerificationStatus
    reviewer_id: str | None = None


class PrivilegePayload(APIModel):
    document_id: str | None = None
    classification_id: str | None = None
    classification: PrivilegeClass | None = None
    tainted_count: int = Field(0, ge=0)
    policy_id: str | None = None


class MaintenancePayload(APIModel):
    task_id: str
    namespace: str
    status: OperationStatus
    progress_fraction: float | None = Field(None, ge=0.0, le=1.0)
    error: str | None = None
```

Required server event types:

- Run/query: `run.started`, `run.progress`, `run.paused`, `run.resumed`,
  `run.canceled`, `run.completed`, `run.failed`, `query.completed`.
- Ledger and steering: `ledger.event`, `steering.requested`,
  `steering.applied`, `steering.rejected`.
- Verification: `verification.updated`, `verification.batch_progress`,
  `review_queue.updated`.
- Privilege/policy: `privilege.updated`, `taint.updated`, `policy.updated`,
  `privilege_log.generated`.
- Issues/proof: `issue.created`, `issue.updated`, `issue.deleted`,
  `issue.linked`, `proof_state.updated`.
- Documents/quant/gaps: `document.annotation_created`,
  `document.spans_updated`, `quant.updated`, `quant.conflict_detected`,
  `gap.created`, `gap.resolved`, `gap.dismissed`.
- Freshness/maintenance: `freshness.updated`, `matter.health_updated`,
  `maintenance.started`, `maintenance.progress`, `maintenance.completed`,
  `maintenance.failed`.

Delivery semantics:

- Server assigns monotonic `event_id` per matter. Clients reconnect with
  `last_event_id`.
- WebSocket events are notification/read-model updates, not command channels.
  Mutations still go through REST endpoints for audit, idempotency, and auth.
- Privileged payloads must be redacted unless the socket token has
  `matter:privilege`.

## OpenAPI Requirements

FastAPI/OpenAPI must include:

- Stable `operationId` values such as `verifyAssertion`, `getReviewQueue`,
  `queryMatter`, `setPrivilegeClassification`, and `triggerMaintenance`.
- Tags matching contract sections: `Verification`, `Query`, `Steering`,
  `Privilege`, `Issues`, `Documents`, `Quant`, `Dashboard`, `Gaps`,
  `Maintenance`, `Operations`, `WebSocket`.
- `securitySchemes` for bearer JWT and per-operation security requirements.
- Standard `ErrorResponse` registered for every non-2xx response.
- Examples for every request model and one representative response per endpoint.
- Vendor extensions generated from the endpoint matrix:
  `x-rateLimit-bucket`, `x-db-reads`, `x-db-writes`, and `x-async-mode`.
- `deprecated: true` on compatibility aliases.
- No untyped `dict` response models for public endpoints except extension fields
  named `metadata`, `policy_manifest`, `llm_usage`, or `details`.
- ISO 8601 UTC for datetimes. Date-only legal event fields use `YYYY-MM-DD`
  plus `date_precision` when needed.
- Documentation text must state that candidate intelligence is not verified
  legal truth and that clean policy outputs may omit privileged support.
