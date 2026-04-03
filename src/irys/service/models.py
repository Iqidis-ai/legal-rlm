"""Pydantic models for API request/response schemas."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Investigation job status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class InvestigateRequest(BaseModel):
    """Request to start an investigation."""
    query: str = Field(..., description="Investigation query")
    s3_prefix: str = Field(..., description="S3 prefix containing documents")
    callback_url: Optional[str] = Field(
        None, description="URL to POST results when complete"
    )
    options: Optional[dict[str, Any]] = Field(
        default_factory=dict, description="Additional investigation options"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the key payment terms and obligations?",
                "s3_prefix": "documents/contracts/acme-deal",
                "callback_url": "https://your-service.com/webhook/investigation",
            }
        }


class InvestigateResponse(BaseModel):
    """Response from starting an investigation."""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    message: str = Field(..., description="Status message")
    estimated_seconds: Optional[int] = Field(
        None, description="Estimated completion time"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "inv_abc123",
                "status": "pending",
                "message": "Investigation queued",
                "estimated_seconds": 60,
            }
        }


class JobResult(BaseModel):
    """Complete investigation result."""
    job_id: str
    status: JobStatus
    query: str
    s3_prefix: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    # Results (only when completed)
    analysis: Optional[str] = None
    citations: Optional[list[dict[str, Any]]] = None
    entities: Optional[dict[str, Any]] = None
    documents_processed: int = 0
    error: Optional[str] = None

    # Matter model (when enable_matter_model=True)
    matter_id: Optional[str] = None
    run_id: Optional[str] = None
    corpus_key: Optional[str] = None  # Stable corpus identity; matter DB is keyed to this


# === Matter Model API Models ===

class MatterStatsResponse(BaseModel):
    """Matter model stats."""
    matter_id: str
    assertion_count: int
    open_gap_count: int
    open_issue_count: int
    actor_count: int
    quant_fact_count: int
    pending_clarifications: int
    recent_runs: int


class StopRunRequest(BaseModel):
    """Request to stop a running investigation."""
    pass


class RedirectRunRequest(BaseModel):
    """Request to redirect investigation to a specific issue."""
    issue_id: str = Field(..., description="Issue ID to redirect investigation toward")


class AnswerClarificationRequest(BaseModel):
    """User answer to a clarification question."""
    answer_text: str = Field(..., description="User's answer to the clarification question")


class TrustOverrideRequest(BaseModel):
    """User trust override for a document (SO-3 trust steering, SO-5 calibration)."""
    document_pattern: str = Field(
        ...,
        description="Document filename or relative path to override (exact match)",
    )
    trust_level: str = Field(
        ...,
        description="Trust level: 'low' (force ALLEGED), 'normal' (reset to auto), 'high' (promote ALLEGED→OPERATIVE)",
    )
    note: Optional[str] = Field(None, description="Reason for the override")


class CorrectAssertionRequest(BaseModel):
    """User correction to an assertion's belief state (SO-2 belief revision)."""
    new_belief_state: str = Field(
        ...,
        description="New belief state: alleged/argued/admitted/operative/performed/"
                    "disputed/superseded/withdrawn/inferred/resolved/unknown",
    )
    confidence: float = Field(0.8, ge=0.0, le=1.0, description="Confidence level 0.0–1.0")
    note: str = Field(..., description="Reason for the correction")


class SearchRequest(BaseModel):
    """Request for quick search."""
    query: str = Field(..., description="Search query")
    s3_prefix: str = Field(..., description="S3 prefix containing documents")
    max_results: int = Field(20, ge=1, le=100, description="Maximum results")


class SearchResult(BaseModel):
    """Single search result."""
    file: str
    page: Optional[int] = None
    text: str
    context: Optional[str] = None
    score: Optional[float] = None


class SearchResponse(BaseModel):
    """Search response."""
    results: list[SearchResult]
    total_matches: int
    documents_searched: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str
    gemini_connected: bool
    s3_connected: bool
    active_jobs: int
    temp_storage_mb: float
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None


# === File Upload Models ===


class UploadInvestigateResponse(BaseModel):
    """Response from file upload investigation."""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    message: str = Field(..., description="Status message")
    files_received: int = Field(..., description="Number of files uploaded")
    estimated_seconds: Optional[int] = Field(
        None, description="Estimated completion time"
    )


class UploadSearchResponse(BaseModel):
    """Response from file upload search."""
    results: list[SearchResult]
    total_matches: int
    files_searched: int


class SyncInvestigateResponse(BaseModel):
    """Response from synchronous investigation."""
    query: str
    analysis: str
    citations: list[dict[str, Any]] = []
    entities: dict[str, Any] = {}
    documents_processed: int
    duration_seconds: float
    s3_prefix: Optional[str] = Field(None, description="S3 prefix if files were kept")


# === S3 URL Models ===


class UrlWithMetadata(BaseModel):
    """URL with optional metadata for file type detection."""
    url: str = Field(..., description="Document URL")
    name: Optional[str] = Field(None, description="Original filename (e.g., 'Contract.pdf')")
    mime: Optional[str] = Field(None, description="MIME type (e.g., 'application/pdf')")


# Type alias for URL input - can be string or object with metadata
UrlInput = str | UrlWithMetadata


class S3UrlsInvestigateRequest(BaseModel):
    """Request to investigate documents by URLs (S3 or HTTP)."""
    query: str = Field(..., description="Investigation query")
    s3_urls: list[UrlInput] = Field(
        ...,
        description=(
            "List of document URLs. Each item can be a string URL or an object with "
            "'url', 'name' (filename), and 'mime' (MIME type) fields. "
            "Supports S3 URLs and generic HTTP(S) URLs including presigned URLs."
        ),
        min_length=1,
        max_length=50,
    )
    callback_url: Optional[str] = Field(
        None, description="URL to POST results when complete"
    )
    options: Optional[dict[str, Any]] = Field(
        default_factory=dict, description="Additional investigation options"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the key payment terms and obligations?",
                "s3_urls": [
                    "s3://my-bucket/contracts/main_contract.pdf",
                    {
                        "url": "https://bucket.s3.amazonaws.com/abc123?X-Amz-Signature=...",
                        "name": "Contract.pdf",
                        "mime": "application/pdf"
                    },
                ],
                "callback_url": "https://your-service.com/webhook/investigation",
            }
        }


class S3UrlsSearchRequest(BaseModel):
    """Request for quick search by URLs (S3 or HTTP)."""
    query: str = Field(..., description="Search query")
    s3_urls: list[UrlInput] = Field(
        ...,
        description=(
            "List of document URLs. Each item can be a string URL or an object with "
            "'url', 'name' (filename), and 'mime' (MIME type) fields."
        ),
        min_length=1,
        max_length=50,
    )
    max_results: int = Field(20, ge=1, le=100, description="Maximum results")

