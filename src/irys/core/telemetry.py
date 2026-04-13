"""Investigation-level cost and latency telemetry.

Tracks per-step, per-operation metrics during an investigation.
Created per investigate() call — no shared state between concurrent investigations.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
import uuid


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# StepOperation — one LLM call or one external search call
# ---------------------------------------------------------------------------

@dataclass
class StepOperation:
    """A single operation within an investigation step."""

    type: str  # "llm" | "ext_search" | "ocr"
    started_at: datetime = field(default_factory=_utcnow)
    latency_ms: int = 0

    # LLM fields (populated when type == "llm")
    tier: str = ""  # "LITE" | "FLASH" | "PRO"
    model_id: str = ""
    prompt_tokens: int = 0
    thinking_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0  # cached_content_token_count from usageMetadata
    total_tokens: int = 0  # total_token_count from usageMetadata
    cost_usd: float = 0.0
    cached: bool = False

    # External search fields (populated when type == "ext_search")
    service: str = ""  # "tavily" | "courtlistener" | "mistral-ocr"
    query: str = ""
    result_count: int = 0
    usage_raw: Optional[dict] = None

    # OCR fields (populated when type == "ocr")
    file_name: str = ""   # name of the file that was OCR'd
    file_type: str = ""   # "png" | "jpg" | "jpeg" | "pdf" | "docx"
    page_count: int = 0   # pages returned by Mistral OCR
    timed_out: bool = False  # True if the OCR call hit the timeout

    def to_dict(self) -> dict[str, Any]:
        base = {
            "type": self.type,
            "started_at": self.started_at.isoformat(),
            "latency_ms": self.latency_ms,
        }
        if self.type == "llm":
            base.update({
                "tier": self.tier,
                "model_id": self.model_id,
                "prompt_tokens": self.prompt_tokens,
                "thinking_tokens": self.thinking_tokens,
                "output_tokens": self.output_tokens,
                "cached_tokens": self.cached_tokens,
                "total_tokens": self.total_tokens,
                "cost_usd": self.cost_usd,
                "cached": self.cached,
            })
        elif self.type == "ext_search":
            base.update({
                "service": self.service,
                "query": self.query,
                "result_count": self.result_count,
                "usage_raw": self.usage_raw,
                "cost_usd": self.cost_usd,
            })
        elif self.type == "ocr":
            base.update({
                "service": self.service,
                "file_name": self.file_name,
                "file_type": self.file_type,
                "page_count": self.page_count,
                "timed_out": self.timed_out,
                "cost_usd": self.cost_usd,
            })
        return base

    def details_dict(self) -> dict[str, Any]:
        """Return type-specific details (for DB JSON column)."""
        if self.type == "llm":
            return {
                "tier": self.tier,
                "model_id": self.model_id,
                "prompt_tokens": self.prompt_tokens,
                "thinking_tokens": self.thinking_tokens,
                "output_tokens": self.output_tokens,
                "cached_tokens": self.cached_tokens,
                "total_tokens": self.total_tokens,
                "cost_usd": self.cost_usd,
                "cached": self.cached,
            }
        elif self.type == "ext_search":
            return {
                "service": self.service,
                "query": self.query,
                "result_count": self.result_count,
                "usage_raw": self.usage_raw,
                "cost_usd": self.cost_usd,
            }
        elif self.type == "ocr":
            return {
                "service": self.service,
                "file_name": self.file_name,
                "file_type": self.file_type,
                "page_count": self.page_count,
                "timed_out": self.timed_out,
                "cost_usd": self.cost_usd,
            }
        return {}


# ---------------------------------------------------------------------------
# InvestigationStep — one logical phase step (planning, analyze_document, etc.)
# ---------------------------------------------------------------------------

@dataclass
class InvestigationStep:
    """A single step within an investigation."""

    seq: int
    step_name: str  # "planning" | "analyze_document" | "sufficiency_check" | "synthesize" | ...
    phase: str  # "planning" | "investigation_loop" | "synthesis"
    started_at: datetime = field(default_factory=_utcnow)
    step_latency_ms: int = 0
    operations: list[StepOperation] = field(default_factory=list)

    def finish(self) -> None:
        """Compute step_latency_ms from started_at to now."""
        elapsed = _utcnow() - self.started_at
        self.step_latency_ms = int(elapsed.total_seconds() * 1000)

    def add_operation(self, op: StepOperation) -> None:
        self.operations.append(op)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "step_name": self.step_name,
            "phase": self.phase,
            "started_at": self.started_at.isoformat(),
            "step_latency_ms": self.step_latency_ms,
            "operations": [op.to_dict() for op in self.operations],
        }


# ---------------------------------------------------------------------------
# TelemetrySummary — output of finalize(), what gets stored and logged
# ---------------------------------------------------------------------------

@dataclass
class TelemetrySummary:
    """Finalized telemetry output for logging and DB persistence."""

    investigation_id: str
    message_id: Optional[str]
    user_id: Optional[str]
    started_at: datetime
    completed_at: datetime
    status: str
    setup_duration_ms: int          # document download / preparation before investigate()
    total_duration_ms: int
    total_cost_usd: float
    total_steps: int
    phase_breakdown: dict[str, dict[str, Any]]
    steps: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "investigation_id": self.investigation_id,
            "message_id": self.message_id,
            "user_id": self.user_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "status": self.status,
            "setup_duration_ms": self.setup_duration_ms,
            "total_duration_ms": self.total_duration_ms,
            "total_cost_usd": self.total_cost_usd,
            "total_steps": self.total_steps,
            "phase_breakdown": self.phase_breakdown,
            "steps": self.steps,
        }


# ---------------------------------------------------------------------------
# InvestigationTelemetry — top-level container created per investigate() call
# ---------------------------------------------------------------------------

@dataclass
class InvestigationTelemetry:
    """Per-investigation telemetry collector.

    Create at the start of investigate(), call begin_step() / end_step()
    around each phase, and finalize() at the end.
    """

    investigation_id: str = field(default_factory=lambda: f"inv_{uuid.uuid4().hex[:8]}")
    message_id: Optional[str] = None
    user_id: Optional[str] = None
    started_at: datetime = field(default_factory=_utcnow)
    completed_at: Optional[datetime] = None
    status: Optional[str] = None
    setup_duration_ms: int = 0      # document download / preparation before investigate()
    steps: list[InvestigationStep] = field(default_factory=list)
    _seq_counter: int = field(default=0, repr=False)

    def begin_step(self, step_name: str, phase: str) -> InvestigationStep:
        """Start a new step and return it (caller passes to GeminiClient)."""
        self._seq_counter += 1
        step = InvestigationStep(
            seq=self._seq_counter,
            step_name=step_name,
            phase=phase,
        )
        self.steps.append(step)
        return step

    def end_step(self, step: InvestigationStep) -> None:
        """Finalize a step's latency."""
        step.finish()

    def finalize(self, status: str = "completed") -> TelemetrySummary:
        """Compute totals and return a TelemetrySummary."""
        self.completed_at = _utcnow()
        self.status = status

        total_duration_ms = int(
            (self.completed_at - self.started_at).total_seconds() * 1000
        )

        total_cost_usd = 0.0
        phase_map: dict[str, dict[str, Any]] = {}

        for step in self.steps:
            # Accumulate cost from operations
            for op in step.operations:
                total_cost_usd += op.cost_usd

            # Phase breakdown
            if step.phase not in phase_map:
                phase_map[step.phase] = {"duration_ms": 0, "step_count": 0}
            phase_map[step.phase]["duration_ms"] += step.step_latency_ms
            phase_map[step.phase]["step_count"] += 1

        return TelemetrySummary(
            investigation_id=self.investigation_id,
            message_id=self.message_id,
            user_id=self.user_id,
            started_at=self.started_at,
            completed_at=self.completed_at,
            status=status,
            setup_duration_ms=self.setup_duration_ms,
            total_duration_ms=total_duration_ms,
            total_cost_usd=round(total_cost_usd, 6),
            total_steps=len(self.steps),
            phase_breakdown=phase_map,
            steps=[s.to_dict() for s in self.steps],
        )
