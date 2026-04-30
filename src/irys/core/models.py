"""Model tiering system for Gemini API access.

Tier Strategy:
- NANO: Flash-Lite with short output budget for triage/classification tasks
         (doc type, actor spotting, section headers, relevance scoring)
- LITE: Flash-Lite for full document reading and assertion extraction
- FLASH: Flash-Lite Preview for intelligent tasks (search decisions, routing, planning)
- PRO: Pro for final synthesis (polished legal output)

Cost optimization (D-007):
- Context caching: pass cached_content=<name> to complete() to reuse a cached prefix
  (90% discount on cache reads). Create the cached resource with create_cached_content().
  Typical cached prefixes: system instructions, matter summary, active issue tree.
- Google Batch API (50% off, 24h turnaround) is a planned optimization for cold-path
  NANO/LITE document ingestion; not yet implemented in this module.
- FLASH + PRO: realtime only (user-interactive)
"""

from contextvars import ContextVar, Token
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
import asyncio
import os
import logging
import time

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

PRICING_SOURCE_URL = "https://ai.google.dev/gemini-api/docs/pricing"
PRICING_VERIFIED_AT = "2026-04-30"

# P0.1 provenance bridge: GeminiClient.complete() stamps the active call's
# identity here so downstream writers (MatterRuntimeAdapter._auto_provenance)
# can pick up llm_call_id, model_id, model_tier, and prompt_hash without
# threading them through every call site. Cleared when complete() returns.
ACTIVE_LLM_CALL: "ContextVar[dict[str, Any] | None]" = ContextVar(
    "irys_active_llm_call",
    default=None,
)


class ModelTier(Enum):
    """Model tiers for different task complexities."""
    NANO = "nano"      # Triage: doc classification, actor spotting, relevance scoring
    LITE = "lite"      # Workhorse: full document reading, assertion extraction
    FLASH = "flash"    # Intelligent: search, routing, planning, gap detection
    PRO = "pro"        # Synthesis: final polished output, legal research


@dataclass
class ModelConfig:
    """Configuration for a model tier."""
    model_id: str
    thinking_level: str = ""
    temperature: float = 1.0
    max_output_tokens: int = 8192
    cost_per_1m_input: float = 0.25  # Default Gemini 3.1 Flash-Lite Preview pricing (verified Apr 10, 2026)
    cost_per_1m_output: float = 1.50
    large_context_threshold: Optional[int] = None
    cost_per_1m_input_large: Optional[float] = None
    cost_per_1m_output_large: Optional[float] = None

    def pricing_for_prompt_tokens(self, total_prompt_tokens: int) -> tuple[float, float, float]:
        """Return (input_rate, cache_read_rate, output_rate) for a request size.

        Gemini Pro models have a higher rate once prompt size exceeds 200k tokens.
        Flash, Flash-Lite, and Flash-Lite Preview are flat-rate for the text-only
        calls this app makes.
        Cache reads are billed at 10% of the active input rate.
        """
        input_rate = self.cost_per_1m_input
        output_rate = self.cost_per_1m_output
        if (
            self.large_context_threshold is not None
            and total_prompt_tokens > self.large_context_threshold
        ):
            input_rate = self.cost_per_1m_input_large or input_rate
            output_rate = self.cost_per_1m_output_large or output_rate
        return input_rate, input_rate * 0.10, output_rate


# Model configurations per tier
# Pricing verified April 30, 2026 against Google Gemini API pricing docs.
# NANO and LITE use the same model (gemini-2.5-flash-lite) — distinction is token budget:
#   NANO: short output (≤2048 tokens) for triage/classification tasks
#   LITE: full output (≤8192 tokens) for document reading and extraction
# FLASH uses gemini-3.1-flash-lite-preview for a stronger but still lower-cost
# routing/planning tier than Gemini 2.5 Flash.
MODEL_CONFIGS: dict[ModelTier, ModelConfig] = {
    ModelTier.NANO: ModelConfig(
        model_id="gemini-2.5-flash-lite",
        thinking_level="",
        max_output_tokens=2048,  # Triage tasks only; narrow scope
        cost_per_1m_input=0.10,
        cost_per_1m_output=0.40,
    ),
    ModelTier.LITE: ModelConfig(
        model_id="gemini-2.5-flash-lite",
        thinking_level="",
        max_output_tokens=8192,
        cost_per_1m_input=0.10,
        cost_per_1m_output=0.40,
    ),
    ModelTier.FLASH: ModelConfig(
        model_id="gemini-3.1-flash-lite-preview",
        thinking_level="",
        max_output_tokens=16384,
        cost_per_1m_input=0.25,
        cost_per_1m_output=1.50,
    ),
    ModelTier.PRO: ModelConfig(
        model_id="gemini-3.1-pro-preview",
        thinking_level="",
        max_output_tokens=65536,
        cost_per_1m_input=2.00,
        cost_per_1m_output=12.00,
        large_context_threshold=200_000,
        cost_per_1m_input_large=4.00,
        cost_per_1m_output_large=18.00,
    ),
}


def estimate_usage_cost(
    tier: Optional["ModelTier"],
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    tool_use_prompt_tokens: int = 0,
) -> float:
    """Estimate Gemini API cost for one request or an accumulated usage bucket."""
    mc = MODEL_CONFIGS.get(tier or ModelTier.FLASH, MODEL_CONFIGS[ModelTier.FLASH])
    total_prompt_tokens = input_tokens + cache_read_tokens + tool_use_prompt_tokens
    input_rate, cache_rate, output_rate = mc.pricing_for_prompt_tokens(total_prompt_tokens)
    return (
        input_tokens * input_rate / 1_000_000
        + cache_read_tokens * cache_rate / 1_000_000
        + tool_use_prompt_tokens * input_rate / 1_000_000
        + output_tokens * output_rate / 1_000_000
    )


@dataclass
class LLMCallRecord:
    """One Gemini API request for audit and UI rollups.

    P0.1 Provenance Lite (SO-2): call_id is a stable UUID minted before
    the request, so downstream provenance_event rows can reference the
    call that produced them. prompt_hash/response_hash are SHA-256
    digests of the effective request payload and response text so a
    future audit can verify reproducibility.
    """
    model_tier: str
    model_id: str
    input_tokens: int
    cache_read_tokens: int
    tool_use_prompt_tokens: int
    thinking_tokens: int
    output_tokens: int
    total_prompt_tokens: int
    estimated_cost_usd: float
    latency_ms: int
    success: bool
    error_kind: Optional[str] = None
    usage_label: Optional[str] = None
    matter_id: Optional[str] = None
    run_id: Optional[str] = None
    call_id: Optional[str] = None
    prompt_hash: Optional[str] = None
    response_hash: Optional[str] = None


@dataclass
class UsageStats:
    """Token usage and cost tracking per tier.

    input_tokens: non-cached prompt tokens (billed at full input rate).
    cache_read_tokens: prompt tokens served from a Gemini cached-content resource
        (billed at 10% of input rate).
    output_tokens: generated tokens.
    """
    input_tokens: int = 0
    cache_read_tokens: int = 0  # From response.usage_metadata.cached_content_token_count
    tool_use_prompt_tokens: int = 0
    thinking_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    estimated_cost_usd: float = 0.0
    # Tier is set at construction time so cost_per_1m_* can be looked up
    tier: Optional["ModelTier"] = field(default=None, repr=False)

    @property
    def estimated_cost(self) -> float:
        """Cumulative estimated cost across all requests in this bucket."""
        return self.estimated_cost_usd

    @property
    def total_prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens

    @property
    def total_processed_tokens(self) -> int:
        return (
            self.input_tokens
            + self.cache_read_tokens
            + self.tool_use_prompt_tokens
            + self.thinking_tokens
            + self.output_tokens
        )

    def add(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        tool_use_prompt_tokens: int = 0,
        thinking_tokens: int = 0,
        estimated_cost_usd: Optional[float] = None,
    ):
        """Add tokens from a request."""
        self.input_tokens += input_tokens
        self.cache_read_tokens += cache_read_tokens
        self.tool_use_prompt_tokens += tool_use_prompt_tokens
        self.thinking_tokens += thinking_tokens
        self.output_tokens += output_tokens
        self.requests += 1
        self.estimated_cost_usd += (
            estimated_cost_usd
            if estimated_cost_usd is not None
            else estimate_usage_cost(
                self.tier,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                tool_use_prompt_tokens=tool_use_prompt_tokens,
            )
        )

    def copy(self) -> "UsageStats":
        return UsageStats(
            input_tokens=self.input_tokens,
            cache_read_tokens=self.cache_read_tokens,
            tool_use_prompt_tokens=self.tool_use_prompt_tokens,
            thinking_tokens=self.thinking_tokens,
            output_tokens=self.output_tokens,
            requests=self.requests,
            estimated_cost_usd=self.estimated_cost_usd,
            tier=self.tier,
        )

    def delta(self, previous: Optional["UsageStats"]) -> "UsageStats":
        prev = previous or UsageStats(tier=self.tier)
        return UsageStats(
            input_tokens=max(self.input_tokens - prev.input_tokens, 0),
            cache_read_tokens=max(self.cache_read_tokens - prev.cache_read_tokens, 0),
            tool_use_prompt_tokens=max(
                self.tool_use_prompt_tokens - prev.tool_use_prompt_tokens, 0
            ),
            thinking_tokens=max(self.thinking_tokens - prev.thinking_tokens, 0),
            output_tokens=max(self.output_tokens - prev.output_tokens, 0),
            requests=max(self.requests - prev.requests, 0),
            estimated_cost_usd=max(self.estimated_cost_usd - prev.estimated_cost_usd, 0.0),
            tier=self.tier,
        )

    def to_dict(self, *, model_id: Optional[str] = None) -> dict[str, Any]:
        return {
            "model_id": model_id,
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "tool_use_prompt_tokens": self.tool_use_prompt_tokens,
            "thinking_tokens": self.thinking_tokens,
            "output_tokens": self.output_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_processed_tokens": self.total_processed_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
        }


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, requests_per_minute: int = 60, burst_size: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.tokens = float(burst_size)
        self.last_update: Optional[float] = None  # Lazy init
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        """Lazy initialization of lock to avoid event loop issues."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def acquire(self):
        """Acquire a token, waiting if necessary."""
        async with self._get_lock():
            now = asyncio.get_event_loop().time()

            # Initialize last_update on first call
            if self.last_update is None:
                self.last_update = now

            time_passed = now - self.last_update
            self.last_update = now

            # Replenish tokens based on time passed
            tokens_to_add = time_passed * (self.requests_per_minute / 60.0)
            self.tokens = min(float(self.burst_size), self.tokens + tokens_to_add)

            if self.tokens < 1:
                # Wait for a token
                wait_time = (1 - self.tokens) / (self.requests_per_minute / 60.0)
                logger.debug(f"Rate limit: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                self.tokens = 0.0
            else:
                self.tokens -= 1


class GeminiClient:
    """Tiered Gemini client for RLM operations with timeout, retry, and rate limiting."""

    DEFAULT_TIMEOUT = 120.0  # 2 minutes
    MAX_RETRIES = 3
    DEFAULT_RPM = 60  # Requests per minute
    DEFAULT_BURST = 10  # Burst size

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        requests_per_minute: int = DEFAULT_RPM,
        burst_size: int = DEFAULT_BURST,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY required")

        self.client = genai.Client(api_key=self.api_key)
        self.timeout = timeout
        self._usage: dict[ModelTier, UsageStats] = {t: UsageStats(tier=t) for t in ModelTier}
        self._rate_limiter = RateLimiter(requests_per_minute, burst_size)
        self._usage_context: ContextVar[dict[str, Any] | None] = ContextVar(
            "gemini_usage_context",
            default=None,
        )

    @staticmethod
    def _parse_usage_metadata(response: Any, prompt: str) -> tuple[int, int, int, int, int]:
        """Parse token counts from a Gemini response.

        Returns (input_tokens, output_tokens, cache_read_tokens, thinking_tokens,
        tool_use_prompt_tokens) where:
        - input_tokens: non-cached prompt tokens (billed at full input rate)
        - output_tokens: generated tokens
        - cache_read_tokens: prompt tokens served from cache (billed at 10% of input rate)
        - thinking_tokens: hidden reasoning tokens reported by Gemini
        - tool_use_prompt_tokens: prompt tokens consumed by tool use

        Gemini's prompt_token_count is the TOTAL prompt including cached tokens.
        Non-cached input = prompt_token_count - cached_content_token_count.
        Falls back to char/4 estimation when usage_metadata is unavailable.
        """
        um = getattr(response, "usage_metadata", None)
        if um is not None:
            total_prompt = getattr(um, "prompt_token_count", None) or 0
            actual_output = (
                getattr(um, "candidates_token_count", None)
                or getattr(um, "response_token_count", None)
                or 0
            )
            actual_cache = getattr(um, "cached_content_token_count", None) or 0
            actual_thinking = getattr(um, "thoughts_token_count", None) or 0
            actual_tool_use = getattr(um, "tool_use_prompt_token_count", None) or 0
            actual_input = max(total_prompt - actual_cache, 0)
        else:
            actual_input = len(prompt) // 4
            actual_output = len(getattr(response, "text", "") or "") // 4
            actual_cache = 0
            actual_thinking = 0
            actual_tool_use = 0
        return (
            actual_input,
            actual_output,
            actual_cache,
            actual_thinking,
            actual_tool_use,
        )

    def _get_config(
        self, tier: ModelTier, *, json_mode: bool = False,
    ) -> types.GenerateContentConfig:
        """Get generation config for a tier."""
        mc = MODEL_CONFIGS[tier]
        config = types.GenerateContentConfig(
            temperature=mc.temperature,
            max_output_tokens=mc.max_output_tokens,
        )
        if json_mode:
            config.response_mime_type = "application/json"
        if mc.thinking_level:
            config.thinking_config = types.ThinkingConfig(thinking_level=mc.thinking_level)
        return config

    def begin_usage_context(
        self,
        *,
        matter_id: Optional[str],
        run_id: Optional[str],
        recorder: Optional[Callable[[LLMCallRecord], None]],
    ) -> Token:
        """Attach matter/run context so complete() can emit auditable call records."""
        return self._usage_context.set(
            {"matter_id": matter_id, "run_id": run_id, "recorder": recorder}
        )

    def end_usage_context(self, token: Token) -> None:
        self._usage_context.reset(token)

    def snapshot_usage(self) -> dict[ModelTier, UsageStats]:
        """Return a copy of current cumulative usage for later delta calculation."""
        return {tier: stats.copy() for tier, stats in self._usage.items()}

    def get_usage_delta(
        self,
        snapshot: Optional[dict[ModelTier, UsageStats]] = None,
    ) -> dict[str, Any]:
        """Summarize usage since a prior snapshot."""
        before = snapshot or {}
        by_tier: dict[str, dict[str, Any]] = {}
        totals = UsageStats()
        successful_requests = 0
        failed_requests = 0

        for tier in ModelTier:
            delta = self._usage[tier].delta(before.get(tier))
            if delta.requests <= 0:
                continue
            successful_requests += delta.requests
            totals.input_tokens += delta.input_tokens
            totals.cache_read_tokens += delta.cache_read_tokens
            totals.tool_use_prompt_tokens += delta.tool_use_prompt_tokens
            totals.thinking_tokens += delta.thinking_tokens
            totals.output_tokens += delta.output_tokens
            totals.requests += delta.requests
            totals.estimated_cost_usd += delta.estimated_cost_usd
            by_tier[tier.value] = delta.to_dict(model_id=MODEL_CONFIGS[tier].model_id)

        return {
            "request_count": totals.requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "input_tokens": totals.input_tokens,
            "cache_read_tokens": totals.cache_read_tokens,
            "tool_use_prompt_tokens": totals.tool_use_prompt_tokens,
            "thinking_tokens": totals.thinking_tokens,
            "output_tokens": totals.output_tokens,
            "total_prompt_tokens": totals.total_prompt_tokens,
            "total_processed_tokens": totals.total_processed_tokens,
            "estimated_cost_usd": round(totals.estimated_cost_usd, 6),
            "by_tier": by_tier,
            "pricing_source": PRICING_SOURCE_URL,
            "pricing_verified_at": PRICING_VERIFIED_AT,
        }

    def _record_call(self, record: LLMCallRecord) -> None:
        ctx = self._usage_context.get()
        if not ctx:
            return
        recorder = ctx.get("recorder")
        if recorder is None:
            return
        record.matter_id = ctx.get("matter_id")
        record.run_id = ctx.get("run_id")
        try:
            recorder(record)
        except Exception as exc:
            logger.debug("LLM usage recorder failed: %s", exc)

    @staticmethod
    def _build_chat_history(
        conversation_history: Optional[list[dict[str, str]]],
    ) -> list[types.Content]:
        """Convert prior visible turns into Gemini chat history content."""
        history: list[types.Content] = []
        for turn in conversation_history or []:
            user_text = str(turn.get("query") or "").strip()
            assistant_text = str(turn.get("answer") or "").strip()
            if user_text:
                history.append(
                    types.UserContent(parts=[types.Part(text=user_text)])
                )
            if assistant_text:
                history.append(
                    types.ModelContent(parts=[types.Part(text=assistant_text)])
                )
        return history

    async def complete(
        self,
        prompt: str,
        tier: ModelTier = ModelTier.FLASH,
        system_prompt: Optional[str] = None,
        tools: Optional[list] = None,
        timeout: Optional[float] = None,
        cached_content: Optional[str] = None,
        json_mode: bool = False,
        usage_label: Optional[str] = None,
        conversation_history: Optional[list[dict[str, str]]] = None,
    ) -> str:
        """Generate completion using specified tier with timeout.

        Args:
            prompt: The user prompt.
            tier: Model tier to use (NANO/LITE/FLASH/PRO).
            system_prompt: Optional system-level instruction prepended to prompt.
            tools: Optional tool definitions for function calling.
            timeout: Per-call timeout override (seconds).
            cached_content: Optional Gemini cached-content name (resource ID returned
                by the caching API). When provided, the cached prefix is reused and
                Gemini charges only the 10% cache-read rate instead of full input cost.
                Use for hot reusable prefixes: system instructions, matter summaries,
                active issue tree. Do NOT cache individual document content.
        """
        mc = MODEL_CONFIGS[tier]
        config = self._get_config(tier, json_mode=json_mode)
        request_timeout = timeout or self.timeout

        if tools:
            config.tools = tools

        if cached_content:
            config.cached_content = cached_content
        if system_prompt:
            config.system_instruction = system_prompt

        request_text = prompt

        # P0.1 provenance: mint a stable call_id and hash the effective
        # request payload before dispatch so both failure and success
        # paths share the same identity.
        import hashlib
        import uuid as _uuid

        _call_id = _uuid.uuid4().hex
        _prompt_hash_input = request_text
        if system_prompt:
            _prompt_hash_input = f"{system_prompt}\n---\n{request_text}"
        _prompt_hash = hashlib.sha256(
            _prompt_hash_input.encode("utf-8", errors="replace")
        ).hexdigest()

        # P0.1 provenance bridge: publish call identity so synchronous
        # writers that run inside the same task (all Matter writes do)
        # can attribute the LLM call without having to thread the id
        # through every signature.
        _active_token = ACTIVE_LLM_CALL.set({
            "call_id": _call_id,
            "model_id": mc.model_id,
            "model_tier": tier.value,
            "prompt_hash": _prompt_hash,
            "usage_label": usage_label,
        })

        logger.debug(f"Calling {mc.model_id} with {len(prompt)} chars")

        # Acquire rate limit token
        await self._rate_limiter.acquire()
        started_at = time.perf_counter()

        try:
            if conversation_history:
                history = self._build_chat_history(conversation_history)
                chat = self.client.chats.create(
                    model=mc.model_id,
                    config=config,
                    history=history,
                )
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        chat.send_message,
                        request_text,
                    ),
                    timeout=request_timeout,
                )
            else:
                contents = [
                    types.Content(
                        role="user",
                        parts=[types.Part(text=request_text)],
                    )
                ]
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.client.models.generate_content,
                        model=mc.model_id,
                        contents=contents,
                        config=config,
                    ),
                    timeout=request_timeout,
                )
        except asyncio.TimeoutError:
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            self._record_call(
                LLMCallRecord(
                    model_tier=tier.value,
                    model_id=mc.model_id,
                    usage_label=usage_label,
                    input_tokens=0,
                    cache_read_tokens=0,
                    tool_use_prompt_tokens=0,
                    thinking_tokens=0,
                    output_tokens=0,
                    total_prompt_tokens=0,
                    estimated_cost_usd=0.0,
                    latency_ms=latency_ms,
                    success=False,
                    error_kind="TimeoutError",
                    call_id=_call_id,
                    prompt_hash=_prompt_hash,
                )
            )
            logger.error(f"API call to {mc.model_id} timed out after {request_timeout}s")
            raise TimeoutError(f"API call timed out after {request_timeout}s")
        except Exception as exc:
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            self._record_call(
                LLMCallRecord(
                    model_tier=tier.value,
                    model_id=mc.model_id,
                    usage_label=usage_label,
                    input_tokens=0,
                    cache_read_tokens=0,
                    tool_use_prompt_tokens=0,
                    thinking_tokens=0,
                    output_tokens=0,
                    total_prompt_tokens=0,
                    estimated_cost_usd=0.0,
                    latency_ms=latency_ms,
                    success=False,
                    error_kind=type(exc).__name__,
                    call_id=_call_id,
                    prompt_hash=_prompt_hash,
                )
            )
            raise

        (
            actual_input,
            actual_output,
            actual_cache,
            actual_thinking,
            actual_tool_use,
        ) = self._parse_usage_metadata(response, prompt)
        call_cost = estimate_usage_cost(
            tier,
            input_tokens=actual_input,
            output_tokens=actual_output,
            cache_read_tokens=actual_cache,
            tool_use_prompt_tokens=actual_tool_use,
        )
        self._usage[tier].add(
            actual_input,
            actual_output,
            cache_read_tokens=actual_cache,
            tool_use_prompt_tokens=actual_tool_use,
            thinking_tokens=actual_thinking,
            estimated_cost_usd=call_cost,
        )
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        _response_text = response.text or ""
        _response_hash = hashlib.sha256(
            _response_text.encode("utf-8", errors="replace")
        ).hexdigest()
        self._record_call(
            LLMCallRecord(
                model_tier=tier.value,
                model_id=mc.model_id,
                usage_label=usage_label,
                input_tokens=actual_input,
                cache_read_tokens=actual_cache,
                tool_use_prompt_tokens=actual_tool_use,
                thinking_tokens=actual_thinking,
                output_tokens=actual_output,
                total_prompt_tokens=actual_input + actual_cache,
                estimated_cost_usd=call_cost,
                latency_ms=latency_ms,
                success=True,
                call_id=_call_id,
                prompt_hash=_prompt_hash,
                response_hash=_response_hash,
            )
        )
        # P0.1: intentionally do NOT reset ACTIVE_LLM_CALL here — writers
        # that run after complete() returns (extraction → record_fact)
        # must still see the originating call identity. Next complete()
        # call overwrites it. The _active_token is unused; keeping it
        # for symmetry with the set() call.
        del _active_token

        logger.debug(f"Got response: {len(response.text) if response.text else 0} chars")
        return response.text

    async def complete_with_retry(
        self,
        prompt: str,
        tier: ModelTier = ModelTier.FLASH,
        system_prompt: Optional[str] = None,
        max_retries: int = MAX_RETRIES,
    ) -> str:
        """Complete with exponential backoff retry."""
        last_error = None

        for attempt in range(max_retries):
            try:
                return await self.complete(prompt, tier, system_prompt)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)

        logger.error(f"All {max_retries} attempts failed")
        raise last_error

