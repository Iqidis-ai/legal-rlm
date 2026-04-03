"""Model tiering system for Gemini API access.

Tier Strategy:
- NANO: Flash-Lite with short output budget for triage/classification tasks
         (doc type, actor spotting, section headers, relevance scoring)
- LITE: Flash-Lite for full document reading and assertion extraction
- FLASH: Flash for intelligent tasks (search decisions, routing, planning)
- PRO: Pro for final synthesis (polished legal output)

Cost optimization (D-007):
- Context caching: pass cached_content=<name> to complete() to reuse a cached prefix
  (90% discount on cache reads). Create the cached resource with create_cached_content().
  Typical cached prefixes: system instructions, matter summary, active issue tree.
- Google Batch API (50% off, 24h turnaround) is a planned optimization for cold-path
  NANO/LITE document ingestion; not yet implemented in this module.
- FLASH + PRO: realtime only (user-interactive)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
import asyncio
import os
import logging

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


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
    cost_per_1m_input: float = 0.30  # Default Gemini 2.5 Flash pricing (verified Apr 2026)
    cost_per_1m_output: float = 2.50


# Model configurations per tier
# Pricing verified April 2026 (see experiments/EXPERIMENTS.md EXP-009, D-007)
# NANO and LITE use the same model (gemini-2.5-flash-lite) — distinction is token budget:
#   NANO: short output (≤2048 tokens) for triage/classification tasks
#   LITE: full output (≤8192 tokens) for document reading and extraction
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
        model_id="gemini-2.5-flash",
        thinking_level="",
        max_output_tokens=16384,
        cost_per_1m_input=0.30,
        cost_per_1m_output=2.50,
    ),
    ModelTier.PRO: ModelConfig(
        model_id="gemini-2.5-pro",
        thinking_level="",
        max_output_tokens=32768,
        cost_per_1m_input=1.25,
        cost_per_1m_output=10.00,
    ),
}


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
    output_tokens: int = 0
    requests: int = 0
    # Tier is set at construction time so cost_per_1m_* can be looked up
    tier: Optional["ModelTier"] = field(default=None, repr=False)

    @property
    def estimated_cost(self) -> float:
        """Estimate cost using per-tier pricing.

        Cache reads are billed at 10% of the input rate (Google Gemini caching pricing).
        Non-cached input tokens are billed at full input rate.
        Note: cache creation and storage costs are not tracked here.
        """
        if self.tier is not None and self.tier in MODEL_CONFIGS:
            mc = MODEL_CONFIGS[self.tier]
            return (
                self.input_tokens * mc.cost_per_1m_input / 1_000_000
                + self.cache_read_tokens * mc.cost_per_1m_input * 0.10 / 1_000_000
                + self.output_tokens * mc.cost_per_1m_output / 1_000_000
            )
        # Fallback: Flash pricing
        return (
            self.input_tokens * 0.30 / 1_000_000
            + self.cache_read_tokens * 0.30 * 0.10 / 1_000_000
            + self.output_tokens * 2.50 / 1_000_000
        )

    def add(self, input_tokens: int, output_tokens: int, cache_read_tokens: int = 0):
        """Add tokens from a request."""
        self.input_tokens += input_tokens
        self.cache_read_tokens += cache_read_tokens
        self.output_tokens += output_tokens
        self.requests += 1


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


@dataclass
class ThinkingCallback:
    """Callback for streaming thinking steps."""
    on_thinking: Optional[Callable[[str], None]] = None
    on_search: Optional[Callable[[str], None]] = None
    on_finding: Optional[Callable[[str, str], None]] = None
    on_replan: Optional[Callable[[str], None]] = None
    on_citation: Optional[Callable[[str, str, str], None]] = None


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

    @staticmethod
    def _parse_usage_metadata(response: Any, prompt: str) -> tuple[int, int, int]:
        """Parse token counts from a Gemini response.

        Returns (input_tokens, output_tokens, cache_read_tokens) where:
        - input_tokens: non-cached prompt tokens (billed at full input rate)
        - output_tokens: generated tokens
        - cache_read_tokens: prompt tokens served from cache (billed at 10% of input rate)

        Gemini's prompt_token_count is the TOTAL prompt including cached tokens.
        Non-cached input = prompt_token_count - cached_content_token_count.
        Falls back to char/4 estimation when usage_metadata is unavailable.
        """
        um = getattr(response, "usage_metadata", None)
        if um is not None:
            total_prompt = getattr(um, "prompt_token_count", None) or 0
            actual_output = getattr(um, "candidates_token_count", None) or 0
            actual_cache = getattr(um, "cached_content_token_count", None) or 0
            actual_input = max(total_prompt - actual_cache, 0)
        else:
            actual_input = len(prompt) // 4
            actual_output = len(getattr(response, "text", "") or "") // 4
            actual_cache = 0
        return actual_input, actual_output, actual_cache

    def _get_config(self, tier: ModelTier) -> types.GenerateContentConfig:
        """Get generation config for a tier."""
        mc = MODEL_CONFIGS[tier]
        config = types.GenerateContentConfig(
            temperature=mc.temperature,
            max_output_tokens=mc.max_output_tokens,
        )
        if mc.thinking_level:
            config.thinking_config = types.ThinkingConfig(thinking_level=mc.thinking_level)
        return config

    async def complete(
        self,
        prompt: str,
        tier: ModelTier = ModelTier.FLASH,
        system_prompt: Optional[str] = None,
        tools: Optional[list] = None,
        timeout: Optional[float] = None,
        cached_content: Optional[str] = None,
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
        config = self._get_config(tier)
        request_timeout = timeout or self.timeout

        if tools:
            config.tools = tools

        if cached_content:
            config.cached_content = cached_content

        contents = []
        if system_prompt:
            contents.append(types.Content(
                role="user",
                parts=[types.Part(text=f"System: {system_prompt}\n\nUser: {prompt}")]
            ))
        else:
            contents.append(types.Content(
                role="user",
                parts=[types.Part(text=prompt)]
            ))

        logger.debug(f"Calling {mc.model_id} with {len(prompt)} chars")

        # Acquire rate limit token
        await self._rate_limiter.acquire()

        try:
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
            logger.error(f"API call to {mc.model_id} timed out after {request_timeout}s")
            raise TimeoutError(f"API call timed out after {request_timeout}s")

        actual_input, actual_output, actual_cache = self._parse_usage_metadata(response, prompt)
        self._usage[tier].add(actual_input, actual_output, cache_read_tokens=actual_cache)

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

    async def complete_with_history(
        self,
        messages: list[dict],
        tier: ModelTier = ModelTier.FLASH,
    ) -> str:
        """Generate completion with conversation history."""
        mc = MODEL_CONFIGS[tier]
        config = self._get_config(tier)

        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(
                role=role,
                parts=[types.Part(text=msg["content"])]
            ))

        # Acquire rate limit token
        await self._rate_limiter.acquire()

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.models.generate_content,
                    model=mc.model_id,
                    contents=contents,
                    config=config,
                ),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"API call timed out after {self.timeout}s")

        self._usage[tier].requests += 1
        return response.text

    async def batch_complete(
        self,
        prompts: list[str],
        tier: ModelTier = ModelTier.LITE,
        max_concurrent: int = 5,
    ) -> list[str]:
        """Process multiple prompts in parallel."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_one(prompt: str) -> str:
            async with semaphore:
                return await self.complete(prompt, tier=tier)

        tasks = [process_one(p) for p in prompts]
        return await asyncio.gather(*tasks)

    def create_cached_content(
        self,
        content: str,
        tier: ModelTier = ModelTier.FLASH,
        ttl_seconds: int = 300,
        display_name: Optional[str] = None,
    ) -> Optional[str]:
        """Create a Gemini cached content resource and return its name (resource ID).

        Use this for hot reusable prefixes that are injected into many calls within
        a run: system instructions, matter summaries, active issue tree text.
        Pass the returned name as `cached_content` in subsequent `complete()` calls.

        Charges: cache write is 1.25x input cost; cache reads cost 10% of input cost.
        Minimum cache size is 32,768 tokens. TTL default is 5 minutes (300s).

        Returns None on failure (caching is a performance optimization, never critical).
        """
        try:
            mc = MODEL_CONFIGS[tier]
            cached = self.client.caches.create(
                model=mc.model_id,
                config=types.CreateCachedContentConfig(
                    contents=[types.Content(
                        role="user",
                        parts=[types.Part(text=content)],
                    )],
                    ttl=f"{ttl_seconds}s",
                    display_name=display_name,
                ),
            )
            return cached.name
        except Exception as e:
            logger.warning("Context caching failed (non-critical): %s", e)
            return None

    def get_usage(self) -> dict[str, UsageStats]:
        """Get usage statistics per tier."""
        return {tier.value: stats for tier, stats in self._usage.items()}

    def get_total_cost(self) -> float:
        """Get total estimated cost across all tiers."""
        total = 0.0
        for tier, stats in self._usage.items():
            total += stats.estimated_cost
        return total

    def reset_usage(self):
        """Reset usage counters."""
        self._usage = {t: UsageStats(tier=t) for t in ModelTier}
