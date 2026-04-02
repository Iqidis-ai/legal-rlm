"""Model tiering system for Gemini API access.

Tier Strategy:
- LITE: Flash-Lite for workhorse tasks (reading, extraction, basic processing)
- FLASH: Flash for intelligent tasks (search decisions, routing, planning)
- PRO: Pro for final synthesis (polished legal output)
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
    LITE = "lite"      # Workhorse: reading, extraction
    FLASH = "flash"    # Intelligent: search, routing, planning
    PRO = "pro"        # Synthesis: final polished output


@dataclass
class ModelConfig:
    """Configuration for a model tier."""
    model_id: str
    thinking_level: str = ""
    temperature: float = 1.0
    max_output_tokens: int = 8192
    cost_per_1m_input: float = 0.075  # Default Gemini 2.5 Flash pricing
    cost_per_1m_output: float = 0.30


# Model configurations per tier
MODEL_CONFIGS: dict[ModelTier, ModelConfig] = {
    ModelTier.LITE: ModelConfig(
        model_id="gemini-2.5-flash-lite",
        thinking_level="",
        max_output_tokens=8192,  # Increased from 4096 to reduce truncation
        cost_per_1m_input=0.01875,  # 1/4 of flash
        cost_per_1m_output=0.075,
    ),
    ModelTier.FLASH: ModelConfig(
        model_id="gemini-2.5-flash",
        thinking_level="",
        max_output_tokens=16384,  # Increased from 8192 to reduce JSON truncation
        cost_per_1m_input=0.075,
        cost_per_1m_output=0.30,
    ),
    ModelTier.PRO: ModelConfig(
        model_id="gemini-2.5-pro",
        thinking_level="",
        max_output_tokens=32768,  # Increased from 16384 for thorough analysis
        cost_per_1m_input=1.25,
        cost_per_1m_output=5.00,
    ),
}


@dataclass
class UsageStats:
    """Token usage and cost tracking."""
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0

    @property
    def estimated_cost(self) -> float:
        """Estimate cost based on default Flash pricing."""
        return (
            self.input_tokens * 0.075 / 1_000_000 +
            self.output_tokens * 0.30 / 1_000_000
        )

    def add(self, input_tokens: int, output_tokens: int):
        """Add tokens from a request."""
        self.input_tokens += input_tokens
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
        self._usage: dict[ModelTier, UsageStats] = {t: UsageStats() for t in ModelTier}
        self._rate_limiter = RateLimiter(requests_per_minute, burst_size)

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
    ) -> str:
        """Generate completion using specified tier with timeout."""
        mc = MODEL_CONFIGS[tier]
        config = self._get_config(tier)
        request_timeout = timeout or self.timeout

        if tools:
            config.tools = tools

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

        # Track usage
        self._usage[tier].requests += 1
        # Estimate tokens (actual count would require response metadata)
        estimated_input = len(prompt) // 4
        estimated_output = len(response.text) // 4 if response.text else 0
        self._usage[tier].add(estimated_input, estimated_output)

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

    def get_usage(self) -> dict[str, UsageStats]:
        """Get usage statistics per tier."""
        return {tier.value: stats for tier, stats in self._usage.items()}

    def get_total_cost(self) -> float:
        """Get total estimated cost across all tiers."""
        total = 0.0
        for tier, stats in self._usage.items():
            mc = MODEL_CONFIGS[tier]
            total += stats.input_tokens * mc.cost_per_1m_input / 1_000_000
            total += stats.output_tokens * mc.cost_per_1m_output / 1_000_000
        return total

    def reset_usage(self):
        """Reset usage counters."""
        self._usage = {t: UsageStats() for t in ModelTier}
