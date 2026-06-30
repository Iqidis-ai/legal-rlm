"""Anthropic fallback provider for Irys LLM calls.

Final-layer fallback that mirrors `GeminiClient.complete()` and is wired in as
the last resort in GeminiClient.complete() — used AFTER the Gemini API and
Vertex AI fallbacks are exhausted (e.g. when every Gemini model returns an
empty/safety-blocked response such as PROHIBITED_CONTENT, which Gemini hard-blocks
and cannot be disabled via safety settings).

Activation: set ANTHROPIC_API_KEY in the environment. When unset, the fallback is
silently disabled and behaviour is unchanged.
"""

from typing import Optional, Any
import asyncio
import os
import logging
import time

from .models import ModelTier, SYSTEM_PROMPT_PRO, SYSTEM_PROMPT_FLASH, SYSTEM_PROMPT_WORKER

logger = logging.getLogger(__name__)


# Per-tier Anthropic model + limits (latest catalog as of 2026-06).
# Adjust here without touching call sites.
ANTHROPIC_MODELS: dict[ModelTier, str] = {
    ModelTier.LITE: "claude-haiku-4-5",    # fast/cheap workhorse
    ModelTier.FLASH: "claude-sonnet-4-6",  # balanced intelligence
    ModelTier.PRO: "claude-opus-4-6",      # Use faster sonnet for now
    # ModelTier.PRO: "claude-opus-4-8",      # flagship synthesis
}

# Max output tokens per model (kept at each model's maximum). Haiku 4.5 and
# Sonnet 4.6 cap at 64K; Opus 4.8 caps at 128K. Streaming is used so these
# large limits don't trip the SDK's non-streaming long-request guard.
ANTHROPIC_MAX_TOKENS: dict[ModelTier, int] = {
    ModelTier.LITE: 64000,
    ModelTier.FLASH: 64000,
    ModelTier.PRO: 128000,
}

# USD per 1M tokens (input, output).
ANTHROPIC_PRICING: dict[ModelTier, tuple] = {
    ModelTier.LITE: (1.00, 5.00),    # Haiku 4.5
    ModelTier.FLASH: (3.00, 15.00),  # Sonnet 4.6
    ModelTier.PRO: (5.00, 25.00),    # Opus 4.8
}

TIER_TIMEOUTS: dict[ModelTier, float] = {
    ModelTier.LITE: 80.0,
    ModelTier.FLASH: 120.0,
    ModelTier.PRO: 120.0,
}


def _system_prompt_for(tier: ModelTier, system_prompt: Optional[str]) -> str:
    if system_prompt is not None:
        return system_prompt
    if tier == ModelTier.PRO:
        return SYSTEM_PROMPT_PRO
    if tier == ModelTier.FLASH:
        return SYSTEM_PROMPT_FLASH
    return SYSTEM_PROMPT_WORKER


class AnthropicClient:
    """Anthropic-backed client mirroring GeminiClient.complete()'s interface."""

    def __init__(self, api_key: Optional[str] = None, timeout: float = 120.0):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY required")
        # Lazy import so the package is only needed when this fallback is used.
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("anthropic package not installed — run `pip install anthropic`") from e
        self._anthropic = anthropic
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        self.timeout = timeout

    async def complete(
        self,
        prompt: str,
        tier: ModelTier = ModelTier.FLASH,
        system_prompt: Optional[str] = None,
        tools: Optional[list] = None,
        timeout: Optional[float] = None,
        overall_timeout: Optional[float] = None,
        use_cache: bool = True,
        active_step: Optional[Any] = None,
        trace_ctx: Optional[Any] = None,
        generation_name: Optional[str] = None,
    ) -> str:
        """Generate a completion via Anthropic. Signature matches GeminiClient.complete().

        `tools` is accepted for interface parity but ignored (Anthropic tool schema
        differs from Gemini); the current fallback call sites use plain text/JSON.
        `use_cache` and `overall_timeout` are accepted for parity and currently
        unused (caching and overall-timeout are handled by the calling GeminiClient).
        """
        model = ANTHROPIC_MODELS[tier]
        max_tokens = ANTHROPIC_MAX_TOKENS[tier]
        request_timeout = timeout if timeout is not None else TIER_TIMEOUTS.get(tier, self.timeout)
        if tools:
            logger.debug("AnthropicClient.complete: tools provided but ignored (parity-only)")

        call_start = time.monotonic()

        async def _run():
            # Streaming is required for large max_tokens (avoids the SDK's
            # non-streaming long-request guard) and yields final usage metadata.
            # temperature is intentionally omitted: newer Claude models (e.g.
            # opus-4-8) deprecate it and reject the parameter with a 400.
            async with self.client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                system=_system_prompt_for(tier, system_prompt),
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for _ in stream.text_stream:
                    pass
                return await stream.get_final_message()

        if request_timeout and request_timeout > 0:
            final = await asyncio.wait_for(_run(), timeout=request_timeout)
        else:
            final = await _run()
        call_latency_ms = int((time.monotonic() - call_start) * 1000)

        text = "".join(
            getattr(block, "text", "") for block in (final.content or [])
            if getattr(block, "type", None) == "text"
        )
        if not text:
            stop = getattr(final, "stop_reason", "unknown")
            raise RuntimeError(f"{model} returned empty response (stop_reason={stop})")

        usage = getattr(final, "usage", None)
        in_tok = getattr(usage, "input_tokens", 0) or 0 if usage else len(prompt) // 4
        out_tok = getattr(usage, "output_tokens", 0) or 0 if usage else len(text) // 4
        in_rate, out_rate = ANTHROPIC_PRICING[tier]
        cost = in_tok * in_rate / 1_000_000 + out_tok * out_rate / 1_000_000

        if active_step is not None:
            from .telemetry import StepOperation
            active_step.add_operation(StepOperation(
                type="llm", latency_ms=call_latency_ms, tier=tier.value.upper(),
                model_id=model, prompt_tokens=in_tok, thinking_tokens=0,
                output_tokens=out_tok, total_tokens=in_tok + out_tok,
                cost_usd=round(cost, 6), cached=False,
            ))
        if trace_ctx is not None:
            trace_ctx.record_generation(
                name=generation_name or f"{tier.value}_completion_anthropic",
                model=model, input=prompt, output=text,
                usage={"input": in_tok, "output": out_tok, "total": in_tok + out_tok, "unit": "TOKENS"},
                metadata={"tier": tier.value, "provider": "anthropic"},
            )
        return text
