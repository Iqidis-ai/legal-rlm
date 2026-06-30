"""OpenAI fallback provider for Irys LLM calls.

Final-layer fallback that mirrors `GeminiClient.complete()` (and the sibling
AnthropicClient) so it can serve as an additional last-resort provider when the
Gemini API + Vertex AI chain is exhausted (e.g. PROHIBITED_CONTENT hard-blocks).

NOT integrated yet. To enable:
  1. `pip install openai`
  2. set OPENAI_API_KEY in the environment
  3. wire OpenAIClient.complete(...) into the exhaustion path in
     GeminiClient.complete() (alongside / after the Anthropic fallback).
"""

from typing import Optional, Any
import asyncio
import os
import logging
import time

from .models import ModelTier, SYSTEM_PROMPT_PRO, SYSTEM_PROMPT_FLASH, SYSTEM_PROMPT_WORKER

logger = logging.getLogger(__name__)


# Per-tier OpenAI model + limits (latest catalog as of 2026-06).
# These are reasoning models: they reject `temperature` and require
# `max_completion_tokens` (not `max_tokens`). Adjust here without touching call sites.
OPENAI_MODELS: dict[ModelTier, str] = {
    ModelTier.LITE: "gpt-5.4-mini",  # fast/cheap workhorse
    ModelTier.FLASH: "gpt-5.4",      # balanced intelligence
    ModelTier.PRO: "gpt-5.4",        # use faster 5.4 for now (was gpt-5.5)
}

# Max completion tokens (includes reasoning + visible output). All GPT-5.x cap
# at 128K; kept large since callers may truncate. Streaming is used regardless.
OPENAI_MAX_TOKENS: dict[ModelTier, int] = {
    ModelTier.LITE: 64000,
    ModelTier.FLASH: 64000,
    ModelTier.PRO: 128000,
}

# Reasoning effort per tier (lower = faster/cheaper). Supported: none/minimal/
# low/medium/high/xhigh depending on model.
OPENAI_REASONING_EFFORT: dict[ModelTier, str] = {
    ModelTier.LITE: "low",
    ModelTier.FLASH: "low",
    ModelTier.PRO: "medium",
}

# USD per 1M tokens (input, output). VERIFY against current OpenAI pricing.
OPENAI_PRICING: dict[ModelTier, tuple] = {
    ModelTier.LITE: (0.25, 2.00),
    ModelTier.FLASH: (1.25, 10.00),
    ModelTier.PRO: (1.25, 10.00),
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


class OpenAIClient:
    """OpenAI-backed client mirroring GeminiClient.complete()'s interface."""

    def __init__(self, api_key: Optional[str] = None, timeout: float = 120.0):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY required")
        # Lazy import so the package is only needed when this fallback is used.
        try:
            import openai
        except ImportError as e:
            raise ImportError("openai package not installed — run `pip install openai`") from e
        self._openai = openai
        self.client = openai.AsyncOpenAI(api_key=self.api_key)
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
        """Generate a completion via OpenAI. Signature matches GeminiClient.complete().

        `tools` is accepted for interface parity but ignored (OpenAI tool schema
        differs from Gemini); the current fallback call sites use plain text/JSON.
        `use_cache` and `overall_timeout` are accepted for parity and currently
        unused (caching and overall-timeout are handled by the calling GeminiClient).
        """
        model = OPENAI_MODELS[tier]
        max_tokens = OPENAI_MAX_TOKENS[tier]
        request_timeout = timeout if timeout is not None else TIER_TIMEOUTS.get(tier, self.timeout)
        if tools:
            logger.debug("OpenAIClient.complete: tools provided but ignored (parity-only)")

        call_start = time.monotonic()

        async def _run():
            # Streaming with usage so large max_completion_tokens don't stall on a
            # single blocking call; temperature omitted (reasoning models reject it).
            stream = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _system_prompt_for(tier, system_prompt)},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=max_tokens,
                reasoning_effort=OPENAI_REASONING_EFFORT[tier],
                stream=True,
                stream_options={"include_usage": True},
            )
            parts, usage = [], None
            async for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        parts.append(delta.content)
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
            return "".join(parts), usage

        if request_timeout and request_timeout > 0:
            text, usage = await asyncio.wait_for(_run(), timeout=request_timeout)
        else:
            text, usage = await _run()
        call_latency_ms = int((time.monotonic() - call_start) * 1000)

        if not text:
            raise RuntimeError(f"{model} returned empty response")

        in_tok = getattr(usage, "prompt_tokens", 0) or 0 if usage else len(prompt) // 4
        out_tok = getattr(usage, "completion_tokens", 0) or 0 if usage else len(text) // 4
        in_rate, out_rate = OPENAI_PRICING[tier]
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
                name=generation_name or f"{tier.value}_completion_openai",
                model=model, input=prompt, output=text,
                usage={"input": in_tok, "output": out_tok, "total": in_tok + out_tok, "unit": "TOKENS"},
                metadata={"tier": tier.value, "provider": "openai"},
            )
        return text
