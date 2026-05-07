"""Tests for UsageStats token accounting and cost estimation."""
from types import SimpleNamespace
import pytest
from irys.core.models import UsageStats, ModelTier, MODEL_CONFIGS, GeminiClient


class TestUsageStatsCostEstimation:
    def test_pro_tier_uses_current_model_limit(self):
        """PRO tier uses current model's documented output cap."""
        mc = MODEL_CONFIGS[ModelTier.PRO]
        assert mc.model_id == "gemini-3.1-flash-lite-preview"
        assert mc.max_output_tokens == 65_536

    def test_nano_tier_uses_lite_pricing(self):
        """NANO uses same flash-lite model/price as LITE."""
        stats = UsageStats(tier=ModelTier.NANO)
        stats.add(input_tokens=1_000_000, output_tokens=1_000_000)
        mc = MODEL_CONFIGS[ModelTier.NANO]
        expected = mc.cost_per_1m_input + mc.cost_per_1m_output
        assert abs(stats.estimated_cost - expected) < 1e-9

    def test_cache_read_tokens_billed_at_10_percent(self):
        """Cache-read tokens are charged at 10% of the input rate."""
        stats = UsageStats(tier=ModelTier.LITE)
        stats.add(input_tokens=0, output_tokens=0, cache_read_tokens=1_000_000)
        mc = MODEL_CONFIGS[ModelTier.LITE]
        expected = mc.cost_per_1m_input * 0.10
        assert abs(stats.estimated_cost - expected) < 1e-9

    def test_cache_read_cheaper_than_normal_input(self):
        """1M cached input tokens must cost less than 1M non-cached input tokens."""
        stats_cached = UsageStats(tier=ModelTier.FLASH)
        stats_cached.add(input_tokens=0, output_tokens=0, cache_read_tokens=1_000_000)

        stats_normal = UsageStats(tier=ModelTier.FLASH)
        stats_normal.add(input_tokens=1_000_000, output_tokens=0)

        assert stats_cached.estimated_cost < stats_normal.estimated_cost

    def test_input_and_cache_combined_cost(self):
        """Non-cached input + cached input each billed at their respective rates."""
        stats = UsageStats(tier=ModelTier.FLASH)
        stats.add(input_tokens=500_000, output_tokens=0, cache_read_tokens=500_000)
        mc = MODEL_CONFIGS[ModelTier.FLASH]
        expected = (
            500_000 * mc.cost_per_1m_input / 1_000_000
            + 500_000 * mc.cost_per_1m_input * 0.10 / 1_000_000
        )
        assert abs(stats.estimated_cost - expected) < 1e-9

    def test_cache_read_tokens_default_zero(self):
        """add() with no cache_read_tokens kwarg defaults to zero."""
        stats = UsageStats(tier=ModelTier.LITE)
        stats.add(input_tokens=100, output_tokens=50)
        assert stats.cache_read_tokens == 0

    def test_pro_uses_flash_lite_flat_pricing(self):
        """PRO tier (currently Flash-Lite) uses flat pricing, no large-context premium."""
        mc = MODEL_CONFIGS[ModelTier.PRO]
        assert mc.large_context_threshold is None
        stats = UsageStats(tier=ModelTier.PRO)
        stats.add(input_tokens=250_000, output_tokens=100_000)
        expected = (
            250_000 * mc.cost_per_1m_input / 1_000_000
            + 100_000 * mc.cost_per_1m_output / 1_000_000
        )
        assert abs(stats.estimated_cost - expected) < 1e-9

    def test_pro_cache_reads_at_10_percent(self):
        """PRO cache reads at 10% of flat input rate."""
        mc = MODEL_CONFIGS[ModelTier.PRO]
        stats = UsageStats(tier=ModelTier.PRO)
        stats.add(input_tokens=0, output_tokens=0, cache_read_tokens=250_000)
        expected = 250_000 * mc.cost_per_1m_input * 0.10 / 1_000_000
        assert abs(stats.estimated_cost - expected) < 1e-9

    def test_fallback_cost_when_tier_is_none(self):
        """When tier is None, falls back to Flash pricing (not crashes)."""
        stats = UsageStats()
        stats.add(input_tokens=1_000_000, output_tokens=1_000_000)
        # Fallback: Flash $0.25 in + $1.50 out
        expected = 0.25 + 1.50
        assert abs(stats.estimated_cost - expected) < 1e-9

    def test_accumulation(self):
        """Multiple add() calls accumulate correctly."""
        stats = UsageStats(tier=ModelTier.LITE)
        stats.add(
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=200,
            tool_use_prompt_tokens=25,
            thinking_tokens=10,
        )
        stats.add(
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=200,
            tool_use_prompt_tokens=25,
            thinking_tokens=10,
        )
        assert stats.input_tokens == 200
        assert stats.output_tokens == 100
        assert stats.cache_read_tokens == 400
        assert stats.tool_use_prompt_tokens == 50
        assert stats.thinking_tokens == 20
        assert stats.total_processed_tokens == 770
        assert stats.requests == 2


class TestCompleteUsageMetadataSplit:
    """Regression tests for the prompt_token_count split in GeminiClient._parse_usage_metadata.

    These tests call the actual static method on GeminiClient so that any regression
    in that method is immediately caught.
    """

    def _make_response(
        self,
        prompt=500,
        candidates=100,
        cached=200,
        thoughts=0,
        tool_use=0,
    ):
        um = SimpleNamespace(
            prompt_token_count=prompt,
            candidates_token_count=candidates,
            cached_content_token_count=cached,
            thoughts_token_count=thoughts,
            tool_use_prompt_token_count=tool_use,
        )
        return SimpleNamespace(usage_metadata=um, text="result")

    def test_cached_tokens_subtracted_from_prompt(self):
        """Non-cached input = total_prompt - cached (the prior double-count bug regression)."""
        response = self._make_response(prompt=500, candidates=100, cached=200)
        actual_input, actual_output, actual_cache, actual_thinking, actual_tool_use = GeminiClient._parse_usage_metadata(
            response, "prompt text"
        )
        assert actual_input == 300   # 500 - 200
        assert actual_cache == 200
        assert actual_output == 100
        assert actual_thinking == 0
        assert actual_tool_use == 0

    def test_no_cache_all_tokens_are_input(self):
        """When cached_content_token_count is 0, all prompt tokens are non-cached."""
        response = self._make_response(prompt=400, candidates=80, cached=0)
        actual_input, actual_output, actual_cache, actual_thinking, actual_tool_use = GeminiClient._parse_usage_metadata(
            response, "prompt text"
        )
        assert actual_input == 400
        assert actual_cache == 0
        assert actual_thinking == 0
        assert actual_tool_use == 0

    def test_missing_metadata_fields_default_to_zero(self):
        """getattr fallback handles missing attributes without AttributeError."""
        response = SimpleNamespace(usage_metadata=SimpleNamespace(), text="result")
        actual_input, actual_output, actual_cache, actual_thinking, actual_tool_use = GeminiClient._parse_usage_metadata(
            response, "prompt text"
        )
        assert actual_input == 0
        assert actual_output == 0
        assert actual_cache == 0
        assert actual_thinking == 0
        assert actual_tool_use == 0

    def test_full_cache_hit_input_is_zero(self):
        """If all prompt tokens came from cache, non-cached input is 0 (not negative)."""
        response = self._make_response(prompt=300, candidates=50, cached=300)
        actual_input, actual_output, actual_cache, actual_thinking, actual_tool_use = GeminiClient._parse_usage_metadata(
            response, "prompt text"
        )
        assert actual_input == 0  # max(300-300, 0) = 0, not negative
        assert actual_cache == 300
        assert actual_thinking == 0
        assert actual_tool_use == 0

    def test_no_usage_metadata_falls_back_to_char_estimate(self):
        """When usage_metadata is absent, token count is estimated from char length."""
        response = SimpleNamespace(text="hello world")  # no usage_metadata attr
        actual_input, actual_output, actual_cache, actual_thinking, actual_tool_use = GeminiClient._parse_usage_metadata(
            response, "x" * 400  # 400 chars → 100 tokens estimate
        )
        assert actual_input == 100   # 400 // 4
        assert actual_output == 2    # len("hello world") // 4 = 2
        assert actual_cache == 0
        assert actual_thinking == 0
        assert actual_tool_use == 0

    def test_thinking_and_tool_use_tokens_are_split_out(self):
        """Gemini thoughts/tool tokens are tracked separately from visible output."""
        response = self._make_response(
            prompt=500,
            candidates=120,
            cached=100,
            thoughts=45,
            tool_use=30,
        )
        actual_input, actual_output, actual_cache, actual_thinking, actual_tool_use = (
            GeminiClient._parse_usage_metadata(response, "prompt text")
        )
        assert actual_input == 400
        assert actual_output == 120
        assert actual_cache == 100
        assert actual_thinking == 45
        assert actual_tool_use == 30
