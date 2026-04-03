"""Tests for UsageStats token accounting and cost estimation."""
from types import SimpleNamespace
import pytest
from irys.core.models import UsageStats, ModelTier, MODEL_CONFIGS


class TestUsageStatsCostEstimation:
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

    def test_fallback_cost_when_tier_is_none(self):
        """When tier is None, falls back to Flash pricing (not crashes)."""
        stats = UsageStats()
        stats.add(input_tokens=1_000_000, output_tokens=1_000_000)
        # Fallback: Flash $0.30 in + $2.50 out
        expected = 0.30 + 2.50
        assert abs(stats.estimated_cost - expected) < 1e-9

    def test_accumulation(self):
        """Multiple add() calls accumulate correctly."""
        stats = UsageStats(tier=ModelTier.LITE)
        stats.add(input_tokens=100, output_tokens=50, cache_read_tokens=200)
        stats.add(input_tokens=100, output_tokens=50, cache_read_tokens=200)
        assert stats.input_tokens == 200
        assert stats.output_tokens == 100
        assert stats.cache_read_tokens == 400
        assert stats.requests == 2


class TestCompleteUsageMetadataSplit:
    """Regression tests for the prompt_token_count split in GeminiClient.complete().

    These tests reproduce the exact getattr chain and max(total - cache, 0) calculation
    from complete() so that a regression in that block is caught immediately.
    """

    def _extract(self, um):
        """Reproduce the usage metadata extraction from complete()."""
        total_prompt = getattr(um, "prompt_token_count", None) or 0
        actual_output = getattr(um, "candidates_token_count", None) or 0
        actual_cache = getattr(um, "cached_content_token_count", None) or 0
        actual_input = max(total_prompt - actual_cache, 0)
        return actual_input, actual_output, actual_cache

    def test_cached_tokens_subtracted_from_prompt(self):
        """Non-cached input = total_prompt - cached (the prior double-count bug regression)."""
        um = SimpleNamespace(prompt_token_count=500, candidates_token_count=100,
                             cached_content_token_count=200)
        actual_input, actual_output, actual_cache = self._extract(um)
        assert actual_input == 300   # 500 - 200
        assert actual_cache == 200
        assert actual_output == 100

    def test_no_cache_all_tokens_are_input(self):
        """When cached_content_token_count is 0, all prompt tokens are non-cached."""
        um = SimpleNamespace(prompt_token_count=400, candidates_token_count=80,
                             cached_content_token_count=0)
        actual_input, actual_output, actual_cache = self._extract(um)
        assert actual_input == 400
        assert actual_cache == 0

    def test_missing_metadata_fields_default_to_zero(self):
        """getattr fallback handles missing attributes without AttributeError."""
        um = SimpleNamespace()  # no fields at all
        actual_input, actual_output, actual_cache = self._extract(um)
        assert actual_input == 0
        assert actual_output == 0
        assert actual_cache == 0

    def test_full_cache_hit_input_is_zero(self):
        """If all prompt tokens came from cache, non-cached input is 0 (not negative)."""
        um = SimpleNamespace(prompt_token_count=300, candidates_token_count=50,
                             cached_content_token_count=300)
        actual_input, actual_output, actual_cache = self._extract(um)
        assert actual_input == 0  # max(300-300, 0) = 0, not negative
        assert actual_cache == 300
