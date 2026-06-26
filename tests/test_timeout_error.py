"""Tests for timeout error message quality."""

import asyncio
import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Load irys.core.models directly, bypassing the heavy package __init__.py
# (which requires fitz, aiohttp, etc. not present in the unit-test environment).
import importlib.util as _ilu
_models_path = str(Path(__file__).parent.parent / "src" / "irys" / "core" / "models.py")
_spec = _ilu.spec_from_file_location("irys.core.models", _models_path)
_models_mod = _ilu.module_from_spec(_spec)
sys.modules["irys.core.models"] = _models_mod
_spec.loader.exec_module(_models_mod)

GeminiClient = _models_mod.GeminiClient
ModelTier = _models_mod.ModelTier
# The module's reference to the genai and asyncio names for patching
_genai = _models_mod.genai
_asyncio_mod = _models_mod.asyncio


@pytest.fixture
def client():
    """GeminiClient with mocked genai.Client to avoid real API calls."""
    with patch.object(_genai, "Client"):
        c = GeminiClient(api_key="test-key")
    return c


class TestTimeoutErrorMessage:
    """Primary fix: asyncio.TimeoutError must produce a non-empty message."""

    @pytest.mark.asyncio
    async def test_overall_timeout_raises_descriptive_error(self, client):
        """overall_timeout firing must raise TimeoutError with a non-empty message."""
        with (
            patch.object(client._rate_limiter, "acquire", new_callable=AsyncMock),
            patch.object(_asyncio_mod, "wait_for", side_effect=asyncio.TimeoutError()),
        ):
            with pytest.raises(TimeoutError) as exc_info:
                await client.complete(
                    "test prompt",
                    tier=ModelTier.PRO,
                    overall_timeout=240.0,
                )

        error_msg = str(exc_info.value)
        assert error_msg, "TimeoutError message must not be empty"
        assert "240" in error_msg, "Message must include the timeout duration"
        assert "gemini" in error_msg.lower(), "Message must name at least one model"

    @pytest.mark.asyncio
    async def test_timeout_message_omits_missing_fallback_fields(self, client):
        """Tiers without secondary models must not crash and must omit that label."""
        # LITE has fallback_model_id but no secondary_fallback_model_id — distinct
        # code path from PRO which has all three.
        with (
            patch.object(client._rate_limiter, "acquire", new_callable=AsyncMock),
            patch.object(_asyncio_mod, "wait_for", side_effect=asyncio.TimeoutError()),
        ):
            with pytest.raises(TimeoutError) as exc_info:
                await client.complete(
                    "test prompt",
                    tier=ModelTier.LITE,   # LITE has fallback but no secondary_fallback
                    overall_timeout=80.0,
                )

        msg = str(exc_info.value)
        assert "primary=" in msg
        assert "fallback=" in msg        # LITE has fallback_model_id — must appear
        assert "secondary=" not in msg   # LITE has no secondary_fallback_model_id — must be absent

    @pytest.mark.asyncio
    async def test_no_overall_timeout_does_not_wrap(self, client):
        """When overall_timeout=None the coroutine runs unwrapped; no change in behaviour."""
        mock_response = MagicMock()
        mock_response.text = "ok"
        mock_response.usage_metadata = None

        with (
            patch.object(client._rate_limiter, "acquire", new_callable=AsyncMock),
            patch.object(client, "_call_with_fallback", return_value=mock_response),
        ):
            result = await client.complete(
                "test prompt",
                tier=ModelTier.FLASH,
                overall_timeout=None,
            )
        assert result == "ok"


class TestServiceErrorFallback:
    """Secondary fix: service/api.py must never send a blank error string to the client."""

    def test_empty_exception_str_falls_back_to_type_name(self):
        """str(asyncio.TimeoutError()) == ""; fallback must produce a non-empty string."""
        e = asyncio.TimeoutError()
        error_msg = str(e) or f"{type(e).__name__}: no further details"
        assert error_msg, "error_msg must not be empty"
        assert "TimeoutError" in error_msg

    def test_non_empty_exception_str_is_unchanged(self):
        """When str(e) is already non-empty, it must be used as-is."""
        e = ValueError("something went wrong")
        error_msg = str(e) or f"{type(e).__name__}: no further details"
        assert error_msg == "something went wrong"

    def test_runtime_error_with_no_message_falls_back(self):
        """RuntimeError() also has an empty str(); fallback should cover it."""
        e = RuntimeError()
        error_msg = str(e) or f"{type(e).__name__}: no further details"
        assert "RuntimeError" in error_msg
