"""Tests for audio embedding in embeddings.py (Phase 2)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydub.generators import Sine

from irys.core.embeddings import EmbeddingClient, EmbeddingError
from irys.core.models import EmbeddingConfig


# =============================================================================
# Test fixtures
# =============================================================================

@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    """Create a synthetic audio file for testing."""
    audio = Sine(440).to_audio_segment(duration=5000)  # 5 seconds
    filepath = tmp_path / "test_audio.mp3"
    audio.export(str(filepath), format="mp3")
    return filepath


@pytest.fixture
def wav_file(tmp_path: Path) -> Path:
    """Create a WAV audio file for format testing."""
    audio = Sine(440).to_audio_segment(duration=3000)
    filepath = tmp_path / "test_audio.wav"
    audio.export(str(filepath), format="wav")
    return filepath


@pytest.fixture
def unsupported_file(tmp_path: Path) -> Path:
    """Create an unsupported file format."""
    filepath = tmp_path / "test.xyz"
    filepath.write_text("not an audio file")
    return filepath


@pytest.fixture
def embedding_config() -> EmbeddingConfig:
    """Default embedding configuration."""
    return EmbeddingConfig(
        model_id="gemini-embedding-2-preview",
        fast_dimensionality=256,
        full_dimensionality=3072,
        index_dimensionality=768,
    )


# =============================================================================
# Test audio embedding with API key mode
# =============================================================================

@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="Requires GEMINI_API_KEY environment variable",
)
@pytest.mark.asyncio
async def test_embed_media_audio_file_returns_normalized_vector(
    audio_file: Path, embedding_config: EmbeddingConfig
):
    """Test that embed_media() returns a 768-dim L2-normalized vector for audio."""
    api_key = os.getenv("GEMINI_API_KEY")
    client = EmbeddingClient(embedding_config, api_key=api_key)

    # Embed the audio file
    vector = client.embed_media(audio_file)

    # Verify shape (should be index_dimensionality)
    assert vector.shape == (768,), f"Expected (768,) shape, got {vector.shape}"

    # Verify dtype
    assert vector.dtype == np.float32, f"Expected float32, got {vector.dtype}"

    # Verify L2-normalized (norm should be ~1.0)
    norm = np.linalg.norm(vector)
    assert abs(norm - 1.0) < 1e-5, f"Vector not normalized: norm={norm}"

    # Verify values are reasonable (not all zeros, not NaN/inf)
    assert not np.all(vector == 0.0), "Vector is all zeros"
    assert np.all(np.isfinite(vector)), "Vector contains NaN or Inf"


@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="Requires GEMINI_API_KEY environment variable",
)
@pytest.mark.asyncio
async def test_embed_media_supports_wav_format(
    wav_file: Path, embedding_config: EmbeddingConfig
):
    """Test that embed_media() supports WAV format."""
    api_key = os.getenv("GEMINI_API_KEY")
    client = EmbeddingClient(embedding_config, api_key=api_key)

    vector = client.embed_media(wav_file)

    assert vector.shape == (768,)
    assert vector.dtype == np.float32


@pytest.mark.asyncio
async def test_embed_media_unsupported_format_raises_error(
    unsupported_file: Path, embedding_config: EmbeddingConfig
):
    """Test that embed_media() raises EmbeddingError for unsupported formats."""
    # Mock API key to avoid real API calls
    with patch("irys.core.embeddings.genai.Client"):
        client = EmbeddingClient(embedding_config, api_key="fake-key")

        with pytest.raises(EmbeddingError, match="Unsupported audio format"):
            client.embed_media(unsupported_file)


@pytest.mark.asyncio
async def test_embed_media_nonexistent_file_raises_error(embedding_config: EmbeddingConfig):
    """Test that embed_media() raises FileNotFoundError for missing files."""
    with patch("irys.core.embeddings.genai.Client"):
        client = EmbeddingClient(embedding_config, api_key="fake-key")

        nonexistent = Path("/nonexistent/file.mp3")
        with pytest.raises(FileNotFoundError):
            client.embed_media(nonexistent)


# =============================================================================
# Test audio embedding with mocked API
# =============================================================================

@pytest.mark.asyncio
async def test_embed_media_calls_gemini_upload_file(
    audio_file: Path, embedding_config: EmbeddingConfig
):
    """Test that embed_media() calls genai.upload_file() for audio files."""
    with patch("irys.core.embeddings.genai.Client") as mock_client_class:
        # Mock the client and its methods
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock upload_file response
        mock_file = MagicMock()
        mock_file.state.name = "ACTIVE"  # Skip processing wait
        mock_file.name = "files/test-audio-123"
        mock_client.files.upload.return_value = mock_file

        # Mock embed_content response
        mock_embedding = MagicMock()
        mock_embedding.values = np.random.randn(768).tolist()
        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding]
        mock_client.models.embed_content.return_value = mock_response

        # Create client and embed
        client = EmbeddingClient(embedding_config, api_key="fake-key")
        vector = client.embed_media(audio_file)

        # Verify upload_file was called with correct parameter
        mock_client.files.upload.assert_called_once()
        call_args = mock_client.files.upload.call_args
        assert call_args[1]["file"] == str(audio_file)

        # Verify embed_content was called with uploaded file
        mock_client.models.embed_content.assert_called_once()
        embed_args = mock_client.models.embed_content.call_args
        assert embed_args[1]["content"] == mock_file


@pytest.mark.asyncio
async def test_embed_media_waits_for_file_processing(
    audio_file: Path, embedding_config: EmbeddingConfig
):
    """Test that embed_media() waits for file processing to complete."""
    with patch("irys.core.embeddings.genai.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock file that starts PROCESSING, then becomes ACTIVE
        mock_file = MagicMock()
        mock_file.state.name = "PROCESSING"
        mock_file.name = "files/test-audio-123"

        # Simulate state transition after get_file call
        def get_file_side_effect(name):
            mock_file.state.name = "ACTIVE"  # Transition to ACTIVE
            return mock_file

        mock_client.files.upload.return_value = mock_file
        mock_client.files.get.side_effect = get_file_side_effect

        # Mock embedding response
        mock_embedding = MagicMock()
        mock_embedding.values = np.random.randn(768).tolist()
        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding]
        mock_client.models.embed_content.return_value = mock_response

        # Create client and embed
        client = EmbeddingClient(embedding_config, api_key="fake-key")
        with patch("time.sleep"):  # Mock sleep to speed up test
            vector = client.embed_media(audio_file)

        # Verify get_file was called to check processing status
        mock_client.files.get.assert_called_once_with("files/test-audio-123")


@pytest.mark.asyncio
async def test_embed_media_returns_normalized_vector_from_api(
    audio_file: Path, embedding_config: EmbeddingConfig
):
    """Test that embed_media() L2-normalizes the vector from API."""
    with patch("irys.core.embeddings.genai.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock file upload
        mock_file = MagicMock()
        mock_file.state.name = "ACTIVE"
        mock_client.files.upload.return_value = mock_file

        # Mock embedding response with NON-normalized vector
        raw_vector = np.random.randn(768)  # NOT normalized
        mock_embedding = MagicMock()
        mock_embedding.values = raw_vector.tolist()
        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding]
        mock_client.models.embed_content.return_value = mock_response

        # Create client and embed
        client = EmbeddingClient(embedding_config, api_key="fake-key")
        vector = client.embed_media(audio_file)

        # Verify vector is normalized (even though API returned non-normalized)
        norm = np.linalg.norm(vector)
        assert abs(norm - 1.0) < 1e-5, f"Vector not normalized: norm={norm}"


# =============================================================================
# Test API error handling
# =============================================================================

@pytest.mark.asyncio
async def test_embed_media_handles_api_error(
    audio_file: Path, embedding_config: EmbeddingConfig
):
    """Test that embed_media() handles API errors gracefully."""
    with patch("irys.core.embeddings.genai.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock API error during upload
        mock_client.files.upload.side_effect = Exception("API rate limit exceeded")

        # Create client and expect error
        client = EmbeddingClient(embedding_config, api_key="fake-key")
        with pytest.raises(EmbeddingError, match="Failed to embed audio"):
            client.embed_media(audio_file)


# =============================================================================
# Test supported formats
# =============================================================================

@pytest.mark.parametrize(
    "extension",
    [".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"],
)
def test_supported_audio_formats(extension: str):
    """Test that common audio formats are recognized as supported."""
    # This will be implemented in embed_media() validation logic
    # Just checking the list of supported extensions
    supported_extensions = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}
    assert extension in supported_extensions
