"""Tests for audio chunking in media_pipeline.py (Phase 2)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from pydub import AudioSegment
from pydub.generators import Sine

from irys.core.media_pipeline import ChunkRecord, process_audio


# =============================================================================
# Test fixtures - Generate synthetic audio files
# =============================================================================

@pytest.fixture
def audio_90s(tmp_path: Path) -> Path:
    """Create a synthetic 90-second audio file for testing."""
    # Generate 90s of 440Hz sine wave
    audio = Sine(440).to_audio_segment(duration=90000)  # milliseconds
    filepath = tmp_path / "test_90s.mp3"
    audio.export(str(filepath), format="mp3")
    return filepath


@pytest.fixture
def audio_15s(tmp_path: Path) -> Path:
    """Create a synthetic 15-second audio file (short file test)."""
    audio = Sine(440).to_audio_segment(duration=15000)
    filepath = tmp_path / "test_15s.mp3"
    audio.export(str(filepath), format="mp3")
    return filepath


@pytest.fixture
def audio_60s(tmp_path: Path) -> Path:
    """Create a synthetic 60-second audio file (exact 2x chunk size)."""
    audio = Sine(440).to_audio_segment(duration=60000)
    filepath = tmp_path / "test_60s.mp3"
    audio.export(str(filepath), format="mp3")
    return filepath


# =============================================================================
# Test audio chunking with 30s segments and 5s overlap
# =============================================================================

@pytest.mark.asyncio
async def test_audio_chunking_90s_file_returns_3_chunks(audio_90s: Path):
    """Test that a 90-second file produces 3 chunks with correct timestamps.

    Expected chunks with 30s duration and 5s overlap:
    - Chunk 0: 0-30s
    - Chunk 1: 25-55s (starts 5s before end of chunk 0)
    - Chunk 2: 50-80s (starts 5s before end of chunk 1)

    Note: Final chunk may be shorter if it extends past file duration.
    """
    chunks = await process_audio(audio_90s, chunk_duration_s=30, overlap_s=5)

    assert len(chunks) == 3, "90s file should produce 3 chunks (30s each with 5s overlap)"

    # Chunk 0: 0-30s
    assert chunks[0].chunk_index == 0
    assert chunks[0].start_time_s == 0.0
    assert chunks[0].end_time_s == 30.0

    # Chunk 1: 25-55s
    assert chunks[1].chunk_index == 1
    assert chunks[1].start_time_s == 25.0
    assert chunks[1].end_time_s == 55.0

    # Chunk 2: 50-80s
    assert chunks[2].chunk_index == 2
    assert chunks[2].start_time_s == 50.0
    assert chunks[2].end_time_s == 80.0


@pytest.mark.asyncio
async def test_audio_chunking_short_file_returns_single_chunk(audio_15s: Path):
    """Test that a 15-second file produces 1 chunk (no splitting needed)."""
    chunks = await process_audio(audio_15s, chunk_duration_s=30, overlap_s=5)

    assert len(chunks) == 1, "15s file should produce 1 chunk (shorter than chunk duration)"

    # Single chunk: 0-15s
    assert chunks[0].chunk_index == 0
    assert chunks[0].start_time_s == 0.0
    assert chunks[0].end_time_s == 15.0


@pytest.mark.asyncio
async def test_audio_chunking_exact_multiple_returns_correct_chunks(audio_60s: Path):
    """Test that a 60-second file (exact 2x chunk size) produces correct chunks."""
    chunks = await process_audio(audio_60s, chunk_duration_s=30, overlap_s=5)

    # 60s with 30s chunks and 5s overlap should produce 2 chunks
    assert len(chunks) == 2

    # Chunk 0: 0-30s
    assert chunks[0].start_time_s == 0.0
    assert chunks[0].end_time_s == 30.0

    # Chunk 1: 25-55s (but clamped to 60s max)
    assert chunks[1].start_time_s == 25.0
    assert chunks[1].end_time_s == 55.0


# =============================================================================
# Test ChunkRecord field population
# =============================================================================

@pytest.mark.asyncio
async def test_audio_chunk_record_has_correct_asset_type(audio_90s: Path):
    """Verify that audio chunks have asset_type='audio'."""
    chunks = await process_audio(audio_90s)

    for chunk in chunks:
        assert chunk.asset_type == "audio", "All audio chunks must have asset_type='audio'"


@pytest.mark.asyncio
async def test_audio_chunk_record_has_timestamps(audio_90s: Path):
    """Verify that audio chunks have start_time_s and end_time_s populated."""
    chunks = await process_audio(audio_90s)

    for chunk in chunks:
        assert chunk.start_time_s is not None, "start_time_s must be populated"
        assert chunk.end_time_s is not None, "end_time_s must be populated"
        assert chunk.start_time_s >= 0.0, "start_time_s must be non-negative"
        assert chunk.end_time_s > chunk.start_time_s, "end_time_s must be > start_time_s"


@pytest.mark.asyncio
async def test_audio_chunk_record_has_correct_asset_path(audio_90s: Path):
    """Verify that chunks store the original file path (not temp segment files)."""
    chunks = await process_audio(audio_90s)

    for chunk in chunks:
        assert chunk.asset_path == str(audio_90s), "asset_path must be original file path"


@pytest.mark.asyncio
async def test_audio_chunk_record_text_content_empty_at_index_time(audio_90s: Path):
    """Verify that text_content is empty at indexing time (filled at retrieval)."""
    chunks = await process_audio(audio_90s)

    for chunk in chunks:
        assert chunk.text_content == "", "text_content should be empty at index time"


@pytest.mark.asyncio
async def test_audio_chunk_record_has_unique_chunk_ids(audio_90s: Path):
    """Verify that each chunk has a unique chunk_id."""
    chunks = await process_audio(audio_90s)

    chunk_ids = [chunk.chunk_id for chunk in chunks]
    assert len(chunk_ids) == len(set(chunk_ids)), "All chunk_ids must be unique"


@pytest.mark.asyncio
async def test_audio_chunk_record_page_number_is_none(audio_90s: Path):
    """Verify that audio chunks have page_number=None (only for PDF/DOCX)."""
    chunks = await process_audio(audio_90s)

    for chunk in chunks:
        assert chunk.page_number is None, "Audio chunks should not have page numbers"


@pytest.mark.asyncio
async def test_audio_chunk_record_start_char_is_none(audio_90s: Path):
    """Verify that audio chunks have start_char=None (only for text chunks)."""
    chunks = await process_audio(audio_90s)

    for chunk in chunks:
        assert chunk.start_char is None, "Audio chunks should not have start_char"
        assert chunk.end_char is None, "Audio chunks should not have end_char"


# =============================================================================
# Test edge cases and error handling
# =============================================================================

@pytest.mark.asyncio
async def test_audio_chunking_nonexistent_file_raises_error():
    """Test that processing a nonexistent file raises FileNotFoundError."""
    nonexistent = Path("/nonexistent/file.mp3")

    with pytest.raises(FileNotFoundError):
        await process_audio(nonexistent)


@pytest.mark.asyncio
async def test_audio_chunking_invalid_chunk_duration_raises_error(audio_90s: Path):
    """Test that invalid chunk duration (<=0) raises ValueError."""
    with pytest.raises(ValueError, match="chunk_duration_s must be positive"):
        await process_audio(audio_90s, chunk_duration_s=0)

    with pytest.raises(ValueError, match="chunk_duration_s must be positive"):
        await process_audio(audio_90s, chunk_duration_s=-10)


@pytest.mark.asyncio
async def test_audio_chunking_invalid_overlap_raises_error(audio_90s: Path):
    """Test that invalid overlap (negative or >= chunk_duration) raises ValueError."""
    with pytest.raises(ValueError, match="overlap_s must be non-negative"):
        await process_audio(audio_90s, chunk_duration_s=30, overlap_s=-5)

    with pytest.raises(ValueError, match="overlap_s must be less than chunk_duration_s"):
        await process_audio(audio_90s, chunk_duration_s=30, overlap_s=30)

    with pytest.raises(ValueError, match="overlap_s must be less than chunk_duration_s"):
        await process_audio(audio_90s, chunk_duration_s=30, overlap_s=35)


# =============================================================================
# Test metadata population
# =============================================================================

@pytest.mark.asyncio
async def test_audio_chunk_metadata_contains_duration(audio_90s: Path):
    """Verify that chunk metadata includes the audio file duration."""
    chunks = await process_audio(audio_90s)

    for chunk in chunks:
        assert "file_duration_s" in chunk.metadata, "Metadata should include file duration"
        assert chunk.metadata["file_duration_s"] == pytest.approx(90.0, abs=0.1)


@pytest.mark.asyncio
async def test_audio_chunk_metadata_contains_format(audio_90s: Path):
    """Verify that chunk metadata includes the audio format."""
    chunks = await process_audio(audio_90s)

    for chunk in chunks:
        assert "format" in chunk.metadata, "Metadata should include audio format"
        assert chunk.metadata["format"] == "mp3"
