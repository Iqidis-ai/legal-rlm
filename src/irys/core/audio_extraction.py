"""Audio extraction at retrieval time — Phase 2.

Hybrid approach: audio embeddings for retrieval, Gemini extraction at retrieval time.

This module provides:
- extract_audio_segment(): Extract audio segment from file using timestamps
- extract_audio_insights(): Pass audio to Gemini Flash for factual extraction
- AudioInsights: Structured extraction result dataclass

No transcription service required — Gemini handles audio understanding natively.
"""

from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Confidence mapping (standardized per Critical Safeguard #5)
# =============================================================================

CONFIDENCE_MAP = {
    "low": 0.3,
    "medium": 0.6,
    "high": 0.9,
}


# =============================================================================
# AudioInsights dataclass
# =============================================================================

@dataclass(frozen=True)
class AudioInsights:
    """Structured extraction result from audio chunk.

    Fields:
        summary: Concise factual summary (2-3 sentences) focused on query relevance.
        spoken_content: Key quotes or statements relevant to the query.
        entities: Named entities mentioned (people, orgs, dates, locations).
        confidence: Extraction confidence (0.0-1.0 float, NOT string).
        temporal_refs: Dates, times, or temporal markers mentioned.
        error: Optional error message if extraction failed.
    """

    summary: str
    spoken_content: str
    entities: list[str]
    confidence: float  # 0.0-1.0 (NOT "low/medium/high")
    temporal_refs: list[str]
    error: Optional[str] = None

    def is_low_confidence(self) -> bool:
        """Return True if confidence is below 0.5."""
        return self.confidence < 0.5


# =============================================================================
# Audio segment extraction
# =============================================================================

async def extract_audio_segment(
    file_path: Path,
    start_time_s: float,
    end_time_s: float,
) -> bytes:
    """Extract audio segment from file using timestamps.

    Uses pydub to load the full audio file and extract the specified segment.

    Args:
        file_path: Absolute path to the audio file.
        start_time_s: Start timestamp in seconds.
        end_time_s: End timestamp in seconds.

    Returns:
        Raw audio bytes of the extracted segment (MP3 format).

    Raises:
        FileNotFoundError: If the audio file does not exist.
        ValueError: If timestamps are invalid.
    """
    # Validate file exists
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # Validate timestamps
    if start_time_s < 0:
        raise ValueError(f"start_time_s must be non-negative, got {start_time_s}")
    if end_time_s <= start_time_s:
        raise ValueError(
            f"end_time_s ({end_time_s}) must be > start_time_s ({start_time_s})"
        )

    # Import pydub here to avoid dependency at module load
    from pydub import AudioSegment

    # Load full audio file
    logger.debug("Loading audio file for segment extraction: %s", file_path)
    audio = AudioSegment.from_file(str(file_path))

    # Convert timestamps to milliseconds (pydub uses ms)
    start_ms = int(start_time_s * 1000)
    end_ms = int(end_time_s * 1000)

    # Extract segment
    logger.debug("Extracting segment: %d-%d ms", start_ms, end_ms)
    segment = audio[start_ms:end_ms]

    # Export segment to bytes (MP3 format)
    buffer = io.BytesIO()
    segment.export(buffer, format="mp3")
    audio_bytes = buffer.getvalue()

    logger.info(
        "Extracted audio segment: %.1fs-%.1fs, %d bytes",
        start_time_s,
        end_time_s,
        len(audio_bytes),
    )
    return audio_bytes


# =============================================================================
# Audio insights extraction using Gemini Flash
# =============================================================================

async def extract_audio_insights(
    audio_bytes: bytes,
    gemini_client,  # GeminiClient from models.py
    query: str,
) -> AudioInsights:
    """Extract insights from audio using Gemini 2.5 Flash.

    Query-aware extraction: focuses on content relevant to the investigation query.
    Uses Gemini's native multimodal understanding (no transcription service).

    Args:
        audio_bytes: Raw audio bytes (MP3 format).
        gemini_client: GeminiClient instance (from models.py, FLASH tier).
        query: Investigation query for query-aware extraction.

    Returns:
        AudioInsights with summary, entities, confidence, and temporal references.

    Fallback behavior on failure:
        Returns AudioInsights with confidence=0.0, summary="[Audio extraction failed]",
        and error field populated with exception message.
    """
    from .models import ModelTier  # Import here to avoid circular dependency

    # Import Gemini types for file handling
    import time
    from google import genai

    try:
        # Upload audio bytes to Gemini for processing
        logger.debug("Uploading audio segment to Gemini for extraction")

        # Write bytes to temporary file for upload
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        try:
            # Upload and wait for processing
            uploaded_file = gemini_client._client.files.upload(file=tmp_path)

            # Wait for file processing
            while uploaded_file.state.name == "PROCESSING":
                time.sleep(1)
                uploaded_file = gemini_client._client.files.get(uploaded_file.name)

            if uploaded_file.state.name != "ACTIVE":
                raise RuntimeError(
                    f"File processing failed. State: {uploaded_file.state.name}"
                )

            # Build extraction prompt (import from prompts.py)
            from ..rlm.prompts import P_EXTRACT_AUDIO_INSIGHTS

            prompt = P_EXTRACT_AUDIO_INSIGHTS.format(query=query)

            # Call Gemini Flash with audio file and prompt
            logger.debug("Calling Gemini Flash for audio insight extraction")
            response = await gemini_client.complete(
                prompt,
                tier=ModelTier.FLASH,
                temperature=0.0,  # Factual extraction
                contents=[uploaded_file],  # Pass audio file
            )

            # Parse JSON response
            import json

            result = json.loads(response)

            # Map confidence string to float (Critical Safeguard #5)
            confidence_str = result.get("confidence", "medium")
            confidence_float = CONFIDENCE_MAP.get(confidence_str, 0.5)

            # Build AudioInsights
            insights = AudioInsights(
                summary=result.get("summary", ""),
                spoken_content=result.get("spoken_content", ""),
                entities=result.get("entities", []),
                confidence=confidence_float,
                temporal_refs=result.get("temporal_refs", []),
                error=None,
            )

            logger.info(
                "Audio insights extracted: confidence=%.2f, %d entities",
                insights.confidence,
                len(insights.entities),
            )
            return insights

        finally:
            # Clean up temporary file
            import os

            os.unlink(tmp_path)

    except Exception as exc:
        # Fallback behavior (Critical Safeguard #3)
        logger.error("Audio extraction failed: %s", exc)
        return AudioInsights(
            summary=f"[Audio extraction failed: {str(exc)}]",
            spoken_content="",
            entities=[],
            confidence=0.0,  # Zero confidence on failure
            temporal_refs=[],
            error=str(exc),
        )
