"""Multimodal content detection for PDF and DOCX documents.

Decides whether a document's extracted text is sufficient or whether
Mistral OCR should be called to re-extract from the raw bytes.

Ported from text-extract-multimodal-detection.ts (TypeScript reference).
All logic is pure Python — no I/O, easy to unit-test.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MultimodalDetectionConfig:
    """Tunable thresholds for multimodal detection scoring."""

    # Characters-per-page thresholds — match the TS defaults exactly
    extremely_high_confidence_text_threshold: int = 125   # < this → +0.55 confidence
    high_confidence_text_threshold: int = 200              # < this → +0.40 confidence
    medium_confidence_text_threshold: int = 400            # < this → +0.20 confidence

    # Ratio of "rendered" pages (pages with substantive text) to total pages
    min_text_extraction_rate: float = 0.7   # below → +0.30 confidence

    # Minimum confidence to flag as requiring OCR
    multimodal_threshold: float = 0.45


DEFAULT_CONFIG = MultimodalDetectionConfig()


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class MultimodalDetectionResult:
    """Output of detect_multimodal_content()."""

    requires_multimodal: bool
    confidence: float
    reasons: list[str]
    metrics: dict  # avg_text_per_page, text_extraction_rate, total_pages, etc.


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect_multimodal_content(
    text: str,
    page_count: int,
    rendered_pages: int = 0,
    pdf_info: dict | None = None,
    config: MultimodalDetectionConfig = DEFAULT_CONFIG,
) -> MultimodalDetectionResult:
    """Decide whether a PDF/DOCX document needs OCR.

    Args:
        text:           Full extracted text (all pages concatenated).
        page_count:     Total page count reported by the parser.
        rendered_pages: Number of pages that yielded substantive text
                        (pages with > 50 chars).  Pass 0 if unknown.
        pdf_info:       Optional dict of PDF metadata (Creator, Producer …).
        config:         Detection thresholds.

    Returns:
        MultimodalDetectionResult
    """
    try:
        metrics = _analyze_text_metrics(text, page_count, rendered_pages)
        suspicious = _detect_suspicious_patterns(text)
        indicators = _analyze_metadata(pdf_info or {})
        confidence = _calculate_confidence(metrics, suspicious, indicators, config)
        reasons = _generate_reasons(metrics, suspicious, indicators, confidence, config)

        result = MultimodalDetectionResult(
            requires_multimodal=confidence >= config.multimodal_threshold,
            confidence=confidence,
            reasons=reasons,
            metrics={**metrics, "suspicious_patterns": suspicious, "metadata_indicators": indicators},
        )

        logger.debug(
            "Multimodal detection: requires=%s confidence=%.2f avg_chars_per_page=%.0f",
            result.requires_multimodal,
            result.confidence,
            metrics["avg_text_per_page"],
        )
        return result

    except Exception as exc:  # noqa: BLE001
        logger.warning("Multimodal detection failed, defaulting to text-only: %s", exc)
        return MultimodalDetectionResult(
            requires_multimodal=False,
            confidence=0.0,
            reasons=["Detection failed — defaulting to text-only processing"],
            metrics={
                "avg_text_per_page": 0,
                "text_extraction_rate": 0,
                "total_pages": page_count,
                "total_text_length": len(text),
                "suspicious_patterns": [],
                "metadata_indicators": [],
            },
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _analyze_text_metrics(text: str, page_count: int, rendered_pages: int) -> dict:
    total_pages = max(page_count, 1)
    total_text_length = len(text)
    avg_text_per_page = total_text_length / total_pages
    # text_extraction_rate: fraction of pages that yielded text
    # If caller didn't supply rendered_pages, derive it from the average density
    if rendered_pages <= 0:
        # Estimate: pages with avg density > 50 chars are "rendered"
        text_extraction_rate = 1.0 if avg_text_per_page > 50 else 0.0
    else:
        text_extraction_rate = rendered_pages / total_pages
    return {
        "avg_text_per_page": avg_text_per_page,
        "text_extraction_rate": text_extraction_rate,
        "total_pages": total_pages,
        "total_text_length": total_text_length,
    }


def _detect_suspicious_patterns(text: str) -> list[str]:
    """Port of detectSuspiciousTextPatterns from TS."""
    patterns: list[str] = []
    if not text:
        return patterns

    # High frequency of non-standard characters (OCR artifacts)
    artifact_count = len(re.findall(r"[^\w\s.,!?;:()\-'\"]", text))
    if text and artifact_count > len(text) * 0.05:
        patterns.append("High frequency of OCR artifacts detected")

    # Excessive whitespace runs (common in scanned docs)
    if re.search(r"\s{5,}", text):
        patterns.append("Excessive whitespace patterns detected")

    # Fragmented text: single chars separated by spaces
    fragmented = re.findall(r"\b\w\s+\w\s+\w\b", text)
    if len(fragmented) > 10:
        patterns.append("Fragmented text patterns detected")

    # Mostly numeric content (charts / tables)
    no_ws = re.sub(r"\s", "", text)
    if no_ws:
        digit_ratio = len(re.findall(r"\d", no_ws)) / len(no_ws)
        if digit_ratio > 0.3:
            patterns.append("High numeric content ratio suggests charts/tables")

    # Short lines (visual layouts)
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if lines:
        short = [ln for ln in lines if len(ln.strip()) < 20]
        if len(short) / len(lines) >= 0.5:
            patterns.append("High ratio of short text lines detected")

        few_word = [ln for ln in lines if len(ln.strip().split()) <= 6]
        if len(few_word) / len(lines) >= 0.6:
            patterns.append("High ratio of lines with few words detected")

    return patterns


def _analyze_metadata(pdf_info: dict) -> list[str]:
    """Port of analyzeMetadata from TS — checks Creator/Producer for scanner keywords."""
    indicators: list[str] = []
    try:
        creator = str(pdf_info.get("creator") or pdf_info.get("Creator") or "").lower()
        producer = str(pdf_info.get("producer") or pdf_info.get("Producer") or "").lower()
        scanner_kw = {"scan", "scanner", "ocr", "image", "photo", "camera"}
        if any(kw in creator or kw in producer for kw in scanner_kw):
            indicators.append("Scanner/OCR software detected in metadata")

        title = str(pdf_info.get("title") or pdf_info.get("Title") or "").lower()
        if "scan" in title:
            indicators.append("Document title suggests scanned content")
    except Exception:
        pass
    return indicators


def _calculate_confidence(
    metrics: dict,
    suspicious: list[str],
    indicators: list[str],
    config: MultimodalDetectionConfig,
) -> float:
    """Confidence scoring — matches TS weights exactly."""
    confidence = 0.0
    avg = metrics["avg_text_per_page"]

    # Text density (up to 0.55)
    if avg < config.extremely_high_confidence_text_threshold:
        confidence += 0.55
    elif avg < config.high_confidence_text_threshold:
        confidence += 0.40
    elif avg < config.medium_confidence_text_threshold:
        confidence += 0.20

    # Text extraction rate (up to 0.30)
    rate = metrics["text_extraction_rate"]
    if rate < config.min_text_extraction_rate:
        confidence += 0.30
    elif rate < 0.9:
        confidence += 0.15

    # Suspicious patterns (up to 0.20, 0.05 each)
    confidence += min(len(suspicious) * 0.05, 0.20)

    # Metadata indicators (up to 0.10, 0.05 each)
    confidence += min(len(indicators) * 0.05, 0.10)

    return min(confidence, 1.0)


def _generate_reasons(
    metrics: dict,
    suspicious: list[str],
    indicators: list[str],
    confidence: float,
    config: MultimodalDetectionConfig,
) -> list[str]:
    reasons: list[str] = []
    if confidence >= 0.8:
        reasons.append("High confidence multimodal content detected")
    elif confidence >= 0.5:
        reasons.append("Medium confidence multimodal content detected")
    else:
        reasons.append("Low confidence — likely text-only document")

    avg = metrics["avg_text_per_page"]
    if avg < config.high_confidence_text_threshold:
        reasons.append(f"Very low text density: {avg:.1f} chars/page")
    elif avg < config.medium_confidence_text_threshold:
        reasons.append(f"Low text density: {avg:.1f} chars/page")

    rate = metrics["text_extraction_rate"]
    if rate < config.min_text_extraction_rate:
        reasons.append(f"Poor text extraction rate: {rate * 100:.1f}%")

    reasons.extend(suspicious)
    reasons.extend(indicators)
    return reasons
