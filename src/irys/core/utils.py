"""Utility functions for the Irys system.

Covers: Logging, Telemetry, Validation, String Similarity.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime

# Configure logging
logger = logging.getLogger("irys")


# =============================================================================
# Logging & Telemetry
# =============================================================================

@dataclass
class TelemetryEvent:
    """A telemetry event."""
    event_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: dict = field(default_factory=dict)
    duration_ms: Optional[int] = None


class TelemetryCollector:
    """Collect and export telemetry data."""

    def __init__(self):
        self._events: list[TelemetryEvent] = []
        self._start_times: dict[str, float] = {}

    def start_operation(self, operation_id: str):
        """Start timing an operation."""
        self._start_times[operation_id] = time.time()

    def end_operation(self, operation_id: str, event_type: str, data: dict = None):
        """End an operation and record the event."""
        start_time = self._start_times.pop(operation_id, None)
        duration_ms = None
        if start_time:
            duration_ms = int((time.time() - start_time) * 1000)

        self._events.append(TelemetryEvent(
            event_type=event_type,
            data=data or {},
            duration_ms=duration_ms,
        ))

    def record(self, event_type: str, data: dict = None):
        """Record a simple event."""
        self._events.append(TelemetryEvent(
            event_type=event_type,
            data=data or {},
        ))

    def get_events(self) -> list[TelemetryEvent]:
        """Get all events."""
        return self._events.copy()

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        by_type: dict[str, list[int]] = {}

        for event in self._events:
            if event.event_type not in by_type:
                by_type[event.event_type] = []
            if event.duration_ms:
                by_type[event.event_type].append(event.duration_ms)

        summary = {}
        for event_type, durations in by_type.items():
            if durations:
                summary[event_type] = {
                    "count": len(durations),
                    "avg_ms": sum(durations) // len(durations),
                    "min_ms": min(durations),
                    "max_ms": max(durations),
                }
            else:
                summary[event_type] = {"count": len([e for e in self._events if e.event_type == event_type])}

        return summary

    def clear(self):
        """Clear all events."""
        self._events.clear()
        self._start_times.clear()


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Configure logging for the system."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


# =============================================================================
# Validation Utilities
# =============================================================================

def validate_query(query: str) -> tuple[bool, list[str]]:
    """Validate a query string."""
    issues = []

    if not query or not query.strip():
        issues.append("Query cannot be empty")
        return False, issues

    if len(query) < 5:
        issues.append("Query is too short (minimum 5 characters)")

    if len(query) > 2000:
        issues.append("Query is too long (maximum 2000 characters)")

    return len(issues) == 0, issues


def validate_file_path(path: str) -> tuple[bool, list[str]]:
    """Validate a file path."""
    from pathlib import Path
    issues = []

    try:
        p = Path(path)
        if not p.exists():
            issues.append(f"Path does not exist: {path}")
        elif not p.is_file() and not p.is_dir():
            issues.append(f"Path is neither file nor directory: {path}")
    except Exception as e:
        issues.append(f"Invalid path: {e}")

    return len(issues) == 0, issues


# =============================================================================
# String Similarity
# =============================================================================

def jaccard_similarity(s1: str, s2: str) -> float:
    """Calculate Jaccard similarity of word sets."""
    words1 = set(s1.lower().split())
    words2 = set(s2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0.0
