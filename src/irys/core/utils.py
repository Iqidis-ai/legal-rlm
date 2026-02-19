"""Utility functions for the Irys system.

Covers: Error Recovery, Logging, Retry Logic, Batch Processing, Text Utilities.
"""

import asyncio
import logging
import functools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar, Optional, Awaitable
from datetime import datetime
import re

T = TypeVar('T')

# Configure logging
logger = logging.getLogger("irys")


# =============================================================================
# Unit 32: Error Recovery
# =============================================================================

class RetryableError(Exception):
    """An error that can be retried."""
    pass


class FatalError(Exception):
    """An error that should not be retried."""
    pass


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    retryable_exceptions: tuple = (RetryableError, TimeoutError, ConnectionError)


def retry_async(config: Optional[RetryConfig] = None):
    """Decorator for retrying async functions with exponential backoff."""
    config = config or RetryConfig()

    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            delay = config.initial_delay

            for attempt in range(config.max_retries + 1):
                try:
                    return await fn(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e
                    if attempt < config.max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{config.max_retries + 1} failed for {fn.__name__}: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        await asyncio.sleep(delay)
                        delay = min(delay * config.exponential_base, config.max_delay)
                    else:
                        logger.error(f"All retries exhausted for {fn.__name__}: {e}")
                except FatalError:
                    raise
                except Exception as e:
                    logger.error(f"Non-retryable error in {fn.__name__}: {e}")
                    raise

            raise last_exception

        return wrapper
    return decorator


def retry_sync(config: Optional[RetryConfig] = None):
    """Decorator for retrying sync functions with exponential backoff."""
    config = config or RetryConfig()

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            delay = config.initial_delay

            for attempt in range(config.max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e
                    if attempt < config.max_retries:
                        logger.warning(f"Retry {attempt + 1}: {e}")
                        time.sleep(delay)
                        delay = min(delay * config.exponential_base, config.max_delay)
                except Exception:
                    raise

            raise last_exception

        return wrapper
    return decorator


# =============================================================================
# Unit 33: Logging & Telemetry
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
# Unit 34: Text Utilities
# =============================================================================

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove control characters
    text = ''.join(c for c in text if ord(c) >= 32 or c in '\n\t')
    return text.strip()


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to max length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def extract_sentences(text: str) -> list[str]:
    """Extract sentences from text."""
    # Simple sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def extract_numbers(text: str) -> list[str]:
    """Extract numbers (including currency) from text."""
    # Match various number formats
    patterns = [
        r'\$[\d,]+(?:\.\d{2})?',  # Currency
        r'[\d,]+(?:\.\d+)?',  # Regular numbers
    ]
    numbers = []
    for pattern in patterns:
        numbers.extend(re.findall(pattern, text))
    return numbers


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    return ' '.join(text.split())


# =============================================================================
# Unit 35: Batch Processing
# =============================================================================

async def batch_process(
    items: list[T],
    processor: Callable[[T], Awaitable[Any]],
    batch_size: int = 10,
    delay_between_batches: float = 0.1,
) -> list[Any]:
    """Process items in batches."""
    results = []

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await asyncio.gather(
            *[processor(item) for item in batch],
            return_exceptions=True,
        )
        results.extend(batch_results)

        if i + batch_size < len(items):
            await asyncio.sleep(delay_between_batches)

    return results


def chunk_list(items: list[T], chunk_size: int) -> list[list[T]]:
    """Split a list into chunks."""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


async def parallel_map(
    items: list[T],
    fn: Callable[[T], Awaitable[Any]],
    max_concurrent: int = 10,
) -> list[Any]:
    """Map function over items with concurrency limit."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_fn(item: T) -> Any:
        async with semaphore:
            return await fn(item)

    return await asyncio.gather(*[limited_fn(item) for item in items])


# =============================================================================
# Unit 36: Progress Tracking
# =============================================================================

@dataclass
class ProgressTracker:
    """Track progress of multi-step operations."""
    total: int
    current: int = 0
    description: str = ""
    started_at: datetime = field(default_factory=datetime.now)
    _callbacks: list = field(default_factory=list)

    def advance(self, n: int = 1, description: str = None):
        """Advance progress."""
        self.current = min(self.current + n, self.total)
        if description:
            self.description = description
        self._notify()

    def set(self, current: int, description: str = None):
        """Set progress to specific value."""
        self.current = min(current, self.total)
        if description:
            self.description = description
        self._notify()

    @property
    def percent(self) -> float:
        """Get progress percentage."""
        return (self.current / self.total * 100) if self.total > 0 else 0

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time."""
        return (datetime.now() - self.started_at).total_seconds()

    @property
    def eta_seconds(self) -> Optional[float]:
        """Estimate time remaining."""
        if self.current == 0:
            return None
        rate = self.current / self.elapsed_seconds
        remaining = self.total - self.current
        return remaining / rate if rate > 0 else None

    def add_callback(self, callback: Callable[["ProgressTracker"], None]):
        """Add progress callback."""
        self._callbacks.append(callback)

    def _notify(self):
        """Notify callbacks of progress."""
        for callback in self._callbacks:
            try:
                callback(self)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get current status."""
        return {
            "current": self.current,
            "total": self.total,
            "percent": round(self.percent, 1),
            "description": self.description,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "eta_seconds": round(self.eta_seconds, 1) if self.eta_seconds else None,
        }


# =============================================================================
# Unit 37: Configuration Management
# =============================================================================

@dataclass
class SystemConfig:
    """System-wide configuration."""
    # Model settings
    default_model_tier: str = "flash"
    max_tokens: int = 8000

    # Investigation settings
    max_depth: int = 5
    max_leads_per_level: int = 5
    min_lead_priority: float = 0.3

    # Performance settings
    parallel_reads: int = 5
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600

    # Output settings
    default_output_format: str = "markdown"
    include_citations: bool = True
    include_thinking_trace: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "SystemConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }


# =============================================================================
# Unit 38: Validation Utilities
# =============================================================================

def validate_query(query: str) -> tuple[bool, list[str]]:
    """Validate a query string."""
    issues = []

    if not query or not query.strip():
        issues.append("Query cannot be empty")
        return False, issues

    # if len(query) < 5:
    #     issues.append("Query is too short (minimum 5 characters)")

    # if len(query) > 2000:
    #     issues.append("Query is too long (maximum 2000 characters)")

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
# Unit 39: Date/Time Utilities
# =============================================================================

def parse_date_flexible(date_str: str) -> Optional[datetime]:
    """Parse dates in various formats."""
    formats = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%d/%m/%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%Y%m%d",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue

    return None


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


# =============================================================================
# Unit 40: String Similarity
# =============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein edit distance."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def similarity_ratio(s1: str, s2: str) -> float:
    """Calculate string similarity ratio (0-1)."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    distance = levenshtein_distance(s1.lower(), s2.lower())
    max_len = max(len(s1), len(s2))
    return 1 - (distance / max_len)


def jaccard_similarity(s1: str, s2: str) -> float:
    """Calculate Jaccard similarity of word sets."""
    words1 = set(s1.lower().split())
    words2 = set(s2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0.0
