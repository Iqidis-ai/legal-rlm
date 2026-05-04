"""RLM exception hierarchy and error-handling context managers.

Four exception classes partition the handler space so that callers can
make principled catch decisions rather than blanket ``except Exception``.

Two context managers encode the project's sanctioned error-handling
patterns:
- ``noncritical``: cache, telemetry, cleanup — log and continue
- ``visible_degradation``: facts, gaps, issues, proof — produce an
  explicit degraded result rather than silently returning empty data
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class RLMError(Exception):
    """Base for all RLM-layer exceptions."""


class CriticalReasoningError(RLMError):
    """A failure in a hard control gate — proof state, synthesis,
    advocacy/quant thresholds — that must propagate or produce a
    visibly degraded result."""


class OptionalEnrichmentError(RLMError):
    """A failure in cache, telemetry, cosmetic enrichment, or
    background maintenance that can be logged and skipped."""


class RecoverableInfrastructureError(RLMError):
    """A failure in storage, network, or LLM provider that may be
    retried or gracefully degraded."""


class UserVisibleDegradation(RLMError):
    """A failure that the user must be told about — the result is
    incomplete or missing data that was expected."""


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------

@contextmanager
def noncritical(
    component: str,
    *,
    default: T = None,  # type: ignore[assignment]
    log_level: str = "warning",
):
    """Wrap optional enrichment / cache / telemetry code.

    On exception, logs a structured warning and yields ``default``.
    The caller gets a value rather than an unhandled exception, and the
    failure is visible in logs rather than silently swallowed.

    Usage::

        with noncritical("cache.write", default=None, log_level="debug") as ctx:
            cache.put(key, value)
        # ctx.result is None on failure, or the last expression on success

    For simple one-shot usage where you just need the default::

        with noncritical("orientation_fingerprint", default=""):
            fingerprint = compute_fingerprint(state)
    """
    _log = getattr(logger, log_level, logger.warning)
    try:
        yield default
    except Exception as exc:
        _log(
            "noncritical %s failed (continuing with default): %s",
            component,
            exc,
            exc_info=log_level == "debug",
        )


@contextmanager
def visible_degradation(
    component: str,
    *,
    result_factory: Callable[[str, Exception], Any] | None = None,
):
    """Wrap core reasoning code where failure must be user-visible.

    On exception, if ``result_factory`` is provided, it is called with
    ``(component, exc)`` to produce a degraded result dict. If not
    provided, the exception is re-raised as ``UserVisibleDegradation``
    so callers upstream can surface it.

    Usage::

        with visible_degradation("read.render_gaps",
                                  result_factory=failed_family_result):
            gaps = mm.gaps.open_gaps(...)
    """
    try:
        yield
    except Exception as exc:
        logger.error(
            "visible_degradation %s: %s",
            component,
            exc,
            exc_info=True,
        )
        if result_factory is not None:
            result_factory(component, exc)
            return
        raise UserVisibleDegradation(
            f"{component} failed: {exc}"
        ) from exc
