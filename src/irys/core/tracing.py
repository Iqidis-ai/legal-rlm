"""Observability tracing layer for LLM calls.

Provides a swappable tracing backend (Langfuse, or NoOp when disabled).
All Langfuse-specific code is isolated here — the rest of the codebase
only interacts with TracingContext and TracingProvider.

Enable by setting LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY env vars.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class SpanHandle:
    """Opaque handle to a span/trace returned by the provider."""

    def __init__(self, raw: Any = None):
        self._raw = raw


class TracingProvider(ABC):
    """Abstract tracing backend. Swap Langfuse for anything."""

    @abstractmethod
    def start_trace(self, *, trace_id: str, name: str, metadata: Optional[dict] = None,
                    user_id: Optional[str] = None, session_id: Optional[str] = None,
                    tags: Optional[list[str]] = None) -> SpanHandle: ...

    @abstractmethod
    def start_span(self, parent: SpanHandle, *, name: str,
                   metadata: Optional[dict] = None) -> SpanHandle: ...

    @abstractmethod
    def record_generation(self, parent: SpanHandle, *, name: str, model: str,
                          input: Any, output: Any, usage: Optional[dict] = None,
                          metadata: Optional[dict] = None) -> None: ...

    @abstractmethod
    def end_span(self, span: SpanHandle, *, metadata: Optional[dict] = None,
                 status: str = "ok") -> None: ...

    @abstractmethod
    def end_trace(self, trace: SpanHandle, *, metadata: Optional[dict] = None,
                  status: str = "ok") -> None: ...

    @abstractmethod
    def flush(self) -> None:
        """Flush pending events (important for serverless / shutdown)."""
        ...


# ---------------------------------------------------------------------------
# NoOp implementation (zero overhead)
# ---------------------------------------------------------------------------

_NOOP_HANDLE = SpanHandle(None)


class NoOpProvider(TracingProvider):
    """No-op provider when tracing is disabled."""

    def start_trace(self, **kwargs) -> SpanHandle:
        return _NOOP_HANDLE

    def start_span(self, parent, **kwargs) -> SpanHandle:
        return _NOOP_HANDLE

    def record_generation(self, parent, **kwargs) -> None:
        pass

    def end_span(self, span, **kwargs) -> None:
        pass

    def end_trace(self, trace, **kwargs) -> None:
        pass

    def flush(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Langfuse implementation
# ---------------------------------------------------------------------------


class LangfuseProvider(TracingProvider):
    """Langfuse-backed tracing provider (v4 SDK / OTEL-based).

    Args:
        public_key: Langfuse public key (or LANGFUSE_PUBLIC_KEY env var)
        secret_key: Langfuse secret key (or LANGFUSE_SECRET_KEY env var)
        host: Langfuse base URL (or LANGFUSE_HOST env var)
        default_tags: Tags automatically added to every trace (e.g. ["ar-service", "env:production"])
    """

    def __init__(self, public_key: Optional[str] = None,
                 secret_key: Optional[str] = None, host: Optional[str] = None,
                 default_tags: Optional[list[str]] = None):
        try:
            from langfuse import Langfuse, propagate_attributes
        except ImportError:
            raise ImportError(
                "langfuse package is required for LangfuseProvider. "
                "Install with: pip install langfuse"
            )
        self._client = Langfuse(
            public_key=public_key or os.environ.get("LANGFUSE_PUBLIC_KEY"),
            secret_key=secret_key or os.environ.get("LANGFUSE_SECRET_KEY"),
            base_url=host or os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        self._propagate_attributes = propagate_attributes
        self._default_tags: list[str] = default_tags or []
        logger.info("Langfuse tracing provider initialized (tags=%s)", self._default_tags)

    def start_trace(self, *, trace_id: str, name: str, metadata: Optional[dict] = None,
                    user_id: Optional[str] = None, session_id: Optional[str] = None,
                    tags: Optional[list[str]] = None) -> SpanHandle:
        # Merge default tags with per-trace tags
        all_tags = list(self._default_tags)
        if tags:
            all_tags.extend(t for t in tags if t not in all_tags)

        try:
            # Create a deterministic trace ID from our investigation ID
            lf_trace_id = self._client.create_trace_id(seed=trace_id)

            # Create the root observation as the "current" span (sets OTEL context)
            root_cm = self._client.start_as_current_observation(
                name=name,
                trace_context={"trace_id": lf_trace_id},
                metadata=metadata,
            )
            root_span = root_cm.__enter__()

            # Propagate tags + user/session to all children via OTEL context
            attrs_cm = self._propagate_attributes(
                tags=all_tags or None,
                user_id=user_id,
                session_id=session_id,
            )
            attrs_cm.__enter__()

            handle = SpanHandle(root_span)
            handle._context_managers = [root_cm, attrs_cm]
            return handle

        except Exception as e:
            logger.warning(f"Failed to start Langfuse trace: {e}")
            return _NOOP_HANDLE

    def start_span(self, parent: SpanHandle, *, name: str,
                   metadata: Optional[dict] = None) -> SpanHandle:
        raw = parent._raw
        if raw is None:
            return _NOOP_HANDLE
        try:
            child = raw.start_observation(name=name, metadata=metadata)
            return SpanHandle(child)
        except Exception as e:
            logger.warning(f"Failed to start Langfuse span '{name}': {e}")
            return _NOOP_HANDLE

    def record_generation(self, parent: SpanHandle, *, name: str, model: str,
                          input: Any, output: Any, usage: Optional[dict] = None,
                          metadata: Optional[dict] = None) -> None:
        raw = parent._raw
        if raw is None:
            return
        try:
            # Convert usage dict to Langfuse v4 format
            usage_details = None
            if usage:
                usage_details = {
                    "input": usage.get("promptTokens") or usage.get("input_tokens") or 0,
                    "output": usage.get("completionTokens") or usage.get("output_tokens") or 0,
                    "total": usage.get("totalTokens") or usage.get("total_tokens") or 0,
                }

            gen = raw.start_observation(
                name=name, as_type="generation", model=model,
                input=input, output=output,
                usage_details=usage_details, metadata=metadata,
            )
            gen.end()
        except Exception as e:
            logger.warning(f"Failed to record Langfuse generation '{name}': {e}")

    def end_span(self, span: SpanHandle, *, metadata: Optional[dict] = None,
                 status: str = "ok") -> None:
        raw = span._raw
        if raw is None:
            return
        try:
            kw: dict[str, Any] = {}
            if metadata:
                kw["metadata"] = metadata
            if status == "error":
                kw["level"] = "ERROR"
            if kw:
                raw.update(**kw)
            raw.end()
        except Exception as e:
            logger.warning(f"Failed to end Langfuse span: {e}")

    def end_trace(self, trace: SpanHandle, *, metadata: Optional[dict] = None,
                  status: str = "ok") -> None:
        raw = trace._raw
        if raw is None:
            return
        try:
            kw: dict[str, Any] = {}
            if metadata:
                kw["metadata"] = metadata
            if status == "error":
                kw["level"] = "ERROR"
            if kw:
                raw.update(**kw)
            raw.end()

            # Exit context managers in reverse order (attrs_cm, then root_cm)
            for cm in reversed(getattr(trace, '_context_managers', [])):
                try:
                    cm.__exit__(None, None, None)
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Failed to end Langfuse trace: {e}")

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception as e:
            logger.warning(f"Failed to flush Langfuse: {e}")


# ---------------------------------------------------------------------------
# TracingContext — passed through the call chain
# ---------------------------------------------------------------------------


class TracingContext:
    """Lightweight context passed through the call chain.

    Wraps a provider + current span so callers don't need to know
    about Langfuse (or any other backend) directly.
    """

    def __init__(self, provider: TracingProvider, current_span: SpanHandle):
        self._provider = provider
        self._current = current_span

    @property
    def span_handle(self) -> SpanHandle:
        return self._current

    def span(self, name: str, **metadata) -> "TracingContextSpan":
        """Create a child span. Use as a context manager.

        Example::

            async with trace_ctx.span("planning") as child_ctx:
                await decisions.create_plan(..., trace_ctx=child_ctx)
        """
        return TracingContextSpan(self._provider, self._current, name, metadata or None)

    def record_generation(
        self,
        *,
        name: str,
        model: str,
        input: Any,
        output: Any,
        usage: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Record an LLM generation on the current span."""
        self._provider.record_generation(
            self._current, name=name, model=model, input=input,
            output=output, usage=usage, metadata=metadata,
        )

    def child_context(self, name: str, metadata: Optional[dict] = None) -> "TracingContext":
        """Create a child TracingContext with a new span (non-context-manager)."""
        child_span = self._provider.start_span(self._current, name=name, metadata=metadata)
        return TracingContext(self._provider, child_span)

    def end(self, metadata: Optional[dict] = None, status: str = "ok") -> None:
        """End the current span."""
        self._provider.end_span(self._current, metadata=metadata, status=status)


class TracingContextSpan:
    """Context manager that creates a child span and yields a TracingContext."""

    def __init__(self, provider: TracingProvider, parent: SpanHandle,
                 name: str, metadata: Optional[dict]):
        self._provider = provider
        self._parent = parent
        self._name = name
        self._metadata = metadata
        self._span: Optional[SpanHandle] = None

    def __enter__(self) -> TracingContext:
        self._span = self._provider.start_span(
            self._parent, name=self._name, metadata=self._metadata,
        )
        return TracingContext(self._provider, self._span)

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "error" if exc_type else "ok"
        error_meta = {"error": str(exc_val)} if exc_val else None
        self._provider.end_span(self._span, metadata=error_meta, status=status)
        return False

    async def __aenter__(self) -> TracingContext:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.__exit__(exc_type, exc_val, exc_tb)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_tracing_provider(
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    host: Optional[str] = None,
    default_tags: Optional[list[str]] = None,
) -> TracingProvider:
    """Create a tracing provider based on environment configuration.

    Returns LangfuseProvider if keys are available, otherwise NoOpProvider.

    Args:
        default_tags: Tags automatically added to every trace
                      (e.g. ["ar-service", "env:production"])
    """
    pk = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
    sk = secret_key or os.environ.get("LANGFUSE_SECRET_KEY")

    if pk and sk:
        try:
            return LangfuseProvider(public_key=pk, secret_key=sk, host=host,
                                   default_tags=default_tags)
        except ImportError:
            logger.warning("langfuse not installed, falling back to NoOpProvider")
            return NoOpProvider()
        except Exception as e:
            logger.warning(f"Failed to initialize Langfuse: {e}, falling back to NoOpProvider")
            return NoOpProvider()

    return NoOpProvider()
