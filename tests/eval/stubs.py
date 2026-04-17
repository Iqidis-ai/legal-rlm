"""Scripted GeminiClient for deterministic engine-stub tests.

Never calls a real network. Returns canned strings indexed by usage_label, or a
fallback default. Records every call in a list so tests can assert on which
engine paths fired without needing real LLM responses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ScriptedCall:
    prompt: str
    usage_label: Optional[str]
    tier: Any
    system_prompt: Optional[str]
    json_mode: bool


class ScriptedGeminiClient:
    """Drop-in GeminiClient replacement for tests.

    Tests supply a dict {usage_label: response_text} and optionally a fallback.
    complete() returns the mapped text; unknown labels use the fallback and
    record the miss so tests can choose to fail loudly.
    """

    def __init__(
        self,
        responses: Optional[dict[str, str]] = None,
        fallback: str = "[]",
    ) -> None:
        self.responses: dict[str, str] = dict(responses or {})
        self.fallback = fallback
        self.calls: list[ScriptedCall] = []
        self.unknown_labels: list[Optional[str]] = []

    async def complete(
        self,
        prompt: str,
        tier: Any = None,
        system_prompt: Optional[str] = None,
        tools: Optional[list] = None,
        timeout: Optional[float] = None,
        cached_content: Optional[str] = None,
        json_mode: bool = False,
        usage_label: Optional[str] = None,
        conversation_history: Optional[list[dict[str, str]]] = None,
    ) -> str:
        self.calls.append(
            ScriptedCall(
                prompt=prompt,
                usage_label=usage_label,
                tier=tier,
                system_prompt=system_prompt,
                json_mode=json_mode,
            )
        )
        if usage_label in self.responses:
            return self.responses[usage_label]
        self.unknown_labels.append(usage_label)
        return self.fallback

    def begin_usage_context(self, *, matter_id=None, run_id=None, recorder=None):
        return object()  # opaque token; no-op in stub

    def end_usage_context(self, token) -> None:
        return None

    def snapshot_usage(self) -> dict:
        return {}

    def get_usage_delta(self, snapshot=None) -> dict:
        return {"request_count": len(self.calls)}
