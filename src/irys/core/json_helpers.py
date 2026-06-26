"""JSON parsing and recovery helpers shared across the irys package."""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def parse_json_safe(text: str) -> Optional[dict]:
    """Safely parse JSON from LLM response."""
    if not text:
        return None

    # Try direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        # Valid JSON but not a dict (e.g. bare int/string like "94") — fall through
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code blocks
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            try:
                parsed = json.loads(text[start:end].strip())
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

    # Try to extract JSON from generic code blocks
    if "```" in text:
        start = text.find("```") + 3
        # Skip language identifier if present
        newline = text.find("\n", start)
        if newline > start:
            start = newline + 1
        end = text.find("```", start)
        if end > start:
            try:
                parsed = json.loads(text[start:end].strip())
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

    # Try to find JSON object in text
    brace_start = text.find("{")
    brace_end = text.rfind("}") + 1
    if brace_start >= 0 and brace_end > brace_start:
        try:
            parsed = json.loads(text[brace_start:brace_end])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return None



def _salvage_truncated_json(text: str) -> Optional[dict]:
    """Last-resort JSON recovery for model output truncated mid-stream.

    Walks from the first '{', tracks string/bracket state, trims to the last
    cleanly-closed value, strips dangling commas, appends missing closers,
    then re-parses. Returns the recovered dict only if it contains a 'facts'
    key (the schema lists facts first, so it survives mid-array truncation).

    Only invoked from extract_facts after both attempts fail with parse_failed.
    Never called from parse_json_safe.
    """
    if not text:
        return None

    start = text.find("{")
    if start < 0:
        return None
    text = text[start:]

    # Pass 1 — find the last position where bracket depth returned to 0
    depth = 0
    in_string = False
    escape_next = False
    last_clean_pos = 0

    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                last_clean_pos = i + 1

    if last_clean_pos == 0:
        # Never balanced — truncated before any close; try to recover anyway
        last_clean_pos = len(text)

    truncated = text[:last_clean_pos].rstrip()
    if truncated.endswith(","):
        truncated = truncated[:-1]

    # Pass 2 — determine which closers are missing, handling unclosed strings.
    # Helper: walk s, return (stack_of_closers, in_string_at_end, last_open_pos).
    def _scan(s):
        stk = []
        in_str = False
        esc = False
        last_open = -1
        for idx, ch in enumerate(s):
            if esc:
                esc = False
                continue
            if ch == "\\" and in_str:
                esc = True
                continue
            if ch == '"':
                if not in_str:
                    last_open = idx
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                stk.append("}")
            elif ch == "[":
                stk.append("]")
            elif ch in "}]" and stk:
                stk.pop()
        return stk, in_str, last_open

    stack, open_string, last_open = _scan(truncated)

    if open_string and last_open >= 0:
        # Truncated mid-string — backtrack to before the unclosed opening quote,
        # drop any trailing comma, then recompute the stack for that prefix.
        truncated = truncated[:last_open].rstrip()
        if truncated.endswith(","):
            truncated = truncated[:-1].rstrip()
        stack, _, _ = _scan(truncated)

    recovered = truncated + "".join(reversed(stack))

    try:
        parsed = json.loads(recovered)
        if isinstance(parsed, dict) and "facts" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    return None



