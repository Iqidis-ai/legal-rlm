"""Irys RLM Gradio UI — 6-panel matter intelligence dashboard.

Panels:
  1. Overview      — landing page: stats, weakest issues, gaps, SO metrics
  2. Run / Output  — live investigation with streaming reasoning trace
  3. Issues        — SO-4: issue tree with coverage, proof state, predicates
  4. Assertions    — SO-2: typed assertion table with corrections
  5. Gaps & Steer  — SO-7/3: missing docs, clarifications, steering controls
  6. Quant         — SO-6: payment reconciliation, damages, numeric conflicts

Architecture: in-process for local dev (InProcessBackend), HTTP for deployed service (HttpBackend).
"""

import asyncio
import concurrent.futures
import html
import logging
import os
import pathlib
import queue
import re
import shutil
import threading
import time
from collections import defaultdict
from typing import Any, Generator, Optional

import gradio as gr

from .backends.in_process import InProcessBackend
from ..rlm.state import normalize_research_mode

logger = logging.getLogger(__name__)

# Dedicated thread pool for running async backend calls from sync Gradio callbacks.
# InProcessBackend methods are async-in-signature but do synchronous SQLite work with
# no internal awaits — a ThreadPoolExecutor lets multiple panel refreshes run in
# parallel, which a single shared event loop would serialize.  asyncio.run() overhead
# per call is ~0.5 ms and is worth the concurrency gain.
_ASYNC_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=8, thread_name_prefix="irys_async"
)


def _run_async(coro, timeout: float = 30):
    """Run a coroutine from a sync context without conflicting with existing loops."""
    future = _ASYNC_EXECUTOR.submit(asyncio.run, coro)
    return future.result(timeout=timeout)


# ---------------------------------------------------------------------------
# Storage mode helpers
# ---------------------------------------------------------------------------

def _get_storage_mode() -> str:
    return os.getenv("IRYS_STORAGE_MODE", "local")


def _is_hash_filename(name: str) -> bool:
    base = pathlib.Path(name).stem
    return len(base) >= 32 and all(c in "0123456789abcdef" for c in base.lower())




# ---------------------------------------------------------------------------
# S3 matter-folder helpers (S3 mode — no local persistence)
# ---------------------------------------------------------------------------

def _s3_bucket() -> str:
    return os.getenv("S3_BUCKET", "")


def _s3_matters_base_prefix() -> str:
    prefix = os.getenv("S3_PREFIX", "").strip("/")
    return f"{prefix}/matters" if prefix else "matters"


def _get_s3_client():
    import boto3
    return boto3.client(
        "s3",
        region_name=os.getenv("S3_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID") or None,
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY") or None,
    )


def _sanitize_matter_name(name: str) -> str:
    """Turn a human matter name into a safe S3 key segment."""
    import re as _re
    name = name.strip()
    name = _re.sub(r"[^\w\s\-\.]", "", name)
    name = _re.sub(r"\s+", "_", name)
    return name[:80] or "untitled"


def _list_s3_matter_names() -> list[str]:
    """List matter names as common prefixes under matters/ in S3."""
    bucket = _s3_bucket()
    if not bucket:
        return []
    try:
        s3 = _get_s3_client()
        base = _s3_matters_base_prefix() + "/"
        paginator = s3.get_paginator("list_objects_v2")
        names: list[str] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=base, Delimiter="/"):
            for cp in page.get("CommonPrefixes", []):
                seg = cp["Prefix"][len(base):].rstrip("/")
                if seg:
                    names.append(seg.replace("_", " "))
        return sorted(names)
    except Exception:
        return []


def _s3_matter_doc_count(matter_name: str) -> str:
    """Count documents in an S3 matter prefix."""
    bucket = _s3_bucket()
    if not bucket:
        return "0 documents"
    try:
        s3 = _get_s3_client()
        safe = _sanitize_matter_name(matter_name)
        prefix = f"{_s3_matters_base_prefix()}/{safe}/"
        paginator = s3.get_paginator("list_objects_v2")
        count = sum(
            1
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix)
            for obj in page.get("Contents", [])
            if any(obj["Key"].lower().endswith(ext) for ext in (".pdf", ".docx", ".doc", ".txt", ".md"))
        )
        return f"{count} document{'s' if count != 1 else ''}"
    except Exception:
        return "? documents"


def _upload_files_to_s3_matter(uploaded_files: list, name: str) -> tuple[str, str]:
    """Upload Gradio files to S3 under matters/<name>/.

    Appends to existing matter if the name already exists. Per-file failures
    are isolated (do not abort the batch), retried with exponential backoff,
    and reported in the status string. Filename collisions get a numeric
    suffix instead of silently overwriting.
    Returns (display_name, status_message).
    """
    bucket = _s3_bucket()
    if not bucket:
        return name, "S3_BUCKET not configured — cannot upload"
    safe = _sanitize_matter_name(name)
    prefix = f"{_s3_matters_base_prefix()}/{safe}"
    s3 = _get_s3_client()

    # Pre-load existing keys so we can avoid silent overwrite on collision.
    used_keys: set[str] = set()
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
            for obj in page.get("Contents", []):
                used_keys.add(obj["Key"])
    except Exception as exc:
        logging.warning("S3 list before upload failed for '%s': %s", safe, exc)

    saved = 0
    failed: list[tuple[str, str]] = []
    for f in uploaded_files:
        try:
            if isinstance(f, str):
                actual_path = pathlib.Path(f)
                display_name = actual_path.name
            else:
                actual_path = pathlib.Path(f.name)
                display_name = actual_path.name
                orig = getattr(f, "orig_name", None)
                if orig and not _is_hash_filename(pathlib.Path(orig).name):
                    display_name = pathlib.Path(orig).name
        except Exception as exc:
            failed.append((str(f)[:80], f"resolve path: {exc}"))
            continue

        key = f"{prefix}/{display_name}"
        if key in used_keys:
            stem = pathlib.Path(display_name).stem
            ext = pathlib.Path(display_name).suffix
            n = 2
            while True:
                cand_name = f"{stem} ({n}){ext}"
                cand_key = f"{prefix}/{cand_name}"
                if cand_key not in used_keys:
                    display_name = cand_name
                    key = cand_key
                    break
                n += 1
        used_keys.add(key)

        last_err: Optional[Exception] = None
        ok = False
        for attempt in range(3):
            try:
                s3.upload_file(str(actual_path), bucket, key)
                ok = True
                break
            except Exception as exc:
                last_err = exc
                if attempt < 2:
                    time.sleep(0.5 * (2 ** attempt))
        if ok:
            saved += 1
        else:
            failed.append((display_name, str(last_err) if last_err else "unknown"))
            logging.error(
                "S3 upload failed after retries for '%s' -> %s: %s",
                display_name, key, last_err,
            )

    count = _s3_matter_doc_count(name)
    display = safe.replace("_", " ")
    total = saved + len(failed)
    if failed:
        sample = "; ".join(f"{n}: {e}" for n, e in failed[:3])
        more = f" (+{len(failed) - 3} more)" if len(failed) > 3 else ""
        msg = (
            f"Saved {saved}/{total} file(s) to '{safe}' — {count} total. "
            f"{len(failed)} failed: {sample}{more}"
        )
    else:
        msg = f"Saved {saved} file(s) to '{safe}' — {count} total"
    return display, msg


def _download_s3_matter_to_temp(matter_name: str, session_id: str) -> pathlib.Path:
    """Download all docs from an S3 matter to a stable temp dir keyed by session_id.

    Existing document files are replaced with the current S3 contents; the
    .irys/ subdirectory (matter DB) is left untouched so warm cache carries
    over between runs on the same matter.
    """
    import tempfile
    bucket = _s3_bucket()
    safe = _sanitize_matter_name(matter_name)
    prefix = f"{_s3_matters_base_prefix()}/{safe}/"
    temp_dir = pathlib.Path(tempfile.gettempdir()) / "irys" / session_id
    temp_dir.mkdir(parents=True, exist_ok=True)
    # Remove stale document files from a prior run but keep .irys/ (matter DB).
    for item in temp_dir.iterdir():
        if item.name == ".irys":
            continue
        if item.is_file():
            item.unlink(missing_ok=True)
        elif item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
    s3 = _get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = key[len(prefix):]
            if not filename or "/" in filename:
                continue  # skip sub-prefixes
            dest = temp_dir / filename
            s3.download_file(bucket, key, str(dest))
    return temp_dir


def _list_s3_matter_files(matter_name: str) -> list[str]:
    """List document filenames in an S3 matter (flat, no sub-prefixes)."""
    bucket = _s3_bucket()
    if not bucket or not matter_name:
        return []
    try:
        s3 = _get_s3_client()
        safe = _sanitize_matter_name(matter_name)
        prefix = f"{_s3_matters_base_prefix()}/{safe}/"
        paginator = s3.get_paginator("list_objects_v2")
        files: list[str] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                filename = obj["Key"][len(prefix):]
                if filename and "/" not in filename:
                    files.append(filename)
        return sorted(files)
    except Exception:
        return []


def _delete_s3_matter_file(matter_name: str, filename: str) -> tuple[list[str], str]:
    """Delete a single file from a matter. Returns (updated file list, status)."""
    bucket = _s3_bucket()
    if not bucket:
        return [], "S3_BUCKET not configured"
    if not filename:
        return _list_s3_matter_files(matter_name), "No file selected"
    try:
        s3 = _get_s3_client()
        safe = _sanitize_matter_name(matter_name)
        key = f"{_s3_matters_base_prefix()}/{safe}/{filename}"
        s3.delete_object(Bucket=bucket, Key=key)
        files = _list_s3_matter_files(matter_name)
        return files, f"Deleted '{filename}' — {len(files)} file(s) remaining"
    except Exception as e:
        return _list_s3_matter_files(matter_name), f"Delete failed: {e}"


def _delete_s3_matter(matter_name: str) -> tuple[list[str], str]:
    """Delete all files in a matter (removes the whole matter folder from S3).
    Returns (updated matter list, status)."""
    bucket = _s3_bucket()
    if not bucket:
        return _list_s3_matter_names(), "S3_BUCKET not configured"
    if not matter_name:
        return _list_s3_matter_names(), "No matter selected"
    try:
        s3 = _get_s3_client()
        safe = _sanitize_matter_name(matter_name)
        prefix = f"{_s3_matters_base_prefix()}/{safe}/"
        paginator = s3.get_paginator("list_objects_v2")
        to_delete: list[dict] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                to_delete.append({"Key": obj["Key"]})
        if to_delete:
            for i in range(0, len(to_delete), 1000):
                s3.delete_objects(Bucket=bucket, Delete={"Objects": to_delete[i:i + 1000]})
        matters = _list_s3_matter_names()
        return matters, f"Deleted matter '{matter_name}' ({len(to_delete)} file(s) removed)"
    except Exception as e:
        return _list_s3_matter_names(), f"Delete failed: {e}"


# ---------------------------------------------------------------------------
# Workspace management
# ---------------------------------------------------------------------------


def _clear_matter(path: str) -> tuple[str, str]:
    """Delete the .irys/ model data for a matter folder (keeps documents).
    Returns (folder_name, status_message)."""
    if not path or not path.strip():
        return "", "No folder selected."
    folder = pathlib.Path(path.strip())
    irys_dir = folder / ".irys"
    if irys_dir.exists():
        shutil.rmtree(str(irys_dir))
        return folder.name, f"Cleared analysis for '{folder.name}'. Documents kept. Next run starts fresh."
    return folder.name, f"No analysis data found in '{folder.name}'."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STEP_ICONS = {
    "THINKING": "💭", "SEARCHING": "🔍", "READING": "📄",
    "SYNTHESIZING": "⚡", "VERIFY": "✓", "REPLAN": "↺", "ANSWER": "✅",
    "SEARCH": "🔍", "READ": "📄", "SYNTH": "⚡",
}

_SESSION_TURN_LIMIT = 4
_SESSION_QUERY_LIMIT = 600
_SESSION_ANSWER_LIMIT = 3000
_AUTO_APPENDIX_SECTIONS = (
    ("Source Calibration Advisory", "Auto-generated by SO-5 advocacy gate"),
    ("Financial Analysis", "Auto-generated by SO-6 threshold gate"),
)


def _fmt_coverage(frac: Optional[float]) -> str:
    if frac is None:
        return "—"
    return f"{frac:.0%}"


def _fmt_research_mode_label(value: Any) -> str:
    return normalize_research_mode(value).replace("_", " ").title()


# Attorney-facing labels for cascade terminal families. Kept free of
# developer jargon — "Deep investigation" rather than "investigate",
# "Quick summary" rather than "read" — per the UI/UX memory that the
# attorney audience should never see internal family tokens.
_ROUTE_LABELS = {
    "investigate": "Deep investigation",
    "read": "Quick summary",
    "query": "Direct lookup",
    "trace": "Reasoning trace",
    "steer": "Correction preview",
    "compare": "Change comparison",
    "scenario": "What-if analysis",
    "deliverable": "Document draft",
    "clarify": "Clarification requested",
    "pleasantry": "Friendly reply",
}


def _fmt_thinking_steps_fallback(state: Any) -> str:
    """Render state.thinking_steps as a reasoning trace when neither
    streaming call_thinking nor ledger events are available. This
    happens on cheap-path families (read, query, trace, steer,
    compare, scenario, deliverable, clarify) where no run session
    runs — api._seed_cheap_path_trace populates thinking_steps
    directly on the state so the UI tab isn't blank. Returns empty
    string when there are no steps.
    """
    try:
        steps = getattr(state, "thinking_steps", None) or []
        if not steps:
            return ""
        lines: list[str] = []
        for idx, s in enumerate(steps, start=1):
            content = getattr(s, "content", "") or ""
            if not content:
                continue
            # Attorney-facing — skip the developer prefixes like [T].
            lines.append(f"{idx}. {content}")
        return "\n".join(lines)
    except Exception:
        return ""


def _fmt_route_chip(state: Any) -> str | None:
    """Return a short route chip ("via Quick summary") for the status
    line, or None if no cascade route is available. When the classifier
    family and terminal family differ (escalation happened) show both:
    e.g. "via Quick summary → Deep investigation". This makes the cost
    cascade decision visible without exposing internal family tokens.
    """
    try:
        findings = getattr(state, "findings", {}) or {}
        route = findings.get("route") if isinstance(findings, dict) else None
        if not route:
            return None
        classifier = str(route.get("classifier_family") or "").strip()
        terminal = str(route.get("terminal_family") or "").strip()
        # Internal-only families that attorneys shouldn't see named.
        if terminal in {"read_infra_failure"} or not terminal:
            return None
        term_label = _ROUTE_LABELS.get(terminal, terminal.title())
        if classifier and classifier != terminal:
            cls_label = _ROUTE_LABELS.get(classifier, classifier.title())
            return f"via {cls_label} → {term_label}"
        return f"via {term_label}"
    except Exception:
        return None


def _fmt_ledger_event(event: dict) -> str | None:
    """Format a single ledger event dict into a human-readable trace line.

    Returns None for synthetic sentinel dicts (e.g. run_terminal, error) that
    stream_run_events() appends after the real persisted events.
    """
    if "event_type" not in event:
        return None  # terminal sentinel {"event":"run_terminal"} or {"error":...}
    seq = event.get("seq_no", "?")
    etype = event.get("event_type", "?")
    summary = event.get("summary", "")
    why = event.get("why", "")
    line = f"#{seq} {etype} | {summary}"
    if why:
        line += f"\n  ↳ {why}"
    return line


def _clip_for_context(value: Any, limit: int) -> str:
    text = "" if value is None else str(value).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 16)].rstrip() + "\n...[truncated]"


def _build_conversation_history(turns: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return bounded prior visible turns for same-session multi-turn continuity."""
    history: list[dict[str, str]] = []
    for turn in turns[-_SESSION_TURN_LIMIT:]:
        query = _clip_for_context(turn.get("query"), _SESSION_QUERY_LIMIT)
        answer = _clip_for_context(turn.get("answer"), _SESSION_ANSWER_LIMIT)
        if query or answer:
            history.append({"query": query, "answer": answer})
    return history


def _build_chat_messages(
    turns: list[dict[str, str]],
    *,
    pending_user: Optional[str] = None,
    pending_assistant: Optional[str] = None,
) -> list[dict[str, str]]:
    """Render visible conversation history for the run UI."""
    messages: list[dict[str, str]] = []
    for turn in turns[-_SESSION_TURN_LIMIT:]:
        query = str(turn.get("query") or "").strip()
        answer = str(turn.get("answer") or "").strip()
        if query:
            messages.append({"role": "user", "content": query})
        if answer:
            messages.append({"role": "assistant", "content": answer})
    if pending_user:
        messages.append({"role": "user", "content": pending_user})
    if pending_assistant:
        messages.append({"role": "assistant", "content": pending_assistant})
    return messages


def _split_run_output_sections(output: Any) -> tuple[str, str]:
    """Split the main answer from auto-generated run appendices.

    Auto-appended SO-5 / SO-6 sections are useful, but they should live in a
    collapsible diagnostics surface rather than inside the main conversational
    answer.
    """
    text = "" if output is None else str(output).strip()
    if not text:
        return "", ""

    starts: list[int] = []
    for section_name, marker in _AUTO_APPENDIX_SECTIONS:
        marker_idx = text.find(marker)
        if marker_idx < 0:
            continue
        header_pat = re.compile(
            rf"(?im)^(?:#+\s*)?{re.escape(section_name)}\s*$"
        )
        section_starts = [
            match.start() for match in header_pat.finditer(text) if match.start() <= marker_idx
        ]
        starts.append(max(section_starts) if section_starts else marker_idx)

    if not starts:
        return text, ""

    split_at = min(starts)
    return text[:split_at].rstrip(), text[split_at:].strip()


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _llm_token_breakdown(metrics: dict[str, Any]) -> str:
    parts = [f"{_safe_int(metrics.get('input_tokens')):,} input"]
    cache_tokens = _safe_int(metrics.get("cache_read_tokens"))
    tool_use_tokens = _safe_int(metrics.get("tool_use_prompt_tokens"))
    thinking_tokens = _safe_int(metrics.get("thinking_tokens"))
    output_tokens = _safe_int(metrics.get("output_tokens"))
    if cache_tokens:
        parts.append(f"{cache_tokens:,} cache")
    if tool_use_tokens:
        parts.append(f"{tool_use_tokens:,} tool")
    if thinking_tokens:
        parts.append(f"{thinking_tokens:,} thinking")
    parts.append(f"{output_tokens:,} output")
    return " / ".join(parts)


def _truncate(value: Any, limit: int = 48) -> str:
    # Readability-first UI rule: never cut visible text off in the dashboard.
    return "" if value is None else str(value)


def _fmt_span_label(span_id: Optional[str]) -> str:
    """Attorney-readable rendering of a source span.

    - "page:3" → "Page 3"
    - "para:12" / "section:4.2" → "Paragraph 12" / "Section 4.2"
    - UUID-ish strings → "—" (not useful to the reader)
    - other human-ish labels → passed through trimmed
    """
    s = (span_id or "").strip()
    if not s:
        return "—"
    if ":" in s:
        kind, tail = s.split(":", 1)
        kind = kind.lower()
        pretty = {
            "page": "Page",
            "para": "Paragraph",
            "paragraph": "Paragraph",
            "section": "Section",
            "clause": "Clause",
            "line": "Line",
        }.get(kind)
        if pretty:
            return f"{pretty} {tail}"
    # Treat long hex/uuid-looking strings as uninformative.
    if len(s) >= 24 and all(c in "0123456789abcdef-" for c in s.lower()):
        return "—"
    return s if len(s) <= 40 else s[:37] + "…"


def _fmt_percent_html(value: Optional[float]) -> str:
    if value is None:
        return "&mdash;"
    return f"{value:.0%}"


def _fmt_money(value: Any, decimals: int = 4) -> str:
    amount = _safe_float(value, 0.0)
    return f"${amount:,.{decimals}f}"


def _fmt_money_short(value: Any) -> str:
    amount = _safe_float(value, 0.0)
    return f"${amount:,.2f}"


def _metric_card(title: str, value: str, detail: str = "", tone: str = "default") -> str:
    detail_html = f"<div class='viz-card-detail'>{detail}</div>" if detail else ""
    return (
        f"<div class='viz-card tone-{_escape(tone)}'>"
        f"<div class='viz-card-title'>{_escape(title)}</div>"
        f"<div class='viz-card-value'>{value}</div>"
        f"{detail_html}"
        f"</div>"
    )


def _bar_row(label: str, value: float, maximum: float, meta: str = "", tone: str = "blue") -> str:
    pct = 0.0 if maximum <= 0 else min(100.0, max(4.0, (value / maximum) * 100.0))
    meta_html = f"<div class='viz-bar-meta'>{meta}</div>" if meta else ""
    return (
        "<div class='viz-bar-row'>"
        f"<div class='viz-bar-label'>{_escape(label)}</div>"
        "<div class='viz-bar-track'>"
        f"<div class='viz-bar-fill tone-{_escape(tone)}' style='width:{pct:.1f}%'></div>"
        "</div>"
        f"{meta_html}"
        "</div>"
    )


_DS_CSS = """<style>
.intel-panel{
  background:var(--background-fill-secondary,#f8fafc);
  border:1px solid var(--block-border-color,#e2e8f0);
  border-radius:14px;padding:16px;
  box-shadow:0 2px 12px rgba(0,0,0,.06);
  display:flex;flex-direction:column;gap:10px;
  width:100%;box-sizing:border-box;overflow:hidden;
}
.intel-panel .intel-panel-title{
  font-size:11px;font-weight:700;
  color:var(--body-text-color,#0f172a);
  letter-spacing:.08em;text-transform:uppercase;
  padding-bottom:6px;
  border-bottom:1px solid var(--block-border-color,#e2e8f0);
  margin-bottom:2px;
}
.intel-panel .viz-shell{display:flex;flex-direction:column;gap:8px;}
.intel-panel .viz-card-grid{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(90px,1fr));
  gap:6px;
}
.intel-panel .viz-card{
  border:1px solid var(--block-border-color,#e2e8f0);
  border-radius:10px;padding:8px 6px;
  background:var(--background-fill-primary,#fff);
  overflow:hidden;
}
.intel-panel .viz-card.tone-amber{border-color:rgba(217,119,6,.4);}
.intel-panel .viz-card.tone-green{border-color:rgba(21,128,61,.4);}
.intel-panel .viz-card.tone-red{border-color:rgba(185,28,28,.4);}
.intel-panel .viz-card-title{
  font-size:9px;letter-spacing:.07em;text-transform:uppercase;
  color:var(--body-text-color-subdued,#64748b);margin-bottom:2px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}
.intel-panel .viz-card-value{
  font-size:18px;font-weight:700;
  color:var(--body-text-color,#0f172a);line-height:1.1;
}
.intel-panel .viz-card-detail{
  font-size:10px;color:var(--body-text-color-subdued,#64748b);margin-top:3px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}
.intel-panel .viz-two-col{display:grid;grid-template-columns:1fr;gap:6px;}
.intel-panel .viz-panel{
  border:1px solid var(--block-border-color,#e2e8f0);
  border-radius:10px;padding:10px;
  background:var(--background-fill-primary,#fff);
}
.intel-panel .viz-panel-title{
  font-size:11px;font-weight:700;
  color:var(--body-text-color,#0f172a);margin-bottom:6px;
}
.intel-panel .viz-subtitle{
  font-size:10px;font-weight:700;
  color:var(--body-text-color-subdued,#64748b);margin-bottom:3px;
}
.intel-panel .viz-footnote{font-size:10px;color:var(--body-text-color-subdued,#64748b);margin-top:4px;}
.intel-panel .viz-bar-row{
  display:flex;flex-direction:column;gap:2px;
  padding:3px 0;
  border-bottom:1px solid var(--block-border-color,#e2e8f0);
}
.intel-panel .viz-bar-label{
  font-size:10px;color:var(--body-text-color,#334155);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}
.intel-panel .viz-bar-track{
  height:5px;border-radius:999px;
  background:var(--block-border-color,#e2e8f0);
  overflow:hidden;margin:2px 0;
}
.intel-panel .viz-bar-fill{height:100%;border-radius:999px;}
.intel-panel .viz-bar-fill.tone-green{background:#16a34a;}
.intel-panel .viz-bar-fill.tone-amber{background:#d97706;}
.intel-panel .viz-bar-fill.tone-blue{background:var(--color-accent,#2563eb);}
.intel-panel .viz-bar-fill.tone-red{background:#b91c1c;}
.intel-panel .viz-bar-meta{font-size:9px;color:var(--body-text-color-subdued,#64748b);}
.intel-panel .viz-list-row{
  display:flex;justify-content:space-between;align-items:flex-start;
  gap:6px;padding:5px 0;
  border-bottom:1px solid var(--block-border-color,#e2e8f0);
  font-size:11px;color:var(--body-text-color,#334155);
}
.intel-panel .viz-list-row strong{
  white-space:nowrap;color:var(--body-text-color-subdued,#64748b);
}
.intel-panel .viz-list-columns{display:grid;grid-template-columns:1fr 1fr;gap:6px;}
.intel-panel .viz-list-columns ul{
  margin:3px 0 0;padding-left:12px;
  font-size:10px;color:var(--body-text-color-subdued,#64748b);
}
.intel-panel .viz-list-columns li{margin-bottom:3px;}
.intel-panel .viz-empty{
  border:1px dashed var(--block-border-color,#cbd5e1);
  border-radius:8px;padding:10px;
  color:var(--body-text-color-subdued,#64748b);font-size:11px;
}
</style>"""


def _fmt_overview_panel(data: dict) -> str:
    if not data:
        return (
            _DS_CSS
            + "<div class='intel-panel'>"
            "<div class='intel-panel-title'>Matter Intelligence</div>"
            "<div class='viz-empty'>"
            "Run your first investigation to see matter intelligence here."
            "</div>"
            "</div>"
        )

    stats = data.get("stats", {})
    so = data.get("so_metrics", {})
    llm = stats.get("llm", {}) if isinstance(stats.get("llm"), dict) else {}
    llm_totals = llm.get("totals", {}) if isinstance(llm, dict) else {}
    coverage_report = data.get("coverage_report", []) or []
    weakest = data.get("weakest_issues", []) or []
    gaps = data.get("top_gaps", []) or []
    clarifications = data.get("pending_clarifications", []) or []

    cards = [
        _metric_card("Assertions", f"{_safe_int(stats.get('assertion_count', 0)):,}"),
        _metric_card(
            "Open Issues",
            f"{_safe_int(stats.get('open_issue_count', 0)):,}",
            detail=f"Gaps: {_safe_int(stats.get('open_gap_count', 0)):,}",
        ),
        _metric_card(
            "Actors",
            f"{_safe_int(stats.get('actor_count', 0)):,}",
            detail=f"Quant facts: {_safe_int(stats.get('quant_fact_count', 0)):,}",
        ),
        _metric_card(
            "LLM Cost",
            _fmt_money(llm_totals.get("estimated_cost_usd", 0.0)),
            detail=f"{_safe_int(llm_totals.get('request_count', 0)):,} calls",
            tone="amber",
        ),
        _metric_card(
            "Issue Coverage",
            _fmt_percent_html(so.get("issue_coverage_avg")),
            detail=f"Reuse: {_fmt_percent_html(so.get('reuse_rate'))}",
            tone="green",
        ),
        _metric_card(
            "Calibration",
            _fmt_percent_html(so.get("source_role_known_rate")),
            detail=f"Structured: {_fmt_percent_html(so.get('assertion_structure_rate'))}",
        ),
    ]

    coverage_rows: list[str] = []
    coverage_sorted = sorted(
        coverage_report,
        key=lambda issue: float(issue.get("coverage_fraction", 0.0)),
    )
    for issue in coverage_sorted:
        frac = _safe_float(issue.get("coverage_fraction", 0.0))
        meta = f"{_fmt_percent_html(frac)} coverage"
        if issue.get("has_proof_gap"):
            meta += " | proof gap"
        coverage_rows.append(
            _bar_row(issue.get("title") or issue.get("id") or "Issue", frac, 1.0, meta, tone="green")
        )

    tier_rows: list[str] = []
    by_tier = llm_totals.get("by_tier", {}) if isinstance(llm_totals, dict) else {}
    tier_items = list(by_tier.items()) if isinstance(by_tier, dict) else []
    tier_max = max(
        (_safe_float(item[1].get("estimated_cost_usd", 0.0)) for item in tier_items),
        default=0.0,
    )
    for tier_name, tier_data in sorted(
        tier_items,
        key=lambda item: _safe_float(item[1].get("estimated_cost_usd", 0.0)),
        reverse=True,
    ):
        tier_rows.append(
            _bar_row(
                tier_name.upper(),
                _safe_float(tier_data.get("estimated_cost_usd", 0.0)),
                tier_max or 1.0,
                meta=(
                    f"{_fmt_money(tier_data.get('estimated_cost_usd', 0.0))} | "
                    f"{_safe_int(tier_data.get('requests', 0)):,} calls"
                ),
                tone="amber",
            )
        )

    # Domain composition panel
    domain_comp = data.get("domain_composition", {})
    domain_facets = domain_comp.get("facets", []) if isinstance(domain_comp, dict) else []
    primary_domain = domain_comp.get("primary_domain_profile_id", "legal") if isinstance(domain_comp, dict) else "legal"
    domain_facet_rows: list[str] = []
    for facet in domain_facets:
        pid = facet.get("domain_profile_id", "unknown")
        conf = _safe_float(facet.get("confidence", 0.0))
        label = pid.replace("_", " ").title()
        if pid == primary_domain:
            label += " (primary)"
        review_roles = facet.get("requires_review_roles")
        meta = _fmt_percent_html(conf)
        if review_roles:
            meta += f" | review: {', '.join(review_roles[:3])}"
        domain_facet_rows.append(
            _bar_row(label, conf, 1.0, meta, tone="green")
        )

    gap_items = "".join(
        f"<li>{_escape(g.get('description') or g.get('gap_type') or 'Gap')}</li>"
        for g in gaps
    ) or "<li>No open gaps.</li>"
    clarification_items = "".join(
        "<li>"
        f"{_escape(c.get('question_text') or c.get('question') or 'Clarification')}"
        "</li>"
        for c in clarifications
    ) or "<li>No pending clarifications.</li>"

    pricing_source = _escape(llm_totals.get("pricing_source", ""))

    return (
        _DS_CSS
        + "<div class='intel-panel'>"
        "<div class='intel-panel-title'>Matter Intelligence</div>"
        "<div class='viz-shell'>"
        "<div class='viz-card-grid'>"
        + "".join(cards)
        + "</div>"
        + "<div class='viz-two-col'>"
        + "<div class='viz-panel'>"
        + "<div class='viz-panel-title'>Coverage distribution</div>"
        + (
            "".join(coverage_rows)
            if coverage_rows
            else "<div class='viz-empty'>Issue coverage will appear after investigation.</div>"
        )
        + "</div>"
        + "<div class='viz-panel'>"
        + "<div class='viz-panel-title'>LLM spend by tier</div>"
        + (
            "".join(tier_rows)
            if tier_rows
            else "<div class='viz-empty'>No LLM usage recorded yet.</div>"
        )
        + (
            f"<div class='viz-footnote'>Pricing source: <a href='{pricing_source}' target='_blank'>Google Gemini API pricing</a></div>"
            if pricing_source
            else ""
        )
        + "</div>"
        + "</div>"
        + "<div class='viz-panel'>"
        + "<div class='viz-panel-title'>Domain composition</div>"
        + (
            "".join(domain_facet_rows)
            if domain_facet_rows
            else "<div class='viz-empty'>Default profile. Domain detection runs on document ingest.</div>"
        )
        + "</div>"
        + "<div class='viz-two-col'>"
        + "<div class='viz-panel'>"
        + "<div class='viz-panel-title'>Weakest issues</div>"
        + (
            "".join(
                "<div class='viz-list-row'>"
                f"<span>{_escape(issue.get('title') or issue.get('id') or 'Issue')}</span>"
                f"<strong>{_fmt_percent_html(_safe_float(issue.get('coverage_fraction', 0.0)))}</strong>"
                "</div>"
                for issue in weakest
            )
            if weakest
            else "<div class='viz-empty'>No issue coverage data yet.</div>"
        )
        + "</div>"
        + "<div class='viz-panel'>"
        + "<div class='viz-panel-title'>Open work</div>"
        + "<div class='viz-list-columns'>"
        + "<div><div class='viz-subtitle'>Gaps</div><ul>"
        + gap_items
        + "</ul></div>"
        + "<div><div class='viz-subtitle'>Clarifications</div><ul>"
        + clarification_items
        + "</ul></div>"
        + "</div>"
        + "</div>"
        + "</div>"
        + "</div>"
        + "</div>"
    )


def _fmt_trust_notice(issues: list) -> str:
    """P0.2 + P0.5 visibility banner: tell the attorney up front
    which issues the synthesis memo is hedging on. Mirrors the
    logic of engine._build_trust_abstention_block — if verified
    support is zero or a proof gap exists, the memo refuses a
    definitive claim, and the attorney should see that in-app
    rather than only inside the memo prose.
    """
    if not issues:
        return ""
    total = len(issues)
    fully_verified = 0
    candidate_only = 0
    gap_blocked = 0
    no_support = 0
    issue_names: list[str] = []
    for issue in issues:
        v = _safe_int(issue.get("verified_supporting_count", 0))
        c = _safe_int(issue.get("candidate_supporting_count", 0))
        has_gap = bool(issue.get("has_proof_gap"))
        title = issue.get("title") or "Untitled"
        if v > 0 and not has_gap:
            fully_verified += 1
            continue
        if has_gap:
            gap_blocked += 1
            issue_names.append(f"{title} (proof gap)")
        elif v == 0 and c > 0:
            candidate_only += 1
            issue_names.append(f"{title} (candidate-only)")
        else:
            no_support += 1
            issue_names.append(f"{title} (unsupported)")
    hedged = candidate_only + gap_blocked + no_support
    if hedged == 0:
        # Everything proved — quiet green banner.
        return (
            "<div style='padding:10px 14px;border-radius:8px;"
            "background:#dcfce7;color:#14532d;border-left:4px solid #15803d;"
            "margin-bottom:12px;font-size:13px;'>"
            f"<strong>All {total} issue(s) have verified support.</strong> "
            "The memo makes definitive claims only where the attorney has "
            "signed off. No hedging needed."
            "</div>"
        )
    # Hedging banner. List up to 4 issue names; summarize the rest.
    shown = issue_names[:4]
    remainder = len(issue_names) - len(shown)
    detail = "; ".join(_escape(n) for n in shown)
    if remainder > 0:
        detail += f"; and {remainder} more"
    return (
        "<div style='padding:10px 14px;border-radius:8px;"
        "background:#fef3c7;color:#78350f;border-left:4px solid #b45309;"
        "margin-bottom:12px;font-size:13px;'>"
        f"<strong>⚠ Hedging on {hedged} of {total} issue(s).</strong> "
        f"The memo frames findings as <em>provisional</em> or <em>unresolved</em> "
        f"on: {detail}. Verify supporting facts in the Review Inbox to promote "
        f"these to definitive claims."
        "</div>"
    )


def _fmt_issues_panel(issues: list) -> str:
    if not issues:
        return "<div class='viz-empty'>No open issues.</div>"

    rows: list[str] = []
    for issue in sorted(
        issues,
        key=lambda item: (item.get("depth", 0), item.get("coverage_fraction", 0.0)),
    ):
        depth = max(0, _safe_int(issue.get("depth", 0)))
        coverage = max(0.0, min(1.0, _safe_float(issue.get("coverage_fraction", 0.0))))
        # P0.2 two-lane coverage: verified is the attorney-signed-off
        # lane; advisory is candidate+verified combined. Show both so
        # the attorney sees how much is actually proved vs. just
        # extracted.
        verified_cov = max(0.0, min(1.0, _safe_float(issue.get("verified_coverage_fraction", 0.0))))
        verified_cnt = _safe_int(issue.get("verified_supporting_count", 0))
        candidate_cnt = _safe_int(issue.get("candidate_supporting_count", 0))
        title = _escape(issue.get("title") or issue.get("id") or "Issue")
        proof = _escape(issue.get("proof_status", "none"))
        attack = _safe_int(issue.get("attacking_count", 0))
        contested = _safe_int(issue.get("contested_predicates", 0))
        blocked = _safe_int(issue.get("blocked_predicates", 0))
        details = [
            f"{verified_cnt} verified",
            f"{candidate_cnt} candidate",
            f"{attack} attack" if attack else "",
        ]
        if contested:
            details.append(f"{contested} disputed")
        if blocked:
            details.append(f"{blocked} blocked")
        details_str = " | ".join(d for d in details if d)
        gap_marker = ""
        if issue.get("has_proof_gap"):
            gap_marker = (
                "<span style='margin-left:8px;padding:1px 6px;border-radius:8px;"
                "background:#fee2e2;color:#991b1b;font-size:10px;font-weight:700;"
                "text-transform:uppercase;letter-spacing:0.05em;'>proof gap</span>"
            )
        # Two overlaid bars: thin dark verified bar inside a wider
        # light candidate-advisory bar. Legend below the track.
        verified_pct = max(0.0, verified_cov * 100)
        advisory_pct = max(0.0, coverage * 100)
        rows.append(
            f"<div class='issue-row' style='--issue-indent:{depth * 18}px'>"
            f"<div class='issue-head'>"
            f"<span class='proof-pill proof-{proof}'>{proof}</span>"
            f"<span class='issue-title'>{title}</span>{gap_marker}"
            f"<span class='issue-pct' title='Verified / Advisory'>"
            f"{verified_cov:.0%} / {coverage:.0%}</span>"
            f"</div>"
            f"<div class='issue-track' style='position:relative;background:#f3f4f6;"
            f"height:8px;border-radius:4px;overflow:hidden;'>"
            f"<div style='position:absolute;inset:0 auto 0 0;width:{advisory_pct:.1f}%;"
            f"background:#bfdbfe;'></div>"
            f"<div style='position:absolute;inset:0 auto 0 0;width:{verified_pct:.1f}%;"
            f"background:#1d4ed8;'></div>"
            f"</div>"
            f"<div class='issue-meta'>{_escape(details_str)}</div>"
            "</div>"
        )
    # Tiny legend row at the top so the reader knows what the two
    # shades mean.
    legend = (
        "<div style='display:flex;gap:16px;font-size:11px;color:#6b7280;"
        "margin-bottom:8px;padding:0 4px;'>"
        "<span><span style='display:inline-block;width:10px;height:8px;"
        "background:#1d4ed8;border-radius:2px;vertical-align:middle;'></span>"
        " Verified coverage</span>"
        "<span><span style='display:inline-block;width:10px;height:8px;"
        "background:#bfdbfe;border-radius:2px;vertical-align:middle;'></span>"
        " Advisory (candidate + verified)</span>"
        "</div>"
    )
    notice = _fmt_trust_notice(issues)
    return (
        "<div class='viz-shell'>"
        + notice
        + legend
        + "<div class='issues-stack'>" + "".join(rows) + "</div></div>"
    )


_MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
_QUARTER_STARTS = {"01": "Q1", "04": "Q2", "07": "Q3", "10": "Q4"}


def _display_date(iso_date: str, precision: str | None) -> str:
    """Render an ISO date according to its precision for human-friendly display."""
    raw = (iso_date or "").strip()
    if not raw or raw.lower() in {"null", "none"}:
        return "Undated"
    if len(raw) < 10:
        return raw
    try:
        year, month, day = raw[:10].split("-")
    except ValueError:
        return raw
    if not year.isdigit():
        return raw
    if precision == "year":
        return year
    if precision == "quarter":
        if month.isdigit():
            return f"{_QUARTER_STARTS.get(month, 'Q?')} {year}"
        return raw
    if precision == "month":
        if month.isdigit():
            m_idx = int(month)
            m_name = _MONTH_NAMES[m_idx] if 1 <= m_idx <= 12 else month
            return f"{m_name} {year}"
        return raw
    # "day" or unknown precision — show full date in readable form
    if month.isdigit() and day.isdigit():
        m_idx = int(month)
        m_name = _MONTH_NAMES[m_idx] if 1 <= m_idx <= 12 else month
        return f"{m_name} {int(day)}, {year}"
    return raw


def _friendly_source_label(raw: Optional[str]) -> str:
    """Attorney-readable rendering of a source doc reference.

    - empty/None → "—"
    - 32+ char hex-only (looks like an internal id) → "—"
    - anything with "/" or "\\" → basename
    - otherwise → trimmed as-is
    """
    s = (raw or "").strip()
    if not s:
        return "—"
    if len(s) >= 24 and all(c in "0123456789abcdef-" for c in s.lower()):
        return "—"
    if "/" in s or "\\" in s:
        norm = s.replace("\\", "/")
        return norm.rsplit("/", 1)[-1] or norm
    return s if len(s) <= 80 else s[:77] + "…"


def _provenance_tier_label(event: dict[str, Any]) -> str:
    tier = str(event.get("model_tier") or "").strip()
    if tier:
        return f"{tier.upper()} tier"
    return "Irys"


def _fmt_timeline_panel(events: list[dict]) -> str:
    if not events:
        return "<div class='viz-empty'>No timeline events available.</div>"
    withheld_total = sum(1 for e in events if e.get("withheld"))
    items: list[str] = []
    for event in events:
        raw_date = event.get("date") or ""
        precision = event.get("date_precision")
        date = _escape(_display_date(raw_date, precision) if raw_date else "Undated")
        is_withheld = bool(event.get("withheld"))
        if is_withheld:
            title = "<em style='color:#b45309;'>Withheld under clean policy</em>"
            source_doc = "[withheld]"
        else:
            title = _escape(event.get("event") or "Event")
            source_doc = _escape(_friendly_source_label(event.get("source_doc")))
        kind = _escape(event.get("kind") or "event")
        subject = _escape(event.get("subject") or "")
        meta_parts = [kind, source_doc]
        if subject and not is_withheld:
            meta_parts.append(subject)
        bar_style = (
            "background:#fef3c7;border-left:3px solid #b45309;"
            if is_withheld else ""
        )
        items.append(
            f"<div class='timeline-item' style='{bar_style}'>"
            f"<div class='timeline-date'>{date}</div>"
            "<div class='timeline-line'><span class='timeline-dot'></span></div>"
            "<div class='timeline-body'>"
            f"<div class='timeline-title'>{title}</div>"
            f"<div class='timeline-meta'>{_escape(' | '.join(meta_parts))}</div>"
            "</div>"
            "</div>"
        )
    header = ""
    if withheld_total:
        header = (
            "<div style='padding:8px 12px;border-radius:6px;"
            "background:#fef3c7;color:#78350f;font-size:12px;"
            "margin-bottom:12px;border-left:3px solid #b45309;'>"
            f"<strong>{withheld_total} event(s) withheld</strong> under clean "
            "policy — their dates are preserved in this timeline but the "
            "event text and source are hidden. Privileged docs are never "
            "shown in clean mode."
            "</div>"
        )
    return (
        "<div class='viz-shell'>"
        + header
        + "<div class='timeline-list'>"
        + "".join(items)
        + "</div></div>"
    )


def _fmt_evidence_matrix_panel(matrix: dict) -> str:
    if not matrix or not matrix.get("issues") or not matrix.get("sources"):
        return "<div class='viz-empty'>Evidence matrix will populate after issues are linked to sources.</div>"

    issues = list(matrix.get("issues", []))
    sources = list(matrix.get("sources", []))
    issue_totals = (
        matrix.get("issue_totals", {}) if isinstance(matrix.get("issue_totals"), dict) else {}
    )
    source_totals = (
        matrix.get("source_totals", {}) if isinstance(matrix.get("source_totals"), dict) else {}
    )
    cells = matrix.get("cells", {}) if isinstance(matrix.get("cells"), dict) else {}

    issues.sort(
        key=lambda issue: -(
            _safe_int(issue_totals.get(issue["id"], {}).get("supporting", 0))
            + _safe_int(issue_totals.get(issue["id"], {}).get("attacking", 0))
        )
    )
    sources.sort(
        key=lambda source: -(
            _safe_int(source_totals.get(source, {}).get("supporting", 0))
            + _safe_int(source_totals.get(source, {}).get("attacking", 0))
        )
    )
    max_total = 1
    for issue in issues:
        for source in sources:
            total = _safe_int(cells.get(issue["id"], {}).get(source, {}).get("total", 0))
            max_total = max(max_total, total)

    # Adversarial #9 fix: render source columns with basenames, and
    # lift the "[withheld]" placeholder into a visually distinct
    # amber column so the attorney knows the collapse happened.
    def _src_header(source: str) -> str:
        if source == "[withheld]":
            return (
                "<th title='Clean-policy withheld: one or more privileged "
                "documents collapsed here' "
                "style='background:#fef3c7;color:#78350f;'>"
                "⚠ Withheld</th>"
            )
        label = _friendly_source_label(source)
        return f"<th title='{_escape(source)}'>{_escape(label)}</th>"
    header = "".join(_src_header(source) for source in sources)
    rows: list[str] = []
    detail_rows: list[str] = []
    for issue in issues:
        row_cells: list[str] = []
        issue_id = issue["id"]
        for source in sources:
            cell = cells.get(issue_id, {}).get(source, {})
            support = _safe_int(cell.get("supporting", 0))
            attack = _safe_int(cell.get("attacking", 0))
            total = _safe_int(cell.get("total", 0))
            alpha = 0.12 + (0.55 * total / max_total if total else 0.0)
            if support and attack:
                background = (
                    f"linear-gradient(90deg, rgba(19,122,78,{alpha:.2f}) 0%, "
                    f"rgba(19,122,78,{alpha:.2f}) 50%, rgba(171,52,40,{alpha:.2f}) 50%, "
                    f"rgba(171,52,40,{alpha:.2f}) 100%)"
                )
            elif support:
                background = f"rgba(19,122,78,{alpha:.2f})"
            elif attack:
                background = f"rgba(171,52,40,{alpha:.2f})"
            else:
                background = "rgba(148,163,184,0.08)"
            tooltip = f"support: {support}, attack: {attack}, total: {total}"
            row_cells.append(
                "<td class='matrix-cell' "
                f"style='background:{background}' title='{_escape(tooltip)}'>{total or ''}</td>"
            )
            if total:
                _src_label = (
                    "⚠ Withheld" if source == "[withheld]"
                    else _friendly_source_label(source)
                )
                detail_rows.append(
                    "<tr>"
                    f"<td>{_escape(issue.get('title') or issue_id)}</td>"
                    f"<td>{_escape(_src_label)}</td>"
                    f"<td>{support}</td>"
                    f"<td>{attack}</td>"
                    f"<td>{total}</td>"
                    "</tr>"
                )
        rows.append(
            "<tr>"
            f"<th title='{_escape(issue.get('title') or issue_id)}'>"
            f"{_escape(issue.get('title') or issue_id)}</th>"
            + "".join(row_cells)
            + "</tr>"
        )

    issue_totals_rows = "".join(
        "<tr>"
        f"<td>{_escape(issue.get('title') or issue.get('id') or 'Issue')}</td>"
        f"<td>{_safe_int(issue_totals.get(issue.get('id'), {}).get('supporting', 0))}</td>"
        f"<td>{_safe_int(issue_totals.get(issue.get('id'), {}).get('attacking', 0))}</td>"
        f"<td>{_safe_int(issue_totals.get(issue.get('id'), {}).get('supporting', 0)) + _safe_int(issue_totals.get(issue.get('id'), {}).get('attacking', 0))}</td>"
        "</tr>"
        for issue in issues
    )
    source_totals_rows = "".join(
        "<tr>"
        f"<td>{_escape('⚠ Withheld' if source == '[withheld]' else _friendly_source_label(source))}</td>"
        f"<td>{_safe_int(source_totals.get(source, {}).get('supporting', 0))}</td>"
        f"<td>{_safe_int(source_totals.get(source, {}).get('attacking', 0))}</td>"
        f"<td>{_safe_int(source_totals.get(source, {}).get('supporting', 0)) + _safe_int(source_totals.get(source, {}).get('attacking', 0))}</td>"
        "</tr>"
        for source in sources
    )

    return (
        "<div class='viz-shell'>"
        "<div class='viz-panel-title'>Support and attack by issue/source</div>"
        "<div class='viz-footnote'>Green = support, red = attack, split cell = both.</div>"
        "<div class='matrix-wrap matrix-wrap-heatmap'><table class='matrix-table evidence-matrix-table'><thead><tr><th>Issue</th>"
        + header
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
        + "<div class='viz-two-col'>"
        + "<div class='viz-panel'><div class='viz-subtitle'>Issue totals</div>"
        + "<div class='matrix-wrap'><table class='analytics-table'><thead><tr>"
        + "<th>Issue</th><th>Support</th><th>Attack</th><th>Total</th>"
        + "</tr></thead><tbody>"
        + (issue_totals_rows or "<tr><td colspan='4'>No issue totals available.</td></tr>")
        + "</tbody></table></div></div>"
        + "<div class='viz-panel'><div class='viz-subtitle'>Source totals</div>"
        + "<div class='matrix-wrap'><table class='analytics-table'><thead><tr>"
        + "<th>Source</th><th>Support</th><th>Attack</th><th>Total</th>"
        + "</tr></thead><tbody>"
        + (source_totals_rows or "<tr><td colspan='4'>No source totals available.</td></tr>")
        + "</tbody></table></div></div>"
        + "</div>"
        + "<div class='viz-panel'><div class='viz-subtitle'>Issue/source detail</div>"
        + "<div class='matrix-wrap'><table class='analytics-table'><thead><tr>"
        + "<th>Issue</th><th>Source</th><th>Support</th><th>Attack</th><th>Total</th>"
        + "</tr></thead><tbody>"
        + (("".join(detail_rows)) or "<tr><td colspan='5'>No linked evidence cells yet.</td></tr>")
        + "</tbody></table></div></div></div>"
    )


_PROOF_PANEL_LABELS = {
    "legal": {
        "issues_tracked": "Issues Reviewed",
        "avg_sufficiency": "Evidence Coverage",
        "gaps": "Evidence Gaps",
        "predicates": "Elements",
        "support": "Supporting",
        "attack": "Contrary",
        "balance": "Net Support",
        "advocacy_badge": "one-sided sources",
    },
    "finance": {
        "issues_tracked": "Theses Reviewed",
        "avg_sufficiency": "Evidence Coverage",
        "gaps": "Diligence Gaps",
        "predicates": "Criteria",
        "support": "Corroborating",
        "attack": "Contrary",
        "balance": "Net Support",
        "advocacy_badge": "management only",
    },
    "coding": {
        "issues_tracked": "Requirements Reviewed",
        "avg_sufficiency": "Evidence Coverage",
        "gaps": "Verification Gaps",
        "predicates": "Conditions",
        "support": "Confirming",
        "attack": "Contradicting",
        "balance": "Net Support",
        "advocacy_badge": "author only",
    },
    "academic_research": {
        "issues_tracked": "Questions Reviewed",
        "avg_sufficiency": "Evidence Coverage",
        "gaps": "Evidence Gaps",
        "predicates": "Criteria",
        "support": "Supporting",
        "attack": "Contradicting",
        "balance": "Net Support",
        "advocacy_badge": "single-source",
    },
    "biomedical": {
        "issues_tracked": "Endpoints Reviewed",
        "avg_sufficiency": "Evidence Coverage",
        "gaps": "Evidence Gaps",
        "predicates": "Criteria",
        "support": "Supporting",
        "attack": "Contrary",
        "balance": "Net Support",
        "advocacy_badge": "sponsor only",
    },
}


def _fmt_proof_state_panel(summary: dict, issues: list, domain: str = "legal") -> str:
    """Render the proof state analysis panel (SO-2 / SO-4)."""
    if not issues:
        return "<div class='viz-empty'>No evidence coverage data yet. Run an investigation first.</div>"

    labels = _PROOF_PANEL_LABELS.get(domain, _PROOF_PANEL_LABELS["legal"])
    total = summary.get("total_issues_tracked", 0)
    avg_suf = summary.get("avg_sufficiency", 0.0)
    by_status = summary.get("by_status", {})
    gap_count = summary.get("gap_count", 0)

    status_colors = {
        "sufficient": "#16a34a",
        "partial": "#d97706",
        "insufficient": "#dc2626",
        "none": "#94a3b8",
    }

    header = (
        "<div style='display:flex;gap:24px;flex-wrap:wrap;margin-bottom:16px;'>"
        f"<div style='text-align:center;'>"
        f"<div style='font-size:28px;font-weight:700;color:#1e293b;'>{total}</div>"
        f"<div style='font-size:11px;color:#6b7280;'>{_escape(labels['issues_tracked'])}</div></div>"
        f"<div style='text-align:center;'>"
        f"<div style='font-size:28px;font-weight:700;color:{'#16a34a' if avg_suf > 0.5 else '#d97706' if avg_suf > 0.25 else '#dc2626'};'>"
        f"{avg_suf:.0%}</div>"
        f"<div style='font-size:11px;color:#6b7280;'>{_escape(labels['avg_sufficiency'])}</div></div>"
        f"<div style='text-align:center;'>"
        f"<div style='font-size:28px;font-weight:700;color:#dc2626;'>{gap_count}</div>"
        f"<div style='font-size:11px;color:#6b7280;'>{_escape(labels['gaps'])}</div></div>"
    )
    for status, count in sorted(by_status.items()):
        color = status_colors.get(status, "#94a3b8")
        header += (
            f"<div style='text-align:center;'>"
            f"<div style='font-size:20px;font-weight:600;color:{color};'>{count}</div>"
            f"<div style='font-size:11px;color:#6b7280;'>{_escape(status.title())}</div></div>"
        )
    header += "</div>"

    rows_html = ""
    for ps in issues:
        issue_id = ps.get("issue_id", "?")
        issue_title = ps.get("issue_title") or issue_id
        sufficiency = max(0.0, min(1.0, _safe_float(ps.get("sufficiency", 0.0))))
        proof_status = ps.get("proof_status", "none")
        supporting = _safe_int(ps.get("supporting_count", 0))
        attacking = _safe_int(ps.get("attacking_count", 0))
        total_preds = _safe_int(ps.get("total_predicate_count", 0))
        satisfied_preds = _safe_int(ps.get("satisfied_predicate_count", 0))
        trust_support = _safe_float(ps.get("trust_weighted_support", 0.0))
        trust_attack = _safe_float(ps.get("trust_weighted_attack", 0.0))
        advocacy_only = ps.get("advocacy_only", False)

        status_color = status_colors.get(proof_status, "#94a3b8")
        suf_pct = sufficiency * 100

        display_title = _escape(issue_title[:40])
        pred_text = f"{satisfied_preds}/{total_preds}" if total_preds else "—"
        balance = trust_support - trust_attack
        balance_color = "#16a34a" if balance > 0.1 else "#dc2626" if balance < -0.1 else "#6b7280"

        adv_badge = ""
        if advocacy_only:
            adv_badge = (
                "<span style='margin-left:6px;padding:1px 5px;border-radius:8px;"
                "background:#fef3c7;color:#92400e;font-size:9px;font-weight:700;"
                f"text-transform:uppercase;'>{_escape(labels['advocacy_badge'])}</span>"
            )

        rows_html += (
            f"<tr>"
            f"<td style='max-width:220px;overflow:hidden;text-overflow:ellipsis;"
            f"white-space:nowrap;' title='{_escape(issue_title)}'>{display_title}{adv_badge}</td>"
            f"<td><span style='display:inline-block;padding:1px 8px;border-radius:8px;"
            f"background:{status_color}22;color:{status_color};font-size:11px;"
            f"font-weight:600;'>{_escape(proof_status)}</span></td>"
            f"<td style='width:120px;'>"
            f"<div style='position:relative;height:8px;background:#f3f4f6;border-radius:4px;"
            f"overflow:hidden;'>"
            f"<div style='position:absolute;inset:0 auto 0 0;width:{suf_pct:.1f}%;"
            f"background:{status_color};border-radius:4px;'></div></div>"
            f"<span style='font-size:10px;color:#6b7280;'>{sufficiency:.0%}</span></td>"
            f"<td style='text-align:center;'>{pred_text}</td>"
            f"<td style='text-align:center;color:#16a34a;'>{supporting}</td>"
            f"<td style='text-align:center;color:#dc2626;'>{attacking}</td>"
            f"<td style='text-align:center;color:{balance_color};font-weight:600;'>"
            f"{balance:+.2f}</td>"
            f"</tr>"
        )

    table = (
        "<div style='overflow-x:auto;'>"
        "<table style='width:100%;border-collapse:collapse;font-size:13px;'>"
        "<thead><tr style='border-bottom:2px solid #e2e8f0;text-align:left;'>"
        "<th style='padding:6px 8px;'>Issue</th>"
        "<th style='padding:6px 8px;'>Status</th>"
        "<th style='padding:6px 8px;'>Coverage</th>"
        f"<th style='padding:6px 8px;text-align:center;'>{_escape(labels['predicates'])}</th>"
        f"<th style='padding:6px 8px;text-align:center;'>{_escape(labels['support'])}</th>"
        f"<th style='padding:6px 8px;text-align:center;'>{_escape(labels['attack'])}</th>"
        f"<th style='padding:6px 8px;text-align:center;'>{_escape(labels['balance'])}</th>"
        "</tr></thead><tbody>"
        + rows_html
        + "</tbody></table></div>"
    )

    return f"<div class='viz-shell'>{header}{table}</div>"


def _fmt_communication_map_panel(graph: dict) -> str:
    actors = list(graph.get("actors", []) or [])
    documents = list(graph.get("documents", []) or [])
    edges = list(graph.get("actor_document_edges", []) or [])
    actor_actor_edges = list(graph.get("actor_actor_edges", []) or [])
    if not actors or not documents or not edges:
        return "<div class='viz-empty'>No communication graph available yet.</div>"

    actor_weights: dict[str, int] = defaultdict(int)
    document_weights: dict[str, int] = defaultdict(int)
    for edge in edges:
        actor_id = edge.get("actor_id")
        document_id = edge.get("document_id")
        count = _safe_int(edge.get("occurrence_count", 0))
        if actor_id:
            actor_weights[actor_id] += count
        if document_id:
            document_weights[document_id] += count

    actor_lookup = {actor.get("id"): actor for actor in actors}
    actor_ids = sorted(actor_weights, key=lambda key: actor_weights[key], reverse=True)[:12]
    doc_ids = sorted(document_weights, key=lambda key: document_weights[key], reverse=True)[:14]
    actor_index = {actor_id: idx for idx, actor_id in enumerate(actor_ids)}
    doc_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    filtered_edges = [
        edge
        for edge in edges
        if edge.get("actor_id") in actor_index and edge.get("document_id") in doc_index
    ]
    if not filtered_edges:
        return "<div class='viz-empty'>Communication graph has no dense connections to render.</div>"

    width = 920
    height = max(320, 80 + max(len(actor_ids), len(doc_ids)) * 44)
    actor_y = {
        actor_id: 50 + idx * ((height - 100) / max(1, len(actor_ids) - 1 or 1))
        for actor_id, idx in actor_index.items()
    }
    doc_y = {
        doc_id: 50 + idx * ((height - 100) / max(1, len(doc_ids) - 1 or 1))
        for doc_id, idx in doc_index.items()
    }
    max_edge = max(
        (_safe_int(edge.get("occurrence_count", 0)) for edge in filtered_edges),
        default=1,
    )
    max_actor = max((actor_weights[actor_id] for actor_id in actor_ids), default=1)
    max_doc = max((document_weights[doc_id] for doc_id in doc_ids), default=1)

    svg_lines: list[str] = []
    for edge in filtered_edges:
        actor_id = edge.get("actor_id")
        document_id = edge.get("document_id")
        count = _safe_int(edge.get("occurrence_count", 0))
        opacity = 0.18 + (0.60 * count / max_edge)
        stroke_width = 1.0 + (3.5 * count / max_edge)
        svg_lines.append(
            f"<line x1='170' y1='{actor_y[actor_id]:.1f}' x2='730' y2='{doc_y[document_id]:.1f}' "
            f"stroke='rgba(37,99,235,{opacity:.2f})' stroke-width='{stroke_width:.1f}' />"
        )

    svg_nodes: list[str] = []
    for actor_id in actor_ids:
        actor = actor_lookup.get(actor_id, {})
        radius = 10 + (14 * actor_weights[actor_id] / max_actor)
        svg_nodes.append(
            f"<circle cx='140' cy='{actor_y[actor_id]:.1f}' r='{radius:.1f}' class='comm-actor-node' />"
            f"<text x='28' y='{actor_y[actor_id] + 4:.1f}' class='comm-label comm-label-left'>"
            f"{_escape(actor.get('name') or actor_id)}</text>"
        )
    for doc_id in doc_ids:
        radius = 9 + (12 * document_weights[doc_id] / max_doc)
        svg_nodes.append(
            f"<circle cx='760' cy='{doc_y[doc_id]:.1f}' r='{radius:.1f}' class='comm-doc-node' />"
            f"<text x='788' y='{doc_y[doc_id] + 4:.1f}' class='comm-label'>"
            f"{_escape(doc_id)}</text>"
        )

    actor_doc_counts: dict[str, int] = defaultdict(int)
    document_actor_counts: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        actor_id = edge.get("actor_id")
        document_id = edge.get("document_id")
        if actor_id and document_id:
            actor_doc_counts[actor_id] += 1
            document_actor_counts[document_id].add(actor_id)

    actor_rows = "".join(
        "<div class='viz-list-row'>"
        f"<span>{_escape(actor_lookup.get(actor_id, {}).get('name') or actor_id)}</span>"
        f"<strong>{actor_weights[actor_id]} mentions | {actor_doc_counts.get(actor_id, 0)} linked docs</strong>"
        "</div>"
        for actor_id in sorted(actor_weights, key=lambda key: actor_weights[key], reverse=True)
    )
    doc_rows = "".join(
        "<div class='viz-list-row'>"
        f"<span>{_escape(doc_id)}</span>"
        f"<strong>{document_weights[doc_id]} links | {len(document_actor_counts.get(doc_id, set()))} actors</strong>"
        "</div>"
        for doc_id in sorted(document_weights, key=lambda key: document_weights[key], reverse=True)
    )
    pair_rows = "".join(
        "<details class='viz-detail'><summary>"
        f"{_escape(actor_lookup.get(edge.get('actor_a_id'), {}).get('name') or edge.get('actor_a_id') or '?')}"
        f" x "
        f"{_escape(actor_lookup.get(edge.get('actor_b_id'), {}).get('name') or edge.get('actor_b_id') or '?')}"
        f" | {_safe_int(edge.get('shared_documents', 0))} shared docs"
        "</summary>"
        + "<div class='viz-detail-block'><strong>Shared documents:</strong><ul>"
        + "".join(f"<li>{_escape(doc)}</li>" for doc in (edge.get("documents") or []))
        + "</ul></div></details>"
        for edge in sorted(
            actor_actor_edges,
            key=lambda item: _safe_int(item.get("shared_documents", 0)),
            reverse=True,
        )
    ) or "<div class='viz-empty'>No repeated actor co-appearance detected yet.</div>"
    edge_rows = "".join(
        "<tr>"
        f"<td>{_escape(actor_lookup.get(edge.get('actor_id'), {}).get('name') or edge.get('actor_id') or '?')}</td>"
        f"<td>{_escape(edge.get('document_id') or '(unknown)')}</td>"
        f"<td>{_safe_int(edge.get('occurrence_count', 0))}</td>"
        "</tr>"
        for edge in sorted(
            edges,
            key=lambda item: _safe_int(item.get('occurrence_count', 0)),
            reverse=True,
        )
    )
    visual_note = ""
    if len(actor_weights) > len(actor_ids) or len(document_weights) > len(doc_ids):
        visual_note = (
            "<div class='viz-footnote'>The SVG highlights the densest actor/document slice. "
            "Full actor, document, and edge detail is listed below.</div>"
        )

    return (
        "<div class='viz-shell'>"
        "<div class='viz-panel-title'>Actor/document communication map</div>"
        "<div class='viz-footnote'>Actors on the left, documents on the right, edge width = co-occurrence count.</div>"
        f"<svg class='comm-graph' viewBox='0 0 {width} {height}' role='img'>"
        + "".join(svg_lines)
        + "".join(svg_nodes)
        + "</svg>"
        + visual_note
        + "<div class='viz-two-col'>"
        + "<div class='viz-panel'><div class='viz-subtitle'>Actors</div>"
        + actor_rows
        + "</div>"
        + "<div class='viz-panel'><div class='viz-subtitle'>Documents</div>"
        + doc_rows
        + "</div></div>"
        + "<div class='viz-panel'>"
        + "<div class='viz-subtitle'>Strongest actor pairs</div>"
        + pair_rows
        + "</div>"
        + "<div class='viz-panel'><div class='viz-subtitle'>Actor/document edge detail</div>"
        + "<div class='matrix-wrap'><table class='analytics-table'><thead><tr>"
        + "<th>Actor</th><th>Document</th><th>Mentions</th>"
        + "</tr></thead><tbody>"
        + (edge_rows or "<tr><td colspan='3'>No actor/document links yet.</td></tr>")
        + "</tbody></table></div></div></div>"
    )


def _fmt_llm_analytics_panel(
    summary: dict,
    calls: list[dict],
    breakdown: Optional[dict] = None,
    anomalies: Optional[list[dict]] = None,
) -> str:
    if not summary and not calls and not breakdown:
        return "<div class='viz-empty'>No LLM analytics available yet.</div>"

    b_totals = (breakdown or {}).get("totals") or {}
    request_count = _safe_int(
        b_totals.get("request_count") or summary.get("request_count", 0)
    )
    total_cost = _safe_float(
        b_totals.get("estimated_cost_usd"),
        _safe_float(summary.get("estimated_cost_usd", 0.0)),
    )
    cache_hit_rate = b_totals.get("cache_hit_rate")
    success_rate = b_totals.get("success_rate")
    p95_latency = b_totals.get("p95_latency_ms")
    avg_latency_b = b_totals.get("avg_latency_ms")
    monthly_burn = _safe_float((breakdown or {}).get("estimated_monthly_burn_usd", 0.0))

    def _token_total(key: str) -> int:
        total = _safe_int(b_totals.get(key) or summary.get(key, 0))
        if total or not calls:
            return total
        return sum(_safe_int(call.get(key, 0)) for call in calls)

    input_tokens = _token_total("input_tokens")
    cache_tokens = _token_total("cache_read_tokens")
    tool_use_tokens = _token_total("tool_use_prompt_tokens")
    thinking_tokens = _token_total("thinking_tokens")
    output_tokens = _token_total("output_tokens")
    input_detail_parts: list[str] = []
    if cache_tokens:
        input_detail_parts.append(f"{cache_tokens:,} cached")
    if tool_use_tokens:
        input_detail_parts.append(f"{tool_use_tokens:,} tool")
    input_detail = " / ".join(input_detail_parts) if input_detail_parts else "Non-cached prompt tokens"

    # Fall back to per-call computation when breakdown is absent.
    total_latency = 0.0
    latency_count = 0
    sampled_failures = 0
    for call in calls:
        latency = _safe_float(call.get("latency_ms"), 0.0)
        if latency > 0:
            total_latency += latency
            latency_count += 1
        if not call.get("success", True):
            sampled_failures += 1
    avg_latency = (
        float(avg_latency_b) if avg_latency_b is not None
        else (total_latency / latency_count if latency_count else 0.0)
    )
    # Matter-wide failure count when breakdown is present; otherwise fall
    # back to the sampled recent-calls count (noted as such in the card).
    if success_rate is not None and request_count:
        fail_count = max(0, round(request_count * (1 - success_rate)))
        fail_source = "matter-wide"
    else:
        fail_count = sampled_failures
        fail_source = "recent rows"

    cards = [
        _metric_card("Calls", f"{request_count:,}", detail=f"{len(calls):,} recent rows"),
        _metric_card("Spend", _fmt_money(total_cost), tone="amber"),
        _metric_card(
            "Input",
            f"{input_tokens:,}",
            tone="blue",
            detail=input_detail,
        ),
        _metric_card(
            "Output",
            f"{output_tokens:,}",
            tone="blue",
            detail="Generated tokens",
        ),
        _metric_card(
            "Thinking",
            f"{thinking_tokens:,}",
            tone="blue",
            detail="Internal reasoning tokens",
        ),
        _metric_card(
            "Monthly burn",
            _fmt_money(monthly_burn),
            tone="amber",
            detail="Projected from last 7 days",
        ),
        _metric_card(
            "Cache hit rate",
            (f"{cache_hit_rate * 100:.1f}%" if cache_hit_rate is not None else "—"),
            tone="blue",
            detail="Cached input / prompt tokens",
        ),
        _metric_card(
            "P95 latency",
            (f"{int(p95_latency):,} ms" if p95_latency else f"{avg_latency:,.0f} ms avg"),
            tone="blue",
        ),
        _metric_card(
            "Failures",
            f"{fail_count:,}",
            tone="red",
            detail=(
                f"{(1 - success_rate) * 100:.1f}% of matter"
                if success_rate is not None else f"in {fail_source}"
            ),
        ),
    ]

    # Per-stage panel: prefer richer breakdown data when present.
    by_stage = (breakdown or {}).get("by_stage") or []
    if by_stage:
        stage_max = max((s.get("estimated_cost_usd") or 0) for s in by_stage)
        stage_rows = "".join(
            _bar_row(
                s.get("stage") or "unknown",
                _safe_float(s.get("estimated_cost_usd")),
                stage_max or 1.0,
                meta=(
                    f"{_fmt_money(s.get('estimated_cost_usd'))} · "
                    f"{_safe_int(s.get('request_count')):,} calls · "
                    + (
                        f"cache {int((s.get('cache_hit_rate') or 0) * 100)}% · "
                        if s.get("cache_hit_rate") is not None else ""
                    )
                    + (
                        f"{_safe_int(s.get('avg_latency_ms')):,} ms avg"
                        if s.get("avg_latency_ms") else "no latency"
                    )
                ),
                tone="amber",
            )
            for s in by_stage
        )
    else:
        stage_costs: dict[str, float] = defaultdict(float)
        stage_calls: dict[str, int] = defaultdict(int)
        for call in calls:
            label = call.get("usage_label") or "unknown"
            stage_costs[label] += _safe_float(call.get("estimated_cost_usd", 0.0))
            stage_calls[label] += 1
        stage_max = max(stage_costs.values(), default=0.0)
        stage_rows = "".join(
            _bar_row(
                stage, cost, stage_max or 1.0,
                meta=f"{_fmt_money(cost)} · {stage_calls[stage]} calls",
                tone="amber",
            )
            for stage, cost in sorted(
                stage_costs.items(), key=lambda item: item[1], reverse=True,
            )
        ) or "<div class='viz-empty'>No per-stage cost data yet.</div>"

    # Per-tier panel.
    by_tier = (breakdown or {}).get("by_tier") or []
    if by_tier:
        tier_max = max((t.get("estimated_cost_usd") or 0) for t in by_tier)
        model_rows = "".join(
            _bar_row(
                str(t.get("model_tier") or "unknown").upper(),
                _safe_float(t.get("estimated_cost_usd")),
                tier_max or 1.0,
                meta=(
                    f"{_fmt_money(t.get('estimated_cost_usd'))} · "
                    f"{_safe_int(t.get('request_count')):,} calls · "
                    + (
                        f"cache {int((t.get('cache_hit_rate') or 0) * 100)}%"
                        if t.get("cache_hit_rate") is not None else "no cache data"
                    )
                ),
                tone="blue",
            )
            for t in by_tier
        )
    else:
        model_costs: dict[str, float] = defaultdict(float)
        for call in calls:
            model_costs[call.get("model_tier") or "unknown"] += _safe_float(
                call.get("estimated_cost_usd", 0.0)
            )
        model_max = max(model_costs.values(), default=0.0)
        model_rows = "".join(
            _bar_row(
                str(model).upper(), cost, model_max or 1.0,
                meta=_fmt_money(cost), tone="blue",
            )
            for model, cost in sorted(
                model_costs.items(), key=lambda item: item[1], reverse=True,
            )
        ) or "<div class='viz-empty'>No model usage yet.</div>"

    # Anomalies section.
    anomalies_block = ""
    if anomalies:
        def _z_cell(val: Any) -> str:
            return f"{val}σ" if val is not None else "—"
        anomaly_rows = "".join(
            "<tr>"
            f"<td>{_escape(a.get('created_at') or '')}</td>"
            f"<td>{_escape(a.get('usage_label') or 'unknown')}</td>"
            f"<td>{_escape((a.get('model_tier') or 'unknown').upper())}</td>"
            f"<td>{_fmt_money(a.get('estimated_cost_usd', 0.0))}</td>"
            f"<td>{_escape(_z_cell(a.get('cost_z')))}</td>"
            f"<td>{_safe_int(a.get('latency_ms', 0)):,} ms</td>"
            f"<td>{_escape(_z_cell(a.get('latency_z')))}</td>"
            f"<td>{_fmt_money(a.get('baseline_cost', 0.0))}</td>"
            "</tr>"
            for a in anomalies
        )
        anomalies_block = (
            "<div class='viz-panel'>"
            "<div class='viz-panel-title'>Cost &amp; latency anomalies</div>"
            "<div class='viz-panel-sub'>Calls more than 2σ above their tier mean — investigation targets.</div>"
            "<div class='matrix-wrap'><table class='analytics-table'><thead><tr>"
            "<th>Time</th><th>Stage</th><th>Tier</th><th>Cost</th><th>Cost z</th>"
            "<th>Latency</th><th>Lat z</th><th>Tier avg cost</th>"
            "</tr></thead><tbody>"
            + anomaly_rows
            + "</tbody></table></div></div>"
        )

    table_rows = "".join(
        "<tr>"
        f"<td>{_escape(call.get('created_at') or '')}</td>"
        f"<td>{_escape(call.get('usage_label') or 'unknown')}</td>"
        f"<td>{_escape((call.get('model_tier') or 'unknown').upper())}</td>"
        f"<td>{_safe_int(call.get('input_tokens', 0)):,}</td>"
        f"<td>{_safe_int(call.get('cache_read_tokens', 0)):,}</td>"
        f"<td>{_safe_int(call.get('tool_use_prompt_tokens', 0)):,}</td>"
        f"<td>{_safe_int(call.get('thinking_tokens', 0)):,}</td>"
        f"<td>{_safe_int(call.get('output_tokens', 0)):,}</td>"
        f"<td>{_safe_int(call.get('latency_ms', 0)):,} ms</td>"
        f"<td>{_fmt_money(call.get('estimated_cost_usd', 0.0))}</td>"
        f"<td>{_escape(call.get('run_id') or '')}</td>"
        f"<td>{'ok' if call.get('success', True) else _escape(call.get('error_kind') or 'error')}</td>"
        "</tr>"
        for call in calls
    )

    pricing_source = _escape(
        (breakdown or {}).get("pricing_source")
        or summary.get("pricing_source", "")
    )

    return (
        "<div class='viz-shell'>"
        "<div class='viz-card-grid'>"
        + "".join(cards)
        + "</div>"
        + "<div class='viz-two-col'>"
        + "<div class='viz-panel'><div class='viz-panel-title'>Cost by stage</div>"
        + stage_rows
        + "</div>"
        + "<div class='viz-panel'><div class='viz-panel-title'>Cost by tier</div>"
        + model_rows
        + "</div>"
        + "</div>"
        + anomalies_block
        + "<div class='viz-panel'><div class='viz-panel-title'>Recent calls</div>"
        + "<div class='matrix-wrap'><table class='analytics-table'><thead><tr>"
        + "<th>Time</th><th>Stage</th><th>Tier</th><th>Input</th><th>Cache</th>"
        + "<th>Tool</th><th>Think</th><th>Output</th><th>Latency</th><th>Cost</th><th>Run</th><th>Status</th></tr></thead><tbody>"
        + (table_rows or "<tr><td colspan='12'>No recent calls.</td></tr>")
        + "</tbody></table></div>"
        + (
            f"<div class='viz-footnote'>Pricing source: <a href='{pricing_source}' target='_blank'>Google Gemini API pricing</a></div>"
            if pricing_source
            else ""
        )
        + "</div></div>"
    )


def _fmt_quant_panel(
    payment_recon: dict,
    invoice_chain: list,
    amount_conflicts: list,
    damages: list,
) -> str:
    has_recon = bool(payment_recon and payment_recon.get("invoiced") is not None)
    has_invoice_chain = bool(invoice_chain)
    has_amount_conflicts = bool(amount_conflicts)
    has_damages = bool(damages)
    if not has_recon and not has_invoice_chain and not has_amount_conflicts and not has_damages:
        return "<div class='viz-empty'>No quantitative facts extracted yet.</div>"

    cards: list[str] = []
    if has_recon:
        cards.extend(
            [
                _metric_card("Invoiced", _fmt_money_short(payment_recon.get("invoiced", 0.0))),
                _metric_card("Paid", _fmt_money_short(payment_recon.get("paid", 0.0)), tone="green"),
                _metric_card(
                    "Disputed",
                    _fmt_money_short(payment_recon.get("disputed", 0.0)),
                    tone="amber",
                ),
                _metric_card(
                    "Net exposure",
                    _fmt_money_short(payment_recon.get("exposure", 0.0)),
                    tone="red",
                ),
            ]
        )

    max_amount = max(
        (_safe_float(row.get("claimed_amount", 0.0)) for row in damages),
        default=0.0,
    )
    damage_rows = "".join(
        _bar_row(
            row.get("component") or "(uncategorized)",
            _safe_float(row.get("claimed_amount", 0.0)),
            max_amount or 1.0,
            meta=(
                f"{_fmt_money_short(row.get('claimed_amount', 0.0))} | "
                f"{_safe_int(row.get('source_count', 0))} sources | "
                f"{len(row.get('conflicts', []) or [])} conflicts"
            ),
            tone="red",
        )
        for row in sorted(
            damages,
            key=lambda item: _safe_float(item.get("claimed_amount", 0.0)),
            reverse=True,
        )
    ) or "<div class='viz-empty'>No damages waterfall available.</div>"

    category_rows = ""
    if has_recon and isinstance(payment_recon.get("by_category"), dict):
        category_rows = "".join(
            "<tr>"
            f"<td>{_escape(subject_type)}</td>"
            f"<td>{_fmt_money_short(values.get('total', 0.0))}</td>"
            f"<td>{_safe_int(values.get('count', 0))}</td>"
            "</tr>"
            for subject_type, values in payment_recon.get("by_category", {}).items()
        )

    source_span_rows = ""
    if has_recon:
        source_span_rows = "".join(
            "<tr>"
            f"<td>{_escape(span.get('subject_type') or '')}</td>"
            f"<td>{_escape(span.get('subject_id') or '')}</td>"
            f"<td>{_fmt_money_short(span.get('amount', 0.0))}</td>"
            f"<td>{_escape(_fmt_span_label(span.get('span_id')))}</td>"
            "</tr>"
            for span in payment_recon.get("source_spans", []) or []
        )

    invoice_rows = "".join(
        "<tr>"
        f"<td>{_escape(invoice.get('invoice_id') or '(unlabeled)')}</td>"
        f"<td>{_fmt_money_short(invoice.get('invoiced', 0.0))}</td>"
        f"<td>{_fmt_money_short(invoice.get('paid', 0.0))}</td>"
        f"<td>{_fmt_money_short(invoice.get('outstanding', 0.0))}</td>"
        f"<td>{'<br>'.join(_escape(_fmt_span_label(span.get('span_id'))) for span in (invoice.get('source_spans') or []))}</td>"
        "</tr>"
        for invoice in invoice_chain
    )

    damage_details = "".join(
        (
            "<details class='viz-detail'><summary>"
            f"{_escape(row.get('component') or '(uncategorized)')} | "
            f"{_fmt_money_short(row.get('claimed_amount', 0.0))} | "
            f"{_safe_int(row.get('source_count', 0))} source entries"
            "</summary>"
            + (
                "<div class='viz-detail-block'><strong>Conflicting values:</strong><ul>"
                + "".join(f"<li>{_escape(str(conflict))}</li>" for conflict in (row.get("conflicts") or []))
                + "</ul></div>"
                if row.get("conflicts")
                else ""
            )
            + "<div class='viz-detail-block'><strong>Source entries:</strong><ul>"
            + "".join(
                "<li>"
                f"{_fmt_money_short(entry.get('amount_value', 0.0))}"
                + (
                    f" · {_escape(_fmt_span_label(entry.get('span_id')))}"
                    if _fmt_span_label(entry.get("span_id")) != "—"
                    else ""
                )
                + f"<br>{_escape(entry.get('raw_text') or '')}"
                "</li>"
                for entry in (row.get("amounts") or [])
            )
            + "</ul></div></details>"
        )
        for row in damages
    )

    amount_conflict_details = "".join(
        "<details class='viz-detail'><summary>"
        f"{_escape(conflict.get('subject_type') or 'amount')} | "
        f"{_escape(conflict.get('subject_id') or '(unlabeled)')} | "
        f"{_escape(conflict.get('currency') or '')}"
        "</summary>"
        + "<div class='viz-detail-block'><strong>Values:</strong> "
        + ", ".join(_fmt_money_short(value) for value in (conflict.get("values") or []))
        + "</div>"
        + (
            "<div class='viz-detail-block'><strong>Raw texts:</strong><ul>"
            + "".join(
                f"<li>{_escape(text)}</li>"
                for text in ((conflict.get("raw_texts") or []) if isinstance(conflict.get("raw_texts"), list) else [])
            )
            + "</ul></div>"
            if conflict.get("raw_texts")
            else ""
        )
        + "</details>"
        for conflict in amount_conflicts
    )

    return (
        "<div class='viz-shell'>"
        + ("<div class='viz-card-grid'>" + "".join(cards) + "</div>" if cards else "")
        + "<div class='viz-panel'><div class='viz-panel-title'>Damages waterfall</div>"
        + damage_rows
        + "</div>"
        + (
            "<div class='viz-two-col'>"
            + "<div class='viz-panel'><div class='viz-panel-title'>Category totals</div>"
            + (
                "<div class='matrix-wrap'><table class='analytics-table'><thead><tr>"
                "<th>Category</th><th>Total</th><th>Facts</th></tr></thead><tbody>"
                + category_rows
                + "</tbody></table></div>"
                if category_rows
                else "<div class='viz-empty'>No category totals available.</div>"
            )
            + "</div>"
            + "<div class='viz-panel'><div class='viz-panel-title'>Invoice reconciliation</div>"
            + (
                "<div class='matrix-wrap'><table class='analytics-table'><thead><tr>"
                "<th>Invoice</th><th>Invoiced</th><th>Paid</th><th>Outstanding</th><th>Source spans</th>"
                "</tr></thead><tbody>"
                + invoice_rows
                + "</tbody></table></div>"
                if invoice_rows
                else "<div class='viz-empty'>No invoice chain available.</div>"
            )
            + "</div></div>"
        )
        + (
            "<div class='viz-panel'><div class='viz-panel-title'>Payment grounding</div>"
            + (
                "<div class='matrix-wrap'><table class='analytics-table'><thead><tr>"
                "<th>Type</th><th>Subject</th><th>Amount</th><th>Location</th>"
                "</tr></thead><tbody>"
                + source_span_rows
                + "</tbody></table></div>"
                if source_span_rows
                else "<div class='viz-empty'>No source span grounding available.</div>"
            )
            + "</div>"
            if has_recon
            else ""
        )
        + (
            "<div class='viz-panel'><div class='viz-panel-title'>Damages source detail</div>"
            + (damage_details or "<div class='viz-empty'>No component detail available.</div>")
            + "</div>"
            if has_damages
            else ""
        )
        + (
            "<div class='viz-panel'><div class='viz-panel-title'>Amount conflicts</div>"
            + (amount_conflict_details or "<div class='viz-empty'>No amount conflicts detected.</div>")
            + "</div>"
            if has_amount_conflicts
            else ""
        )
        + "</div>"
    )


def _fmt_overview(data: dict) -> str:
    if not data:
        return "No matter loaded."
    stats = data.get("stats", {})
    so = data.get("so_metrics", {})
    llm = stats.get("llm", {}) if isinstance(stats.get("llm"), dict) else {}
    llm_totals = llm.get("totals", {}) if isinstance(llm, dict) else {}
    lines = [
        "## Matter Overview",
        f"**Assertions:** {stats.get('assertion_count', 0)}  |  "
        f"**Issues:** {stats.get('open_issue_count', 0)}  |  "
        f"**Gaps:** {stats.get('open_gap_count', 0)}",
        f"**Actors:** {stats.get('actor_count', 0)}  |  "
        f"**Quant facts:** {stats.get('quant_fact_count', 0)}  |  "
        f"**Pending clarifications:** {stats.get('pending_clarifications', 0)}",
    ]
    if llm_totals.get("request_count", 0):
        lines.append(
            f"**LLM Calls:** {llm_totals.get('request_count', 0)}  |  "
            f"**Tokens by type:** {_llm_token_breakdown(llm_totals)}  |  "
            f"**Est. Cost:** ${float(llm_totals.get('estimated_cost_usd', 0.0) or 0.0):.4f}"
        )
        by_tier = llm_totals.get("by_tier", {})
        if isinstance(by_tier, dict) and by_tier:
            tier_parts = []
            for tier_name, tier_data in by_tier.items():
                tier_parts.append(
                    f"{tier_name.upper()}: {tier_data.get('requests', 0)} calls / "
                    f"${float(tier_data.get('estimated_cost_usd', 0.0) or 0.0):.4f}"
                )
            lines.append("**By Tier:** " + "  |  ".join(tier_parts[:4]))
        last_run = llm.get("last_run", {})
        if isinstance(last_run, dict) and last_run.get("request_count", 0):
            lines.append(
                f"**Last Run LLM:** {last_run.get('request_count', 0)} calls  |  "
                f"{_llm_token_breakdown(last_run)}  |  "
                f"${float(last_run.get('estimated_cost_usd', 0.0) or 0.0):.4f}"
            )

    # SO metrics
    cov = so.get("issue_coverage_avg")
    reuse = so.get("reuse_rate")
    struct = so.get("assertion_structure_rate")
    src = so.get("source_role_known_rate")
    lines.append(
        f"\n**Reuse rate:** {f'{reuse:.1%}' if reuse is not None else '—'}  |  "
        f"**Structured facts:** {f'{struct:.1%}' if struct is not None else '—'}  |  "
        f"**Issue coverage:** {f'{cov:.1%}' if cov is not None else '—'}  |  "
        f"**Source calibration:** {f'{src:.1%}' if src is not None else '—'}"
    )

    # Weakest issues
    weakest = data.get("weakest_issues", [])
    if weakest:
        lines.append("\n### Weakest Issues (proof gaps)")
        for issue in weakest[:5]:
            title = issue.get("title") or issue.get("id", "?")
            frac = issue.get("coverage_fraction")
            gap = " ⚠️" if issue.get("has_proof_gap") else ""
            lines.append(f"- **{title}** — {_fmt_coverage(frac)}{gap}")

    # Top gaps
    top_gaps = data.get("top_gaps", [])
    if top_gaps:
        lines.append("\n### Open Gaps")
        for gap in top_gaps[:5]:
            desc = gap.get("description") or gap.get("gap_type", "—")
            lines.append(f"- {desc}")

    # Pending clarifications
    clarifications = data.get("pending_clarifications", [])
    if clarifications:
        lines.append("\n### Pending Clarifications")
        for c in clarifications[:5]:
            q = c.get("question_text") or c.get("question", "—")
            lines.append(f"- {q}")

    return "\n".join(lines)


def _fmt_issues(issues: list) -> str:
    if not issues:
        return "No open issues."

    # Build lookup for tree rendering
    by_id = {i["id"]: i for i in issues}
    # Sort by depth then coverage so tree structure is visible
    sorted_issues = sorted(issues, key=lambda i: (i.get("depth", 0), i.get("coverage_fraction", 0)))

    # Render as indented list with coverage bars
    _proof_icons = {
        "strong": "🟢", "partial": "🟡", "weak": "🟠",
        "gap": "🔴", "none": "⚪",
    }
    lines = []
    for iss in sorted_issues:
        depth = iss.get("depth", 0)
        indent = "  " * depth
        title = iss.get("title") or iss.get("id", "?")
        cov = iss.get("coverage_fraction", 0)
        proof = iss.get("proof_status", "none")
        icon = _proof_icons.get(proof, "⚪")
        sup = iss.get("supporting_count", 0)
        atk = iss.get("attacking_count", 0)
        burden = iss.get("burden_side") or ""
        burden_str = f" [{burden}]" if burden else ""

        # Coverage bar: 10 chars wide
        filled = int(cov * 10)
        bar = "█" * filled + "░" * (10 - filled)

        line = f"{indent}{icon} **{title}**{burden_str}  {bar} {cov:.0%}"
        details = []
        if sup:
            details.append(f"{sup} supporting")
        if atk:
            details.append(f"{atk} attacking")
        pred_cnt = iss.get("predicate_count", 0)
        if pred_cnt:
            details.append(f"{pred_cnt} elements")
        contested = iss.get("contested_predicates", 0)
        blocked = iss.get("blocked_predicates", 0)
        if contested:
            details.append(f"⚠️ {contested} disputed")
        if blocked:
            details.append(f"🚫 {blocked} blocked")

        # Subtree info for parent issues
        subtree_cov = iss.get("subtree_coverage")
        if subtree_cov is not None:
            weakest = iss.get("weakest_leaf_coverage", 0)
            size = iss.get("subtree_size", 0)
            details.append(f"subtree: {subtree_cov:.0%} avg, weakest {weakest:.0%}, {size} issues")

        if details:
            line += f"  *({', '.join(details)})*"
        lines.append(line)

    return "\n".join(lines)


_TRUST_ICONS = {
    "OPERATIVE": "🟢", "AUTHORITATIVE": "🔵", "PROCEDURAL": "⚪",
    "INFORMAL": "🟡", "DRAFT": "🟡", "POST_HOC": "🟠", "ADVOCACY": "🔴",
}

_DOMAIN_SOURCE_LABELS: dict[str, dict[str, str]] = {
    "finance": {
        "operative": "Audited Filing", "authoritative": "Regulatory",
        "procedural": "Compliance", "informal": "Market Commentary",
        "draft": "Preliminary", "advocacy": "Analyst Opinion",
        "post_hoc": "Retrospective", "unknown": "Unclassified",
    },
    "coding": {
        "operative": "Specification", "authoritative": "Documentation",
        "procedural": "Standard", "informal": "Comment/Discussion",
        "draft": "RFC/Proposal", "advocacy": "Opinion/Blog",
        "post_hoc": "Post-mortem", "unknown": "Unclassified",
    },
    "academic_research": {
        "operative": "Primary Source", "authoritative": "Systematic Review",
        "procedural": "Protocol", "informal": "Grey Literature",
        "draft": "Preprint", "advocacy": "Editorial/Opinion",
        "post_hoc": "Retrospective", "unknown": "Unclassified",
    },
    "biomedical": {
        "operative": "Primary Evidence", "authoritative": "Meta-Analysis",
        "procedural": "Guideline", "informal": "Correspondence",
        "draft": "Preprint", "advocacy": "Expert Opinion",
        "post_hoc": "Retrospective", "unknown": "Unclassified",
    },
}


def _domain_source_label(role: str, domain: str) -> str:
    profile_map = _DOMAIN_SOURCE_LABELS.get(domain, {})
    return profile_map.get(role.lower(), role.replace("_", " ").title())


_DOMAIN_SPEECH_ACT_LABELS: dict[str, dict[str, str]] = {
    "finance": {
        "alleged": "Reported", "argued": "Projected", "denied": "Disputed",
        "admitted": "Disclosed", "ordered": "Mandated", "performed": "Executed",
        "paid": "Settled", "requested": "Filed", "threatened": "Warned",
        "promised": "Committed", "estimated": "Estimated", "calculated": "Modeled",
        "observed": "Measured", "testified": "Attested", "stipulated": "Agreed",
        "amended": "Restated", "waived": "Waived", "terminated": "Closed",
        "inferred": "Derived", "operative": "Contractual", "extracted": "Extracted",
    },
    "coding": {
        "alleged": "Claimed", "argued": "Proposed", "denied": "Rejected",
        "admitted": "Confirmed", "ordered": "Required", "performed": "Implemented",
        "paid": "Delivered", "requested": "Requested", "threatened": "Flagged",
        "promised": "Planned", "estimated": "Estimated", "calculated": "Computed",
        "observed": "Traced", "testified": "Documented", "stipulated": "Specified",
        "amended": "Patched", "waived": "Deferred", "terminated": "Deprecated",
        "inferred": "Inferred", "operative": "Defined", "extracted": "Extracted",
    },
    "academic_research": {
        "alleged": "Hypothesized", "argued": "Argued", "denied": "Refuted",
        "admitted": "Accepted", "ordered": "Prescribed", "performed": "Conducted",
        "paid": "Funded", "requested": "Proposed", "threatened": "Cautioned",
        "promised": "Predicted", "estimated": "Estimated", "calculated": "Computed",
        "observed": "Observed", "testified": "Reported", "stipulated": "Defined",
        "amended": "Corrected", "waived": "Excluded", "terminated": "Retracted",
        "inferred": "Derived", "operative": "Methodological", "extracted": "Extracted",
    },
    "biomedical": {
        "alleged": "Reported", "argued": "Hypothesized", "denied": "Contradicted",
        "admitted": "Acknowledged", "ordered": "Prescribed", "performed": "Administered",
        "paid": "Fulfilled", "requested": "Recommended", "threatened": "Warned",
        "promised": "Indicated", "estimated": "Estimated", "calculated": "Modeled",
        "observed": "Observed", "testified": "Documented", "stipulated": "Specified",
        "amended": "Revised", "waived": "Excluded", "terminated": "Discontinued",
        "inferred": "Inferred", "operative": "Labeled", "extracted": "Extracted",
    },
}


def _domain_speech_act_label(act: str, domain: str) -> str:
    profile_map = _DOMAIN_SPEECH_ACT_LABELS.get(domain, {})
    return profile_map.get(act.lower(), act.replace("_", " ").title())


def _trust_icon(role: str) -> str:
    """Return a colored dot indicating source trust level."""
    return _TRUST_ICONS.get(role.upper(), "⚪") if role else "⚪"


_VERIFICATION_PILL = {
    "verified":  ("✓ Verified",   "#15803d", "#dcfce7"),
    "candidate": ("Needs review", "#92400e", "#fef3c7"),
    "stale":     ("Pulled back",  "#475569", "#e2e8f0"),
    "rejected":  ("Rejected",     "#991b1b", "#fee2e2"),
}


def _verification_pill(status: str | None) -> str:
    """Attorney-readable pill for verification_status."""
    label, fg, bg = _VERIFICATION_PILL.get(
        (status or "candidate").lower(),
        ("Needs review", "#92400e", "#fef3c7"),
    )
    return (
        f"<span style='display:inline-block;padding:1px 8px;border-radius:10px;"
        f"background:{bg};color:{fg};font-size:11px;font-weight:600;"
        f"letter-spacing:0.02em;'>{label}</span>"
    )


_DOMAIN_BELIEF_LABELS: dict[str, dict[str, str]] = {
    "finance": {
        "alleged": "Reported", "argued": "Projected", "admitted": "Confirmed",
        "operative": "Established", "performed": "Executed",
        "not_performed": "Unexecuted", "disputed": "Challenged",
        "superseded": "Revised", "withdrawn": "Retracted",
        "inferred": "Derived", "resolved": "Settled", "unknown": "Undetermined",
    },
    "coding": {
        "alleged": "Claimed", "argued": "Asserted", "admitted": "Acknowledged",
        "operative": "Established", "performed": "Implemented",
        "not_performed": "Not Implemented", "disputed": "Contested",
        "superseded": "Deprecated", "withdrawn": "Removed",
        "inferred": "Inferred", "resolved": "Resolved", "unknown": "Unknown",
    },
    "academic_research": {
        "alleged": "Hypothesized", "argued": "Argued", "admitted": "Accepted",
        "operative": "Established", "performed": "Demonstrated",
        "not_performed": "Not Demonstrated", "disputed": "Disputed",
        "superseded": "Superseded", "withdrawn": "Retracted",
        "inferred": "Derived", "resolved": "Confirmed", "unknown": "Unknown",
    },
    "biomedical": {
        "alleged": "Reported", "argued": "Argued", "admitted": "Acknowledged",
        "operative": "Established", "performed": "Observed",
        "not_performed": "Not Observed", "disputed": "Contested",
        "superseded": "Superseded", "withdrawn": "Withdrawn",
        "inferred": "Inferred", "resolved": "Confirmed", "unknown": "Undetermined",
    },
}


def _domain_belief_label(state_raw: str, domain: str) -> str:
    profile_map = _DOMAIN_BELIEF_LABELS.get(domain, {})
    return profile_map.get(state_raw.lower(), state_raw.replace("_", " ").title())


_LEGAL_BELIEF_DESCRIPTIONS: dict[str, str] = {
    "alleged": "claimed but not proven",
    "argued": "legal argument, not fact",
    "admitted": "acknowledged by opposing party",
    "operative": "from a binding document",
    "performed": "action that occurred",
    "not_performed": "action did not occur",
    "disputed": "parties disagree",
    "superseded": "replaced by later document",
    "withdrawn": "retracted by source",
    "inferred": "deduced from other facts",
    "resolved": "settled or decided",
    "unknown": "status cannot be determined",
}

_DOMAIN_BELIEF_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "finance": {
        "alleged": "reported but unverified",
        "argued": "projected or forecast",
        "admitted": "confirmed by counterparty",
        "operative": "from audited filing",
        "performed": "transaction executed",
        "not_performed": "transaction not executed",
        "disputed": "challenged or contested",
        "superseded": "revised in later filing",
        "withdrawn": "retracted by issuer",
        "inferred": "derived from data",
        "resolved": "settled or finalized",
        "unknown": "status undetermined",
    },
    "coding": {
        "alleged": "claimed in documentation",
        "argued": "asserted without test coverage",
        "admitted": "acknowledged by maintainer",
        "operative": "established in codebase",
        "performed": "implemented and tested",
        "not_performed": "not implemented",
        "disputed": "contested in review",
        "superseded": "deprecated by newer version",
        "withdrawn": "removed from codebase",
        "inferred": "inferred from behavior",
        "resolved": "fixed or resolved",
        "unknown": "status unknown",
    },
    "academic_research": {
        "alleged": "hypothesized",
        "argued": "argued in literature",
        "admitted": "accepted by community",
        "operative": "established finding",
        "performed": "demonstrated experimentally",
        "not_performed": "not demonstrated",
        "disputed": "disputed in literature",
        "superseded": "superseded by later study",
        "withdrawn": "retracted by authors",
        "inferred": "derived from analysis",
        "resolved": "confirmed by replication",
        "unknown": "status unknown",
    },
    "biomedical": {
        "alleged": "reported in study",
        "argued": "argued in publication",
        "admitted": "acknowledged by investigators",
        "operative": "established in guidelines",
        "performed": "observed in trial",
        "not_performed": "not observed",
        "disputed": "contested in literature",
        "superseded": "superseded by later evidence",
        "withdrawn": "withdrawn by authors",
        "inferred": "inferred from data",
        "resolved": "confirmed by meta-analysis",
        "unknown": "status undetermined",
    },
}

_CORRECTION_STATES = [
    "alleged", "argued", "admitted", "operative", "performed",
    "not_performed", "disputed", "superseded", "withdrawn",
    "inferred", "resolved", "unknown",
]


def _correction_dropdown_choices(domain: str = "legal") -> list[tuple[str, str]]:
    descriptions = _DOMAIN_BELIEF_DESCRIPTIONS.get(domain, _LEGAL_BELIEF_DESCRIPTIONS)
    choices = []
    for state in _CORRECTION_STATES:
        label = _domain_belief_label(state, domain)
        desc = descriptions.get(state, "")
        display = f"{label} — {desc}" if desc else label
        choices.append((display, state))
    return choices


def _fmt_assertions(assertions: list, domain: str = "legal") -> str:
    if not assertions:
        return "<div class='viz-empty'>No assertions recorded yet.</div>"
    rows_html = ""
    for a in assertions:
        assertion_id = _escape(a.get("id", "?"))
        prop = _escape(a.get("proposition_text") or "")
        state_raw = a.get("belief_state") or "—"
        state_cls = state_raw.lower() if state_raw != "—" else "unknown"
        state_label = _domain_belief_label(state_raw, domain) if state_raw != "—" else "—"
        state_cell = (
            f"<span class='belief-pill belief-{state_cls}'>{_escape(state_label)}</span>"
            if state_raw != "—" else "—"
        )
        conf = f"{float(a.get('confidence', 0)):.2f}" if a.get("confidence") is not None else "—"
        src_roles = a.get("source_roles", [])
        if len(src_roles) > 1:
            best = min(
                src_roles,
                key=lambda r: list(_TRUST_ICONS).index(r.upper())
                if r.upper() in _TRUST_ICONS else 99,
            )
            icon = _trust_icon(best)
            labels = [_domain_source_label(r, domain) for r in src_roles]
            src = _escape(f"MULTI[{','.join(labels)}]")
        elif src_roles:
            icon = _trust_icon(src_roles[0])
            src = _escape(_domain_source_label(src_roles[0], domain))
        else:
            src_role = a.get("source_role") or a.get("primary_source_role") or "—"
            icon = _trust_icon(src_role)
            src = _escape(_domain_source_label(src_role, domain) if src_role != "—" else "—")
        _speech_raw = a.get("speech_act") or a.get("primary_speech_act") or "—"
        speech = _escape(_domain_speech_act_label(_speech_raw, domain) if _speech_raw != "—" else "—")
        # Verification pill as a compact prefix on the proposition
        # cell so the new responsive table layout keeps room for
        # human-review signal alongside the clickable-row fact-edit.
        vpill = _verification_pill(a.get("verification_status"))
        reviewer = a.get("reviewed_by_kind")
        review_meta_html = ""
        if reviewer and a.get("verification_status") in ("verified", "rejected"):
            review_meta_html = (
                f" <span style='font-size:10px;color:#6b7280;'>"
                f"by {_escape(reviewer)}"
                + (f" — {_escape(a.get('rejection_reason') or '')}"
                   if a.get("rejection_reason") else "")
                + "</span>"
            )
        prop_cell = f"{vpill} {prop}{review_meta_html}".lstrip()
        rows_html += (
            f"<tr class='assertions-row' onclick='irysSelectFact(\"{assertion_id}\")' style='cursor:pointer'>"
            f"<td style='text-align:center'>{icon}</td>"
            f"<td>{prop_cell}</td>"
            f"<td>{state_cell}</td>"
            f"<td style='text-align:right'>{conf}</td>"
            f"<td>{src}</td>"
            f"<td>{speech}</td>"
            f"<td><code style='font-size:10px'>{assertion_id}</code></td>"
            "</tr>"
        )
    return (
        "<div class='matrix-wrap'>"
        "<table class='analytics-table' style='table-layout:fixed;width:100%'>"
        "<colgroup>"
        "<col style='width:3%'>"
        "<col style='width:39%'>"
        "<col style='width:10%'>"
        "<col style='width:8%'>"
        "<col style='width:13%'>"
        "<col style='width:12%'>"
        "<col style='width:15%'>"
        "</colgroup>"
        "<thead><tr>"
        "<th></th><th>Proposition</th><th>State</th>"
        "<th>Conf</th><th>Source</th><th>Speech</th><th>ID</th>"
        "</tr></thead>"
        "<tbody>" + rows_html + "</tbody>"
        "</table></div>"
    )


_ASSUMPTION_STATUS_ICONS = {
    "provisional": "⏳", "confirmed": "✅", "invalidated": "❌",
}

def _fmt_assumptions(assumptions: list) -> str:
    if not assumptions:
        return "*No assumptions recorded yet.*"
    lines = ["These are the working assumptions Irys is using. "
             "If any are wrong, the conclusions that depend on them may change.\n"]
    for a in assumptions:
        status = a.get("status", "provisional")
        icon = _ASSUMPTION_STATUS_ICONS.get(status, "⏳")
        stmt = a.get("statement") or "?"
        cond = a.get("invalidation_condition") or ""
        line = f"- {icon} **{stmt}**"
        if cond:
            line += f"  \n  *Would be invalidated if: {cond}*"
        rationale = a.get("rationale") or ""
        if rationale:
            line += f"  \n  *Rationale: {rationale}*"
        lines.append(line)
    return "\n".join(lines)


def _fmt_gaps(gaps: list, clarifications: list) -> str:
    parts = []
    if gaps:
        parts.append("### Open Gaps")
        for g in gaps:
            desc = g.get("description") or g.get("gap_type", "?")
            mat = g.get("materiality_score") or g.get("materiality") or ""
            mat_str = f" [materiality: {mat:.2f}]" if isinstance(mat, (int, float)) else (f" [{mat}]" if mat else "")
            parts.append(f"- {desc}{mat_str}")
    if clarifications:
        parts.append("\n### Pending Clarifications")
        for c in clarifications:
            q = c.get("question_text") or c.get("question", "?")
            impact = c.get("expected_impact") or ""
            parts.append(f"- **{q}**" + (f"\n  *Impact: {impact}*" if impact else ""))
    return "\n".join(parts) if parts else "No open gaps or clarifications."


_REVIEW_BUCKET_LABELS = {
    0: ("Gap blocker", "#b91c1c"),       # red — gap-blocked issue
    1: ("Disputed", "#b45309"),          # amber — contradicted
    2: ("Supports a claim", "#2563eb"),  # blue — issue-linked
    3: ("Element of proof", "#0d9488"),  # teal — predicate
    4: ("Number", "#7c3aed"),            # purple — quant
    5: ("Citation / Authority", "#6b7280"),  # grey — authority
    6: ("Other", "#94a3b8"),             # light grey
}


def _fmt_privilege_banner(audience: str) -> str:
    """Visible indicator of the active privilege mode (UI-6). Rendered
    at the top of the sidebar so a reviewer cannot forget they've opted
    into internal mode before sharing a screen / exporting."""
    if audience == "internal":
        return (
            "<div style='padding:8px 12px;border-radius:6px;"
            "background:#fee2e2;color:#991b1b;font-size:12px;"
            "margin-bottom:8px;border-left:3px solid #b91c1c;font-weight:600;'>"
            "Internal mode — privileged content is unredacted. "
            "Do not share this view with external parties."
            "</div>"
        )
    return (
        "<div style='padding:8px 12px;border-radius:6px;"
        "background:#ecfdf5;color:#065f46;font-size:12px;"
        "margin-bottom:8px;border-left:3px solid #059669;'>"
        "Clean mode — privileged content shows as [withheld]. "
        "Safe for external review."
        "</div>"
    )


def _fmt_review_count_badge(total: int, bucket_counts: dict[int, int]) -> str:
    """Render the sidebar review-queue badge from precomputed counts.
    Shared by load_review_count_badge and load_post_review_snapshot so
    both paths produce identical HTML (OPT-2b)."""
    if total == 0:
        return (
            "<div style='padding:8px 12px;border-radius:6px;"
            "background:#dcfce7;color:#14532d;font-size:12px;"
            "margin-bottom:8px;border-left:3px solid #15803d;'>"
            "✓ All findings reviewed."
            "</div>"
        )
    lines: list[str] = []
    for bucket in sorted(bucket_counts):
        label, color = _REVIEW_BUCKET_LABELS.get(
            bucket, _REVIEW_BUCKET_LABELS[6],
        )
        count = bucket_counts[bucket]
        lines.append(
            f"<span style='display:inline-block;padding:1px 6px;"
            f"border-radius:4px;background:{color};color:white;"
            f"font-size:10px;font-weight:700;margin-right:4px;'>"
            f"{label}: {count}</span>"
        )
    return (
        "<div style='padding:8px 12px;border-radius:6px;"
        "background:#fef3c7;color:#78350f;font-size:12px;"
        "margin-bottom:8px;border-left:3px solid #b45309;'>"
        f"<strong>{total} finding(s) need review.</strong> "
        "Open the <em>Review Inbox</em> below to verify or reject."
        f"<div style='margin-top:6px;'>{' '.join(lines)}</div>"
        "</div>"
    )


def _review_bucket_badge(bucket: int, score: float) -> str:
    label, color = _REVIEW_BUCKET_LABELS.get(
        int(bucket or 6), _REVIEW_BUCKET_LABELS[6],
    )
    meter = ""
    if score and score > 0:
        meter = f" <span style='opacity:0.7;font-size:11px;'>· materiality {score:.0%}</span>"
    return (
        f"<span style='display:inline-block;padding:2px 8px;border-radius:10px;"
        f"background:{color};color:white;font-size:11px;font-weight:600;"
        f"text-transform:uppercase;letter-spacing:0.03em;'>{label}</span>{meter}"
    )


def _fmt_review_queue(queue: list[dict]) -> str:
    """Render the prioritized review queue in attorney-readable form.

    No raw IDs, no internal store names. Each row shows:
      - bucket badge (proof-critical / contradicted / issue-linked / etc.)
      - the finding text or citation
      - the kind tag (Fact / Number / Citation / Predicate)
      - a copyable target id truncated for selection — hidden text, not shown
    """
    if not queue:
        return (
            "<div class='viz-empty'>"
            "No findings need review. The matter model is fully reviewed "
            "or has no AI-extracted material yet."
            "</div>"
        )
    rows_html = []
    for row in queue:
        bucket = row.get("priority_bucket", 6)
        score = row.get("priority_score", 0.0)
        kind = row.get("target_kind", "unknown")
        # Pick the right context field per kind.
        text = (
            row.get("proposition_text")
            or row.get("quant_raw_text")
            or row.get("authority_citation")
            or row.get("predicate_description")
            or "(no preview)"
        )
        kind_pretty = {
            "assertion": "Fact",
            "assertion_occurrence": "Raw utterance",
            "evidence_edge": "Evidence link",
            "issue_predicate": "Proof element",
            "quant_fact": "Number",
            "authority": "Citation / Authority",
            "document_card": "Document classification",
        }.get(kind, kind.replace("_", " ").title())
        truncated = _truncate(text, 180)
        rows_html.append(
            f"<div style='padding:10px 12px;border-left:3px solid #e5e7eb;"
            f"margin-bottom:8px;background:#f9fafb;border-radius:0 6px 6px 0;'>"
            f"<div style='margin-bottom:4px;'>"
            f"{_review_bucket_badge(bucket, score)}"
            f"<span style='margin-left:10px;font-size:12px;color:#6b7280;"
            f"text-transform:uppercase;letter-spacing:0.03em;'>{kind_pretty}</span>"
            f"</div>"
            f"<div style='color:#1f2937;line-height:1.45;'>{_escape(truncated)}</div>"
            f"</div>"
        )
    return (
        "<div style='max-height:540px;overflow-y:auto;padding:4px;'>"
        f"<div style='margin-bottom:12px;font-size:13px;color:#6b7280;'>"
        f"{len(queue)} item(s) awaiting review — ranked by proof impact."
        f"</div>"
        + "\n".join(rows_html)
        + "</div>"
    )


def _fmt_source_drawer(
    target_kind: str,
    target_id: str,
    provenance_rows: list[dict],
    verification_events: list[dict],
) -> str:
    """Attorney-readable "where did this come from + what's been done
    to it" drawer. Combines P0.1 provenance (AI write trail) with
    P0.3 verification events (human review history)."""
    if not provenance_rows and not verification_events:
        return (
            "<div class='viz-empty'>No source or review history for this finding. "
            "Try an item created during a live investigation.</div>"
        )
    sections: list[str] = []

    # Attorney-readable labels for reviewers and status transitions.
    reviewer_labels = {
        "user": "You",
        "attorney": "Attorney",
        "system": "Irys",
        "import": "Bulk import",
    }
    status_labels = {
        "verified": "verified",
        "rejected": "rejected",
        "stale": "pulled back for re-check",
        "candidate": "flagged for review",
    }

    # --- Source (where it came from) ---
    if provenance_rows:
        lines = [
            "<div style='margin-bottom:16px;'>"
            "<div style='font-size:12px;color:#6b7280;text-transform:uppercase;"
            "letter-spacing:0.04em;font-weight:600;margin-bottom:6px;'>"
            "Where this came from</div>"
        ]
        for ev in provenance_rows[:5]:
            doc = ev.get("source_document_ref") or "—"
            span_raw = ev.get("source_span_id")
            span_status = ev.get("source_span_status") or "unknown"
            when = (ev.get("created_at") or "")[:10]  # YYYY-MM-DD, date only
            tier_label = _provenance_tier_label(ev)
            span_html = ""
            if span_raw:
                span_html = (
                    f" <span style='font-size:11px;color:#6b7280;'>"
                    f"· section {_escape(_truncate(span_raw, 20))}</span>"
                )
            elif span_status == "missing":
                span_html = (
                    " <span style='color:#b45309;font-size:11px;'>"
                    "(whole document, no specific section)</span>"
                )
            lines.append(
                "<div style='padding:8px 10px;background:#f9fafb;"
                "border-left:3px solid #3b82f6;border-radius:0 6px 6px 0;"
                "margin-bottom:6px;'>"
                f"<div style='color:#1f2937;font-weight:500;'>"
                f"{_escape(doc)}{span_html}</div>"
                f"<div style='color:#6b7280;font-size:12px;margin-top:2px;'>"
                f"Captured via <strong>{_escape(tier_label)}</strong>"
                f" on {_escape(when)}</div>"
                "</div>"
            )
        if len(provenance_rows) > 5:
            lines.append(
                f"<div style='font-size:11px;color:#9ca3af;margin-top:4px;'>"
                f"+ {len(provenance_rows) - 5} more extraction event(s)</div>"
            )
        lines.append("</div>")
        sections.append("".join(lines))

    # --- Review history (what's been done to it) ---
    if verification_events:
        lines = [
            "<div>"
            "<div style='font-size:12px;color:#6b7280;text-transform:uppercase;"
            "letter-spacing:0.04em;font-weight:600;margin-bottom:6px;'>"
            "Review history</div>"
        ]
        for ev in verification_events[:20]:
            new_s = ev.get("new_status") or "—"
            reviewer_key = ev.get("reviewed_by_kind") or ev.get("actor_kind") or "system"
            actor_label = reviewer_labels.get(
                reviewer_key, reviewer_key.replace("_", " ").title(),
            )
            note = ev.get("note") or ev.get("review_note") or ""
            reason = ev.get("rejection_reason") or ""
            when = (ev.get("created_at") or "")[:16].replace("T", " ")
            tint = {
                "verified": "#15803d",
                "rejected": "#991b1b",
                "stale": "#475569",
                "candidate": "#92400e",
            }.get(new_s, "#1f2937")
            action_label = status_labels.get(
                new_s, new_s.replace("_", " "),
            )
            detail_bits: list[str] = []
            if reason:
                detail_bits.append(f"Reason: {_escape(reason)}")
            if note and note != "ai_rewrite_revival":
                detail_bits.append(f"Note: {_escape(note)}")
            detail_html = ""
            if detail_bits:
                detail_html = (
                    "<div style='color:#6b7280;font-size:12px;margin-top:2px;'>"
                    + " · ".join(detail_bits) + "</div>"
                )
            lines.append(
                "<div style='padding:8px 10px;background:#f9fafb;"
                f"border-left:3px solid {tint};border-radius:0 6px 6px 0;"
                "margin-bottom:6px;'>"
                f"<div style='color:#1f2937;'>"
                f"<span style='color:{tint};font-weight:600;'>{_escape(action_label).capitalize()}</span>"
                f" by <strong>{_escape(actor_label)}</strong>"
                f" <span style='font-weight:400;color:#6b7280;font-size:12px;'>"
                f"· {_escape(when)}</span></div>"
                f"{detail_html}</div>"
            )
        lines.append("</div>")
        sections.append("".join(lines))

    return (
        "<div style='max-height:420px;overflow-y:auto;'>"
        + "\n".join(sections)
        + "</div>"
    )


def _review_queue_choices(queue: list[dict]) -> list[tuple[str, str]]:
    """Build dropdown (label, value) pairs. Label is attorney-readable;
    value is the internal "kind:id" handle used by verify/reject."""
    choices = []
    for row in queue:
        kind = row.get("target_kind") or ""
        tid = row.get("target_id") or ""
        text = (
            row.get("proposition_text")
            or row.get("quant_raw_text")
            or row.get("authority_citation")
            or row.get("predicate_description")
            or "(no preview)"
        )
        kind_pretty = {
            "assertion": "Fact",
            "evidence_edge": "Evidence link",
            "issue_predicate": "Element of proof",
            "quant_fact": "Number",
            "authority": "Citation / Authority",
            "document_card": "Document classification",
            "assertion_occurrence": "Quoted utterance",
        }.get(kind, kind.replace("_", " ").title())
        label = f"{kind_pretty}: {_truncate(text, 90)}"
        choices.append((label, f"{kind}:{tid}"))
    return choices


def _fmt_steering(actions: list) -> str:
    """Format get_ledger_steering_surface() output as actionable recommendations."""
    if not actions:
        return "No steering recommendations available."
    lines = ["### Steering Recommendations\n"]
    for a in actions:
        action_type = a.get("action_type", "unknown")
        description = a.get("description", "")
        rationale = a.get("rationale", "")
        priority = a.get("priority", "")
        priority_str = f" **[{priority.upper()}]**" if priority else ""
        lines.append(f"**{action_type}**{priority_str}: {description}")
        if rationale:
            lines.append(f"  > {rationale}")
        # Show action params useful for the UI (issue_id, gap_id, assertion_id)
        params = a.get("params", {})
        if params:
            param_str = " | ".join(f"`{k}: {str(v)}`" for k, v in params.items() if v)
            lines.append(f"  *Params: {param_str}*")
        lines.append("")
    return "\n".join(lines)


def _fmt_quant(payment_recon: dict, damages: list) -> str:
    """Format quant reconciliation and damages waterfall (SO-6)."""
    parts = []

    # Payment reconciliation — keys match reconcile_payment_chain() output:
    # invoiced, paid, disputed, exposure, currency
    if payment_recon and payment_recon.get("invoiced") is not None:
        inv = payment_recon.get("invoiced", 0)
        paid = payment_recon.get("paid", 0)
        disputed = payment_recon.get("disputed", 0)
        exp = payment_recon.get("exposure", 0)
        currency = payment_recon.get("currency", "USD")
        parts.append("### Payment Reconciliation")
        parts.append(f"| Metric | Amount ({currency}) |")
        parts.append("|--------|--------|")
        parts.append(f"| Total Invoiced | {inv:,.2f}" if isinstance(inv, (int, float)) else f"| Total Invoiced | {inv}")
        parts.append(f"| Total Paid | {paid:,.2f}" if isinstance(paid, (int, float)) else f"| Total Paid | {paid}")
        if disputed:
            parts.append(f"| Disputed | {disputed:,.2f}" if isinstance(disputed, (int, float)) else f"| Disputed | {disputed}")
        parts.append(f"| **Net Exposure** | **{exp:,.2f}**" if isinstance(exp, (int, float)) else f"| Net Exposure | {exp}")

    # Damages waterfall
    if damages:
        parts.append("\n### Damages Waterfall")
        parts.append("| Component | Claimed | Sources | Conflicts |")
        parts.append("|-----------|---------|---------|-----------|")
        for d in damages:
            comp = d.get("component") or "(uncategorised)"
            amt = d.get("claimed_amount", 0)
            amt_str = f"{amt:,.2f}" if isinstance(amt, (int, float)) else str(amt)
            srcs = d.get("source_count", 0)
            conflicts = len(d.get("conflicts", []))
            conflict_str = f"⚠️ {conflicts}" if conflicts else "—"
            parts.append(f"| {comp} | {amt_str} | {srcs} | {conflict_str} |")

    return "\n".join(parts) if parts else "No quantitative facts extracted yet. Run an investigation first."


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------


class AppState:
    """Global app state — designed for single-user dev use.

    NOTE: thinking_log, citations_log, update_queue, and is_running are
    instance-level (not session-scoped). For a multi-user deployment,
    these should be moved into gr.State session objects. For a local dev
    tool with one user, this is acceptable.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._backend: Optional[InProcessBackend] = None
        self.thinking_log: list[str] = []
        self.citations_log: list[str] = []
        self.update_queue: queue.Queue = queue.Queue()
        self.is_running = False
        self.final_output = ""
        self.final_diagnostics = ""
        self.current_matter_id: Optional[str] = None
        self.current_run_id: Optional[str] = None
        self.current_repo_path: Optional[str] = None
        self.current_research_mode: str = "simple"
        # UI-6: privilege mode. "clean" redacts privileged content in
        # timeline / evidence matrix / etc. (safe default). "internal"
        # bypasses the gate for attorney-only workspaces. Changed via
        # the sidebar toggle; a visual banner surfaces the active mode.
        self.policy_audience: str = "clean"
        self.session_turns: list[dict[str, str]] = []
        self._last_resume_error: Optional[str] = None  # set by do_resume() on failure
        self._irys_ref = None  # weak ref to active Irys instance for stop
        # Stop event: set by stop_investigation() to signal early-stop before
        # the first engine step fires (when run_session may not exist yet).
        self._stop_event = threading.Event()

    def backend(self) -> InProcessBackend:
        if self._backend is None:
            self._backend = InProcessBackend(api_key=self.api_key)
        return self._backend

    def _make_on_step(self, update_q: queue.Queue, thinking: list):
        """Return a thinking-step callback that also captures run_id early."""
        def _callback(step):
            icon = STEP_ICONS.get(
                step.step_type.name if hasattr(step.step_type, "name") else str(step.step_type),
                "•",
            )
            line = f"{icon} {step.display}"
            thinking.append(line)
            update_q.put(("thinking", line))
            # Capture run_id + matter_id on the first step if not yet known
            if self.current_run_id is None and self._irys_ref is not None:
                try:
                    engine = self._irys_ref._engine
                    if engine and engine._matter_model:
                        _mm = engine._matter_model
                        _run_row = _mm.db.execute(
                            "SELECT id FROM run_session WHERE matter_id=? AND status='running'"
                            " AND (objective IS NULL OR objective NOT IN"
                            " ('manual_flush','background_flush'))"
                            " ORDER BY started_at DESC LIMIT 1",
                            (_mm.matter_id,),
                        ).fetchone()
                        runs = [dict(_run_row)] if _run_row else []
                        if runs:
                            self.current_run_id = runs[0]["id"]
                            self.current_matter_id = engine._matter_model.matter_id
                except Exception:
                    pass
        return _callback

    def _run_thread(
        self,
        query: str,
        repo_path: str,
        update_q: queue.Queue,
        thinking: list,
        citations: list,
        research_mode: str,
        conversation_history: Optional[list[dict[str, str]]] = None,
        resume_matter_id: Optional[str] = None,
        resume_run_id: Optional[str] = None,
    ):
        """Run investigation in a background thread via InProcessBackend.

        Delegates entirely to InProcessBackend.run_investigation_thread() so the
        Run tab goes through the UIBackend rather than calling irys internals directly.
        """
        backend = self.backend()
        if not isinstance(backend, InProcessBackend):
            update_q.put(("error",
                "Run tab requires InProcessBackend. "
                "HttpBackend does not support local streaming investigations — "
                "it connects to a running FastAPI service that requires S3-backed repos."))
            return
        backend.run_investigation_thread(
            query,
            repo_path,
            update_q,
            thinking,
            citations,
            research_mode=research_mode,
            conversation_history=conversation_history,
            resume_matter_id=resume_matter_id,
            resume_run_id=resume_run_id,
            on_irys_created=lambda irys: setattr(self, "_irys_ref", irys),
            on_step=self._make_on_step(update_q, thinking),
            set_current_run_id=lambda rid: setattr(self, "current_run_id", rid),
            set_current_matter_id=lambda mid: setattr(self, "current_matter_id", mid),
            set_final_output=lambda o: setattr(self, "final_output", o),
            stop_event=self._stop_event,
        )

    def stream_investigation(
        self, query: str, repo_path: str, research_mode: str = "deep"
    ) -> Generator[tuple, None, None]:
        """Generator yielding (output, trace, citations, status, matter_id) tuples."""
        # Per-call local state (mitigates global AppState race for concurrent calls)
        call_thinking: list[str] = []
        call_citations: list[str] = []
        call_queue: queue.Queue = queue.Queue()
        self.thinking_log = call_thinking
        self.citations_log = call_citations
        self.update_queue = call_queue
        self.final_output = ""
        self.final_diagnostics = ""
        previous_run_id = self.current_run_id
        previous_matter_id = self.current_matter_id
        previous_repo_path = self.current_repo_path
        try:
            resolved_repo_path = str(pathlib.Path(repo_path).resolve()) if repo_path else None
        except Exception:
            resolved_repo_path = repo_path
        user_query = query.strip()
        if previous_repo_path and resolved_repo_path and previous_repo_path != resolved_repo_path:
            self.session_turns = []
        conversation_history = (
            _build_conversation_history(self.session_turns)
            if (
                user_query
                and resolved_repo_path
                and previous_repo_path == resolved_repo_path
                and self.session_turns
            )
            else []
        )
        normalized_mode = normalize_research_mode(research_mode)
        should_resume_follow_up = bool(
            previous_run_id
            and previous_matter_id
            and previous_repo_path
            and resolved_repo_path
            and previous_repo_path == resolved_repo_path
        )
        self.current_run_id = None
        self.current_repo_path = resolved_repo_path
        self.current_research_mode = normalized_mode
        # Create a fresh per-call stop event so that stopping one investigation
        # cannot interfere with a subsequent one (shared-event reuse race).
        # stop_investigation() always sets self._stop_event, which after this
        # line points to THIS call's event — not a previous call's.
        self._stop_event = threading.Event()

        if not repo_path or not __import__("pathlib").Path(repo_path).exists():
            yield ("", "", "", "❌ Invalid repository path", "")
            return
        if not user_query:
            yield ("", "", "", "❌ Please enter a query", "")
            return
        if not self.api_key:
            yield ("", "", "", "❌ No GEMINI_API_KEY. Set the env var or pass --api-key", "")
            return

        self.is_running = True
        thread = threading.Thread(
            target=self._run_thread,
            args=(
                user_query or query,
                repo_path,
                call_queue,
                call_thinking,
                call_citations,
                normalized_mode,
                conversation_history or None,
                previous_matter_id if should_resume_follow_up else None,
                previous_run_id if should_resume_follow_up else None,
            ),
            daemon=True,
        )
        thread.start()

        start_time = time.time()

        # The streaming loop reads exclusively from per-call local variables
        # (call_thinking, call_citations, call_queue) so that a second concurrent
        # call cannot overwrite this generator's data sources.  self.* fields are
        # written for the stop button (single-user dev tool, see class docstring).
        while self.is_running:
            try:
                update_type, data = call_queue.get(timeout=0.5)
                elapsed = time.time() - start_time

                if update_type == "thinking":
                    status = (
                        f"⏳  {_fmt_research_mode_label(self.current_research_mode)} | "
                        f"{elapsed:.0f}s | {len(call_thinking)} steps | {len(call_citations)} citations"
                    )
                    yield (
                        _build_chat_messages(
                            self.session_turns,
                            pending_user=user_query,
                            pending_assistant="*Investigating...*",
                        ),
                        "\n".join(call_thinking),
                        "\n".join(call_citations) or "—",
                        status,
                        self.current_matter_id or "—",
                    )

                elif update_type == "complete":
                    self.is_running = False
                    # NOTE: current_run_id is NOT cleared here — kept for post-run
                    # redirect and panel refreshes. Cleared only when a new investigation
                    # starts (at the top of stream_investigation).
                    state = data
                    elapsed = time.time() - start_time
                    summary = state.get_summary()
                    metrics = summary.get("metrics", {})
                    true_rate = metrics.get("true_reuse_rate")
                    rate_str = f"{true_rate:.1%}" if true_rate is not None else "—"
                    llm_calls = int(metrics.get("llm_request_count", 0) or 0)
                    llm_cost = float(metrics.get("llm_estimated_cost_usd", 0.0) or 0.0)
                    mode_label = _fmt_research_mode_label(getattr(state, "research_mode", None))
                    _route_chip = _fmt_route_chip(state)
                    status = (
                        f"✅  {mode_label} | {elapsed:.0f}s | "
                        f"Docs: {state.documents_read} ({state.documents_from_cache} cached) | "
                        f"Reuse: {rate_str} | "
                        f"LLM: {llm_calls} calls / ${llm_cost:.4f}"
                        + (f" | {_route_chip}" if _route_chip else "")
                    )
                    # Replace raw thinking trace with structured ledger events so the
                    # Reasoning Trace tab shows durable, matter-model-backed content.
                    structured_trace = "\n".join(call_thinking)  # fallback
                    if self.current_matter_id and self.current_run_id:
                        try:
                            events = _run_async(
                                self.backend().get_run_events(
                                    self.current_matter_id,
                                    self.current_run_id,
                                )
                            )
                            if events:
                                formatted = [
                                    _fmt_ledger_event(ev) for ev in events
                                ]
                                lines = [f for f in formatted if f is not None]
                                if lines:
                                    structured_trace = "\n".join(lines)
                        except Exception:
                            pass  # keep raw thinking fallback
                    # Cheap-path families (read, query, trace, etc.)
                    # don't run the full engine and therefore don't
                    # stream thinking or emit ledger events. Fall back
                    # to state.thinking_steps (seeded by api._seed_cheap_path_trace)
                    # so the Reasoning Trace tab isn't blank on those flows.
                    if not structured_trace.strip():
                        structured_trace = _fmt_thinking_steps_fallback(state) or structured_trace
                    main_output, diagnostics_output = _split_run_output_sections(self.final_output)
                    self.final_output = main_output or self.final_output
                    self.final_diagnostics = diagnostics_output
                    if state.status == "completed" and self.final_output:
                        self.session_turns.append({"query": user_query, "answer": self.final_output})
                        self.session_turns = self.session_turns[-_SESSION_TURN_LIMIT:]
                    supporting_text = "\n".join(call_citations) or "â€”"
                    if self.final_diagnostics:
                        supporting_text = (
                            "## Run Diagnostics & Safeguards\n\n"
                            + self.final_diagnostics
                            + ("\n\n---\n\n" + supporting_text if supporting_text and supporting_text != "â€”" else "")
                        )
                    yield (
                        _build_chat_messages(self.session_turns),
                        structured_trace,
                        "\n".join(call_citations) or "—",
                        status,
                        self.current_matter_id or "—",
                    )
                    return

                elif update_type == "error":
                    self.is_running = False
                    yield (
                        _build_chat_messages(
                            self.session_turns,
                            pending_user=user_query,
                            pending_assistant=f"❌ {data}",
                        ),
                        "\n".join(call_thinking),
                        "",
                        f"❌ {data}",
                        "—",
                    )
                    return

            except queue.Empty:
                if self.is_running:
                    elapsed = time.time() - start_time
                    status = (
                        f"⏳  {_fmt_research_mode_label(self.current_research_mode)} | "
                        f"{elapsed:.0f}s | {len(call_thinking)} steps"
                    )
                    yield (
                        _build_chat_messages(
                            self.session_turns,
                            pending_user=user_query,
                            pending_assistant="*Investigating...*",
                        ),
                        "\n".join(call_thinking),
                        "\n".join(call_citations) or "—",
                        status,
                        self.current_matter_id or "—",
                    )

        thread.join(timeout=2)

    def stream_investigation_session(
        self, query: str, repo_path: str, research_mode: str = "deep"
    ) -> Generator[tuple, None, None]:
        """Session-scoped run stream with full trace and follow-up continuity."""
        call_thinking: list[str] = []
        call_citations: list[str] = []
        call_queue: queue.Queue = queue.Queue()
        self.thinking_log = call_thinking
        self.citations_log = call_citations
        self.update_queue = call_queue
        self.final_output = ""
        self.final_diagnostics = ""

        previous_run_id = self.current_run_id
        previous_matter_id = self.current_matter_id
        previous_repo_path = self.current_repo_path
        try:
            resolved_repo_path = str(pathlib.Path(repo_path).resolve()) if repo_path else None
        except Exception:
            resolved_repo_path = repo_path

        user_query = query.strip()
        if previous_repo_path and resolved_repo_path and previous_repo_path != resolved_repo_path:
            self.session_turns = []

        conversation_history = (
            _build_conversation_history(self.session_turns)
            if (
                user_query
                and resolved_repo_path
                and previous_repo_path == resolved_repo_path
                and self.session_turns
            )
            else []
        )
        normalized_mode = normalize_research_mode(research_mode)
        should_resume_follow_up = bool(
            previous_run_id
            and previous_matter_id
            and previous_repo_path
            and resolved_repo_path
            and previous_repo_path == resolved_repo_path
        )

        self.current_run_id = None
        self.current_repo_path = resolved_repo_path
        self.current_research_mode = normalized_mode
        self._stop_event = threading.Event()

        if not repo_path or not __import__("pathlib").Path(repo_path).exists():
            yield ("", "", "", "Invalid repository path", "")
            return
        if not user_query:
            yield ("", "", "", "Please enter a query", "")
            return
        if not self.api_key:
            yield ("", "", "", "No GEMINI_API_KEY. Set the env var or pass --api-key", "")
            return

        self.is_running = True
        thread = threading.Thread(
            target=self._run_thread,
            args=(
                user_query or query,
                repo_path,
                call_queue,
                call_thinking,
                call_citations,
                normalized_mode,
                conversation_history or None,
                previous_matter_id if should_resume_follow_up else None,
                previous_run_id if should_resume_follow_up else None,
            ),
            daemon=True,
        )
        thread.start()

        start_time = time.time()

        while self.is_running:
            try:
                update_type, data = call_queue.get(timeout=0.5)
                elapsed = time.time() - start_time

                if update_type == "thinking":
                    status = (
                        f"⏳  {_fmt_research_mode_label(self.current_research_mode)} | "
                        f"{elapsed:.0f}s | {len(call_thinking)} steps | {len(call_citations)} citations"
                    )
                    yield (
                        _build_chat_messages(
                            self.session_turns,
                            pending_user=user_query,
                            pending_assistant="*Investigating...*",
                        ),
                        "\n".join(call_thinking),
                        "\n".join(call_citations) or "—",
                        status,
                        self.current_matter_id or "—",
                    )
                    continue

                if update_type == "complete":
                    self.is_running = False
                    state = data
                    elapsed = time.time() - start_time
                    summary = state.get_summary()
                    metrics = summary.get("metrics", {})
                    true_rate = metrics.get("true_reuse_rate")
                    rate_str = f"{true_rate:.1%}" if true_rate is not None else "—"
                    llm_calls = int(metrics.get("llm_request_count", 0) or 0)
                    llm_cost = float(metrics.get("llm_estimated_cost_usd", 0.0) or 0.0)
                    mode_label = _fmt_research_mode_label(getattr(state, "research_mode", None))
                    _route_chip = _fmt_route_chip(state)
                    status = (
                        f"✅  {mode_label} | {elapsed:.0f}s | "
                        f"Docs: {state.documents_read} ({state.documents_from_cache} cached) | "
                        f"Reuse: {rate_str} | "
                        f"LLM: {llm_calls} calls / ${llm_cost:.4f}"
                        + (f" | {_route_chip}" if _route_chip else "")
                    )

                    structured_trace = "\n".join(call_thinking)
                    if self.current_matter_id and self.current_run_id:
                        try:
                            events = _run_async(
                                self.backend().get_run_events(
                                    self.current_matter_id,
                                    self.current_run_id,
                                )
                            )
                            if events:
                                lines = [
                                    formatted
                                    for formatted in (_fmt_ledger_event(ev) for ev in events)
                                    if formatted is not None
                                ]
                                if lines:
                                    structured_trace = "\n".join(lines)
                        except Exception:
                            pass
                    # adv#13 Finding #3: the session-stream path
                    # drifted from the other stream path. Cheap-path
                    # families seed state.thinking_steps via
                    # api._seed_cheap_path_trace but never stream
                    # thinking and never start a run — so without
                    # this fallback the Reasoning Trace tab stays
                    # blank on read/query/trace/steer/etc. answers.
                    if not structured_trace.strip():
                        structured_trace = _fmt_thinking_steps_fallback(state) or structured_trace

                    main_output, diagnostics_output = _split_run_output_sections(self.final_output)
                    self.final_output = main_output or self.final_output
                    self.final_diagnostics = diagnostics_output

                    if state.status == "completed" and self.final_output:
                        self.session_turns.append({"query": user_query, "answer": self.final_output})
                        self.session_turns = self.session_turns[-_SESSION_TURN_LIMIT:]

                    supporting_text = "\n".join(call_citations) or "—"
                    if self.final_diagnostics:
                        supporting_text = (
                            "## Run Diagnostics & Safeguards\n\n"
                            + self.final_diagnostics
                            + ("\n\n---\n\n" + supporting_text if supporting_text and supporting_text != "—" else "")
                        )

                    yield (
                        _build_chat_messages(self.session_turns),
                        structured_trace,
                        supporting_text,
                        status,
                        self.current_matter_id or "—",
                    )
                    return

                if update_type == "error":
                    self.is_running = False
                    yield (
                        _build_chat_messages(
                            self.session_turns,
                            pending_user=user_query,
                            pending_assistant=f"❌ {data}",
                        ),
                        "\n".join(call_thinking),
                        "",
                        f"❌ {data}",
                        "—",
                    )
                    return

            except queue.Empty:
                if self.is_running:
                    elapsed = time.time() - start_time
                    status = (
                        f"⏳  {_fmt_research_mode_label(self.current_research_mode)} | "
                        f"{elapsed:.0f}s | {len(call_thinking)} steps"
                    )
                    yield (
                        _build_chat_messages(
                            self.session_turns,
                            pending_user=user_query,
                            pending_assistant="*Investigating...*",
                        ),
                        "\n".join(call_thinking),
                        "\n".join(call_citations) or "—",
                        status,
                        self.current_matter_id or "—",
                    )

        thread.join(timeout=2)

    def stop_investigation(self):
        """Stop the running investigation.

        Sets is_running=False and _stop_event (breaks the UI generator before
        run_session exists). Routes stop through backend().stop_run() so both
        InProcessBackend and HttpBackend are covered — no direct engine access.

        Does NOT clear current_run_id so the redirect button in Gaps & Steering
        remains usable after stop (redirect must be sent before the engine fully
        halts; the run transitions away from 'running' once the engine iteration
        completes).  current_run_id is cleared only when a new investigation starts.

        Early-stop race: if stop is pressed before the first on_step fires
        (current_run_id is still None), falls back to querying the DB via _irys_ref.
        """
        self.is_running = False
        self._stop_event.set()  # breaks UI generator before run_session exists

        matter_id = self.current_matter_id
        run_id = self.current_run_id

        if matter_id and run_id:
            # Canonical path: submit directly to the shared executor (bounded at 8
            # workers) for fire-and-forget. is_running=False + _stop_event are
            # already set above, so the UI generator exits; the engine picks up
            # the stop flag on its next iteration. Discarding the future silences
            # any exception from the stop call (idempotent stop is safe).
            _ASYNC_EXECUTOR.submit(asyncio.run, self.backend().stop_run(matter_id, run_id))
        elif self._irys_ref is not None:
            # Early-stop race: no run_id yet — fall back to direct DB query.
            # Submit to the bounded executor (same pool as canonical path) so
            # the click handler returns without blocking on SQLite's busy_timeout.
            irys_ref = self._irys_ref  # snapshot before task runs
            def _early_stop() -> None:
                try:
                    engine = irys_ref._engine
                    if engine and engine._matter_model:
                        row = engine._matter_model.db.execute(
                            "SELECT id FROM run_session WHERE matter_id=?"
                            " AND status='running'"
                            " AND (objective IS NULL OR objective NOT IN"
                            " ('manual_flush','background_flush'))"
                            " ORDER BY started_at DESC LIMIT 1",
                            (engine._matter_model.matter_id,),
                        ).fetchone()
                        if row:
                            engine._matter_model.ledger.request_stop(row["id"])
                except Exception:
                    pass
            _ASYNC_EXECUTOR.submit(_early_stop)
        # current_run_id intentionally NOT cleared here — see docstring.
        return gr.update()

    def load_overview(self, matter_id: str) -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded. Run an investigation first.</div>"
        try:
            data = _run_async(self.backend().get_overview(matter_id))
            return _fmt_overview_panel(data)
        except Exception as exc:
            return f"<div class='viz-empty'>Error loading overview: {_escape(exc)}</div>"

    def load_issues(self, matter_id: str) -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            issues = _run_async(self.backend().list_issues(matter_id))
            return _fmt_issues_panel(issues)
        except Exception as exc:
            return f"<div class='viz-empty'>Error loading issues: {_escape(exc)}</div>"

    def load_assertions(self, matter_id: str) -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            backend = self.backend()
            assertions = _run_async(backend.list_assertions(matter_id, limit=50))
            domain = "legal"
            try:
                overview = _run_async(backend.get_overview(matter_id))
                dc = overview.get("domain_composition", {}) if isinstance(overview, dict) else {}
                domain = dc.get("primary_domain_profile_id", "legal") if isinstance(dc, dict) else "legal"
            except Exception:
                pass
            return _fmt_assertions(assertions, domain=domain)
        except Exception as exc:
            return f"<div class='viz-empty'>Error loading assertions: {_escape(str(exc))}</div>"

    def load_review_queue(self, matter_id: str) -> tuple[str, gr.update]:
        """Return (html_render, dropdown_update) for the review queue.

        The dropdown is keyed as "kind:id" strings so one selection
        drives both verify and reject. Returning a gr.update keeps
        the dropdown live without the handler needing to rewire it.
        """
        if not matter_id or matter_id == "—":
            return (
                "<div class='viz-empty'>No matter loaded.</div>",
                gr.update(choices=[], value=None),
            )
        try:
            queue = _run_async(self.backend().get_review_queue(matter_id, limit=100))
        except Exception as exc:
            logger.warning("Review queue load failed: %s", exc)
            return (
                "<div class='viz-empty'>Could not load the review queue. "
                "Try refreshing after the current investigation finishes.</div>",
                gr.update(choices=[], value=None),
            )
        html = _fmt_review_queue(queue)
        choices = _review_queue_choices(queue)
        value = choices[0][1] if choices else None
        return html, gr.update(choices=choices, value=value)

    def do_verify_target(
        self,
        matter_id: str,
        target_handle: str,
        review_note: str,
    ) -> str:
        """target_handle is a "kind:id" string from the dropdown."""
        if not matter_id or matter_id == "—":
            return "⚠️ No matter loaded."
        if not target_handle or ":" not in target_handle:
            return "⚠️ Select an item from the review queue first."
        kind, tid = target_handle.split(":", 1)
        # Snapshot queue size before the write so the success copy
        # can be honest about how many items a single verify actually
        # clears — adversarial #9 finding #4 regression. One assertion
        # verify often clears the assertion + its system-inferred edge
        # + its occurrence, so the queue drops by >1.
        try:
            before_count = _run_async(
                self.backend().count_review_queue(matter_id)
            ).get("total")
        except Exception:
            before_count = None
        try:
            _run_async(self.backend().verify_target(
                matter_id, kind, tid,
                reviewed_by_kind="user",
                reviewed_by_id="",  # attorney-readable label in audit
                review_note=(review_note or "").strip() or None,
            ))
        except ValueError as exc:
            return f"⚠️ {exc}"
        except Exception as exc:
            logger.warning("Verify failed for %s:%s — %s", kind, tid, exc)
            return "⚠️ Verify didn't go through. Please try again."
        if before_count is not None:
            try:
                after_count = _run_async(
                    self.backend().count_review_queue(matter_id)
                ).get("total", 0)
                cleared = max(0, before_count - after_count)
                if cleared > 1 and kind == "assertion":
                    return (
                        f"✅ Verified — also promoted {cleared - 1} related "
                        f"item(s) (the fact's supporting evidence link) so the "
                        "verified-coverage bar moves up cleanly."
                    )
            except Exception:
                pass
        return "✅ Verified — the verified-coverage bar moves up and the queue shrinks."

    def do_reject_target(
        self,
        matter_id: str,
        target_handle: str,
        rejection_reason: str,
    ) -> str:
        if not matter_id or matter_id == "—":
            return "⚠️ No matter loaded."
        if not target_handle or ":" not in target_handle:
            return "⚠️ Select an item from the review queue first."
        reason = (rejection_reason or "").strip()
        if not reason:
            return "⚠️ Rejection needs a reason — explain why it's wrong."
        kind, tid = target_handle.split(":", 1)
        try:
            _run_async(self.backend().reject_target(
                matter_id, kind, tid,
                rejection_reason=reason,
                reviewed_by_kind="user",
                reviewed_by_id="",
            ))
        except ValueError as exc:
            return f"⚠️ {exc}"
        except Exception as exc:
            logger.warning("Reject failed for %s:%s — %s", kind, tid, exc)
            return "⚠️ Reject didn't go through. Please try again."
        return (
            "✅ Rejected — dependent evidence and numbers were pulled back "
            "for re-check so the memo won't rely on them."
        )

    def load_review_count_badge(self, matter_id: str) -> str:
        """Small HTML chip showing how many findings are awaiting
        review, broken down by the top buckets. Renders quiet when
        the queue is empty.

        Uses the lightweight count endpoint rather than pulling the
        full 500-row queue just to bucket-count it — ~10× cheaper on
        active matters (OPT-2)."""
        if not matter_id or matter_id == "—":
            return ""
        try:
            counts = _run_async(self.backend().count_review_queue(matter_id))
        except Exception:
            return ""
        total = int(counts.get("total", 0) or 0)
        bucket_counts = {
            int(k): int(v) for k, v in (counts.get("by_bucket") or {}).items()
        }
        return _fmt_review_count_badge(total, bucket_counts)

    def load_post_review_snapshot(
        self, matter_id: str, target_handle: str,
    ) -> dict:
        """OPT-2b: one round-trip fetch for the seven panels that the
        verify/reject handlers refresh. Each underlying backend read
        runs concurrently inside a single event loop, so the 7
        sequential `_run_async` spawns collapse to one.

        Returns a dict with formatted HTML strings and the dropdown
        update so the caller can unpack straight into Gradio outputs.
        """
        if not matter_id or matter_id == "—":
            empty_html = "<div class='viz-empty'>No matter loaded.</div>"
            return {
                "queue_html": empty_html,
                "dropdown": gr.update(choices=[], value=None),
                "drawer_html": empty_html,
                "badge_html": "",
                "assertions_html": empty_html,
                "issues_html": empty_html,
                "overview_html": (
                    "<div class='viz-empty'>No matter loaded. "
                    "Run an investigation first.</div>"
                ),
            }
        backend = self.backend()
        kind_tid = target_handle.split(":", 1) if target_handle and ":" in target_handle else None

        async def _gather():
            tasks = [
                backend.get_review_queue(matter_id, limit=100),
                backend.count_review_queue(matter_id),
                backend.list_assertions(matter_id, limit=50),
                backend.list_issues(matter_id),
                backend.get_overview(matter_id),
            ]
            if kind_tid is not None:
                kind, tid = kind_tid
                tasks.append(backend.get_provenance(matter_id, kind, tid))
                tasks.append(backend.get_verification_events(matter_id, kind, tid))
            return await asyncio.gather(*tasks, return_exceptions=True)

        results = _run_async(_gather())
        queue, counts, assertions, issues, overview = results[:5]

        # One slow/failed read should not nuke the whole panel refresh
        # (graceful degradation), but per codex_opt2_review.txt we must
        # NOT silently substitute empty states — an empty queue renders
        # as "All findings reviewed", a false-clean UI. Render an
        # explicit error block per failed panel instead.
        def _err_html(label: str, exc: BaseException) -> str:
            return (
                "<div class='viz-empty'>"
                f"Couldn't load {_escape(label)}: {_escape(exc)}. "
                "Try the panel's own Refresh button."
                "</div>"
            )

        if isinstance(queue, BaseException):
            queue_html = _err_html("the review queue", queue)
            dropdown = gr.update(choices=[], value=None)
        else:
            queue_html = _fmt_review_queue(queue)
            choices = _review_queue_choices(queue)
            dropdown_value = choices[0][1] if choices else None
            dropdown = gr.update(choices=choices, value=dropdown_value)

        if isinstance(counts, BaseException):
            badge_html = (
                "<div style='padding:8px 12px;border-radius:6px;"
                "background:#fee2e2;color:#991b1b;font-size:12px;"
                "margin-bottom:8px;border-left:3px solid #b91c1c;'>"
                f"Badge unavailable: {_escape(counts)}. "
                "Hit refresh to retry."
                "</div>"
            )
        else:
            total = int(counts.get("total", 0) or 0)
            bucket_counts = {
                int(k): int(v) for k, v in (counts.get("by_bucket") or {}).items()
            }
            badge_html = _fmt_review_count_badge(total, bucket_counts)

        if kind_tid is not None:
            kind, tid = kind_tid
            prov, events = results[5], results[6]
            if isinstance(prov, BaseException) or isinstance(events, BaseException):
                drawer_html = _err_html(
                    "source & history",
                    prov if isinstance(prov, BaseException) else events,
                )
            else:
                drawer_html = _fmt_source_drawer(kind, tid, prov, events)
        else:
            drawer_html = (
                "<div class='viz-empty'>Pick a finding and click "
                "<em>Show source &amp; history</em> to see where it "
                "came from and every review action on it.</div>"
            )

        _ov_domain = "legal"
        if not isinstance(overview, BaseException) and isinstance(overview, dict):
            _dc = overview.get("domain_composition", {})
            if isinstance(_dc, dict):
                _ov_domain = _dc.get("primary_domain_profile_id", "legal")

        assertions_html = (
            _err_html("assertions", assertions)
            if isinstance(assertions, BaseException)
            else _fmt_assertions(assertions, domain=_ov_domain)
        )
        issues_html = (
            _err_html("issues", issues)
            if isinstance(issues, BaseException)
            else _fmt_issues_panel(issues)
        )
        overview_html = (
            _err_html("the overview", overview)
            if isinstance(overview, BaseException)
            else _fmt_overview_panel(overview)
        )

        return {
            "queue_html": queue_html,
            "dropdown": dropdown,
            "drawer_html": drawer_html,
            "badge_html": badge_html,
            "assertions_html": assertions_html,
            "issues_html": issues_html,
            "overview_html": overview_html,
        }

    def load_source_drawer(
        self, matter_id: str, target_handle: str,
    ) -> str:
        """Load provenance + verification history for the selected
        review-queue target."""
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        if not target_handle or ":" not in target_handle:
            return (
                "<div class='viz-empty'>Pick a finding and click "
                "<em>Show source &amp; history</em> to see where it "
                "came from and every review action on it.</div>"
            )
        kind, tid = target_handle.split(":", 1)
        try:
            prov = _run_async(self.backend().get_provenance(matter_id, kind, tid))
        except Exception as exc:
            logger.debug("Provenance fetch failed: %s", exc)
            prov = []
        try:
            events = _run_async(
                self.backend().get_verification_events(matter_id, kind, tid)
            )
        except Exception as exc:
            logger.debug("Verification events fetch failed: %s", exc)
            events = []
        return _fmt_source_drawer(kind, tid, prov, events)

    def do_bulk_verify_by_document(
        self,
        matter_id: str,
        document_ref: str,
    ) -> str:
        if not matter_id or matter_id == "—":
            return "⚠️ No matter loaded."
        ref = (document_ref or "").strip()
        if not ref:
            return "⚠️ Enter a document name or path."
        try:
            ids = _run_async(self.backend().bulk_verify_by_document(
                matter_id, ref,
                reviewed_by_kind="user", reviewed_by_id="",
            ))
        except ValueError as exc:
            return f"⚠️ {exc}"
        except Exception as exc:
            logger.warning("Bulk verify failed for %s — %s", ref, exc)
            return "⚠️ Bulk verify didn't go through. Please try again."
        if not ids:
            return (
                f"No candidate findings matched <strong>{_escape(ref)}</strong>. "
                "Check the document name exactly as it appears in the Sources list."
            )
        return f"✅ Verified {len(ids)} fact(s) sourced from {_escape(ref)}."

    def load_batch_review_facts(
        self,
        matter_id: str,
        document_ref: str,
    ) -> tuple[gr.update, gr.update, gr.update, gr.update]:
        """Load candidate facts for a document so the attorney can
        review + selectively verify them. Returns gr.update tuples
        for (CheckboxGroup, metadata HTML, verify-selected button,
        result markdown). Everything stays hidden until a real
        document is picked AND it has candidate facts.
        """
        hide = (
            gr.update(choices=[], value=[], visible=False),
            gr.update(value="", visible=False),
            gr.update(visible=False),
            gr.update(value="", visible=False),
        )
        if not matter_id or matter_id == "—":
            return hide
        ref = (document_ref or "").strip()
        if not ref:
            return hide
        try:
            rows = _run_async(
                self.backend().list_candidate_assertions_for_document(
                    matter_id, ref,
                )
            )
        except Exception as exc:
            logger.warning(
                "load_batch_review_facts failed for %s: %s", ref, exc,
            )
            return (
                gr.update(choices=[], value=[], visible=False),
                gr.update(value="", visible=False),
                gr.update(visible=False),
                gr.update(
                    value=(
                        "⚠️ Couldn't load candidate facts for this "
                        "document. Check the document name."
                    ),
                    visible=True,
                ),
            )
        if not rows:
            return (
                gr.update(choices=[], value=[], visible=False),
                gr.update(value="", visible=False),
                gr.update(visible=False),
                gr.update(
                    value=(
                        "No candidate facts remain for this document "
                        "— everything has already been verified or "
                        "rejected."
                    ),
                    visible=True,
                ),
            )
        _batch_domain = "legal"
        try:
            _ov = _run_async(self.backend().get_overview(matter_id))
            _dc = _ov.get("domain_composition", {}) if isinstance(_ov, dict) else {}
            _batch_domain = _dc.get("primary_domain_profile_id", "legal") if isinstance(_dc, dict) else "legal"
        except Exception:
            pass
        choices: list[tuple[str, str]] = []
        detail_lines: list[str] = []
        for r in rows:
            prop = str(r.get("proposition_text") or "").strip() or "(no text)"
            _raw_role = str(r.get("primary_source_role") or "unknown")
            role = _domain_source_label(_raw_role, _batch_domain).upper()
            _raw_speech = str(r.get("primary_speech_act") or "").lower() or "extracted"
            speech_act = _domain_speech_act_label(_raw_speech, _batch_domain)
            conf_raw = r.get("confidence")
            try:
                conf_str = f"{float(conf_raw):.0%}" if conf_raw is not None else "—"
            except (TypeError, ValueError):
                conf_str = "—"
            occ_count = int(r.get("occurrence_count") or 1)
            occ_str = f", appears {occ_count}×" if occ_count > 1 else ""
            short = prop[:180] + ("…" if len(prop) > 180 else "")
            label = (
                f"[{role}] {short}  —  {speech_act}, "
                f"confidence {conf_str}{occ_str}"
            )
            choices.append((label, str(r.get("id"))))
            detail_lines.append(
                f"<li><strong>{_escape(short)}</strong> — "
                f"<em>{_escape(role)}</em> · "
                f"{_escape(speech_act)} · "
                f"confidence {_escape(conf_str)}{_escape(occ_str)}</li>"
            )
        # Pre-check every id — the intent is "verify all" with
        # opt-out. Attorney unchecks to exclude.
        all_ids = [aid for _, aid in choices]
        summary_html = (
            "<div class='viz-note'>"
            f"<strong>{len(rows)} candidate fact(s)</strong> from "
            f"<code>{_escape(ref)}</code>. Every fact is pre-selected "
            "— uncheck any you don't want to verify, then click "
            "<strong>Verify selected</strong>."
            "</div>"
        )
        return (
            gr.update(choices=choices, value=all_ids, visible=True),
            gr.update(value=summary_html, visible=True),
            gr.update(visible=True),
            gr.update(value="", visible=False),
        )

    def do_batch_verify_selected(
        self,
        matter_id: str,
        selected_ids: list[str],
    ) -> tuple[gr.update, gr.update, gr.update, gr.update]:
        """Verify the attorney-selected subset. Returns the same 4
        gr.update tuple shape as load_batch_review_facts so the UI
        binding can wire one event to all four components.
        """
        if not matter_id or matter_id == "—":
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(value="⚠️ No matter loaded.", visible=True),
            )
        ids = [str(i) for i in (selected_ids or []) if i]
        if not ids:
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(
                    value=(
                        "⚠️ Nothing selected. Check at least one fact "
                        "to verify — or use \"Verify all from document\" "
                        "above for the one-shot action."
                    ),
                    visible=True,
                ),
            )
        try:
            verified = _run_async(
                self.backend().bulk_verify_assertion_ids(
                    matter_id, ids,
                    reviewed_by_kind="user", reviewed_by_id="",
                )
            )
        except ValueError as exc:
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(value=f"⚠️ {exc}", visible=True),
            )
        except Exception as exc:
            logger.warning("Batch verify failed: %s", exc)
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(
                    value=(
                        "⚠️ Batch verify didn't go through. Please "
                        "try again."
                    ),
                    visible=True,
                ),
            )
        skipped = len(ids) - len(verified)
        msg = f"✅ Verified {len(verified)} selected fact(s)."
        if skipped:
            msg += (
                f" Skipped {skipped} that were already verified "
                "or no longer candidate — refresh the list."
            )
        # Hide the checklist after success so the UI returns to
        # "load new doc" mode.
        return (
            gr.update(choices=[], value=[], visible=False),
            gr.update(value="", visible=False),
            gr.update(visible=False),
            gr.update(value=msg, visible=True),
        )

    def load_gaps(self, matter_id: str) -> tuple[str, str]:
        """Return (gaps_and_steering_markdown, top_redirect_issue_id).

        The second value auto-populates the Redirect form's issue_id field so
        the steering surface is actionable without manual copy-paste (SO-3).
        """
        if not matter_id or matter_id == "—":
            return "No matter loaded.", ""
        try:
            gaps = _run_async(self.backend().list_gaps(matter_id))
            clarifications = _run_async(self.backend().list_clarifications(matter_id))
            gap_section = _fmt_gaps(gaps, clarifications)
        except Exception as exc:
            gap_section = f"⚠️ Error loading gaps: {exc}"
        actions: list = []
        try:
            run_id = getattr(self, "current_run_id", None)
            actions = _run_async(self.backend().get_steering_surface(matter_id, run_id=run_id))
            steering_section = _fmt_steering(actions)
        except Exception as exc:
            steering_section = f"⚠️ Steering surface error: {exc}"
        sections = [gap_section]
        if steering_section:
            sections.append("\n" + steering_section)
        # Extract the highest-priority redirect_focus issue_id to auto-populate the form.
        top_redirect_issue = ""
        for action in actions:
            if action.get("action_type") == "redirect_focus":
                top_redirect_issue = action.get("params", {}).get("issue_id", "")
                break
        return "\n".join(sections), top_redirect_issue

    def load_assumptions(self, matter_id: str) -> str:
        if not matter_id or matter_id == "—":
            return "No matter loaded."
        try:
            assumptions = _run_async(self.backend().list_assumptions(matter_id))
            return _fmt_assumptions(assumptions)
        except Exception as exc:
            return f"Error loading assumptions: {exc}"

    def load_quant(self, matter_id: str) -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            quant_data = _run_async(self.backend().get_quant_summary(matter_id))
            return _fmt_quant_panel(
                quant_data.get("payment_reconciliation", {}),
                quant_data.get("invoice_reconciliation", []),
                quant_data.get("amount_conflicts", []),
                quant_data.get("damages_waterfall", []),
            )
        except Exception as exc:
            return f"<div class='viz-empty'>Error loading quantitative data: {_escape(exc)}</div>"

    def load_timeline(self, matter_id: str) -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            events = _run_async(self.backend().get_timeline(
                matter_id, limit=200, policy_audience=self.policy_audience,
            ))
            return _fmt_timeline_panel(events)
        except Exception as exc:
            return f"<div class='viz-empty'>Error loading timeline: {_escape(exc)}</div>"

    def load_evidence_matrix(self, matter_id: str) -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            matrix = _run_async(self.backend().get_evidence_matrix(
                matter_id, policy_audience=self.policy_audience,
            ))
            return _fmt_evidence_matrix_panel(matrix)
        except Exception as exc:
            return f"<div class='viz-empty'>Error loading evidence matrix: {_escape(exc)}</div>"

    def load_proof_state(self, matter_id: str) -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            data = _run_async(self.backend().get_proof_state_summary(matter_id))
            summary = data.get("summary", {}) if isinstance(data, dict) else {}
            issues = data.get("issues", []) if isinstance(data, dict) else []
            domain = "legal"
            try:
                ov = _run_async(self.backend().get_overview(matter_id))
                dc = ov.get("domain_composition", {}) if isinstance(ov, dict) else {}
                domain = dc.get("primary_domain_profile_id", "legal") if isinstance(dc, dict) else "legal"
            except Exception:
                pass
            return _fmt_proof_state_panel(summary, issues, domain)
        except Exception as exc:
            return f"<div class='viz-empty'>Error loading proof state: {_escape(exc)}</div>"

    def set_policy_audience(self, label: str) -> str:
        """Called when the sidebar privilege-mode toggle flips. Returns
        a visible banner HTML so the reviewer always knows which mode
        is active — silent switching would be a disclosure risk."""
        audience = "internal" if str(label).strip().startswith("Internal") else "clean"
        self.policy_audience = audience
        return _fmt_privilege_banner(audience)

    def load_document_picker_choices(self, matter_id: str) -> list:
        """Return labelled choices for the bulk-verify Dropdown, newest
        pending first. Labels include pending/verified counts so
        reviewers can triage without opening the doc list."""
        if not matter_id or matter_id == "—":
            return []
        try:
            rows = _run_async(
                self.backend().list_reviewable_documents(matter_id)
            )
        except Exception:
            return []
        choices = []
        for row in rows:
            path = row.get("path") or ""
            if not path:
                continue
            pending = int(row.get("pending") or 0)
            verified = int(row.get("verified") or 0)
            if pending > 0:
                label = f"{path}  ({pending} pending)"
            elif verified > 0:
                label = f"{path}  ({verified} reviewed ✓)"
            else:
                label = path
            choices.append((label, path))
        return choices

    def load_communication_map(self, matter_id: str) -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            graph = _run_async(self.backend().get_communication_map(matter_id))
            return _fmt_communication_map_panel(graph)
        except Exception as exc:
            return f"<div class='viz-empty'>Error loading communication map: {_escape(exc)}</div>"

    def load_llm_analytics(self, matter_id: str) -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            overview = _run_async(self.backend().get_overview(matter_id))
            stats = overview.get("stats", {}) if isinstance(overview, dict) else {}
            llm = stats.get("llm", {}) if isinstance(stats.get("llm"), dict) else {}
            summary = llm.get("totals", {}) if isinstance(llm, dict) else {}
            calls = _run_async(self.backend().list_llm_calls(matter_id, limit=250))
            breakdown = _run_async(self.backend().get_cost_breakdown(matter_id))
            anomalies = _run_async(
                self.backend().get_cost_anomalies(matter_id, limit=10)
            )
            return _fmt_llm_analytics_panel(summary, calls, breakdown, anomalies)
        except Exception as exc:
            return f"<div class='viz-empty'>Error loading LLM analytics: {_escape(exc)}</div>"

    def do_correct_assertion(
        self, matter_id: str, assertion_id: str, new_state: str, reason: str
    ) -> str:
        if not matter_id:
            return "Load a matter before applying a correction."
        if not assertion_id:
            return "Copy a Fact ID from the Extracted Facts table above and paste it into the form."
        if not new_state:
            return "Pick the corrected characterization from the dropdown."
        try:
            result = _run_async(
                self.backend().correct_assertion(
                    matter_id, assertion_id, new_state, reason,
                    run_id=getattr(self, "current_run_id", None),
                )
            )
            if isinstance(result, dict) and result.get("status") == "error":
                return f"❌ {result.get('detail', result)}"
            return f"✅ Corrected: {result}"
        except Exception as exc:
            return f"❌ Error: {exc}"

    def do_resume(self, matter_id: str, run_id: str, research_mode: str = "deep") -> str:
        if not matter_id:
            return "Load a matter before resuming an investigation."
        if not run_id:
            return "No run to resume — start an investigation first."
        normalized_mode = normalize_research_mode(research_mode)
        self.current_research_mode = normalized_mode
        # Launch resume in the executor (fire-and-forget): investigation can be
        # much longer than the 30s _run_async default timeout. The run will complete
        # in the background; the user can monitor progress via Overview / Ledger Events.
        # Pre-validation errors (not interrupted, no checkpoint) are surfaced in the
        # result dict but swallowed here — user can check Overview/status.
        mid, rid = matter_id, run_id  # snapshot before task runs
        def _do_resume() -> None:
            try:
                result = asyncio.run(
                    self.backend().resume_run(
                        mid,
                        rid,
                        research_mode=normalized_mode,
                    )
                )
                if isinstance(result, dict):
                    if result.get("status") == "error":
                        # Surface pre-validation errors to a discoverable attribute
                        self._last_resume_error = result.get("detail", "Unknown error")
                        return
                    new_rid = result.get("new_run_id")
                    if new_rid:
                        self.current_run_id = new_rid
                        self._last_resume_error = None
            except Exception as exc:
                self._last_resume_error = str(exc)
        self._last_resume_error = None  # clear stale error before submit (LOW r69)
        future = _ASYNC_EXECUTOR.submit(_do_resume)
        # r96 MEDIUM: wait briefly so fast pre-validation failures surface immediately.
        # Pre-validation (run exists, is interrupted, has checkpoint) completes in
        # milliseconds; real investigations take seconds-to-minutes and will time out.
        import concurrent.futures as _cf
        try:
            future.result(timeout=0.5)
            # Completed within 0.5s — fast pre-validation failure or very small repo.
            if self._last_resume_error:
                return f"❌ Resume failed: {self._last_resume_error}"
        except _cf.TimeoutError:
            pass  # Still running — genuine investigation launch
        return f"⏳ Resume of run {run_id} launched — monitor via Overview or Ledger Events."

    def do_redirect(self, matter_id: str, run_id: str, issue_id: str) -> str:
        if not matter_id:
            return "Load a matter before redirecting the investigation."
        if not run_id:
            return "No active investigation to redirect — start one first."
        if not issue_id:
            return "Pick an issue to focus on from the suggestions above."
        try:
            result = _run_async(
                self.backend().redirect_run(matter_id, run_id, issue_id)
            )
            if isinstance(result, dict) and result.get("status") == "error":
                return f"❌ {result.get('detail', result)}"
            msg = f"✅ Redirect requested → issue {result.get('issue_id', issue_id) if isinstance(result, dict) else issue_id}"
            if isinstance(result, dict) and result.get("issue_title"):
                msg += f" ({result['issue_title']})"
            if isinstance(result, dict) and result.get("note"):
                msg += f"\n⚠️ {result['note']}"
            return msg
        except Exception as exc:
            return f"❌ Error: {exc}"


# ---------------------------------------------------------------------------
# Matter workspace UI helpers
# ---------------------------------------------------------------------------

def _fmt_matter_card(name: str, files: list[str]) -> str:
    """Render a styled HTML card for the selected matter."""
    if not name:
        return (
            "<div class='matter-empty-state'>"
            "<div class='matter-empty-title'>No matter selected</div>"
            "<div class='matter-empty-sub'>Choose an existing matter from the dropdown above, "
            "or create a new one below</div>"
            "</div>"
        )
    count = len(files)
    badge = f"{count} document{'s' if count != 1 else ''}"
    if files:
        chips = "".join(
            f"<div class='matter-file-chip'>"
            f"<span class='matter-file-icon'>📄</span>"
            f"<span>{_escape(f)}</span>"
            f"</div>"
            for f in files
        )
        files_html = f"<div class='matter-card-files'>{chips}</div>"
    else:
        files_html = (
            "<div style='font-size:12px;color:#94a3b8;margin-top:2px;'>"
            "No documents yet — upload some below"
            "</div>"
        )
    return (
        "<div class='matter-card'>"
        "<div class='matter-card-header'>"
        f"<span class='matter-card-name'>📁 {_escape(name)}</span>"
        f"<span class='matter-card-badge'>{badge}</span>"
        "</div>"
        f"{files_html}"
        "</div>"
    )


def _fmt_ws_status(msg: str, kind: str = "ok") -> str:
    """Render a styled status pill. kind: ok | err | info"""
    if not msg:
        return ""
    tone = {"ok": "ws-ok", "err": "ws-err", "info": "ws-info", "warn": "ws-warn"}.get(kind, "ws-info")
    return f"<div class='ws-status {tone}'>{_escape(msg)}</div>"


# ---------------------------------------------------------------------------
# Gradio app construction
# ---------------------------------------------------------------------------


_theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
)
_css = """
    .mono textarea { font-family: 'JetBrains Mono', monospace; font-size: 12px; }
    .status-bar textarea { font-weight: 600; font-size: 13px; }
    .compact-id { font-size: 11px !important; }
    .compact-id textarea { font-size: 11px; color: var(--body-text-color-subdued, #888); }
    footer { display: none !important; }

    /* ── Matter workspace ─────────────────────────────────── */
    .matter-card {
        border: 1px solid var(--block-border-color, #dbe4ef);
        border-radius: 14px; padding: 16px 18px;
        background: var(--background-fill-primary, #fff);
        box-shadow: 0 2px 10px rgba(15,23,42,0.06); margin-bottom: 2px;
    }
    .matter-card-header {
        display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;
    }
    .matter-card-name { font-size: 15px; font-weight: 700; color: var(--body-text-color, #0f172a); }
    .matter-card-badge {
        font-size: 12px; font-weight: 600; color: var(--color-accent, #2563eb);
        background: rgba(37,99,235,0.08); border-radius: 999px; padding: 2px 10px;
    }
    .matter-card-files { display: flex; flex-direction: column; gap: 4px; }
    .matter-file-chip {
        display: flex; align-items: center; gap: 8px; font-size: 12px;
        color: var(--body-text-color, #334155);
        background: var(--background-fill-secondary, #f8fafc);
        border: 1px solid var(--block-border-color, #e2e8f0);
        border-radius: 8px; padding: 5px 10px;
    }
    .matter-file-icon { color: var(--body-text-color-subdued, #64748b); flex-shrink: 0; }
    .matter-empty-state {
        border: 2px dashed var(--block-border-color, #cbd5e1);
        border-radius: 14px; padding: 32px 20px;
        text-align: center; background: var(--background-fill-secondary, #f8fafc);
    }
    .matter-empty-title { font-size: 15px; font-weight: 600; color: var(--body-text-color-subdued, #64748b); margin-bottom: 6px; }
    .matter-empty-sub { font-size: 13px; color: var(--body-text-color-subdued, #94a3b8); }
    /* Status badges: intentional semantic colors, not theme variables */
    .ws-status { font-size: 12px; border-radius: 8px; padding: 7px 12px; margin-top: 4px; }
    .ws-ok  { color: #15803d; background: #f0fdf4; border: 1px solid #bbf7d0; }
    .ws-err { color: #b91c1c; background: #fef2f2; border: 1px solid #fecaca; }
    .ws-info{ color: #1d4ed8; background: #eff6ff; border: 1px solid #bfdbfe; }
    .ws-warn{ color: #92400e; background: #fffbeb; border: 1px solid #fde68a; }
    .gap-highlight { background: #fef3c7; border-radius: 6px; padding: 8px; }

    /* ── Shared viz shell ─────────────────────────────────── */
    .viz-shell { display: flex; flex-direction: column; gap: 12px; min-width: 0; }
    .viz-empty {
        border: 1px dashed var(--block-border-color, #cbd5e1); border-radius: 12px; padding: 14px;
        color: var(--body-text-color-subdued, #64748b);
        background: var(--background-fill-secondary, #f8fafc);
    }

    /* ── Stat cards ───────────────────────────────────────── */
    .viz-card-grid {
        display: grid; gap: 10px;
        grid-template-columns: repeat(auto-fit, minmax(135px, 1fr));
    }
    .viz-card {
        border: 1px solid var(--block-border-color, #dbe4ef);
        border-radius: 14px; padding: 12px 14px;
        background: var(--background-fill-primary, #fff);
        box-shadow: 0 2px 8px rgba(15,23,42,0.05);
        min-width: 0;
    }
    .viz-card.tone-amber { border-color: rgba(217,119,6,0.25); }
    .viz-card.tone-green { border-color: rgba(21,128,61,0.25); }
    .viz-card.tone-red   { border-color: rgba(185,28,28,0.25); }
    .viz-card-title {
        font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase;
        color: var(--body-text-color-subdued, #64748b);
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .viz-card-value {
        font-size: 24px; font-weight: 700;
        color: var(--body-text-color, #0f172a); margin-top: 4px;
    }
    .viz-card-detail {
        font-size: 12px; color: var(--body-text-color-subdued, #475569); margin-top: 6px;
    }

    /* ── Two-column grid ─────────────────────────────────── */
    .viz-two-col {
        display: grid; gap: 12px;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        min-width: 0;
    }

    /* ── Panel card ──────────────────────────────────────── */
    .viz-panel {
        border: 1px solid var(--block-border-color, #dbe4ef);
        border-radius: 14px; padding: 14px;
        background: var(--background-fill-primary, #fff);
        min-width: 0;
    }
    .viz-panel-title {
        font-size: 13px; font-weight: 700;
        color: var(--body-text-color, #0f172a); margin-bottom: 10px;
    }
    .viz-subtitle {
        font-size: 12px; font-weight: 700;
        color: var(--body-text-color-subdued, #475569); margin-bottom: 6px;
    }
    .viz-footnote { font-size: 11px; color: var(--body-text-color-subdued, #64748b); margin-top: 8px; }

    /* ── List rows ───────────────────────────────────────── */
    .viz-list-row {
        display: flex; justify-content: space-between; align-items: flex-start;
        gap: 12px; padding: 8px 0;
        border-bottom: 1px solid var(--block-border-color, #eef2f7);
        font-size: 12px; color: var(--body-text-color, #334155);
        min-width: 0;
    }
    .viz-list-row span, .viz-list-row strong { white-space: normal; word-break: break-word; min-width: 0; }
    .viz-list-row:last-child { border-bottom: none; }
    .viz-list-columns {
        display: grid; gap: 14px;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    }
    .viz-list-columns ul { margin: 0; padding-left: 18px; color: var(--body-text-color, #334155); }
    .viz-list-columns li { margin-bottom: 6px; }

    /* ── Bar rows ────────────────────────────────────────── */
    .viz-bar-row {
        display: grid; gap: 8px; align-items: center;
        grid-template-columns: minmax(80px, 1fr) minmax(80px, 2fr) minmax(80px, auto);
        margin-bottom: 8px; min-width: 0;
    }
    .viz-bar-label {
        font-size: 12px; color: var(--body-text-color, #334155);
        white-space: normal; word-break: break-word; min-width: 0;
    }
    .viz-bar-track {
        height: 10px; border-radius: 999px;
        background: var(--block-border-color, #e2e8f0); overflow: hidden; min-width: 0;
    }
    .viz-bar-fill { height: 100%; border-radius: 999px; }
    .viz-bar-fill.tone-blue  { background: linear-gradient(90deg, var(--color-accent,#2563eb), #38bdf8); }
    .viz-bar-fill.tone-amber { background: linear-gradient(90deg, #d97706, #f59e0b); }
    .viz-bar-fill.tone-green { background: linear-gradient(90deg, #15803d, #22c55e); }
    .viz-bar-fill.tone-red   { background: linear-gradient(90deg, #b91c1c, #ef4444); }
    .viz-bar-meta { font-size: 12px; color: var(--body-text-color-subdued, #64748b); text-align: right; min-width: 0; word-break: break-word; }

    /* ── Issues tree ─────────────────────────────────────── */
    .issues-stack { display: flex; flex-direction: column; gap: 10px; min-width: 0; }
    .issue-row {
        padding: 10px 12px 12px calc(12px + var(--issue-indent));
        border: 1px solid var(--block-border-color, #e2e8f0); border-radius: 12px;
        background: var(--background-fill-primary, #fff); min-width: 0;
    }
    .issue-head { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; min-width: 0; }
    .proof-pill {
        border-radius: 999px; padding: 2px 8px; font-size: 10px;
        font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; flex-shrink: 0;
    }
    .proof-strong { background: rgba(21,128,61,0.12); color: #166534; }
    .proof-partial { background: rgba(217,119,6,0.12); color: #b45309; }
    .proof-weak    { background: rgba(249,115,22,0.12); color: #c2410c; }
    .proof-gap     { background: rgba(185,28,28,0.12); color: #b91c1c; }
    .proof-none    { background: rgba(148,163,184,0.18); color: #475569; }
    .issue-title {
        flex: 1; min-width: 0; font-size: 13px; font-weight: 600;
        color: var(--body-text-color, #0f172a); white-space: normal; word-break: break-word;
    }
    .issue-pct { font-size: 12px; color: var(--body-text-color-subdued, #475569); flex-shrink: 0; }
    .issue-track {
        height: 8px; border-radius: 999px;
        background: var(--block-border-color, #e2e8f0); overflow: hidden;
    }
    .issue-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--color-accent,#1d4ed8), #22c55e); }
    .issue-meta { font-size: 12px; color: var(--body-text-color-subdued, #64748b); margin-top: 8px; }

    /* ── Timeline ────────────────────────────────────────── */
    .timeline-list { position: relative; display: flex; flex-direction: column; gap: 12px; min-width: 0; }
    .timeline-item {
        display: grid; gap: 12px; align-items: start;
        grid-template-columns: 100px 18px minmax(0, 1fr);
    }
    .timeline-date {
        font-size: 12px; font-weight: 700;
        color: var(--body-text-color, #334155); padding-top: 2px; word-break: break-word;
    }
    .timeline-line { position: relative; min-height: 56px; }
    .timeline-line::before {
        content: ''; position: absolute; left: 8px; top: 0; bottom: -12px;
        width: 2px; background: var(--block-border-color, #dbe4ef);
    }
    .timeline-dot {
        position: absolute; left: 2px; top: 6px; width: 14px; height: 14px;
        border-radius: 50%; background: var(--color-accent, #2563eb);
        box-shadow: 0 0 0 4px rgba(37,99,235,0.12);
    }
    .timeline-body {
        border: 1px solid var(--block-border-color, #dbe4ef); border-radius: 12px; padding: 10px 12px;
        background: var(--background-fill-primary, #fff); min-width: 0;
    }
    .timeline-title {
        font-size: 13px; font-weight: 600; color: var(--body-text-color, #0f172a);
        white-space: normal; word-break: break-word;
    }
    .timeline-meta {
        font-size: 12px; color: var(--body-text-color-subdued, #64748b);
        margin-top: 6px; white-space: normal; word-break: break-word;
    }

    /* ── Evidence / analytics tables ─────────────────────── */
    .matrix-wrap { overflow: auto; max-width: 100%; }
    .matrix-wrap-heatmap {
        overflow: auto; max-width: 100%; max-height: 72vh;
        border: 1px solid var(--block-border-color, #dbe4ef); border-radius: 12px;
        background: var(--background-fill-primary, #fff);
    }
    .matrix-table, .analytics-table {
        width: 100%; border-collapse: separate; border-spacing: 0; font-size: 12px;
    }
    .matrix-table th, .matrix-table td, .analytics-table th, .analytics-table td {
        border-bottom: 1px solid var(--block-border-color, #e2e8f0);
        padding: 8px 10px; text-align: left;
        white-space: normal; word-break: break-word; vertical-align: top;
    }
    .matrix-table thead th, .analytics-table thead th {
        position: sticky; top: 0; z-index: 1;
        background: var(--background-fill-secondary, #f8fafc);
        color: var(--body-text-color, #334155);
        font-weight: 700;
    }
    .matrix-cell { min-width: 52px; text-align: center !important; font-weight: 700; color: var(--body-text-color, #0f172a); }
    .evidence-matrix-table { width: max-content; min-width: max-content; table-layout: fixed; }
    .evidence-matrix-table thead th { min-width: 170px; max-width: 220px; z-index: 3; }
    .evidence-matrix-table thead th:first-child {
        min-width: 220px; max-width: 300px; left: 0; z-index: 5;
        box-shadow: 2px 0 0 var(--block-border-color, #dbe4ef);
    }
    .evidence-matrix-table tbody th {
        position: sticky; left: 0; min-width: 220px; max-width: 300px; z-index: 2;
        background: var(--background-fill-secondary, #f8fafc);
        box-shadow: 2px 0 0 var(--block-border-color, #dbe4ef);
    }
    .evidence-matrix-table td.matrix-cell { min-width: 72px; width: 72px; text-align: center !important; }

    /* ── Communication graph ─────────────────────────────── */
    .comm-graph {
        width: 100%; height: auto;
        border: 1px solid var(--block-border-color, #dbe4ef); border-radius: 14px;
        background: var(--background-fill-secondary, #f8fafc); overflow: visible;
    }
    .comm-actor-node { fill: var(--color-accent, #1d4ed8); opacity: 0.9; }
    .comm-doc-node { fill: #0f766e; opacity: 0.85; }
    .comm-label { font-size: 11px; fill: var(--body-text-color, #334155); font-family: 'Inter', sans-serif; }
    .comm-label-left { text-anchor: start; }

    /* ── Expandable detail ───────────────────────────────── */
    .viz-detail { border-top: 1px solid var(--block-border-color, #e2e8f0); padding: 10px 0; }
    .viz-detail:first-child { border-top: none; }
    .viz-detail summary { cursor: pointer; font-weight: 600; color: var(--body-text-color, #0f172a); }
    .viz-detail-block { margin-top: 10px; font-size: 12px; color: var(--body-text-color, #334155); }
    .viz-detail-block ul { margin: 6px 0 0 0; padding-left: 18px; }
    .viz-detail-block li { margin-bottom: 6px; }

    /* ── Sidebar column transparency ─────────────────────── */
    .intelligence-sidebar {
        --block-background-fill: transparent;
        --block-border-color: transparent;
        --block-border-width: 0px;
        --block-shadow: none;
        --block-padding: 0px;
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    .intelligence-sidebar > div,
    .intelligence-sidebar > div > div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }

    /* ── Global responsive safety net ───────────────────── */
    .gradio-container * { box-sizing: border-box; }
    .gradio-container td, .gradio-container th { overflow-wrap: break-word; }
    .gradio-container img { max-width: 100%; height: auto; }

    /* ── Assertions: clickable rows ──────────────────────── */
    .assertions-row:hover { background: var(--background-fill-secondary, #f8fafc); }

    /* ── Belief state pills ──────────────────────────────── */
    .belief-pill {
        display: inline-block; border-radius: 999px; padding: 2px 8px;
        font-size: 10px; font-weight: 700; text-transform: capitalize;
        letter-spacing: 0.04em;
    }
    .belief-alleged    { background: rgba(217,119,6,0.12);  color: #b45309; }
    .belief-argued     { background: rgba(30,64,175,0.10);  color: #1e40af; }
    .belief-admitted   { background: rgba(21,128,61,0.12);  color: #166534; }
    .belief-operative  { background: rgba(6,95,70,0.12);    color: #065f46; }
    .belief-performed  { background: rgba(15,118,110,0.12); color: #0f766e; }
    .belief-disputed   { background: rgba(194,65,12,0.12);  color: #c2410c; }
    .belief-superseded { background: rgba(71,85,105,0.12);  color: #475569; }
    .belief-withdrawn  { background: rgba(107,114,128,0.12);color: #6b7280; }
    .belief-inferred   { background: rgba(109,40,217,0.12); color: #6d28d9; }
    .belief-resolved   { background: rgba(55,48,163,0.12);  color: #3730a3; }
    .belief-unknown    { background: rgba(148,163,184,0.18);color: #475569; }
    .belief-legend { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; }

    /* ── Correction result card ──────────────────────────── */
    .correction-result {
        display: flex; align-items: flex-start; gap: 10px;
        padding: 10px 14px; border-radius: 8px; font-size: 13px;
        border: 1px solid;
    }
    .correction-ok  { background: rgba(21,128,61,0.08); border-color: rgba(21,128,61,0.25); color: #166534; }
    .correction-err { background: rgba(185,28,28,0.08); border-color: rgba(185,28,28,0.25); color: #b91c1c; }
    .correction-icon { font-size: 18px; line-height: 1; flex-shrink: 0; }
    .correction-detail { color: var(--body-text-color, #334155); font-size: 12px; }
    .correction-result-empty { color: var(--body-text-color-subdued, #64748b); font-size: 12px; font-style: italic; }
    """


def create_app(api_key: Optional[str] = None) -> gr.Blocks:
    state = AppState(api_key=api_key)
    _s3_mode = _get_storage_mode() == "s3"

    with gr.Blocks(title="Irys — Matter Intelligence") as demo:

        # Hidden matter_id state — auto-populated, never shown prominently
        matter_id_box = gr.Textbox(visible=False)

        # ==================================================================
        # HEADER
        # ==================================================================
        gr.Markdown(
            "# Irys\n"
            "Analyze your matter. Point to a folder of documents and "
            "ask questions in plain English."
        )

        # ==================================================================
        # INPUT
        # ==================================================================
        # Hidden textbox holds the resolved local path in both modes.
        repo_path = gr.Textbox(visible=False)

        if _s3_mode:
            # --- Matter selector row ---
            with gr.Row():
                matter_dropdown = gr.Dropdown(
                    label="Select matter",
                    choices=_list_s3_matter_names(),
                    value=None,
                    allow_custom_value=False,
                    scale=5,
                    interactive=True,
                )
                refresh_matters_btn = gr.Button("↻  Refresh", variant="secondary", scale=1, min_width=100)

            # --- Matter card: shows selected matter name, doc count, and file list ---
            matter_card_html = gr.HTML(_fmt_matter_card("", []))

            # --- Workspace: visible only when a matter is selected ---
            matter_workspace = gr.Group(visible=False)
            with matter_workspace:
                with gr.Row():
                    matter_files_dropdown = gr.Dropdown(
                        label="Select file to remove",
                        choices=[],
                        value=None,
                        allow_custom_value=False,
                        scale=4,
                        interactive=True,
                    )
                    delete_file_btn = gr.Button("Remove file", variant="stop", scale=1, min_width=130)
                with gr.Row():
                    file_upload = gr.File(
                        label="Add files",
                        file_count="multiple",
                        scale=4,
                    )
                    add_files_btn = gr.Button("Upload Files", variant="primary", scale=1, min_width=120)
                with gr.Row():
                    folder_upload = gr.File(
                        label="Add folder (select any file inside the folder)",
                        file_count="multiple",
                        scale=4,
                        elem_id="folder-upload",
                    )
                    add_folder_btn = gr.Button("Upload Folder", variant="primary", scale=1, min_width=120)
                with gr.Accordion("Danger zone", open=False):
                    delete_matter_btn = gr.Button("Delete entire matter", variant="stop")
                file_manage_status = gr.HTML()

            # --- Create new matter ---
            with gr.Accordion("+ Create new matter", open=False):
                matter_name_input = gr.Textbox(
                    label="Matter name",
                    placeholder="e.g. Smith v Jones 2024",
                )
                file_upload_new = gr.File(
                    label="Initial files (optional)",
                    file_count="multiple",
                )
                folder_upload_new = gr.File(
                    label="Initial folder (optional)",
                    file_count="multiple",
                    elem_id="folder-upload-new",
                )
                save_matter_btn = gr.Button("Create matter", variant="primary")
                upload_status = gr.HTML()

            browse_btn = None
        else:
            with gr.Row():
                folder_path_box = gr.Textbox(
                    label="Matter folder",
                    placeholder="Paste a path or click Browse",
                    scale=4,
                )
                browse_btn = gr.Button("Browse", variant="secondary", scale=1, min_width=80)
                matter_status = gr.Textbox(
                    label="",
                    interactive=False,
                    scale=1,
                    elem_classes=["compact-id"],
                )
                file_upload = None

        with gr.Row():
            query = gr.Textbox(
                label="Question",
                placeholder="What are the key claims and defenses? What damages are alleged?",
                lines=2,
                scale=4,
            )
            research_mode = gr.Dropdown(
                label="Research Mode",
                choices=[
                    ("Simple", "simple"),
                    ("Deep", "deep"),
                    ("Sebih Special", "sebih_special"),
                ],
                value="simple",
                scale=1,
                min_width=220,
            )
        with gr.Row():
            submit_btn = gr.Button("Investigate", variant="primary", scale=4)
            stop_btn = gr.Button("Stop", variant="stop", scale=1)
            clear_matter_btn = gr.Button("Reset matter", variant="stop", size="sm", scale=1)

        status_box = gr.Textbox(
            label="Status",
            interactive=False,
            elem_classes=["status-bar"],
        )

        # ==================================================================
        # MAIN AREA: Analysis (wide) + Intelligence Sidebar (narrow)
        # ==================================================================
        with gr.Row():

            # ---------- LEFT: Analysis output ----------
            with gr.Column(scale=3):
                run_output = gr.Chatbot(
                    label="Conversation",
                    value=[
                        {
                            "role": "assistant",
                            "content": (
                                "Select documents above, type a question, and click **Investigate**. "
                                "Irys will read every document, extract structured facts, map the issues, "
                                "and answer in a multi-turn thread."
                            ),
                        }
                    ],
                    height=420,
                )

                with gr.Accordion("Reasoning Trace — watch Irys think step by step", open=False):
                    trace_box = gr.Textbox(
                        label="Live reasoning steps during investigation, structured ledger after",
                        lines=25,
                        interactive=False,
                        autoscroll=True,
                        elem_classes=["mono"],
                    )

                with gr.Accordion("Sources, Citations & Run Diagnostics", open=False):
                    citations_box = gr.Textbox(
                        label="Documents, pages, and auto-generated run safeguards",
                        lines=12,
                        interactive=False,
                    )

            # ---------- RIGHT: Intelligence sidebar ----------
            with gr.Column(scale=1, min_width=280, elem_classes=["intelligence-sidebar"]):
                # UI-6 privilege mode toggle + P0.3 review badge
                # preserved at the top of the new dashboard sidebar.
                # The collaborator's UI-audit overhaul restructured
                # the sidebar around _fmt_overview_panel, but the
                # privilege toggle (clean vs. internal) and review-
                # queue count chip are operational controls that
                # attorneys need visible — not information to hide
                # behind the overview.
                privilege_banner_md = gr.HTML(_fmt_privilege_banner("clean"))
                privilege_toggle = gr.Radio(
                    choices=[
                        "Clean mode (hide privileged)",
                        "Internal mode (show all)",
                    ],
                    value="Clean mode (hide privileged)",
                    label="Privilege mode",
                    info=(
                        "Clean redacts privileged material in the timeline "
                        "and evidence matrix. Flip to Internal only in a "
                        "private attorney workspace."
                    ),
                )
                review_badge_md = gr.HTML("")

                # New dashboard overview panel (collaborator's UI
                # audit). The dark navy panel is self-contained
                # inside the HTML — no Gradio column styling needed.
                overview_md = gr.HTML(_fmt_overview_panel({}))
                # Hidden components kept for callback compatibility
                # with existing wiring.
                issues_md = gr.HTML(visible=False)
                gaps_md = gr.Markdown(visible=False)
                assumptions_md = gr.Markdown(visible=False)

                refresh_sidebar_btn = gr.Button("↻ Refresh", variant="secondary", size="sm")

        # ==================================================================
        # DETAIL ACCORDIONS (below main area)
        # ==================================================================

        with gr.Accordion("Extracted Facts — every fact from your documents, with source and confidence", open=False):
            gr.Markdown(
                "Each fact shows where it came from, how confident Irys is, and how it's characterized "
                "(e.g., *alleged* in a complaint vs. *operative* in a signed contract). "
                "If something is wrong, correct it below — Irys will automatically update any "
                "conclusions that depended on that fact."
            )
            # gr.HTML (not gr.Markdown) so the trust-pill and belief-
            # pill styled HTML table renders without sanitize_html
            # stripping inline styles.
            assertions_md = gr.HTML(
                "<div class='viz-empty'>Facts will appear here after an investigation.</div>"
            )
            refresh_assertions_btn = gr.Button("Refresh Facts", variant="secondary", size="sm")

            with gr.Accordion("Correct a fact", open=False):
                gr.Markdown(
                    "Copy a Fact ID from the table above, choose the correct characterization, "
                    "and explain why. Irys will propagate the correction through its analysis."
                )
                with gr.Row():
                    correction_assertion_id = gr.Textbox(
                        label="Fact ID (copy from table above)", scale=2,
                    )
                    correction_new_state = gr.Dropdown(
                        label="Correct characterization",
                        choices=_correction_dropdown_choices("legal"),
                        scale=1,
                    )
                gr.HTML(
                    "<div class='belief-legend'>"
                    + "".join(
                        f"<span class='belief-pill belief-{s}'>{s}</span> "
                        for s in [
                            "alleged", "argued", "admitted", "operative", "performed",
                            "disputed", "superseded", "withdrawn", "inferred", "resolved",
                        ]
                    )
                    + "</div>"
                )
                correction_reason = gr.Textbox(
                    label="Why is this correction needed?",
                    placeholder="e.g. This is from the signed contract, not the complaint",
                    lines=2,
                )
                correction_btn = gr.Button("Apply Correction", variant="primary")
                correction_result = gr.HTML("<div class='correction-result-empty'>Apply a correction to see the result here.</div>")

        # ==================================================================
        # REVIEW INBOX — what AI extractions need the attorney's sign-off
        # ==================================================================

        with gr.Accordion(
            "Review Inbox — findings awaiting your sign-off",
            open=False,
        ):
            gr.Markdown(
                "Irys flags every AI-extracted finding as a **candidate** "
                "until you verify it. Proof-critical findings, contradicted "
                "facts, and issue-linked evidence appear first. "
                "Verify what's right, reject what's wrong — rejected findings "
                "are removed from clean synthesis and their downstream "
                "evidence is marked stale immediately."
            )
            review_queue_html = gr.HTML(
                "<div class='viz-empty'>Run an investigation to populate the review queue.</div>"
            )
            refresh_review_btn = gr.Button("Refresh review queue", variant="secondary", size="sm")

            gr.Markdown("---")
            gr.Markdown("#### Verify or reject a finding")
            review_target = gr.Dropdown(
                label="Pick a finding from the queue above",
                choices=[],
                value=None,
                allow_custom_value=False,
            )
            with gr.Accordion(
                "Where this came from + review history",
                open=False,
            ):
                source_drawer_html = gr.HTML(
                    "<div class='viz-empty'>Pick a finding and click "
                    "<em>Show source &amp; history</em> to see the "
                    "source document, how it was extracted, and every "
                    "review action on it.</div>"
                )
                show_source_btn = gr.Button(
                    "Show source & history",
                    variant="secondary", size="sm",
                )
            with gr.Row():
                verify_note = gr.Textbox(
                    label="Verification note (optional)",
                    placeholder="e.g. Confirmed in signed MSA §4.2",
                    scale=4,
                )
                verify_btn = gr.Button("✓ Verify", variant="primary", scale=1, min_width=120)
            with gr.Row():
                reject_reason = gr.Textbox(
                    label="Rejection reason (required)",
                    placeholder="e.g. Misread — the contract says 30 days, not 15",
                    scale=4,
                )
                reject_btn = gr.Button("✗ Reject", variant="stop", scale=1, min_width=120)
            review_action_result = gr.Markdown("")

            gr.Markdown("---")
            gr.Markdown("#### Bulk verify a whole document")
            gr.Markdown(
                "When you've reviewed an entire contract or pleading, "
                "approve every candidate fact from it in one action. "
                "Pick a document from the list — docs with the most "
                "pending facts are at the top."
            )
            with gr.Row():
                # UI-6: searchable dropdown fed by
                # list_reviewable_documents. Labels include pending /
                # reviewed counts so attorneys can triage the queue by
                # document without opening every one. allow_custom_value
                # so typing a path that isn't listed still works.
                bulk_doc_ref = gr.Dropdown(
                    label="Document",
                    choices=[],
                    value=None,
                    allow_custom_value=True,
                    filterable=True,
                    scale=4,
                )
                review_facts_btn = gr.Button(
                    "Review facts first", variant="secondary",
                    scale=1, min_width=160,
                )
                bulk_verify_btn = gr.Button(
                    "Verify all from document", variant="primary",
                    scale=1, min_width=180,
                )
                refresh_doc_picker_btn = gr.Button(
                    "↻", variant="secondary", scale=0, min_width=40,
                    size="sm",
                )
            bulk_verify_result = gr.Markdown("")
            # Batch-review panel — renders the candidate facts from
            # the selected document with full metadata, pre-checked,
            # so the attorney can uncheck anything they don't want
            # to promote and then verify the subset in one action.
            batch_review_facts = gr.CheckboxGroup(
                label="Candidate facts from this document (uncheck any you don't want to verify)",
                choices=[],
                value=[],
                visible=False,
                interactive=True,
            )
            batch_review_metadata = gr.HTML(visible=False)
            batch_verify_selected_btn = gr.Button(
                "Verify selected", variant="primary",
                visible=False,
            )
            batch_review_result = gr.Markdown("", visible=False)

        with gr.Accordion("Financials — payments, damages, and numeric disputes", open=False):
            gr.Markdown(
                "Invoices, payments, damages claims, and numeric conflicts — "
                "pulled directly from your documents and reconciled. "
                "If two documents disagree on an amount, Irys flags the conflict."
            )
            quant_md = gr.HTML("<div class='viz-empty'>Financial data will appear here after an investigation.</div>")
            refresh_quant_btn = gr.Button("Refresh Financials", variant="secondary", size="sm")

        with gr.Accordion("Timeline — dated events across the matter", open=False):
            gr.Markdown(
                "Chronological events from dated facts and claims pinned to a point in time."
            )
            timeline_html = gr.HTML("<div class='viz-empty'>Timeline events will appear here after an investigation.</div>")
            refresh_timeline_btn = gr.Button("Refresh Timeline", variant="secondary", size="sm")

        with gr.Accordion("Evidence Matrix — which documents support which issues", open=False):
            gr.Markdown(
                "Rows are issues, columns are source documents, and cells show how strongly "
                "each document supports or attacks the issue."
            )
            evidence_matrix_html = gr.HTML("<div class='viz-empty'>Evidence matrix will appear here after an investigation.</div>")
            refresh_evidence_btn = gr.Button("Refresh Evidence Matrix", variant="secondary", size="sm")

        with gr.Accordion("Proof State — sufficiency and predicate coverage by issue", open=False):
            gr.Markdown(
                "Per-issue proof analysis: sufficiency scores, predicate coverage, "
                "trust-weighted support vs. attack balance, and advocacy-only warnings."
            )
            proof_state_html = gr.HTML("<div class='viz-empty'>Proof state will appear here after an investigation.</div>")
            refresh_proof_btn = gr.Button("Refresh Proof State", variant="secondary", size="sm")

        with gr.Accordion("Communication Graph — who appears in which documents", open=False):
            gr.Markdown(
                "Maps which people and companies appear in which documents and highlights "
                "repeated pairings (e.g. frequent correspondents)."
            )
            communication_html = gr.HTML("<div class='viz-empty'>Communication graph will appear here after an investigation.</div>")
            refresh_comm_btn = gr.Button("Refresh Communication Graph", variant="secondary", size="sm")

        with gr.Accordion("LLM Analytics — cost, latency, and stage mix", open=False):
            gr.Markdown(
                "Shows recent LLM calls, spend by stage, and the current tier mix."
            )
            llm_analytics_html = gr.HTML("<div class='viz-empty'>LLM analytics will appear here after an investigation.</div>")
            refresh_llm_btn = gr.Button("Refresh LLM Analytics", variant="secondary", size="sm")

        with gr.Accordion("Steering Controls — redirect or resume an investigation", open=False):
            gr.Markdown(
                "**Redirect:** If Irys is investigating the wrong thing, redirect it to focus "
                "on a specific issue. The investigation will shift focus at the next step.\n\n"
                "**Resume:** If you stopped an investigation, you can pick up where it left off."
            )
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Redirect to a Specific Issue")
                    redirect_issue_id = gr.Textbox(
                        label="Issue ID (copy from Issues panel, or auto-filled from gap analysis)",
                        placeholder="Auto-populated when you click Refresh All",
                    )
                    redirect_btn = gr.Button("Redirect Investigation", variant="primary")
                    redirect_result = gr.Textbox(label="Result", interactive=False)
                with gr.Column():
                    gr.Markdown("#### Resume Stopped Investigation")
                    gr.Markdown(
                        "Click below to continue the last investigation from where it was stopped. "
                        "If you changed the research mode above, the resumed run will use that budget."
                    )
                    resume_btn = gr.Button("Resume Last Investigation", variant="primary")
                    resume_result = gr.Textbox(label="Result", interactive=False)

        # ==================================================================
        # WIRING
        # ==================================================================

        # --- Folder detection: check for existing .irys/ when path changes ---
        def _check_folder(path: str) -> str:
            if not path or not path.strip():
                return ""
            p = pathlib.Path(path.strip())
            if not p.exists():
                return "Folder not found"
            if not p.is_dir():
                return "Not a folder"
            doc_count = sum(1 for f in p.iterdir() if f.is_file() and not f.name.startswith("."))
            has_model = (p / ".irys" / "matter.sqlite3").exists()
            if has_model:
                return f"Existing matter ({doc_count} files) — continuing from previous analysis"
            return f"New matter ({doc_count} files) — will start fresh"

        def _browse_folder() -> tuple[str, str]:
            """Open native folder picker dialog and return (path, status)."""
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                folder = filedialog.askdirectory(title="Select matter folder")
                root.destroy()
                if folder:
                    return folder, _check_folder(folder)
                return gr.update(), gr.update()
            except Exception:
                return gr.update(), "Folder picker unavailable — paste a path instead"

        if _s3_mode:
            # Select a matter → show card, reveal workspace, populate file list
            def _on_matter_select(name):
                if not name:
                    return (
                        "",
                        _fmt_matter_card("", []),
                        gr.update(visible=False),
                        gr.update(choices=[], value=None),
                    )
                files = _list_s3_matter_files(name)
                return (
                    name,
                    _fmt_matter_card(name, files),
                    gr.update(visible=True),
                    gr.update(choices=files, value=None),
                )

            matter_dropdown.change(
                fn=_on_matter_select,
                inputs=[matter_dropdown],
                outputs=[repo_path, matter_card_html, matter_workspace, matter_files_dropdown],
            )

            # Refresh matter list from S3
            refresh_matters_btn.click(
                fn=lambda: gr.update(choices=_list_s3_matter_names()),
                inputs=[],
                outputs=[matter_dropdown],
            )

            # Remove a single file from the selected matter
            def _on_delete_file(matter_name, filename):
                if not matter_name:
                    return gr.update(), gr.update(), _fmt_ws_status("No matter selected", "err")
                if not filename:
                    return gr.update(), gr.update(), _fmt_ws_status("Select a file to remove first", "info")
                files, _ = _delete_s3_matter_file(matter_name, filename)
                return (
                    gr.update(choices=files, value=None),
                    _fmt_matter_card(matter_name, files),
                    _fmt_ws_status(f"Removed '{filename}'", "ok"),
                )

            delete_file_btn.click(
                fn=_on_delete_file,
                inputs=[repo_path, matter_files_dropdown],
                outputs=[matter_files_dropdown, matter_card_html, file_manage_status],
            )

            # Delete entire matter from S3
            def _on_delete_matter(matter_name):
                if not matter_name:
                    return gr.update(), None, _fmt_matter_card("", []), gr.update(visible=False), gr.update(choices=[], value=None), _fmt_ws_status("No matter selected", "err")
                matters, _ = _delete_s3_matter(matter_name)
                return (
                    gr.update(choices=matters, value=None),
                    None,
                    _fmt_matter_card("", []),
                    gr.update(visible=False),
                    gr.update(choices=[], value=None),
                    _fmt_ws_status(f"Deleted matter '{matter_name}'", "ok"),
                )

            delete_matter_btn.click(
                fn=_on_delete_matter,
                inputs=[repo_path],
                outputs=[matter_dropdown, repo_path, matter_card_html, matter_workspace, matter_files_dropdown, file_manage_status],
            )

            # Add files to the currently selected matter
            def _on_add_files(files, matter_name):
                if not matter_name:
                    return gr.update(), _fmt_ws_status("Select a matter first", "err"), gr.update(), gr.update()
                if not files:
                    return gr.update(), _fmt_ws_status("No files selected", "info"), gr.update(), gr.update()
                try:
                    _, status_msg = _upload_files_to_s3_matter(files, matter_name.strip())
                    updated = _list_s3_matter_files(matter_name.strip())
                    tone = "warn" if "failed" in status_msg.lower() else "ok"
                    return (
                        gr.update(choices=updated, value=None),
                        _fmt_ws_status(status_msg, tone),
                        None,
                        _fmt_matter_card(matter_name, updated),
                    )
                except Exception as e:
                    return gr.update(), _fmt_ws_status(f"Upload failed: {e}", "err"), gr.update(), gr.update()

            add_files_btn.click(
                fn=_on_add_files,
                inputs=[file_upload, repo_path],
                outputs=[matter_files_dropdown, file_manage_status, file_upload, matter_card_html],
            )

            def _on_add_folder(files, matter_name):
                if not matter_name:
                    return gr.update(), _fmt_ws_status("Select a matter first", "err"), gr.update(), gr.update()
                if not files:
                    return gr.update(), _fmt_ws_status("No folder selected", "info"), gr.update(), gr.update()
                try:
                    _, status_msg = _upload_files_to_s3_matter(files, matter_name.strip())
                    updated = _list_s3_matter_files(matter_name.strip())
                    tone = "warn" if "failed" in status_msg.lower() else "ok"
                    return (
                        gr.update(choices=updated, value=None),
                        _fmt_ws_status(status_msg, tone),
                        None,
                        _fmt_matter_card(matter_name, updated),
                    )
                except Exception as e:
                    return gr.update(), _fmt_ws_status(f"Upload failed: {e}", "err"), gr.update(), gr.update()

            add_folder_btn.click(
                fn=_on_add_folder,
                inputs=[folder_upload, repo_path],
                outputs=[matter_files_dropdown, file_manage_status, folder_upload, matter_card_html],
            )

            # Create a new matter (with optional initial files/folder), then select it
            def _on_save_matter(files, folder_files, name):
                if not name or not name.strip():
                    return gr.update(), "", _fmt_ws_status("Enter a matter name first", "err"), gr.update(visible=False), gr.update(choices=[], value=None), gr.update()
                try:
                    all_files = (files or []) + (folder_files or [])
                    status_msg = ""
                    if all_files:
                        display_name, status_msg = _upload_files_to_s3_matter(all_files, name.strip())
                    else:
                        display_name = _sanitize_matter_name(name.strip()).replace("_", " ")
                    names = _list_s3_matter_names()
                    new_files = _list_s3_matter_files(display_name)
                    if status_msg and "failed" in status_msg.lower():
                        ok_msg = _fmt_ws_status(f"Created '{display_name}' — {status_msg}", "warn")
                    else:
                        ok_msg = _fmt_ws_status(f"Created '{display_name}'", "ok")
                    return (
                        gr.update(choices=names, value=display_name),
                        display_name,
                        ok_msg,
                        gr.update(visible=True),
                        gr.update(choices=new_files, value=None),
                        _fmt_matter_card(display_name, new_files),
                    )
                except Exception as e:
                    return gr.update(), "", _fmt_ws_status(f"Create failed: {e}", "err"), gr.update(visible=False), gr.update(choices=[], value=None), gr.update()

            save_matter_btn.click(
                fn=_on_save_matter,
                inputs=[file_upload_new, folder_upload_new, matter_name_input],
                outputs=[matter_dropdown, repo_path, upload_status, matter_workspace, matter_files_dropdown, matter_card_html],
            )
        else:
            browse_btn.click(
                fn=_browse_folder,
                inputs=[],
                outputs=[folder_path_box, matter_status],
            )

            folder_path_box.change(
                fn=_check_folder,
                inputs=[folder_path_box],
                outputs=[matter_status],
            )

            # Keep repo_path hidden state in sync with the visible textbox
            folder_path_box.change(
                fn=lambda p: p,
                inputs=[folder_path_box],
                outputs=[repo_path],
            )

        # Clear removes .irys/ so next run starts fresh
        def _clear_and_report(path: str) -> tuple[str, str]:
            if not path or not path.strip():
                return "", "No folder selected"
            result = _clear_matter(path)
            return _check_folder(path), result[1]

        if not _s3_mode:
            clear_matter_btn.click(
                fn=_clear_and_report,
                inputs=[repo_path],
                outputs=[matter_status, status_box],
            )

        # --- Sidebar refresh (all panels at once) ---
        def _refresh_all(mid):
            overview = state.load_overview(mid)
            issues = state.load_issues(mid)
            gaps_text, top_issue = state.load_gaps(mid)
            assumptions = state.load_assumptions(mid)
            assertions = state.load_assertions(mid)
            quant = state.load_quant(mid)
            timeline = state.load_timeline(mid)
            evidence = state.load_evidence_matrix(mid)
            communication = state.load_communication_map(mid)
            llm_analytics = state.load_llm_analytics(mid)
            proof_state = state.load_proof_state(mid)
            review_badge = state.load_review_count_badge(mid)
            doc_choices = state.load_document_picker_choices(mid)
            domain = "legal"
            if mid and mid != "—":
                try:
                    ov_data = _run_async(state.backend().get_overview(mid))
                    dc = ov_data.get("domain_composition", {}) if isinstance(ov_data, dict) else {}
                    domain = dc.get("primary_domain_profile_id", "legal") if isinstance(dc, dict) else "legal"
                except Exception:
                    pass
            return (
                review_badge,
                overview,
                issues,
                gaps_text,
                assumptions,
                assertions,
                quant,
                timeline,
                evidence,
                communication,
                llm_analytics,
                proof_state,
                top_issue,
                gr.update(choices=doc_choices),
                gr.update(choices=_correction_dropdown_choices(domain), value=None),
            )

        # --- Investigation stream ---
        run_outputs = [run_output, trace_box, citations_box, status_box, matter_id_box]

        if _s3_mode:
            import uuid as _uuid

            def _stream_s3(query_text, matter_name, mode):
                if not matter_name or not matter_name.strip():
                    yield ("", "", "", "❌ Select a matter first", "")
                    return
                # Stable per-matter session_id so the .irys/ matter DB at
                # temp_dir/.irys/matter.sqlite3 survives between runs on the
                # same matter (warm cache → quick summary routing on re-queries).
                import hashlib as _hashlib
                session_id = _hashlib.sha256(matter_name.strip().lower().encode()).hexdigest()[:16]
                try:
                    temp_dir = _download_s3_matter_to_temp(matter_name.strip(), session_id)
                except Exception as e:
                    yield ("", "", "", f"❌ Failed to load matter: {e}", "")
                    return
                try:
                    yield from state.stream_investigation_session(query_text, str(temp_dir), mode)
                finally:
                    # Remove document files but preserve .irys/ (matter DB) for the next run.
                    if temp_dir.exists():
                        for _item in temp_dir.iterdir():
                            if _item.name == ".irys":
                                continue
                            if _item.is_file():
                                _item.unlink(missing_ok=True)
                            elif _item.is_dir():
                                shutil.rmtree(_item, ignore_errors=True)

            submit_btn.click(
                fn=_stream_s3,
                inputs=[query, repo_path, research_mode],
                outputs=run_outputs,
            ).then(
                fn=_refresh_all,
                inputs=[matter_id_box],
                outputs=[
                    review_badge_md,
                    overview_md,
                    issues_md,
                    gaps_md,
                    assumptions_md,
                    assertions_md,
                    quant_md,
                    timeline_html,
                    evidence_matrix_html,
                    communication_html,
                    llm_analytics_html,
                    proof_state_html,
                    redirect_issue_id,
                    bulk_doc_ref,
                    correction_new_state,
                ],
            )
        else:
            submit_btn.click(
                fn=state.stream_investigation_session,
                inputs=[query, repo_path, research_mode],
                outputs=run_outputs,
            ).then(
                fn=_refresh_all,
                inputs=[matter_id_box],
                outputs=[
                    review_badge_md,
                    overview_md,
                    issues_md,
                    gaps_md,
                    assumptions_md,
                    assertions_md,
                    quant_md,
                    timeline_html,
                    evidence_matrix_html,
                    communication_html,
                    llm_analytics_html,
                    proof_state_html,
                    redirect_issue_id,
                    bulk_doc_ref,
                    correction_new_state,
                ],
            )
        stop_btn.click(fn=state.stop_investigation, inputs=[], outputs=[])

        refresh_sidebar_btn.click(
            fn=_refresh_all,
            inputs=[matter_id_box],
            outputs=[
                review_badge_md,
                overview_md,
                issues_md,
                gaps_md,
                assumptions_md,
                assertions_md,
                quant_md,
                timeline_html,
                evidence_matrix_html,
                communication_html,
                llm_analytics_html,
                proof_state_html,
                redirect_issue_id,
                bulk_doc_ref,
                correction_new_state,
            ],
        )

        # --- Detail panel refreshes ---
        refresh_assertions_btn.click(
            fn=lambda mid: state.load_assertions(mid),
            inputs=[matter_id_box],
            outputs=[assertions_md],
        )
        refresh_quant_btn.click(
            fn=lambda mid: state.load_quant(mid),
            inputs=[matter_id_box],
            outputs=[quant_md],
        )
        refresh_timeline_btn.click(
            fn=lambda mid: state.load_timeline(mid),
            inputs=[matter_id_box],
            outputs=[timeline_html],
        )
        refresh_evidence_btn.click(
            fn=lambda mid: state.load_evidence_matrix(mid),
            inputs=[matter_id_box],
            outputs=[evidence_matrix_html],
        )
        refresh_proof_btn.click(
            fn=lambda mid: state.load_proof_state(mid),
            inputs=[matter_id_box],
            outputs=[proof_state_html],
        )
        refresh_comm_btn.click(
            fn=lambda mid: state.load_communication_map(mid),
            inputs=[matter_id_box],
            outputs=[communication_html],
        )
        refresh_llm_btn.click(
            fn=lambda mid: state.load_llm_analytics(mid),
            inputs=[matter_id_box],
            outputs=[llm_analytics_html],
        )

        # --- UI-6 privilege mode toggle + document picker ---
        def _on_privilege_toggle(label, mid):
            # Flip AppState.policy_audience, re-render the banner, and
            # re-load the two panels that actually honour the audience
            # flag (timeline + evidence matrix). Other panels don't
            # depend on audience so we leave them alone to keep the
            # refresh tight.
            banner = state.set_policy_audience(label)
            return (
                banner,
                state.load_timeline(mid),
                state.load_evidence_matrix(mid),
            )

        privilege_toggle.change(
            fn=_on_privilege_toggle,
            inputs=[privilege_toggle, matter_id_box],
            outputs=[privilege_banner_md, timeline_html, evidence_matrix_html],
        )

        refresh_doc_picker_btn.click(
            fn=lambda mid: gr.update(
                choices=state.load_document_picker_choices(mid),
            ),
            inputs=[matter_id_box],
            outputs=[bulk_doc_ref],
        )

        # --- Correction ---
        def _fmt_correction_result(result_text: str) -> str:
            ok = result_text.startswith("✅")
            body = result_text[2:].strip() if result_text[:2] in ("✅", "❌") else result_text
            label = "Correction applied" if ok else "Correction failed"
            icon = "✅" if ok else "❌"
            css_class = "correction-ok" if ok else "correction-err"
            return (
                f"<div class='correction-result {css_class}'>"
                f"<span class='correction-icon'>{icon}</span>"
                f"<div><strong>{label}</strong><br>"
                f"<span class='correction-detail'>{_escape(body)}</span></div>"
                f"</div>"
            )

        def _correct_and_refresh(mid, aid, new_state_str, reason):
            result_text = state.do_correct_assertion(mid, aid, new_state_str, reason)
            if result_text.startswith("\u2705"):
                return (
                    _fmt_correction_result(result_text),
                    state.load_assertions(mid),
                    state.load_issues(mid),
                    state.load_overview(mid),
                )
            return _fmt_correction_result(result_text), gr.update(), gr.update(), gr.update()

        correction_btn.click(
            fn=_correct_and_refresh,
            inputs=[matter_id_box, correction_assertion_id, correction_new_state, correction_reason],
            outputs=[correction_result, assertions_md, issues_md, overview_md],
        )

        # --- Review Inbox wiring ---
        def _refresh_review_and_drawer(mid):
            queue_html, dropdown_update = state.load_review_queue(mid)
            new_value = dropdown_update.get("value") if isinstance(dropdown_update, dict) else None
            drawer = state.load_source_drawer(mid, new_value or "")
            badge = state.load_review_count_badge(mid)
            return queue_html, dropdown_update, drawer, badge

        refresh_review_btn.click(
            fn=_refresh_review_and_drawer,
            inputs=[matter_id_box],
            outputs=[review_queue_html, review_target, source_drawer_html, review_badge_md],
        )

        # Drawer is refreshed when (a) the queue is refreshed, (b) the
        # user explicitly clicks the "Show source" button below, or
        # (c) a verify/reject action fires. We deliberately do NOT
        # wire review_target.change to the drawer: in Gradio 6.3
        # .change() fires on function-driven value updates as well as
        # real user input, which would overwrite the "show the review
        # event I just wrote" state right after verify/reject.
        show_source_btn.click(
            fn=state.load_source_drawer,
            inputs=[matter_id_box, review_target],
            outputs=[source_drawer_html],
        )

        def _verify_and_refresh(mid, target, note):
            # Snapshot target before the write so the drawer shows the
            # transition we just made (prior_target).
            prior_target = target
            result = state.do_verify_target(mid, target, note)
            snap = state.load_post_review_snapshot(mid, prior_target)
            doc_choices = state.load_document_picker_choices(mid)
            return (
                result,
                snap["queue_html"],
                snap["dropdown"],
                snap["drawer_html"],
                snap["badge_html"],
                snap["assertions_html"],
                snap["issues_html"],
                snap["overview_html"],
                "",  # clear the note field
                gr.update(choices=doc_choices),
            )

        verify_btn.click(
            fn=_verify_and_refresh,
            inputs=[matter_id_box, review_target, verify_note],
            outputs=[
                review_action_result, review_queue_html, review_target,
                source_drawer_html, review_badge_md,
                assertions_md, issues_md, overview_md, verify_note,
                bulk_doc_ref,
            ],
        )

        def _reject_and_refresh(mid, target, reason):
            prior_target = target
            result = state.do_reject_target(mid, target, reason)
            snap = state.load_post_review_snapshot(mid, prior_target)
            doc_choices = state.load_document_picker_choices(mid)
            return (
                result,
                snap["queue_html"],
                snap["dropdown"],
                snap["drawer_html"],
                snap["badge_html"],
                snap["assertions_html"],
                snap["issues_html"],
                snap["overview_html"],
                "",  # clear the reason field
                gr.update(choices=doc_choices),
            )

        reject_btn.click(
            fn=_reject_and_refresh,
            inputs=[matter_id_box, review_target, reject_reason],
            outputs=[
                review_action_result, review_queue_html, review_target,
                source_drawer_html, review_badge_md,
                assertions_md, issues_md, overview_md, reject_reason,
                bulk_doc_ref,
            ],
        )

        def _bulk_verify_doc_and_refresh(mid, doc_ref):
            result = state.do_bulk_verify_by_document(mid, doc_ref)
            queue_html, dropdown_update = state.load_review_queue(mid)
            return (
                result, queue_html, dropdown_update,
                state.load_review_count_badge(mid),
                state.load_assertions(mid),
                state.load_issues(mid),
                state.load_overview(mid),
            )

        bulk_verify_btn.click(
            fn=_bulk_verify_doc_and_refresh,
            inputs=[matter_id_box, bulk_doc_ref],
            outputs=[
                bulk_verify_result, review_queue_html, review_target,
                review_badge_md,
                assertions_md, issues_md, overview_md,
            ],
        )

        # Batch-review: "Review facts first" loads the checkable list;
        # "Verify selected" commits the uncheck-filtered subset. Hides
        # the bulk_verify_result markdown so the two flows' status
        # lines don't stack on top of each other.
        review_facts_btn.click(
            fn=state.load_batch_review_facts,
            inputs=[matter_id_box, bulk_doc_ref],
            outputs=[
                batch_review_facts, batch_review_metadata,
                batch_verify_selected_btn, batch_review_result,
            ],
        )

        def _batch_verify_and_refresh(mid, selected):
            updates = state.do_batch_verify_selected(mid, selected)
            return (
                *updates,
                state.load_review_queue(mid)[0],
                state.load_review_queue(mid)[1],
                state.load_review_count_badge(mid),
                state.load_assertions(mid),
                state.load_issues(mid),
                state.load_overview(mid),
            )

        batch_verify_selected_btn.click(
            fn=_batch_verify_and_refresh,
            inputs=[matter_id_box, batch_review_facts],
            outputs=[
                batch_review_facts, batch_review_metadata,
                batch_verify_selected_btn, batch_review_result,
                review_queue_html, review_target,
                review_badge_md,
                assertions_md, issues_md, overview_md,
            ],
        )

        # --- Steering: redirect uses current run_id automatically ---
        redirect_btn.click(
            fn=lambda mid, issue_id: state.do_redirect(mid, state.current_run_id or "", issue_id),
            inputs=[matter_id_box, redirect_issue_id],
            outputs=[redirect_result],
        )

        # --- Steering: resume uses current run_id automatically ---
        resume_btn.click(
            fn=lambda mid, mode: state.do_resume(mid, state.current_run_id or "", mode),
            inputs=[matter_id_box, research_mode],
            outputs=[resume_result],
        )

        # Inject JS: clicking an assertions row fills the Fact ID textbox
        _fact_select_js = """
() => {
    window.irysSelectFact = function(id) {
        const labels = document.querySelectorAll('label');
        for (const lbl of labels) {
            if (lbl.textContent.includes('Fact ID')) {
                const box = lbl.closest('.block')?.querySelector('textarea, input[type=text]');
                if (box) {
                    // adv#14 Finding #5: pick the prototype matching
                    // the actual element. The old code OR'd both and
                    // always picked HTMLInputElement — calling that
                    // setter on a textarea raises an illegal-invocation
                    // error and the click does nothing. Gradio renders
                    // Textbox(lines=1) as input and Textbox(lines≥2)
                    // as textarea — we must handle both.
                    const proto = (box instanceof HTMLTextAreaElement)
                        ? window.HTMLTextAreaElement.prototype
                        : window.HTMLInputElement.prototype;
                    const desc = Object.getOwnPropertyDescriptor(proto, 'value');
                    if (desc && desc.set) {
                        desc.set.call(box, id);
                        box.dispatchEvent(new Event('input', { bubbles: true }));
                        box.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                    break;
                }
            }
        }
    };
}
"""
        demo.load(fn=None, js=_fact_select_js)

        # Inject JS to enable folder selection on the folder-upload inputs
        if _s3_mode:
            _folder_upload_js = """
() => {
    const applyFolderAttr = () => {
        ['folder-upload', 'folder-upload-new'].forEach(id => {
            const el = document.getElementById(id);
            if (!el) return;
            el.querySelectorAll('input[type=file]').forEach(inp => {
                inp.setAttribute('webkitdirectory', '');
                inp.setAttribute('directory', '');
                inp.setAttribute('multiple', '');
            });
        });
    };
    applyFolderAttr();
    new MutationObserver(applyFolderAttr).observe(document.body, {childList: true, subtree: true});
}
"""
            demo.load(fn=None, js=_folder_upload_js)

    return demo


def _load_dotenv() -> None:
    """Load .env file from project root into os.environ (stdlib only)."""
    env_path = pathlib.Path(__file__).resolve().parents[3] / ".env"
    if not env_path.is_file():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key and key not in os.environ:
                os.environ[key] = value


def main():
    import argparse

    _load_dotenv()

    parser = argparse.ArgumentParser(description="Irys RLM UI")
    parser.add_argument("--api-key", help="Gemini API key")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("No GEMINI_API_KEY set — pass --api-key or set the env var")

    demo = create_app(api_key=api_key)
    demo.launch(server_port=args.port, share=args.share, theme=_theme, css=_css)


if __name__ == "__main__":
    main()
