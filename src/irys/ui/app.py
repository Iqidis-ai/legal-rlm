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
import json
import logging
import math
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
    from botocore.config import Config
    # max_pool_connections defaults to 10, which throttles the 16-way parallel
    # upload/download paths. Bump to 32 so threaded callers don't queue on the
    # connection pool for matter transfers with hundreds of files.
    return boto3.client(
        "s3",
        region_name=os.getenv("S3_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID") or None,
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY") or None,
        config=Config(max_pool_connections=32),
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
    except Exception as exc:
        logger.warning("Failed to list S3 matter names: %s", exc)
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
    except Exception as exc:
        logger.warning("Failed to count S3 matter docs for %r: %s", matter_name, exc)
        return "? documents"


def _sanitize_s3_relpath(relpath: str) -> str:
    """Sanitize a browser-supplied relative path for use as an S3 key suffix.

    Splits on / (and \\), drops empty / `.` / `..` segments to block traversal,
    strips whitespace per segment. Returns "" when nothing usable remains.
    """
    parts: list[str] = []
    for raw in relpath.replace("\\", "/").split("/"):
        part = raw.strip()
        if not part or part in (".", ".."):
            continue
        parts.append(part)
    return "/".join(parts)


def _upload_files_to_s3_matter(
    uploaded_files: list,
    name: str,
    relpath_json: Optional[str] = None,
) -> tuple[str, str]:
    """Upload Gradio files to S3 under matters/<name>/.

    Appends to existing matter if the name already exists. Per-file failures
    are isolated (do not abort the batch), retried with exponential backoff,
    and reported in the status string. Filename collisions get a numeric
    suffix instead of silently overwriting.

    When `relpath_json` is provided (a JSON object of `{leaf_name: [relpath,
    ...]}` produced client-side from `webkitRelativePath`), folder structure
    is preserved by using the relpath as the S3 key suffix. Multiple files
    sharing a leaf name in different subfolders are matched in upload order
    by popping the head of each leaf's relpath list.

    Returns (display_name, status_message).
    """
    bucket = _s3_bucket()
    if not bucket:
        return name, "S3_BUCKET not configured — cannot upload"
    safe = _sanitize_matter_name(name)
    prefix = f"{_s3_matters_base_prefix()}/{safe}"
    s3 = _get_s3_client()

    # Parse relpath map from JS (best-effort; falls back to flat upload).
    relpath_map: dict[str, list[str]] = {}
    if relpath_json:
        try:
            raw = json.loads(relpath_json)
            if isinstance(raw, dict):
                for leaf, paths in raw.items():
                    if isinstance(paths, list):
                        relpath_map[leaf] = [str(p) for p in paths]
                    elif isinstance(paths, str):
                        relpath_map[leaf] = [paths]
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logging.warning("Could not parse folder relpath JSON: %s", exc)

    # Pre-load existing keys so we can avoid silent overwrite on collision.
    used_keys: set[str] = set()
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
            for obj in page.get("Contents", []):
                used_keys.add(obj["Key"])
    except Exception as exc:
        logging.warning("S3 list before upload failed for '%s': %s", safe, exc)

    # Phase 1 — serial key allocation: relpath_map is consumed in upload order
    # (so JS-supplied relpaths line up with their files) and `used_keys` is a
    # shared collision tracker. Both are racy under threads, so we resolve
    # final keys here before fanning out the actual PUTs.
    plan: list[tuple[pathlib.Path, str, str]] = []  # (source path, display name, full key)
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

        # If JS supplied a relpath for this leaf name, use it as the key suffix
        # so subdirectory structure survives the round-trip through S3.
        key_suffix = display_name
        bucket_paths = relpath_map.get(display_name)
        if bucket_paths:
            cleaned = _sanitize_s3_relpath(bucket_paths.pop(0))
            if cleaned:
                key_suffix = cleaned

        key = f"{prefix}/{key_suffix}"
        if key in used_keys:
            stem = pathlib.Path(key_suffix).stem
            ext = pathlib.Path(key_suffix).suffix
            parent = pathlib.Path(key_suffix).parent
            n = 2
            while True:
                cand_leaf = f"{stem} ({n}){ext}"
                cand_suffix = (
                    str(parent / cand_leaf) if str(parent) not in (".", "") else cand_leaf
                )
                cand_key = f"{prefix}/{cand_suffix}"
                if cand_key not in used_keys:
                    display_name = cand_leaf
                    key_suffix = cand_suffix
                    key = cand_key
                    break
                n += 1
        used_keys.add(key)
        plan.append((actual_path, display_name, key))

    # Phase 2 — parallel upload. boto3's botocore client is documented as
    # thread-safe for separate API calls, so 16 concurrent PUTs of small PDFs
    # easily saturate residential bandwidth without exhausting the connection
    # pool (default 10) plus the safety margin we configure at client init.
    def _upload_one(item: tuple[pathlib.Path, str, str]) -> tuple[bool, str, str, str]:
        src, name_for_log, target_key = item
        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                s3.upload_file(str(src), bucket, target_key)
                return True, name_for_log, target_key, ""
            except Exception as exc:
                last_err = exc
                if attempt < 2:
                    time.sleep(0.5 * (2 ** attempt))
        return False, name_for_log, target_key, str(last_err) if last_err else "unknown"

    saved = 0
    if plan:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(16, len(plan)), thread_name_prefix="s3_upload"
        ) as pool:
            for ok, name_for_log, target_key, err in pool.map(_upload_one, plan):
                if ok:
                    saved += 1
                else:
                    failed.append((name_for_log, err))
                    logging.error(
                        "S3 upload failed after retries for '%s' -> %s: %s",
                        name_for_log, target_key, err,
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
    over between runs on the same matter. Subdirectory structure is recreated
    locally so files uploaded under nested folders remain reachable.
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
    download_jobs: list[tuple[str, pathlib.Path]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            relpath = key[len(prefix):]
            if not relpath or relpath.endswith("/"):
                continue
            dest = temp_dir / relpath
            dest.parent.mkdir(parents=True, exist_ok=True)
            download_jobs.append((key, dest))

    def _download_one(job: tuple[str, pathlib.Path]) -> None:
        key, dest = job
        s3.download_file(bucket, key, str(dest))

    if download_jobs:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(16, len(download_jobs)), thread_name_prefix="s3_download"
        ) as pool:
            # Materialize results so any per-file exception surfaces as before.
            list(pool.map(_download_one, download_jobs))
    return temp_dir


def _list_s3_matter_files(matter_name: str) -> list[str]:
    """List document relative paths in an S3 matter, including nested folders."""
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
                relpath = obj["Key"][len(prefix):]
                if relpath and not relpath.endswith("/"):
                    files.append(relpath)
        return sorted(files)
    except Exception as exc:
        logger.warning("Failed to list S3 matter files for %r: %s", matter_name, exc)
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
    ("Source Calibration Advisory", "Auto-generated by SO-5 reliance gate"),
    ("Financial Analysis", "Auto-generated by SO-6 threshold gate"),
)

_DOMAIN_CREATE_PRESETS: dict[str, dict[str, Any]] = {
    "legal": {
        "label": "Legal Matter",
        "placeholder": "e.g. Smith v Jones 2024",
        "objective_templates": [
            "Analyze liability exposure",
            "Assess damages quantum",
            "Evaluate procedural posture",
        ],
        "document_labels": ["Pleading", "Contract", "Correspondence", "Court Order", "Discovery"],
        "taint_default": "public_clean",
    },
    "finance": {
        "label": "Financial Analysis",
        "placeholder": "e.g. ACME Corp Q4 2025 Thesis",
        "objective_templates": [
            "Evaluate investment thesis",
            "Assess risk exposure",
            "Analyze revenue sustainability",
            "Model valuation scenarios",
        ],
        "document_labels": ["10-K", "10-Q", "Earnings Transcript", "Analyst Report", "Proxy"],
        "taint_default": "public_clean",
    },
    "coding": {
        "label": "Software Analysis",
        "placeholder": "e.g. Auth Service Migration",
        "objective_templates": [
            "Analyze architecture fitness",
            "Assess migration risk",
            "Evaluate test coverage",
        ],
        "document_labels": ["Source Code", "Design Doc", "RFC", "Test Suite", "Config"],
        "taint_default": "public_clean",
    },
    "academic_research": {
        "label": "Research Analysis",
        "placeholder": "e.g. mRNA Delivery Mechanisms Review",
        "objective_templates": [
            "Synthesize state of the art",
            "Identify methodological gaps",
            "Evaluate replication status",
        ],
        "document_labels": ["Journal Article", "Preprint", "Dataset", "Protocol", "Review"],
        "taint_default": "public_clean",
    },
    "biomedical": {
        "label": "Clinical/Biomedical Analysis",
        "placeholder": "e.g. Drug X Phase III Safety Review",
        "objective_templates": [
            "Evaluate efficacy endpoints",
            "Assess safety signals",
            "Analyze regulatory pathway",
            "Review trial design",
        ],
        "document_labels": ["Clinical Trial", "FDA Document", "Label/PI", "Case Report", "Guideline"],
        "taint_default": "patient_deidentified",
    },
}


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
    except Exception as exc:
        logger.warning("Failed to format thinking steps: %s", exc)
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
    except Exception as exc:
        logger.warning("Failed to format route chip: %s", exc)
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


def _fmt_output_envelope_summary(envelope_dict: dict) -> str:
    """Format the OutputEnvelope's quality metadata for display in the citations panel."""
    if not envelope_dict or not isinstance(envelope_dict, dict):
        return ""
    parts: list[str] = []
    wk = str(envelope_dict.get("workflow_kind") or "")
    shape = str(envelope_dict.get("output_shape") or "")
    if wk or shape:
        parts.append(f"**Workflow**: {wk} → {shape}")
    blocking = list(envelope_dict.get("blocking_issues") or [])
    warns = list(envelope_dict.get("warnings") or [])
    review = bool(envelope_dict.get("review_required"))
    validations = list(envelope_dict.get("validation_results") or [])
    if not blocking and not warns and not review and not validations:
        parts.append("✅ All output quality checks passed.")
    else:
        if review:
            parts.append("⚠️ **Review required** — output flagged for human review.")
        for issue in blocking:
            parts.append(f"🚫 {issue}")
        for w in warns:
            parts.append(f"⚠️ {w}")
        for vr in validations:
            if not isinstance(vr, dict):
                continue
            name = str(vr.get("validator") or "check")
            passed = bool(vr.get("passed"))
            try:
                score = float(vr.get("score") or 0)
            except (TypeError, ValueError):
                score = 0.0
            icon = "✅" if passed else "❌"
            parts.append(f"{icon} {name}: {'passed' if passed else 'failed'} (score: {score:.2f})")
    dep_hash = envelope_dict.get("dependency_manifest_hash")
    if dep_hash and isinstance(dep_hash, str):
        parts.append(f"*Manifest*: `{dep_hash[:16]}…`")
    return "\n".join(parts)


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _error_html(data: Any) -> str | None:
    """If data is an error dict from the HTTP backend, return visible error HTML. Otherwise None."""
    if isinstance(data, dict) and "error" in data:
        return f"<div class='viz-empty'>Backend error: {_escape(data['error'])}</div>"
    return None


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


def _safe_list(value: Any) -> list:
    return value if isinstance(value, list) else []


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
        return "—"
    return f"{value:.0%}"


def _fmt_money(value: Any, decimals: int = 4) -> str:
    amount = _safe_float(value, 0.0)
    return f"${amount:,.{decimals}f}"


def _fmt_money_short(value: Any) -> str:
    amount = _safe_float(value, 0.0)
    return f"${amount:,.2f}"


def _metric_card(title: str, value: str, detail: str = "", tone: str = "default") -> str:
    detail_html = f"<div class='viz-card-detail'>{_escape(detail)}</div>" if detail else ""
    return (
        f"<div class='viz-card tone-{_escape(tone)}'>"
        f"<div class='viz-card-title'>{_escape(title)}</div>"
        f"<div class='viz-card-value'>{_escape(value)}</div>"
        f"{detail_html}"
        f"</div>"
    )


def _bar_row(label: str, value: float, maximum: float, meta: str = "", tone: str = "blue") -> str:
    pct = 0.0 if maximum <= 0 else min(100.0, max(4.0, (value / maximum) * 100.0))
    meta_html = f"<div class='viz-bar-meta'>{_escape(meta)}</div>" if meta else ""
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


_OVERVIEW_PANEL_LABELS = {
    "legal": {
        "assertions": "Assertions",
        "open_issues": "Open Issues",
        "actors": "Actors",
        "quant_facts": "Quant facts",
        "version_chains": "Version chains",
        "issue_coverage": "Issue Coverage",
        "calibration": "Calibration",
        "coverage_dist": "Coverage distribution",
        "weakest": "Weakest issues",
        "issue_fallback": "Issue",
        "gaps": "Gaps",
        "clarifications": "Clarifications",
        "open_work": "Open work",
    },
    "finance": {
        "assertions": "Claims",
        "open_issues": "Open Theses",
        "actors": "Entities",
        "quant_facts": "Quant facts",
        "version_chains": "Filing versions",
        "issue_coverage": "Thesis Coverage",
        "calibration": "Calibration",
        "coverage_dist": "Coverage distribution",
        "weakest": "Weakest theses",
        "issue_fallback": "Thesis",
        "gaps": "Diligence gaps",
        "clarifications": "Clarifications",
        "open_work": "Open work",
    },
    "coding": {
        "assertions": "Findings",
        "open_issues": "Open Hypotheses",
        "actors": "Components",
        "quant_facts": "Metrics",
        "version_chains": "Artifact versions",
        "issue_coverage": "Hypothesis Coverage",
        "calibration": "Calibration",
        "coverage_dist": "Coverage distribution",
        "weakest": "Weakest hypotheses",
        "issue_fallback": "Hypothesis",
        "gaps": "Verification gaps",
        "clarifications": "Clarifications",
        "open_work": "Open work",
    },
    "academic_research": {
        "assertions": "Claims",
        "open_issues": "Open Questions",
        "actors": "Authors",
        "quant_facts": "Quant facts",
        "version_chains": "Source versions",
        "issue_coverage": "Claim Coverage",
        "calibration": "Calibration",
        "coverage_dist": "Coverage distribution",
        "weakest": "Weakest claims",
        "issue_fallback": "Claim",
        "gaps": "Evidence gaps",
        "clarifications": "Clarifications",
        "open_work": "Open work",
    },
    "biomedical": {
        "assertions": "Findings",
        "open_issues": "Open Questions",
        "actors": "Entities",
        "quant_facts": "Quant facts",
        "version_chains": "Record versions",
        "issue_coverage": "Finding Coverage",
        "calibration": "Calibration",
        "coverage_dist": "Coverage distribution",
        "weakest": "Weakest findings",
        "issue_fallback": "Finding",
        "gaps": "Evidence gaps",
        "clarifications": "Clarifications",
        "open_work": "Open work",
    },
}


def _fmt_overview_panel(data: dict, domain: str = "legal") -> str:
    if err := _error_html(data):
        return err
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

    labels = _OVERVIEW_PANEL_LABELS.get(domain, _OVERVIEW_PANEL_LABELS["legal"])
    stats = data.get("stats", {})
    so = data.get("so_metrics", {})
    llm = stats.get("llm", {}) if isinstance(stats.get("llm"), dict) else {}
    llm_totals = llm.get("totals", {}) if isinstance(llm, dict) else {}
    coverage_report = data.get("coverage_report", []) or []
    weakest = data.get("weakest_issues", []) or []
    gaps = data.get("top_gaps", []) or []
    clarifications = data.get("pending_clarifications", []) or []

    cards = [
        _metric_card(labels["assertions"], f"{_safe_int(stats.get('assertion_count', 0)):,}"),
        _metric_card(
            labels["open_issues"],
            f"{_safe_int(stats.get('open_issue_count', 0)):,}",
            detail=(
                f"{labels['gaps']}: {_safe_int(stats.get('open_gap_count', 0)):,}"
                + (f" | Conflicts: {_safe_int(data.get('contradiction_count', 0)):,}"
                   if data.get("contradiction_count") else "")
            ),
        ),
        _metric_card(
            labels["actors"],
            f"{_safe_int(stats.get('actor_count', 0)):,}",
            detail=(
                f"{labels['quant_facts']}: {_safe_int(stats.get('quant_fact_count', 0)):,}"
                + (f" | {labels['version_chains']}: {_safe_int(data.get('version_chain_count', 0)):,}"
                   if data.get("version_chain_count") else "")
            ),
        ),
        _metric_card(
            "LLM Cost",
            _fmt_money(llm_totals.get("estimated_cost_usd", 0.0)),
            detail=f"{_safe_int(llm_totals.get('request_count', 0)):,} calls",
            tone="amber",
        ),
        _metric_card(
            labels["issue_coverage"],
            _fmt_percent_html(so.get("issue_coverage_avg")),
            detail=f"Reuse: {_fmt_percent_html(so.get('reuse_rate'))}",
            tone="green",
        ),
        _metric_card(
            labels["calibration"],
            _fmt_percent_html(so.get("source_role_known_rate")),
            detail=f"Structured: {_fmt_percent_html(so.get('assertion_structure_rate'))}",
        ),
    ]

    coverage_rows: list[str] = []
    coverage_sorted = sorted(
        (item for item in coverage_report if isinstance(item, dict)),
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

    gap_labels = _GAP_LABELS.get(primary_domain, _GAP_LABELS["legal"])

    def _gap_li(g: dict) -> str:
        gt = str(g.get("gap_type") or "")
        pill = gap_labels.get(gt)
        if isinstance(pill, tuple):
            label, cls = pill
        else:
            label, cls = gt.replace("_", " ").title() or "Gap", "pill-neutral"
        desc = _escape(str(g.get("description") or label))
        mat = g.get("materiality_score") or g.get("materiality")
        mat_tag = (
            f" <span style='font-size:10px;color:#6b7280'>({float(mat):.1f})</span>"
            if isinstance(mat, (int, float)) and mat else ""
        )
        return f"<li><span class='pill {cls}' style='font-size:9px;padding:1px 5px'>{_escape(label)}</span> {desc}{mat_tag}</li>"
    gap_items = "".join(_gap_li(g) for g in gaps if isinstance(g, dict)) or "<li>No open gaps.</li>"
    clarification_items = "".join(
        "<li>"
        f"{_escape(c.get('question_text') or c.get('question') or 'Clarification')}"
        "</li>"
        for c in clarifications if isinstance(c, dict)
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
        + f"<div class='viz-panel-title'>{_escape(labels['coverage_dist'])}</div>"
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
        + f"<div class='viz-panel-title'>{_escape(labels['weakest'])}</div>"
        + (
            "".join(
                "<div class='viz-list-row'>"
                f"<span>{_escape(issue.get('title') or issue.get('id') or labels['issue_fallback'])}</span>"
                f"<strong>{_fmt_percent_html(_safe_float(issue.get('coverage_fraction', 0.0)))}</strong>"
                "</div>"
                for issue in weakest if isinstance(issue, dict)
            )
            if weakest
            else "<div class='viz-empty'>No issue coverage data yet.</div>"
        )
        + "</div>"
        + "<div class='viz-panel'>"
        + f"<div class='viz-panel-title'>{_escape(labels['open_work'])}</div>"
        + "<div class='viz-list-columns'>"
        + f"<div><div class='viz-subtitle'>{_escape(labels['gaps'])}</div><ul>"
        + gap_items
        + "</ul></div>"
        + f"<div><div class='viz-subtitle'>{_escape(labels['clarifications'])}</div><ul>"
        + clarification_items
        + "</ul></div>"
        + "</div>"
        + "</div>"
        + "</div>"
        + "</div>"
        + "</div>"
    )


_TRUST_NOTICE_LABELS = {
    "legal": {
        "issues_word": "issue(s)",
        "reviewer": "attorney",
        "proof_gap": "proof gap",
        "candidate_only": "candidate-only",
        "unsupported": "unsupported",
        "all_verified": "All {total} {issues_word} have verified support. The memo makes definitive claims only where the {reviewer} has signed off. No hedging needed.",
        "hedging": "Hedging on {hedged} of {total} {issues_word}. The memo frames findings as provisional or unresolved on: {detail}. Verify supporting facts in the Review Inbox to promote these to definitive claims.",
    },
    "finance": {
        "issues_word": "thesis(es)",
        "reviewer": "analyst",
        "proof_gap": "evidence gap",
        "candidate_only": "unconfirmed",
        "unsupported": "unsourced",
        "all_verified": "All {total} {issues_word} have confirmed support. The report makes definitive claims only where the {reviewer} has signed off. No hedging needed.",
        "hedging": "Hedging on {hedged} of {total} {issues_word}. The report frames findings as provisional or unresolved on: {detail}. Verify supporting facts in the Review Inbox to promote these to definitive claims.",
    },
    "coding": {
        "issues_word": "hypothesis(es)",
        "reviewer": "engineer",
        "proof_gap": "verification gap",
        "candidate_only": "unverified",
        "unsupported": "no evidence",
        "all_verified": "All {total} {issues_word} have confirmed support. The analysis makes definitive claims only where the {reviewer} has signed off. No hedging needed.",
        "hedging": "Hedging on {hedged} of {total} {issues_word}. The analysis frames findings as provisional or unresolved on: {detail}. Verify supporting facts in the Review Inbox to promote these to definitive claims.",
    },
    "academic_research": {
        "issues_word": "claim(s)",
        "reviewer": "reviewer",
        "proof_gap": "evidence gap",
        "candidate_only": "unverified",
        "unsupported": "uncited",
        "all_verified": "All {total} {issues_word} have verified support. The review makes definitive claims only where the {reviewer} has signed off. No hedging needed.",
        "hedging": "Hedging on {hedged} of {total} {issues_word}. The review frames findings as provisional or unresolved on: {detail}. Verify supporting facts in the Review Inbox to promote these to definitive claims.",
    },
    "biomedical": {
        "issues_word": "finding(s)",
        "reviewer": "clinician",
        "proof_gap": "evidence gap",
        "candidate_only": "unverified",
        "unsupported": "unsourced",
        "all_verified": "All {total} {issues_word} have verified support. The summary makes definitive claims only where the {reviewer} has signed off. No hedging needed.",
        "hedging": "Hedging on {hedged} of {total} {issues_word}. The summary frames findings as provisional or unresolved on: {detail}. Verify supporting facts in the Review Inbox to promote these to definitive claims.",
    },
}


def _fmt_trust_notice(issues: list, domain: str = "legal") -> str:
    if not issues:
        return ""
    labels = _TRUST_NOTICE_LABELS.get(domain, _TRUST_NOTICE_LABELS["legal"])
    total = len(issues)
    fully_verified = 0
    candidate_only = 0
    gap_blocked = 0
    no_support = 0
    issue_names: list[str] = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        v = _safe_int(issue.get("verified_supporting_count", 0))
        c = _safe_int(issue.get("candidate_supporting_count", 0))
        has_gap = bool(issue.get("has_proof_gap"))
        title = issue.get("title") or "Untitled"
        if v > 0 and not has_gap:
            fully_verified += 1
            continue
        if has_gap:
            gap_blocked += 1
            issue_names.append(f"{title} ({labels['proof_gap']})")
        elif v == 0 and c > 0:
            candidate_only += 1
            issue_names.append(f"{title} ({labels['candidate_only']})")
        else:
            no_support += 1
            issue_names.append(f"{title} ({labels['unsupported']})")
    hedged = candidate_only + gap_blocked + no_support
    if hedged == 0:
        msg = labels["all_verified"].format(
            total=total, issues_word=labels["issues_word"], reviewer=labels["reviewer"],
        )
        return (
            "<div style='padding:10px 14px;border-radius:8px;"
            "background:#dcfce7;color:#14532d;border-left:4px solid #15803d;"
            "margin-bottom:12px;font-size:13px;'>"
            f"<strong>{_escape(msg)}</strong>"
            "</div>"
        )
    shown = issue_names[:4]
    remainder = len(issue_names) - len(shown)
    detail = "; ".join(_escape(n) for n in shown)
    if remainder > 0:
        detail += f"; and {remainder} more"
    msg = labels["hedging"].format(
        hedged=hedged, total=total, issues_word=labels["issues_word"], detail=detail,
    )
    return (
        "<div style='padding:10px 14px;border-radius:8px;"
        "background:#fef3c7;color:#78350f;border-left:4px solid #b45309;"
        "margin-bottom:12px;font-size:13px;'>"
        f"<strong>⚠ {_escape(str(hedged))} of {_escape(str(total))} {_escape(labels['issues_word'])}</strong> "
        f"— {msg}"
        "</div>"
    )


_ISSUES_PANEL_LABELS = {
    "legal": {
        "empty": "No open issues.",
        "issue_fallback": "Issue",
        "verified": "verified",
        "candidate": "candidate",
        "attack": "attack",
        "disputed": "disputed",
        "blocked": "blocked",
        "proof_gap": "proof gap",
        "verified_cov": "Verified coverage",
        "advisory_cov": "Advisory (candidate + verified)",
    },
    "finance": {
        "empty": "No open theses.",
        "issue_fallback": "Thesis",
        "verified": "confirmed",
        "candidate": "candidate",
        "attack": "contradiction",
        "disputed": "disputed",
        "blocked": "blocked",
        "proof_gap": "evidence gap",
        "verified_cov": "Confirmed coverage",
        "advisory_cov": "Advisory (candidate + confirmed)",
    },
    "coding": {
        "empty": "No open hypotheses.",
        "issue_fallback": "Hypothesis",
        "verified": "confirmed",
        "candidate": "candidate",
        "attack": "refutation",
        "disputed": "disputed",
        "blocked": "blocked",
        "proof_gap": "verification gap",
        "verified_cov": "Confirmed coverage",
        "advisory_cov": "Advisory (candidate + confirmed)",
    },
    "academic_research": {
        "empty": "No open claims.",
        "issue_fallback": "Claim",
        "verified": "verified",
        "candidate": "candidate",
        "attack": "challenge",
        "disputed": "disputed",
        "blocked": "blocked",
        "proof_gap": "evidence gap",
        "verified_cov": "Verified coverage",
        "advisory_cov": "Advisory (candidate + verified)",
    },
    "biomedical": {
        "empty": "No open findings.",
        "issue_fallback": "Finding",
        "verified": "verified",
        "candidate": "candidate",
        "attack": "contradiction",
        "disputed": "disputed",
        "blocked": "blocked",
        "proof_gap": "evidence gap",
        "verified_cov": "Verified coverage",
        "advisory_cov": "Advisory (candidate + verified)",
    },
}


def _fmt_issues_panel(issues: list, domain: str = "legal") -> str:
    labels = _ISSUES_PANEL_LABELS.get(domain, _ISSUES_PANEL_LABELS["legal"])
    if not issues:
        return f"<div class='viz-empty'>{_escape(labels['empty'])}</div>"

    rows: list[str] = []
    for issue in sorted(
        (item for item in issues if isinstance(item, dict)),
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
        issue_id = issue.get("id") or ""
        title = _escape(issue.get("title") or issue_id or labels["issue_fallback"])
        short_issue_id = _escape(str(issue_id)[:12])
        _KNOWN_PROOFS = {"none", "partial", "sufficient", "proved", "contested", "blocked", "unknown"}
        _raw_proof = (issue.get("proof_status") or "none").lower()
        proof = _raw_proof if _raw_proof in _KNOWN_PROOFS else "none"
        attack = _safe_int(issue.get("attacking_count", 0))
        contested = _safe_int(issue.get("contested_predicates", 0))
        blocked = _safe_int(issue.get("blocked_predicates", 0))
        details = [
            f"{verified_cnt} {labels['verified']}",
            f"{candidate_cnt} {labels['candidate']}",
            f"{attack} {labels['attack']}" if attack else "",
        ]
        if contested:
            details.append(f"{contested} {labels['disputed']}")
        if blocked:
            details.append(f"{blocked} {labels['blocked']}")
        details_str = " | ".join(d for d in details if d)
        gap_marker = ""
        if issue.get("has_proof_gap"):
            gap_marker = (
                "<span style='margin-left:8px;padding:1px 6px;border-radius:8px;"
                "background:#fee2e2;color:#991b1b;font-size:10px;font-weight:700;"
                f"text-transform:uppercase;letter-spacing:0.05em;'>{_escape(labels['proof_gap'])}</span>"
            )
        # Two overlaid bars: thin dark verified bar inside a wider
        # light candidate-advisory bar. Legend below the track.
        verified_pct = max(0.0, verified_cov * 100)
        advisory_pct = max(0.0, coverage * 100)
        rows.append(
            f"<div class='issue-row' style='--issue-indent:{depth * 18}px'>"
            f"<div class='issue-head'>"
            f"<span class='proof-pill proof-{proof}'>{proof}</span>"
            f"<span class='issue-title'>{title}</span>"
            f"<span style='font-size:9px;color:#9ca3af;font-family:monospace;margin-left:6px;' title='{_escape(str(issue_id))}'>{short_issue_id}</span>"
            f"{gap_marker}"
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
        f" {_escape(labels['verified_cov'])}</span>"
        "<span><span style='display:inline-block;width:10px;height:8px;"
        "background:#bfdbfe;border-radius:2px;vertical-align:middle;'></span>"
        f" {_escape(labels['advisory_cov'])}</span>"
        "</div>"
    )
    notice = _fmt_trust_notice(issues, domain=domain)
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


_TIMELINE_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "empty": "No timeline events available.",
        "withheld_title": "Withheld under clean policy",
        "withheld_notice": (
            "under clean policy — their dates are preserved in this "
            "timeline but the event text and source are hidden. "
            "Privileged docs are never shown in clean mode."
        ),
    },
    "finance": {
        "empty": "No timeline events available.",
        "withheld_title": "Withheld under compliance policy",
        "withheld_notice": (
            "under compliance policy — their dates are preserved "
            "but the event text and source are hidden. Restricted "
            "sources are never shown in clean mode."
        ),
    },
    "coding": {
        "empty": "No timeline events available.",
        "withheld_title": "Withheld under content policy",
        "withheld_notice": (
            "under content policy — their dates are preserved "
            "but the event text and source are hidden. Restricted "
            "artifacts are never shown in clean mode."
        ),
    },
    "academic_research": {
        "empty": "No timeline events available.",
        "withheld_title": "Withheld under review policy",
        "withheld_notice": (
            "under review policy — their dates are preserved "
            "but the event text and source are hidden. Embargoed "
            "sources are never shown in clean mode."
        ),
    },
    "biomedical": {
        "empty": "No timeline events available.",
        "withheld_title": "Withheld under compliance policy",
        "withheld_notice": (
            "under compliance policy — their dates are preserved "
            "but the event text and source are hidden. Protected "
            "sources are never shown in clean mode."
        ),
    },
}


def _fmt_timeline_panel(events: list[dict], domain: str = "legal") -> str:
    L = _TIMELINE_LABELS.get(domain, _TIMELINE_LABELS["legal"])
    if not events:
        return f"<div class='viz-empty'>{L['empty']}</div>"
    withheld_total = sum(1 for e in events if isinstance(e, dict) and e.get("withheld"))
    items: list[str] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        raw_date = event.get("date") or ""
        precision = event.get("date_precision")
        date = _escape(_display_date(raw_date, precision) if raw_date else "Undated")
        is_withheld = bool(event.get("withheld"))
        if is_withheld:
            title = f"<em style='color:#b45309;'>{L['withheld_title']}</em>"
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
            f"<strong>{withheld_total} event(s) withheld</strong> "
            f"{L['withheld_notice']}"
            "</div>"
        )
    return (
        "<div class='viz-shell'>"
        + header
        + "<div class='timeline-list'>"
        + "".join(items)
        + "</div></div>"
    )


_EVIDENCE_MATRIX_LABELS = {
    "legal": {
        "title": "Support and attack by issue/source",
        "legend": "Green = support, red = attack, split cell = both.",
        "issue": "Issue",
        "source": "Source",
        "support": "Support",
        "attack": "Attack",
        "total": "Total",
        "issue_totals": "Issue totals",
        "source_totals": "Source totals",
        "detail": "Issue/source detail",
        "no_issue_totals": "No issue totals available.",
        "no_source_totals": "No source totals available.",
        "no_cells": "No linked evidence cells yet.",
        "empty": "Evidence matrix will populate after issues are linked to sources.",
    },
    "finance": {
        "title": "Corroboration and contradiction by thesis/source",
        "legend": "Green = corroboration, red = contradiction, split cell = both.",
        "issue": "Thesis",
        "source": "Source",
        "support": "Corroboration",
        "attack": "Contradiction",
        "total": "Total",
        "issue_totals": "Thesis totals",
        "source_totals": "Source totals",
        "detail": "Thesis/source detail",
        "no_issue_totals": "No thesis totals available.",
        "no_source_totals": "No source totals available.",
        "no_cells": "No linked evidence cells yet.",
        "empty": "Evidence matrix will populate after theses are linked to sources.",
    },
    "coding": {
        "title": "Confirmation and refutation by hypothesis/artifact",
        "legend": "Green = confirmation, red = refutation, split cell = both.",
        "issue": "Hypothesis",
        "source": "Artifact",
        "support": "Confirmation",
        "attack": "Refutation",
        "total": "Total",
        "issue_totals": "Hypothesis totals",
        "source_totals": "Artifact totals",
        "detail": "Hypothesis/artifact detail",
        "no_issue_totals": "No hypothesis totals available.",
        "no_source_totals": "No artifact totals available.",
        "no_cells": "No linked evidence cells yet.",
        "empty": "Evidence matrix will populate after hypotheses are linked to artifacts.",
    },
    "academic_research": {
        "title": "Support and challenge by claim/source",
        "legend": "Green = support, red = challenge, split cell = both.",
        "issue": "Claim",
        "source": "Source",
        "support": "Support",
        "attack": "Challenge",
        "total": "Total",
        "issue_totals": "Claim totals",
        "source_totals": "Source totals",
        "detail": "Claim/source detail",
        "no_issue_totals": "No claim totals available.",
        "no_source_totals": "No source totals available.",
        "no_cells": "No linked evidence cells yet.",
        "empty": "Evidence matrix will populate after claims are linked to sources.",
    },
    "biomedical": {
        "title": "Support and contradiction by finding/source",
        "legend": "Green = support, red = contradiction, split cell = both.",
        "issue": "Finding",
        "source": "Source",
        "support": "Support",
        "attack": "Contradiction",
        "total": "Total",
        "issue_totals": "Finding totals",
        "source_totals": "Source totals",
        "detail": "Finding/source detail",
        "no_issue_totals": "No finding totals available.",
        "no_source_totals": "No source totals available.",
        "no_cells": "No linked evidence cells yet.",
        "empty": "Evidence matrix will populate after findings are linked to sources.",
    },
}


def _fmt_evidence_matrix_panel(matrix: dict, domain: str = "legal") -> str:
    labels = _EVIDENCE_MATRIX_LABELS.get(domain, _EVIDENCE_MATRIX_LABELS["legal"])
    if not matrix or not matrix.get("issues") or not matrix.get("sources"):
        return f"<div class='viz-empty'>{_escape(labels['empty'])}</div>"

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
        f"<div class='viz-panel-title'>{_escape(labels['title'])}</div>"
        f"<div class='viz-footnote'>{_escape(labels['legend'])}</div>"
        f"<div class='matrix-wrap matrix-wrap-heatmap'><table class='matrix-table evidence-matrix-table'><thead><tr><th>{_escape(labels['issue'])}</th>"
        + header
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
        + "<div class='viz-two-col'>"
        + f"<div class='viz-panel'><div class='viz-subtitle'>{_escape(labels['issue_totals'])}</div>"
        + "<div class='matrix-wrap'><table class='analytics-table'><thead><tr>"
        + f"<th>{_escape(labels['issue'])}</th><th>{_escape(labels['support'])}</th><th>{_escape(labels['attack'])}</th><th>{_escape(labels['total'])}</th>"
        + "</tr></thead><tbody>"
        + (issue_totals_rows or f"<tr><td colspan='4'>{_escape(labels['no_issue_totals'])}</td></tr>")
        + "</tbody></table></div></div>"
        + f"<div class='viz-panel'><div class='viz-subtitle'>{_escape(labels['source_totals'])}</div>"
        + "<div class='matrix-wrap'><table class='analytics-table'><thead><tr>"
        + f"<th>{_escape(labels['source'])}</th><th>{_escape(labels['support'])}</th><th>{_escape(labels['attack'])}</th><th>{_escape(labels['total'])}</th>"
        + "</tr></thead><tbody>"
        + (source_totals_rows or f"<tr><td colspan='4'>{_escape(labels['no_source_totals'])}</td></tr>")
        + "</tbody></table></div></div>"
        + "</div>"
        + f"<div class='viz-panel'><div class='viz-subtitle'>{_escape(labels['detail'])}</div>"
        + "<div class='matrix-wrap'><table class='analytics-table'><thead><tr>"
        + f"<th>{_escape(labels['issue'])}</th><th>{_escape(labels['source'])}</th><th>{_escape(labels['support'])}</th><th>{_escape(labels['attack'])}</th><th>{_escape(labels['total'])}</th>"
        + "</tr></thead><tbody>"
        + (("".join(detail_rows)) or f"<tr><td colspan='5'>{_escape(labels['no_cells'])}</td></tr>")
        + "</tbody></table></div></div></div>"
    )


# Source Calibration Inspector (SO-4, SO-5, SO-7)

_SOURCE_CALIBRATION_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Source Calibration Inspector",
        "subtitle": "are conclusions grounded in the right source types?",
        "issue_col": "Issue",
        "support_col": "Support",
        "attack_col": "Attack",
        "diversity_col": "Source Diversity",
        "warnings_col": "Warnings",
        "doc_col": "Document",
        "role_col": "Source Role",
        "gap_col": "Missing Source Type",
        "materiality_col": "Materiality",
        "empty": "No source calibration data — run an investigation first.",
        "single_source": "single-source",
        "advocacy_only": "advocacy-only",
        "no_attack_tested": "no attack tested",
        "unknown_role": "unknown source role",
    },
    "finance": {
        "title": "Source Type Assessment",
        "subtitle": "are conclusions grounded in the right data sources?",
        "issue_col": "Position",
        "support_col": "Support",
        "attack_col": "Challenge",
        "diversity_col": "Data Diversity",
        "warnings_col": "Warnings",
        "doc_col": "Filing",
        "role_col": "Source Type",
        "gap_col": "Missing Data Type",
        "materiality_col": "Materiality",
        "empty": "No source assessment data — run an analysis first.",
        "single_source": "single-source",
        "advocacy_only": "management-only",
        "no_attack_tested": "no independent review",
        "unknown_role": "unknown source type",
    },
    "coding": {
        "title": "Evidence Source Assessment",
        "subtitle": "are conclusions grounded in the right artifact types?",
        "issue_col": "Objective",
        "support_col": "Support",
        "attack_col": "Challenge",
        "diversity_col": "Artifact Diversity",
        "warnings_col": "Warnings",
        "doc_col": "Artifact",
        "role_col": "Artifact Role",
        "gap_col": "Missing Artifact Type",
        "materiality_col": "Materiality",
        "empty": "No source assessment data — run an analysis first.",
        "single_source": "single-source",
        "advocacy_only": "author-only",
        "no_attack_tested": "no test evidence",
        "unknown_role": "unknown artifact role",
    },
    "academic_research": {
        "title": "Citation Source Assessment",
        "subtitle": "are conclusions grounded in the right source classes?",
        "issue_col": "Research Question",
        "support_col": "Support",
        "attack_col": "Challenge",
        "diversity_col": "Citation Diversity",
        "warnings_col": "Warnings",
        "doc_col": "Publication",
        "role_col": "Source Class",
        "gap_col": "Missing Source Class",
        "materiality_col": "Materiality",
        "empty": "No citation assessment data — run an analysis first.",
        "single_source": "single-source",
        "advocacy_only": "preprint-only",
        "no_attack_tested": "no replication",
        "unknown_role": "unknown source class",
    },
    "biomedical": {
        "title": "Clinical Evidence Assessment",
        "subtitle": "are conclusions grounded in the right evidence classes?",
        "issue_col": "Clinical Question",
        "support_col": "Support",
        "attack_col": "Challenge",
        "diversity_col": "Evidence Diversity",
        "warnings_col": "Warnings",
        "doc_col": "Record",
        "role_col": "Evidence Class",
        "gap_col": "Missing Evidence Class",
        "materiality_col": "Materiality",
        "empty": "No evidence assessment data — run an analysis first.",
        "single_source": "single-source",
        "advocacy_only": "sponsor-only",
        "no_attack_tested": "no independent trial",
        "unknown_role": "unknown evidence class",
    },
}


def _fmt_source_calibration(data: dict, domain: str = "legal") -> str:
    L = _SOURCE_CALIBRATION_LABELS.get(domain, _SOURCE_CALIBRATION_LABELS["legal"])
    if not data or not isinstance(data, dict):
        return f"<div class='viz-empty'>{_escape(L['empty'])}</div>"
    if err := _error_html(data):
        return err

    matrix = data.get("evidence_matrix", {})
    if not isinstance(matrix, dict):
        matrix = {}
    coverage = _safe_list(data.get("coverage_report"))
    profile = data.get("domain_profile", {})
    if not isinstance(profile, dict):
        profile = {}
    gap_wb = data.get("gap_workbench", {})
    if not isinstance(gap_wb, dict):
        gap_wb = {}
    docs = _safe_list(data.get("reviewable_documents"))

    source_roles = _safe_list(profile.get("source_roles"))
    trust_weights = profile.get("composed_trust_weights", {})
    if not isinstance(trust_weights, dict):
        trust_weights = {}
    profile_id = profile.get("profile_id", "unknown")

    issue_totals = matrix.get("issue_totals", {})
    if not isinstance(issue_totals, dict):
        issue_totals = {}
    source_totals = matrix.get("source_totals", {})
    if not isinstance(source_totals, dict):
        source_totals = {}
    cells = matrix.get("cells", {})
    if not isinstance(cells, dict):
        cells = {}
    issues = _safe_list(matrix.get("issues"))
    sources = _safe_list(matrix.get("sources"))

    doc_type_map: dict[str, str] = {}
    for d in docs:
        if isinstance(d, dict):
            path = str(d.get("path", ""))
            dtype = str(d.get("doc_type", "unknown"))
            if path:
                doc_type_map[path] = dtype

    parts: list[str] = []
    parts.append("<div style='margin-bottom:16px;'>")
    parts.append(
        f"<h3 style='margin:0 0 4px;'>{_escape(L['title'])}</h3>"
        f"<div style='font-size:12px;color:#6b7280;margin-bottom:12px;'>{_escape(L['subtitle'])}</div>"
    )

    # Calibration Summary
    known_roles = len([r for r in source_roles if not isinstance(r, dict)])
    total_sources = len(sources)
    unique_doc_types = len(set(doc_type_map.values())) if doc_type_map else 0
    parts.append(
        "<div style='margin-bottom:12px;padding:10px;background:#f0fdf4;border:1px solid #bbf7d0;border-radius:6px;'>"
        "<div style='font-weight:700;font-size:13px;margin-bottom:6px;'>Calibration Summary</div>"
        f"<div style='font-size:12px;'>Profile: <strong>{_escape(str(profile_id))}</strong> "
        f"| Known source roles: <strong>{known_roles}</strong> "
        f"| Source documents: <strong>{total_sources}</strong> "
        f"| Distinct source types: <strong>{unique_doc_types}</strong> "
        f"| Issues covered: <strong>{len(issues)}</strong></div>"
    )
    if trust_weights:
        parts.append("<div style='font-size:11px;margin-top:4px;color:#6b7280;'>Trust weights: ")
        tw_parts = []
        for k, v in list(trust_weights.items())[:6]:
            val = float(v) if isinstance(v, (int, float)) and math.isfinite(float(v)) else 0.0
            tw_parts.append(f"{_escape(str(k)[:20])}={val:.2f}")
        parts.append(", ".join(tw_parts))
        if len(trust_weights) > 6:
            parts.append(f" +{len(trust_weights)-6} more")
        parts.append("</div>")
    parts.append("</div>")

    # Issue Source Mix table
    if issues:
        parts.append(
            "<div style='margin-bottom:12px;'>"
            "<div style='font-weight:700;font-size:13px;margin-bottom:6px;'>Issue Source Mix</div>"
            "<div class='table-wrap'><table class='viz-table'>"
            f"<thead><tr><th>{_escape(L['issue_col'])}</th>"
            f"<th>{_escape(L['support_col'])}</th>"
            f"<th>{_escape(L['attack_col'])}</th>"
            f"<th>{_escape(L['diversity_col'])}</th>"
            f"<th>{_escape(L['warnings_col'])}</th></tr></thead><tbody>"
        )
        for iss in issues[:30]:
            if not isinstance(iss, dict):
                continue
            iid = iss.get("id", "")
            title = _escape(str(iss.get("title", iid))[:50])
            totals = issue_totals.get(iid, {})
            if not isinstance(totals, dict):
                totals = {}
            sup = int(totals.get("supporting", 0)) if isinstance(totals.get("supporting"), (int, float)) else 0
            atk = int(totals.get("attacking", 0)) if isinstance(totals.get("attacking"), (int, float)) else 0

            issue_cells = cells.get(iid, {})
            if not isinstance(issue_cells, dict):
                issue_cells = {}
            unique_sources = len(issue_cells)
            issue_doc_types = {doc_type_map.get(src, "unknown") for src in issue_cells}
            type_diversity = len(issue_doc_types)

            warnings = []
            if unique_sources <= 1 and sup > 0:
                warnings.append(L["single_source"])
            if atk == 0 and sup > 0:
                warnings.append(L["no_attack_tested"])
            if type_diversity <= 1 and unique_sources > 1:
                warnings.append("single type")

            warn_html = ""
            if warnings:
                warn_html = " ".join(
                    f"<span style='display:inline-block;padding:1px 6px;background:#fef3c7;border:1px solid #fde68a;"
                    f"border-radius:3px;font-size:10px;color:#92400e;margin:1px;'>{_escape(w)}</span>"
                    for w in warnings
                )

            parts.append(
                f"<tr><td title='{_escape(str(iid)[:40])}'>{title}</td>"
                f"<td style='text-align:center;'>{sup}</td>"
                f"<td style='text-align:center;'>{atk}</td>"
                f"<td style='text-align:center;'>{unique_sources} ({type_diversity} types)</td>"
                f"<td>{warn_html}</td></tr>"
            )
        if len(issues) > 30:
            parts.append(f"<tr><td colspan='5' style='color:#9ca3af;text-align:center;'>+{len(issues)-30} more issues</td></tr>")
        parts.append("</tbody></table></div></div>")

    # Missing Source Obligations from gap workbench
    gaps = _safe_list(gap_wb.get("items"))
    source_gaps = [g for g in gaps if isinstance(g, dict) and str(g.get("gap_type", "")).startswith("MISSING_")]
    if source_gaps:
        parts.append(
            "<div style='margin-bottom:12px;'>"
            "<div style='font-weight:700;font-size:13px;margin-bottom:6px;'>Missing Source Obligations</div>"
            "<div class='table-wrap'><table class='viz-table'>"
            f"<thead><tr><th>{_escape(L['gap_col'])}</th>"
            f"<th>{_escape(L['issue_col'])}</th>"
            f"<th>{_escape(L['materiality_col'])}</th></tr></thead><tbody>"
        )
        for g in source_gaps[:20]:
            gap_type = _escape(str(g.get("gap_type", ""))[:40].replace("MISSING_", "").replace("_", " ").title())
            desc = _escape(str(g.get("description", ""))[:60])
            mat = _safe_float(g.get("materiality_score") or g.get("materiality") or 0)
            mat_color = "#dc2626" if mat >= 0.7 else "#f59e0b" if mat >= 0.4 else "#6b7280"
            issue_title = _escape(str(g.get("issue_title", g.get("affected_issue_id", "")))[:40])
            parts.append(
                f"<tr><td title='{desc}'>{gap_type}</td>"
                f"<td>{issue_title}</td>"
                f"<td style='color:{mat_color};font-weight:600;'>{mat:.2f}</td></tr>"
            )
        if len(source_gaps) > 20:
            parts.append(f"<tr><td colspan='3' style='color:#9ca3af;text-align:center;'>+{len(source_gaps)-20} more</td></tr>")
        parts.append("</tbody></table></div></div>")

    parts.append("</div>")
    return "".join(parts)


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

        short_iid = _escape(str(issue_id)[:12])
        rows_html += (
            f"<tr>"
            f"<td style='max-width:220px;overflow:hidden;text-overflow:ellipsis;"
            f"white-space:nowrap;' title='{_escape(issue_title)}'>{display_title}{adv_badge}"
            f"<span style='font-size:9px;color:#9ca3af;font-family:monospace;margin-left:6px;'"
            f" title='{_escape(str(issue_id))}'>{short_iid}</span></td>"
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


_AUTHORITY_PANEL_LABELS = {
    "legal": {
        "title": "Authorities & Precedent",
        "item": "Authority",
        "type_label": "Type",
        "weight_label": "Weight",
        "jurisdiction_label": "Jurisdiction",
        "linked_issues": "Linked Issues",
        "empty": "No authorities cited yet. Run an investigation to discover relevant case law, statutes, and rules.",
    },
    "finance": {
        "title": "Regulatory & Standards References",
        "item": "Reference",
        "type_label": "Type",
        "weight_label": "Authority",
        "jurisdiction_label": "Jurisdiction",
        "linked_issues": "Linked Theses",
        "empty": "No regulatory references found yet. Run an investigation to discover relevant standards and rulings.",
    },
    "coding": {
        "title": "Specifications & Standards",
        "item": "Specification",
        "type_label": "Type",
        "weight_label": "Authority",
        "jurisdiction_label": "Scope",
        "linked_issues": "Linked Requirements",
        "empty": "No specifications referenced yet. Run an investigation to discover relevant RFCs, docs, and standards.",
    },
    "academic_research": {
        "title": "Cited Works & Methodologies",
        "item": "Citation",
        "type_label": "Type",
        "weight_label": "Impact",
        "jurisdiction_label": "Field",
        "linked_issues": "Linked Questions",
        "empty": "No cited works found yet. Run an investigation to discover relevant papers and methodologies.",
    },
    "biomedical": {
        "title": "Clinical Guidelines & Protocols",
        "item": "Guideline",
        "type_label": "Type",
        "weight_label": "Evidence Level",
        "jurisdiction_label": "Jurisdiction",
        "linked_issues": "Linked Endpoints",
        "empty": "No clinical guidelines found yet. Run an investigation to discover relevant protocols and guidance.",
    },
}


def _fmt_authority_panel(data: dict, domain: str = "legal") -> str:
    """Render the authority / reference network panel (SO-4)."""
    if not data or not isinstance(data, dict):
        labels = _AUTHORITY_PANEL_LABELS.get(domain, _AUTHORITY_PANEL_LABELS["legal"])
        return f"<div class='viz-empty'>{_escape(labels['empty'])}</div>"
    if err := _error_html(data):
        return err
    authorities = data.get("authorities", [])
    issue_links = data.get("issue_links", {})
    labels = _AUTHORITY_PANEL_LABELS.get(domain, _AUTHORITY_PANEL_LABELS["legal"])

    if not authorities:
        return f"<div class='viz-empty'>{_escape(labels['empty'])}</div>"

    by_type: dict[str, int] = {}
    by_weight: dict[str, int] = {}
    for auth in authorities:
        if not isinstance(auth, dict):
            continue
        t = auth.get("authority_type", "unknown")
        w = auth.get("weight", "unknown")
        by_type[t] = by_type.get(t, 0) + 1
        by_weight[w] = by_weight.get(w, 0) + 1

    weight_colors = {
        "binding": "#16a34a",
        "persuasive": "#2563eb",
        "neutral": "#6b7280",
        "unknown": "#94a3b8",
    }

    header = (
        "<div style='display:flex;gap:24px;flex-wrap:wrap;margin-bottom:16px;'>"
        f"<div style='text-align:center;'>"
        f"<div style='font-size:28px;font-weight:700;color:#1e293b;'>{len(authorities)}</div>"
        f"<div style='font-size:11px;color:#6b7280;'>Total</div></div>"
    )
    for w, count in sorted(by_weight.items()):
        color = weight_colors.get(w, "#94a3b8")
        header += (
            f"<div style='text-align:center;'>"
            f"<div style='font-size:20px;font-weight:600;color:{color};'>{count}</div>"
            f"<div style='font-size:11px;color:#6b7280;'>{_escape(w.title())}</div></div>"
        )
    for t, count in sorted(by_type.items()):
        header += (
            f"<div style='text-align:center;'>"
            f"<div style='font-size:16px;font-weight:500;color:#475569;'>{count}</div>"
            f"<div style='font-size:11px;color:#6b7280;'>{_escape(t.title())}</div></div>"
        )
    header += "</div>"

    relevance_colors = {"supporting": "#16a34a", "attacking": "#dc2626", "neutral": "#6b7280"}

    rows_html = ""
    for auth in authorities:
        if not isinstance(auth, dict):
            continue
        aid = auth.get("id", "")
        citation = auth.get("citation", "—")
        name = auth.get("name") or ""
        auth_type = auth.get("authority_type", "unknown")
        weight = auth.get("weight", "unknown")
        jurisdiction = auth.get("jurisdiction") or "—"
        w_color = weight_colors.get(weight, "#94a3b8")

        links = issue_links.get(aid, [])
        if links:
            link_badges = " ".join(
                f"<span style='display:inline-block;padding:1px 6px;border-radius:8px;"
                f"background:{relevance_colors.get(lk.get('relevance', 'neutral'), '#6b7280')}18;"
                f"color:{relevance_colors.get(lk.get('relevance', 'neutral'), '#6b7280')};"
                f"font-size:10px;margin:1px;' title='{_escape(lk.get('relevance', 'neutral'))} · ID: {_escape(lk.get('issue_id', '?'))}'>"
                f"{_escape((lk.get('issue_title') or lk.get('issue_id', '?'))[:20])}</span>"
                for lk in links if isinstance(lk, dict)
            )
        else:
            link_badges = "<span style='color:#94a3b8;font-size:11px;'>none</span>"

        display_citation = _escape(citation[:50])
        display_name = f"<div style='font-size:10px;color:#6b7280;'>{_escape(name[:40])}</div>" if name else ""

        short_id = _escape(str(aid)[:12])
        rows_html += (
            f"<tr>"
            f"<td style='font-size:10px;color:#9ca3af;font-family:monospace' title='{_escape(str(aid))}'>{short_id}</td>"
            f"<td style='max-width:240px;' title='{_escape(citation)}'>"
            f"{display_citation}{display_name}</td>"
            f"<td style='font-size:11px;'>{_escape(auth_type)}</td>"
            f"<td><span style='display:inline-block;padding:1px 8px;border-radius:8px;"
            f"background:{w_color}22;color:{w_color};font-size:11px;"
            f"font-weight:600;'>{_escape(weight)}</span></td>"
            f"<td style='font-size:11px;'>{_escape(jurisdiction)}</td>"
            f"<td>{link_badges}</td>"
            f"</tr>"
        )

    table = (
        "<div style='overflow-x:auto;'>"
        "<table style='width:100%;border-collapse:collapse;font-size:13px;'>"
        "<thead><tr style='border-bottom:2px solid #e2e8f0;text-align:left;'>"
        "<th style='padding:6px 8px;'>ID</th>"
        f"<th style='padding:6px 8px;'>{_escape(labels['item'])}</th>"
        f"<th style='padding:6px 8px;'>{_escape(labels['type_label'])}</th>"
        f"<th style='padding:6px 8px;'>{_escape(labels['weight_label'])}</th>"
        f"<th style='padding:6px 8px;'>{_escape(labels['jurisdiction_label'])}</th>"
        f"<th style='padding:6px 8px;'>{_escape(labels['linked_issues'])}</th>"
        "</tr></thead><tbody>"
        + rows_html
        + "</tbody></table></div>"
    )

    return f"<div class='viz-shell'>{header}{table}</div>"


_DOC_PANEL_LABELS = {
    "legal": {
        "empty": "No document profiles yet. Run an investigation to analyze your documents.",
        "total": "Total Documents", "ingested": "Ingested", "profiled": "Profiled",
        "restricted": "Privileged", "doc_col": "Document", "type_col": "Type",
        "side_col": "Source Side", "author_col": "Author",
    },
    "finance": {
        "empty": "No document profiles yet. Run an investigation to analyze your filings.",
        "total": "Total Filings", "ingested": "Ingested", "profiled": "Profiled",
        "restricted": "MNPI-Flagged", "doc_col": "Filing", "type_col": "Type",
        "side_col": "Source", "author_col": "Issuer",
    },
    "coding": {
        "empty": "No source profiles yet. Run an investigation to analyze your codebase.",
        "total": "Total Sources", "ingested": "Indexed", "profiled": "Profiled",
        "restricted": "Restricted", "doc_col": "Source", "type_col": "Type",
        "side_col": "Scope", "author_col": "Author",
    },
    "academic_research": {
        "empty": "No document profiles yet. Run an investigation to analyze your papers.",
        "total": "Total Papers", "ingested": "Ingested", "profiled": "Profiled",
        "restricted": "Embargoed", "doc_col": "Paper", "type_col": "Type",
        "side_col": "Source", "author_col": "Author",
    },
    "biomedical": {
        "empty": "No document profiles yet. Run an investigation to analyze your reports.",
        "total": "Total Reports", "ingested": "Ingested", "profiled": "Profiled",
        "restricted": "Patient-Protected", "doc_col": "Report", "type_col": "Type",
        "side_col": "Source", "author_col": "Author",
    },
}


def _fmt_document_intelligence_panel(data: dict, domain: str = "legal", trust_overrides: list | None = None) -> str:
    """Render the document intelligence panel (SO-5)."""
    if not data or not isinstance(data, dict):
        labels = _DOC_PANEL_LABELS.get(domain, _DOC_PANEL_LABELS["legal"])
        return f"<div class='viz-empty'>{_escape(labels['empty'])}</div>"
    if err := _error_html(data):
        return err
    cards = data.get("cards", [])
    total_inv = data.get("total_inventory", 0)
    ingested = data.get("ingested_count", 0)
    labels = _DOC_PANEL_LABELS.get(domain, _DOC_PANEL_LABELS["legal"])

    if not cards and total_inv == 0:
        return f"<div class='viz-empty'>{_escape(labels['empty'])}</div>"

    by_type: dict[str, int] = {}
    by_side: dict[str, int] = {}
    priv_count = 0
    for c in cards:
        t = c.get("doc_type") or "unknown"
        by_type[t] = by_type.get(t, 0) + 1
        s = c.get("source_side") or "unknown"
        by_side[s] = by_side.get(s, 0) + 1
        if c.get("privilege_flag"):
            priv_count += 1

    header = (
        "<div style='display:flex;gap:24px;flex-wrap:wrap;margin-bottom:16px;'>"
        f"<div style='text-align:center;'>"
        f"<div style='font-size:28px;font-weight:700;color:#1e293b;'>{total_inv}</div>"
        f"<div style='font-size:11px;color:#6b7280;'>{_escape(labels['total'])}</div></div>"
        f"<div style='text-align:center;'>"
        f"<div style='font-size:28px;font-weight:700;color:#16a34a;'>{ingested}</div>"
        f"<div style='font-size:11px;color:#6b7280;'>{_escape(labels['ingested'])}</div></div>"
        f"<div style='text-align:center;'>"
        f"<div style='font-size:28px;font-weight:700;color:#2563eb;'>{len(cards)}</div>"
        f"<div style='font-size:11px;color:#6b7280;'>{_escape(labels['profiled'])}</div></div>"
    )
    if priv_count:
        header += (
            f"<div style='text-align:center;'>"
            f"<div style='font-size:20px;font-weight:600;color:#dc2626;'>{priv_count}</div>"
            f"<div style='font-size:11px;color:#6b7280;'>{_escape(labels['restricted'])}</div></div>"
        )
    for t, count in sorted(by_type.items(), key=lambda x: -x[1])[:6]:
        header += (
            f"<div style='text-align:center;'>"
            f"<div style='font-size:16px;font-weight:500;color:#475569;'>{count}</div>"
            f"<div style='font-size:11px;color:#6b7280;'>{_escape(t.replace('_', ' ').title())}</div></div>"
        )
    header += "</div>"

    operative_colors = {
        "operative": "#16a34a", "superseded": "#d97706", "draft": "#6b7280",
        "unknown": "#94a3b8", "revoked": "#dc2626",
    }

    rows_html = ""
    for c in cards:
        path = c.get("relative_path") or "—"
        title = c.get("title") or ""
        doc_type = c.get("doc_type") or "—"
        source_side = c.get("source_side") or "—"
        author = c.get("author") or "—"
        operative = c.get("operative_status") or "unknown"
        priv = c.get("privilege_flag", False)
        salience = _safe_float(c.get("salience_score", 0.0))
        flags = c.get("unresolved_flags") or []
        flag_count = len(flags) if isinstance(flags, list) else 0

        op_color = operative_colors.get(operative, "#94a3b8")
        display_path = _escape(path.rsplit("/", 1)[-1][:30])
        display_title = f"<div style='font-size:10px;color:#6b7280;'>{_escape(title[:35])}</div>" if title else ""

        priv_badge = ""
        if priv:
            priv_badge = (
                "<span style='margin-left:4px;padding:1px 5px;border-radius:8px;"
                "background:#fef2f2;color:#dc2626;font-size:9px;font-weight:700;"
                "text-transform:uppercase;'>restricted</span>"
            )
        flag_badge = ""
        if flag_count:
            flag_badge = (
                f"<span style='margin-left:4px;padding:1px 5px;border-radius:8px;"
                f"background:#fef3c7;color:#92400e;font-size:9px;font-weight:700;'>"
                f"{flag_count} flag{'s' if flag_count > 1 else ''}</span>"
            )

        rows_html += (
            f"<tr>"
            f"<td style='max-width:200px;overflow:hidden;text-overflow:ellipsis;"
            f"white-space:nowrap;' title='{_escape(path)}'>{display_path}{display_title}{priv_badge}</td>"
            f"<td style='font-size:11px;'>{_escape(doc_type.replace('_', ' '))}</td>"
            f"<td style='font-size:11px;'>{_escape(source_side)}</td>"
            f"<td style='font-size:11px;max-width:100px;overflow:hidden;"
            f"text-overflow:ellipsis;white-space:nowrap;'>{_escape(author)}</td>"
            f"<td><span style='display:inline-block;padding:1px 8px;border-radius:8px;"
            f"background:{op_color}22;color:{op_color};font-size:11px;"
            f"font-weight:600;'>{_escape(operative)}</span></td>"
            f"<td style='text-align:center;font-size:11px;'>{salience:.2f}</td>"
            f"<td>{flag_badge}</td>"
            f"</tr>"
        )

    table = (
        "<div style='overflow-x:auto;'>"
        "<table style='width:100%;border-collapse:collapse;font-size:13px;'>"
        "<thead><tr style='border-bottom:2px solid #e2e8f0;text-align:left;'>"
        f"<th style='padding:6px 8px;'>{_escape(labels['doc_col'])}</th>"
        f"<th style='padding:6px 8px;'>{_escape(labels['type_col'])}</th>"
        f"<th style='padding:6px 8px;'>{_escape(labels['side_col'])}</th>"
        f"<th style='padding:6px 8px;'>{_escape(labels['author_col'])}</th>"
        "<th style='padding:6px 8px;'>Status</th>"
        "<th style='padding:6px 8px;text-align:center;'>Salience</th>"
        "<th style='padding:6px 8px;'>Flags</th>"
        "</tr></thead><tbody>"
        + rows_html
        + "</tbody></table></div>"
    )

    trust_section = ""
    _overrides = trust_overrides or []
    if _overrides:
        trust_dist: dict[str, int] = {}
        for ov in _overrides:
            if not isinstance(ov, dict):
                continue
            lvl = str(ov.get("trust_level", "normal")).lower()
            trust_dist[lvl] = trust_dist.get(lvl, 0) + 1
        _trust_colors = {"high": "#16a34a", "normal": "#2563eb", "low": "#dc2626"}
        pills = []
        for lvl in ("high", "normal", "low"):
            cnt = trust_dist.get(lvl, 0)
            if cnt:
                color = _trust_colors.get(lvl, "#6b7280")
                pills.append(
                    f"<span style='display:inline-block;padding:2px 10px;border-radius:8px;"
                    f"background:{color}18;color:{color};font-size:12px;font-weight:600;"
                    f"margin-right:8px;'>{cnt} {lvl}</span>"
                )
        if pills:
            trust_section = (
                "<div style='margin-bottom:12px;padding:8px 12px;border-radius:8px;"
                "background:#f8fafc;border:1px solid #e2e8f0;'>"
                "<span style='font-size:11px;font-weight:600;color:#475569;"
                "margin-right:8px;'>Source Trust:</span>"
                + "".join(pills)
                + "</div>"
            )

    return f"<div class='viz-shell'>{header}{trust_section}{table}</div>"


_BELIEF_REVISION_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Truth Maintenance Trail",
        "assertion_col": "Assertion",
        "transition_col": "State Change",
        "cause_col": "Cause",
        "confidence_col": "Confidence",
        "empty": "No belief revisions recorded. The assertion graph has not been corrected or challenged.",
    },
    "finance": {
        "title": "Fact Revision Audit",
        "assertion_col": "Financial Fact",
        "transition_col": "Status Change",
        "cause_col": "Trigger",
        "confidence_col": "Confidence",
        "empty": "No revisions recorded. Financial facts have not been challenged or restated.",
    },
    "coding": {
        "title": "Knowledge Base Revisions",
        "assertion_col": "Technical Fact",
        "transition_col": "Status Change",
        "cause_col": "Trigger",
        "confidence_col": "Confidence",
        "empty": "No revisions recorded. Extracted technical facts are unchanged.",
    },
    "academic_research": {
        "title": "Evidence Revision History",
        "assertion_col": "Research Finding",
        "transition_col": "Evidence Status",
        "cause_col": "Revision Cause",
        "confidence_col": "Confidence",
        "empty": "No revisions recorded. Research findings have not been challenged or superseded.",
    },
    "biomedical": {
        "title": "Clinical Evidence Revisions",
        "assertion_col": "Clinical Finding",
        "transition_col": "Evidence Status",
        "cause_col": "Revision Cause",
        "confidence_col": "Confidence",
        "empty": "No revisions recorded. Clinical evidence has not been revised or contradicted.",
    },
}


def _fmt_belief_revision_panel(revisions: list[dict], domain: str = "legal") -> str:
    labels = _BELIEF_REVISION_LABELS.get(domain, _BELIEF_REVISION_LABELS["legal"])
    if not revisions:
        return f"<div class='viz-empty'>{labels['empty']}</div>"

    rows_html = ""
    for rev in revisions:
        if not isinstance(rev, dict):
            continue
        text = _escape(str(rev.get("proposition_text") or "—")[:120])
        old_state = _escape(str(rev.get("old_belief_state") or "?"))
        new_state = _escape(str(rev.get("new_belief_state") or "?"))
        cause = _escape(str(rev.get("cause") or "—"))
        old_conf = _safe_float(rev.get("old_confidence"), 0)
        new_conf = _safe_float(rev.get("new_confidence"), 0)
        delta = new_conf - old_conf
        delta_icon = "▲" if delta > 0 else ("▼" if delta < 0 else "—")
        delta_class = "tone-green" if delta > 0 else ("tone-red" if delta < 0 else "")
        transition = (
            f"<span class='pill pill-neutral'>{old_state}</span>"
            f" → "
            f"<span class='pill pill-blue'>{new_state}</span>"
        )
        rows_html += (
            f"<tr><td title='{text}'>{text}</td>"
            f"<td>{transition}</td>"
            f"<td>{cause}</td>"
            f"<td class='{delta_class}'>{delta_icon} {new_conf:.2f}</td></tr>"
        )

    count = len(revisions)
    header = f"<div class='viz-header'><strong>{labels['title']}</strong> — {count} revision{'s' if count != 1 else ''}</div>"
    table = (
        "<div class='table-wrap'><table class='viz-table'>"
        f"<thead><tr><th>{labels['assertion_col']}</th><th>{labels['transition_col']}</th>"
        f"<th>{labels['cause_col']}</th><th>{labels['confidence_col']}</th></tr></thead>"
        "<tbody>"
        + rows_html
        + "</tbody></table></div>"
    )
    return f"<div class='viz-shell'>{header}{table}</div>"


_CONTRADICTION_PANEL_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Contradiction Analysis",
        "attacker_col": "Challenging Assertion",
        "attacked_col": "Challenged Assertion",
        "link_col": "Conflict Type",
        "status_col": "Status",
        "empty": "No active contradictions found. The assertion graph has no unresolved conflicts.",
    },
    "finance": {
        "title": "Conflicting Financial Claims",
        "attacker_col": "Contrary Claim",
        "attacked_col": "Challenged Claim",
        "link_col": "Conflict Type",
        "status_col": "Status",
        "empty": "No conflicting financial claims detected.",
    },
    "coding": {
        "title": "Conflicting Technical Facts",
        "attacker_col": "Contradicting Fact",
        "attacked_col": "Contradicted Fact",
        "link_col": "Conflict Type",
        "status_col": "Status",
        "empty": "No conflicting technical facts detected.",
    },
    "academic_research": {
        "title": "Conflicting Research Findings",
        "attacker_col": "Contradicting Finding",
        "attacked_col": "Contradicted Finding",
        "link_col": "Conflict Type",
        "status_col": "Status",
        "empty": "No conflicting research findings detected.",
    },
    "biomedical": {
        "title": "Conflicting Clinical Evidence",
        "attacker_col": "Contradicting Evidence",
        "attacked_col": "Contradicted Evidence",
        "link_col": "Conflict Type",
        "status_col": "Status",
        "empty": "No conflicting clinical evidence detected.",
    },
}


def _fmt_contradiction_panel(contradictions: list[dict], domain: str = "legal") -> str:
    labels = _CONTRADICTION_PANEL_LABELS.get(domain, _CONTRADICTION_PANEL_LABELS["legal"])
    if not contradictions:
        return f"<div class='viz-empty'>{labels['empty']}</div>"

    _BELIEF_PILLS = {
        "operative": "pill-green",
        "admitted": "pill-green",
        "alleged": "pill-neutral",
        "argued": "pill-neutral",
        "inferred": "pill-neutral",
        "disputed": "pill-orange",
        "superseded": "pill-red",
        "withdrawn": "pill-red",
        "not_performed": "pill-red",
    }

    rows_html = ""
    for c in contradictions:
        if not isinstance(c, dict):
            continue
        attacker_prop = _escape(str(c.get("attacker_prop") or "—")[:120])
        attacked_prop = _escape(str(c.get("attacked_prop") or "—")[:120])
        link_type = _escape(str(c.get("link_type") or "—"))
        attacker_belief = str(c.get("attacker_belief") or "unknown")
        attacked_belief = str(c.get("attacked_belief") or "unknown")
        a_pill = _BELIEF_PILLS.get(attacker_belief, "pill-neutral")
        d_pill = _BELIEF_PILLS.get(attacked_belief, "pill-neutral")
        status = "Open Conflict"
        status_class = "pill-orange"
        if attacked_belief == "disputed":
            status = "Disputed"
            status_class = "pill-orange"
        elif attacked_belief in ("superseded", "withdrawn", "resolved"):
            status = "Resolved"
            status_class = "pill-green"

        rows_html += (
            f"<tr>"
            f"<td title='{attacker_prop}'>{attacker_prop} "
            f"<span class='pill {a_pill}'>{_escape(attacker_belief)}</span></td>"
            f"<td title='{attacked_prop}'>{attacked_prop} "
            f"<span class='pill {d_pill}'>{_escape(attacked_belief)}</span></td>"
            f"<td><span class='pill pill-blue'>{link_type}</span></td>"
            f"<td><span class='pill {status_class}'>{status}</span></td>"
            f"</tr>"
        )

    if not rows_html:
        return f"<div class='viz-empty'>{labels['empty']}</div>"
    count = rows_html.count("<tr>")
    header = (
        f"<div class='viz-header'><strong>{labels['title']}</strong>"
        f" — {count} active conflict{'s' if count != 1 else ''}</div>"
    )
    table = (
        "<div class='table-wrap'><table class='viz-table'>"
        f"<thead><tr><th>{labels['attacker_col']}</th><th>{labels['attacked_col']}</th>"
        f"<th>{labels['link_col']}</th><th>{labels['status_col']}</th></tr></thead>"
        "<tbody>" + rows_html + "</tbody></table></div>"
    )
    return f"<div class='viz-shell'>{header}{table}</div>"


# ---------------------------------------------------------------------------
# Document Version Chains panel (SO-5 sourcing transparency)
# ---------------------------------------------------------------------------

_DOCUMENT_VERSION_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Document Version Chains",
        "doc_col": "Document",
        "status_col": "Version Status",
        "empty": "No document version chains detected. Documents have not been grouped into version families.",
    },
    "finance": {
        "title": "Filing Version History",
        "doc_col": "Filing / Report",
        "status_col": "Version Status",
        "empty": "No filing version chains detected.",
    },
    "coding": {
        "title": "Artifact Version Chains",
        "doc_col": "Artifact",
        "status_col": "Version Status",
        "empty": "No artifact version chains detected.",
    },
    "academic_research": {
        "title": "Manuscript Version History",
        "doc_col": "Manuscript / Dataset",
        "status_col": "Version Status",
        "empty": "No manuscript version chains detected.",
    },
    "biomedical": {
        "title": "Protocol Version History",
        "doc_col": "Protocol / Study Document",
        "status_col": "Version Status",
        "empty": "No protocol version chains detected.",
    },
}


def _fmt_document_versions_panel(families: list[dict], domain: str = "legal") -> str:
    labels = _DOCUMENT_VERSION_LABELS.get(domain, _DOCUMENT_VERSION_LABELS["legal"])
    if not families:
        return f"<div class='viz-empty'>{labels['empty']}</div>"

    families_html = ""
    total_docs = 0
    for fam in families:
        if not isinstance(fam, dict):
            continue
        members = fam.get("members") or []
        if not members:
            continue
        total_docs += len(members)
        rows = ""
        for m in members:
            if not isinstance(m, dict):
                continue
            path = m.get("relative_path") or "—"
            short_name = _escape(path.rsplit("/", 1)[-1] if "/" in str(path) else str(path))
            full_path = _escape(str(path))
            is_op = m.get("is_operative", False)
            if is_op:
                badge = "<span class='pill pill-green'>Operative (Current)</span>"
            else:
                badge = "<span class='pill pill-neutral'>Superseded</span>"
            rows += (
                f"<tr>"
                f"<td title='{full_path}'>{short_name}</td>"
                f"<td>{badge}</td>"
                f"</tr>"
            )
        if rows:
            families_html += (
                f"<table class='viz-table' style='margin-bottom:12px'>"
                f"<thead><tr><th>{labels['doc_col']}</th>"
                f"<th>{labels['status_col']}</th></tr></thead>"
                f"<tbody>{rows}</tbody></table>"
            )

    if not families_html:
        return f"<div class='viz-empty'>{labels['empty']}</div>"

    fam_count = len([f for f in families if isinstance(f, dict) and (f.get("members") or [])])
    header = (
        f"<div class='viz-header'><strong>{labels['title']}</strong>"
        f" — {fam_count} version chain{'s' if fam_count != 1 else ''}, "
        f"{total_docs} document{'s' if total_docs != 1 else ''}</div>"
    )
    return f"<div class='viz-shell'>{header}<div class='table-wrap'>{families_html}</div></div>"


_OPERATIVE_VERSION_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Operative Version Lookup",
        "operative": "Operative (current)",
        "superseded": "Superseded by",
        "same": "This document is already the operative version.",
        "empty": "Enter a document ID to check its operative version.",
    },
    "finance": {
        "title": "Current Filing Lookup",
        "operative": "Current filing",
        "superseded": "Superseded by",
        "same": "This filing is already the current version.",
        "empty": "Enter a filing ID to check its current version.",
    },
    "coding": {
        "title": "Current Artifact Lookup",
        "operative": "Current artifact",
        "superseded": "Superseded by",
        "same": "This artifact is already the current version.",
        "empty": "Enter an artifact ID to check its current version.",
    },
    "academic_research": {
        "title": "Current Manuscript Lookup",
        "operative": "Current version",
        "superseded": "Superseded by",
        "same": "This manuscript is already the current version.",
        "empty": "Enter a document ID to check its current version.",
    },
    "biomedical": {
        "title": "Active Protocol Lookup",
        "operative": "Active protocol",
        "superseded": "Superseded by",
        "same": "This protocol is already the active version.",
        "empty": "Enter a protocol ID to check its active version.",
    },
}


def _fmt_operative_version_result(data: dict, domain: str = "legal") -> str:
    L = _OPERATIVE_VERSION_LABELS.get(domain, _OPERATIVE_VERSION_LABELS["legal"])
    if not data or not isinstance(data, dict):
        return f"<div class='viz-empty'>{L['empty']}</div>"
    if err := _error_html(data):
        return err

    doc_id = _escape(str(data.get("doc_id", "")))
    operative_id = _escape(str(data.get("operative_doc_id", "")))
    is_operative = data.get("is_operative", True)

    if is_operative:
        return (
            f"<div style='padding:8px;background:#f0fdf4;border:1px solid #bbf7d0;border-radius:6px;'>"
            f"<strong>{doc_id}</strong> — {_escape(L['same'])}</div>"
        )
    return (
        f"<div style='padding:8px;background:#fef3c7;border:1px solid #fde68a;border-radius:6px;'>"
        f"<strong>{doc_id}</strong> {_escape(L['superseded'])}: "
        f"<strong style='color:#059669;'>{operative_id}</strong> ({_escape(L['operative'])})</div>"
    )


# ---------------------------------------------------------------------------
# Quantitative Threshold Violations panel (SO-6)
# ---------------------------------------------------------------------------

_QUANT_THRESHOLD_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Financial Health Alerts",
        "threshold_col": "Threshold",
        "level_col": "Severity",
        "desc_col": "Finding",
        "empty": "No quantitative threshold violations. Financial figures are consistent.",
    },
    "finance": {
        "title": "Financial Risk Alerts",
        "threshold_col": "Risk Metric",
        "level_col": "Severity",
        "desc_col": "Finding",
        "empty": "No financial risk thresholds breached.",
    },
    "coding": {
        "title": "Metric Threshold Alerts",
        "threshold_col": "Metric",
        "level_col": "Severity",
        "desc_col": "Finding",
        "empty": "No quantitative thresholds breached.",
    },
    "academic_research": {
        "title": "Statistical Threshold Alerts",
        "threshold_col": "Threshold",
        "level_col": "Severity",
        "desc_col": "Finding",
        "empty": "No statistical thresholds breached.",
    },
    "biomedical": {
        "title": "Clinical Threshold Alerts",
        "threshold_col": "Threshold",
        "level_col": "Severity",
        "desc_col": "Finding",
        "empty": "No clinical quantitative thresholds breached.",
    },
}

_SEVERITY_PILLS = {
    "HIGH": "pill-red",
    "MED": "pill-orange",
    "LOW": "pill-neutral",
}


def _fmt_quant_thresholds_panel(violations: list[dict], domain: str = "legal") -> str:
    labels = _QUANT_THRESHOLD_LABELS.get(domain, _QUANT_THRESHOLD_LABELS["legal"])
    if not violations:
        return f"<div class='viz-empty'>{labels['empty']}</div>"

    rows_html = ""
    for v in violations:
        if not isinstance(v, dict):
            continue
        threshold = _escape(str(v.get("threshold") or "—").replace("_", " ").title())
        level = str(v.get("level") or "LOW")
        pill_class = _SEVERITY_PILLS.get(level, "pill-neutral")
        desc = _escape(str(v.get("description") or "—")[:200])
        raw_amt = v.get("amount")
        amt_cell = ""
        if raw_amt is not None:
            try:
                amt_cell = f"<td style='text-align:right;font-family:monospace;font-size:12px;'>{float(raw_amt):,.2f}</td>"
            except (ValueError, TypeError):
                amt_cell = "<td>—</td>"
        else:
            amt_cell = "<td>—</td>"
        rows_html += (
            f"<tr>"
            f"<td>{threshold}</td>"
            f"<td><span class='pill {pill_class}'>{_escape(level)}</span></td>"
            f"<td>{desc}</td>"
            f"{amt_cell}"
            f"</tr>"
        )

    if not rows_html:
        return f"<div class='viz-empty'>{labels['empty']}</div>"

    count = rows_html.count("<tr>")
    high_count = sum(1 for v in violations if isinstance(v, dict) and v.get("level") == "HIGH")
    severity_note = f" ({high_count} HIGH)" if high_count else ""
    header = (
        f"<div class='viz-header'><strong>{labels['title']}</strong>"
        f" — {count} violation{'s' if count != 1 else ''}{severity_note}</div>"
    )
    table = (
        "<div class='table-wrap'><table class='viz-table'>"
        f"<thead><tr><th>{labels['threshold_col']}</th><th>{labels['level_col']}</th>"
        f"<th>{labels['desc_col']}</th><th style='text-align:right;'>Amount</th></tr></thead>"
        "<tbody>" + rows_html + "</tbody></table></div>"
    )
    return f"<div class='viz-shell'>{header}{table}</div>"


_SYSTEM_HEALTH_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Truth Maintenance Health",
        "good": "All systems healthy — no oscillating assertions, dispute rate within bounds.",
        "attention": "Attention needed — review disputed assertions and oscillation patterns.",
        "row_assertions": "Total Assertions",
        "row_disputed": "Disputed",
        "row_dispute_rate": "Dispute Rate",
        "row_revisions": "Belief Revisions",
        "row_gaps": "Open Gaps",
        "row_contradictions": "Contradictions",
        "row_versions": "Version Chains",
        "row_oscillating": "Oscillating Assertions",
    },
    "finance": {
        "title": "Analysis Health",
        "good": "All systems healthy — positions are stable, no oscillating claims.",
        "attention": "Attention needed — review disputed positions and oscillation patterns.",
        "row_assertions": "Total Claims",
        "row_disputed": "Disputed",
        "row_dispute_rate": "Dispute Rate",
        "row_revisions": "Position Revisions",
        "row_gaps": "Open Gaps",
        "row_contradictions": "Contradictions",
        "row_versions": "Version Chains",
        "row_oscillating": "Oscillating Claims",
    },
    "coding": {
        "title": "Analysis Health",
        "good": "All systems healthy — no oscillating findings, dispute rate within bounds.",
        "attention": "Attention needed — review disputed findings and oscillation patterns.",
        "row_assertions": "Total Findings",
        "row_disputed": "Disputed",
        "row_dispute_rate": "Dispute Rate",
        "row_revisions": "Finding Revisions",
        "row_gaps": "Open Gaps",
        "row_contradictions": "Contradictions",
        "row_versions": "Version Chains",
        "row_oscillating": "Oscillating Findings",
    },
    "academic_research": {
        "title": "Analysis Health",
        "good": "All systems healthy — no oscillating claims, dispute rate within bounds.",
        "attention": "Attention needed — review disputed claims and oscillation patterns.",
        "row_assertions": "Total Claims",
        "row_disputed": "Disputed",
        "row_dispute_rate": "Dispute Rate",
        "row_revisions": "Claim Revisions",
        "row_gaps": "Open Gaps",
        "row_contradictions": "Contradictions",
        "row_versions": "Version Chains",
        "row_oscillating": "Oscillating Claims",
    },
    "biomedical": {
        "title": "Analysis Health",
        "good": "All systems healthy — no oscillating findings, dispute rate within bounds.",
        "attention": "Attention needed — review disputed findings and oscillation patterns.",
        "row_assertions": "Total Findings",
        "row_disputed": "Disputed",
        "row_dispute_rate": "Dispute Rate",
        "row_revisions": "Finding Revisions",
        "row_gaps": "Open Gaps",
        "row_contradictions": "Contradictions",
        "row_versions": "Version Chains",
        "row_oscillating": "Oscillating Findings",
    },
}


def _fmt_system_health_panel(health: dict, domain: str = "legal") -> str:
    labels = _SYSTEM_HEALTH_LABELS.get(domain, _SYSTEM_HEALTH_LABELS["legal"])
    if not health or not isinstance(health, dict):
        return "<div class='viz-empty'>No health data available yet.</div>"

    score = health.get("health_score", "good")
    pill_class = "pill-green" if score == "good" else "pill-orange"
    banner_text = labels["good"] if score == "good" else labels["attention"]
    header = (
        f"<div class='viz-header'><strong>{labels['title']}</strong>"
        f" — <span class='pill {pill_class}'>{_escape(score.replace('_', ' ').title())}</span></div>"
        f"<div style='padding:4px 8px;font-size:0.9em;color:#666;'>{banner_text}</div>"
    )

    dispute_rate = health.get("disputed_fraction", 0)
    rate_pct = f"{dispute_rate * 100:.1f}%"
    rate_pill = "pill-green" if dispute_rate < 0.15 else ("pill-orange" if dispute_rate < 0.3 else "pill-red")

    osc = health.get("oscillating_count", 0)
    osc_pill = "pill-green" if osc == 0 else "pill-red"

    rows = (
        f"<tr><td>{labels['row_assertions']}</td><td><strong>{health.get('assertion_count', 0)}</strong></td></tr>"
        f"<tr><td>{labels['row_disputed']}</td><td>{health.get('disputed_count', 0)}</td></tr>"
        f"<tr><td>{labels['row_dispute_rate']}</td><td><span class='pill {rate_pill}'>{rate_pct}</span></td></tr>"
        f"<tr><td>{labels['row_revisions']}</td><td>{health.get('revision_count', 0)}</td></tr>"
        f"<tr><td>{labels['row_gaps']}</td><td>{health.get('open_gap_count', 0)}</td></tr>"
        f"<tr><td>{labels['row_contradictions']}</td><td>{health.get('contradiction_count', 0)}</td></tr>"
        f"<tr><td>{labels['row_versions']}</td><td>{health.get('version_chain_count', 0)}</td></tr>"
        f"<tr><td>{labels['row_oscillating']}</td><td><span class='pill {osc_pill}'>{osc}</span></td></tr>"
    )

    table = (
        "<div class='table-wrap'><table class='viz-table'>"
        "<thead><tr><th>Metric</th><th>Value</th></tr></thead>"
        "<tbody>" + rows + "</tbody></table></div>"
    )
    return f"<div class='viz-shell'>{header}{table}</div>"


_SO_SCORECARD_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Sacred Outcomes Scorecard",
        "subtitle": "Tracks whether the matter model meets quantitative success criteria.",
        "so1": "SO-1 Durability",
        "so2_struct": "SO-2 Structure",
        "so2_revision": "SO-2 Revisions",
        "so2_provenance": "SO-2 Provenance",
        "so3": "SO-3 Steerability",
        "so4": "SO-4 Coverage",
        "so5": "SO-5 Calibration",
        "so6": "SO-6 Quant Source",
        "so7": "SO-7 Gap Surface",
    },
    "finance": {
        "title": "Analysis Quality Scorecard",
        "subtitle": "Tracks whether the analysis model meets quality criteria.",
        "so1": "Durability",
        "so2_struct": "Structure Rate",
        "so2_revision": "Position Revisions",
        "so2_provenance": "Provenance",
        "so3": "Steerability",
        "so4": "Coverage",
        "so5": "Source Calibration",
        "so6": "Numeric Sourcing",
        "so7": "Gap Surface",
    },
    "coding": {
        "title": "Analysis Quality Scorecard",
        "subtitle": "Tracks whether the analysis model meets quality criteria.",
        "so1": "Durability",
        "so2_struct": "Structure Rate",
        "so2_revision": "Finding Revisions",
        "so2_provenance": "Provenance",
        "so3": "Steerability",
        "so4": "Coverage",
        "so5": "Source Calibration",
        "so6": "Numeric Sourcing",
        "so7": "Gap Surface",
    },
    "academic_research": {
        "title": "Analysis Quality Scorecard",
        "subtitle": "Tracks whether the analysis model meets quality criteria.",
        "so1": "Durability",
        "so2_struct": "Structure Rate",
        "so2_revision": "Claim Revisions",
        "so2_provenance": "Provenance",
        "so3": "Steerability",
        "so4": "Coverage",
        "so5": "Source Calibration",
        "so6": "Numeric Sourcing",
        "so7": "Gap Surface",
    },
    "biomedical": {
        "title": "Analysis Quality Scorecard",
        "subtitle": "Tracks whether the analysis model meets quality criteria.",
        "so1": "Durability",
        "so2_struct": "Structure Rate",
        "so2_revision": "Finding Revisions",
        "so2_provenance": "Provenance",
        "so3": "Steerability",
        "so4": "Coverage",
        "so5": "Source Calibration",
        "so6": "Numeric Sourcing",
        "so7": "Gap Surface",
    },
}


_RUN_STATUS_COLORS: dict[str, str] = {
    "completed": "#22c55e",
    "running": "#3b82f6",
    "stopped": "#eab308",
    "failed": "#ef4444",
    "error": "#ef4444",
}

_OPERATION_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "query": "Investigation",
        "clarification_answer": "Clarification",
        "redirect": "Redirect",
        "resume": "Resume",
        "correction": "Correction",
        "verify": "Verification",
    },
    "finance": {
        "query": "Analysis",
        "clarification_answer": "Data Request",
        "redirect": "Rebalance",
        "resume": "Resume",
        "correction": "Adjustment",
        "verify": "Reconciliation",
    },
    "coding": {
        "query": "Analysis",
        "clarification_answer": "Clarification",
        "redirect": "Redirect",
        "resume": "Resume",
        "correction": "Patch",
        "verify": "Verification",
    },
    "academic_research": {
        "query": "Literature Review",
        "clarification_answer": "Clarification",
        "redirect": "Reorientation",
        "resume": "Resume",
        "correction": "Revision",
        "verify": "Peer Review",
    },
    "biomedical": {
        "query": "Case Review",
        "clarification_answer": "Clinical Query",
        "redirect": "Differential Pivot",
        "resume": "Resume",
        "correction": "Amendment",
        "verify": "Validation",
    },
}

_INVESTIGATION_HISTORY_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Investigation History",
        "empty": "No investigation runs recorded yet.",
        "col_query": "Query",
        "col_type": "Type",
        "col_status": "Status",
        "col_mode": "Mode",
        "col_llm": "LLM Calls",
        "col_cost": "Cost",
        "col_reuse": "Reuse",
        "col_started": "Started",
        "run_unit": "run",
        "runs_unit": "runs",
    },
    "finance": {
        "title": "Analysis History",
        "empty": "No analysis runs recorded yet.",
        "col_query": "Query",
        "col_type": "Type",
        "col_status": "Status",
        "col_mode": "Mode",
        "col_llm": "Model Calls",
        "col_cost": "Cost",
        "col_reuse": "Cache Rate",
        "col_started": "Started",
        "run_unit": "run",
        "runs_unit": "runs",
    },
    "coding": {
        "title": "Analysis History",
        "empty": "No analysis runs recorded yet.",
        "col_query": "Query",
        "col_type": "Type",
        "col_status": "Status",
        "col_mode": "Mode",
        "col_llm": "LLM Calls",
        "col_cost": "Cost",
        "col_reuse": "Cache Rate",
        "col_started": "Started",
        "run_unit": "run",
        "runs_unit": "runs",
    },
    "academic_research": {
        "title": "Research Session History",
        "empty": "No research sessions recorded yet.",
        "col_query": "Research Question",
        "col_type": "Type",
        "col_status": "Status",
        "col_mode": "Mode",
        "col_llm": "LLM Calls",
        "col_cost": "Cost",
        "col_reuse": "Reuse",
        "col_started": "Started",
        "run_unit": "session",
        "runs_unit": "sessions",
    },
    "biomedical": {
        "title": "Case Review History",
        "empty": "No case reviews recorded yet.",
        "col_query": "Clinical Query",
        "col_type": "Type",
        "col_status": "Status",
        "col_mode": "Mode",
        "col_llm": "Model Calls",
        "col_cost": "Cost",
        "col_reuse": "Reuse",
        "col_started": "Started",
        "run_unit": "review",
        "runs_unit": "reviews",
    },
}


def _fmt_investigation_history_panel(runs: list, domain: str = "legal") -> str:
    L = _INVESTIGATION_HISTORY_LABELS.get(domain, _INVESTIGATION_HISTORY_LABELS["legal"])
    OL = _OPERATION_LABELS.get(domain, _OPERATION_LABELS["legal"])
    if not runs or not isinstance(runs, list):
        return f"<div class='viz-empty'>{L['empty']}</div>"

    valid = [r for r in runs if isinstance(r, dict)]
    if not valid:
        return f"<div class='viz-empty'>{L['empty']}</div>"

    count = len(valid)
    unit = L["run_unit"] if count == 1 else L["runs_unit"]
    parts = [
        f"<h3 style='margin:0 0 8px 0;'>{L['title']}</h3>",
        f"<div style='color:#666;font-size:0.9em;margin-bottom:8px;'>{count} {unit} recorded</div>",
        "<table style='border-collapse:collapse;width:100%;font-size:0.85em;'>",
        "<tr style='background:#f1f5f9;'>"
        f"<th style='text-align:left;padding:4px 8px;'>{L['col_query']}</th>"
        f"<th style='text-align:left;padding:4px 8px;'>{L['col_type']}</th>"
        f"<th style='text-align:left;padding:4px 8px;'>{L['col_status']}</th>"
        f"<th style='text-align:left;padding:4px 8px;'>{L['col_mode']}</th>"
        f"<th style='text-align:left;padding:4px 8px;'>{L['col_llm']}</th>"
        f"<th style='text-align:left;padding:4px 8px;'>{L['col_cost']}</th>"
        f"<th style='text-align:left;padding:4px 8px;'>{L['col_reuse']}</th>"
        f"<th style='text-align:left;padding:4px 8px;'>{L['col_started']}</th>"
        "</tr>",
    ]
    for run in valid:
        query = _escape(str(run.get("query", "—"))[:60])
        op_type = str(run.get("operation_type", "query"))
        op_label = _escape(OL.get(op_type, op_type.replace("_", " ").title()))
        status = str(run.get("status", "unknown"))
        status_color = _RUN_STATUS_COLORS.get(status, "#94a3b8")
        status_display = _escape(status)
        mode = _escape(str(run.get("research_mode", "—")))
        llm_calls = int(run.get("llm_request_count", 0)) if isinstance(run.get("llm_request_count"), (int, float)) else 0
        avoided = int(run.get("llm_calls_avoided", 0)) if isinstance(run.get("llm_calls_avoided"), (int, float)) else 0
        calls_str = f"{llm_calls}"
        if avoided > 0:
            calls_str += f" <span style='color:#22c55e;font-size:0.85em;'>({avoided} cached)</span>"
        cost = float(run.get("llm_estimated_cost_usd", 0)) if isinstance(run.get("llm_estimated_cost_usd"), (int, float)) else 0.0
        cost_str = f"${cost:.4f}" if cost > 0 else "—"
        reuse = run.get("reuse_rate")
        reuse_str = f"{float(reuse) * 100:.0f}%" if isinstance(reuse, (int, float)) and reuse is not None else "—"
        started = _escape(str(run.get("started_at", "?"))[:19])
        parts.append(
            f"<tr>"
            f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{query}</td>"
            f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{op_label}</td>"
            f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>"
            f"<span style='background:{status_color};color:white;padding:1px 8px;border-radius:8px;font-size:0.85em;'>{status_display}</span></td>"
            f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{mode}</td>"
            f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{calls_str}</td>"
            f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{cost_str}</td>"
            f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{reuse_str}</td>"
            f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{started}</td>"
            f"</tr>"
        )
    parts.append("</table>")
    return "\n".join(parts)


_DOMAIN_PROFILE_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Domain Profile — Legal",
        "empty": "No domain profile loaded.",
        "kernel_header": "Neutral Kernel Mappings",
        "trust_header": "Source Trust Weights",
        "roles_header": "Source Roles",
        "belief_header": "Belief States",
        "taint_header": "Taint Classes",
        "speech_header": "Speech Acts",
    },
    "finance": {
        "title": "Domain Profile — Finance",
        "empty": "No domain profile loaded.",
        "kernel_header": "Neutral Kernel Mappings",
        "trust_header": "Source Trust Weights",
        "roles_header": "Source Roles",
        "belief_header": "Belief States",
        "taint_header": "Taint Classes",
        "speech_header": "Speech Acts",
    },
    "coding": {
        "title": "Domain Profile — Coding",
        "empty": "No domain profile loaded.",
        "kernel_header": "Neutral Kernel Mappings",
        "trust_header": "Source Trust Weights",
        "roles_header": "Component Roles",
        "belief_header": "Belief States",
        "taint_header": "Sensitivity Classes",
        "speech_header": "Speech Acts",
    },
    "academic_research": {
        "title": "Domain Profile — Academic Research",
        "empty": "No domain profile loaded.",
        "kernel_header": "Neutral Kernel Mappings",
        "trust_header": "Source Trust Weights",
        "roles_header": "Source Roles",
        "belief_header": "Evidence States",
        "taint_header": "Access Classes",
        "speech_header": "Speech Acts",
    },
    "biomedical": {
        "title": "Domain Profile — Biomedical",
        "empty": "No domain profile loaded.",
        "kernel_header": "Neutral Kernel Mappings",
        "trust_header": "Source Trust Weights",
        "roles_header": "Evidence Sources",
        "belief_header": "Clinical States",
        "taint_header": "Sensitivity Classes",
        "speech_header": "Speech Acts",
    },
}


_DOCUMENT_CARD_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Document Card",
        "empty": "No document card has been profiled for this source yet.",
        "type": "Document Type",
        "source_side": "Source Side",
        "author": "Author",
        "sender": "Sender",
        "recipient": "Recipient",
        "purpose": "Purpose",
        "rhetorical": "Rhetorical Posture",
        "reliability": "Reliability",
        "operative": "Operative Status",
        "privilege": "Privilege",
        "flags": "Unresolved Flags",
        "source_role": "Source Role",
        "dates": "Key Dates",
        "privileged": "PRIVILEGED",
        "not_privileged": "Not privileged",
    },
    "finance": {
        "title": "Filing Card",
        "empty": "No filing card has been profiled for this document yet.",
        "type": "Document Type",
        "source_side": "Issuer / Counterparty",
        "author": "Preparer",
        "sender": "Sender",
        "recipient": "Recipient",
        "purpose": "Purpose",
        "rhetorical": "Presentation Posture",
        "reliability": "Reliability",
        "operative": "Effective Status",
        "privilege": "Confidentiality",
        "flags": "Unresolved Issues",
        "source_role": "Source Role",
        "dates": "Key Dates",
        "privileged": "CONFIDENTIAL",
        "not_privileged": "Not restricted",
    },
    "coding": {
        "title": "Artifact Card",
        "empty": "No artifact card has been profiled for this source yet.",
        "type": "Artifact Type",
        "source_side": "Component",
        "author": "Author",
        "sender": "Sender",
        "recipient": "Recipient",
        "purpose": "Purpose",
        "rhetorical": "Documentation Style",
        "reliability": "Reliability",
        "operative": "Current Status",
        "privilege": "Access Level",
        "flags": "Unresolved Issues",
        "source_role": "Source Role",
        "dates": "Key Dates",
        "privileged": "RESTRICTED",
        "not_privileged": "Public",
    },
    "academic_research": {
        "title": "Citation Card",
        "empty": "No citation card has been profiled for this source yet.",
        "type": "Source Type",
        "source_side": "Affiliation",
        "author": "Author(s)",
        "sender": "Sender",
        "recipient": "Recipient",
        "purpose": "Purpose",
        "rhetorical": "Argumentative Stance",
        "reliability": "Reliability",
        "operative": "Publication Status",
        "privilege": "Access",
        "flags": "Unresolved Issues",
        "source_role": "Source Role",
        "dates": "Key Dates",
        "privileged": "EMBARGOED",
        "not_privileged": "Open access",
    },
    "biomedical": {
        "title": "Clinical Record Card",
        "empty": "No clinical record card has been profiled for this source yet.",
        "type": "Record Type",
        "source_side": "Facility / Provider",
        "author": "Author",
        "sender": "Sender",
        "recipient": "Recipient",
        "purpose": "Purpose",
        "rhetorical": "Clinical Stance",
        "reliability": "Reliability",
        "operative": "Record Status",
        "privilege": "Patient Privacy",
        "flags": "Unresolved Issues",
        "source_role": "Source Role",
        "dates": "Key Dates",
        "privileged": "PROTECTED",
        "not_privileged": "Not restricted",
    },
}


def _fmt_document_card(data: dict, domain: str = "legal") -> str:
    L = _DOCUMENT_CARD_LABELS.get(domain, _DOCUMENT_CARD_LABELS["legal"])
    if not data or not isinstance(data, dict):
        return f"<div class='viz-empty'>{_escape(L['empty'])}</div>"
    if err := _error_html(data):
        return err
    card = data.get("card")
    if not card or not isinstance(card, dict):
        return f"<div class='viz-empty'>{_escape(L['empty'])}</div>"

    title = _escape(str(card.get("title", "") or ""))
    doc_type = _escape(str(card.get("doc_type", "") or ""))
    doc_subtype = _escape(str(card.get("doc_subtype", "") or ""))
    source_side = _escape(str(card.get("source_side", "") or ""))
    author = _escape(str(card.get("author", "") or ""))
    sender = _escape(str(card.get("sender", "") or ""))
    recipient = _escape(str(card.get("recipient", "") or ""))
    purpose = _escape(str(card.get("purpose", "") or ""))
    rhetorical = _escape(str(card.get("rhetorical_posture", "") or ""))
    reliability = _escape(str(card.get("reliability_posture", "") or ""))
    operative = _escape(str(card.get("operative_status", "") or ""))
    source_role = _escape(str(card.get("source_role", "") or ""))
    privilege = card.get("privilege_flag")

    parts: list[str] = []
    parts.append("<div style='margin-bottom:16px;'>")

    # Title + privilege badge
    priv_badge = ""
    if privilege:
        priv_badge = (
            f" <span style='display:inline-block;padding:2px 10px;background:#dc2626;"
            f"color:white;font-weight:700;font-size:11px;border-radius:4px;'>"
            f"{_escape(L['privileged'])}</span>"
        )
    parts.append(
        f"<h3 style='margin:0 0 8px;'>{_escape(L['title'])}: "
        f"{title or '<em>untitled</em>'}{priv_badge}</h3>"
    )

    # Identity section
    identity_rows = []
    if doc_type:
        type_str = doc_type
        if doc_subtype:
            type_str += f" / {doc_subtype}"
        identity_rows.append((L["type"], type_str))
    if source_side:
        identity_rows.append((L["source_side"], source_side))
    if source_role:
        identity_rows.append((L["source_role"], source_role))
    if operative:
        op_color = "#059669" if operative.lower() in ("operative", "effective", "current") else "#d97706"
        identity_rows.append((L["operative"], f"<span style='color:{op_color};font-weight:600;'>{operative}</span>"))

    if identity_rows:
        parts.append(
            "<div style='margin-bottom:12px;padding:10px;background:#f0fdf4;"
            "border:1px solid #bbf7d0;border-radius:6px;'>"
        )
        for label, val in identity_rows:
            parts.append(
                f"<div style='font-size:12px;margin-bottom:2px;'>"
                f"<strong>{_escape(label)}:</strong> {val}</div>"
            )
        parts.append("</div>")

    # Parties section
    party_rows = []
    if author:
        party_rows.append((L["author"], author))
    if sender:
        party_rows.append((L["sender"], sender))
    if recipient:
        party_rows.append((L["recipient"], recipient))

    if party_rows:
        parts.append("<div style='margin-bottom:12px;'>")
        for label, val in party_rows:
            parts.append(
                f"<div style='font-size:12px;margin-bottom:2px;'>"
                f"<strong>{_escape(label)}:</strong> {val}</div>"
            )
        parts.append("</div>")

    # Dates section
    date_fields = [
        ("creation_date", "Created"),
        ("sent_date", "Sent"),
        ("effective_date", "Effective"),
        ("discovery_date", "Discovered"),
    ]
    date_parts = []
    for field, label in date_fields:
        val = card.get(field)
        if val:
            date_parts.append(f"{label}: {_escape(str(val)[:20])}")
    if date_parts:
        parts.append(
            f"<div style='font-size:12px;margin-bottom:8px;color:#6b7280;'>"
            f"<strong>{_escape(L['dates'])}:</strong> {' | '.join(date_parts)}</div>"
        )

    # Purpose and posture
    if purpose:
        parts.append(
            f"<div style='font-size:12px;margin-bottom:4px;'>"
            f"<strong>{_escape(L['purpose'])}:</strong> {purpose}</div>"
        )
    if rhetorical:
        parts.append(
            f"<div style='font-size:12px;margin-bottom:4px;'>"
            f"<strong>{_escape(L['rhetorical'])}:</strong> {rhetorical}</div>"
        )
    if reliability:
        rel_color = "#059669" if reliability.lower() in ("high", "authoritative") else "#d97706" if reliability.lower() in ("medium", "moderate") else "#6b7280"
        parts.append(
            f"<div style='font-size:12px;margin-bottom:4px;'>"
            f"<strong>{_escape(L['reliability'])}:</strong> "
            f"<span style='color:{rel_color};font-weight:600;'>{reliability}</span></div>"
        )

    # Privilege status
    if not privilege:
        parts.append(
            f"<div style='font-size:11px;color:#6b7280;margin-bottom:4px;'>"
            f"{_escape(L['privilege'])}: {_escape(L['not_privileged'])}</div>"
        )

    # Unresolved flags
    flags_raw = card.get("unresolved_flags")
    flags: list[str] = []
    if isinstance(flags_raw, list):
        flags = [_escape(str(f)[:80]) for f in flags_raw if isinstance(f, str)]
    elif isinstance(flags_raw, str):
        try:
            parsed = json.loads(flags_raw)
            if isinstance(parsed, list):
                flags = [_escape(str(f)[:80]) for f in parsed if isinstance(f, str)]
        except (ValueError, TypeError):
            pass

    if flags:
        flag_badges = " ".join(
            f"<span style='display:inline-block;padding:2px 8px;background:#fef3c7;"
            f"border:1px solid #fde68a;border-radius:4px;font-size:11px;"
            f"color:#92400e;margin:2px;'>{f}</span>"
            for f in flags[:8]
        )
        parts.append(
            f"<div style='margin-top:6px;'>"
            f"<strong style='font-size:12px;'>{_escape(L['flags'])}:</strong> {flag_badges}</div>"
        )

    parts.append("</div>")
    return "".join(parts)


_DOC_TRIAGE_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Document Triage Queue",
        "empty": "All documents have been profiled.",
        "header_path": "Document",
        "header_type": "Type",
        "header_size": "Size",
        "header_salience": "Salience",
        "header_status": "Ingest Status",
    },
    "finance": {
        "title": "Document Triage Queue",
        "empty": "All financial documents have been profiled.",
        "header_path": "Document",
        "header_type": "Type",
        "header_size": "Size",
        "header_salience": "Salience",
        "header_status": "Ingest Status",
    },
    "coding": {
        "title": "Artifact Triage Queue",
        "empty": "All code artifacts have been profiled.",
        "header_path": "Artifact",
        "header_type": "Type",
        "header_size": "Size",
        "header_salience": "Salience",
        "header_status": "Ingest Status",
    },
    "academic_research": {
        "title": "Document Triage Queue",
        "empty": "All research documents have been profiled.",
        "header_path": "Document",
        "header_type": "Type",
        "header_size": "Size",
        "header_salience": "Salience",
        "header_status": "Ingest Status",
    },
    "biomedical": {
        "title": "Document Triage Queue",
        "empty": "All biomedical documents have been profiled.",
        "header_path": "Document",
        "header_type": "Type",
        "header_size": "Size",
        "header_salience": "Salience",
        "header_status": "Ingest Status",
    },
}


def _fmt_doc_triage_panel(docs: list, domain: str = "legal") -> str:
    labels = _DOC_TRIAGE_LABELS.get(domain, _DOC_TRIAGE_LABELS["legal"])
    if not docs or not isinstance(docs, list):
        return f"<div class='viz-empty'>{labels['empty']}</div>"

    valid = [d for d in docs if isinstance(d, dict)]
    if not valid:
        return f"<div class='viz-empty'>{labels['empty']}</div>"

    def _fmt_size(b):
        if not isinstance(b, (int, float)) or b <= 0:
            return "—"
        if b < 1024:
            return f"{int(b)} B"
        if b < 1024 * 1024:
            return f"{b / 1024:.1f} KB"
        return f"{b / (1024 * 1024):.1f} MB"

    parts = [
        f"<h3 style='margin:0 0 8px 0;'>{_escape(labels['title'])}</h3>",
        f"<div style='color:#666;font-size:0.9em;margin-bottom:8px;'>{len(valid)} document{'s' if len(valid) != 1 else ''} pending profiling</div>",
        "<table style='border-collapse:collapse;width:100%;font-size:0.9em;'>",
        f"<tr style='background:#f1f5f9;'>"
        f"<th style='text-align:left;padding:4px 8px;'>{_escape(labels['header_path'])}</th>"
        f"<th style='text-align:left;padding:4px 8px;'>{_escape(labels['header_type'])}</th>"
        f"<th style='text-align:left;padding:4px 8px;'>{_escape(labels['header_size'])}</th>"
        f"<th style='text-align:left;padding:4px 8px;'>{_escape(labels['header_salience'])}</th>"
        f"<th style='text-align:left;padding:4px 8px;'>{_escape(labels['header_status'])}</th>"
        f"</tr>",
    ]
    for doc in valid:
        path = _escape(str(doc.get("relative_path", "—")))
        ftype = _escape(str(doc.get("file_type", "—")))
        size = _fmt_size(doc.get("size_bytes"))
        salience = float(doc.get("salience_score", 0)) if isinstance(doc.get("salience_score"), (int, float)) else 0.0
        sal_color = "#22c55e" if salience >= 0.7 else "#eab308" if salience >= 0.4 else "#94a3b8"
        ingest = _escape(str(doc.get("ingest_status", "—")))
        parts.append(
            f"<tr>"
            f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{path}</td>"
            f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{ftype}</td>"
            f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{size}</td>"
            f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>"
            f"<span style='color:{sal_color};font-weight:600;'>{salience:.2f}</span></td>"
            f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{ingest}</td>"
            f"</tr>"
        )
    parts.append("</table>")
    return "\n".join(parts)


_DOC_CONSOLE_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Document Review Console",
        "candidates": "Candidate facts",
        "verified": "Verified",
        "rejected": "Rejected",
        "linked_issues": "Linked Issues",
        "actors": "Actors & Roles",
        "no_card": "No document profile available yet.",
    },
    "finance": {
        "title": "Document Review Console",
        "candidates": "Candidate data points",
        "verified": "Verified",
        "rejected": "Rejected",
        "linked_issues": "Linked Positions",
        "actors": "Entities & Roles",
        "no_card": "No document profile available yet.",
    },
    "coding": {
        "title": "Artifact Review Console",
        "candidates": "Candidate findings",
        "verified": "Verified",
        "rejected": "Rejected",
        "linked_issues": "Linked Requirements",
        "actors": "Components & Roles",
        "no_card": "No artifact profile available yet.",
    },
    "academic_research": {
        "title": "Source Review Console",
        "candidates": "Candidate claims",
        "verified": "Verified",
        "rejected": "Rejected",
        "linked_issues": "Linked Claims",
        "actors": "Authors & Roles",
        "no_card": "No source profile available yet.",
    },
    "biomedical": {
        "title": "Record Review Console",
        "candidates": "Candidate findings",
        "verified": "Verified",
        "rejected": "Rejected",
        "linked_issues": "Linked Findings",
        "actors": "Providers & Roles",
        "no_card": "No record profile available yet.",
    },
}


def _fmt_document_console(data: dict, domain: str = "legal") -> str:
    if not data or not isinstance(data, dict):
        return "<div class='viz-empty'>No document data available.</div>"
    if err := _error_html(data):
        return err

    L = _DOC_CONSOLE_LABELS.get(domain, _DOC_CONSOLE_LABELS["legal"])
    ref = _escape(str(data.get("document_ref", "")))
    card = data.get("card", {})
    if not isinstance(card, dict):
        card = {}
    cand_count = int(data.get("candidate_count", 0))
    ver_count = int(data.get("verified_count", 0))
    rej_count = int(data.get("rejected_count", 0))
    total = cand_count + ver_count + rej_count
    candidates = data.get("candidates", [])
    linked_issues = data.get("linked_issues", [])
    actor_roles = data.get("actor_roles", [])

    parts = [
        f"<div style='margin-bottom:16px;'>",
        f"<h3 style='margin:0 0 8px;'>{_escape(L['title'])}: {ref}</h3>",
    ]

    # Card metadata
    if card:
        doc_type = _escape(str(card.get("doc_type", "unknown")))
        privilege = card.get("privilege_flag")
        priv_label = "Privileged" if privilege else "Not privileged"
        priv_color = "#dc2626" if privilege else "#059669"
        parties = _escape(str(card.get("parties_summary", "") or ""))
        date_range = _escape(str(card.get("date_range", "") or ""))

        parts.append(
            f"<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:12px;"
            f"padding:10px;background:#f9fafb;border-radius:8px;'>"
            f"<div><div style='font-size:11px;color:#6b7280;font-weight:600;'>Type</div>"
            f"<div style='font-size:13px;'>{doc_type}</div></div>"
            f"<div><div style='font-size:11px;color:#6b7280;font-weight:600;'>Privilege</div>"
            f"<div style='font-size:13px;color:{priv_color};font-weight:600;'>{_escape(priv_label)}</div></div>"
            f"<div><div style='font-size:11px;color:#6b7280;font-weight:600;'>Parties</div>"
            f"<div style='font-size:13px;'>{parties or '—'}</div></div>"
            f"<div><div style='font-size:11px;color:#6b7280;font-weight:600;'>Date Range</div>"
            f"<div style='font-size:13px;'>{date_range or '—'}</div></div>"
            f"</div>"
        )
    else:
        parts.append(f"<div style='color:#9ca3af;font-style:italic;margin-bottom:8px;'>{_escape(L['no_card'])}</div>")

    # Counts summary
    ver_pct = round(ver_count / total * 100) if total > 0 else 0
    parts.append(
        f"<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px;'>"
    )
    for label, val, color in [
        (L["candidates"], str(cand_count), "#f59e0b" if cand_count > 0 else "#059669"),
        (L["verified"], str(ver_count), "#059669"),
        (L["rejected"], str(rej_count), "#dc2626" if rej_count > 0 else "#6b7280"),
    ]:
        parts.append(
            f"<div style='text-align:center;padding:10px;background:#f9fafb;border-radius:8px;'>"
            f"<div style='font-size:22px;font-weight:700;color:{color};'>{_escape(val)}</div>"
            f"<div style='font-size:11px;color:#6b7280;font-weight:600;margin-top:2px;'>"
            f"{_escape(label)}</div></div>"
        )
    parts.append("</div>")

    # Progress bar
    parts.append(
        f"<div style='margin-bottom:16px;'>"
        f"<div style='font-size:12px;color:#6b7280;margin-bottom:4px;font-weight:600;'>"
        f"Review progress: {ver_pct}% verified</div>"
        f"<div style='width:100%;height:12px;background:#e5e7eb;border-radius:6px;'>"
        f"<div style='width:{ver_pct}%;height:100%;background:#059669;"
        f"border-radius:6px;transition:width 0.3s;'></div></div></div>"
    )

    # Linked issues
    if linked_issues:
        parts.append(
            f"<div style='margin-bottom:12px;'>"
            f"<div style='font-size:13px;font-weight:700;color:#1f2937;margin-bottom:6px;'>"
            f"{_escape(L['linked_issues'])} ({len(linked_issues)})</div>"
            f"<div style='display:flex;flex-wrap:wrap;gap:6px;'>"
        )
        for iss in linked_issues[:10]:
            if not isinstance(iss, dict):
                continue
            title = _escape(str(iss.get("title", ""))[:40])
            mat = float(iss.get("materiality", 0))
            mat_color = "#dc2626" if mat >= 0.7 else "#f59e0b" if mat >= 0.4 else "#6b7280"
            parts.append(
                f"<span style='display:inline-block;padding:3px 10px;border-radius:10px;"
                f"background:#f3f4f6;font-size:12px;border-left:3px solid {mat_color};'>"
                f"{title}</span>"
            )
        parts.append("</div></div>")

    # Actor roles
    if actor_roles:
        parts.append(
            f"<div style='margin-bottom:12px;'>"
            f"<div style='font-size:13px;font-weight:700;color:#1f2937;margin-bottom:6px;'>"
            f"{_escape(L['actors'])} ({len(actor_roles)})</div>"
            f"<div style='display:flex;flex-wrap:wrap;gap:6px;'>"
        )
        for ar in actor_roles[:8]:
            if not isinstance(ar, dict):
                continue
            name = _escape(str(ar.get("actor_name", "")))
            role = _escape(str(ar.get("role", "")))
            parts.append(
                f"<span style='display:inline-block;padding:3px 10px;border-radius:10px;"
                f"background:#ede9fe;font-size:12px;'>"
                f"<strong>{name}</strong> — {role}</span>"
            )
        parts.append("</div></div>")

    # Candidate facts table
    if candidates:
        parts.append(
            f"<div style='margin-bottom:8px;'>"
            f"<div style='font-size:13px;font-weight:700;color:#1f2937;margin-bottom:6px;'>"
            f"{_escape(L['candidates'])} ({cand_count})</div>"
            f"<div style='max-height:300px;overflow-y:auto;'>"
            f"<table style='width:100%;border-collapse:collapse;font-size:12px;'>"
            f"<tr style='background:#f8fafc;'>"
            f"<th style='padding:4px 8px;text-align:left;'>Proposition</th>"
            f"<th style='padding:4px 8px;text-align:left;'>Belief State</th>"
            f"<th style='padding:4px 8px;text-align:left;'>Confidence</th>"
            f"</tr>"
        )
        for c in candidates[:30]:
            if not isinstance(c, dict):
                continue
            prop = _escape(str(c.get("proposition_text", ""))[:80])
            bs = _escape(str(c.get("belief_state", "")))
            conf = _safe_float(c.get("confidence", 0))
            conf_color = "#059669" if conf >= 0.7 else "#f59e0b" if conf >= 0.4 else "#dc2626"
            parts.append(
                f"<tr>"
                f"<td style='padding:4px 8px;border-bottom:1px solid #e5e7eb;'>{prop}</td>"
                f"<td style='padding:4px 8px;border-bottom:1px solid #e5e7eb;'>{bs}</td>"
                f"<td style='padding:4px 8px;border-bottom:1px solid #e5e7eb;"
                f"color:{conf_color};font-weight:600;'>{conf:.2f}</td>"
                f"</tr>"
            )
        if cand_count > 30:
            parts.append(
                f"<tr><td colspan='3' style='padding:4px 8px;color:#9ca3af;'>"
                f"+{cand_count - 30} more (use bulk verify to approve all)</td></tr>"
            )
        parts.append("</table></div></div>")

    parts.append("</div>")
    return "".join(parts)


_TAINT_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Sensitivity & Taint Summary",
        "empty": "No taint records found — all objects are clean.",
        "class_header": "By Sensitivity Class",
        "kind_header": "By Object Kind",
        "recent_header": "Recent Taint Events",
    },
    "finance": {
        "title": "Sensitivity & Taint Summary",
        "empty": "No taint records found — all objects are clean.",
        "class_header": "By Sensitivity Class",
        "kind_header": "By Object Kind",
        "recent_header": "Recent Taint Events",
    },
    "coding": {
        "title": "Sensitivity & Taint Summary",
        "empty": "No sensitivity records found — all artifacts are clean.",
        "class_header": "By Sensitivity Class",
        "kind_header": "By Artifact Kind",
        "recent_header": "Recent Sensitivity Events",
    },
    "academic_research": {
        "title": "Sensitivity & Access Summary",
        "empty": "No access restrictions found — all documents are clean.",
        "class_header": "By Access Class",
        "kind_header": "By Object Kind",
        "recent_header": "Recent Access Events",
    },
    "biomedical": {
        "title": "Sensitivity & Taint Summary",
        "empty": "No sensitivity records found — all artifacts are clean.",
        "class_header": "By Sensitivity Class",
        "kind_header": "By Artifact Kind",
        "recent_header": "Recent Sensitivity Events",
    },
}

_TAINT_CLASS_COLORS: dict[str, str] = {
    "public_clean": "#22c55e",
    "clean_with_withheld": "#86efac",
    "internal_work_product": "#eab308",
    "privileged": "#f97316",
    "sealed_privileged": "#ef4444",
    "material_nonpublic": "#ef4444",
    "confidential_counterparty": "#ef4444",
    "phi": "#dc2626",
    "clinical_trial_confidential": "#dc2626",
    "regulatory_confidential": "#f97316",
    "security_sensitive": "#f97316",
    "secret_or_credential": "#dc2626",
    "embargoed_research": "#f97316",
    "human_subjects_sensitive": "#dc2626",
    "license_restricted": "#eab308",
    "unknown_taint": "#94a3b8",
}


def _fmt_taint_summary_panel(data: dict, domain: str = "legal") -> str:
    labels = _TAINT_LABELS.get(domain, _TAINT_LABELS["legal"])
    if not data or not isinstance(data, dict):
        return f"<div class='viz-empty'>{labels['empty']}</div>"
    if err := _error_html(data):
        return err

    total = int(data.get("total", 0)) if isinstance(data.get("total"), (int, float)) else 0
    if total == 0:
        return f"<div class='viz-empty'>{labels['empty']}</div>"

    parts = [
        f"<h3 style='margin:0 0 8px 0;'>{_escape(labels['title'])}</h3>",
        f"<div style='color:#666;font-size:0.9em;margin-bottom:8px;'>{total} taint record{'s' if total != 1 else ''} across matter</div>",
    ]

    by_class = data.get("by_class", [])
    if isinstance(by_class, list) and by_class:
        parts.append(f"<h4 style='margin:12px 0 4px 0;'>{_escape(labels['class_header'])}</h4>")
        parts.append("<table style='border-collapse:collapse;width:100%;font-size:0.9em;'>")
        parts.append("<tr style='background:#f1f5f9;'><th style='text-align:left;padding:4px 8px;'>Class</th><th style='text-align:left;padding:4px 8px;'>Count</th><th style='text-align:left;padding:4px 8px;'>Bar</th></tr>")
        max_cnt = max((int(r.get("count", 0)) for r in by_class if isinstance(r, dict)), default=1) or 1
        for row in by_class:
            if not isinstance(row, dict):
                continue
            tc = _escape(str(row.get("taint_class", "?")))
            cnt = int(row.get("count", 0)) if isinstance(row.get("count"), (int, float)) else 0
            color = _TAINT_CLASS_COLORS.get(str(row.get("taint_class", "")), "#94a3b8")
            bar = int((cnt / max_cnt) * 100)
            parts.append(
                f"<tr><td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>"
                f"<span style='display:inline-block;background:{color};color:white;padding:1px 8px;border-radius:8px;font-size:0.85em;'>{tc}</span></td>"
                f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{cnt}</td>"
                f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>"
                f"<div style='background:#e2e8f0;border-radius:4px;height:12px;width:100px;'>"
                f"<div style='background:{color};border-radius:4px;height:12px;width:{bar}px;'></div></div></td></tr>"
            )
        parts.append("</table>")

    by_kind = data.get("by_kind", [])
    if isinstance(by_kind, list) and by_kind:
        parts.append(f"<h4 style='margin:12px 0 4px 0;'>{_escape(labels['kind_header'])}</h4>")
        pills = []
        for row in by_kind:
            if not isinstance(row, dict):
                continue
            kind = _escape(str(row.get("target_kind", "?")))
            cnt = int(row.get("count", 0)) if isinstance(row.get("count"), (int, float)) else 0
            pills.append(f"<span style='display:inline-block;background:#e2e8f0;padding:2px 10px;border-radius:10px;margin:2px 4px;font-size:0.85em;'>{kind}: {cnt}</span>")
        parts.append("<div style='margin-bottom:8px;'>" + "".join(pills) + "</div>")

    recent = data.get("recent", [])
    if isinstance(recent, list) and recent:
        valid_recent = [r for r in recent if isinstance(r, dict)]
        if valid_recent:
            parts.append(f"<h4 style='margin:12px 0 4px 0;'>{_escape(labels['recent_header'])}</h4>")
            parts.append("<table style='border-collapse:collapse;width:100%;font-size:0.85em;'>")
            parts.append("<tr style='background:#f1f5f9;'><th style='text-align:left;padding:4px 8px;'>Object</th><th style='text-align:left;padding:4px 8px;'>Class</th><th style='text-align:left;padding:4px 8px;'>Reason</th><th style='text-align:left;padding:4px 8px;'>When</th></tr>")
            for row in valid_recent[:20]:
                kind = _escape(str(row.get("target_kind", "?")))
                tid = _escape(str(row.get("target_id", "?"))[:40])
                tc = _escape(str(row.get("taint_class", "?")))
                reason = _escape(str(row.get("derivation_reason", "—"))[:60])
                when = _escape(str(row.get("created_at", "?"))[:19])
                color = _TAINT_CLASS_COLORS.get(str(row.get("taint_class", "")), "#94a3b8")
                parts.append(
                    f"<tr><td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{kind}/{tid}</td>"
                    f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>"
                    f"<span style='background:{color};color:white;padding:1px 6px;border-radius:6px;font-size:0.85em;'>{tc}</span></td>"
                    f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{reason}</td>"
                    f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{when}</td></tr>"
                )
            parts.append("</table>")

    return "\n".join(parts)


_SENSITIVITY_REVIEW_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Sensitivity Reclassification Review",
        "subtitle": "Review and reclassify document privilege flags to control downstream evidence handling.",
        "doc_id": "Document ID",
        "action": "Reclassify",
        "stale_action": "Mark Stale",
        "privilege_flag": "Privileged",
        "non_privilege": "Non-Privileged",
        "stale_reason": "Stale reason",
        "result_header": "Result",
        "staled_count": "Dependents staled",
        "empty": "No reclassification actions taken yet.",
    },
    "finance": {
        "title": "MNPI Reclassification Review",
        "subtitle": "Review and reclassify material non-public information flags on artifacts.",
        "doc_id": "Artifact ID",
        "action": "Reclassify",
        "stale_action": "Mark Stale",
        "privilege_flag": "Restricted",
        "non_privilege": "Public",
        "stale_reason": "Stale reason",
        "result_header": "Result",
        "staled_count": "Dependents staled",
        "empty": "No reclassification actions taken yet.",
    },
    "coding": {
        "title": "Security Classification Review",
        "subtitle": "Review and reclassify security sensitivity flags on artifacts.",
        "doc_id": "Artifact ID",
        "action": "Reclassify",
        "stale_action": "Mark Stale",
        "privilege_flag": "Secret/Internal",
        "non_privilege": "Public",
        "stale_reason": "Stale reason",
        "result_header": "Result",
        "staled_count": "Dependents staled",
        "empty": "No reclassification actions taken yet.",
    },
    "academic_research": {
        "title": "Access Reclassification Review",
        "subtitle": "Review and reclassify embargo or confidentiality flags on documents.",
        "doc_id": "Document ID",
        "action": "Reclassify",
        "stale_action": "Mark Stale",
        "privilege_flag": "Embargoed",
        "non_privilege": "Public",
        "stale_reason": "Stale reason",
        "result_header": "Result",
        "staled_count": "Dependents staled",
        "empty": "No reclassification actions taken yet.",
    },
    "biomedical": {
        "title": "PHI Reclassification Review",
        "subtitle": "Review and reclassify protected health information flags on artifacts.",
        "doc_id": "Artifact ID",
        "action": "Reclassify",
        "stale_action": "Mark Stale",
        "privilege_flag": "PHI/Restricted",
        "non_privilege": "Public",
        "stale_reason": "Stale reason",
        "result_header": "Result",
        "staled_count": "Dependents staled",
        "empty": "No reclassification actions taken yet.",
    },
}


def _fmt_sensitivity_review_result(data: dict, domain: str = "legal") -> str:
    labels = _SENSITIVITY_REVIEW_LABELS.get(domain, _SENSITIVITY_REVIEW_LABELS["legal"])
    if not data or not isinstance(data, dict):
        return f"<div class='viz-empty'>{labels['empty']}</div>"
    if err := _error_html(data):
        return err

    parts = [
        f"<h3 style='margin:0 0 8px 0;'>{_escape(labels['result_header'])}</h3>",
    ]

    doc_id = data.get("doc_id") or data.get("span_id", "")
    staled = data.get("staled_count", 0)
    if not isinstance(staled, (int, float)) or not math.isfinite(float(staled)):
        staled = 0

    parts.append(
        f"<div style='padding:8px;background:#f0fdf4;border:1px solid #bbf7d0;border-radius:6px;'>"
        f"<strong>{_escape(labels['doc_id'])}:</strong> {_escape(str(doc_id))} · "
        f"<strong>{_escape(labels['staled_count'])}:</strong> {int(staled)}"
        f"</div>"
    )

    return "\n".join(parts)


def _fmt_domain_profile_panel(summary: dict, domain: str = "legal") -> str:
    labels = _DOMAIN_PROFILE_LABELS.get(domain, _DOMAIN_PROFILE_LABELS["legal"])
    if not summary or not isinstance(summary, dict):
        return f"<div class='viz-empty'>{labels['empty']}</div>"

    status = summary.get("status", "unknown")
    if status == "not_found":
        return f"<div class='viz-empty'>{labels['empty']}</div>"

    pid = _escape(summary.get("profile_id", "unknown"))
    pver = int(summary.get("profile_version", 1)) if isinstance(summary.get("profile_version"), (int, float)) else 1
    pkind = _escape(summary.get("profile_kind", pid))
    is_primary = summary.get("is_primary", False)
    is_fallback = summary.get("is_fallback", False)

    primary_badge = "<span style='background:#2563eb;color:white;padding:2px 8px;border-radius:10px;font-size:0.8em;margin-left:8px;'>PRIMARY</span>" if is_primary else ""

    parts = []
    if is_fallback and pid == "legal":
        parts.append(
            "<div style='background:#fef3c7;border:1px solid #f59e0b;border-radius:8px;"
            "padding:10px 14px;margin-bottom:12px;'>"
            "<div style='font-weight:700;font-size:13px;color:#92400e;margin-bottom:4px;'>"
            "&#9888; Domain profile fallback active</div>"
            "<div style='font-size:12px;color:#78350f;'>"
            "No domain composition detected — using legal profile as default. "
            "Synthesis, trust weights, and issue classification may use legal-specific "
            "terminology. Run an investigation with domain-appropriate documents to "
            "auto-detect the correct profile.</div></div>"
        )
    parts.extend([
        f"<div style='margin-bottom:16px;'>",
        f"<h3 style='margin:0 0 4px 0;'>{_escape(labels['title'])}{primary_badge}</h3>",
        f"<div style='color:#666;font-size:0.9em;'>Profile: <b>{pid}</b> v{pver} · Kind: <b>{pkind}</b> · Status: <b>{_escape(status)}</b></div>",
        f"</div>",
    ])

    kernel = summary.get("neutral_kernel", {})
    if isinstance(kernel, dict) and kernel:
        parts.append(f"<h4 style='margin:12px 0 4px 0;'>{_escape(labels['kernel_header'])}</h4>")
        parts.append("<table style='border-collapse:collapse;width:100%;font-size:0.9em;'>")
        parts.append("<tr style='background:#f1f5f9;'><th style='text-align:left;padding:4px 8px;'>Canonical</th><th style='text-align:left;padding:4px 8px;'>Domain Term</th></tr>")
        for canonical, domain_term in sorted(kernel.items()):
            parts.append(f"<tr><td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{_escape(str(canonical))}</td>"
                         f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{_escape(str(domain_term))}</td></tr>")
        parts.append("</table>")

    tw = summary.get("composed_trust_weights", {})
    if isinstance(tw, dict) and tw:
        parts.append(f"<h4 style='margin:12px 0 4px 0;'>{_escape(labels['trust_header'])}</h4>")
        parts.append("<table style='border-collapse:collapse;width:100%;font-size:0.9em;'>")
        parts.append("<tr style='background:#f1f5f9;'><th style='text-align:left;padding:4px 8px;'>Source Role</th><th style='text-align:left;padding:4px 8px;'>Weight</th><th style='text-align:left;padding:4px 8px;'>Bar</th></tr>")
        for role, weight in sorted(tw.items(), key=lambda x: -float(x[1]) if isinstance(x[1], (int, float)) else 0):
            w = float(weight) if isinstance(weight, (int, float)) else 0.0
            bar_width = int(w * 100)
            color = "#22c55e" if w >= 0.7 else "#eab308" if w >= 0.5 else "#ef4444"
            parts.append(
                f"<tr><td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{_escape(str(role))}</td>"
                f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{w:.2f}</td>"
                f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>"
                f"<div style='background:#e2e8f0;border-radius:4px;height:12px;width:100px;'>"
                f"<div style='background:{color};border-radius:4px;height:12px;width:{bar_width}px;'></div>"
                f"</div></td></tr>"
            )
        parts.append("</table>")

    for field, header_key in [
        ("source_roles", "roles_header"),
        ("belief_states", "belief_header"),
        ("taint_classes", "taint_header"),
        ("speech_acts", "speech_header"),
    ]:
        items = summary.get(field, [])
        if isinstance(items, list) and items:
            parts.append(f"<h4 style='margin:12px 0 4px 0;'>{_escape(labels[header_key])}</h4>")
            pills = []
            for item in items:
                if not isinstance(item, str):
                    continue
                display = _escape(item.replace("_", " ").title())
                pills.append(f"<span style='display:inline-block;background:#e2e8f0;padding:2px 10px;border-radius:10px;margin:2px 4px 2px 0;font-size:0.85em;'>{display}</span>")
            parts.append("<div style='margin-bottom:8px;'>" + "".join(pills) + "</div>")

    facets = summary.get("facets", [])
    if isinstance(facets, list) and facets:
        parts.append("<h4 style='margin:12px 0 4px 0;'>Domain Facets</h4>")
        parts.append("<table style='border-collapse:collapse;width:100%;font-size:0.9em;'>")
        parts.append("<tr style='background:#f1f5f9;'><th style='text-align:left;padding:4px 8px;'>Profile</th><th style='text-align:left;padding:4px 8px;'>Version</th><th style='text-align:left;padding:4px 8px;'>Confidence</th><th style='text-align:left;padding:4px 8px;'>Status</th></tr>")
        for facet in facets:
            if not isinstance(facet, dict):
                continue
            fpid = _escape(str(facet.get("domain_profile_id", "?")))
            fver = _escape(str(facet.get("domain_profile_version", "?")))
            fconf = float(facet.get("confidence", 0)) if isinstance(facet.get("confidence"), (int, float)) else 0.0
            fstatus = _escape(str(facet.get("status", "?")))
            parts.append(
                f"<tr><td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{fpid}</td>"
                f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{fver}</td>"
                f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{fconf:.2f}</td>"
                f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{fstatus}</td></tr>"
            )
        parts.append("</table>")

    return "\n".join(parts)


_DOMAIN_COMPOSITION_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Domain Composition",
        "empty": "No domain composition data available yet.",
        "primary": "Primary Domain",
        "facets": "Active Facets",
        "trust_weights": "Composed Trust Weights",
        "events": "Recent Detection Events",
        "no_events": "No detection events recorded.",
    },
    "finance": {
        "title": "Domain Composition",
        "empty": "No domain composition data available yet.",
        "primary": "Primary Domain",
        "facets": "Active Facets",
        "trust_weights": "Composed Reliability Weights",
        "events": "Recent Detection Events",
        "no_events": "No detection events recorded.",
    },
    "coding": {
        "title": "Domain Composition",
        "empty": "No domain composition data available yet.",
        "primary": "Primary Domain",
        "facets": "Active Facets",
        "trust_weights": "Composed Confidence Weights",
        "events": "Recent Detection Events",
        "no_events": "No detection events recorded.",
    },
    "academic_research": {
        "title": "Domain Composition",
        "empty": "No domain composition data available yet.",
        "primary": "Primary Domain",
        "facets": "Active Facets",
        "trust_weights": "Composed Authority Weights",
        "events": "Recent Detection Events",
        "no_events": "No detection events recorded.",
    },
    "biomedical": {
        "title": "Domain Composition",
        "empty": "No domain composition data available yet.",
        "primary": "Primary Domain",
        "facets": "Active Facets",
        "trust_weights": "Composed Evidence Weights",
        "events": "Recent Detection Events",
        "no_events": "No detection events recorded.",
    },
}


def _fmt_domain_composition_panel(data: dict, domain: str = "legal") -> str:
    L = _DOMAIN_COMPOSITION_LABELS.get(domain, _DOMAIN_COMPOSITION_LABELS["legal"])
    if not data or not isinstance(data, dict):
        return f"<div class='viz-empty'>{L['empty']}</div>"
    if err := _error_html(data):
        return err

    primary = _escape(str(data.get("primary_domain_profile_id") or "unknown"))
    facets = data.get("facets") or {}
    tw = data.get("composed_trust_weights") or {}
    events = data.get("detection_events") or []

    primary_badge = (
        f"<span style='display:inline-block;padding:2px 10px;border-radius:8px;"
        f"background:#2563eb22;color:#2563eb;font-size:13px;font-weight:600;'>"
        f"{primary}</span>"
    )

    facet_pills = ""
    if isinstance(facets, dict) and facets:
        pills = []
        for facet_name, facet_val in facets.items():
            if not isinstance(facet_name, str):
                continue
            weight = float(facet_val) if isinstance(facet_val, (int, float)) else 0.0
            opacity = max(0.3, min(1.0, weight))
            pills.append(
                f"<span style='display:inline-block;padding:2px 8px;border-radius:6px;"
                f"background:#6366f122;color:#6366f1;font-size:12px;margin:2px;"
                f"opacity:{opacity:.2f};'>{_escape(facet_name)}"
                f" <span style='font-size:10px;opacity:0.7'>{weight:.0%}</span></span>"
            )
        facet_pills = " ".join(pills)
    else:
        facet_pills = "<span style='color:#9ca3af;font-size:12px;'>No facets detected</span>"

    tw_rows = ""
    if isinstance(tw, dict) and tw:
        for role, weight in sorted(tw.items(), key=lambda x: float(x[1]) if isinstance(x[1], (int, float)) else 0, reverse=True):
            w = float(weight) if isinstance(weight, (int, float)) else 0.0
            bar_pct = max(0, min(100, w * 100))
            tw_rows += (
                f"<tr><td style='padding:3px 8px;font-size:12px;'>{_escape(str(role))}</td>"
                f"<td style='padding:3px 8px;width:60%;'>"
                f"<div style='background:#f3f4f6;height:14px;border-radius:4px;overflow:hidden;'>"
                f"<div style='background:#6366f1;height:100%;width:{bar_pct:.1f}%;border-radius:4px;'></div>"
                f"</div></td>"
                f"<td style='padding:3px 8px;font-size:11px;color:#6b7280;text-align:right;'>{w:.2f}</td></tr>"
            )

    event_rows = ""
    for evt in events[:20]:
        if not isinstance(evt, dict):
            continue
        profile = _escape(str(evt.get("candidate_profile_id") or "?"))
        _raw_conf = evt.get("confidence")
        conf = float(_raw_conf) if isinstance(_raw_conf, (int, float)) else 0.0
        target = _escape(f"{evt.get('target_kind', '?')}:{str(evt.get('target_id', '?'))[:16]}")
        ts = _escape(str(evt.get("created_at") or "")[:19])
        conf_color = "#16a34a" if conf >= 0.7 else ("#eab308" if conf >= 0.4 else "#6b7280")
        event_rows += (
            f"<tr><td style='padding:3px 8px;font-size:11px;'>{ts}</td>"
            f"<td style='padding:3px 8px;font-size:11px;'>{profile}</td>"
            f"<td style='padding:3px 8px;'>"
            f"<span style='color:{conf_color};font-weight:600;font-size:11px;'>{conf:.0%}</span></td>"
            f"<td style='padding:3px 8px;font-size:10px;color:#6b7280;font-family:monospace;'>{target}</td></tr>"
        )

    parts = [
        "<div class='viz-shell'>",
        f"<div class='viz-header'><strong>{L['title']}</strong></div>",
        f"<div style='padding:8px 12px;'>"
        f"<div style='margin-bottom:12px;'><strong>{L['primary']}:</strong> {primary_badge}</div>"
        f"<div style='margin-bottom:12px;'><strong>{L['facets']}:</strong><div style='margin-top:4px;'>{facet_pills}</div></div>",
    ]

    if tw_rows:
        parts.append(
            f"<div style='margin-bottom:12px;'><strong>{L['trust_weights']}:</strong>"
            "<div class='table-wrap'><table class='viz-table'>"
            "<thead><tr><th>Role</th><th>Weight</th><th></th></tr></thead>"
            f"<tbody>{tw_rows}</tbody></table></div></div>"
        )

    if event_rows:
        parts.append(
            f"<div><strong>{L['events']}:</strong>"
            "<div class='table-wrap'><table class='viz-table'>"
            "<thead><tr><th>Time</th><th>Profile</th><th>Confidence</th><th>Target</th></tr></thead>"
            f"<tbody>{event_rows}</tbody></table></div></div>"
        )
    elif not events:
        parts.append(f"<div style='color:#9ca3af;font-size:12px;'>{L['no_events']}</div>")

    parts.append("</div></div>")
    return "\n".join(parts)


def _fmt_so_scorecard_panel(so: dict, domain: str = "legal") -> str:
    labels = _SO_SCORECARD_LABELS.get(domain, _SO_SCORECARD_LABELS["legal"])
    if not so or not isinstance(so, dict):
        return "<div class='viz-empty'>No scorecard data available yet.</div>"

    targets = so.get("targets", {})
    met = so.get("targets_met", {})

    def _row(label: str, metric_key: str, is_bool: bool = False) -> str:
        val = so.get(metric_key)
        target = targets.get(metric_key)
        passed = met.get(metric_key)
        if val is None:
            val_str = "—"
        elif is_bool:
            val_str = "Yes" if val else "No"
        else:
            try:
                fval = float(val)
                val_str = f"{fval * 100:.1f}%" if math.isfinite(fval) else "—"
            except (TypeError, ValueError):
                val_str = "—"
        if target is None:
            tgt_str = "—"
        elif isinstance(target, bool):
            tgt_str = "Yes" if target else "No"
        else:
            try:
                tval = float(target)
                tgt_str = f"{tval * 100:.0f}%" if math.isfinite(tval) else "—"
            except (TypeError, ValueError):
                tgt_str = "—"
        if passed is True:
            pill = "<span class='pill pill-green'>Pass</span>"
        elif passed is False:
            pill = "<span class='pill pill-red'>Fail</span>"
        else:
            pill = "<span class='pill pill-neutral'>N/A</span>"
        return f"<tr><td>{_escape(label)}</td><td>{val_str}</td><td>{tgt_str}</td><td>{pill}</td></tr>"

    rows = (
        _row(labels["so1"], "reuse_rate")
        + _row(labels["so2_struct"], "assertion_structure_rate")
        + _row(labels["so2_revision"], "belief_revision", is_bool=True)
        + _row(labels["so2_provenance"], "provenance_attribution_rate")
        + _row(labels["so3"], "steerability", is_bool=True)
        + _row(labels["so4"], "issue_coverage_avg")
        + _row(labels["so5"], "source_role_known_rate")
        + _row(labels["so6"], "numeric_extraction_rate")
        + _row(labels["so7"], "gap_surface_ratio")
    )

    passed_count = sum(1 for v in met.values() if v is True)
    total_count = sum(1 for v in met.values() if v is not None)
    score_pill = "pill-green" if passed_count == total_count and total_count > 0 else "pill-orange"

    header = (
        f"<div class='viz-header'><strong>{labels['title']}</strong>"
        f" — <span class='pill {score_pill}'>{passed_count}/{total_count} passing</span></div>"
        f"<div style='padding:4px 8px;font-size:0.9em;color:#666;'>{labels['subtitle']}</div>"
    )
    table = (
        "<div class='table-wrap'><table class='viz-table'>"
        "<thead><tr><th>Metric</th><th>Current</th><th>Target</th><th>Status</th></tr></thead>"
        "<tbody>" + rows + "</tbody></table></div>"
    )
    return f"<div class='viz-shell'>{header}{table}</div>"


_TRUST_LEVEL_PILLS = {
    "low": ("Low", "pill-red"),
    "normal": ("Normal", "pill-neutral"),
    "high": ("High", "pill-green"),
}


_TRUST_OVERRIDES_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "empty": "No trust overrides set.",
        "title": "Document Trust Overrides",
        "col_doc": "Document", "col_trust": "Trust", "col_reason": "Reason", "col_set": "Set",
    },
    "finance": {
        "empty": "No trust overrides set.",
        "title": "Source Trust Overrides",
        "col_doc": "Source", "col_trust": "Trust", "col_reason": "Reason", "col_set": "Set",
    },
    "coding": {
        "empty": "No trust overrides set.",
        "title": "Artifact Trust Overrides",
        "col_doc": "Artifact", "col_trust": "Trust", "col_reason": "Reason", "col_set": "Set",
    },
    "academic_research": {
        "empty": "No trust overrides set.",
        "title": "Source Trust Overrides",
        "col_doc": "Source", "col_trust": "Trust", "col_reason": "Reason", "col_set": "Set",
    },
    "biomedical": {
        "empty": "No trust overrides set.",
        "title": "Source Trust Overrides",
        "col_doc": "Source", "col_trust": "Trust", "col_reason": "Reason", "col_set": "Set",
    },
}


def _fmt_trust_overrides(overrides: list[dict], domain: str = "legal") -> str:
    L = _TRUST_OVERRIDES_LABELS.get(domain, _TRUST_OVERRIDES_LABELS["legal"])
    if not overrides:
        return f"<div class='viz-empty'>{L['empty']}</div>"
    rows = ""
    for ov in overrides:
        if not isinstance(ov, dict):
            continue
        pattern = _escape(str(ov.get("document_pattern", "—")))
        level = str(ov.get("trust_level", "normal"))
        label, cls = _TRUST_LEVEL_PILLS.get(level, ("Unknown", "pill-neutral"))
        note = _escape(str(ov.get("note") or "—")[:120])
        created = _escape(str(ov.get("created_at", "—"))[:19])
        rows += (
            f"<tr>"
            f"<td>{pattern}</td>"
            f"<td><span class='pill {cls}'>{label}</span></td>"
            f"<td>{note}</td>"
            f"<td style='font-size:11px;color:#6b7280'>{created}</td>"
            f"</tr>"
        )
    if not rows:
        return f"<div class='viz-empty'>{L['empty']}</div>"
    count = rows.count("<tr>")
    header = f"<div class='viz-header'><strong>{L['title']}</strong> — {count} active</div>"
    table = (
        "<div class='table-wrap'><table class='viz-table'>"
        f"<thead><tr><th>{L['col_doc']}</th><th>{L['col_trust']}</th><th>{L['col_reason']}</th><th>{L['col_set']}</th></tr></thead>"
        "<tbody>" + rows + "</tbody></table></div>"
    )
    return f"<div class='viz-shell'>{header}{table}</div>"


_DECISION_MAKER_LABELS = {
    "judge": "Judge", "partner": "Partner", "client": "Client",
    "mediator": "Mediator", "arbitrator": "Arbitrator",
    "regulator": "Regulator", "unknown": "Other",
}
_OBJECTIVE_LABELS = {
    "motion_practice": "Motion Practice", "settlement": "Settlement",
    "diligence": "Due Diligence", "audit": "Audit", "advisory": "Advisory",
    "trial_prep": "Trial Prep", "regulatory_response": "Regulatory Response",
    "transactional": "Transactional", "unknown": "Other",
}

_DOMAIN_DECISION_MAKER_CHOICES: dict[str, list[tuple[str, str]]] = {
    "legal": [
        ("Judge", "judge"), ("Partner", "partner"), ("Client", "client"),
        ("Mediator", "mediator"), ("Arbitrator", "arbitrator"),
        ("Regulator", "regulator"), ("Other", "unknown"),
    ],
    "finance": [
        ("Portfolio Manager", "portfolio_manager"), ("Risk Officer", "risk_officer"),
        ("Compliance Officer", "compliance_officer"), ("Auditor", "auditor"),
        ("Board Member", "board_member"), ("Regulator", "regulator"),
        ("Investor", "investor"), ("Other", "unknown"),
    ],
    "coding": [
        ("Tech Lead", "tech_lead"), ("Product Manager", "product_manager"),
        ("DevOps Engineer", "devops"), ("Security Engineer", "security"),
        ("QA Engineer", "qa"), ("Architect", "architect"), ("Other", "unknown"),
    ],
    "academic_research": [
        ("Principal Investigator", "pi"), ("Co-Investigator", "co_investigator"),
        ("Peer Reviewer", "peer_reviewer"), ("Funding Agency", "funding_agency"),
        ("Ethics Board", "ethics_board"), ("Department Head", "department_head"),
        ("Other", "unknown"),
    ],
    "biomedical": [
        ("Clinician", "clinician"), ("Trial Investigator", "trial_investigator"),
        ("IRB / Ethics Board", "irb"), ("Regulator", "regulator"),
        ("Patient Advocate", "patient_advocate"), ("Sponsor", "sponsor"),
        ("Other", "unknown"),
    ],
}

_DOMAIN_OBJECTIVE_CHOICES: dict[str, list[tuple[str, str]]] = {
    "legal": [
        ("Motion Practice", "motion_practice"), ("Settlement", "settlement"),
        ("Due Diligence", "diligence"), ("Audit", "audit"),
        ("Advisory", "advisory"), ("Trial Prep", "trial_prep"),
        ("Regulatory Response", "regulatory_response"),
        ("Transactional", "transactional"), ("Other", "unknown"),
    ],
    "finance": [
        ("Due Diligence", "diligence"), ("Audit", "audit"),
        ("Compliance Review", "compliance_review"),
        ("Portfolio Analysis", "portfolio_analysis"),
        ("Risk Assessment", "risk_assessment"),
        ("Valuation", "valuation"), ("Regulatory Filing", "regulatory_filing"),
        ("Other", "unknown"),
    ],
    "coding": [
        ("Bug Triage", "bug_triage"), ("Code Review", "code_review"),
        ("Architecture Decision", "architecture_decision"),
        ("Release Readiness", "release_readiness"),
        ("Security Audit", "security_audit"),
        ("Performance Analysis", "performance_analysis"), ("Other", "unknown"),
    ],
    "academic_research": [
        ("Literature Review", "literature_review"),
        ("Methodology Audit", "methodology_audit"),
        ("Grant Writing", "grant_writing"), ("Replication Study", "replication"),
        ("Systematic Review", "systematic_review"),
        ("Ethics Review", "ethics_review"), ("Other", "unknown"),
    ],
    "biomedical": [
        ("Clinical Trial Assessment", "clinical_trial"),
        ("Drug Safety Review", "drug_safety"),
        ("Protocol Review", "protocol_review"),
        ("Regulatory Submission", "regulatory_submission"),
        ("Literature Synthesis", "literature_synthesis"),
        ("Diagnostic Workup", "diagnostic_workup"), ("Other", "unknown"),
    ],
}


def _decision_maker_choices_for_domain(domain: str) -> list[tuple[str, str]]:
    return _DOMAIN_DECISION_MAKER_CHOICES.get(
        domain, _DOMAIN_DECISION_MAKER_CHOICES["legal"]
    )


def _objective_choices_for_domain(domain: str) -> list[tuple[str, str]]:
    return _DOMAIN_OBJECTIVE_CHOICES.get(
        domain, _DOMAIN_OBJECTIVE_CHOICES["legal"]
    )


def _fmt_decision_context(ctx: "dict | None", domain: str = "legal") -> str:
    if not ctx or not isinstance(ctx, dict):
        return "<div class='viz-empty'>No decision context set. Set one above to adjust how Irys frames its analysis.</div>"
    maker_choices = dict(
        (v, k) for k, v in _decision_maker_choices_for_domain(domain)
    )
    obj_choices = dict(
        (v, k) for k, v in _objective_choices_for_domain(domain)
    )
    maker_raw = ctx.get("decision_maker_type", "")
    obj_raw = ctx.get("objective", "")
    maker = maker_choices.get(maker_raw, _DECISION_MAKER_LABELS.get(maker_raw, maker_raw or "—"))
    obj = obj_choices.get(obj_raw, _OBJECTIVE_LABELS.get(obj_raw, obj_raw or "—"))
    name = _escape(ctx.get("decision_maker_name") or "—")
    notes = _escape(ctx.get("strategic_notes") or "—")
    narrow = "Yes" if ctx.get("scope_narrow") else "No"
    updated = _escape(str(ctx.get("updated_at", "—"))[:19])
    rows = (
        f"<tr><td><strong>Decision-maker</strong></td><td>{_escape(maker)}</td></tr>"
        f"<tr><td><strong>Name</strong></td><td>{name}</td></tr>"
        f"<tr><td><strong>Objective</strong></td><td>{_escape(obj)}</td></tr>"
        f"<tr><td><strong>Strategic notes</strong></td><td>{notes}</td></tr>"
        f"<tr><td><strong>Narrow scope</strong></td><td>{narrow}</td></tr>"
        f"<tr><td><strong>Last updated</strong></td><td style='font-size:11px;color:#6b7280'>{updated}</td></tr>"
    )
    header = "<div class='viz-header'><strong>Active Decision Context</strong></div>"
    table = f"<div class='table-wrap'><table class='viz-table'><tbody>{rows}</tbody></table></div>"
    return f"<div class='viz-shell'>{header}{table}</div>"


_ANNOTATION_TYPE_PILLS = {
    "strategic": ("Strategic", "pill-blue"),
    "reliability": ("Reliability", "pill-amber"),
    "scope": ("Scope", "pill-neutral"),
}


_ANNOTATIONS_PANEL_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "empty_cta": "No document notes yet. Add one below to guide the investigation.",
        "empty": "No document notes yet.",
        "title": "Document Notes",
        "col_doc": "Document", "col_type": "Type", "col_note": "Note", "col_added": "Added",
    },
    "finance": {
        "empty_cta": "No source notes yet. Add one below to guide the analysis.",
        "empty": "No source notes yet.",
        "title": "Source Notes",
        "col_doc": "Source", "col_type": "Type", "col_note": "Note", "col_added": "Added",
    },
    "coding": {
        "empty_cta": "No artifact notes yet. Add one below to guide the investigation.",
        "empty": "No artifact notes yet.",
        "title": "Artifact Notes",
        "col_doc": "Artifact", "col_type": "Type", "col_note": "Note", "col_added": "Added",
    },
    "academic_research": {
        "empty_cta": "No source notes yet. Add one below to guide the review.",
        "empty": "No source notes yet.",
        "title": "Source Notes",
        "col_doc": "Source", "col_type": "Type", "col_note": "Note", "col_added": "Added",
    },
    "biomedical": {
        "empty_cta": "No source notes yet. Add one below to guide the investigation.",
        "empty": "No source notes yet.",
        "title": "Source Notes",
        "col_doc": "Source", "col_type": "Type", "col_note": "Note", "col_added": "Added",
    },
}


def _fmt_annotations_panel(annotations: list[dict], domain: str = "legal") -> str:
    L = _ANNOTATIONS_PANEL_LABELS.get(domain, _ANNOTATIONS_PANEL_LABELS["legal"])
    if not annotations:
        return f"<div class='viz-empty'>{L['empty_cta']}</div>"
    rows = ""
    for a in annotations:
        if not isinstance(a, dict):
            continue
        doc = _escape(str(a.get("document_pattern", "—")))
        text = _escape(str(a.get("annotation_text", "—"))[:200])
        ann_type = str(a.get("annotation_type", "strategic"))
        label, cls = _ANNOTATION_TYPE_PILLS.get(ann_type, ("Note", "pill-neutral"))
        created = _escape(str(a.get("created_at", "—"))[:19])
        ann_id = _escape(str(a.get("annotation_id") or a.get("id") or "—")[:12])
        rows += (
            f"<tr>"
            f"<td>{doc}</td>"
            f"<td><span class='pill {cls}'>{label}</span></td>"
            f"<td>{text}</td>"
            f"<td style='font-size:11px;color:#6b7280'>{created}</td>"
            f"<td style='font-size:10px;color:#9ca3af;font-family:monospace'>{ann_id}</td>"
            f"</tr>"
        )
    if not rows:
        return f"<div class='viz-empty'>{L['empty']}</div>"
    count = rows.count("<tr>")
    header = f"<div class='viz-header'><strong>{L['title']}</strong> — {count} note{'s' if count != 1 else ''}</div>"
    table = (
        "<div class='table-wrap'><table class='viz-table'>"
        f"<thead><tr><th>{L['col_doc']}</th><th>{L['col_type']}</th><th>{L['col_note']}</th><th>{L['col_added']}</th><th>ID</th></tr></thead>"
        "<tbody>" + rows + "</tbody></table></div>"
    )
    return f"<div class='viz-shell'>{header}{table}</div>"


_ACTOR_TYPE_LABELS: dict[str, dict[str, str]] = {
    "legal": {"title": "Potential Duplicate Parties", "empty": "No duplicate parties detected."},
    "finance": {"title": "Potential Duplicate Entities", "empty": "No duplicate entities detected."},
    "coding": {"title": "Potential Duplicate Contributors", "empty": "No duplicate contributors detected."},
    "academic_research": {"title": "Potential Duplicate Authors", "empty": "No duplicate authors detected."},
    "biomedical": {"title": "Potential Duplicate Subjects", "empty": "No duplicate subjects detected."},
}


def _fmt_duplicate_actors_panel(pairs: list[dict], domain: str = "legal") -> str:
    labels = _ACTOR_TYPE_LABELS.get(domain, _ACTOR_TYPE_LABELS["legal"])
    if not pairs:
        return f"<div class='viz-empty'>{labels['empty']}</div>"
    rows = ""
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        a = pair.get("actor_a", {}) or {}
        b = pair.get("actor_b", {}) or {}
        prefix = _escape(pair.get("shared_prefix", ""))
        rows += (
            "<tr>"
            f"<td><strong>{_escape(a.get('canonical_name', ''))}</strong>"
            f"<br><span class='dim'>{_escape(a.get('id', '')[:12])}</span></td>"
            f"<td><strong>{_escape(b.get('canonical_name', ''))}</strong>"
            f"<br><span class='dim'>{_escape(b.get('id', '')[:12])}</span></td>"
            f"<td>{prefix}</td>"
            f"<td>{_escape(a.get('actor_type', ''))}</td>"
            "</tr>"
        )
    if not rows:
        return f"<div class='viz-empty'>{labels['empty']}</div>"
    return (
        "<div class='viz-shell'>"
        f"<div class='viz-panel-title'>{labels['title']}</div>"
        "<div class='viz-footnote'>Copy the IDs below into the merge fields to consolidate duplicates.</div>"
        "<div class='matrix-wrap'><table class='analytics-table'><thead><tr>"
        "<th>Actor A</th><th>Actor B</th><th>Shared Prefix</th><th>Type</th>"
        "</tr></thead><tbody>"
        + rows
        + "</tbody></table></div></div>"
    )


_COMMUNICATION_MAP_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "empty": "No communication graph available yet.",
        "sparse": "Communication graph has no dense connections to render.",
        "title": "Actor/document communication map",
        "subtitle": "Actors on the left, documents on the right, edge width = co-occurrence count.",
        "footnote_slice": "The SVG highlights the densest actor/document slice. Full actor, document, and edge detail is listed below.",
        "actors": "Actors", "documents": "Documents",
        "pairs": "Strongest actor pairs",
        "edge_detail": "Actor/document edge detail",
        "col_actor": "Actor", "col_document": "Document", "col_mentions": "Mentions",
        "no_edges": "No actor/document links yet.",
        "no_pairs": "No repeated actor co-appearance detected yet.",
        "linked_docs": "linked docs", "mentions": "mentions", "links": "links", "actors_word": "actors",
    },
    "finance": {
        "empty": "No communication graph available yet.",
        "sparse": "Communication graph has no dense connections to render.",
        "title": "Entity/source communication map",
        "subtitle": "Entities on the left, sources on the right, edge width = co-occurrence count.",
        "footnote_slice": "The SVG highlights the densest entity/source slice. Full entity, source, and edge detail is listed below.",
        "actors": "Entities", "documents": "Sources",
        "pairs": "Strongest entity pairs",
        "edge_detail": "Entity/source edge detail",
        "col_actor": "Entity", "col_document": "Source", "col_mentions": "Mentions",
        "no_edges": "No entity/source links yet.",
        "no_pairs": "No repeated entity co-appearance detected yet.",
        "linked_docs": "linked sources", "mentions": "mentions", "links": "links", "actors_word": "entities",
    },
    "coding": {
        "empty": "No communication graph available yet.",
        "sparse": "Communication graph has no dense connections to render.",
        "title": "Component/artifact communication map",
        "subtitle": "Components on the left, artifacts on the right, edge width = co-occurrence count.",
        "footnote_slice": "The SVG highlights the densest component/artifact slice. Full detail is listed below.",
        "actors": "Components", "documents": "Artifacts",
        "pairs": "Strongest component pairs",
        "edge_detail": "Component/artifact edge detail",
        "col_actor": "Component", "col_document": "Artifact", "col_mentions": "Mentions",
        "no_edges": "No component/artifact links yet.",
        "no_pairs": "No repeated component co-appearance detected yet.",
        "linked_docs": "linked artifacts", "mentions": "mentions", "links": "links", "actors_word": "components",
    },
    "academic_research": {
        "empty": "No communication graph available yet.",
        "sparse": "Communication graph has no dense connections to render.",
        "title": "Author/source communication map",
        "subtitle": "Authors on the left, sources on the right, edge width = co-occurrence count.",
        "footnote_slice": "The SVG highlights the densest author/source slice. Full detail is listed below.",
        "actors": "Authors", "documents": "Sources",
        "pairs": "Strongest author pairs",
        "edge_detail": "Author/source edge detail",
        "col_actor": "Author", "col_document": "Source", "col_mentions": "Mentions",
        "no_edges": "No author/source links yet.",
        "no_pairs": "No repeated author co-appearance detected yet.",
        "linked_docs": "linked sources", "mentions": "mentions", "links": "links", "actors_word": "authors",
    },
    "biomedical": {
        "empty": "No communication graph available yet.",
        "sparse": "Communication graph has no dense connections to render.",
        "title": "Entity/source communication map",
        "subtitle": "Entities on the left, sources on the right, edge width = co-occurrence count.",
        "footnote_slice": "The SVG highlights the densest entity/source slice. Full detail is listed below.",
        "actors": "Entities", "documents": "Sources",
        "pairs": "Strongest entity pairs",
        "edge_detail": "Entity/source edge detail",
        "col_actor": "Entity", "col_document": "Source", "col_mentions": "Mentions",
        "no_edges": "No entity/source links yet.",
        "no_pairs": "No repeated entity co-appearance detected yet.",
        "linked_docs": "linked sources", "mentions": "mentions", "links": "links", "actors_word": "entities",
    },
}


def _fmt_communication_map_panel(graph: dict, domain: str = "legal") -> str:
    L = _COMMUNICATION_MAP_LABELS.get(domain, _COMMUNICATION_MAP_LABELS["legal"])
    actors = list(graph.get("actors", []) or [])
    documents = list(graph.get("documents", []) or [])
    edges = list(graph.get("actor_document_edges", []) or [])
    actor_actor_edges = list(graph.get("actor_actor_edges", []) or [])
    if not actors or not documents or not edges:
        return f"<div class='viz-empty'>{L['empty']}</div>"

    actor_weights: dict[str, int] = defaultdict(int)
    document_weights: dict[str, int] = defaultdict(int)
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        actor_id = edge.get("actor_id")
        document_id = edge.get("document_id")
        count = _safe_int(edge.get("occurrence_count", 0))
        if actor_id:
            actor_weights[actor_id] += count
        if document_id:
            document_weights[document_id] += count

    actor_lookup = {actor.get("id"): actor for actor in actors if isinstance(actor, dict)}
    actor_ids = sorted(actor_weights, key=lambda key: actor_weights[key], reverse=True)[:12]
    doc_ids = sorted(document_weights, key=lambda key: document_weights[key], reverse=True)[:14]
    actor_index = {actor_id: idx for idx, actor_id in enumerate(actor_ids)}
    doc_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    filtered_edges = [
        edge
        for edge in edges
        if isinstance(edge, dict) and edge.get("actor_id") in actor_index and edge.get("document_id") in doc_index
    ]
    if not filtered_edges:
        return f"<div class='viz-empty'>{L['sparse']}</div>"

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
        if not isinstance(edge, dict):
            continue
        actor_id = edge.get("actor_id")
        document_id = edge.get("document_id")
        if actor_id and document_id:
            actor_doc_counts[actor_id] += 1
            document_actor_counts[document_id].add(actor_id)

    actor_rows = "".join(
        "<div class='viz-list-row'>"
        f"<span>{_escape(actor_lookup.get(actor_id, {}).get('name') or actor_id)}</span>"
        f"<strong>{actor_weights[actor_id]} {L['mentions']} | {actor_doc_counts.get(actor_id, 0)} {L['linked_docs']}</strong>"
        "</div>"
        for actor_id in sorted(actor_weights, key=lambda key: actor_weights[key], reverse=True)
    )
    doc_rows = "".join(
        "<div class='viz-list-row'>"
        f"<span>{_escape(doc_id)}</span>"
        f"<strong>{document_weights[doc_id]} {L['links']} | {len(document_actor_counts.get(doc_id, set()))} {L['actors_word']}</strong>"
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
    ) or f"<div class='viz-empty'>{L['no_pairs']}</div>"
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
        visual_note = f"<div class='viz-footnote'>{L['footnote_slice']}</div>"

    return (
        "<div class='viz-shell'>"
        f"<div class='viz-panel-title'>{L['title']}</div>"
        f"<div class='viz-footnote'>{L['subtitle']}</div>"
        f"<svg class='comm-graph' viewBox='0 0 {width} {height}' role='img'>"
        + "".join(svg_lines)
        + "".join(svg_nodes)
        + "</svg>"
        + visual_note
        + "<div class='viz-two-col'>"
        + f"<div class='viz-panel'><div class='viz-subtitle'>{L['actors']}</div>"
        + actor_rows
        + "</div>"
        + f"<div class='viz-panel'><div class='viz-subtitle'>{L['documents']}</div>"
        + doc_rows
        + "</div></div>"
        + "<div class='viz-panel'>"
        + f"<div class='viz-subtitle'>{L['pairs']}</div>"
        + pair_rows
        + "</div>"
        + f"<div class='viz-panel'><div class='viz-subtitle'>{L['edge_detail']}</div>"
        + "<div class='matrix-wrap'><table class='analytics-table'><thead><tr>"
        + f"<th>{L['col_actor']}</th><th>{L['col_document']}</th><th>{L['col_mentions']}</th>"
        + "</tr></thead><tbody>"
        + (edge_rows or f"<tr><td colspan='3'>{L['no_edges']}</td></tr>")
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
        return sum(_safe_int(call.get(key, 0)) for call in calls if isinstance(call, dict))

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
        if not isinstance(call, dict):
            continue
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
        stage_max = max((s.get("estimated_cost_usd") or 0) for s in by_stage if isinstance(s, dict))
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
            for s in by_stage if isinstance(s, dict)
        )
    else:
        stage_costs: dict[str, float] = defaultdict(float)
        stage_calls: dict[str, int] = defaultdict(int)
        for call in calls:
            if not isinstance(call, dict):
                continue
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
        tier_max = max((t.get("estimated_cost_usd") or 0) for t in by_tier if isinstance(t, dict))
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
            for t in by_tier if isinstance(t, dict)
        )
    else:
        model_costs: dict[str, float] = defaultdict(float)
        for call in calls:
            if not isinstance(call, dict):
                continue
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
            for a in anomalies if isinstance(a, dict)
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
        for call in calls if isinstance(call, dict)
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


_QUANT_PANEL_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "empty": "No quantitative facts extracted yet.",
        "invoiced": "Invoiced", "paid": "Paid", "disputed": "Disputed", "exposure": "Net exposure",
        "waterfall": "Damages waterfall", "no_waterfall": "No damages waterfall available.",
        "categories": "Category totals", "no_categories": "No category totals available.",
        "invoice_recon": "Invoice reconciliation", "no_invoices": "No invoice chain available.",
        "grounding": "Payment grounding", "no_grounding": "No source span grounding available.",
        "conflicts": "Amount conflicts", "no_conflicts": "No amount conflicts detected.",
        "damage_detail": "Damage detail",
    },
    "finance": {
        "empty": "No quantitative facts extracted yet.",
        "invoiced": "Billed", "paid": "Settled", "disputed": "Disputed", "exposure": "Net exposure",
        "waterfall": "Amount breakdown", "no_waterfall": "No amount breakdown available.",
        "categories": "Category totals", "no_categories": "No category totals available.",
        "invoice_recon": "Transaction reconciliation", "no_invoices": "No transaction chain available.",
        "grounding": "Transaction grounding", "no_grounding": "No source grounding available.",
        "conflicts": "Amount conflicts", "no_conflicts": "No amount conflicts detected.",
        "damage_detail": "Amount detail",
    },
    "coding": {
        "empty": "No quantitative facts extracted yet.",
        "invoiced": "Allocated", "paid": "Consumed", "disputed": "Disputed", "exposure": "Net remaining",
        "waterfall": "Metric breakdown", "no_waterfall": "No metric breakdown available.",
        "categories": "Category totals", "no_categories": "No category totals available.",
        "invoice_recon": "Resource reconciliation", "no_invoices": "No resource chain available.",
        "grounding": "Metric grounding", "no_grounding": "No source grounding available.",
        "conflicts": "Metric conflicts", "no_conflicts": "No metric conflicts detected.",
        "damage_detail": "Metric detail",
    },
    "academic_research": {
        "empty": "No quantitative facts extracted yet.",
        "invoiced": "Budgeted", "paid": "Spent", "disputed": "Disputed", "exposure": "Net remaining",
        "waterfall": "Amount breakdown", "no_waterfall": "No amount breakdown available.",
        "categories": "Category totals", "no_categories": "No category totals available.",
        "invoice_recon": "Funding reconciliation", "no_invoices": "No funding chain available.",
        "grounding": "Funding grounding", "no_grounding": "No source grounding available.",
        "conflicts": "Amount conflicts", "no_conflicts": "No amount conflicts detected.",
        "damage_detail": "Amount detail",
    },
    "biomedical": {
        "empty": "No quantitative facts extracted yet.",
        "invoiced": "Charged", "paid": "Paid", "disputed": "Disputed", "exposure": "Net exposure",
        "waterfall": "Cost breakdown", "no_waterfall": "No cost breakdown available.",
        "categories": "Category totals", "no_categories": "No category totals available.",
        "invoice_recon": "Cost reconciliation", "no_invoices": "No cost chain available.",
        "grounding": "Cost grounding", "no_grounding": "No source grounding available.",
        "conflicts": "Amount conflicts", "no_conflicts": "No amount conflicts detected.",
        "damage_detail": "Cost detail",
    },
}


_QUANT_ONTOLOGY_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Metric Ontology Workbench",
        "approved": "Approved",
        "pending": "Pending classification",
        "empty": "No quantitative facts extracted yet. Run an investigation to populate metrics.",
        "coverage": "Ontology coverage",
        "classify": "Classify",
        "metric_type": "Raw metric type",
        "canonical": "Canonical metric",
    },
    "finance": {
        "title": "Financial Metric Ontology",
        "approved": "Approved",
        "pending": "Pending classification",
        "empty": "No financial metrics extracted yet.",
        "coverage": "Metric ontology coverage",
        "classify": "Classify",
        "metric_type": "Raw metric type",
        "canonical": "Canonical metric",
    },
    "coding": {
        "title": "Engineering Metric Ontology",
        "approved": "Approved",
        "pending": "Pending classification",
        "empty": "No engineering metrics extracted yet.",
        "coverage": "Metric ontology coverage",
        "classify": "Classify",
        "metric_type": "Raw metric type",
        "canonical": "Canonical metric",
    },
    "academic_research": {
        "title": "Research Metric Ontology",
        "approved": "Approved",
        "pending": "Pending classification",
        "empty": "No research metrics extracted yet.",
        "coverage": "Metric ontology coverage",
        "classify": "Classify",
        "metric_type": "Raw metric type",
        "canonical": "Canonical metric",
    },
    "biomedical": {
        "title": "Clinical Metric Ontology",
        "approved": "Approved",
        "pending": "Pending classification",
        "empty": "No clinical metrics extracted yet.",
        "coverage": "Metric ontology coverage",
        "classify": "Classify",
        "metric_type": "Raw metric type",
        "canonical": "Canonical metric",
    },
}


def _fmt_quant_ontology(data: dict, domain: str = "legal") -> str:
    labels = _QUANT_ONTOLOGY_LABELS.get(domain, _QUANT_ONTOLOGY_LABELS["legal"])
    if err := _error_html(data):
        return err
    if not data or not data.get("metric_groups"):
        return f"<p style='color:#888;'>{_escape(labels['empty'])}</p>"

    groups = data["metric_groups"]
    approved_count = data.get("approved_count", 0)
    total = data.get("total_metric_types", 0)
    coverage_frac = data.get("coverage_fraction", 0)

    pct = int(coverage_frac * 100)
    bar_color = "#22c55e" if pct >= 70 else "#f59e0b" if pct >= 30 else "#dc2626"

    parts = [
        f"<h3 style='margin:0 0 8px;'>{_escape(labels['title'])}</h3>",
        f"<div style='display:flex;gap:24px;margin-bottom:12px;'>",
        f"<div><b>{_escape(labels['coverage'])}:</b> {int(approved_count)}/{int(total)} ({int(pct)}%)</div>",
        f"</div>",
        f"<div style='background:#e5e7eb;border-radius:4px;height:8px;margin-bottom:16px;'>",
        f"<div style='background:{bar_color};height:8px;border-radius:4px;width:{pct}%;'></div>",
        f"</div>",
    ]

    for g in groups:
        if not isinstance(g, dict):
            continue
        mt = _escape(str(g.get("metric_type", "")))
        cm = g.get("canonical_metric")
        approved = g.get("approved", False)
        fc = g.get("fact_count", 0)
        tv = g.get("total_value", 0)

        badge_color = "#22c55e" if approved else "#f59e0b"
        badge_text = _escape(labels["approved"]) if approved else _escape(labels["pending"])
        canonical_display = _escape(str(cm)) if cm else "<i>unclassified</i>"

        parts.append(
            f"<div style='border:1px solid #e5e7eb;border-radius:6px;padding:10px;margin-bottom:8px;'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
            f"<div><b>{mt}</b> → {canonical_display}</div>"
            f"<span style='background:{badge_color};color:white;padding:2px 8px;border-radius:10px;"
            f"font-size:0.8em;'>{badge_text}</span>"
            f"</div>"
            f"<div style='color:#6b7280;font-size:0.85em;margin-top:4px;'>"
            f"{int(fc)} facts"
        )
        if tv:
            parts.append(f" · Total value: {float(tv):,.2f}")
        parts.append("</div>")

        sample_facts = g.get("sample_facts", [])
        if sample_facts:
            parts.append("<div style='margin-top:6px;font-size:0.82em;color:#4b5563;'>")
            for sf in sample_facts[:3]:
                if not isinstance(sf, dict):
                    continue
                raw = _escape(str(sf.get("raw_text", ""))[:80])
                parts.append(f"<div>· {raw}</div>")
            parts.append("</div>")

        parts.append("</div>")

    return "\n".join(parts)


_CANONICAL_METRICS: dict[str, list[str]] = {
    "legal": [
        "accounts_receivable", "accounts_payable", "invoice_amount",
        "payment_amount", "outstanding_balance", "damages_claimed",
        "settlement_amount", "interest_rate", "deadline_date", "contract_term",
    ],
    "finance": [
        "revenue", "gross_margin", "operating_income", "net_income", "eps",
        "free_cash_flow", "arr", "deferred_revenue", "rpo", "capex",
        "cash_and_equivalents",
    ],
    "coding": [
        "latency", "error_rate", "throughput", "memory_usage", "cpu_usage",
        "test_coverage", "build_time", "defect_count",
    ],
    "academic_research": [
        "sample_size", "effect_size", "p_value", "confidence_interval",
        "accuracy", "precision", "recall", "f1_score", "auc",
    ],
    "biomedical": [
        "hazard_ratio", "odds_ratio", "relative_risk", "confidence_interval",
        "p_value", "sample_size", "dosage", "adverse_event_rate",
        "overall_survival", "progression_free_survival",
    ],
}


def _canonical_metric_choices(domain: str = "legal") -> list[str]:
    return _CANONICAL_METRICS.get(domain, _CANONICAL_METRICS["legal"])


_ANSWER_AUDIT_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Answer Audit Trail",
        "fresh": "Fresh",
        "stale": "Stale",
        "policy_limited": "Policy-limited",
        "unknown": "Unknown",
        "empty": "No answer audits available. Run an investigation to generate dependency manifests.",
        "evidence": "Consumed evidence",
        "changes": "Changes since answer",
        "missingness": "Missingness dependencies",
        "domain_policy": "Domain & policy",
    },
    "finance": {
        "title": "Analysis Audit Trail",
        "fresh": "Fresh",
        "stale": "Stale",
        "policy_limited": "Policy-limited",
        "unknown": "Unknown",
        "empty": "No analysis audits available.",
        "evidence": "Consumed data",
        "changes": "Changes since analysis",
        "missingness": "Missingness dependencies",
        "domain_policy": "Domain & policy",
    },
    "coding": {
        "title": "Reasoning Audit Trail",
        "fresh": "Fresh",
        "stale": "Stale",
        "policy_limited": "Policy-limited",
        "unknown": "Unknown",
        "empty": "No reasoning audits available.",
        "evidence": "Consumed artifacts",
        "changes": "Changes since analysis",
        "missingness": "Missingness dependencies",
        "domain_policy": "Domain & policy",
    },
    "academic_research": {
        "title": "Research Audit Trail",
        "fresh": "Fresh",
        "stale": "Stale",
        "policy_limited": "Policy-limited",
        "unknown": "Unknown",
        "empty": "No research audits available.",
        "evidence": "Consumed sources",
        "changes": "Changes since analysis",
        "missingness": "Missingness dependencies",
        "domain_policy": "Domain & policy",
    },
    "biomedical": {
        "title": "Clinical Audit Trail",
        "fresh": "Fresh",
        "stale": "Stale",
        "policy_limited": "Policy-limited",
        "unknown": "Unknown",
        "empty": "No clinical audits available.",
        "evidence": "Consumed evidence",
        "changes": "Changes since analysis",
        "missingness": "Missingness dependencies",
        "domain_policy": "Domain & policy",
    },
}


def _fmt_answer_audit(data: dict, domain: str = "legal") -> str:
    labels = _ANSWER_AUDIT_LABELS.get(domain, _ANSWER_AUDIT_LABELS["legal"])
    if not data or not isinstance(data, dict):
        return f"<p style='color:#888;'>{_escape(labels['empty'])}</p>"
    if err := _error_html(data):
        return err
    audits = data.get("audits", []) if isinstance(data, dict) else []
    if not audits:
        return f"<p style='color:#888;'>{_escape(labels['empty'])}</p>"

    total = int(data.get("total_manifests", 0))
    parts = [
        f"<h3 style='margin:0 0 8px;'>{_escape(labels['title'])}</h3>",
        f"<div style='color:#6b7280;margin-bottom:12px;'>"
        f"{len(audits)} of {total} manifests shown</div>",
    ]

    badge_colors = {
        "fresh": "#22c55e",
        "stale": "#dc2626",
        "policy_limited": "#f59e0b",
        "unknown": "#6b7280",
    }

    for audit in audits:
        if not isinstance(audit, dict):
            continue

        mh = _escape(str(audit.get("manifest_hash", ""))[:16])
        purpose = _escape(str(audit.get("purpose", "")))
        created = _escape(str(audit.get("created_at", ""))[:19])
        badge = audit.get("status_badge", "unknown")
        badge_label = _escape(labels.get(badge, badge))
        badge_color = badge_colors.get(badge, "#6b7280")

        profile = _escape(str(audit.get("domain_profile_id", "")))
        audience = _escape(str(audit.get("policy_audience", "")))
        taint = _escape(str(audit.get("taint_class", "")))
        obj_count = audit.get("object_dependency_count", 0)
        neg_count = audit.get("negative_dependency_count", 0)

        parts.append(
            f"<div style='border:1px solid #e5e7eb;border-radius:6px;padding:12px;margin-bottom:10px;'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
            f"<div><b>{purpose}</b> <span style='color:#9ca3af;font-size:0.8em;'>{mh}…</span></div>"
            f"<span style='background:{badge_color};color:white;padding:2px 10px;border-radius:10px;"
            f"font-size:0.8em;'>{badge_label}</span>"
            f"</div>"
            f"<div style='color:#6b7280;font-size:0.85em;margin-top:4px;'>"
            f"{created} · {int(obj_count)} evidence deps · {int(neg_count)} missingness deps"
            f"</div>"
        )

        parts.append(
            f"<div style='margin-top:6px;font-size:0.85em;'>"
            f"<b>{_escape(labels['domain_policy'])}:</b> "
            f"Profile: {profile} · Audience: {audience} · Taint: {taint}"
            f"</div>"
        )

        stale_reasons = audit.get("stale_reasons", [])
        if stale_reasons:
            parts.append(
                f"<div style='margin-top:6px;font-size:0.82em;color:#dc2626;'>"
                f"<b>{_escape(labels['changes'])}:</b>"
            )
            for reason in stale_reasons[:5]:
                if not isinstance(reason, str):
                    continue
                parts.append(f"<div>· {_escape(reason[:120])}</div>")
            parts.append("</div>")

        obj_groups = audit.get("object_groups", {})
        if isinstance(obj_groups, dict) and obj_groups:
            parts.append(
                f"<div style='margin-top:6px;font-size:0.82em;color:#4b5563;'>"
                f"<b>{_escape(labels['evidence'])}:</b>"
            )
            for kind, items in obj_groups.items():
                if not isinstance(items, list):
                    continue
                parts.append(f"<div>{_escape(str(kind))}: {len(items)} dep(s)</div>")
            parts.append("</div>")

        neg_deps = audit.get("negative_dependencies", [])
        if neg_deps:
            parts.append(
                f"<div style='margin-top:6px;font-size:0.82em;color:#7c3aed;'>"
                f"<b>{_escape(labels['missingness'])}:</b>"
            )
            for nd in neg_deps[:5]:
                if not isinstance(nd, dict):
                    continue
                ns = _escape(str(nd.get("namespace", "")))
                pred = _escape(str(nd.get("query_predicate", ""))[:60])
                parts.append(f"<div>· {ns}: {pred}</div>")
            parts.append("</div>")

        parts.append("</div>")

    return "\n".join(parts)


_OBJECTIVE_COVERAGE_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Objective Coverage",
        "empty": "No objectives defined yet. Run an investigation to generate the issue tree.",
        "objective": "Claim / Issue",
        "criteria": "Legal Elements",
        "support": "Supporting Assertions",
        "gaps": "Open Gaps",
        "coverage": "Coverage",
        "covered": "Covered",
        "thin": "Thin Coverage",
        "blocked": "Blocked",
        "missing": "Missing Evidence",
        "contradicted": "Contradicted",
    },
    "finance": {
        "title": "Thesis Coverage",
        "empty": "No thesis points defined. Run an investigation to generate the analysis tree.",
        "objective": "Thesis Point",
        "criteria": "Key Metrics / Criteria",
        "support": "Supporting Evidence",
        "gaps": "Open Gaps",
        "coverage": "Coverage",
        "covered": "Covered",
        "thin": "Thin Coverage",
        "blocked": "Blocked",
        "missing": "Missing Data",
        "contradicted": "Contradicted",
    },
    "coding": {
        "title": "Requirement Coverage",
        "empty": "No requirements defined. Run an investigation to generate the requirement tree.",
        "objective": "Requirement",
        "criteria": "Acceptance Criteria",
        "support": "Supporting Evidence",
        "gaps": "Open Gaps",
        "coverage": "Coverage",
        "covered": "Covered",
        "thin": "Thin Coverage",
        "blocked": "Blocked",
        "missing": "Missing Tests / Evidence",
        "contradicted": "Contradicted",
    },
    "academic_research": {
        "title": "Research Question Coverage",
        "empty": "No research questions defined. Run an investigation to generate the question tree.",
        "objective": "Research Question",
        "criteria": "Criteria / Hypotheses",
        "support": "Supporting Findings",
        "gaps": "Open Gaps",
        "coverage": "Coverage",
        "covered": "Covered",
        "thin": "Thin Coverage",
        "blocked": "Blocked",
        "missing": "Missing Evidence",
        "contradicted": "Contradicted",
    },
    "biomedical": {
        "title": "Endpoint Coverage",
        "empty": "No endpoints defined. Run an investigation to generate the assessment tree.",
        "objective": "Endpoint / Outcome",
        "criteria": "Assessment Criteria",
        "support": "Supporting Evidence",
        "gaps": "Open Gaps",
        "coverage": "Coverage",
        "covered": "Covered",
        "thin": "Thin Coverage",
        "blocked": "Blocked",
        "missing": "Missing Data",
        "contradicted": "Contradicted",
    },
}


def _fmt_objective_coverage(data: dict, domain: str = "legal") -> str:
    labels = _OBJECTIVE_COVERAGE_LABELS.get(domain, _OBJECTIVE_COVERAGE_LABELS["legal"])
    if not isinstance(data, dict):
        return f"<p style='color:#888;'>{_escape(labels['empty'])}</p>"
    if err := _error_html(data):
        return err

    objectives = data.get("objectives", [])
    if not isinstance(objectives, list) or not objectives:
        return f"<p style='color:#888;'>{_escape(labels['empty'])}</p>"

    summary = data.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}

    _raw_total = data.get("total", 0)
    total = int(_raw_total) if isinstance(_raw_total, (int, float)) else 0

    badge_colors = {
        "covered": "#22c55e",
        "thin": "#f59e0b",
        "blocked": "#dc2626",
        "missing": "#6b7280",
        "contradicted": "#7c3aed",
    }

    def _safe_count(key: str) -> int:
        v = summary.get(key, 0)
        return int(v) if isinstance(v, (int, float)) else 0

    parts = [
        f"<h3 style='margin:0 0 8px;'>{_escape(labels['title'])}</h3>",
        f"<div style='display:flex;gap:12px;margin-bottom:12px;flex-wrap:wrap;'>",
    ]
    for badge_key in ("covered", "thin", "blocked", "missing", "contradicted"):
        cnt = _safe_count(badge_key)
        color = badge_colors.get(badge_key, "#6b7280")
        lbl = _escape(labels.get(badge_key, badge_key))
        parts.append(
            f"<span style='background:{color};color:white;padding:2px 10px;"
            f"border-radius:10px;font-size:0.85em;'>{cnt} {lbl}</span>"
        )
    parts.append(f"<span style='color:#6b7280;font-size:0.85em;padding:2px 0;'>{total} total</span>")
    parts.append("</div>")

    for obj in objectives:
        if not isinstance(obj, dict):
            continue

        title = _escape(str(obj.get("title", "")))
        oid = _escape(str(obj.get("id", ""))[:16])
        full_oid = _escape(str(obj.get("id", "")))
        issue_type = _escape(str(obj.get("issue_type", "")))
        badge = obj.get("coverage_badge", "missing")
        badge_label = _escape(labels.get(badge, badge))
        badge_color = badge_colors.get(badge, "#6b7280")
        mat = obj.get("materiality", 0.5)
        materiality = float(mat) if isinstance(mat, (int, float)) else 0.5
        if not math.isfinite(materiality):
            materiality = 0.5
        materiality = max(0.0, min(1.0, materiality))
        cov_frac = obj.get("coverage_fraction", 0.0)
        coverage = float(cov_frac) if isinstance(cov_frac, (int, float)) else 0.0
        if not math.isfinite(coverage):
            coverage = 0.0
        coverage = max(0.0, min(1.0, coverage))
        cov_pct = int(coverage * 100)
        supporting = obj.get("supporting_count", 0)
        supporting_ct = int(supporting) if isinstance(supporting, (int, float)) else 0
        supporting_ct = max(0, supporting_ct)

        pred_total = obj.get("predicate_total", 0)
        pt = int(pred_total) if isinstance(pred_total, (int, float)) else 0
        pred_sat = obj.get("predicate_satisfied", 0)
        ps = int(pred_sat) if isinstance(pred_sat, (int, float)) else 0

        burden = _escape(str(obj.get("burden_side") or ""))

        parts.append(
            f"<div style='border:1px solid #e5e7eb;border-radius:8px;padding:12px;margin-bottom:10px;'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
            f"<div>"
            f"<b>{title}</b>"
            f" <span style='color:#9ca3af;font-size:0.8em;'>{issue_type}</span>"
        )
        if burden:
            parts.append(f" <span style='color:#6b7280;font-size:0.75em;'>({burden})</span>")
        parts.append(
            f"</div>"
            f"<span style='background:{badge_color};color:white;padding:2px 10px;"
            f"border-radius:10px;font-size:0.8em;'>{badge_label}</span>"
            f"</div>"
        )

        parts.append(
            f"<div style='display:flex;gap:16px;margin-top:6px;font-size:0.85em;color:#4b5563;'>"
            f"<span>{_escape(labels['coverage'])}: {cov_pct}%</span>"
            f"<span>{_escape(labels['support'])}: {supporting_ct}</span>"
            f"<span>{_escape(labels['criteria'])}: {ps}/{pt}</span>"
            f"<span>Materiality: {int(materiality * 100)}%</span>"
            f"<span style='color:#9ca3af;font-size:0.85em;cursor:pointer;' title='{full_oid}'>{oid}…</span>"
            f"</div>"
        )

        cov_bar_color = badge_color
        parts.append(
            f"<div style='margin-top:6px;background:#e5e7eb;border-radius:4px;height:6px;'>"
            f"<div style='background:{cov_bar_color};width:{cov_pct}%;height:100%;"
            f"border-radius:4px;transition:width 0.3s;'></div></div>"
        )

        predicates = obj.get("predicates", [])
        if isinstance(predicates, list) and predicates:
            pred_status_colors = {
                "open": "#3b82f6",
                "resolved": "#22c55e",
                "blocked": "#dc2626",
                "contested": "#f59e0b",
            }
            parts.append(
                f"<div style='margin-top:8px;font-size:0.82em;'>"
                f"<b>{_escape(labels['criteria'])}:</b>"
            )
            for pred in predicates[:10]:
                if not isinstance(pred, dict):
                    continue
                pdesc = _escape(str(pred.get("description", ""))[:100])
                pstatus = str(pred.get("status", "open"))
                pcolor = pred_status_colors.get(pstatus, "#6b7280")
                parts.append(
                    f"<div style='margin:2px 0;'>"
                    f"<span style='color:{pcolor};'>●</span> {pdesc}"
                    f" <span style='color:{pcolor};font-size:0.85em;'>[{_escape(pstatus)}]</span>"
                    f"</div>"
                )
            parts.append("</div>")

        gaps = obj.get("gaps", [])
        if isinstance(gaps, list) and gaps:
            parts.append(
                f"<div style='margin-top:6px;font-size:0.82em;color:#dc2626;'>"
                f"<b>{_escape(labels['gaps'])}:</b>"
            )
            for gap in gaps[:5]:
                if not isinstance(gap, dict):
                    continue
                gdesc = _escape(str(gap.get("description", ""))[:100])
                gtype = _escape(str(gap.get("gap_type", "")))
                parts.append(f"<div>· {gtype}: {gdesc}</div>")
            parts.append("</div>")

        parts.append("</div>")

    return "\n".join(parts)


_KNOWLEDGE_SEED_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Reusable Intelligence Seeds",
        "empty": "No knowledge seeds available for this matter.",
        "promotable": "Promotable",
        "matter_local": "Accepted",
        "rejected": "Rejected",
        "seed_kind": "Kind",
        "source": "Source Matter",
        "domain": "Domain Profile",
        "review": "Review Note",
    },
    "finance": {
        "title": "Reusable Analytical Seeds",
        "empty": "No analytical seeds available for this portfolio.",
        "promotable": "Promotable",
        "matter_local": "Accepted",
        "rejected": "Rejected",
        "seed_kind": "Seed Type",
        "source": "Source Portfolio",
        "domain": "Domain Profile",
        "review": "Review Note",
    },
    "coding": {
        "title": "Reusable Pattern Seeds",
        "empty": "No pattern seeds available for this codebase.",
        "promotable": "Promotable",
        "matter_local": "Accepted",
        "rejected": "Rejected",
        "seed_kind": "Pattern Type",
        "source": "Source Codebase",
        "domain": "Domain Profile",
        "review": "Review Note",
    },
    "academic_research": {
        "title": "Reusable Methodology Seeds",
        "empty": "No methodology seeds available for this study.",
        "promotable": "Promotable",
        "matter_local": "Accepted",
        "rejected": "Rejected",
        "seed_kind": "Methodology Type",
        "source": "Source Study",
        "domain": "Domain Profile",
        "review": "Review Note",
    },
    "biomedical": {
        "title": "Reusable Clinical Seeds",
        "empty": "No clinical seeds available for this protocol.",
        "promotable": "Promotable",
        "matter_local": "Accepted",
        "rejected": "Rejected",
        "seed_kind": "Seed Category",
        "source": "Source Protocol",
        "domain": "Domain Profile",
        "review": "Review Note",
    },
}


def _fmt_knowledge_seeds(data: dict, domain: str = "legal") -> str:
    labels = _KNOWLEDGE_SEED_LABELS.get(domain, _KNOWLEDGE_SEED_LABELS["legal"])
    if not isinstance(data, dict):
        return f"<p style='color:#888;'>{_escape(labels['empty'])}</p>"
    if err := _error_html(data):
        return err

    _raw_total = data.get("total", 0)
    total = int(_raw_total) if isinstance(_raw_total, (int, float)) else 0
    counts = data.get("counts", {})
    if not isinstance(counts, dict):
        counts = {}
    promotable = data.get("promotable", [])
    accepted = data.get("accepted", [])
    rejected = data.get("rejected", [])

    if total == 0:
        return f"<p style='color:#888;'>{_escape(labels['empty'])}</p>"

    _raw_p = counts.get("promotable", 0)
    p_count = int(_raw_p) if isinstance(_raw_p, (int, float)) else 0
    _raw_a = counts.get("matter_local", 0)
    a_count = int(_raw_a) if isinstance(_raw_a, (int, float)) else 0
    _raw_r = counts.get("rejected", 0)
    r_count = int(_raw_r) if isinstance(_raw_r, (int, float)) else 0

    parts = [
        f"<h3 style='margin:0 0 8px;'>{_escape(labels['title'])}</h3>",
        f"<div style='display:flex;gap:16px;margin-bottom:12px;'>",
        f"<span style='background:#3b82f6;color:white;padding:2px 10px;border-radius:10px;"
        f"font-size:0.85em;'>{p_count} {_escape(labels['promotable'])}</span>",
        f"<span style='background:#22c55e;color:white;padding:2px 10px;border-radius:10px;"
        f"font-size:0.85em;'>{a_count} {_escape(labels['matter_local'])}</span>",
        f"<span style='background:#dc2626;color:white;padding:2px 10px;border-radius:10px;"
        f"font-size:0.85em;'>{r_count} {_escape(labels['rejected'])}</span>",
        f"</div>",
    ]

    status_colors = {
        "promotable": "#3b82f6",
        "matter_local": "#22c55e",
        "rejected": "#dc2626",
    }

    for group_label, group_items, status_key in [
        (labels["promotable"], promotable, "promotable"),
        (labels["matter_local"], accepted, "matter_local"),
        (labels["rejected"], rejected, "rejected"),
    ]:
        if not isinstance(group_items, list) or not group_items:
            continue
        parts.append(
            f"<div style='margin-top:8px;'>"
            f"<b style='color:{status_colors[status_key]};'>{_escape(group_label)}</b>"
            f"</div>"
        )
        for seed in group_items:
            if not isinstance(seed, dict):
                continue
            sid = _escape(str(seed.get("id", ""))[:16])
            kind = _escape(str(seed.get("seed_kind", "")))
            source = _escape(str(seed.get("source_matter_id", "") or "—")[:20])
            profile = _escape(str(seed.get("domain_profile_id", "")))
            created = _escape(str(seed.get("created_at", ""))[:19])
            note = _escape(str(seed.get("review_note", "") or ""))

            parts.append(
                f"<div style='border:1px solid #e5e7eb;border-radius:6px;padding:10px;margin:4px 0;'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                f"<div><b>{kind}</b> <span style='color:#9ca3af;font-size:0.8em;'>{sid}…</span></div>"
                f"<span style='color:{status_colors[status_key]};font-size:0.82em;'>"
                f"{_escape(labels[status_key])}</span>"
                f"</div>"
                f"<div style='color:#6b7280;font-size:0.85em;margin-top:4px;'>"
                f"{_escape(labels['source'])}: {source} · {_escape(labels['domain'])}: {profile} · {created}"
                f"</div>"
            )
            if note:
                parts.append(
                    f"<div style='color:#4b5563;font-size:0.82em;margin-top:4px;'>"
                    f"<b>{_escape(labels['review'])}:</b> {note[:200]}"
                    f"</div>"
                )
            parts.append("</div>")

    return "\n".join(parts)


_QUANT_FACT_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Extracted Numeric Facts",
        "empty": "No quantitative facts extracted yet. Run an investigation over financial documents.",
        "amount": "Monetary Amounts",
        "date": "Dates",
        "date_range": "Date Ranges",
        "rate": "Rates / Percentages",
        "balance": "Balances",
        "count": "Counts",
        "conflict": "Conflicted",
        "source": "Source",
        "subject": "Subject",
    },
    "finance": {
        "title": "Extracted Financial Data Points",
        "empty": "No data points extracted yet. Run an investigation over financial documents.",
        "amount": "Dollar Amounts",
        "date": "Dates",
        "date_range": "Date Ranges",
        "rate": "Rates / Yields",
        "balance": "Account Balances",
        "count": "Counts",
        "conflict": "Conflicted",
        "source": "Source",
        "subject": "Entity / Instrument",
    },
    "coding": {
        "title": "Extracted Metrics",
        "empty": "No metrics extracted yet. Run an investigation.",
        "amount": "Numeric Values",
        "date": "Dates",
        "date_range": "Date Ranges",
        "rate": "Percentages",
        "balance": "Counters",
        "count": "Counts",
        "conflict": "Conflicted",
        "source": "Source",
        "subject": "Component",
    },
    "academic_research": {
        "title": "Extracted Quantitative Data",
        "empty": "No quantitative data extracted yet. Run an investigation.",
        "amount": "Measurements",
        "date": "Dates",
        "date_range": "Study Periods",
        "rate": "Effect Sizes / Rates",
        "balance": "Baselines",
        "count": "Sample Sizes",
        "conflict": "Conflicted",
        "source": "Source",
        "subject": "Variable / Endpoint",
    },
    "biomedical": {
        "title": "Extracted Clinical Data",
        "empty": "No clinical data extracted yet. Run an investigation.",
        "amount": "Dosages / Measurements",
        "date": "Dates",
        "date_range": "Treatment Periods",
        "rate": "Rates / Hazard Ratios",
        "balance": "Baselines",
        "count": "Patient Counts",
        "conflict": "Conflicted",
        "source": "Source",
        "subject": "Endpoint / Biomarker",
    },
}


def _fmt_quant_facts(data: dict, domain: str = "legal") -> str:
    labels = _QUANT_FACT_LABELS.get(domain, _QUANT_FACT_LABELS["legal"])
    if not isinstance(data, dict):
        return f"<p style='color:#888;'>{_escape(labels['empty'])}</p>"
    if err := _error_html(data):
        return err

    by_kind = data.get("by_kind", [])
    if not isinstance(by_kind, list) or not by_kind:
        return f"<p style='color:#888;'>{_escape(labels['empty'])}</p>"

    def _safe_int(v, default: int = 0) -> int:
        if isinstance(v, (int, float)) and math.isfinite(v):
            return max(0, int(v))
        return default

    total = _safe_int(data.get("total", 0))
    total_conflicted = _safe_int(data.get("total_conflicted", 0))

    parts = [
        f"<h3 style='margin:0 0 8px;'>{_escape(labels['title'])}</h3>",
        f"<div style='display:flex;gap:12px;margin-bottom:12px;flex-wrap:wrap;'>",
        f"<span style='color:#4b5563;font-size:0.85em;'>{total} facts</span>",
    ]
    if total_conflicted > 0:
        parts.append(
            f"<span style='background:#dc2626;color:white;padding:2px 10px;"
            f"border-radius:10px;font-size:0.85em;'>"
            f"{total_conflicted} {_escape(labels['conflict'])}</span>"
        )
    parts.append("</div>")

    kind_colors = {
        "amount": "#22c55e", "date": "#3b82f6", "date_range": "#6366f1",
        "rate": "#f59e0b", "balance": "#8b5cf6", "count": "#6b7280",
    }

    for group in by_kind:
        if not isinstance(group, dict):
            continue
        kind = str(group.get("kind", "unknown"))
        facts = group.get("facts", [])
        if not isinstance(facts, list) or not facts:
            continue
        count = _safe_int(group.get("count"), len(facts))
        conflicted = _safe_int(group.get("conflicted"))
        kind_label = _escape(labels.get(kind, kind.replace("_", " ").title()))
        kind_color = kind_colors.get(kind, "#6b7280")

        parts.append(
            f"<div style='border:1px solid #e5e7eb;border-radius:8px;padding:12px;margin-bottom:10px;'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
            f"<div><b style='color:{kind_color};'>{kind_label}</b>"
            f" <span style='color:#9ca3af;font-size:0.85em;'>({count})</span></div>"
        )
        if conflicted > 0:
            parts.append(
                f"<span style='background:#dc2626;color:white;padding:2px 8px;"
                f"border-radius:8px;font-size:0.78em;'>{conflicted} conflict(s)</span>"
            )
        parts.append("</div>")

        parts.append("<div style='margin-top:8px;font-size:0.82em;'>")
        for fact in facts[:20]:
            if not isinstance(fact, dict):
                continue
            raw = _escape(str(fact.get("raw_text", ""))[:120])
            amount = fact.get("amount_value")
            currency = _escape(str(fact.get("currency") or ""))
            subj_type = _escape(str(fact.get("subject_type") or ""))
            subj_id = _escape(str(fact.get("subject_id") or "")[:40])
            has_conflict = fact.get("has_conflict", False)
            fid = _escape(str(fact.get("id", ""))[:16])

            amount_display = ""
            if amount is not None and isinstance(amount, (int, float)) and math.isfinite(amount):
                if currency:
                    amount_display = f"{currency} {amount:,.2f}"
                else:
                    amount_display = f"{amount:,.2f}"
            elif fact.get("rate_value") is not None:
                rv = fact["rate_value"]
                if isinstance(rv, (int, float)) and math.isfinite(rv):
                    amount_display = f"{rv:.2%}" if abs(rv) < 10 else f"{rv:,.2f}"
            elif fact.get("date_value"):
                amount_display = _escape(str(fact["date_value"])[:20])

            conflict_badge = ""
            if has_conflict:
                conflict_badge = (
                    " <span style='background:#fecaca;color:#991b1b;padding:1px 6px;"
                    "border-radius:4px;font-size:0.82em;'>conflict</span>"
                )

            subject_info = ""
            if subj_type:
                subject_info = f" <span style='color:#6b7280;'>({subj_type}"
                if subj_id:
                    subject_info += f": {subj_id}"
                subject_info += ")</span>"

            parts.append(
                f"<div style='margin:3px 0;padding:4px 8px;background:#f9fafb;"
                f"border-radius:4px;border-left:3px solid {kind_color};'>"
                f"<span style='font-weight:500;'>{_escape(amount_display)}</span>"
                f"{conflict_badge}{subject_info}"
                f" <span style='color:#9ca3af;font-size:0.85em;'>{raw}</span>"
                f" <span style='color:#d1d5db;font-size:0.75em;cursor:pointer;' title='{fid}'>{fid}</span>"
                f"</div>"
            )
        if len(facts) > 20:
            parts.append(f"<div style='color:#9ca3af;margin-top:4px;'>... and {len(facts) - 20} more</div>")
        parts.append("</div></div>")

    return "\n".join(parts)


_DECISION_LEVERAGE_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Decision Leverage Map",
        "subtitle": "What to review next to shift the case outcome",
        "empty": "No leverage points identified yet. Run an investigation first.",
        "weak_objective": "Weak Objective",
        "unreviewed_assumption": "Unreviewed Assumption",
        "tainted_evidence": "Tainted Evidence",
        "quant_conflict": "Numeric Conflict",
        "open_gap": "Open Gap",
        "pending_review": "Pending Review",
        "blocker": "Blocker",
        "impact": "Impact",
        "action": "Recommended Action",
    },
    "finance": {
        "title": "Decision Leverage Map",
        "subtitle": "What to review next to shift the investment thesis",
        "empty": "No leverage points identified yet. Run an investigation first.",
        "weak_objective": "Weak Thesis Point",
        "unreviewed_assumption": "Unverified Assumption",
        "tainted_evidence": "Compromised Data",
        "quant_conflict": "Numeric Discrepancy",
        "open_gap": "Information Gap",
        "pending_review": "Pending Verification",
        "blocker": "Blocker",
        "impact": "Impact",
        "action": "Recommended Action",
    },
    "coding": {
        "title": "Decision Leverage Map",
        "subtitle": "What to review next to improve code quality",
        "empty": "No leverage points identified yet. Run an investigation first.",
        "weak_objective": "Weak Requirement",
        "unreviewed_assumption": "Unverified Assumption",
        "tainted_evidence": "Unreliable Source",
        "quant_conflict": "Metric Conflict",
        "open_gap": "Coverage Gap",
        "pending_review": "Pending Review",
        "blocker": "Blocker",
        "impact": "Impact",
        "action": "Recommended Action",
    },
    "academic_research": {
        "title": "Decision Leverage Map",
        "subtitle": "What to review next to strengthen the research position",
        "empty": "No leverage points identified yet. Run an investigation first.",
        "weak_objective": "Weak Hypothesis Support",
        "unreviewed_assumption": "Untested Assumption",
        "tainted_evidence": "Questionable Source",
        "quant_conflict": "Statistical Conflict",
        "open_gap": "Literature Gap",
        "pending_review": "Pending Verification",
        "blocker": "Blocker",
        "impact": "Impact",
        "action": "Recommended Action",
    },
    "biomedical": {
        "title": "Decision Leverage Map",
        "subtitle": "What to review next to shift clinical conclusions",
        "empty": "No leverage points identified yet. Run an investigation first.",
        "weak_objective": "Weak Endpoint",
        "unreviewed_assumption": "Unverified Assumption",
        "tainted_evidence": "Compromised Data",
        "quant_conflict": "Measurement Conflict",
        "open_gap": "Evidence Gap",
        "pending_review": "Pending Review",
        "blocker": "Blocker",
        "impact": "Impact",
        "action": "Recommended Action",
    },
}


def _fmt_decision_leverage(data: dict, domain: str = "legal") -> str:
    labels = _DECISION_LEVERAGE_LABELS.get(domain, _DECISION_LEVERAGE_LABELS["legal"])
    if not isinstance(data, dict):
        return f"<p style='color:#888;'>{_escape(labels['empty'])}</p>"
    if err := _error_html(data):
        return err

    items = data.get("items", [])
    if not isinstance(items, list) or not items:
        return f"<p style='color:#888;'>{_escape(labels['empty'])}</p>"

    kind_colors = {
        "weak_objective": "#ef4444",
        "unreviewed_assumption": "#f59e0b",
        "tainted_evidence": "#dc2626",
        "quant_conflict": "#8b5cf6",
        "open_gap": "#3b82f6",
        "pending_review": "#6b7280",
    }

    def _safe_float(v, default: float = 0.0) -> float:
        if isinstance(v, (int, float)) and math.isfinite(v):
            return max(0.0, min(1.0, float(v)))
        return default

    total = data.get("total", len(items))
    if not isinstance(total, (int, float)) or not math.isfinite(total):
        total = len(items)

    parts = [
        f"<div style='margin-bottom:12px;'>",
        f"<h4 style='margin:0 0 4px 0;'>{_escape(labels['title'])}</h4>",
        f"<p style='color:#6b7280;margin:0 0 12px 0;font-size:0.9em;'>"
        f"{_escape(labels['subtitle'])} &mdash; {int(total)} item(s)</p>",
    ]

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", ""))
        color = kind_colors.get(kind, "#6b7280")
        kind_label = _escape(labels.get(kind, kind.replace("_", " ").title()))
        title = _escape(str(item.get("title", ""))[:120])
        blocker = _escape(str(item.get("blocker", ""))[:80])
        impact = _safe_float(item.get("impact"))
        detail = _escape(str(item.get("detail", ""))[:200])
        action = _escape(str(item.get("action", ""))[:200])
        item_id = _escape(str(item.get("id", ""))[:60])

        impact_pct = int(impact * 100)
        impact_color = "#dc2626" if impact >= 0.7 else "#f59e0b" if impact >= 0.4 else "#6b7280"

        parts.append(
            f"<div style='margin:6px 0;padding:10px 14px;background:#f9fafb;"
            f"border-radius:6px;border-left:4px solid {color};'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
            f"<span style='font-weight:600;font-size:0.95em;'>"
            f"<span style='color:{color};font-size:0.8em;'>{idx + 1}.</span> "
            f"{title}</span>"
            f"<span style='background:{impact_color};color:white;padding:2px 8px;"
            f"border-radius:10px;font-size:0.8em;font-weight:500;'>"
            f"{impact_pct}% {_escape(labels['impact'].lower())}</span>"
            f"</div>"
            f"<div style='margin-top:4px;font-size:0.85em;color:#4b5563;'>"
            f"<span style='background:{color}22;color:{color};padding:1px 6px;"
            f"border-radius:3px;font-size:0.85em;'>{kind_label}</span>"
            f" &middot; {_escape(labels['blocker'])}: {blocker}"
            f"</div>"
            f"<div style='margin-top:4px;font-size:0.85em;color:#6b7280;'>{detail}</div>"
            f"<div style='margin-top:4px;font-size:0.85em;color:#059669;font-weight:500;'>"
            f"&#x2794; {action}</div>"
        )
        if item_id:
            parts.append(
                f"<div style='margin-top:2px;font-size:0.75em;color:#d1d5db;"
                f"cursor:pointer;' title='{item_id}'>{item_id}</div>"
            )
        parts.append("</div>")

    parts.append("</div>")
    return "\n".join(parts)


_OUTPUT_QUALITY_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Output Quality Contract",
        "subtitle": "Is this analysis ready for professional reliance?",
        "empty": "No investigation data available. Run an investigation first.",
        "ready": "Ready for reliance",
        "caution": "Proceed with caution",
        "blocked": "Not ready — action required",
        "run_header": "Recent Investigations",
        "obligations_header": "Reliance Obligations",
        "manifest_header": "Dependency Freshness",
    },
    "finance": {
        "title": "Output Quality Contract",
        "subtitle": "Is this analysis ready for investment reliance?",
        "empty": "No investigation data available. Run an analysis first.",
        "ready": "Ready for reliance",
        "caution": "Proceed with caution",
        "blocked": "Not ready — action required",
        "run_header": "Recent Analyses",
        "obligations_header": "Reliance Obligations",
        "manifest_header": "Dependency Freshness",
    },
    "coding": {
        "title": "Output Quality Contract",
        "subtitle": "Is this assessment ready for implementation?",
        "empty": "No investigation data available. Run an assessment first.",
        "ready": "Ready for action",
        "caution": "Proceed with caution",
        "blocked": "Not ready — action required",
        "run_header": "Recent Assessments",
        "obligations_header": "Quality Obligations",
        "manifest_header": "Dependency Freshness",
    },
    "academic_research": {
        "title": "Output Quality Contract",
        "subtitle": "Is this literature review ready for citation?",
        "empty": "No investigation data available. Run a review first.",
        "ready": "Ready for citation",
        "caution": "Proceed with caution",
        "blocked": "Not ready — action required",
        "run_header": "Recent Reviews",
        "obligations_header": "Scholarly Obligations",
        "manifest_header": "Dependency Freshness",
    },
    "biomedical": {
        "title": "Output Quality Contract",
        "subtitle": "Is this clinical analysis ready for reliance?",
        "empty": "No investigation data available. Run a clinical analysis first.",
        "ready": "Ready for reliance",
        "caution": "Proceed with caution",
        "blocked": "Not ready — action required",
        "run_header": "Recent Analyses",
        "obligations_header": "Clinical Obligations",
        "manifest_header": "Dependency Freshness",
    },
}


def _fmt_output_quality(data: dict, domain: str = "legal") -> str:
    labels = _OUTPUT_QUALITY_LABELS.get(domain, _OUTPUT_QUALITY_LABELS["legal"])
    if not isinstance(data, dict):
        return f"<div class='viz-empty'>{_escape(labels['empty'])}</div>"
    if err := _error_html(data):
        return err

    readiness = data.get("readiness", "unknown")
    blocker_count = data.get("blocker_count", 0)
    runs = data.get("runs", [])
    obligations = data.get("obligations", [])
    manifest_count = data.get("manifest_count", 0)
    manifest_fresh = data.get("manifest_fresh", False)
    stale_count = data.get("stale_manifest_count", 0)
    summary = data.get("summary", {})

    if not runs and not obligations:
        return f"<div class='viz-empty'>{_escape(labels['empty'])}</div>"

    readiness_colors = {
        "ready": ("#16a34a", labels["ready"]),
        "caution": ("#d97706", labels["caution"]),
        "blocked": ("#dc2626", labels["blocked"]),
    }
    r_color, r_label = readiness_colors.get(readiness, ("#6b7280", readiness))

    parts = [
        "<div class='viz-shell'>",
        f"<div class='viz-header'>"
        f"<strong>{_escape(labels['title'])}</strong> &mdash; "
        f"{_escape(labels['subtitle'])}</div>",
        f"<div style='padding:12px 16px;background:{r_color}11;"
        f"border-left:4px solid {r_color};margin:8px 0;border-radius:4px;'>"
        f"<span style='color:{r_color};font-weight:700;font-size:1.1em;'>"
        f"{'&#x2705;' if readiness == 'ready' else '&#x26A0;&#xFE0F;' if readiness == 'caution' else '&#x274C;'} "
        f"{_escape(r_label)}</span>",
    ]
    if blocker_count:
        parts.append(
            f"<span style='margin-left:12px;color:#6b7280;font-size:0.9em;'>"
            f"{blocker_count} blocker{'s' if blocker_count != 1 else ''}</span>"
        )
    avg_cov = summary.get("avg_coverage", 0)
    if isinstance(avg_cov, (int, float)) and math.isfinite(avg_cov):
        parts.append(
            f"<span style='margin-left:12px;color:#6b7280;font-size:0.9em;'>"
            f"Avg coverage: {int(avg_cov * 100)}%</span>"
        )
    parts.append("</div>")

    if obligations:
        parts.append(
            f"<h4 style='margin:12px 0 6px 0;font-size:0.95em;'>"
            f"{_escape(labels['obligations_header'])}</h4>"
        )
        for ob in obligations:
            if not isinstance(ob, dict):
                continue
            satisfied = ob.get("satisfied", False)
            severity = ob.get("severity", "medium")
            name = _escape(str(ob.get("name", ""))[:120])
            item_count = ob.get("item_count", 0)
            if satisfied:
                icon = "&#x2705;"
                bg = "#f0fdf4"
                border = "#16a34a"
            else:
                sev_colors = {"high": "#dc2626", "medium": "#d97706", "low": "#6b7280"}
                border = sev_colors.get(severity, "#6b7280")
                icon = "&#x274C;" if severity == "high" else "&#x26A0;&#xFE0F;"
                bg = "#fef2f2" if severity == "high" else "#fffbeb"
            count_badge = ""
            if not satisfied and item_count:
                count_badge = (
                    f" <span style='background:{border};color:white;padding:1px 6px;"
                    f"border-radius:10px;font-size:0.8em;'>{item_count}</span>"
                )
            parts.append(
                f"<div style='padding:6px 12px;margin:3px 0;background:{bg};"
                f"border-left:3px solid {border};border-radius:3px;font-size:0.9em;'>"
                f"{icon} {name}{count_badge}</div>"
            )

    if runs:
        parts.append(
            f"<h4 style='margin:12px 0 6px 0;font-size:0.95em;'>"
            f"{_escape(labels['run_header'])}</h4>"
            "<div class='matrix-wrap'><table class='analytics-table'>"
            "<thead><tr><th>Status</th><th>Query</th><th>Mode</th>"
            "<th>Events</th><th>Cache Reuse</th><th>Summary</th></tr></thead><tbody>"
        )
        for r in runs[:5]:
            if not isinstance(r, dict):
                continue
            status = r.get("status", "unknown")
            status_colors = {
                "completed": "#16a34a", "running": "#2563eb",
                "failed": "#dc2626", "stopped": "#d97706",
            }
            s_color = status_colors.get(status, "#6b7280")
            query = _escape((r.get("query") or "")[:50])
            mode = _escape(r.get("research_mode", ""))
            events = r.get("event_count", 0)
            reuse = r.get("cache_reuse_rate", 0)
            if isinstance(reuse, (int, float)) and math.isfinite(reuse):
                reuse_pct = f"{int(reuse * 100)}%"
            else:
                reuse_pct = "—"
            comp_summary = _escape((r.get("completion_summary") or "")[:80])
            parts.append(
                f"<tr><td><span style='color:{s_color};font-weight:600;'>"
                f"{_escape(status)}</span></td>"
                f"<td>{query}</td><td>{mode}</td>"
                f"<td style='text-align:center;'>{events}</td>"
                f"<td style='text-align:center;'>{reuse_pct}</td>"
                f"<td style='font-size:0.85em;color:#6b7280;'>{comp_summary}</td></tr>"
            )
        parts.append("</tbody></table></div>")

    manifest_icon = "&#x2705;" if manifest_fresh else "&#x26A0;&#xFE0F;"
    parts.append(
        f"<h4 style='margin:12px 0 6px 0;font-size:0.95em;'>"
        f"{_escape(labels['manifest_header'])}</h4>"
        f"<div style='font-size:0.9em;color:#6b7280;'>"
        f"{manifest_icon} {manifest_count} manifest(s) tracked"
    )
    if stale_count:
        parts.append(
            f" &middot; <span style='color:#dc2626;font-weight:600;'>"
            f"{stale_count} stale</span>"
        )
    parts.append("</div></div>")
    return "\n".join(parts)


_DELIVERABLE_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Deliverable Builder",
        "subtitle": "Assemble a memo, letter, or outline from verified evidence",
        "empty": "No issues available. Run an investigation first.",
        "gate_ready": "Reliance gate passed — deliverable can be produced",
        "gate_blocked": "Reliance gate blocked — resolve blockers before producing deliverables",
        "gate_caution": "Reliance gate: caution — review blockers before proceeding",
        "issues_header": "Issues in Scope",
        "verified": "verified facts",
        "sources": "source documents",
    },
    "finance": {
        "title": "Report Builder",
        "subtitle": "Assemble a report or briefing from verified analysis",
        "empty": "No theses available. Run an analysis first.",
        "gate_ready": "Reliance gate passed — report can be produced",
        "gate_blocked": "Reliance gate blocked — resolve blockers first",
        "gate_caution": "Reliance gate: caution — review blockers before proceeding",
        "issues_header": "Theses in Scope",
        "verified": "verified data points",
        "sources": "source documents",
    },
    "coding": {
        "title": "Assessment Builder",
        "subtitle": "Assemble an assessment from verified findings",
        "empty": "No tasks available. Run an assessment first.",
        "gate_ready": "Quality gate passed — assessment can be produced",
        "gate_blocked": "Quality gate blocked — resolve issues first",
        "gate_caution": "Quality gate: caution — review issues before proceeding",
        "issues_header": "Tasks in Scope",
        "verified": "verified findings",
        "sources": "source files",
    },
    "academic_research": {
        "title": "Manuscript Builder",
        "subtitle": "Assemble a literature review from verified claims",
        "empty": "No questions available. Run a review first.",
        "gate_ready": "Citation gate passed — manuscript can be produced",
        "gate_blocked": "Citation gate blocked — resolve gaps first",
        "gate_caution": "Citation gate: caution — review gaps before proceeding",
        "issues_header": "Questions in Scope",
        "verified": "verified claims",
        "sources": "source papers",
    },
    "biomedical": {
        "title": "Clinical Summary Builder",
        "subtitle": "Assemble a clinical summary from verified findings",
        "empty": "No hypotheses available. Run a clinical analysis first.",
        "gate_ready": "Clinical gate passed — summary can be produced",
        "gate_blocked": "Clinical gate blocked — resolve blockers first",
        "gate_caution": "Clinical gate: caution — review blockers before proceeding",
        "issues_header": "Hypotheses in Scope",
        "verified": "verified findings",
        "sources": "source records",
    },
}


def _fmt_deliverable_workbench(data: dict, domain: str = "legal") -> str:
    labels = _DELIVERABLE_LABELS.get(domain, _DELIVERABLE_LABELS["legal"])
    if not isinstance(data, dict):
        return f"<div class='viz-empty'>{_escape(labels['empty'])}</div>"
    if err := _error_html(data):
        return err

    issues = data.get("issues", [])
    gate = data.get("reliance_gate", "unknown")
    blocker_count = data.get("blocker_count", 0)
    total_verified = data.get("total_verified_assertions", 0)
    total_sources = data.get("total_source_documents", 0)

    if not issues:
        return f"<div class='viz-empty'>{_escape(labels['empty'])}</div>"

    gate_configs = {
        "ready": ("#16a34a", labels["gate_ready"], "&#x2705;"),
        "caution": ("#d97706", labels["gate_caution"], "&#x26A0;&#xFE0F;"),
        "blocked": ("#dc2626", labels["gate_blocked"], "&#x274C;"),
    }
    g_color, g_label, g_icon = gate_configs.get(gate, ("#6b7280", gate, "&#x2753;"))

    parts = [
        "<div class='viz-shell'>",
        f"<div class='viz-header'><strong>{_escape(labels['title'])}</strong>"
        f" &mdash; {_escape(labels['subtitle'])}</div>",
        f"<div style='padding:10px 16px;background:{g_color}11;"
        f"border-left:4px solid {g_color};margin:8px 0;border-radius:4px;'>"
        f"<span style='color:{g_color};font-weight:700;'>{g_icon} {_escape(g_label)}</span>",
    ]
    if blocker_count:
        parts.append(
            f" <span style='color:#6b7280;font-size:0.9em;'>"
            f"({blocker_count} blocker{'s' if blocker_count != 1 else ''})</span>"
        )
    parts.append("</div>")

    parts.append(
        f"<div style='margin:8px 0;font-size:0.9em;color:#6b7280;'>"
        f"<strong>{len(issues)}</strong> issue(s) in scope &middot; "
        f"<strong>{total_verified}</strong> {_escape(labels['verified'])} &middot; "
        f"<strong>{total_sources}</strong> {_escape(labels['sources'])}</div>"
    )

    parts.append(
        f"<h4 style='margin:12px 0 6px 0;font-size:0.95em;'>"
        f"{_escape(labels['issues_header'])}</h4>"
    )

    for iss in issues:
        if not isinstance(iss, dict):
            continue
        title = _escape((iss.get("title") or "")[:120])
        iid = _escape((iss.get("issue_id") or "")[:40])
        v_count = iss.get("verified_assertion_count", 0)
        sources = iss.get("source_documents", [])
        mat = iss.get("materiality", 0)
        if isinstance(mat, (int, float)) and math.isfinite(mat):
            mat_pct = int(mat * 100)
        else:
            mat_pct = 0

        has_evidence = v_count > 0
        border_color = "#16a34a" if has_evidence else "#d97706"
        bg = "#f0fdf4" if has_evidence else "#fffbeb"

        parts.append(
            f"<div style='padding:8px 14px;margin:4px 0;background:{bg};"
            f"border-left:3px solid {border_color};border-radius:4px;'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
            f"<strong>{title}</strong>"
            f"<span style='font-size:0.8em;color:#6b7280;'>materiality: {mat_pct}%</span>"
            f"</div>"
            f"<div style='margin-top:4px;font-size:0.85em;color:#4b5563;'>"
            f"{'&#x2705;' if has_evidence else '&#x26A0;&#xFE0F;'} "
            f"{v_count} {_escape(labels['verified'])}"
        )
        if sources:
            src_list = ", ".join(_escape(s[:30]) for s in sources[:5])
            parts.append(f" &middot; Sources: {src_list}")
        parts.append("</div>")

        assertions = iss.get("verified_assertions", [])
        if assertions:
            parts.append("<div style='margin-top:4px;padding-left:12px;'>")
            for va in assertions[:5]:
                if not isinstance(va, dict):
                    continue
                prop = _escape((va.get("proposition") or "")[:120])
                bs = _escape(va.get("belief_state", ""))
                parts.append(
                    f"<div style='font-size:0.82em;color:#6b7280;margin:2px 0;'>"
                    f"&bull; {prop} <span style='color:#2563eb;'>({bs})</span></div>"
                )
            if len(assertions) > 5:
                parts.append(
                    f"<div style='font-size:0.8em;color:#9ca3af;'>"
                    f"… and {len(assertions) - 5} more</div>"
                )
            parts.append("</div>")

        parts.append("</div>")

    parts.append("</div>")
    return "\n".join(parts)


_SCENARIO_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Scenario Branches",
        "subtitle": "Explore alternative case theories with persistent what-if branches",
        "empty": "No scenario branches yet. Create one to explore a hypothetical.",
        "active": "Active",
        "archived": "Archived",
        "assumptions": "Assumptions",
        "created": "Created",
        "branch_header": "Active Branches",
        "notes": "Notes",
        "create_hint": "Enter a name and assumptions to create a new scenario branch.",
    },
    "finance": {
        "title": "Scenario Branches",
        "subtitle": "Explore alternative investment cases with persistent what-if branches",
        "empty": "No scenario branches yet. Create one to model a hypothetical.",
        "active": "Active",
        "archived": "Archived",
        "assumptions": "Assumptions",
        "created": "Created",
        "branch_header": "Active Branches",
        "notes": "Notes",
        "create_hint": "Enter a name and assumptions to create a new scenario branch.",
    },
    "coding": {
        "title": "Design Paths",
        "subtitle": "Explore alternative design paths with persistent what-if branches",
        "empty": "No design paths yet. Create one to explore a hypothetical.",
        "active": "Active",
        "archived": "Archived",
        "assumptions": "Assumptions",
        "created": "Created",
        "branch_header": "Active Paths",
        "notes": "Notes",
        "create_hint": "Enter a name and assumptions to create a new design path.",
    },
    "academic_research": {
        "title": "Hypothesis Branches",
        "subtitle": "Explore alternative hypotheses with persistent what-if branches",
        "empty": "No hypothesis branches yet. Create one to test an alternative.",
        "active": "Active",
        "archived": "Archived",
        "assumptions": "Assumptions",
        "created": "Created",
        "branch_header": "Active Hypotheses",
        "notes": "Notes",
        "create_hint": "Enter a name and assumptions to create a new hypothesis branch.",
    },
    "biomedical": {
        "title": "Clinical Interpretations",
        "subtitle": "Explore alternative clinical interpretations with what-if branches",
        "empty": "No clinical interpretation branches yet. Create one to explore a differential.",
        "active": "Active",
        "archived": "Archived",
        "assumptions": "Assumptions",
        "created": "Created",
        "branch_header": "Active Interpretations",
        "notes": "Notes",
        "create_hint": "Enter a name and assumptions to create a new interpretation branch.",
    },
}


def _fmt_scenario_workbench(data: dict, domain: str = "legal") -> str:
    labels = _SCENARIO_LABELS.get(domain, _SCENARIO_LABELS["legal"])
    if not isinstance(data, dict):
        return f"<div class='viz-empty'>{_escape(labels['empty'])}</div>"
    if err := _error_html(data):
        return err

    branches = data.get("branches", [])
    active_count = data.get("active_count", 0)
    archived_count = data.get("archived_count", 0)

    if not branches:
        return f"<div class='viz-empty'>{_escape(labels['empty'])}</div>"

    parts = [
        "<div class='viz-shell'>",
        f"<div class='viz-header'><strong>{_escape(labels['title'])}</strong>"
        f" &mdash; {_escape(labels['subtitle'])}</div>",
        f"<div style='margin:8px 0;font-size:0.9em;color:#6b7280;'>"
        f"<strong>{active_count}</strong> {_escape(labels['active'])} &middot; "
        f"<strong>{archived_count}</strong> {_escape(labels['archived'])}</div>",
        f"<h4 style='margin:12px 0 6px 0;font-size:0.95em;'>"
        f"{_escape(labels['branch_header'])}</h4>",
    ]

    for branch in branches:
        if not isinstance(branch, dict):
            continue
        name = _escape((branch.get("name") or "")[:80])
        bid = _escape((branch.get("id") or branch.get("branch_id") or "")[:40])
        status = _escape(branch.get("status", ""))
        created = _escape((branch.get("created_at") or "")[:19])
        notes = _escape((branch.get("notes") or "")[:200])
        assumptions = branch.get("assumptions", [])

        status_color = "#16a34a" if status == "active" else "#6b7280"
        parts.append(
            f"<div style='padding:8px 14px;margin:4px 0;background:#f8fafc;"
            f"border-left:3px solid {status_color};border-radius:4px;'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
            f"<strong>{name}</strong>"
            f"<span style='font-size:0.8em;color:{status_color};'>{status}</span>"
            f"</div>"
        )
        if created:
            parts.append(
                f"<div style='font-size:0.8em;color:#9ca3af;margin-top:2px;'>"
                f"{_escape(labels['created'])}: {created}</div>"
            )
        if assumptions and isinstance(assumptions, list):
            parts.append(
                f"<div style='margin-top:4px;font-size:0.85em;color:#4b5563;'>"
                f"<strong>{_escape(labels['assumptions'])}:</strong></div>"
                f"<div style='padding-left:12px;'>"
            )
            for assumption in assumptions[:5]:
                if isinstance(assumption, dict):
                    text = _escape((assumption.get("text") or assumption.get("assumption") or str(assumption))[:120])
                else:
                    text = _escape(str(assumption)[:120])
                parts.append(
                    f"<div style='font-size:0.82em;color:#6b7280;margin:2px 0;'>"
                    f"&bull; {text}</div>"
                )
            if len(assumptions) > 5:
                parts.append(
                    f"<div style='font-size:0.8em;color:#9ca3af;'>"
                    f"… and {len(assumptions) - 5} more</div>"
                )
            parts.append("</div>")
        if notes:
            parts.append(
                f"<div style='font-size:0.82em;color:#6b7280;margin-top:4px;'>"
                f"<em>{_escape(labels['notes'])}: {notes}</em></div>"
            )
        parts.append("</div>")

    parts.append("</div>")
    return "\n".join(parts)


_SCENARIO_COMPARE_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Scenario Comparison",
        "subtitle": "Side-by-side analysis of branch vs baseline matter state.",
        "belief_changes": "Belief State Changes",
        "suppressions": "Suppressed Items",
        "new_gaps": "New Gaps",
        "resolved_gaps": "Resolved Gaps",
        "new_assertions": "New Assertions",
        "assumptions": "Branch Assumptions",
        "empty": "Select a branch and click Compare to see the impact analysis.",
        "no_changes": "No changes — this branch has no deltas applied.",
    },
    "finance": {
        "title": "Scenario Comparison",
        "subtitle": "Side-by-side analysis of scenario vs base case.",
        "belief_changes": "Belief State Changes", "suppressions": "Suppressed Items",
        "new_gaps": "New Diligence Gaps", "resolved_gaps": "Resolved Gaps",
        "new_assertions": "New Claims", "assumptions": "Scenario Assumptions",
        "empty": "Select a scenario and click Compare to see the impact.",
        "no_changes": "No changes — this scenario has no deltas applied.",
    },
    "coding": {
        "title": "Path Comparison",
        "subtitle": "Side-by-side analysis of design path vs baseline.",
        "belief_changes": "Belief State Changes", "suppressions": "Suppressed Items",
        "new_gaps": "New Gaps", "resolved_gaps": "Resolved Gaps",
        "new_assertions": "New Claims", "assumptions": "Path Assumptions",
        "empty": "Select a path and click Compare to see the impact.",
        "no_changes": "No changes — this path has no deltas applied.",
    },
    "academic_research": {
        "title": "Hypothesis Comparison",
        "subtitle": "Side-by-side analysis of hypothesis vs baseline.",
        "belief_changes": "Belief State Changes", "suppressions": "Suppressed Items",
        "new_gaps": "New Evidence Gaps", "resolved_gaps": "Resolved Gaps",
        "new_assertions": "New Findings", "assumptions": "Hypothesis Assumptions",
        "empty": "Select a hypothesis and click Compare to see the impact.",
        "no_changes": "No changes — this hypothesis has no deltas applied.",
    },
    "biomedical": {
        "title": "Interpretation Comparison",
        "subtitle": "Side-by-side analysis of clinical interpretation vs baseline.",
        "belief_changes": "Belief State Changes", "suppressions": "Suppressed Items",
        "new_gaps": "New Evidence Gaps", "resolved_gaps": "Resolved Gaps",
        "new_assertions": "New Findings", "assumptions": "Interpretation Assumptions",
        "empty": "Select an interpretation and click Compare to see the impact.",
        "no_changes": "No changes — this interpretation has no deltas applied.",
    },
}


_SNAPSHOT_HISTORY_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Scenario Snapshot History",
        "subtitle": "Chronological record of branch evaluation snapshots.",
        "snapshot_id": "Snapshot",
        "delta_count": "Deltas evaluated",
        "created_at": "Computed at",
        "empty": "No snapshots recorded for this branch yet.",
    },
    "finance": {
        "title": "Scenario Snapshot History",
        "subtitle": "Chronological record of scenario evaluation snapshots.",
        "snapshot_id": "Snapshot",
        "delta_count": "Deltas evaluated",
        "created_at": "Computed at",
        "empty": "No snapshots recorded for this scenario yet.",
    },
    "coding": {
        "title": "Path Snapshot History",
        "subtitle": "Chronological record of design path evaluation snapshots.",
        "snapshot_id": "Snapshot",
        "delta_count": "Deltas evaluated",
        "created_at": "Computed at",
        "empty": "No snapshots recorded for this path yet.",
    },
    "academic_research": {
        "title": "Hypothesis Snapshot History",
        "subtitle": "Chronological record of hypothesis evaluation snapshots.",
        "snapshot_id": "Snapshot",
        "delta_count": "Deltas evaluated",
        "created_at": "Computed at",
        "empty": "No snapshots recorded for this hypothesis yet.",
    },
    "biomedical": {
        "title": "Interpretation Snapshot History",
        "subtitle": "Chronological record of interpretation evaluation snapshots.",
        "snapshot_id": "Snapshot",
        "delta_count": "Deltas evaluated",
        "created_at": "Computed at",
        "empty": "No snapshots recorded for this interpretation yet.",
    },
}


def _fmt_scenario_snapshot_history(data: dict, domain: str = "legal") -> str:
    L = _SNAPSHOT_HISTORY_LABELS.get(domain, _SNAPSHOT_HISTORY_LABELS["legal"])
    if not data or not isinstance(data, dict):
        return f"<div class='viz-empty'>{L['empty']}</div>"
    if err := _error_html(data):
        return err

    snapshots = data.get("snapshots", [])
    if not isinstance(snapshots, list) or not snapshots:
        return f"<div class='viz-empty'>{L['empty']}</div>"

    branch_id = _escape(str(data.get("branch_id", "")))
    count = len(snapshots)

    parts = [
        f"<h3 style='margin:0 0 8px 0;'>{_escape(L['title'])}</h3>",
        f"<div style='color:#666;font-size:0.9em;margin-bottom:8px;'>"
        f"{_escape(L['subtitle'])} Branch: {branch_id} · {count} snapshot{'s' if count != 1 else ''}</div>",
        "<table style='border-collapse:collapse;width:100%;font-size:0.9em;'>",
        f"<tr style='background:#f1f5f9;'>"
        f"<th style='text-align:left;padding:4px 8px;'>{_escape(L['snapshot_id'])}</th>"
        f"<th style='text-align:left;padding:4px 8px;'>{_escape(L['delta_count'])}</th>"
        f"<th style='text-align:left;padding:4px 8px;'>{_escape(L['created_at'])}</th></tr>",
    ]

    for snap in snapshots:
        if not isinstance(snap, dict):
            continue
        sid = _escape(str(snap.get("id", "?")))
        dc = snap.get("delta_count", 0)
        if not isinstance(dc, (int, float)) or not math.isfinite(float(dc)):
            dc = 0
        created = _escape(str(snap.get("created_at", "?")))
        parts.append(
            f"<tr style='border-bottom:1px solid #e2e8f0;'>"
            f"<td style='padding:4px 8px;font-family:monospace;font-size:0.85em;'>{sid[:12]}</td>"
            f"<td style='padding:4px 8px;'>{int(dc)}</td>"
            f"<td style='padding:4px 8px;'>{created}</td></tr>"
        )

    parts.append("</table>")
    return "\n".join(parts)


_SCENARIO_DELTA_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Scenario Delta Log",
        "subtitle": "Changes applied to this branch without mutating the baseline matter.",
        "empty": "No deltas applied to this branch yet.",
        "col_op": "Operation", "col_kind": "Target Type",
        "col_target": "Target ID", "col_time": "Applied",
        "col_by": "By",
    },
    "finance": {
        "title": "Scenario Delta Log",
        "subtitle": "Changes applied to this scenario without mutating the base case.",
        "empty": "No deltas applied to this scenario yet.",
        "col_op": "Operation", "col_kind": "Target Type",
        "col_target": "Target ID", "col_time": "Applied",
        "col_by": "By",
    },
    "coding": {
        "title": "Design Path Delta Log",
        "subtitle": "Changes applied to this path without mutating the baseline.",
        "empty": "No deltas applied to this design path yet.",
        "col_op": "Operation", "col_kind": "Target Type",
        "col_target": "Target ID", "col_time": "Applied",
        "col_by": "By",
    },
    "academic_research": {
        "title": "Hypothesis Delta Log",
        "subtitle": "Changes applied to this hypothesis without mutating the baseline.",
        "empty": "No deltas applied to this hypothesis yet.",
        "col_op": "Operation", "col_kind": "Target Type",
        "col_target": "Target ID", "col_time": "Applied",
        "col_by": "By",
    },
    "biomedical": {
        "title": "Interpretation Delta Log",
        "subtitle": "Changes applied to this interpretation without mutating the baseline.",
        "empty": "No deltas applied to this interpretation yet.",
        "col_op": "Operation", "col_kind": "Target Type",
        "col_target": "Target ID", "col_time": "Applied",
        "col_by": "By",
    },
}


def _fmt_scenario_deltas(deltas: list, domain: str = "legal") -> str:
    L = _SCENARIO_DELTA_LABELS.get(domain, _SCENARIO_DELTA_LABELS["legal"])
    if not deltas or not isinstance(deltas, list):
        return f"<div class='viz-empty'>{_escape(L['empty'])}</div>"

    safe = [d for d in deltas if isinstance(d, dict)]
    if not safe:
        return f"<div class='viz-empty'>{_escape(L['empty'])}</div>"

    _OP_COLORS = {
        "override_belief": "#2563eb",
        "suppress": "#dc2626",
        "add_gap": "#d97706",
        "resolve_gap": "#059669",
        "add_assertion": "#7c3aed",
        "assume": "#6b7280",
    }

    parts = [
        f"<h3 style='margin:0 0 8px 0;'>{_escape(L['title'])}</h3>",
        f"<div style='color:#666;font-size:0.9em;margin-bottom:8px;'>"
        f"{_escape(L['subtitle'])} {len(safe)} delta{'s' if len(safe) != 1 else ''}</div>",
        "<table style='border-collapse:collapse;width:100%;font-size:0.9em;'>",
        f"<tr style='background:#f1f5f9;'>"
        f"<th style='text-align:left;padding:4px 8px;'>{_escape(L['col_op'])}</th>"
        f"<th style='text-align:left;padding:4px 8px;'>{_escape(L['col_kind'])}</th>"
        f"<th style='text-align:left;padding:4px 8px;'>{_escape(L['col_target'])}</th>"
        f"<th style='text-align:left;padding:4px 8px;'>Detail</th>"
        f"<th style='text-align:left;padding:4px 8px;'>{_escape(L['col_time'])}</th>"
        f"<th style='text-align:left;padding:4px 8px;'>{_escape(L['col_by'])}</th></tr>",
    ]

    _PAYLOAD_SUMMARY_KEYS = ("new_belief", "description", "gap_type", "reason", "text")

    for d in safe:
        op = _escape(str(d.get("operation", "?")))
        kind = _escape(str(d.get("target_kind", "?")))
        tid = _escape(str(d.get("target_id", "?"))[:16])
        created = _escape(str(d.get("created_at", "?")))
        by = _escape(str(d.get("created_by", "user")))
        color = _OP_COLORS.get(d.get("operation", ""), "#6b7280")
        payload = d.get("payload", {})
        detail_parts = []
        if isinstance(payload, dict):
            for pk in _PAYLOAD_SUMMARY_KEYS:
                pv = payload.get(pk)
                if pv:
                    detail_parts.append(f"{_escape(pk)}: {_escape(str(pv)[:40])}")
        detail = "; ".join(detail_parts) if detail_parts else "—"
        parts.append(
            f"<tr style='border-bottom:1px solid #e2e8f0;'>"
            f"<td style='padding:4px 8px;'>"
            f"<span style='color:{color};font-weight:600;'>{op}</span></td>"
            f"<td style='padding:4px 8px;'>{kind}</td>"
            f"<td style='padding:4px 8px;font-family:monospace;font-size:0.85em;'>{tid}</td>"
            f"<td style='padding:4px 8px;font-size:0.85em;color:#4b5563;'>{detail}</td>"
            f"<td style='padding:4px 8px;'>{created}</td>"
            f"<td style='padding:4px 8px;'>{by}</td></tr>"
        )

    parts.append("</table>")
    return "\n".join(parts)


def _fmt_scenario_comparison(data: dict, domain: str = "legal") -> str:
    L = _SCENARIO_COMPARE_LABELS.get(domain, _SCENARIO_COMPARE_LABELS["legal"])
    if not data or not isinstance(data, dict):
        return f"<div class='viz-empty'>{_escape(L['empty'])}</div>"
    if err := _error_html(data):
        return err

    delta_count = _safe_int(data.get("delta_count", 0))
    if delta_count == 0:
        return f"<div class='viz-empty'>{_escape(L['no_changes'])}</div>"

    branch_name = _escape(str(data.get("branch_name", ""))[:80])
    status = _escape(str(data.get("branch_status", "")))

    header = (
        f"<div class='viz-shell'>"
        f"<div class='intel-panel-title'>{_escape(L['title'])}</div>"
        f"<div style='font-size:0.8em;color:#6b7280;margin-bottom:8px;'>{_escape(L['subtitle'])}</div>"
        f"<div style='display:flex;gap:16px;align-items:center;margin-bottom:12px;flex-wrap:wrap;'>"
        f"<div><strong>Branch:</strong> {branch_name}</div>"
        f"<div style='font-size:0.85em;color:#6b7280;'>Status: {status} | Deltas: {delta_count}</div>"
        f"</div>"
    )

    body = ""

    assumptions = data.get("assumptions", [])
    if isinstance(assumptions, list) and assumptions:
        body += f"<div style='margin-bottom:10px;'><strong>{_escape(L['assumptions'])}:</strong>"
        for a in assumptions[:10]:
            if not isinstance(a, dict):
                continue
            text = _escape(str(a.get("text") or a.get("assumption") or "")[:150])
            body += f"<div style='font-size:0.82em;color:#6b7280;padding-left:12px;'>&bull; {text}</div>"
        body += "</div>"

    belief_changes = data.get("belief_changes", [])
    if isinstance(belief_changes, list) and belief_changes:
        body += (
            f"<div style='margin-bottom:10px;'>"
            f"<strong style='color:#d97706;'>{_escape(L['belief_changes'])} ({len(belief_changes)}):</strong>"
        )
        for bc in belief_changes[:10]:
            if not isinstance(bc, dict):
                continue
            prop = _escape(str(bc.get("proposition", ""))[:150])
            bl = _escape(str(bc.get("baseline_belief", "")))
            br = _escape(str(bc.get("branch_belief", "")))
            body += (
                f"<div style='font-size:0.82em;padding:3px 0;border-bottom:1px solid #f3f4f6;'>"
                f"{prop} — <span style='color:#dc2626;'>{bl}</span>"
                f" → <span style='color:#16a34a;'>{br}</span></div>"
            )
        body += "</div>"

    suppressions = data.get("suppressions", [])
    if isinstance(suppressions, list) and suppressions:
        body += (
            f"<div style='margin-bottom:10px;'>"
            f"<strong style='color:#dc2626;'>{_escape(L['suppressions'])} ({len(suppressions)}):</strong>"
        )
        for s in suppressions[:10]:
            if not isinstance(s, dict):
                continue
            kind = _escape(str(s.get("target_kind", "")))
            label = _escape(str(s.get("label", s.get("target_id", "")))[:150])
            body += f"<div style='font-size:0.82em;color:#b91c1c;padding-left:12px;'>&#9888; [{kind}] {label}</div>"
        body += "</div>"

    new_gaps = data.get("new_gaps", [])
    if isinstance(new_gaps, list) and new_gaps:
        body += (
            f"<div style='margin-bottom:10px;'>"
            f"<strong style='color:#dc2626;'>{_escape(L['new_gaps'])} ({len(new_gaps)}):</strong>"
        )
        for g in new_gaps[:10]:
            if not isinstance(g, dict):
                continue
            desc = _escape(str(g.get("description", ""))[:150])
            gtype = _escape(str(g.get("gap_type", "")))
            body += f"<div style='font-size:0.82em;color:#b91c1c;padding-left:12px;'>&#9888; [{gtype}] {desc}</div>"
        body += "</div>"

    resolved_gaps = data.get("resolved_gaps", [])
    if isinstance(resolved_gaps, list) and resolved_gaps:
        body += (
            f"<div style='margin-bottom:10px;'>"
            f"<strong style='color:#16a34a;'>{_escape(L['resolved_gaps'])} ({len(resolved_gaps)}):</strong>"
        )
        for rg in resolved_gaps[:10]:
            if not isinstance(rg, dict):
                continue
            reason = _escape(str(rg.get("reason", ""))[:150])
            body += f"<div style='font-size:0.82em;color:#047857;padding-left:12px;'>&#10003; {reason}</div>"
        body += "</div>"

    new_assertions = data.get("new_assertions", [])
    if isinstance(new_assertions, list) and new_assertions:
        body += (
            f"<div style='margin-bottom:10px;'>"
            f"<strong style='color:#2563eb;'>{_escape(L['new_assertions'])} ({len(new_assertions)}):</strong>"
        )
        for na in new_assertions[:10]:
            if not isinstance(na, dict):
                continue
            prop = _escape(str(na.get("proposition", ""))[:150])
            bs = _escape(str(na.get("belief_state", "")))
            body += f"<div style='font-size:0.82em;padding-left:12px;'>+ {prop} [{bs}]</div>"
        body += "</div>"

    return f"{header}{body}</div>"


_ALTERNATIVE_THEORY_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Alternative Theories",
        "subtitle": "Competing interpretations derived from the matter graph",
        "empty": "No alternative theories yet — add assertions and evidence first.",
        "support": "Supporting",
        "attack": "Attacking",
        "assumptions": "Assumptions",
        "gaps": "Open Gaps",
        "confidence": "Confidence",
        "taint": "Tainted",
        "discriminators": "Key Discriminators",
        "promote": "Promote to Scenario",
    },
    "finance": {
        "title": "Competing Theses",
        "subtitle": "Alternative investment interpretations from the analysis model",
        "empty": "No competing theses yet — add positions and evidence first.",
        "support": "Supporting",
        "attack": "Contradicting",
        "assumptions": "Assumptions",
        "gaps": "Open Gaps",
        "confidence": "Confidence",
        "taint": "Tainted",
        "discriminators": "Key Questions",
        "promote": "Promote to Scenario",
    },
    "coding": {
        "title": "Alternative Interpretations",
        "subtitle": "Competing explanations from the analysis model",
        "empty": "No alternative interpretations yet — add findings first.",
        "support": "Supporting",
        "attack": "Contradicting",
        "assumptions": "Assumptions",
        "gaps": "Open Gaps",
        "confidence": "Confidence",
        "taint": "Tainted",
        "discriminators": "Key Questions",
        "promote": "Promote to Design Path",
    },
    "academic_research": {
        "title": "Rival Hypotheses",
        "subtitle": "Competing interpretations from the research model",
        "empty": "No rival hypotheses yet — add claims and evidence first.",
        "support": "Supporting",
        "attack": "Contradicting",
        "assumptions": "Assumptions",
        "gaps": "Open Gaps",
        "confidence": "Confidence",
        "taint": "Tainted",
        "discriminators": "Discriminating Questions",
        "promote": "Promote to Hypothesis Branch",
    },
    "biomedical": {
        "title": "Differential Diagnoses",
        "subtitle": "Competing clinical interpretations from the analysis model",
        "empty": "No differential diagnoses yet — add findings and evidence first.",
        "support": "Supporting",
        "attack": "Contradicting",
        "assumptions": "Assumptions",
        "gaps": "Open Gaps",
        "confidence": "Confidence",
        "taint": "Tainted",
        "discriminators": "Key Discriminators",
        "promote": "Promote to Clinical Branch",
    },
}


def _fmt_alternative_theories(data: dict, domain: str = "legal") -> str:
    L = _ALTERNATIVE_THEORY_LABELS.get(domain, _ALTERNATIVE_THEORY_LABELS["legal"])
    if not isinstance(data, dict):
        return f"<div class='viz-empty'>{_escape(L['empty'])}</div>"
    if err := _error_html(data):
        return err

    theories = data.get("theories", [])
    if not theories or not isinstance(theories, list):
        return f"<div class='viz-empty'>{_escape(L['empty'])}</div>"

    parts = [
        "<div class='viz-shell'>",
        f"<div class='viz-header'><strong>{_escape(L['title'])}</strong>"
        f" &mdash; {_escape(L['subtitle'])}</div>",
        f"<div style='margin:8px 0;font-size:0.9em;color:#6b7280;'>"
        f"<strong>{len(theories)}</strong> theories identified</div>",
    ]

    stance_colors = {
        "supporting": "#16a34a",
        "opposing": "#dc2626",
        "uncertain": "#d97706",
        "missing": "#6b7280",
    }

    for theory in theories:
        if not isinstance(theory, dict):
            continue
        label = _escape(str(theory.get("label", ""))[:120])
        stance = str(theory.get("stance", "")).lower()
        color = stance_colors.get(stance, "#6b7280")

        supporting = _safe_int(theory.get("supporting_assertions", 0))
        attacking_count = _safe_int(theory.get("attacking_assertions", 0))
        assumptions_count = _safe_int(theory.get("assumptions", 0))
        gaps_count = _safe_int(theory.get("open_gaps", 0))
        conf = theory.get("confidence_range", [0.0, 0.0])
        if not isinstance(conf, list) or len(conf) < 2:
            conf = [0.0, 0.0]
        try:
            c_lo = float(conf[0])
            c_hi = float(conf[1])
            if not (math.isfinite(c_lo) and math.isfinite(c_hi)):
                c_lo, c_hi = 0.0, 0.0
        except (TypeError, ValueError):
            c_lo, c_hi = 0.0, 0.0

        taint = theory.get("taint_summary", {})
        taint_count = 0
        if isinstance(taint, dict):
            tc = taint.get("tainted_assertion_count", 0)
            try:
                taint_count = int(tc) if isinstance(tc, (int, float)) and math.isfinite(float(tc)) else 0
            except (TypeError, ValueError):
                taint_count = 0

        src_mix = theory.get("source_role_mix", {})
        if not isinstance(src_mix, dict):
            src_mix = {}
        mix_parts = []
        for role, count in list(src_mix.items())[:5]:
            mix_parts.append(f"{_escape(str(role))}: {_safe_int(count)}")
        mix_str = ", ".join(mix_parts) if mix_parts else "—"

        parts.append(
            f"<div style='padding:10px 14px;margin:6px 0;background:#f8fafc;"
            f"border-left:4px solid {color};border-radius:4px;'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
            f"<strong style='font-size:0.95em;'>{label}</strong>"
            f"<span class='pill' style='background:{color};color:#fff;font-size:0.75em;'>"
            f"{_escape(stance)}</span></div>"
            f"<div style='display:flex;gap:16px;margin-top:6px;font-size:0.85em;color:#4b5563;'>"
            f"<span>{_escape(L['support'])}: <strong>{supporting}</strong></span>"
            f"<span>{_escape(L['attack'])}: <strong>{attacking_count}</strong></span>"
            f"<span>{_escape(L['assumptions'])}: <strong>{assumptions_count}</strong></span>"
            f"<span>{_escape(L['gaps'])}: <strong>{gaps_count}</strong></span>"
            f"<span>{_escape(L['confidence'])}: {c_lo:.0%}–{c_hi:.0%}</span>"
            f"</div>"
        )

        if taint_count > 0:
            parts.append(
                f"<div style='font-size:0.8em;color:#dc2626;margin-top:4px;'>"
                f"⚠ {taint_count} {_escape(L['taint'])} assertion(s)</div>"
            )

        if mix_str != "—":
            parts.append(
                f"<div style='font-size:0.8em;color:#6b7280;margin-top:2px;'>"
                f"Sources: {mix_str}</div>"
            )

        disc_qs = theory.get("discriminator_questions", [])
        if isinstance(disc_qs, list) and disc_qs:
            parts.append(
                f"<div style='margin-top:6px;font-size:0.82em;color:#4b5563;'>"
                f"<strong>{_escape(L['discriminators'])}:</strong></div>"
            )
            for q in disc_qs[:5]:
                if not isinstance(q, str):
                    continue
                parts.append(
                    f"<div style='padding-left:12px;font-size:0.8em;color:#6b7280;margin:2px 0;'>"
                    f"&bull; {_escape(q[:200])}</div>"
                )

        parts.append("</div>")

    parts.append("</div>")
    return "\n".join(parts)


_MANIFEST_INSPECTOR_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Dependency Manifests",
        "subtitle": "Exact evidence and freshness behind each analysis output",
        "empty": "No dependency manifests recorded yet.",
        "fresh": "Fresh",
        "stale": "Stale",
        "taint_blocked": "Taint-Blocked",
        "policy_limited": "Policy-Limited",
        "unknown": "Unknown",
        "objects": "Objects",
        "negative": "Negative Deps",
        "profile": "Profile",
        "audience": "Audience",
    },
    "finance": {
        "title": "Dependency Manifests",
        "subtitle": "Exact evidence and freshness behind each analysis output",
        "empty": "No dependency manifests recorded yet.",
        "fresh": "Fresh",
        "stale": "Stale",
        "taint_blocked": "Taint-Blocked",
        "policy_limited": "Policy-Limited",
        "unknown": "Unknown",
        "objects": "Objects",
        "negative": "Negative Deps",
        "profile": "Profile",
        "audience": "Audience",
    },
    "coding": {
        "title": "Dependency Manifests",
        "subtitle": "Exact evidence and freshness behind each analysis output",
        "empty": "No dependency manifests recorded yet.",
        "fresh": "Fresh",
        "stale": "Stale",
        "taint_blocked": "Taint-Blocked",
        "policy_limited": "Policy-Limited",
        "unknown": "Unknown",
        "objects": "Objects",
        "negative": "Negative Deps",
        "profile": "Profile",
        "audience": "Audience",
    },
    "academic_research": {
        "title": "Dependency Manifests",
        "subtitle": "Exact evidence and freshness behind each analysis output",
        "empty": "No dependency manifests recorded yet.",
        "fresh": "Fresh",
        "stale": "Stale",
        "taint_blocked": "Taint-Blocked",
        "policy_limited": "Policy-Limited",
        "unknown": "Unknown",
        "objects": "Objects",
        "negative": "Negative Deps",
        "profile": "Profile",
        "audience": "Audience",
    },
    "biomedical": {
        "title": "Dependency Manifests",
        "subtitle": "Exact evidence and freshness behind each analysis output",
        "empty": "No dependency manifests recorded yet.",
        "fresh": "Fresh",
        "stale": "Stale",
        "taint_blocked": "Taint-Blocked",
        "policy_limited": "Policy-Limited",
        "unknown": "Unknown",
        "objects": "Objects",
        "negative": "Negative Deps",
        "profile": "Profile",
        "audience": "Audience",
    },
}


def _fmt_manifest_inspector(data: dict, domain: str = "legal") -> str:
    L = _MANIFEST_INSPECTOR_LABELS.get(domain, _MANIFEST_INSPECTOR_LABELS["legal"])
    if not isinstance(data, dict):
        return f"<div class='viz-empty'>{_escape(L['empty'])}</div>"
    if err := _error_html(data):
        return err

    manifests = data.get("manifests", [])
    if not manifests or not isinstance(manifests, list):
        return f"<div class='viz-empty'>{_escape(L['empty'])}</div>"

    total = data.get("total_count", 0)
    fresh = data.get("fresh_count", 0)
    stale = data.get("stale_count", 0)

    def _safe_count(v: object) -> int:
        try:
            iv = int(v)  # type: ignore[arg-type]
            return iv if isinstance(iv, int) and iv >= 0 else 0
        except (TypeError, ValueError):
            return 0

    total = _safe_count(total)
    fresh = _safe_count(fresh)
    stale = _safe_count(stale)

    status_styles = {
        "fresh": ("pill-green", L["fresh"]),
        "stale": ("pill-red", L["stale"]),
        "taint_blocked": ("pill-orange", L["taint_blocked"]),
        "policy_limited": ("pill-orange", L["policy_limited"]),
        "unknown": ("pill-neutral", L["unknown"]),
    }

    parts = [
        "<div class='viz-shell'>",
        f"<div class='viz-header'><strong>{_escape(L['title'])}</strong>"
        f" &mdash; {_escape(L['subtitle'])}</div>",
        f"<div style='margin:8px 0;font-size:0.9em;color:#6b7280;'>"
        f"<strong>{total}</strong> total &middot; "
        f"<span class='pill pill-green'>{fresh} {_escape(L['fresh'])}</span> "
        f"<span class='pill pill-red'>{stale} {_escape(L['stale'])}</span>"
        f"</div>",
    ]

    for m in manifests:
        if not isinstance(m, dict):
            continue
        mh = _escape(str(m.get("manifest_hash", ""))[:16])
        purpose = _escape(str(m.get("purpose", ""))[:80])
        created = _escape(str(m.get("created_at", ""))[:19])
        status = str(m.get("status", "unknown")).lower()
        pill_cls, pill_label = status_styles.get(status, ("pill-neutral", "Unknown"))
        profile_id = _escape(str(m.get("domain_profile_id", ""))[:30])
        audience = _escape(str(m.get("policy_audience", ""))[:20])
        obj_count = _safe_count(m.get("object_dependency_count", 0))
        neg_count = _safe_count(m.get("negative_dependency_count", 0))

        border_color = "#16a34a" if status == "fresh" else "#dc2626" if status == "stale" else "#d97706"

        parts.append(
            f"<div style='padding:10px 14px;margin:6px 0;background:#f8fafc;"
            f"border-left:4px solid {border_color};border-radius:4px;'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
            f"<strong style='font-family:monospace;font-size:0.9em;'>{mh}…</strong>"
            f"<span class='pill {pill_cls}'>{_escape(pill_label)}</span></div>"
        )

        if purpose:
            parts.append(
                f"<div style='font-size:0.85em;color:#4b5563;margin-top:4px;'>{purpose}</div>"
            )

        parts.append(
            f"<div style='display:flex;gap:16px;margin-top:6px;font-size:0.82em;color:#6b7280;'>"
            f"<span>{_escape(L['objects'])}: <strong>{obj_count}</strong></span>"
            f"<span>{_escape(L['negative'])}: <strong>{neg_count}</strong></span>"
            f"<span>{_escape(L['profile'])}: {profile_id}</span>"
            f"<span>{_escape(L['audience'])}: {audience}</span>"
            f"</div>"
        )

        if created:
            parts.append(
                f"<div style='font-size:0.78em;color:#9ca3af;margin-top:2px;'>{created}</div>"
            )

        obj_by_kind = m.get("consumed_objects_by_kind", {})
        if isinstance(obj_by_kind, dict) and obj_by_kind:
            kind_parts = []
            for kind, cnt in list(obj_by_kind.items())[:8]:
                kind_parts.append(f"{_escape(str(kind))}: {_safe_count(cnt)}")
            parts.append(
                f"<div style='font-size:0.8em;color:#6b7280;margin-top:4px;'>"
                f"By kind: {', '.join(kind_parts)}</div>"
            )

        stale_ns = m.get("stale_namespaces", [])
        if isinstance(stale_ns, list) and stale_ns:
            parts.append(
                f"<div style='margin-top:6px;font-size:0.8em;color:#dc2626;'>"
                f"<strong>Stale namespaces:</strong></div>"
            )
            for ns in stale_ns[:5]:
                if not isinstance(ns, dict):
                    continue
                reason = _escape(str(ns.get("reason", ""))[:200])
                parts.append(
                    f"<div style='padding-left:12px;font-size:0.78em;color:#b91c1c;margin:2px 0;'>"
                    f"&bull; {reason}</div>"
                )

        stale_reasons = m.get("stale_reasons", [])
        if isinstance(stale_reasons, list) and stale_reasons and status != "fresh":
            non_ns_reasons = [r for r in stale_reasons if isinstance(r, str) and "namespace" not in r]
            for r in non_ns_reasons[:3]:
                parts.append(
                    f"<div style='font-size:0.78em;color:#b91c1c;padding-left:12px;'>"
                    f"⚠ {_escape(r[:200])}</div>"
                )

        parts.append("</div>")

    parts.append("</div>")
    return "\n".join(parts)


# ── Freshness Report ────────────────────────────────────────────────
_FRESHNESS_REPORT_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Namespace Freshness",
        "subtitle": "Revision state of every matter intelligence namespace",
        "empty": "No freshness data available. Load a matter first.",
        "fresh": "Fresh", "missing": "Not Initialized", "stale": "Stale",
        "hot": "Hot-Answerable", "not_hot": "Pending Work",
        "active_runs": "Active Investigations", "last_update": "Last Updated",
    },
    "finance": {
        "title": "Data Freshness",
        "subtitle": "Revision state of every analysis namespace",
        "empty": "No freshness data available. Load a matter first.",
        "fresh": "Current", "missing": "Not Initialized", "stale": "Stale",
        "hot": "Ready", "not_hot": "Pending Updates",
        "active_runs": "Active Analyses", "last_update": "Last Updated",
    },
    "coding": {
        "title": "Index Freshness",
        "subtitle": "Revision state of every codebase analysis namespace",
        "empty": "No freshness data. Load a project first.",
        "fresh": "Current", "missing": "Not Indexed", "stale": "Stale",
        "hot": "Index Current", "not_hot": "Indexing Required",
        "active_runs": "Active Scans", "last_update": "Last Updated",
    },
    "academic_research": {
        "title": "Corpus Freshness",
        "subtitle": "Revision state of every research namespace",
        "empty": "No freshness data. Load a corpus first.",
        "fresh": "Current", "missing": "Not Populated", "stale": "Stale",
        "hot": "Corpus Ready", "not_hot": "Updates Pending",
        "active_runs": "Active Reviews", "last_update": "Last Updated",
    },
    "biomedical": {
        "title": "Evidence Freshness",
        "subtitle": "Revision state of every clinical evidence namespace",
        "empty": "No freshness data. Load a dataset first.",
        "fresh": "Current", "missing": "Not Populated", "stale": "Stale",
        "hot": "Evidence Ready", "not_hot": "Updates Pending",
        "active_runs": "Active Analyses", "last_update": "Last Updated",
    },
}


def _fmt_freshness_report(data: dict, domain: str = "legal") -> str:
    L = _FRESHNESS_REPORT_LABELS.get(domain, _FRESHNESS_REPORT_LABELS["legal"])
    if not data or not isinstance(data, dict):
        return f"<div class='viz-empty'>{_escape(L['empty'])}</div>"
    if err := _error_html(data):
        return err

    namespaces = data.get("namespaces", [])
    if not isinstance(namespaces, list) or not namespaces:
        return f"<div class='viz-empty'>{_escape(L['empty'])}</div>"

    stale_list = data.get("stale_namespaces", [])
    is_hot = data.get("is_hot_answerable", False)
    active_runs = int(data.get("active_run_count", 0)) if isinstance(data.get("active_run_count"), (int, float)) else 0
    last_update = _escape(str(data.get("last_update_at", ""))[:19])
    ns_count = len(namespaces)
    fresh_count = sum(1 for ns in namespaces if isinstance(ns, dict) and ns.get("state") == "fresh")
    missing_count = ns_count - fresh_count

    hot_pill = (
        f"<span class='pill pill-green'>{_escape(L['hot'])}</span>"
        if is_hot
        else f"<span class='pill pill-orange'>{_escape(L['not_hot'])}</span>"
    )

    parts = [
        "<div class='viz-shell'>",
        f"<div class='viz-header'><strong>{_escape(L['title'])}</strong>"
        f" &mdash; {_escape(L['subtitle'])}</div>",
        f"<div style='margin:8px 0;font-size:0.9em;color:#6b7280;'>"
        f"{hot_pill} &middot; "
        f"<span class='pill pill-green'>{fresh_count} {_escape(L['fresh'])}</span> "
        f"<span class='pill pill-neutral'>{missing_count} {_escape(L['missing'])}</span>"
        f"</div>",
    ]

    if active_runs > 0:
        parts.append(
            f"<div style='font-size:0.85em;color:#6b7280;margin-bottom:6px;'>"
            f"{_escape(L['active_runs'])}: <strong>{active_runs}</strong></div>"
        )
    if last_update:
        parts.append(
            f"<div style='font-size:0.82em;color:#9ca3af;margin-bottom:8px;'>"
            f"{_escape(L['last_update'])}: {last_update}</div>"
        )

    parts.append(
        "<table style='width:100%;border-collapse:collapse;font-size:0.85em;'>"
        "<tr style='border-bottom:1px solid #e5e7eb;'>"
        "<th style='text-align:left;padding:4px 8px;color:#6b7280;'>Namespace</th>"
        "<th style='text-align:center;padding:4px 8px;color:#6b7280;'>Rev</th>"
        "<th style='text-align:center;padding:4px 8px;color:#6b7280;'>State</th>"
        "<th style='text-align:right;padding:4px 8px;color:#6b7280;'>Updated</th>"
        "</tr>"
    )

    for ns in namespaces:
        if not isinstance(ns, dict):
            continue
        name = _escape(str(ns.get("namespace", "")))
        rev = int(ns.get("revision", 0)) if isinstance(ns.get("revision"), (int, float)) else 0
        state = str(ns.get("state", "missing")).lower()
        updated = _escape(str(ns.get("updated_at", "—"))[:19])

        if state == "fresh":
            pill = f"<span class='pill pill-green'>{_escape(L['fresh'])}</span>"
        elif state == "stale":
            pill = f"<span class='pill pill-red'>{_escape(L['stale'])}</span>"
        else:
            pill = f"<span class='pill pill-neutral'>{_escape(L['missing'])}</span>"

        parts.append(
            f"<tr style='border-bottom:1px solid #f3f4f6;'>"
            f"<td style='padding:4px 8px;font-family:monospace;font-size:0.9em;'>{name}</td>"
            f"<td style='text-align:center;padding:4px 8px;'>{rev}</td>"
            f"<td style='text-align:center;padding:4px 8px;'>{pill}</td>"
            f"<td style='text-align:right;padding:4px 8px;font-size:0.85em;color:#9ca3af;'>{updated}</td>"
            f"</tr>"
        )

    parts.append("</table></div>")
    return "\n".join(parts)


# ── Reasoning Cache Stats ────────────────────────────────────────────
_CACHE_STATS_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Reasoning Cache",
        "subtitle": "Hit rates by reasoning stage — reuse of prior analysis",
        "empty": "No cache data available. Run an investigation first.",
        "stage": "Stage", "entries": "Entries", "hits": "Hits",
        "hit_rate": "Hit Rate", "trust_rev": "Trust Revision",
        "overall": "Overall Hit Rate",
    },
    "finance": {
        "title": "Analysis Cache",
        "subtitle": "Hit rates by analysis stage — reuse of prior computations",
        "empty": "No cache data. Run an analysis first.",
        "stage": "Stage", "entries": "Entries", "hits": "Hits",
        "hit_rate": "Hit Rate", "trust_rev": "Trust Revision",
        "overall": "Overall Hit Rate",
    },
    "coding": {
        "title": "Reasoning Cache",
        "subtitle": "Hit rates by analysis stage — reuse of prior scans",
        "empty": "No cache data. Run a scan first.",
        "stage": "Stage", "entries": "Entries", "hits": "Hits",
        "hit_rate": "Hit Rate", "trust_rev": "Trust Revision",
        "overall": "Overall Hit Rate",
    },
    "academic_research": {
        "title": "Reasoning Cache",
        "subtitle": "Hit rates by research stage — reuse of prior synthesis",
        "empty": "No cache data. Run a review first.",
        "stage": "Stage", "entries": "Entries", "hits": "Hits",
        "hit_rate": "Hit Rate", "trust_rev": "Trust Revision",
        "overall": "Overall Hit Rate",
    },
    "biomedical": {
        "title": "Evidence Cache",
        "subtitle": "Hit rates by reasoning stage — reuse of prior evidence review",
        "empty": "No cache data. Run an analysis first.",
        "stage": "Stage", "entries": "Entries", "hits": "Hits",
        "hit_rate": "Hit Rate", "trust_rev": "Trust Revision",
        "overall": "Overall Hit Rate",
    },
}


def _fmt_cache_stats(data: dict, domain: str = "legal") -> str:
    L = _CACHE_STATS_LABELS.get(domain, _CACHE_STATS_LABELS["legal"])
    if not data or not isinstance(data, dict):
        return f"<div class='viz-empty'>{_escape(L['empty'])}</div>"
    if err := _error_html(data):
        return err

    stages = data.get("stages", [])
    if not isinstance(stages, list) or not stages:
        return f"<div class='viz-empty'>{_escape(L['empty'])}</div>"

    trust_rev = int(data.get("trust_revision", 0)) if isinstance(data.get("trust_revision"), (int, float)) else 0
    total_entries = int(data.get("total_entries", 0)) if isinstance(data.get("total_entries"), (int, float)) else 0
    overall_rate = float(data.get("overall_hit_rate", 0.0)) if isinstance(data.get("overall_hit_rate"), (int, float)) else 0.0

    parts = [
        "<div class='viz-shell'>",
        f"<div class='viz-header'><strong>{_escape(L['title'])}</strong>"
        f" &mdash; {_escape(L['subtitle'])}</div>",
        f"<div style='margin:8px 0;font-size:0.9em;color:#6b7280;'>"
        f"<span class='pill pill-neutral'>{_escape(L['trust_rev'])}: {trust_rev}</span> &middot; "
        f"<span class='pill pill-neutral'>{total_entries} {_escape(L['entries'])}</span> &middot; "
        f"<span class='pill pill-green'>{_escape(L['overall'])}: {overall_rate:.1%}</span>"
        f"</div>",
        "<table style='width:100%;border-collapse:collapse;font-size:0.85em;'>"
        "<tr style='border-bottom:1px solid #e5e7eb;'>"
        f"<th style='text-align:left;padding:4px 8px;color:#6b7280;'>{_escape(L['stage'])}</th>"
        f"<th style='text-align:center;padding:4px 8px;color:#6b7280;'>{_escape(L['entries'])}</th>"
        f"<th style='text-align:center;padding:4px 8px;color:#6b7280;'>{_escape(L['hits'])}</th>"
        f"<th style='text-align:right;padding:4px 8px;color:#6b7280;'>{_escape(L['hit_rate'])}</th>"
        "</tr>",
    ]

    for stage in stages:
        if not isinstance(stage, dict):
            continue
        name = _escape(str(stage.get("stage", "")))
        entries = int(stage.get("total_entries", 0)) if isinstance(stage.get("total_entries"), (int, float)) else 0
        hits = int(stage.get("hit_count", 0)) if isinstance(stage.get("hit_count"), (int, float)) else 0
        rate = float(stage.get("hit_rate", 0.0)) if isinstance(stage.get("hit_rate"), (int, float)) else 0.0

        if rate >= 0.5:
            rate_pill = f"<span class='pill pill-green'>{rate:.1%}</span>"
        elif rate > 0:
            rate_pill = f"<span class='pill pill-orange'>{rate:.1%}</span>"
        else:
            rate_pill = f"<span class='pill pill-neutral'>{rate:.1%}</span>"

        parts.append(
            f"<tr style='border-bottom:1px solid #f3f4f6;'>"
            f"<td style='padding:4px 8px;font-family:monospace;font-size:0.9em;'>{name}</td>"
            f"<td style='text-align:center;padding:4px 8px;'>{entries}</td>"
            f"<td style='text-align:center;padding:4px 8px;'>{hits}</td>"
            f"<td style='text-align:right;padding:4px 8px;'>{rate_pill}</td>"
            f"</tr>"
        )

    parts.append("</table></div>")
    return "\n".join(parts)


# ── LLM Usage Summary ────────────────────────────────────────────────
_LLM_USAGE_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "LLM Usage",
        "subtitle": "Token consumption and cost across reasoning tiers",
        "empty": "No LLM usage data. Run an investigation first.",
        "requests": "Requests", "success": "Successful", "failed": "Failed",
        "input": "Input Tokens", "output": "Output Tokens",
        "cache_read": "Cache-Read Tokens", "thinking": "Thinking Tokens",
        "cost": "Est. Cost", "tier": "Model Tier",
    },
    "finance": {
        "title": "LLM Usage",
        "subtitle": "Token consumption and cost across analysis tiers",
        "empty": "No usage data. Run an analysis first.",
        "requests": "Requests", "success": "Successful", "failed": "Failed",
        "input": "Input Tokens", "output": "Output Tokens",
        "cache_read": "Cache-Read Tokens", "thinking": "Thinking Tokens",
        "cost": "Est. Cost", "tier": "Model Tier",
    },
    "coding": {
        "title": "LLM Usage",
        "subtitle": "Token consumption and cost across scan tiers",
        "empty": "No usage data. Run a scan first.",
        "requests": "Requests", "success": "Successful", "failed": "Failed",
        "input": "Input Tokens", "output": "Output Tokens",
        "cache_read": "Cache-Read Tokens", "thinking": "Thinking Tokens",
        "cost": "Est. Cost", "tier": "Model Tier",
    },
    "academic_research": {
        "title": "LLM Usage",
        "subtitle": "Token consumption and cost across research tiers",
        "empty": "No usage data. Run a review first.",
        "requests": "Requests", "success": "Successful", "failed": "Failed",
        "input": "Input Tokens", "output": "Output Tokens",
        "cache_read": "Cache-Read Tokens", "thinking": "Thinking Tokens",
        "cost": "Est. Cost", "tier": "Model Tier",
    },
    "biomedical": {
        "title": "LLM Usage",
        "subtitle": "Token consumption and cost across evidence tiers",
        "empty": "No usage data. Run an analysis first.",
        "requests": "Requests", "success": "Successful", "failed": "Failed",
        "input": "Input Tokens", "output": "Output Tokens",
        "cache_read": "Cache-Read Tokens", "thinking": "Thinking Tokens",
        "cost": "Est. Cost", "tier": "Model Tier",
    },
}


def _fmt_llm_usage(data: dict, domain: str = "legal") -> str:
    L = _LLM_USAGE_LABELS.get(domain, _LLM_USAGE_LABELS["legal"])
    if not data or not isinstance(data, dict):
        return f"<div class='viz-empty'>{_escape(L['empty'])}</div>"
    if err := _error_html(data):
        return err

    req_count = int(data.get("request_count", 0)) if isinstance(data.get("request_count"), (int, float)) else 0
    if req_count == 0:
        return f"<div class='viz-empty'>{_escape(L['empty'])}</div>"

    success = int(data.get("successful_requests", 0)) if isinstance(data.get("successful_requests"), (int, float)) else 0
    failed = int(data.get("failed_requests", 0)) if isinstance(data.get("failed_requests"), (int, float)) else 0
    input_tok = int(data.get("input_tokens", 0)) if isinstance(data.get("input_tokens"), (int, float)) else 0
    output_tok = int(data.get("output_tokens", 0)) if isinstance(data.get("output_tokens"), (int, float)) else 0
    cache_tok = int(data.get("cache_read_tokens", 0)) if isinstance(data.get("cache_read_tokens"), (int, float)) else 0
    thinking_tok = int(data.get("thinking_tokens", 0)) if isinstance(data.get("thinking_tokens"), (int, float)) else 0
    cost = float(data.get("estimated_cost_usd", 0.0)) if isinstance(data.get("estimated_cost_usd"), (int, float)) else 0.0

    fail_pill = (
        f" <span class='pill pill-red'>{failed} {_escape(L['failed'])}</span>"
        if failed > 0 else ""
    )

    parts = [
        "<div class='viz-shell'>",
        f"<div class='viz-header'><strong>{_escape(L['title'])}</strong>"
        f" &mdash; {_escape(L['subtitle'])}</div>",
        f"<div style='margin:8px 0;font-size:0.9em;color:#6b7280;'>"
        f"<span class='pill pill-green'>{success} {_escape(L['success'])}</span>"
        f"{fail_pill} &middot; "
        f"<span class='pill pill-neutral'>{_escape(L['cost'])}: ${cost:.4f}</span>"
        f"</div>",
        "<div style='display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px;'>",
        _metric_card(L["input"], f"{input_tok:,}"),
        _metric_card(L["output"], f"{output_tok:,}"),
        _metric_card(L["cache_read"], f"{cache_tok:,}"),
        _metric_card(L["thinking"], f"{thinking_tok:,}"),
        "</div>",
    ]

    by_tier = data.get("by_tier", {})
    if isinstance(by_tier, dict) and by_tier:
        parts.append(
            "<table style='width:100%;border-collapse:collapse;font-size:0.85em;'>"
            "<tr style='border-bottom:1px solid #e5e7eb;'>"
            f"<th style='text-align:left;padding:4px 8px;color:#6b7280;'>{_escape(L['tier'])}</th>"
            f"<th style='text-align:center;padding:4px 8px;color:#6b7280;'>{_escape(L['requests'])}</th>"
            f"<th style='text-align:center;padding:4px 8px;color:#6b7280;'>{_escape(L['input'])}</th>"
            f"<th style='text-align:center;padding:4px 8px;color:#6b7280;'>{_escape(L['output'])}</th>"
            f"<th style='text-align:right;padding:4px 8px;color:#6b7280;'>{_escape(L['cost'])}</th>"
            "</tr>"
        )
        for tier_name, tier_data in sorted(by_tier.items()):
            if not isinstance(tier_data, dict):
                continue
            t_req = int(tier_data.get("requests", 0)) if isinstance(tier_data.get("requests"), (int, float)) else 0
            t_in = int(tier_data.get("input_tokens", 0)) if isinstance(tier_data.get("input_tokens"), (int, float)) else 0
            t_out = int(tier_data.get("output_tokens", 0)) if isinstance(tier_data.get("output_tokens"), (int, float)) else 0
            t_cost = float(tier_data.get("estimated_cost_usd", 0.0)) if isinstance(tier_data.get("estimated_cost_usd"), (int, float)) else 0.0
            parts.append(
                f"<tr style='border-bottom:1px solid #f3f4f6;'>"
                f"<td style='padding:4px 8px;font-family:monospace;font-size:0.9em;'>{_escape(str(tier_name))}</td>"
                f"<td style='text-align:center;padding:4px 8px;'>{t_req:,}</td>"
                f"<td style='text-align:center;padding:4px 8px;'>{t_in:,}</td>"
                f"<td style='text-align:center;padding:4px 8px;'>{t_out:,}</td>"
                f"<td style='text-align:right;padding:4px 8px;'>${t_cost:.4f}</td>"
                f"</tr>"
            )
        parts.append("</table>")

    parts.append("</div>")
    return "\n".join(parts)


# ── Steering Impact Preview ──────────────────────────────────────────
_IMPACT_PREVIEW_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Steering Impact Preview",
        "subtitle": "Project the effect of a correction before committing it.",
        "before": "Current State",
        "after": "Projected State",
        "coverage": "Issue Coverage",
        "gaps": "Open Gaps",
        "contradictions": "Contradictions",
        "disputed": "Disputed Assertions",
        "readiness": "Readiness",
        "followups": "Recommended Follow-ups",
        "empty": "Select a steering action and click Preview to see projected impact.",
        "invalid": "Invalid action — check warnings below.",
    },
    "finance": {
        "title": "Action Impact Preview",
        "subtitle": "Project the effect of a correction before committing it.",
        "before": "Current Position",
        "after": "Projected Position",
        "coverage": "Thesis Coverage",
        "gaps": "Open Diligence Gaps",
        "contradictions": "Contradictions",
        "disputed": "Disputed Claims",
        "readiness": "Readiness",
        "followups": "Recommended Follow-ups",
        "empty": "Select an action and click Preview to see projected impact.",
        "invalid": "Invalid action — check warnings below.",
    },
    "coding": {
        "title": "Action Impact Preview",
        "subtitle": "Project the effect of a correction before committing it.",
        "before": "Current State",
        "after": "Projected State",
        "coverage": "Requirement Coverage",
        "gaps": "Open Verification Gaps",
        "contradictions": "Conflicts",
        "disputed": "Disputed Claims",
        "readiness": "Readiness",
        "followups": "Recommended Follow-ups",
        "empty": "Select an action and click Preview to see projected impact.",
        "invalid": "Invalid action — check warnings below.",
    },
    "academic_research": {
        "title": "Action Impact Preview",
        "subtitle": "Project the effect of a correction before committing it.",
        "before": "Current State",
        "after": "Projected State",
        "coverage": "Question Coverage",
        "gaps": "Open Evidence Gaps",
        "contradictions": "Contradictions",
        "disputed": "Disputed Findings",
        "readiness": "Readiness",
        "followups": "Recommended Follow-ups",
        "empty": "Select an action and click Preview to see projected impact.",
        "invalid": "Invalid action — check warnings below.",
    },
    "biomedical": {
        "title": "Action Impact Preview",
        "subtitle": "Project the effect of a correction before committing it.",
        "before": "Current State",
        "after": "Projected State",
        "coverage": "Mechanism Coverage",
        "gaps": "Open Evidence Gaps",
        "contradictions": "Contradictions",
        "disputed": "Disputed Findings",
        "readiness": "Readiness",
        "followups": "Recommended Follow-ups",
        "empty": "Select an action and click Preview to see projected impact.",
        "invalid": "Invalid action — check warnings below.",
    },
}


def _fmt_impact_preview(data: dict, domain: str = "legal") -> str:
    L = _IMPACT_PREVIEW_LABELS.get(domain, _IMPACT_PREVIEW_LABELS["legal"])

    if not data:
        return f"<div class='viz-empty'>{_escape(L['empty'])}</div>"
    if err := _error_html(data):
        return err

    if not data.get("valid", False):
        warnings = data.get("warnings", [])
        warn_html = "".join(
            f"<div style='color:#dc2626;font-size:0.85em;'>&#9888; {_escape(str(w)[:200])}</div>"
            for w in warnings if isinstance(w, str)
        )
        return (
            f"<div class='viz-shell'><div class='intel-panel-title'>{_escape(L['title'])}</div>"
            f"<div style='color:#dc2626;font-weight:600;'>{_escape(L['invalid'])}</div>"
            f"{warn_html}</div>"
        )

    before = data.get("before", {})
    after = data.get("after", {})
    deltas = data.get("deltas", {})

    if not isinstance(before, dict):
        before = {}
    if not isinstance(after, dict):
        after = {}
    if not isinstance(deltas, dict):
        deltas = {}

    def _safe_float(v: object) -> float:
        try:
            fv = float(v)  # type: ignore[arg-type]
            return fv if math.isfinite(fv) else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _safe_int(v: object) -> int:
        try:
            iv = int(v)  # type: ignore[arg-type]
            return iv if isinstance(iv, int) and iv >= 0 else 0
        except (TypeError, ValueError):
            return 0

    def _delta_badge(before_val: object, after_val: object, lower_is_better: bool = True) -> str:
        bv = _safe_float(before_val)
        av = _safe_float(after_val)
        diff = av - bv
        if abs(diff) < 0.0001:
            return "<span style='color:#6b7280;'>&#8212;</span>"
        improved = (diff < 0) if lower_is_better else (diff > 0)
        color = "#16a34a" if improved else "#dc2626"
        arrow = "&#9660;" if diff < 0 else "&#9650;"
        return f"<span style='color:{color};font-weight:600;'>{arrow} {abs(diff):.2f}</span>"

    def _metric_row(label: str, bv: object, av: object, is_pct: bool = False, lower_is_better: bool = True) -> str:
        if is_pct:
            b_str = f"{_safe_float(bv) * 100:.1f}%"
            a_str = f"{_safe_float(av) * 100:.1f}%"
        else:
            b_str = str(_safe_int(bv))
            a_str = str(_safe_int(av))
        badge = _delta_badge(bv, av, lower_is_better=lower_is_better)
        return (
            f"<tr><td>{_escape(label)}</td>"
            f"<td style='text-align:center;'>{b_str}</td>"
            f"<td style='text-align:center;'>{a_str}</td>"
            f"<td style='text-align:center;'>{badge}</td></tr>"
        )

    rows = (
        _metric_row(L["coverage"], before.get("issue_coverage_avg"), after.get("issue_coverage_avg"), is_pct=True, lower_is_better=False)
        + _metric_row(L["gaps"], before.get("open_gap_count"), after.get("open_gap_count"))
        + _metric_row(L["contradictions"], before.get("contradiction_count"), after.get("contradiction_count"))
        + _metric_row(L["disputed"], before.get("disputed_count"), after.get("disputed_count"))
    )

    before_readiness = _escape(str(before.get("readiness", "unknown")))
    after_readiness = _escape(str(after.get("readiness", "unknown")))
    readiness_color = "#16a34a" if after_readiness == "good" else "#d97706"
    rows += (
        f"<tr><td>{_escape(L['readiness'])}</td>"
        f"<td style='text-align:center;'>{before_readiness}</td>"
        f"<td style='text-align:center;color:{readiness_color};font-weight:600;'>{after_readiness}</td>"
        f"<td></td></tr>"
    )

    action_type = _escape(str(data.get("action_type", "")))

    table = (
        f"<div class='viz-shell'>"
        f"<div class='intel-panel-title'>{_escape(L['title'])}</div>"
        f"<div style='font-size:0.8em;color:#6b7280;margin-bottom:8px;'>"
        f"{_escape(L['subtitle'])} Action: <strong>{action_type}</strong></div>"
        f"<div class='table-wrap'><table class='viz-table'>"
        f"<thead><tr><th>Metric</th><th>{_escape(L['before'])}</th>"
        f"<th>{_escape(L['after'])}</th><th>Delta</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )

    affected_objs = deltas.get("affected_objectives", [])
    affected_asns = deltas.get("affected_assertions", [])
    if isinstance(affected_objs, list) and affected_objs:
        table += (
            f"<div style='margin-top:8px;font-size:0.8em;color:#6b7280;'>"
            f"Affected objectives: {len(affected_objs)}</div>"
        )
    if isinstance(affected_asns, list) and affected_asns:
        table += (
            f"<div style='font-size:0.8em;color:#6b7280;'>"
            f"Affected assertions: {len(affected_asns)}</div>"
        )

    followups = data.get("recommended_followups", [])
    if isinstance(followups, list) and followups:
        fu_items = "".join(
            f"<li>{_escape(str(f)[:200])}</li>"
            for f in followups if isinstance(f, str)
        )
        table += (
            f"<div style='margin-top:10px;'>"
            f"<div class='viz-subtitle'>{_escape(L['followups'])}</div>"
            f"<ul style='margin:4px 0;padding-left:18px;font-size:0.85em;'>{fu_items}</ul></div>"
        )

    warnings = data.get("warnings", [])
    if isinstance(warnings, list) and warnings:
        for w in warnings:
            if isinstance(w, str):
                table += (
                    f"<div style='color:#d97706;font-size:0.8em;margin-top:4px;'>"
                    f"&#9888; {_escape(str(w)[:200])}</div>"
                )

    table += "</div>"
    return table


# ── Domain Investigation Readiness ───────────────────────────────────
_DOMAIN_READINESS_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Domain Investigation Readiness",
        "subtitle": "Cross-domain acceptance gate for investigation quality.",
        "ready": "Ready",
        "partial": "Partial",
        "blocked": "Blocked",
        "overall": "Overall Readiness",
        "profile": "Profile",
        "status": "Status",
        "detection": "Detection",
        "source_roles": "Source Calibration",
        "assertions": "Assertion Quality",
        "coverage": "Coverage",
        "quant": "Quant",
        "gaps": "Gaps",
        "steering": "Steering",
        "repairs": "Recommended Repairs",
        "cross_domain": "Cross-Domain Findings",
        "empty": "No readiness data. Load a matter and run an investigation first.",
    },
    "finance": {
        "title": "Domain Investigation Readiness",
        "subtitle": "Cross-domain acceptance gate for analysis quality.",
        "ready": "Ready", "partial": "Partial", "blocked": "Blocked",
        "overall": "Overall Readiness", "profile": "Profile", "status": "Status",
        "detection": "Detection", "source_roles": "Source Calibration",
        "assertions": "Assertion Quality", "coverage": "Coverage",
        "quant": "Quant", "gaps": "Gaps", "steering": "Steering",
        "repairs": "Recommended Repairs", "cross_domain": "Cross-Domain Findings",
        "empty": "No readiness data. Load a matter and run an investigation first.",
    },
    "coding": {
        "title": "Domain Investigation Readiness",
        "subtitle": "Cross-domain acceptance gate for analysis quality.",
        "ready": "Ready", "partial": "Partial", "blocked": "Blocked",
        "overall": "Overall Readiness", "profile": "Profile", "status": "Status",
        "detection": "Detection", "source_roles": "Source Calibration",
        "assertions": "Assertion Quality", "coverage": "Coverage",
        "quant": "Metrics", "gaps": "Gaps", "steering": "Steering",
        "repairs": "Recommended Repairs", "cross_domain": "Cross-Domain Findings",
        "empty": "No readiness data. Load a matter and run an investigation first.",
    },
    "academic_research": {
        "title": "Domain Investigation Readiness",
        "subtitle": "Cross-domain acceptance gate for research quality.",
        "ready": "Ready", "partial": "Partial", "blocked": "Blocked",
        "overall": "Overall Readiness", "profile": "Profile", "status": "Status",
        "detection": "Detection", "source_roles": "Source Calibration",
        "assertions": "Assertion Quality", "coverage": "Coverage",
        "quant": "Quant", "gaps": "Gaps", "steering": "Steering",
        "repairs": "Recommended Repairs", "cross_domain": "Cross-Domain Findings",
        "empty": "No readiness data. Load a matter and run an investigation first.",
    },
    "biomedical": {
        "title": "Domain Investigation Readiness",
        "subtitle": "Cross-domain acceptance gate for clinical quality.",
        "ready": "Ready", "partial": "Partial", "blocked": "Blocked",
        "overall": "Overall Readiness", "profile": "Profile", "status": "Status",
        "detection": "Detection", "source_roles": "Source Calibration",
        "assertions": "Assertion Quality", "coverage": "Coverage",
        "quant": "Quant", "gaps": "Gaps", "steering": "Steering",
        "repairs": "Recommended Repairs", "cross_domain": "Cross-Domain Findings",
        "empty": "No readiness data. Load a matter and run an investigation first.",
    },
}


def _fmt_domain_readiness(data: dict, domain: str = "legal") -> str:
    L = _DOMAIN_READINESS_LABELS.get(domain, _DOMAIN_READINESS_LABELS["legal"])

    if not data or not isinstance(data, dict):
        return f"<div class='viz-empty'>{_escape(L['empty'])}</div>"
    if err := _error_html(data):
        return err

    profiles = data.get("profiles", [])
    if not isinstance(profiles, list) or not profiles:
        return f"<div class='viz-empty'>{_escape(L['empty'])}</div>"

    overall = _escape(str(data.get("overall_status", "unknown")))
    status_colors = {"ready": "#16a34a", "partial": "#d97706", "blocked": "#dc2626"}
    overall_color = status_colors.get(data.get("overall_status", ""), "#6b7280")

    ready_count = sum(1 for p in profiles if isinstance(p, dict) and p.get("status") == "ready")
    partial_count = sum(1 for p in profiles if isinstance(p, dict) and p.get("status") == "partial")
    blocked_count = sum(1 for p in profiles if isinstance(p, dict) and p.get("status") == "blocked")

    def _check(d: dict, key: str = "pass") -> str:
        if not isinstance(d, dict):
            return "&#10060;"
        return "&#9989;" if d.get(key, False) else "&#10060;"

    header = (
        f"<div class='viz-shell'>"
        f"<div class='intel-panel-title'>{_escape(L['title'])}</div>"
        f"<div style='font-size:0.8em;color:#6b7280;margin-bottom:10px;'>{_escape(L['subtitle'])}</div>"
        f"<div style='display:flex;gap:16px;align-items:center;margin-bottom:12px;'>"
        f"<div style='font-size:1.1em;font-weight:700;color:{overall_color};'>"
        f"{_escape(L['overall'])}: {overall}</div>"
        f"<div style='font-size:0.85em;color:#6b7280;'>"
        f"{_escape(L['ready'])}: {ready_count} | {_escape(L['partial'])}: {partial_count} | "
        f"{_escape(L['blocked'])}: {blocked_count}</div>"
        f"</div>"
    )

    rows = ""
    for p in profiles:
        if not isinstance(p, dict):
            continue
        pid = _escape(str(p.get("profile_id", "?"))[:30])
        pstatus = str(p.get("status", "unknown"))
        pcolor = status_colors.get(pstatus, "#6b7280")
        sr = p.get("source_role_calibration", {})
        aq = p.get("assertion_quality", {})
        oc = p.get("objective_coverage", {})
        qc = p.get("quantitative_coverage", {})
        gm = p.get("gap_modeling", {})
        st = p.get("steering_readiness", {})
        dd = p.get("domain_detection", {})
        det_conf = 0.0
        if isinstance(dd, dict):
            try:
                det_conf = float(dd.get("confidence", 0.0))
            except (TypeError, ValueError):
                det_conf = 0.0
        if not math.isfinite(det_conf):
            det_conf = 0.0

        rows += (
            f"<tr>"
            f"<td><strong>{pid}</strong></td>"
            f"<td style='color:{pcolor};font-weight:600;'>{_escape(pstatus)}</td>"
            f"<td style='text-align:center;'>{det_conf:.0%}</td>"
            f"<td style='text-align:center;'>{_check(sr)}</td>"
            f"<td style='text-align:center;'>{_check(aq)}</td>"
            f"<td style='text-align:center;'>{_check(oc)}</td>"
            f"<td style='text-align:center;'>{_check(qc)}</td>"
            f"<td style='text-align:center;'>{_check(gm)}</td>"
            f"<td style='text-align:center;'>{_check(st)}</td>"
            f"</tr>"
        )

    table = (
        f"<div class='table-wrap'><table class='viz-table'>"
        f"<thead><tr>"
        f"<th>{_escape(L['profile'])}</th>"
        f"<th>{_escape(L['status'])}</th>"
        f"<th>{_escape(L['detection'])}</th>"
        f"<th>{_escape(L['source_roles'])}</th>"
        f"<th>{_escape(L['assertions'])}</th>"
        f"<th>{_escape(L['coverage'])}</th>"
        f"<th>{_escape(L['quant'])}</th>"
        f"<th>{_escape(L['gaps'])}</th>"
        f"<th>{_escape(L['steering'])}</th>"
        f"</tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )

    repairs_html = ""
    for p in profiles:
        if not isinstance(p, dict):
            continue
        reps = p.get("recommended_repairs", [])
        if isinstance(reps, list) and reps:
            pid = _escape(str(p.get("profile_id", "?"))[:30])
            items = "".join(
                f"<li>{_escape(str(r)[:200])}</li>"
                for r in reps if isinstance(r, str)
            )
            repairs_html += (
                f"<div style='margin-top:6px;'>"
                f"<strong>{pid}:</strong>"
                f"<ul style='margin:2px 0;padding-left:18px;font-size:0.85em;'>{items}</ul></div>"
            )

    if repairs_html:
        repairs_html = (
            f"<div style='margin-top:12px;'>"
            f"<div class='viz-subtitle'>{_escape(L['repairs'])}</div>"
            f"{repairs_html}</div>"
        )

    cross_domain = data.get("cross_domain_findings", [])
    cross_html = ""
    if isinstance(cross_domain, list) and cross_domain:
        cd_items = ""
        for finding in cross_domain:
            if not isinstance(finding, dict):
                continue
            sev = _escape(str(finding.get("severity", ""))[:20])
            msg = _escape(str(finding.get("message", ""))[:200])
            kind = _escape(str(finding.get("kind", ""))[:40])
            sev_color = "#dc2626" if sev == "high" else "#d97706"
            cd_items += (
                f"<div style='font-size:0.85em;padding:4px 0;'>"
                f"<span style='color:{sev_color};font-weight:600;'>[{sev}]</span> "
                f"<span style='color:#6b7280;'>{kind}</span> — {msg}</div>"
            )
        cross_html = (
            f"<div style='margin-top:12px;'>"
            f"<div class='viz-subtitle'>{_escape(L['cross_domain'])}</div>"
            f"{cd_items}</div>"
        )

    return f"{header}{table}{repairs_html}{cross_html}</div>"


# ── Issue Brief Compiler ─────────────────────────────────────────────
_ISSUE_BRIEF_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Issue Brief",
        "subtitle": "Structured analysis compiled from the matter model.",
        "reliance": "Reliance Gate",
        "sections": "Issue Sections",
        "assertions": "Assertions",
        "supporting": "Supporting",
        "attacking": "Attacking",
        "gaps": "Proof Gaps",
        "contradictions": "Contradictions",
        "sources": "Source Documents",
        "empty": "No issue brief data. Run an investigation to populate the matter model.",
        "no_sections": "No issues found. Run an investigation to generate issue analysis.",
    },
    "finance": {
        "title": "Analytical Memo",
        "subtitle": "Structured analysis compiled from the matter model.",
        "reliance": "Reliance Gate", "sections": "Thesis Sections",
        "assertions": "Claims", "supporting": "Corroborating",
        "attacking": "Contradicting", "gaps": "Diligence Gaps",
        "contradictions": "Contradictions", "sources": "Source Filings",
        "empty": "No analytical memo data. Run an investigation first.",
        "no_sections": "No theses found. Run an investigation first.",
    },
    "coding": {
        "title": "Investigation Report",
        "subtitle": "Structured analysis compiled from the matter model.",
        "reliance": "Reliance Gate", "sections": "Requirement Sections",
        "assertions": "Claims", "supporting": "Supporting",
        "attacking": "Conflicting", "gaps": "Verification Gaps",
        "contradictions": "Conflicts", "sources": "Source Files",
        "empty": "No report data. Run an investigation first.",
        "no_sections": "No requirements found. Run an investigation first.",
    },
    "academic_research": {
        "title": "Research Brief",
        "subtitle": "Structured analysis compiled from the matter model.",
        "reliance": "Reliance Gate", "sections": "Question Sections",
        "assertions": "Findings", "supporting": "Supporting",
        "attacking": "Contradicting", "gaps": "Evidence Gaps",
        "contradictions": "Contradictions", "sources": "Source Papers",
        "empty": "No research brief data. Run an investigation first.",
        "no_sections": "No research questions found. Run an investigation first.",
    },
    "biomedical": {
        "title": "Evidence Summary",
        "subtitle": "Structured analysis compiled from the matter model.",
        "reliance": "Reliance Gate", "sections": "Mechanism Sections",
        "assertions": "Findings", "supporting": "Supporting",
        "attacking": "Contradicting", "gaps": "Evidence Gaps",
        "contradictions": "Contradictions", "sources": "Source Reports",
        "empty": "No evidence summary data. Run an investigation first.",
        "no_sections": "No mechanisms found. Run an investigation first.",
    },
}


def _fmt_issue_brief(data: dict, domain: str = "legal") -> str:
    L = _ISSUE_BRIEF_LABELS.get(domain, _ISSUE_BRIEF_LABELS["legal"])

    if not data or not isinstance(data, dict):
        return f"<div class='viz-empty'>{_escape(L['empty'])}</div>"
    if err := _error_html(data):
        return err

    sections = data.get("sections", [])
    if not isinstance(sections, list) or not sections:
        return f"<div class='viz-empty'>{_escape(L['no_sections'])}</div>"

    reliance = _escape(str(data.get("reliance_gate", "unknown")))
    gate_colors = {"ready": "#16a34a", "caution": "#d97706", "blocked": "#dc2626"}
    gate_color = gate_colors.get(data.get("reliance_gate", ""), "#6b7280")

    total_a = _safe_int(data.get("total_assertions", 0))
    total_g = _safe_int(data.get("total_gaps", 0))
    total_c = _safe_int(data.get("total_contradictions", 0))
    total_s = _safe_int(data.get("total_source_documents", 0))

    header = (
        f"<div class='viz-shell'>"
        f"<div class='intel-panel-title'>{_escape(L['title'])}</div>"
        f"<div style='font-size:0.8em;color:#6b7280;margin-bottom:8px;'>{_escape(L['subtitle'])}</div>"
        f"<div style='display:flex;gap:16px;align-items:center;margin-bottom:12px;flex-wrap:wrap;'>"
        f"<div><strong>{_escape(L['reliance'])}:</strong> "
        f"<span style='color:{gate_color};font-weight:600;'>{reliance}</span></div>"
        f"<div style='font-size:0.85em;color:#6b7280;'>"
        f"{_escape(L['assertions'])}: {total_a} | "
        f"{_escape(L['gaps'])}: {total_g} | "
        f"{_escape(L['contradictions'])}: {total_c} | "
        f"{_escape(L['sources'])}: {total_s}</div>"
        f"</div>"
    )

    sections_html = ""
    for sec in sections:
        if not isinstance(sec, dict):
            continue
        title = _escape(str(sec.get("title", "Untitled"))[:200])
        materiality = 0.0
        try:
            mv = float(sec.get("materiality", 0.0))
            materiality = mv if math.isfinite(mv) else 0.0
        except (TypeError, ValueError):
            pass

        sup_count = _safe_int(sec.get("supporting_count", 0))
        atk_count = _safe_int(sec.get("attacking_count", 0))

        sections_html += (
            f"<div style='border:1px solid var(--border-color-primary,#e5e7eb);"
            f"border-radius:6px;padding:10px;margin-bottom:8px;'>"
            f"<div style='font-weight:600;margin-bottom:4px;'>{title}</div>"
            f"<div style='font-size:0.8em;color:#6b7280;margin-bottom:6px;'>"
            f"Materiality: {materiality:.0%} | "
            f"{_escape(L['supporting'])}: {sup_count} | "
            f"{_escape(L['attacking'])}: {atk_count}</div>"
        )

        assertions = sec.get("assertions", [])
        if isinstance(assertions, list) and assertions:
            sections_html += "<div style='margin-bottom:6px;'>"
            for a in assertions[:10]:
                if not isinstance(a, dict):
                    continue
                prop = _escape(str(a.get("proposition", ""))[:200])
                belief = _escape(str(a.get("belief_state", "")))
                edge = str(a.get("edge_type", "supports"))
                edge_color = "#16a34a" if edge == "supports" else "#dc2626"
                roles = a.get("source_roles", [])
                role_str = ", ".join(_escape(str(r)) for r in roles[:3]) if isinstance(roles, list) else ""
                sections_html += (
                    f"<div style='font-size:0.82em;padding:3px 0;border-bottom:1px solid #f3f4f6;'>"
                    f"<span style='color:{edge_color};'>&#9679;</span> {prop}"
                    f" <span style='color:#9ca3af;font-size:0.85em;'>[{belief}]</span>"
                )
                if role_str:
                    sections_html += f" <span style='color:#6b7280;font-size:0.8em;'>({role_str})</span>"
                sections_html += "</div>"
            sections_html += "</div>"

        gaps = sec.get("gaps", [])
        if isinstance(gaps, list) and gaps:
            sections_html += (
                f"<div style='margin-top:4px;font-size:0.8em;color:#dc2626;'>"
                f"<strong>{_escape(L['gaps'])} ({len(gaps)}):</strong></div>"
            )
            for g in gaps[:5]:
                if not isinstance(g, dict):
                    continue
                gdesc = _escape(str(g.get("description", ""))[:150])
                gtype = _escape(str(g.get("gap_type", ""))[:30])
                sections_html += (
                    f"<div style='font-size:0.78em;color:#b91c1c;padding-left:12px;'>"
                    f"&#9888; [{gtype}] {gdesc}</div>"
                )

        contradictions = sec.get("contradictions", [])
        if isinstance(contradictions, list) and contradictions:
            sections_html += (
                f"<div style='margin-top:4px;font-size:0.8em;color:#d97706;'>"
                f"<strong>{_escape(L['contradictions'])} ({len(contradictions)}):</strong></div>"
            )
            for c in contradictions[:3]:
                if not isinstance(c, dict):
                    continue
                ap = _escape(str(c.get("attacker_prop", ""))[:100])
                dp = _escape(str(c.get("attacked_prop", ""))[:100])
                sections_html += (
                    f"<div style='font-size:0.78em;color:#92400e;padding-left:12px;'>"
                    f"&#9650; &ldquo;{ap}&rdquo; vs &ldquo;{dp}&rdquo;</div>"
                )

        source_docs = sec.get("source_documents", [])
        if isinstance(source_docs, list) and source_docs:
            doc_list = ", ".join(_escape(str(d)[:60]) for d in source_docs[:5] if isinstance(d, str))
            sections_html += (
                f"<div style='margin-top:4px;font-size:0.78em;color:#6b7280;'>"
                f"{_escape(L['sources'])}: {doc_list}</div>"
            )

        sections_html += "</div>"

    return f"{header}{sections_html}</div>"


def _fmt_quant_panel(
    payment_recon: dict,
    invoice_chain: list,
    amount_conflicts: list,
    damages: list,
    domain: str = "legal",
) -> str:
    L = _QUANT_PANEL_LABELS.get(domain, _QUANT_PANEL_LABELS["legal"])
    has_recon = bool(payment_recon and payment_recon.get("invoiced") is not None)
    has_invoice_chain = bool(invoice_chain)
    has_amount_conflicts = bool(amount_conflicts)
    has_damages = bool(damages)
    if not has_recon and not has_invoice_chain and not has_amount_conflicts and not has_damages:
        return f"<div class='viz-empty'>{L['empty']}</div>"

    cards: list[str] = []
    if has_recon:
        cards.extend(
            [
                _metric_card(L["invoiced"], _fmt_money_short(payment_recon.get("invoiced", 0.0))),
                _metric_card(L["paid"], _fmt_money_short(payment_recon.get("paid", 0.0)), tone="green"),
                _metric_card(
                    L["disputed"],
                    _fmt_money_short(payment_recon.get("disputed", 0.0)),
                    tone="amber",
                ),
                _metric_card(
                    L["exposure"],
                    _fmt_money_short(payment_recon.get("exposure", 0.0)),
                    tone="red",
                ),
            ]
        )

    max_amount = max(
        (_safe_float(row.get("claimed_amount", 0.0)) for row in damages if isinstance(row, dict)),
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
            (d for d in damages if isinstance(d, dict)),
            key=lambda item: _safe_float(item.get("claimed_amount", 0.0)),
            reverse=True,
        )
    ) or f"<div class='viz-empty'>{L['no_waterfall']}</div>"

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
            for span in payment_recon.get("source_spans", []) or [] if isinstance(span, dict)
        )

    invoice_rows = "".join(
        "<tr>"
        f"<td>{_escape(invoice.get('invoice_id') or '(unlabeled)')}</td>"
        f"<td>{_fmt_money_short(invoice.get('invoiced', 0.0))}</td>"
        f"<td>{_fmt_money_short(invoice.get('paid', 0.0))}</td>"
        f"<td>{_fmt_money_short(invoice.get('outstanding', 0.0))}</td>"
        f"<td>{'<br>'.join(_escape(_fmt_span_label(span.get('span_id'))) for span in (invoice.get('source_spans') or []))}</td>"
        "</tr>"
        for invoice in invoice_chain if isinstance(invoice, dict)
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
        for row in damages if isinstance(row, dict)
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
        for conflict in amount_conflicts if isinstance(conflict, dict)
    )

    return (
        "<div class='viz-shell'>"
        + ("<div class='viz-card-grid'>" + "".join(cards) + "</div>" if cards else "")
        + f"<div class='viz-panel'><div class='viz-panel-title'>{L['waterfall']}</div>"
        + damage_rows
        + "</div>"
        + (
            "<div class='viz-two-col'>"
            + f"<div class='viz-panel'><div class='viz-panel-title'>{L['categories']}</div>"
            + (
                "<div class='matrix-wrap'><table class='analytics-table'><thead><tr>"
                "<th>Category</th><th>Total</th><th>Facts</th></tr></thead><tbody>"
                + category_rows
                + "</tbody></table></div>"
                if category_rows
                else f"<div class='viz-empty'>{L['no_categories']}</div>"
            )
            + "</div>"
            + f"<div class='viz-panel'><div class='viz-panel-title'>{L['invoice_recon']}</div>"
            + (
                "<div class='matrix-wrap'><table class='analytics-table'><thead><tr>"
                f"<th>Invoice</th><th>{L['invoiced']}</th><th>{L['paid']}</th><th>Outstanding</th><th>Source spans</th>"
                "</tr></thead><tbody>"
                + invoice_rows
                + "</tbody></table></div>"
                if invoice_rows
                else f"<div class='viz-empty'>{L['no_invoices']}</div>"
            )
            + "</div></div>"
        )
        + (
            f"<div class='viz-panel'><div class='viz-panel-title'>{L['grounding']}</div>"
            + (
                "<div class='matrix-wrap'><table class='analytics-table'><thead><tr>"
                "<th>Type</th><th>Subject</th><th>Amount</th><th>Location</th>"
                "</tr></thead><tbody>"
                + source_span_rows
                + "</tbody></table></div>"
                if source_span_rows
                else f"<div class='viz-empty'>{L['no_grounding']}</div>"
            )
            + "</div>"
            if has_recon
            else ""
        )
        + (
            f"<div class='viz-panel'><div class='viz-panel-title'>{L['damage_detail']}</div>"
            + (damage_details or "<div class='viz-empty'>No component detail available.</div>")
            + "</div>"
            if has_damages
            else ""
        )
        + (
            f"<div class='viz-panel'><div class='viz-panel-title'>{L['conflicts']}</div>"
            + (amount_conflict_details or f"<div class='viz-empty'>{L['no_conflicts']}</div>")
            + "</div>"
            if has_amount_conflicts
            else ""
        )
        + "</div>"
    )


_EXPORT_OVERVIEW_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Matter Overview",
        "assertions": "Assertions", "issues": "Issues", "gaps": "Gaps", "actors": "Actors",
        "weakest": "Weakest Issues (proof gaps)", "open_gaps": "Open Gaps",
        "clarifications": "Pending Clarifications",
    },
    "finance": {
        "title": "Analysis Overview",
        "assertions": "Claims", "issues": "Theses", "gaps": "Evidence gaps", "actors": "Entities",
        "weakest": "Weakest Theses (evidence gaps)", "open_gaps": "Open Evidence Gaps",
        "clarifications": "Pending Clarifications",
    },
    "coding": {
        "title": "Investigation Overview",
        "assertions": "Findings", "issues": "Hypotheses", "gaps": "Verification gaps", "actors": "Components",
        "weakest": "Weakest Hypotheses (verification gaps)", "open_gaps": "Open Verification Gaps",
        "clarifications": "Pending Clarifications",
    },
    "academic_research": {
        "title": "Research Overview",
        "assertions": "Claims", "issues": "Questions", "gaps": "Gaps", "actors": "Authors",
        "weakest": "Weakest Claims (evidence gaps)", "open_gaps": "Open Gaps",
        "clarifications": "Pending Clarifications",
    },
    "biomedical": {
        "title": "Investigation Overview",
        "assertions": "Findings", "issues": "Questions", "gaps": "Gaps", "actors": "Entities",
        "weakest": "Weakest Findings (evidence gaps)", "open_gaps": "Open Gaps",
        "clarifications": "Pending Clarifications",
    },
}


def _fmt_overview(data: dict, domain: str = "legal") -> str:
    if not data:
        return "No matter loaded."
    if err := _error_html(data):
        return err
    L = _EXPORT_OVERVIEW_LABELS.get(domain, _EXPORT_OVERVIEW_LABELS["legal"])
    stats = data.get("stats", {})
    so = data.get("so_metrics", {})
    llm = stats.get("llm", {}) if isinstance(stats.get("llm"), dict) else {}
    llm_totals = llm.get("totals", {}) if isinstance(llm, dict) else {}
    lines = [
        f"## {L['title']}",
        f"**{L['assertions']}:** {stats.get('assertion_count', 0)}  |  "
        f"**{L['issues']}:** {stats.get('open_issue_count', 0)}  |  "
        f"**{L['gaps']}:** {stats.get('open_gap_count', 0)}",
        f"**{L['actors']}:** {stats.get('actor_count', 0)}  |  "
        f"**Quant facts:** {stats.get('quant_fact_count', 0)}  |  "
        f"**Contradictions:** {data.get('contradiction_count', 0)}  |  "
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
        lines.append(f"\n### {L['weakest']}")
        for issue in weakest[:5]:
            if not isinstance(issue, dict):
                continue
            title = issue.get("title") or issue.get("id", "?")
            frac = issue.get("coverage_fraction")
            gap = " ⚠️" if issue.get("has_proof_gap") else ""
            lines.append(f"- **{title}** — {_fmt_coverage(frac)}{gap}")

    # Top gaps
    top_gaps = data.get("top_gaps", [])
    if top_gaps:
        lines.append(f"\n### {L['open_gaps']}")
        for gap in top_gaps[:5]:
            if not isinstance(gap, dict):
                continue
            desc = gap.get("description") or gap.get("gap_type", "—")
            lines.append(f"- {desc}")

    # Pending clarifications
    clarifications = data.get("pending_clarifications", [])
    if clarifications:
        lines.append(f"\n### {L['clarifications']}")
        for c in clarifications[:5]:
            if not isinstance(c, dict):
                continue
            q = c.get("question_text") or c.get("question", "—")
            lines.append(f"- {q}")

    return "\n".join(lines)


def _fmt_issues(issues: list, domain: str = "legal") -> str:
    L = _ISSUES_PANEL_LABELS.get(domain, _ISSUES_PANEL_LABELS["legal"])
    if not issues:
        return L["empty"]

    # Build lookup for tree rendering
    by_id = {i["id"]: i for i in issues if isinstance(i, dict) and "id" in i}
    sorted_issues = sorted(
        (i for i in issues if isinstance(i, dict)),
        key=lambda i: (i.get("depth", 0), i.get("coverage_fraction", 0)),
    )

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
    "legal": {
        "operative": "Operative", "authoritative": "Authoritative",
        "procedural": "Procedural", "informal": "Informal",
        "draft": "Draft", "advocacy": "Advocacy",
        "post_hoc": "Post Hoc", "unknown": "Unclassified",
    },
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
    "legal": {
        "alleged": "Alleged", "argued": "Argued", "denied": "Denied",
        "admitted": "Admitted", "ordered": "Ordered", "performed": "Performed",
        "paid": "Paid", "requested": "Requested", "threatened": "Threatened",
        "promised": "Promised", "estimated": "Estimated", "calculated": "Calculated",
        "observed": "Observed", "testified": "Testified", "stipulated": "Stipulated",
        "amended": "Amended", "waived": "Waived", "terminated": "Terminated",
        "inferred": "Inferred", "operative": "Operative", "extracted": "Extracted",
    },
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
    "legal": {
        "alleged": "Alleged", "argued": "Argued", "admitted": "Admitted",
        "operative": "Operative", "performed": "Performed",
        "not_performed": "Not Performed", "disputed": "Disputed",
        "superseded": "Superseded", "withdrawn": "Withdrawn",
        "inferred": "Inferred", "resolved": "Resolved", "unknown": "Unknown",
    },
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
        if not isinstance(a, dict):
            continue
        assertion_id = _escape(a.get("id", "?"))
        prop = _escape(a.get("proposition_text") or "")
        state_raw = a.get("belief_state") or "—"
        _KNOWN_BELIEFS_A = {"accepted", "rejected", "undetermined", "disputed", "provisional", "confirmed", "hypothetical",
                            "alleged", "argued", "admitted", "operative", "performed", "superseded", "withdrawn", "inferred", "resolved"}
        state_cls = state_raw.lower() if state_raw.lower() in _KNOWN_BELIEFS_A else "unknown"
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
        docs = a.get("documents") or []
        doc_label = ""
        if docs:
            short_docs = [d.rsplit("/", 1)[-1] for d in docs[:3]]
            doc_label = (
                f"<div style='font-size:10px;color:#6b7280;margin-top:2px'>"
                f"Source: {_escape(', '.join(short_docs))}"
                + (f" +{len(docs)-3} more" if len(docs) > 3 else "")
                + "</div>"
            )
        prop_cell = f"{vpill} {prop}{review_meta_html}{doc_label}".lstrip()
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


_ISSUE_ASSERTIONS_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "supporting": "Supporting Evidence",
        "attacking": "Attacking Evidence",
        "neutral": "Neutral / Background",
        "empty": "No linked assertions found for this issue.",
    },
    "finance": {
        "supporting": "Corroborating Data",
        "attacking": "Contradicting Data",
        "neutral": "Contextual Data",
        "empty": "No linked data points found for this objective.",
    },
    "coding": {
        "supporting": "Supporting Findings",
        "attacking": "Contradicting Findings",
        "neutral": "Contextual Findings",
        "empty": "No linked findings for this objective.",
    },
    "academic_research": {
        "supporting": "Supporting Findings",
        "attacking": "Contradicting Findings",
        "neutral": "Contextual References",
        "empty": "No linked findings for this question.",
    },
    "biomedical": {
        "supporting": "Supporting Evidence",
        "attacking": "Contradicting Evidence",
        "neutral": "Contextual Evidence",
        "empty": "No linked evidence for this hypothesis.",
    },
}


_ISSUE_CLOSURE_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Issue Closure Workbench",
        "ready": "Ready to rely on",
        "blocked": "Not ready — action needed",
        "coverage": "Evidence Coverage",
        "verified": "Verified facts",
        "pending": "Pending review",
        "gaps": "Open Gaps",
        "blockers": "Blockers",
        "sources": "Source Agreement",
    },
    "finance": {
        "title": "Position Closure Workbench",
        "ready": "Ready to rely on",
        "blocked": "Not ready — action needed",
        "coverage": "Evidence Coverage",
        "verified": "Verified data points",
        "pending": "Pending review",
        "gaps": "Open Gaps",
        "blockers": "Blockers",
        "sources": "Source Agreement",
    },
    "coding": {
        "title": "Requirement Closure Workbench",
        "ready": "Ready to close",
        "blocked": "Not ready — action needed",
        "coverage": "Evidence Coverage",
        "verified": "Verified findings",
        "pending": "Pending review",
        "gaps": "Open Gaps",
        "blockers": "Blockers",
        "sources": "Source Agreement",
    },
    "academic_research": {
        "title": "Claim Closure Workbench",
        "ready": "Sufficiently supported",
        "blocked": "Insufficiently supported",
        "coverage": "Evidence Coverage",
        "verified": "Verified citations",
        "pending": "Pending review",
        "gaps": "Open Gaps",
        "blockers": "Blockers",
        "sources": "Source Agreement",
    },
    "biomedical": {
        "title": "Finding Closure Workbench",
        "ready": "Sufficiently evidenced",
        "blocked": "Insufficiently evidenced",
        "coverage": "Evidence Coverage",
        "verified": "Verified records",
        "pending": "Pending review",
        "gaps": "Open Gaps",
        "blockers": "Blockers",
        "sources": "Source Agreement",
    },
}


def _fmt_issue_closure_workbench(data: dict, domain: str = "legal") -> str:
    if not data or not isinstance(data, dict):
        return "<div class='viz-empty'>No closure data available.</div>"
    if err := _error_html(data):
        return err

    L = _ISSUE_CLOSURE_LABELS.get(domain, _ISSUE_CLOSURE_LABELS["legal"])
    title = _escape(str(data.get("title", "")))
    readiness = data.get("readiness", "blocked")
    coverage = _safe_float(data.get("coverage_fraction", 0))
    supporting = int(data.get("supporting_count", 0))
    predicates = int(data.get("predicate_count", 0))
    verified = int(data.get("verified_count", 0))
    pending = int(data.get("pending_count", 0))
    blockers = data.get("blockers", [])
    gaps = data.get("gaps", [])

    ready_color = "#059669" if readiness == "ready" else "#dc2626"
    ready_label = L["ready"] if readiness == "ready" else L["blocked"]
    coverage_pct = min(coverage * 100, 100)
    cov_color = "#059669" if coverage >= 0.7 else "#f59e0b" if coverage >= 0.4 else "#dc2626"

    parts = [
        f"<div style='margin-bottom:16px;'>",
        f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;'>",
        f"<h3 style='margin:0;'>{L['title']}: {title}</h3>",
        f"<span style='display:inline-block;padding:4px 14px;border-radius:12px;"
        f"background:{ready_color};color:white;font-weight:700;font-size:13px;'>"
        f"{_escape(ready_label)}</span>",
        f"</div>",
    ]

    # Coverage bar
    parts.append(
        f"<div style='margin-bottom:12px;'>"
        f"<div style='font-size:12px;color:#6b7280;margin-bottom:4px;font-weight:600;'>"
        f"{L['coverage']}: {coverage_pct:.0f}% "
        f"({supporting} supporting / {predicates} element{'s' if predicates != 1 else ''})</div>"
        f"<div style='width:100%;height:12px;background:#e5e7eb;border-radius:6px;'>"
        f"<div style='width:{coverage_pct:.0f}%;height:100%;background:{cov_color};"
        f"border-radius:6px;transition:width 0.3s;'></div></div>"
        f"</div>"
    )

    # Verification status
    parts.append(
        f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px;'>"
        f"<div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:8px 12px;'>"
        f"<div style='font-size:11px;color:#166534;font-weight:600;'>{L['verified']}</div>"
        f"<div style='font-size:20px;font-weight:700;color:#059669;'>{verified}</div></div>"
        f"<div style='background:#fef3c7;border:1px solid #fde68a;border-radius:8px;padding:8px 12px;'>"
        f"<div style='font-size:11px;color:#92400e;font-weight:600;'>{L['pending']}</div>"
        f"<div style='font-size:20px;font-weight:700;color:#d97706;'>{pending}</div></div>"
        f"</div>"
    )

    # Blockers
    if blockers:
        blocker_items = "".join(
            f"<li style='margin:3px 0;color:#991b1b;'>{_escape(str(b))}</li>"
            for b in blockers if isinstance(b, str)
        )
        parts.append(
            f"<div style='background:#fef2f2;border:1px solid #fecaca;border-radius:8px;"
            f"padding:8px 12px;margin-bottom:12px;'>"
            f"<div style='font-size:12px;font-weight:700;color:#991b1b;margin-bottom:4px;'>"
            f"{L['blockers']} ({len(blockers)})</div>"
            f"<ul style='margin:0;padding-left:18px;font-size:12px;'>{blocker_items}</ul></div>"
        )

    # Gaps
    if gaps:
        gap_items = ""
        for g in gaps[:5]:
            if not isinstance(g, dict):
                continue
            gdesc = _escape(str(g.get("description", "")))
            gtype = _escape(str(g.get("gap_type", "")).replace("_", " ").title())
            gmat = _safe_float(g.get("materiality_score", 0))
            gap_items += (
                f"<div style='padding:4px 0;border-bottom:1px solid #f3f4f6;font-size:12px;'>"
                f"<span class='pill pill-neutral' style='font-size:10px;'>{gtype}</span> "
                f"{gdesc} <span style='color:#6b7280;'>({gmat:.2f})</span></div>"
            )
        parts.append(
            f"<div style='margin-bottom:12px;'>"
            f"<div style='font-size:12px;font-weight:700;color:#374151;margin-bottom:4px;'>"
            f"{L['gaps']} ({len(gaps)})</div>{gap_items}</div>"
        )

    # Source agreement summary
    source_agreement = data.get("source_agreement", [])
    if source_agreement:
        src_rows = ""
        for sa in source_agreement[:6]:
            if not isinstance(sa, dict):
                continue
            doc = _escape(str(sa.get("doc_label", "")))
            role = _escape(str(sa.get("source_role", "")))
            sup = int(sa.get("supports", 0))
            att = int(sa.get("attacks", 0))
            balance = "&#9989;" if sup > att else "&#9888;&#65039;" if att > 0 else "&#8212;"
            src_rows += (
                f"<tr><td style='padding:2px 6px;font-size:12px;'>{doc}</td>"
                f"<td style='padding:2px 6px;font-size:12px;'>{role}</td>"
                f"<td style='padding:2px 6px;font-size:12px;text-align:center;'>{sup}</td>"
                f"<td style='padding:2px 6px;font-size:12px;text-align:center;'>{att}</td>"
                f"<td style='padding:2px 6px;font-size:12px;text-align:center;'>{balance}</td></tr>"
            )
        if src_rows:
            parts.append(
                f"<div style='margin-bottom:8px;'>"
                f"<div style='font-size:12px;font-weight:700;color:#374151;margin-bottom:4px;'>"
                f"{L['sources']}</div>"
                f"<table style='width:100%;border-collapse:collapse;'>"
                f"<thead><tr style='background:#f1f5f9;'>"
                f"<th style='text-align:left;padding:2px 6px;font-size:11px;'>Document</th>"
                f"<th style='text-align:left;padding:2px 6px;font-size:11px;'>Role</th>"
                f"<th style='text-align:center;padding:2px 6px;font-size:11px;'>Support</th>"
                f"<th style='text-align:center;padding:2px 6px;font-size:11px;'>Attack</th>"
                f"<th style='text-align:center;padding:2px 6px;font-size:11px;'>Balance</th>"
                f"</tr></thead><tbody>{src_rows}</tbody></table></div>"
            )

    parts.append("</div>")
    return "".join(parts)


def _fmt_issue_assertions(
    assertions: list, issue_id: str, authorities: list | None = None, domain: str = "legal"
) -> str:
    L = _ISSUE_ASSERTIONS_LABELS.get(domain, _ISSUE_ASSERTIONS_LABELS["legal"])
    if not assertions and not authorities:
        return f"<div class='viz-empty'>{L['empty']}</div>"

    groups: dict[str, list[dict]] = {"supporting": [], "attacking": [], "neutral": []}
    for a in assertions:
        if not isinstance(a, dict):
            continue
        rel = (a.get("relation_type") or "neutral").lower()
        bucket = rel if rel in groups else "neutral"
        groups[bucket].append(a)

    parts = [f"<div class='viz-shell'><div style='padding:8px 12px;'>"]
    parts.append(
        f"<div style='font-size:12px;color:#6b7280;margin-bottom:8px;'>"
        f"Issue: <code>{_escape(str(issue_id)[:24])}</code> · "
        f"{len(assertions)} linked assertion(s)</div>"
    )

    section_styles = {
        "supporting": ("#16a34a", "#f0fdf4"),
        "attacking": ("#dc2626", "#fef2f2"),
        "neutral": ("#6b7280", "#f9fafb"),
    }

    for bucket in ("supporting", "attacking", "neutral"):
        items = groups[bucket]
        if not items:
            continue
        color, bg = section_styles[bucket]
        label = L.get(bucket, bucket.title())
        parts.append(
            f"<div style='margin-bottom:12px;'>"
            f"<div style='font-weight:600;font-size:13px;color:{color};margin-bottom:4px;'>"
            f"{_escape(label)} ({len(items)})</div>"
        )
        for a in items:
            prop = _escape((a.get("proposition_text") or "—")[:200])
            belief = a.get("belief_state") or "undetermined"
            belief_label = _escape(_domain_belief_label(belief, domain))
            _KNOWN_BELIEFS = {"accepted", "rejected", "undetermined", "disputed", "provisional", "confirmed", "hypothetical"}
            belief_cls = belief.lower() if belief.lower() in _KNOWN_BELIEFS else "undetermined"
            conf = a.get("confidence")
            conf_str = f"{float(conf):.2f}" if isinstance(conf, (int, float)) else "—"
            aid = _escape(str(a.get("id", "?"))[:12])
            parts.append(
                f"<div style='padding:6px 10px;margin-bottom:4px;border-radius:6px;"
                f"background:{bg};border-left:3px solid {color};font-size:12px;'>"
                f"<div>{prop}</div>"
                f"<div style='font-size:11px;color:#6b7280;margin-top:2px;'>"
                f"<span class='belief-pill belief-{belief_cls}'>{belief_label}</span>"
                f" · Confidence: {conf_str}"
                f" · <code style='font-size:10px;'>{aid}</code></div>"
                f"</div>"
            )
        parts.append("</div>")

    if authorities:
        auth_items = [a for a in authorities if isinstance(a, dict)]
        if auth_items:
            parts.append(
                "<div style='margin-top:12px;border-top:1px solid #e5e7eb;padding-top:8px;'>"
                "<div style='font-weight:600;font-size:13px;color:#7c3aed;margin-bottom:4px;'>"
                f"Linked Authorities ({len(auth_items)})</div>"
            )
            for auth in auth_items:
                cite = _escape((auth.get("citation") or "—")[:120])
                atype = _escape(str(auth.get("authority_type") or "?"))
                weight = _escape(str(auth.get("weight") or "?"))
                relevance = _escape(str(auth.get("relevance") or "neutral"))
                rel_color = "#16a34a" if relevance == "supporting" else (
                    "#dc2626" if relevance == "opposing" else "#6b7280"
                )
                parts.append(
                    f"<div style='padding:4px 10px;margin-bottom:3px;border-radius:6px;"
                    f"background:#f5f3ff;border-left:3px solid #7c3aed;font-size:12px;'>"
                    f"<div>{cite}</div>"
                    f"<div style='font-size:11px;color:#6b7280;margin-top:2px;'>"
                    f"{atype} · {weight} · "
                    f"<span style='color:{rel_color};font-weight:600;'>{relevance}</span></div>"
                    f"</div>"
                )
            parts.append("</div>")

    parts.append("</div></div>")
    return "\n".join(parts)


_SOURCE_AGREEMENT_LABELS: dict[str, dict[str, str]] = {
    "legal": {"title": "Source Agreement Analysis", "doc_col": "Document", "role_col": "Source Role", "empty": "No source data available for this issue."},
    "finance": {"title": "Source Agreement Analysis", "doc_col": "Filing / Report", "role_col": "Source Type", "empty": "No source data available for this thesis."},
    "coding": {"title": "Source Agreement Analysis", "doc_col": "Spec / Artifact", "role_col": "Source Type", "empty": "No source data available for this task."},
    "academic_research": {"title": "Source Agreement Analysis", "doc_col": "Paper / Source", "role_col": "Source Type", "empty": "No source data available for this question."},
    "biomedical": {"title": "Source Agreement Analysis", "doc_col": "Record / Source", "role_col": "Source Type", "empty": "No source data available for this hypothesis."},
}


def _fmt_source_agreement(sources: list, domain: str = "legal") -> str:
    L = _SOURCE_AGREEMENT_LABELS.get(domain, _SOURCE_AGREEMENT_LABELS["legal"])
    if not sources:
        return f"<div class='viz-empty'>{L['empty']}</div>"
    rows_html = ""
    for s in sources:
        if not isinstance(s, dict):
            continue
        label = _escape(str(s.get("doc_label", "—"))[:50])
        role = _escape(str(s.get("source_role", "unknown")))
        supports = int(s.get("supports", 0))
        attacks = int(s.get("attacks", 0))
        total = supports + attacks
        bar_w = min(total * 8, 120)
        sup_pct = (supports / total * 100) if total else 0
        atk_pct = 100 - sup_pct
        bar = (
            f"<div style='display:inline-flex;height:10px;width:{bar_w}px;border-radius:3px;overflow:hidden'>"
            f"<div style='width:{sup_pct:.0f}%;background:#16a34a'></div>"
            f"<div style='width:{atk_pct:.0f}%;background:#dc2626'></div>"
            "</div>"
        ) if total else ""
        rows_html += (
            "<tr>"
            f"<td style='font-size:11px'>{label}</td>"
            f"<td style='font-size:11px'>{role}</td>"
            f"<td style='font-size:11px;color:#16a34a;font-weight:600'>{supports}</td>"
            f"<td style='font-size:11px;color:#dc2626;font-weight:600'>{attacks}</td>"
            f"<td>{bar}</td>"
            "</tr>"
        )
    if not rows_html:
        return f"<div class='viz-empty'>{L['empty']}</div>"
    return (
        f"<div class='viz-shell'><div class='viz-header'><strong>{_escape(L['title'])}</strong></div>"
        "<div class='table-wrap'><table class='viz-table'>"
        f"<thead><tr><th>{_escape(L['doc_col'])}</th><th>{_escape(L['role_col'])}</th>"
        "<th>Supports</th><th>Attacks</th><th>Balance</th></tr></thead>"
        "<tbody>" + rows_html + "</tbody></table></div></div>"
    )


_ASSERTION_GRAPH_LABELS: dict[str, dict[str, str]] = {
    "legal": {"title": "Assertion Relationship Map", "empty": "No assertions linked to this issue."},
    "finance": {"title": "Finding Relationship Map", "empty": "No findings linked to this thesis."},
    "coding": {"title": "Claim Relationship Map", "empty": "No claims linked to this task."},
    "academic_research": {"title": "Claim Relationship Map", "empty": "No claims linked to this question."},
    "biomedical": {"title": "Finding Relationship Map", "empty": "No findings linked to this hypothesis."},
}

_GRAPH_LINK_COLORS: dict[str, str] = {
    "supports": "#16a34a", "corroborates": "#16a34a",
    "attacks": "#dc2626", "contradicts": "#dc2626",
    "supersedes": "#7c3aed",
}


def _fmt_assertion_graph(graph: dict, domain: str = "legal") -> str:
    L = _ASSERTION_GRAPH_LABELS.get(domain, _ASSERTION_GRAPH_LABELS["legal"])
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    if not nodes:
        return f"<div class='viz-empty'>{L['empty']}</div>"

    supporting = []
    attacking = []
    other = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        rel = (n.get("relation_type") or "").lower()
        if rel in ("supports", "establishes"):
            supporting.append(n)
        elif rel in ("attacks", "negates"):
            attacking.append(n)
        else:
            other.append(n)

    def _node_card(n: dict, color: str, bg: str) -> str:
        aid = _escape(str(n.get("id", ""))[:16])
        prop = _escape(str(n.get("proposition_text", "—"))[:80])
        belief = n.get("belief_state", "undetermined")
        belief_label = _escape(_domain_belief_label(belief, domain))
        conf = n.get("confidence", 0)
        try:
            conf_val = float(conf)
        except (TypeError, ValueError):
            conf_val = 0.0
        return (
            f"<div style='padding:6px 10px;margin:3px 0;border-radius:6px;"
            f"background:{bg};border-left:3px solid {color};font-size:11px'>"
            f"<div>{prop}</div>"
            f"<div style='font-size:10px;color:#6b7280;margin-top:2px'>"
            f"<code>{aid}</code> · {belief_label} · {conf_val:.2f}</div></div>"
        )

    parts = [f"<div class='viz-shell'><div class='viz-header'><strong>{_escape(L['title'])}</strong></div>"]

    if supporting:
        parts.append(
            "<div style='margin-bottom:12px'>"
            f"<div style='font-weight:600;font-size:12px;color:#16a34a;margin-bottom:4px'>Supporting ({len(supporting)})</div>"
        )
        for n in supporting:
            parts.append(_node_card(n, "#16a34a", "#f0fdf4"))
        parts.append("</div>")

    if attacking:
        parts.append(
            "<div style='margin-bottom:12px'>"
            f"<div style='font-weight:600;font-size:12px;color:#dc2626;margin-bottom:4px'>Attacking ({len(attacking)})</div>"
        )
        for n in attacking:
            parts.append(_node_card(n, "#dc2626", "#fef2f2"))
        parts.append("</div>")

    if other:
        parts.append(
            "<div style='margin-bottom:12px'>"
            f"<div style='font-weight:600;font-size:12px;color:#6b7280;margin-bottom:4px'>Other ({len(other)})</div>"
        )
        for n in other:
            parts.append(_node_card(n, "#6b7280", "#f9fafb"))
        parts.append("</div>")

    if edges:
        edge_valid = [e for e in edges if isinstance(e, dict)]
        if edge_valid:
            node_map = {n["id"]: _escape(str(n.get("proposition_text", "—"))[:40])
                        for n in nodes if isinstance(n, dict) and n.get("id")}
            parts.append(
                "<div style='margin-top:8px;border-top:1px solid #e5e7eb;padding-top:8px'>"
                "<div style='font-weight:600;font-size:12px;color:#374151;margin-bottom:4px'>"
                f"Inter-Assertion Links ({len(edge_valid)})</div>"
            )
            for e in edge_valid[:20]:
                src = _escape(str(e.get("src", ""))[:16])
                dst = _escape(str(e.get("dst", ""))[:16])
                lt = str(e.get("link_type", "?")).lower()
                color = _GRAPH_LINK_COLORS.get(lt, "#6b7280")
                src_label = node_map.get(e.get("src", ""), src)
                dst_label = node_map.get(e.get("dst", ""), dst)
                parts.append(
                    f"<div style='font-size:11px;padding:2px 0'>"
                    f"{src_label} "
                    f"<span style='color:{color};font-weight:600'>→ {_escape(lt)} →</span> "
                    f"{dst_label}</div>"
                )
            parts.append("</div>")

    parts.append("</div>")
    return "\n".join(parts)


_ASSUMPTION_STATUS_PILLS: dict[str, tuple[str, str]] = {
    "provisional": ("Provisional", "pill-orange"),
    "confirmed": ("Confirmed", "pill-green"),
    "invalidated": ("Invalidated", "pill-red"),
}

_ASSUMPTION_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Working Assumptions",
        "empty": "No assumptions recorded yet. Irys will log assumptions as it builds its analysis.",
    },
    "finance": {
        "title": "Working Assumptions",
        "empty": "No assumptions recorded yet. Irys will log assumptions as it builds its analysis.",
    },
    "coding": {
        "title": "Working Assumptions",
        "empty": "No assumptions recorded yet. Irys will log assumptions as it builds its analysis.",
    },
    "academic_research": {
        "title": "Working Assumptions",
        "empty": "No assumptions recorded yet. Irys will log assumptions as it builds its analysis.",
    },
    "biomedical": {
        "title": "Working Assumptions",
        "empty": "No assumptions recorded yet. Irys will log assumptions as it builds its analysis.",
    },
}


def _fmt_assumptions(assumptions: list, domain: str = "legal") -> str:
    labels = _ASSUMPTION_LABELS.get(domain, _ASSUMPTION_LABELS["legal"])
    if not assumptions:
        return f"<div class='viz-empty'>{labels['empty']}</div>"
    rows = ""
    prov_count = 0
    conf_count = 0
    inv_count = 0
    for a in assumptions:
        if not isinstance(a, dict):
            continue
        status = str(a.get("status") or "provisional")
        if status == "provisional":
            prov_count += 1
        elif status == "confirmed":
            conf_count += 1
        elif status == "invalidated":
            inv_count += 1
        pill_text, pill_cls = _ASSUMPTION_STATUS_PILLS.get(status, (_escape(status.replace("_", " ").title()), "pill-neutral"))
        stmt = _escape(str(a.get("statement") or "?"))
        cond = _escape(str(a.get("invalidation_condition") or ""))
        rationale = _escape(str(a.get("rationale") or ""))
        detail = ""
        if cond:
            detail += f"<div style='font-size:11px;color:#dc2626;margin-top:2px'>Invalidated if: {cond}</div>"
        if rationale:
            detail += f"<div style='font-size:11px;color:#6b7280;margin-top:2px'>Rationale: {rationale}</div>"
        target_count = a.get("linked_target_count", 0)
        if target_count:
            targets = a.get("linked_targets", [])
            target_types = set()
            for t in targets:
                if isinstance(t, dict):
                    target_types.add(t.get("target_type", "item"))
            type_str = ", ".join(sorted(target_types)) if target_types else "items"
            detail += (
                f"<div style='font-size:10px;color:#7c3aed;margin-top:2px'>"
                f"Linked to {target_count} {type_str}</div>"
            )
        aid = _escape(str(a.get("id", "?"))[:16])
        full_aid = _escape(str(a.get("id", "?")))
        rows += (
            "<tr>"
            f"<td><span class='pill {pill_cls}'>{pill_text}</span></td>"
            f"<td><strong>{stmt}</strong>{detail}</td>"
            f"<td><code style='font-size:10px;cursor:pointer;' title='{full_aid}'>{aid}</code></td>"
            "</tr>"
        )
    if not rows:
        return f"<div class='viz-empty'>{labels['empty']}</div>"
    total = prov_count + conf_count + inv_count
    subtitle = f"{total} assumption{'s' if total != 1 else ''}"
    if inv_count:
        subtitle += f" ({inv_count} invalidated)"
    return (
        "<div class='viz-shell'>"
        f"<div class='viz-header'><strong>{labels['title']}</strong> — {subtitle}</div>"
        "<div class='table-wrap'><table class='viz-table'>"
        "<thead><tr><th>Status</th><th>Assumption</th><th>ID</th></tr></thead>"
        "<tbody>" + rows + "</tbody></table></div>"
        "</div>"
    )


_ASSUMPTION_REVIEW_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Assumption Review Workbench",
        "subtitle": "Lifecycle review of every assumption underlying the analysis.",
        "provisional": "Provisional",
        "confirmed": "Confirmed",
        "invalidated": "Invalidated",
        "linked": "Linked targets",
        "empty": "No assumptions recorded. Irys will log assumptions as it builds the matter model.",
    },
    "finance": {
        "title": "Thesis Assumption Review",
        "subtitle": "Lifecycle review of every assumption underlying the thesis.",
        "provisional": "Provisional",
        "confirmed": "Confirmed",
        "invalidated": "Invalidated",
        "linked": "Linked targets",
        "empty": "No assumptions recorded. Irys will log assumptions during analysis.",
    },
    "coding": {
        "title": "Design Assumption Review",
        "subtitle": "Lifecycle review of every design assumption.",
        "provisional": "Provisional",
        "confirmed": "Confirmed",
        "invalidated": "Invalidated",
        "linked": "Linked targets",
        "empty": "No assumptions recorded. Irys will log assumptions during investigation.",
    },
    "academic_research": {
        "title": "Methodology Assumption Review",
        "subtitle": "Lifecycle review of every methodology assumption.",
        "provisional": "Provisional",
        "confirmed": "Confirmed",
        "invalidated": "Invalidated",
        "linked": "Linked targets",
        "empty": "No assumptions recorded. Irys will log assumptions during research.",
    },
    "biomedical": {
        "title": "Mechanism Assumption Review",
        "subtitle": "Lifecycle review of every clinical and mechanism assumption.",
        "provisional": "Provisional",
        "confirmed": "Confirmed",
        "invalidated": "Invalidated",
        "linked": "Linked targets",
        "empty": "No assumptions recorded. Irys will log assumptions during analysis.",
    },
}


def _fmt_assumption_review_workbench(data: dict, domain: str = "legal") -> str:
    L = _ASSUMPTION_REVIEW_LABELS.get(domain, _ASSUMPTION_REVIEW_LABELS["legal"])

    if not data or not isinstance(data, dict):
        return f"<div class='viz-empty'>{_escape(L['empty'])}</div>"
    if err := _error_html(data):
        return err

    counts = data.get("counts", {})
    if not isinstance(counts, dict):
        counts = {}
    total = _safe_int(data.get("total", 0))
    prov_n = _safe_int(counts.get("provisional", 0))
    conf_n = _safe_int(counts.get("confirmed", 0))
    inv_n = _safe_int(counts.get("invalidated", 0))

    if total == 0:
        return f"<div class='viz-empty'>{_escape(L['empty'])}</div>"

    header = (
        f"<div class='viz-shell'>"
        f"<div class='intel-panel-title'>{_escape(L['title'])}</div>"
        f"<div style='font-size:0.8em;color:#6b7280;margin-bottom:8px;'>{_escape(L['subtitle'])}</div>"
        f"<div style='display:flex;gap:16px;align-items:center;margin-bottom:12px;flex-wrap:wrap;'>"
        f"<div style='font-size:0.85em;'>"
        f"<span class='pill pill-neutral'>{_escape(L['provisional'])}: {prov_n}</span> "
        f"<span class='pill pill-green'>{_escape(L['confirmed'])}: {conf_n}</span> "
        f"<span class='pill pill-red'>{_escape(L['invalidated'])}: {inv_n}</span>"
        f"</div>"
        f"<div style='font-size:0.8em;color:#9ca3af;'>Total: {total}</div>"
        f"</div>"
    )

    def _render_bucket(items: list, bucket_label: str, color: str) -> str:
        if not isinstance(items, list) or not items:
            return ""
        html = (
            f"<div style='margin-bottom:12px;'>"
            f"<div style='font-weight:600;color:{color};margin-bottom:4px;font-size:0.9em;'>"
            f"{_escape(bucket_label)} ({len(items)})</div>"
        )
        for a in items[:25]:
            if not isinstance(a, dict):
                continue
            stmt = _escape(str(a.get("statement") or "?")[:250])
            aid = _escape(str(a.get("id") or "?")[:16])
            full_aid = _escape(str(a.get("id") or "?"))
            rationale = _escape(str(a.get("rationale") or "")[:200])
            cond = _escape(str(a.get("invalidation_condition") or "")[:200])
            target_count = _safe_int(a.get("linked_target_count", 0))

            html += (
                f"<div style='border:1px solid var(--border-color-primary,#e5e7eb);"
                f"border-radius:6px;padding:8px;margin-bottom:6px;'>"
                f"<div style='font-weight:500;'>{stmt}</div>"
                f"<div style='font-size:0.78em;color:#9ca3af;margin-top:2px;'>"
                f"ID: <code style='font-size:10px;cursor:pointer;' title='{full_aid}'>{aid}</code></div>"
            )
            if rationale:
                html += f"<div style='font-size:0.8em;color:#6b7280;margin-top:3px;'>Rationale: {rationale}</div>"
            if cond:
                html += f"<div style='font-size:0.8em;color:#dc2626;margin-top:2px;'>Invalidated if: {cond}</div>"
            if target_count:
                targets = a.get("linked_targets", [])
                target_types = set()
                for t in (targets if isinstance(targets, list) else []):
                    if isinstance(t, dict):
                        target_types.add(t.get("target_type", "item"))
                type_str = ", ".join(sorted(target_types)) if target_types else "items"
                html += (
                    f"<div style='font-size:0.78em;color:#7c3aed;margin-top:2px;'>"
                    f"{_escape(L['linked'])}: {target_count} {_escape(type_str)}</div>"
                )
            html += "</div>"
        html += "</div>"
        return html

    body = ""
    body += _render_bucket(data.get("provisional", []), L["provisional"], "#6b7280")
    body += _render_bucket(data.get("invalidated", []), L["invalidated"], "#dc2626")
    body += _render_bucket(data.get("confirmed", []), L["confirmed"], "#16a34a")

    return f"{header}{body}</div>"


_GAP_LABELS: dict[str, dict[str, object]] = {
    "legal": {
        "title": "Open Gaps & Missingness",
        "empty": "No open gaps or pending clarifications.",
        "missing_document": ("Missing Document", "pill-red"),
        "missing_metadata": ("Missing Metadata", "pill-orange"),
        "missing_issue_predicate": ("Missing Element of Proof", "pill-orange"),
        "missing_authority": ("Missing Authority", "pill-orange"),
        "missing_user_context": ("Missing Context", "pill-neutral"),
        "missing_quantitative_input": ("Missing Amount", "pill-orange"),
        "unresolved_contradiction": ("Unresolved Conflict", "pill-red"),
        "expected_absent_attachment": ("Expected Exhibit", "pill-orange"),
        "expected_absent_notice": ("Expected Notice", "pill-orange"),
    },
    "finance": {
        "title": "Open Gaps & Missing Data",
        "empty": "No open data gaps or pending queries.",
        "missing_document": ("Missing Filing", "pill-red"),
        "missing_metadata": ("Missing Metadata", "pill-orange"),
        "missing_issue_predicate": ("Missing Condition", "pill-orange"),
        "missing_authority": ("Missing Regulation", "pill-orange"),
        "missing_user_context": ("Missing Context", "pill-neutral"),
        "missing_quantitative_input": ("Missing Figure", "pill-orange"),
        "unresolved_contradiction": ("Discrepancy", "pill-red"),
        "expected_absent_attachment": ("Expected Schedule", "pill-orange"),
        "expected_absent_notice": ("Expected Disclosure", "pill-orange"),
    },
    "coding": {
        "title": "Open Gaps & Missing Information",
        "empty": "No open gaps or pending clarifications.",
        "missing_document": ("Missing Spec", "pill-red"),
        "missing_metadata": ("Missing Metadata", "pill-orange"),
        "missing_issue_predicate": ("Missing Acceptance Criteria", "pill-orange"),
        "missing_authority": ("Missing Reference", "pill-orange"),
        "missing_user_context": ("Missing Context", "pill-neutral"),
        "missing_quantitative_input": ("Missing Metric", "pill-orange"),
        "unresolved_contradiction": ("Conflicting Requirements", "pill-red"),
        "expected_absent_attachment": ("Expected Artifact", "pill-orange"),
        "expected_absent_notice": ("Expected Notification", "pill-orange"),
    },
    "academic_research": {
        "title": "Open Gaps & Missing Evidence",
        "empty": "No open gaps or pending clarifications.",
        "missing_document": ("Missing Source", "pill-red"),
        "missing_metadata": ("Missing Metadata", "pill-orange"),
        "missing_issue_predicate": ("Missing Hypothesis Element", "pill-orange"),
        "missing_authority": ("Missing Citation", "pill-orange"),
        "missing_user_context": ("Missing Context", "pill-neutral"),
        "missing_quantitative_input": ("Missing Data Point", "pill-orange"),
        "unresolved_contradiction": ("Conflicting Findings", "pill-red"),
        "expected_absent_attachment": ("Expected Appendix", "pill-orange"),
        "expected_absent_notice": ("Expected Disclosure", "pill-orange"),
    },
    "biomedical": {
        "title": "Open Gaps & Missing Data",
        "empty": "No open gaps or pending queries.",
        "missing_document": ("Missing Record", "pill-red"),
        "missing_metadata": ("Missing Metadata", "pill-orange"),
        "missing_issue_predicate": ("Missing Diagnostic Criterion", "pill-orange"),
        "missing_authority": ("Missing Protocol Reference", "pill-orange"),
        "missing_user_context": ("Missing Patient Context", "pill-neutral"),
        "missing_quantitative_input": ("Missing Lab Value", "pill-orange"),
        "unresolved_contradiction": ("Conflicting Results", "pill-red"),
        "expected_absent_attachment": ("Expected Imaging", "pill-orange"),
        "expected_absent_notice": ("Expected Consent", "pill-orange"),
    },
}


_MISSING_DOC_TYPES = frozenset({
    "missing_document", "expected_absent_attachment", "expected_absent_notice",
})

_MISSING_DOC_HEADER: dict[str, str] = {
    "legal": "Missing Documents & Exhibits",
    "finance": "Missing Filings & Reports",
    "coding": "Missing Specs & Artifacts",
    "academic_research": "Missing Sources & Appendices",
    "biomedical": "Missing Records & Disclosures",
}


_GAP_WORKBENCH_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Gap-to-Action Workbench",
        "empty": "No open gaps or pending clarifications.",
        "investigate": "Investigate",
        "request_document": "Request document",
        "clarify": "Answer clarification",
        "escalate": "Escalate",
        "affected": "Affected issues",
        "source": "Missing source",
        "clarifications": "Clarifications",
    },
    "finance": {
        "title": "Gap-to-Action Workbench",
        "empty": "No open gaps or pending items.",
        "investigate": "Investigate",
        "request_document": "Request filing",
        "clarify": "Answer question",
        "escalate": "Escalate",
        "affected": "Affected positions",
        "source": "Missing filing",
        "clarifications": "Pending questions",
    },
    "coding": {
        "title": "Gap-to-Action Workbench",
        "empty": "No open gaps or pending items.",
        "investigate": "Investigate",
        "request_document": "Request spec",
        "clarify": "Answer question",
        "escalate": "Escalate",
        "affected": "Affected requirements",
        "source": "Missing specification",
        "clarifications": "Pending questions",
    },
    "academic_research": {
        "title": "Gap-to-Action Workbench",
        "empty": "No open gaps or pending items.",
        "investigate": "Investigate",
        "request_document": "Request source",
        "clarify": "Answer question",
        "escalate": "Escalate",
        "affected": "Affected claims",
        "source": "Missing reference",
        "clarifications": "Pending questions",
    },
    "biomedical": {
        "title": "Gap-to-Action Workbench",
        "empty": "No open gaps or pending items.",
        "investigate": "Investigate",
        "request_document": "Request record",
        "clarify": "Answer question",
        "escalate": "Escalate",
        "affected": "Affected findings",
        "source": "Missing record",
        "clarifications": "Pending questions",
    },
}

_GAP_ACTION_COLORS: dict[str, str] = {
    "investigate": "#2563eb",
    "request_document": "#d97706",
    "clarify": "#7c3aed",
    "escalate": "#dc2626",
}


def _fmt_gap_workbench(payload: dict, domain: str = "legal") -> str:
    labels = _GAP_WORKBENCH_LABELS.get(domain, _GAP_WORKBENCH_LABELS["legal"])
    items = payload.get("items", []) if isinstance(payload, dict) else []
    if not items:
        return f"<div class='viz-empty'>{labels['empty']}</div>"

    cards: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        action = item.get("recommended_next_action", "investigate")
        action_label = labels.get(action, action.replace("_", " ").title())
        action_color = _GAP_ACTION_COLORS.get(action, "#6b7280")
        desc = _escape(str(item.get("description", "")))
        typ = _escape(str(item.get("type", "")).replace("_", " ").title())
        mat = _safe_float(item.get("materiality_score", 0))
        mat_color = "#dc2626" if mat >= 0.7 else "#f59e0b" if mat >= 0.4 else "#6b7280"
        source = _escape(str(item.get("missing_source_suggestion", "")))

        issue_parts = []
        for iss in (item.get("affected_issues") or [])[:4]:
            if not isinstance(iss, dict):
                continue
            t = iss.get("title") or iss.get("affected_id", "")[:8]
            issue_parts.append(_escape(str(t)))
        issues_html = ", ".join(issue_parts) if issue_parts else "<span style='color:#9ca3af'>None linked</span>"

        clar_items = []
        for c in (item.get("pending_clarifications") or []):
            if not isinstance(c, dict):
                continue
            q = _escape(str(c.get("question_text", "")))
            if q:
                clar_items.append(f"<li style='margin:2px 0;'>{q}</li>")
        clar_html = (
            f"<ul style='margin:4px 0;padding-left:16px;font-size:12px;'>{''.join(clar_items)}</ul>"
            if clar_items
            else "<span style='color:#9ca3af;font-size:12px;'>None pending</span>"
        )

        cards.append(
            f"<div style='border-bottom:1px solid #e5e7eb;padding:12px 4px;'>"
            f"<div style='display:flex;gap:8px;align-items:center;justify-content:space-between;margin-bottom:6px;'>"
            f"<span class='pill pill-neutral'>{typ}</span>"
            f"<span style='font-weight:700;color:{mat_color};'>{mat:.2f}</span>"
            f"<span style='display:inline-block;padding:2px 10px;border-radius:10px;"
            f"background:{action_color};color:white;font-size:11px;font-weight:600;'>"
            f"{_escape(action_label)}</span>"
            f"</div>"
            f"<div style='color:#1f2937;line-height:1.5;margin-bottom:8px;'>{desc}</div>"
            f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;font-size:12px;'>"
            f"<div><div style='font-size:11px;color:#6b7280;font-weight:700;margin-bottom:2px;'>"
            f"{labels['affected']}</div><div>{issues_html}</div></div>"
            f"<div><div style='font-size:11px;color:#6b7280;font-weight:700;margin-bottom:2px;'>"
            f"{labels['source']}</div><div style='color:#374151;'>{source}</div></div>"
            f"<div><div style='font-size:11px;color:#6b7280;font-weight:700;margin-bottom:2px;'>"
            f"{labels['clarifications']}</div>{clar_html}</div>"
            f"</div></div>"
        )

    return (
        f"<div style='max-height:620px;overflow-y:auto;'>"
        f"<div class='viz-header'><strong>{labels['title']}</strong>"
        f" &mdash; {len(cards)} open gap(s)</div>"
        + "".join(cards)
        + "</div>"
    )


_READINESS_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Investigation Readiness",
        "ready": "Ready for reliance",
        "caution": "Proceed with caution",
        "blocked": "Not ready — action required",
        "issues": "Issues tracked",
        "avg_coverage": "Average coverage",
        "gaps": "Open gaps",
        "contradictions": "Contradictions",
        "clarifications": "Pending clarifications",
        "review": "Pending review",
        "low_coverage": "Insufficient evidence coverage",
        "proof_gap": "Missing element of proof",
        "unverified_critical": "Critical issues lack verified facts",
        "contradictions_blocker": "Unresolved contradictions",
        "pending_clarifications_blocker": "Pending clarifications",
        "high_materiality_gaps": "High-materiality gaps",
    },
    "finance": {
        "title": "Analysis Readiness",
        "ready": "Ready for reliance",
        "caution": "Proceed with caution",
        "blocked": "Not ready — action required",
        "issues": "Positions tracked",
        "avg_coverage": "Average coverage",
        "gaps": "Open gaps",
        "contradictions": "Contradictions",
        "clarifications": "Pending clarifications",
        "review": "Pending review",
        "low_coverage": "Insufficient data coverage",
        "proof_gap": "Missing element of proof",
        "unverified_critical": "Critical positions lack verified data",
        "contradictions_blocker": "Unresolved contradictions",
        "pending_clarifications_blocker": "Pending clarifications",
        "high_materiality_gaps": "High-materiality gaps",
    },
    "coding": {
        "title": "Analysis Readiness",
        "ready": "Ready for reliance",
        "caution": "Proceed with caution",
        "blocked": "Not ready — action required",
        "issues": "Requirements tracked",
        "avg_coverage": "Average coverage",
        "gaps": "Open gaps",
        "contradictions": "Contradictions",
        "clarifications": "Pending clarifications",
        "review": "Pending review",
        "low_coverage": "Insufficient evidence coverage",
        "proof_gap": "Missing element of proof",
        "unverified_critical": "Critical requirements lack verified findings",
        "contradictions_blocker": "Unresolved contradictions",
        "pending_clarifications_blocker": "Pending clarifications",
        "high_materiality_gaps": "High-materiality gaps",
    },
    "academic_research": {
        "title": "Research Readiness",
        "ready": "Ready for reliance",
        "caution": "Proceed with caution",
        "blocked": "Not ready — action required",
        "issues": "Claims tracked",
        "avg_coverage": "Average coverage",
        "gaps": "Open gaps",
        "contradictions": "Contradictions",
        "clarifications": "Pending clarifications",
        "review": "Pending review",
        "low_coverage": "Insufficient citation coverage",
        "proof_gap": "Missing element of proof",
        "unverified_critical": "Critical claims lack verified citations",
        "contradictions_blocker": "Unresolved contradictions",
        "pending_clarifications_blocker": "Pending clarifications",
        "high_materiality_gaps": "High-materiality gaps",
    },
    "biomedical": {
        "title": "Assessment Readiness",
        "ready": "Ready for reliance",
        "caution": "Proceed with caution",
        "blocked": "Not ready — action required",
        "issues": "Findings tracked",
        "avg_coverage": "Average coverage",
        "gaps": "Open gaps",
        "contradictions": "Contradictions",
        "clarifications": "Pending clarifications",
        "review": "Pending review",
        "low_coverage": "Insufficient evidence coverage",
        "proof_gap": "Missing element of proof",
        "unverified_critical": "Critical findings lack verified records",
        "contradictions_blocker": "Unresolved contradictions",
        "pending_clarifications_blocker": "Pending clarifications",
        "high_materiality_gaps": "High-materiality gaps",
    },
}

_READINESS_SEVERITY_COLORS: dict[str, str] = {
    "high": "#dc2626",
    "medium": "#f59e0b",
    "low": "#6b7280",
}


def _fmt_readiness_panel(data: dict, domain: str = "legal") -> str:
    if not data or not isinstance(data, dict):
        return "<div class='viz-empty'>No readiness data available.</div>"
    if err := _error_html(data):
        return err

    L = _READINESS_LABELS.get(domain, _READINESS_LABELS["legal"])
    readiness = data.get("readiness", "blocked")
    blockers = data.get("blockers", [])
    summary = data.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}

    ready_colors = {"ready": "#059669", "caution": "#f59e0b", "blocked": "#dc2626"}
    ready_color = ready_colors.get(readiness, "#dc2626")
    ready_label = L.get(readiness, readiness)

    issue_count = int(summary.get("issue_count", 0))
    avg_cov = _safe_float(summary.get("avg_coverage", 0))
    avg_cov_pct = min(avg_cov * 100, 100)
    cov_color = "#059669" if avg_cov >= 0.7 else "#f59e0b" if avg_cov >= 0.4 else "#dc2626"
    gap_count = int(summary.get("open_gap_count", 0))
    contradiction_count = int(summary.get("contradiction_count", 0))
    pending_clar = int(summary.get("pending_clarifications", 0))
    pending_rev = int(summary.get("pending_review", 0))

    parts = [
        f"<div style='margin-bottom:16px;'>",
        f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;'>",
        f"<h3 style='margin:0;'>{_escape(L['title'])}</h3>",
        f"<span style='display:inline-block;padding:4px 14px;border-radius:12px;"
        f"background:{ready_color};color:white;font-weight:700;font-size:13px;'>"
        f"{_escape(ready_label)}</span>",
        f"</div>",
    ]

    # Summary metrics grid
    parts.append(
        f"<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px;'>"
    )
    metric_items = [
        (L["issues"], str(issue_count), "#1f2937"),
        (L["avg_coverage"], f"{avg_cov_pct:.0f}%", cov_color),
        (L["gaps"], str(gap_count), "#dc2626" if gap_count > 0 else "#059669"),
        (L["contradictions"], str(contradiction_count), "#dc2626" if contradiction_count > 0 else "#059669"),
        (L["clarifications"], str(pending_clar), "#f59e0b" if pending_clar > 0 else "#059669"),
        (L["review"], str(pending_rev), "#f59e0b" if pending_rev > 0 else "#059669"),
    ]
    for label, val, color in metric_items:
        parts.append(
            f"<div style='text-align:center;padding:10px;background:#f9fafb;border-radius:8px;'>"
            f"<div style='font-size:22px;font-weight:700;color:{color};'>{_escape(val)}</div>"
            f"<div style='font-size:11px;color:#6b7280;font-weight:600;margin-top:2px;'>"
            f"{_escape(label)}</div></div>"
        )
    parts.append("</div>")

    # Coverage bar
    parts.append(
        f"<div style='margin-bottom:16px;'>"
        f"<div style='font-size:12px;color:#6b7280;margin-bottom:4px;font-weight:600;'>"
        f"{_escape(L['avg_coverage'])}</div>"
        f"<div style='width:100%;height:12px;background:#e5e7eb;border-radius:6px;'>"
        f"<div style='width:{avg_cov_pct:.0f}%;height:100%;background:{cov_color};"
        f"border-radius:6px;transition:width 0.3s;'></div></div></div>"
    )

    # Blockers section
    if blockers:
        parts.append(
            f"<div style='margin-bottom:8px;'>"
            f"<div style='font-size:13px;font-weight:700;color:#1f2937;margin-bottom:8px;'>"
            f"Blockers ({len(blockers)})</div>"
        )
        for b in blockers:
            if not isinstance(b, dict):
                continue
            severity = b.get("severity", "medium")
            sev_color = _READINESS_SEVERITY_COLORS.get(severity, "#6b7280")
            label = _escape(str(b.get("label", "")))
            btype = b.get("type", "")
            items = b.get("items", [])

            parts.append(
                f"<div style='border-left:3px solid {sev_color};padding:8px 12px;"
                f"margin-bottom:8px;background:#fefefe;border-radius:0 6px 6px 0;'>"
                f"<div style='display:flex;gap:8px;align-items:center;'>"
                f"<span style='display:inline-block;padding:1px 8px;border-radius:8px;"
                f"background:{sev_color};color:white;font-size:10px;font-weight:700;"
                f"text-transform:uppercase;'>{_escape(severity)}</span>"
                f"<span style='font-size:13px;color:#1f2937;'>{label}</span>"
                f"</div>"
            )

            if items and btype in ("low_coverage", "proof_gap", "unverified_critical"):
                parts.append("<ul style='margin:4px 0 0;padding-left:18px;font-size:12px;color:#374151;'>")
                for it in items[:5]:
                    if not isinstance(it, dict):
                        continue
                    t = _escape(str(it.get("title", it.get("issue_id", "")[:8])))
                    parts.append(f"<li style='margin:2px 0;'>{t}</li>")
                if len(items) > 5:
                    parts.append(f"<li style='margin:2px 0;color:#9ca3af;'>+{len(items)-5} more</li>")
                parts.append("</ul>")
            elif items and btype == "high_materiality_gaps":
                parts.append("<ul style='margin:4px 0 0;padding-left:18px;font-size:12px;color:#374151;'>")
                for it in items[:5]:
                    if not isinstance(it, dict):
                        continue
                    desc = _escape(str(it.get("description", "")))[:80]
                    parts.append(f"<li style='margin:2px 0;'>{desc}</li>")
                if len(items) > 5:
                    parts.append(f"<li style='margin:2px 0;color:#9ca3af;'>+{len(items)-5} more</li>")
                parts.append("</ul>")

            parts.append("</div>")
        parts.append("</div>")
    else:
        parts.append(
            "<div style='text-align:center;padding:16px;color:#059669;font-weight:600;'>"
            "No blockers detected — investigation is ready for reliance."
            "</div>"
        )

    parts.append("</div>")
    return "".join(parts)


# Query Context Inspector (SO-1, SO-3, SO-4, SO-7)

_QUERY_CONTEXT_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Case Context",
        "subtitle": "what Irys will carry into the next legal analysis",
        "issues": "Issues",
        "gaps": "Evidence Gaps",
        "weakest": "Weakest Issue",
        "actors": "Parties & Witnesses",
        "predicates": "Typed Claim Signals",
        "empty": "No query context available — run an investigation first.",
    },
    "finance": {
        "title": "Diligence Context",
        "subtitle": "what Irys will carry into the next financial analysis",
        "issues": "Diligence Questions",
        "gaps": "Diligence Gaps",
        "weakest": "Weakest Diligence Question",
        "actors": "Companies & Executives",
        "predicates": "Metric Relationship Signals",
        "empty": "No query context available — run an investigation first.",
    },
    "coding": {
        "title": "Codebase Context",
        "subtitle": "what Irys will carry into the next engineering analysis",
        "issues": "Engineering Objectives",
        "gaps": "Implementation Gaps",
        "weakest": "Weakest Objective",
        "actors": "Components & Owners",
        "predicates": "Dependency Signals",
        "empty": "No query context available — run an investigation first.",
    },
    "academic_research": {
        "title": "Research Context",
        "subtitle": "what Irys will carry into the next literature analysis",
        "issues": "Research Questions",
        "gaps": "Literature Gaps",
        "weakest": "Weakest Research Question",
        "actors": "Authors & Institutions",
        "predicates": "Claim Relationship Signals",
        "empty": "No query context available — run an investigation first.",
    },
    "biomedical": {
        "title": "Clinical Evidence Context",
        "subtitle": "what Irys will carry into the next biomedical analysis",
        "issues": "Clinical Questions",
        "gaps": "Evidence Gaps",
        "weakest": "Weakest Clinical Question",
        "actors": "Sponsors, Regulators & Cohorts",
        "predicates": "Clinical Relationship Signals",
        "empty": "No query context available — run an investigation first.",
    },
}


def _fmt_query_context(data: dict, domain: str = "legal") -> str:
    L = _QUERY_CONTEXT_LABELS.get(domain, _QUERY_CONTEXT_LABELS["legal"])
    if not data or not isinstance(data, dict):
        return f"<div class='viz-empty'>{_escape(L['empty'])}</div>"
    if err := _error_html(data):
        return err

    assertions = int(data.get("existing_assertion_count", 0)) if isinstance(data.get("existing_assertion_count"), (int, float)) and math.isfinite(float(data.get("existing_assertion_count", 0))) else 0
    actors = int(data.get("existing_actor_count", 0)) if isinstance(data.get("existing_actor_count"), (int, float)) and math.isfinite(float(data.get("existing_actor_count", 0))) else 0
    known_docs = _safe_list(data.get("known_document_ids"))
    doc_count = len(known_docs)
    card_count = int(data.get("document_card_count", 0)) if isinstance(data.get("document_card_count"), (int, float)) and math.isfinite(float(data.get("document_card_count", 0))) else 0
    assumption_count = len(_safe_list(data.get("active_assumptions")))

    open_issues = _safe_list(data.get("open_issues"))
    open_gaps = _safe_list(data.get("open_gaps"))
    weakest_id = data.get("weakest_issue_id")
    known_actors = _safe_list(data.get("known_actors"))
    clarifications = _safe_list(data.get("answered_clarifications"))
    annotations = _safe_list(data.get("document_annotations"))
    predicates = _safe_list(data.get("key_predicates"))
    domain_facets = _safe_list(data.get("domain_facets"))
    trust_weights = data.get("composed_trust_weights", {}) or {}
    if not isinstance(trust_weights, dict):
        trust_weights = {}
    profile_id = data.get("primary_domain_profile_id") or "none"

    parts: list[str] = []
    parts.append(f"<div style='margin-bottom:16px;'>")
    parts.append(
        f"<h3 style='margin:0 0 4px;'>{_escape(L['title'])}</h3>"
        f"<div style='font-size:12px;color:#6b7280;margin-bottom:12px;'>{_escape(L['subtitle'])}</div>"
    )

    # KPI row
    kpi_style = "display:inline-block;padding:6px 14px;margin:0 6px 6px 0;border-radius:8px;background:#f3f4f6;font-size:13px;"
    parts.append("<div style='margin-bottom:12px;'>")
    parts.append(f"<span style='{kpi_style}'><strong>{assertions}</strong> assertions</span>")
    parts.append(f"<span style='{kpi_style}'><strong>{actors}</strong> actors</span>")
    parts.append(f"<span style='{kpi_style}'><strong>{doc_count}</strong> documents</span>")
    parts.append(f"<span style='{kpi_style}'><strong>{card_count}</strong> document cards</span>")
    parts.append(f"<span style='{kpi_style}'><strong>{assumption_count}</strong> active assumptions</span>")
    parts.append("</div>")

    # Next Run Focus
    parts.append(
        "<div style='margin-bottom:12px;padding:10px;background:#fef3c7;border:1px solid #fde68a;border-radius:6px;'>"
        "<div style='font-weight:700;font-size:13px;margin-bottom:6px;'>Next Run Focus</div>"
    )
    if weakest_id:
        parts.append(f"<div style='font-size:12px;'>{_escape(L['weakest'])}: <strong>{_escape(str(weakest_id)[:40])}</strong></div>")
    parts.append(f"<div style='font-size:12px;'>{_escape(L['issues'])}: <strong>{len(open_issues)}</strong> open</div>")
    parts.append(f"<div style='font-size:12px;'>{_escape(L['gaps'])}: <strong>{len(open_gaps)}</strong> open</div>")
    if open_gaps:
        parts.append("<ul style='margin:4px 0 0;padding-left:18px;font-size:11px;color:#92400e;'>")
        for g in open_gaps[:5]:
            if not isinstance(g, dict):
                continue
            desc = _escape(str(g.get("description", g.get("gap_type", "")))[:60])
            parts.append(f"<li>{desc}</li>")
        if len(open_gaps) > 5:
            parts.append(f"<li style='color:#9ca3af;'>+{len(open_gaps)-5} more</li>")
        parts.append("</ul>")
    parts.append("</div>")

    # Reuse Inputs
    parts.append(
        "<div style='margin-bottom:12px;padding:10px;background:#f0fdf4;border:1px solid #bbf7d0;border-radius:6px;'>"
        "<div style='font-weight:700;font-size:13px;margin-bottom:6px;'>Reuse Inputs</div>"
    )
    parts.append(f"<div style='font-size:12px;'>Known documents: <strong>{len(known_docs)}</strong></div>")
    str_actors = [a for a in known_actors if not isinstance(a, dict)]
    if str_actors:
        parts.append(f"<div style='font-size:12px;'>{_escape(L['actors'])}: ")
        actor_tags = ", ".join(_escape(str(a)[:30]) for a in str_actors[:10])
        parts.append(f"<span style='color:#059669;'>{actor_tags}</span>")
        if len(str_actors) > 10:
            parts.append(f" <span style='color:#9ca3af;'>+{len(str_actors)-10} more</span>")
        parts.append("</div>")
    parts.append(f"<div style='font-size:12px;'>Answered clarifications: <strong>{len(clarifications)}</strong></div>")
    parts.append(f"<div style='font-size:12px;'>Document annotations: <strong>{len(annotations)}</strong></div>")
    parts.append("</div>")

    # Typed Graph Signals
    str_predicates = [p for p in predicates if not isinstance(p, dict)]
    if str_predicates:
        parts.append(
            "<div style='margin-bottom:12px;padding:10px;background:#ede9fe;border:1px solid #c4b5fd;border-radius:6px;'>"
            f"<div style='font-weight:700;font-size:13px;margin-bottom:6px;'>{_escape(L['predicates'])}</div>"
        )
        parts.append("<div style='display:flex;flex-wrap:wrap;gap:4px;'>")
        for p in str_predicates[:15]:
            parts.append(
                f"<span style='display:inline-block;padding:2px 8px;background:#ddd6fe;border-radius:4px;"
                f"font-size:11px;color:#5b21b6;'>{_escape(str(p)[:40])}</span>"
            )
        if len(str_predicates) > 15:
            parts.append(f"<span style='font-size:11px;color:#9ca3af;'>+{len(str_predicates)-15} more</span>")
        parts.append("</div></div>")

    # Domain Calibration
    parts.append(
        "<div style='padding:10px;background:#e0f2fe;border:1px solid #7dd3fc;border-radius:6px;'>"
        "<div style='font-weight:700;font-size:13px;margin-bottom:6px;'>Domain Calibration</div>"
        f"<div style='font-size:12px;'>Primary profile: <strong>{_escape(str(profile_id))}</strong></div>"
    )
    if domain_facets:
        parts.append("<div style='font-size:12px;margin-top:4px;'>Active facets: ")
        facet_tags = []
        for f in domain_facets[:8]:
            if not isinstance(f, dict):
                continue
            fname = _escape(str(f.get("facet_name", f.get("name", "")))[:30])
            if fname:
                facet_tags.append(fname)
        parts.append(", ".join(facet_tags) if facet_tags else "none")
        if len(domain_facets) > 8:
            parts.append(f" +{len(domain_facets)-8} more")
        parts.append("</div>")
    if trust_weights:
        parts.append("<div style='font-size:12px;margin-top:4px;'>Trust weights: ")
        tw_parts = []
        for k, v in list(trust_weights.items())[:8]:
            val = float(v) if isinstance(v, (int, float)) and math.isfinite(float(v)) else 0.0
            tw_parts.append(f"{_escape(str(k)[:20])}={val:.2f}")
        parts.append(", ".join(tw_parts))
        if len(trust_weights) > 8:
            parts.append(f" +{len(trust_weights)-8} more")
        parts.append("</div>")
    parts.append("</div>")

    parts.append("</div>")
    return "".join(parts)


def _fmt_gaps(gaps: list, clarifications: list, domain: str = "legal", issue_titles: dict | None = None) -> str:
    labels = _GAP_LABELS.get(domain, _GAP_LABELS["legal"])
    _titles = issue_titles or {}
    parts: list[str] = []
    if gaps:
        doc_gaps = [
            g for g in gaps
            if isinstance(g, dict) and g.get("gap_type") in _MISSING_DOC_TYPES
        ]
        if doc_gaps:
            doc_header = _MISSING_DOC_HEADER.get(domain, _MISSING_DOC_HEADER["legal"])
            checklist = ""
            for dg in doc_gaps:
                desc = _escape(str(dg.get("description") or "?"))
                mat = _safe_float(dg.get("materiality_score") or dg.get("materiality") or 0)
                urgency = "color:#dc2626;font-weight:700" if mat >= 0.7 else "color:#92400e" if mat >= 0.4 else "color:#6b7280"
                dep_names = []
                for d in (dg.get("dependencies") or [])[:3]:
                    if not isinstance(d, dict):
                        continue
                    t = _titles.get(d.get("affected_id", ""), "")
                    if t:
                        dep_names.append(_escape(t[:25]))
                dep_note = f" — blocks: {', '.join(dep_names)}" if dep_names else ""
                checklist += (
                    f"<li style='margin:4px 0;{urgency}'>"
                    f"{'&#9744;' if mat >= 0.5 else '&#9634;'} {desc}{dep_note}</li>"
                )
            parts.append(
                "<div style='background:#fef2f2;border:1px solid #fecaca;border-radius:8px;"
                "padding:10px 14px;margin-bottom:12px;'>"
                f"<div style='font-weight:700;font-size:13px;color:#991b1b;margin-bottom:6px;'>"
                f"{_escape(doc_header)} ({len(doc_gaps)})</div>"
                f"<ul style='list-style:none;padding:0;margin:0;'>{checklist}</ul>"
                "</div>"
            )
        gap_rows = ""
        for g in gaps:
            if not isinstance(g, dict):
                continue
            desc = _escape(str(g.get("description") or g.get("gap_type", "?")))
            gap_type = str(g.get("gap_type") or "")
            pill = labels.get(gap_type)
            if isinstance(pill, tuple):
                label, cls = pill
            else:
                label, cls = gap_type.replace("_", " ").title(), "pill-neutral"
            mat = g.get("materiality_score") or g.get("materiality") or 0
            mat_val = float(mat) if isinstance(mat, (int, float)) else 0.0
            mat_pct = min(mat_val * 100, 100)
            mat_bar = (
                f"<div style='width:60px;height:8px;background:#e5e7eb;border-radius:4px;display:inline-block;vertical-align:middle'>"
                f"<div style='width:{mat_pct:.0f}%;height:100%;background:{'#dc2626' if mat_val >= 0.7 else '#f59e0b' if mat_val >= 0.4 else '#6b7280'};border-radius:4px'></div>"
                f"</div> {mat_val:.2f}"
            )
            deps = g.get("dependencies") or []
            dep_str = ""
            if deps:
                dep_parts = []
                for d in deps[:4]:
                    if not isinstance(d, dict):
                        continue
                    a_type = d.get("affected_type", "?")
                    a_id = d.get("affected_id", "")
                    title = _titles.get(a_id, "")
                    if title:
                        dep_parts.append(f"{_escape(title[:30])}")
                    else:
                        dep_parts.append(_escape(a_type))
                dep_str = f"<span style='font-size:10px;color:#6b7280'>{', '.join(dep_parts)}</span>"
            gap_id = _escape(str(g.get("id", "?"))[:16])
            full_gap_id = _escape(str(g.get("id", "?")))
            gap_rows += (
                f"<tr>"
                f"<td><span class='pill {cls}'>{_escape(label)}</span></td>"
                f"<td>{desc}</td>"
                f"<td>{mat_bar}</td>"
                f"<td>{dep_str}</td>"
                f"<td><code style='font-size:10px;cursor:pointer;' title='{full_gap_id}'>{gap_id}</code></td>"
                f"</tr>"
            )
        gap_count = len([g for g in gaps if isinstance(g, dict)])
        parts.append(
            f"<div class='viz-header'><strong>{labels['title']}</strong> — {gap_count} unresolved</div>"
            "<div class='table-wrap'><table class='viz-table'>"
            "<thead><tr><th>Type</th><th>Description</th><th>Materiality</th><th>Affects</th><th>ID</th></tr></thead>"
            "<tbody>" + gap_rows + "</tbody></table></div>"
        )
    if clarifications:
        clar_items = ""
        for c in clarifications:
            if not isinstance(c, dict):
                continue
            q = _escape(str(c.get("question_text") or c.get("question", "?")))
            impact = _escape(str(c.get("expected_impact") or ""))
            clar_items += (
                f"<li><strong>{q}</strong>"
                + (f"<br><span style='font-size:11px;color:#6b7280'>Impact: {impact}</span>" if impact else "")
                + "</li>"
            )
        parts.append(
            "<div class='viz-header' style='margin-top:12px'><strong>Pending Clarifications</strong></div>"
            "<ul style='margin:4px 0;padding-left:20px'>" + clar_items + "</ul>"
        )
    return (
        "<div class='viz-shell'>" + "".join(parts) + "</div>"
        if parts else f"<div class='viz-empty'>{labels['empty']}</div>"
    )


_REVIEW_BUCKET_LABELS: dict[str, dict[int, tuple[str, str]]] = {
    "legal": {
        0: ("Gap blocker", "#b91c1c"),
        1: ("Disputed", "#b45309"),
        2: ("Supports a claim", "#2563eb"),
        3: ("Element of proof", "#0d9488"),
        4: ("Number", "#7c3aed"),
        5: ("Citation / Authority", "#6b7280"),
        6: ("Other", "#94a3b8"),
    },
    "finance": {
        0: ("Gap blocker", "#b91c1c"),
        1: ("Disputed", "#b45309"),
        2: ("Supports a position", "#2563eb"),
        3: ("Compliance element", "#0d9488"),
        4: ("Figure", "#7c3aed"),
        5: ("Reference / Standard", "#6b7280"),
        6: ("Other", "#94a3b8"),
    },
    "coding": {
        0: ("Gap blocker", "#b91c1c"),
        1: ("Disputed", "#b45309"),
        2: ("Supports a finding", "#2563eb"),
        3: ("Acceptance criterion", "#0d9488"),
        4: ("Metric", "#7c3aed"),
        5: ("Spec / Standard", "#6b7280"),
        6: ("Other", "#94a3b8"),
    },
    "academic_research": {
        0: ("Gap blocker", "#b91c1c"),
        1: ("Disputed", "#b45309"),
        2: ("Supports a hypothesis", "#2563eb"),
        3: ("Methodological criterion", "#0d9488"),
        4: ("Statistic", "#7c3aed"),
        5: ("Citation / Reference", "#6b7280"),
        6: ("Other", "#94a3b8"),
    },
    "biomedical": {
        0: ("Gap blocker", "#b91c1c"),
        1: ("Disputed", "#b45309"),
        2: ("Supports a conclusion", "#2563eb"),
        3: ("Endpoint criterion", "#0d9488"),
        4: ("Measurement", "#7c3aed"),
        5: ("Protocol / Guideline", "#6b7280"),
        6: ("Other", "#94a3b8"),
    },
}

_REVIEW_KIND_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "assertion": "Fact",
        "assertion_occurrence": "Raw utterance",
        "evidence_edge": "Evidence link",
        "issue_predicate": "Proof element",
        "quant_fact": "Number",
        "authority": "Citation / Authority",
        "document_card": "Document classification",
    },
    "finance": {
        "assertion": "Finding",
        "assertion_occurrence": "Raw extract",
        "evidence_edge": "Evidence link",
        "issue_predicate": "Compliance element",
        "quant_fact": "Figure",
        "authority": "Reference / Standard",
        "document_card": "Document classification",
    },
    "coding": {
        "assertion": "Finding",
        "assertion_occurrence": "Raw snippet",
        "evidence_edge": "Evidence link",
        "issue_predicate": "Acceptance criterion",
        "quant_fact": "Metric",
        "authority": "Spec / Standard",
        "document_card": "Artifact classification",
    },
    "academic_research": {
        "assertion": "Claim",
        "assertion_occurrence": "Raw excerpt",
        "evidence_edge": "Evidence link",
        "issue_predicate": "Methodological criterion",
        "quant_fact": "Statistic",
        "authority": "Citation / Reference",
        "document_card": "Source classification",
    },
    "biomedical": {
        "assertion": "Finding",
        "assertion_occurrence": "Raw observation",
        "evidence_edge": "Evidence link",
        "issue_predicate": "Endpoint criterion",
        "quant_fact": "Measurement",
        "authority": "Protocol / Guideline",
        "document_card": "Source classification",
    },
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


def _fmt_review_count_badge(total: int, bucket_counts: dict[int, int], domain: str = "legal") -> str:
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
    bucket_labels = _REVIEW_BUCKET_LABELS.get(domain, _REVIEW_BUCKET_LABELS["legal"])
    lines: list[str] = []
    for bucket in sorted(bucket_counts):
        label, color = bucket_labels.get(
            bucket, bucket_labels[6],
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


def _review_bucket_badge(bucket: int, score: float, domain: str = "legal") -> str:
    bucket_labels = _REVIEW_BUCKET_LABELS.get(domain, _REVIEW_BUCKET_LABELS["legal"])
    label, color = bucket_labels.get(
        int(bucket) if bucket is not None else 6, bucket_labels[6],
    )
    meter = ""
    if score and score > 0:
        meter = f" <span style='opacity:0.7;font-size:11px;'>· materiality {score:.0%}</span>"
    return (
        f"<span style='display:inline-block;padding:2px 8px;border-radius:10px;"
        f"background:{color};color:white;font-size:11px;font-weight:600;"
        f"text-transform:uppercase;letter-spacing:0.03em;'>{label}</span>{meter}"
    )


def _fmt_review_queue(queue: list[dict], domain: str = "legal") -> str:
    """Render the prioritized review queue in domain-professional-readable form.

    No raw IDs, no internal store names. Each row shows:
      - bucket badge (proof-critical / contradicted / issue-linked / etc.)
      - the finding text or citation
      - the kind tag (domain-aware label)
      - a copyable target id truncated for selection — hidden text, not shown
    """
    if not queue:
        return (
            "<div class='viz-empty'>"
            "No findings need review. The matter model is fully reviewed "
            "or has no AI-extracted material yet."
            "</div>"
        )
    kind_labels = _REVIEW_KIND_LABELS.get(domain, _REVIEW_KIND_LABELS["legal"])
    rows_html = []
    for row in queue:
        if not isinstance(row, dict):
            continue
        bucket = row.get("priority_bucket", 6)
        score = row.get("priority_score", 0.0)
        kind = row.get("target_kind", "unknown")
        text = (
            row.get("proposition_text")
            or row.get("quant_raw_text")
            or row.get("authority_citation")
            or row.get("predicate_description")
            or "(no preview)"
        )
        kind_pretty = kind_labels.get(kind, kind.replace("_", " ").title())
        truncated = _truncate(text, 180)
        source_doc = row.get("source_doc_label")
        source_section = row.get("source_section_label") or row.get("source_span_id")
        source_html = ""
        if source_doc:
            section_part = (
                f" &middot; {_escape(_truncate(str(source_section), 48))}"
                if source_section
                else ""
            )
            source_html = (
                f"<div style='margin-top:4px;font-size:12px;color:#6b7280;'>"
                f"&#128196; {_escape(_truncate(str(source_doc), 110))}{section_part}"
                f"</div>"
            )
        rows_html.append(
            f"<div style='padding:10px 12px;border-left:3px solid #e5e7eb;"
            f"margin-bottom:8px;background:#f9fafb;border-radius:0 6px 6px 0;'>"
            f"<div style='margin-bottom:4px;'>"
            f"{_review_bucket_badge(bucket, score, domain)}"
            f"<span style='margin-left:10px;font-size:12px;color:#6b7280;"
            f"text-transform:uppercase;letter-spacing:0.03em;'>{kind_pretty}</span>"
            f"</div>"
            f"<div style='color:#1f2937;line-height:1.45;'>{_escape(truncated)}</div>"
            f"{source_html}"
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


_SOURCE_DRAWER_REVIEWER_LABELS: dict[str, dict[str, str]] = {
    "legal": {"user": "You", "attorney": "Attorney", "system": "Irys", "import": "Bulk import"},
    "finance": {"user": "You", "attorney": "Analyst", "system": "Irys", "import": "Bulk import"},
    "coding": {"user": "You", "attorney": "Engineer", "system": "Irys", "import": "Bulk import"},
    "academic_research": {"user": "You", "attorney": "Reviewer", "system": "Irys", "import": "Bulk import"},
    "biomedical": {"user": "You", "attorney": "Clinician", "system": "Irys", "import": "Bulk import"},
}


def _fmt_source_drawer(
    target_kind: str,
    target_id: str,
    provenance_rows: list[dict],
    verification_events: list[dict],
    domain: str = "legal",
) -> str:
    """Professional-readable "where did this come from + what's been done
    to it" drawer. Combines P0.1 provenance (AI write trail) with
    P0.3 verification events (human review history)."""
    if not provenance_rows and not verification_events:
        return (
            "<div class='viz-empty'>No source or review history for this finding. "
            "Try an item created during a live investigation.</div>"
        )
    sections: list[str] = []

    reviewer_labels = _SOURCE_DRAWER_REVIEWER_LABELS.get(
        domain, _SOURCE_DRAWER_REVIEWER_LABELS["legal"]
    )
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
            if not isinstance(ev, dict):
                continue
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
            if not isinstance(ev, dict):
                continue
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
        if not isinstance(row, dict):
            continue
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
        doc = row.get("source_doc_label")
        doc_part = f" [{_truncate(str(doc), 36)}]" if doc else ""
        label = f"{kind_pretty}: {_truncate(text, 90)}{doc_part}"
        choices.append((label, f"{kind}:{tid}"))
    return choices


def _fmt_steering(actions: list, domain: str = "legal") -> str:
    """Format get_ledger_steering_surface() output as actionable recommendations."""
    L = _STEERING_ACTION_LABELS.get(domain, _STEERING_ACTION_LABELS["legal"])
    if not actions:
        return L["empty"]
    lines = [f"### {L['title']}\n"]
    for a in actions:
        if not isinstance(a, dict):
            continue
        action_type = a.get("action_type", "unknown")
        action_label = L.get(action_type, action_type.replace("_", " ").title())
        description = _escape(a.get("description", ""))
        rationale = _escape(a.get("rationale", ""))
        priority = _escape(str(a.get("priority", "")))
        priority_str = f" **[{priority.upper()}]**" if priority else ""
        lines.append(f"**{action_label}**{priority_str}: {description}")
        if rationale:
            lines.append(f"  > {rationale}")
        params = a.get("params", {})
        if isinstance(params, dict) and params:
            param_str = " | ".join(f"`{_escape(str(k))}: {_escape(str(v))}`" for k, v in params.items() if v)
            lines.append(f"  *Params: {param_str}*")
        lines.append("")
    return "\n".join(lines)


_STEERING_ACTION_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Recommended Next Steps",
        "empty": "No recommendations. The matter model is in good shape.",
        "correct_assertion": "Correct a Fact",
        "force_belief_state": "Override Belief State",
        "redirect_focus": "Redirect Investigation",
        "supply_document": "Supply Missing Document",
        "answer_clarification": "Answer Clarification",
        "set_trust_override": "Adjust Source Trust",
    },
    "finance": {
        "title": "Recommended Actions",
        "empty": "No recommendations. The analysis model is consistent.",
        "correct_assertion": "Correct a Finding",
        "force_belief_state": "Override Assessment",
        "redirect_focus": "Redirect Analysis",
        "supply_document": "Supply Missing Data",
        "answer_clarification": "Clarify Assumption",
        "set_trust_override": "Adjust Source Reliability",
    },
    "coding": {
        "title": "Suggested Improvements",
        "empty": "No suggestions. The code model is consistent.",
        "correct_assertion": "Correct Finding",
        "force_belief_state": "Override Status",
        "redirect_focus": "Redirect Focus",
        "supply_document": "Supply Missing File",
        "answer_clarification": "Clarify Requirement",
        "set_trust_override": "Adjust Source Trust",
    },
    "academic_research": {
        "title": "Research Recommendations",
        "empty": "No recommendations. The literature model is consistent.",
        "correct_assertion": "Correct Claim",
        "force_belief_state": "Override Assessment",
        "redirect_focus": "Redirect Inquiry",
        "supply_document": "Supply Missing Source",
        "answer_clarification": "Clarify Methodology",
        "set_trust_override": "Adjust Source Authority",
    },
    "biomedical": {
        "title": "Clinical Recommendations",
        "empty": "No recommendations. The clinical model is consistent.",
        "correct_assertion": "Correct Finding",
        "force_belief_state": "Override Determination",
        "redirect_focus": "Redirect Analysis",
        "supply_document": "Supply Missing Record",
        "answer_clarification": "Clarify Protocol",
        "set_trust_override": "Adjust Source Weight",
    },
}

_PRIORITY_PILLS: dict[str, tuple[str, str]] = {
    "high": ("HIGH", "pill-red"),
    "medium": ("MEDIUM", "pill-orange"),
    "low": ("LOW", "pill-neutral"),
}


def _fmt_steering_panel(actions: list[dict], domain: str = "legal") -> str:
    labels = _STEERING_ACTION_LABELS.get(domain, _STEERING_ACTION_LABELS["legal"])
    if not actions:
        return f"<div class='viz-empty'>{labels['empty']}</div>"
    rows = ""
    for a in actions:
        if not isinstance(a, dict):
            continue
        action_type = str(a.get("action_type", "unknown"))
        description = _escape(a.get("description", ""))
        rationale = _escape(a.get("rationale", ""))
        impact = _escape(a.get("impact", ""))
        priority = a.get("priority", "low")
        pill_text, pill_cls = _PRIORITY_PILLS.get(priority, ("—", "pill-neutral"))
        action_label = _escape(labels.get(action_type, action_type.replace("_", " ").title()))
        params = a.get("params") or {}
        param_html = ""
        if isinstance(params, dict) and params:
            param_items = " ".join(
                f"<code>{_escape(k)}={_escape(str(v)[:40])}</code>"
                for k, v in params.items() if v
            )
            param_html = f"<div style='margin-top:4px;font-size:11px;color:#6b7280'>{param_items}</div>"
        rows += (
            "<tr>"
            f"<td><span class='pill {pill_cls}'>{pill_text}</span></td>"
            f"<td><strong>{action_label}</strong></td>"
            f"<td>{description}"
            + (f"<br><span style='font-size:11px;color:#6b7280'>{rationale}</span>" if rationale else "")
            + param_html
            + "</td>"
            f"<td style='font-size:12px'>{impact}</td>"
            "</tr>"
        )
    if not rows:
        return f"<div class='viz-empty'>{labels['empty']}</div>"
    count = len([a for a in actions if isinstance(a, dict)])
    high_count = sum(1 for a in actions if isinstance(a, dict) and a.get("priority") == "high")
    subtitle = f"{count} recommendation{'s' if count != 1 else ''}"
    if high_count:
        subtitle += f" ({high_count} high priority)"
    return (
        "<div class='viz-shell'>"
        f"<div class='viz-header'><strong>{labels['title']}</strong> — {subtitle}</div>"
        "<div class='matrix-wrap'><table class='analytics-table'><thead><tr>"
        "<th>Priority</th><th>Action</th><th>Details</th><th>Impact</th>"
        "</tr></thead><tbody>"
        + rows
        + "</tbody></table></div></div>"
    )


_BELIEF_STATE_COLORS: dict[str, str] = {
    "accepted": "#16a34a",
    "supported": "#2563eb",
    "disputed": "#d97706",
    "rejected": "#dc2626",
    "undetermined": "#6b7280",
}

_ORIGIN_LABELS: dict[str, dict[str, str]] = {
    "legal": {"ai_extracted": "AI-Extracted", "attorney_annotated": "Attorney Note",
              "system_inferred": "System Inferred", "imported": "Imported", "legacy_backfill": "Backfill"},
    "finance": {"ai_extracted": "AI-Extracted", "attorney_annotated": "Analyst Note",
                "system_inferred": "System Inferred", "imported": "Imported", "legacy_backfill": "Backfill"},
    "coding": {"ai_extracted": "AI-Extracted", "attorney_annotated": "Engineer Note",
               "system_inferred": "System Inferred", "imported": "Imported", "legacy_backfill": "Backfill"},
    "academic_research": {"ai_extracted": "AI-Extracted", "attorney_annotated": "Reviewer Note",
                          "system_inferred": "System Inferred", "imported": "Imported", "legacy_backfill": "Backfill"},
    "biomedical": {"ai_extracted": "AI-Extracted", "attorney_annotated": "Clinician Note",
                   "system_inferred": "System Inferred", "imported": "Imported", "legacy_backfill": "Backfill"},
}

_LINKED_ISSUES_LABELS: dict[str, dict[str, str]] = {
    "legal": {"header": "Linked Issues", "supports": "supports", "attacks": "attacks", "establishes": "establishes", "negates": "negates"},
    "finance": {"header": "Linked Theses", "supports": "supports", "attacks": "challenges", "establishes": "establishes", "negates": "negates"},
    "coding": {"header": "Linked Tasks", "supports": "supports", "attacks": "blocks", "establishes": "establishes", "negates": "negates"},
    "academic_research": {"header": "Linked Questions", "supports": "supports", "attacks": "contradicts", "establishes": "establishes", "negates": "negates"},
    "biomedical": {"header": "Linked Hypotheses", "supports": "supports", "attacks": "contradicts", "establishes": "establishes", "negates": "negates"},
}


_ASSERTION_TRACE_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Assertion Impact Trace",
        "sources": "Source Documents",
        "issues": "Affected Issues",
        "dependents": "Dependent Assertions",
        "revisions": "Belief Revision History",
        "verification": "Verification Status",
    },
    "finance": {
        "title": "Data Point Impact Trace",
        "sources": "Source Documents",
        "issues": "Affected Positions",
        "dependents": "Dependent Data Points",
        "revisions": "Revision History",
        "verification": "Verification Status",
    },
    "coding": {
        "title": "Finding Impact Trace",
        "sources": "Source Artifacts",
        "issues": "Affected Requirements",
        "dependents": "Dependent Findings",
        "revisions": "Revision History",
        "verification": "Verification Status",
    },
    "academic_research": {
        "title": "Claim Impact Trace",
        "sources": "Source Citations",
        "issues": "Affected Claims",
        "dependents": "Dependent Claims",
        "revisions": "Revision History",
        "verification": "Verification Status",
    },
    "biomedical": {
        "title": "Finding Impact Trace",
        "sources": "Source Records",
        "issues": "Affected Findings",
        "dependents": "Dependent Findings",
        "revisions": "Revision History",
        "verification": "Verification Status",
    },
}


def _fmt_assertion_trace(data: dict, domain: str = "legal") -> str:
    if not data or not isinstance(data, dict):
        return "<div class='viz-empty'>No trace data available.</div>"
    if err := _error_html(data):
        return err

    L = _ASSERTION_TRACE_LABELS.get(domain, _ASSERTION_TRACE_LABELS["legal"])
    prop = _escape(str(data.get("proposition_text", "")))
    bs = _escape(str(data.get("belief_state", "")))
    conf = _safe_float(data.get("confidence", 0))
    speech = _escape(str(data.get("speech_act", "")))
    impact = data.get("impact_summary", {})
    if not isinstance(impact, dict):
        impact = {}

    bs_colors = {
        "operative": "#059669", "admitted": "#059669", "resolved": "#059669",
        "alleged": "#f59e0b", "argued": "#f59e0b", "inferred": "#f59e0b",
        "disputed": "#dc2626", "withdrawn": "#9ca3af", "superseded": "#9ca3af",
    }
    bs_color = bs_colors.get(bs.lower(), "#6b7280")

    parts = [
        f"<div style='margin-bottom:16px;'>",
        f"<h3 style='margin:0 0 8px;'>{_escape(L['title'])}</h3>",
        f"<div style='padding:12px;background:#f9fafb;border-radius:8px;margin-bottom:12px;"
        f"border-left:4px solid {bs_color};'>"
        f"<div style='font-size:14px;color:#1f2937;line-height:1.5;margin-bottom:6px;'>{prop}</div>"
        f"<div style='display:flex;gap:12px;font-size:12px;color:#6b7280;'>"
        f"<span>Belief: <strong style='color:{bs_color};'>{bs}</strong></span>"
        f"<span>Confidence: <strong>{conf:.2f}</strong></span>"
        f"<span>Speech act: <strong>{speech}</strong></span>"
        f"</div></div>",
    ]

    # Impact summary metrics
    parts.append(
        f"<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:16px;'>"
    )
    for label, val in [
        (L["sources"], str(impact.get("source_doc_count", 0))),
        (L["issues"], str(impact.get("issues_affected", 0))),
        (L["dependents"], str(impact.get("dependents_count", 0))),
        (L["revisions"], str(impact.get("revision_count", 0))),
    ]:
        parts.append(
            f"<div style='text-align:center;padding:8px;background:#f9fafb;border-radius:8px;'>"
            f"<div style='font-size:20px;font-weight:700;color:#1f2937;'>{_escape(val)}</div>"
            f"<div style='font-size:11px;color:#6b7280;'>{_escape(label)}</div></div>"
        )
    parts.append("</div>")

    # Source documents
    sources = data.get("source_documents", [])
    if sources:
        parts.append(
            f"<div style='margin-bottom:12px;'>"
            f"<div style='font-size:13px;font-weight:700;margin-bottom:6px;'>"
            f"{_escape(L['sources'])} ({len(sources)})</div>"
        )
        for s in sources[:8]:
            if not isinstance(s, dict):
                continue
            doc = _escape(str(s.get("document_label", "—")))
            sec = _escape(str(s.get("section_label", "") or ""))
            parts.append(
                f"<div style='padding:4px 8px;font-size:12px;border-bottom:1px solid #f3f4f6;'>"
                f"&#128196; {doc}"
                + (f" <span style='color:#6b7280;'>({sec})</span>" if sec else "")
                + "</div>"
            )
        parts.append("</div>")

    # Affected issues
    issues = data.get("affected_issues", [])
    if issues:
        parts.append(
            f"<div style='margin-bottom:12px;'>"
            f"<div style='font-size:13px;font-weight:700;margin-bottom:6px;'>"
            f"{_escape(L['issues'])} ({len(issues)})</div>"
            f"<div style='display:flex;flex-wrap:wrap;gap:6px;'>"
        )
        for iss in issues[:8]:
            if not isinstance(iss, dict):
                continue
            title = _escape(str(iss.get("title", ""))[:40])
            mat = float(iss.get("materiality", 0))
            mat_color = "#dc2626" if mat >= 0.7 else "#f59e0b" if mat >= 0.4 else "#6b7280"
            parts.append(
                f"<span style='display:inline-block;padding:3px 10px;border-radius:10px;"
                f"background:#f3f4f6;font-size:12px;border-left:3px solid {mat_color};'>"
                f"{title}</span>"
            )
        parts.append("</div></div>")

    # Dependent assertions
    deps = data.get("dependent_assertions", [])
    if deps:
        parts.append(
            f"<div style='margin-bottom:12px;'>"
            f"<div style='font-size:13px;font-weight:700;margin-bottom:6px;'>"
            f"{_escape(L['dependents'])} ({len(deps)})</div>"
        )
        for d in deps[:5]:
            if not isinstance(d, dict):
                continue
            dprop = _escape(str(d.get("proposition_text", ""))[:60])
            dbs = _escape(str(d.get("belief_state", "")))
            dbs_color = bs_colors.get(dbs.lower(), "#6b7280")
            parts.append(
                f"<div style='padding:4px 8px;font-size:12px;border-bottom:1px solid #f3f4f6;'>"
                f"{dprop} <span style='color:{dbs_color};font-weight:600;'>({dbs})</span></div>"
            )
        parts.append("</div>")

    # Revision history
    revisions = data.get("revision_history", [])
    if revisions:
        parts.append(
            f"<div style='margin-bottom:12px;'>"
            f"<div style='font-size:13px;font-weight:700;margin-bottom:6px;'>"
            f"{_escape(L['revisions'])}</div>"
        )
        for r in revisions[:5]:
            if not isinstance(r, dict):
                continue
            old = _escape(str(r.get("old_state", "")))
            new = _escape(str(r.get("new_state", "")))
            cause = _escape(str(r.get("cause", "")))
            parts.append(
                f"<div style='padding:4px 8px;font-size:12px;border-bottom:1px solid #f3f4f6;'>"
                f"<strong>{old}</strong> → <strong>{new}</strong>"
                f" <span style='color:#6b7280;'>({cause})</span></div>"
            )
        parts.append("</div>")

    # Verification status
    ver = data.get("verification", {})
    if isinstance(ver, dict) and ver:
        vstatus = _escape(str(ver.get("status", "candidate")))
        vcolor = "#059669" if vstatus == "verified" else "#dc2626" if vstatus == "rejected" else "#f59e0b"
        parts.append(
            f"<div style='margin-bottom:8px;'>"
            f"<div style='font-size:13px;font-weight:700;margin-bottom:4px;'>"
            f"{_escape(L['verification'])}</div>"
            f"<span style='display:inline-block;padding:3px 12px;border-radius:10px;"
            f"background:{vcolor};color:white;font-size:12px;font-weight:600;'>"
            f"{vstatus}</span></div>"
        )

    parts.append("</div>")
    return "".join(parts)


def _fmt_assertion_inspector(health: dict, history: list | None = None, domain: str = "legal") -> str:
    if health.get("error"):
        return f"<div class='viz-empty'>Assertion not found.</div>"
    aid = _escape(health.get("assertion_id", "?"))
    prop = _escape(health.get("proposition_text", ""))
    belief = health.get("belief_state", "undetermined")
    belief_label = _domain_belief_label(belief, domain) if belief else "—"
    belief_color = _BELIEF_STATE_COLORS.get(belief, "#6b7280")
    conf = health.get("confidence", 0)
    conf_val = float(conf) if isinstance(conf, (int, float)) else 0.0
    oscillating = health.get("oscillating", False)
    support_count = health.get("support_count", 0)
    attack_count = health.get("attack_count", 0)
    has_superseding = health.get("has_superseding", False)
    support_roles = health.get("support_source_roles", [])
    attack_roles = health.get("attack_source_roles", [])

    osc_badge = (
        "<span style='background:#dc2626;color:white;padding:2px 6px;border-radius:4px;font-size:10px;margin-left:6px'>OSCILLATING</span>"
        if oscillating else ""
    )
    supersede_badge = (
        "<span style='background:#7c3aed;color:white;padding:2px 6px;border-radius:4px;font-size:10px;margin-left:6px'>SUPERSEDED</span>"
        if has_superseding else ""
    )

    header = (
        "<div style='margin-bottom:12px'>"
        f"<div style='font-size:13px;font-weight:600;margin-bottom:4px'>{prop}</div>"
        f"<div style='font-size:11px;color:#6b7280'>ID: <code>{aid}</code></div>"
        f"<div style='margin-top:6px'>"
        f"<span style='background:{belief_color};color:white;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600'>"
        f"{_escape(belief_label.upper())}</span>"
        f" <span style='font-size:11px;color:#6b7280'>confidence {conf_val:.2f}</span>"
        f"{osc_badge}{supersede_badge}"
        "</div></div>"
    )

    role_pills = lambda roles: " ".join(
        f"<span style='background:#e5e7eb;padding:1px 5px;border-radius:3px;font-size:10px'>{_escape(r)}</span>"
        for r in (roles[:5] if isinstance(roles, list) else [])
    )

    evidence_html = (
        "<div style='display:flex;gap:16px;margin-bottom:12px'>"
        f"<div style='flex:1;padding:8px;background:#f0fdf4;border-radius:6px;border-left:3px solid #16a34a'>"
        f"<div style='font-size:10px;color:#6b7280;text-transform:uppercase'>Supporting</div>"
        f"<div style='font-size:18px;font-weight:600;color:#16a34a'>{int(support_count) if isinstance(support_count, (int, float)) else 0}</div>"
        f"<div style='margin-top:4px'>{role_pills(support_roles)}</div></div>"
        f"<div style='flex:1;padding:8px;background:#fef2f2;border-radius:6px;border-left:3px solid #dc2626'>"
        f"<div style='font-size:10px;color:#6b7280;text-transform:uppercase'>Attacking</div>"
        f"<div style='font-size:18px;font-weight:600;color:#dc2626'>{int(attack_count) if isinstance(attack_count, (int, float)) else 0}</div>"
        f"<div style='margin-top:4px'>{role_pills(attack_roles)}</div></div>"
        "</div>"
    )

    linked_issues = health.get("linked_issues", [])
    linked_html = ""
    if linked_issues:
        _li_labels = _LINKED_ISSUES_LABELS.get(domain, _LINKED_ISSUES_LABELS["legal"])
        _rel_colors = {"supports": "#16a34a", "establishes": "#16a34a", "attacks": "#dc2626", "negates": "#dc2626"}
        li_rows = ""
        for li in linked_issues:
            if not isinstance(li, dict):
                continue
            iid = _escape(str(li.get("id", ""))[:20])
            title = _escape(str(li.get("title", "—"))[:60])
            rel = str(li.get("relation_type", "supports")).lower()
            rel_label = _escape(_li_labels.get(rel, rel.replace("_", " ")))
            rel_color = _rel_colors.get(rel, "#6b7280")
            status = _escape(str(li.get("status", "open")))
            materiality = li.get("materiality", 0)
            try:
                mat_val = float(materiality)
            except (TypeError, ValueError):
                mat_val = 0.0
            li_rows += (
                "<tr>"
                f"<td style='font-size:11px'>{title}</td>"
                f"<td><span style='background:{rel_color};color:white;padding:1px 6px;border-radius:3px;font-size:10px'>{rel_label}</span></td>"
                f"<td style='font-size:11px'>{status}</td>"
                f"<td style='font-size:11px'>{mat_val:.2f}</td>"
                f"<td style='font-size:10px;color:#6b7280'><code>{iid}</code></td>"
                "</tr>"
            )
        if li_rows:
            linked_html = (
                f"<div class='viz-header' style='margin-top:8px'><strong>{_escape(_li_labels['header'])}</strong></div>"
                "<div class='table-wrap'><table class='viz-table'>"
                "<thead><tr><th>Title</th><th>Relation</th><th>Status</th><th>Materiality</th><th>ID</th></tr></thead>"
                "<tbody>" + li_rows + "</tbody></table></div>"
            )

    provenance = health.get("provenance", [])
    if provenance:
        prov_rows = ""
        for p in provenance:
            if not isinstance(p, dict):
                continue
            event_kind = _escape(str(p.get("event_kind", "unknown")))
            writer = _escape(str(p.get("writer_name", "—")))
            model_id = _escape(str(p.get("model_id") or "—"))
            model_tier = _escape(str(p.get("model_tier") or "—"))
            ts = _escape(str(p.get("created_at") or "—"))
            _ol = _ORIGIN_LABELS.get(domain, _ORIGIN_LABELS["legal"])
            origin = _escape(_ol.get(event_kind, event_kind.replace("_", " ").title()))
            source_ref = _escape(str(p.get("source_document_ref") or ""))
            span_status = p.get("source_span_status", "")
            span_badge = ""
            if span_status == "present":
                span_badge = "<span style='color:#16a34a;font-size:10px'> (span linked)</span>"
            elif span_status == "missing":
                span_badge = "<span style='color:#d97706;font-size:10px'> (span missing)</span>"
            elif span_status == "not_applicable":
                span_badge = "<span style='color:#6b7280;font-size:10px'> (n/a)</span>"
            prov_rows += (
                "<tr>"
                f"<td style='font-size:11px;white-space:nowrap'>{ts[:19]}</td>"
                f"<td><span class='pill pill-neutral' style='font-size:10px;padding:1px 5px'>{origin}</span></td>"
                f"<td style='font-size:11px'>{writer}</td>"
                f"<td style='font-size:11px'>{model_id}{(' / ' + model_tier) if model_tier != '—' else ''}</td>"
                f"<td style='font-size:11px'>{source_ref}{span_badge}</td>"
                "</tr>"
            )
        if not prov_rows:
            provenance = []
    if provenance and prov_rows:
        prov_html = (
            "<div class='viz-header' style='margin-top:8px'><strong>Provenance Trail</strong></div>"
            "<div class='table-wrap'><table class='viz-table'>"
            "<thead><tr><th>Timestamp</th><th>Origin</th><th>Writer</th><th>Model</th><th>Source</th></tr></thead>"
            "<tbody>" + prov_rows + "</tbody></table></div>"
        )
    else:
        prov_html = "<div style='font-size:11px;color:#6b7280;margin-top:8px'>No provenance events recorded for this assertion.</div>"

    history_html = ""
    if history:
        hist_rows = ""
        for h in (history or []):
            if not isinstance(h, dict):
                continue
            ts = _escape(str(h.get("created_at") or "—"))
            field = _escape(str(h.get("changed_field") or "—"))
            old_val = _escape(str(h.get("old_value") or "—")[:60])
            new_val = _escape(str(h.get("new_value") or "—")[:60])
            cause = _escape(str(h.get("cause") or "—"))
            actor = _escape(str(h.get("actor_kind") or ""))
            actor_ref = _escape(str(h.get("actor_ref") or ""))
            actor_cell = f"{actor}" + (f":{actor_ref}" if actor_ref else "")
            hist_rows += (
                "<tr>"
                f"<td style='font-size:11px;white-space:nowrap'>{ts[:19]}</td>"
                f"<td style='font-size:11px'>{field}</td>"
                f"<td style='font-size:11px;color:#6b7280'>{old_val}</td>"
                f"<td style='font-size:11px;font-weight:600'>{new_val}</td>"
                f"<td style='font-size:11px'>{cause}</td>"
                f"<td style='font-size:11px;color:#6b7280'>{actor_cell}</td>"
                "</tr>"
            )
        if hist_rows:
            history_html = (
                "<div class='viz-header' style='margin-top:12px'><strong>Revision History</strong></div>"
                "<div class='table-wrap'><table class='viz-table'>"
                "<thead><tr><th>When</th><th>Field</th><th>Old</th><th>New</th><th>Cause</th><th>Actor</th></tr></thead>"
                "<tbody>" + hist_rows + "</tbody></table></div>"
            )

    return (
        "<div class='viz-shell'>"
        + header
        + evidence_html
        + linked_html
        + prov_html
        + history_html
        + "</div>"
    )


_POLICY_ACTION_COLORS: dict[str, tuple[str, str]] = {
    "allow": ("#16a34a", "white"),
    "block": ("#dc2626", "white"),
    "withhold": ("#d97706", "white"),
}

_CONTENT_POLICY_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "title": "Content Policy Audit",
        "empty": "No content policy decisions recorded yet.",
    },
    "finance": {
        "title": "Content Policy Audit",
        "empty": "No content policy decisions recorded yet.",
    },
    "coding": {
        "title": "Content Policy Audit",
        "empty": "No content policy decisions recorded yet.",
    },
    "academic_research": {
        "title": "Content Policy Audit",
        "empty": "No content policy decisions recorded yet.",
    },
    "biomedical": {
        "title": "Content Policy Audit",
        "empty": "No content policy decisions recorded yet.",
    },
}


def _fmt_content_policy_panel(decisions: list[dict], domain: str = "legal") -> str:
    labels = _CONTENT_POLICY_LABELS.get(domain, _CONTENT_POLICY_LABELS["legal"])
    if not decisions:
        return f"<div class='viz-empty'>{labels['empty']}</div>"
    rows = ""
    allow_count = 0
    block_count = 0
    withhold_count = 0
    for d in decisions:
        if not isinstance(d, dict):
            continue
        action = str(d.get("action") or "unknown")
        if action == "allow":
            allow_count += 1
        elif action == "block":
            block_count += 1
        elif action == "withhold":
            withhold_count += 1
        bg, fg = _POLICY_ACTION_COLORS.get(action, ("#6b7280", "white"))
        action_pill = f"<span style='background:{bg};color:{fg};padding:2px 6px;border-radius:4px;font-size:10px;font-weight:600'>{_escape(action.upper())}</span>"
        purpose = _escape(str(d.get("purpose") or "—"))
        target_kind = _escape(str(d.get("target_kind") or "—"))
        target_id = _escape(str(d.get("target_id") or "—"))[:24]
        reason = _escape(str(d.get("reason_code") or "—"))
        trust = _escape(str(d.get("trust_bucket") or "—"))
        audience = _escape(str(d.get("policy_audience") or "—"))
        ts = _escape(str(d.get("created_at") or "—"))
        priv_flag = d.get("privilege_flag")
        priv_badge = ""
        if priv_flag == 1 or priv_flag is True:
            priv_badge = " <span style='background:#7c3aed;color:white;padding:1px 4px;border-radius:3px;font-size:9px'>PRIV</span>"
        rows += (
            "<tr>"
            f"<td style='font-size:11px;white-space:nowrap'>{ts[:19]}</td>"
            f"<td>{action_pill}</td>"
            f"<td style='font-size:11px'>{purpose}</td>"
            f"<td style='font-size:11px'>{target_kind}:{target_id}</td>"
            f"<td style='font-size:11px'>{reason}</td>"
            f"<td style='font-size:11px'>{trust}{priv_badge}</td>"
            f"<td style='font-size:11px'>{audience}</td>"
            "</tr>"
        )
    if not rows:
        return f"<div class='viz-empty'>{labels['empty']}</div>"
    total = allow_count + block_count + withhold_count
    subtitle = f"{total} decisions"
    if block_count:
        subtitle += f" ({block_count} blocked)"
    if withhold_count:
        subtitle += f" ({withhold_count} withheld)"
    return (
        "<div class='viz-shell'>"
        f"<div class='viz-header'><strong>{labels['title']}</strong> — {subtitle}</div>"
        "<div style='display:flex;gap:12px;margin-bottom:8px'>"
        f"<div style='font-size:11px'><span style='color:#16a34a;font-weight:600'>{allow_count}</span> allowed</div>"
        f"<div style='font-size:11px'><span style='color:#dc2626;font-weight:600'>{block_count}</span> blocked</div>"
        f"<div style='font-size:11px'><span style='color:#d97706;font-weight:600'>{withhold_count}</span> withheld</div>"
        "</div>"
        "<div class='table-wrap'><table class='viz-table'>"
        "<thead><tr><th>When</th><th>Action</th><th>Purpose</th><th>Target</th><th>Reason</th><th>Trust</th><th>Audience</th></tr></thead>"
        "<tbody>" + rows + "</tbody></table></div>"
        "</div>"
    )


_EXPORT_QUANT_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "recon_title": "Payment Reconciliation",
        "invoiced": "Total Invoiced", "paid": "Total Paid",
        "disputed": "Disputed", "exposure": "Net Exposure",
        "waterfall_title": "Damages Waterfall",
        "empty": "No quantitative facts extracted yet. Run an investigation first.",
    },
    "finance": {
        "recon_title": "Transaction Reconciliation",
        "invoiced": "Total Billed", "paid": "Total Settled",
        "disputed": "Disputed", "exposure": "Net Exposure",
        "waterfall_title": "Amount Breakdown",
        "empty": "No quantitative facts extracted yet. Run an analysis first.",
    },
    "coding": {
        "recon_title": "Resource Reconciliation",
        "invoiced": "Total Allocated", "paid": "Total Consumed",
        "disputed": "Disputed", "exposure": "Net Remaining",
        "waterfall_title": "Metric Breakdown",
        "empty": "No quantitative facts extracted yet. Run an investigation first.",
    },
    "academic_research": {
        "recon_title": "Funding Reconciliation",
        "invoiced": "Total Budgeted", "paid": "Total Spent",
        "disputed": "Disputed", "exposure": "Net Remaining",
        "waterfall_title": "Amount Breakdown",
        "empty": "No quantitative facts extracted yet. Run an analysis first.",
    },
    "biomedical": {
        "recon_title": "Cost Reconciliation",
        "invoiced": "Total Charged", "paid": "Total Paid",
        "disputed": "Disputed", "exposure": "Net Exposure",
        "waterfall_title": "Amount Breakdown",
        "empty": "No quantitative facts extracted yet. Run an investigation first.",
    },
}


def _fmt_quant(payment_recon: dict, damages: list, domain: str = "legal") -> str:
    """Format quant reconciliation and damages waterfall (SO-6)."""
    L = _EXPORT_QUANT_LABELS.get(domain, _EXPORT_QUANT_LABELS["legal"])
    parts = []

    if payment_recon and payment_recon.get("invoiced") is not None:
        inv = payment_recon.get("invoiced", 0)
        paid = payment_recon.get("paid", 0)
        disputed = payment_recon.get("disputed", 0)
        exp = payment_recon.get("exposure", 0)
        currency = payment_recon.get("currency", "USD")
        parts.append(f"### {L['recon_title']}")
        parts.append(f"| Metric | Amount ({currency}) |")
        parts.append("|--------|--------|")
        parts.append(f"| {L['invoiced']} | {inv:,.2f}" if isinstance(inv, (int, float)) else f"| {L['invoiced']} | {inv}")
        parts.append(f"| {L['paid']} | {paid:,.2f}" if isinstance(paid, (int, float)) else f"| {L['paid']} | {paid}")
        if disputed:
            parts.append(f"| {L['disputed']} | {disputed:,.2f}" if isinstance(disputed, (int, float)) else f"| {L['disputed']} | {disputed}")
        parts.append(f"| **{L['exposure']}** | **{exp:,.2f}**" if isinstance(exp, (int, float)) else f"| {L['exposure']} | {exp}")

    if damages:
        parts.append(f"\n### {L['waterfall_title']}")
        parts.append("| Component | Claimed | Sources | Conflicts |")
        parts.append("|-----------|---------|---------|-----------|")
        for d in damages:
            if not isinstance(d, dict):
                continue
            comp = d.get("component") or "(uncategorised)"
            amt = d.get("claimed_amount", 0)
            amt_str = f"{amt:,.2f}" if isinstance(amt, (int, float)) else str(amt)
            srcs = d.get("source_count", 0)
            conflicts = len(d.get("conflicts", []))
            conflict_str = f"⚠️ {conflicts}" if conflicts else "—"
            parts.append(f"| {comp} | {amt_str} | {srcs} | {conflict_str} |")

    return "\n".join(parts) if parts else L["empty"]


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
                except Exception as _exc:
                    logger.warning("run_id capture failed: %s", _exc)
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
        except Exception as exc:
            logger.warning("Path resolution failed for %r: %s", repo_path, exc)
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
                        except Exception as _exc:
                            logger.warning("structured trace format failed: %s", _exc)
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
                    supporting_text = "\n".join(call_citations) or "—"
                    _envelope = (
                        getattr(state, "findings", {}) or {}
                    ).get("output_envelope")
                    _envelope_md = _fmt_output_envelope_summary(_envelope) if _envelope else ""
                    if _envelope_md or self.final_diagnostics:
                        _diag_parts: list[str] = []
                        if _envelope_md:
                            _diag_parts.append("## Output Quality\n\n" + _envelope_md)
                        if self.final_diagnostics:
                            _diag_parts.append("## Run Diagnostics & Safeguards\n\n" + self.final_diagnostics)
                        supporting_text = (
                            "\n\n---\n\n".join(_diag_parts)
                            + ("\n\n---\n\n" + supporting_text if supporting_text and supporting_text != "—" else "")
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
        except Exception as exc:
            logger.warning("Path resolution failed for %r: %s", repo_path, exc)
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
                        except Exception as _exc:
                            logger.warning("session-stream structured trace failed: %s", _exc)
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
                    _envelope = (
                        getattr(state, "findings", {}) or {}
                    ).get("output_envelope")
                    _envelope_md = _fmt_output_envelope_summary(_envelope) if _envelope else ""
                    if _envelope_md or self.final_diagnostics:
                        _diag_parts: list[str] = []
                        if _envelope_md:
                            _diag_parts.append("## Output Quality\n\n" + _envelope_md)
                        if self.final_diagnostics:
                            _diag_parts.append("## Run Diagnostics & Safeguards\n\n" + self.final_diagnostics)
                        supporting_text = (
                            "\n\n---\n\n".join(_diag_parts)
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
                except Exception as _exc:
                    logger.warning("early stop request failed: %s", _exc)
            _ASYNC_EXECUTOR.submit(_early_stop)
        # current_run_id intentionally NOT cleared here — see docstring.
        return gr.update()

    def load_overview(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded. Run an investigation first.</div>"
        try:
            data = _run_async(self.backend().get_overview(matter_id))
            return _fmt_overview_panel(data, domain=domain)
        except Exception as exc:
            logger.warning("load_overview: %s", exc)
            return f"<div class='viz-empty'>Error loading overview: {_escape(str(exc))}</div>"

    def load_issues(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            issues = _run_async(self.backend().list_issues(matter_id))
            return _fmt_issues_panel(issues, domain=domain)
        except Exception as exc:
            logger.warning("load_issues: %s", exc)
            return f"<div class='viz-empty'>Error loading issues: {_escape(str(exc))}</div>"

    def load_assertions(self, matter_id: str) -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            backend = self.backend()
            assertions = _run_async(backend.list_assertions(matter_id, limit=50))
            domain = self._detect_domain(matter_id)
            return _fmt_assertions(assertions, domain=domain)
        except Exception as exc:
            logger.warning("load_assertions: %s", exc)
            return f"<div class='viz-empty'>Error loading assertions: {_escape(str(exc))}</div>"

    def search_assertions(self, matter_id: str, query: str) -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        if not query or len(query.strip()) < 2:
            return self.load_assertions(matter_id)
        try:
            backend = self.backend()
            results = _run_async(backend.search_assertions(matter_id, query.strip(), limit=20))
            domain = self._detect_domain(matter_id)
            if not results:
                return f"<div class='viz-empty'>No facts matching “{_escape(query.strip())}”.</div>"
            return _fmt_assertions(results, domain=domain)
        except Exception as exc:
            logger.warning("search_assertions: %s", exc)
            return f"<div class='viz-empty'>Error searching: {_escape(str(exc))}</div>"

    def inspect_assertion(self, matter_id: str, assertion_id: str) -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        aid = (assertion_id or "").strip()
        if not aid:
            return "<div class='viz-empty'>Enter an assertion ID to inspect.</div>"
        try:
            backend = self.backend()
            health = _run_async(backend.get_assertion_health(matter_id, aid))
            history_resp = _run_async(backend.get_assertion_history(matter_id, aid, limit=20))
            history = history_resp.get("history", []) if isinstance(history_resp, dict) else []
            domain = self._detect_domain(matter_id)
            return _fmt_assertion_inspector(health, history=history, domain=domain)
        except Exception as exc:
            logger.warning("inspect_assertion: %s", exc)
            return f"<div class='viz-empty'>Error inspecting assertion: {_escape(str(exc))}</div>"

    def load_assertion_trace(self, matter_id: str, assertion_id: str) -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        aid = (assertion_id or "").strip()
        if not aid:
            return "<div class='viz-empty'>Enter an assertion ID to trace.</div>"
        try:
            domain = self._detect_domain(matter_id)
            data = _run_async(self.backend().get_assertion_trace(matter_id, aid))
            return _fmt_assertion_trace(data, domain=domain)
        except Exception as exc:
            logger.warning("Assertion trace failed: %s", exc)
            return f"<div class='viz-empty'>Error tracing assertion: {_escape(str(exc))}</div>"

    def load_issue_assertions(self, matter_id: str, issue_id: str) -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        iid = (issue_id or "").strip()
        if not iid:
            return "<div class='viz-empty'>Enter an issue ID to see its linked evidence.</div>"
        try:
            backend = self.backend()
            assertions = _run_async(backend.get_issue_assertions(matter_id, iid))
            authorities = []
            try:
                authorities = _run_async(backend.get_issue_authorities(matter_id, iid))
            except Exception as exc:
                logger.warning("get_issue_authorities failed: %s", exc)
            if not assertions and not authorities:
                return f"<div class='viz-empty'>No assertions or authorities linked to issue {_escape(iid[:12])}.</div>"
            domain = self._detect_domain(matter_id)
            return _fmt_issue_assertions(assertions, iid, authorities=authorities, domain=domain)
        except Exception as exc:
            logger.warning("load_issue_assertions failed: %s", exc)
            return f"<div class='viz-empty'>Error: {_escape(str(exc))}</div>"

    def load_issue_closure_workbench(self, matter_id: str, issue_id: str) -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        iid = (issue_id or "").strip()
        if not iid:
            return "<div class='viz-empty'>Select an issue to see closure status.</div>"
        try:
            data = _run_async(self.backend().get_issue_closure_workbench(matter_id, iid))
            domain = self._detect_domain(matter_id)
            return _fmt_issue_closure_workbench(data, domain=domain)
        except Exception as exc:
            logger.warning("load_issue_closure_workbench failed: %s", exc)
            return f"<div class='viz-empty'>Error: {_escape(str(exc))}</div>"

    def load_source_agreement(self, matter_id: str, issue_id: str) -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        iid = (issue_id or "").strip()
        if not iid:
            return "<div class='viz-empty'>Enter an issue ID to see source agreement.</div>"
        try:
            sources = _run_async(self.backend().get_source_agreement(matter_id, iid))
            domain = self._detect_domain(matter_id)
            return _fmt_source_agreement(sources, domain=domain)
        except Exception as exc:
            logger.warning("load_source_agreement failed: %s", exc)
            return f"<div class='viz-empty'>Error: {_escape(str(exc))}</div>"

    def load_assertion_graph(self, matter_id: str, issue_id: str) -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        iid = (issue_id or "").strip()
        if not iid:
            return "<div class='viz-empty'>Enter an issue ID to see assertion relationships.</div>"
        try:
            graph = _run_async(self.backend().get_assertion_graph(matter_id, iid))
            domain = self._detect_domain(matter_id)
            return _fmt_assertion_graph(graph, domain=domain)
        except Exception as exc:
            logger.warning("load_assertion_graph failed: %s", exc)
            return f"<div class='viz-empty'>Error: {_escape(str(exc))}</div>"

    def load_content_policy_audit(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            decisions = _run_async(self.backend().list_content_policy_decisions(matter_id, limit=50))
            return _fmt_content_policy_panel(decisions, domain=domain)
        except Exception as exc:
            logger.warning("load_content_policy_audit: %s", exc)
            return f"<div class='viz-empty'>Error loading content policy audit: {_escape(str(exc))}</div>"

    def load_review_queue(self, matter_id: str, domain: str = "legal") -> tuple[str, gr.update]:
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
        html = _fmt_review_queue(queue, domain=domain)
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
        except Exception as _exc:
            logger.warning("review queue count (pre-verify) failed: %s", _exc)
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
            except Exception as _exc:
                logger.warning("review queue count (post-verify) failed: %s", _exc)
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

    def load_review_count_badge(self, matter_id: str, domain: str = "legal") -> str:
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
        except Exception as exc:
            logger.warning("Failed to load review queue count for %s: %s", matter_id, exc)
            return ""
        total = int(counts.get("total", 0) or 0)
        bucket_counts = {
            int(k): int(v) for k, v in (counts.get("by_bucket") or {}).items()
        }
        return _fmt_review_count_badge(total, bucket_counts, domain=domain)

    def load_post_review_snapshot(
        self, matter_id: str, target_handle: str, domain: str = "legal",
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
                f"Couldn't load {_escape(label)}: {_escape(str(exc))}. "
                "Try the panel's own Refresh button."
                "</div>"
            )

        if isinstance(queue, BaseException):
            queue_html = _err_html("the review queue", queue)
            dropdown = gr.update(choices=[], value=None)
        else:
            queue_html = _fmt_review_queue(queue, domain=domain)
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
            badge_html = _fmt_review_count_badge(total, bucket_counts, domain=domain)

        _ov_domain = "legal"
        if not isinstance(overview, BaseException) and isinstance(overview, dict):
            _dc = overview.get("domain_composition", {})
            if isinstance(_dc, dict):
                _ov_domain = _dc.get("primary_domain_profile_id", "legal")

        if kind_tid is not None:
            kind, tid = kind_tid
            prov, events = results[5], results[6]
            if isinstance(prov, BaseException) or isinstance(events, BaseException):
                drawer_html = _err_html(
                    "source & history",
                    prov if isinstance(prov, BaseException) else events,
                )
            else:
                drawer_html = _fmt_source_drawer(kind, tid, prov, events, domain=_ov_domain)
        else:
            drawer_html = (
                "<div class='viz-empty'>Pick a finding and click "
                "<em>Show source &amp; history</em> to see where it "
                "came from and every review action on it.</div>"
            )

        assertions_html = (
            _err_html("assertions", assertions)
            if isinstance(assertions, BaseException)
            else _fmt_assertions(assertions, domain=_ov_domain)
        )
        issues_html = (
            _err_html("issues", issues)
            if isinstance(issues, BaseException)
            else _fmt_issues_panel(issues, domain=_ov_domain)
        )
        overview_html = (
            _err_html("the overview", overview)
            if isinstance(overview, BaseException)
            else _fmt_overview_panel(overview, domain=_ov_domain)
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
        self, matter_id: str, target_handle: str, domain: str = "legal",
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
            logger.warning("load_source_drawer: %s", exc)
            logger.debug("Provenance fetch failed: %s", exc)
            prov = []
        try:
            events = _run_async(
                self.backend().get_verification_events(matter_id, kind, tid)
            )
        except Exception as exc:
            logger.warning("load_source_drawer: %s", exc)
            logger.debug("Verification events fetch failed: %s", exc)
            events = []
        return _fmt_source_drawer(kind, tid, prov, events, domain=domain)

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

    def do_bulk_verify_by_span(
        self,
        matter_id: str,
        span_id: str,
    ) -> str:
        if not matter_id or matter_id == "—":
            return "⚠️ No matter loaded."
        sid = (span_id or "").strip()
        if not sid:
            return "⚠️ Enter a section or span reference."
        try:
            ids = _run_async(self.backend().bulk_verify_by_span(
                matter_id, sid,
                reviewed_by_kind="user", reviewed_by_id="",
            ))
        except ValueError as exc:
            return f"⚠️ {exc}"
        except Exception as exc:
            logger.warning("Bulk span verify failed for %s — %s", sid, exc)
            return "⚠️ Bulk span verify didn't go through. Please try again."
        if not ids:
            return (
                f"No candidate findings matched section <strong>{_escape(sid)}</strong>. "
                "Check the section reference as it appears in the document structure."
            )
        return f"✅ Verified {len(ids)} fact(s) from section {_escape(sid)}."

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
        _batch_domain = self._detect_domain(matter_id)
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
        _domain = self._detect_domain(matter_id)
        try:
            gaps = _run_async(self.backend().list_gaps(matter_id))
            clarifications = _run_async(self.backend().list_clarifications(matter_id))
            gap_section = _fmt_gaps(gaps, clarifications, domain=_domain)
        except Exception as exc:
            logger.warning("load_gaps: %s", exc)
            gap_section = f"⚠️ Error loading gaps: {exc}"
        actions: list = []
        try:
            run_id = getattr(self, "current_run_id", None)
            actions = _run_async(self.backend().get_steering_surface(matter_id, run_id=run_id))
            steering_section = _fmt_steering(actions, domain=_domain)
        except Exception as exc:
            logger.warning("load_gaps: %s", exc)
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

    def load_steering_panel(self, matter_id: str, domain: str = "legal") -> tuple[str, list]:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>", []
        try:
            run_id = getattr(self, "current_run_id", None)
            actions = _run_async(self.backend().get_steering_surface(matter_id, run_id=run_id))
            panel_html = _fmt_steering_panel(actions, domain)
            labels = _STEERING_ACTION_LABELS.get(domain, _STEERING_ACTION_LABELS["legal"])
            choices: list[tuple[str, str]] = []
            for idx, a in enumerate(actions):
                if not isinstance(a, dict):
                    continue
                action_type = a.get("action_type", "unknown")
                priority = a.get("priority", "low")
                action_label = labels.get(action_type, action_type.replace("_", " ").title())
                desc_short = (a.get("description") or "")[:60]
                choice_label = f"[{priority.upper()}] {action_label}: {desc_short}"
                needs_input = action_type in ("answer_clarification", "correct_assertion")
                choice_value = json.dumps({
                    "action_type": action_type,
                    "params": a.get("params") or {},
                    "needs_input": needs_input,
                })
                choices.append((choice_label, choice_value))
            return panel_html, choices
        except Exception as exc:
            logger.warning("Steering panel load failed: %s", exc)
            return f"<div class='viz-empty'>Error loading recommendations: {_escape(str(exc))}</div>", []

    def execute_steering_action(
        self, matter_id: str, action_json: str, user_input: str
    ) -> str:
        if not matter_id or matter_id == "—":
            return "Load a matter first."
        if not action_json:
            return "Select an action from the dropdown."
        try:
            action = json.loads(action_json)
        except (json.JSONDecodeError, TypeError):
            return "Invalid action data — refresh recommendations and try again."
        action_type = action.get("action_type", "")
        params = action.get("params") or {}
        run_id = getattr(self, "current_run_id", None) or ""

        if action_type == "redirect_focus":
            issue_id = params.get("issue_id", "")
            return self.do_redirect(matter_id, run_id, issue_id)

        if action_type == "answer_clarification":
            question_id = params.get("question_id", "")
            answer_text = (user_input or "").strip()
            if not answer_text:
                return "Enter your answer in the text box before executing."
            return self.do_answer_clarification(matter_id, question_id, answer_text)

        if action_type in ("force_belief_state", "correct_assertion"):
            assertion_id = params.get("assertion_id", "")
            new_state = params.get("new_state", "disputed")
            reason = (user_input or "").strip() or params.get("note", "Steering action")
            return self.do_correct_assertion(matter_id, assertion_id, new_state, reason)

        if action_type == "set_trust_override":
            doc_pattern = params.get("document_pattern", "")
            trust_level = params.get("trust_level", "low")
            note = (user_input or "").strip() or "Steering recommendation"
            return self.do_set_trust_override(matter_id, doc_pattern, trust_level, note)

        if action_type == "supply_document":
            desc = _escape(params.get("description", ""))
            return (
                f"Document needed: \"{desc}\". "
                "Upload the document in the Documents tab, then click Refresh Recommendations."
            )

        return f"Unknown action type: {_escape(action_type)}"

    def load_gaps_detail(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            gaps = _run_async(self.backend().list_gaps(matter_id))
            clarifications = _run_async(self.backend().list_clarifications(matter_id))
            issues = _run_async(self.backend().list_issues(matter_id))
            issue_titles = {
                i["id"]: i.get("title", "")
                for i in issues if isinstance(i, dict) and i.get("id")
            }
            return _fmt_gaps(gaps, clarifications, domain=domain, issue_titles=issue_titles)
        except Exception as exc:
            logger.warning("load_gaps_detail: %s", exc)
            return f"<div class='viz-empty'>Error loading gaps: {_escape(str(exc))}</div>"

    def load_gap_workbench(self, matter_id: str) -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            domain = self._detect_domain(matter_id)
            payload = _run_async(self.backend().get_gap_workbench(matter_id, limit=50))
            return _fmt_gap_workbench(payload, domain=domain)
        except Exception as exc:
            logger.warning("Gap workbench load failed: %s", exc)
            return f"<div class='viz-empty'>Error loading gap workbench: {_escape(str(exc))}</div>"

    def load_investigation_readiness(self, matter_id: str) -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            domain = self._detect_domain(matter_id)
            data = _run_async(self.backend().get_investigation_readiness(matter_id))
            return _fmt_readiness_panel(data, domain=domain)
        except Exception as exc:
            logger.warning("Readiness panel load failed: %s", exc)
            return f"<div class='viz-empty'>Error loading readiness: {_escape(str(exc))}</div>"

    def load_query_context(self, matter_id: str) -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            domain = self._detect_domain(matter_id)
            data = _run_async(self.backend().get_query_context(matter_id))
            return _fmt_query_context(data, domain=domain)
        except Exception as exc:
            logger.warning("Query context load failed: %s", exc)
            return f"<div class='viz-empty'>Error loading query context: {_escape(str(exc))}</div>"

    def load_source_calibration(self, matter_id: str) -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            domain = self._detect_domain(matter_id)
            data = _run_async(self.backend().get_source_calibration(matter_id))
            return _fmt_source_calibration(data, domain=domain)
        except Exception as exc:
            logger.warning("Source calibration load failed: %s", exc)
            return f"<div class='viz-empty'>Error loading source calibration: {_escape(str(exc))}</div>"

    def resolve_gap(self, matter_id: str, gap_id: str, resolution_note: str) -> tuple[str, str]:
        if not matter_id or matter_id == "—":
            return "Load a matter first.", ""
        gid = (gap_id or "").strip()
        if not gid:
            return "Enter a gap ID.", ""
        try:
            resolved = _run_async(self.backend().resolve_gap(matter_id, gid, resolution_note.strip()))
            if not resolved:
                return f"Gap {_escape(gid[:16])} not found or already resolved.", ""
            domain = self._detect_domain(matter_id)
            html = self.load_gaps_detail(matter_id, domain)
            return f"Resolved gap {_escape(gid[:16])}.", html
        except Exception as exc:
            logger.warning("resolve_gap failed: %s", exc)
            return f"Error: {_escape(str(exc))}", ""

    def escalate_gap(self, matter_id: str, gap_id: str) -> tuple[str, str]:
        if not matter_id or matter_id == "—":
            return "Load a matter first.", ""
        gid = (gap_id or "").strip()
        if not gid:
            return "Select a gap to escalate.", ""
        try:
            escalated = _run_async(self.backend().escalate_gap(matter_id, gid))
            if not escalated:
                return f"Gap {_escape(gid[:16])} not found or already resolved.", ""
            wb = self.load_gap_workbench(matter_id)
            return f"Escalated gap {_escape(gid[:16])} to maximum priority.", wb
        except Exception as exc:
            logger.warning("escalate_gap failed: %s", exc)
            return f"Error: {_escape(str(exc))}", ""

    def get_gap_choices(self, matter_id: str) -> list[tuple[str, str]]:
        if not matter_id or matter_id == "—":
            return []
        try:
            payload = _run_async(self.backend().get_gap_workbench(matter_id, limit=50))
            items = payload.get("items", []) if isinstance(payload, dict) else []
            choices = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                gid = item.get("gap_id", "")
                desc = str(item.get("description", ""))[:60]
                mat = _safe_float(item.get("materiality_score", 0))
                choices.append((f"[{mat:.2f}] {desc}", gid))
            return choices
        except Exception as exc:
            logger.warning("get_gap_choices failed: %s", exc)
            return []

    def load_assumptions(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            assumptions = _run_async(self.backend().list_assumptions(matter_id))
            return _fmt_assumptions(assumptions, domain=domain)
        except Exception as exc:
            logger.warning("load_assumptions: %s", exc)
            return f"<div class='viz-empty'>Error loading assumptions: {_escape(str(exc))}</div>"

    def load_assumption_review(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            data = _run_async(self.backend().get_assumption_review(matter_id))
            return _fmt_assumption_review_workbench(data, domain=domain)
        except Exception as exc:
            logger.warning("load_assumption_review: %s", exc)
            return f"<div class='viz-empty'>Error loading assumption review: {_escape(str(exc))}</div>"

    def update_assumption_status(self, matter_id: str, assumption_id: str, action: str, reason: str) -> tuple[str, str]:
        if not matter_id or matter_id == "—":
            return "Load a matter first.", ""
        aid = (assumption_id or "").strip()
        if not aid:
            return "Enter an assumption ID.", ""
        if not action:
            return "Select an action.", ""
        try:
            result = _run_async(self.backend().review_assumption(
                matter_id, aid, action, reason.strip()
            ))
            if not isinstance(result, dict):
                return "Unexpected response.", ""
            if result.get("error"):
                return f"Error: {_escape(str(result['error']))}", ""
            actions = result.get("actions", [])
            msg = " · ".join(str(a) for a in actions if isinstance(a, str))
            domain = self._detect_domain(matter_id)
            html = self.load_assumptions(matter_id, domain)
            return f"Reviewed: {_escape(msg)}", html
        except ValueError as ve:
            return f"Invalid: {_escape(str(ve))}", ""
        except Exception as exc:
            logger.warning("update_assumption_status failed: %s", exc)
            return f"Error: {_escape(str(exc))}", ""

    def set_issue_priority(self, matter_id: str, issue_id: str, priority: str) -> str:
        if not matter_id or matter_id == "—":
            return "Load a matter first."
        iid = (issue_id or "").strip()
        if not iid:
            return "Enter an issue ID."
        if not priority:
            return "Select a priority level."
        try:
            updated = _run_async(self.backend().set_issue_priority(matter_id, iid, priority))
            if not updated:
                return f"Issue {_escape(iid[:20])} not found."
            label = {"critical": "Critical", "high": "High", "medium": "Medium", "low": "Low"}.get(priority, priority)
            return f"Set {_escape(iid[:20])} to {label} priority."
        except Exception as exc:
            logger.warning("set_issue_priority failed: %s", exc)
            return f"Error: {_escape(str(exc))}"

    def load_quant(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            quant_data = _run_async(self.backend().get_quant_summary(matter_id))
            return _fmt_quant_panel(
                quant_data.get("payment_reconciliation", {}),
                quant_data.get("invoice_reconciliation", []),
                quant_data.get("amount_conflicts", []),
                quant_data.get("damages_waterfall", []),
                domain=domain,
            )
        except Exception as exc:
            logger.warning("load_quant: %s", exc)
            return f"<div class='viz-empty'>Error loading quantitative data: {_escape(str(exc))}</div>"

    def do_detect_quant_conflicts(self, matter_id: str) -> tuple[str, str]:
        if not matter_id or matter_id == "—":
            return "Load a matter first.", ""
        try:
            gap_ids = _run_async(self.backend().detect_quant_conflicts(matter_id))
            if gap_ids:
                msg = f"Detected {len(gap_ids)} new conflict(s). Gaps recorded, assertions marked DISPUTED."
            else:
                msg = "No new conflicts detected. All amounts are consistent."
            refreshed = self.load_quant(matter_id, domain=self._detect_domain(matter_id))
            return msg, refreshed
        except Exception as exc:
            logger.warning("do_detect_quant_conflicts: %s", exc)
            return f"Error: {_escape(str(exc))}", ""

    def load_quant_facts(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            data = _run_async(self.backend().get_quant_facts(matter_id))
            if not isinstance(data, dict):
                logger.warning("load_quant_facts: expected dict, got %s", type(data).__name__)
                data = {}
            return _fmt_quant_facts(data, domain=domain)
        except Exception as exc:
            logger.warning("load_quant_facts failed: %s", exc)
            return f"<div class='viz-empty'>Error: {_escape(str(exc))}</div>"

    def load_decision_leverage(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            data = _run_async(self.backend().get_decision_leverage(matter_id))
            if not isinstance(data, dict):
                logger.warning("load_decision_leverage: expected dict, got %s", type(data).__name__)
                data = {}
            return _fmt_decision_leverage(data, domain=domain)
        except Exception as exc:
            logger.warning("load_decision_leverage failed: %s", exc)
            return f"<div class='viz-empty'>Error: {_escape(str(exc))}</div>"

    def load_output_quality(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            run_id = getattr(self, "current_run_id", None)
            data = _run_async(self.backend().get_output_quality(matter_id, run_id=run_id))
            if not isinstance(data, dict):
                logger.warning("load_output_quality: expected dict, got %s", type(data).__name__)
                data = {}
            return _fmt_output_quality(data, domain=domain)
        except Exception as exc:
            logger.warning("Output quality workbench load failed: %s", exc)
            return f"<div class='viz-empty'>Error: {_escape(str(exc))}</div>"

    def load_deliverable_workbench(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            data = _run_async(self.backend().get_deliverable_workbench(matter_id))
            if not isinstance(data, dict):
                logger.warning("load_deliverable_workbench: expected dict, got %s", type(data).__name__)
                data = {}
            return _fmt_deliverable_workbench(data, domain=domain)
        except Exception as exc:
            logger.warning("Deliverable workbench load failed: %s", exc)
            return f"<div class='viz-empty'>Error: {_escape(str(exc))}</div>"

    def load_scenario_workbench(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            data = _run_async(self.backend().get_scenario_workbench(matter_id))
            if not isinstance(data, dict):
                logger.warning("load_scenario_workbench: expected dict, got %s", type(data).__name__)
                data = {}
            return _fmt_scenario_workbench(data, domain=domain)
        except Exception as exc:
            logger.warning("Scenario workbench load failed: %s", exc)
            return f"<div class='viz-empty'>Error: {_escape(str(exc))}</div>"

    def load_scenario_comparison(self, matter_id: str, branch_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        bid = (branch_id or "").strip()
        if not bid:
            return "<div class='viz-empty'>Enter a branch ID to compare.</div>"
        try:
            data = _run_async(self.backend().compare_scenario_to_baseline(matter_id, bid))
            return _fmt_scenario_comparison(data, domain=domain)
        except Exception as exc:
            logger.warning("load_scenario_comparison: %s", exc)
            return f"<div class='viz-empty'>Error comparing scenario: {_escape(str(exc))}</div>"

    def load_scenario_snapshot_history(
        self, matter_id: str, branch_id: str, domain: str = "legal",
    ) -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        bid = (branch_id or "").strip()
        if not bid:
            return "<div class='viz-empty'>Enter a branch ID to view snapshot history.</div>"
        try:
            data = _run_async(self.backend().list_scenario_snapshots(matter_id, bid))
            return _fmt_scenario_snapshot_history(data, domain=domain)
        except Exception as exc:
            logger.warning("load_scenario_snapshot_history: %s", exc)
            return f"<div class='viz-empty'>Error loading snapshots: {_escape(str(exc))}</div>"

    def load_scenario_deltas(
        self, matter_id: str, branch_id: str, domain: str = "legal",
    ) -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        bid = (branch_id or "").strip()
        if not bid:
            return "<div class='viz-empty'>Enter a branch ID to view deltas.</div>"
        try:
            deltas = _run_async(self.backend().list_scenario_deltas(matter_id, bid))
            if not isinstance(deltas, list):
                logger.warning("load_scenario_deltas: expected list, got %s", type(deltas).__name__)
                deltas = []
            return _fmt_scenario_deltas(deltas, domain=domain)
        except Exception as exc:
            logger.warning("load_scenario_deltas: %s", exc)
            return f"<div class='viz-empty'>Error loading deltas: {_escape(str(exc))}</div>"

    def apply_scenario_delta_ui(
        self, matter_id: str, branch_id: str,
        target_kind: str, target_id: str, operation: str,
        payload_json: str = "",
    ) -> tuple[str, str]:
        if not matter_id or matter_id == "—":
            return "No matter loaded.", ""
        bid = (branch_id or "").strip()
        if not bid:
            return "Enter a branch ID.", ""
        tkind = (target_kind or "").strip()
        tid = (target_id or "").strip()
        op = (operation or "").strip()
        if not tkind or not tid or not op:
            return "All fields (target kind, target ID, operation) are required.", ""
        payload: dict = {}
        if payload_json and payload_json.strip():
            try:
                payload = json.loads(payload_json.strip())
                if not isinstance(payload, dict):
                    return "Payload must be a JSON object.", ""
            except (json.JSONDecodeError, TypeError):
                return "Payload must be valid JSON.", ""
        try:
            result = _run_async(
                self.backend().apply_scenario_delta(matter_id, bid, tkind, tid, op, payload)
            )
            if not isinstance(result, dict):
                logger.warning("apply_scenario_delta_ui: expected dict, got %s", type(result).__name__)
                return "Unexpected response.", ""
            if result.get("error"):
                return f"Error: {_escape(str(result['error']))}", ""
            total = result.get("total_deltas", 0)
            domain = self._detect_domain(matter_id)
            refreshed = self.load_scenario_deltas(matter_id, bid, domain=domain)
            return f"Delta applied ({op} on {tkind}). {total} total deltas.", refreshed
        except Exception as exc:
            logger.warning("apply_scenario_delta_ui: %s", exc)
            return f"Error: {_escape(str(exc))}", ""

    def compute_scenario_snapshot_ui(
        self, matter_id: str, branch_id: str,
    ) -> tuple[str, str]:
        if not matter_id or matter_id == "—":
            return "No matter loaded.", ""
        bid = (branch_id or "").strip()
        if not bid:
            return "Enter a branch ID.", ""
        try:
            result = _run_async(
                self.backend().compute_scenario_snapshot(matter_id, bid)
            )
            if not isinstance(result, dict):
                logger.warning("compute_scenario_snapshot_ui: expected dict, got %s", type(result).__name__)
                return "Unexpected response.", ""
            if result.get("error"):
                return f"Error: {_escape(str(result['error']))}", ""
            dc = result.get("delta_count", 0)
            warns = result.get("warnings", [])
            warn_text = f" Warnings: {', '.join(_escape(str(w)) for w in warns)}" if warns else ""
            domain = self._detect_domain(matter_id)
            refreshed = self.load_scenario_snapshot_history(matter_id, bid, domain=domain)
            return f"Snapshot computed ({dc} deltas evaluated).{warn_text}", refreshed
        except Exception as exc:
            logger.warning("compute_scenario_snapshot_ui: %s", exc)
            return f"Error: {_escape(str(exc))}", ""

    def archive_scenario_branch_ui(
        self, matter_id: str, branch_id: str,
    ) -> tuple[str, str]:
        if not matter_id or matter_id == "—":
            return "No matter loaded.", ""
        bid = (branch_id or "").strip()
        if not bid:
            return "Enter a branch ID.", ""
        try:
            result = _run_async(
                self.backend().archive_scenario_branch(matter_id, bid)
            )
            if isinstance(result, bool):
                status = "Branch archived." if result else "Branch not found or already archived."
            elif isinstance(result, dict):
                if result.get("error"):
                    status = f"Error: {_escape(str(result['error']))}"
                else:
                    status = "Archived."
            else:
                logger.warning("archive_scenario_branch_ui: unexpected type %s", type(result).__name__)
                status = "Unexpected response type."
            domain = self._detect_domain(matter_id)
            refreshed = self.load_scenario_workbench(matter_id, domain=domain)
            return status, refreshed
        except Exception as exc:
            logger.warning("archive_scenario_branch_ui: %s", exc)
            return f"Error: {_escape(str(exc))}", ""

    def load_alternative_theories(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            data = _run_async(self.backend().get_alternative_theory_portfolio(matter_id))
            if not isinstance(data, dict):
                logger.warning("load_alternative_theories: expected dict, got %s", type(data).__name__)
                data = {}
            return _fmt_alternative_theories(data, domain=domain)
        except Exception as exc:
            logger.warning("Alternative theories load failed: %s", exc)
            return f"<div class='viz-empty'>Error: {_escape(str(exc))}</div>"

    def load_manifest_inspector(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            data = _run_async(self.backend().get_dependency_manifest_inspector(matter_id))
            if not isinstance(data, dict):
                logger.warning("load_manifest_inspector: expected dict, got %s", type(data).__name__)
                data = {}
            return _fmt_manifest_inspector(data, domain=domain)
        except Exception as exc:
            logger.warning("Manifest inspector load failed: %s", exc)
            return f"<div class='viz-empty'>Error: {_escape(str(exc))}</div>"

    def load_freshness_report(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            data = _run_async(self.backend().get_freshness_report(matter_id))
            if not isinstance(data, dict):
                logger.warning("load_freshness_report: expected dict, got %s", type(data).__name__)
                data = {}
            return _fmt_freshness_report(data, domain=domain)
        except Exception as exc:
            logger.warning("Freshness report load failed: %s", exc)
            return f"<div class='viz-empty'>Error: {_escape(str(exc))}</div>"

    def load_cache_stats(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            data = _run_async(self.backend().get_cache_stats(matter_id))
            if not isinstance(data, dict):
                logger.warning("load_cache_stats: expected dict, got %s", type(data).__name__)
                data = {}
            return _fmt_cache_stats(data, domain=domain)
        except Exception as exc:
            logger.warning("Cache stats load failed: %s", exc)
            return f"<div class='viz-empty'>Error: {_escape(str(exc))}</div>"

    def load_llm_usage(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            data = _run_async(self.backend().get_llm_usage(matter_id))
            if not isinstance(data, dict):
                logger.warning("load_llm_usage: expected dict, got %s", type(data).__name__)
                data = {}
            return _fmt_llm_usage(data, domain=domain)
        except Exception as exc:
            logger.warning("LLM usage load failed: %s", exc)
            return f"<div class='viz-empty'>Error: {_escape(str(exc))}</div>"

    def load_impact_preview(
        self, matter_id: str, action_type: str, payload_json: str, domain: str = "legal",
    ) -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        if not action_type:
            return _fmt_impact_preview({}, domain=domain)
        try:
            import json as _json
            payload = _json.loads(payload_json) if payload_json else {}
        except (ValueError, TypeError):
            return (
                "<div class='viz-empty' style='color:#dc2626;'>"
                "Invalid payload JSON. Enter a valid JSON object.</div>"
            )
        try:
            data = _run_async(
                self.backend().get_steering_impact_preview(matter_id, action_type, payload)
            )
            if not isinstance(data, dict):
                logger.warning("load_impact_preview: expected dict, got %s", type(data).__name__)
                data = {}
            return _fmt_impact_preview(data, domain=domain)
        except Exception as exc:
            logger.warning("Impact preview load failed: %s", exc)
            return f"<div class='viz-empty'>Error: {_escape(str(exc))}</div>"

    def load_domain_readiness(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            data = _run_async(self.backend().get_domain_investigation_readiness(matter_id))
            if not isinstance(data, dict):
                logger.warning("load_domain_readiness: expected dict, got %s", type(data).__name__)
                data = {}
            return _fmt_domain_readiness(data, domain=domain)
        except Exception as exc:
            logger.warning("Domain readiness load failed: %s", exc)
            return f"<div class='viz-empty'>Error: {_escape(str(exc))}</div>"

    def load_issue_brief(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            data = _run_async(self.backend().compile_issue_brief(matter_id))
            if not isinstance(data, dict):
                logger.warning("load_issue_brief: expected dict, got %s", type(data).__name__)
                data = {}
            return _fmt_issue_brief(data, domain=domain)
        except Exception as exc:
            logger.warning("Issue brief load failed: %s", exc)
            return f"<div class='viz-empty'>Error: {_escape(str(exc))}</div>"

    def create_scenario_branch_ui(self, matter_id: str, name: str, assumptions_text: str, notes: str = "") -> str:
        if not matter_id or matter_id == "—":
            return "No matter loaded."
        if not name or not name.strip():
            return "Branch name is required."
        assumptions = [{"text": a.strip()} for a in assumptions_text.split("\n") if a.strip()]
        try:
            result = _run_async(self.backend().create_scenario_branch(
                matter_id,
                {"name": name.strip(), "assumptions": assumptions, "notes": notes.strip()},
            ))
            if isinstance(result, dict) and result.get("branch_id"):
                return f"Created branch: {_escape(result.get('name', name))}"
            return f"Error: {_escape(str(result))}"
        except Exception as exc:
            logger.warning("create_scenario_branch_ui failed: %s", exc)
            return f"Error: {_escape(str(exc))}"

    def load_quant_ontology(self, matter_id: str, domain: str = "legal") -> tuple[str, Any]:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>", gr.update(choices=[])
        try:
            data = _run_async(self.backend().get_quant_ontology(matter_id))
            html = _fmt_quant_ontology(data if isinstance(data, dict) else {}, domain=domain)
            raw_choices: list[str] = []
            for g in (data.get("metric_groups", []) if isinstance(data, dict) else []):
                if not isinstance(g, dict):
                    continue
                mt = g.get("metric_type", "")
                if mt and not g.get("approved"):
                    raw_choices.append(str(mt))
            return html, gr.update(choices=raw_choices)
        except Exception as exc:
            logger.warning("load_quant_ontology failed: %s", exc)
            return f"<div class='viz-empty'>Error: {_escape(str(exc))}</div>", gr.update(choices=[])

    def approve_quant_alias(
        self, matter_id: str, raw_label: str, canonical_metric: str, unit: str = "",
    ) -> tuple[str, str, Any]:
        if not matter_id or matter_id == "—":
            return "Load a matter first.", "", gr.update(choices=[])
        if not raw_label or not canonical_metric:
            return "Select a raw metric and enter a canonical metric name.", "", gr.update(choices=[])
        try:
            _run_async(self.backend().approve_metric_alias(
                matter_id, raw_label.strip(), canonical_metric.strip(),
                unit=unit.strip() or None,
            ))
            domain = self._detect_domain(matter_id)
            refreshed_html, dropdown_update = self.load_quant_ontology(matter_id, domain=domain)
            return (
                f"Approved: **{_escape(raw_label)}** → **{_escape(canonical_metric)}**",
                refreshed_html,
                dropdown_update,
            )
        except Exception as exc:
            logger.warning("approve_quant_alias failed: %s", exc)
            return f"Error: {_escape(str(exc))}", "", gr.update(choices=[])

    def load_timeline(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            events = _run_async(self.backend().get_timeline(
                matter_id, limit=200, policy_audience=self.policy_audience,
            ))
            return _fmt_timeline_panel(events, domain=domain)
        except Exception as exc:
            logger.warning("load_timeline: %s", exc)
            return f"<div class='viz-empty'>Error loading timeline: {_escape(str(exc))}</div>"

    def load_answer_audits(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            data = _run_async(self.backend().get_answer_audits(matter_id))
            return _fmt_answer_audit(data if isinstance(data, dict) else {}, domain=domain)
        except Exception as exc:
            logger.warning("load_answer_audits failed: %s", exc)
            return f"<div class='viz-empty'>Error: {_escape(str(exc))}</div>"

    def load_evidence_matrix(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            matrix = _run_async(self.backend().get_evidence_matrix(
                matter_id, policy_audience=self.policy_audience,
            ))
            return _fmt_evidence_matrix_panel(matrix, domain=domain)
        except Exception as exc:
            logger.warning("load_evidence_matrix: %s", exc)
            return f"<div class='viz-empty'>Error loading evidence matrix: {_escape(str(exc))}</div>"

    def _detect_domain(self, matter_id: str) -> str:
        if not matter_id or matter_id == "—":
            return "legal"
        try:
            ov = _run_async(self.backend().get_overview(matter_id))
            dc = ov.get("domain_composition", {}) if isinstance(ov, dict) else {}
            return dc.get("primary_domain_profile_id", "legal") if isinstance(dc, dict) else "legal"
        except Exception as exc:
            logger.warning("Failed to detect domain for %s: %s", matter_id, exc)
            return "legal"

    def load_proof_state(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            data = _run_async(self.backend().get_proof_state_summary(matter_id))
            summary = data.get("summary", {}) if isinstance(data, dict) else {}
            issues = data.get("issues", []) if isinstance(data, dict) else []
            return _fmt_proof_state_panel(summary, issues, domain)
        except Exception as exc:
            logger.warning("load_proof_state: %s", exc)
            return f"<div class='viz-empty'>Error loading proof state: {_escape(str(exc))}</div>"

    def recompute_proof_state(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            _run_async(self.backend().compute_proof_state(matter_id))
            return self.load_proof_state(matter_id, domain)
        except Exception as exc:
            logger.warning("recompute_proof_state: %s", exc)
            return f"<div class='viz-empty'>Error recomputing proof state: {_escape(str(exc))}</div>"

    def recompute_issue_proof_state(self, matter_id: str, issue_id: str) -> tuple[str, str]:
        if not matter_id or matter_id == "—":
            return "Load a matter first.", ""
        iid = (issue_id or "").strip()
        if not iid:
            return "Enter an issue ID.", ""
        try:
            result = _run_async(self.backend().compute_issue_proof_state(matter_id, iid))
            status = result.get("proof_status", "unknown") if isinstance(result, dict) else "done"
            suf = result.get("sufficiency", "?") if isinstance(result, dict) else "?"
            domain = self._detect_domain(matter_id)
            html = self.load_proof_state(matter_id, domain)
            return f"Recomputed issue {_escape(iid[:12])}: {_escape(status)} ({_escape(str(suf))})", html
        except Exception as exc:
            logger.warning("recompute_issue_proof_state failed: %s", exc)
            return f"Error: {_escape(str(exc))}", ""

    def load_authority_network(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            data = _run_async(self.backend().get_authority_network(matter_id))
            return _fmt_authority_panel(data, domain)
        except Exception as exc:
            logger.warning("load_authority_network: %s", exc)
            return f"<div class='viz-empty'>Error loading authorities: {_escape(str(exc))}</div>"

    def search_authorities(self, matter_id: str, query: str) -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        q = (query or "").strip()
        if len(q) < 2:
            return self.load_authority_network(matter_id, domain=self._detect_domain(matter_id))
        try:
            results = _run_async(self.backend().search_authorities(matter_id, q, limit=20))
            domain = self._detect_domain(matter_id)
            if not results:
                return f"<div class='viz-empty'>No authorities matching “{_escape(q)}”.</div>"
            wrapped = {"authorities": results, "issue_links": {}}
            return _fmt_authority_panel(wrapped, domain)
        except Exception as exc:
            logger.warning("search_authorities failed: %s", exc)
            return f"<div class='viz-empty'>Error searching: {_escape(str(exc))}</div>"

    def do_upsert_authority(
        self, matter_id: str, citation: str, authority_type: str,
        name: str, jurisdiction: str, weight: str,
    ) -> tuple[str, str]:
        if not matter_id or matter_id == "—":
            return "No matter loaded.", ""
        citation = (citation or "").strip()
        if not citation:
            return "Citation is required.", ""
        try:
            result = _run_async(self.backend().upsert_authority(
                matter_id, citation,
                authority_type=authority_type or "case",
                name=name or None,
                jurisdiction=jurisdiction or None,
                weight=weight or "persuasive",
            ))
            is_new = result.get("is_new", True)
            aid = result.get("authority_id", "?")
            verb = "Added" if is_new else "Updated"
            domain = self._detect_domain(matter_id)
            html = self.load_authority_network(matter_id, domain)
            return f"{verb} authority {_escape(aid[:12])}", html
        except Exception as exc:
            logger.warning("upsert_authority failed for %s: %s", matter_id, exc)
            return f"Error: {_escape(str(exc))}", ""

    def do_link_authority_issue(
        self, matter_id: str, authority_id: str, issue_id: str, relevance: str,
    ) -> tuple[str, str]:
        if not matter_id or matter_id == "—":
            return "No matter loaded.", ""
        authority_id = (authority_id or "").strip()
        issue_id = (issue_id or "").strip()
        if not authority_id or not issue_id:
            return "Both authority ID and issue ID are required.", ""
        try:
            _run_async(self.backend().link_authority_to_issue(
                matter_id, authority_id, issue_id,
                relevance=relevance or "supporting",
            ))
            domain = self._detect_domain(matter_id)
            html = self.load_authority_network(matter_id, domain)
            return f"Linked {_escape(authority_id[:12])} → {_escape(issue_id[:12])} ({_escape(relevance or '')})", html
        except Exception as exc:
            logger.warning("link_authority_to_issue failed: %s", exc)
            return f"Error: {_escape(str(exc))}", ""

    def do_unlink_authority_issue(
        self, matter_id: str, authority_id: str, issue_id: str,
    ) -> tuple[str, str]:
        if not matter_id or matter_id == "—":
            return "No matter loaded.", ""
        authority_id = (authority_id or "").strip()
        issue_id = (issue_id or "").strip()
        if not authority_id or not issue_id:
            return "Both authority ID and issue ID are required.", ""
        try:
            _run_async(self.backend().unlink_authority_from_issue(
                matter_id, authority_id, issue_id,
            ))
            domain = self._detect_domain(matter_id)
            html = self.load_authority_network(matter_id, domain)
            return f"Unlinked {_escape(authority_id[:12])} from {_escape(issue_id[:12])}", html
        except Exception as exc:
            logger.warning("unlink_authority_from_issue failed: %s", exc)
            return f"Error: {_escape(str(exc))}", ""

    def load_document_intelligence(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            data = _run_async(self.backend().get_document_intelligence(matter_id))
            overrides = _run_async(self.backend().list_trust_overrides(matter_id))
            return _fmt_document_intelligence_panel(data, domain, trust_overrides=overrides)
        except Exception as exc:
            logger.warning("load_document_intelligence: %s", exc)
            return f"<div class='viz-empty'>Error loading document intelligence: {_escape(str(exc))}</div>"

    def load_belief_revisions(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            data = _run_async(self.backend().list_belief_revisions(matter_id))
            return _fmt_belief_revision_panel(data, domain)
        except Exception as exc:
            logger.warning("load_belief_revisions: %s", exc)
            return f"<div class='viz-empty'>Error loading belief revisions: {_escape(str(exc))}</div>"

    def load_contradictions(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            data = _run_async(self.backend().get_contradictions(matter_id))
            return _fmt_contradiction_panel(data, domain)
        except Exception as exc:
            logger.warning("load_contradictions: %s", exc)
            return f"<div class='viz-empty'>Error loading contradictions: {_escape(str(exc))}</div>"

    def mine_and_load_contradictions(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            _run_async(self.backend().mine_contradictions(matter_id))
            data = _run_async(self.backend().get_contradictions(matter_id))
            return _fmt_contradiction_panel(data, domain)
        except Exception as exc:
            logger.warning("mine_and_load_contradictions: %s", exc)
            return f"<div class='viz-empty'>Error mining contradictions: {_escape(str(exc))}</div>"

    def resolve_contradiction(
        self, matter_id: str, attacker_id: str, attacked_id: str,
        decision: str, rationale: str,
    ) -> tuple[str, str]:
        if not matter_id or matter_id == "—":
            return "Load a matter first.", ""
        if not attacker_id or not attacked_id:
            return "Enter both assertion IDs from a contradiction pair.", ""
        if not decision:
            return "Select a resolution decision.", ""
        try:
            result = _run_async(self.backend().resolve_contradiction(
                matter_id, attacker_id.strip(), attacked_id.strip(),
                decision, rationale.strip(),
            ))
            if not isinstance(result, dict):
                return "Unexpected response.", ""
            if result.get("error"):
                return f"Error: {_escape(str(result['error']))}", ""
            actions = result.get("actions", [])
            msg = " · ".join(str(a) for a in actions if isinstance(a, str))
            domain = self._detect_domain(matter_id)
            refreshed = self.load_contradictions(matter_id, domain=domain)
            return f"Resolved: {_escape(msg)}", refreshed
        except Exception as exc:
            logger.warning("resolve_contradiction failed: %s", exc)
            return f"Error: {_escape(str(exc))}", ""

    def load_objective_coverage(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            data = _run_async(self.backend().get_objective_coverage(matter_id))
            if not isinstance(data, dict):
                logger.warning("load_objective_coverage: expected dict, got %s", type(data).__name__)
                data = {}
            return _fmt_objective_coverage(data, domain=domain)
        except Exception as exc:
            logger.warning("load_objective_coverage failed: %s", exc)
            return f"<div class='viz-empty'>Error: {_escape(str(exc))}</div>"

    def set_criterion_status(
        self, matter_id: str, predicate_id: str, status: str, reason: str,
    ) -> tuple[str, str]:
        if not matter_id or matter_id == "—":
            return "Load a matter first.", ""
        pid = (predicate_id or "").strip()
        if not pid:
            return "Enter a criterion ID.", ""
        if not status:
            return "Select a status.", ""
        try:
            result = _run_async(self.backend().set_criterion_status(
                matter_id, pid, status, (reason or "").strip(),
            ))
            if not isinstance(result, dict):
                return "Unexpected response.", ""
            if result.get("error"):
                return f"Error: {_escape(str(result['error']))}", ""
            domain = self._detect_domain(matter_id)
            html = self.load_objective_coverage(matter_id, domain)
            return f"Criterion {_escape(pid[:16])} → {_escape(status)}", html
        except Exception as exc:
            logger.warning("set_criterion_status failed: %s", exc)
            return f"Error: {_escape(str(exc))}", ""

    def add_criterion(
        self, matter_id: str, objective_id: str, description: str, burden_side: str,
    ) -> tuple[str, str]:
        if not matter_id or matter_id == "—":
            return "Load a matter first.", ""
        oid = (objective_id or "").strip()
        desc = (description or "").strip()
        if not oid:
            return "Enter an objective ID.", ""
        if not desc:
            return "Enter a description.", ""
        try:
            result = _run_async(self.backend().add_criterion(
                matter_id, oid, desc, (burden_side or "").strip(),
            ))
            if not isinstance(result, dict):
                return "Unexpected response.", ""
            if result.get("error"):
                return f"Error: {_escape(str(result['error']))}", ""
            pid = result.get("predicate_id", "")
            domain = self._detect_domain(matter_id)
            html = self.load_objective_coverage(matter_id, domain)
            return f"Added criterion {_escape(str(pid)[:16])} to {_escape(oid[:16])}", html
        except Exception as exc:
            logger.warning("add_criterion failed: %s", exc)
            return f"Error: {_escape(str(exc))}", ""

    def load_knowledge_seeds(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            data = _run_async(self.backend().get_knowledge_seeds(matter_id))
            return _fmt_knowledge_seeds(data if isinstance(data, dict) else {}, domain=domain)
        except Exception as exc:
            logger.warning("load_knowledge_seeds failed: %s", exc)
            return f"<div class='viz-empty'>Error: {_escape(str(exc))}</div>"

    def review_knowledge_seed(
        self, matter_id: str, seed_id: str, decision: str, review_note: str,
    ) -> tuple[str, str]:
        if not matter_id or matter_id == "—":
            return "Load a matter first.", ""
        if not seed_id:
            return "Enter a seed ID.", ""
        if not decision:
            return "Select a review decision.", ""
        try:
            result = _run_async(self.backend().review_knowledge_seed(
                matter_id, seed_id.strip(), decision, review_note.strip(),
            ))
            if not isinstance(result, dict):
                return "Unexpected response.", ""
            if result.get("error"):
                return f"Error: {_escape(str(result['error']))}", ""
            domain = self._detect_domain(matter_id)
            refreshed = self.load_knowledge_seeds(matter_id, domain=domain)
            return f"Reviewed: {_escape(result.get('seed_kind', ''))} → {_escape(decision)}", refreshed
        except Exception as exc:
            logger.warning("review_knowledge_seed failed: %s", exc)
            return f"Error: {_escape(str(exc))}", ""

    def promote_knowledge_seed_ui(
        self, matter_id: str, seed_kind: str,
        domain_profile_id: str, payload_json: str,
    ) -> tuple[str, str]:
        if not matter_id or matter_id == "—":
            return "Load a matter first.", ""
        if not seed_kind:
            return "Select a seed kind.", ""
        if not domain_profile_id or not domain_profile_id.strip():
            return "Enter a domain profile ID.", ""
        payload_str = (payload_json or "{}").strip()
        try:
            json.loads(payload_str)
        except (json.JSONDecodeError, TypeError):
            return "Payload must be valid JSON.", ""
        try:
            result = _run_async(self.backend().promote_knowledge_seed(
                matter_id, seed_kind, domain_profile_id.strip(),
                payload_str, source_matter_id=matter_id,
            ))
            if not isinstance(result, dict):
                logger.warning("promote_knowledge_seed_ui: expected dict, got %s", type(result).__name__)
                return "Unexpected response.", ""
            if result.get("error"):
                return f"Error: {_escape(str(result['error']))}", ""
            domain = self._detect_domain(matter_id)
            refreshed = self.load_knowledge_seeds(matter_id, domain=domain)
            return f"Seed promoted: {_escape(seed_kind)} (ID: {_escape(str(result.get('seed_id', ''))[:12])})", refreshed
        except Exception as exc:
            logger.warning("promote_knowledge_seed_ui: %s", exc)
            return f"Error: {_escape(str(exc))}", ""

    def load_document_versions(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            data = _run_async(self.backend().get_document_versions(matter_id))
            return _fmt_document_versions_panel(data, domain)
        except Exception as exc:
            logger.warning("load_document_versions: %s", exc)
            return f"<div class='viz-empty'>Error loading document versions: {_escape(str(exc))}</div>"

    def detect_and_load_document_versions(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            _run_async(self.backend().refresh_document_families(matter_id))
            data = _run_async(self.backend().get_document_versions(matter_id))
            return _fmt_document_versions_panel(data, domain)
        except Exception as exc:
            logger.warning("detect_and_load_document_versions: %s", exc)
            return f"<div class='viz-empty'>Error detecting version chains: {_escape(str(exc))}</div>"

    def lookup_operative_version(
        self, matter_id: str, doc_id: str, domain: str = "legal",
    ) -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        if not doc_id or not doc_id.strip():
            return "<div class='viz-empty'>Please enter a document ID.</div>"
        try:
            data = _run_async(self.backend().get_operative_document_version(
                matter_id, doc_id.strip(),
            ))
            return _fmt_operative_version_result(data, domain)
        except Exception as exc:
            logger.warning("lookup_operative_version: %s", exc)
            return f"<div class='viz-empty'>Error looking up operative version: {_escape(str(exc))}</div>"

    def load_quant_thresholds(
        self, matter_id: str, domain: str = "legal",
        exposure_high: float = 10_000.0, disputed_fraction_min: float = 0.10,
    ) -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            data = _run_async(self.backend().get_quant_thresholds(
                matter_id, exposure_high=exposure_high,
                disputed_fraction_min=disputed_fraction_min,
            ))
            return _fmt_quant_thresholds_panel(data, domain)
        except Exception as exc:
            logger.warning("load_quant_thresholds: %s", exc)
            return f"<div class='viz-empty'>Error loading quant thresholds: {_escape(str(exc))}</div>"

    def load_system_health(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            data = _run_async(self.backend().get_system_health(matter_id))
            return _fmt_system_health_panel(data, domain)
        except Exception as exc:
            logger.warning("load_system_health: %s", exc)
            return f"<div class='viz-empty'>Error loading system health: {_escape(str(exc))}</div>"

    def flush_pending_propagation(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            result = _run_async(self.backend().flush_pending(matter_id))
            revised = result.get("revised_count", 0) if isinstance(result, dict) else 0
            health_html = self.load_system_health(matter_id, domain)
            status = f"<div class='status-badge' style='background:#059669;color:white;padding:4px 10px;border-radius:4px;margin-bottom:8px;display:inline-block;'>Flush complete — {revised} assertion(s) revised</div>"
            return status + health_html
        except Exception as exc:
            logger.warning("flush_pending_propagation: %s", exc)
            return f"<div class='viz-empty'>Error flushing pending propagation: {_escape(str(exc))}</div>"

    def load_so_scorecard(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            data = _run_async(self.backend().get_so_scorecard(matter_id))
            return _fmt_so_scorecard_panel(data, domain)
        except Exception as exc:
            logger.warning("load_so_scorecard: %s", exc)
            return f"<div class='viz-empty'>Error loading SO scorecard: {_escape(str(exc))}</div>"

    def load_domain_profile(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            data = _run_async(self.backend().get_domain_profile_summary(matter_id))
            return _fmt_domain_profile_panel(data, domain)
        except Exception as exc:
            logger.warning("load_domain_profile: %s", exc)
            return f"<div class='viz-empty'>Error loading domain profile: {_escape(str(exc))}</div>"

    def load_domain_composition(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            data = _run_async(self.backend().get_domain_composition(matter_id))
            return _fmt_domain_composition_panel(data, domain)
        except Exception as exc:
            logger.warning("load_domain_composition: %s", exc)
            return f"<div class='viz-empty'>Error loading domain composition: {_escape(str(exc))}</div>"

    def load_document_console(self, matter_id: str, document_ref: str) -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        ref = (document_ref or "").strip()
        if not ref:
            return "<div class='viz-empty'>Select a document to review.</div>"
        try:
            domain = self._detect_domain(matter_id)
            data = _run_async(self.backend().get_document_console(matter_id, ref))
            return _fmt_document_console(data, domain=domain)
        except Exception as exc:
            logger.warning("Document console load failed: %s", exc)
            return f"<div class='viz-empty'>Error loading document console: {_escape(str(exc))}</div>"

    def get_reviewable_doc_choices(self, matter_id: str) -> list[tuple[str, str]]:
        if not matter_id or matter_id == "—":
            return []
        try:
            docs = _run_async(self.backend().list_reviewable_documents(matter_id))
            choices = []
            for d in docs:
                if not isinstance(d, dict):
                    continue
                path = str(d.get("path", ""))
                pending = int(d.get("pending", 0))
                verified = int(d.get("verified", 0))
                choices.append((f"{path} ({pending} pending, {verified} verified)", path))
            return choices
        except Exception as exc:
            logger.warning("get_reviewable_doc_choices failed: %s", exc)
            return []

    def load_document_triage(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            data = _run_async(self.backend().list_documents_needing_profile(matter_id))
            return _fmt_doc_triage_panel(data, domain)
        except Exception as exc:
            logger.warning("load_document_triage: %s", exc)
            return f"<div class='viz-empty'>Error loading document triage: {_escape(str(exc))}</div>"

    def load_document_card(self, matter_id: str, document_ref: str) -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        ref = (document_ref or "").strip().replace("\\", "/")
        if not ref:
            return "<div class='viz-empty'>Select a document to view its card.</div>"
        try:
            domain = self._detect_domain(matter_id)
            data = _run_async(
                self.backend().get_document_card(matter_id, relative_path=ref)
            )
            if not isinstance(data, dict):
                logger.warning("load_document_card: expected dict, got %s", type(data).__name__)
                data = {}
            return _fmt_document_card(data, domain=domain)
        except Exception as exc:
            logger.warning("load_document_card failed: %s", exc)
            return f"<div class='viz-empty'>Error loading document card: {_escape(str(exc))}</div>"

    def get_document_card_choices(self, matter_id: str) -> list:
        if not matter_id or matter_id == "—":
            return []
        try:
            data = _run_async(self.backend().list_reviewable_documents(matter_id))
            if not isinstance(data, list):
                return []
            choices = []
            for doc in data:
                if not isinstance(doc, dict):
                    continue
                path = doc.get("path", "")
                if not path:
                    continue
                dtype = doc.get("doc_type", "")
                label = f"{path} ({dtype})" if dtype else str(path)
                choices.append((label, str(path)))
            return choices
        except Exception as exc:
            logger.warning("get_document_card_choices: %s", exc)
            return []

    def correct_document_card(
        self, matter_id: str, document_ref: str,
        doc_type: str, source_role: str, privilege_flag: bool,
        operative_status: str, flags_text: str,
    ) -> tuple[str, str]:
        if not matter_id or matter_id == "—":
            return "No matter loaded.", ""
        ref = (document_ref or "").strip().replace("\\", "/")
        if not ref:
            return "Select a document first.", ""
        try:
            inv_row = _run_async(
                self.backend().get_document_card(matter_id, relative_path=ref)
            )
            doc_id = ref
            if isinstance(inv_row, dict) and isinstance(inv_row.get("card"), dict):
                doc_id = inv_row["card"].get("doc_id", ref)

            current_priv = False
            if isinstance(inv_row, dict) and isinstance(inv_row.get("card"), dict):
                current_priv = bool(inv_row["card"].get("privilege_flag"))

            fields: dict = {}
            if doc_type and doc_type.strip():
                fields["doc_type"] = doc_type.strip()
            if source_role and source_role.strip():
                fields["source_role"] = source_role.strip()
            if bool(privilege_flag) != current_priv:
                fields["privilege_flag"] = bool(privilege_flag)
            if operative_status and operative_status.strip():
                fields["operative_status"] = operative_status.strip()
            if flags_text and flags_text.strip():
                fields["unresolved_flags"] = [
                    f.strip() for f in flags_text.strip().split(",") if f.strip()
                ]

            result = _run_async(
                self.backend().patch_document_card(matter_id, doc_id, fields)
            )
            if not isinstance(result, dict):
                logger.warning("correct_document_card: expected dict, got %s", type(result).__name__)
                return "Unexpected response.", ""

            if result.get("error"):
                return f"Error: {_escape(str(result['error']))}", ""

            changed = result.get("changed_fields", [])
            staled = result.get("staled_count", 0)
            if not changed:
                status = "No changes detected."
            else:
                status = (
                    f"Updated {', '.join(changed)}."
                    f" {staled} downstream target(s) marked stale."
                )
            card_html = self.load_document_card(matter_id, document_ref)
            return status, card_html
        except Exception as exc:
            logger.warning("correct_document_card failed: %s", exc)
            return f"Error: {_escape(str(exc))}", ""

    def load_taint_summary(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            data = _run_async(self.backend().get_taint_summary(matter_id))
            return _fmt_taint_summary_panel(data, domain)
        except Exception as exc:
            logger.warning("load_taint_summary: %s", exc)
            return f"<div class='viz-empty'>Error loading taint summary: {_escape(str(exc))}</div>"

    def reclassify_document_sensitivity(
        self, matter_id: str, doc_id: str, privilege_flag: bool, domain: str = "legal",
    ) -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        if not doc_id or not doc_id.strip():
            return "<div class='viz-empty'>Please enter a document ID.</div>"
        try:
            data = _run_async(self.backend().reclassify_document_sensitivity(
                matter_id, doc_id.strip(), privilege_flag,
            ))
            return _fmt_sensitivity_review_result(data, domain)
        except Exception as exc:
            logger.warning("reclassify_document_sensitivity: %s", exc)
            return f"<div class='viz-empty'>Error reclassifying: {_escape(str(exc))}</div>"

    def mark_document_stale_action(
        self, matter_id: str, doc_id: str, reason: str, domain: str = "legal",
    ) -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        if not doc_id or not doc_id.strip():
            return "<div class='viz-empty'>Please enter a document ID.</div>"
        try:
            data = _run_async(self.backend().mark_document_stale(
                matter_id, doc_id.strip(), reason.strip() or "manual_stale",
            ))
            return _fmt_sensitivity_review_result(data, domain)
        except Exception as exc:
            logger.warning("mark_document_stale_action: %s", exc)
            return f"<div class='viz-empty'>Error marking stale: {_escape(str(exc))}</div>"

    def mark_span_stale_action(
        self, matter_id: str, span_id: str, reason: str, domain: str = "legal",
    ) -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        if not span_id or not span_id.strip():
            return "<div class='viz-empty'>Please enter a span ID.</div>"
        try:
            data = _run_async(self.backend().mark_span_stale(
                matter_id, span_id.strip(), reason.strip() or "manual_span_stale",
            ))
            return _fmt_sensitivity_review_result(data, domain)
        except Exception as exc:
            logger.warning("mark_span_stale_action: %s", exc)
            return f"<div class='viz-empty'>Error marking span stale: {_escape(str(exc))}</div>"

    def load_investigation_history(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            runs = _run_async(self.backend().list_runs(matter_id, limit=20))
            return _fmt_investigation_history_panel(runs, domain)
        except Exception as exc:
            logger.warning("load_investigation_history: %s", exc)
            return f"<div class='viz-empty'>Error loading investigation history: {_escape(str(exc))}</div>"

    def load_clarification_choices(self, matter_id: str) -> list:
        if not matter_id or matter_id == "—":
            self._clarification_cache = {}
            self._clarification_cache_matter = ""
            return []
        try:
            items = _run_async(self.backend().list_clarifications(matter_id))
            self._clarification_cache = {
                c.get("id", ""): c for c in items
                if isinstance(c, dict) and c.get("status") == "pending"
            }
            self._clarification_cache_matter = matter_id
            return [
                (f"{c.get('question_text', '?')[:80]}", c.get("id", ""))
                for c in items
                if isinstance(c, dict) and c.get("status") == "pending"
            ]
        except Exception as exc:
            logger.warning("Failed to load clarification choices for %s: %s", matter_id, exc)
            self._clarification_cache = {}
            self._clarification_cache_matter = ""
            return []

    def get_clarification_context(self, matter_id: str, question_id: str) -> str:
        if not question_id or not matter_id or matter_id == "—":
            return ""
        cache = getattr(self, "_clarification_cache", {})
        if getattr(self, "_clarification_cache_matter", "") != matter_id:
            return ""
        c = cache.get(question_id)
        if not c or not isinstance(c, dict):
            return ""
        parts = []
        full_q = _escape(str(c.get("question_text", "")))
        if full_q:
            parts.append(f"<div style='font-weight:600;margin-bottom:6px;'>{full_q}</div>")
        why = _escape(str(c.get("why_it_matters") or ""))
        if why:
            parts.append(
                f"<div style='background:#eff6ff;border-left:3px solid #3b82f6;padding:6px 10px;"
                f"border-radius:4px;margin-bottom:6px;font-size:12px;'>"
                f"<strong>Why it matters:</strong> {why}</div>"
            )
        impact = _escape(str(c.get("expected_impact") or ""))
        if impact:
            parts.append(
                f"<div style='background:#f0fdf4;border-left:3px solid #22c55e;padding:6px 10px;"
                f"border-radius:4px;margin-bottom:6px;font-size:12px;'>"
                f"<strong>Expected impact:</strong> {impact}</div>"
            )
        gap_id = c.get("gap_id")
        if gap_id:
            parts.append(
                f"<div style='font-size:11px;color:#6b7280;'>"
                f"Linked gap: <code>{_escape(str(gap_id)[:16])}</code></div>"
            )
        return "".join(parts) if parts else ""

    def get_domain_dropdown_updates(self, matter_id: str) -> tuple:
        domain = self._detect_domain(matter_id)
        maker_choices = _decision_maker_choices_for_domain(domain)
        obj_choices = _objective_choices_for_domain(domain)
        return (
            gr.update(choices=maker_choices, value=None),
            gr.update(choices=obj_choices, value=None),
        )

    def do_answer_clarification(
        self, matter_id: str, question_id: str, answer_text: str
    ) -> str:
        if not matter_id or matter_id == "—":
            return "Load a matter first."
        if not question_id:
            return "Select a clarification question first."
        if not answer_text or not answer_text.strip():
            return "Provide an answer text."
        try:
            found = _run_async(self.backend().answer_clarification(
                matter_id, question_id, answer_text.strip()
            ))
            if found:
                return "Answer recorded successfully."
            return "Clarification question not found — it may have already been answered."
        except Exception as exc:
            logger.warning("do_answer_clarification: %s", exc)
            return f"Error: {exc}"

    def do_generate_clarifications(self, matter_id: str) -> str:
        if not matter_id or matter_id == "—":
            return "Load a matter first."
        try:
            ids = _run_async(self.backend().generate_clarifications(matter_id))
            if ids:
                return f"Generated {len(ids)} clarification question{'s' if len(ids) != 1 else ''}."
            return "No new questions generated — all high-materiality gaps already have pending questions."
        except Exception as exc:
            logger.warning("do_generate_clarifications: %s", exc)
            return f"Error: {exc}"

    def load_trust_overrides(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            data = _run_async(self.backend().list_trust_overrides(matter_id))
            return _fmt_trust_overrides(data, domain=domain)
        except Exception as exc:
            logger.warning("load_trust_overrides: %s", exc)
            return f"<div class='viz-empty'>Error loading trust overrides: {_escape(str(exc))}</div>"

    def do_set_trust_override(
        self, matter_id: str, document_pattern: str, trust_level: str, note: str
    ) -> str:
        if not matter_id or matter_id == "—":
            return "Load a matter first."
        if not document_pattern or not document_pattern.strip():
            return "Enter a document pattern."
        if trust_level not in ("low", "normal", "high"):
            return "Select a valid trust level."
        try:
            override_id = _run_async(self.backend().set_trust_override(
                matter_id, document_pattern.strip(), trust_level, note=(note or "").strip()
            ))
            if override_id:
                return f"Trust override set (ID: {override_id[:8]}…). Belief revision triggered on affected assertions."
            return "Override set."
        except Exception as exc:
            logger.warning("do_set_trust_override: %s", exc)
            return f"Error: {exc}"

    def do_delete_trust_override(self, matter_id: str, document_pattern: str) -> tuple[str, str]:
        if not matter_id or matter_id == "—":
            return "Load a matter first.", ""
        if not document_pattern or not document_pattern.strip():
            return "Enter the document pattern to remove.", ""
        try:
            deleted = _run_async(self.backend().delete_trust_override(
                matter_id, document_pattern.strip()
            ))
            if deleted:
                refreshed = self.load_trust_overrides(matter_id, domain=self._detect_domain(matter_id))
                return "Trust override removed.", refreshed
            return "Override not found — check the document pattern.", ""
        except Exception as exc:
            logger.warning("do_delete_trust_override: %s", exc)
            return f"Error: {_escape(str(exc))}", ""

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
        except Exception as exc:
            logger.warning("Failed to load reviewable documents for %s: %s", matter_id, exc)
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

    def load_communication_map(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            graph = _run_async(self.backend().get_communication_map(matter_id))
            return _fmt_communication_map_panel(graph, domain=domain)
        except Exception as exc:
            logger.warning("load_communication_map: %s", exc)
            return f"<div class='viz-empty'>Error loading communication map: {_escape(str(exc))}</div>"

    def load_duplicate_actors(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            pairs = _run_async(self.backend().find_duplicate_actors(matter_id))
            return _fmt_duplicate_actors_panel(pairs, domain)
        except Exception as exc:
            logger.warning("load_duplicate_actors: %s", exc)
            return f"<div class='viz-empty'>Error scanning for duplicates: {_escape(str(exc))}</div>"

    def do_merge_actors(self, matter_id: str, keep_id: str, merge_id: str, confirmed: bool = False) -> tuple[str, str]:
        if not matter_id or matter_id == "—":
            return "Load a matter first.", ""
        if not keep_id or not keep_id.strip():
            return "Enter the ID of the actor to keep.", ""
        if not merge_id or not merge_id.strip():
            return "Enter the ID of the actor to merge away.", ""
        if keep_id.strip() == merge_id.strip():
            return "Cannot merge an actor with itself.", ""
        if not confirmed:
            return "Check the confirmation box before merging.", ""
        try:
            _run_async(self.backend().merge_actors(matter_id, keep_id.strip(), merge_id.strip()))
            refreshed = self.load_duplicate_actors(matter_id)
            return f"Merged {merge_id.strip()[:12]} into {keep_id.strip()[:12]}.", refreshed
        except Exception as exc:
            logger.warning("do_merge_actors: %s", exc)
            return f"Error: {_escape(str(exc))}", ""

    def do_resolve_actor(self, matter_id: str, name: str) -> str:
        if not matter_id or matter_id == "—":
            return "No matter loaded."
        name = (name or "").strip()
        if not name:
            return "Enter an actor name to look up."
        try:
            result = _run_async(self.backend().resolve_actor(matter_id, name))
            actor_id = result.get("actor_id")
            if not actor_id:
                return f"No actor found matching '{_escape(name)}'."
            actor = result.get("actor") or {}
            display = _escape(actor.get("name", actor_id) if isinstance(actor, dict) else str(actor_id))
            return f"Resolved: **{display}** · ID: `{_escape(str(actor_id)[:24])}`"
        except Exception as exc:
            logger.warning("resolve_actor failed: %s", exc)
            return f"Error: {_escape(str(exc))}"

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
            logger.warning("load_llm_analytics: %s", exc)
            return f"<div class='viz-empty'>Error loading LLM analytics: {_escape(str(exc))}</div>"

    def load_decision_context(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            ctx = _run_async(self.backend().get_decision_context(matter_id))
            return _fmt_decision_context(ctx, domain=domain)
        except Exception as exc:
            logger.warning("load_decision_context: %s", exc)
            return f"<div class='viz-empty'>Error: {_escape(str(exc))}</div>"

    def do_set_decision_context(
        self, matter_id: str, maker_type: str, objective: str,
        name: str, notes: str, narrow: bool,
    ) -> tuple[str, str]:
        if not matter_id or matter_id == "—":
            return "Load a matter first.", ""
        try:
            _run_async(self.backend().set_decision_context(
                matter_id,
                decision_maker_type=maker_type or None,
                decision_maker_name=name.strip() or None,
                objective=objective or None,
                strategic_notes=notes.strip() or None,
                scope_narrow=narrow,
            ))
            domain = self._detect_domain(matter_id)
            return "Decision context updated.", self.load_decision_context(matter_id, domain=domain)
        except Exception as exc:
            logger.warning("do_set_decision_context: %s", exc)
            return f"Error: {_escape(str(exc))}", ""

    def do_clear_decision_context(self, matter_id: str) -> tuple[str, str]:
        if not matter_id or matter_id == "—":
            return "Load a matter first.", ""
        try:
            _run_async(self.backend().clear_decision_context(matter_id))
            return "Decision context cleared.", "<div class='viz-empty'>No decision context set.</div>"
        except Exception as exc:
            logger.warning("do_clear_decision_context: %s", exc)
            return f"Error: {_escape(str(exc))}", ""

    def load_annotations(self, matter_id: str, domain: str = "legal") -> str:
        if not matter_id or matter_id == "—":
            return "<div class='viz-empty'>No matter loaded.</div>"
        try:
            annotations = _run_async(self.backend().list_annotations(matter_id))
            return _fmt_annotations_panel(annotations, domain=domain)
        except Exception as exc:
            logger.warning("load_annotations: %s", exc)
            return f"<div class='viz-empty'>Error: {_escape(str(exc))}</div>"

    def do_add_annotation(self, matter_id: str, doc: str, text: str, ann_type: str) -> tuple[str, str]:
        if not matter_id or matter_id == "—":
            return "Load a matter first.", ""
        if not doc or not doc.strip():
            return "Enter a document name or pattern.", ""
        if not text or not text.strip():
            return "Enter your note text.", ""
        try:
            _run_async(self.backend().add_annotation(
                matter_id, doc.strip(), text.strip(), ann_type or "strategic",
            ))
            refreshed = self.load_annotations(matter_id, domain=self._detect_domain(matter_id))
            return "Note added.", refreshed
        except Exception as exc:
            logger.warning("do_add_annotation: %s", exc)
            return f"Error: {_escape(str(exc))}", ""

    def do_delete_annotation(self, matter_id: str, annotation_id: str) -> tuple[str, str]:
        if not matter_id or matter_id == "—":
            return "Load a matter first.", ""
        if not annotation_id or not annotation_id.strip():
            return "Enter the annotation ID to delete.", ""
        try:
            deleted = _run_async(self.backend().delete_annotation(matter_id, annotation_id.strip()))
            if deleted:
                refreshed = self.load_annotations(matter_id, domain=self._detect_domain(matter_id))
                return "Note deleted.", refreshed
            return "Annotation not found — check the ID.", ""
        except Exception as exc:
            logger.warning("do_delete_annotation: %s", exc)
            return f"Error: {_escape(str(exc))}", ""

    _EXPORT_LABELS: dict[str, dict[str, str]] = {
        "legal": {"title": "MATTER SUMMARY REPORT", "issues": "ISSUES BY EVIDENCE COVERAGE (weakest first)",
                  "assertions": "KEY ASSERTIONS", "gaps": "OPEN GAPS", "contradictions": "CONTRADICTIONS",
                  "proof": "PROOF STATE BY ISSUE", "authorities": "AUTHORITIES & REFERENCES",
                  "financial": "FINANCIAL RECONCILIATION", "threshold_alerts": "FINANCIAL HEALTH ALERTS",
                  "issue_unit": "issues", "assertion_unit": "assertions"},
        "finance": {"title": "ANALYSIS SUMMARY REPORT", "issues": "THESES BY EVIDENCE COVERAGE (weakest first)",
                    "assertions": "KEY FINDINGS", "gaps": "OPEN GAPS", "contradictions": "DATA CONFLICTS",
                    "proof": "PROOF STATE BY THESIS", "authorities": "REFERENCES & SOURCES",
                    "financial": "FINANCIAL RECONCILIATION", "threshold_alerts": "FINANCIAL RISK ALERTS",
                    "issue_unit": "theses", "assertion_unit": "findings"},
        "coding": {"title": "ANALYSIS SUMMARY REPORT", "issues": "HYPOTHESES BY EVIDENCE COVERAGE (weakest first)",
                   "assertions": "KEY FINDINGS", "gaps": "OPEN GAPS", "contradictions": "CONTRADICTIONS",
                   "proof": "PROOF STATE BY HYPOTHESIS", "authorities": "REFERENCES & DOCUMENTATION",
                   "financial": "METRIC RECONCILIATION", "threshold_alerts": "METRIC THRESHOLD ALERTS",
                   "issue_unit": "hypotheses", "assertion_unit": "findings"},
        "academic_research": {"title": "RESEARCH SESSION SUMMARY", "issues": "CLAIMS BY EVIDENCE COVERAGE (weakest first)",
                              "assertions": "KEY CLAIMS", "gaps": "OPEN GAPS", "contradictions": "CONFLICTING FINDINGS",
                              "proof": "PROOF STATE BY CLAIM", "authorities": "CITATIONS & SOURCES",
                              "financial": "QUANTITATIVE RECONCILIATION", "threshold_alerts": "QUANTITATIVE THRESHOLD ALERTS",
                              "issue_unit": "claims", "assertion_unit": "claims"},
        "biomedical": {"title": "CASE REVIEW SUMMARY", "issues": "FINDINGS BY EVIDENCE COVERAGE (weakest first)",
                       "assertions": "KEY FINDINGS", "gaps": "OPEN GAPS", "contradictions": "CONFLICTING EVIDENCE",
                       "proof": "PROOF STATE BY FINDING", "authorities": "CLINICAL REFERENCES",
                       "financial": "QUANTITATIVE RECONCILIATION", "threshold_alerts": "CLINICAL THRESHOLD ALERTS",
                       "issue_unit": "findings", "assertion_unit": "findings"},
    }

    def export_summary_report(self, matter_id: str) -> "str | None":
        if not matter_id or matter_id == "—":
            raise gr.Error("Load a matter first before exporting.")
        try:
            data = _run_async(self.backend().export_matter_summary(matter_id))
        except Exception as exc:
            logger.warning("Export failed: %s", exc)
            raise gr.Error(f"Export failed: {exc}")
        if not data or not isinstance(data, dict):
            raise gr.Error("Export returned empty data.")
        import json as _json
        import tempfile as _tempfile
        domain = self._detect_domain(matter_id)
        el = self._EXPORT_LABELS.get(domain, self._EXPORT_LABELS["legal"])
        lines: list[str] = []
        lines.append(el["title"])
        lines.append(f"Matter ID: {data.get('matter_id', '—')}")
        lines.append(f"Generated: {data.get('generated_at', '—')}")
        stats = data.get("stats", {})
        lines.append(f"\n{'='*60}")
        lines.append("OVERVIEW")
        lines.append(f"{'='*60}")
        lines.append(f"  {el['assertion_unit'].title()}:     {stats.get('assertion_count', 0)}")
        lines.append(f"  Open {el['issue_unit']}:    {stats.get('open_issue_count', 0)}")
        lines.append(f"  Open gaps:      {stats.get('open_gap_count', 0)}")
        lines.append(f"  Actors:         {stats.get('actor_count', 0)}")
        lines.append(f"  Clarifications: {stats.get('pending_clarifications', 0)} pending")
        issues = data.get("issues", [])
        if issues:
            lines.append(f"\n{'='*60}")
            lines.append(el["issues"])
            lines.append(f"{'='*60}")
            for iss in issues[:20]:
                cov = float(iss.get("coverage_fraction", 0))
                gap_flag = " [PROOF GAP]" if iss.get("has_proof_gap") else ""
                lines.append(
                    f"  [{cov*100:.0f}%] {iss.get('title', '—')[:80]}{gap_flag}"
                )
        assertions = data.get("assertions", [])
        if assertions:
            lines.append(f"\n{'='*60}")
            lines.append(f"{el['assertions']} (first 50)")
            lines.append(f"{'='*60}")
            for a in assertions[:50]:
                bs = a.get("belief_state", "—")
                src = a.get("document_id") or "—"
                text = (a.get("proposition_text") or "—")[:100]
                lines.append(f"  [{bs}] {text}")
                lines.append(f"         Source: {src}")
        gaps = data.get("gaps", [])
        if gaps:
            lines.append(f"\n{'='*60}")
            lines.append(el["gaps"])
            lines.append(f"{'='*60}")
            for g in gaps[:20]:
                lines.append(f"  - {g.get('description', '—')[:100]}")
        contradictions = data.get("contradictions", [])
        if contradictions:
            lines.append(f"\n{'='*60}")
            lines.append(el["contradictions"])
            lines.append(f"{'='*60}")
            for c in contradictions[:15]:
                lines.append(f"  - {c.get('a_text', '—')[:60]}")
                lines.append(f"    vs. {c.get('b_text', '—')[:60]}")
        so = data.get("so_metrics", {})
        if so:
            lines.append(f"\n{'='*60}")
            lines.append("SACRED OUTCOME SCORECARD")
            lines.append(f"{'='*60}")
            targets_met = so.get("targets_met", {})
            for key, val in so.items():
                if key in ("targets", "targets_met"):
                    continue
                if val is None:
                    continue
                passed = targets_met.get(key)
                status = " ✓" if passed else (" ✗" if passed is False else "")
                if isinstance(val, float):
                    lines.append(f"  {key}: {val:.3f}{status}")
                elif isinstance(val, bool):
                    lines.append(f"  {key}: {'Yes' if val else 'No'}{status}")
                else:
                    lines.append(f"  {key}: {val}{status}")
        try:
            proof = _run_async(self.backend().get_proof_state_summary(matter_id))
            proof_issues = proof.get("issues", []) if isinstance(proof, dict) else []
            if proof_issues:
                lines.append(f"\n{'='*60}")
                lines.append(el["proof"])
                lines.append(f"{'='*60}")
                summary = proof.get("summary", {}) if isinstance(proof, dict) else {}
                lines.append(f"  Average sufficiency: {summary.get('avg_sufficiency', 0):.0%}")
                for ps in proof_issues[:20]:
                    if not isinstance(ps, dict):
                        continue
                    title = (ps.get("issue_title") or ps.get("issue_id") or "—")[:60]
                    status = ps.get("proof_status", "?")
                    suf = ps.get("sufficiency", 0)
                    suf_str = f"{float(suf):.0%}" if isinstance(suf, (int, float)) else str(suf)
                    sup = ps.get("supporting_count", 0)
                    att = ps.get("attacking_count", 0)
                    lines.append(f"  [{status}] {title}  ({suf_str}, +{sup}/-{att})")
        except Exception as exc:
            logger.warning("Export proof state section failed: %s", exc)

        try:
            auth_data = _run_async(self.backend().get_authority_network(matter_id))
            authorities = auth_data.get("authorities", []) if isinstance(auth_data, dict) else []
            if authorities:
                lines.append(f"\n{'='*60}")
                lines.append(el["authorities"])
                lines.append(f"{'='*60}")
                for auth in authorities[:30]:
                    if not isinstance(auth, dict):
                        continue
                    cite = (auth.get("citation") or "—")[:80]
                    atype = auth.get("authority_type", "?")
                    weight = auth.get("weight", "?")
                    n_links = len(auth.get("issue_links", []) or [])
                    lines.append(f"  [{atype}/{weight}] {cite}  ({n_links} linked issues)")
        except Exception as exc:
            logger.warning("Export authority section failed: %s", exc)

        try:
            quant = _run_async(self.backend().get_quant_summary(matter_id))
            recon = quant.get("payment_reconciliation", {}) if isinstance(quant, dict) else {}
            if recon and recon.get("invoiced") is not None:
                lines.append(f"\n{'='*60}")
                lines.append(el["financial"])
                lines.append(f"{'='*60}")
                ccy = recon.get("currency", "USD")
                for label, key in [("Invoiced", "invoiced"), ("Paid", "paid"),
                                   ("Disputed", "disputed"), ("Exposure", "exposure")]:
                    val = recon.get(key, 0)
                    if isinstance(val, (int, float)):
                        lines.append(f"  {label}: {ccy} {val:,.2f}")
            waterfall = quant.get("damages_waterfall", []) if isinstance(quant, dict) else []
            if waterfall:
                lines.append(f"\n  Damages Waterfall:")
                for d in waterfall[:15]:
                    if not isinstance(d, dict):
                        continue
                    comp = d.get("component") or "(uncategorised)"
                    amt = d.get("claimed_amount", 0)
                    amt_str = f"{amt:,.2f}" if isinstance(amt, (int, float)) else str(amt)
                    conflicts = len(d.get("conflicts", []) or [])
                    flag = f" [!{conflicts} conflicts]" if conflicts else ""
                    lines.append(f"    {comp}: {amt_str}{flag}")
        except Exception as exc:
            logger.warning("Export financial section failed: %s", exc)

        try:
            violations = _run_async(self.backend().get_quant_thresholds(matter_id))
            if violations:
                lines.append(f"\n{'='*60}")
                lines.append(el["threshold_alerts"])
                lines.append(f"{'='*60}")
                for v in violations:
                    if not isinstance(v, dict):
                        continue
                    level = v.get("level", "?")
                    desc = (v.get("description") or "—")[:120]
                    amt = v.get("amount")
                    amt_str = f" ({amt:,.2f})" if isinstance(amt, (int, float)) else ""
                    lines.append(f"  [{level}] {desc}{amt_str}")
        except Exception as exc:
            logger.warning("Export threshold alerts section failed: %s", exc)

        lines.append(f"\n{'='*60}")
        lines.append("END OF REPORT")
        lines.append(f"{'='*60}\n")
        content = "\n".join(lines)
        fd = _tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", prefix=f"irys_report_{matter_id[:8]}_",
            delete=False, encoding="utf-8",
        )
        fd.write(content)
        fd.close()
        return fd.name

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
            logger.warning("do_correct_assertion: %s", exc)
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
                logger.warning("_do_resume: %s", exc)
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
            logger.warning("do_redirect: %s", exc)
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
                folder_upload_relpaths = gr.Textbox(
                    visible=False,
                    elem_id="folder-upload-relpaths",
                    value="",
                )
                with gr.Accordion("Danger zone", open=False):
                    delete_matter_btn = gr.Button("Delete entire matter", variant="stop")
                file_manage_status = gr.HTML()

            # --- Create new matter ---
            with gr.Accordion("+ Create new matter", open=False):
                domain_choices = [
                    (v["label"], k) for k, v in _DOMAIN_CREATE_PRESETS.items()
                ]
                domain_selector = gr.Dropdown(
                    label="Matter type",
                    choices=domain_choices,
                    value="legal",
                    interactive=True,
                )
                matter_name_input = gr.Textbox(
                    label="Matter name",
                    placeholder=_DOMAIN_CREATE_PRESETS["legal"]["placeholder"],
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
                folder_upload_new_relpaths = gr.Textbox(
                    visible=False,
                    elem_id="folder-upload-new-relpaths",
                    value="",
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

                with gr.Row():
                    refresh_sidebar_btn = gr.Button("↻ Refresh", variant="secondary", size="sm")
                    export_report_btn = gr.Button("Download Report", variant="secondary", size="sm")
                export_report_file = gr.File(label="Report", visible=False)

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
            with gr.Row():
                assertion_search_box = gr.Textbox(
                    label="Search facts",
                    placeholder="Type to filter facts by content…",
                    scale=4,
                )
                assertion_search_btn = gr.Button("Search", variant="secondary", size="sm", scale=1)
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
        # ASSERTION INSPECTOR — deep-dive provenance for a single fact
        # ==================================================================

        with gr.Accordion(
            "Assertion Inspector — trace how a fact was derived and who supports it",
            open=False,
        ):
            gr.Markdown(
                "Copy a Fact ID from the table above and click **Inspect** to see its "
                "full provenance trail: which documents it came from, which model extracted it, "
                "and how many other facts support or attack it."
            )
            with gr.Row():
                inspector_assertion_id = gr.Textbox(
                    label="Fact ID (copy from table above)",
                    placeholder="e.g. a-3f8c…",
                    scale=3,
                )
                inspect_btn = gr.Button("Inspect", variant="primary", size="sm", scale=1)
            inspector_html = gr.HTML(
                "<div class='viz-empty'>Enter an assertion ID to see its provenance and evidence summary.</div>"
            )
            trace_btn = gr.Button("Show Impact Trace", variant="secondary", size="sm")
            trace_html = gr.HTML(
                "<div class='viz-empty'>Click <em>Show Impact Trace</em> to see "
                "source documents, affected issues, dependent facts, and revision history.</div>"
            )

        # ==================================================================
        # ISSUE EVIDENCE DRILL-DOWN — see assertions linked to a specific issue
        # ==================================================================

        with gr.Accordion(
            "Issue Evidence — drill into the facts backing a specific issue",
            open=False,
        ):
            gr.Markdown(
                "Copy an Issue ID from the Issues panel or Proof State table "
                "and click **Show Evidence** to see all assertions linked to that "
                "issue, grouped by supporting, attacking, and neutral."
            )
            with gr.Row():
                issue_drilldown_id = gr.Textbox(
                    label="Issue ID",
                    placeholder="Paste issue ID from issues panel or proof state table",
                    scale=3,
                )
                issue_drilldown_btn = gr.Button("Show Evidence", variant="primary", size="sm", scale=1)
            issue_assertions_html = gr.HTML(
                "<div class='viz-empty'>Enter an issue ID to see its linked evidence.</div>"
            )
            source_agreement_html = gr.HTML(
                "<div class='viz-empty'>Source agreement analysis will appear after selecting an issue.</div>"
            )
            assertion_graph_html = gr.HTML(
                "<div class='viz-empty'>Assertion relationship map will appear after selecting an issue.</div>"
            )
            issue_closure_html = gr.HTML(
                "<div class='viz-empty'>Issue closure workbench will appear after selecting an issue.</div>"
            )

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
            with gr.Row():
                bulk_span_ref = gr.Textbox(
                    label="Section / span reference",
                    placeholder="e.g. Section 4.2, clause-ref, or span ID",
                    scale=4,
                )
                bulk_span_verify_btn = gr.Button(
                    "Verify all from section", variant="primary",
                    scale=1, min_width=180,
                )
            bulk_span_result = gr.Markdown("")
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

        with gr.Accordion("Investigation Readiness — should you rely on this analysis?", open=True):
            gr.Markdown(
                "Overall readiness assessment: open blockers, coverage gaps, unresolved "
                "contradictions, pending clarifications, and unverified critical facts. "
                "Green means ready for reliance; red means action is required first."
            )
            readiness_html = gr.HTML(
                "<div class='viz-empty'>Readiness will appear after investigation.</div>"
            )
            refresh_readiness_btn = gr.Button(
                "Refresh Readiness", variant="secondary", size="sm",
            )

        with gr.Accordion("Next Run Context — what Irys will carry into the next investigation", open=True):
            gr.Markdown(
                "Snapshot of the engine's query context: what assertions, actors, documents, "
                "gaps, and domain calibration Irys will carry into the next run. "
                "Use this to verify the system's starting knowledge before re-investigating."
            )
            query_context_html = gr.HTML(
                "<div class='viz-empty'>Query context will appear after investigation.</div>"
            )
            refresh_query_context_btn = gr.Button(
                "Refresh Context", variant="secondary", size="sm",
            )

        with gr.Accordion("Gap-to-Action Workbench — consolidated gap review", open=True):
            gr.Markdown(
                "Consolidated view of every open gap with affected issues, missing sources, "
                "pending clarifications, and a recommended next action. Higher materiality "
                "gaps appear first."
            )
            gap_workbench_html = gr.HTML(
                "<div class='viz-empty'>Gap workbench will appear after investigation.</div>"
            )
            refresh_gap_workbench_btn = gr.Button(
                "Refresh Workbench", variant="secondary", size="sm",
            )

        with gr.Accordion("Gaps & Missingness — what the system knows it does not know", open=False):
            gr.Markdown(
                "Every gap represents something the matter model needs but does not have: "
                "a missing document, an unresolved numeric conflict, a missing element of proof. "
                "Higher materiality means the gap is more likely to affect conclusions."
            )
            gaps_detail_html = gr.HTML("<div class='viz-empty'>Gaps will appear here after an investigation.</div>")
            refresh_gaps_btn = gr.Button("Refresh Gaps", variant="secondary", size="sm")
            with gr.Accordion("Gap Actions — resolve or escalate", open=False):
                gap_action_dropdown = gr.Dropdown(
                    label="Select a gap",
                    choices=[],
                    interactive=True,
                    scale=4,
                )
                refresh_gap_choices_btn = gr.Button(
                    "Refresh gap list", variant="secondary", size="sm",
                )
                with gr.Row():
                    resolve_gap_note_input = gr.Textbox(
                        label="Resolution note (optional)", scale=4,
                    )
                    resolve_gap_btn = gr.Button("Resolve Gap", variant="primary", size="sm", scale=1)
                    escalate_gap_btn = gr.Button("Escalate", variant="stop", size="sm", scale=1)
                resolve_gap_result = gr.Markdown("")

        with gr.Accordion("Next Steps — prioritized recommendations for your review", open=False):
            gr.Markdown(
                "Actionable recommendations derived from the current matter state: "
                "conflicts to resolve, issues needing evidence, documents to supply, "
                "and clarifications to answer. Ranked by impact on analysis quality."
            )
            steering_panel_html = gr.HTML("<div class='viz-empty'>Recommendations will appear here after an investigation.</div>")
            refresh_steering_btn = gr.Button("Refresh Recommendations", variant="secondary", size="sm")
            with gr.Accordion("Quick Execute — act on a recommendation directly", open=False):
                gr.Markdown(
                    "Select a recommendation from the dropdown, optionally provide "
                    "additional input (required for clarifications and corrections), "
                    "then click Execute to apply it immediately."
                )
                with gr.Row():
                    steering_action_dropdown = gr.Dropdown(
                        label="Select action to execute",
                        choices=[],
                        scale=4,
                    )
                    steering_action_input = gr.Textbox(
                        label="Additional input (answer text or correction reason)",
                        placeholder="Required for clarifications and corrections",
                        scale=3,
                    )
                    steering_execute_btn = gr.Button(
                        "Execute", variant="primary", size="sm", scale=1,
                    )
                steering_execute_result = gr.Textbox(
                    label="Result", interactive=False,
                )

        with gr.Accordion("Adjust issue priority", open=False):
            gr.Markdown(
                "Override the computed priority of an issue. Critical and high-priority "
                "issues receive deeper investigation and are surfaced first in reports."
            )
            with gr.Row():
                priority_issue_id_input = gr.Textbox(
                    label="Issue ID (copy from the Issues panel)", scale=3,
                )
                priority_dropdown = gr.Dropdown(
                    label="Priority",
                    choices=[
                        ("Critical — must investigate deeply", "critical"),
                        ("High — important to the outcome", "high"),
                        ("Medium — standard investigation", "medium"),
                        ("Low — background concern only", "low"),
                    ],
                    scale=2,
                )
                priority_apply_btn = gr.Button("Set Priority", variant="primary", size="sm", scale=1)
            priority_result = gr.Markdown("")

        with gr.Accordion("Working Assumptions — what Irys is taking as given", open=False):
            gr.Markdown(
                "Irys logs every assumption it makes during analysis. If an assumption "
                "is wrong, the conclusions that depend on it may change. Invalidated "
                "assumptions are flagged so you can see what shifted."
            )
            assumptions_detail_html = gr.HTML("<div class='viz-empty'>Assumptions will appear here after an investigation.</div>")
            refresh_assumptions_btn = gr.Button("Refresh Assumptions", variant="secondary", size="sm")
            with gr.Accordion("Challenge or confirm an assumption", open=False):
                with gr.Row():
                    assumption_id_input = gr.Textbox(
                        label="Assumption ID (copy from table above)", scale=3,
                    )
                    assumption_action_dropdown = gr.Dropdown(
                        label="Action",
                        choices=[
                            ("Confirm — mark as validated", "confirmed"),
                            ("Invalidate — mark as disproved", "invalidated"),
                            ("Reset to provisional", "provisional"),
                        ],
                        scale=2,
                    )
                    assumption_reason_input = gr.Textbox(
                        label="Reason (optional)", scale=3,
                    )
                    assumption_action_btn = gr.Button("Apply", variant="primary", size="sm", scale=1)
                assumption_action_result = gr.Markdown("")

        with gr.Accordion("Assumption Review Workbench — lifecycle buckets with impact links", open=False):
            gr.Markdown(
                "Assumptions grouped by lifecycle status: provisional, confirmed, "
                "and invalidated. Each shows linked targets (predicates, issues) "
                "and impact on downstream analysis. Use the controls above to "
                "confirm, invalidate, or reset assumptions."
            )
            assumption_review_html = gr.HTML(
                "<div class='viz-empty'>Assumption review will appear here after an investigation.</div>"
            )
            refresh_assumption_review_btn = gr.Button(
                "Refresh Assumption Review", variant="secondary", size="sm",
            )

        with gr.Accordion("Financials — payments, damages, and numeric disputes", open=False):
            gr.Markdown(
                "Invoices, payments, damages claims, and numeric conflicts — "
                "pulled directly from your documents and reconciled. "
                "If two documents disagree on an amount, Irys flags the conflict."
            )
            quant_md = gr.HTML("<div class='viz-empty'>Financial data will appear here after an investigation.</div>")
            with gr.Row():
                refresh_quant_btn = gr.Button("Refresh Financials", variant="secondary", size="sm")
                detect_conflicts_btn = gr.Button("Detect Amount Conflicts", variant="primary", size="sm")
            detect_conflicts_result = gr.Markdown("")
            with gr.Accordion("Metric Ontology — classify extracted numeric facts", open=False):
                quant_ontology_html = gr.HTML(
                    "<div class='viz-empty'>Metric ontology will appear after an investigation.</div>"
                )
                refresh_quant_ontology_btn = gr.Button("Refresh Metric Ontology", variant="secondary", size="sm")
                with gr.Row():
                    quant_alias_raw_dropdown = gr.Dropdown(
                        label="Raw metric", choices=[], interactive=True, scale=2,
                    )
                    quant_alias_canonical_dropdown = gr.Dropdown(
                        label="Canonical metric",
                        choices=_canonical_metric_choices("legal"),
                        allow_custom_value=True, interactive=True, scale=2,
                    )
                    quant_alias_unit_input = gr.Textbox(
                        label="Unit (optional)", placeholder="e.g. USD, days, %", scale=1,
                    )
                    approve_quant_alias_btn = gr.Button("Approve", variant="primary", size="sm", scale=1)
                quant_alias_result = gr.Markdown("")

            with gr.Accordion("Quant Fact Review — inspect every extracted number", open=False):
                gr.Markdown(
                    "Every numeric fact extracted from documents: amounts, dates, rates, counts. "
                    "Grouped by type with conflict detection. Review individual facts and their "
                    "source linkage to ensure the numbers driving the analysis are correct."
                )
                quant_facts_html = gr.HTML(
                    "<div class='viz-empty'>Quant facts will appear after an investigation.</div>"
                )
                refresh_quant_facts_btn = gr.Button("Refresh Quant Facts", variant="secondary", size="sm")

        with gr.Accordion("Decision Leverage Map — what to review next", open=False):
            gr.Markdown(
                "Ranked leverage points: the interventions most likely to change the outcome. "
                "Connects weak objectives, unreviewed assumptions, tainted evidence, numeric "
                "conflicts, open gaps, and pending reviews into a single prioritized view."
            )
            decision_leverage_html = gr.HTML(
                "<div class='viz-empty'>Leverage map will appear after an investigation.</div>"
            )
            refresh_leverage_btn = gr.Button("Refresh Leverage Map", variant="secondary", size="sm")

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

        with gr.Accordion("Source Calibration — are conclusions grounded in the right source types?", open=False):
            gr.Markdown(
                "Evaluates whether each issue is supported by a diverse, professionally "
                "appropriate mix of source types. Flags single-source reliance, "
                "advocacy-only support, and missing source obligations."
            )
            source_calibration_html = gr.HTML(
                "<div class='viz-empty'>Source calibration will appear after investigation.</div>"
            )
            refresh_source_calibration_btn = gr.Button(
                "Refresh Calibration", variant="secondary", size="sm",
            )

        with gr.Accordion("Proof State — sufficiency and predicate coverage by issue", open=False):
            gr.Markdown(
                "Per-issue proof analysis: sufficiency scores, predicate coverage, "
                "trust-weighted support vs. attack balance, and advocacy-only warnings."
            )
            proof_state_html = gr.HTML("<div class='viz-empty'>Proof state will appear here after an investigation.</div>")
            with gr.Row():
                refresh_proof_btn = gr.Button("Refresh Proof State", variant="secondary", size="sm")
                recompute_proof_btn = gr.Button("Recompute Proof State", variant="primary", size="sm")
            with gr.Accordion("Recompute single issue", open=False):
                with gr.Row():
                    proof_issue_id_input = gr.Textbox(label="Issue ID", placeholder="Paste issue ID from table above", scale=3)
                    recompute_issue_proof_btn = gr.Button("Recompute Issue", variant="secondary", size="sm", scale=1)
                recompute_issue_result = gr.Markdown("")

        with gr.Accordion("Authorities & References — cited sources of law, standards, and precedent", open=False):
            gr.Markdown(
                "Authorities, standards, and references cited in the analysis, "
                "with their weight, type, jurisdiction, and links to relevant issues."
            )
            with gr.Row():
                authority_search_box = gr.Textbox(
                    label="Search authorities",
                    placeholder="Search by citation or name…",
                    scale=4,
                )
                authority_search_btn = gr.Button("Search", variant="secondary", size="sm", scale=1)
            authority_html = gr.HTML("<div class='viz-empty'>Authority data will appear here after an investigation.</div>")
            refresh_authority_btn = gr.Button("Refresh Authorities", variant="secondary", size="sm")

            with gr.Accordion("Add or update an authority", open=False):
                with gr.Row():
                    auth_citation = gr.Textbox(label="Citation", placeholder="e.g. Smith v. Jones, 123 F.3d 456", scale=3)
                    auth_name = gr.Textbox(label="Short name (optional)", placeholder="e.g. Smith v. Jones", scale=2)
                with gr.Row():
                    auth_type = gr.Dropdown(
                        label="Type",
                        choices=[
                            ("Case law", "case"), ("Statute", "statute"),
                            ("Regulation", "regulation"), ("Rule", "rule"),
                            ("Secondary source", "secondary"),
                        ],
                        value="case", scale=1,
                    )
                    auth_weight = gr.Dropdown(
                        label="Weight",
                        choices=[
                            ("Binding", "binding"), ("Persuasive", "persuasive"),
                            ("Neutral", "neutral"),
                        ],
                        value="persuasive", scale=1,
                    )
                    auth_jurisdiction = gr.Textbox(label="Jurisdiction (optional)", placeholder="e.g. 9th Cir.", scale=1)
                upsert_authority_btn = gr.Button("Add / Update Authority", variant="primary", size="sm")
                upsert_authority_result = gr.Markdown("")

            with gr.Accordion("Link / unlink authority ↔ issue", open=False):
                with gr.Row():
                    link_auth_id = gr.Textbox(label="Authority ID", placeholder="Copy from table above", scale=2)
                    link_issue_id = gr.Textbox(label="Issue ID", placeholder="Issue to link", scale=2)
                    link_relevance = gr.Dropdown(
                        label="Relevance",
                        choices=[
                            ("Supporting", "supporting"), ("Attacking", "attacking"),
                            ("Neutral", "neutral"),
                        ],
                        value="supporting", scale=1,
                    )
                with gr.Row():
                    link_authority_btn = gr.Button("Link", variant="primary", size="sm")
                    unlink_authority_btn = gr.Button("Unlink", variant="stop", size="sm")
                link_authority_result = gr.Markdown("")

        with gr.Accordion("Document Intelligence — what the system knows about each source", open=False):
            gr.Markdown(
                "Per-document profile cards: type classification, source side, author, "
                "operative status, privilege flags, salience score, and unresolved issues."
            )
            doc_intel_html = gr.HTML("<div class='viz-empty'>Document intelligence will appear here after an investigation.</div>")
            refresh_doc_intel_btn = gr.Button("Refresh Documents", variant="secondary", size="sm")

            with gr.Accordion("Document Notes — attach strategic guidance to documents", open=False):
                gr.Markdown(
                    "Notes you add here are injected into the reasoning engine's context "
                    "so Irys can use your domain knowledge during investigation. "
                    "Use *strategic* for broad guidance, *reliability* for source quality notes, "
                    "or *scope* to mark documents as irrelevant."
                )
                annotations_html = gr.HTML("<div class='viz-empty'>No notes yet.</div>")
                refresh_annotations_btn = gr.Button("Refresh Notes", variant="secondary", size="sm")
                gr.Markdown("#### Add a note")
                with gr.Row():
                    annotation_doc = gr.Textbox(label="Document name or pattern", placeholder="e.g. contract.pdf", scale=3)
                    annotation_type = gr.Dropdown(
                        label="Type",
                        choices=[("Strategic guidance", "strategic"), ("Reliability concern", "reliability"), ("Scope / relevance", "scope")],
                        value="strategic",
                        scale=1,
                    )
                annotation_text = gr.Textbox(label="Your note", placeholder="e.g. This report overstates damages — focus on §4 corrections", lines=2)
                add_annotation_btn = gr.Button("Add Note", variant="primary", size="sm")
                annotation_result = gr.Markdown("")
                gr.Markdown("#### Remove a note")
                with gr.Row():
                    delete_annotation_id = gr.Textbox(label="Annotation ID", placeholder="Copy from ID column above", scale=3)
                    delete_annotation_btn = gr.Button("Delete Note", variant="stop", size="sm", scale=1)

            with gr.Accordion("Document Card Inspector — structured intelligence for a single source", open=False):
                gr.Markdown(
                    "Select a document to view its intelligence card: type classification, "
                    "source side, authorship, operative status, privilege flags, rhetorical "
                    "posture, reliability, and unresolved issues."
                )
                doc_card_selector = gr.Dropdown(
                    label="Select document",
                    choices=[],
                    interactive=True,
                    allow_custom_value=True,
                )
                with gr.Row():
                    load_doc_card_btn = gr.Button(
                        "Load Card", variant="primary", size="sm", scale=1,
                    )
                    refresh_doc_card_choices_btn = gr.Button(
                        "Refresh document list", variant="secondary", size="sm", scale=1,
                    )
                document_card_html = gr.HTML(
                    "<div class='viz-empty'>Select a document to view its card.</div>"
                )

                with gr.Accordion("Correct Card Classification — fix type, role, or status", open=False):
                    gr.Markdown(
                        "If the AI-profiled classification is wrong, correct it here. "
                        "Changes to type, role, or status will mark downstream "
                        "assertions and evidence as stale for re-evaluation."
                    )
                    with gr.Row():
                        card_edit_doc_type = gr.Textbox(
                            label="Document type",
                            placeholder="e.g. contract, memo, filing, report",
                            scale=1,
                        )
                        card_edit_source_role = gr.Textbox(
                            label="Source role",
                            placeholder="e.g. operative, advocacy, informal",
                            scale=1,
                        )
                    with gr.Row():
                        card_edit_operative = gr.Textbox(
                            label="Operative status",
                            placeholder="e.g. operative, draft, superseded",
                            scale=1,
                        )
                        card_edit_privilege = gr.Checkbox(
                            label="Privileged / restricted",
                            value=False,
                            scale=1,
                        )
                    card_edit_flags = gr.Textbox(
                        label="Unresolved flags (comma-separated)",
                        placeholder="e.g. missing_exhibit, date_conflict",
                    )
                    card_edit_btn = gr.Button(
                        "Apply Corrections", variant="primary", size="sm",
                    )
                    card_edit_result = gr.Markdown("")

            with gr.Accordion("Document Review Console — deep-dive into a single source", open=False):
                gr.Markdown(
                    "Select a document to see its full profile: type, privilege status, "
                    "linked issues, actors, and every candidate fact ready for review."
                )
                doc_console_selector = gr.Dropdown(
                    label="Select document",
                    choices=[],
                    interactive=True,
                    allow_custom_value=True,
                )
                refresh_doc_console_choices_btn = gr.Button(
                    "Refresh document list", variant="secondary", size="sm",
                )
                doc_console_html = gr.HTML(
                    "<div class='viz-empty'>Select a document to review.</div>"
                )
                load_doc_console_btn = gr.Button(
                    "Load Document Console", variant="primary", size="sm",
                )

        with gr.Accordion("Belief Revisions — how the system's understanding has changed over time", open=False):
            gr.Markdown(
                "Every time an assertion's belief state or confidence changes, the revision "
                "is logged here. Shows the full audit trail of truth maintenance."
            )
            belief_revision_html = gr.HTML("<div class='viz-empty'>Belief revisions will appear here after an investigation.</div>")
            refresh_belief_btn = gr.Button("Refresh Belief Revisions", variant="secondary", size="sm")

        with gr.Accordion("Contradiction Analysis — conflicting assertions and open disputes", open=False):
            gr.Markdown(
                "Active contradiction pairs in the assertion graph. Shows which facts challenge "
                "each other, their belief states, and whether the conflict is resolved or open."
            )
            contradiction_html = gr.HTML("<div class='viz-empty'>Contradiction analysis will appear here after an investigation.</div>")
            with gr.Row():
                refresh_contradiction_btn = gr.Button("Refresh Contradictions", variant="secondary", size="sm")
                mine_contradiction_btn = gr.Button("Run Contradiction Mining", variant="primary", size="sm")
            with gr.Accordion("Resolve a contradiction", open=False):
                gr.Markdown(
                    "Paste the attacker and attacked assertion IDs from a conflict pair above, "
                    "choose how to resolve the disagreement, and provide your reasoning."
                )
                with gr.Row():
                    resolve_attacker_id = gr.Textbox(
                        label="Attacker assertion ID", placeholder="Paste attacker ID", scale=2,
                    )
                    resolve_attacked_id = gr.Textbox(
                        label="Attacked assertion ID", placeholder="Paste attacked ID", scale=2,
                    )
                with gr.Row():
                    resolve_decision = gr.Dropdown(
                        label="Resolution decision",
                        choices=[
                            ("Prefer attacker — supersede attacked", "prefer_attacker"),
                            ("Prefer attacked — supersede attacker", "prefer_attacked"),
                            ("Mark both as disputed", "mark_both_disputed"),
                            ("Request more evidence", "request_evidence"),
                        ],
                        interactive=True, scale=2,
                    )
                    resolve_rationale = gr.Textbox(
                        label="Rationale", placeholder="Why this decision?", scale=3,
                    )
                    resolve_btn = gr.Button("Resolve", variant="primary", size="sm", scale=1)
                resolve_result = gr.Markdown("")

        with gr.Accordion("Document Version Chains — which documents supersede each other", open=False):
            gr.Markdown(
                "Groups documents that are versions of each other (e.g. contract_v1.pdf → contract_v2.pdf). "
                "The operative (current) version is highlighted so you know which document to cite."
            )
            doc_versions_html = gr.HTML("<div class='viz-empty'>Document version chains will appear here after an investigation.</div>")
            with gr.Row():
                refresh_doc_versions_btn = gr.Button("Refresh Document Versions", variant="secondary", size="sm")
                detect_versions_btn = gr.Button("Detect Version Chains", variant="primary", size="sm")
            with gr.Accordion("Look up operative version for a document", open=False):
                with gr.Row():
                    operative_doc_id_input = gr.Textbox(
                        label="Document ID", placeholder="Enter a document ID...", scale=3,
                    )
                    operative_lookup_btn = gr.Button(
                        "Find Operative Version", variant="secondary", size="sm", scale=1,
                    )
                operative_version_html = gr.HTML(
                    "<div class='viz-empty'>Enter a document ID to find its operative version.</div>"
                )

        with gr.Accordion("Document Triage — documents awaiting profiling", open=False):
            gr.Markdown(
                "Shows documents that have been ingested but not yet fully profiled. "
                "Higher-salience documents are listed first. Profiling enriches the matter "
                "model with document-level intelligence that improves reasoning quality."
            )
            doc_triage_html = gr.HTML("<div class='viz-empty'>Document triage queue will appear here after an investigation.</div>")
            refresh_doc_triage_btn = gr.Button("Refresh Document Triage", variant="secondary", size="sm")

        with gr.Accordion("Financial Health Alerts — quantitative threshold violations", open=False):
            gr.Markdown(
                "Detects financial exposure, disputed amount fractions, and numeric conflicts. "
                "Each alert indicates a threshold breach that requires attention in the analysis."
            )
            with gr.Accordion("Configure thresholds", open=False):
                with gr.Row():
                    exposure_high_input = gr.Number(
                        label="Exposure HIGH threshold",
                        value=10000.0, minimum=0, step=1000,
                        info="Amounts above this are flagged HIGH severity",
                        scale=1,
                    )
                    disputed_fraction_input = gr.Number(
                        label="Disputed fraction threshold",
                        value=0.10, minimum=0, maximum=1.0, step=0.01,
                        info="Fraction of invoiced amount that triggers an alert",
                        scale=1,
                    )
            quant_thresholds_html = gr.HTML("<div class='viz-empty'>Financial health alerts will appear here after an investigation.</div>")
            refresh_quant_thresholds_btn = gr.Button("Refresh Financial Health", variant="secondary", size="sm")

        with gr.Accordion("System Health — truth maintenance diagnostics", open=False):
            gr.Markdown(
                "Monitors assertion stability: dispute rate, belief oscillation, open gaps, "
                "and contradictions. A healthy system has low dispute rates and zero oscillating assertions."
            )
            system_health_html = gr.HTML("<div class='viz-empty'>System health diagnostics will appear here after an investigation.</div>")
            with gr.Row():
                refresh_system_health_btn = gr.Button("Refresh System Health", variant="secondary", size="sm")
                flush_pending_btn = gr.Button("Flush Pending Propagation", variant="primary", size="sm")

        with gr.Accordion("SO Scorecard — Sacred Outcome metrics vs targets", open=False):
            gr.Markdown(
                "Shows each Sacred Outcome metric with its current value, target threshold, "
                "and pass/fail status. Tracks analysis quality across all key dimensions."
            )
            so_scorecard_html = gr.HTML("<div class='viz-empty'>SO scorecard will appear here after an investigation.</div>")
            refresh_so_scorecard_btn = gr.Button("Refresh SO Scorecard", variant="secondary", size="sm")

        with gr.Accordion("Objective Coverage — what are we proving and how far along", open=False):
            gr.Markdown(
                "Each objective represents something Irys is trying to prove, decide, or assess. "
                "Coverage shows how well the evidence supports each objective, which criteria are "
                "satisfied, and where gaps remain. This is the central map of analysis progress."
            )
            objective_coverage_html = gr.HTML("<div class='viz-empty'>Objective coverage will appear here after an investigation.</div>")
            refresh_objective_coverage_btn = gr.Button("Refresh Coverage", variant="secondary", size="sm")

            with gr.Accordion("Manage criteria", open=False):
                with gr.Row():
                    criterion_predicate_id = gr.Textbox(label="Criterion ID", placeholder="Paste the criterion/predicate ID", scale=2)
                    criterion_status_dropdown = gr.Dropdown(
                        choices=[
                            ("Open", "open"),
                            ("Resolved", "resolved"),
                            ("Contested", "contested"),
                            ("Blocked", "blocked"),
                        ],
                        label="Status",
                        value="open",
                        scale=1,
                    )
                criterion_reason_input = gr.Textbox(label="Reason (optional)", placeholder="Why this status change?")
                set_criterion_btn = gr.Button("Set Criterion Status", variant="primary", size="sm")
                criterion_result = gr.Textbox(label="Result", interactive=False, visible=True)

                gr.Markdown("---")
                with gr.Row():
                    add_criterion_objective_id = gr.Textbox(label="Objective ID", placeholder="Paste the objective/issue ID", scale=2)
                    add_criterion_burden = gr.Textbox(label="Burden side (optional)", placeholder="e.g. plaintiff, defendant", scale=1)
                add_criterion_desc = gr.Textbox(label="Criterion description", placeholder="What must be proven?")
                add_criterion_btn = gr.Button("Add Criterion", variant="primary", size="sm")
                add_criterion_result = gr.Textbox(label="Result", interactive=False, visible=True)

        with gr.Accordion("Output Quality Contract — is this analysis reliance-ready?", open=False):
            gr.Markdown(
                "Aggregated quality assessment: which professional obligations are satisfied, "
                "recent investigation run summaries, dependency manifest freshness, and any "
                "remaining blockers that must be resolved before the output can be relied upon."
            )
            output_quality_html = gr.HTML(
                "<div class='viz-empty'>Quality contract will appear after an investigation.</div>"
            )
            refresh_output_quality_btn = gr.Button(
                "Refresh Quality Contract", variant="secondary", size="sm",
            )

        with gr.Accordion("Deliverable Builder — assemble work product from verified evidence", open=False):
            gr.Markdown(
                "Preview what a professional deliverable would contain: verified assertions "
                "for each issue, cited source documents, and the current reliance gate status. "
                "Issues with no verified evidence are flagged so you know what to resolve first."
            )
            deliverable_workbench_html = gr.HTML(
                "<div class='viz-empty'>Deliverable preview will appear after an investigation.</div>"
            )
            refresh_deliverable_btn = gr.Button(
                "Refresh Deliverable Preview", variant="secondary", size="sm",
            )

        with gr.Accordion("Issue Brief — compiled analysis with provenance and gap disclosure", open=False):
            gr.Markdown(
                "A structured brief compiled from the matter model showing all assertions "
                "per issue with belief state, source roles, proof gaps, contradictions, "
                "and source document citations. Suitable for review and export."
            )
            issue_brief_html = gr.HTML(
                "<div class='viz-empty'>Issue brief will appear after an investigation.</div>"
            )
            refresh_brief_btn = gr.Button(
                "Compile Issue Brief", variant="primary", size="sm",
            )

        with gr.Accordion("Scenario Branches — persistent what-if counterfactual analysis", open=False):
            gr.Markdown(
                "Create named scenario branches to explore alternative interpretations. "
                "Each branch records its assumptions and can be compared against the baseline. "
                "Branches persist across sessions for ongoing analysis."
            )
            scenario_workbench_html = gr.HTML(
                "<div class='viz-empty'>Scenario branches will appear here.</div>"
            )
            with gr.Row():
                scenario_name_input = gr.Textbox(
                    label="Branch Name", placeholder="e.g. Contract is void",
                    scale=2,
                )
                scenario_notes_input = gr.Textbox(
                    label="Notes (optional)", placeholder="Why this scenario matters",
                    scale=2,
                )
            scenario_assumptions_input = gr.Textbox(
                label="Assumptions (one per line)",
                placeholder="The contract was signed under duress\nThe statute of limitations has run",
                lines=3,
            )
            with gr.Row():
                scenario_create_btn = gr.Button(
                    "Create Branch", variant="primary", size="sm",
                )
                refresh_scenario_btn = gr.Button(
                    "Refresh Branches", variant="secondary", size="sm",
                )
            scenario_create_result = gr.Textbox(
                label="Result", interactive=False, visible=True,
            )
            with gr.Accordion("Compare branch to baseline", open=False):
                with gr.Row():
                    scenario_compare_branch_id = gr.Textbox(
                        label="Branch ID (copy from branch list above)", scale=3,
                    )
                    scenario_compare_btn = gr.Button(
                        "Compare to Baseline", variant="primary", size="sm", scale=1,
                    )
                scenario_comparison_html = gr.HTML(
                    "<div class='viz-empty'>Select a branch and click Compare to see the impact analysis.</div>"
                )
            with gr.Accordion("Snapshot history — chronological evaluation log", open=False):
                with gr.Row():
                    snapshot_history_branch_id = gr.Textbox(
                        label="Branch ID", placeholder="Enter branch ID...", scale=3,
                    )
                    load_snapshots_btn = gr.Button(
                        "Load Snapshot History", variant="secondary", size="sm", scale=1,
                    )
                snapshot_history_html = gr.HTML(
                    "<div class='viz-empty'>Enter a branch ID and click Load to see snapshot history.</div>"
                )
            with gr.Accordion("Delta log — changes applied to a branch", open=False):
                with gr.Row():
                    delta_log_branch_id = gr.Textbox(
                        label="Branch ID", placeholder="Enter branch ID...", scale=3,
                    )
                    load_deltas_btn = gr.Button(
                        "Load Deltas", variant="secondary", size="sm", scale=1,
                    )
                delta_log_html = gr.HTML(
                    "<div class='viz-empty'>Enter a branch ID and click Load to see applied deltas.</div>"
                )
            with gr.Accordion("Apply delta — add a what-if change to a branch", open=False):
                with gr.Row():
                    delta_branch_id = gr.Textbox(
                        label="Branch ID", placeholder="Enter branch ID...", scale=2,
                    )
                    delta_target_kind = gr.Dropdown(
                        choices=["assertion", "issue", "predicate", "quant_fact",
                                 "authority", "gap"],
                        label="Target Kind", scale=1,
                    )
                with gr.Row():
                    delta_target_id = gr.Textbox(
                        label="Target ID", placeholder="ID of target to modify", scale=2,
                    )
                    delta_operation = gr.Dropdown(
                        choices=["override_belief", "suppress", "add_gap",
                                 "resolve_gap", "add_assertion", "assume"],
                        label="Operation", scale=1,
                    )
                delta_payload = gr.Textbox(
                    label="Payload (JSON, optional)",
                    placeholder='e.g. {"new_belief": "accepted"} or {"description": "missing receipts"}',
                    lines=2,
                )
                apply_delta_btn = gr.Button("Apply Delta", variant="primary", size="sm")
                apply_delta_result = gr.Textbox(label="Result", interactive=False)
            with gr.Accordion("Branch actions — compute snapshot or archive", open=False):
                with gr.Row():
                    action_branch_id = gr.Textbox(
                        label="Branch ID", placeholder="Enter branch ID...", scale=3,
                    )
                    compute_snapshot_btn = gr.Button(
                        "Compute Snapshot", variant="primary", size="sm", scale=1,
                    )
                    archive_branch_btn = gr.Button(
                        "Archive Branch", variant="stop", size="sm", scale=1,
                    )
                branch_action_result = gr.Textbox(label="Result", interactive=False)

        with gr.Accordion("Alternative Theories — competing interpretations from the matter graph", open=False):
            gr.Markdown(
                "Irys derives the strongest competing interpretations from the current matter model. "
                "Each theory shows its supporting and attacking assertions, key assumptions, open gaps, "
                "and discriminator questions that would help distinguish between theories."
            )
            alt_theories_html = gr.HTML(
                "<div class='viz-empty'>Alternative theories will appear here after an investigation.</div>"
            )
            refresh_alt_theories_btn = gr.Button(
                "Refresh Theories", variant="secondary", size="sm",
            )

        with gr.Accordion("Dependency Manifests — exact evidence and freshness behind each output", open=False):
            gr.Markdown(
                "Every analysis output is backed by a dependency manifest that tracks "
                "exactly which objects were consumed, which namespaces are stale, "
                "and whether taint or policy constraints apply."
            )
            manifest_inspector_html = gr.HTML(
                "<div class='viz-empty'>Dependency manifests will appear here after an investigation.</div>"
            )
            refresh_manifest_btn = gr.Button(
                "Refresh Manifests", variant="secondary", size="sm",
            )

        with gr.Accordion("Namespace Freshness — revision state of every intelligence namespace", open=False):
            gr.Markdown(
                "Shows which namespaces have been populated and their revision numbers. "
                "A hot-answerable matter has all namespaces fresh with no active investigations."
            )
            freshness_report_html = gr.HTML(
                "<div class='viz-empty'>Freshness report will appear here after loading a matter.</div>"
            )
            refresh_freshness_btn = gr.Button(
                "Refresh Freshness", variant="secondary", size="sm",
            )

        with gr.Accordion("Reasoning Cache — hit rates by stage showing reuse of prior analysis", open=False):
            gr.Markdown(
                "Shows how effectively the reasoning cache is being reused across stages. "
                "Higher hit rates mean faster investigations and less redundant computation."
            )
            cache_stats_html = gr.HTML(
                "<div class='viz-empty'>Cache stats will appear here after loading a matter.</div>"
            )
            refresh_cache_stats_btn = gr.Button(
                "Refresh Cache Stats", variant="secondary", size="sm",
            )

        with gr.Accordion("LLM Usage — token consumption and cost across reasoning tiers", open=False):
            gr.Markdown(
                "Aggregated token counts, costs, and per-tier breakdowns for all LLM calls "
                "in this matter. Useful for monitoring compute spend and identifying optimization targets."
            )
            llm_usage_html = gr.HTML(
                "<div class='viz-empty'>LLM usage will appear here after loading a matter.</div>"
            )
            refresh_llm_usage_btn = gr.Button(
                "Refresh LLM Usage", variant="secondary", size="sm",
            )

        with gr.Accordion("Steering Impact Preview — project the effect of corrections before committing", open=False):
            gr.Markdown(
                "Select a steering action type, provide the relevant payload as JSON, "
                "and preview the projected impact on coverage, gaps, contradictions, "
                "and readiness before committing the change."
            )
            with gr.Row():
                impact_action_type = gr.Dropdown(
                    choices=["resolve_gap", "correct_assertion", "resolve_contradiction",
                             "approve_metric_alias", "escalate_gap"],
                    label="Action Type",
                    value="resolve_gap",
                )
                impact_payload = gr.Textbox(
                    label="Payload (JSON)",
                    placeholder='{"gap_id": "..."}',
                    lines=2,
                )
            impact_preview_btn = gr.Button(
                "Preview Impact", variant="primary", size="sm",
            )
            impact_preview_html = gr.HTML(
                "<div class='viz-empty'>Select a steering action and click Preview to see projected impact.</div>"
            )

        with gr.Accordion("Domain Readiness — cross-domain investigation acceptance gate", open=False):
            gr.Markdown(
                "Validates whether the matter model substrate works across all five "
                "domain profiles: legal, finance, coding, academic research, biomedical. "
                "Checks assertion quality, source calibration, coverage, quantitative "
                "extraction, gap modeling, and steering readiness per profile."
            )
            domain_readiness_html = gr.HTML(
                "<div class='viz-empty'>Domain readiness will appear here after an investigation.</div>"
            )
            refresh_readiness_btn = gr.Button(
                "Refresh Domain Readiness", variant="secondary", size="sm",
            )

        with gr.Accordion("Answer Audit — freshness, sources, and policy for every answer", open=False):
            gr.Markdown(
                "Every answer Irys produces is backed by a dependency manifest that tracks "
                "exactly which evidence was consumed, under which domain profile, and whether "
                "anything has changed since the answer was generated. Stale answers are flagged."
            )
            answer_audit_html = gr.HTML("<div class='viz-empty'>Answer audit trail will appear here after an investigation.</div>")
            refresh_answer_audit_btn = gr.Button("Refresh Answer Audit", variant="secondary", size="sm")

        with gr.Accordion("Knowledge Reuse — cross-matter intelligence seeds", open=False):
            gr.Markdown(
                "When Irys resolves contradictions, approves metric classifications, or calibrates "
                "domain weights, those decisions can be promoted as reusable seeds. New matters "
                "inherit approved seeds, accelerating analysis and reducing repeated reasoning."
            )
            knowledge_seeds_html = gr.HTML("<div class='viz-empty'>Knowledge seeds will appear here after an investigation.</div>")
            refresh_knowledge_seeds_btn = gr.Button("Refresh Seeds", variant="secondary", size="sm")

            with gr.Accordion("Review a seed", open=False):
                seed_id_input = gr.Textbox(label="Seed ID", placeholder="Paste the seed ID to review")
                seed_decision_input = gr.Dropdown(
                    choices=[
                        ("Accept for this matter", "matter_local"),
                        ("Keep as promotable", "promotable"),
                        ("Reject", "rejected"),
                    ],
                    label="Decision",
                    value="matter_local",
                )
                seed_review_note_input = gr.Textbox(label="Review note (optional)", placeholder="Why this decision?")
                seed_review_btn = gr.Button("Submit Review", variant="primary", size="sm")
                seed_review_status = gr.Textbox(label="Status", interactive=False, visible=True)

            with gr.Accordion("Promote intelligence as a reusable seed", open=False):
                with gr.Row():
                    promote_seed_kind = gr.Dropdown(
                        choices=[
                            ("Contradiction resolution", "contradiction_resolution"),
                            ("Metric classification", "metric_classification"),
                            ("Domain weight", "domain_weight"),
                            ("Source calibration", "source_calibration"),
                            ("Custom", "custom"),
                        ],
                        label="Seed Kind", scale=2,
                    )
                    promote_domain_profile = gr.Textbox(
                        label="Domain Profile ID", placeholder="e.g. legal:1",
                        scale=1,
                    )
                promote_payload = gr.Textbox(
                    label="Payload (JSON)", placeholder='{"key": "value"}',
                    lines=2,
                )
                promote_seed_btn = gr.Button("Promote Seed", variant="primary", size="sm")
                promote_seed_result = gr.Textbox(label="Result", interactive=False)

        with gr.Accordion("Domain Profile — how Irys interprets this subject area", open=False):
            gr.Markdown(
                "Shows the active domain profile: which vocabulary maps concepts to domain terms, "
                "how sources are weighted for trustworthiness, and what sensitivity classes apply. "
                "Multi-domain matters blend profiles by detection confidence."
            )
            domain_profile_html = gr.HTML("<div class='viz-empty'>Domain profile will appear here after an investigation.</div>")
            refresh_domain_profile_btn = gr.Button("Refresh Domain Profile", variant="secondary", size="sm")

            with gr.Accordion("Domain Composition — how domain detection blends profiles", open=False):
                gr.Markdown(
                    "Shows which domain facets are active, how trust weights are composed "
                    "across detected domains, and recent detection events that shaped the profile."
                )
                domain_composition_html = gr.HTML("<div class='viz-empty'>Domain composition will appear here after an investigation.</div>")
                refresh_domain_composition_btn = gr.Button("Refresh Composition", variant="secondary", size="sm")

        with gr.Accordion("Content Policy Audit — what Irys allowed, blocked, or withheld", open=False):
            gr.Markdown(
                "Every time Irys decides whether to include or redact content for a given "
                "audience mode, the decision is logged here. Use this to audit privilege "
                "redaction behavior and verify that sensitive material is handled correctly."
            )
            content_policy_html = gr.HTML("<div class='viz-empty'>Content policy audit will appear here after an investigation.</div>")
            refresh_content_policy_btn = gr.Button("Refresh Content Policy Audit", variant="secondary", size="sm")

        with gr.Accordion("Sensitivity & Taint — what content is restricted or flagged", open=False):
            gr.Markdown(
                "Shows how content sensitivity is tracked across the matter model. Each object "
                "(document, assertion, entity) can carry taint classifications that control what "
                "can be disclosed in different audience modes."
            )
            taint_summary_html = gr.HTML("<div class='viz-empty'>Taint summary will appear here after an investigation.</div>")
            refresh_taint_btn = gr.Button("Refresh Taint Summary", variant="secondary", size="sm")

            with gr.Accordion("Sensitivity Reclassification — change privilege or staleness flags", open=False):
                gr.Markdown(
                    "Reclassify a document's sensitivity flag or mark a document/span as stale. "
                    "Reclassification stales all downstream dependents (assertions, evidence edges, "
                    "quantitative facts) so they are re-evaluated under the new classification."
                )
                sensitivity_result_html = gr.HTML("<div class='viz-empty'>Results will appear here after an action.</div>")
                with gr.Row():
                    sensitivity_doc_id_input = gr.Textbox(
                        label="Document ID", placeholder="Enter document ID...",
                        scale=3,
                    )
                    sensitivity_flag_checkbox = gr.Checkbox(
                        label="Mark as privileged/restricted", value=False,
                    )
                    reclassify_btn = gr.Button("Reclassify Sensitivity", variant="primary", size="sm", scale=1)
                with gr.Row():
                    stale_doc_id_input = gr.Textbox(
                        label="Document ID (stale)", placeholder="Enter document ID to mark stale...",
                        scale=2,
                    )
                    stale_reason_input = gr.Textbox(
                        label="Reason", placeholder="e.g. superseded by v2",
                        scale=2,
                    )
                    mark_doc_stale_btn = gr.Button("Mark Document Stale", variant="secondary", size="sm", scale=1)
                with gr.Row():
                    stale_span_id_input = gr.Textbox(
                        label="Span ID (stale)", placeholder="Enter span ID to mark stale...",
                        scale=2,
                    )
                    span_stale_reason_input = gr.Textbox(
                        label="Reason", placeholder="e.g. clause amended",
                        scale=2,
                    )
                    mark_span_stale_btn = gr.Button("Mark Span Stale", variant="secondary", size="sm", scale=1)

        with gr.Accordion("Communication Graph — who appears in which documents", open=False):
            gr.Markdown(
                "Maps which people and companies appear in which documents and highlights "
                "repeated pairings (e.g. frequent correspondents)."
            )
            communication_html = gr.HTML("<div class='viz-empty'>Communication graph will appear here after an investigation.</div>")
            refresh_comm_btn = gr.Button("Refresh Communication Graph", variant="secondary", size="sm")

        with gr.Accordion("Actor Resolution — detect and merge duplicate entities", open=False):
            gr.Markdown(
                "Finds actors whose names overlap (e.g. 'Acme Inc' vs. 'Acme Corporation'). "
                "Merging consolidates all references so evidence is not split across duplicates."
            )
            actor_duplicates_html = gr.HTML("<div class='viz-empty'>Scan for duplicates to see results.</div>")
            with gr.Row():
                scan_duplicates_btn = gr.Button("Scan for Duplicates", variant="primary", size="sm")
                refresh_duplicates_btn = gr.Button("Refresh", variant="secondary", size="sm")
            with gr.Row():
                merge_keep_id = gr.Textbox(label="Keep actor ID", placeholder="ID of the actor to keep")
                merge_discard_id = gr.Textbox(label="Merge (discard) actor ID", placeholder="ID of the actor to merge away")
            merge_confirm = gr.Checkbox(label="I understand this permanently merges these actors", value=False)
            merge_actors_btn = gr.Button("Merge Actors", variant="stop", size="sm")
            merge_result = gr.Markdown("")
            gr.Markdown("#### Look up actor by name")
            with gr.Row():
                resolve_actor_name = gr.Textbox(label="Actor name", placeholder="e.g. Acme Inc", scale=3)
                resolve_actor_btn = gr.Button("Resolve", variant="secondary", size="sm", scale=1)
            resolve_actor_result = gr.Markdown("")

        with gr.Accordion("LLM Analytics — cost, latency, and stage mix", open=False):
            gr.Markdown(
                "Shows recent LLM calls, spend by stage, and the current tier mix."
            )
            llm_analytics_html = gr.HTML("<div class='viz-empty'>LLM analytics will appear here after an investigation.</div>")
            refresh_llm_btn = gr.Button("Refresh LLM Analytics", variant="secondary", size="sm")

        with gr.Accordion("Investigation History — past runs and their outcomes", open=False):
            gr.Markdown(
                "Shows every investigation run recorded on this matter: what was queried, "
                "how much it cost, how many LLM calls were made vs cached, and the "
                "matter model reuse rate. Higher reuse means the durable matter model "
                "is paying off across runs."
            )
            investigation_history_html = gr.HTML("<div class='viz-empty'>Investigation history will appear here after loading a matter.</div>")
            refresh_history_btn = gr.Button("Refresh Investigation History", variant="secondary", size="sm")

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
            gr.Markdown("---")
            gr.Markdown("#### Answer a Clarification")
            gr.Markdown(
                "The system may generate clarification questions when it encounters gaps "
                "in the available evidence. Select a pending question and provide your answer."
            )
            with gr.Row():
                clarification_dropdown = gr.Dropdown(
                    label="Pending Clarification",
                    choices=[],
                    interactive=True,
                )
                clarification_answer_input = gr.Textbox(
                    label="Your Answer",
                    placeholder="Type your answer to the selected clarification question...",
                    lines=2,
                )
            clarification_context_html = gr.HTML("")
            with gr.Row():
                answer_clarification_btn = gr.Button("Submit Answer", variant="primary", size="sm")
                generate_clarifications_btn = gr.Button("Generate Questions from Gaps", variant="secondary", size="sm")
            answer_clarification_result = gr.Textbox(label="Result", interactive=False)
            gr.Markdown("---")
            gr.Markdown("#### Document Trust Overrides")
            gr.Markdown(
                "Override how much the system trusts a specific document. "
                "**Low** demotes all assertions from that document to alleged status. "
                "**High** promotes them to operative. **Normal** resets to automatic calibration."
            )
            trust_overrides_html = gr.HTML("<div class='viz-empty'>No trust overrides set.</div>")
            with gr.Row():
                trust_doc_pattern = gr.Textbox(
                    label="Document Pattern",
                    placeholder="e.g. contract_v1.pdf",
                )
                trust_level_dropdown = gr.Dropdown(
                    label="Trust Level",
                    choices=[("Low — demote to alleged", "low"), ("Normal — reset to auto", "normal"), ("High — promote to operative", "high")],
                    value="low",
                )
            trust_note = gr.Textbox(
                label="Reason (optional)",
                placeholder="Why are you overriding trust for this document?",
            )
            with gr.Row():
                set_trust_btn = gr.Button("Set Trust Override", variant="primary", size="sm")
                refresh_trust_btn = gr.Button("Refresh Overrides", variant="secondary", size="sm")
            trust_override_result = gr.Textbox(label="Result", interactive=False)
            gr.Markdown("#### Remove a trust override")
            with gr.Row():
                delete_trust_pattern = gr.Textbox(
                    label="Document pattern to remove",
                    placeholder="e.g. contract.pdf (must match exactly)",
                    scale=3,
                )
                delete_trust_btn = gr.Button("Delete Override", variant="stop", size="sm", scale=1)
            gr.Markdown("---")
            gr.Markdown("#### Decision Context — who is the decision-maker and what are they trying to do?")
            gr.Markdown(
                "Setting a decision context adjusts how Irys frames its synthesis and recommendations. "
                "The record model and assertions are not changed — only the presentation layer adapts."
            )
            decision_context_html = gr.HTML("<div class='viz-empty'>No decision context set.</div>")
            with gr.Row():
                dc_maker_type = gr.Dropdown(
                    label="Decision-maker role",
                    choices=[
                        ("Judge", "judge"), ("Partner", "partner"), ("Client", "client"),
                        ("Mediator", "mediator"), ("Arbitrator", "arbitrator"),
                        ("Regulator", "regulator"), ("Other", "unknown"),
                    ],
                    value=None,
                    scale=1,
                )
                dc_objective = gr.Dropdown(
                    label="Objective",
                    choices=[
                        ("Motion practice", "motion_practice"), ("Settlement", "settlement"),
                        ("Due diligence", "diligence"), ("Audit", "audit"),
                        ("Advisory", "advisory"), ("Trial prep", "trial_prep"),
                        ("Regulatory response", "regulatory_response"),
                        ("Transactional", "transactional"), ("Other", "unknown"),
                    ],
                    value=None,
                    scale=1,
                )
            dc_name = gr.Textbox(label="Decision-maker name (optional)", placeholder="e.g. Judge Martinez")
            dc_notes = gr.Textbox(label="Strategic notes (optional)", placeholder="Focus on damages claims under §10.2", lines=2)
            dc_narrow = gr.Checkbox(label="Narrow scope — only surface issues directly relevant to the objective", value=False)
            with gr.Row():
                set_dc_btn = gr.Button("Set Context", variant="primary", size="sm")
                clear_dc_btn = gr.Button("Clear Context", variant="secondary", size="sm")
                refresh_dc_btn = gr.Button("Refresh", variant="secondary", size="sm")
            dc_result = gr.Markdown("")

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
            except Exception as exc:
                logger.warning("Folder picker failed: %s", exc)
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
                    logger.warning("_on_add_files: %s", e)
                    return gr.update(), _fmt_ws_status(f"Upload failed: {e}", "err"), gr.update(), gr.update()

            add_files_btn.click(
                fn=_on_add_files,
                inputs=[file_upload, repo_path],
                outputs=[matter_files_dropdown, file_manage_status, file_upload, matter_card_html],
            )

            def _on_add_folder(files, relpaths_json, matter_name):
                if not matter_name:
                    return gr.update(), _fmt_ws_status("Select a matter first", "err"), gr.update(), gr.update(), ""
                if not files:
                    return gr.update(), _fmt_ws_status("No folder selected", "info"), gr.update(), gr.update(), ""
                try:
                    _, status_msg = _upload_files_to_s3_matter(
                        files, matter_name.strip(), relpath_json=relpaths_json
                    )
                    updated = _list_s3_matter_files(matter_name.strip())
                    tone = "warn" if "failed" in status_msg.lower() else "ok"
                    return (
                        gr.update(choices=updated, value=None),
                        _fmt_ws_status(status_msg, tone),
                        None,
                        _fmt_matter_card(matter_name, updated),
                        "",
                    )
                except Exception as e:
                    logger.warning("_on_add_folder: %s", e)
                    return gr.update(), _fmt_ws_status(f"Upload failed: {e}", "err"), gr.update(), gr.update(), ""

            add_folder_btn.click(
                fn=_on_add_folder,
                inputs=[folder_upload, folder_upload_relpaths, repo_path],
                outputs=[matter_files_dropdown, file_manage_status, folder_upload, matter_card_html, folder_upload_relpaths],
            )

            # Update placeholder when domain changes
            def _on_domain_change(domain_val):
                preset = _DOMAIN_CREATE_PRESETS.get(domain_val, _DOMAIN_CREATE_PRESETS["legal"])
                return gr.update(placeholder=preset["placeholder"])

            domain_selector.change(
                fn=_on_domain_change,
                inputs=[domain_selector],
                outputs=[matter_name_input],
            )

            # Create a new matter (with optional initial files/folder), then select it
            def _on_save_matter(files, folder_files, folder_relpaths_json, name, domain_val):
                if not name or not name.strip():
                    return gr.update(), "", _fmt_ws_status("Enter a matter name first", "err"), gr.update(visible=False), gr.update(choices=[], value=None), gr.update(), ""
                try:
                    all_files = (files or []) + (folder_files or [])
                    status_msg = ""
                    if all_files:
                        display_name, status_msg = _upload_files_to_s3_matter(
                            all_files, name.strip(), relpath_json=folder_relpaths_json
                        )
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
                        "",
                    )
                except Exception as e:
                    logger.warning("_on_save_matter: %s", e)
                    return gr.update(), "", _fmt_ws_status(f"Create failed: {e}", "err"), gr.update(visible=False), gr.update(choices=[], value=None), gr.update(), ""

            save_matter_btn.click(
                fn=_on_save_matter,
                inputs=[file_upload_new, folder_upload_new, folder_upload_new_relpaths, matter_name_input, domain_selector],
                outputs=[matter_dropdown, repo_path, upload_status, matter_workspace, matter_files_dropdown, matter_card_html, folder_upload_new_relpaths],
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
            from concurrent.futures import ThreadPoolExecutor

            domain = state._detect_domain(mid)

            with ThreadPoolExecutor(max_workers=6) as pool:
                f_overview = pool.submit(state.load_overview, mid, domain)
                f_issues = pool.submit(state.load_issues, mid, domain)
                f_gaps = pool.submit(state.load_gaps, mid)
                f_assumptions = pool.submit(state.load_assumptions, mid, domain)
                f_assertions = pool.submit(state.load_assertions, mid)
                f_quant = pool.submit(state.load_quant, mid, domain)
                f_timeline = pool.submit(state.load_timeline, mid, domain)
                f_evidence = pool.submit(state.load_evidence_matrix, mid, domain=domain)
                f_communication = pool.submit(state.load_communication_map, mid, domain)
                f_llm = pool.submit(state.load_llm_analytics, mid)
                f_proof = pool.submit(state.load_proof_state, mid, domain=domain)
                f_authority = pool.submit(state.load_authority_network, mid, domain=domain)
                f_doc_intel = pool.submit(state.load_document_intelligence, mid, domain=domain)
                f_belief = pool.submit(state.load_belief_revisions, mid, domain=domain)
                f_contradictions = pool.submit(state.load_contradictions, mid, domain=domain)
                f_doc_versions = pool.submit(state.load_document_versions, mid, domain=domain)
                f_quant_thresh = pool.submit(state.load_quant_thresholds, mid, domain=domain)
                f_sys_health = pool.submit(state.load_system_health, mid, domain=domain)
                f_so_scorecard = pool.submit(state.load_so_scorecard, mid, domain=domain)
                f_review = pool.submit(state.load_review_count_badge, mid, domain)
                f_docs = pool.submit(state.load_document_picker_choices, mid)

            gaps_text, top_issue = f_gaps.result()
            return (
                f_review.result(),
                f_overview.result(),
                f_issues.result(),
                gaps_text,
                f_assumptions.result(),
                f_assertions.result(),
                f_quant.result(),
                f_timeline.result(),
                f_evidence.result(),
                f_communication.result(),
                f_llm.result(),
                f_proof.result(),
                f_authority.result(),
                f_doc_intel.result(),
                f_belief.result(),
                f_contradictions.result(),
                f_doc_versions.result(),
                f_quant_thresh.result(),
                f_sys_health.result(),
                f_so_scorecard.result(),
                top_issue,
                gr.update(choices=f_docs.result()),
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
                    logger.warning("_stream_s3: %s", e)
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
                    authority_html,
                    doc_intel_html,
                    belief_revision_html,
                    contradiction_html,
                    doc_versions_html,
                    quant_thresholds_html,
                    system_health_html,
                    so_scorecard_html,
                    redirect_issue_id,
                    bulk_doc_ref,
                    correction_new_state,
                ],
            ).then(
                fn=lambda mid: state.load_assumptions(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[assumptions_detail_html],
            ).then(
                fn=lambda mid: state.load_content_policy_audit(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[content_policy_html],
            ).then(
                fn=lambda mid: state.load_gaps_detail(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[gaps_detail_html],
            ).then(
                fn=lambda mid: state.load_gap_workbench(mid),
                inputs=[matter_id_box],
                outputs=[gap_workbench_html],
            ).then(
                fn=lambda mid: state.load_investigation_readiness(mid),
                inputs=[matter_id_box],
                outputs=[readiness_html],
            ).then(
                fn=lambda mid: state.load_query_context(mid),
                inputs=[matter_id_box],
                outputs=[query_context_html],
            ).then(
                fn=lambda mid: state.load_source_calibration(mid),
                inputs=[matter_id_box],
                outputs=[source_calibration_html],
            ).then(
                fn=lambda mid: state.load_domain_profile(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[domain_profile_html],
            ).then(
                fn=lambda mid: state.load_document_triage(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[doc_triage_html],
            ).then(
                fn=lambda mid: state.load_taint_summary(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[taint_summary_html],
            ).then(
                fn=lambda mid: gr.update(choices=state.get_document_card_choices(mid)),
                inputs=[matter_id_box],
                outputs=[doc_card_selector],
            ).then(
                fn=lambda mid: state.load_investigation_history(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[investigation_history_html],
            ).then(
                fn=lambda mid: state.get_domain_dropdown_updates(mid),
                inputs=[matter_id_box],
                outputs=[dc_maker_type, dc_objective],
            ).then(
                fn=lambda mid: state.load_domain_composition(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[domain_composition_html],
            ).then(
                fn=lambda mid: state.load_objective_coverage(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[objective_coverage_html],
            ).then(
                fn=lambda mid: state.load_knowledge_seeds(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[knowledge_seeds_html],
            ).then(
                fn=lambda mid: state.load_quant_facts(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[quant_facts_html],
            ).then(
                fn=lambda mid: state.load_decision_leverage(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[decision_leverage_html],
            ).then(
                fn=lambda mid: (
                    lambda r: (r[0], gr.update(choices=r[1], value=None))
                )(state.load_steering_panel(mid, domain=state._detect_domain(mid))),
                inputs=[matter_id_box],
                outputs=[steering_panel_html, steering_action_dropdown],
            ).then(
                fn=lambda mid: state.load_output_quality(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[output_quality_html],
            ).then(
                fn=lambda mid: state.load_deliverable_workbench(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[deliverable_workbench_html],
            ).then(
                fn=lambda mid: state.load_scenario_workbench(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[scenario_workbench_html],
            ).then(
                fn=lambda mid: state.load_alternative_theories(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[alt_theories_html],
            ).then(
                fn=lambda mid: state.load_manifest_inspector(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[manifest_inspector_html],
            ).then(
                fn=lambda mid: state.load_freshness_report(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[freshness_report_html],
            ).then(
                fn=lambda mid: state.load_cache_stats(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[cache_stats_html],
            ).then(
                fn=lambda mid: state.load_llm_usage(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[llm_usage_html],
            ).then(
                fn=lambda mid: state.load_domain_readiness(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[domain_readiness_html],
            ).then(
                fn=lambda mid: state.load_issue_brief(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[issue_brief_html],
            ).then(
                fn=lambda mid: state.load_assumption_review(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[assumption_review_html],
            ).then(
                fn=lambda mid: state.load_quant_ontology(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[quant_ontology_html, quant_alias_raw_dropdown],
            ).then(
                fn=lambda mid: state.load_answer_audits(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[answer_audit_html],
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
                    authority_html,
                    doc_intel_html,
                    belief_revision_html,
                    contradiction_html,
                    doc_versions_html,
                    quant_thresholds_html,
                    system_health_html,
                    so_scorecard_html,
                    redirect_issue_id,
                    bulk_doc_ref,
                    correction_new_state,
                ],
            ).then(
                fn=lambda mid: state.load_assumptions(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[assumptions_detail_html],
            ).then(
                fn=lambda mid: state.load_content_policy_audit(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[content_policy_html],
            ).then(
                fn=lambda mid: state.load_gaps_detail(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[gaps_detail_html],
            ).then(
                fn=lambda mid: state.load_gap_workbench(mid),
                inputs=[matter_id_box],
                outputs=[gap_workbench_html],
            ).then(
                fn=lambda mid: state.load_investigation_readiness(mid),
                inputs=[matter_id_box],
                outputs=[readiness_html],
            ).then(
                fn=lambda mid: state.load_query_context(mid),
                inputs=[matter_id_box],
                outputs=[query_context_html],
            ).then(
                fn=lambda mid: state.load_source_calibration(mid),
                inputs=[matter_id_box],
                outputs=[source_calibration_html],
            ).then(
                fn=lambda mid: state.load_domain_profile(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[domain_profile_html],
            ).then(
                fn=lambda mid: state.load_document_triage(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[doc_triage_html],
            ).then(
                fn=lambda mid: state.load_taint_summary(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[taint_summary_html],
            ).then(
                fn=lambda mid: gr.update(choices=state.get_document_card_choices(mid)),
                inputs=[matter_id_box],
                outputs=[doc_card_selector],
            ).then(
                fn=lambda mid: state.load_investigation_history(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[investigation_history_html],
            ).then(
                fn=lambda mid: state.get_domain_dropdown_updates(mid),
                inputs=[matter_id_box],
                outputs=[dc_maker_type, dc_objective],
            ).then(
                fn=lambda mid: state.load_domain_composition(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[domain_composition_html],
            ).then(
                fn=lambda mid: state.load_objective_coverage(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[objective_coverage_html],
            ).then(
                fn=lambda mid: state.load_knowledge_seeds(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[knowledge_seeds_html],
            ).then(
                fn=lambda mid: state.load_quant_facts(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[quant_facts_html],
            ).then(
                fn=lambda mid: state.load_decision_leverage(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[decision_leverage_html],
            ).then(
                fn=lambda mid: (
                    lambda r: (r[0], gr.update(choices=r[1], value=None))
                )(state.load_steering_panel(mid, domain=state._detect_domain(mid))),
                inputs=[matter_id_box],
                outputs=[steering_panel_html, steering_action_dropdown],
            ).then(
                fn=lambda mid: state.load_output_quality(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[output_quality_html],
            ).then(
                fn=lambda mid: state.load_deliverable_workbench(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[deliverable_workbench_html],
            ).then(
                fn=lambda mid: state.load_scenario_workbench(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[scenario_workbench_html],
            ).then(
                fn=lambda mid: state.load_alternative_theories(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[alt_theories_html],
            ).then(
                fn=lambda mid: state.load_manifest_inspector(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[manifest_inspector_html],
            ).then(
                fn=lambda mid: state.load_freshness_report(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[freshness_report_html],
            ).then(
                fn=lambda mid: state.load_cache_stats(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[cache_stats_html],
            ).then(
                fn=lambda mid: state.load_llm_usage(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[llm_usage_html],
            ).then(
                fn=lambda mid: state.load_domain_readiness(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[domain_readiness_html],
            ).then(
                fn=lambda mid: state.load_issue_brief(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[issue_brief_html],
            ).then(
                fn=lambda mid: state.load_assumption_review(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[assumption_review_html],
            ).then(
                fn=lambda mid: state.load_quant_ontology(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[quant_ontology_html, quant_alias_raw_dropdown],
            ).then(
                fn=lambda mid: state.load_answer_audits(mid, domain=state._detect_domain(mid)),
                inputs=[matter_id_box],
                outputs=[answer_audit_html],
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
                authority_html,
                doc_intel_html,
                belief_revision_html,
                contradiction_html,
                doc_versions_html,
                quant_thresholds_html,
                system_health_html,
                so_scorecard_html,
                redirect_issue_id,
                bulk_doc_ref,
                correction_new_state,
            ],
        ).then(
            fn=lambda mid: gr.update(choices=state.load_clarification_choices(mid)),
            inputs=[matter_id_box],
            outputs=[clarification_dropdown],
        ).then(
            fn=lambda mid: state.load_assumptions(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[assumptions_detail_html],
        ).then(
            fn=lambda mid: state.load_content_policy_audit(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[content_policy_html],
        ).then(
            fn=lambda mid: state.load_gaps_detail(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[gaps_detail_html],
        ).then(
            fn=lambda mid: state.load_gap_workbench(mid),
            inputs=[matter_id_box],
            outputs=[gap_workbench_html],
        ).then(
            fn=lambda mid: state.load_investigation_readiness(mid),
            inputs=[matter_id_box],
            outputs=[readiness_html],
        ).then(
            fn=lambda mid: state.load_query_context(mid),
            inputs=[matter_id_box],
            outputs=[query_context_html],
        ).then(
            fn=lambda mid: state.load_source_calibration(mid),
            inputs=[matter_id_box],
            outputs=[source_calibration_html],
        ).then(
            fn=lambda mid: state.load_domain_profile(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[domain_profile_html],
        ).then(
            fn=lambda mid: state.load_document_triage(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[doc_triage_html],
        ).then(
            fn=lambda mid: state.load_taint_summary(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[taint_summary_html],
        ).then(
            fn=lambda mid: gr.update(choices=state.get_document_card_choices(mid)),
            inputs=[matter_id_box],
            outputs=[doc_card_selector],
        ).then(
            fn=lambda mid: state.load_investigation_history(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[investigation_history_html],
        ).then(
            fn=lambda mid: state.get_domain_dropdown_updates(mid),
            inputs=[matter_id_box],
            outputs=[dc_maker_type, dc_objective],
        ).then(
            fn=lambda mid: state.load_domain_composition(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[domain_composition_html],
        ).then(
            fn=lambda mid: state.load_quant_ontology(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[quant_ontology_html, quant_alias_raw_dropdown],
        ).then(
            fn=lambda mid: state.load_objective_coverage(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[objective_coverage_html],
        ).then(
            fn=lambda mid: state.load_answer_audits(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[answer_audit_html],
        ).then(
            fn=lambda mid: state.load_knowledge_seeds(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[knowledge_seeds_html],
        ).then(
            fn=lambda mid: state.load_quant_facts(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[quant_facts_html],
        ).then(
            fn=lambda mid: state.load_decision_leverage(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[decision_leverage_html],
        ).then(
            fn=lambda mid: (
                lambda r: (r[0], gr.update(choices=r[1], value=None))
            )(state.load_steering_panel(mid, domain=state._detect_domain(mid))),
            inputs=[matter_id_box],
            outputs=[steering_panel_html, steering_action_dropdown],
        ).then(
            fn=lambda mid: state.load_output_quality(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[output_quality_html],
        ).then(
            fn=lambda mid: state.load_deliverable_workbench(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[deliverable_workbench_html],
        ).then(
            fn=lambda mid: state.load_scenario_workbench(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[scenario_workbench_html],
        ).then(
            fn=lambda mid: state.load_alternative_theories(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[alt_theories_html],
        ).then(
            fn=lambda mid: state.load_manifest_inspector(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[manifest_inspector_html],
        ).then(
            fn=lambda mid: state.load_freshness_report(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[freshness_report_html],
        ).then(
            fn=lambda mid: state.load_cache_stats(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[cache_stats_html],
        ).then(
            fn=lambda mid: state.load_llm_usage(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[llm_usage_html],
        ).then(
            fn=lambda mid: state.load_domain_readiness(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[domain_readiness_html],
        ).then(
            fn=lambda mid: state.load_issue_brief(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[issue_brief_html],
        ).then(
            fn=lambda mid: state.load_assumption_review(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[assumption_review_html],
        )

        export_report_btn.click(
            fn=lambda mid: state.export_summary_report(mid),
            inputs=[matter_id_box],
            outputs=[export_report_file],
        )
        refresh_annotations_btn.click(
            fn=lambda mid: state.load_annotations(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[annotations_html],
        )
        add_annotation_btn.click(
            fn=lambda mid, doc, text, ann_type: state.do_add_annotation(mid, doc, text, ann_type),
            inputs=[matter_id_box, annotation_doc, annotation_text, annotation_type],
            outputs=[annotation_result, annotations_html],
        )
        delete_annotation_btn.click(
            fn=lambda mid, ann_id: state.do_delete_annotation(mid, ann_id),
            inputs=[matter_id_box, delete_annotation_id],
            outputs=[annotation_result, annotations_html],
        )
        refresh_dc_btn.click(
            fn=lambda mid: state.load_decision_context(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[decision_context_html],
        )
        set_dc_btn.click(
            fn=lambda mid, mt, obj, nm, nt, nr: state.do_set_decision_context(mid, mt, obj, nm, nt, nr),
            inputs=[matter_id_box, dc_maker_type, dc_objective, dc_name, dc_notes, dc_narrow],
            outputs=[dc_result, decision_context_html],
        )
        clear_dc_btn.click(
            fn=lambda mid: state.do_clear_decision_context(mid),
            inputs=[matter_id_box],
            outputs=[dc_result, decision_context_html],
        )

        # --- Detail panel refreshes ---
        refresh_assertions_btn.click(
            fn=lambda mid: state.load_assertions(mid),
            inputs=[matter_id_box],
            outputs=[assertions_md],
        )
        assertion_search_btn.click(
            fn=lambda mid, q: state.search_assertions(mid, q),
            inputs=[matter_id_box, assertion_search_box],
            outputs=[assertions_md],
        )
        assertion_search_box.submit(
            fn=lambda mid, q: state.search_assertions(mid, q),
            inputs=[matter_id_box, assertion_search_box],
            outputs=[assertions_md],
        )
        refresh_readiness_btn.click(
            fn=lambda mid: state.load_investigation_readiness(mid),
            inputs=[matter_id_box],
            outputs=[readiness_html],
        )
        refresh_query_context_btn.click(
            fn=lambda mid: state.load_query_context(mid),
            inputs=[matter_id_box],
            outputs=[query_context_html],
        )
        refresh_source_calibration_btn.click(
            fn=lambda mid: state.load_source_calibration(mid),
            inputs=[matter_id_box],
            outputs=[source_calibration_html],
        )
        refresh_gap_workbench_btn.click(
            fn=lambda mid: state.load_gap_workbench(mid),
            inputs=[matter_id_box],
            outputs=[gap_workbench_html],
        )
        refresh_gaps_btn.click(
            fn=lambda mid: state.load_gaps_detail(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[gaps_detail_html],
        )
        refresh_gap_choices_btn.click(
            fn=lambda mid: gr.update(choices=state.get_gap_choices(mid)),
            inputs=[matter_id_box],
            outputs=[gap_action_dropdown],
        )
        resolve_gap_btn.click(
            fn=lambda mid, gid, note: state.resolve_gap(mid, gid, note),
            inputs=[matter_id_box, gap_action_dropdown, resolve_gap_note_input],
            outputs=[resolve_gap_result, gap_workbench_html],
        )
        escalate_gap_btn.click(
            fn=lambda mid, gid: state.escalate_gap(mid, gid),
            inputs=[matter_id_box, gap_action_dropdown],
            outputs=[resolve_gap_result, gap_workbench_html],
        )
        refresh_steering_btn.click(
            fn=lambda mid: (
                lambda r: (r[0], gr.update(choices=r[1], value=None))
            )(state.load_steering_panel(mid, domain=state._detect_domain(mid))),
            inputs=[matter_id_box],
            outputs=[steering_panel_html, steering_action_dropdown],
        )
        steering_execute_btn.click(
            fn=lambda mid, action_json, user_input: state.execute_steering_action(
                mid, action_json, user_input,
            ),
            inputs=[matter_id_box, steering_action_dropdown, steering_action_input],
            outputs=[steering_execute_result],
        ).then(
            fn=lambda mid: (
                lambda r: (r[0], gr.update(choices=r[1], value=None))
            )(state.load_steering_panel(mid, domain=state._detect_domain(mid))),
            inputs=[matter_id_box],
            outputs=[steering_panel_html, steering_action_dropdown],
        ).then(
            fn=lambda mid: state.load_gaps_detail(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[gaps_detail_html],
        ).then(
            fn=lambda mid: state.load_investigation_readiness(mid),
            inputs=[matter_id_box],
            outputs=[readiness_html],
        ).then(
            fn=lambda mid: state.load_query_context(mid),
            inputs=[matter_id_box],
            outputs=[query_context_html],
        ).then(
            fn=lambda mid: state.load_source_calibration(mid),
            inputs=[matter_id_box],
            outputs=[source_calibration_html],
        ).then(
            fn=lambda mid: state.load_objective_coverage(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[objective_coverage_html],
        ).then(
            fn=lambda mid: state.load_decision_leverage(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[decision_leverage_html],
        )
        refresh_assumptions_btn.click(
            fn=lambda mid: state.load_assumptions(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[assumptions_detail_html],
        )
        refresh_assumption_review_btn.click(
            fn=lambda mid: state.load_assumption_review(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[assumption_review_html],
        )
        assumption_action_btn.click(
            fn=lambda mid, aid, action, reason: state.update_assumption_status(mid, aid, action, reason),
            inputs=[matter_id_box, assumption_id_input, assumption_action_dropdown, assumption_reason_input],
            outputs=[assumption_action_result, assumptions_detail_html],
        ).then(
            fn=lambda mid: state.load_objective_coverage(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[objective_coverage_html],
        ).then(
            fn=lambda mid: state.load_assumption_review(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[assumption_review_html],
        )
        priority_apply_btn.click(
            fn=lambda mid, iid, p: state.set_issue_priority(mid, iid, p),
            inputs=[matter_id_box, priority_issue_id_input, priority_dropdown],
            outputs=[priority_result],
        )
        refresh_quant_btn.click(
            fn=lambda mid: state.load_quant(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[quant_md],
        )
        detect_conflicts_btn.click(
            fn=lambda mid: state.do_detect_quant_conflicts(mid),
            inputs=[matter_id_box],
            outputs=[detect_conflicts_result, quant_md],
        )
        refresh_quant_ontology_btn.click(
            fn=lambda mid: state.load_quant_ontology(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[quant_ontology_html, quant_alias_raw_dropdown],
        )
        approve_quant_alias_btn.click(
            fn=lambda mid, raw, canonical, unit: state.approve_quant_alias(mid, raw, canonical, unit),
            inputs=[matter_id_box, quant_alias_raw_dropdown, quant_alias_canonical_dropdown, quant_alias_unit_input],
            outputs=[quant_alias_result, quant_ontology_html, quant_alias_raw_dropdown],
        )
        refresh_quant_facts_btn.click(
            fn=lambda mid: state.load_quant_facts(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[quant_facts_html],
        )
        refresh_leverage_btn.click(
            fn=lambda mid: state.load_decision_leverage(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[decision_leverage_html],
        )
        refresh_timeline_btn.click(
            fn=lambda mid: state.load_timeline(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[timeline_html],
        )
        refresh_evidence_btn.click(
            fn=lambda mid: state.load_evidence_matrix(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[evidence_matrix_html],
        )
        refresh_proof_btn.click(
            fn=lambda mid: state.load_proof_state(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[proof_state_html],
        )
        recompute_proof_btn.click(
            fn=lambda mid: state.recompute_proof_state(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[proof_state_html],
        ).then(
            fn=lambda mid: state.load_so_scorecard(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[so_scorecard_html],
        ).then(
            fn=lambda mid: state.load_overview(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[overview_md],
        )
        recompute_issue_proof_btn.click(
            fn=lambda mid, iid: state.recompute_issue_proof_state(mid, iid),
            inputs=[matter_id_box, proof_issue_id_input],
            outputs=[recompute_issue_result, proof_state_html],
        )
        refresh_authority_btn.click(
            fn=lambda mid: state.load_authority_network(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[authority_html],
        )
        authority_search_btn.click(
            fn=lambda mid, q: state.search_authorities(mid, q),
            inputs=[matter_id_box, authority_search_box],
            outputs=[authority_html],
        )
        authority_search_box.submit(
            fn=lambda mid, q: state.search_authorities(mid, q),
            inputs=[matter_id_box, authority_search_box],
            outputs=[authority_html],
        )
        upsert_authority_btn.click(
            fn=lambda mid, cit, atype, name, jur, wt: state.do_upsert_authority(mid, cit, atype, name, jur, wt),
            inputs=[matter_id_box, auth_citation, auth_type, auth_name, auth_jurisdiction, auth_weight],
            outputs=[upsert_authority_result, authority_html],
        )
        link_authority_btn.click(
            fn=lambda mid, aid, iid, rel: state.do_link_authority_issue(mid, aid, iid, rel),
            inputs=[matter_id_box, link_auth_id, link_issue_id, link_relevance],
            outputs=[link_authority_result, authority_html],
        )
        unlink_authority_btn.click(
            fn=lambda mid, aid, iid: state.do_unlink_authority_issue(mid, aid, iid),
            inputs=[matter_id_box, link_auth_id, link_issue_id],
            outputs=[link_authority_result, authority_html],
        )
        refresh_doc_intel_btn.click(
            fn=lambda mid: state.load_document_intelligence(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[doc_intel_html],
        )
        refresh_doc_console_choices_btn.click(
            fn=lambda mid: gr.update(choices=state.get_reviewable_doc_choices(mid)),
            inputs=[matter_id_box],
            outputs=[doc_console_selector],
        )
        load_doc_console_btn.click(
            fn=lambda mid, doc: state.load_document_console(mid, doc),
            inputs=[matter_id_box, doc_console_selector],
            outputs=[doc_console_html],
        )
        load_doc_card_btn.click(
            fn=lambda mid, doc: state.load_document_card(mid, doc),
            inputs=[matter_id_box, doc_card_selector],
            outputs=[document_card_html],
        )
        refresh_doc_card_choices_btn.click(
            fn=lambda mid: gr.update(choices=state.get_document_card_choices(mid)),
            inputs=[matter_id_box],
            outputs=[doc_card_selector],
        )
        card_edit_btn.click(
            fn=lambda mid, doc, dt, sr, pf, os_, fl: state.correct_document_card(
                mid, doc, dt, sr, pf, os_, fl,
            ),
            inputs=[
                matter_id_box, doc_card_selector,
                card_edit_doc_type, card_edit_source_role,
                card_edit_privilege, card_edit_operative,
                card_edit_flags,
            ],
            outputs=[card_edit_result, document_card_html],
        )
        refresh_belief_btn.click(
            fn=lambda mid: state.load_belief_revisions(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[belief_revision_html],
        )
        refresh_contradiction_btn.click(
            fn=lambda mid: state.load_contradictions(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[contradiction_html],
        )
        mine_contradiction_btn.click(
            fn=lambda mid: state.mine_and_load_contradictions(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[contradiction_html],
        )
        resolve_btn.click(
            fn=lambda mid, att, atd, dec, rat: state.resolve_contradiction(mid, att, atd, dec, rat),
            inputs=[matter_id_box, resolve_attacker_id, resolve_attacked_id, resolve_decision, resolve_rationale],
            outputs=[resolve_result, contradiction_html],
        )
        refresh_doc_versions_btn.click(
            fn=lambda mid: state.load_document_versions(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[doc_versions_html],
        )
        detect_versions_btn.click(
            fn=lambda mid: state.detect_and_load_document_versions(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[doc_versions_html],
        )
        operative_lookup_btn.click(
            fn=lambda mid, did: state.lookup_operative_version(mid, did, domain=state._detect_domain(mid)),
            inputs=[matter_id_box, operative_doc_id_input],
            outputs=[operative_version_html],
        )
        refresh_quant_thresholds_btn.click(
            fn=lambda mid, eh, df: state.load_quant_thresholds(
                mid, domain=state._detect_domain(mid),
                exposure_high=float(eh) if eh else 10_000.0,
                disputed_fraction_min=float(df) if df else 0.10,
            ),
            inputs=[matter_id_box, exposure_high_input, disputed_fraction_input],
            outputs=[quant_thresholds_html],
        )
        refresh_system_health_btn.click(
            fn=lambda mid: state.load_system_health(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[system_health_html],
        )
        flush_pending_btn.click(
            fn=lambda mid: state.flush_pending_propagation(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[system_health_html],
        ).then(
            fn=lambda mid: state.load_proof_state(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[proof_state_html],
        ).then(
            fn=lambda mid: state.load_belief_revisions(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[belief_revision_html],
        )
        refresh_so_scorecard_btn.click(
            fn=lambda mid: state.load_so_scorecard(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[so_scorecard_html],
        )
        refresh_objective_coverage_btn.click(
            fn=lambda mid: state.load_objective_coverage(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[objective_coverage_html],
        )
        set_criterion_btn.click(
            fn=lambda mid, pid, st, reason: state.set_criterion_status(mid, pid, st, reason),
            inputs=[matter_id_box, criterion_predicate_id, criterion_status_dropdown, criterion_reason_input],
            outputs=[criterion_result, objective_coverage_html],
        )
        add_criterion_btn.click(
            fn=lambda mid, oid, desc, burden: state.add_criterion(mid, oid, desc, burden),
            inputs=[matter_id_box, add_criterion_objective_id, add_criterion_desc, add_criterion_burden],
            outputs=[add_criterion_result, objective_coverage_html],
        )
        refresh_output_quality_btn.click(
            fn=lambda mid: state.load_output_quality(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[output_quality_html],
        )
        refresh_deliverable_btn.click(
            fn=lambda mid: state.load_deliverable_workbench(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[deliverable_workbench_html],
        )
        refresh_brief_btn.click(
            fn=lambda mid: state.load_issue_brief(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[issue_brief_html],
        )
        refresh_scenario_btn.click(
            fn=lambda mid: state.load_scenario_workbench(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[scenario_workbench_html],
        )
        scenario_create_btn.click(
            fn=lambda mid, name, assumptions, notes: state.create_scenario_branch_ui(mid, name, assumptions, notes),
            inputs=[matter_id_box, scenario_name_input, scenario_assumptions_input, scenario_notes_input],
            outputs=[scenario_create_result],
        ).then(
            fn=lambda mid: state.load_scenario_workbench(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[scenario_workbench_html],
        )
        scenario_compare_btn.click(
            fn=lambda mid, bid: state.load_scenario_comparison(mid, bid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box, scenario_compare_branch_id],
            outputs=[scenario_comparison_html],
        )
        load_snapshots_btn.click(
            fn=lambda mid, bid: state.load_scenario_snapshot_history(mid, bid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box, snapshot_history_branch_id],
            outputs=[snapshot_history_html],
        )
        load_deltas_btn.click(
            fn=lambda mid, bid: state.load_scenario_deltas(mid, bid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box, delta_log_branch_id],
            outputs=[delta_log_html],
        )
        apply_delta_btn.click(
            fn=lambda mid, bid, tk, tid, op, pj: state.apply_scenario_delta_ui(mid, bid, tk, tid, op, pj),
            inputs=[matter_id_box, delta_branch_id, delta_target_kind, delta_target_id, delta_operation, delta_payload],
            outputs=[apply_delta_result, delta_log_html],
        )
        compute_snapshot_btn.click(
            fn=lambda mid, bid: state.compute_scenario_snapshot_ui(mid, bid),
            inputs=[matter_id_box, action_branch_id],
            outputs=[branch_action_result, snapshot_history_html],
        )
        archive_branch_btn.click(
            fn=lambda mid, bid: state.archive_scenario_branch_ui(mid, bid),
            inputs=[matter_id_box, action_branch_id],
            outputs=[branch_action_result, scenario_workbench_html],
        )
        refresh_alt_theories_btn.click(
            fn=lambda mid: state.load_alternative_theories(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[alt_theories_html],
        )
        refresh_manifest_btn.click(
            fn=lambda mid: state.load_manifest_inspector(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[manifest_inspector_html],
        )
        refresh_freshness_btn.click(
            fn=lambda mid: state.load_freshness_report(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[freshness_report_html],
        )
        refresh_cache_stats_btn.click(
            fn=lambda mid: state.load_cache_stats(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[cache_stats_html],
        )
        refresh_llm_usage_btn.click(
            fn=lambda mid: state.load_llm_usage(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[llm_usage_html],
        )
        impact_preview_btn.click(
            fn=lambda mid, at, pj: state.load_impact_preview(
                mid, at, pj, domain=state._detect_domain(mid),
            ),
            inputs=[matter_id_box, impact_action_type, impact_payload],
            outputs=[impact_preview_html],
        )
        refresh_readiness_btn.click(
            fn=lambda mid: state.load_domain_readiness(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[domain_readiness_html],
        )
        refresh_answer_audit_btn.click(
            fn=lambda mid: state.load_answer_audits(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[answer_audit_html],
        )
        refresh_knowledge_seeds_btn.click(
            fn=lambda mid: state.load_knowledge_seeds(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[knowledge_seeds_html],
        )
        seed_review_btn.click(
            fn=lambda mid, sid, dec, note: state.review_knowledge_seed(mid, sid, dec, note),
            inputs=[matter_id_box, seed_id_input, seed_decision_input, seed_review_note_input],
            outputs=[seed_review_status, knowledge_seeds_html],
        )
        promote_seed_btn.click(
            fn=lambda mid, kind, dp, pj: state.promote_knowledge_seed_ui(mid, kind, dp, pj),
            inputs=[matter_id_box, promote_seed_kind, promote_domain_profile, promote_payload],
            outputs=[promote_seed_result, knowledge_seeds_html],
        )
        refresh_domain_profile_btn.click(
            fn=lambda mid: state.load_domain_profile(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[domain_profile_html],
        )
        refresh_domain_composition_btn.click(
            fn=lambda mid: state.load_domain_composition(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[domain_composition_html],
        )
        refresh_doc_triage_btn.click(
            fn=lambda mid: state.load_document_triage(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[doc_triage_html],
        )
        refresh_taint_btn.click(
            fn=lambda mid: state.load_taint_summary(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[taint_summary_html],
        )
        reclassify_btn.click(
            fn=lambda mid, did, flag: state.reclassify_document_sensitivity(
                mid, did, flag, domain=state._detect_domain(mid),
            ),
            inputs=[matter_id_box, sensitivity_doc_id_input, sensitivity_flag_checkbox],
            outputs=[sensitivity_result_html],
        ).then(
            fn=lambda mid: state.load_taint_summary(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[taint_summary_html],
        )
        mark_doc_stale_btn.click(
            fn=lambda mid, did, reason: state.mark_document_stale_action(
                mid, did, reason, domain=state._detect_domain(mid),
            ),
            inputs=[matter_id_box, stale_doc_id_input, stale_reason_input],
            outputs=[sensitivity_result_html],
        ).then(
            fn=lambda mid: state.load_taint_summary(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[taint_summary_html],
        )
        mark_span_stale_btn.click(
            fn=lambda mid, sid, reason: state.mark_span_stale_action(
                mid, sid, reason, domain=state._detect_domain(mid),
            ),
            inputs=[matter_id_box, stale_span_id_input, span_stale_reason_input],
            outputs=[sensitivity_result_html],
        ).then(
            fn=lambda mid: state.load_taint_summary(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[taint_summary_html],
        )
        refresh_content_policy_btn.click(
            fn=lambda mid: state.load_content_policy_audit(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[content_policy_html],
        )
        refresh_comm_btn.click(
            fn=lambda mid: state.load_communication_map(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[communication_html],
        )
        scan_duplicates_btn.click(
            fn=lambda mid: state.load_duplicate_actors(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[actor_duplicates_html],
        )
        refresh_duplicates_btn.click(
            fn=lambda mid: state.load_duplicate_actors(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[actor_duplicates_html],
        )
        merge_actors_btn.click(
            fn=lambda mid, keep, merge, confirmed: state.do_merge_actors(mid, keep, merge, confirmed),
            inputs=[matter_id_box, merge_keep_id, merge_discard_id, merge_confirm],
            outputs=[merge_result, actor_duplicates_html],
        )
        resolve_actor_btn.click(
            fn=lambda mid, name: state.do_resolve_actor(mid, name),
            inputs=[matter_id_box, resolve_actor_name],
            outputs=[resolve_actor_result],
        )
        refresh_llm_btn.click(
            fn=lambda mid: state.load_llm_analytics(mid),
            inputs=[matter_id_box],
            outputs=[llm_analytics_html],
        )
        refresh_history_btn.click(
            fn=lambda mid: state.load_investigation_history(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[investigation_history_html],
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
                state.load_timeline(mid, domain=state._detect_domain(mid)),
                state.load_evidence_matrix(mid, domain=state._detect_domain(mid)),
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
                    state.load_issues(mid, domain=state._detect_domain(mid)),
                    state.load_overview(mid, domain=state._detect_domain(mid)),
                )
            return _fmt_correction_result(result_text), gr.update(), gr.update(), gr.update()

        correction_btn.click(
            fn=_correct_and_refresh,
            inputs=[matter_id_box, correction_assertion_id, correction_new_state, correction_reason],
            outputs=[correction_result, assertions_md, issues_md, overview_md],
        )

        # --- Assertion Inspector wiring ---
        inspect_btn.click(
            fn=lambda mid, aid: state.inspect_assertion(mid, aid),
            inputs=[matter_id_box, inspector_assertion_id],
            outputs=[inspector_html],
        )
        trace_btn.click(
            fn=lambda mid, aid: state.load_assertion_trace(mid, aid),
            inputs=[matter_id_box, inspector_assertion_id],
            outputs=[trace_html],
        )
        inspector_assertion_id.submit(
            fn=lambda mid, aid: state.inspect_assertion(mid, aid),
            inputs=[matter_id_box, inspector_assertion_id],
            outputs=[inspector_html],
        )

        # --- Issue Evidence Drill-Down wiring ---
        issue_drilldown_btn.click(
            fn=lambda mid, iid: state.load_issue_assertions(mid, iid),
            inputs=[matter_id_box, issue_drilldown_id],
            outputs=[issue_assertions_html],
        ).then(
            fn=lambda mid, iid: state.load_source_agreement(mid, iid),
            inputs=[matter_id_box, issue_drilldown_id],
            outputs=[source_agreement_html],
        ).then(
            fn=lambda mid, iid: state.load_assertion_graph(mid, iid),
            inputs=[matter_id_box, issue_drilldown_id],
            outputs=[assertion_graph_html],
        ).then(
            fn=lambda mid, iid: state.load_issue_closure_workbench(mid, iid),
            inputs=[matter_id_box, issue_drilldown_id],
            outputs=[issue_closure_html],
        )
        issue_drilldown_id.submit(
            fn=lambda mid, iid: state.load_issue_assertions(mid, iid),
            inputs=[matter_id_box, issue_drilldown_id],
            outputs=[issue_assertions_html],
        ).then(
            fn=lambda mid, iid: state.load_source_agreement(mid, iid),
            inputs=[matter_id_box, issue_drilldown_id],
            outputs=[source_agreement_html],
        ).then(
            fn=lambda mid, iid: state.load_assertion_graph(mid, iid),
            inputs=[matter_id_box, issue_drilldown_id],
            outputs=[assertion_graph_html],
        )

        # --- Review Inbox wiring ---
        def _refresh_review_and_drawer(mid):
            domain = state._detect_domain(mid)
            queue_html, dropdown_update = state.load_review_queue(mid, domain=domain)
            new_value = dropdown_update.get("value") if isinstance(dropdown_update, dict) else None
            drawer = state.load_source_drawer(mid, new_value or "", domain=domain)
            badge = state.load_review_count_badge(mid, domain=domain)
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
            fn=lambda mid, tgt: state.load_source_drawer(mid, tgt, domain=state._detect_domain(mid)),
            inputs=[matter_id_box, review_target],
            outputs=[source_drawer_html],
        )

        def _verify_and_refresh(mid, target, note):
            prior_target = target
            result = state.do_verify_target(mid, target, note)
            domain = state._detect_domain(mid)
            snap = state.load_post_review_snapshot(mid, prior_target, domain=domain)
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
            domain = state._detect_domain(mid)
            snap = state.load_post_review_snapshot(mid, prior_target, domain=domain)
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
            domain = state._detect_domain(mid)
            queue_html, dropdown_update = state.load_review_queue(mid, domain=domain)
            return (
                result, queue_html, dropdown_update,
                state.load_review_count_badge(mid, domain=domain),
                state.load_assertions(mid),
                state.load_issues(mid, domain=domain),
                state.load_overview(mid, domain=domain),
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

        def _bulk_verify_span_and_refresh(mid, span_ref):
            result = state.do_bulk_verify_by_span(mid, span_ref)
            domain = state._detect_domain(mid)
            queue_html, dropdown_update = state.load_review_queue(mid, domain=domain)
            return (
                result, queue_html, dropdown_update,
                state.load_review_count_badge(mid, domain=domain),
                state.load_assertions(mid),
                state.load_issues(mid, domain=domain),
                state.load_overview(mid, domain=domain),
            )

        bulk_span_verify_btn.click(
            fn=_bulk_verify_span_and_refresh,
            inputs=[matter_id_box, bulk_span_ref],
            outputs=[
                bulk_span_result, review_queue_html, review_target,
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
            domain = state._detect_domain(mid)
            queue_html, dropdown_update = state.load_review_queue(mid, domain=domain)
            return (
                *updates,
                queue_html,
                dropdown_update,
                state.load_review_count_badge(mid, domain=domain),
                state.load_assertions(mid),
                state.load_issues(mid, domain=domain),
                state.load_overview(mid, domain=domain),
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
                authority_html,
                doc_intel_html,
                belief_revision_html,
                contradiction_html,
                doc_versions_html,
                quant_thresholds_html,
                system_health_html,
                so_scorecard_html,
                redirect_issue_id,
                bulk_doc_ref,
                correction_new_state,
            ],
        ).then(
            fn=lambda mid: state.load_assumptions(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[assumptions_detail_html],
        ).then(
            fn=lambda mid: state.load_content_policy_audit(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[content_policy_html],
        ).then(
            fn=lambda mid: state.load_gaps_detail(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[gaps_detail_html],
        ).then(
            fn=lambda mid: state.load_gap_workbench(mid),
            inputs=[matter_id_box],
            outputs=[gap_workbench_html],
        ).then(
            fn=lambda mid: state.load_investigation_readiness(mid),
            inputs=[matter_id_box],
            outputs=[readiness_html],
        ).then(
            fn=lambda mid: state.load_query_context(mid),
            inputs=[matter_id_box],
            outputs=[query_context_html],
        ).then(
            fn=lambda mid: state.load_source_calibration(mid),
            inputs=[matter_id_box],
            outputs=[source_calibration_html],
        ).then(
            fn=lambda mid: state.load_domain_profile(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[domain_profile_html],
        ).then(
            fn=lambda mid: state.load_document_triage(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[doc_triage_html],
        ).then(
            fn=lambda mid: state.load_taint_summary(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[taint_summary_html],
        ).then(
            fn=lambda mid: gr.update(choices=state.get_document_card_choices(mid)),
            inputs=[matter_id_box],
            outputs=[doc_card_selector],
        ).then(
            fn=lambda mid: state.load_investigation_history(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[investigation_history_html],
        ).then(
            fn=lambda mid: state.get_domain_dropdown_updates(mid),
            inputs=[matter_id_box],
            outputs=[dc_maker_type, dc_objective],
        ).then(
            fn=lambda mid: state.load_domain_composition(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[domain_composition_html],
        ).then(
            fn=lambda mid: state.load_objective_coverage(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[objective_coverage_html],
        ).then(
            fn=lambda mid: state.load_knowledge_seeds(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[knowledge_seeds_html],
        ).then(
            fn=lambda mid: state.load_quant_facts(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[quant_facts_html],
        ).then(
            fn=lambda mid: state.load_decision_leverage(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[decision_leverage_html],
        ).then(
            fn=lambda mid: (
                lambda r: (r[0], gr.update(choices=r[1], value=None))
            )(state.load_steering_panel(mid, domain=state._detect_domain(mid))),
            inputs=[matter_id_box],
            outputs=[steering_panel_html, steering_action_dropdown],
        ).then(
            fn=lambda mid: state.load_output_quality(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[output_quality_html],
        ).then(
            fn=lambda mid: state.load_deliverable_workbench(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[deliverable_workbench_html],
        ).then(
            fn=lambda mid: state.load_scenario_workbench(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[scenario_workbench_html],
        ).then(
            fn=lambda mid: state.load_alternative_theories(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[alt_theories_html],
        ).then(
            fn=lambda mid: state.load_manifest_inspector(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[manifest_inspector_html],
        ).then(
            fn=lambda mid: state.load_freshness_report(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[freshness_report_html],
        ).then(
            fn=lambda mid: state.load_cache_stats(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[cache_stats_html],
        ).then(
            fn=lambda mid: state.load_llm_usage(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[llm_usage_html],
        ).then(
            fn=lambda mid: state.load_domain_readiness(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[domain_readiness_html],
        ).then(
            fn=lambda mid: state.load_issue_brief(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[issue_brief_html],
        ).then(
            fn=lambda mid: state.load_assumption_review(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[assumption_review_html],
        )

        clarification_dropdown.change(
            fn=lambda mid, qid: state.get_clarification_context(mid, qid),
            inputs=[matter_id_box, clarification_dropdown],
            outputs=[clarification_context_html],
        )

        answer_clarification_btn.click(
            fn=lambda mid, qid, ans: state.do_answer_clarification(mid, qid, ans),
            inputs=[matter_id_box, clarification_dropdown, clarification_answer_input],
            outputs=[answer_clarification_result],
        ).then(
            fn=lambda mid: gr.update(choices=state.load_clarification_choices(mid), value=None),
            inputs=[matter_id_box],
            outputs=[clarification_dropdown],
        ).then(
            fn=lambda: "",
            inputs=[],
            outputs=[clarification_context_html],
        )

        generate_clarifications_btn.click(
            fn=lambda mid: state.do_generate_clarifications(mid),
            inputs=[matter_id_box],
            outputs=[answer_clarification_result],
        ).then(
            fn=lambda mid: gr.update(choices=state.load_clarification_choices(mid), value=None),
            inputs=[matter_id_box],
            outputs=[clarification_dropdown],
        )

        set_trust_btn.click(
            fn=lambda mid, pat, lvl, note: state.do_set_trust_override(mid, pat, lvl, note),
            inputs=[matter_id_box, trust_doc_pattern, trust_level_dropdown, trust_note],
            outputs=[trust_override_result],
        ).then(
            fn=lambda mid: state.load_trust_overrides(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[trust_overrides_html],
        )
        refresh_trust_btn.click(
            fn=lambda mid: state.load_trust_overrides(mid, domain=state._detect_domain(mid)),
            inputs=[matter_id_box],
            outputs=[trust_overrides_html],
        )
        delete_trust_btn.click(
            fn=lambda mid, pat: state.do_delete_trust_override(mid, pat),
            inputs=[matter_id_box, delete_trust_pattern],
            outputs=[trust_override_result, trust_overrides_html],
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

        # Inject JS to enable folder selection on the folder-upload inputs and
        # to capture each file's webkitRelativePath into a hidden Textbox so the
        # server can preserve subdirectory structure on S3 upload. We post a
        # JSON map of {leafFilename: [relpath, ...]} keyed by leaf name to
        # disambiguate duplicate filenames living in different subfolders.
        if _s3_mode:
            _folder_upload_js = """
() => {
    const widgets = ['folder-upload', 'folder-upload-new'];

    const writeRelpathMap = (widgetId, fileList) => {
        const map = {};
        Array.from(fileList || []).forEach(f => {
            const rp = f.webkitRelativePath || f.name;
            const leaf = f.name;
            if (!map[leaf]) map[leaf] = [];
            map[leaf].push(rp);
        });
        const target = document.getElementById(widgetId + '-relpaths');
        if (!target) return;
        const box = target.querySelector('textarea, input[type=text]');
        if (!box) return;
        const desc = Object.getOwnPropertyDescriptor(box.constructor.prototype, 'value');
        if (!desc || !desc.set) return;
        desc.set.call(box, JSON.stringify(map));
        box.dispatchEvent(new Event('input', { bubbles: true }));
        box.dispatchEvent(new Event('change', { bubbles: true }));
    };

    const setupWidget = (widgetId) => {
        const wrapper = document.getElementById(widgetId);
        if (!wrapper) return;
        wrapper.querySelectorAll('input[type=file]').forEach(inp => {
            inp.setAttribute('webkitdirectory', '');
            inp.setAttribute('directory', '');
            inp.setAttribute('multiple', '');
            if (inp.dataset.relpathBound === '1') return;
            inp.dataset.relpathBound = '1';
            inp.addEventListener('change', () => {
                writeRelpathMap(widgetId, inp.files);
            });
        });
    };

    const apply = () => widgets.forEach(setupWidget);
    apply();
    new MutationObserver(apply).observe(document.body, { childList: true, subtree: true });
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
