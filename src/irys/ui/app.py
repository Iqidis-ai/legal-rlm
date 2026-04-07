"""Irys RLM Gradio UI — 6-panel legal intelligence dashboard.

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
import os
import queue
import threading
import time
from typing import Generator, Optional

import gradio as gr

from .backends.in_process import InProcessBackend

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
# Helpers
# ---------------------------------------------------------------------------

STEP_ICONS = {
    "THINKING": "💭", "SEARCHING": "🔍", "READING": "📄",
    "SYNTHESIZING": "⚡", "VERIFY": "✓", "REPLAN": "↺", "ANSWER": "✅",
    "SEARCH": "🔍", "READ": "📄", "SYNTH": "⚡",
}


def _fmt_coverage(frac: Optional[float]) -> str:
    if frac is None:
        return "—"
    return f"{frac:.0%}"


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


def _fmt_overview(data: dict) -> str:
    if not data:
        return "No matter loaded."
    stats = data.get("stats", {})
    so = data.get("so_metrics", {})
    lines = [
        "## Matter Overview",
        f"**Assertions:** {stats.get('assertion_count', 0)}  |  "
        f"**Issues:** {stats.get('open_issue_count', 0)}  |  "
        f"**Gaps:** {stats.get('open_gap_count', 0)}",
        f"**Actors:** {stats.get('actor_count', 0)}  |  "
        f"**Quant facts:** {stats.get('quant_fact_count', 0)}  |  "
        f"**Pending clarifications:** {stats.get('pending_clarifications', 0)}",
    ]

    # SO metrics
    cov = so.get("issue_coverage_avg")
    reuse = so.get("reuse_rate")
    struct = so.get("assertion_structure_rate")
    src = so.get("source_role_known_rate")
    lines.append(
        f"\n**SO-1 reuse:** {f'{reuse:.1%}' if reuse is not None else '—'}  |  "
        f"**SO-2 structure:** {f'{struct:.1%}' if struct is not None else '—'}  |  "
        f"**SO-4 coverage:** {f'{cov:.1%}' if cov is not None else '—'}  |  "
        f"**SO-5 source calibration:** {f'{src:.1%}' if src is not None else '—'}"
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
    lines = [
        "| ID (paste to redirect) | Issue | Coverage | Proof | Sup | Atk |",
        "|------------------------|-------|----------|-------|-----|-----|",
    ]
    for iss in issues:
        issue_id = iss.get("id", "?")
        title = (iss.get("title") or issue_id)[:38]
        cov = _fmt_coverage(iss.get("coverage_fraction"))
        proof = iss.get("proof_status") or "—"
        sup = iss.get("supporting_count", "—")
        atk = iss.get("attacking_count", "—")
        # Full ID shown — backend get_issue() requires exact match; truncated IDs silently fail.
        lines.append(f"| `{issue_id}` | {title} | {cov} | {proof} | {sup} | {atk} |")
    return "\n".join(lines)


def _fmt_assertions(assertions: list) -> str:
    if not assertions:
        return "No assertions."
    lines = [
        "| ID (paste to correct) | Proposition | State | Conf | Source | Speech |",
        "|-----------------------|-------------|-------|------|--------|--------|",
    ]
    for a in assertions:
        assertion_id = a.get("id", "?")
        # Full ID — backend correct_assertion() requires exact match.
        prop = (a.get("proposition_text") or "")[:55]
        state = a.get("belief_state") or "—"
        conf = f"{float(a.get('confidence', 0)):.2f}" if a.get("confidence") is not None else "—"
        # list_recent() returns source_roles (list of distinct roles) + primary_source_role.
        # Show MULTI-SOURCE[...] when an assertion spans multiple source types (SO-5).
        src_roles = a.get("source_roles", [])
        if len(src_roles) > 1:
            src = f"MULTI-SOURCE[{','.join(src_roles)}]"
        elif src_roles:
            src = src_roles[0]
        else:
            src = a.get("source_role") or a.get("primary_source_role") or "—"
        speech = a.get("speech_act") or a.get("primary_speech_act") or "—"
        lines.append(f"| `{assertion_id}` | {prop} | {state} | {conf} | {src} | {speech} |")
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
            lines.append(f"  > {rationale[:120]}")
        # Show action params useful for the UI (issue_id, gap_id, assertion_id)
        params = a.get("params", {})
        if params:
            param_str = " | ".join(f"`{k}: {str(v)[:40]}`" for k, v in params.items() if v)
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
        self.current_matter_id: Optional[str] = None
        self.current_run_id: Optional[str] = None
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
            on_irys_created=lambda irys: setattr(self, "_irys_ref", irys),
            on_step=self._make_on_step(update_q, thinking),
            set_current_run_id=lambda rid: setattr(self, "current_run_id", rid),
            set_current_matter_id=lambda mid: setattr(self, "current_matter_id", mid),
            set_final_output=lambda o: setattr(self, "final_output", o),
            stop_event=self._stop_event,
        )

    def stream_investigation(
        self, query: str, repo_path: str
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
        self.current_run_id = None
        # Create a fresh per-call stop event so that stopping one investigation
        # cannot interfere with a subsequent one (shared-event reuse race).
        # stop_investigation() always sets self._stop_event, which after this
        # line points to THIS call's event — not a previous call's.
        self._stop_event = threading.Event()

        if not repo_path or not __import__("pathlib").Path(repo_path).exists():
            yield ("", "", "", "❌ Invalid repository path", "")
            return
        if not query.strip():
            yield ("", "", "", "❌ Please enter a query", "")
            return
        if not self.api_key:
            yield ("", "", "", "❌ No GEMINI_API_KEY. Set the env var or pass --api-key", "")
            return

        self.is_running = True
        thread = threading.Thread(
            target=self._run_thread,
            args=(query, repo_path, call_queue, call_thinking, call_citations),
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
                    status = f"⏳  {elapsed:.0f}s | {len(call_thinking)} steps | {len(call_citations)} citations"
                    yield (
                        "*Investigating...*",
                        "\n".join(call_thinking[-80:]),
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
                    status = (
                        f"✅  {elapsed:.0f}s | "
                        f"Docs: {state.documents_read} ({state.documents_from_cache} cached) | "
                        f"Reuse: {rate_str}"
                    )
                    # Replace raw thinking trace with structured ledger events so the
                    # Reasoning Trace tab shows durable, matter-model-backed content.
                    structured_trace = "\n".join(call_thinking[-80:])  # fallback
                    if self.current_matter_id and self.current_run_id:
                        try:
                            events: list[dict] = []
                            async def _collect_events(mid: str, rid: str) -> list[dict]:
                                collected: list[dict] = []
                                async for ev in self.backend().stream_run_events(mid, rid):
                                    collected.append(ev)
                                    if len(collected) >= 500:
                                        break
                                return collected
                            events = _run_async(
                                _collect_events(self.current_matter_id, self.current_run_id)
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
                    yield (
                        self.final_output,
                        structured_trace,
                        "\n".join(call_citations) or "—",
                        status,
                        self.current_matter_id or "—",
                    )
                    return

                elif update_type == "error":
                    self.is_running = False
                    yield (
                        "",
                        "\n".join(call_thinking[-80:]),
                        "",
                        f"❌ {data}",
                        "—",
                    )
                    return

            except queue.Empty:
                if self.is_running:
                    elapsed = time.time() - start_time
                    status = f"⏳  {elapsed:.0f}s | {len(call_thinking)} steps"
                    yield (
                        "*Investigating...*",
                        "\n".join(call_thinking[-80:]),
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
            # Run in the executor so the click handler returns without blocking
            # on SQLite's 5s busy_timeout.
            irys_ref = self._irys_ref  # snapshot before thread runs
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
            threading.Thread(target=_early_stop, daemon=True).start()
        # current_run_id intentionally NOT cleared here — see docstring.
        return gr.update()

    def load_overview(self, matter_id: str) -> str:
        if not matter_id or matter_id == "—":
            return "No matter loaded. Run an investigation first."
        try:
            data = _run_async(self.backend().get_overview(matter_id))
            return _fmt_overview(data)
        except Exception as exc:
            return f"Error loading overview: {exc}"

    def load_issues(self, matter_id: str) -> str:
        if not matter_id or matter_id == "—":
            return "No matter loaded."
        try:
            issues = _run_async(self.backend().list_issues(matter_id))
            return _fmt_issues(issues)
        except Exception as exc:
            return f"Error loading issues: {exc}"

    def load_assertions(self, matter_id: str) -> str:
        if not matter_id or matter_id == "—":
            return "No matter loaded."
        try:
            assertions = _run_async(self.backend().list_assertions(matter_id, limit=50))
            return _fmt_assertions(assertions)
        except Exception as exc:
            return f"Error loading assertions: {exc}"

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

    def load_quant(self, matter_id: str) -> str:
        if not matter_id or matter_id == "—":
            return "No matter loaded."
        try:
            quant_data = _run_async(self.backend().get_quant_summary(matter_id))
            return _fmt_quant(
                quant_data.get("payment_reconciliation", {}),
                quant_data.get("damages_waterfall", []),
            )
        except Exception as exc:
            return f"⚠️ Error loading quantitative data: {exc}"

    def do_correct_assertion(
        self, matter_id: str, assertion_id: str, new_state: str, reason: str
    ) -> str:
        if not matter_id or not assertion_id:
            return "Provide matter ID and assertion ID."
        if not new_state:
            return "Select a belief state."
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

    def do_redirect(self, matter_id: str, run_id: str, issue_id: str) -> str:
        if not matter_id or not run_id or not issue_id:
            return "Provide matter ID, run ID, and issue ID."
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
# Gradio app construction
# ---------------------------------------------------------------------------


def create_app(api_key: Optional[str] = None) -> gr.Blocks:
    state = AppState(api_key=api_key)

    with gr.Blocks(title="Irys RLM") as demo:
        gr.Markdown("# Irys RLM — Legal Intelligence System")

        # Shared matter_id state (populated after a run completes)
        matter_id_box = gr.Textbox(
            label="Active Matter ID",
            placeholder="Populated after first run",
            interactive=False,
            scale=1,
        )

        with gr.Tabs():

            # ============================================================
            # Tab 1: Run / Investigate
            # ============================================================
            with gr.TabItem("Run", id="run"):
                gr.Markdown(
                    "Point to a folder of legal documents and ask a question. "
                    "**Same folder = same matter model** — each run builds on prior state."
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        repo_path = gr.Textbox(
                            label="Repository Path",
                            placeholder="Full path to folder containing legal documents",
                        )
                        query = gr.Textbox(
                            label="Query",
                            placeholder="What do you want to investigate?",
                            lines=3,
                        )
                        with gr.Row():
                            submit_btn = gr.Button("Investigate", variant="primary", scale=4)
                            stop_btn = gr.Button("Stop", variant="stop", scale=1)
                    with gr.Column(scale=1):
                        status_box = gr.Textbox(
                            label="Status",
                            lines=3,
                            interactive=False,
                            elem_classes=["status-bar"],
                        )

                with gr.Tabs():
                    with gr.TabItem("Analysis"):
                        run_output = gr.Markdown()
                    with gr.TabItem("Reasoning Trace (live)"):
                        trace_box = gr.Textbox(
                            label="Ledger Events (structured after run; live steps during run)",
                            lines=35,
                            interactive=False,
                            autoscroll=True,
                            elem_classes=["mono"],
                        )
                    with gr.TabItem("Citations"):
                        citations_box = gr.Textbox(
                            label="Sources",
                            lines=20,
                            interactive=False,
                        )

                run_outputs = [run_output, trace_box, citations_box, status_box, matter_id_box]

                submit_btn.click(
                    fn=state.stream_investigation,
                    inputs=[query, repo_path],
                    outputs=run_outputs,
                )
                stop_btn.click(fn=state.stop_investigation, inputs=[], outputs=[])

                gr.Examples(
                    examples=[
                        ["What are the key claims and defenses in this dispute?", ""],
                        ["What damages are claimed and what is the evidentiary basis?", ""],
                        ["Who are the key parties and what documents are missing?", ""],
                    ],
                    inputs=[query, repo_path],
                )

            # ============================================================
            # Tab 2: Overview
            # ============================================================
            with gr.TabItem("Overview", id="overview"):
                gr.Markdown("**Landing page** — matter state, SO metrics, weakest issues, open gaps.")
                with gr.Row():
                    refresh_overview_btn = gr.Button("Refresh Overview", variant="secondary")
                overview_md = gr.Markdown("Run an investigation first to populate this panel.")

                def _refresh_overview(mid):
                    return state.load_overview(mid)

                refresh_overview_btn.click(
                    fn=_refresh_overview,
                    inputs=[matter_id_box],
                    outputs=[overview_md],
                )

            # ============================================================
            # Tab 3: Issues / Proof (SO-4)
            # ============================================================
            with gr.TabItem("Issues", id="issues"):
                gr.Markdown("**SO-4** — Issue tree with coverage fraction, proof status, supporting/attacking counts.")
                with gr.Row():
                    refresh_issues_btn = gr.Button("Refresh Issues", variant="secondary")
                issues_md = gr.Markdown("Run an investigation first.")

                def _refresh_issues(mid):
                    return state.load_issues(mid)

                refresh_issues_btn.click(
                    fn=_refresh_issues,
                    inputs=[matter_id_box],
                    outputs=[issues_md],
                )

            # ============================================================
            # Tab 4: Assertions / Evidence (SO-2)
            # ============================================================
            with gr.TabItem("Assertions", id="assertions"):
                gr.Markdown("**SO-2** — Typed assertion table. Correct assertions inline.")
                with gr.Row():
                    refresh_assertions_btn = gr.Button("Refresh Assertions", variant="secondary")
                assertions_md = gr.Markdown("Run an investigation first.")

                refresh_assertions_btn.click(
                    fn=lambda mid: state.load_assertions(mid),
                    inputs=[matter_id_box],
                    outputs=[assertions_md],
                )

                gr.Markdown("### Correct an Assertion")
                with gr.Row():
                    correction_assertion_id = gr.Textbox(label="Assertion ID", scale=2)
                    correction_new_state = gr.Dropdown(
                        label="New Belief State",
                        choices=[
                            "alleged", "argued", "admitted", "operative", "performed",
                            "disputed", "superseded", "withdrawn", "inferred", "resolved",
                        ],
                        scale=1,
                    )
                correction_reason = gr.Textbox(label="Reason", lines=2)
                correction_btn = gr.Button("Apply Correction", variant="primary")
                correction_result = gr.Textbox(label="Result", interactive=False)

                def _correct_and_refresh(mid, aid, new_state_str, reason):
                    """Apply correction and refresh assertions + issues + overview (SO-2).

                    Belief revision after a correction can change proof states (which
                    affects issue coverage), so all three panels must refresh.
                    """
                    result_text = state.do_correct_assertion(mid, aid, new_state_str, reason)
                    if result_text.startswith("✅"):
                        return (
                            result_text,
                            state.load_assertions(mid),
                            state.load_issues(mid),
                            state.load_overview(mid),
                        )
                    return result_text, gr.update(), gr.update(), gr.update()

                correction_btn.click(
                    fn=_correct_and_refresh,
                    inputs=[matter_id_box, correction_assertion_id, correction_new_state, correction_reason],
                    outputs=[correction_result, assertions_md, issues_md, overview_md],
                )

            # ============================================================
            # Tab 5: Gaps & Steering (SO-7 + SO-3)
            # ============================================================
            with gr.TabItem("Gaps & Steering", id="gaps"):
                gr.Markdown("**SO-7** — Open gaps and missing documents. **SO-3** — Redirect and steering controls.")
                with gr.Row():
                    refresh_gaps_btn = gr.Button("Refresh Gaps", variant="secondary")
                gaps_md = gr.Markdown("Run an investigation first.")

                gr.Markdown("### Redirect Investigation")
                gr.Markdown(
                    "Redirect the current run to a specific issue. "
                    "**Refresh Gaps** auto-populates the Issue ID from the top steering recommendation."
                )
                with gr.Row():
                    redirect_run_id = gr.Textbox(label="Run ID", scale=2)
                    fill_run_id_btn = gr.Button("← Use Active Run", scale=1)
                    redirect_issue_id = gr.Textbox(label="Issue ID to redirect toward", scale=2)

                # Registered here (after redirect_issue_id is defined) so that
                # load_gaps() can auto-populate the redirect form from the top
                # steering recommendation (makes the steering surface actionable — SO-3).
                refresh_gaps_btn.click(
                    fn=lambda mid: state.load_gaps(mid),
                    inputs=[matter_id_box],
                    outputs=[gaps_md, redirect_issue_id],
                )
                redirect_btn = gr.Button("Redirect", variant="primary")
                redirect_result = gr.Textbox(label="Result", interactive=False)

                # Populate run_id from the active investigation (SO-3 steerability).
                fill_run_id_btn.click(
                    fn=lambda: state.current_run_id or "",
                    inputs=[],
                    outputs=[redirect_run_id],
                )
                redirect_btn.click(
                    fn=state.do_redirect,
                    inputs=[matter_id_box, redirect_run_id, redirect_issue_id],
                    outputs=[redirect_result],
                )

            # ============================================================
            # Tab 6: Quant (SO-6)
            # ============================================================
            with gr.TabItem("Quant", id="quant"):
                gr.Markdown(
                    "**SO-6** — Quantitative intelligence: payment reconciliation, "
                    "damages waterfall, numeric conflicts. Numbers from the matter model, "
                    "not extracted from prose."
                )
                with gr.Row():
                    refresh_quant_btn = gr.Button("Refresh Quant", variant="secondary")
                quant_md = gr.Markdown("Run an investigation first.")

                refresh_quant_btn.click(
                    fn=lambda mid: state.load_quant(mid),
                    inputs=[matter_id_box],
                    outputs=[quant_md],
                )

    return demo


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Irys RLM UI")
    parser.add_argument("--api-key", help="Gemini API key")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("⚠️  No GEMINI_API_KEY — set it or pass --api-key")

    demo = create_app(api_key=api_key)
    demo.launch(
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        css=".mono textarea { font-family: monospace; font-size: 12px; }",
    )


if __name__ == "__main__":
    main()
