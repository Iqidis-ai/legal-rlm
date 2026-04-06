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

# Dedicated executor for running async backend calls from sync Gradio callbacks.
# asyncio.run() can conflict with Gradio's internal event loop in some versions;
# using a thread + new event loop is reliably safe.
_ASYNC_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="irys_async")


def _run_async(coro):
    """Run a coroutine from a sync context without conflicting with existing loops."""
    future = _ASYNC_EXECUTOR.submit(asyncio.run, coro)
    return future.result(timeout=30)


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
        f"\n**SO-1 reuse:** {f'{reuse:.1%}' if reuse else '—'}  |  "
        f"**SO-2 structure:** {f'{struct:.1%}' if struct else '—'}  |  "
        f"**SO-4 coverage:** {f'{cov:.1%}' if cov else '—'}  |  "
        f"**SO-5 source calibration:** {f'{src:.1%}' if src else '—'}"
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
    lines = ["| Issue | Coverage | Proof | Supporting | Attacking |",
             "|-------|----------|-------|------------|-----------|"]
    for iss in issues:
        title = (iss.get("title") or iss.get("id", "?"))[:40]
        cov = _fmt_coverage(iss.get("coverage_fraction"))
        proof = iss.get("proof_status") or "—"
        sup = iss.get("supporting_count", "—")
        atk = iss.get("attacking_count", "—")
        lines.append(f"| {title} | {cov} | {proof} | {sup} | {atk} |")
    return "\n".join(lines)


def _fmt_assertions(assertions: list) -> str:
    if not assertions:
        return "No assertions."
    lines = ["| Proposition | State | Confidence | Source | Speech Act |",
             "|-------------|-------|------------|--------|------------|"]
    for a in assertions:
        prop = (a.get("proposition_text") or "")[:60]
        state = a.get("belief_state") or "—"
        conf = f"{float(a.get('confidence', 0)):.2f}" if a.get("confidence") is not None else "—"
        src = a.get("source_role") or "—"
        speech = a.get("speech_act") or "—"
        lines.append(f"| {prop} | {state} | {conf} | {src} | {speech} |")
    return "\n".join(lines)


def _fmt_gaps(gaps: list, clarifications: list) -> str:
    parts = []
    if gaps:
        parts.append("### Open Gaps")
        for g in gaps:
            desc = g.get("description") or g.get("gap_type", "?")
            mat = g.get("materiality") or ""
            mat_str = f" [{mat}]" if mat else ""
            parts.append(f"- {desc}{mat_str}")
    if clarifications:
        parts.append("\n### Pending Clarifications")
        for c in clarifications:
            q = c.get("question_text") or c.get("question", "?")
            impact = c.get("expected_impact") or ""
            parts.append(f"- **{q}**" + (f"\n  *Impact: {impact}*" if impact else ""))
    return "\n".join(parts) if parts else "No open gaps or clarifications."


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
                        runs = engine._matter_model.ledger.recent_runs(1)
                        if runs and runs[0].get("status") == "running":
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
        """Run investigation in a background thread."""
        async def _inner():
            irys = self.backend()._get_irys()
            self._irys_ref = irys
            irys.on_step(self._make_on_step(update_q, thinking))
            try:
                result = await irys.investigate(query, repo_path)
                state = result.state
                engine = irys._engine
                mm = engine._matter_model if engine else None
                self.current_matter_id = mm.matter_id if mm else None
                self.current_run_id = getattr(state, "_run_id", None)
                citations.extend(
                    f"[{i+1}] {c.document}" + (f", p.{c.page}" if c.page else "")
                    for i, c in enumerate(state.citations)
                )
                self.final_output = result.output
                update_q.put(("complete", state))
            except Exception as exc:
                update_q.put(("error", str(exc)))

        asyncio.run(_inner())

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

        while self.is_running:
            try:
                update_type, data = self.update_queue.get(timeout=0.5)
                elapsed = time.time() - start_time

                if update_type == "thinking":
                    status = f"⏳  {elapsed:.0f}s | {len(self.thinking_log)} steps | {len(self.citations_log)} citations"
                    yield (
                        "*Investigating...*",
                        "\n".join(self.thinking_log[-80:]),
                        "\n".join(self.citations_log) or "—",
                        status,
                        self.current_matter_id or "—",
                    )

                elif update_type == "complete":
                    self.is_running = False
                    state = data
                    elapsed = time.time() - start_time
                    summary = state.get_summary()
                    avoided = summary.get("llm_calls_avoided", 0)
                    required = summary.get("llm_calls_required", 0)
                    true_rate = summary.get("true_reuse_rate")
                    rate_str = f"{true_rate:.1%}" if true_rate is not None else "—"
                    status = (
                        f"✅  {elapsed:.0f}s | "
                        f"Docs: {state.documents_read} ({state.documents_from_cache} cached) | "
                        f"Reuse: {rate_str}"
                    )
                    yield (
                        self.final_output,
                        "\n".join(self.thinking_log[-80:]),
                        "\n".join(self.citations_log) or "—",
                        status,
                        self.current_matter_id or "—",
                    )
                    return

                elif update_type == "error":
                    self.is_running = False
                    yield (
                        "",
                        "\n".join(self.thinking_log[-80:]),
                        "",
                        f"❌ {data}",
                        "—",
                    )
                    return

            except queue.Empty:
                if self.is_running:
                    elapsed = time.time() - start_time
                    status = f"⏳  {elapsed:.0f}s | {len(self.thinking_log)} steps"
                    yield (
                        "*Investigating...*",
                        "\n".join(self.thinking_log[-80:]),
                        "\n".join(self.citations_log) or "—",
                        status,
                        self.current_matter_id or "—",
                    )

        thread.join(timeout=2)

    def stop_investigation(self):
        """Stop the running investigation.

        Sets is_running=False to break the UI generator, and calls
        ledger.request_stop() on the run_id so the engine honors it on
        the next iteration check.
        """
        self.is_running = False
        run_id = self.current_run_id
        if run_id and self._irys_ref:
            try:
                engine = self._irys_ref._engine
                if engine and engine._matter_model:
                    engine._matter_model.ledger.request_stop(run_id)
            except Exception:
                pass
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

    def load_gaps(self, matter_id: str) -> str:
        if not matter_id or matter_id == "—":
            return "No matter loaded."
        try:
            gaps = _run_async(self.backend().list_gaps(matter_id))
            clarifications = _run_async(self.backend().list_clarifications(matter_id))
            return _fmt_gaps(gaps, clarifications)
        except Exception as exc:
            return f"Error loading gaps: {exc}"

    def do_correct_assertion(
        self, matter_id: str, assertion_id: str, new_state: str, reason: str
    ) -> str:
        if not matter_id or not assertion_id:
            return "Provide matter ID and assertion ID."
        try:
            result = _run_async(
                self.backend().correct_assertion(matter_id, assertion_id, new_state, reason)
            )
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
            return f"✅ Redirected: {result}"
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
                            label="Thinking Steps",
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
                        choices=["accepted", "rejected", "disputed", "superseded", "withdrawn"],
                        scale=1,
                    )
                correction_reason = gr.Textbox(label="Reason", lines=2)
                correction_btn = gr.Button("Apply Correction", variant="primary")
                correction_result = gr.Textbox(label="Result", interactive=False)

                correction_btn.click(
                    fn=state.do_correct_assertion,
                    inputs=[matter_id_box, correction_assertion_id, correction_new_state, correction_reason],
                    outputs=[correction_result],
                )

            # ============================================================
            # Tab 5: Gaps & Steering (SO-7 + SO-3)
            # ============================================================
            with gr.TabItem("Gaps & Steering", id="gaps"):
                gr.Markdown("**SO-7** — Open gaps and missing documents. **SO-3** — Redirect and steering controls.")
                with gr.Row():
                    refresh_gaps_btn = gr.Button("Refresh Gaps", variant="secondary")
                gaps_md = gr.Markdown("Run an investigation first.")

                refresh_gaps_btn.click(
                    fn=lambda mid: state.load_gaps(mid),
                    inputs=[matter_id_box],
                    outputs=[gaps_md],
                )

                gr.Markdown("### Redirect Investigation")
                gr.Markdown("Redirect the current run to focus on a specific issue.")
                with gr.Row():
                    redirect_run_id = gr.Textbox(label="Run ID (from Active Matter)", scale=2)
                    redirect_issue_id = gr.Textbox(label="Issue ID to redirect toward", scale=2)
                redirect_btn = gr.Button("Redirect", variant="primary")
                redirect_result = gr.Textbox(label="Result", interactive=False)

                redirect_btn.click(
                    fn=state.do_redirect,
                    inputs=[matter_id_box, redirect_run_id, redirect_issue_id],
                    outputs=[redirect_result],
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
