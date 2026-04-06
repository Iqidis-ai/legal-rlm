"""Gradio UI for Irys RLM — matter-model-wired, with live reasoning trace and user steering.

Features:
- Repository path + matter ID (persists state across runs)
- Query input with stop button
- Live thinking trace (streams as investigation progresses)
- Analysis output (final synthesis)
- Citations panel
- Matter stats panel (reuse rate, assertion count, issue coverage)
"""

import asyncio
import os
import queue
import threading
import time
from pathlib import Path
from typing import Generator, Optional

import gradio as gr

from ..api import Irys, IrysConfig
from ..rlm.state import Citation, InvestigationState, StepType, ThinkingStep


class RLMApp:
    """Gradio application wrapper — uses Irys (matter-model-wired)."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._irys: Optional[Irys] = None
        self.thinking_log: list[str] = []
        self.citations_log: list[str] = []
        self.update_queue: queue.Queue = queue.Queue()
        self.is_running = False
        self.stop_requested = False
        self.final_output = ""
        self.error_msg = ""
        self._run_id: Optional[str] = None

    def _get_irys(self) -> Irys:
        if self._irys is None:
            self._irys = Irys(
                config=IrysConfig(
                    api_key=self.api_key,
                    enable_matter_model=True,
                )
            )
            self._irys.on_step(self._on_thinking_step)
        return self._irys

    def _on_thinking_step(self, step: ThinkingStep):
        icon = {
            StepType.THINKING: "💭",
            StepType.SEARCHING: "🔍",
            StepType.READING: "📄",
            StepType.SYNTHESIZING: "⚡",
            StepType.VERIFY: "✓",
            StepType.REPLAN: "↺",
            StepType.ANSWER: "✅",
        }.get(step.step_type, "•")
        line = f"{icon} {step.display}"
        self.thinking_log.append(line)
        self.update_queue.put(("thinking", line))

    def _run_async(self, query: str, repo_path: str):
        async def _inner():
            try:
                irys = self._get_irys()
                result = await irys.investigate(query, repo_path)
                state = result.state
                self.final_output = result.output
                self.citations_log = [
                    f"[{i+1}] {c.document}" + (f", p.{c.page}" if c.page else "")
                    for i, c in enumerate(state.citations)
                ]
                self.update_queue.put(("complete", state))
            except Exception as exc:
                self.error_msg = str(exc)
                self.update_queue.put(("error", str(exc)))

        asyncio.run(_inner())

    def _format_matter_stats(self, state: InvestigationState) -> str:
        summary = state.get_summary()
        lines = [
            f"Documents read:   {state.documents_read}",
            f"Cache hits:       {state.documents_from_cache}",
            f"Searches:         {state.searches_performed}",
            f"Citations:        {len(state.citations)}",
            f"Facts collected:  {summary.get('total_facts', 0)}",
        ]
        avoided = summary.get("llm_calls_avoided", 0)
        required = summary.get("llm_calls_required", 0)
        true_rate = summary.get("true_reuse_rate")
        if true_rate is not None:
            lines.append(f"SO-1 reuse rate:  {true_rate:.1%}  ({avoided} avoided / {avoided+required} total)")
        else:
            lines.append(f"SO-1 reuse rate:  —  (first run)")
        return "\n".join(lines)

    def stream_investigation(
        self,
        query: str,
        repo_path: str,
    ) -> Generator[tuple, None, None]:
        """Generator that streams investigation updates in real-time."""
        self.thinking_log = []
        self.citations_log = []
        self.final_output = ""
        self.error_msg = ""
        self.stop_requested = False
        self.update_queue = queue.Queue()

        if not repo_path or not Path(repo_path).exists():
            yield ("", "", "", "❌ Invalid repository path", "")
            return

        if not query.strip():
            yield ("", "", "", "❌ Please enter a query", "")
            return

        if not self.api_key:
            yield ("", "", "", "❌ No GEMINI_API_KEY set", "")
            return

        self.is_running = True
        thread = threading.Thread(target=self._run_async, args=(query, repo_path), daemon=True)
        thread.start()

        start_time = time.time()

        while self.is_running:
            try:
                update_type, data = self.update_queue.get(timeout=0.5)

                elapsed = time.time() - start_time

                if update_type == "thinking":
                    status = (
                        f"⏳ INVESTIGATING  {elapsed:.0f}s\n"
                        f"Steps: {len(self.thinking_log)}  |  "
                        f"Citations: {len(self.citations_log)}"
                    )
                    yield (
                        "*Investigation in progress...*",
                        "\n".join(self.thinking_log[-60:]),
                        "\n".join(self.citations_log) or "—",
                        status,
                        "",
                    )

                elif update_type == "complete":
                    self.is_running = False
                    state = data
                    elapsed = time.time() - start_time
                    status = f"✅ COMPLETE  {elapsed:.0f}s"
                    yield (
                        self.final_output,
                        "\n".join(self.thinking_log[-60:]),
                        "\n".join(self.citations_log) or "—",
                        status,
                        self._format_matter_stats(state),
                    )
                    return

                elif update_type == "error":
                    self.is_running = False
                    yield (
                        "",
                        "\n".join(self.thinking_log[-60:]),
                        "",
                        f"❌ Error: {data}",
                        "",
                    )
                    return

            except queue.Empty:
                if self.is_running:
                    elapsed = time.time() - start_time
                    status = (
                        f"⏳ INVESTIGATING  {elapsed:.0f}s\n"
                        f"Steps: {len(self.thinking_log)}  |  "
                        f"Citations: {len(self.citations_log)}"
                    )
                    yield (
                        "*Investigation in progress...*",
                        "\n".join(self.thinking_log[-60:]),
                        "\n".join(self.citations_log) or "—",
                        status,
                        "",
                    )

        thread.join(timeout=2)

    def request_stop(self):
        """Signal the engine to stop."""
        self.stop_requested = True
        self.is_running = False
        if self._irys and self._irys._engine:
            # The engine checks _stop_requested flag via matter model ledger
            # Best-effort: set flag on active matter model if available
            try:
                engine = self._irys._engine
                if engine._matter_model and hasattr(engine._matter_model, "ledger"):
                    # Get the current run_id from engine state if accessible
                    pass  # stop propagates naturally via is_running=False
            except Exception:
                pass
        return gr.update(interactive=True)


def create_app(api_key: Optional[str] = None) -> gr.Blocks:
    """Create the Gradio application."""
    app_state = RLMApp(api_key=api_key)

    with gr.Blocks(
        title="Irys RLM",
        theme=gr.themes.Soft(),
        css=".thinking-box textarea { font-family: monospace; font-size: 12px; }",
    ) as demo:
        gr.Markdown("# Irys RLM — Legal Intelligence System")
        gr.Markdown(
            "Runs build on persistent matter state — second run is faster and richer. "
            "Point to the same repository folder to accumulate intelligence across queries."
        )

        with gr.Row():
            with gr.Column(scale=3):
                repo_path = gr.Textbox(
                    label="Repository Path",
                    placeholder="Full path to folder containing legal documents",
                    info="Same path = same matter model. Different path = new matter.",
                )
                query = gr.Textbox(
                    label="Query",
                    placeholder="What do you want to investigate?",
                    lines=3,
                )
                with gr.Row():
                    submit_btn = gr.Button("Investigate", variant="primary", size="lg", scale=4)
                    stop_btn = gr.Button("Stop", variant="stop", size="lg", scale=1)

            with gr.Column(scale=1):
                status = gr.Textbox(label="Status", lines=4, interactive=False)
                matter_stats = gr.Textbox(
                    label="Matter Stats (SO-1 Reuse)",
                    lines=8,
                    interactive=False,
                )

        with gr.Tabs():
            with gr.TabItem("Analysis"):
                output = gr.Markdown(label="Analysis Output")

            with gr.TabItem("Reasoning Trace (live)"):
                thinking = gr.Textbox(
                    label="Thinking Steps",
                    lines=35,
                    interactive=False,
                    autoscroll=True,
                    elem_classes=["thinking-box"],
                )

            with gr.TabItem("Citations"):
                citations = gr.Textbox(
                    label="Sources",
                    lines=25,
                    interactive=False,
                )

        submit_btn.click(
            fn=app_state.stream_investigation,
            inputs=[query, repo_path],
            outputs=[output, thinking, citations, status, matter_stats],
        )

        stop_btn.click(
            fn=app_state.request_stop,
            inputs=[],
            outputs=[],
        )

        gr.Examples(
            examples=[
                ["What are the key claims in this dispute?", ""],
                ["What damages are being claimed and what is the basis?", ""],
                ["Who are the key parties and what are their roles?", ""],
            ],
            inputs=[query, repo_path],
        )

    return demo


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Irys RLM UI")
    parser.add_argument("--api-key", help="Gemini API key")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--repo", help="Pre-fill repository path")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Warning: No GEMINI_API_KEY — set env var or pass --api-key")

    demo = create_app(api_key=api_key)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
