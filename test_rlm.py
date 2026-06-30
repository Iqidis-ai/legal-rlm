#!/usr/bin/env python3
"""Test the RLM engine end-to-end."""

import sys
import os
import asyncio

sys.path.insert(0, "src")

# Require API key from environment
if not os.environ.get("GEMINI_API_KEY"):
    print("ERROR: GEMINI_API_KEY environment variable required")
    sys.exit(1)

from irys.core.models import GeminiClient
from irys.rlm.engine import RLMEngine, RLMConfig
from irys.rlm.state import ThinkingStep, Citation

# Callback to print thinking steps
def on_step(step: ThinkingStep):
    # Avoid unicode issues on Windows
    type_map = {
        "thinking": "[THINK]",
        "search": "[SEARCH]",
        "reading": "[READ]",
        "finding": "[FIND]",
        "replan": "[REPLAN]",
        "synthesis": "[SYNTH]",
        "error": "[ERROR]",
    }
    prefix = type_map.get(step.step_type.value, "[?]")
    indent = "  " * step.depth
    content = step.content.encode('ascii', 'replace').decode('ascii')
    print(f"{indent}{prefix} {content}")

def on_citation(citation: Citation):
    doc = citation.document.encode('ascii', 'replace').decode('ascii')
    text = citation.text[:50].encode('ascii', 'replace').decode('ascii')
    print(f"  [CITE] {doc}, p. {citation.page}: {text}...")

async def main():
    print("=" * 60)
    print("RLM ENGINE TEST")
    print("=" * 60)

    client = GeminiClient()

    # Reduce scope for quick test
    config = RLMConfig(
        max_depth=2,
        max_leads_per_level=2,
        max_documents_per_search=3,
    )

    engine = RLMEngine(
        gemini_client=client,
        config=config,
        on_step=on_step,
        on_citation=on_citation,
    )

    query = "What are the key claims in this dispute?"
    repo_path = r"C:\Users\devan\Downloads\CITIOM v Gulfstream"

    print(f"\nQuery: {query}")
    print(f"Repository: {repo_path}")
    print("\n" + "-" * 60)
    print("INVESTIGATION LOG:")
    print("-" * 60)

    try:
        state = await engine.investigate(query, repo_path)

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Status: {state.status}")
        print(f"Documents read: {state.documents_read}")
        print(f"Searches: {state.searches_performed}")
        print(f"Citations: {len(state.citations)}")
        print(f"Max depth: {state.max_depth_reached}")

        if state.duration_seconds:
            print(f"Duration: {state.duration_seconds:.1f}s")

        print("\n" + "-" * 60)
        print("FINAL OUTPUT:")
        print("-" * 60)
        output = state.findings.get("final_output", "No output")
        # Handle unicode for Windows
        output = output.encode('ascii', 'replace').decode('ascii')
        print(output[:2000])
        if len(output) > 2000:
            print(f"\n... [{len(output) - 2000} more chars]")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
