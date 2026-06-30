"""
Detailed RLM debugging script - captures every step for comparison with Claude Code.
"""

import sys
import os
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()

from irys.core.models import GeminiClient
from irys.core.repository import MatterRepository
from irys.rlm.engine import RLMEngine, RLMConfig
from irys.rlm.state import StepType

# Detailed logging
debug_log = []

def log_step(step_type: str, content: str, details: dict = None):
    """Log a step with timestamp."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "step_type": step_type,
        "content": content,
        "details": details or {}
    }
    debug_log.append(entry)
    print(f"[{step_type}] {content}")
    if details:
        for k, v in details.items():
            if isinstance(v, str) and len(v) > 200:
                v = v[:200] + "..."
            print(f"    {k}: {v}")


def on_step_callback(step):
    """Callback for each RLM step."""
    log_step(
        step.step_type.value,
        step.content,
        step.details
    )


async def run_debug_query(query: str, repo_path: str):
    """Run a query with detailed logging."""
    print("=" * 80)
    print(f"QUERY: {query}")
    print(f"REPO: {repo_path}")
    print("=" * 80)

    # Initialize
    client = GeminiClient()
    config = RLMConfig(
        max_depth=3,
        max_leads_per_level=3,
        max_iterations=10,
    )
    engine = RLMEngine(
        gemini_client=client,
        config=config,
        on_step=on_step_callback,
    )

    repo = MatterRepository(repo_path)

    # Log initial state
    stats = repo.get_stats()
    log_step("INIT", f"Repository loaded", {
        "total_files": stats.total_files,
        "file_types": stats.files_by_type,
    })

    # Run investigation
    start_time = time.time()
    state = await engine.investigate(query, repo_path)
    duration = time.time() - start_time

    # Log final state
    log_step("COMPLETE", f"Investigation finished in {duration:.1f}s", {
        "documents_read": state.documents_read,
        "searches_performed": state.searches_performed,
        "citations": len(state.citations),
        "facts_found": len(state.findings.get("accumulated_facts", [])),
    })

    # Log all facts found
    print("\n" + "=" * 80)
    print("FACTS EXTRACTED:")
    print("=" * 80)
    for i, fact in enumerate(state.findings.get("accumulated_facts", [])[:30]):
        print(f"  [{i+1}] {fact[:150]}...")

    # Log all citations
    print("\n" + "=" * 80)
    print("CITATIONS:")
    print("=" * 80)
    for c in state.citations[:15]:
        print(f"  - {Path(c.document).name}: {c.text[:100]}...")

    # Log final output
    print("\n" + "=" * 80)
    print("FINAL OUTPUT:")
    print("=" * 80)
    output = state.findings.get("final_output", "No output")
    print(output[:3000])

    # Check for key figures
    print("\n" + "=" * 80)
    print("KEY FIGURE CHECK:")
    print("=" * 80)
    all_text = str(state.findings) + output
    checks = [
        ("$835,931 (initial cost)", "835,931" in all_text),
        ("$3,879,972.80 (actual cost)", "3,879,972" in all_text),
        ("2,800 hours (initial)", "2,800" in all_text),
        ("26,409 hours (actual)", "26,409" in all_text),
        ("9.4x overrun", "9.4" in all_text),
        ("10 months downtime", "10 month" in all_text.lower() or "nearly 10" in all_text.lower()),
    ]
    for label, found in checks:
        status = "[OK]" if found else "[MISS]"
        print(f"  {status} {label}")

    # Save debug log
    log_file = f"debug_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_file, "w") as f:
        json.dump({
            "query": query,
            "duration": duration,
            "steps": debug_log,
            "facts": state.findings.get("accumulated_facts", []),
            "citations": [{"doc": c.document, "text": c.text} for c in state.citations],
        }, f, indent=2)
    print(f"\nDebug log saved to: {log_file}")

    return state


async def main():
    query = "What was Gulfstream's initial cost estimate and how did it compare to actual costs?"
    repo_path = r"C:\Users\devan\Downloads\CITIOM v Gulfstream\documents"

    await run_debug_query(query, repo_path)


if __name__ == "__main__":
    asyncio.run(main())
