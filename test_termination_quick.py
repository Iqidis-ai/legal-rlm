"""Quick test for early termination - 2 queries only."""

import sys
import os
import asyncio
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Require API key from environment - never hardcode keys
if not os.environ.get("GEMINI_API_KEY"):
    print("ERROR: GEMINI_API_KEY environment variable required")
    sys.exit(1)

from irys.core.models import GeminiClient
from irys.rlm.engine import RLMEngine, RLMConfig
from irys.rlm.state import classify_query

REPO_PATH = r"C:\Users\devan\Downloads\CITIOM v Gulfstream\documents"

# Test just 2 queries - factual vs analytical
TEST_QUERIES = [
    ("What is the serial number of the aircraft?", "factual"),
    ("What are the key issues in this dispute?", "analytical"),
]


async def run_test(query: str, expected_type: str) -> dict:
    """Run a single query and collect metrics."""
    print(f"\n{'='*60}")
    print(f"Query: {query}")

    classification = classify_query(query)
    print(f"Classified as: {classification['type']} (complexity: {classification['complexity']})")

    client = GeminiClient(api_key=os.environ["GEMINI_API_KEY"])
    config = RLMConfig(max_iterations=10, min_depth=1)

    termination_reason = ""

    def on_step(step):
        nonlocal termination_reason
        if "Early termination" in step.content:
            termination_reason = step.content
        # Only print key steps
        if step.step_type.value in ["thinking", "replan"]:
            print(f"  {step.display[:80]}")

    engine = RLMEngine(gemini_client=client, config=config, on_step=on_step)

    start = time.time()
    state = await engine.investigate(query, REPO_PATH)
    elapsed = time.time() - start

    print(f"\nResults:")
    print(f"  Type: {classification['type']}")
    print(f"  Iterations: {len(state.facts_per_iteration)}")
    print(f"  Facts/iter: {state.facts_per_iteration}")
    print(f"  Citations: {len(state.citations)}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Termination: {termination_reason or 'Max iterations/no leads'}")

    return {
        "query_type": classification["type"],
        "iterations": len(state.facts_per_iteration),
        "elapsed": elapsed,
        "termination": termination_reason,
    }


async def main():
    print("="*60)
    print("EARLY TERMINATION QUICK TEST")
    print("="*60)

    results = []
    for query, expected in TEST_QUERIES:
        result = await run_test(query, expected)
        results.append(result)

    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)

    factual = [r for r in results if r["query_type"] == "factual"]
    analytical = [r for r in results if r["query_type"] == "analytical"]

    if factual and analytical:
        f_iters = factual[0]["iterations"]
        a_iters = analytical[0]["iterations"]
        print(f"Factual iterations:   {f_iters}")
        print(f"Analytical iterations: {a_iters}")

        if f_iters < a_iters:
            print("\n✓ SUCCESS: Factual queries terminate faster than analytical!")
        elif f_iters == a_iters:
            print("\n○ SAME: Both terminated at same iteration (may be OK)")
        else:
            print("\n✗ UNEXPECTED: Factual took longer than analytical")


if __name__ == "__main__":
    asyncio.run(main())
