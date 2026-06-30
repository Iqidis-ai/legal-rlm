"""Test early termination improvements.

Tests:
1. Diminishing returns detection
2. Query complexity-aware termination thresholds
"""

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
from irys.rlm.state import classify_query, QueryType

REPO_PATH = r"C:\Users\devan\Downloads\CITIOM v Gulfstream\documents"

# Test queries of different types
TEST_QUERIES = [
    # Factual - should terminate quickly
    ("What is the serial number of the aircraft?", "factual"),
    ("When did the inspection begin?", "factual"),

    # Analytical - medium termination
    ("What are the key issues in this dispute?", "analytical"),

    # Evaluative - thorough investigation
    ("Evaluate the strengths and weaknesses of CITIOM's claims.", "evaluative"),
]


async def run_single_test(query: str, expected_type: str) -> dict:
    """Run a single query and collect metrics."""
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"Expected type: {expected_type}")

    # Classify query
    classification = classify_query(query)
    print(f"Detected type: {classification['type']}, complexity: {classification['complexity']}")

    client = GeminiClient(api_key=os.environ["GEMINI_API_KEY"])
    config = RLMConfig(
        max_iterations=10,  # Cap iterations for testing
        min_depth=1,
    )

    termination_reasons = []

    def on_step(step):
        if "Early termination" in step.content:
            termination_reasons.append(step.content)
        print(f"  {step.display}")

    engine = RLMEngine(
        gemini_client=client,
        config=config,
        on_step=on_step,
    )

    start = time.time()
    state = await engine.investigate(query, REPO_PATH)
    elapsed = time.time() - start

    result = {
        "query": query,
        "expected_type": expected_type,
        "detected_type": classification["type"],
        "complexity": classification["complexity"],
        "iterations": len(state.facts_per_iteration),
        "facts_per_iteration": state.facts_per_iteration,
        "total_facts": len(state.findings.get("accumulated_facts", [])),
        "citations": len(state.citations),
        "elapsed_seconds": round(elapsed, 1),
        "termination_reason": termination_reasons[-1] if termination_reasons else "Max iterations",
    }

    print(f"\nResults:")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Facts per iteration: {result['facts_per_iteration']}")
    print(f"  Total facts: {result['total_facts']}")
    print(f"  Citations: {result['citations']}")
    print(f"  Time: {result['elapsed_seconds']}s")
    print(f"  Termination: {result['termination_reason']}")

    return result


async def main():
    print("="*60)
    print("EARLY TERMINATION TEST")
    print("="*60)
    print("\nTesting query complexity awareness and diminishing returns...\n")

    results = []

    for query, expected_type in TEST_QUERIES:
        try:
            result = await run_single_test(query, expected_type)
            results.append(result)
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"query": query, "error": str(e)})

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Query Type':<12} {'Iters':<6} {'Facts':<8} {'Citations':<10} {'Time':<8}")
    print("-"*60)

    for r in results:
        if "error" not in r:
            print(f"{r['detected_type']:<12} {r['iterations']:<6} {r['total_facts']:<8} {r['citations']:<10} {r['elapsed_seconds']:<8}s")

    # Verify expectations
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)

    factual_results = [r for r in results if r.get("detected_type") == "factual" and "error" not in r]
    analytical_results = [r for r in results if r.get("detected_type") == "analytical" and "error" not in r]
    evaluative_results = [r for r in results if r.get("detected_type") == "evaluative" and "error" not in r]

    if factual_results and analytical_results:
        avg_factual_iters = sum(r["iterations"] for r in factual_results) / len(factual_results)
        avg_analytical_iters = sum(r["iterations"] for r in analytical_results) / len(analytical_results)

        if avg_factual_iters < avg_analytical_iters:
            print("✓ Factual queries terminate faster than analytical (as expected)")
        else:
            print("✗ Factual queries NOT terminating faster - check thresholds")

    # Check diminishing returns
    diminishing_detected = any("Diminishing" in r.get("termination_reason", "") for r in results)
    if diminishing_detected:
        print("✓ Diminishing returns detection triggered")
    else:
        print("○ Diminishing returns not triggered (may be OK if queries found enough)")


if __name__ == "__main__":
    asyncio.run(main())
