"""Test early termination + JSON truncation fixes."""

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

# Test queries of different complexity
TEST_QUERIES = [
    "What is the serial number of the aircraft?",  # Factual - should terminate fast
    "What are the key claims in this dispute?",    # Analytical - medium
]

json_parse_errors = 0

async def run_test(query: str) -> dict:
    """Run a single query and collect metrics."""
    global json_parse_errors
    json_parse_errors = 0

    print(f"\n{'='*70}")
    print(f"QUERY: {query}")

    classification = classify_query(query)
    print(f"Type: {classification['type']} | Complexity: {classification['complexity']}")

    client = GeminiClient(api_key=os.environ["GEMINI_API_KEY"])
    config = RLMConfig(max_iterations=10, min_depth=1)

    termination_reason = ""

    # Track JSON parse errors by patching the engine's parser
    original_parse = None

    def on_step(step):
        nonlocal termination_reason
        content = step.display
        if "Early termination" in step.content:
            termination_reason = step.content
        if "JSON parse failed" in content:
            global json_parse_errors
            json_parse_errors += 1
        # Print condensed output
        if step.step_type.value in ["thinking"]:
            print(f"  {content[:100]}")

    engine = RLMEngine(gemini_client=client, config=config, on_step=on_step)

    start = time.time()
    state = await engine.investigate(query, REPO_PATH)
    elapsed = time.time() - start

    result = {
        "query": query,
        "type": classification["type"],
        "iterations": len(state.facts_per_iteration),
        "facts_per_iter": state.facts_per_iteration,
        "total_facts": len(state.findings.get("accumulated_facts", [])),
        "citations": len(state.citations),
        "elapsed": round(elapsed, 1),
        "termination": termination_reason or "Max iterations",
    }

    print(f"\nRESULTS:")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Facts/iteration: {result['facts_per_iter']}")
    print(f"  Total facts: {result['total_facts']}")
    print(f"  Citations: {result['citations']}")
    print(f"  Time: {result['elapsed']}s")
    print(f"  Termination: {result['termination']}")

    return result


async def main():
    print("="*70)
    print("TESTING: Early Termination + JSON Truncation Fixes")
    print("="*70)
    print(f"Token limits: LITE=8192, FLASH=16384, PRO=32768")
    print(f"Fact limits: 15 (search), 20 (deep read)")

    results = []
    for query in TEST_QUERIES:
        try:
            result = await run_test(query)
            results.append(result)
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"query": query, "error": str(e)})

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Type':<12} {'Iters':<6} {'Facts':<8} {'Citations':<10} {'Time':<10}")
    print("-"*50)

    for r in results:
        if "error" not in r:
            print(f"{r['type']:<12} {r['iterations']:<6} {r['total_facts']:<8} {r['citations']:<10} {r['elapsed']:<10}s")

    # Verify early termination worked
    factual = [r for r in results if r.get("type") == "factual" and "error" not in r]
    analytical = [r for r in results if r.get("type") == "analytical" and "error" not in r]

    print("\nVERIFICATION:")
    if factual:
        print(f"  Factual query iterations: {factual[0]['iterations']}")
        if factual[0]['iterations'] <= 2:
            print("  ✓ Factual terminated quickly (as expected)")
        else:
            print("  ✗ Factual took too many iterations")

    if factual and analytical:
        if factual[0]['iterations'] <= analytical[0]['iterations']:
            print("  ✓ Factual terminated faster/equal to analytical")


if __name__ == "__main__":
    asyncio.run(main())
