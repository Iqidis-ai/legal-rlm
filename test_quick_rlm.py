"""Quick RLM test with minimal depth."""

import sys
import os
import asyncio
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()

from irys.core.models import GeminiClient
from irys.rlm.engine import RLMEngine, RLMConfig

REPO_PATH = r"C:\Users\devan\Downloads\CITIOM v Gulfstream\documents"


async def test():
    print("Quick RLM test with minimal depth...")
    start_time = time.time()

    client = GeminiClient()
    # Very shallow investigation for quick test
    config = RLMConfig(
        max_depth=1,  # Just 1 level deep
        max_leads_per_level=2,  # Only 2 leads
        parallel_reads=2,  # Only read 2 docs in parallel
        max_iterations=3,  # Limit iterations
    )

    step_count = [0]

    def on_step(step):
        elapsed = time.time() - start_time
        step_count[0] += 1
        content = step.content[:80] if len(step.content) > 80 else step.content
        print(f"[{elapsed:6.1f}s] [{step_count[0]:2d}] [{step.step_type.value:10s}] {content}")

    engine = RLMEngine(client, config=config, on_step=on_step)

    query = "What was Gulfstream's initial cost estimate for the 192-month inspection?"

    print(f"\nQuery: {query}")
    print("=" * 80)

    try:
        state = await engine.investigate(query, REPO_PATH)

        elapsed = time.time() - start_time
        print(f"\n{'=' * 80}")
        print(f"COMPLETED in {elapsed:.1f}s")
        print(f"Status: {state.status}")
        print(f"Documents Read: {state.documents_read}")
        print(f"Searches: {state.searches_performed}")
        print(f"Citations: {len(state.citations)}")

        if state.citations:
            print("\n--- CITATIONS ---")
            for i, cit in enumerate(state.citations[:3]):
                print(f"{i+1}. {cit.document} p.{cit.page}: {cit.text[:100]}...")

        # Check for final output
        final_output = state.findings.get("final_output", "")
        if final_output:
            print("\n--- FINAL OUTPUT ---")
            print(final_output[:2000])

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\nERROR after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test())
