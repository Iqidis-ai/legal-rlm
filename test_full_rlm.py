"""Full RLM investigation test."""

import sys
import os
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()

from irys.core.models import GeminiClient
from irys.rlm.engine import RLMEngine, RLMConfig

REPO_PATH = r"C:\Users\devan\Downloads\CITIOM v Gulfstream\documents"


async def test_investigation():
    """Run a full RLM investigation."""
    print("Initializing RLM...")
    client = GeminiClient()
    config = RLMConfig(max_depth=3, max_leads_per_level=3)

    def on_step(step):
        print(f"[{step.step_type.value}] {step.content[:100] if len(step.content) > 100 else step.content}")

    engine = RLMEngine(client, config=config, on_step=on_step)

    query = "What evidence supports CITIOM's position that Gulfstream's initial cost estimate was unreasonably low?"

    print(f"\nQuery: {query}")
    print("=" * 70)

    try:
        state = await engine.investigate(query, REPO_PATH)

        print("\n" + "=" * 70)
        print("INVESTIGATION RESULTS:")
        print("=" * 70)
        print(f"Status: {state.status}")
        print(f"Documents Read: {state.documents_read}")
        print(f"Searches Performed: {state.searches_performed}")
        print(f"Citations: {len(state.citations)}")
        print(f"Max Depth Reached: {state.max_depth_reached}")

        if state.hypothesis:
            print(f"\nFinal Hypothesis: {state.hypothesis}")

        print(f"\nAccumulated Facts: {len(state.findings.get('accumulated_facts', []))}")

        if state.citations:
            print("\n--- TOP CITATIONS ---")
            for i, cit in enumerate(state.citations[:5]):
                print(f"\n{i+1}. {cit.document} (p.{cit.page})")
                print(f"   Text: {cit.text[:150]}...")

        if state.synthesis:
            print("\n--- SYNTHESIS ---")
            print(state.synthesis[:1500])

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_investigation())
