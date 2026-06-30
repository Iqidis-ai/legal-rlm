"""Test LLM responses to see what we're getting."""

import sys
import os
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()

from irys.core.models import GeminiClient, ModelTier
from irys.rlm.engine import ORIENTATION_PROMPT
from irys.core.repository import MatterRepository

REPO_PATH = r"C:\Users\devan\Downloads\CITIOM v Gulfstream\documents"


async def test():
    print("Testing LLM responses...")

    client = GeminiClient()
    repo = MatterRepository(REPO_PATH)
    stats = repo.get_stats()
    structure = repo.get_structure()
    structure_str = "\n".join(f"  {folder}: {count} files" for folder, count in list(structure.items())[:10])

    query = "What evidence supports CITIOM's claim that Gulfstream's initial estimate was unreasonably low?"

    prompt = ORIENTATION_PROMPT.format(
        structure=structure_str,
        total_files=stats.total_files,
        query=query,
    )

    print(f"Prompt length: {len(prompt)} chars")
    print(f"First 500 chars of prompt:\n{prompt[:500]}...")
    print("\n" + "=" * 70)

    try:
        print("Calling Gemini FLASH model...")
        response = await client.complete(prompt, tier=ModelTier.FLASH, timeout=60)
        print(f"\nResponse length: {len(response) if response else 0} chars")
        print(f"\nFull response:\n{response}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test())
