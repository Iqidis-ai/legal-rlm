"""Debug script to trace RLM search behavior."""

import sys
import os
import asyncio
import re

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()

from irys.core.repository import MatterRepository
from irys.core.models import GeminiClient, ModelTier
from irys.rlm.engine import RLMEngine, RLMConfig, ORIENTATION_PROMPT

REPO_PATH = r"C:\Users\devan\Downloads\CITIOM v Gulfstream\documents"

async def debug_search():
    """Debug the search flow."""
    repo = MatterRepository(REPO_PATH)

    # Test 1: Basic search works
    print("=" * 60)
    print("TEST 1: Direct search for 'Gulfstream'")
    results = repo.search("Gulfstream", context_lines=3)
    print(f"Found {len(results.hits)} hits")
    if results.hits:
        print(f"First hit: {results.hits[0].filename}")

    # Test 2: Search for more specific terms
    print("\n" + "=" * 60)
    print("TEST 2: Search for '192-month'")
    results = repo.search("192-month", context_lines=3)
    print(f"Found {len(results.hits)} hits")

    # Test 3: What does the LLM generate for initial searches?
    print("\n" + "=" * 60)
    print("TEST 3: LLM-generated search terms")

    client = GeminiClient()
    stats = repo.get_stats()
    structure = repo.get_structure()
    structure_str = "\n".join(f"  {folder}: {count} files" for folder, count in structure.items())

    query = "What evidence supports CITIOM's claim that Gulfstream's initial estimate of $3.1M was unreasonably low?"

    prompt = ORIENTATION_PROMPT.format(
        structure=structure_str,
        total_files=stats.total_files,
        query=query,
    )

    response = await client.complete(prompt, tier=ModelTier.FLASH)
    print("Raw LLM response (first 2000 chars):")
    print(response[:2000])

    # Parse the initial_searches
    import json
    try:
        # Try to find JSON in response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            plan = json.loads(json_match.group())
            searches = plan.get("initial_searches", [])
            print(f"\nInitial searches from LLM: {searches}")

            # Test each search term
            for term in searches[:5]:
                if isinstance(term, str):
                    print(f"\n--- Testing search term: '{term}' ---")

                    # Apply same extraction logic as engine
                    extracted = extract_search_term(term)
                    print(f"After extraction: '{extracted}'")

                    results = repo.search(extracted, context_lines=3)
                    print(f"Results: {len(results.hits)} hits")
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")

    # Test 4: Run full RLM investigation with verbose logging
    print("\n" + "=" * 60)
    print("TEST 4: Full RLM investigation (verbose)")

    config = RLMConfig(max_depth=2, max_leads_per_level=3)

    steps_logged = []

    def log_step(step):
        steps_logged.append(step)
        print(f"[{step.step_type.value}] {step.content}")
        if step.details:
            print(f"  Details: {str(step.details)[:200]}")

    engine = RLMEngine(client, config=config, on_step=log_step)

    try:
        state = await engine.investigate(query, REPO_PATH)
        print(f"\n--- Investigation Complete ---")
        print(f"Status: {state.status}")
        print(f"Documents read: {state.documents_read}")
        print(f"Searches performed: {state.searches_performed}")
        print(f"Citations: {len(state.citations)}")

        # Print all leads and their status
        print(f"\n--- Leads ---")
        for lead in state.leads[:10]:
            print(f"  [{lead.status}] {lead.description}")
    except Exception as e:
        print(f"Investigation failed: {e}")
        import traceback
        traceback.print_exc()


def extract_search_term(lead_description: str) -> str:
    """Same logic as engine._extract_search_term - now returns SINGLE term."""
    # Remove common prefixes
    prefixes = ["Search for:", "Investigate:", "Find:", "Look for:"]
    term = lead_description
    for prefix in prefixes:
        if term.startswith(prefix):
            term = term[len(prefix):].strip()
            break

    # Remove boolean operators and quotes
    term = re.sub(r'\bAND\b|\bOR\b|\bNOT\b', ' ', term, flags=re.IGNORECASE)
    term = re.sub(r'[\'\"()]', ' ', term)
    term = re.sub(r'\s+', ' ', term).strip()

    # Extract all meaningful words
    stopwords = {'the', 'and', 'for', 'with', 'that', 'this', 'from', 'into',
                 'about', 'which', 'when', 'where', 'what', 'how', 'who', 'why',
                 'any', 'all', 'each', 'between', 'related', 'regarding', 'concerning'}

    words = [w for w in term.split()
             if len(w) > 2 and w.lower() not in stopwords and not w.startswith('$')]

    if not words:
        words = [w for w in term.split() if len(w) > 3]

    if not words:
        return term[:50]

    # Priority: specific terms > generic terms
    priority_words = []

    # 1. Specific dollar amounts or numbers
    for w in words:
        if re.match(r'^\$?[\d,.]+[MKBmkb]?$', w):
            priority_words.append(w)

    # 2. Proper nouns / entity names (capitalized words)
    for w in words:
        if w[0].isupper() and len(w) > 2:
            priority_words.append(w)

    # 3. Legal-specific terms
    legal_terms = {'contract', 'agreement', 'breach', 'damages', 'liability',
                   'warranty', 'negligence', 'fraud', 'misrepresentation',
                   'estimate', 'inspection', 'maintenance', 'invoice', 'payment'}
    for w in words:
        if w.lower() in legal_terms:
            priority_words.append(w)

    if priority_words:
        return priority_words[0]

    return words[0]


if __name__ == "__main__":
    asyncio.run(debug_search())
