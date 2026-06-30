"""Quick test to verify search term extraction fix."""

import sys
import os
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from irys.core.repository import MatterRepository

REPO_PATH = r"C:\Users\devan\Downloads\CITIOM v Gulfstream\documents"

def extract_search_term(lead_description: str) -> str:
    """Same logic as engine._extract_search_term - now returns SINGLE term."""
    prefixes = ["Search for:", "Investigate:", "Find:", "Look for:"]
    term = lead_description
    for prefix in prefixes:
        if term.startswith(prefix):
            term = term[len(prefix):].strip()
            break

    term = re.sub(r'\bAND\b|\bOR\b|\bNOT\b', ' ', term, flags=re.IGNORECASE)
    term = re.sub(r'[\'\"()]', ' ', term)
    term = re.sub(r'\s+', ' ', term).strip()

    stopwords = {'the', 'and', 'for', 'with', 'that', 'this', 'from', 'into',
                 'about', 'which', 'when', 'where', 'what', 'how', 'who', 'why',
                 'any', 'all', 'each', 'between', 'related', 'regarding', 'concerning'}

    words = [w for w in term.split()
             if len(w) > 2 and w.lower() not in stopwords and not w.startswith('$')]

    if not words:
        words = [w for w in term.split() if len(w) > 3]

    if not words:
        return term[:50]

    priority_words = []

    for w in words:
        if re.match(r'^\$?[\d,.]+[MKBmkb]?$', w):
            priority_words.append(w)

    for w in words:
        if w[0].isupper() and len(w) > 2:
            priority_words.append(w)

    legal_terms = {'contract', 'agreement', 'breach', 'damages', 'liability',
                   'warranty', 'negligence', 'fraud', 'misrepresentation',
                   'estimate', 'inspection', 'maintenance', 'invoice', 'payment'}
    for w in words:
        if w.lower() in legal_terms:
            priority_words.append(w)

    if priority_words:
        return priority_words[0]

    return words[0]


# Test cases
test_leads = [
    'Search for: CITIOM AND Gulfstream AND "3.1M" OR "$3.1M" OR "3.1 million" AND estimate OR quote OR proposal OR bid',
    '"unreasonably low" AND estimate OR cost OR proposal',
    'variance AND cost AND estimate AND CITIOM AND Gulfstream',
    'expert report AND estimate AND cost AND (CITIOM OR Gulfstream)',
    'methodology AND estimate AND Gulfstream',
    'industry standard AND cost OR estimate',
    'damages AND reliance AND estimate AND (CITIOM OR Gulfstream)',
    'Search for: 192-month inspection estimate',
    'Search for: Bassim warranty claims',
]

print("Testing search term extraction and search:")
print("=" * 70)

repo = MatterRepository(REPO_PATH)

for lead in test_leads:
    extracted = extract_search_term(lead)
    results = repo.search(extracted, context_lines=1)
    print(f"\nOriginal: {lead[:60]}...")
    print(f"Extracted: '{extracted}'")
    print(f"Search hits: {len(results.hits)}")
    if results.hits:
        print(f"First match file: {results.hits[0].filename[:50]}")

print("\n" + "=" * 70)
print("Testing direct searches for key terms:")

direct_terms = ["CITIOM", "Gulfstream", "estimate", "inspection", "192-month", "warranty", "Bassim"]

for term in direct_terms:
    results = repo.search(term, context_lines=1)
    print(f"'{term}': {len(results.hits)} hits")
