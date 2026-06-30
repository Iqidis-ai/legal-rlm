"""Test document cache performance."""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from irys.core.repository import MatterRepository

REPO_PATH = r"C:\Users\devan\Downloads\CITIOM v Gulfstream\documents"


def test():
    print("Testing document cache performance...")

    repo = MatterRepository(REPO_PATH)

    # Multiple searches with same repo
    terms = ["Gulfstream", "CITIOM", "estimate", "inspection", "warranty"]

    for term in terms:
        start = time.time()
        results = repo.search(term, context_lines=2)
        elapsed = time.time() - start
        print(f"{elapsed:6.2f}s - '{term}': {len(results.hits)} hits (cache size: {len(repo._doc_cache)})")

    # Second round should be faster
    print("\nSecond round (should use cache):")
    for term in terms:
        start = time.time()
        results = repo.search(term, context_lines=2)
        elapsed = time.time() - start
        print(f"{elapsed:6.2f}s - '{term}': {len(results.hits)} hits")


if __name__ == "__main__":
    test()
