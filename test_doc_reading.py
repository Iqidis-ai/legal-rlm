"""Test document reading speed."""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from irys.core.repository import MatterRepository
from irys.core.reader import DocumentReader

REPO_PATH = r"C:\Users\devan\Downloads\CITIOM v Gulfstream\documents"


def test():
    print("Testing document reading speed...")
    repo = MatterRepository(REPO_PATH)

    # List files
    files = list(repo.list_files())
    print(f"Total files: {len(files)}")

    # Test reading a few files
    reader = DocumentReader()

    for f in files[:5]:
        start = time.time()
        try:
            content = reader.read(f.path)
            elapsed = time.time() - start
            print(f"{elapsed:6.2f}s - {f.path.name[:50]} ({len(content.full_text)} chars)")
        except Exception as e:
            elapsed = time.time() - start
            print(f"{elapsed:6.2f}s - {f.path.name[:50]} ERROR: {e}")

    # Test search
    print("\nTesting search...")
    start = time.time()
    results = repo.search("192-month", context_lines=2)
    elapsed = time.time() - start
    print(f"Search completed in {elapsed:.2f}s, {len(results.hits)} hits")

    if results.hits:
        hit = results.hits[0]
        print(f"First hit: {hit.filename}")
        print(f"Text: {hit.match_text[:150]}...")


if __name__ == "__main__":
    test()
