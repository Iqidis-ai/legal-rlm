#!/usr/bin/env python3
"""Quick test to verify basic functionality."""

import sys
sys.path.insert(0, "src")

from pathlib import Path

# Test 1: Document Reader
print("=" * 60)
print("TEST 1: Document Reader")
print("=" * 60)

from irys.core.reader import DocumentReader

reader = DocumentReader()
test_repo = Path(r"C:\Users\devan\Downloads\CITIOM v Gulfstream")

# Find a PDF to test
pdfs = list(test_repo.glob("**/*.pdf"))[:1]
if pdfs:
    print(f"Testing with: {pdfs[0].name}")
    try:
        doc = reader.read(pdfs[0])
        print(f"  File type: {doc.file_type}")
        print(f"  Pages: {doc.page_count}")
        print(f"  Total chars: {doc.total_chars}")
        print(f"  First 200 chars: {doc.full_text[:200]}...")
        print("  [OK] PDF reading works!")
    except Exception as e:
        print(f"  [ERROR] {e}")

# Test 2: Repository
print("\n" + "=" * 60)
print("TEST 2: Matter Repository")
print("=" * 60)

from irys.core.repository import MatterRepository

try:
    repo = MatterRepository(test_repo)
    stats = repo.get_stats()
    print(f"  Total files: {stats.total_files}")
    print(f"  Size: {stats.size_mb:.1f} MB")
    print(f"  File types: {stats.files_by_type}")
    print(f"  Folders: {len(stats.folders)}")
    print("  [OK] Repository works!")
except Exception as e:
    print(f"  [ERROR] {e}")

# Test 3: Search
print("\n" + "=" * 60)
print("TEST 3: Document Search")
print("=" * 60)

try:
    results = repo.search("breach")
    print(f"  Query: 'breach'")
    print(f"  Total matches: {results.total_matches}")
    print(f"  Files searched: {results.files_searched}")
    if results.hits:
        hit = results.hits[0]
        print(f"  First hit: {hit.filename}, p. {hit.page_num}")
        print(f"  Match: {hit.match_text[:100]}...")
    print("  [OK] Search works!")
except Exception as e:
    print(f"  [ERROR] {e}")

# Test 4: Gemini Client
print("\n" + "=" * 60)
print("TEST 4: Gemini Client")
print("=" * 60)

import os
if not os.environ.get("GEMINI_API_KEY"):
    print("  [SKIP] GEMINI_API_KEY not set, skipping Gemini test")
else:
    from irys.core.models import GeminiClient, ModelTier

    try:
        client = GeminiClient()
        print("  Client initialized")
        print("  Testing Flash-Lite...")

        import asyncio

        async def test_gemini():
            response = await client.complete(
                "Say 'Hello, legal world!' in exactly those words.",
                tier=ModelTier.LITE
            )
            return response

        response = asyncio.run(test_gemini())
        print(f"  Response: {response[:100]}...")
        print("  [OK] Gemini works!")
    except Exception as e:
        print(f"  [ERROR] {e}")

print("\n" + "=" * 60)
print("BASIC TESTS COMPLETE")
print("=" * 60)
