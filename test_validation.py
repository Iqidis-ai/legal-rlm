#!/usr/bin/env python3
"""Validation tests that verify actual output quality, not just completion."""

import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / "src"))

from irys.rlm.engine import RLMEngine, RLMConfig
from irys.core.models import GeminiClient
from irys.core.repository import MatterRepository

# Get absolute path to project root
PROJECT_ROOT = Path(__file__).parent.absolute()

# Test cases with expected content in answers
TEST_CASES = [
    {
        "name": "Small Repo - Contract Fee",
        "repo": str(PROJECT_ROOT / "test_documents"),
        "query": "What is the monthly fee in the contract?",
        "expected_in_answer": ["$65,000", "65,000", "amended", "March"],
        "min_answer_length": 100,
    },
    {
        "name": "Small Repo - Parties",
        "repo": str(PROJECT_ROOT / "test_documents"),
        "query": "Who are the parties in the contract?",
        "expected_in_answer": ["ACME", "TechServices", "Provider", "Client"],
        "min_answer_length": 50,
    },
]

# External search test cases (use UGC repo if available)
EXTERNAL_SEARCH_TESTS = [
    {
        "name": "External Search - Case Law and Regulations",
        "repo": "C:/Users/devan/OneDrive/Desktop/Test/UGC",
        "query": "What case law and regulatory guidance exists on creator contracts and affiliate marketing disclosures?",
        "expected_citations_contain": ["Case Law"],  # At minimum should have case law
        "min_citations": 1,
    },
]


async def run_test(test_case: dict) -> dict:
    """Run a single test case and validate output."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_case['name']}")
    print(f"Query: {test_case['query']}")
    print(f"{'='*60}")

    repo_path = test_case["repo"]
    if not Path(repo_path).exists():
        return {
            "name": test_case["name"],
            "status": "SKIPPED",
            "reason": f"Repo not found: {repo_path}",
        }

    # Track citations
    citations_found = []

    def on_citation(citation):
        citations_found.append(citation)

    try:
        client = GeminiClient()
        config = RLMConfig(max_iterations=5)
        engine = RLMEngine(gemini_client=client, config=config, on_citation=on_citation)

        result = await engine.investigate(test_case["query"], repo_path)
        answer = result.findings.get("final_output", "") or ""

        # Validation checks
        errors = []

        # Check answer length
        min_length = test_case.get("min_answer_length", 50)
        if len(answer) < min_length:
            errors.append(f"Answer too short: {len(answer)} chars (min: {min_length})")

        # Check expected content in answer
        expected = test_case.get("expected_in_answer", [])
        for term in expected:
            if term.lower() not in answer.lower():
                errors.append(f"Expected term not found in answer: '{term}'")

        # Check expected citation content
        expected_cites = test_case.get("expected_citations_contain", [])
        for cite_type in expected_cites:
            found = any(cite_type.lower() in str(c.document).lower() for c in citations_found)
            if not found:
                errors.append(f"Expected citation type not found: '{cite_type}'")

        # Check minimum citations
        min_cites = test_case.get("min_citations", 0)
        if len(citations_found) < min_cites:
            errors.append(f"Not enough citations: {len(citations_found)} (min: {min_cites})")

        # Print results
        print(f"\nAnswer ({len(answer)} chars):")
        print(answer[:500] + "..." if len(answer) > 500 else answer)

        print(f"\nCitations ({len(citations_found)}):")
        for c in citations_found[:5]:
            print(f"  - {c.document}")

        if errors:
            print(f"\n[FAIL] VALIDATION ERRORS:")
            for e in errors:
                print(f"  - {e}")
            return {
                "name": test_case["name"],
                "status": "FAILED",
                "errors": errors,
                "answer_length": len(answer),
                "citations": len(citations_found),
            }
        else:
            print(f"\n[PASS] Answer contains expected content")
            return {
                "name": test_case["name"],
                "status": "PASSED",
                "answer_length": len(answer),
                "citations": len(citations_found),
            }

    except Exception as e:
        print(f"\n[ERROR]: {e}")
        return {
            "name": test_case["name"],
            "status": "ERROR",
            "error": str(e),
        }


async def main():
    """Run all validation tests."""
    print("\n" + "="*60)
    print("RLM OUTPUT VALIDATION TESTS")
    print("="*60)

    results = []

    # Run basic tests
    for test in TEST_CASES:
        result = await run_test(test)
        results.append(result)

    # Run external search tests (optional)
    for test in EXTERNAL_SEARCH_TESTS:
        if Path(test["repo"]).exists():
            result = await run_test(test)
            results.append(result)
        else:
            print(f"\nSKIPPED: {test['name']} (repo not found)")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    passed = sum(1 for r in results if r["status"] == "PASSED")
    failed = sum(1 for r in results if r["status"] == "FAILED")
    errors = sum(1 for r in results if r["status"] == "ERROR")
    skipped = sum(1 for r in results if r["status"] == "SKIPPED")

    for r in results:
        status_icon = "[PASS]" if r["status"] == "PASSED" else "[FAIL]" if r["status"] in ["FAILED", "ERROR"] else "[SKIP]"
        print(f"  {status_icon} {r['name']}: {r['status']}")

    print(f"\nTotal: {len(results)} | Passed: {passed} | Failed: {failed} | Errors: {errors} | Skipped: {skipped}")

    if failed > 0 or errors > 0:
        sys.exit(1)

    print("\n[SUCCESS] All validation tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
