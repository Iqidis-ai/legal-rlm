"""Run comparative tests for RLM vs Claude Code vs Codex."""

import sys
import os
import asyncio
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()

from irys.core.models import GeminiClient
from irys.rlm.engine import RLMEngine, RLMConfig

REPO_PATH = r"C:\Users\devan\Downloads\CITIOM v Gulfstream\documents"

# Load queries
with open("legal_queries.json", "r") as f:
    queries_data = json.load(f)

# Sample queries from each category (2 from each = 12 total)
SAMPLE_QUERIES = [
    # Factual
    queries_data["categories"]["factual_extraction"][0],  # Initial cost estimate
    queries_data["categories"]["factual_extraction"][5],  # Days out of service

    # Multi-doc synthesis
    queries_data["categories"]["multi_document_synthesis"][0],  # Evidence across docs
    queries_data["categories"]["multi_document_synthesis"][10], # Compare briefs

    # Timeline
    queries_data["categories"]["timeline_construction"][0],  # Complete timeline
    queries_data["categories"]["timeline_construction"][6],   # When acknowledged

    # Contradiction
    queries_data["categories"]["contradiction_detection"][0],  # Proposal vs actual
    queries_data["categories"]["contradiction_detection"][2],  # Internal vs position

    # Legal analysis
    queries_data["categories"]["legal_analysis"][0],  # Elements to prove
    queries_data["categories"]["legal_analysis"][5],  # Duty of care

    # Evidence assessment
    queries_data["categories"]["evidence_assessment"][0],  # Strongest evidence
    queries_data["categories"]["evidence_assessment"][4],  # Most compelling
]


async def test_rlm_query(query: str, query_num: int) -> dict:
    """Test a single query with RLM."""
    print(f"\n{'='*70}")
    print(f"Query {query_num}: {query[:60]}...")
    print("="*70)

    client = GeminiClient()
    config = RLMConfig(
        max_depth=2,
        max_leads_per_level=3,
        parallel_reads=3,
        max_iterations=4,
    )

    start_time = time.time()

    def on_step(step):
        elapsed = time.time() - start_time
        content = step.content[:60] if len(step.content) > 60 else step.content
        print(f"  [{elapsed:5.1f}s] [{step.step_type.value:10s}] {content}")

    engine = RLMEngine(client, config=config, on_step=on_step)

    try:
        state = await engine.investigate(query, REPO_PATH)
        elapsed = time.time() - start_time

        result = {
            "query": query,
            "status": state.status,
            "documents_read": state.documents_read,
            "searches": state.searches_performed,
            "citations": len(state.citations),
            "elapsed_seconds": round(elapsed, 1),
            "output": state.findings.get("final_output", "")[:3000],
            "hypothesis": state.hypothesis,
            "error": None,
        }

        print(f"\n  Status: {state.status}")
        print(f"  Documents: {state.documents_read}, Searches: {state.searches_performed}, Citations: {len(state.citations)}")
        print(f"  Time: {elapsed:.1f}s")

        return result

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ERROR after {elapsed:.1f}s: {e}")
        return {
            "query": query,
            "status": "error",
            "error": str(e),
            "elapsed_seconds": round(elapsed, 1),
        }


async def run_rlm_tests():
    """Run all RLM tests."""
    print("="*70)
    print("RLM COMPARATIVE TEST")
    print(f"Testing {len(SAMPLE_QUERIES)} queries")
    print("="*70)

    results = []
    total_start = time.time()

    for i, query in enumerate(SAMPLE_QUERIES, 1):
        result = await test_rlm_query(query, i)
        results.append(result)

        # Save intermediate results
        with open("rlm_test_results.json", "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_queries": len(SAMPLE_QUERIES),
                "completed": len(results),
                "results": results,
            }, f, indent=2)

    total_elapsed = time.time() - total_start

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total queries: {len(results)}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")

    successful = [r for r in results if r["status"] == "completed"]
    failed = [r for r in results if r["status"] != "completed"]

    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        avg_citations = sum(r["citations"] for r in successful) / len(successful)
        avg_docs = sum(r["documents_read"] for r in successful) / len(successful)
        avg_time = sum(r["elapsed_seconds"] for r in successful) / len(successful)
        print(f"Average citations: {avg_citations:.1f}")
        print(f"Average documents: {avg_docs:.1f}")
        print(f"Average time: {avg_time:.1f}s")

    # Print each result's output preview
    print("\n" + "="*70)
    print("OUTPUT PREVIEWS")
    print("="*70)

    for i, r in enumerate(results, 1):
        print(f"\n--- Query {i}: {r['query'][:50]}... ---")
        if r.get("output"):
            print(r["output"][:500] + "...")
        elif r.get("error"):
            print(f"ERROR: {r['error']}")
        else:
            print("No output")

    return results


async def run_limited_tests(count: int = 3):
    """Run limited number of RLM tests."""
    print("="*70)
    print(f"RLM TEST (Limited to {count} queries)")
    print("="*70)

    results = []
    total_start = time.time()

    for i, query in enumerate(SAMPLE_QUERIES[:count], 1):
        result = await test_rlm_query(query, i)
        results.append(result)

    total_elapsed = time.time() - total_start

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total queries: {len(results)}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")

    successful = [r for r in results if r["status"] == "completed"]
    print(f"Successful: {len(successful)}/{len(results)}")

    if successful:
        avg_citations = sum(r["citations"] for r in successful) / len(successful)
        print(f"Average citations: {avg_citations:.1f}")

    return results


if __name__ == "__main__":
    import sys
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    asyncio.run(run_limited_tests(count))
