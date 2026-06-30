"""Run complete 100-query test with all outputs saved."""

import sys
import os

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()

from irys.core.models import GeminiClient
from irys.rlm.engine import RLMEngine, RLMConfig

REPO_PATH = r"C:\Users\devan\Downloads\CITIOM v Gulfstream\documents"
OUTPUT_DIR = Path("test_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load all queries
with open("legal_queries.json", "r") as f:
    queries_data = json.load(f)

ALL_QUERIES = queries_data["all_queries"]


async def test_single_query(query: str, query_num: int, total: int) -> dict:
    """Test a single query and return results."""
    print(f"\n{'='*70}")
    print(f"[{query_num}/{total}] {query[:60]}...")
    print("="*70)

    client = GeminiClient()
    config = RLMConfig(
        max_depth=3,
        max_leads_per_level=3,
        parallel_reads=3,
        max_iterations=5,
    )

    start_time = time.time()
    steps = []

    def on_step(step):
        elapsed = time.time() - start_time
        step_info = {
            "time": round(elapsed, 1),
            "type": step.step_type.value,
            "content": step.content[:200],
        }
        steps.append(step_info)
        print(f"  [{elapsed:5.1f}s] [{step.step_type.value:10s}] {step.content[:50]}")

    engine = RLMEngine(client, config=config, on_step=on_step)

    try:
        state = await engine.investigate(query, REPO_PATH)
        elapsed = time.time() - start_time

        # Extract all relevant data
        result = {
            "query_num": query_num,
            "query": query,
            "status": state.status,
            "elapsed_seconds": round(elapsed, 1),
            "documents_read": state.documents_read,
            "searches_performed": state.searches_performed,
            "citations_count": len(state.citations),
            "max_depth_reached": state.max_depth_reached,
            "hypothesis": state.hypothesis,
            "final_output": state.findings.get("final_output", ""),
            "accumulated_facts": state.findings.get("accumulated_facts", []),
            "citations": [
                {
                    "document": c.document,
                    "page": c.page,
                    "text": c.text[:500],
                    "verified": c.verified,
                }
                for c in state.citations[:20]  # Top 20 citations
            ],
            "steps": steps,
            "error": None,
        }

        print(f"\n  COMPLETED: {state.documents_read} docs, {len(state.citations)} citations, {elapsed:.1f}s")

        # Save individual result
        result_file = OUTPUT_DIR / f"query_{query_num:03d}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return result

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n  ERROR after {elapsed:.1f}s: {e}")

        result = {
            "query_num": query_num,
            "query": query,
            "status": "error",
            "elapsed_seconds": round(elapsed, 1),
            "error": str(e),
            "steps": steps,
        }

        # Save error result
        result_file = OUTPUT_DIR / f"query_{query_num:03d}_error.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return result


async def run_all_tests():
    """Run all 100 queries."""
    print("="*70)
    print("COMPLETE RLM TEST - 100 QUERIES")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print("="*70)

    results = []
    total_start = time.time()

    # Track progress
    progress_file = OUTPUT_DIR / "progress.json"

    for i, query in enumerate(ALL_QUERIES, 1):
        result = await test_single_query(query, i, len(ALL_QUERIES))
        results.append(result)

        # Save progress after each query
        progress = {
            "timestamp": datetime.now().isoformat(),
            "completed": len(results),
            "total": len(ALL_QUERIES),
            "elapsed_minutes": round((time.time() - total_start) / 60, 1),
            "successful": len([r for r in results if r["status"] == "completed"]),
            "failed": len([r for r in results if r["status"] != "completed"]),
        }
        with open(progress_file, "w") as f:
            json.dump(progress, f, indent=2)

        # Save full results periodically
        if i % 10 == 0:
            save_summary(results, total_start)

    # Final save
    save_summary(results, total_start)

    total_elapsed = time.time() - total_start
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"Results saved to: {OUTPUT_DIR.absolute()}")

    return results


def save_summary(results: list, start_time: float):
    """Save summary of all results."""
    elapsed = time.time() - start_time

    successful = [r for r in results if r["status"] == "completed"]
    failed = [r for r in results if r["status"] != "completed"]

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_queries": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "elapsed_minutes": round(elapsed / 60, 1),
        "avg_time_per_query": round(elapsed / len(results), 1) if results else 0,
        "statistics": {},
        "results": results,
    }

    if successful:
        summary["statistics"] = {
            "avg_citations": round(sum(r["citations_count"] for r in successful) / len(successful), 1),
            "avg_documents": round(sum(r["documents_read"] for r in successful) / len(successful), 1),
            "avg_searches": round(sum(r["searches_performed"] for r in successful) / len(successful), 1),
            "avg_time_seconds": round(sum(r["elapsed_seconds"] for r in successful) / len(successful), 1),
            "total_citations": sum(r["citations_count"] for r in successful),
            "total_documents": sum(r["documents_read"] for r in successful),
        }

    summary_file = OUTPUT_DIR / "full_results.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  [Saved progress: {len(results)} queries, {len(successful)} successful]")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
