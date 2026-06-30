"""Run Codex test on all 100 queries - saves results for comparison."""

import sys
import os
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

OUTPUT_DIR = Path("test_results/codex")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load queries
with open("legal_queries.json", "r") as f:
    queries_data = json.load(f)

ALL_QUERIES = queries_data["all_queries"]

SYSTEM_CONTEXT = """You are a senior legal analyst reviewing the CITIOM v. Gulfstream case.

Key case facts:
- CITIOM owns a Gulfstream G550 aircraft (SN 5174)
- Gulfstream performed a 192-month inspection starting January 29, 2024
- Original estimate: $835,931, 2,800 manhours, 35-45 business days
- Actual: work ballooned to 26,000+ manhours, completed November 29, 2024
- CITIOM claims Gulfstream's estimate was unreasonably low and misrepresented
- Key documents: Customer Support Proposal Q-11876 (June 2, 2023), Expert Reports from Danny Farnham and Leonard B

Provide a thorough legal analysis based on the facts of this case."""


def test_codex_query(query: str, query_num: int, total: int) -> dict:
    """Test a single query with Codex."""
    print(f"\n[{query_num}/{total}] {query[:60]}...")

    full_prompt = f"""{SYSTEM_CONTEXT}

QUESTION: {query}

Provide a detailed legal analysis addressing this question. Include:
1. Direct answer to the question
2. Relevant legal reasoning
3. Key facts that support your analysis
4. Any caveats or limitations"""

    start_time = time.time()

    try:
        # Run codex exec (use full path for Windows compatibility)
        codex_path = r"C:\Users\devan\AppData\Roaming\npm\codex.cmd"
        result = subprocess.run(
            [codex_path, "exec", full_prompt],
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout
            encoding='utf-8',
            errors='replace',  # Replace undecodable chars
        )

        elapsed = time.time() - start_time
        output = result.stdout.strip()

        response = {
            "query_num": query_num,
            "query": query,
            "status": "completed" if result.returncode == 0 else "error",
            "elapsed_seconds": round(elapsed, 1),
            "output": output,
            "error": result.stderr if result.returncode != 0 else None,
        }

        print(f"  Completed in {elapsed:.1f}s ({len(output)} chars)")

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        response = {
            "query_num": query_num,
            "query": query,
            "status": "timeout",
            "elapsed_seconds": round(elapsed, 1),
            "output": "",
            "error": "Timeout after 300s",
        }
        print(f"  TIMEOUT after {elapsed:.1f}s")

    except Exception as e:
        elapsed = time.time() - start_time
        response = {
            "query_num": query_num,
            "query": query,
            "status": "error",
            "elapsed_seconds": round(elapsed, 1),
            "output": "",
            "error": str(e),
        }
        print(f"  ERROR: {e}")

    # Save individual result
    result_file = OUTPUT_DIR / f"query_{query_num:03d}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(response, f, indent=2, ensure_ascii=False)

    return response


def run_all_codex_tests():
    """Run all queries through Codex."""
    print("="*70)
    print("CODEX TEST - 100 QUERIES")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Output: {OUTPUT_DIR.absolute()}")
    print("="*70)

    results = []
    total_start = time.time()
    progress_file = OUTPUT_DIR / "progress.json"

    for i, query in enumerate(ALL_QUERIES, 1):
        result = test_codex_query(query, i, len(ALL_QUERIES))
        results.append(result)

        # Save progress
        progress = {
            "timestamp": datetime.now().isoformat(),
            "system": "codex",
            "completed": len(results),
            "total": len(ALL_QUERIES),
            "elapsed_minutes": round((time.time() - total_start) / 60, 1),
            "successful": len([r for r in results if r["status"] == "completed"]),
            "failed": len([r for r in results if r["status"] != "completed"]),
        }
        with open(progress_file, "w") as f:
            json.dump(progress, f, indent=2)

        # Save full results every 10
        if i % 10 == 0:
            save_summary(results, total_start)

    save_summary(results, total_start)

    print(f"\nCOMPLETE: {len(results)} queries in {(time.time()-total_start)/60:.1f} min")
    return results


def save_summary(results: list, start_time: float):
    """Save summary."""
    successful = [r for r in results if r["status"] == "completed"]

    summary = {
        "system": "codex",
        "timestamp": datetime.now().isoformat(),
        "total_queries": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "elapsed_minutes": round((time.time() - start_time) / 60, 1),
        "results": results,
    }

    if successful:
        summary["avg_time_seconds"] = round(
            sum(r["elapsed_seconds"] for r in successful) / len(successful), 1
        )

    with open(OUTPUT_DIR / "full_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    run_all_codex_tests()
