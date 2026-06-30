"""
Evaluation Loop Runner for Irys RLM

Usage:
    python run_eval_loop.py baseline     # Run baseline and save
    python run_eval_loop.py test         # Run test and compare to baseline
    python run_eval_loop.py quick        # Quick 3-query test
    python run_eval_loop.py compare      # Show baseline vs latest comparison
"""

import sys
import os
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

# Paths
REPO_PATH = r"C:\Users\devan\Downloads\CITIOM v Gulfstream\documents"
QUERIES_FILE = "legal_queries.json"
BASELINE_FILE = "eval_baseline.json"
LATEST_FILE = "eval_latest.json"
RESULTS_DIR = Path("eval_results")

# Sample queries (2 from each category for quick eval)
def get_sample_queries(count_per_category=2):
    with open(QUERIES_FILE, "r") as f:
        data = json.load(f)

    queries = []
    for category, items in data["categories"].items():
        for item in items[:count_per_category]:
            queries.append({"category": category, "query": item})
    return queries


async def run_single_query(query: str, category: str) -> dict:
    """Run a single query and return metrics."""
    client = GeminiClient()
    config = RLMConfig(
        max_depth=3,
        max_leads_per_level=3,
        max_iterations=10,
    )
    engine = RLMEngine(gemini_client=client, config=config)

    start = time.time()
    try:
        state = await engine.investigate(query, REPO_PATH)
        duration = time.time() - start

        return {
            "category": category,
            "query": query,
            "status": "completed",
            "duration_seconds": round(duration, 2),
            "documents_read": state.documents_read,
            "searches_performed": state.searches_performed,
            "facts_found": len(state.findings.get("accumulated_facts", [])),
            "citations": len(state.citations),
            "output_length": len(state.findings.get("final_output", "")),
            "error": None,
        }
    except Exception as e:
        duration = time.time() - start
        return {
            "category": category,
            "query": query,
            "status": "error",
            "duration_seconds": round(duration, 2),
            "documents_read": 0,
            "searches_performed": 0,
            "facts_found": 0,
            "citations": 0,
            "output_length": 0,
            "error": str(e),
        }


async def run_eval(queries: list, label: str) -> dict:
    """Run evaluation on a set of queries."""
    print(f"\n{'='*70}")
    print(f"RUNNING EVAL: {label}")
    print(f"Queries: {len(queries)}")
    print(f"{'='*70}\n")

    results = []
    total_start = time.time()

    for i, q in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {q['category']}: {q['query'][:50]}...")
        result = await run_single_query(q["query"], q["category"])
        print(f"    -> {result['status']} in {result['duration_seconds']}s, {result['citations']} citations")
        results.append(result)

    total_duration = time.time() - total_start

    # Calculate summary stats
    completed = [r for r in results if r["status"] == "completed"]

    summary = {
        "label": label,
        "timestamp": datetime.now().isoformat(),
        "total_queries": len(queries),
        "completed": len(completed),
        "failed": len(queries) - len(completed),
        "total_duration_seconds": round(total_duration, 2),
        "avg_duration_seconds": round(sum(r["duration_seconds"] for r in results) / len(results), 2) if results else 0,
        "avg_citations": round(sum(r["citations"] for r in completed) / len(completed), 2) if completed else 0,
        "avg_docs_read": round(sum(r["documents_read"] for r in completed) / len(completed), 2) if completed else 0,
        "avg_facts": round(sum(r["facts_found"] for r in completed) / len(completed), 2) if completed else 0,
        "docs_found_rate": round(sum(1 for r in completed if r["documents_read"] > 0) / len(completed) * 100, 1) if completed else 0,
    }

    # Stats by category
    category_stats = {}
    for cat in set(r["category"] for r in results):
        cat_results = [r for r in results if r["category"] == cat]
        cat_completed = [r for r in cat_results if r["status"] == "completed"]
        category_stats[cat] = {
            "total": len(cat_results),
            "completed": len(cat_completed),
            "avg_duration": round(sum(r["duration_seconds"] for r in cat_results) / len(cat_results), 2) if cat_results else 0,
            "avg_citations": round(sum(r["citations"] for r in cat_completed) / len(cat_completed), 2) if cat_completed else 0,
        }

    return {
        "summary": summary,
        "category_stats": category_stats,
        "results": results,
    }


def print_summary(data: dict):
    """Print evaluation summary."""
    s = data["summary"]
    print(f"\n{'='*70}")
    print(f"SUMMARY: {s['label']}")
    print(f"{'='*70}")
    print(f"Timestamp: {s['timestamp']}")
    print(f"Queries: {s['completed']}/{s['total_queries']} completed")
    print(f"Total time: {s['total_duration_seconds']}s ({s['total_duration_seconds']/60:.1f} min)")
    print(f"Avg time per query: {s['avg_duration_seconds']}s")
    print(f"Avg citations: {s['avg_citations']}")
    print(f"Avg docs read: {s['avg_docs_read']}")
    print(f"Docs found rate: {s['docs_found_rate']}%")

    print(f"\nBy Category:")
    for cat, stats in data["category_stats"].items():
        print(f"  {cat[:20]:20} | {stats['completed']}/{stats['total']} | {stats['avg_duration']}s | {stats['avg_citations']} cites")


def compare_results(baseline: dict, latest: dict):
    """Compare baseline to latest results."""
    b = baseline["summary"]
    l = latest["summary"]

    print(f"\n{'='*70}")
    print("COMPARISON: Baseline vs Latest")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'Baseline':>12} {'Latest':>12} {'Change':>12}")
    print("-" * 65)

    def delta(old, new, lower_better=True):
        if old == 0:
            return "N/A"
        pct = ((new - old) / old) * 100
        symbol = "v" if (pct < 0 and lower_better) or (pct > 0 and not lower_better) else "^" if pct != 0 else "="
        return f"{pct:+.1f}% {symbol}"

    print(f"{'Avg Duration (s)':<25} {b['avg_duration_seconds']:>12} {l['avg_duration_seconds']:>12} {delta(b['avg_duration_seconds'], l['avg_duration_seconds'], True):>12}")
    print(f"{'Avg Citations':<25} {b['avg_citations']:>12} {l['avg_citations']:>12} {delta(b['avg_citations'], l['avg_citations'], False):>12}")
    print(f"{'Avg Docs Read':<25} {b['avg_docs_read']:>12} {l['avg_docs_read']:>12} {delta(b['avg_docs_read'], l['avg_docs_read'], False):>12}")
    print(f"{'Docs Found Rate (%)':<25} {b['docs_found_rate']:>12} {l['docs_found_rate']:>12} {delta(b['docs_found_rate'], l['docs_found_rate'], False):>12}")
    print(f"{'Completed':<25} {b['completed']:>12} {l['completed']:>12} {delta(b['completed'], l['completed'], False):>12}")


async def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1].lower()

    RESULTS_DIR.mkdir(exist_ok=True)

    if cmd == "baseline":
        queries = get_sample_queries(2)  # 12 queries (2 per category)
        data = await run_eval(queries, "BASELINE")

        with open(BASELINE_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nBaseline saved to {BASELINE_FILE}")
        print_summary(data)

    elif cmd == "test":
        queries = get_sample_queries(2)
        data = await run_eval(queries, "LATEST TEST")

        with open(LATEST_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to {LATEST_FILE}")
        print_summary(data)

        # Compare to baseline if exists
        if os.path.exists(BASELINE_FILE):
            with open(BASELINE_FILE, "r") as f:
                baseline = json.load(f)
            compare_results(baseline, data)

    elif cmd == "quick":
        queries = get_sample_queries(1)[:3]  # Just 3 queries
        data = await run_eval(queries, "QUICK TEST")
        print_summary(data)

    elif cmd == "compare":
        if not os.path.exists(BASELINE_FILE) or not os.path.exists(LATEST_FILE):
            print("Need both baseline and latest results. Run 'baseline' and 'test' first.")
            return

        with open(BASELINE_FILE, "r") as f:
            baseline = json.load(f)
        with open(LATEST_FILE, "r") as f:
            latest = json.load(f)

        print_summary(baseline)
        print_summary(latest)
        compare_results(baseline, latest)

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    asyncio.run(main())
