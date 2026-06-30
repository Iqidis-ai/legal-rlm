"""Compare results from RLM, Codex, and Claude Code tests."""

import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class QueryResult:
    """Result from a single query."""
    query_num: int
    query: str
    status: str
    elapsed_seconds: float
    output: str
    error: Optional[str] = None

    # RLM-specific fields
    citations: int = 0
    documents_read: int = 0
    iterations: int = 0


@dataclass
class SystemResults:
    """All results from one system."""
    system: str
    total_queries: int
    successful: int
    failed: int
    elapsed_minutes: float
    avg_time_seconds: float
    results: List[QueryResult]


def load_results(results_dir: Path, system: str) -> Optional[SystemResults]:
    """Load results from a system's output directory."""
    full_results_file = results_dir / "full_results.json"

    if not full_results_file.exists():
        print(f"No results found for {system} at {full_results_file}")
        return None

    with open(full_results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for r in data.get("results", []):
        qr = QueryResult(
            query_num=r.get("query_num", 0),
            query=r.get("query", ""),
            status=r.get("status", ""),
            elapsed_seconds=r.get("elapsed_seconds", 0),
            output=r.get("final_output", r.get("output", "")),
            error=r.get("error"),
            citations=r.get("citations_count", r.get("citations", 0)),
            documents_read=r.get("documents_read", 0),
            iterations=r.get("iterations", 0),
        )
        results.append(qr)

    return SystemResults(
        system=system,
        total_queries=data.get("total_queries", 0),
        successful=data.get("successful", 0),
        failed=data.get("failed", 0),
        elapsed_minutes=data.get("elapsed_minutes", 0),
        avg_time_seconds=data.get("avg_time_seconds", 0),
        results=results,
    )


def load_all_results() -> Dict[str, SystemResults]:
    """Load results from all systems."""
    base_dir = Path("test_results")

    systems = {
        "rlm": base_dir,  # RLM results in test_results/
        "codex": base_dir / "codex",
        "claude_code": base_dir / "claude_code",
    }

    all_results = {}
    for name, path in systems.items():
        results = load_results(path, name)
        if results:
            all_results[name] = results

    return all_results


def print_summary(all_results: Dict[str, SystemResults]):
    """Print comparison summary."""
    print("=" * 80)
    print("TEST RESULTS COMPARISON")
    print("=" * 80)

    for system, results in all_results.items():
        print(f"\n{system.upper()}")
        print("-" * 40)
        print(f"  Total queries: {results.total_queries}")
        print(f"  Successful:    {results.successful}")
        print(f"  Failed:        {results.failed}")
        print(f"  Total time:    {results.elapsed_minutes:.1f} min")
        print(f"  Avg per query: {results.avg_time_seconds:.1f}s")

        # RLM-specific stats
        if system == "rlm" and results.results:
            total_citations = sum(r.citations if isinstance(r.citations, int) else 0 for r in results.results)
            total_docs = sum(r.documents_read for r in results.results)
            avg_citations = total_citations / len(results.results) if results.results else 0
            print(f"  Total citations: {total_citations}")
            print(f"  Avg citations:   {avg_citations:.1f}")
            print(f"  Total docs read: {total_docs}")


def compare_query(query_num: int, all_results: Dict[str, SystemResults]) -> dict:
    """Compare outputs for a specific query across systems."""
    comparison = {"query_num": query_num}

    for system, results in all_results.items():
        for r in results.results:
            if r.query_num == query_num:
                comparison[system] = {
                    "status": r.status,
                    "time": r.elapsed_seconds,
                    "output_length": len(r.output),
                    "error": r.error,
                }
                if system == "rlm":
                    comparison[system]["citations"] = r.citations
                    comparison[system]["documents_read"] = r.documents_read
                break

    return comparison


def export_comparison(all_results: Dict[str, SystemResults], output_file: str = "comparison_report.json"):
    """Export detailed comparison to JSON."""
    report = {
        "summary": {},
        "per_query": [],
    }

    # Summary stats
    for system, results in all_results.items():
        report["summary"][system] = {
            "total_queries": results.total_queries,
            "successful": results.successful,
            "failed": results.failed,
            "elapsed_minutes": results.elapsed_minutes,
            "avg_time_seconds": results.avg_time_seconds,
        }

    # Per-query comparison
    max_queries = max(r.total_queries for r in all_results.values()) if all_results else 0
    for i in range(1, max_queries + 1):
        report["per_query"].append(compare_query(i, all_results))

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nComparison report saved to: {output_file}")
    return report


def analyze_quality(all_results: Dict[str, SystemResults]):
    """Analyze output quality metrics."""
    print("\n" + "=" * 80)
    print("QUALITY ANALYSIS")
    print("=" * 80)

    for system, results in all_results.items():
        outputs = [r.output for r in results.results if r.status == "completed"]

        if not outputs:
            print(f"\n{system.upper()}: No completed outputs")
            continue

        avg_length = sum(len(o) for o in outputs) / len(outputs)

        # Count specific document references
        doc_refs = sum(1 for o in outputs if any(
            term in o.lower() for term in [".pdf", ".docx", "page", "exhibit"]
        ))

        # Count specific citations (numbers, dates, amounts)
        specific_refs = sum(1 for o in outputs if any(
            c in o for c in ["$", "2024", "2023", "SN 5174", "Q-11876"]
        ))

        print(f"\n{system.upper()}")
        print("-" * 40)
        print(f"  Avg output length: {avg_length:.0f} chars")
        print(f"  Outputs with doc refs: {doc_refs}/{len(outputs)}")
        print(f"  Outputs with specifics: {specific_refs}/{len(outputs)}")


def main():
    """Main comparison function."""
    all_results = load_all_results()

    if not all_results:
        print("No results found. Run the tests first.")
        return

    print_summary(all_results)
    analyze_quality(all_results)
    export_comparison(all_results)


if __name__ == "__main__":
    main()
