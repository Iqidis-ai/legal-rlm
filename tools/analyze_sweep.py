"""Analyze Harvey LAB sweep results and generate Codex review package.

Reads scores.json files from the LAB results directory, aggregates metrics,
identifies systemic failure patterns, and writes a structured report for
Codex architectural review.

Usage:
    python tools/analyze_sweep.py [--sweep-id 20260507-03] [--codex-review]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1].parent / "harvey-labs"
RESULTS_DIR = LAB_ROOT / "results"


def collect_scores(sweep_prefix: str | None = None) -> list[dict]:
    scores = []
    for scores_file in RESULTS_DIR.rglob("scores.json"):
        if sweep_prefix and sweep_prefix not in str(scores_file):
            continue
        try:
            data = json.loads(scores_file.read_text(encoding="utf-8"))
            data["_path"] = str(scores_file)
            scores.append(data)
        except Exception:
            continue
    return scores


def deduplicate_best(scores: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    for s in scores:
        task = s.get("task", "")
        if task not in best or s.get("n_passed", 0) > best[task].get("n_passed", 0):
            best[task] = s
    return sorted(best.values(), key=lambda x: x.get("task", ""))


def analyze(scores: list[dict]) -> dict:
    by_pa: dict[str, list] = defaultdict(list)
    all_failures: list[dict] = []
    all_passes: list[dict] = []

    for s in scores:
        task = s.get("task", "unknown")
        pa = task.split("/")[0] if "/" in task else "unknown"
        n_passed = s.get("n_passed", 0)
        n_total = s.get("n_criteria", 1)
        pct = round(n_passed / n_total * 100, 1) if n_total else 0

        by_pa[pa].append({
            "task": task,
            "passed": n_passed,
            "total": n_total,
            "pct": pct,
            "all_pass": s.get("all_pass", False),
        })

        for cr in s.get("criteria_results", []):
            entry = {
                "task": task,
                "pa": pa,
                "criterion_id": cr.get("id", ""),
                "title": cr.get("title", ""),
                "reasoning": cr.get("reasoning", ""),
            }
            if cr.get("verdict") == "fail":
                all_failures.append(entry)
            else:
                all_passes.append(entry)

    pa_summary = {}
    for pa, tasks in sorted(by_pa.items()):
        total_passed = sum(t["passed"] for t in tasks)
        total_criteria = sum(t["total"] for t in tasks)
        all_pass_count = sum(1 for t in tasks if t["all_pass"])
        pa_summary[pa] = {
            "tasks_scored": len(tasks),
            "criteria_passed": total_passed,
            "criteria_total": total_criteria,
            "pct": round(total_passed / total_criteria * 100, 1) if total_criteria else 0,
            "all_pass_tasks": all_pass_count,
            "per_task": sorted(tasks, key=lambda x: -x["pct"]),
        }

    failure_categories = categorize_failures(all_failures)

    global_passed = sum(v["criteria_passed"] for v in pa_summary.values())
    global_total = sum(v["criteria_total"] for v in pa_summary.values())

    return {
        "global": {
            "tasks_scored": len(scores),
            "practice_areas": len(pa_summary),
            "criteria_passed": global_passed,
            "criteria_total": global_total,
            "pct": round(global_passed / global_total * 100, 1) if global_total else 0,
            "all_pass_tasks": sum(v["all_pass_tasks"] for v in pa_summary.values()),
        },
        "by_practice_area": pa_summary,
        "failure_categories": failure_categories,
        "total_failures": len(all_failures),
        "total_passes": len(all_passes),
    }


def categorize_failures(failures: list[dict]) -> dict[str, int]:
    categories: dict[str, int] = defaultdict(int)
    for f in failures:
        title = f.get("title", "").lower()
        reasoning = f.get("reasoning", "").lower()

        if any(k in title for k in ("quantif", "dollar", "$", "cost", "calculation")):
            categories["quantitative_analysis_missing"] += 1
        elif any(k in title for k in ("risk rating", "red", "yellow", "green", "color")):
            categories["risk_rating_missing"] += 1
        elif any(k in title for k in ("recommend", "counterproposal", "specific")):
            categories["recommendation_missing"] += 1
        elif any(k in title for k in ("impact", "consequence", "effect")):
            categories["impact_analysis_missing"] += 1
        elif any(k in title for k in ("identified", "identified:")):
            categories["issue_not_identified"] += 1
        elif "does not" in reasoning or "not mention" in reasoning or "entirely omits" in reasoning:
            categories["content_gap"] += 1
        else:
            categories["other"] += 1

    return dict(sorted(categories.items(), key=lambda x: -x[1]))


def generate_codex_review_prompt(analysis: dict, sweep_id: str) -> str:
    g = analysis["global"]
    lines = [
        f"# Harvey LAB Sweep Results — Codex Architectural Review",
        f"",
        f"## Sweep: {sweep_id}",
        f"## Global: {g['criteria_passed']}/{g['criteria_total']} criteria passed ({g['pct']}%)",
        f"## Tasks scored: {g['tasks_scored']} across {g['practice_areas']} practice areas",
        f"## Perfect-score tasks: {g['all_pass_tasks']}/{g['tasks_scored']}",
        f"",
        f"## Per Practice Area:",
    ]

    for pa, data in sorted(analysis["by_practice_area"].items(), key=lambda x: -x[1]["pct"]):
        lines.append(f"- **{pa}**: {data['criteria_passed']}/{data['criteria_total']} ({data['pct']}%) — {data['tasks_scored']} tasks, {data['all_pass_tasks']} perfect")

    lines.append("")
    lines.append("## Failure Category Breakdown:")
    for cat, count in analysis["failure_categories"].items():
        lines.append(f"- {cat}: {count}")

    lines.append("")
    lines.append("## Worst Tasks (bottom 10):")
    all_tasks = []
    for pa_data in analysis["by_practice_area"].values():
        all_tasks.extend(pa_data["per_task"])
    for t in sorted(all_tasks, key=lambda x: x["pct"])[:10]:
        lines.append(f"- {t['task']}: {t['passed']}/{t['total']} ({t['pct']}%)")

    lines.append("")
    lines.append("## Best Tasks (top 10):")
    for t in sorted(all_tasks, key=lambda x: -x["pct"])[:10]:
        lines.append(f"- {t['task']}: {t['passed']}/{t['total']} ({t['pct']}%)")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-id", default="20260507-03")
    parser.add_argument("--codex-review", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    scores = collect_scores(args.sweep_id)
    if not scores:
        print(f"No scores found for sweep {args.sweep_id}")
        return 1

    scores = deduplicate_best(scores)
    analysis = analyze(scores)

    if args.json:
        print(json.dumps(analysis, indent=2))
        return 0

    g = analysis["global"]
    print(f"\n{'='*60}")
    print(f"HARVEY LAB SWEEP ANALYSIS — {args.sweep_id}")
    print(f"{'='*60}")
    print(f"  Tasks scored:     {g['tasks_scored']}")
    print(f"  Practice areas:   {g['practice_areas']}")
    print(f"  Criteria passed:  {g['criteria_passed']}/{g['criteria_total']} ({g['pct']}%)")
    print(f"  Perfect tasks:    {g['all_pass_tasks']}/{g['tasks_scored']}")
    print()

    print("  Per Practice Area:")
    for pa, data in sorted(analysis["by_practice_area"].items(), key=lambda x: -x[1]["pct"]):
        print(f"    {pa:45s} {data['criteria_passed']:>4}/{data['criteria_total']:<4} ({data['pct']:>5.1f}%)  [{data['tasks_scored']} tasks]")

    print()
    print("  Failure Categories:")
    for cat, count in analysis["failure_categories"].items():
        print(f"    {cat:40s} {count:>5}")

    if args.codex_review:
        prompt = generate_codex_review_prompt(analysis, args.sweep_id)
        out_path = Path("tools") / "sweep_review_prompt.md"
        out_path.write_text(prompt, encoding="utf-8")
        print(f"\n  Codex review prompt written to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
