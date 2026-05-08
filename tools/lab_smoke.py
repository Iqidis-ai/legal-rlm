"""Tiered Harvey LAB benchmark runner.

Tier 1 (smoke, every iteration):
    - Build the candidate pool: all known-failing tasks + N random tasks
      uniformly sampled across all 24 practice areas (default N=30).
    - Cap the pool at `--cap-total` (default 40). If the pool is larger,
      uniformly sub-sample down to the cap.
    - Random sampling uses a fresh seed each run — do NOT freeze the
      sample, that's how overfitting to a sample starts.
    - Run, score, update ledger.

The 40-task cap is an iteration-speed throttle (set 2026-05-08 while
substrate scores are still ramping). Lift the cap (raise --cap-total)
once Tier 1 scores get strong; eventually retire the cap entirely and
return to "all failing + 50 random".

Tier 2 (confirmation, only when Tier 1 looks clean):
    - Full benchmark across all 989 tasks. Used as the gate before declaring
      a structural change ship-able.

Usage:
    # Tier 1 smoke (default, 40-cap)
    python tools/lab_smoke.py

    # Tier 1 with a wider cap (once scores improve)
    python tools/lab_smoke.py --cap-total 80 --random 50

    # Tier 2 full sweep
    python tools/lab_smoke.py --tier 2

    # Refresh the failing-tasks ledger from the latest scored runs across
    # the harvey-labs results dir
    python tools/lab_smoke.py --refresh-ledger

The failing-tasks ledger lives at tools/lab_failing_tasks_ledger.json; it is
updated after every smoke or full run so subsequent smokes pick up the
freshly-failing set.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAB_ROOT_DEFAULT = PROJECT_ROOT.parent / "harvey-labs"
LEDGER_PATH = PROJECT_ROOT / "tools" / "lab_failing_tasks_ledger.json"


# ---------------------------------------------------------------------------
# Task discovery
# ---------------------------------------------------------------------------


def discover_all_tasks(lab_root: Path) -> list[str]:
    """Walk harvey-labs/tasks/<practice-area>/<task-id>/ and return ids.

    Matches the task-id format `practice-area/task-name` used by lab_benchmark.py.
    """
    tasks_root = lab_root / "tasks"
    if not tasks_root.is_dir():
        raise FileNotFoundError(f"harvey-labs tasks dir not found: {tasks_root}")
    out: list[str] = []
    for area in sorted(tasks_root.iterdir()):
        if not area.is_dir() or area.name.startswith("."):
            continue
        for task in sorted(area.iterdir()):
            if not task.is_dir() or task.name.startswith("."):
                continue
            # Heuristic: a real task dir has either `documents/`, `task.json`,
            # or `criteria.json` inside.
            if (task / "documents").is_dir() or (task / "task.json").exists() \
                    or (task / "criteria.json").exists():
                out.append(f"{area.name}/{task.name}")
    return out


# ---------------------------------------------------------------------------
# Failing-tasks ledger
# ---------------------------------------------------------------------------


def load_ledger() -> dict:
    if not LEDGER_PATH.exists():
        return {"failing_tasks": {}, "last_refreshed_at": None}
    try:
        return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"failing_tasks": {}, "last_refreshed_at": None}


def save_ledger(ledger: dict) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEDGER_PATH.write_text(json.dumps(ledger, indent=2), encoding="utf-8")


def refresh_ledger_from_results(
    lab_root: Path, threshold: float = 0.5,
) -> dict:
    """Scan harvey-labs/results/ for the most-recent score per task and
    rebuild the failing list (any task below `threshold` of criteria)."""
    results_root = lab_root / "results"
    if not results_root.is_dir():
        return {"failing_tasks": {}, "last_refreshed_at": _now_iso()}

    failing: dict[str, dict] = {}
    for area in sorted(results_root.iterdir()):
        if not area.is_dir() or area.name.startswith("."):
            continue
        for task in sorted(area.iterdir()):
            if not task.is_dir() or task.name.startswith("."):
                continue
            irys_simple = task / "irys-rlm-simple"
            if not irys_simple.is_dir():
                continue
            # Pick most recent run that has scores.json
            best_run, best_scores = None, None
            for run in sorted(irys_simple.iterdir(), reverse=True):
                sf = run / "scores.json"
                if sf.exists():
                    try:
                        d = json.loads(sf.read_text(encoding="utf-8"))
                        best_run = run.name
                        best_scores = d
                        break
                    except Exception:
                        continue
            if not best_scores:
                continue
            n_passed = int(best_scores.get("n_passed", 0))
            n_criteria = int(best_scores.get("n_criteria", 0)) or 1
            ratio = n_passed / n_criteria
            if ratio < threshold:
                tid = f"{area.name}/{task.name}"
                failing[tid] = {
                    "ratio": round(ratio, 4),
                    "n_passed": n_passed,
                    "n_criteria": n_criteria,
                    "scored_run": best_run,
                }
    return {
        "failing_tasks": failing,
        "last_refreshed_at": _now_iso(),
        "threshold": threshold,
    }


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------


def pick_smoke_tasks(
    all_tasks: list[str],
    failing_tasks: list[str],
    n_random: int = 30,
    cap_total: int = 40,
) -> list[str]:
    """Tier-1 selection: failing tasks + N random, capped at cap_total.

    Iteration speed throttle (2026-05-08): build the candidate pool from
    `failing_tasks ∪ N_random`, then if the pool exceeds `cap_total`,
    uniformly subsample down to `cap_total`. This keeps cycles fast while
    we're far from passing — once scores get strong we lift the cap and
    return to full Tier 1.

    Random sampling uses the system RNG (fresh seed per run by default).
    Avoids cherry-picking — prevents unconscious overfitting to a fixed sample.
    """
    rng = random.Random()  # seeded from system entropy
    failing_set = set(failing_tasks) & set(all_tasks)
    pool = [t for t in all_tasks if t not in failing_set]
    n = min(n_random, len(pool))
    sampled = rng.sample(pool, n)
    candidates = sorted(failing_set) + sampled
    if len(candidates) > cap_total:
        # Subsample down to cap_total — uniform across the pool, NOT
        # failing-prioritized, so we don't always run the same failing
        # tasks every cycle (that would be its own form of overfitting).
        candidates = rng.sample(candidates, cap_total)
        candidates.sort()  # stable order for logging
    return candidates


# ---------------------------------------------------------------------------
# Runner integration
# ---------------------------------------------------------------------------


def run_tasks(
    task_ids: list[str],
    *,
    lab_root: Path,
    research_mode: str = "simple",
    judge_model: str = "gemini-3.1-flash-lite-preview",
    concurrency: int = 6,
    log_path: Optional[Path] = None,
) -> int:
    """Run all tasks in ONE lab_benchmark.py process via --tasks-file.

    Single-process means: one matter init, one Gemini client, one agent
    registry, one async event loop with N concurrent investigations.
    Saves ~60s/task of Python startup overhead vs the per-task approach.
    """
    runner = PROJECT_ROOT / "tools" / "lab_benchmark.py"
    log_path = log_path or (PROJECT_ROOT / "tools" / "lab_smoke_run.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Write task list to a temp file for --tasks-file
    tasks_file = PROJECT_ROOT / "tools" / "lab_smoke_tasks.txt"
    tasks_file.write_text("\n".join(task_ids) + "\n", encoding="utf-8")

    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(f"=== smoke run {_now_iso()} ===\n")
        logf.write(f"tasks: {len(task_ids)} (single process, asyncio.gather)\n")
        logf.write(
            f"mode: {research_mode}, judge: {judge_model}, "
            f"concurrency: {concurrency}\n\n"
        )

    cmd = [
        sys.executable, str(runner),
        "--tasks-file", str(tasks_file),
        "--research-mode", research_mode,
        "--auto-score",
        "--concurrency", str(concurrency),
        "--judge-model", judge_model,
    ]
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"command: {' '.join(cmd)}\n\n")
        r = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT), stdout=logf, stderr=subprocess.STDOUT,
        )
    return r.returncode


def run_full_sweep(
    *, lab_root: Path,
    research_mode: str = "simple",
    judge_model: str = "gemini-3.1-flash-lite-preview",
    concurrency: int = 6,
    log_path: Optional[Path] = None,
) -> int:
    """Tier 2: invoke lab_benchmark.py --all once."""
    runner = PROJECT_ROOT / "tools" / "lab_benchmark.py"
    log_path = log_path or (PROJECT_ROOT / "tools" / "lab_full_sweep.log")
    cmd = [
        sys.executable, str(runner),
        "--all",
        "--research-mode", research_mode,
        "--auto-score",
        "--concurrency", str(concurrency),
        "--judge-model", judge_model,
    ]
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(f"=== full sweep {_now_iso()} ===\n")
    with log_path.open("a", encoding="utf-8") as logf:
        return subprocess.run(cmd, cwd=str(PROJECT_ROOT), stdout=logf, stderr=subprocess.STDOUT).returncode


# ---------------------------------------------------------------------------
# Score aggregation after run
# ---------------------------------------------------------------------------


def score_summary_for_tasks(
    task_ids: list[str], lab_root: Path,
) -> tuple[int, int, list[tuple[str, int, int]]]:
    """Read most-recent scores.json per task and return (passed, criteria, per-task)."""
    results_root = lab_root / "results"
    total_p, total_c = 0, 0
    rows: list[tuple[str, int, int]] = []
    for tid in task_ids:
        area, name = tid.split("/", 1)
        irys = results_root / area / name / "irys-rlm-simple"
        if not irys.is_dir():
            continue
        for run in sorted(irys.iterdir(), reverse=True):
            sf = run / "scores.json"
            if sf.exists():
                try:
                    d = json.loads(sf.read_text(encoding="utf-8"))
                    p = int(d.get("n_passed", 0))
                    c = int(d.get("n_criteria", 0))
                    rows.append((tid, p, c))
                    total_p += p
                    total_c += c
                    break
                except Exception:
                    continue
    return total_p, total_c, rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiered Harvey LAB benchmark runner")
    parser.add_argument("--tier", type=int, choices=[1, 2], default=1,
                        help="1 = smoke (failing + N random, capped); 2 = full sweep")
    parser.add_argument("--random", type=int, default=30,
                        help="Number of random tasks added to the candidate pool")
    parser.add_argument("--cap-total", type=int, default=40,
                        help="Hard cap on Tier 1 task count (uniform sub-sample if pool > cap)")
    parser.add_argument("--failing-threshold", type=float, default=0.5,
                        help="Tasks below this criteria-pass ratio are 'failing'")
    parser.add_argument("--lab-root", default=str(LAB_ROOT_DEFAULT),
                        help="harvey-labs repo path")
    parser.add_argument("--research-mode", default="simple",
                        choices=["simple", "deep", "sebih_special"])
    parser.add_argument("--judge-model", default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--refresh-ledger", action="store_true",
                        help="Recompute failing-tasks ledger from results/ and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print tasks that would run; do not invoke")

    args = parser.parse_args()
    lab_root = Path(args.lab_root).resolve()

    if args.refresh_ledger:
        ledger = refresh_ledger_from_results(lab_root, threshold=args.failing_threshold)
        save_ledger(ledger)
        n = len(ledger["failing_tasks"])
        print(f"Ledger refreshed: {n} failing tasks (< {args.failing_threshold*100:.0f}% pass)")
        for tid, info in sorted(ledger["failing_tasks"].items(),
                                key=lambda kv: kv[1]["ratio"])[:20]:
            print(f"  {info['ratio']*100:5.1f}%  {tid}")
        return

    if args.tier == 2:
        rc = run_full_sweep(
            lab_root=lab_root,
            research_mode=args.research_mode,
            judge_model=args.judge_model,
            concurrency=args.concurrency,
        )
        # Refresh ledger after full sweep
        ledger = refresh_ledger_from_results(lab_root, threshold=args.failing_threshold)
        save_ledger(ledger)
        print(f"Tier 2 complete (rc={rc}). Ledger: {len(ledger['failing_tasks'])} failing.")
        return

    # Tier 1: smoke
    all_tasks = discover_all_tasks(lab_root)
    ledger = load_ledger()
    failing = list(ledger.get("failing_tasks", {}).keys())
    smoke = pick_smoke_tasks(
        all_tasks, failing, n_random=args.random, cap_total=args.cap_total,
    )

    smoke_set = set(smoke)
    n_failing_in_smoke = len(set(failing) & smoke_set)
    n_random_in_smoke = len(smoke) - n_failing_in_smoke
    print(f"Tier 1 smoke set: {len(smoke)} tasks "
          f"(cap={args.cap_total}; "
          f"{n_failing_in_smoke} failing + {n_random_in_smoke} random)")

    if args.dry_run:
        for t in smoke:
            tag = "[FAIL]" if t in failing else "[rand]"
            print(f"  {tag} {t}")
        return

    rc = run_tasks(
        smoke, lab_root=lab_root,
        research_mode=args.research_mode,
        judge_model=args.judge_model,
        concurrency=args.concurrency,
    )

    # Score summary + ledger refresh
    p, c, rows = score_summary_for_tasks(smoke, lab_root)
    pct = (100 * p / c) if c else 0
    print(f"\nTier 1 result: {p}/{c} = {pct:.1f}% across {len(rows)} scored tasks (rc={rc})")

    new_ledger = refresh_ledger_from_results(lab_root, threshold=args.failing_threshold)
    save_ledger(new_ledger)
    fixed = set(failing) - set(new_ledger["failing_tasks"].keys())
    new_failing = set(new_ledger["failing_tasks"].keys()) - set(failing)
    if fixed:
        print(f"Tasks moved from failing -> passing ({len(fixed)}):")
        for t in sorted(fixed):
            print(f"  + {t}")
    if new_failing:
        print(f"Tasks newly failing ({len(new_failing)}):")
        for t in sorted(new_failing):
            print(f"  - {t}")
    if not new_ledger["failing_tasks"]:
        print("\n[OK] All known-failing tasks now pass. "
              "Consider escalating to Tier 2 (full sweep) for stability check.")


if __name__ == "__main__":
    main()
