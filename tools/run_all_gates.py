"""Tiered benchmark gate runner for the Irys build pipeline.

Gate 1 (Fast): Deterministic contract checks — no LLM, milliseconds.
  - Long-context mixed pack (16 queries)
  - Corpus eval ontology validation (133 queries across 5 domains)

Gate 2 (Medium): Corpus eval contract checks with domain-specific validation.
  - All 5 JSONL corpus benchmarks checked for routing correctness

Gate 3 (Full): Harvey LAB end-to-end (requires --full flag + API key).
  - Runs specified Harvey LAB tasks with Irys and scores them

Usage:
    python tools/run_all_gates.py                  # Gate 1 only (fast)
    python tools/run_all_gates.py --medium          # Gates 1 + 2
    python tools/run_all_gates.py --full --task corporate-ma/extract-change-of-control-provisions
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.eval.benchmark_loader import (
    load_benchmark_pack,
    load_corpus_benchmark,
    list_benchmark_packs,
    run_pack_contract_checks,
)
from irys.rlm.governance import infer_task_spec


_CORPUS_FILES = {
    "legal": "legal_corpus_eval.jsonl",
    "finance": "finance_corpus_eval.jsonl",
    "aiml": "aiml_corpus_eval.jsonl",
    "coding": "coding_corpus_eval.jsonl",
    "regression": "legal_tester_regression_eval.jsonl",
}

_DOMAIN_MAP = {
    "legal": "legal",
    "finance": "finance",
    "aiml": "academic_research",
    "coding": "coding",
    "regression": "legal",
}


def _gate1_contract_checks() -> tuple[int, int]:
    """Run deterministic contract checks on all benchmark packs."""
    passed = 0
    failed = 0
    packs = list_benchmark_packs()
    for pack_name in packs:
        pack = load_benchmark_pack(pack_name)
        results = run_pack_contract_checks(pack)
        for r in results:
            if r.passed:
                passed += 1
            else:
                failed += 1
                print(f"  FAIL {pack_name}/{r.query_id}: {', '.join(r.failures)}")
    return passed, failed


def _gate2_corpus_ontology() -> tuple[int, int]:
    """Validate corpus eval queries route to sensible task specs."""
    passed = 0
    failed = 0
    for label, filename in _CORPUS_FILES.items():
        domain = _DOMAIN_MAP[label]
        try:
            queries = load_corpus_benchmark(filename)
        except FileNotFoundError:
            print(f"  SKIP {label}: {filename} not found")
            continue
        for q in queries:
            spec = infer_task_spec(q.query, domain)
            issues: list[str] = []
            if spec.task_type == "unknown":
                issues.append(f"task_type=unknown for '{q.query[:60]}...'")
            if spec.answer_shape == "unknown":
                issues.append(f"answer_shape=unknown for '{q.query[:60]}...'")
            if not spec.required_evidence:
                issues.append(f"no required_evidence for '{q.query[:60]}...'")
            if issues:
                failed += 1
                for issue in issues:
                    print(f"  FAIL {label}/{q.id}: {issue}")
            else:
                passed += 1
    return passed, failed


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    run_medium = "--medium" in args or "--full" in args
    run_full = "--full" in args
    task_id = None
    if "--task" in args:
        idx = args.index("--task")
        if idx + 1 < len(args):
            task_id = args[idx + 1]

    total_passed = 0
    total_failed = 0

    # Gate 1: Contract checks
    print("=" * 60)
    print("GATE 1: Deterministic Contract Checks (no LLM)")
    print("=" * 60)
    t0 = time.time()
    p, f = _gate1_contract_checks()
    total_passed += p
    total_failed += f
    dt = time.time() - t0
    status = "PASS" if f == 0 else "FAIL"
    print(f"  Gate 1: {p}/{p+f} passed ({dt:.1f}s) — {status}")
    if f > 0 and not run_medium:
        print("\nGate 1 failed. Fix contract check failures before proceeding.")
        return 1

    # Gate 2: Corpus eval ontology validation
    if run_medium:
        print()
        print("=" * 60)
        print("GATE 2: Corpus Eval Ontology Validation (no LLM)")
        print("=" * 60)
        t0 = time.time()
        p, f = _gate2_corpus_ontology()
        total_passed += p
        total_failed += f
        dt = time.time() - t0
        status = "PASS" if f == 0 else "FAIL"
        print(f"  Gate 2: {p}/{p+f} passed ({dt:.1f}s) — {status}")
        if f > 0 and not run_full:
            print("\nGate 2 failed. Fix ontology routing before proceeding.")
            return 1

    # Gate 3: Harvey LAB end-to-end
    if run_full:
        print()
        print("=" * 60)
        print("GATE 3: Harvey LAB End-to-End Benchmark")
        print("=" * 60)
        if not task_id:
            print("  ERROR: --full requires --task <task-id>")
            return 2
        print(f"  Task: {task_id}")
        print("  (Delegate to tools/lab_benchmark.py)")
        import subprocess
        cmd = [
            sys.executable, str(REPO_ROOT / "tools" / "lab_benchmark.py"),
            "--task", task_id, "--auto-score", "--research-mode", "simple",
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            total_failed += 1
            print(f"  Gate 3: FAIL (exit code {result.returncode})")
        else:
            total_passed += 1
            print("  Gate 3: PASS")

    # Summary
    print()
    print("=" * 60)
    print("GATE SUMMARY")
    print("=" * 60)
    print(f"  Total checks: {total_passed + total_failed}")
    print(f"  Passed:       {total_passed}")
    print(f"  Failed:       {total_failed}")
    return 1 if total_failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
