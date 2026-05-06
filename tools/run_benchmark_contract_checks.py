"""Run cheap task-ontology contract checks for a benchmark pack.

This is intentionally LLM-free and source-free. It verifies that benchmark
queries map to the expected task contracts before any expensive run.

Usage:
    python tools/run_benchmark_contract_checks.py long_context_mixed_v1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.eval.benchmark_loader import (  # noqa: E402
    load_benchmark_pack,
    run_pack_contract_checks,
)


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    pack_name = args[0] if args else "long_context_mixed_v1"
    pack = load_benchmark_pack(pack_name)
    results = run_pack_contract_checks(pack)
    failed = False
    for result in results:
        row = {
            "pack": pack.name,
            "query_id": result.query_id,
            "passed": result.passed,
            "failures": list(result.failures),
            "observed": result.observed,
        }
        print(json.dumps(row, sort_keys=True))
        failed = failed or not result.passed
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
