"""MVP.1 runtime budget gate.

Loads every fixture manifest, runs both store mode and engine-stub mode once
under a strict wall-clock budget. MVP.1 acceptance criterion #4 requires the
whole fixture suite to complete in under 5 minutes; the realistic budget for
in-memory SQLite plus a scripted client is well under a minute.
"""

import time

import pytest

from tests.eval.harness import run_engine_stub_mode, run_store_mode
from tests.eval.invariants import run_invariants
from tests.eval.loader import list_fixtures, load_fixture


pytestmark = pytest.mark.eval


_BUDGET_SECONDS = 300.0  # AC #4


def test_full_suite_under_budget() -> None:
    fixtures = list_fixtures()
    assert fixtures, "at least one fixture manifest must exist"
    start = time.perf_counter()
    for name in fixtures:
        fixture = load_fixture(name)
        store_result = run_store_mode(fixture)
        run_invariants(name, "store", store_result)
        stub_result = run_engine_stub_mode(fixture)
        run_invariants(name, "engine_stub", stub_result)
    elapsed = time.perf_counter() - start
    assert elapsed < _BUDGET_SECONDS, (
        f"MVP.1 fixture suite exceeded {_BUDGET_SECONDS:.0f}s budget: "
        f"{elapsed:.2f}s"
    )
