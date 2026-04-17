"""Store-mode eval harness tests.

Each synthetic micro-matter loads as a JSON manifest, seeds an in-memory
MatterModel, runs declared maintenance steps, and executes every activatable
invariant. Failure messages identify fixture + mode + invariant.
"""

import pytest

from tests.eval.harness import run_store_mode
from tests.eval.invariants import run_invariants
from tests.eval.loader import list_fixtures, load_fixture


pytestmark = pytest.mark.eval


@pytest.mark.parametrize("fixture_name", list_fixtures())
def test_store_mode_invariants(fixture_name: str) -> None:
    fixture = load_fixture(fixture_name)
    result = run_store_mode(fixture)
    # Store mode cannot satisfy context-packet invariants that require the
    # engine; those skip with a clear reason instead of silently passing.
    skipped = run_invariants(fixture_name, "store", result)
    skipped_names = {name for name, _ in skipped}
    # Every fixture must declare at least one invariant that runs in store
    # mode; otherwise the fixture has no regression value here.
    runnable = [
        spec for spec in fixture.invariants
        if spec.name not in skipped_names and spec.group != "context_packet"
    ]
    assert runnable, (
        f"fixture {fixture_name!r} declares no store-mode-runnable "
        "invariants; add one or convert this fixture to engine-stub-only. "
        f"skipped: {skipped}"
    )
