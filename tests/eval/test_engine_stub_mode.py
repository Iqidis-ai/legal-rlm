"""Engine-stub-mode eval harness tests.

Same fixture manifests as store mode, but with an RLMEngine wired to a
ScriptedGeminiClient. Activates the context_packet invariants that need
_assemble_context_packet / _build_capped_gap_section / _build_issue_focus_block.
"""

import pytest

from tests.eval.harness import run_engine_stub_mode
from tests.eval.invariants import run_invariants
from tests.eval.loader import list_fixtures, load_fixture


pytestmark = pytest.mark.eval


@pytest.mark.parametrize("fixture_name", list_fixtures())
def test_engine_stub_mode_invariants(fixture_name: str) -> None:
    fixture = load_fixture(fixture_name)
    result = run_engine_stub_mode(fixture)
    run_invariants(fixture_name, "engine_stub", result)
