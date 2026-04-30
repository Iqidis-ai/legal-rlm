"""Regression tests for memory-surface manifest coverage."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LINTER_PATH = ROOT / "tools" / "check_memory_namespace_coverage.py"

spec = importlib.util.spec_from_file_location("memory_surface_linter", LINTER_PATH)
assert spec is not None and spec.loader is not None
memory_surface_linter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memory_surface_linter)


def _manifest() -> dict:
    return json.loads((ROOT / "memory_surface_manifest.json").read_text(encoding="utf-8"))


def test_manifest_covers_discovered_user_facing_family_handlers():
    manifest = _manifest()
    declared = {
        item["surface"]
        for item in memory_surface_linter.iter_surface_entries(
            manifest, "read_surfaces", "read_surface_groups"
        )
    }

    discovered = memory_surface_linter.discovered_family_read_surfaces()
    assert discovered <= declared
    assert "TraceFamilyHandler.run" in declared


def test_manifest_covers_discovered_canonical_writers():
    manifest = _manifest()
    declared = {
        item["surface"]
        for item in memory_surface_linter.iter_surface_entries(
            manifest, "write_surfaces", "write_surface_groups"
        )
    }

    discovered = memory_surface_linter.discovered_writer_surfaces()
    assert discovered <= declared
    assert "QuantStore.record" in declared


def test_manifest_writer_namespaces_match_derived_write_tables():
    manifest = _manifest()
    protocol_text = (ROOT / "MEMORY_COORDINATION_PROTOCOL.md").read_text(
        encoding="utf-8"
    )

    assert memory_surface_linter.writer_namespace_mismatches(
        manifest, protocol_text
    ) == []


def test_manifest_rejects_overbroad_writer_namespaces():
    manifest = copy.deepcopy(_manifest())
    protocol_text = (ROOT / "MEMORY_COORDINATION_PROTOCOL.md").read_text(
        encoding="utf-8"
    )
    target = next(
        item
        for item in manifest["write_surfaces"]
        if item["surface"] == "ClarificationStore.answer_question"
    )
    target["writes"] = sorted(set(target["writes"]) | {"claims"})

    issues = memory_surface_linter.writer_namespace_mismatches(
        manifest, protocol_text
    )

    assert "ClarificationStore.answer_question:extra:claims" in issues
