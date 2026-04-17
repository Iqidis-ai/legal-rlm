"""JSON manifest loader for eval fixtures.

Loads a fixture manifest file from tests/eval/fixtures/<name>.json into the
typed FixtureSpec dataclass from schema.py. Unknown fields raise — we want a
loud failure when a manifest drifts from the schema.
"""

from __future__ import annotations

import json
from pathlib import Path

from .schema import (
    FixtureAssertion,
    FixtureDocument,
    FixtureGap,
    FixtureIssue,
    FixtureIssueLink,
    FixtureMaintenanceStep,
    FixtureSpec,
    InvariantSpec,
)


_FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _tuple_of(cls, items):
    return tuple(cls(**item) for item in (items or []))


def load_fixture(name: str) -> FixtureSpec:
    path = _FIXTURE_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No eval fixture at {path}")
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    return FixtureSpec(
        name=data["name"],
        intent=data["intent"],
        sacred_outcomes=tuple(data.get("sacred_outcomes", [])),
        documents=_tuple_of(FixtureDocument, data.get("documents")),
        assertions=_tuple_of(FixtureAssertion, data.get("assertions")),
        issues=tuple(
            FixtureIssue(
                alias=i["alias"],
                title=i["title"],
                issue_type=i.get("issue_type", "claim"),
                materiality=i.get("materiality", 0.6),
                salience=i.get("salience", 0.5),
                burden_side=i.get("burden_side"),
                predicates=tuple(i.get("predicates", [])),
            )
            for i in data.get("issues", [])
        ),
        issue_links=_tuple_of(FixtureIssueLink, data.get("issue_links")),
        gaps=_tuple_of(FixtureGap, data.get("gaps")),
        maintenance=tuple(
            FixtureMaintenanceStep(
                step=m["step"], payload=m.get("payload", {})
            )
            for m in data.get("maintenance", [])
        ),
        scripted_responses=dict(data.get("scripted_responses", {})),
        invariants=tuple(
            InvariantSpec(
                name=iv["name"],
                group=iv["group"],
                requires=tuple(iv.get("requires", [])),
                params=iv.get("params", {}),
            )
            for iv in data.get("invariants", [])
        ),
    )


def list_fixtures() -> list[str]:
    return sorted(p.stem for p in _FIXTURE_DIR.glob("*.json"))
