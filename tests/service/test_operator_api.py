"""API tests for the new operator-substrate observability endpoints."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from irys.matter import MatterModel
from irys.service.api import _active_matter_models, app


MATTER_ID = "test_matter_op_api"


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def matter():
    m = MatterModel.open_in_memory()
    _active_matter_models[MATTER_ID] = m
    yield m
    _active_matter_models.pop(MATTER_ID, None)


def _seed_invocation_with_artifact(
    matter, *, agent_id="obligation_coverage_matrix",
    run_id="run-1", invocation_id="inv-1", artifact_id="a-1",
    status="success", artifact_kind="obligation.coverage_matrix.v1",
    artifact_payload=None, requirement="optional",
):
    matter.db.execute(
        "INSERT INTO sub_agent_invocation (id, matter_id, run_id, agent_id, "
        "agent_version, phase, status, requirement, invocation_at, input_hash, "
        "latency_ms, cost_estimate, llm_calls, success_bool, capability_tags_json, "
        "token_estimate) VALUES (?, ?, ?, ?, 1, 'pre_synthesis', ?, ?, "
        "'2026-05-08T00:00:00', 'h0', 100, 0.001, 1, 1, '[]', 500)",
        (invocation_id, matter.matter_id, run_id, agent_id, status, requirement),
    )
    matter.db.execute(
        "INSERT INTO agent_artifact (id, matter_id, invocation_id, "
        "artifact_kind, artifact_key, payload_json, synthesis_visibility, "
        "verification_state, created_at) VALUES (?, ?, ?, ?, ?, ?, "
        "'answer_ingredient', 'verified', '2026-05-08T00:00:01')",
        (artifact_id, matter.matter_id, invocation_id, artifact_kind,
         f"k:{artifact_id}",
         json.dumps(artifact_payload or {"schema_ref": artifact_kind})),
    )


def test_get_operator_invocations_lists_recent(client, matter):
    _seed_invocation_with_artifact(matter)
    resp = client.get(f"/matter/{MATTER_ID}/operator-invocations")
    assert resp.status_code == 200
    body = resp.json()
    assert body["matter_id"] == matter.matter_id
    assert len(body["invocations"]) == 1
    inv = body["invocations"][0]
    assert inv["agent_id"] == "obligation_coverage_matrix"
    assert inv["artifact_count"] == 1
    assert "obligation.coverage_matrix.v1" in inv["artifact_kinds"]


def test_get_operator_invocations_filters(client, matter):
    _seed_invocation_with_artifact(matter, agent_id="hhi_calc",
                                   invocation_id="inv-A", artifact_id="a-A",
                                   artifact_kind="hhi.calculation")
    _seed_invocation_with_artifact(matter, agent_id="term_grid_op",
                                   invocation_id="inv-B", artifact_id="a-B",
                                   artifact_kind="term_grid.v1")
    resp = client.get(
        f"/matter/{MATTER_ID}/operator-invocations?agent_id=hhi_calc",
    )
    body = resp.json()
    assert len(body["invocations"]) == 1
    assert body["invocations"][0]["agent_id"] == "hhi_calc"


def test_get_operator_invocations_validates_requirement(client, matter):
    resp = client.get(
        f"/matter/{MATTER_ID}/operator-invocations?requirement=bogus",
    )
    assert resp.status_code == 422


def test_get_operator_artifacts_payload_summary(client, matter):
    _seed_invocation_with_artifact(
        matter, artifact_payload={
            "schema_ref": "obligation.coverage_matrix.v1",
            "n_total": 5, "n_met": 3, "n_critical_missing": 1,
            "n_required_missing": 1, "matrix_status": "success",
        },
    )
    resp = client.get(f"/matter/{MATTER_ID}/operator-artifacts")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["artifacts"]) == 1
    art = body["artifacts"][0]
    assert art["payload_summary"]["n_total"] == 5
    assert art["payload"] is None  # default include_payload=false


def test_get_operator_artifacts_with_payload(client, matter):
    _seed_invocation_with_artifact(matter)
    resp = client.get(
        f"/matter/{MATTER_ID}/operator-artifacts?include_payload=true",
    )
    body = resp.json()
    assert body["artifacts"][0]["payload"] is not None


def test_get_single_operator_artifact(client, matter):
    _seed_invocation_with_artifact(
        matter, artifact_id="art-fetch",
        artifact_payload={"schema_ref": "term_grid.v1", "rows": [{"x": 1}]},
        artifact_kind="term_grid.v1",
    )
    resp = client.get(
        f"/matter/{MATTER_ID}/operator-artifacts/art-fetch",
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["artifact"]["payload"]["schema_ref"] == "term_grid.v1"


def test_get_single_operator_artifact_404(client, matter):
    resp = client.get(f"/matter/{MATTER_ID}/operator-artifacts/none")
    assert resp.status_code == 404


def test_get_requirement_coverage_returns_latest_matrix(client, matter):
    _seed_invocation_with_artifact(
        matter,
        artifact_payload={
            "schema_ref": "obligation.coverage_matrix.v1",
            "n_total": 4, "n_met": 4, "n_critical_missing": 0,
            "rows": [], "matrix_status": "success",
        },
    )
    resp = client.get(f"/matter/{MATTER_ID}/requirement-coverage")
    assert resp.status_code == 200
    body = resp.json()
    assert body["matrix"]["n_total"] == 4
    assert body["validator_failure"] is None


def test_get_requirement_coverage_empty_matter(client, matter):
    resp = client.get(f"/matter/{MATTER_ID}/requirement-coverage")
    assert resp.status_code == 200
    body = resp.json()
    assert body["matrix"] is None
    assert body["validator_failure"] is None


def test_get_term_grid_aggregates_rows(client, matter):
    _seed_invocation_with_artifact(
        matter, agent_id="term_grid_op", invocation_id="inv-tg",
        artifact_id="art-tg", artifact_kind="term_grid.v1",
        artifact_payload={
            "schema_ref": "term_grid.v1",
            "topics": ["covenant", "exception"],
            "rows": [
                {"row_id": "r1", "topic": "covenant",
                 "actor": "Borrower", "obligation_or_right": "must"},
                {"row_id": "r2", "topic": "exception",
                 "actor": "Lender", "obligation_or_right": "may"},
            ],
        },
    )
    resp = client.get(f"/matter/{MATTER_ID}/term-grid")
    body = resp.json()
    assert body["n_rows"] == 2
    assert "covenant" in body["topics"]


def test_get_term_grid_topic_filter(client, matter):
    _seed_invocation_with_artifact(
        matter, agent_id="term_grid_op", invocation_id="inv-tg2",
        artifact_id="art-tg2", artifact_kind="term_grid.v1",
        artifact_payload={
            "schema_ref": "term_grid.v1", "topics": ["covenant", "exception"],
            "rows": [
                {"row_id": "r1", "topic": "covenant", "actor": "Borrower"},
                {"row_id": "r2", "topic": "exception", "actor": "Lender"},
            ],
        },
    )
    resp = client.get(
        f"/matter/{MATTER_ID}/term-grid?topic=covenant",
    )
    body = resp.json()
    assert body["n_rows"] == 1
    assert body["rows"][0]["topic"] == "covenant"


def test_get_operator_summary_aggregates(client, matter):
    _seed_invocation_with_artifact(matter, agent_id="op_a",
                                    invocation_id="inv-a1", artifact_id="aa1")
    _seed_invocation_with_artifact(matter, agent_id="op_a",
                                    invocation_id="inv-a2", artifact_id="aa2")
    _seed_invocation_with_artifact(matter, agent_id="op_b",
                                    invocation_id="inv-b1", artifact_id="ab1",
                                    status="error",
                                    requirement="blocking_validator")
    resp = client.get(f"/matter/{MATTER_ID}/operator-summary")
    body = resp.json()
    assert body["n_invocations"] == 3
    assert body["n_success"] == 2
    assert body["n_failed"] == 1
    assert body["n_blocking_failures"] == 1
    by_agent = {a["agent_id"]: a["n"] for a in body["by_agent"]}
    assert by_agent["op_a"] == 2
    assert by_agent["op_b"] == 1


def test_anti_gaming_no_benchmark_metadata_in_responses(client, matter):
    """Anti-gaming sentinel: scan all 6 endpoint responses for
    benchmark identifiers / rubric vocabulary leaking through."""
    _seed_invocation_with_artifact(matter)
    forbidden = ("harvey-lab", "lab_task_id", "rubric_id", "C-001")
    for path in (
        f"/matter/{MATTER_ID}/operator-invocations",
        f"/matter/{MATTER_ID}/operator-artifacts",
        f"/matter/{MATTER_ID}/requirement-coverage",
        f"/matter/{MATTER_ID}/term-grid",
        f"/matter/{MATTER_ID}/operator-summary",
    ):
        resp = client.get(path)
        body = json.dumps(resp.json()).lower()
        for f in forbidden:
            assert f not in body, f"{f!r} leaked into {path}"
