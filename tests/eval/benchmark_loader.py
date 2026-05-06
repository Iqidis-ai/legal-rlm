"""Loaders and cheap contract checks for long-context benchmark packs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from irys.rlm.governance import infer_task_spec

from .benchmark_schema import (
    BenchmarkPack,
    BenchmarkQuery,
    BenchmarkRunPolicy,
    ContractCheckResult,
    CorpusBenchmarkQuery,
)


_BENCHMARK_PACK_DIR = Path(__file__).parent / "benchmark_packs"
_REPO_ROOT = Path(__file__).parents[2]
_CORPUS_BENCHMARK_DIR = _REPO_ROOT / "benchmarks"


def _as_tuple(value) -> tuple[str, ...]:
    return tuple(str(item) for item in (value or ()))


def load_benchmark_pack(name: str) -> BenchmarkPack:
    path = _BENCHMARK_PACK_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No benchmark pack at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    policy = data.get("run_policy") or {}
    return BenchmarkPack(
        name=data["name"],
        intent=data["intent"],
        run_policy=BenchmarkRunPolicy(
            default_slice_size=int(policy.get("default_slice_size", 5)),
            max_expensive_queries_per_slice=int(
                policy.get("max_expensive_queries_per_slice", 2)
            ),
            cheap_contract_checks_first=bool(
                policy.get("cheap_contract_checks_first", True)
            ),
            record_runtime_breakdown=bool(
                policy.get("record_runtime_breakdown", True)
            ),
        ),
        queries=tuple(
            BenchmarkQuery(
                id=item["id"],
                family=item["family"],
                style=item["style"],
                query=item["query"],
                expected_task_type=item["expected_task_type"],
                expected_answer_shape=item["expected_answer_shape"],
                cached_state_allowed=bool(item["cached_state_allowed"]),
                required_evidence=_as_tuple(item.get("required_evidence")),
                forbidden_output_patterns=_as_tuple(
                    item.get("forbidden_output_patterns")
                ),
                expected_negative_statuses=_as_tuple(
                    item.get("expected_negative_statuses")
                ),
                max_runtime_seconds=item.get("max_runtime_seconds"),
            )
            for item in data.get("queries", ())
        ),
    )


def list_benchmark_packs() -> list[str]:
    return sorted(path.stem for path in _BENCHMARK_PACK_DIR.glob("*.json"))


def load_corpus_benchmark(filename: str) -> tuple[CorpusBenchmarkQuery, ...]:
    path = _CORPUS_BENCHMARK_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"No corpus benchmark at {path}")
    rows: list[CorpusBenchmarkQuery] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        data = json.loads(line)
        try:
            rows.append(
                CorpusBenchmarkQuery(
                    id=data["id"],
                    query=data["query"],
                    category=data["category"],
                    difficulty=data["difficulty"],
                    required_capabilities=_as_tuple(data.get("required_capabilities")),
                    min_documents_needed=int(data.get("min_documents_needed", 0)),
                    gold_answer_sketch=data["gold_answer_sketch"],
                    ontology_stress=_as_tuple(data.get("ontology_stress")),
                )
            )
        except KeyError as exc:
            raise ValueError(f"{path}:{lineno} missing required key {exc}") from exc
    return tuple(rows)


def run_pack_contract_checks(
    pack: BenchmarkPack,
    *,
    domain: str = "legal",
) -> tuple[ContractCheckResult, ...]:
    """Check benchmark metadata against deterministic task ontology.

    This intentionally does not call an LLM or read matter folders. It protects
    the cheapest contract layer before any expensive long-context run.
    """

    results: list[ContractCheckResult] = []
    for query in pack.queries:
        spec = infer_task_spec(query.query, domain)
        failures: list[str] = []
        observed = {
            "task_type": spec.task_type,
            "answer_shape": spec.answer_shape,
            "required_evidence": list(spec.required_evidence),
            "cached_state_allowed": spec.cached_state_allowed,
            "fresh_extraction_required": spec.fresh_extraction_required,
        }
        if spec.task_type != query.expected_task_type:
            failures.append(
                f"task_type {spec.task_type} != {query.expected_task_type}"
            )
        if spec.answer_shape != query.expected_answer_shape:
            failures.append(
                f"answer_shape {spec.answer_shape} != {query.expected_answer_shape}"
            )
        if tuple(spec.required_evidence) != query.required_evidence:
            failures.append(
                "required_evidence "
                f"{tuple(spec.required_evidence)} != {query.required_evidence}"
            )
        if spec.cached_state_allowed is not query.cached_state_allowed:
            failures.append(
                "cached_state_allowed "
                f"{spec.cached_state_allowed} != {query.cached_state_allowed}"
            )
        results.append(
            ContractCheckResult(
                query_id=query.id,
                passed=not failures,
                failures=tuple(failures),
                observed=observed,
            )
        )
    return tuple(results)


def check_forbidden_output_patterns(
    query: BenchmarkQuery,
    output_text: str,
) -> ContractCheckResult:
    lowered = (output_text or "").lower()
    hits = [
        pattern
        for pattern in query.forbidden_output_patterns
        if pattern.lower() in lowered
    ]
    return ContractCheckResult(
        query_id=query.id,
        passed=not hits,
        failures=tuple(f"forbidden output pattern: {pattern}" for pattern in hits),
        observed={"forbidden_patterns_hit": hits},
    )


def ensure_unique_ids(items: Iterable[BenchmarkQuery | CorpusBenchmarkQuery]) -> None:
    ids = [item.id for item in items]
    duplicates = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate benchmark ids: {', '.join(duplicates)}")
