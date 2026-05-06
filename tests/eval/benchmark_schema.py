"""Typed schema for long-context benchmark query packs.

These packs are metadata-first: they let us run cheap ontology and route
contract checks before spending on full end-to-end long-context investigations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BenchmarkRunPolicy:
    default_slice_size: int = 5
    max_expensive_queries_per_slice: int = 2
    cheap_contract_checks_first: bool = True
    record_runtime_breakdown: bool = True


@dataclass(frozen=True)
class BenchmarkQuery:
    id: str
    family: str
    style: str
    query: str
    expected_task_type: str
    expected_answer_shape: str
    cached_state_allowed: bool
    required_evidence: tuple[str, ...]
    forbidden_output_patterns: tuple[str, ...] = ()
    expected_negative_statuses: tuple[str, ...] = ()
    max_runtime_seconds: int | None = None


@dataclass(frozen=True)
class BenchmarkPack:
    name: str
    intent: str
    run_policy: BenchmarkRunPolicy
    queries: tuple[BenchmarkQuery, ...]

    def ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.queries)

    def families(self) -> set[str]:
        return {item.family for item in self.queries}

    def styles(self) -> set[str]:
        return {item.style for item in self.queries}


@dataclass(frozen=True)
class CorpusBenchmarkQuery:
    id: str
    query: str
    category: str
    difficulty: str
    required_capabilities: tuple[str, ...]
    min_documents_needed: int
    gold_answer_sketch: str
    ontology_stress: tuple[str, ...]


@dataclass(frozen=True)
class ContractCheckResult:
    query_id: str
    passed: bool
    failures: tuple[str, ...] = ()
    observed: dict[str, Any] = field(default_factory=dict)
