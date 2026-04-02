# Experiments

Reverse chronological. All entries must have a corresponding entry in `ledger.jsonl`.
Only Codex-validated conclusions are recorded as findings.

---

## EXP-000 — Ledger Initialization (2026-04-02)

**Status:** INIT
**Purpose:** No experiments have been run yet. Prior roundtable cycles were implementation
work, not controlled experiments with measurable outcomes vs. baselines.
**What we learned:** N/A — baseline state.
**Next:** First real experiment will be a baseline measurement of the current pipeline
(EXP-001) to establish ground truth before the matter model refactor begins.

---

## Planned Experiments (not yet run)

### EXP-001: Current Pipeline Baseline
**Purpose:** Measure the current recursive pipeline on a standardized legal matter corpus
to establish baseline metrics before Phase 5 refactoring.
**Metrics to capture:** query latency, token cost per query, assertion count, citation
accuracy, issue coverage %, reuse rate (expected: 0% since no persistence).
**Why it matters:** Without a baseline, we cannot claim the matter model refactor is an
improvement. Every subsequent experiment compares against this.

### EXP-002: SQLite Matter Model Throughput
**Purpose:** Validate A-007 — does SQLite handle target scale (100k assertions, 10k documents)
within latency budget?
**Metrics to capture:** write throughput (assertions/sec), read latency for graph traversal,
total storage size per matter.
**Why it matters:** Architecture decision D-004 depends on this. If SQLite fails, we need
PostgreSQL before building anything else on top of it.

### EXP-003: Source-Role Classification Accuracy
**Purpose:** Validate A-006 — can document type + metadata → source role be inferred with
>85% accuracy?
**Setup:** Hand-label 100 documents from test corpus with ground-truth source roles.
Run inference. Measure precision/recall per role type.
**Why it matters:** Source-role modeling is a Priority 0 capability (SO-5). If inference
accuracy is low, the architecture must include a human-annotation pathway.
