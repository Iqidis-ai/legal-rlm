# Irys RLM Project Context

Last curated: 2026-04-23

This is the single maintained context file for coding agents. It replaces the
old ignored root planning files such as `SYSTEM_STATE.md`, `.claude/CLAUDE.md`,
`DECISIONS.md`, `ASSUMPTIONS.md`, `IDEAL_PRODUCT_SPEC.md`,
`CONSOLIDATED_IMPLEMENTATION_SPEC.md`, `ENHANCEMENT_DESIGN.md`, and the
topic-specific design notes that used to live in the repository root.

When this file conflicts with code, tests, or Git history, trust the code first.
Older roadmap language was aspirational and often stale.

## Product Mission

Irys RLM is a legal intelligence system that builds and maintains a durable
matter model from messy, adversarial, incomplete legal inputs.

The core product is the matter model, not the answer text. Every text answer,
chart, timeline, research packet, and memo should be a downstream view over the
matter model substrate.

## Sacred Outcomes

- SO-1: Durable Matter Model. Repeated work should reuse persistent matter
  intelligence instead of rediscovering stable structure.
- SO-2: Typed Assertion Graph with Truth Maintenance. Facts are stored as
  typed assertions with source role, speech act, support/attack links, and
  revisable belief state.
- SO-3: User-Steerable Reasoning. Users can interrupt, redirect, correct,
  annotate, and review the matter model without restarting from scratch.
- SO-4: Issue-Driven Architecture. Retrieval and synthesis move issue coverage
  forward rather than matching query tokens generically.
- SO-5: Source-Aware Intelligence. Advocacy, operative, authoritative,
  informal, draft, and post-hoc sources are calibrated differently.
- SO-6: Quantitative Intelligence as First-Class. Amounts, dates, balances,
  rates, formulas, payments, and damages assumptions are structured data.
- SO-7: Missingness Is Modeled. Missing documents, missing metadata, proof
  gaps, absent authorities, and unresolved contradictions are explicit objects.

## Current Architecture

The front door is the Answerability-Governed Cost Cascade. A compact
answerability snapshot is classified before the system spends on a full
investigation. The classifier emits an `ExecutionContract`; downstream handlers
and termination logic read that same contract.

Current query families:

| Family | Handler | Purpose |
| --- | --- | --- |
| `investigate` | `RLMEngine.investigate()` | Full recursive investigation for fresh or unresolved questions. |
| `read` | `ReadFamilyHandler` | Synthesize from existing matter state and conversation. |
| `query` | `QueryFamilyHandler` | Direct matter-table lookup or enumeration. |
| `trace` | `TraceFamilyHandler` | Explain provenance or why the system said something. |
| `steer` | `SteerFamilyHandler` | Parse corrections, annotations, overrides, and previews. |
| `compare` | `CompareFamilyHandler` | Compare current state to prior run state. |
| `scenario` | `ScenarioFamilyHandler` | One-turn counterfactual read with temporary assumptions. |
| `deliverable` | `DeliverableFamilyHandler` | Structured legal deliverables such as privilege logs. |
| `clarify` | No LLM answer | Ask for a clarification when the target is ambiguous. |

The cascade code lives in `src/irys/rlm/governance.py`. Keep that module as the
single substrate for routing, execution contracts, and family handlers unless a
concrete scaling reason appears.

## Workflow Engine Direction

The next frontier is not "more prompts." The engine needs explicit workflow
state so different work products can be planned, checked, resumed, and improved
without hiding their requirements inside one generic synthesis pass.

The core distinction:

- Analysis workflows answer "what is true, what is supported, what is missing,
  and how confident are we?"
- Drafting workflows answer "what document should exist, for which audience,
  with which required sections, citations, procedural constraints, privilege
  boundaries, and review gates?"
- Solution workflows answer "what should we do, under which assumptions, with
  which alternatives, tradeoffs, risks, dependencies, and validation tests?"
- Lookup, trace, compare, steer, scenario, clarify, and deliverable routes are
  execution families, but their outputs still need workflow contracts.

`ExecutionContract` now carries:

- `family`: the route that will execute.
- `workflow_kind`: the kind of work being performed, such as `analysis`,
  `drafting`, `solution`, `lookup`, `trace`, or `steering`.
- `output_contract`: a small machine-readable contract describing the expected
  output shape and hard requirements.

`InvestigationState` now has durable workflow primitives:

- `RunObjective`: user goal, workflow kind, output shape, audience, policy
  audience, success criteria, constraints, and source query.
- `Obligation`: a required condition such as citation support, element
  coverage, authority coverage, procedural compliance, privilege safety,
  gap disclosure, or review-before-service.
- `WorkingSet`: the verified/candidate assertions, issues, gaps, documents,
  authorities, assumptions, and dependency manifest hash the workflow may use.
- `PlanAction`: planned operator steps that target obligations.
- `ValidationResult`: validator output, blocking issues, warnings, score, and
  per-obligation status.
- `OutputEnvelope`: auditable wrapper around final user-facing text, including
  workflow kind, output shape, objective id, dependency manifest hash, validation
  results, blockers, warnings, and review-required status.

The intended architecture is an objective/obligation planner rather than a
rigid mode switch. A user may ask for a draft, but the system should build an
objective, derive obligations, assemble a working set, choose operators, render
the output, validate it, and either ship with caveats or loop back to fill the
blocking gaps. The same mechanism applies to solution design and research
review; only the obligation templates and validators change.

Near-term implementation order:

1. Populate `RunObjective` and default obligations from the classifier contract
   at run start.
2. Emit synthesis and sufficiency-probe answers through `OutputEnvelope` while
   preserving the existing `final_output` compatibility field.
3. Build workflow-specific obligation templates for drafts, solutions, and
   analysis memos.
4. Teach context assembly to emit a `WorkingSet` plus dependency manifest, not
   only prompt text.
5. Add validators that recompute obligations after rendering, starting with
   citation coverage, unsupported factual claims, open proof gaps, authority
   coverage, and privilege leakage.
6. Record validation results as ledger/output events so failed drafts and
   solution plans become reusable training signals for future runs.
7. Add quantitative-statistical obligations for quant-heavy work: distribution
   checks, outlier detection, reconciliation variance, confidence intervals,
   scenario/sensitivity bands, and numeric assumption audit trails.

## Canonical Entry Points

- UI: `python -m irys.ui.app`
- Service: `uvicorn irys.service.api:app --host 0.0.0.0 --port 8000`
- Python API: `from irys import Irys`
- FastAPI app: `src/irys/service/api.py`
- Gradio app: `src/irys/ui/app.py`
- API contract and target roadmap: `API_CONTRACTS.md`

Do not recreate root wrapper scripts such as `run_server.py` unless there is a
specific product reason. The module and uvicorn entry points are the maintained
surface.

## Repo Shape

- `src/irys/core`: Gemini client, pricing, usage accounting, reader,
  repository, and search.
- `src/irys/rlm`: investigation engine, checkpoints, state, governance, and
  research-mode budget control.
- `src/irys/matter`: SQLite schema, canonical stores, rollups, evidence and
  timeline views, quantitative stores, and reasoning ledger.
- `src/irys/service`: FastAPI service layer and S3-backed repository support.
- `src/irys/ui`: Gradio UI plus in-process and HTTP backends.
- `tests`: matter-model, service, eval harness, governance, termination, and
  usage coverage.

## Implemented Snapshot

Major capabilities present in the codebase:

- SQLite-backed matter model with schema migrations and persistent run sessions.
- Assertion, issue, actor, gap, proof, quantitative, trust, review, and
  reasoning-ledger stores.
- LLM telemetry persisted per request in `llm_call`, with cost and usage rollups.
- Review queue and verification APIs for promoting or rejecting candidate
  intelligence.
- Content-policy guardrails for clean-mode reads, synthesis, timeline, evidence
  matrix, and exports.
- Cost cascade routing for investigate/read/query/trace/steer/compare/scenario/
  deliverable/clarify flows.
- Workflow contract metadata and checkpoint-safe objective, obligation,
  working-set, plan-action, validation-result, and output-envelope primitives.
- Coverage-driven lead planning and proof-gap surfacing.
- Gradio dashboard with matter intelligence panels, steering, review, trust
  controls, privilege mode, and cost visibility.

## Targets And Metrics

The measurable SO targets used by `MatterModel.get_so_metrics()` are:

- `assertion_structure_rate >= 1.0`
- `source_role_known_rate >= 0.9`
- `issue_coverage_avg >= 0.8`
- `reuse_rate >= 0.7` on repeated queries over a stable matter
- `numeric_extraction_rate >= 0.9`
- `provenance_attribution_rate >= 0.9`
- `steerability == True`
- `belief_revision == True`

These are product-health targets, not unit-test substitutes. A matter with too
little data can legitimately report `None` for some metrics.

## Known Constraints

- OCR is not implemented. PDFs must contain machine-readable text.
- Gemini is the active production provider path.
- Authority and case-law lookup is scaffolded but not wired to an external
  authority corpus.
- Scenario assumptions are one-turn overrides, not persisted sessions.
- The `/api/v1/*` shape in `API_CONTRACTS.md` is still a target contract. The
  current FastAPI app exposes unprefixed runtime routes.
- Full test runs in this OneDrive-backed environment can hit temp-directory
  cleanup permission issues. Targeted tests are usually more reliable.

## Maintenance Rules

- Keep root source files minimal: README, API contract, project metadata,
  deployment files, `src`, `tests`, and `docs`.
- Keep scratch prompts, review notes, benchmark output, pytest temp trees,
  pycache, and local agent artifacts out of Git and out of the root.
- Add durable context here rather than creating new root-level planning docs.
- Prefer updating code, tests, and API contract over preserving old roadmap
  prose.
