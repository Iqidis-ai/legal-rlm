# Irys RLM Project Context

Last curated: 2026-05-03

This is the single maintained context file for coding agents. It replaces the
old ignored root planning files such as `SYSTEM_STATE.md`, `.claude/CLAUDE.md`,
`DECISIONS.md`, `ASSUMPTIONS.md`, `IDEAL_PRODUCT_SPEC.md`,
`CONSOLIDATED_IMPLEMENTATION_SPEC.md`, `ENHANCEMENT_DESIGN.md`, and the
topic-specific design notes that used to live in the repository root.

When this file conflicts with code, tests, or Git history, trust the code first.
Older roadmap language was aspirational and often stale.

## Product Mission

Irys RLM is a durable reasoning substrate that builds and maintains structured
matter models from messy, adversarial, incomplete inputs across complex domains.

The core product is the matter model, not the answer text. Every text answer,
chart, timeline, research packet, and memo should be a downstream view over the
matter model substrate.

## Target Domains

The system is legal-flavored today but the underlying machinery is
domain-general. The neutral kernel (claim, objective_node, entity, artifact,
support_edge, criteria, gaps) maps across all target domains.

| Domain | Artifacts | Claims | Objective Nodes | Entities | Source Calibration |
| --- | --- | --- | --- | --- | --- |
| Legal | Pleadings, contracts, exhibits | Typed assertions with belief state | Legal issues, elements to prove | Parties, witnesses, judges | Advocacy vs operative vs authoritative |
| Finance | 10-K, transcripts, analyst reports | Financial statements, risk factors, guidance | Investment thesis, compliance questions | Companies, executives, analysts, regulators | Management vs auditor vs analyst vs regulator |
| Coding | Source files, PRs, test suites | Behavior claims, design decisions, bug reports | Feature requirements, technical debt items | Developers, components, services | Author vs reviewer vs automated test |
| Academic Research | Papers, datasets, preprints | Hypotheses, findings, methodology claims | Research questions, methodology concerns | Researchers, institutions, funding bodies | Peer-reviewed vs preprint vs replication |
| Biomedical | Trial reports, FDA filings, lab results | Clinical findings, mechanism hypotheses | Disease mechanisms, treatment efficacy | Patient cohorts, compounds, genes | Phase III vs case report vs in-vitro |

Immediate expansion targets: **legal, finance, coding, academic research,
biomedical sciences** — then eventually everything.

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
- SO-5: Source-Aware Intelligence. Source roles are calibrated differently per
  domain — advocacy/operative/authoritative in legal, management/auditor/analyst
  in finance, peer-reviewed/preprint/replication in research.
- SO-6: Quantitative Intelligence as First-Class. Amounts, dates, balances,
  rates, formulas, payments, and domain-specific metrics are structured data.
- SO-7: Missingness Is Modeled. Missing documents, missing metadata, proof
  gaps, absent authorities, and unresolved contradictions are explicit objects.

These sacred outcomes are domain-general. The moat is the reasoning substrate
plus embedded domain-professional thinking, not any single domain vocabulary.

## Domain Portability Architecture

The memory broker provides domain portability through three mechanisms:

- **Neutral Kernel**: Legal names (assertion, issue, actor, evidence_edge) map
  to domain-neutral names (claim, objective_node, entity, support_edge). The
  `TAINT_KIND_ALIASES` in `MemoryBrokerStore` canonicalize both vocabularies.
- **Domain Profiles**: Each matter carries a `domain_profile` declaring its
  domain binding (legal, finance, code, research, biomedical) with a JSON
  profile and content hash. Profile version tracks schema evolution.
- **Profile Mappings**: Cross-domain compatibility declared per
  `(source_profile, target_profile, target_kind, target_namespace)`. Enables
  cross-domain reasoning (e.g., financial fraud = legal + finance profiles).

All five domain profiles are installed: `legal:1`, `finance:1`, `coding:1`,
`academic_research:1`, `biomedical:1`. Each carries full vocabulary (source
roles, belief states, trust weights, taint classes, speech acts). Identity and
cross-domain compatible mappings are registered for all 10 bidirectional pairs
across 8 target kinds. Domain detection signals score content against all
profiles, and the composition substrate merges active vocabularies.

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
| `deliverable` | `DeliverableFamilyHandler` | Structured deliverables such as privilege logs. |
| `clarify` | No LLM answer | Ask for a clarification when the target is ambiguous. |

The cascade code lives in `src/irys/rlm/governance.py`. Keep that module as the
single substrate for routing, execution contracts, and family handlers unless a
concrete scaling reason appears.

## Memory Broker System

The memory broker provides transactional coordination for the shared mutable
matter model substrate. Three dimensions:

1. **Freshness**: Namespace revision counters (41 required namespaces) give O(1)
   staleness checks. CAS writes under `BEGIN IMMEDIATE` prevent TOCTOU gaps.
   Dependency manifests (not yet materialized) will make every read provably
   fresh and every answer auditable.

2. **Taint Isolation**: Object taint lattice (`public_clean` < `clean_with_withheld`
   < `internal_work_product` < `sealed_privileged` < `unknown_taint`).
   Transitive filtering in `build_query_context` excludes tainted objects from
   all orientation fields. Domain-critical for legal (privilege), finance
   (material non-public information), biomedical (patient data), and research
   (embargoed results).

3. **Auditability**: Every user-facing answer should trace back to exactly which
   matter objects it consumed, at which revision, under which domain profile.
   The `OutputEnvelope` carries a `dependency_manifest_hash` field; the
   `dependency_manifest` table and `DependencyManifest` class persist the full
   audit trail of what each output consumed.

Substrate tables (schema v59-v61): `namespace_revision`, `object_taint`,
`domain_profile`, `profile_mapping`. Current state: 1 pilot brokered CAS writer
(clarifications), 95 legacy writers, semantic cache quarantine fails closed.

## Workflow Engine Direction

The engine needs explicit workflow state so different work products can be
planned, checked, resumed, and improved without hiding their requirements inside
one generic synthesis pass.

`ExecutionContract` carries: `family`, `workflow_kind`, `output_contract`.

`InvestigationState` has durable workflow primitives: `RunObjective`,
`Obligation`, `WorkingSet`, `PlanAction`, `ValidationResult`, `OutputEnvelope`.

Synthesis receives a mandatory Workflow Quality Contract section. After
synthesis, the engine runs one focused repair pass for fixable structural misses.

The foreground cold path keeps corpus mapping separate from answer latency.
Cheap inventory questions answer from repository path metadata. Substantive
questions rank files by path/name signals before deep-reading a foreground slice.

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
  timeline views, quantitative stores, memory broker, and reasoning ledger.
- `src/irys/service`: FastAPI service layer and S3-backed repository support.
- `src/irys/ui`: Gradio UI plus in-process and HTTP backends.
- `tests`: matter-model, service, eval harness, governance, termination, and
  usage coverage.
- `docs`: PROJECT_CONTEXT (this file), IMPLEMENTED_REASONING_SYSTEM_SCHEMA
  (portable ontology reference).

## Implemented Snapshot

Major capabilities present in the codebase:

- SQLite-backed matter model with schema migrations (v61) and persistent run sessions.
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
- Mandatory workflow-quality synthesis context plus a one-pass repair loop for
  fixable output-validator failures before final emission.
- Repository-inventory fast path for count/list questions answered from
  filenames and folders without LLM calls, profiling, or deep reading.
- Path/name-aware foreground deep-read selection for large corpora.
- Coverage-driven lead planning and proof-gap surfacing.
- Memory broker substrate (namespace revision CAS, object taint with domain
  profile binding, semantic cache quarantine, taint-aware context assembly).
- Gradio dashboard with matter intelligence panels (proof state, authority
  network, document intelligence, evidence matrix, communication map, timeline,
  quant, LLM analytics), steering, review, trust controls, privilege mode, and
  cost visibility.
- Portable ontology reference (`docs/IMPLEMENTED_REASONING_SYSTEM_SCHEMA.md`)
  documenting the general-purpose substrate vs. legal-specific overlays.

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
- All five domain profiles are implemented with full vocabulary, trust weights,
  and cross-domain mappings. LLM prompts are domain-neutral with per-domain
  synthesis preambles. Extraction (deep-read) prompts inject domain-specific
  vocabulary from `_DOMAIN_DEEP_READ_VOCABULARY`. UI labels are domain-aware
  across all panels. Domain detection is cached per-run on `InvestigationState`.

## Maintenance Rules

- Keep root source files minimal: README, API contract, project metadata,
  deployment files, `src`, `tests`, and `docs`.
- Keep scratch prompts, review notes, benchmark output, pytest temp trees,
  pycache, and local agent artifacts out of Git and out of the root.
- Add durable context here rather than creating new root-level planning docs.
- Prefer updating code, tests, and API contract over preserving old roadmap
  prose.
