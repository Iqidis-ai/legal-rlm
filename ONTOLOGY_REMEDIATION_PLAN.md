# Ontology Remediation Plan

This plan is the working contract for fixing the Sebih Special tester failures without turning the system into a legal-only patch pile. The goal is a general-purpose investigation engine whose routes, evidence objects, answer contracts, validators, and failure states all agree about the work being requested.

## North Star

The system must answer this before it spends or ships:

1. What typed task did the user request?
2. What evidence objects would make that task answerable?
3. Do we already have those objects at sufficient quality?
4. If not, what source-grounded work must run?
5. If the answer is negative or absent, what exact status applies?

Cheap paths are allowed only after this check. They are implementation routes, not semantic substitutes for the work.

## Primary Failure Classes

### Route Substitution

Observed harm:
- Contract-to-contract comparisons became run-delta reports.
- Document-grounded enumerations became `List Documents` or `List Actors` stubs.
- Source extraction questions became matter-state summaries.

Required fix:
- Route selection must be gated by a task ontology.
- `compare` means run/matter-state delta only.
- `query` means raw state-table lookup only.
- `read` means synthesis over already-answerable matter state only.
- Extraction, validation, absence checks, and document comparisons must go to `investigate` unless the exact typed evidence objects already exist.

Success condition:
- The route audit includes a task spec.
- The final workflow contract names the task, operation, required evidence objects, and answer shape.
- Regression tests fail if a document comparison returns `## What changed` or if signatory extraction returns actor rows without names/titles.

Failure harm:
- Users get fast, polished non-answers.
- Quality gates may pass because a malformed output has citations or gap language.
- Repeated tests produce inconsistent artifacts because the wrong subsystem is being exercised.

### Evidence Object Confusion

Observed harm:
- "Signatories" were treated as generic actors.
- "Defined terms" were treated as documents.
- "Every reference" was treated as document inventory.
- Numeric mentions were treated as payment reconciliation facts.

Required fix:
- Promote high-value extracted objects into typed evidence surfaces:
  - `defined_term`
  - `signature_block`
  - `section_ref`
  - `cross_reference`
  - `schedule_entry`
  - `redaction_marker`
  - `authority_validation`
  - `quant_fact`
  - `absence_status`
- Answer contracts must require the right evidence object, not just any citation.

Success condition:
- A task can declare required evidence objects.
- Synthesis sees the evidence contract.
- Validators inspect task-shaped output, not just generic citations.

Failure harm:
- The system appears grounded because it cites documents while omitting the requested objects.
- Matter-state caches become misleading because they contain related but insufficient facts.

### Missingness Collapse

Observed harm:
- False-premise probes, out-of-matter questions, and genuinely missing-source cases all became "we lack the operative text."
- "All documents ingested" was confused with "the answer exists in state."

Required fix:
- Use distinct absence statuses:
  - `not_searched`
  - `searched_not_found`
  - `found_unverified`
  - `found_verified`
  - `conflicting_evidence`
  - `out_of_matter`
  - `false_premise_likely`
  - `source_missing`
- Premise-check tasks must distinguish source-missing from searched-not-found and false-premise-likely.

Success condition:
- A Section 19.7 ESG-covenant probe can say the searched materials do not support the premise.
- An out-of-matter entity can be labeled as not part of the matter instead of asking for more documents.
- A genuinely unavailable source can still be labeled source-missing.

Failure harm:
- The system refuses correctly in a shallow sense but gives the wrong legal/product reason.
- Testers cannot tell whether the engine searched and found nothing, never searched, or lacks documents.

### Diagnostic Corruption

Observed harm:
- SO-6 appended financial analysis into unrelated legal answers.
- Repeated numeric mentions accumulated into fake payment totals.
- The final answer body stopped matching the user's requested task.

Required fix:
- Diagnostics belong in validator/UI side channels unless the task itself asks for them.
- Quant reconciliation must require a compatible task and subject model.
- Numeric conflicts must not group unrelated null-subject amounts into a matter-wide conflict.

Success condition:
- SO-6 diagnostics are recorded but do not mutate unrelated final answers.
- Damages/payment tasks can still surface quant warnings.
- Non-financial legal tasks stay task-shaped.

Failure harm:
- The system produces confident-looking but nonsensical financial blocks.
- Any document-heavy litigation matter can poison unrelated outputs with accumulated numeric facts.

### Test Harness Fragility

Observed harm:
- Tests passed but pytest hung during cache teardown.
- Tests using `tmp_path` hit inaccessible global temp directories.
- Interrupted runs left CPU-burning orphan Python processes.

Required fix:
- Disable pytest cache provider in repo config.
- Put pytest temp dirs under a repo-local or known writable path.
- Avoid broad integration tests during tight loops unless the harness is healthy.
- Use surgical tests and static checks by default.

Success condition:
- Governance, engine bridge, and API cascade test targets complete in seconds.
- No orphan Python processes remain after interrupted runs.

Failure harm:
- Engineering velocity collapses.
- Test results become untrustworthy because "hang" is confused with product failure.

## Execution Phases

### Phase 1: Guardrails Now

Already in progress:
- Add deterministic `TaskSpec` inference.
- Force document-grounded tasks away from cheap routes.
- Preserve `task_spec` in execution contracts.
- Move SO-6 from answer mutation to diagnostics.
- Disable pytest cache provider and use a safe basetemp.

Exit criteria:
- Governance tests pass quickly.
- Engine bridge tests pass quickly.
- API cascade tests pass quickly.
- Static checks pass.

### Phase 2: Workflow Contract Propagation

Build:
- Add task ontology details to the workflow quality section.
- Add task-specific workflow obligations.
- Add validator directives for:
  - task evidence contract
  - required evidence objects
  - absence status

Exit criteria:
- Synthesis prompt cannot miss that the user asked for a premise check, signatory table, defined-term inventory, or document comparison.
- Tests assert that premise-check tasks include absence vocabulary and required evidence objects.

### Phase 3: Typed Evidence Stores

Build first:
- Section index
- Defined terms
- Signature blocks
- Cross-references
- Schedule/exhibit entries

Then:
- Case/statute authority validation objects
- Redaction marker categories
- Quant subject ontology and reconciliation scopes

Exit criteria:
- `query` may answer only when the exact typed store is populated and the task contract allows it.
- Otherwise it escalates to source-grounded extraction.

### Phase 4: Absence and Premise Semantics

Build:
- `AbsenceStatus` model and store.
- Search coverage records: target, terms, documents searched, status, confidence.
- Synthesis rules for false premise / out of matter / source missing.

Exit criteria:
- False premise probes do not become missing-document memos.
- Out-of-matter questions do not imply future documents are expected.
- Missing-source cases remain possible and explicit.

### Phase 5: Task-Specific Validators

Build validators:
- `document_comparison_has_per_document_rows`
- `defined_term_inventory_has_terms_definitions_sections`
- `signatory_answer_has_name_title_entity`
- `reference_inventory_has_spans`
- `premise_check_has_absence_status`
- `authority_validation_used_authority_lookup`
- `quant_reconciliation_has_subject_scope`

Exit criteria:
- Generic `citation_floor` and `gap_disclosure` remain baseline checks, not the main quality signal.
- Wrong refusals and wrong cheap-path artifacts fail validation.

### Phase 6: Golden Tester Harness

Build:
- Convert Delek and Whitfield tester items into fixtures.
- Each item declares:
  - expected task type
  - forbidden routes
  - required evidence objects
  - known canonical facts
  - false-premise or out-of-matter status where applicable
  - forbidden output patterns

Exit criteria:
- Tester failures become repeatable regression checks.
- The system can improve without re-reading spreadsheet notes by hand.

### Phase 7: Long-Context Benchmark Loop

Build:
- A benchmark query pack that deliberately mixes:
  - prior Delek transactional queries already tested
  - prior Whitfield litigation queries already tested
  - Wilson-style synthesis prompts that require clean, concise user-facing judgment
  - quantitative prompts that require subject-scoped reconciliation
  - full-inventory prompts that stress complete enumeration
  - adversarial premise and out-of-matter prompts
  - cross-document comparisons over multiple long agreements
- Per-query metadata:
  - expected task type
  - expected answer shape
  - required evidence objects
  - forbidden route artifacts
  - forbidden output patterns
  - known canonical facts or canonical negative statuses
  - maximum acceptable runtime tier for the route
- A result ledger that records:
  - route selected
  - task spec
  - documents read
  - LLM calls
  - output validators
  - runtime
  - failure class
  - regression priority

Exit criteria:
- The benchmark pack can run in bounded slices rather than one all-night run.
- Any output that repeats a known tester failure is classified automatically.
- Long-context failures produce a next engineering action, not only a score.

Failure harm:
- We optimize only for narrow unit tests and miss the actual long-context behaviors.
- The system improves routing but still fails synthesis, quant reconciliation, or exhaustive extraction.
- Tester regressions reappear because no mixed benchmark exercises them together.

### Phase 8: Benchmark-Driven Repair Loop

Loop:
1. Run a bounded benchmark slice.
2. Classify each failure into route, retrieval, extraction, synthesis, validation, or persistence.
3. Fix the highest-harm failure with the smallest durable abstraction.
4. Add or update a regression fixture before rerunning.
5. Promote repeated ad hoc facts into typed evidence objects only when the benchmark proves they are recurring.

Success condition:
- Each benchmark failure either becomes a fixed regression or a tracked product limitation.
- The system gets better on both legal and non-legal task ontology dimensions.
- Fast routes remain fast, but only for semantically eligible tasks.

Failure condition:
- A fix only changes prompt wording and leaves route/evidence contracts unchanged.
- A validator rewards a refusal just because it has citations or gap language.
- Quant or synthesis diagnostics mutate the final answer instead of producing side-channel telemetry.

## Benchmark Query Families

### Transactional Long-Context

Purpose:
- Validate document comparison, defined-term inventories, signatory extraction, schedule/exhibit analysis, cross-reference inventories, and false-premise resistance across long agreements.

Representative prompts:
- Compare schedules across two or three agreements.
- Extract all Article 1 defined terms beginning with a specified letter.
- Inventory every defined term across all agreements.
- Identify every signatory by name, title, and entity.
- Find every reference to a named collateral agreement.
- Pull a section that likely does not exist and label the premise status.

Primary harm if failed:
- The product looks fluent but misses the actual contract object the user asked for.

### Litigation Long-Context

Purpose:
- Validate procedural history, deposition extraction, property identification, case-law validation, damages analysis, and contradiction surfacing over large messy matter folders.

Representative prompts:
- Reconstruct a timeline from pleadings, orders, and deposition excerpts.
- Identify the property at issue and surface address/legal-description conflicts.
- Validate named cases and reporter citations.
- Explain money at stake using subject-scoped amounts.
- Draft a deposition prep outline without contaminating it with unrelated diagnostics.

Primary harm if failed:
- The system confuses advocacy, record facts, court posture, and financial exposure.

### Wilson-Style Synthesis

Purpose:
- Test whether the engine can convert a large evidence state into a short, direct, decision-useful answer without losing caveats or inventing certainty.

Representative prompts:
- Give the single biggest practical risk.
- Draft a short GC status memo with no headings and a word cap.
- Explain the recommended next move in plain language.
- Summarize what matters most, what is unknown, and what should be checked next.

Primary harm if failed:
- The system may extract facts correctly but fail the actual user-facing synthesis task.

### Quantitative Analysis

Purpose:
- Verify that numeric facts are grouped by subject, metric, period, unit, and source role before any reconciliation.

Representative prompts:
- How much money is at stake?
- Reconcile contract price, paid amount, outstanding balance, improvements, fees, and damages.
- Identify numeric conflicts, but only within comparable scopes.
- Explain whether repeated mentions corroborate a figure or conflict with it.

Primary harm if failed:
- The system fabricates exposure totals by summing every number seen in the matter.

### Adversarial Absence

Purpose:
- Force the system to distinguish searched-not-found, false-premise-likely, out-of-matter, source-missing, and found-unverified.

Representative prompts:
- Ask for a non-existent section.
- Ask about an entity outside the matter.
- Ask for a cited authority that may not exist.
- Ask for a provision that exists in a different document but not the target document.

Primary harm if failed:
- The system refuses with the wrong reason and trains users to think missing sources are the problem.

## Benchmark Operating Rules

- Never run the entire long-context suite while diagnosing harness instability.
- Run slices by family and cap the number of expensive investigations per pass.
- Every slow test must report why it is slow: routing, retrieval, extraction, model calls, validation, or teardown.
- A benchmark timeout is a product signal only after pytest/process teardown has been ruled out.
- Use fixture-level expected route and task-spec checks before spending on full LLM runs.
- Promote a query to expensive end-to-end testing only after the cheap contract checks pass.

## Operating Rules

- Do not run broad tests while the harness is suspect.
- Always kill orphan Python processes after interrupted test runs.
- Prefer deterministic route guards over prompt-only routing fixes.
- Prefer typed stores and validators over prose heuristics.
- Keep legal-specific objects as domain overlays on general task/evidence/status machinery.
- Do not treat ingestion coverage as answerability.
- Do not treat a proof gap as a missing document unless source coverage supports that status.

## Immediate Next Actions

1. Wire benchmark packs into typed loaders and contract-check runners.
2. Promote typed evidence records into persistent stores in small slices.
3. Add extraction operators for defined terms, signature blocks, cross-references, sections, schedules, and quant facts.
4. Convert every high-harm tester failure into a regression with expected task, route, required evidence, and forbidden output patterns.
5. Run benchmark slices, classify failures, and fix the highest-harm recurrent failure before expanding scope.

## Detailed Continuation Plan

This section is the work queue after the first remediation slice. It is intentionally broader than legal contracts: legal is the proving ground, but the reusable kernel is task -> evidence object -> source coverage -> synthesis contract -> validator -> review status.

### Lane A: General Task Ontology

Goal:
- Replace prompt-only route interpretation with a durable task language that can describe legal, finance, coding, academic, and biomedical work.

Build:
- Task facets:
  - operation: `lookup`, `extract`, `compare`, `reconcile`, `validate`, `synthesize`, `draft`, `verify_absence`
  - object class: `document`, `section`, `term`, `actor`, `authority`, `metric`, `event`, `artifact`, `claim`, `relationship`
  - evidence requirement: `source_span`, `typed_record`, `authority_lookup`, `search_coverage`, `reconciliation_scope`
  - freshness policy: cached eligible, stale-check required, fresh extraction required
  - answer shape: table, matrix, memo, timeline, side-by-side comparison, negative-status answer, work product
- Domain overlays:
  - legal overlay: contracts, pleadings, orders, authorities, signatories, defined terms
  - finance overlay: filings, metrics, periods, GAAP/non-GAAP, reconciliation scopes
  - coding overlay: files, symbols, commits, tests, runtime errors
  - academic overlay: papers, methods, findings, effect sizes, citations
  - biomedical overlay: studies, outcomes, adverse events, regulatory artifacts

Success:
- The same task machinery can describe "list every defined term," "reconcile Datadog quarterly revenue," and "map every API route" without legal-specific branching at the kernel level.

Failure harm:
- We fix Delek/Whitfield while creating a second brittle legal-only classifier.

Next implementation:
- Extend `TaskSpec` into a richer schema without breaking current tests.
- Add a compatibility adapter from current deterministic `infer_task_spec()` to the richer schema.
- Keep deterministic guardrails before any LLM classifier.

### Lane B: Typed Evidence Persistence

Goal:
- Make `query` eligible only when the exact requested evidence object already exists, with source and verification state.

Build order:
1. `section_index`: document, section label, span, hierarchy, heading text.
2. `defined_term`: document, term, definition span, first-defined location, aliases.
3. `signature_block`: document, entity, signer, title/capacity, date, span.
4. `cross_reference`: source document/span, target label, normalized target, reference type.
5. `schedule_entry`: document, schedule label, title, status active/reserved/redacted, span.
6. `authority_validation`: citation/name, jurisdiction, status, source of validation, treatment.
7. `quant_fact`: subject, metric, value, unit/currency, period, role, source span.
8. `absence_status`: target, status, search coverage, terms searched, confidence.

Success:
- A signatory query can be answered from `signature_block` records or escalates.
- A defined-term inventory can be answered from `defined_term` records or escalates.
- A damages query cannot use unscoped numeric mentions as a reconciliation.

Failure harm:
- Cached state continues to look useful while lacking the actual objects needed to answer.

Next implementation:
- Add lightweight store APIs for the highest-value records before full UI integration.
- Keep dataclasses and validation tests as compatibility anchors during migration.

### Lane C: Extraction Operators

Goal:
- Turn source-grounded task specs into targeted extraction work, not generic deep reads.

Build:
- Section/schedule parser operator:
  - use document text spans and headings
  - record hierarchy and active/reserved status
  - preserve page/section provenance
- Defined-term extractor:
  - target specific documents/sections
  - supports prefix filters like "terms beginning with T"
  - emits term records and absence statuses for missing target docs/sections
- Signature extractor:
  - targets signature pages and execution blocks
  - separates entity, signer name, title, capacity, and date
  - rejects actor-only rows
- Reference inventory extractor:
  - finds exact target labels and known aliases
  - emits per-reference spans, not document stubs
- Authority validator:
  - requires external/authority lookup when task says validate
  - records verified/rejected/not-found status
- Quant reconciliation operator:
  - builds subject and metric scope before aggregation
  - groups repeated mentions as corroboration, not new payments

Success:
- R2/R3 failures stop being regenerated as "generic investigation memo" because the operator knows the required object shape before synthesis.

Failure harm:
- The engine still performs broad investigation and hopes synthesis reconstructs the right table from scattered facts.

### Lane D: Synthesis Quality

Goal:
- Prevent reasoning/output mismatch and wrong refusal framing.

Build:
- Evidence manifest passed to synthesis:
  - required objects requested
  - objects found
  - objects searched but not found
  - unresolved conflicts
  - source coverage
- Synthesis self-check before final output:
  - Does the output answer the declared task?
  - Does it use the required answer shape?
  - Does it contradict extracted reasoning trace?
  - Does it claim source missing when search status says searched-not-found?
  - Does it include unrelated diagnostics?
- Negative-answer template:
  - `Status: false_premise_likely`
  - `Search coverage: documents/sections/terms searched`
  - `What was found instead`
  - `What would change the status`

Success:
- If trace found T-defined terms, the answer cannot say the Lion agreement is missing.
- If a provision likely does not exist, the answer says that rather than asking for operative text.

Failure harm:
- The system can do the hard work internally and still ship the wrong answer.

### Lane E: Validators and Quality Gates

Goal:
- Replace generic "has a citation" and "mentions a gap" with task-shaped validation.

Validators:
- `document_comparison_has_per_document_rows`
- `defined_term_inventory_has_terms_definitions_sections`
- `signatory_answer_has_name_title_entity`
- `reference_inventory_has_spans`
- `premise_check_has_absence_status`
- `out_of_matter_check_has_membership_status`
- `authority_validation_has_lookup_status`
- `quant_reconciliation_has_scope`
- `synthesis_does_not_include_diagnostics`
- `format_constraints_preserved`
- `trace_output_alignment`

Success:
- A polished refusal with citations fails if the refusal reason is wrong.
- A direct lookup table fails if it lacks the requested typed fields.

Failure harm:
- The system continues passing the wrong outputs because they are formatted cleanly.

### Lane F: Benchmark and Evaluation Loop

Goal:
- Make improvements measurable on long-context behavior without creating hour-long local failures.

Build:
- Typed benchmark pack schema.
- Contract-check runner:
  - task spec match
  - route eligibility
  - required evidence object declaration
  - forbidden output patterns
- Stub-mode synthesis validator tests.
- Bounded end-to-end runner:
  - run only N expensive queries per slice
  - record route, runtime, docs read, LLM calls, validators, failure class
  - write JSONL ledger
- Failure triage dashboard:
  - route failure
  - retrieval failure
  - extraction failure
  - synthesis failure
  - validator false pass
  - runtime/harness failure

Success:
- We can rerun Delek/Whitfield-style queries cheaply for routing/contracts and selectively for full outputs.
- New failures produce a concrete engineering backlog item.

Failure harm:
- We keep discovering the same failures manually in spreadsheets.

### Lane G: Runtime and Test Discipline

Goal:
- Keep the engineering loop fast enough to support continuous improvement.

Rules:
- Contract checks first; expensive LLM/matter-folder runs only after cheap checks pass.
- Run benchmark slices, not full suites, during development.
- Treat pytest teardown hangs as harness failures until proven otherwise.
- Keep all temp/cache paths repo-local or ignored.
- Capture runtime breakdown in benchmark results.

Success:
- Daily work uses seconds-to-tens-of-seconds checks, with deliberate long runs only when needed.

Failure harm:
- Test friction hides product failures and slows every fix.

### Lane H: Product Behavior Targets

The system should get to these behaviors:

- Contract comparison: always returns side-by-side contract differences, never run deltas.
- Complete enumeration: always either completes an inventory or reports partial coverage with missing spans.
- Direct lookup: only uses state tables for actual state-table questions.
- Matter model reuse: used for follow-ups and summaries only when the task is cache eligible.
- False premise: says searched-not-found or false-premise-likely with coverage.
- Out of matter: says out-of-matter with membership search, not missing documents.
- Authority validation: validates or rejects authorities through the appropriate lookup path.
- Quant analysis: reconciles only comparable scoped numbers.
- Synthesis: concise answer shape and user constraints are preserved.
- Diagnostics: never contaminate final work product text.

### High-Harm Backlog

1. Persistent stores for `defined_term`, `signature_block`, `cross_reference`, `absence_status`, and scoped `quant_fact`.
2. Task-specific extraction operators for those stores.
3. Synthesis manifest that reconciles reasoning trace vs final answer.
4. Validator suite for every known tester failure class.
5. End-to-end benchmark runner with runtime ledger.
6. Route audit UI/API fields showing task spec and terminal family.
7. Freshness and invalidation policy for typed stores.
8. General-purpose ontology schema that legal, finance, coding, research, and biomedical overlays can all use.
