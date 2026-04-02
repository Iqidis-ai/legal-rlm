# Irys RLM — Project CLAUDE.md

This is the project-level governance document for Irys RLM. It extends the global CLAUDE.md
and defines the sacred outcomes, success criteria, and domain-specific Swarm Build rules for
this project. In all conflicts, project memory wins, then this file, then global CLAUDE.md.

---

## Project Mission

Build a legal intelligence system that continuously constructs, maintains, and improves a
living model of a legal matter from messy, adversarial, incomplete, evolving inputs.

The core product is NOT the answer text. The core product is the **matter model**.

Every text answer, chart, timeline, research packet, and memo is a downstream view over
an evolving matter model substrate. If that substrate is weak, everything built on top of
it is fragile.

Full product specification: `IDEAL_PRODUCT_SPEC.md` in the repo root.

---

## Sacred Outcomes

These are non-negotiable. Every architectural decision, every sprint, every refactor must
serve at least one of these. If it serves none, it should not happen.

### SO-1: Durable Matter Model

Every query reads from and writes to a persistent matter model. The system never rediscovers
stable structure from scratch. Cold-path ingestion builds durable structure. Hot-path queries
mostly read and selectively refresh. Recompute waste is treated as a product failure.

**Test:** Run the same query twice. The second run must be faster and produce richer results
by reading from persisted state, not by repeating all the same LLM calls.

### SO-2: Typed Assertion Graph with Truth Maintenance

Facts are stored as typed assertions with: speech-act classification (alleged/argued/admitted/
operative/performed/disputed/superseded/withdrawn/inferred/resolved), source role, source side,
temporal scope, support links, attack links, and revisable status. No flat fact strings.
The system can revise beliefs when new evidence or user corrections arrive. This is a
truth-maintenance system, not an append-only notes system.

**Test:** A user correction changes not just that assertion but also downstream conclusions
that depended on it. The dependency graph is traversed and updated.

### SO-3: User-Steerable Reasoning

Users can see the active reasoning state, interrupt execution mid-run, redirect work to a
different issue branch, mark a source as low-trust or high-trust, annotate documents, correct
factual assumptions, and supply strategic context. The reasoning ledger is structured,
user-facing, and actionable — not raw hidden chain-of-thought.

**Test:** A user can stop the investigation after Phase 2 iteration 3, tell the system that
a line of reasoning is strategically irrelevant, and have the system redirect without
restarting from scratch.

### SO-4: Issue-Driven Architecture

The issue model is the backbone of retrieval and synthesis. Every retrieval serves issue
coverage. The system maintains a structured issue tree (claims, defenses, elements, conditions,
burdens, authorities, damages components). Each issue knows what supports it, what attacks it,
what remains unknown, and what law governs it. Retrieval is targeted to move issue coverage
forward, not to match query tokens.

**Test:** Given a matter with 3 active claims, the system can report per-claim evidence
coverage, identify which claims are well-supported vs. proof-gap-exposed, and prioritize
retrieval toward the weakest-covered claim elements.

### SO-5: Source-Aware Intelligence

The system distinguishes source roles: advocacy, operative, procedural, authoritative,
informal, draft, post-hoc explanatory. It maintains a durable actor/contact store with
role resolution, alias handling, and communication patterns. It calibrates trust based on
source role and does not amplify advocacy material simply because it has clear argumentative
form.

**Test:** Given a complaint allegation and a signed contract clause asserting the same
proposition, the system treats them differently — the complaint as "alleged by plaintiff,"
the contract as "operative." This distinction must be preserved in the assertion graph and
visible in output.

### SO-6: Quantitative Intelligence as First-Class

Numbers are stored structurally — amounts, currencies, dates, date ranges, rates, balances,
invoice chains, payment events, formulas, damages assumptions, reconciliations, conflicts.
Numbers are not left trapped in prose. The system can perform damages modeling, timeline
analysis, payment reconciliation, and numeric conflict detection.

**Test:** Given documents with payment histories, the system can produce a reconciliation
showing what was invoiced, what was paid, what is disputed, and what the claimed exposure is,
grounded in source spans.

### SO-7: Missingness Is Modeled, Not Ignored

The system maintains a gap store: missing documents, missing metadata, missing issue
predicates, missing authorities, missing quantitative inputs, unresolved contradictions,
expected-but-absent attachments. The system surfaces these gaps proactively. Absence is
evidence that is analyzed, not a void that is quietly skipped.

**Test:** Given a matter where a signed amendment is referenced but not present, the system
explicitly flags the missing document as a gap, identifies which conclusions depend on it,
and asks a targeted clarification question with expected impact stated.

---

## Success Criteria

These are how we know we are winning. Not fluency. Not length. Not citation count.

| Criterion | Description | Target |
|-----------|-------------|--------|
| Reuse rate | % of matter intelligence reused from persistent store vs. recomputed | >70% on repeated queries |
| Issue coverage | % of active issue elements with at least one supporting/attacking assertion | >80% |
| Assertion structure | % of extracted facts stored as typed assertions with full metadata | 100% |
| Source calibration | Advocacy vs. operative distinction precision | >90% |
| Gap detection recall | % of known missing documents flagged by gap store | >85% |
| User steerability | User can interrupt and redirect mid-run | Yes/No — must be Yes |
| Belief revision | User correction propagates through dependency graph | Yes/No — must be Yes |
| Numeric extraction | Numbers stored structurally with source spans | >90% of numeric facts |

---

## Current State (as of 2026-04-02)

### What Exists

- Recursive search-read-synthesize pipeline (4 roundtable cycles complete)
- Document reading: PDF/DOCX/TXT via PyMuPDF and python-docx
- RLM engine: orient → investigate loop → citation verify → synthesize
- 3-tier Gemini model usage: LITE / FLASH / PRO
- Output formatters: Markdown, HTML, JSON, Text
- Service layer with FastAPI, S3 repository support
- Gradio UI
- 87 tests passing across 6 test files
- Branch: SebihSpecial

### What Is Missing (Priority Order)

The current system is a prototype pipeline. It is NOT a matter intelligence substrate.

**Priority 0 — Must build first:**
- Persistent matter model (canonical stores for matter, repository, document cards, actors, assertions, evidence, issues, assumptions, gaps, quant, authorities, work-product, reasoning ledger)
- Typed assertion graph with truth maintenance
- Issue model as investigation backbone
- Actor/contact store with role resolution
- Source-role and agenda modeling
- User-visible reasoning ledger
- User steering and interruptibility
- Clarification engine
- Gap store with structured missingness tracking
- Belief revision / incremental recompute

**Priority 1 — After substrate:**
- Decision-context overlays (judge, partner, client goals)
- Legal research layer (authorities as structured objects)
- Quantitative intelligence layer
- Proof-aware and adversarial reasoning
- Background maintenance loops
- Better attention allocation

**Priority 2 — After structure:**
- Visual work product (timelines, matrices, communication maps, damages waterfalls)
- Advanced presentation surfaces
- Richer workflow modes

---

## Architecture Principles

1. Every expensive computation either improves the durable matter model or reads from it.
   If it does neither, it is likely waste.

2. The five reasoning layers must never collapse into one stream:
   - What the record says (record model)
   - What likely happened (reality model)
   - What can be proved (proof model)
   - What the law says (legal model)
   - What the decision-maker cares about (decision-context model)

3. Assertions, evidence, and conclusions are distinct types. Never promote an assertion
   to a conclusion without an evidence layer between them.

4. Source role must be assigned before trust can be calibrated. Do not weight advocacy
   material as if it were operative content.

5. Missingness is information. An absent document is a finding, not a void.

6. User corrections route to the correct layer: factual corrections → canonical matter
   memory; strategic instructions → decision-context; presentation preferences → output
   mode. Never dump all user input into one bucket.

---

## Swarm Build Extensions (Domain-Specific)

### Swarm Build Reviewer Persona Extensions

**Correctness Engineer — also check:**
- Source-role attribution: is the speech act type (alleged/operative/argued) assigned correctly?
- Assertion dependency graph: does belief revision propagate correctly when an upstream
  assertion changes?
- Citation provenance: do all citations trace to exact source spans?
- Layer separation: are assertions, evidence, and conclusions stored in distinct structures?
- Gap detection: is missingness being surfaced or silently skipped?

**Scaling Expert — also evaluate:**
- Matter model size at scale: 10,000+ document repositories, 100,000+ assertion graphs
- Multi-matter isolation: does state from matter A bleed into matter B?
- Incremental update cost: when one new document arrives, how much recompute is triggered?
- Actor resolution at scale: alias matching and identity resolution across large corpora

**Architecture Theorist — derive from:**
- Truth-maintenance system theory (Doyle 1979, de Kleer 1986) — how should belief revision
  propagate through an assertion dependency graph?
- Legal epistemology — the assertion/evidence/conclusion layering mirrors how evidence law
  distinguishes admissibility, weight, and sufficiency
- Information-theoretic retrieval — issue-coverage-driven retrieval as entropy reduction over
  the issue model

**Add Persona 9: Legal Intelligence Auditor**
A senior litigation attorney with 20 years of experience reviewing the system's outputs for
practical legal soundness:
- Would a partner trust this analysis? Why or why not?
- Are advocacy sources being over-amplified? (The most dangerous failure mode)
- Are proof gaps visible in the output, or hidden under confident-sounding prose?
- Are assumptions surfaced where they carry the analysis?
- Is the distinction between "what the record says" and "what likely happened" maintained?
- Are missing documents flagged where they matter?
- Would this analysis survive adversarial scrutiny?

Trigger: Any time output will be used for legal work product review.

### Experiment Logging Extensions

Legal intelligence experiments should log:
- `matter_type`: litigation / transactional / regulatory / diligence
- `corpus_size`: number of documents, approximate pages
- `query_type`: factual / procedural / analytical / comparative / evaluative
- `issue_coverage_before`: % issue elements covered before run
- `issue_coverage_after`: % issue elements covered after run
- `assertion_count`: total assertions in graph after run
- `gap_count`: total gaps identified
- `reuse_rate`: % of intelligence reused from persistent store
- `source_role_distribution`: breakdown of advocacy vs. operative vs. authoritative sources

---

## Anti-Patterns for This Project

Specific to Irys RLM — these are the failure modes the spec explicitly calls out:

1. Adding more prompt complexity instead of improving persistent state
2. Adding more recursive search instead of building a real issue model
3. Storing extracted "facts" as free text when a typed assertion is needed
4. Treating user chat history as a substitute for durable matter memory
5. Optimizing for memo fluency before fixing the underlying intelligence substrate
6. Merging strategy, truth, and preference into one store
7. Over-amplifying advocate-authored material (pleadings, briefs, demand letters)
8. Hiding uncertainty to make the answer look cleaner
9. Ignoring missingness because the system can still produce prose
10. Spending expensive tokens to rediscover stable structure already in the matter model
