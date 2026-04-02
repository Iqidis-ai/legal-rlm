# Project Status

Last updated: 2026-04-02
Branch: SebihSpecial

---

## Current Phase

**Phase 5: Intelligence Substrate** — Building the persistent matter model and typed assertion
graph. This is Priority 0 work from IDEAL_PRODUCT_SPEC.md §36.1.

The 4 roundtable cycles produced a functional recursive pipeline. Now we are re-architecting
around a durable matter model substrate. This is not cosmetic — it requires fundamental
structural additions.

---

## What Is Done

| Component | Status | Notes |
|-----------|--------|-------|
| Document reading (PDF/DOCX/TXT) | Complete | reader.py, handles newlines correctly |
| Full-text search + ranking | Complete | search.py, legal synonym expansion |
| Document clustering (TF-IDF) | Complete | clustering.py |
| LRU + response caching | Complete | cache.py |
| Gemini model tiering (LITE/FLASH/PRO) | Complete | models.py |
| RLM engine (recursive pipeline) | Complete | engine.py — orient/investigate/verify/synthesize |
| Investigation state | Complete | state.py — 20+ data classes |
| Investigation templates | Complete | templates.py — 5 built-in |
| Output formatters | Complete | Markdown, HTML, JSON, Text |
| High-level API (Irys class) | Complete | api.py — async + sync |
| Service layer (FastAPI + S3) | Complete | service/ |
| Gradio UI | Complete | ui/app.py |
| Test suite | MISSING FROM REPO | test files described in ROUNDTABLE_PROGRESS.md were never committed. Only test_simple.py exists (requires GEMINI_API_KEY). |
| Documentation | Complete | README, API_DOCS, DEPLOY |

---

## What Is Missing (Priority Order)

### Priority 0 — Intelligence Substrate (MUST BUILD FIRST)

| Component | Status | Spec Reference |
|-----------|--------|----------------|
| Persistent matter model + canonical stores | NOT STARTED | §13, §31 |
| Matter registry | NOT STARTED | §13.1 |
| Repository inventory store (durable, hash-based) | NOT STARTED | §13.2 |
| Document card store (author, role, posture, operative status) | NOT STARTED | §13.3, §15 |
| Span store (exact source grounding) | NOT STARTED | §13.4 |
| Actor/contact store (aliases, roles, comm patterns) | NOT STARTED | §13.5, §16 |
| Typed assertion graph | NOT STARTED | §13.6, §17 |
| Evidence store (support/attack/corroboration) | NOT STARTED | §13.7 |
| Issue model / structured issue tree | NOT STARTED | §13.8, §19 |
| Assumption store | NOT STARTED | §13.9 |
| Gap store (structured missingness) | NOT STARTED | §13.10, §22 |
| Quant store (amounts, dates, formulas) | NOT STARTED | §13.11, §29 |
| Authority store | NOT STARTED | §13.12, §28 |
| Decision-context store | NOT STARTED | §13.13, §21 |
| Work-product store | NOT STARTED | §13.14 |
| Reasoning ledger (structured, user-facing) | NOT STARTED | §13.15, §24 |
| Belief revision / truth maintenance | NOT STARTED | §17, §8.5 |
| User steering + interruptibility | NOT STARTED | §24, §5.3 |
| Clarification engine | NOT STARTED | §23 |
| Source-role / agenda modeling | NOT STARTED | §15, §8.3 |
| Repository intelligence (families, versions, missing companions) | PARTIAL | §14 |

### Priority 1 — Higher-Order Reasoning (after substrate)

| Component | Status | Spec Reference |
|-----------|--------|----------------|
| Decision-context overlays | NOT STARTED | §21 |
| Legal research layer (authorities as objects) | NOT STARTED | §28 |
| Quantitative intelligence layer | NOT STARTED | §29 |
| Proof-aware reasoning | NOT STARTED | §18 |
| Adversarial / source-calibration reasoning | NOT STARTED | §8.2, §8.3 |
| Attention allocation | MINIMAL | §12.8 |
| Background maintenance loops | NOT STARTED | §26 |

### Priority 2 — Presentation (after structure)

| Component | Status | Spec Reference |
|-----------|--------|----------------|
| Timelines from structured state | PARTIAL (prose) | §30 |
| Issue-evidence matrices | NOT STARTED | §30 |
| Damages waterfalls | NOT STARTED | §30 |
| Communication maps | NOT STARTED | §30 |
| Visual analytic surfaces | NOT STARTED | §36.3 |

---

## Active Work

No tasks in progress at session start 2026-04-02. Swarm Build is being initialized.
Next step: Codex design gate for the persistent matter model architecture.

---

## Known Blockers

None currently. Clean state.

---

## Key Metrics (Current)

- Tests passing: 87 / 87
- Persistent matter model: 0% built
- Typed assertion graph: 0% built
- Issue model: 0% built
- Source-role modeling: 0% built
- User steerability: 0% built
