# Irys RLM Benchmark Strategy

## Tiered Priority System

### Tier 1 (Priority 1): Harvey LAB — Full 989-Task Suite
All 24 practice areas, expert-scored with detailed criteria.

| Practice Area | Tasks | Notes |
|---|---|---|
| corporate-ma | 110 | Current focus (CoC extraction); expand to all |
| intellectual-property | 145 | Largest area; contract amendment, licensing, patent |
| corporate-governance | 93 | Board memos, proxy analysis, compliance |
| trusts-estates-private-client | 71 | Trust administration, estate planning |
| litigation-dispute-resolution | 52 | Motion drafting, discovery, case analysis |
| data-privacy-cybersecurity | 42 | GDPR, CCPA, incident response |
| healthcare-life-sciences | 41 | FDA, HIPAA, clinical trial agreements |
| emerging-companies-venture-capital | 41 | Term sheets, SAFEs, board formation |
| funds-asset-management | 42 | Fund formation, LP agreements |
| environmental-esg | 40 | ESG reporting, environmental compliance |
| international-trade-sanctions | 39 | OFAC, export controls, compliance |
| employment-labor | 39 | Employment agreements, NLRA, FLSA |
| banking-finance | 37 | Credit agreements, regulatory compliance |
| tax | 34 | Tax opinions, structuring, controversy |
| arbitration-international-dispute-resolution | 33 | ICC rules, investor-state disputes |
| antitrust-competition | 31 | HSR, merger review, compliance |
| bankruptcy-restructuring | 32 | Chapter 11, creditor rights |
| capital-markets | 31 | Securities offerings, disclosure |
| energy-natural-resources | 31 | E&P, pipeline, renewable energy |
| insurance | 31 | Coverage, claims, regulatory |
| structured-finance-securitization | 29 | CLOs, RMBS, waterfall analysis |
| immigration | 25 | H-1B, EB-1, I-140 processing |
| white-collar-defense-investigations | 21 | Internal investigations, DOJ cooperation |
| real-estate | 42 | PSA review, lease analysis, zoning |

**Infrastructure:** `tools/lab_benchmark.py` supports `--practice-area` and `--all`.
**Blocker:** Gemini API rate limits (503 errors during demand spikes).
**Strategy:** Run practice areas sequentially; retry with backoff.

### Tier 2 (Priority 2): External Long-Context Benchmarks

Ranked by relevance to Irys's 5-domain testing needs:

1. **AA-LCR** (Artificial Analysis) — Multi-domain, ~100K tokens/Q, legal+finance+academic
2. **DocFinQA** (ACL 2024) — Finance, 123K avg words, 7,437 expert QA pairs on SEC filings
3. **LongBench v2** (ACL 2025) — Multi-domain, 503 expert Qs, 8K-2M words
4. **HELMET** (ICLR 2025) — Citation generation directly maps to source-aware synthesis
5. **LegalBench** (NeurIPS 2023) — 162 legal reasoning tasks (shorter context but broad)
6. **ContractEval** (2025) — 41 contract risk categories, builds on CUAD
7. **ScholarQABench** — Multi-paper academic QA, PhD-level ground truth
8. **BioASQ 2025** (CLEF) — 13 years of biomedical QA annotation
9. **FACTS Grounding** (Google) — Tests source grounding, maps to SO-5
10. **InfiniteBench** (ACL 2024) — Stress-test at 100K+ tokens

**Domain coverage matrix:**

| Domain | Primary | Supplementary |
|---|---|---|
| Legal | LegalBench + ContractEval | CUAD, Harvey LAB |
| Finance | DocFinQA | FinBen/PIXIU |
| Coding | LongBench v2 (code subset) | SWE-ContextBench |
| Research | ScholarQABench | L-Eval |
| Biomedical | BioASQ 2025 | PubMedQA |
| Cross-domain | AA-LCR + HELMET | InfiniteBench, RULER |

### Tier 3: Internal Corpus Evals (Already Built)

- `benchmarks/legal_corpus_eval.jsonl` — 30 legal multi-hop queries
- `benchmarks/finance_corpus_eval.jsonl` — 30 finance queries (Datadog SEC filings)
- `benchmarks/aiml_corpus_eval.jsonl` — 30 AI/ML research queries
- `benchmarks/coding_corpus_eval.jsonl` — 30 coding/Kubernetes queries
- `benchmarks/legal_tester_regression_eval.jsonl` — 13 regression queries
- `tests/eval/benchmark_packs/long_context_mixed_v1.json` — 16 mixed queries (contract checks)

## Execution Order

1. Fix Gemini API reliability (backoff/retry or Vertex AI)
2. Run Harvey LAB CoC extraction (score baseline)
3. Run Harvey LAB across all 24 practice areas (batch by area)
4. Integrate DocFinQA for financial long-doc testing
5. Integrate LegalBench for legal reasoning breadth
6. Integrate LongBench v2 for cross-domain coverage
7. Add ScholarQABench + BioASQ for remaining domains
