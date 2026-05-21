# Fact Completeness — Design Spec

**Date:** 2026-05-21  
**Status:** Approved

---

## Problem

Extracted fact strings are stripped claims. The LLM distils a clause into one sentence, dropping the qualifications, conditions, carve-outs, and temporal constraints that immediately surround it in the source text. Synthesis answers questions using these stripped claims and misses governing conditions.

Example:
- Stored: `"Indemnification cap is $5M"`
- Actual clause: `"Indemnification cap is $5M, subject to the gross negligence carve-out in §8.3(b), and resets annually on the contract anniversary."`

The carve-out is dropped. Synthesis answers the damages question without knowing §8.3(b) exists.

---

## Goal

Every fact string must be a self-contained passage that includes the sentence immediately before and the sentence immediately after the core claim as it appears in the source text — preserving qualifications and limitations within the fact itself.

---

## Scope

**One instruction block in one file: `src/irys/rlm/prompts.py`, `P_EXTRACT_FACTS` only.**

### Why only `P_EXTRACT_FACTS`

Three prompts emit `"facts"` arrays:

| Prompt | Source | Include? |
|---|---|---|
| `P_EXTRACT_FACTS` | Full document read | **Yes** |
| `P_ANALYZE_RESULTS` | Search snippet | **No** — model cannot see surrounding lines; adding the instruction would cause hallucination |
| `P_ASSESS_AND_PLAN` / `P_ASSESS_SMALL_REPO` | Cached fact listing | **No** — not extracting new facts, just listing existing ones |

---

## The Instruction

Insert after the `SCOPE DISCIPLINE` block, before the `CRITICAL:` line:

```
FACT COMPLETENESS: Each entry in "facts" must be a self-contained passage, not a
stripped claim. Include the sentence immediately before and the sentence immediately
after the core claim as it appears in the source text — these preserve qualifications,
conditions, carve-outs, and limitations that govern the claim. If a surrounding
sentence adds no qualifying context (e.g. it is a heading or an unrelated clause),
omit it. Minimum: the core claim alone. Maximum: 3 sentences total.
Do NOT apply this to "quotes" — quotes remain verbatim text only.
```

---

## No Schema Changes Required

- `StoredFact.fact` is a plain `str` — longer strings are stored as-is
- FTS5 indexes the full `fact` string with Porter stemmer — longer passages improve BM25 recall
- `EvidencePacker._density()` normalises by `len(fact.fact)` — longer passages are naturally down-weighted per token, correct behaviour
- Content hashes for existing short-form facts are unaffected — they are different strings and coexist in the DB
- No migration needed — old facts stay as-is, new extractions produce richer strings

---

## Test

Add one test to `tests/test_fact_store.py` verifying the instruction is present in `P_EXTRACT_FACTS` and absent from `P_ANALYZE_RESULTS`.
