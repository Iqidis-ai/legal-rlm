# Fact Completeness Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `FACT COMPLETENESS` instruction to `P_EXTRACT_FACTS` so the extraction LLM returns 1-3 sentence passages preserving qualifications and carve-outs, instead of stripped single-clause claims.

**Architecture:** Single prompt instruction block inserted into `P_EXTRACT_FACTS` in `prompts.py`. No schema changes, no code changes outside the prompt and its test. The fact string field on `StoredFact` already accepts arbitrary-length strings; FTS5, EvidencePacker, and `_fact_line` all handle longer strings without modification.

**Tech Stack:** Python, pytest

---

### Task 1: Write failing tests for the new instruction

**Files:**
- Modify: `tests/test_fact_store.py`

- [ ] **Step 1: Add the failing tests**

Open `tests/test_fact_store.py` and append this class at the bottom of the file:

```python
class TestFactCompletenessInstruction:
    """P_EXTRACT_FACTS must contain the FACT COMPLETENESS instruction block.
    P_ANALYZE_RESULTS must NOT — it works from search snippets and cannot see surrounding lines.
    """

    def test_fact_completeness_present_in_extract_facts(self):
        from irys.rlm.prompts import P_EXTRACT_FACTS
        assert "FACT COMPLETENESS" in P_EXTRACT_FACTS

    def test_surrounding_sentence_instruction_present(self):
        from irys.rlm.prompts import P_EXTRACT_FACTS
        assert "sentence immediately before" in P_EXTRACT_FACTS
        assert "sentence immediately after" in P_EXTRACT_FACTS

    def test_three_sentence_cap_present(self):
        from irys.rlm.prompts import P_EXTRACT_FACTS
        assert "3 sentences" in P_EXTRACT_FACTS

    def test_quotes_exemption_present(self):
        from irys.rlm.prompts import P_EXTRACT_FACTS
        assert "quotes" in P_EXTRACT_FACTS

    def test_fact_completeness_absent_from_analyze_results(self):
        """Search snippet path must NOT get this instruction — model cannot see surrounding lines."""
        from irys.rlm.prompts import P_ANALYZE_RESULTS
        assert "FACT COMPLETENESS" not in P_ANALYZE_RESULTS
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_fact_store.py::TestFactCompletenessInstruction -v
```

Expected output — all 5 tests fail with `AssertionError`:
```
FAILED tests/test_fact_store.py::TestFactCompletenessInstruction::test_fact_completeness_present_in_extract_facts
FAILED tests/test_fact_store.py::TestFactCompletenessInstruction::test_surrounding_sentence_instruction_present
FAILED tests/test_fact_store.py::TestFactCompletenessInstruction::test_three_sentence_cap_present
FAILED tests/test_fact_store.py::TestFactCompletenessInstruction::test_quotes_exemption_present
PASSED tests/test_fact_store.py::TestFactCompletenessInstruction::test_fact_completeness_absent_from_analyze_results
```

(The last test passes already because the instruction doesn't exist yet.)

---

### Task 2: Add the FACT COMPLETENESS instruction to P_EXTRACT_FACTS

**Files:**
- Modify: `src/irys/rlm/prompts.py`

The insertion point is after the `SCOPE DISCIPLINE` block and before the `CRITICAL:` line.

Current text at that boundary (around line 406-408):
```
- For quote page numbers, use visible page markers when available; otherwise use null.

CRITICAL: Legal precision is paramount. Extract ALL relevant facts with EXACT values.
```

- [ ] **Step 1: Insert the instruction block**

Replace that boundary with:

```
- For quote page numbers, use visible page markers when available; otherwise use null.

FACT COMPLETENESS: Each entry in "facts" must be a self-contained passage, not a
stripped claim. Include the sentence immediately before and the sentence immediately
after the core claim as it appears in the source text — these preserve qualifications,
conditions, carve-outs, and limitations that govern the claim. If a surrounding
sentence adds no qualifying context (e.g. it is a heading or an unrelated clause),
omit it. Minimum: the core claim alone. Maximum: 3 sentences total.
Do NOT apply this to "quotes" — quotes remain verbatim text only.

CRITICAL: Legal precision is paramount. Extract ALL relevant facts with EXACT values.
```

- [ ] **Step 2: Run the tests to verify they pass**

```
pytest tests/test_fact_store.py::TestFactCompletenessInstruction -v
```

Expected output — all 5 pass:
```
PASSED tests/test_fact_store.py::TestFactCompletenessInstruction::test_fact_completeness_present_in_extract_facts
PASSED tests/test_fact_store.py::TestFactCompletenessInstruction::test_surrounding_sentence_instruction_present
PASSED tests/test_fact_store.py::TestFactCompletenessInstruction::test_three_sentence_cap_present
PASSED tests/test_fact_store.py::TestFactCompletenessInstruction::test_quotes_exemption_present
PASSED tests/test_fact_store.py::TestFactCompletenessInstruction::test_fact_completeness_absent_from_analyze_results
```

- [ ] **Step 3: Run the full test suite to check for regressions**

```
pytest tests/test_fact_store.py tests/test_fact_store_v2.py -v
```

Expected: all tests pass. No failures.

- [ ] **Step 4: Commit**

```bash
git add src/irys/rlm/prompts.py tests/test_fact_store.py
git commit -m "feat(extraction): add FACT COMPLETENESS instruction to P_EXTRACT_FACTS

Instructs extraction LLM to include the sentence immediately before and
after each core claim, preserving qualifications, carve-outs, and
conditions that govern the claim. Capped at 3 sentences per fact entry.

P_ANALYZE_RESULTS is intentionally excluded — it operates on search
snippets where surrounding lines are not visible to the model."
```
