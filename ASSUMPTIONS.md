# Assumption Log

Active assumptions are tracked here. Each assumption records what depends on it, what would
invalidate it, and its current status. Invalidated assumptions must be immediately updated
and downstream work reassessed.

---

## A-002: Google Gemini API is Available and Stable

**Status:** Active
**Source:** D-001
**Assumption:** GEMINI_API_KEY is valid, Gemini API is accessible, model names are stable.
**Depends on:** All LLM calls in engine.py, models.py
**Would invalidate if:** Google changes model names, deprecates gemini-2.5-* family, or
significantly changes pricing
**Downstream impact if wrong:** Model tiering must be reconfigured. May need fallback provider.

---

## A-003: Documents Are Digitally Readable (No OCR Required)

**Status:** Active
**Source:** README limitations section
**Assumption:** PDFs contain machine-readable text (not scanned images). DOCX and TXT are
standard format.
**Depends on:** reader.py, all document processing
**Would invalidate if:** User provides scanned PDFs or legacy .doc files
**Downstream impact if wrong:** OCR layer (Tesseract or cloud OCR) must be added before
text extraction.

---

## A-004: Legal Documents Are in English

**Status:** Active
**Source:** No multi-language requirements stated
**Assumption:** All documents, queries, and outputs are in English. Legal synonyms and
expansion are English-only.
**Depends on:** search.py (legal synonym expansion), all prompts
**Would invalidate if:** International matters require non-English document analysis
**Downstream impact if wrong:** Synonym expansion, prompts, and output formatting all require
localization.

---

## A-005: The Assertion Graph Fits In Memory During a Query Run

**Status:** Provisional — needs validation at scale
**Source:** Current architecture (state.py holds all facts in memory during a run)
**Assumption:** For typical legal matters (hundreds to low thousands of documents), the
full assertion graph for one matter fits comfortably in memory during a single query run.
**Depends on:** state.py design, engine.py investigation loop
**Would invalidate if:** A matter has 10,000+ documents with dense cross-references
**Downstream impact if wrong:** Must implement streaming or paged graph traversal. Matter
model read/write must be more granular.

---

## A-006: Source-Role Can Be Inferred From Document Type and Metadata

**Status:** Provisional — untested
**Source:** IDEAL_PRODUCT_SPEC.md §15
**Assumption:** Document type (complaint, contract, email, court order, deposition) combined
with available metadata (sender, recipient, date) is sufficient signal to infer source role
(advocacy, operative, authoritative, informal) with >85% accuracy using an LLM.
**Depends on:** Document card population, source-role modeling implementation
**Would invalidate if:** Source-role inference accuracy on real legal corpora falls below 85%
**Downstream impact if wrong:** Source-role modeling requires human annotation or more
document-type-specific heuristics.

---

## A-007: SQLite Is Sufficient for Matter Model at Target Scale

**Status:** Proposed — pending validation
**Source:** D-004
**Assumption:** SQLite can handle a matter with: 10,000 documents, 100,000 assertions,
500 actors, 200 issues, without performance degradation on typical query patterns.
**Depends on:** D-004 (SQLite choice), persistent store implementation
**Would invalidate if:** Query latency on assertion graph traversal exceeds 1 second on
a matter with >50,000 assertions
**Downstream impact if wrong:** Must migrate to PostgreSQL or adopt a graph database.
