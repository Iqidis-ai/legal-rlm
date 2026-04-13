# Image & OCR Extraction — Implementation Plan

> **Status: Implemented.** This document reflects the actual implementation.

## Overview

Support for multimodal files — image file types (PNG, JPEG) and PDF/DOCX files that may
contain scanned text or images — is added to the text extraction pipeline. For image files,
and for PDF/DOCX files where multimodal processing is detected as required, the Mistral OCR
API extracts text. OCR calls are wrapped in a configurable timeout so a slow or hung Mistral
call never blocks the main investigation flow. OCR calls are logged following the existing
telemetry patterns (`StepOperation(type="ocr")` → `InvestigationOperation` DB row).

---

## Files Changed

### 0. `src/irys/core/multimodal_detection.py` *(new file)*

Heuristic scoring module that decides whether a PDF or DOCX needs OCR instead of relying
on text already extracted by fitz/python-docx.

**Exported:**
- `MultimodalDetectionConfig` — tunable thresholds (all match the original TS defaults):
  - `extremely_high_confidence_text_threshold`: 125 chars/page → +0.55 confidence
  - `high_confidence_text_threshold`: 200 chars/page → +0.40 confidence
  - `medium_confidence_text_threshold`: 400 chars/page → +0.20 confidence
  - `min_text_extraction_rate`: 0.7 (ratio of pages with >50 chars to total pages) → +0.30
  - `multimodal_threshold`: 0.45 (minimum confidence to flag as needing OCR)
- `MultimodalDetectionResult` — `requires_multimodal`, `confidence`, `reasons`, `metrics`
- `detect_multimodal_content(text, page_count, rendered_pages, pdf_info, config)` — pure
  function, no I/O. `rendered_pages` = count of pages where extracted text > 50 chars
  (computed from fitz page-by-page extraction). `pdf_info` = fitz `doc.metadata` dict.

Internal helpers: `_analyze_text_metrics`, `_detect_suspicious_patterns`, `_analyze_metadata`
(checks Creator/Producer for scanner keywords), `_calculate_confidence`, `_generate_reasons`.
All ported faithfully from `text-extract-multimodal-detection.ts` (now deleted).

---

### 1. `src/irys/service/s3_repository.py`

- Added `image/png`, `image/jpeg`, `image/jpg` to `CONTENT_TYPE_TO_EXT`.
- Added `\x89PNG` and `\xff\xd8\xff` (JPEG) to `MAGIC_BYTES`.
- Updated extension allow-lists in `_download_url`, `_download_http_url`, and the default
  extensions list in `list_documents` to include `.png`, `.jpg`, `.jpeg`.

---

### 2. `src/irys/core/reader.py`

**`OcrCallMetadata` dataclass** (new):
```python
@dataclass
class OcrCallMetadata:
    latency_ms: int
    page_count: int
    timed_out: bool
    file_type: str   # "png" | "jpg" | "jpeg" | "pdf" | "docx"
    file_name: str
```

**`SUPPORTED_EXTENSIONS`** — extended with `.png`, `.jpg`, `.jpeg`.

**`_detect_type_from_magic`** — extended with PNG (`\x89PNG`) and JPEG (`\xff\xd8\xff`) detection.

**`read()` (sync)** — raises `ValueError` for image files with a clear message directing
callers to `read_async()`. All other existing formats unchanged.

**`read_async(path, ocr_timeout=60)` (new async method)** — the main entry point for all
files in the engine. Returns `(DocumentContent, OcrCallMetadata | None)`:
- **Images** → `_read_image_via_ocr()` → `_call_mistral_ocr()` with `image_url` type
- **PDF** → `_read_pdf_with_meta()` for normal extraction + `rendered_pages` count + fitz
  metadata → `detect_multimodal_content()` → if `requires_multimodal` and
  `_pdf_ocr_enabled()`: `_call_mistral_ocr()` with `document_url` type. If OCR returns
  0 chars, falls back to the original fitz extraction rather than returning empty.
- **DOCX** → same pattern: `_read_docx()` first, then multimodal detection with `page_count=1`,
  then OCR if needed, with same fallback logic.
- **TXT/MHT** → delegates to sync `read()`, returns `(doc, None)`.

**`_call_mistral_ocr(data, mime, doc_type, path, file_type, ocr_timeout)` (new shared helper)**:
- `doc_type` is `"image_url"` for images, `"document_url"` for PDF/DOCX.
- Payload key matches `doc_type` value (Mistral API shape).
- Blocking HTTP call (`httpx`) wrapped in `asyncio.to_thread` + `asyncio.wait_for`.
- Never raises. All failure paths return `(_empty_doc(), OcrCallMetadata(...))`:
  - No `MISTRAL_API_KEY` → returns immediately, `latency_ms=0`
  - `asyncio.TimeoutError` → `timed_out=True`
  - Any other exception → `timed_out=False`
- On success: parses `pages[].markdown` (fallback to `pages[].text`).

**`_read_pdf_with_meta(path)`** (new private helper) — runs normal fitz extraction and
additionally returns `rendered_pages` (pages with >50 chars) and `doc.metadata` dict for
multimodal detection scoring.

**`_empty_doc(path, file_type)`** (new static helper) — returns a zero-page `DocumentContent`
used by all OCR failure paths.

**`_pdf_ocr_enabled()`** (module-level function) — reads `IRYS_PDF_OCR_ENABLED` env var.
When `false/0/no/off`: skips multimodal detection and OCR for PDF and DOCX. Images always
go through OCR regardless (no alternative extraction path exists for them).

---

### 3. `src/irys/core/repository.py`

- `OcrCallMetadata` added to imports from `reader`.
- `SUPPORTED_EXTENSIONS` extended with `.png`, `.jpg`, `.jpeg`.
- `_ASYNC_ONLY_EXTENSIONS = {".png", ".jpg", ".jpeg"}` class var added — used to guard
  sync-only code paths.
- `_compute_metadata()` and `get_all_content()` skip image files (they require `read_async`
  and have no sync fallback).
- **`read_async(path, ocr_timeout=60.0)` (new method)** — async wrapper around
  `self.reader.read_async()`. Checks `self._doc_cache` first (returns `(cached, None)` on
  hit — no OCR metadata since no Mistral call was made). Stores result in `_doc_cache` on
  miss (including empty docs from failed OCR, preventing re-calls within the same
  `investigate()` session).

---

### 4. `src/irys/core/telemetry.py`

`StepOperation` extended with four new fields (all default to empty/zero):
```python
file_name: str = ""    # file that was OCR'd
file_type: str = ""    # "png" | "jpg" | "jpeg" | "pdf" | "docx"
page_count: int = 0    # pages returned by Mistral
timed_out: bool = False
```

`"ocr"` branch added to `to_dict()` and `details_dict()` emitting:
`service`, `file_name`, `file_type`, `page_count`, `timed_out`, `cost_usd`.

---

### 5. `src/irys/rlm/engine.py`

`_read_document()` changed from `repo.read()` to `await repo.read_async()`.

If `ocr_meta is not None` after the read, a dedicated `"document_read_ocr"` telemetry step
is created, a single `StepOperation(type="ocr", ...)` is added to it, and the step is ended
immediately. This is self-contained rather than attaching to an existing step — it mirrors
the pattern used for `ext_search` steps.

---

### 6. `src/irys/__init__.py`

`OcrCallMetadata` added to public exports.

---

## Environment Variables

```
MISTRAL_API_KEY=xxx          # Required for OCR. Add to .env, never hardcode.
IRYS_PDF_OCR_ENABLED=false   # Optional. Disables multimodal detection + OCR for PDF/DOCX.
                             # Images always go through OCR regardless of this flag.
IRYS_OCR_CACHE_ENTRIES=200   # Optional. Max entries in the process-level OCR result cache.
                             # Each entry holds the full DocumentContent for one file.
                             # LRU eviction applies once the limit is reached.
                             # Default: 200. Must be a positive integer.
```

---

## Key Behavioural Constraints

| Constraint | Detail |
|---|---|
| Timeout | OCR call never blocks indefinitely. Default 60s via `asyncio.wait_for`. Fail-safe, not fail-loud. |
| Image files | Always go straight to OCR — no fitz, no multimodal detection. |
| PDF/DOCX | Extract text normally first → multimodal detection → OCR only if `requires_multimodal=True` and `IRYS_PDF_OCR_ENABLED` is not false. |
| OCR fallback | If OCR returns 0 chars for a PDF/DOCX, the original fitz/docx extraction is used instead of returning empty content. |
| Shared OCR helper | Both image and PDF/DOCX paths go through the same `_call_mistral_ocr()` in `reader.py`. |
| Sync `read()` | Raises `ValueError` for image files — callers must use `read_async()`. All other formats unchanged. |
| Telemetry | Every Mistral OCR call → one `StepOperation(type="ocr")` on a dedicated `"document_read_ocr"` step. |
| Within-call deduplication | `MatterRepository._doc_cache` caches `read_async()` results (including empty docs from failed OCR). OCR is not called twice for the same file within one `investigate()` call. `InvestigationCache.mark_extracted` provides a second layer of protection at the engine level. |
| Cross-call deduplication | `_GLOBAL_DOC_CACHE` — a module-level `LRUCache[DocumentContent]` in `repository.py`. Keyed by the resolved file path / S3 URL (content-addressable in practice). Survives across `investigate()` calls within the same process lifetime. Bounded by `IRYS_OCR_CACHE_ENTRIES` (default 200 entries). All cache reads and writes are wrapped in `try/except` so a failure is only logged and never blocks the main flow. Each uvicorn worker process maintains its own copy — no cross-process locking needed. |
| Cost | `cost_usd` field left as `0.0` for now. |
