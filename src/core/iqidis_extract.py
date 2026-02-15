"""
Extract Knowledge Graph from Iqidis matter documents.

Supports two modes:
A. Frontend payload mode (extract_from_frontend_payload):
   - Next.js server sends documents list with pre-signed URLs and optional extracted_text
   - Fast path: use extracted_text directly (no download/parse)
   - Slow path: download from signed_url, parse
   - Dedup via content_hash

B. Legacy mode (extract_from_iqidis_matter):
   - Backend queries Iqidis DB for documents, downloads from S3
"""
import os
import time
import queue
import hashlib
import threading
import urllib.request
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import MAX_DOCUMENT_SIZE_BYTES
from .iqidis_data import get_matter_documents, get_matter_by_id
from .s3_fetcher import download_from_s3
from .parsing.document_parser import DocumentParser, ParsedDocument
from .knowledge_graph import KnowledgeGraph

# Auto-detect tier for download parallelism
_IS_PAID_TIER = os.getenv(
    "GEMINI_PAID_TIER", "true").lower() in ("true", "1", "yes")
_DEFAULT_DOWNLOAD_WORKERS = 6 if _IS_PAID_TIER else 3


def _download_and_parse_document(
    doc_info: Dict[str, Any],
    parser: DocumentParser,
    verbose: bool
) -> Tuple[Optional[ParsedDocument], str, Optional[str]]:
    """
    Download and parse a single document.
    Returns (parsed_document, original_name, error_message).
    """
    storage_key = doc_info.get("storage_key")
    original_name = doc_info.get("original_name") or "document"
    mime = doc_info.get("mime") or "application/pdf"
    size_byte = doc_info.get("size_byte")

    if verbose:
        print(f"[{original_name}] Downloading from S3...", flush=True)

    if not storage_key:
        return None, original_name, f"No storage_key for {original_name}"

    if size_byte is not None and size_byte > MAX_DOCUMENT_SIZE_BYTES:
        return None, original_name, f"Skipped {original_name}: exceeds 50MB limit ({size_byte / (1024*1024):.1f}MB)"

    data = download_from_s3(storage_key)
    if not data:
        return None, original_name, f"Failed to download {original_name}"

    if len(data) > MAX_DOCUMENT_SIZE_BYTES:
        return None, original_name, f"Skipped {original_name}: downloaded size exceeds 50MB limit"

    if verbose:
        print(f"[{original_name}] Parsing...", flush=True)

    parsed = parser.parse_from_bytes(data, original_name, mime)
    if not parsed or not parsed.text.strip():
        return None, original_name, f"Could not extract text from {original_name}"

    return parsed, original_name, None


# ---------------------------------------------------------------------------
# Frontend payload helpers
# ---------------------------------------------------------------------------

def _download_from_signed_url(url: str, timeout: int = 120) -> Optional[bytes]:
    """Download document bytes from a pre-signed S3 URL via HTTP GET."""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as e:
        print(f"Download failed from signed URL: {e}", flush=True)
        return None


def _prepare_document_from_payload(
    doc: Dict[str, Any],
    parser: DocumentParser,
    verbose: bool,
) -> Tuple[Optional[ParsedDocument], str, str, Optional[str], Optional[List]]:
    """
    Prepare a ParsedDocument from a frontend payload document entry.

    3-tier processing (fastest → slowest):
      Tier 1: chunks present → skip download/parse/chunk/embed
      Tier 2: extracted_text present → skip download/parse
      Tier 3: full pipeline via signed_url

    Returns (parsed_document, file_name, doc_id, error_message, precomputed_chunks).
    precomputed_chunks is non-None only for Tier 1.
    """
    file_name = doc.get("file_name") or "document"
    doc_id = doc.get("doc_id") or ""
    mime_type = doc.get("mime_type") or "application/pdf"
    size_bytes = doc.get("size_bytes")
    content_hash = doc.get("content_hash")
    signed_url = doc.get("signed_url") or ""
    extracted_text = doc.get("extracted_text")
    chunks = doc.get("chunks")

    # --- TIER 1: pre-computed chunks + embeddings ---
    if chunks and isinstance(chunks, list) and len(chunks) > 0:
        if verbose:
            print(
                f"[{file_name}] Tier 1: {len(chunks)} pre-computed chunks", flush=True)
        # Combine chunk content into full text for structural extraction / entity resolution
        sorted_chunks = sorted(chunks, key=lambda c: c.get("chunk_index", 0))
        full_text = "\n\n".join(c.get("content", "") for c in sorted_chunks)
        if not full_text.strip():
            return None, file_name, doc_id, f"Empty chunk content for {file_name}", None
        file_hash = content_hash or hashlib.sha256(
            full_text.encode("utf-8")).hexdigest()
        parsed = ParsedDocument(
            filepath=f"<precomputed:{file_name}>",
            filename=file_name,
            file_hash=file_hash,
            text=full_text,
            metadata={"source": "precomputed_chunks", "doc_id": doc_id},
            page_count=0,
        )
        return parsed, file_name, doc_id, None, chunks

    # --- TIER 2: text already extracted by frontend ---
    if extracted_text and extracted_text.strip():
        if verbose:
            print(
                f"[{file_name}] Tier 2: pre-extracted text ({len(extracted_text)} chars)", flush=True)
        file_hash = content_hash or hashlib.sha256(
            extracted_text.encode("utf-8")).hexdigest()
        parsed = ParsedDocument(
            filepath=f"<extracted:{file_name}>",
            filename=file_name,
            file_hash=file_hash,
            text=extracted_text,
            metadata={"source": "extracted_text", "doc_id": doc_id},
            page_count=0,
        )
        return parsed, file_name, doc_id, None, None

    # --- TIER 3: download from signed URL and parse ---
    if not signed_url:
        return None, file_name, doc_id, f"No signed_url, extracted_text, or chunks for {file_name}", None

    if size_bytes is not None and size_bytes > MAX_DOCUMENT_SIZE_BYTES:
        return None, file_name, doc_id, (
            f"Skipped {file_name}: exceeds 50MB limit ({size_bytes / (1024*1024):.1f}MB)"
        ), None

    if verbose:
        print(f"[{file_name}] Tier 3: Downloading from signed URL...", flush=True)

    data = _download_from_signed_url(signed_url)
    if not data:
        return None, file_name, doc_id, f"Failed to download {file_name}", None

    if len(data) > MAX_DOCUMENT_SIZE_BYTES:
        return None, file_name, doc_id, f"Skipped {file_name}: downloaded size exceeds 50MB limit", None

    if verbose:
        print(f"[{file_name}] Parsing...", flush=True)

    parsed = parser.parse_from_bytes(data, file_name, mime_type)
    if not parsed or not parsed.text.strip():
        return None, file_name, doc_id, f"Could not extract text from {file_name}", None

    # Override file_hash with content_hash from frontend for consistent dedup
    if content_hash:
        parsed.file_hash = content_hash

    return parsed, file_name, doc_id, None, None


def extract_from_frontend_payload(
    matter_id: str,
    documents: List[Dict[str, Any]],
    options: Dict[str, Any],
    api_key: str,
    verbose: bool = True,
    max_workers: int = _DEFAULT_DOWNLOAD_WORKERS,
    db_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract KG from documents sent by the Next.js frontend.

    Each document may include pre-extracted text (fast path) or a signed URL
    for download (slow path). Deduplication uses content_hash when skip_existing
    is enabled.

    Args:
        matter_id: The Iqidis matter ID (UUID)
        documents: List of document dicts from the frontend payload
        options: Options dict (e.g. {"skip_existing": true})
        api_key: Gemini API key
        verbose: Print progress messages
        max_workers: Parallel download workers for slow-path documents
        db_url: Optional explicit PostgreSQL URL (overrides env-based default)
    """
    def _log(msg: str):
        if verbose:
            print(msg, flush=True)

    skip_existing = bool(options.get("skip_existing", True))

    result: Dict[str, Any] = {
        "success": False,
        "matter_id": matter_id,
        "documents_found": len(documents),
        "documents_processed": 0,
        "documents_skipped": 0,
        "documents_failed": 0,
        "errors": [],
        "stats": {},
    }

    if not documents:
        result["success"] = True
        return result

    _log(f"Received {len(documents)} documents for matter {matter_id}. "
         f"skip_existing={skip_existing}, max_workers={max_workers}")

    start_time = time.time()
    kg = KnowledgeGraph(matter_id, api_key=api_key, db_url=db_url)
    parser = DocumentParser()

    # --- Pre-filter: skip already-processed documents ---
    docs_to_process: List[Dict[str, Any]] = []
    for doc in documents:
        if skip_existing and doc.get("content_hash"):
            existing = kg.extraction_pipeline.db.get_document_by_hash(
                doc["content_hash"])
            if existing and existing.processed_at:
                _log(
                    f"  ⏭ {doc.get('file_name', 'doc')}: already processed, skipping")
                result["documents_skipped"] += 1
                continue
        docs_to_process.append(doc)

    if not docs_to_process:
        elapsed = time.time() - start_time
        result["success"] = True
        result["stats"] = kg.get_stats()
        result["stats"]["total_time_seconds"] = round(elapsed, 1)
        kg.close()
        _log("All documents already processed. Nothing to do.")
        return result

    _log(f"Processing {len(docs_to_process)} documents "
         f"({result['documents_skipped']} skipped)...")

    # Release any open transaction on the original kg so consumer threads
    # don't deadlock on ACCESS EXCLUSIVE locks (e.g. ALTER TABLE migrations).
    try:
        kg.db.conn.commit()
    except Exception:
        pass

    # --- Parallel extraction: multiple consumers with own DB connections ---
    # Up to 3 parallel extractions
    NUM_CONSUMERS = min(3, len(docs_to_process))
    doc_queue: queue.Queue = queue.Queue()
    SENTINEL = None
    result_lock = threading.Lock()

    def _extraction_consumer(consumer_id: int):
        """Consume prepared documents and run KG extraction."""
        import traceback as _tb
        _log(f"  [C{consumer_id}] Starting consumer thread...")
        # Each consumer gets its own KnowledgeGraph (own DB connection)
        try:
            kg_local = KnowledgeGraph(
                matter_id, api_key=api_key, db_url=db_url)
        except Exception as e:
            _log(f"  ✗ [C{consumer_id}] Failed to init KnowledgeGraph: {e}")
            _log(_tb.format_exc())
            # Drain items so other consumers / join() don't hang
            return
        _log(f"  [C{consumer_id}] KnowledgeGraph ready")
        try:
            while True:
                item = doc_queue.get()
                if item is SENTINEL:
                    break
                parsed, file_name, doc_id, chunks = item
                tier = "tier1" if chunks else (
                    "tier2" if parsed.metadata.get("source") == "extracted_text" else "tier3")
                _log(f"  [C{consumer_id}] Extracting {file_name} ({tier})...")
                t0 = time.time()
                try:
                    if chunks:
                        # Tier 1: pre-computed chunks + embeddings
                        kg_local.extraction_pipeline.process_precomputed_chunks(
                            parsed, chunks,
                            skip_if_exists=skip_existing,
                            doc_id=doc_id or None)
                    else:
                        # Tier 2/3: normal processing
                        kg_local.extraction_pipeline.process_parsed_document(
                            parsed, skip_if_exists=skip_existing,
                            doc_id=doc_id or None)
                    elapsed_doc = (time.time() - t0) * 1000
                    with result_lock:
                        result["documents_processed"] += 1
                    _log(f"  ✓ [C{consumer_id}] {file_name} ({tier}): "
                         f"{elapsed_doc:.0f}ms, chunks={len(chunks or [])}")
                except Exception as e:
                    with result_lock:
                        result["documents_failed"] += 1
                        result["errors"].append(f"{file_name}: {str(e)}")
                    _log(
                        f"  ✗ [C{consumer_id}] Failed extraction for {file_name}: {e}")
        finally:
            kg_local.close()

    # Start consumer threads
    consumer_threads = []
    for i in range(NUM_CONSUMERS):
        t = threading.Thread(target=_extraction_consumer,
                             args=(i,), daemon=True)
        t.start()
        consumer_threads.append(t)

    # Prepare documents in parallel (download if needed)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_doc = {
            executor.submit(_prepare_document_from_payload, doc, parser, verbose): doc
            for doc in docs_to_process
        }
        for future in as_completed(future_to_doc):
            doc = future_to_doc[future]
            file_name = doc.get("file_name") or "document"
            try:
                parsed, name, doc_id, error, chunks = future.result()
                if error:
                    with result_lock:
                        result["documents_failed"] += 1
                        result["errors"].append(error)
                    _log(f"  ✗ {name}: {error}")
                elif parsed:
                    tier_label = "Tier 1" if chunks else (
                        "Tier 2" if parsed.metadata.get("source") == "extracted_text" else "Tier 3")
                    _log(
                        f"  ✓ {name}: Ready ({tier_label}), queued for extraction")
                    doc_queue.put((parsed, name, doc_id, chunks))
            except Exception as e:
                with result_lock:
                    result["documents_failed"] += 1
                    result["errors"].append(f"{file_name}: {str(e)}")
                _log(f"  ✗ {file_name}: {e}")

    # Signal all consumers to finish and wait
    for _ in range(NUM_CONSUMERS):
        doc_queue.put(SENTINEL)
    for t in consumer_threads:
        t.join()

    elapsed = time.time() - start_time
    result["stats"] = kg.get_stats()
    result["stats"]["total_time_seconds"] = round(elapsed, 1)
    kg.close()
    result["success"] = result["documents_processed"] > 0 or result["documents_skipped"] > 0
    _log(f"\nCompleted in {elapsed:.1f}s. "
         f"Processed {result['documents_processed']}, "
         f"skipped {result['documents_skipped']}, "
         f"failed {result['documents_failed']}.")
    return result


# ---------------------------------------------------------------------------
# Legacy mode: query Iqidis DB for documents
# ---------------------------------------------------------------------------

def extract_from_iqidis_matter(
    matter_id: str,
    api_key: str,
    verbose: bool = True,
    max_workers: int = _DEFAULT_DOWNLOAD_WORKERS,
) -> Dict[str, Any]:
    """
    Extract entities and relationships from Iqidis matter documents.
    Stores the Knowledge Graph in PostgreSQL (kg_entities, kg_edges, etc.).

    Uses a pipelined approach: extraction starts as soon as documents
    are downloaded and parsed, overlapping I/O with computation.

    Args:
        matter_id: The Iqidis matter ID (UUID)
        api_key: Gemini API key
        verbose: Print progress messages
        max_workers: Number of parallel workers for downloading and parsing
    """
    def _log(msg: str):
        if verbose:
            print(msg, flush=True)

    result = {
        "success": False,
        "matter_id": matter_id,
        "matter_name": "",
        "documents_found": 0,
        "documents_processed": 0,
        "documents_failed": 0,
        "errors": [],
        "stats": {},
    }

    matter = get_matter_by_id(matter_id)
    if not matter:
        result["errors"].append(
            f"Matter {matter_id} not found in Iqidis database")
        return result

    result["matter_name"] = matter.get("matter_name", matter_id)

    docs = get_matter_documents(matter_id)
    result["documents_found"] = len(docs)

    if not docs:
        result["success"] = True
        result["errors"].append("No documents found for this matter")
        return result

    _log(f"Found {len(docs)} documents. Starting pipelined extraction with {max_workers} download workers...")

    start_time = time.time()
    kg = KnowledgeGraph(matter_id, api_key=api_key)
    parser = DocumentParser()

    # Pipelined approach: download/parse in parallel, extract sequentially as docs arrive
    # Use a thread-safe queue to feed parsed docs to the extraction consumer
    doc_queue: queue.Queue = queue.Queue()
    SENTINEL = None  # Signals end of downloads
    result_lock = threading.Lock()  # Protect result dict mutations across threads

    def _extraction_consumer():
        """Consume parsed documents and extract KG (sequential for DB safety)."""
        while True:
            item = doc_queue.get()
            if item is SENTINEL:
                break
            parsed, original_name = item
            try:
                kg.extraction_pipeline.process_parsed_document(
                    parsed, skip_if_exists=True)
                with result_lock:
                    result["documents_processed"] += 1
                _log(f"  ✓ Extracted from {original_name}")
            except Exception as e:
                with result_lock:
                    result["documents_failed"] += 1
                    result["errors"].append(f"{original_name}: {str(e)}")
                _log(f"  ✗ Failed extraction for {original_name}: {e}")

    # Start extraction consumer thread
    consumer_thread = threading.Thread(
        target=_extraction_consumer, daemon=True)
    consumer_thread.start()

    # Download and parse in parallel, feeding results to extraction queue
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_doc = {
            executor.submit(_download_and_parse_document, doc_info, parser, verbose): doc_info
            for doc_info in docs
        }

        for future in as_completed(future_to_doc):
            doc_info = future_to_doc[future]
            original_name = doc_info.get("original_name") or "document"

            try:
                parsed, name, error = future.result()
                if error:
                    with result_lock:
                        result["documents_failed"] += 1
                        result["errors"].append(error)
                    _log(f"  ✗ {name}: {error}")
                elif parsed:
                    _log(
                        f"  ✓ {name}: Downloaded and parsed, queued for extraction")
                    doc_queue.put((parsed, name))
            except Exception as e:
                with result_lock:
                    result["documents_failed"] += 1
                    result["errors"].append(f"{original_name}: {str(e)}")
                _log(f"  ✗ {original_name}: {e}")

    # Signal extraction consumer to finish
    doc_queue.put(SENTINEL)
    consumer_thread.join()

    elapsed = time.time() - start_time
    result["stats"] = kg.get_stats()
    result["stats"]["total_time_seconds"] = round(elapsed, 1)
    kg.close()
    result["success"] = result["documents_processed"] > 0
    _log(
        f"\nCompleted in {elapsed:.1f}s. Processed {result['documents_processed']}/{result['documents_found']} documents.")
    return result
