#!/usr/bin/env python3
"""
RLM Investigation API Load Testing Harness — Enhanced

Enhancements over load_test_rlm.py:
  - Dataset: main_doc.json (1000+ documents with sizeByte metadata)
  - Document size bucketing: Small / Medium / Large
  - Guaranteed Word document inclusion (MIN_WORD_DOCS_PER_REQUEST)
  - Cumulative request size cap (MAX_REQUEST_SIZE_MB)
  - Enriched JSONL log entries with composition metadata
  - Request-composition section in the auto-generated summary report
"""

import json
import time
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Any, Tuple
from statistics import mean, median

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_API_URL         = "https://rlm.iryslegal.com"
DOCUMENT_FILE        = "main_doc.json"
S3_BUCKET_BASE       = "https://iqidis-artifact.s3.us-east-1.amazonaws.com"
TEST_QUERY           = "Test investigation - load testing"
REQUESTS_PER_MINUTE  = 10
TEST_DURATION_MINUTES = 2

# Concurrent burst mode config
CONCURRENT_BURST_COUNT = 5   # Number of requests fired simultaneously
MIN_DOCS_PER_REQUEST = 200
MAX_DOCS_PER_REQUEST = 300
LOG_FILE             = "load_test_enhanced_log.jsonl"
REQUEST_TIMEOUT      = 6000

# Enhanced sampling config
MIN_WORD_DOCS_PER_REQUEST    = 5     # Minimum Word docs guaranteed per request
MAX_REQUEST_SIZE_MB          = 200   # Hard cap on cumulative request payload size
FORCE_LARGE_DOC_PROBABILITY  = 0.15  # Probability of forcing ≥1 large doc per request

# Size-bucket thresholds (bytes)
SMALL_MAX_BYTES  = 500_000           # < 500 KB  → Small
MEDIUM_MAX_BYTES = 10_000_000        # 500 KB – 10 MB → Medium
                                     # > 10 MB          → Large
WORD_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

# ============================================================================
# SEQUENTIAL DOCUMENT CYCLER
# ============================================================================

class DocumentCycler:
    """
    Manages sequential cycling through the full document pool.

    The pool is shuffled once at construction. Documents are yielded
    sequentially across requests. When the end of the pool is reached the
    pool is reshuffled and the index resets, ensuring variety across test
    cycles while preventing the same documents from appearing in consecutive
    requests.
    """

    def __init__(self, documents: List[Dict[str, Any]]) -> None:
        self._pool: List[Dict[str, Any]] = documents[:]
        random.shuffle(self._pool)
        self._index: int = 0
        print(f"DocumentCycler: {len(self._pool)} documents in pool (shuffled)")

    def get_fill_docs(
        self,
        target_count: int,
        exclude_urls: set,
        current_bytes: int,
        max_bytes: int,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Pull up to target_count documents sequentially from the cycled pool.

        Skips documents whose URL is in exclude_urls (already selected) or
        that would push total bytes past max_bytes.  Reshuffles and resets
        the index when the end of the pool is reached.

        At most one full pool traversal is attempted per call to prevent
        infinite loops when very few documents fit under the size cap.

        Args:
            target_count:  Number of additional documents to collect.
            exclude_urls:  Set of already-selected URLs — mutated in-place.
            current_bytes: Cumulative byte total accumulated so far.
            max_bytes:     Hard cap on cumulative bytes.

        Returns:
            Tuple of (selected_docs, updated_total_bytes).
        """
        result: List[Dict[str, Any]] = []
        total_bytes = current_bytes
        examined = 0  # safety: stop after one full traversal per call

        while len(result) < target_count and examined < len(self._pool):
            # Reshuffle and reset when we reach the end of the pool
            if self._index >= len(self._pool):
                random.shuffle(self._pool)
                self._index = 0

            doc = self._pool[self._index]
            self._index += 1
            examined += 1

            url  = doc.get('url')
            size = doc.get('sizeByte', 0)

            if url in exclude_urls:
                continue
            if total_bytes + size > max_bytes:
                continue

            result.append(doc)
            exclude_urls.add(url)
            total_bytes += size

        return result, total_bytes


# ============================================================================
# DOCUMENT LOADING AND DEDUPLICATION
# ============================================================================

def load_documents(filepath: str) -> List[Dict[str, Any]]:
    """
    Load documents from a JSON file and deduplicate by URL.

    Supports both:
      - Top-level list of user objects (each with a 'documents' key)
      - Dict with a top-level 'documents' key (legacy format)

    Returns:
        List of unique document dicts (url, name, mime, sizeByte)
    """
    print(f"Loading documents from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_docs: List[Dict[str, Any]] = []
    if isinstance(data, list):
        for user in data:
            all_docs.extend(user.get('documents', []))
    else:
        all_docs = data.get('documents', [])

    print(f"Loaded {len(all_docs)} documents (across all users)")

    seen_urls: set = set()
    unique_docs: List[Dict[str, Any]] = []
    for doc in all_docs:
        url = doc.get('url')
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_docs.append(doc)

    print(f"After deduplication: {len(unique_docs)} unique documents")
    return unique_docs


# ============================================================================
# DOCUMENT CLASSIFICATION
# ============================================================================

def classify_documents(documents: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Classify documents into size buckets and isolate Word documents.

    Buckets:
        small_documents   — sizeByte < SMALL_MAX_BYTES
        medium_documents  — SMALL_MAX_BYTES <= sizeByte < MEDIUM_MAX_BYTES
        large_documents   — sizeByte >= MEDIUM_MAX_BYTES
        word_documents    — MIME == WORD_MIME (can overlap with size buckets)

    Returns:
        Dict with keys: 'small', 'medium', 'large', 'word', 'all'
    """
    small:  List[Dict[str, Any]] = []
    medium: List[Dict[str, Any]] = []
    large:  List[Dict[str, Any]] = []
    word:   List[Dict[str, Any]] = []

    for doc in documents:
        size = doc.get('sizeByte', 0)
        mime = doc.get('mime', '')

        if mime == WORD_MIME:
            word.append(doc)

        if size < SMALL_MAX_BYTES:
            small.append(doc)
        elif size < MEDIUM_MAX_BYTES:
            medium.append(doc)
        else:
            large.append(doc)

    print(f"\nDocument classification:")
    print(f"  Small  (< {SMALL_MAX_BYTES // 1000} KB):   {len(small):>5}")
    print(f"  Medium (< {MEDIUM_MAX_BYTES // 1_000_000} MB):  {len(medium):>5}")
    print(f"  Large  (>= {MEDIUM_MAX_BYTES // 1_000_000} MB): {len(large):>5}")
    print(f"  Word documents:         {len(word):>5}")
    print()

    return {
        'small':  small,
        'medium': medium,
        'large':  large,
        'word':   word,
        'all':    documents,
    }




# ============================================================================
# REQUEST PAYLOAD BUILDER
# ============================================================================

def build_request_payload(
    buckets: Dict[str, List[Dict[str, Any]]],
    min_docs: int,
    max_docs: int,
    min_word_docs: int,
    max_request_size_mb: float,
    cycler: DocumentCycler,
    force_large_prob: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Build a single request's document list.

    Selection order:
      0. With probability force_large_prob, force-include one large document
         (only if the large bucket is non-empty and size cap allows).
      1. Add Word documents first (random shuffle of word pool), up to
         min_word_docs or all available — whichever is smaller.
      2. Fill remaining slots sequentially via DocumentCycler, respecting
         the max_docs target and max_request_size_mb cap.
      3. Shuffle the final list to remove ordering bias.

    Args:
        buckets:             Output of classify_documents()
        min_docs:            Minimum total documents in the request
        max_docs:            Maximum total documents in the request
        min_word_docs:       Minimum Word documents to include
        max_request_size_mb: Hard cap on cumulative sizeByte (MB)
        cycler:              DocumentCycler instance shared across requests
        force_large_prob:    Probability (0–1) of forcing ≥1 large doc

    Returns:
        Tuple of:
          - selected document dicts (url, name, mime, sizeByte)
          - composition dict {documents_sent, total_request_size_bytes,
                              small_docs, medium_docs, large_docs, word_docs}
    """
    max_bytes    = int(max_request_size_mb * 1024 * 1024)
    target_count = random.randint(min_docs, max_docs)

    selected: List[Dict[str, Any]] = []
    selected_urls: set = set()
    total_bytes = 0

    def _can_add(doc: Dict[str, Any]) -> bool:
        return (
            doc.get('url') not in selected_urls
            and total_bytes + doc.get('sizeByte', 0) <= max_bytes
            and len(selected) < target_count
        )

    # Step 0 — optionally force one large document
    if buckets['large'] and random.random() < force_large_prob:
        large_pool = buckets['large'][:]
        random.shuffle(large_pool)
        for doc in large_pool:
            if _can_add(doc):
                selected.append(doc)
                selected_urls.add(doc['url'])
                total_bytes += doc.get('sizeByte', 0)
                break

    # Step 1 — add Word documents first (unchanged logic)
    word_pool = buckets['word'][:]
    random.shuffle(word_pool)
    for doc in word_pool:
        if len([d for d in selected if d.get('mime') == WORD_MIME]) >= min_word_docs:
            break
        if _can_add(doc):
            selected.append(doc)
            selected_urls.add(doc['url'])
            total_bytes += doc.get('sizeByte', 0)

    # Step 2 — fill remaining slots sequentially via DocumentCycler
    remaining = target_count - len(selected)
    if remaining > 0:
        fill_docs, total_bytes = cycler.get_fill_docs(
            target_count  = remaining,
            exclude_urls  = selected_urls,
            current_bytes = total_bytes,
            max_bytes     = max_bytes,
        )
        selected.extend(fill_docs)

    # Step 3 — shuffle final list
    random.shuffle(selected)

    # Step 4 — compute composition metadata
    word_count   = sum(1 for d in selected if d.get('mime') == WORD_MIME)
    small_count  = sum(1 for d in selected if d.get('sizeByte', 0) < SMALL_MAX_BYTES)
    medium_count = sum(1 for d in selected if SMALL_MAX_BYTES <= d.get('sizeByte', 0) < MEDIUM_MAX_BYTES)
    large_count  = sum(1 for d in selected if d.get('sizeByte', 0) >= MEDIUM_MAX_BYTES)

    composition = {
        "documents_sent":           len(selected),
        "total_request_size_bytes": total_bytes,
        "small_docs":               small_count,
        "medium_docs":              medium_count,
        "large_docs":               large_count,
        "word_docs":                word_count,
    }

    return selected, composition


# ============================================================================
# S3 URL CONSTRUCTION
# ============================================================================

def construct_s3_url(storage_key: str) -> str:
    """Construct S3 URL from storage key."""
    return f"{S3_BUCKET_BASE}/{storage_key}"


def build_s3_url_objects(documents: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Strip documents down to {url, name, mime} for the API payload.

    Supports both new format (url/name/mime/sizeByte) and legacy format
    (storageKey/originalName/mime).
    """
    s3_objects = []
    for doc in documents:
        mime = doc.get('mime')
        url  = doc.get('url')
        name = doc.get('name')

        # Legacy fallback
        if not url:
            storage_key   = doc.get('storageKey')
            original_name = doc.get('originalName')
            if storage_key and original_name:
                url  = construct_s3_url(storage_key)
                name = original_name

        if all([url, name, mime]):
            s3_objects.append({"url": url, "name": name, "mime": mime})

    return s3_objects


# ============================================================================
# API REQUEST
# ============================================================================

def send_investigation_request(
    query: str,
    s3_urls: List[Dict[str, str]],
    timeout: int,
) -> Dict[str, Any]:
    """
    POST an investigation request to the RLM service and measure latency.

    Returns:
        Dict with status_code, latency_seconds, response_size, error
    """
    endpoint = f"{BASE_API_URL}/investigate/urls/stream"
    payload  = {"query": query, "s3_urls": s3_urls}

    start_time = time.time()
    try:
        response = requests.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        latency = time.time() - start_time
        return {
            "status_code":      response.status_code,
            "latency_seconds":  round(latency, 2),
            "response_size":    len(response.content),
            "response_headers": dict(response.headers),
            "response_body":    response.text[:500],
            "error":            None,
        }
    except requests.exceptions.Timeout:
        latency = time.time() - start_time
        return {"status_code": None, "latency_seconds": round(latency, 2),
                "response_size": 0, "error": "Request timeout"}
    except requests.exceptions.RequestException as exc:
        latency = time.time() - start_time
        return {"status_code": None, "latency_seconds": round(latency, 2),
                "response_size": 0, "error": str(exc)}


# ============================================================================
# LOGGING
# ============================================================================

def log_request(log_file: str, log_entry: Dict[str, Any]) -> None:
    """Append a log entry to the JSONL log file."""
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry) + '\n')


# ============================================================================
# PERFORMANCE SUMMARY REPORT
# ============================================================================

def calculate_percentile(values: List[float], percentile: int) -> float:
    """Calculate nth percentile using linear interpolation."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    index = (percentile / 100.0) * len(sorted_vals)
    if index.is_integer():
        return sorted_vals[int(index) - 1]
    lower = sorted_vals[int(index) - 1]
    upper = sorted_vals[int(index)]
    return lower + (upper - lower) * (index - int(index))


def generate_summary_report(log_file: str, total_test_time: float) -> None:
    """
    Read the JSONL log file and print a formatted performance summary.

    Includes a REQUEST COMPOSITION section showing average bucket breakdown
    and Word document counts per request.
    """
    try:
        entries = []
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        if not entries:
            print("\n⚠ WARNING: Log file is empty, cannot generate summary report")
            return

        total_requests    = len(entries)
        successful        = [e for e in entries if e.get('status_code') == 200]
        failed            = [e for e in entries if e.get('status_code') != 200 or e.get('error')]
        success_count     = len(successful)
        success_rate      = (success_count / total_requests * 100) if total_requests > 0 else 0
        actual_rpm        = (total_requests / total_test_time * 60) if total_test_time > 0 else 0

        latencies = [e['latency_seconds'] for e in successful if e.get('latency_seconds') is not None]
        if latencies:
            lat_min    = min(latencies)
            lat_max    = max(latencies)
            lat_mean   = mean(latencies)
            lat_median = median(latencies)
            lat_p95    = calculate_percentile(latencies, 95)
            lat_p99    = calculate_percentile(latencies, 99)
        else:
            lat_min = lat_max = lat_mean = lat_median = lat_p95 = lat_p99 = 0.0

        # Composition averages (all requests, not just successful)
        docs_counts  = [e.get('documents_sent', 0) for e in entries]
        small_counts = [e.get('small_docs', 0)     for e in entries]
        med_counts   = [e.get('medium_docs', 0)    for e in entries]
        large_counts = [e.get('large_docs', 0)     for e in entries]
        word_counts  = [e.get('word_docs', 0)      for e in entries]
        size_bytes   = [e.get('total_request_size_bytes', 0) for e in entries]

        avg = lambda lst: sum(lst) / len(lst) if lst else 0.0

        # Error summary
        error_counts: Dict[str, int] = {}
        for e in failed:
            error  = e.get('error', 'Unknown error')
            status = e.get('status_code', 'None')
            key    = f"{status}: {error}" if error else f"Status {status}"
            error_counts[key] = error_counts.get(key, 0) + 1

        # ---- Print report ----
        print("\n" + "=" * 80)
        print("PERFORMANCE SUMMARY REPORT")
        print("=" * 80)

        print("\nOVERVIEW")
        print("-" * 80)
        print(f"{'Total Requests:':<35} {total_requests:>10}")
        print(f"{'Successful Requests:':<35} {success_count:>10}")
        print(f"{'Failed Requests:':<35} {len(failed):>10}")
        print(f"{'Success Rate:':<35} {success_rate:>9.1f}%")
        print(f"{'Actual Throughput:':<35} {actual_rpm:>9.1f} req/min")

        print("\nLATENCY STATISTICS (seconds)")
        print("-" * 80)
        print(f"{'Minimum:':<35} {lat_min:>10.2f}s")
        print(f"{'Maximum:':<35} {lat_max:>10.2f}s")
        print(f"{'Mean (Average):':<35} {lat_mean:>10.2f}s")
        print(f"{'Median:':<35} {lat_median:>10.2f}s")
        print(f"{'P95 (95th percentile):':<35} {lat_p95:>10.2f}s")
        print(f"{'P99 (99th percentile):':<35} {lat_p99:>10.2f}s")

        print("\nREQUEST COMPOSITION (averages per request)")
        print("-" * 80)
        print(f"{'Avg Total Documents:':<35} {avg(docs_counts):>10.1f}")
        print(f"{'Avg Small Docs:':<35} {avg(small_counts):>10.1f}")
        print(f"{'Avg Medium Docs:':<35} {avg(med_counts):>10.1f}")
        print(f"{'Avg Large Docs:':<35} {avg(large_counts):>10.1f}")
        print(f"{'Avg Word Docs:':<35} {avg(word_counts):>10.1f}")
        print(f"{'Avg Request Size (MB):':<35} {avg(size_bytes) / (1024*1024):>10.2f}")

        if error_counts:
            print("\nERROR SUMMARY")
            print("-" * 80)
            for key, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {count:>5} × {key}")

        print("\n" + "=" * 80)

    except FileNotFoundError:
        print(f"\n⚠ WARNING: Log file '{log_file}' not found")
    except json.JSONDecodeError as exc:
        print(f"\n⚠ WARNING: Invalid JSON in log file: {exc}")
    except Exception as exc:
        print(f"\n⚠ WARNING: Error generating summary report: {exc}")


# ============================================================================
# LOAD TEST SCHEDULER
# ============================================================================

def run_load_test(
    buckets: Dict[str, List[Dict[str, Any]]],
    requests_per_minute: int,
    duration_minutes: int,
    query: str,
    min_docs: int,
    max_docs: int,
    min_word_docs: int,
    max_request_size_mb: float,
    log_file: str,
    timeout: int,
    force_large_prob: float,
) -> None:
    """
    Execute the enhanced load test with controlled request rate.

    For each request:
      1. Calls build_request_payload() with a shared DocumentCycler so docs
         are pulled sequentially across the full dataset.
      2. Strips sizeByte via build_s3_url_objects() before sending.
      3. Logs enriched JSONL entry with composition metadata.
    """
    interval_seconds = 60.0 / requests_per_minute
    total_requests   = requests_per_minute * duration_minutes

    # Create cycler once — shared across all requests
    cycler = DocumentCycler(buckets['all'])

    print(f"\nStarting enhanced load test")
    print(f"  Dataset:               {DOCUMENT_FILE}")
    print(f"  Total unique docs:     {len(buckets['all'])}")
    print(f"  Word docs available:   {len(buckets['word'])}")
    print(f"  Large docs available:  {len(buckets['large'])}")
    print(f"  Force large prob:      {force_large_prob:.0%}")
    print(f"  Rate:                  {requests_per_minute} req/min")
    print(f"  Duration:              {duration_minutes} min")
    print(f"  Total requests:        {total_requests}")
    print(f"  Docs per request:      {min_docs}–{max_docs}")
    print(f"  Min Word docs:         {min_word_docs}")
    print(f"  Max request size:      {max_request_size_mb} MB")
    print(f"  Interval:              {interval_seconds:.1f}s between requests")
    print(f"  Log file:              {log_file}\n")

    # Clear / create log file
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('')

    request_count  = 0
    start_test_time = time.time()

    for i in range(total_requests):
        request_count += 1
        request_id    = f"req_{request_count:03d}"
        iter_start    = time.time()

        # Build payload using enhanced sampling
        selected_docs, composition = build_request_payload(
            buckets, min_docs, max_docs, min_word_docs, max_request_size_mb,
            cycler=cycler, force_large_prob=force_large_prob,
        )
        s3_urls = build_s3_url_objects(selected_docs)

        # Send request
        result = send_investigation_request(query, s3_urls, timeout)

        # Build enriched log entry
        log_entry = {
            "timestamp":                datetime.now().isoformat(),
            "request_id":               request_id,
            "documents_sent":           composition["documents_sent"],
            "total_request_size_bytes": composition["total_request_size_bytes"],
            "small_docs":               composition["small_docs"],
            "medium_docs":              composition["medium_docs"],
            "large_docs":               composition["large_docs"],
            "word_docs":                composition["word_docs"],
            "status_code":              result.get("status_code"),
            "latency_seconds":          result.get("latency_seconds"),
            "response_size":            result.get("response_size"),
            "error":                    result.get("error"),
        }
        log_request(log_file, log_entry)

        # Console summary line
        status     = result.get("status_code", "ERROR")
        latency    = result.get("latency_seconds", 0)
        size_mb    = composition["total_request_size_bytes"] / (1024 * 1024)
        error_msg  = f" → {result.get('error')}" if result.get('error') else ""
        print(
            f"[{request_id}] "
            f"{composition['documents_sent']} docs "
            f"(S:{composition['small_docs']} M:{composition['medium_docs']} "
            f"L:{composition['large_docs']} W:{composition['word_docs']}) "
            f"| {size_mb:.1f} MB | {latency}s | {status}{error_msg}"
        )

        # Maintain interval
        elapsed    = time.time() - iter_start
        sleep_time = max(0, interval_seconds - elapsed)
        if i < total_requests - 1:
            time.sleep(sleep_time)

    total_test_time = time.time() - start_test_time
    generate_summary_report(log_file, total_test_time)

    print(f"\n{'='*80}")
    print(f"Enhanced load test completed")
    print(f"Total requests: {request_count}")
    print(f"Total time:     {total_test_time:.1f}s")
    print(f"Average rate:   {request_count / (total_test_time / 60):.1f} req/min")
    print(f"Log file:       {log_file}")
    print(f"{'='*80}\n")


# ============================================================================
# CONCURRENT BURST MODE
# ============================================================================

def run_concurrent_burst(
    buckets: Dict[str, List[Dict[str, Any]]],
    burst_count: int,
    query: str,
    min_docs: int,
    max_docs: int,
    min_word_docs: int,
    max_request_size_mb: float,
    log_file: str,
    timeout: int,
    force_large_prob: float,
) -> None:
    """
    Fire burst_count requests all at the same time (concurrent) and wait for
    all of them to complete.

    All payloads are built upfront before any request is sent so that the HTTP
    calls start as close to simultaneously as possible.  A shared
    DocumentCycler ensures each request gets a distinct slice of the document
    pool.
    """
    cycler = DocumentCycler(buckets['all'])

    print(f"\nStarting concurrent burst test")
    print(f"  Dataset:               {DOCUMENT_FILE}")
    print(f"  Concurrent requests:   {burst_count}")
    print(f"  Docs per request:      {min_docs}–{max_docs}")
    print(f"  Min Word docs:         {min_word_docs}")
    print(f"  Max request size:      {max_request_size_mb} MB")
    print(f"  Force large prob:      {force_large_prob:.0%}")
    print(f"  Log file:              {log_file}\n")

    # Clear / create log file
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('')

    # Build all payloads upfront (sequential — fast)
    payloads: List[Tuple[str, List[Dict[str, str]], Dict[str, int]]] = []
    for i in range(burst_count):
        selected_docs, composition = build_request_payload(
            buckets, min_docs, max_docs, min_word_docs, max_request_size_mb,
            cycler=cycler, force_large_prob=force_large_prob,
        )
        s3_urls = build_s3_url_objects(selected_docs)
        request_id = f"req_{i + 1:03d}"
        payloads.append((request_id, s3_urls, composition))

    print(f"All {burst_count} payloads built — firing concurrently...\n")

    burst_start = time.time()

    def _worker(args: Tuple[str, List[Dict[str, str]], Dict[str, int]]) -> Dict[str, Any]:
        req_id, s3_urls, composition = args
        result = send_investigation_request(query, s3_urls, timeout)
        return {
            "request_id":               req_id,
            "composition":              composition,
            "result":                   result,
            "timestamp":                datetime.now().isoformat(),
        }

    results = []
    with ThreadPoolExecutor(max_workers=burst_count) as executor:
        futures = {executor.submit(_worker, p): p[0] for p in payloads}
        for future in as_completed(futures):
            entry_data = future.result()
            results.append(entry_data)

            req_id      = entry_data["request_id"]
            composition = entry_data["composition"]
            result      = entry_data["result"]
            status      = result.get("status_code", "ERROR")
            latency     = result.get("latency_seconds", 0)
            size_mb     = composition["total_request_size_bytes"] / (1024 * 1024)
            error_msg   = f" → {result.get('error')}" if result.get("error") else ""

            print(
                f"[{req_id}] "
                f"{composition['documents_sent']} docs "
                f"(S:{composition['small_docs']} M:{composition['medium_docs']} "
                f"L:{composition['large_docs']} W:{composition['word_docs']}) "
                f"| {size_mb:.1f} MB | {latency}s | {status}{error_msg}"
            )

            log_entry = {
                "timestamp":                entry_data["timestamp"],
                "request_id":               req_id,
                "documents_sent":           composition["documents_sent"],
                "total_request_size_bytes": composition["total_request_size_bytes"],
                "small_docs":               composition["small_docs"],
                "medium_docs":              composition["medium_docs"],
                "large_docs":               composition["large_docs"],
                "word_docs":                composition["word_docs"],
                "status_code":              result.get("status_code"),
                "latency_seconds":          result.get("latency_seconds"),
                "response_size":            result.get("response_size"),
                "error":                    result.get("error"),
            }
            log_request(log_file, log_entry)

    total_time = time.time() - burst_start
    generate_summary_report(log_file, total_time)

    print(f"\n{'='*80}")
    print(f"Concurrent burst test completed")
    print(f"Requests fired:  {burst_count}")
    print(f"Wall-clock time: {total_time:.1f}s")
    print(f"Log file:        {log_file}")
    print(f"{'='*80}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main() -> None:
    """Main entry point.

    Prompts for test mode:
      1 — Sequential (original): requests spaced ~6 s apart over 2 minutes
      2 — Concurrent burst:      CONCURRENT_BURST_COUNT requests all at once
    """
    print("=" * 80)
    print("RLM Investigation API Load Testing Harness — Enhanced")
    print("=" * 80)
    print()
    print("Select test mode:")
    print(f"  [1] Sequential  — {REQUESTS_PER_MINUTE} req/min × {TEST_DURATION_MINUTES} min "
          f"({REQUESTS_PER_MINUTE * TEST_DURATION_MINUTES} total, ~{60 // REQUESTS_PER_MINUTE}s apart)")
    print(f"  [2] Concurrent  — {CONCURRENT_BURST_COUNT} requests fired simultaneously")
    print()

    mode = input("Enter mode [1/2] (default 2): ").strip() or "2"

    # Load and deduplicate
    documents = load_documents(DOCUMENT_FILE)
    if not documents:
        print("ERROR: No documents found in dataset")
        return

    # Classify into buckets
    buckets = classify_documents(documents)

    if len(buckets['all']) < MIN_DOCS_PER_REQUEST:
        print(f"ERROR: Not enough documents ({len(buckets['all'])}) for "
              f"minimum request size ({MIN_DOCS_PER_REQUEST})")
        return

    if not buckets['word']:
        print("WARNING: No Word documents found — MIN_WORD_DOCS_PER_REQUEST "
              "will be satisfied with 0 Word docs")

    if mode == "1":
        run_load_test(
            buckets              = buckets,
            requests_per_minute  = REQUESTS_PER_MINUTE,
            duration_minutes     = TEST_DURATION_MINUTES,
            query                = TEST_QUERY,
            min_docs             = MIN_DOCS_PER_REQUEST,
            max_docs             = MAX_DOCS_PER_REQUEST,
            min_word_docs        = MIN_WORD_DOCS_PER_REQUEST,
            max_request_size_mb  = MAX_REQUEST_SIZE_MB,
            log_file             = LOG_FILE,
            timeout              = REQUEST_TIMEOUT,
            force_large_prob     = FORCE_LARGE_DOC_PROBABILITY,
        )
    else:
        run_concurrent_burst(
            buckets              = buckets,
            burst_count          = CONCURRENT_BURST_COUNT,
            query                = TEST_QUERY,
            min_docs             = MIN_DOCS_PER_REQUEST,
            max_docs             = MAX_DOCS_PER_REQUEST,
            min_word_docs        = MIN_WORD_DOCS_PER_REQUEST,
            max_request_size_mb  = MAX_REQUEST_SIZE_MB,
            log_file             = LOG_FILE,
            timeout              = REQUEST_TIMEOUT,
            force_large_prob     = FORCE_LARGE_DOC_PROBABILITY,
        )


if __name__ == "__main__":
    main()
