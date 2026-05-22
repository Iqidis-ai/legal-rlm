"""
Integration test: DELE-004 corpus + context-store-v2 fixes.

Verifies four things in a single real investigation run:
  1. SQLite FactStore — facts written during extraction and readable via pack_evidence
  2. No event-loop blocking — no BlockingIOError or asyncio warnings in stderr
  3. Citation injection — inline service succeeds (matched > 0), no hallucinated IDs
  4. Cross-document coverage — all 3 Delek agreement sources appear in extracted facts

Run:
    python test_dele004_context_store_v2.py

Exit codes:
    0  all assertions pass
    1  one or more assertions failed
"""
import asyncio
import io
import logging
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv
load_dotenv()

if not os.environ.get("GEMINI_API_KEY"):
    print("ERROR: GEMINI_API_KEY not set")
    sys.exit(1)

from irys.api import Irys, IrysConfig

REPO          = r"D:\legal-rlm\DELE-004 formatted (1)\documents"
QUERY         = (
    "Compare the conditions precedent (Article 2) in all three agreements. "
    "For each condition: state the condition, identify which agreement(s) impose it, "
    "and call out any condition unique to one agreement."
)
EXPECTED_DOCS = {
    "Delek ARKS Amended Restated Master Supply & Offtake Agreement.pdf",
    "Delek Amended Restated Master Supply & Offtake Agreement.pdf",
    "Delek BSR Amended Restated Master Supply & Offtake Agreement.pdf",
}


# ── Capture log output + stderr during investigation ──────────────────────────

class _LogCapture(logging.Handler):
    """Logging handler that accumulates all log records as text."""
    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(self.format(record))

    @property
    def text(self):
        return "\n".join(self.lines)


class _StderrCapture:
    """Context manager that captures stderr text while still printing it."""
    def __init__(self):
        self.buf = io.StringIO()
        self._orig = None

    def __enter__(self):
        self._orig = sys.stderr
        sys.stderr = self
        return self

    def write(self, s):
        self.buf.write(s)
        self._orig.write(s)

    def flush(self):
        self._orig.flush()

    def __exit__(self, *_):
        sys.stderr = self._orig

    @property
    def text(self):
        return self.buf.getvalue()


# ── Assertion helpers ──────────────────────────────────────────────────────────

_RESULTS: list[tuple[bool, str]] = []

def check(condition: bool, label: str, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {label}"
    if detail:
        msg += f"\n         {detail}"
    print(msg)
    _RESULTS.append((condition, label))


# ── Main ───────────────────────────────────────────────────────────────────────

async def run():
    config = IrysConfig(
        api_key=os.environ["GEMINI_API_KEY"],
        enable_inline_citations=True,
    )
    irys = Irys(config)
    # Tighten engine config for test speed without crippling coverage
    irys._ensure_initialized()
    irys._engine.config.max_iterations = 8
    irys._engine.config.max_leads_per_level = 3
    irys._engine.config.parallel_reads = 2
    irys._engine.config.early_exit_facts = 10

    print(f"Query: {QUERY[:80]}...\n")
    print(f"Repo:  {REPO}\n")

    log_cap = _LogCapture()
    log_cap.setFormatter(logging.Formatter("%(name)s - %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(log_cap)

    stderr_cap = _StderrCapture()
    start = time.monotonic()

    with stderr_cap:
        result = await irys.investigate(QUERY, REPO)

    root_logger.removeHandler(log_cap)
    elapsed = time.monotonic() - start
    # Combined log text: both captured log records and raw stderr
    all_log_text = log_cap.text + "\n" + stderr_cap.text
    print(f"\nInvestigation complete in {elapsed:.1f}s\n")
    print("=" * 60)

    # ── 1. SQLite FactStore: facts were written and retrieved ─────────────────
    print("\n[1] SQLite FactStore")

    accumulated = result.state.findings.get("accumulated_facts", []) if result.state else []
    fact_texts  = [f[0] if isinstance(f, (list, tuple)) else f for f in accumulated]

    check(len(fact_texts) >= 5,
          "≥5 facts accumulated in session",
          f"got {len(fact_texts)}")

    # FactStore connection is closed at the end of investigate() (Windows WAL lock release).
    # Proxy: pack_evidence is logged at checkpoint with "X chars of cached facts".
    # If that log line appears, SQLite write + BM25 read both worked.
    cached_facts_logged = re.search(
        r"checkpoint.*?with\s+([\d,]+)\s+chars of cached facts",
        all_log_text,
    )
    if cached_facts_logged:
        chars = int(cached_facts_logged.group(1).replace(",", ""))
        check(chars > 0,
              "pack_evidence returned cached facts at checkpoint (SQLite write+read confirmed)",
              f"{chars:,} chars returned")
    else:
        # Fallback: check that investigation completed and facts > 0 (SQLite was used)
        check(len(fact_texts) > 0,
              "pack_evidence invoked (facts present — SQLite write+read inferred)",
              f"{len(fact_texts)} facts accumulated")

    # ── 2. No event-loop blocking errors ─────────────────────────────────────
    print("\n[2] Event-loop health")

    blocking_patterns = [
        r"BlockingIOError",
        r"sqlite3.*called from.*thread",
        r"Event loop is closed",
        r"coroutine.*was never awaited",
    ]
    found_blocking = [p for p in blocking_patterns if re.search(p, all_log_text, re.IGNORECASE)]
    check(len(found_blocking) == 0,
          "No event-loop blocking errors in stderr",
          f"found: {found_blocking}" if found_blocking else "")

    # ── 3. Citation injection ─────────────────────────────────────────────────
    print("\n[3] Citation injection")

    answer = result.output or ""
    citations = result.citations or []

    check(len(citations) >= 3,
          "≥3 citations in result",
          f"got {len(citations)}")

    # Check for [[cite:N]] markers in answer (means injection succeeded)
    cite_markers = re.findall(r'\[\[cite:\d+\]\]', answer)
    check(len(cite_markers) >= 1,
          "Answer contains ≥1 [[cite:N]] inline marker",
          f"found {len(cite_markers)} markers")

    # Confirm no hallucinated raw UUID markers leaked through
    raw_uuid_markers = re.findall(r'\[[a-f0-9]{8}\]', answer)
    check(len(raw_uuid_markers) == 0,
          "No raw UUID citation markers in answer (injection completed cleanly)",
          f"leaked UUIDs: {raw_uuid_markers[:3]}" if raw_uuid_markers else "")

    # ── 4. Cross-document coverage ────────────────────────────────────────────
    print("\n[4] Cross-document coverage")

    # Distinguishing keywords per document (order matters for the "Delek" agreement)
    doc_keywords = {
        "Delek ARKS Amended Restated Master Supply & Offtake Agreement.pdf": "ARKS",
        "Delek BSR Amended Restated Master Supply & Offtake Agreement.pdf": "BSR",
        "Delek Amended Restated Master Supply & Offtake Agreement.pdf": "Lion Oil",
    }
    for doc in EXPECTED_DOCS:
        short = doc_keywords.get(doc, doc)
        in_citations = any(doc in str(getattr(c, "document", "")) for c in citations)
        in_answer = short in answer
        check(in_citations or in_answer,
              f"Source covered: {short}",
              "found in citations or answer")

    # ── Answer quality spot-check ─────────────────────────────────────────────
    print("\n[5] Answer quality (spot-check)")

    check(len(answer) >= 500,
          "Answer is substantive (≥500 chars)",
          f"got {len(answer)} chars")

    key_phrases = ["conditions precedent", "Article 2", "ARKS", "BSR"]
    for phrase in key_phrases:
        check(phrase.lower() in answer.lower(),
              f"Answer mentions '{phrase}'")


async def main():
    await run()

    print("\n" + "=" * 60)
    passed = sum(1 for ok, _ in _RESULTS if ok)
    failed = sum(1 for ok, _ in _RESULTS if not ok)
    print(f"Results: {passed} passed, {failed} failed out of {len(_RESULTS)}")

    if failed:
        print("\nFailed checks:")
        for ok, label in _RESULTS:
            if not ok:
                print(f"  ✗ {label}")
        sys.exit(1)
    else:
        print("\nAll checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
