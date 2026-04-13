"""Test script: download files from URLs and extract text, saving each to a .txt file.

Tests the S3Repository download pipeline and DocumentReader text extraction
end-to-end without invoking the investigation engine.

USAGE:
    python test_extraction.py
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Configure logging to stdout so progress is visible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("test_extraction")

# ── Files to test ────────────────────────────────────────────────────────────
TEST_FILES = [
    {
        "url": "https://iqidis-artifact.s3.us-east-1.amazonaws.com/default/preview/af/cb/d02e2e7120dfeedb137f4daffd2656b39e92383167fd163e98e2ae03831b",
        "name": "paper1.pdf",
        "mime": "application/pdf",
    },
    {
        "url": "https://iqidis-artifact.s3.us-east-1.amazonaws.com/default/preview/6f/dc/30f6d23b37ee4d2c6a1c32aa0af60fa4ca37f45d86a9ae3c4b105286f1e6",
        "name": "paper 2.pdf",
        "mime": "application/pdf",
    },
    {
        "url": "https://iqidis-uploads-production.s3.us-east-1.amazonaws.com/uploads/cc697e13-8cb5-411d-b859-caf474a377fb.pdf",
        "name": "paper 3.pdf",
        "mime": "application/pdf",
    },
    {
        "url": "https://iqidis-artifact.s3.us-east-1.amazonaws.com/default/preview/94/bf/11de44d4856dff0120adcc0c5ba9dbc947329d8e1c6ba4f4ffc913a7276d",
        "name": "Screenshot 2025-10-07 172516.png",
        "mime": "image/png",
    },
    {
        "url": "https://iqidis-artifact.s3.us-east-1.amazonaws.com/default/preview/f0/e0/57fa26ce61de53db5980deb2df6455f5965f4d0954ffb7c0cefad432a985",
        "name": "Handwritten Nurse Notes.pdf",
        "mime": "application/pdf",
    },
    {
        "url": "https://iqidis-artifact.s3.us-east-1.amazonaws.com/default/preview/17/90/fa2aae1a3f7aab43d9991790bad60adf175a0c95a8b044debbe2a5714658",
        "name": "pdf_scanned_ocr.docx",
        "mime": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    },
    {
        "url": "https://iqidis-artifact.s3.us-east-1.amazonaws.com/default/preview/56/db/c38ad88037dbbab1d78406e23a0d6b82e30722fa7552cf2f933f4db7e03a",
        "name": "epa_sample_letter_sent_to_commissioners_dated_february_29_2015.pdf",
        "mime": "application/pdf",
    },
    {
        "url": "https://iqidis-artifact.s3.us-east-1.amazonaws.com/default/preview/3a/83/4199bff573dcad3b836c468966a52c557f34c9f663d71b4e1d646e9884c1",
        "name": "ASSERTION_ARCHITECTURE_SUMMARY.md",
        "mime": "text/markdown",
    },
]
# ─────────────────────────────────────────────────────────────────────────────


def make_output_dir() -> Path:
    """Create a timestamped output folder for this test run."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(f"test_extraction_output/{ts}")
    out.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output folder: {out}")
    return out


def print_separator(label: str = "") -> None:
    width = 70
    if label:
        pad = (width - len(label) - 2) // 2
        print(f"\n{'─' * pad} {label} {'─' * pad}")
    else:
        print("─" * width)


async def run() -> None:
    from src.irys.service.config import ServiceConfig
    from src.irys.service.s3_repository import S3Repository
    from src.irys.core.reader import DocumentReader

    # Minimal config — only needs AWS creds + temp dir. No Gemini key required.
    config = ServiceConfig.from_env()
    config.temp_dir = "test_extraction_output/tmp"
    Path(config.temp_dir).mkdir(parents=True, exist_ok=True)

    out_dir = make_output_dir()
    reader = DocumentReader()

    # ── Step 1: Download all files ────────────────────────────────────────────
    print_separator("DOWNLOAD")
    logger.info(f"Downloading {len(TEST_FILES)} file(s)...")

    s3_repo = S3Repository(
        bucket=config.s3_bucket or "placeholder",
        prefix="",
        config=config,
    )

    job_id = f"extraction_test_{datetime.now().strftime('%H%M%S')}"
    temp_dir = await s3_repo.download_urls_to_temp(job_id, TEST_FILES)
    logger.info(f"Downloaded to: {temp_dir}")

    downloaded = sorted(f for f in temp_dir.glob("*") if f.is_file() and f.name != "_filename_mapping.json")
    logger.info(f"Files on disk: {[f.name for f in downloaded]}")

    # ── Step 2: Extract text from each file ──────────────────────────────────
    print_separator("EXTRACTION")

    results = []
    for file_path in downloaded:
        print_separator(file_path.name)
        logger.info(f"Extracting: {file_path.name}  ({file_path.stat().st_size / 1024:.1f} KB)")

        try:
            # read_async handles all formats:
            #   • images  → straight to Mistral OCR
            #   • PDF/DOCX → normal extraction, then multimodal detection, OCR if needed
            #   • TXT/MHT  → normal sync extraction, no OCR
            doc, ocr_meta = await reader.read_async(file_path)
            text = doc.full_text

            ocr_note = ""
            if ocr_meta is not None:
                if ocr_meta.timed_out:
                    ocr_note = "  [OCR: timed out]"
                else:
                    ocr_note = f"  [OCR: {ocr_meta.page_count} page(s), {ocr_meta.latency_ms}ms]"

            logger.info(f"  type={doc.file_type}  pages={doc.page_count}  chars={doc.total_chars:,}{ocr_note}")

            # Save to output txt file (sanitize filename)
            safe_name = "".join(c if c.isalnum() or c in " ._-" else "_" for c in file_path.stem)
            out_path = out_dir / f"{safe_name}.txt"
            out_path.write_text(text, encoding="utf-8")
            logger.info(f"  ✓ Saved extracted text → {out_path}")

            # Print first 300 chars as preview
            preview = text[:300].replace("\n", " ").strip()
            print(f"\n  Preview: {preview}{'...' if len(text) > 300 else ''}\n")

            results.append({
                "file": file_path.name,
                "status": "ok",
                "chars": doc.total_chars,
                "pages": doc.page_count,
                "out": str(out_path),
                "ocr": ocr_note.strip() if ocr_note else None,
            })

        except Exception as e:
            logger.error(f"  ✗ Extraction failed: {e}")
            results.append({"file": file_path.name, "status": "error", "error": str(e)})

    # ── Step 3: Summary ───────────────────────────────────────────────────────
    print_separator("SUMMARY")
    ok = [r for r in results if r["status"] == "ok"]
    err = [r for r in results if r["status"] == "error"]
    print(f"\n  Downloaded : {len(downloaded)} file(s)")
    print(f"  Extracted  : {len(ok)} succeeded, {len(err)} failed")
    for r in ok:
        ocr_tag = f"  {r['ocr']}" if r.get("ocr") else ""
        print(f"    ✓ {r['file']:40s}  {r['chars']:>8,} chars  {r['pages']} page(s)  → {r['out']}{ocr_tag}")
    for r in err:
        print(f"    ✗ {r['file']:40s}  ERROR: {r['error']}")
    print(f"\n  Output dir : {out_dir.resolve()}\n")

    # Cleanup temp download dir
    import shutil
    shutil.rmtree(temp_dir)
    logger.info("Cleaned up temp dir")


if __name__ == "__main__":
    asyncio.run(run())
