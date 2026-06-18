"""Document reader for PDF, DOCX, DOC, MHT, Markdown, and image files.

Extracts text with page/section preservation for citation tracking.
Image files and scanned PDFs/DOCX files are processed via Mistral OCR.
DOC (legacy binary) files are extracted via antiword when available.
"""

import asyncio
import base64
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import re
import email
from email import policy
from html.parser import HTMLParser
import csv
import io
import openpyxl

import fitz  # PyMuPDF
import httpx
from docx import Document

from .multimodal_detection import detect_multimodal_content, MultimodalDetectionConfig

logger = logging.getLogger(__name__)

# Mistral OCR endpoint
_MISTRAL_OCR_URL = "https://api.mistral.ai/v1/ocr"
_OCR_MODEL = "mistral-ocr-latest"
_OCR_TIMEOUT_SECONDS = 60  # configurable default
# Files larger than this are skipped for OCR to avoid 413 Payload Too Large.
# Base64 encoding adds ~33% overhead, so 35 MB raw → ~47 MB payload.
_MISTRAL_MAX_FILE_BYTES = 35 * 1024 * 1024  # 35 MB

# MIME types for image extensions
_IMAGE_MIME: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


def _get_mistral_api_key() -> Optional[str]:
    return os.environ.get("MISTRAL_API_KEY")


def _pdf_ocr_enabled() -> bool:
    """Return True (default) unless IRYS_PDF_OCR_ENABLED=false/0/no/off.

    Controls whether multimodal detection + OCR fallback runs for PDF and DOCX.
    Images are *always* sent through OCR regardless of this flag — there is no
    alternative text-extraction path for them.
    """
    val = os.environ.get("IRYS_PDF_OCR_ENABLED", "true").strip().lower()
    return val not in {"false", "0", "no", "off"}


class HTMLTextExtractor(HTMLParser):
    """Extract text from HTML, stripping tags."""

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.skip_tags = {'script', 'style', 'head', 'meta', 'link'}
        self.current_skip = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self.skip_tags:
            self.current_skip += 1
        # Add spacing for block elements
        if tag.lower() in {'p', 'div', 'br', 'li', 'tr', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}:
            self.text_parts.append('\n')

    def handle_endtag(self, tag):
        if tag.lower() in self.skip_tags:
            self.current_skip = max(0, self.current_skip - 1)
        if tag.lower() in {'p', 'div', 'li', 'tr', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}:
            self.text_parts.append('\n')

    def handle_data(self, data):
        if self.current_skip == 0:
            self.text_parts.append(data)

    def get_text(self) -> str:
        return ''.join(self.text_parts)


@dataclass
class PageContent:
    """Content from a single page."""
    page_num: int
    text: str


@dataclass
class DocumentContent:
    """Full document content with metadata."""
    path: str
    filename: str
    file_type: str
    page_count: int
    pages: list[PageContent]
    total_chars: int

    @property
    def full_text(self) -> str:
        """Get full document text with page markers."""
        parts = []
        for page in self.pages:
            parts.append(f"\n--- PAGE {page.page_num} ---\n")
            parts.append(page.text)
        return "".join(parts)

    def get_page_range(self, start: int, end: int) -> str:
        """Get text from a page range (1-indexed)."""
        parts = []
        for page in self.pages:
            if start <= page.page_num <= end:
                parts.append(f"\n--- PAGE {page.page_num} ---\n")
                parts.append(page.text)
        return "".join(parts)

    def get_excerpt(self, max_chars: int = 5000) -> str:
        """Get excerpt of document up to max_chars."""
        text = self.full_text
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + f"\n\n[...truncated, {self.total_chars - max_chars} more chars...]"


@dataclass
class OcrCallMetadata:
    """Metadata produced by a Mistral OCR call, forwarded to telemetry."""
    latency_ms: int
    page_count: int
    timed_out: bool
    file_type: str   # "png" | "jpg" | "jpeg" | "pdf" | "docx"
    file_name: str


class DocumentReader:
    """Read and extract text from PDF, DOCX, DOC, TXT, MHT, and image files.

    Image files (PNG, JPEG) go straight to Mistral OCR.
    PDF/DOCX files are read normally first; if multimodal detection
    determines OCR is needed, Mistral OCR is called as a fallback.
    DOC (legacy binary) files are extracted via antiword (must be installed
    on the system).  If antiword is not available, .doc files are skipped
    with a clear error message.

    Note: .rtf format is NOT supported.
    Convert to .docx or .pdf before processing.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".mht", ".mhtml", ".png", ".jpg", ".jpeg",".csv", ".xlsx"}

    @staticmethod
    def _detect_type_from_magic(path: Path) -> str:
        """Detect file type from magic bytes for extensionless files.

        Returns extension string (e.g., '.pdf') or empty string.
        """
        try:
            with open(path, "rb") as f:
                header = f.read(16)
        except Exception:
            return ""

        if header.startswith(b'%PDF'):
            return '.pdf'
        if header.startswith(b'PK\x03\x04'):
            return '.docx'
        if header.startswith(b'\xd0\xcf\x11\xe0'):
            return '.doc'
        if header.startswith(b'{\\rtf'):
            return '.rtf'
        if header.startswith(b'\x89PNG'):
            return '.png'
        if header.startswith(b'\xff\xd8\xff'):
            return '.jpg'
        try:
            with open(path, "rb") as f:
                sample = f.read(1000)
            sample.decode('utf-8')
            return '.txt'
        except (UnicodeDecodeError, Exception):
            pass
        return ''

    def read(self, path: Path | str) -> DocumentContent:
        """Read a document and extract text."""
        path = Path(path)

        if not path.exists():
            # Provide detailed diagnostic info
            parent = path.parent
            parent_exists = parent.exists()
            siblings = list(parent.glob("*"))[:5] if parent_exists else []
            sibling_names = [s.name for s in siblings]

            raise FileNotFoundError(
                f"Document not found: {path}. "
                f"Parent dir exists: {parent_exists}. "
                f"Sample files in parent: {sibling_names if sibling_names else 'none/empty'}"
            )

        suffix = path.suffix.lower()

        # For extensionless files (e.g., hash-named), detect type from magic bytes
        if not suffix:
            suffix = self._detect_type_from_magic(path)

        if suffix == ".pdf":
            return self._read_pdf(path)
        elif suffix == ".docx":
            return self._read_docx(path)
        elif suffix == ".doc":
            return self._read_doc(path)
        elif suffix == ".txt":
            return self._read_txt(path)
        elif suffix == ".md":
            return self._read_md(path)
        elif suffix in {".mht", ".mhtml"}:
            return self._read_mht(path)
        elif suffix == ".csv":
            return self._read_csv(path)
        elif suffix == ".xlsx":
            return self._read_xlsx(path)
        elif suffix in {".png", ".jpg", ".jpeg"}:
            raise ValueError(
                f"Image files must be read via read_async(): {path.name}. "
                f"Use DocumentReader.read_async() or MatterRepository.read_async()."
            )
        elif suffix == ".rtf":
            raise ValueError(
                f"Unsupported legacy format: {suffix}. "
                f"Please convert to .docx or .pdf first."
            )
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def _read_pdf(self, path: Path) -> DocumentContent:
        """Extract text from PDF with page structure."""
        doc = fitz.open(path)
        pages = []
        total_chars = 0

        try:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text()
                text = self._clean_text(text)
                pages.append(PageContent(page_num=page_num, text=text))
                total_chars += len(text)
        finally:
            doc.close()

        return DocumentContent(
            path=str(path),
            filename=path.name,
            file_type="pdf",
            page_count=len(pages),
            pages=pages,
            total_chars=total_chars,
        )

    def _read_docx(self, path: Path) -> DocumentContent:
        """Extract text from DOCX."""
        doc = Document(path)

        # DOCX doesn't have real pages, treat as single page
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n\n".join(paragraphs)
        text = self._clean_text(text)

        # Also extract tables
        for table in doc.tables:
            table_text = []
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells)
                table_text.append(row_text)
            text += "\n\n[TABLE]\n" + "\n".join(table_text) + "\n[/TABLE]\n"

        pages = [PageContent(page_num=1, text=text)]

        return DocumentContent(
            path=str(path),
            filename=path.name,
            file_type="docx",
            page_count=1,
            pages=pages,
            total_chars=len(text),
        )

    # ------------------------------------------------------------------
    # Legacy .doc support via antiword
    # ------------------------------------------------------------------

    @staticmethod
    def _find_antiword() -> Optional[str]:
        """Locate the antiword binary on the system.

        Checks:
        1. PATH via shutil.which (works on Linux, macOS, Windows)
        2. Common Linux package paths
        3. ANTIWORD_PATH environment variable (user override)

        Returns the full path to the binary, or None if not found.
        """
        # User override
        env_path = os.environ.get("ANTIWORD_PATH")
        if env_path and os.path.isfile(env_path):
            return env_path

        # Standard PATH lookup
        found = shutil.which("antiword")
        if found:
            return found

        # Common Linux install locations
        for candidate in ["/usr/bin/antiword", "/usr/local/bin/antiword"]:
            if os.path.isfile(candidate):
                return candidate

        return None

    def _read_doc(self, path: Path) -> DocumentContent:
        """Extract text from legacy .doc (binary Word) files via antiword.

        antiword is a well-tested C utility that reliably extracts text
        from Microsoft Word binary (.doc) files.  It must be installed
        on the system (e.g. ``apt install antiword`` on Debian/Ubuntu).

        If antiword is not available, raises ValueError with install
        instructions so the application can still run — .doc files are
        simply skipped.
        """
        # User-friendly message for all .doc read failures
        _user_msg = f"Could not read {path.name} — file may be corrupted or unsupported. Continuing with other documents."

        antiword_cmd = self._find_antiword()
        if antiword_cmd is None:
            logger.warning(
                "antiword is not installed — cannot read .doc files. "
                "Install with: sudo apt install antiword (Linux) / brew install antiword (macOS), "
                "or set ANTIWORD_PATH env var."
            )
            raise ValueError(_user_msg)

        try:
            result = subprocess.run(
                [antiword_cmd, "-w", "0", str(path)],  # -w 0 = no line wrapping
                capture_output=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            logger.warning("antiword timed out (30s) reading %s", path.name)
            raise ValueError(_user_msg)
        except OSError as exc:
            logger.warning("Failed to run antiword for %s: %s", path.name, exc)
            raise ValueError(_user_msg)

        if result.returncode != 0:
            stderr_msg = result.stderr.decode("utf-8", errors="replace").strip()
            logger.warning(
                "antiword failed on %s (exit code %d): %s",
                path.name, result.returncode, stderr_msg or "unknown error",
            )
            raise ValueError(_user_msg)

        # antiword outputs UTF-8 text by default
        try:
            text = result.stdout.decode("utf-8")
        except UnicodeDecodeError:
            # Fallback: try latin-1 which never fails
            text = result.stdout.decode("latin-1")

        text = self._clean_text(text)

        if not text.strip():
            logger.warning(
                "antiword returned empty text for %s — file may be "
                "image-only or password-protected",
                path.name,
            )

        pages = [PageContent(page_num=1, text=text)]

        return DocumentContent(
            path=str(path),
            filename=path.name,
            file_type="doc",
            page_count=1,
            pages=pages,
            total_chars=len(text),
        )

    def _read_txt(self, path: Path) -> DocumentContent:
        """Read plain text file."""
        text = path.read_text(encoding="utf-8", errors="replace")
        text = self._clean_text(text)

        pages = [PageContent(page_num=1, text=text)]

        return DocumentContent(
            path=str(path),
            filename=path.name,
            file_type="txt",
            page_count=1,
            pages=pages,
            total_chars=len(text),
        )
        
    def _read_csv(self, path: Path) -> DocumentContent:
        """Extract rows from CSV files, formatting rows into structured text for LLM attention."""


        try:
            # Read bytes safely and substitute anomalies using replace
            raw_bytes = path.read_bytes()
            text_data = raw_bytes.decode('utf-8', errors='replace')
            text_stream = io.StringIO(text_data)
            reader = csv.reader(text_stream)
            
            headers = next(reader, None)
            if headers:
                headers = [h.strip() for h in headers if h.strip()]
                output = [f"CSV Structure Layout: Contains columns [{', '.join(headers)}]\n"]
                
                for idx, row in enumerate(reader, start=1):
                    # Combine row positions into key-value strings for semantic retrieval
                    row_str = ", ".join(f"{hdr}: {val.strip()}" for hdr, val in zip(headers, row) if val.strip())
                    if row_str:
                        output.append(f"Row {idx}: {row_str}")
                
                text = "\n".join(output)
            else:
                text = "Empty CSV document."
        except Exception as e:
            logger.error("Failed parsing CSV resource %s: %s", path.name, e)
            text = f"Error extracting structural data rows from CSV matrix: {e}"

        # Clean spaces and unify structural layouts using existing pipeline filters
        text = self._clean_text(text)
        pages = [PageContent(page_num=1, text=text)]

        return DocumentContent(
            path=str(path),
            filename=path.name,
            file_type="csv",  # Passes metadata context down to all engine leads
            page_count=1,
            pages=pages,
            total_chars=len(text),
        )
        
    def _read_xlsx(self, path: Path) -> DocumentContent:
        """Extract rows from Excel workbook sheets (.xlsx), mapping cells explicitly to headers."""
        

        try:
            # data_only=True ensures we extract calculated string/numeric values, not raw formulas
            wb = openpyxl.load_workbook(path, data_only=True)
            output = []

            for sheet in wb.worksheets:
                output.append(f"--- EXCEL WORKSHEET: {sheet.title} ---")
                rows = list(sheet.iter_rows(values_only=True))
                
                # Filter out empty spreadsheets
                if not rows or all(all(cell is None for cell in r) for r in rows):
                    output.append("Empty sheet content.")
                    continue

                # Isolate the first row that actually contains data to establish headers
                headers = None
                header_row_index = 0
                for idx, r in enumerate(rows):
                    if any(cell is not None for cell in r):
                        headers = [str(cell).strip() for cell in r if cell is not None]
                        header_row_index = idx
                        break

                if not headers:
                    headers = [f"Column_{i}" for i in range(len(rows[0]))]

                output.append(f"Columns defined: [{', '.join(headers)}]")
                
                virtual_row_idx = 1
                for r in rows[header_row_index + 1:]:
                    row_parts = []
                    for hdr, cell in zip(headers, r):
                        if cell is not None and str(cell).strip():
                            row_parts.append(f"{hdr}: {str(cell).strip()}")
                    
                    if row_parts:
                        output.append(f"Row {virtual_row_idx}: {', '.join(row_parts)}")
                        virtual_row_idx += 1
                
                output.append("\n") # Section break between different sheets
                
            text = "\n".join(output)
        except Exception as e:
            logger.error("Failed parsing XLSX file %s: %s", path.name, e)
            text = f"Error extracting tabular data from Excel workbook: {e}"

        text = self._clean_text(text)
        pages = [PageContent(page_num=1, text=text)]

        return DocumentContent(
            path=str(path), filename=path.name, file_type="xlsx",
            page_count=1, pages=pages, total_chars=len(text),
        )

    def _read_md(self, path: Path) -> DocumentContent:
        """Read Markdown file as plain text (no rendering, structure preserved)."""
        text = path.read_text(encoding="utf-8", errors="replace")
        text = self._clean_text(text)

        pages = [PageContent(page_num=1, text=text)]

        return DocumentContent(
            path=str(path),
            filename=path.name,
            file_type="md",
            page_count=1,
            pages=pages,
            total_chars=len(text),
        )

    def _read_mht(self, path: Path) -> DocumentContent:
        """Read MIME HTML (MHT/MHTML) file and extract text.

        MHT files are MIME-encoded web archives that contain HTML content.
        We extract the HTML and convert it to plain text.
        """
        # Read the raw content
        raw_content = path.read_bytes()

        # Try to parse as MIME message
        try:
            msg = email.message_from_bytes(raw_content, policy=policy.default)
            html_content = None

            # Walk through MIME parts to find HTML
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    if content_type == 'text/html':
                        payload = part.get_payload(decode=True)
                        if payload:
                            # Try different encodings
                            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                                try:
                                    html_content = payload.decode(encoding)
                                    break
                                except UnicodeDecodeError:
                                    continue
                        break
            else:
                # Single-part message
                payload = msg.get_payload(decode=True)
                if payload:
                    for encoding in ['utf-8', 'latin-1', 'cp1252']:
                        try:
                            html_content = payload.decode(encoding)
                            break
                        except UnicodeDecodeError:
                            continue

            # Fallback: try reading as raw text with HTML
            if not html_content:
                raw_text = raw_content.decode('utf-8', errors='replace')
                # Look for HTML content between markers
                if '<html' in raw_text.lower():
                    html_content = raw_text
        except Exception:
            # Final fallback: read as text and extract what we can
            html_content = path.read_text(encoding='utf-8', errors='replace')

        # Extract text from HTML
        if html_content:
            # Handle quoted-printable encoding artifacts
            html_content = html_content.replace('=\n', '')  # Line continuations
            html_content = re.sub(r'=([0-9A-Fa-f]{2})',
                                  lambda m: chr(int(m.group(1), 16)),
                                  html_content)

            # Use regex-based extraction (more robust than HTMLParser for MHT files)
            # Remove script and style content
            text = re.sub(r'<script[^>]*>.*?</script>', '', html_content,
                          flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text,
                          flags=re.DOTALL | re.IGNORECASE)
            # Strip remaining tags
            text = re.sub(r'<[^>]+>', ' ', text)
        else:
            text = ""

        text = self._clean_text(text)
        pages = [PageContent(page_num=1, text=text)]

        return DocumentContent(
            path=str(path),
            filename=path.name,
            file_type="mht",
            page_count=1,
            pages=pages,
            total_chars=len(text),
        )

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Normalize horizontal whitespace (preserve newlines)
        text = re.sub(r'[^\S\n]+', ' ', text)
        # Clean up spaces around newlines
        text = re.sub(r' ?\n ?', '\n', text)
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def get_page_count(self, path: Path | str) -> int:
        """Get page count without reading full content."""
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            doc = fitz.open(path)
            count = len(doc)
            doc.close()
            return count
        else:
            return 1  # Non-paginated formats

    def can_read(self, path: Path | str) -> bool:
        """Check if file type is supported."""
        path = Path(path)
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    # ------------------------------------------------------------------
    # Async OCR methods
    # ------------------------------------------------------------------

    async def read_async(
        self,
        path: Path | str,
        ocr_timeout: float = _OCR_TIMEOUT_SECONDS,
    ) -> tuple["DocumentContent", Optional["OcrCallMetadata"]]:
        """Async version of read() — required for image files and OCR fallback.

        Returns:
            (DocumentContent, OcrCallMetadata | None)
            OcrCallMetadata is non-None only when a Mistral OCR call was made.
        """
        path = Path(path)

        if not path.exists():
            parent = path.parent
            parent_exists = parent.exists()
            siblings = [s.name for s in list(parent.glob("*"))[:5]] if parent_exists else []
            raise FileNotFoundError(
                f"Document not found: {path}. "
                f"Parent dir exists: {parent_exists}. "
                f"Sample files in parent: {siblings if siblings else 'none/empty'}"
            )

        suffix = path.suffix.lower()
        if not suffix:
            suffix = self._detect_type_from_magic(path)

        # --- Image: straight to Mistral OCR ---
        if suffix in {".png", ".jpg", ".jpeg"}:
            return await self._read_image_via_ocr(path, ocr_timeout)

        # --- PDF: normal extraction → multimodal check → OCR (if enabled) ---
        if suffix == ".pdf":
            doc_content, rendered_pages, pdf_info = await asyncio.to_thread(
                self._read_pdf_with_meta, path
            )

            if _pdf_ocr_enabled():
                detection = detect_multimodal_content(
                    text=doc_content.full_text,
                    page_count=doc_content.page_count,
                    rendered_pages=rendered_pages,
                    pdf_info=pdf_info,
                )
                if detection.requires_multimodal:
                    logger.info(
                        "PDF multimodal OCR triggered for %s (confidence=%.2f): %s",
                        path.name,
                        detection.confidence,
                        detection.reasons[:2],
                    )
                    raw_bytes = path.read_bytes()
                    ocr_doc, ocr_meta = await self._call_mistral_ocr(
                        raw_bytes, "application/pdf", "document_url",
                        path, suffix.lstrip("."), ocr_timeout,
                    )
                    # Fall back to original fitz extraction if OCR returned nothing
                    if ocr_doc.total_chars > 0:
                        return ocr_doc, ocr_meta
                    logger.warning(
                        "OCR returned empty content for %s — using original extraction (%d chars)",
                        path.name, doc_content.total_chars,
                    )

            return doc_content, None

        # --- DOCX: normal extraction → multimodal check → OCR (if enabled) ---
        if suffix == ".docx":
            doc_content = await asyncio.to_thread(self._read_docx, path)

            if _pdf_ocr_enabled():
                detection = detect_multimodal_content(
                    text=doc_content.full_text,
                    page_count=1,
                    rendered_pages=1 if doc_content.total_chars > 50 else 0,
                )
                if detection.requires_multimodal:
                    logger.info(
                        "DOCX multimodal OCR triggered for %s (confidence=%.2f)",
                        path.name, detection.confidence,
                    )
                    raw_bytes = path.read_bytes()
                    ocr_doc, ocr_meta = await self._call_mistral_ocr(
                        raw_bytes,
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        "document_url",
                        path, "docx", ocr_timeout,
                    )
                    # Fall back to original docx extraction if OCR returned nothing
                    if ocr_doc.total_chars > 0:
                        return ocr_doc, ocr_meta
                    logger.warning(
                        "OCR returned empty content for %s — using original extraction (%d chars)",
                        path.name, doc_content.total_chars,
                    )

            return doc_content, None

        # --- All other supported formats: use sync read, no OCR ---
        return self.read(path), None

    async def _read_image_via_ocr(
        self,
        path: Path,
        ocr_timeout: float = _OCR_TIMEOUT_SECONDS,
    ) -> tuple["DocumentContent", "OcrCallMetadata"]:
        """Read an image file by sending it to Mistral OCR as an image_url."""
        suffix = path.suffix.lower()
        mime = _IMAGE_MIME.get(suffix, "image/jpeg")
        raw_bytes = path.read_bytes()
        return await self._call_mistral_ocr(
            raw_bytes, mime, "image_url", path, suffix.lstrip("."), ocr_timeout
        )

    async def _call_mistral_ocr(
        self,
        data: bytes,
        mime: str,
        doc_type: str,   # "document_url" for PDF/DOCX, "image_url" for images
        path: Path,
        file_type: str,
        ocr_timeout: float = _OCR_TIMEOUT_SECONDS,
    ) -> tuple["DocumentContent", "OcrCallMetadata"]:
        """Shared Mistral OCR helper.

        Wraps the blocking HTTP call in asyncio.to_thread + asyncio.wait_for.
        On timeout or API error, returns empty DocumentContent with timed_out=True,
        and logs a warning — never raises.
        """
        api_key = _get_mistral_api_key()
        if not api_key:
            logger.warning("MISTRAL_API_KEY not set — returning empty content for %s", path.name)
            return self._empty_doc(path, file_type), OcrCallMetadata(
                latency_ms=0, page_count=0, timed_out=False, file_type=file_type, file_name=path.name
            )

        b64 = base64.b64encode(data).decode("ascii")
        # Build data URI; image uses image_url key, document uses document_url key
        data_uri = f"data:{mime};base64,{b64}"
        payload: dict = {
            "model": _OCR_MODEL,
            "document": {
                "type": doc_type,
                doc_type: data_uri,  # key matches the type value
            },
        }

        # Skip OCR for files that would exceed Mistral's payload limit.
        # Base64 encoding adds ~33% overhead; 35 MB raw → ~47 MB payload → 413.
        if len(data) > _MISTRAL_MAX_FILE_BYTES:
            size_mb = len(data) / (1024 * 1024)
            logger.warning(
                "Skipping Mistral OCR for %s (%.1f MB > %.0f MB limit) — using original extraction",
                path.name, size_mb, _MISTRAL_MAX_FILE_BYTES / (1024 * 1024),
            )
            return self._empty_doc(path, file_type), OcrCallMetadata(
                latency_ms=0, page_count=0, timed_out=False,
                file_type=file_type, file_name=path.name,
            )

        t0 = time.monotonic()
        timed_out = False

        def _do_request() -> dict:
            with httpx.Client(timeout=ocr_timeout + 5) as client:
                resp = client.post(
                    _MISTRAL_OCR_URL,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
                return resp.json()

        # Retry transient 5xx errors (502 Bad Gateway, 503, 504) up to 2 times.
        _TRANSIENT_STATUS_CODES = {502, 503, 504}
        last_exc: Optional[Exception] = None
        for attempt in range(3):
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(_do_request),
                    timeout=ocr_timeout,
                )
                last_exc = None
                break
            except asyncio.TimeoutError as exc:
                latency_ms = int((time.monotonic() - t0) * 1000)
                logger.warning("Mistral OCR timed out after %.0fs for %s", ocr_timeout, path.name)
                return self._empty_doc(path, file_type), OcrCallMetadata(
                    latency_ms=latency_ms, page_count=0, timed_out=True,
                    file_type=file_type, file_name=path.name,
                )
            except Exception as exc:
                status = getattr(getattr(exc, 'response', None), 'status_code', None)
                if status in _TRANSIENT_STATUS_CODES and attempt < 2:
                    wait = 2 ** attempt  # 1s, 2s
                    logger.warning(
                        "Mistral OCR %d error for %s, retrying in %ds (attempt %d/3)",
                        status, path.name, wait, attempt + 1,
                    )
                    await asyncio.sleep(wait)
                    last_exc = exc
                    continue
                last_exc = exc
                break

        if last_exc is not None:
            latency_ms = int((time.monotonic() - t0) * 1000)
            logger.warning("Mistral OCR error for %s: %s", path.name, last_exc)
            return self._empty_doc(path, file_type), OcrCallMetadata(
                latency_ms=latency_ms, page_count=0, timed_out=False,
                file_type=file_type, file_name=path.name,
            )

        latency_ms = int((time.monotonic() - t0) * 1000)

        # Parse the OCR response — pages array with markdown field
        raw_pages = result.get("pages", [])
        pages: list[PageContent] = []
        total_chars = 0
        for idx, pg in enumerate(raw_pages, start=1):
            text = self._clean_text(pg.get("markdown") or pg.get("text") or "")
            pages.append(PageContent(page_num=idx, text=text))
            total_chars += len(text)

        if not pages:
            # API returned no pages — treat as empty
            logger.warning("Mistral OCR returned no pages for %s", path.name)

        doc_content = DocumentContent(
            path=str(path),
            filename=path.name,
            file_type=file_type,
            page_count=len(pages),
            pages=pages,
            total_chars=total_chars,
        )
        ocr_meta = OcrCallMetadata(
            latency_ms=latency_ms,
            page_count=len(pages),
            timed_out=False,
            file_type=file_type,
            file_name=path.name,
        )
        logger.info(
            "Mistral OCR completed: %s pages, %d chars, %dms (%s)",
            len(pages), total_chars, latency_ms, path.name,
        )
        return doc_content, ocr_meta

    def _read_pdf_with_meta(
        self, path: Path
    ) -> tuple["DocumentContent", int, dict]:
        """Read PDF and also return rendered_pages count and fitz metadata dict.

        rendered_pages = count of pages that yielded > 50 chars of text.
        """
        doc = fitz.open(path)
        pages: list[PageContent] = []
        total_chars = 0
        rendered_pages = 0
        try:
            for page_num, page in enumerate(doc, start=1):
                text = self._clean_text(page.get_text())
                pages.append(PageContent(page_num=page_num, text=text))
                total_chars += len(text)
                if len(text) > 50:
                    rendered_pages += 1
            pdf_info = doc.metadata or {}
        finally:
            doc.close()

        doc_content = DocumentContent(
            path=str(path),
            filename=path.name,
            file_type="pdf",
            page_count=len(pages),
            pages=pages,
            total_chars=total_chars,
        )
        return doc_content, rendered_pages, pdf_info

    @staticmethod
    def _empty_doc(path: Path, file_type: str) -> "DocumentContent":
        """Return an empty DocumentContent (OCR timeout / API error fallback)."""
        return DocumentContent(
            path=str(path),
            filename=path.name,
            file_type=file_type,
            page_count=0,
            pages=[],
            total_chars=0,
        )
