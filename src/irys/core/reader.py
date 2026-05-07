"""Document reader for PDF and DOCX files.

Extracts text with page/section preservation for citation tracking.
"""

from pathlib import Path
from dataclasses import dataclass
import re

import fitz  # PyMuPDF
from docx import Document


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


class DocumentReader:
    """Read and extract text from PDF and DOCX files."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}

    def read(self, path: Path | str) -> DocumentContent:
        """Read a document and extract text."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return self._read_pdf(path)
        elif suffix in {".docx", ".doc"}:
            return self._read_docx(path)
        elif suffix == ".txt":
            return self._read_txt(path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def _read_pdf(self, path: Path) -> DocumentContent:
        """Extract text from PDF with page structure."""
        doc = fitz.open(path)
        pages = []
        total_chars = 0

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            text = self._clean_text(text)
            pages.append(PageContent(page_num=page_num, text=text))
            total_chars += len(text)

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
        """Extract text from DOCX, preserving tracked changes as [DELETED]/[ADDED] markers."""
        doc = Document(path)

        paragraphs = []
        for p in doc.paragraphs:
            para_text = self._extract_paragraph_with_revisions(p)
            if para_text.strip():
                paragraphs.append(para_text)
        text = "\n\n".join(paragraphs)
        text = self._clean_text(text)

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

    @staticmethod
    def _extract_paragraph_with_revisions(paragraph) -> str:
        """Extract paragraph text preserving tracked changes as markup markers."""
        from lxml import etree
        nsmap = {
            'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
        }
        parts = []
        try:
            for elem in paragraph._element.iter():
                tag = etree.QName(elem.tag).localname if isinstance(elem.tag, str) else ""
                if tag == "delText":
                    parts.append(f"[DELETED: {elem.text or ''}]")
                elif tag == "t":
                    parent_tag = ""
                    if elem.getparent() is not None:
                        parent_tag = etree.QName(elem.getparent().tag).localname if isinstance(elem.getparent().tag, str) else ""
                    grandparent_tag = ""
                    if elem.getparent() is not None and elem.getparent().getparent() is not None:
                        gp = elem.getparent().getparent()
                        grandparent_tag = etree.QName(gp.tag).localname if isinstance(gp.tag, str) else ""
                    if grandparent_tag == "ins":
                        parts.append(f"[ADDED: {elem.text or ''}]")
                    else:
                        parts.append(elem.text or "")
        except Exception:
            return paragraph.text or ""
        return "".join(parts) if parts else (paragraph.text or "")

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

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Normalize horizontal whitespace (preserve newlines)
        text = re.sub(r'[^\S\n]+', ' ', text)
        # Clean up spaces around newlines
        text = re.sub(r' ?\n ?', '\n', text)
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def can_read(self, path: Path | str) -> bool:
        """Check if file type is supported."""
        path = Path(path)
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS
