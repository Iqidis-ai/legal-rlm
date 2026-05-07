"""Document reader for PDF, DOCX, and Excel files.

Extracts text with page/section preservation for citation tracking.
"""

from pathlib import Path
from dataclasses import dataclass
import re

import fitz  # PyMuPDF
from docx import Document
from openpyxl import load_workbook
from openpyxl.cell.cell import MergedCell


@dataclass
class PageContent:
    """Content from a single page."""
    page_num: int
    text: str


@dataclass
class TrackedChange:
    """A single tracked change extracted from DOCX XML."""
    deleted_text: str
    added_text: str
    context: str  # surrounding paragraph text for location


@dataclass
class DocumentContent:
    """Full document content with metadata."""
    path: str
    filename: str
    file_type: str
    page_count: int
    pages: list[PageContent]
    total_chars: int
    tracked_changes: list[TrackedChange] | None = None

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

    def get_tracked_change_manifest(self) -> str:
        """Build a structured manifest of all tracked changes for LLM consumption."""
        if not self.tracked_changes:
            return ""
        lines = [
            f"TRACKED CHANGE MANIFEST — {len(self.tracked_changes)} changes extracted from {self.filename}:",
            "Each entry shows text DELETED from the original and text ADDED by the markup.",
            "You MUST create a provision_comparison entry for EVERY change below.",
            "",
        ]
        for i, tc in enumerate(self.tracked_changes, 1):
            lines.append(f"Change #{i}:")
            if tc.deleted_text:
                lines.append(f"  DELETED: \"{tc.deleted_text[:500]}\"")
            if tc.added_text:
                lines.append(f"  ADDED:   \"{tc.added_text[:500]}\"")
            if tc.context:
                lines.append(f"  CONTEXT: ...{tc.context[:250]}...")
            lines.append("")
        return "\n".join(lines)


class DocumentReader:
    """Read and extract text from PDF and DOCX files."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".xlsx", ".xls"}

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
        elif suffix in {".xlsx", ".xls"}:
            return self._read_xlsx(path)
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
                cell_texts = []
                for cell in row.cells:
                    cell_parts = []
                    for p in cell.paragraphs:
                        cell_parts.append(self._extract_paragraph_with_revisions(p))
                    cell_texts.append(" ".join(cell_parts).strip())
                table_text.append(" | ".join(cell_texts))
            text += "\n\n[TABLE]\n" + "\n".join(table_text) + "\n[/TABLE]\n"

        pages = [PageContent(page_num=1, text=text)]

        tracked = self._extract_tracked_changes(doc)

        return DocumentContent(
            path=str(path),
            filename=path.name,
            file_type="docx",
            page_count=1,
            pages=pages,
            total_chars=len(text),
            tracked_changes=tracked if tracked else None,
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

    @staticmethod
    def _extract_tracked_changes(doc: "Document") -> list[TrackedChange]:
        """Walk DOCX XML to extract every tracked change as structured data.

        Walks ALL <w:p> elements in the document body (including table cells,
        headers, footers, footnotes, endnotes) — not just doc.paragraphs which
        only covers top-level body paragraphs.

        Each <w:del>/<w:ins> revision group within a paragraph becomes its own
        TrackedChange entry (atomic, not merged per paragraph).
        """
        from lxml import etree
        W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        changes: list[TrackedChange] = []

        body = doc.element.body
        if body is None:
            return changes

        for para_elem in body.iter(f"{{{W}}}p"):
            del_runs = para_elem.findall(f".//{{{W}}}del")
            ins_runs = para_elem.findall(f".//{{{W}}}ins")
            if not del_runs and not ins_runs:
                continue

            plain_parts: list[str] = []
            for t in para_elem.findall(f".//{{{W}}}t"):
                parent = t.getparent()
                gp = parent.getparent() if parent is not None else None
                gp_tag = etree.QName(gp.tag).localname if gp is not None and isinstance(gp.tag, str) else ""
                if gp_tag in ("ins", "del"):
                    continue
                if t.text:
                    plain_parts.append(t.text)
            context = "".join(plain_parts).strip()[:200]

            rev_groups: list[tuple[str, str]] = []
            for d in del_runs:
                parts: list[str] = []
                for dt in d.findall(f".//{{{W}}}delText"):
                    if dt.text:
                        parts.append(dt.text)
                text = "".join(parts).strip()
                if text:
                    rev_groups.append(("del", text))
            for ins in ins_runs:
                parts = []
                for t in ins.findall(f".//{{{W}}}t"):
                    if t.text:
                        parts.append(t.text)
                text = "".join(parts).strip()
                if text:
                    rev_groups.append(("ins", text))

            if not rev_groups:
                continue

            del_texts = [t for kind, t in rev_groups if kind == "del"]
            ins_texts = [t for kind, t in rev_groups if kind == "ins"]

            if len(del_texts) <= 1 and len(ins_texts) <= 1:
                deleted = del_texts[0] if del_texts else ""
                added = ins_texts[0] if ins_texts else ""
                changes.append(TrackedChange(
                    deleted_text=deleted,
                    added_text=added,
                    context=context,
                ))
            else:
                max_pairs = max(len(del_texts), len(ins_texts))
                for idx in range(max_pairs):
                    deleted = del_texts[idx] if idx < len(del_texts) else ""
                    added = ins_texts[idx] if idx < len(ins_texts) else ""
                    if deleted or added:
                        changes.append(TrackedChange(
                            deleted_text=deleted,
                            added_text=added,
                            context=context,
                        ))

        return changes

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

    def _read_xlsx(self, path: Path) -> DocumentContent:
        """Extract text from Excel workbook, one sheet per page.

        Includes cell references (A1, B2, etc.) for financial data to
        preserve coordinates for reconciliation tasks.
        """
        wb = load_workbook(path, read_only=True, data_only=True)
        pages: list[PageContent] = []
        total_chars = 0

        def _col_letter(ci: int) -> str:
            result = ""
            while ci >= 0:
                result = chr(ci % 26 + ord('A')) + result
                ci = ci // 26 - 1
            return result

        for sheet_idx, sheet_name in enumerate(wb.sheetnames, start=1):
            ws = wb[sheet_name]
            rows_raw: list[list[str]] = []

            for row in ws.iter_rows():
                cell_values: list[str] = []
                for cell in row:
                    if isinstance(cell, MergedCell):
                        cell_values.append("")
                    elif cell.value is None:
                        cell_values.append("")
                    else:
                        cell_values.append(str(cell.value).strip())
                rows_raw.append(cell_values)

            # Strip trailing empty rows
            while rows_raw and all(v == "" for v in rows_raw[-1]):
                rows_raw.pop()

            if not rows_raw:
                continue

            # Determine max column width per column for alignment
            num_cols = max(len(r) for r in rows_raw) if rows_raw else 0
            # Pad short rows
            for r in rows_raw:
                while len(r) < num_cols:
                    r.append("")

            # Strip trailing empty columns
            while num_cols > 0 and all(r[num_cols - 1] == "" for r in rows_raw):
                for r in rows_raw:
                    r.pop()
                num_cols -= 1

            if num_cols == 0:
                continue

            # Compute column widths for readable formatting
            col_widths = [0] * num_cols
            for r in rows_raw:
                for ci, val in enumerate(r):
                    col_widths[ci] = max(col_widths[ci], len(val))
            col_widths = [min(w, 120) for w in col_widths]

            # Build text table with cell references
            lines: list[str] = [f"[SHEET: {sheet_name}]"]
            # Column header reference row
            col_refs = "     " + " | ".join(
                _col_letter(ci).ljust(col_widths[ci])[:col_widths[ci]]
                for ci in range(num_cols)
            )
            lines.append(col_refs)
            lines.append("-" * len(col_refs))
            for row_idx, r in enumerate(rows_raw):
                row_num = row_idx + 1
                padded = [val.ljust(col_widths[ci])[:col_widths[ci]] for ci, val in enumerate(r)]
                lines.append(f"{row_num:4d} " + " | ".join(padded))
                if row_idx == 0:
                    lines.append("---- " + "-+-".join("-" * w for w in col_widths))

            sheet_text = self._clean_text("\n".join(lines))
            pages.append(PageContent(page_num=sheet_idx, text=sheet_text))
            total_chars += len(sheet_text)

        wb.close()

        if not pages:
            # Workbook was empty — return single empty page
            pages = [PageContent(page_num=1, text="[Empty workbook]")]

        return DocumentContent(
            path=str(path),
            filename=path.name,
            file_type="xlsx",
            page_count=len(pages),
            pages=pages,
            total_chars=total_chars,
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
