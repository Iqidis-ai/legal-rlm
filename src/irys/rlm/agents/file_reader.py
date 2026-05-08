"""DocumentFileReader — structural file-reading operator.

Operator Substrate Thesis: Sub-agents are bounded operators. This one
produces durable document-structure artifacts (section maps, tables,
schedules, signature status) that downstream domain operators can
consume without each rediscovering structure on their own.

Per the user's directive ("better file reading sub-agents"), this is
priority structural infrastructure: when more docs land in a corpus,
their structure is parsed once into durable artifacts instead of being
re-derived inside every extractor's prompt.

The agent is fully deterministic — pure regex + heuristic parsing on
DocumentContent.full_text. Zero LLM calls. Cross-domain (works on any
document with text content; legal/finance/coding/research/biomedical).
"""

from __future__ import annotations

import re as _re
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .contracts import (
    AgentArtifact,
    AgentInvocation,
    AgentInvocationResult,
    AgentMatch,
    AgentRequirement,
)


# ---------------------------------------------------------------------------
# Parsers — section map / tables / signatures / schedules / footnotes
# ---------------------------------------------------------------------------


# Section heading patterns. Cover legal numbering (Section 4.01, ARTICLE I,
# 1., 1.1.1), Markdown (#, ##, ###), and "Schedule X / Exhibit X" labels.
_SECTION_PATTERNS = (
    # "Section 4.01(a) — Title", "Section 4.01 Title"
    _re.compile(
        r"^\s*(Section\s+\d+(?:\.\d+)*(?:\([a-z]+\))?)\s*[—\-:.]?\s*(.{0,120})$",
        _re.IGNORECASE | _re.MULTILINE,
    ),
    # "ARTICLE I — Definitions"
    _re.compile(
        r"^\s*(ARTICLE\s+[IVXLC\d]+)\s*[—\-:.]?\s*(.{0,120})$",
        _re.MULTILINE,
    ),
    # Markdown headings "# Title", "## Title", "### Title"
    _re.compile(r"^\s*(#{1,3})\s+(.{1,200})$", _re.MULTILINE),
    # Numbered "1.", "1.1", "1.1.1"
    _re.compile(
        r"^\s*(\d+(?:\.\d+){0,3})\.?\s+([A-Z][^\n]{0,150})$",
        _re.MULTILINE,
    ),
)

_SCHEDULE_RE = _re.compile(
    r"\b(Schedule|Exhibit|Annex|Appendix)\s+"
    r"([A-Z0-9]+(?:[\.\-][A-Z0-9]+)*(?:\([a-zA-Z0-9]+\))?)"
)

_TABLE_LINE_RE = _re.compile(r"^\|.+\|.*$", _re.MULTILINE)

# Footnote body cannot itself contain a bracketed marker — that bounds
# the body so the next [n] match is reachable.
_FOOTNOTE_RE = _re.compile(r"\[\s*(\d+|\*+)\s*\]\s*([^\[\n]{0,200})")

_SIGNATURE_BLOCK_HINTS = (
    "/s/", "By:", "Name:", "Title:", "Signature:", "Authorized Signatory",
    "Executed by", "IN WITNESS WHEREOF",
)


def _detect_sections(text: str, max_results: int = 200) -> list[dict]:
    """Walk the document; produce normalized section entries with line offsets."""
    seen: set[str] = set()
    out: list[dict] = []
    for pattern in _SECTION_PATTERNS:
        for m in pattern.finditer(text):
            label = (m.group(1) or "").strip()
            title = (m.group(2) or "").strip() if m.lastindex and m.lastindex >= 2 else ""
            key = (label.lower(), title.lower()[:80])
            if key in seen:
                continue
            seen.add(key)
            line_start = text.count("\n", 0, m.start()) + 1
            depth = label.count(".") + (label.count("#") if "#" in label else 0)
            out.append({
                "label": label,
                "title": title,
                "line_start": line_start,
                "depth": depth,
                "char_offset": m.start(),
            })
            if len(out) >= max_results:
                return out
    out.sort(key=lambda d: d["char_offset"])
    return out


def _detect_schedules(text: str) -> list[dict]:
    """Find Schedule X / Exhibit Y / Annex Z labels with first-occurrence offset."""
    seen: dict[tuple[str, str], dict] = {}
    for m in _SCHEDULE_RE.finditer(text):
        kind = m.group(1).strip().lower()
        ref = m.group(2).strip()
        key = (kind, ref.lower())
        if key in seen:
            seen[key]["count"] += 1
            continue
        seen[key] = {
            "kind": kind,
            "label": f"{m.group(1).strip()} {ref}",
            "ref": ref,
            "first_offset": m.start(),
            "line_start": text.count("\n", 0, m.start()) + 1,
            "count": 1,
        }
    out = sorted(seen.values(), key=lambda d: d["first_offset"])
    return out


def _detect_tables(text: str) -> list[dict]:
    """Identify markdown-style table blocks. Returns row-count + char window."""
    lines = text.split("\n")
    out: list[dict] = []
    block_start = None
    block_rows = 0
    for i, line in enumerate(lines):
        if _TABLE_LINE_RE.match(line):
            if block_start is None:
                block_start = i
            block_rows += 1
        else:
            if block_start is not None and block_rows >= 2:
                out.append({
                    "line_start": block_start + 1,
                    "line_end": i,
                    "n_rows": block_rows,
                })
            block_start = None
            block_rows = 0
    if block_start is not None and block_rows >= 2:
        out.append({
            "line_start": block_start + 1,
            "line_end": len(lines),
            "n_rows": block_rows,
        })
    return out


def _detect_footnotes(text: str) -> list[dict]:
    """Best-effort footnote indices like [1], [*]."""
    seen: set[str] = set()
    out: list[dict] = []
    for m in _FOOTNOTE_RE.finditer(text):
        marker = (m.group(1) or "").strip()
        body = (m.group(2) or "").strip()
        if marker in seen:
            continue
        seen.add(marker)
        out.append({
            "marker": marker,
            "preview": body[:120],
            "char_offset": m.start(),
        })
    return out


def _detect_signature_status(text: str) -> dict:
    """Signed/unsigned heuristic from common signature-block hints."""
    hits = []
    for hint in _SIGNATURE_BLOCK_HINTS:
        if hint in text:
            hits.append(hint)
    has_filled_signature = "/s/" in text or _re.search(
        r"By:\s*[A-Z][a-z]", text
    ) is not None
    return {
        "has_signature_block": bool(hits),
        "is_signed_heuristic": has_filled_signature,
        "hints_found": hits,
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


@dataclass
class DocumentFileReader:
    """Structural file-reading operator. Cross-domain. Deterministic.

    Reads each document in the matter (capped) and emits durable
    structural artifacts (one per kind per document):
      - document.section_map
      - document.schedule_index
      - document.table_index
      - document.footnote_map
      - document.signature_status
    """

    agent_id: str = "document.file_reader.v1"
    version: int = 1
    enabled: bool = True
    priority: int = 100
    capability_tags: tuple[str, ...] = (
        "parse.document_structure",
        "parse.section_map",
        "parse.table",
        "parse.schedule",
        "parse.footnote",
        "parse.signature",
    )
    supported_domain_profiles: tuple[str, ...] = (
        "legal:1", "finance:1", "coding:1",
        "academic_research:1", "biomedical:1",
    )
    phases: tuple[str, ...] = ("pre_synthesis",)
    exclusive_group: Optional[str] = None
    deterministic: bool = True

    # Caps — keep cost bounded
    max_documents: int = 25
    max_excerpt_chars: int = 80_000
    min_section_count_for_artifact: int = 2

    # ------------------------------------------------------------------
    # match
    # ------------------------------------------------------------------

    def match(self, invocation: AgentInvocation) -> Optional[AgentMatch]:
        family = (invocation.execution_family or "").lower()
        if family and family not in {"investigate", "extract", "compare"}:
            return None
        wp = invocation.work_profile or {}
        n_docs = int(wp.get("document_inventory_count", -1))
        n_section_maps = int(wp.get("section_map_count", -1))
        # Already-parsed docs don't need re-parsing
        if n_docs > 0 and n_section_maps >= n_docs:
            return AgentMatch(
                agent_id=self.agent_id, score=0.10,
                reasons=("documents_already_parsed",),
                requirement=AgentRequirement.OPTIONAL,
                phase="pre_synthesis",
            )
        if n_docs > 0:
            return AgentMatch(
                agent_id=self.agent_id, score=0.90,
                reasons=(f"documents:{n_docs}",),
                requirement=AgentRequirement.OPTIONAL,
                phase="pre_synthesis",
            )
        if n_docs == 0:
            return AgentMatch(
                agent_id=self.agent_id, score=0.10,
                reasons=("no_documents",),
                requirement=AgentRequirement.OPTIONAL,
                phase="pre_synthesis",
            )
        return AgentMatch(
            agent_id=self.agent_id, score=0.80,
            reasons=("no_work_profile",),
            requirement=AgentRequirement.OPTIONAL,
            phase="pre_synthesis",
        )

    # ------------------------------------------------------------------
    # invoke
    # ------------------------------------------------------------------

    async def invoke(
        self,
        invocation: AgentInvocation,
        runtime: Any,
    ) -> AgentInvocationResult:
        """All sync I/O is delegated to asyncio.to_thread so the event
        loop is never blocked by sqlite reads or doc parsing (Codex
        PR-gate HOLD blocker on async safety)."""
        import asyncio as _asyncio
        import time as _time

        t0 = _time.perf_counter()
        warnings: list[str] = []
        try:
            inventory = await _asyncio.to_thread(self._list_inventory, runtime)
        except Exception as exc:
            return AgentInvocationResult(
                status="error",
                error_class=type(exc).__name__,
                error=str(exc)[:300],
                elapsed_ms=int((_time.perf_counter() - t0) * 1000),
            )
        if not inventory:
            return AgentInvocationResult(
                status="success",
                elapsed_ms=int((_time.perf_counter() - t0) * 1000),
                warnings=("no_documents",),
            )

        # Read + parse each document on a worker thread. This keeps the
        # event loop free for other concurrent investigations.
        artifacts: list[AgentArtifact] = []

        def _read_and_parse(inv_row: Mapping[str, Any]) -> tuple[list[AgentArtifact], Optional[str]]:
            doc_id = inv_row.get("id") or inv_row.get("relative_path") or ""
            doc_path = inv_row.get("relative_path") or doc_id
            if not doc_id:
                return ([], None)
            try:
                text = self._read_document_text(runtime, doc_path)
            except Exception as exc:
                return ([], f"read_error:{doc_path}:{type(exc).__name__}")
            if not text:
                return ([], None)
            arts = self._parse_document(
                doc_id=str(doc_id), doc_path=str(doc_path), text=text,
            )
            return (arts, None)

        for inv_row in inventory[: self.max_documents]:
            arts, warn = await _asyncio.to_thread(_read_and_parse, inv_row)
            artifacts.extend(arts)
            if warn:
                warnings.append(warn)

        return AgentInvocationResult(
            status="success",
            artifacts=tuple(artifacts),
            warnings=tuple(warnings),
            elapsed_ms=int((_time.perf_counter() - t0) * 1000),
        )

    # ------------------------------------------------------------------
    # verify_output
    # ------------------------------------------------------------------

    def verify_output(
        self,
        invocation: AgentInvocation,
        result: AgentInvocationResult,
    ) -> AgentInvocationResult:
        if result.status != "success":
            return result
        for a in result.artifacts:
            p = a.payload or {}
            if a.artifact_kind.startswith("document."):
                if "document_id" not in p:
                    return AgentInvocationResult(
                        status="invalid",
                        error_class="MalformedArtifact",
                        error=f"artifact {a.artifact_key} missing document_id",
                        elapsed_ms=result.elapsed_ms,
                        warnings=result.warnings,
                    )
        return result

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _list_inventory(self, runtime: Any) -> list[Mapping[str, Any]]:
        mm = runtime.matter_model
        if mm is None:
            return []
        try:
            rows = mm.db.execute(
                """SELECT id, relative_path FROM document_inventory
                   WHERE matter_id=? ORDER BY relative_path""",
                (mm.matter_id,),
            ).fetchall()
        except Exception:
            return []
        return [dict(r) for r in rows]

    def _read_document_text(self, runtime: Any, path: str) -> str:
        """Best-effort: try repo.read() if attached; otherwise typed_evidence
        text snippets are not available so we return empty.

        For tests, runtime can carry an optional `_test_doc_text` mapping.
        """
        # Test hook
        test_hook = getattr(runtime, "_test_doc_text", None)
        if test_hook and path in test_hook:
            return test_hook[path][: self.max_excerpt_chars]
        # Runtime may carry a repository reference
        repo = getattr(runtime, "_repo", None)
        if repo is None:
            return ""
        try:
            doc = repo.read(path)
        except Exception:
            return ""
        if doc is None:
            return ""
        try:
            text = doc.get_excerpt(max_chars=self.max_excerpt_chars)
        except AttributeError:
            text = (getattr(doc, "full_text", "") or "")[: self.max_excerpt_chars]
        return text or ""

    def _parse_document(
        self, *, doc_id: str, doc_path: str, text: str,
    ) -> list[AgentArtifact]:
        out: list[AgentArtifact] = []

        # Section map
        sections = _detect_sections(text)
        if len(sections) >= self.min_section_count_for_artifact:
            out.append(AgentArtifact(
                artifact_kind="document.section_map",
                artifact_key=f"section_map:{doc_path}",
                payload={
                    "schema_ref": "agent.document.section_map.v1",
                    "document_id": doc_id,
                    "document_path": doc_path,
                    "sections": sections,
                    "n_sections": len(sections),
                },
                label=f"Section map [{doc_path}]: {len(sections)} sections",
                synthesis_visibility="audit_only",
                confidence=0.85,
                verification_state="verified",
            ))

        # Schedule index
        schedules = _detect_schedules(text)
        if schedules:
            out.append(AgentArtifact(
                artifact_kind="document.schedule_index",
                artifact_key=f"schedule_index:{doc_path}",
                payload={
                    "schema_ref": "agent.document.schedule_index.v1",
                    "document_id": doc_id,
                    "document_path": doc_path,
                    "schedules": schedules,
                    "n_schedules": len(schedules),
                },
                label=f"Schedules [{doc_path}]: {len(schedules)} entries",
                synthesis_visibility="audit_only",
                confidence=0.85,
                verification_state="verified",
            ))

        # Table index
        tables = _detect_tables(text)
        if tables:
            out.append(AgentArtifact(
                artifact_kind="document.table_index",
                artifact_key=f"table_index:{doc_path}",
                payload={
                    "schema_ref": "agent.document.table_index.v1",
                    "document_id": doc_id,
                    "document_path": doc_path,
                    "tables": tables,
                    "n_tables": len(tables),
                },
                label=f"Tables [{doc_path}]: {len(tables)} tables",
                synthesis_visibility="audit_only",
                confidence=0.75,
                verification_state="candidate",
            ))

        # Footnote map
        footnotes = _detect_footnotes(text)
        if footnotes:
            out.append(AgentArtifact(
                artifact_kind="document.footnote_map",
                artifact_key=f"footnote_map:{doc_path}",
                payload={
                    "schema_ref": "agent.document.footnote_map.v1",
                    "document_id": doc_id,
                    "document_path": doc_path,
                    "footnotes": footnotes,
                    "n_footnotes": len(footnotes),
                },
                label=f"Footnotes [{doc_path}]: {len(footnotes)} markers",
                synthesis_visibility="audit_only",
                confidence=0.7,
                verification_state="candidate",
            ))

        # Signature status
        sig = _detect_signature_status(text)
        if sig["has_signature_block"]:
            out.append(AgentArtifact(
                artifact_kind="document.signature_status",
                artifact_key=f"signature_status:{doc_path}",
                payload={
                    "schema_ref": "agent.document.signature_status.v1",
                    "document_id": doc_id,
                    "document_path": doc_path,
                    **sig,
                },
                label=(
                    f"Signature [{doc_path}]: "
                    + ("signed" if sig["is_signed_heuristic"] else "unsigned")
                ),
                synthesis_visibility="answer_ingredient",
                confidence=0.7 if sig["is_signed_heuristic"] else 0.5,
                verification_state="candidate",
            ))

        return out
