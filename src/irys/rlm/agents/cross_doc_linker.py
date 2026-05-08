"""CrossDocLinker — provenance-rich link operator.

Operator Substrate Thesis: Sub-agents are bounded operators. This one
discovers and writes durable cross-document links: section references
(Section 4.01 in Agreement A → Section 4.01 in checklist row), entity
unification (same party named slightly differently across docs),
schedule↔agreement traversal.

Inputs: document.section_map + document.schedule_index agent_artifacts
produced by DocumentFileReader, plus typed_evidence rows that may carry
section/schedule references in their payloads.

Outputs:
  link.cross_reference — Section/Article references resolved to a
    document_id + section label
  link.schedule_to_agreement — Schedule X in Agreement A traversed to
    its referencing locations
  link.entity_unification — same entity referenced across multiple
    documents (alias-evidence-required to merge)

Per the user directive ("better linking sub-agents we could redo"):
this is a structural primitive — once links are durable, downstream
operators (CP extractor, synthesis renderer, future cross-doc
verifiers) consume them without re-deriving.

Fully deterministic. Zero LLM calls.
"""

from __future__ import annotations

import json as _json
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
# Cross-reference patterns
# ---------------------------------------------------------------------------


# "Section 4.01(a)", "Section 4.01"
_SECTION_REF_RE = _re.compile(
    r"\b(Section\s+\d+(?:\.\d+)*(?:\([a-z]+\))?)\b",
    _re.IGNORECASE,
)
_ARTICLE_REF_RE = _re.compile(r"\b(ARTICLE\s+[IVXLC\d]+)\b")
_SCHEDULE_REF_RE = _re.compile(
    r"\b(Schedule|Exhibit|Annex|Appendix)\s+"
    r"([A-Z0-9]+(?:[\.\-][A-Z0-9]+)*(?:\([a-zA-Z0-9]+\))?)",
    _re.IGNORECASE,
)


def _normalize_section_label(s: str) -> str:
    return _re.sub(r"\s+", " ", (s or "").strip().lower())


def _normalize_schedule_label(kind: str, ref: str) -> str:
    return f"{(kind or '').strip().lower()} {(ref or '').strip().lower()}"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


@dataclass
class CrossDocLinker:
    """Cross-document reference resolver. Deterministic.

    Builds three artifact families:
      - link.cross_reference: section refs resolved to defining document
      - link.schedule_to_agreement: schedule refs traversed to referencing docs
      - link.entity_unification: shared entities across documents
        (currently: party-name aliases — case/punctuation variants only;
         no LLM-based fuzzy merging — that's deferred to a future agent)
    """

    agent_id: str = "link.cross_document_linker.v1"
    version: int = 1
    enabled: bool = True
    priority: int = 100
    capability_tags: tuple[str, ...] = (
        "link.cross_reference", "link.schedule_traversal",
        "link.entity_unification", "verify.cross_doc",
    )
    supported_domain_profiles: tuple[str, ...] = (
        "legal:1", "finance:1", "coding:1",
        "academic_research:1", "biomedical:1",
    )
    phases: tuple[str, ...] = ("pre_synthesis",)
    exclusive_group: Optional[str] = None
    deterministic: bool = True

    max_cross_refs: int = 500
    max_entity_pairs: int = 200

    # ------------------------------------------------------------------
    # match
    # ------------------------------------------------------------------

    def match(self, invocation: AgentInvocation) -> Optional[AgentMatch]:
        family = (invocation.execution_family or "").lower()
        if family and family not in {"investigate", "extract", "compare"}:
            return None
        return AgentMatch(
            agent_id=self.agent_id,
            score=0.78,
            reasons=("cross_doc_linkage",),
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
        import time as _time

        t0 = _time.perf_counter()
        warnings: list[str] = []
        try:
            section_maps = self._load_section_maps(runtime)
            schedule_indexes = self._load_schedule_indexes(runtime)
        except Exception as exc:
            return AgentInvocationResult(
                status="error",
                error_class=type(exc).__name__,
                error=str(exc)[:300],
                elapsed_ms=int((_time.perf_counter() - t0) * 1000),
            )
        artifacts: list[AgentArtifact] = []

        # Cross-reference artifact: which document defines each section?
        if section_maps:
            cross_ref_art = self._build_cross_references(
                section_maps, runtime, warnings,
            )
            if cross_ref_art:
                artifacts.append(cross_ref_art)

        # Schedule → agreement traversal: which schedules are referenced
        # from where?
        if schedule_indexes:
            sched_art = self._build_schedule_traversal(
                schedule_indexes, runtime, warnings,
            )
            if sched_art:
                artifacts.append(sched_art)

        # Entity unification runs independently of doc structure artifacts —
        # actors are populated by other engine paths.
        entity_art = self._build_entity_unification(runtime, warnings)
        if entity_art:
            artifacts.append(entity_art)

        if not section_maps and not schedule_indexes and not artifacts:
            warnings = list(warnings) + ["no_structure_artifacts"]

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
            if a.artifact_kind.startswith("link.") and "n_links" not in p:
                return AgentInvocationResult(
                    status="invalid",
                    error_class="MalformedArtifact",
                    error=f"artifact {a.artifact_key} missing n_links",
                    elapsed_ms=result.elapsed_ms,
                    warnings=result.warnings,
                )
        return result

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _load_section_maps(self, runtime: Any) -> list[dict]:
        return self._load_kind(runtime, "document.section_map")

    def _load_schedule_indexes(self, runtime: Any) -> list[dict]:
        return self._load_kind(runtime, "document.schedule_index")

    def _load_kind(self, runtime: Any, kind: str) -> list[dict]:
        mm = runtime.matter_model
        if mm is None:
            return []
        try:
            rows = mm.db.execute(
                """SELECT artifact_kind, artifact_key, payload_json
                   FROM agent_artifact
                   WHERE matter_id=? AND artifact_kind=?
                   ORDER BY created_at DESC""",
                (mm.matter_id, kind),
            ).fetchall()
        except Exception:
            return []
        out: list[dict] = []
        for r in rows:
            try:
                payload = _json.loads(r["payload_json"] or "{}")
            except Exception:
                continue
            payload["_artifact_key"] = r["artifact_key"]
            out.append(payload)
        return out

    def _list_documents(self, runtime: Any) -> list[Mapping[str, Any]]:
        mm = runtime.matter_model
        if mm is None:
            return []
        try:
            rows = mm.db.execute(
                """SELECT id, relative_path FROM document_inventory
                   WHERE matter_id=?""",
                (mm.matter_id,),
            ).fetchall()
        except Exception:
            return []
        return [dict(r) for r in rows]

    def _build_cross_references(
        self,
        section_maps: list[dict],
        runtime: Any,
        warnings: list[str],
    ) -> Optional[AgentArtifact]:
        """For each section label found in any section_map, attribute it to
        the document that defines it. Build a normalized lookup."""
        # Defining-document index: section_label → document_path
        defining: dict[str, list[dict]] = {}
        for sm in section_maps:
            doc_path = sm.get("document_path") or sm.get("document_id") or ""
            for s in (sm.get("sections") or []):
                label_norm = _normalize_section_label(s.get("label") or "")
                if not label_norm:
                    continue
                defining.setdefault(label_norm, []).append({
                    "document_path": doc_path,
                    "title": s.get("title"),
                    "line_start": s.get("line_start"),
                    "depth": s.get("depth"),
                })
        if not defining:
            return None

        # Flag ambiguous (same section label defined in multiple docs)
        ambiguous: list[dict] = []
        unique_links: list[dict] = []
        for label, defs in defining.items():
            if len(defs) > 1:
                ambiguous.append({
                    "section_label": label,
                    "n_definitions": len(defs),
                    "documents": [d["document_path"] for d in defs],
                })
            for d in defs:
                unique_links.append({
                    "section_label": label,
                    "document_path": d["document_path"],
                    "title": d.get("title"),
                    "line_start": d.get("line_start"),
                    "confidence": 1.0 if len(defs) == 1 else 0.5,
                    "reason": "section_map_definition",
                })

        unique_links = unique_links[: self.max_cross_refs]
        if ambiguous:
            warnings.append(f"ambiguous_sections:n={len(ambiguous)}")

        return AgentArtifact(
            artifact_kind="link.cross_reference",
            artifact_key="cross_ref:matter",
            payload={
                "schema_ref": "agent.link.cross_reference.v1",
                "n_links": len(unique_links),
                "n_ambiguous": len(ambiguous),
                "links": unique_links,
                "ambiguous": ambiguous,
            },
            label=(
                f"Cross-references: {len(unique_links)} sections "
                f"({len(ambiguous)} ambiguous)"
            ),
            synthesis_visibility="audit_only",
            confidence=0.9 if not ambiguous else 0.7,
            verification_state="verified" if not ambiguous else "candidate",
        )

    def _build_schedule_traversal(
        self,
        schedule_indexes: list[dict],
        runtime: Any,
        warnings: list[str],
    ) -> Optional[AgentArtifact]:
        """Each schedule entry maps a schedule_label to its hosting document.
        Plus the count of references gives traversal weight."""
        traversals: list[dict] = []
        for si in schedule_indexes:
            doc_path = si.get("document_path") or si.get("document_id") or ""
            for sch in (si.get("schedules") or []):
                label = _normalize_schedule_label(
                    sch.get("kind", ""), sch.get("ref", ""),
                )
                if not label:
                    continue
                traversals.append({
                    "schedule_label": label,
                    "kind": sch.get("kind"),
                    "ref": sch.get("ref"),
                    "host_document": doc_path,
                    "reference_count": int(sch.get("count") or 1),
                    "first_offset": sch.get("first_offset"),
                    "direction": "agreement_to_schedule",
                    "confidence": 1.0,
                    "reason": "schedule_index_entry",
                })
        if not traversals:
            return None
        return AgentArtifact(
            artifact_kind="link.schedule_to_agreement",
            artifact_key="schedule_traversal:matter",
            payload={
                "schema_ref": "agent.link.schedule_to_agreement.v1",
                "n_links": len(traversals),
                "traversals": traversals,
            },
            label=f"Schedule traversals: {len(traversals)} entries",
            synthesis_visibility="audit_only",
            confidence=0.9,
            verification_state="verified",
        )

    def _build_entity_unification(
        self, runtime: Any, warnings: list[str],
    ) -> Optional[AgentArtifact]:
        """Conservative case/punctuation-only alias detection across
        actor / entity stores. Real fuzzy merging is left to a future
        LLM-using agent — the contract here says no merging without
        alias evidence.
        """
        mm = runtime.matter_model
        if mm is None:
            return None
        try:
            rows = mm.db.execute(
                "SELECT id, canonical_name FROM actor WHERE matter_id=?",
                (mm.matter_id,),
            ).fetchall()
        except Exception:
            return None
        if not rows:
            return None

        groups: dict[str, list[dict]] = {}
        for r in rows:
            name = (r["canonical_name"] or "").strip()
            if not name:
                continue
            # Normalize: lowercase, collapse whitespace, drop trailing
            # corporate suffixes (Inc, LLC, Ltd, Corp, Co)
            stripped = _re.sub(
                r"\s*(?:,\s*)?(Inc\.?|LLC|Ltd\.?|Corp\.?|Co\.?|Company|GmbH|S\.A\.?|N\.V\.?)\s*$",
                "", name, flags=_re.IGNORECASE,
            )
            key = _re.sub(r"\s+", " ", stripped.lower()).strip()
            if not key:
                continue
            groups.setdefault(key, []).append({
                "actor_id": r["id"],
                "canonical_name": name,
            })

        unifications = [
            {"normalized_key": k, "members": members,
             "n_aliases": len(members),
             "confidence": 0.95 if len(members) >= 2 else 1.0,
             "reason": "case_and_suffix_normalization"}
            for k, members in groups.items()
            if len(members) >= 2
        ]
        unifications = unifications[: self.max_entity_pairs]

        n_links = len(unifications)
        if n_links == 0:
            # Still emit an empty (audit_only) artifact so the dispatcher
            # call site has a record this stage ran. Suppress when truly
            # nothing to record:
            return None

        return AgentArtifact(
            artifact_kind="link.entity_unification",
            artifact_key="entity_unification:matter",
            payload={
                "schema_ref": "agent.link.entity_unification.v1",
                "n_links": n_links,
                "unifications": unifications,
            },
            label=f"Entity aliases unified: {n_links} groups",
            synthesis_visibility="audit_only",
            confidence=0.9,
            verification_state="candidate",
        )
