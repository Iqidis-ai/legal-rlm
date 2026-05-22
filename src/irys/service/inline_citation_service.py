"""Inline citation injection post-processing service.

Single-pass LLM injection pipeline:

  All citation types (document, web, case law) are sent to Gemini Lite
  in one prompt.  The LLM inserts [id] at the end of the sentence whose
  content matches the citation text.

  Case law citations are enriched with CASE_NAME and LEGAL_CITATION
  metadata so the LLM can match paraphrased references (e.g. "the
  Kroger decision" → Kroger Co. v. Carpenter).

A _renumber_citations() step converts raw UUID-fragment markers to
[[cite:1]], [[cite:2]], … in first-appearance order.
"""

import re
import logging
import os
import time
from typing import Any, Optional
from dataclasses import dataclass

logger = logging.getLogger("irys.inline_citation")


# ── Citation budget ──────────────────────────────────────────────────────
# Hard ceiling across all types.  Per-type soft caps only kick in when the
# total exceeds MAX_TOTAL.
MAX_TOTAL_CITATIONS = 165
SOFT_CAP_DOCUMENT = 35
SOFT_CAP_WEB = 30
SOFT_CAP_CASE_LAW = 100

# Per-citation text limit for the injection prompt (chars).  We only need
# enough for the LLM to identify anchors; the full text lives elsewhere.
MAX_CITATION_TEXT_CHARS = 750


INLINE_CITATION_PROMPT = """TASK:

Insert inline citation markers into the ANSWER using only the provided CITATIONS.

ANSWER:
{answer}

CITATIONS:
{citation_block}

INSTRUCTIONS:

CASE LAW citations — place the marker immediately after the mention:
- Whenever a case name, party name, or reporter citation string (e.g. "168 S.W.3d 802") appears in the text, insert the citation ID directly after that mention.
- This applies everywhere: body text, markdown headings, parentheticals, and footnotes.
- Match even when paraphrased (e.g. "the Kroger decision" → Kroger Co. v. Persley [id]).
- If a line contains the full citation (e.g. "*City of Keller v. Wilson*, 168 S.W.3d 802"), place the ID immediately after the reporter string.
- If the same case is mentioned multiple times, insert the marker each time.

DOCUMENT and WEB citations — place the marker at the end of the sentence:
- Read sentence by sentence.
- Insert the citation ID at the END of the sentence if the sentence contains a valid anchor from the citation text.
- Valid anchors: exact numbers, exact dates, named entities, distinct multi-word phrases.
- If multiple citations match a sentence, insert all IDs at the end: [id1] [id2]

GENERAL:
- Do not add a marker where there is no matching anchor.
- Do not invent or modify any citation IDs.
- Leave all text that has no match completely unchanged.

OUTPUT:
Return the full ANSWER text with citation markers inserted.
Return only the ANSWER text. No explanations."""


@dataclass
class SanitizedCitation:
    """Compact citation representation for LLM prompt."""
    id: str
    source_type: str            # "document" | "web" | "case_law"
    filename: str
    page: Optional[int]
    text_excerpt: str
    # Case law enrichment (None for non-case-law)
    case_name: Optional[str] = None
    legal_citation: Optional[str] = None
    court: Optional[str] = None


class InlineCitationService:
    """Single-pass LLM inline citation injection service.

    All citation types are sent in one Gemini Lite call.  Case law
    citations are enriched with case name / legal citation metadata.
    """

    # Regex pattern to extract citation markers like [c89e8135]
    CITATION_MARKER_PATTERN = re.compile(r'\[([a-f0-9]{8})\]')

    @classmethod
    async def inject(
        cls,
        answer: str,
        citations: list,
        config,
        trace_ctx=None,
    ) -> tuple[str, list, dict[str, Any]]:
        """Inject inline citation markers into the answer via single LLM pass.

        Args:
            answer:    The synthesized answer text.
            citations: List of Citation objects from state.citations.
            config:    IrysConfig with enable_inline_citations flag.

        Returns:
            Tuple of (annotated_answer, reordered_citations, diagnostics) where:
            - annotated_answer has [[cite:1]], [[cite:2]], … markers
            - reordered_citations is citations sorted by first-appearance in text
              (unreferenced citations are appended at the end)
            - diagnostics dict with injection metrics for telemetry
            On failure returns (original_answer, original_citations, diagnostics).
        """
        empty_diag: dict[str, Any] = {}
        if not getattr(config, 'enable_inline_citations', False):
            return answer, citations, empty_diag
        if not citations or not answer or not answer.strip():
            return answer, citations, empty_diag

        try:
            selected = cls._select_citations(citations)
            # Validate against IDs actually shown to the LLM, not the full citation list.
            # The LLM can only inject IDs it received; using all citations would allow
            # hallucinated IDs to slip through if they happen to match an unsent citation.
            sanitized_for_ids = cls._sanitize_citations(selected)
            all_ids = {s.id for s in sanitized_for_ids}

            final, diag = await cls._inject_single(answer, selected, config, all_ids, trace_ctx=trace_ctx)

            # Compute matched / unmatched from the final text (before renumbering)
            matched_ids = set(cls.CITATION_MARKER_PATTERN.findall(final))
            fed_ids = {getattr(c, 'id', None) for c in selected} - {None}
            unmatched_ids = fed_ids - matched_ids

            diag.update({
                "total_citations_input": len(citations),
                "citations_fed_to_llm": len(selected),
                "budget_capped": len(citations) > MAX_TOTAL_CITATIONS,
                "by_type_input": {
                    "document": sum(1 for c in citations if getattr(c, 'source_type', 'document') == 'document'),
                    "web": sum(1 for c in citations if getattr(c, 'source_type', 'document') == 'web'),
                    "case_law": sum(1 for c in citations if getattr(c, 'source_type', 'document') == 'case_law'),
                },
                "citations_matched_inline": len(matched_ids),
                "citations_unmatched": len(unmatched_ids),
                "unmatched_details": [
                    {"name": getattr(c, 'document', 'unknown'), "source_type": getattr(c, 'source_type', 'document')}
                    for c in selected if getattr(c, 'id', None) in unmatched_ids
                ],
            })

            # Lean log — counts only
            logger.info(
                "citation_injection_complete: fed=%d matched=%d unmatched=%d latency_ms=%d",
                diag["citations_fed_to_llm"],
                diag["citations_matched_inline"],
                diag["citations_unmatched"],
                diag.get("llm_latency_ms", 0),
            )

            reordered = cls._reorder_by_uuid_appearance(final, citations)
            reordered = cls._trim_external_citations(reordered, matched_ids)
            return cls._renumber_citations(final), reordered, diag

        except Exception as e:
            logger.warning(f"Citation injection failed: {e}, returning original answer")
            return answer, citations, {"error": str(e)}

    @classmethod
    def _reorder_by_uuid_appearance(cls, text_with_uuid_markers: str, citations: list) -> list:
        """Return citations sorted by the first appearance of their UUID in text.

        Citations whose UUIDs were not injected into the text are appended at
        the end in their original order.  This ensures state.citations[N-1]
        always corresponds to [[cite:N]] in the annotated answer.
        """
        cit_by_id = {getattr(c, 'id', None): c for c in citations if getattr(c, 'id', None)}

        seen_ids: list[str] = []
        for match in cls.CITATION_MARKER_PATTERN.finditer(text_with_uuid_markers):
            cid = match.group(1)
            if cid not in seen_ids:
                seen_ids.append(cid)

        referenced: set[str] = set(seen_ids)
        reordered = [cit_by_id[cid] for cid in seen_ids if cid in cit_by_id]
        reordered += [c for c in citations if getattr(c, 'id', None) not in referenced]
        return reordered

    # -------------------------------------------------------------------------
    # External citation trimming — keep matched + small fill for unmatched
    # -------------------------------------------------------------------------

    _FILL_TO = 5  # per-type minimum for case law and web

    @classmethod
    def _trim_external_citations(cls, reordered: list, matched_ids: set) -> list:
        """Drop excess unmatched external citations after injection.

        Rules:
        - Document citations: always kept in full.
        - Case law / web that got an inline marker: always kept.
        - Case law / web with no marker: kept only to fill up to _FILL_TO per
          type, to serve as reference items when inline coverage is low.
        """
        matched     = [c for c in reordered if getattr(c, 'id', None) in matched_ids]
        unmatched   = [c for c in reordered if getattr(c, 'id', None) not in matched_ids]

        unmatched_doc = [c for c in unmatched if c.source_type == "document"]
        unmatched_cl  = [c for c in unmatched if c.source_type == "case_law"]
        unmatched_web = [c for c in unmatched if c.source_type == "web"]

        matched_cl  = sum(1 for c in matched if c.source_type == "case_law")
        matched_web = sum(1 for c in matched if c.source_type == "web")

        extra_cl  = unmatched_cl[:max(0, cls._FILL_TO - matched_cl)]
        extra_web = unmatched_web[:max(0, cls._FILL_TO - matched_web)]

        return matched + unmatched_doc + extra_cl + extra_web

    # -------------------------------------------------------------------------
    # Citation selection — soft caps per type, take all if under budget
    # -------------------------------------------------------------------------

    @classmethod
    def _select_citations(cls, citations: list) -> list:
        """Select citations to send to the LLM, respecting soft caps.

        If total citations ≤ MAX_TOTAL_CITATIONS, take all.
        Otherwise apply per-type soft caps and distribute leftover budget
        to types that still have overflow.
        """
        if len(citations) <= MAX_TOTAL_CITATIONS:
            return list(citations)

        by_type: dict[str, list] = {'document': [], 'web': [], 'case_law': []}
        for c in citations:
            st = getattr(c, 'source_type', 'document')
            by_type.setdefault(st, []).append(c)

        caps = {'document': SOFT_CAP_DOCUMENT, 'web': SOFT_CAP_WEB, 'case_law': SOFT_CAP_CASE_LAW}
        selected: list = []
        overflow: list = []

        for stype, cap in caps.items():
            items = by_type.get(stype, [])
            selected.extend(items[:cap])
            overflow.extend(items[cap:])

        # Distribute remaining budget to overflow items
        remaining = MAX_TOTAL_CITATIONS - len(selected)
        if remaining > 0 and overflow:
            selected.extend(overflow[:remaining])

        return selected

    # -------------------------------------------------------------------------
    # Single-pass LLM injection
    # -------------------------------------------------------------------------

    @classmethod
    async def _inject_single(
        cls,
        answer: str,
        citations: list,
        config,
        all_valid_ids: set,
        trace_ctx=None,
    ) -> tuple[str, dict[str, Any]]:
        """Sanitize, build prompt, call LLM once (GeminiClient handles retries/fallbacks).

        Returns (annotated_text, diagnostics_dict).
        """
        diag: dict[str, Any] = {"validation_passed": False}

        sanitized = cls._sanitize_citations(citations)
        if not sanitized:
            return answer, diag

        citation_block = cls._build_citation_block(sanitized)
        prompt = INLINE_CITATION_PROMPT.format(
            citation_block=citation_block,
            answer=answer,
        )
        diag["prompt_chars"] = len(prompt)

        # Create a telemetry step to capture LLM cost/tokens
        try:
            from ..core.telemetry import InvestigationStep
            telemetry_step = InvestigationStep(seq=0, step_name="citation_injection", phase="post_processing")
        except Exception:
            telemetry_step = None

        t0 = time.monotonic()
        annotated = await cls._call_gemini_lite(prompt, config, active_step=telemetry_step, trace_ctx=trace_ctx)
        diag["llm_latency_ms"] = int((time.monotonic() - t0) * 1000)

        if cls._validate_response(annotated, answer, all_valid_ids):
            diag["validation_passed"] = True
            cls._attach_llm_telemetry(diag, telemetry_step)
            return annotated, diag

        logger.warning("Citation injection failed validation, returning original answer")
        cls._attach_llm_telemetry(diag, telemetry_step)
        return answer, diag

    @staticmethod
    def _attach_llm_telemetry(diag: dict, step) -> None:
        """Extract cost/token data from telemetry step into diagnostics."""
        if step is None or not step.operations:
            return
        # Sum across operations (in case of retry, each call has its own step)
        total_cost = 0.0
        total_prompt_tokens = 0
        total_output_tokens = 0
        for op in step.operations:
            total_cost += getattr(op, 'cost_usd', 0.0)
            total_prompt_tokens += getattr(op, 'prompt_tokens', 0)
            total_output_tokens += getattr(op, 'output_tokens', 0)
        diag["llm_cost_usd"] = round(total_cost, 6)
        diag["llm_prompt_tokens"] = total_prompt_tokens
        diag["llm_output_tokens"] = total_output_tokens

    @classmethod
    def _sanitize_citations(cls, citations: list) -> list[SanitizedCitation]:
        """Extract only the fields the LLM needs for anchor matching."""
        sanitized = []

        for c in citations:
            try:
                cit_id = getattr(c, 'id', None)
                if not cit_id:
                    continue

                source_type = getattr(c, 'source_type', 'document')

                # Document name — strip CourtListener highlight tags
                document = getattr(c, 'document', '') or 'unknown'
                document = re.sub(r'</?mark>', '', document).strip()

                page = getattr(c, 'page', None)

                text = getattr(c, 'text', '') or ''
                context = getattr(c, 'context', '') or ''

                case_name = None
                legal_citation = None
                court = None

                if source_type == 'case_law':
                    case_name, legal_citation, court = cls._extract_case_law_metadata(c)
                    # For case law, snippet (text) is often empty.
                    # Always include context; append snippet if available.
                    text_excerpt = context.strip()
                    snippet = text.strip()[:MAX_CITATION_TEXT_CHARS]
                    if snippet:
                        text_excerpt = f"{text_excerpt}\nSnippet: {snippet}" if text_excerpt else snippet
                    text_excerpt = text_excerpt[:MAX_CITATION_TEXT_CHARS]
                else:
                    text_excerpt = text.strip()[:MAX_CITATION_TEXT_CHARS]

                if not text_excerpt and not case_name:
                    logger.debug("Skipping citation %s (%s): no usable content", cit_id, document)
                    continue

                sanitized.append(SanitizedCitation(
                    id=cit_id,
                    source_type=source_type,
                    filename=document,
                    page=page,
                    text_excerpt=text_excerpt or '',
                    case_name=case_name,
                    legal_citation=legal_citation,
                    court=court,
                ))
            except Exception as e:
                logger.debug(f"Failed to sanitize citation: {e}")
                continue

        return sanitized

    @classmethod
    def _extract_case_law_metadata(cls, cit) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Pull case name, legal citation string, and court from a case law Citation."""
        doc = getattr(cit, 'document', '') or ''
        context = getattr(cit, 'context', '') or ''

        # document is "[Case Law] Smith v. Jones" → strip prefix
        case_name = doc.replace('[Case Law] ', '').strip() or None

        # context is "Citation: 123 F.3d 456 | Court: Tex. 2005"
        legal_citation = None
        court = None
        if 'Citation:' in context:
            raw = context.split('Citation:')[1].split('|')[0].strip()
            if raw and raw.lower() != 'none':
                legal_citation = raw
        if 'Court:' in context:
            court = context.split('Court:')[1].strip() or None

        return case_name, legal_citation, court

    @classmethod
    def _build_citation_block(cls, sanitized: list[SanitizedCitation]) -> str:
        """Build formatted citation block for prompt."""
        blocks = []
        for c in sanitized:
            parts = [f"ID={c.id}"]

            if c.source_type == 'case_law':
                if c.case_name:
                    parts.append(f"CASE_NAME={c.case_name}")
                if c.legal_citation:
                    parts.append(f"LEGAL_CITATION={c.legal_citation}")
                if c.court:
                    parts.append(f"COURT={c.court}")
            elif c.source_type == 'web':
                # Strip the [Web] prefix for cleaner display
                name = c.filename.replace('[Web] ', '')
                parts.append(f"SOURCE={name}")
            else:
                parts.append(f"FILE={c.filename}")
                if c.page is not None:
                    parts.append(f"PAGE={c.page}")

            header = " | ".join(parts)
            if c.text_excerpt:
                block = f"{header}\nTEXT:\n{c.text_excerpt}"
            else:
                block = header
            blocks.append(block)

        return "\n\n".join(blocks)

    @classmethod
    async def _call_gemini_lite(cls, prompt: str, config, active_step=None, trace_ctx=None) -> str:
        """Call Gemini Lite model for citation injection (async, non-blocking)."""
        from ..core.models import GeminiClient, ModelTier

        api_key = getattr(config, 'api_key', None) or os.environ.get("GEMINI_API_KEY")
        client = GeminiClient(api_key=api_key)

        # Low temperature system prompt for deterministic output
        system_prompt = """You insert citation ID markers into a given ANSWER.

You may only insert citation markers in square brackets.
You must not change wording.
You must not remove text.
You must not add new text.
You must not reorder sentences.

You may only use citation IDs provided in the CITATIONS block.
Never invent or modify citation IDs.
If no valid citation matches a sentence, leave it unchanged.

A citation may be inserted only if the sentence shares at least one clear anchor with the citation text.

Valid anchors:
- Exact numbers (e.g., 17.3%, 5.5%, 13.5%)
- Exact dates (e.g., March 2025)
- Named entities (e.g., SCBs, CRAR, CET1)
- Distinct multi-word phrases (two or more consecutive words that match)
- Case names and party names (match even if paraphrased, e.g. "the Kroger decision" matches CASE_NAME=Kroger Co. v. Carpenter)
- Legal citation strings (e.g., 123 F.3d 456)

If multiple citations match a sentence, insert all valid matching citation IDs.

Insert citations at the end of the supported sentence.

Separate multiple citations with a single space:
[id1] [id2]

Do not insert citations without anchor overlap.
Do not insert citations to increase count.

Return the full ANSWER text with citation markers inserted.
Return only the ANSWER text.
Do not include explanations or commentary."""

        return await client.complete(
            prompt=prompt,
            tier=ModelTier.LITE,
            system_prompt=system_prompt,
            timeout=30.0,
            use_cache=False,
            active_step=active_step,
            trace_ctx=trace_ctx,
            generation_name="citation_injection",
        )

    # Pattern to detect comma-separated IDs in brackets (invalid format)
    MULTI_ID_PATTERN = re.compile(r'\[[a-f0-9]{8}(?:,\s*[a-f0-9]{8})+\]')

    @classmethod
    def _validate_response(cls, annotated: str, original: str, valid_ids: set) -> bool:
        """Validate the annotated response."""
        # Rule 1: Non-empty
        if not annotated or not annotated.strip():
            logger.info("Citation validation failed: empty response")
            return False

        # Rule 2: Reject comma-separated IDs in brackets (e.g., [id1, id2])
        if cls.MULTI_ID_PATTERN.search(annotated):
            logger.info("Citation validation failed: comma-separated IDs in brackets")
            return False

        # Rule 3: Extract all citation markers
        found_ids = set(cls.CITATION_MARKER_PATTERN.findall(annotated))

        # Rule 4: All IDs must be valid (no unknown IDs)
        invalid_ids = found_ids - valid_ids
        if invalid_ids:
            logger.info("Citation validation failed: %d hallucinated IDs %s", len(invalid_ids), list(invalid_ids)[:3])
            return False

        # Rule 5: Length check ±15% (generous to handle minor reformatting by LLM)
        annotated_clean = cls.CITATION_MARKER_PATTERN.sub("", annotated)
        original_clean = original.strip()

        len_ratio = len(annotated_clean) / len(original_clean) if original_clean else 0
        if len_ratio < 0.85 or len_ratio > 1.15:
            logger.info("Citation validation failed: length ratio %.2f outside 0.85-1.15", len_ratio)
            return False

        return True

    @classmethod
    def _renumber_citations(cls, annotated: str) -> str:
        """Deterministically renumber citations to [[cite:1]], [[cite:2]], etc."""
        # Find all citation IDs in order of first appearance
        seen_ids = []
        for match in cls.CITATION_MARKER_PATTERN.finditer(annotated):
            cit_id = match.group(1)
            if cit_id not in seen_ids:
                seen_ids.append(cit_id)

        if not seen_ids:
            return annotated

        # Build replacement mapping
        id_to_num = {cit_id: str(i + 1) for i, cit_id in enumerate(seen_ids)}

        # Replace all occurrences
        def replacer(match):
            cit_id = match.group(1)
            return f"[[cite:{id_to_num[cit_id]}]]"
        result = cls.CITATION_MARKER_PATTERN.sub(replacer, annotated)
        logger.info(f"Citation injection successful: {len(seen_ids)} unique citations renumbered")

        return result

