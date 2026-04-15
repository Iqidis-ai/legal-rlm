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
MAX_TOTAL_CITATIONS = 100
SOFT_CAP_DOCUMENT = 35
SOFT_CAP_WEB = 35
SOFT_CAP_CASE_LAW = 30

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

Read the ANSWER sentence by sentence.

For each sentence:
- Insert a citation ID at the END of the sentence if the sentence shares at least one valid anchor with the citation text.
- Valid anchors: exact numbers, exact dates, named entities, distinct multi-word phrases, case names.
- For case law: match by case name, party names, or legal citation string even if paraphrased.
- If multiple citations match, insert all IDs separated by a space: [id1] [id2]
- If no citation matches, leave the sentence unchanged.

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


class InlineCitationService:
    """Single-pass LLM inline citation injection service.

    All citation types are sent in one Gemini Lite call.  Case law
    citations are enriched with case name / legal citation metadata.
    """

    # Regex pattern to extract citation markers like [c89e8135]
    CITATION_MARKER_PATTERN = re.compile(r'\[([a-f0-9]{8})\]')

    @classmethod
    def inject(
        cls,
        answer: str,
        citations: list,
        config,
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
            all_ids = {getattr(c, 'id', None) for c in citations} - {None}

            final, diag = cls._inject_with_retry(answer, selected, config, all_ids)

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
    def _inject_with_retry(
        cls,
        answer: str,
        citations: list,
        config,
        all_valid_ids: set,
    ) -> tuple[str, dict[str, Any]]:
        """Sanitize, build prompt, call LLM with one retry on validation failure.

        Returns (annotated_text, diagnostics_dict).
        """
        diag: dict[str, Any] = {"retries": 0, "validation_passed": False}

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
        annotated = cls._call_gemini_lite(prompt, config, active_step=telemetry_step)
        diag["llm_latency_ms"] = int((time.monotonic() - t0) * 1000)

        if cls._validate_response(annotated, answer, all_valid_ids):
            diag["validation_passed"] = True
            cls._attach_llm_telemetry(diag, telemetry_step)
            return annotated, diag

        logger.info("Citation injection attempt 1 failed validation, retrying")
        diag["retries"] = 1

        # Reset step for retry
        try:
            from ..core.telemetry import InvestigationStep
            telemetry_step = InvestigationStep(seq=0, step_name="citation_injection_retry", phase="post_processing")
        except Exception:
            telemetry_step = None

        t0 = time.monotonic()
        annotated = cls._call_gemini_lite(prompt, config, active_step=telemetry_step)
        diag["llm_latency_ms"] += int((time.monotonic() - t0) * 1000)

        if cls._validate_response(annotated, answer, all_valid_ids):
            diag["validation_passed"] = True
            cls._attach_llm_telemetry(diag, telemetry_step)
            return annotated, diag

        logger.warning("Citation injection failed after retry, returning original answer")
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

                # Text excerpt — cap length for prompt budget
                text = getattr(c, 'text', '') or ''
                text_excerpt = text.strip()[:MAX_CITATION_TEXT_CHARS]

                if not text_excerpt:
                    continue

                # Case law enrichment
                case_name = None
                legal_citation = None
                if source_type == 'case_law':
                    case_name, legal_citation = cls._extract_case_law_metadata(c)

                sanitized.append(SanitizedCitation(
                    id=cit_id,
                    source_type=source_type,
                    filename=document,
                    page=page,
                    text_excerpt=text_excerpt,
                    case_name=case_name,
                    legal_citation=legal_citation,
                ))
            except Exception as e:
                logger.debug(f"Failed to sanitize citation: {e}")
                continue

        return sanitized

    @classmethod
    def _extract_case_law_metadata(cls, cit) -> tuple[Optional[str], Optional[str]]:
        """Pull case name and legal citation string from a case law Citation."""
        doc = getattr(cit, 'document', '') or ''
        context = getattr(cit, 'context', '') or ''

        # document is "[Case Law] Smith v. Jones" → strip prefix + <mark> tags
        case_name = doc.replace('[Case Law] ', '').strip()
        case_name = re.sub(r'</?mark>', '', case_name).strip() or None

        # context is "Citation: 123 F.3d 456 | Court: ..." → extract citation string
        context_clean = re.sub(r'</?mark>', '', context)
        legal_citation = None
        if 'Citation:' in context_clean:
            raw = context_clean.split('Citation:')[1].split('|')[0].strip()
            if raw and raw.lower() != 'none':
                legal_citation = raw

        return case_name, legal_citation

    @classmethod
    def _build_citation_block(cls, sanitized: list[SanitizedCitation]) -> str:
        """Build formatted citation block for prompt."""
        blocks = []
        for c in sanitized:
            parts = [f"ID={c.id}"]

            if c.source_type == 'case_law' and c.case_name:
                parts.append(f"CASE_NAME={c.case_name}")
                if c.legal_citation:
                    parts.append(f"LEGAL_CITATION={c.legal_citation}")
            else:
                parts.append(f"FILE={c.filename}")
                if c.page is not None:
                    parts.append(f"PAGE={c.page}")

            header = " | ".join(parts)
            block = f"{header}\nTEXT:\n{c.text_excerpt}"
            blocks.append(block)

        return "\n\n".join(blocks)

    @classmethod
    def _call_gemini_lite(cls, prompt: str, config, active_step=None) -> str:
        """Call Gemini Lite model for citation injection."""
        import asyncio
        from ..core.models import GeminiClient, ModelTier

        # Get or create client
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

        async def _complete():
            return await client.complete(
                prompt=prompt,
                tier=ModelTier.LITE,
                system_prompt=system_prompt,
                timeout=30.0,
                use_cache=False,
                active_step=active_step,
            )

        # Run async call
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, _complete())
                    return future.result(timeout=35.0)
            else:
                return loop.run_until_complete(_complete())
        except RuntimeError:
            return asyncio.run(_complete())

    # Pattern to detect comma-separated IDs in brackets (invalid format)
    MULTI_ID_PATTERN = re.compile(r'\[[a-f0-9]{8}(?:,\s*[a-f0-9]{8})+\]')

    @classmethod
    def _validate_response(cls, annotated: str, original: str, valid_ids: set) -> bool:
        """Validate the annotated response."""
        # Rule 1: Non-empty
        if not annotated or not annotated.strip():
            logger.debug("Validation failed: empty response")
            return False

        # Rule 2: Reject comma-separated IDs in brackets (e.g., [id1, id2])
        if cls.MULTI_ID_PATTERN.search(annotated):
            logger.debug("Validation failed: found comma-separated citation IDs in brackets")
            return False

        # Rule 3: Extract all citation markers
        found_ids = set(cls.CITATION_MARKER_PATTERN.findall(annotated))

        # Rule 4: All IDs must be valid (no unknown IDs)
        invalid_ids = found_ids - valid_ids
        if invalid_ids:
            logger.debug(f"Validation failed: invalid IDs {invalid_ids}")
            return False

        # Rule 5: Length check ±10%
        # Remove markers for length comparison
        annotated_clean = cls.CITATION_MARKER_PATTERN.sub("", annotated)
        original_clean = original.strip()

        len_ratio = len(annotated_clean) / len(original_clean) if original_clean else 0
        if len_ratio < 0.9 or len_ratio > 1.1:
            logger.debug(f"Validation failed: length ratio {len_ratio:.2f} outside 0.9-1.1")
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

