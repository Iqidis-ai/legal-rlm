"""Inline citation injection post-processing service.

Two-pass injection pipeline:

  Pass 1 — Case law (deterministic, regex):
    Inserts [id] immediately after each case name / citation string in the text.
    No LLM needed — legal writing uses verbatim case names by convention.

  Pass 2 — Documents + Web (LLM, Gemini Lite):
    Inserts [id] at the end of the sentence where content matches.
    The LLM is instructed to leave existing markers from Pass 1 untouched.

Both passes feed into a single _renumber_citations() step that converts
raw UUID-fragment markers to [[cite:1]], [[cite:2]], … in first-appearance order.
"""

import re
import logging
import os
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger("irys.inline_citation")


# Maximum document+web citations sent to the LLM in Pass 2
MAX_DOC_WEB_CITATIONS = 12

# Prompt for Pass 2 — documents + web, end-of-sentence placement.
# The LLM must leave any existing [xxxxxxxx] markers (from Pass 1) untouched.
INLINE_CITATION_PROMPT = """TASK:

Insert inline citation markers into the ANSWER using only the provided CITATIONS.
The ANSWER may already contain citation markers like [xxxxxxxx] — do NOT remove or move them.

ANSWER:
{answer}

CITATIONS:
{citation_block}

INSTRUCTIONS:

Read the ANSWER sentence by sentence.

For each sentence:
- Insert a citation ID at the END of the sentence if the sentence shares at least one valid anchor with the citation text.
- Valid anchors: exact numbers, exact dates, named entities, distinct multi-word phrases.
- If multiple citations match, insert all IDs separated by a space: [id1] [id2]
- If no citation matches, leave the sentence unchanged.
- Do NOT touch existing citation markers already present in the text.

OUTPUT:
Return the full ANSWER text with citation markers inserted.
Return only the ANSWER text. No explanations."""


@dataclass
class SanitizedCitation:
    """Compact citation representation for LLM prompt."""
    id: str
    filename: str
    page: Optional[int]
    text_excerpt: str


class InlineCitationService:
    """Two-pass inline citation injection service.

    Pass 1 — case law citations injected right after the case name (deterministic regex).
    Pass 2 — document + web citations injected at end-of-sentence (Gemini Lite LLM).
    """

    # Regex pattern to extract citation markers like [c89e8135]
    CITATION_MARKER_PATTERN = re.compile(r'\[([a-f0-9]{8})\]')

    @classmethod
    def inject(
        cls,
        answer: str,
        citations: list,
        config,
    ) -> str:
        """Inject inline citation markers into the answer via two-pass pipeline.

        Args:
            answer:    The synthesized answer text.
            citations: List of Citation objects from state.citations.
            config:    IrysConfig with enable_inline_citations flag.

        Returns:
            Answer with [[cite:1]], [[cite:2]], … markers, or original on failure.
        """
        if not getattr(config, 'enable_inline_citations', False):
            return answer
        if not citations or not answer or not answer.strip():
            return answer

        try:
            # Split by source_type (default to "document" for backward compat)
            case_law_cits = [c for c in citations if getattr(c, 'source_type', 'document') == 'case_law']
            doc_web_cits  = [c for c in citations if getattr(c, 'source_type', 'document') != 'case_law']

            # Pass 1: anchor-based injection for case law (deterministic, no LLM)
            after_pass1 = cls._inject_case_law_anchors(answer, case_law_cits)

            # Collect all valid IDs (pass 1 markers already placed + pass 2 candidates)
            all_ids = {getattr(c, 'id', None) for c in citations} - {None}

            # Pass 2: end-of-sentence LLM injection for documents + web
            final = cls._inject_doc_web_with_retry(after_pass1, doc_web_cits, config, all_ids)

            return cls._renumber_citations(final)

        except Exception as e:
            logger.warning(f"Citation injection failed: {e}, returning original answer")
            return answer

    # -------------------------------------------------------------------------
    # Pass 1: case law — deterministic anchor replacement
    # -------------------------------------------------------------------------

    @classmethod
    def _inject_case_law_anchors(cls, text: str, case_law_cits: list) -> str:
        """Insert [id] immediately after each case name or citation string.

        Tries the case name first, then the bare citation string (e.g., '123 F.3d 456').
        Each anchor is matched once per citation to avoid duplicate markers.
        Falls through silently if no match — citation simply won't appear inline.
        """
        for cit in case_law_cits:
            cit_id = getattr(cit, 'id', None)
            if not cit_id:
                continue

            # Already injected (dedup guard for incremental search rounds)
            if f"[{cit_id}]" in text:
                continue

            # Build candidate anchors from the document name and context fields
            doc = getattr(cit, 'document', '') or ''
            context = getattr(cit, 'context', '') or ''

            # document is "[Case Law] Smith v. Jones" → strip prefix and any
            # HTML highlight tags (<mark>…</mark>) injected by CourtListener
            case_name = doc.replace('[Case Law] ', '').strip()
            case_name = re.sub(r'</?mark>', '', case_name).strip()

            # context is "Citation: 123 F.3d 456 | Court: ..." → extract citation string
            # Strip <mark> tags that CourtListener adds for search result highlighting
            context_clean = re.sub(r'</?mark>', '', context)
            citation_str = ''
            if 'Citation:' in context_clean:
                citation_str = context_clean.split('Citation:')[1].split('|')[0].strip()

            anchors = [a for a in [case_name, citation_str] if len(a) > 3]

            injected = False
            for anchor in anchors:
                # Escape for regex; match the anchor not already followed by [id]
                escaped = re.escape(anchor)
                pattern = re.compile(rf'({escaped})(?!\s*\[{re.escape(cit_id)}\])')
                new_text, n = pattern.subn(rf'\1[{cit_id}]', text, count=1)
                if n:
                    text = new_text
                    injected = True
                    break

            if not injected:
                logger.debug(f"Case law anchor not found in text for: {case_name!r}")

        return text

    # -------------------------------------------------------------------------
    # Pass 2: documents + web — LLM end-of-sentence injection
    # -------------------------------------------------------------------------

    @classmethod
    def _inject_doc_web_with_retry(
        cls,
        answer: str,
        doc_web_cits: list,
        config,
        all_valid_ids: set,
    ) -> str:
        """Attempt LLM injection with one retry on validation failure."""
        sanitized = cls._sanitize_citations(doc_web_cits)
        if not sanitized:
            return answer

        citation_block = cls._build_citation_block(sanitized)
        prompt = INLINE_CITATION_PROMPT.format(
            citation_block=citation_block,
            answer=answer,
        )

        annotated = cls._call_gemini_lite(prompt, config)
        if cls._validate_response(annotated, answer, all_valid_ids):
            return annotated

        logger.info("Pass-2 injection attempt 1 failed validation, retrying")
        annotated = cls._call_gemini_lite(prompt, config)
        if cls._validate_response(annotated, answer, all_valid_ids):
            return annotated

        logger.warning("Pass-2 citation injection failed after retry, keeping Pass-1 result")
        return answer

    @classmethod
    def _sanitize_citations(cls, citations: list) -> list[SanitizedCitation]:
        """Extract citation data without truncation."""
        sanitized = []

        for c in citations[:MAX_DOC_WEB_CITATIONS]:
            try:
                # Extract ID
                cit_id = getattr(c, 'id', None)
                if not cit_id:
                    continue

                # Use document path directly; strip CourtListener highlight tags
                document = getattr(c, 'document', '') or 'unknown'
                document = re.sub(r'</?mark>', '', document).strip()

                # Extract page
                page = getattr(c, 'page', None)

                # Use citation text as-is (no truncation)
                text = getattr(c, 'text', '') or ''
                text_excerpt = text.strip()

                if not text_excerpt:
                    continue

                sanitized.append(SanitizedCitation(
                    id=cit_id,
                    filename=document,
                    page=page,
                    text_excerpt=text_excerpt,
                ))
            except Exception as e:
                logger.debug(f"Failed to sanitize citation: {e}")
                continue

        return sanitized

    @classmethod
    def _build_citation_block(cls, sanitized: list[SanitizedCitation]) -> str:
        """Build formatted citation block for prompt."""
        blocks = []
        for c in sanitized:
            page_str = f" | PAGE={c.page}" if c.page is not None else ""
            block = f"ID={c.id} | FILE={c.filename}{page_str}\nTEXT:\n{c.text_excerpt}"
            blocks.append(block)

        return "\n\n".join(blocks)

    @classmethod
    def _call_gemini_lite(cls, prompt: str, config) -> str:
        """Call Gemini Lite model for citation injection."""
        import asyncio
        from ..core.models import GeminiClient, ModelTier

        # Get or create client
        api_key = getattr(config, 'api_key', None) or os.environ.get("GEMINI_API_KEY")
        client = GeminiClient(api_key=api_key)

        # Low temperature system prompt for deterministic output
        system_prompt = """ 
        You insert citation ID markers into a given ANSWER.

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

        Exact numbers (e.g., 17.3%, 5.5%, 13.5%)

        Exact dates (e.g., March 2025)

        Named entities (e.g., SCBs, CRAR, CET1)

        Distinct multi-word phrases (two or more consecutive words that match)

        If multiple citations match a sentence, insert all valid matching citation IDs.

        Insert citations at the end of the supported sentence.

        Separate multiple citations with a single space:
        [id1] [id2]

        Do not insert citations without anchor overlap.
        Do not insert citations to increase count.

        Return the full ANSWER text with citation markers inserted.
        Return only the ANSWER text.
        Do not include explanations or commentary.

        """

        async def _complete():
            return await client.complete(
                prompt=prompt,
                tier=ModelTier.LITE,
                system_prompt=system_prompt,
                timeout=30.0,
                use_cache=False,  # Don't cache injection calls
            )

        # Run async call
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in an async context
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, _complete())
                    return future.result(timeout=35.0)
            else:
                return loop.run_until_complete(_complete())
        except RuntimeError:
            # No event loop, create one
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

