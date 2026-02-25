"""Inline citation injection post-processing service.

Injects citation ID markers into synthesized answers using Gemini Lite,
then deterministically renumbers them to [[cite:1]], [[cite:2]], etc.
"""

import re
import logging
import os
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger("irys.inline_citation")


# Maximum citations to include in prompt (keep prompt compact for Lite model)
MAX_CITATIONS = 12

# Prompt template for Gemini Lite
INLINE_CITATION_PROMPT = """TASK:
Insert inline citation markers into the ANSWER using ONLY the provided CITATIONS.

Rules:
1. Do NOT rewrite or change wording.
2. Only insert citation ID markers in square brackets, like [c89e8135].
3. Use ONLY citation IDs listed below.
4. If support is weak or unclear, do NOT attach a citation.
5. Avoid repeating the same citation in consecutive sentences unless necessary.
6. Attach a citation only to the first sentence that introduces the supported claim.
7. Do not add explanations or commentary.
8. Use SEPARATE brackets for multiple citations: [id1] [id2], NOT [id1, id2].

CITATIONS:
{citation_block}

ANSWER:
{answer}

OUTPUT:
Return ONLY the full answer text with citation ID markers inserted.
No markdown.
No JSON.
No extra text."""


@dataclass
class SanitizedCitation:
    """Compact citation representation for LLM prompt."""
    id: str
    filename: str
    page: Optional[int]
    text_excerpt: str


class InlineCitationService:
    """Service for injecting inline citations into synthesized answers."""

    # Regex pattern to extract citation markers like [c89e8135]
    CITATION_MARKER_PATTERN = re.compile(r'\[([a-f0-9]{8})\]')

    @classmethod
    def inject(
        cls,
        answer: str,
        citations: list,
        config,
    ) -> str:
        """
        Inject inline citation markers into the answer.

        Args:
            answer: The synthesized answer text
            citations: List of Citation objects from state.citations
            config: IrysConfig with enable_inline_citations flag

        Returns:
            Answer with [[cite:1]], [[cite:2]], ... citation markers, or original if injection fails
        """
        # Step A: Early exits
        if not getattr(config, 'enable_inline_citations', False):
            return answer

        if not citations:
            logger.debug("No citations provided, skipping injection")
            return answer

        if not answer or not answer.strip():
            logger.debug("Empty answer, skipping injection")
            return answer

        try:
            return cls._inject_with_retry(answer, citations, config)
        except Exception as e:
            logger.warning(f"Citation injection failed: {e}, returning original answer")
            return answer

    @classmethod
    def _inject_with_retry(cls, answer: str, citations: list, config) -> str:
        """Attempt injection with one retry on validation failure."""
        # Step B: Sanitize citations
        sanitized = cls._sanitize_citations(citations)
        if not sanitized:
            logger.debug("No valid citations after sanitization")
            return answer

        # Build citation ID set for validation
        valid_ids = {c.id for c in sanitized}

        # Step C: Build prompt
        citation_block = cls._build_citation_block(sanitized)
        prompt = INLINE_CITATION_PROMPT.format(
            citation_block=citation_block,
            answer=answer,
        )

        # Step D: Call LLM (attempt 1)
        annotated = cls._call_gemini_lite(prompt, config)

        # Step E: Validate
        if cls._validate_response(annotated, answer, valid_ids):
            # Step F: Renumber and return
            return cls._renumber_citations(annotated)

        # Retry once
        logger.info("First injection attempt failed validation, retrying")
        annotated = cls._call_gemini_lite(prompt, config)

        if cls._validate_response(annotated, answer, valid_ids):
            return cls._renumber_citations(annotated)

        # Both attempts failed
        logger.warning("Citation injection validation failed after retry, using original")
        return answer

    @classmethod
    def _sanitize_citations(cls, citations: list) -> list[SanitizedCitation]:
        """Extract citation data without truncation."""
        sanitized = []

        for c in citations[:MAX_CITATIONS]:
            try:
                # Extract ID
                cit_id = getattr(c, 'id', None)
                if not cit_id:
                    continue

                # Use document path directly
                document = getattr(c, 'document', '') or 'unknown'

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
        system_prompt = "You are a precise citation marker. Insert citation IDs exactly where the text is supported. Do not modify any other text."

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

