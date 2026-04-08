"""Document search - grep-style search across document repository.

No vectors. Direct text matching with context.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .reader import DocumentReader, DocumentContent

logger = logging.getLogger(__name__)

# Document type priority weights for legal document analysis
DOCUMENT_PRIORITY = {
    # High priority - core legal documents
    "contract": 1.5,
    "agreement": 1.5,
    "amendment": 1.4,
    "exhibit": 1.3,
    "declaration": 1.3,
    "affidavit": 1.3,
    "complaint": 1.4,
    "motion": 1.3,
    "order": 1.4,
    "judgment": 1.5,
    # Medium priority - supporting documents
    "memo": 1.2,
    "memorandum": 1.2,
    "letter": 1.1,
    "report": 1.2,
    "analysis": 1.2,
    # Lower priority - correspondence
    "email": 0.9,
    "correspondence": 0.9,
    "note": 0.8,
    "draft": 0.8,
}


def get_document_priority(filename: str) -> float:
    """Get priority weight for a document based on its name/type."""
    filename_lower = filename.lower()

    for doc_type, priority in DOCUMENT_PRIORITY.items():
        if doc_type in filename_lower:
            return priority

    return 1.0  # Default priority


# Legal term synonyms for query expansion
LEGAL_SYNONYMS = {
    "agreement": ["contract", "covenant", "arrangement", "understanding"],
    "contract": ["agreement", "covenant", "arrangement"],
    "breach": ["violation", "default", "non-compliance", "failure"],
    "terminate": ["cancel", "end", "rescind", "void"],
    "liability": ["responsibility", "obligation", "duty"],
    "damages": ["compensation", "remedy", "recovery", "losses"],
    "indemnify": ["hold harmless", "compensate", "reimburse"],
    "confidential": ["proprietary", "secret", "private"],
    "warranty": ["guarantee", "representation", "assurance"],
    "obligation": ["duty", "requirement", "responsibility"],
    "party": ["parties", "signatory", "counterparty"],
    "execute": ["sign", "enter into", "consummate"],
    "material": ["significant", "substantial", "important"],
    "consent": ["approval", "permission", "authorization"],
    "notice": ["notification", "communication", "written notice"],
}


def expand_query(
    query: str,
    max_expansions: int = 3,
    context_terms: list[str] | None = None,
) -> list[str]:
    """Expand a search query with synonyms, related terms, and issue context.

    When *context_terms* are provided (e.g. active predicates or party names),
    the top 1-2 are fused with the original query to produce issue-aware
    expansion variants.
    """
    expanded = [query]
    query_lower = query.lower()

    for term, synonyms in LEGAL_SYNONYMS.items():
        if term in query_lower:
            # Add queries with synonyms
            for synonym in synonyms[:max_expansions]:
                expanded_query = query_lower.replace(term, synonym)
                if expanded_query not in expanded:
                    expanded.append(expanded_query)

    # Issue-aware context fusion: append top context terms to create
    # targeted variants (e.g. "breach" + "payment_obligation" → "breach payment obligation")
    if context_terms:
        for ct in context_terms[:2]:
            ct_clean = ct.replace("_", " ").lower()
            if ct_clean not in query_lower:
                variant = f"{query_lower} {ct_clean}"
                if variant not in expanded:
                    expanded.append(variant)

    return expanded[:max_expansions + 1]


def generate_related_searches(query: str) -> list[str]:
    """Generate related search queries based on the original."""
    related = []
    query_lower = query.lower()

    # Extract key terms (words longer than 3 chars)
    words = [w for w in query_lower.split() if len(w) > 3]

    # Generate phrase variations
    if len(words) >= 2:
        # Pairs of terms
        for i in range(len(words) - 1):
            related.append(f"{words[i]} {words[i+1]}")

    # Add common legal modifiers
    modifiers = ["material", "significant", "breach of", "failure to"]
    for word in words[:2]:
        for modifier in modifiers[:2]:
            if modifier not in query_lower:
                related.append(f"{modifier} {word}")

    return related[:5]


@dataclass
class SearchHit:
    """A single search match."""
    file_path: str
    filename: str
    page_num: int
    line_num: int
    match_text: str
    context_before: list[str] = field(default_factory=list)
    context_after: list[str] = field(default_factory=list)
    score: float = 1.0  # For ranking
    match_count: int = 1  # Matches in this line
    exact_match: bool = False  # Whether this is an exact phrase match

    @property
    def context(self) -> str:
        """Get full context around match."""
        parts = []
        if self.context_before:
            parts.extend(self.context_before)
        parts.append(f">>> {self.match_text}")
        if self.context_after:
            parts.extend(self.context_after)
        return "\n".join(parts)

    @property
    def citation(self) -> str:
        """Get citation string."""
        return f"{self.filename}, p. {self.page_num}"

    @property
    def context_richness(self) -> float:
        """Score based on context quality."""
        score = 0.0
        # More context = better
        score += len(self.context_before) * 0.1
        score += len(self.context_after) * 0.1
        # Longer match text = more specific
        score += min(len(self.match_text) / 100, 0.5)
        return score


@dataclass
class SearchResults:
    """Collection of search results."""
    query: str
    hits: list[SearchHit]
    files_searched: int
    total_matches: int

    def top(self, n: int = 10) -> list[SearchHit]:
        """Get top N results by score."""
        return sorted(self.hits, key=lambda h: h.score, reverse=True)[:n]

    def by_file(self) -> dict[str, list[SearchHit]]:
        """Group results by file."""
        grouped: dict[str, list[SearchHit]] = {}
        for hit in self.hits:
            if hit.file_path not in grouped:
                grouped[hit.file_path] = []
            grouped[hit.file_path].append(hit)
        return grouped


class DocumentSearch:
    """Grep-style search across documents."""

    def __init__(self, reader: Optional[DocumentReader] = None):
        self.reader = reader or DocumentReader()
        self._doc_cache: dict[str, DocumentContent] = {}

    def search(
        self,
        query: str,
        files: list[Path],
        regex: bool = False,
        case_sensitive: bool = False,
        context_lines: int = 2,
        max_workers: int = 10,
    ) -> SearchResults:
        """
        Search for query across all files.

        Args:
            query: Search term or regex pattern
            files: List of file paths to search
            regex: If True, treat query as regex
            case_sensitive: Case sensitive matching
            context_lines: Lines of context before/after match
            max_workers: Max parallel workers

        Returns:
            SearchResults with all matches
        """
        flags = 0 if case_sensitive else re.IGNORECASE

        if regex:
            pattern = re.compile(query, flags)
        else:
            pattern = re.compile(re.escape(query), flags)

        all_hits: list[SearchHit] = []
        files_searched = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._search_file, f, pattern, context_lines, query
                ): f for f in files if self.reader.can_read(f)
            }

            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    hits = future.result()
                    all_hits.extend(hits)
                    files_searched += 1
                except Exception as e:
                    logger.warning(f"Error searching {file_path}: {e}")

        # Score results with multi-factor ranking
        self._score_results(all_hits, query)

        return SearchResults(
            query=query,
            hits=all_hits,
            files_searched=files_searched,
            total_matches=len(all_hits),
        )

    def _score_results(self, hits: list[SearchHit], query: str):
        """Apply multi-factor scoring to search results."""
        if not hits:
            return

        # Calculate file-level statistics
        file_counts: dict[str, int] = {}
        file_page_spread: dict[str, set] = {}

        for hit in hits:
            file_counts[hit.file_path] = file_counts.get(hit.file_path, 0) + 1
            if hit.file_path not in file_page_spread:
                file_page_spread[hit.file_path] = set()
            file_page_spread[hit.file_path].add(hit.page_num)

        # Score each hit
        for hit in hits:
            score = 1.0

            # Factor 1: Match density in file (0.1-0.5)
            density = file_counts[hit.file_path]
            score += min(density * 0.1, 0.5)

            # Factor 2: Page spread (matches on multiple pages = higher relevance)
            page_spread = len(file_page_spread[hit.file_path])
            score += min(page_spread * 0.05, 0.3)

            # Factor 3: Position boost (earlier in document = slightly higher)
            position_factor = 1.0 / (1.0 + hit.page_num * 0.01)
            score += position_factor * 0.2

            # Factor 4: Context richness
            score += hit.context_richness

            # Factor 5: Exact match bonus
            if hit.exact_match:
                score += 0.5

            # Factor 6: Match length bonus (longer matches = more specific)
            query_words = len(query.split())
            match_words = len(hit.match_text.split())
            if match_words >= query_words:
                score += 0.2

            # Factor 7: Document type priority
            doc_priority = get_document_priority(hit.filename)
            score *= doc_priority

            hit.score = score

    def _search_file(
        self,
        file_path: Path,
        pattern: re.Pattern,
        context_lines: int,
        original_query: str = "",
    ) -> list[SearchHit]:
        """Search a single file for pattern matches."""
        # Check cache
        cache_key = str(file_path)
        if cache_key in self._doc_cache:
            doc = self._doc_cache[cache_key]
        else:
            doc = self.reader.read(file_path)
            self._doc_cache[cache_key] = doc

        hits = []

        for page in doc.pages:
            lines = page.text.split('\n')

            for i, line in enumerate(lines):
                matches = list(pattern.finditer(line))
                if matches:
                    # Check for exact phrase match
                    exact_match = original_query.lower() in line.lower() if original_query else False

                    hits.append(SearchHit(
                        file_path=doc.path,
                        filename=doc.filename,
                        page_num=page.page_num,
                        line_num=i + 1,
                        match_text=line.strip(),
                        context_before=[
                            lines[j].strip()
                            for j in range(max(0, i - context_lines), i)
                        ],
                        context_after=[
                            lines[j].strip()
                            for j in range(i + 1, min(len(lines), i + 1 + context_lines))
                        ],
                        match_count=len(matches),
                        exact_match=exact_match,
                    ))

        return hits

    def search_multi(
        self,
        queries: list[str],
        files: list[Path],
        require_all: bool = False,
    ) -> SearchResults:
        """
        Search for multiple terms.

        Args:
            queries: List of search terms
            files: Files to search
            require_all: If True, only return files matching all queries
        """
        all_results = [self.search(q, files) for q in queries]

        if require_all:
            # Find files that appear in all result sets
            file_sets = [
                set(hit.file_path for hit in r.hits)
                for r in all_results
            ]
            common_files = set.intersection(*file_sets) if file_sets else set()

            # Filter hits to only common files
            combined_hits = []
            for r in all_results:
                for hit in r.hits:
                    if hit.file_path in common_files:
                        combined_hits.append(hit)
        else:
            combined_hits = []
            for r in all_results:
                combined_hits.extend(r.hits)

        return SearchResults(
            query=" AND ".join(queries) if require_all else " OR ".join(queries),
            hits=combined_hits,
            files_searched=all_results[0].files_searched if all_results else 0,
            total_matches=len(combined_hits),
        )

    def clear_cache(self):
        """Clear document cache."""
        self._doc_cache.clear()
