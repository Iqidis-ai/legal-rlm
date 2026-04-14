"""Document search - grep-style search across document repository.

No vectors. Direct text matching with context.
Scoring/ranking delegated to LLM decisions layer.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import re
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .reader import DocumentReader, DocumentContent

logger = logging.getLogger(__name__)


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
    match_count: int = 1  # Number of matches in this line

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


@dataclass
class SearchResults:
    """Collection of search results."""
    query: str
    hits: list[SearchHit]
    files_searched: int
    total_matches: int

    def top(self, n: int = 10) -> list[SearchHit]:
        """Get first N results (no scoring - LLM will pick relevant ones)."""
        return self.hits[:n]

    def by_file(self) -> dict[str, list[SearchHit]]:
        """Group results by file."""
        grouped: dict[str, list[SearchHit]] = {}
        for hit in self.hits:
            if hit.file_path not in grouped:
                grouped[hit.file_path] = []
            grouped[hit.file_path].append(hit)
        return grouped

    def format_for_llm(self, max_hits: int = 20) -> str:
        """Format results for LLM consumption."""
        if not self.hits:
            return "No matches found."

        lines = [f"Found {self.total_matches} matches in {self.files_searched} files:"]
        for i, hit in enumerate(self.hits[:max_hits]):
            lines.append(f"\n[{i+1}] {hit.citation}")
            lines.append(hit.context)

        if self.total_matches > max_hits:
            lines.append(f"\n... and {self.total_matches - max_hits} more matches")

        return "\n".join(lines)


class DocumentSearch:
    """Grep-style search across documents.

    Thread-safe: Uses a lock to protect concurrent cache access.
    No scoring - returns raw matches for LLM to evaluate.
    """

    def __init__(self, reader: Optional[DocumentReader] = None, max_cache_size: int = 100):
        self.reader = reader or DocumentReader()
        self._doc_cache: dict[str, DocumentContent] = {}
        self._cache_lock = threading.Lock()
        self._max_cache_size = max_cache_size

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
            SearchResults with all matches (unscored)
        """
        if not query.strip():
            return SearchResults(query=query, hits=[], files_searched=0, total_matches=0)

        flags = 0 if case_sensitive else re.IGNORECASE

        if regex:
            pattern = re.compile(query, flags)
        else:
            pattern = re.compile(re.escape(query), flags)

        all_hits: list[SearchHit] = []
        files_searched = 0

        # Image files (.png/.jpg/.jpeg) require async OCR — they have no extractable
        # text content for grep-style search and must be excluded here.
        _ASYNC_ONLY = {".png", ".jpg", ".jpeg"}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._search_file, f, pattern, context_lines
                ): f for f in files
                if self.reader.can_read(f) and f.suffix.lower() not in _ASYNC_ONLY
            }

            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    hits = future.result()
                    all_hits.extend(hits)
                    files_searched += 1
                except Exception as e:
                    logger.warning(f"Error searching {file_path}: {e}")

        return SearchResults(
            query=query,
            hits=all_hits,
            files_searched=files_searched,
            total_matches=len(all_hits),
        )

    def smart_search(
        self,
        query: str,
        files: list[Path],
        regex: bool = False,
        case_sensitive: bool = False,
        context_lines: int = 2,
        max_workers: int = 10,
    ) -> SearchResults:
        """Search with fallback to OR search on individual terms.

        Tries exact phrase match first. If no results and query has multiple words,
        splits into individual terms and searches for each, deduplicating results.
        """
        # First try exact match
        exact_results = self.search(
            query=query,
            files=files,
            regex=regex,
            case_sensitive=case_sensitive,
            context_lines=context_lines,
            max_workers=max_workers,
        )

        # If we got results, return them
        if exact_results.hits:
            return exact_results

        # Split query into terms
        terms = [t for t in re.split(r"\s+", query.strip()) if t and len(t) > 2]
        if len(terms) <= 1:
            return exact_results

        # Search for each term
        unique_terms = list(dict.fromkeys(terms))  # Preserve order, remove dupes
        all_hits: list[SearchHit] = []
        files_searched = 0

        for term in unique_terms:
            term_results = self.search(
                query=term,
                files=files,
                regex=regex,
                case_sensitive=case_sensitive,
                context_lines=context_lines,
                max_workers=max_workers,
            )
            all_hits.extend(term_results.hits)
            if not files_searched:
                files_searched = term_results.files_searched

        # Deduplicate hits by (file, page, line)
        deduped: dict[tuple[str, int, int], SearchHit] = {}
        for hit in all_hits:
            key = (hit.file_path, hit.page_num, hit.line_num)
            existing = deduped.get(key)
            if existing:
                existing.match_count += hit.match_count
            else:
                deduped[key] = hit

        deduped_hits = list(deduped.values())

        return SearchResults(
            query=" OR ".join(unique_terms),
            hits=deduped_hits,
            files_searched=files_searched,
            total_matches=len(deduped_hits),
        )

    def _search_file(
        self,
        file_path: Path,
        pattern: re.Pattern,
        context_lines: int,
    ) -> list[SearchHit]:
        """Search a single file for pattern matches."""
        cache_key = str(file_path)

        # Check cache with lock
        with self._cache_lock:
            if cache_key in self._doc_cache:
                doc = self._doc_cache[cache_key]
            else:
                doc = self.reader.read(file_path)
                # LRU-style eviction if cache too large
                if len(self._doc_cache) >= self._max_cache_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self._doc_cache))
                    del self._doc_cache[oldest_key]
                self._doc_cache[cache_key] = doc

        hits = []

        for page in doc.pages:
            lines = page.text.split('\n')

            for i, line in enumerate(lines):
                matches = list(pattern.finditer(line))
                if matches:
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
        with self._cache_lock:
            self._doc_cache.clear()

    @property
    def cache_size(self) -> int:
        """Get current cache size."""
        return len(self._doc_cache)
