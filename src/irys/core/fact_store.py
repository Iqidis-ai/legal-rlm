"""Fact Store - Persistent storage for extracted facts.

Stores facts in JSONL format for easy appending and line-by-line reading.
Facts include source citations for traceability.
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class StoredFact:
    """A single fact with source citation."""
    fact: str
    source: str  # filename
    page: Optional[int] = None
    quote: Optional[str] = None  # verbatim quote if available
    category: Optional[str] = None  # financial, timeline, entity, etc.
    extracted: str = ""  # ISO date string
    query_context: Optional[str] = None  # what query led to this extraction

    def __post_init__(self):
        if not self.extracted:
            self.extracted = datetime.now().strftime("%Y-%m-%d")

    def to_json_line(self) -> str:
        """Convert to JSON line for JSONL storage."""
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json_line(cls, line: str) -> "StoredFact":
        """Parse from JSON line."""
        data = json.loads(line.strip())
        return cls(**data)

    def matches_query(self, query_lower: str) -> bool:
        """Simple keyword matching for relevance filtering."""
        fact_lower = self.fact.lower()
        # Check if any significant query words appear in the fact
        query_words = [w for w in query_lower.split() if len(w) > 3]
        return any(word in fact_lower for word in query_words)


class FactStore:
    """
    Persistent fact storage using JSONL format.

    Storage location: {repository_path}/.irys/facts.jsonl

    Usage:
        store = FactStore(repo_path)
        store.load()

        # Add facts
        store.add_fact(StoredFact(fact="Contract value was $2.5M", source="Agreement.pdf", page=3))

        # Get facts relevant to a query
        relevant = store.get_relevant("What was the contract value?")

        # Format for LLM context
        fact_sheet = store.format_for_llm(relevant)
    """

    STORE_DIR = ".irys"
    FACTS_FILE = "facts.jsonl"

    def __init__(self, repository_path: Path):
        self.repository_path = Path(repository_path)
        self.store_dir = self.repository_path / self.STORE_DIR
        self.facts_file = self.store_dir / self.FACTS_FILE
        self._facts: list[StoredFact] = []
        self._loaded = False

    def _ensure_store_dir(self):
        """Create .irys directory if it doesn't exist."""
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> int:
        """Load facts from JSONL file. Returns number of facts loaded."""
        self._facts = []

        if not self.facts_file.exists():
            logger.info(f"No existing fact store at {self.facts_file}")
            self._loaded = True
            return 0

        try:
            with open(self.facts_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        fact = StoredFact.from_json_line(line)
                        self._facts.append(fact)
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(f"Skipping malformed line {line_num}: {e}")

            logger.info(f"Loaded {len(self._facts)} facts from {self.facts_file}")
            self._loaded = True
            return len(self._facts)

        except Exception as e:
            logger.error(f"Failed to load fact store: {e}")
            self._loaded = True
            return 0

    def save(self) -> int:
        """Save all facts to JSONL file. Returns number of facts saved."""
        self._ensure_store_dir()

        try:
            with open(self.facts_file, "w", encoding="utf-8") as f:
                for fact in self._facts:
                    f.write(fact.to_json_line() + "\n")

            logger.info(f"Saved {len(self._facts)} facts to {self.facts_file}")
            return len(self._facts)

        except Exception as e:
            logger.error(f"Failed to save fact store: {e}")
            return 0

    def add_fact(self, fact: StoredFact) -> bool:
        """
        Add a fact if not duplicate.
        Returns True if added, False if duplicate.
        """
        if not self._loaded:
            self.load()

        # Check for duplicates (same fact text from same source)
        fact_normalized = " ".join(fact.fact.lower().split())
        for existing in self._facts:
            existing_normalized = " ".join(existing.fact.lower().split())
            if (existing_normalized == fact_normalized and
                existing.source == fact.source):
                return False

        self._facts.append(fact)
        return True

    def add_facts_from_extraction(
        self,
        extraction: dict,
        source_filename: str,
        query_context: Optional[str] = None,
    ) -> int:
        """
        Add facts from an extract_facts() result.

        Args:
            extraction: Result from decisions.extract_facts()
            source_filename: The document these facts came from
            query_context: The query that led to this extraction

        Returns:
            Number of new facts added
        """
        added = 0

        # Extract plain facts
        for fact_text in extraction.get("facts", []):
            if not fact_text or not isinstance(fact_text, str):
                continue

            fact = StoredFact(
                fact=fact_text,
                source=source_filename,
                query_context=query_context,
            )
            if self.add_fact(fact):
                added += 1

        # Extract facts from quotes (these have page numbers)
        for quote in extraction.get("quotes", []):
            if not isinstance(quote, dict):
                continue

            quote_text = quote.get("text", "")
            if not quote_text:
                continue

            # Create a fact from the quote with its context
            relevance = quote.get("relevance", "")
            fact_text = f"{relevance}: \"{quote_text}\"" if relevance else quote_text

            fact = StoredFact(
                fact=fact_text,
                source=source_filename,
                page=quote.get("page"),
                quote=quote_text,
                query_context=query_context,
            )
            if self.add_fact(fact):
                added += 1

        return added

    def get_all(self) -> list[StoredFact]:
        """Get all stored facts."""
        if not self._loaded:
            self.load()
        return self._facts.copy()

    def get_relevant(self, query: str, max_facts: int = 50) -> list[StoredFact]:
        """
        Get facts relevant to a query using simple keyword matching.

        For more sophisticated relevance, use get_relevant_with_llm().
        """
        if not self._loaded:
            self.load()

        if not self._facts:
            return []

        query_lower = query.lower()
        relevant = [f for f in self._facts if f.matches_query(query_lower)]

        # Sort by source to group facts from same document
        relevant.sort(key=lambda f: (f.source, f.page or 0))

        return relevant[:max_facts]

    def format_for_llm(
        self,
        facts: Optional[list[StoredFact]] = None,
        max_chars: int = 15000,
    ) -> str:
        """
        Format facts as a string for LLM context.

        Args:
            facts: Facts to format (if None, uses all facts)
            max_chars: Maximum characters to include

        Returns:
            Formatted fact sheet string
        """
        if facts is None:
            facts = self.get_all()

        if not facts:
            return ""

        lines = ["=== CACHED FACTS FROM PREVIOUS INVESTIGATIONS ===", ""]
        current_source = None
        chars = 0

        for fact in facts:
            # Group by source document
            if fact.source != current_source:
                source_header = f"\n[{fact.source}]"
                if chars + len(source_header) > max_chars:
                    break
                lines.append(source_header)
                current_source = fact.source
                chars += len(source_header)

            # Format the fact
            page_ref = f" (p.{fact.page})" if fact.page else ""
            fact_line = f"  - {fact.fact}{page_ref}"

            if chars + len(fact_line) > max_chars:
                lines.append(f"\n... and {len(facts) - len(lines)} more facts")
                break

            lines.append(fact_line)
            chars += len(fact_line)

        return "\n".join(lines)

    def get_stats(self) -> dict:
        """Get statistics about the fact store."""
        if not self._loaded:
            self.load()

        sources = set(f.source for f in self._facts)

        return {
            "total_facts": len(self._facts),
            "unique_sources": len(sources),
            "sources": list(sources),
            "store_path": str(self.facts_file),
            "exists": self.facts_file.exists(),
        }

    def clear(self):
        """Clear all facts (in memory only - call save() to persist)."""
        self._facts = []

    def __len__(self) -> int:
        if not self._loaded:
            self.load()
        return len(self._facts)

    def __bool__(self) -> bool:
        return len(self) > 0
