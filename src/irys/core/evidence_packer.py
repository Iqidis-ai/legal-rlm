"""EvidencePacker — token-budget-aware evidence bundle for synthesis.

Algorithm:
  Pass 1: Allocate token budget per source (<=MAX_SOURCE_SHARE=30%).
  Pass 2: Within each source, sort by density DESC and fill until budget exhausted.
  Final: Emit all selected facts sorted by compound_score DESC across sources,
         each prefixed with [scope_type, tier] quality tags.
"""

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from irys.core.fact_store import StoredFact

MAX_SOURCE_SHARE = 0.30
CHARS_PER_TOKEN  = 4


def _density(fact: "StoredFact", compound_score: float) -> float:
    token_est = max(1, len(fact.fact) // CHARS_PER_TOKEN)
    return compound_score / (1 + math.log1p(token_est))


class EvidencePacker:
    """Static helper — call EvidencePacker.pack(facts, query, token_budget)."""

    @staticmethod
    def pack(
        facts: list["StoredFact"],
        query: str,
        token_budget: int,
    ) -> str:
        """Pack facts into a synthesis-ready string within token_budget."""
        if not facts:
            return ""

        char_budget = token_budget * CHARS_PER_TOKEN

        from collections import defaultdict
        by_source: dict[str, list] = defaultdict(list)
        for f in facts:
            by_source[f.source].append(f)

        n_sources = len(by_source)
        source_char_cap = min(
            char_budget * MAX_SOURCE_SHARE,
            char_budget / n_sources * 1.5,
        )

        def quick_score(f: "StoredFact") -> float:
            scope_bonus = {"targeted": 1.0, "prefix": 0.7, "snippet": 0.4}.get(f.scope_type, 0.4)
            imp_s = (f.importance / 100.0) * scope_bonus
            tier_boost = {"core": 1.15, "validated": 1.08, "draft": 1.0}.get(f.tier, 1.0)
            return imp_s * tier_boost

        selected: list[tuple[float, "StoredFact"]] = []
        total_chars = 0

        for source, source_facts in by_source.items():
            if total_chars >= char_budget:
                break
            scored = [(quick_score(f), f) for f in source_facts]
            scored.sort(key=lambda x: _density(x[1], x[0]), reverse=True)

            source_chars = 0
            for score, fact in scored:
                line = EvidencePacker._format_fact(fact)
                if source_chars + len(line) > source_char_cap:
                    break
                if total_chars + len(line) > char_budget:
                    break
                selected.append((score, fact))
                source_chars += len(line)
                total_chars += len(line)

        selected.sort(key=lambda x: x[0], reverse=True)
        return "\n".join(EvidencePacker._format_fact(fact) for _, fact in selected)

    @staticmethod
    def _format_fact(fact: "StoredFact") -> str:
        page_ref = f"  p.{fact.page}" if fact.page else ""
        return (
            f"[{fact.scope_type}, {fact.tier}]   {fact.fact}\n"
            f"                   Source: {fact.source}{page_ref}"
        )
