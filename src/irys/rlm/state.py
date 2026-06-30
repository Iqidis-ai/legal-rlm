"""State management for RLM investigation.

Tracks thinking steps, citations, and investigation progress.
Simplified: All scoring/ranking delegated to LLM decisions layer.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum
from datetime import datetime
from pathlib import Path
import uuid
import json


class StepType(Enum):
    """Types of investigation steps."""
    # Legacy types (kept for backward compat during transition)
    THINKING = "thinking"
    SEARCH = "search"
    READING = "reading"
    FINDING = "finding"
    REPLAN = "replan"
    VERIFY = "verify"
    SYNTHESIS = "synthesis"
    ERROR = "error"
    # New hierarchical event types
    INVESTIGATION_STARTED = "investigation_started"
    PLAN = "plan"
    LEAD_STARTED = "lead_started"
    LEAD_UPDATE = "lead_update"
    LEAD_DONE = "lead_done"
    LEAD_ERROR = "lead_error"
    CHECKPOINT = "checkpoint"
    SYNTHESIS_STARTED = "synthesis_started"
    SYNTHESIS_COMPLETE = "synthesis_complete"


class QueryType(Enum):
    """Types of legal queries."""
    FACTUAL = "factual"  # What happened? When?
    ANALYTICAL = "analytical"  # What does this mean? Implications?
    COMPARATIVE = "comparative"  # How does X compare to Y?
    EVALUATIVE = "evaluative"  # Is this valid? Strengths/weaknesses?
    PROCEDURAL = "procedural"  # What steps? What process?
    UNKNOWN = "unknown"


def classify_query(query: str) -> dict[str, Any]:
    """Classify a legal query by type and extract key terms."""
    query_lower = query.lower()

    factual_keywords = ["what happened", "when did", "who was", "where did", "what is the", "what was the"]
    analytical_keywords = ["what does", "means", "implications", "significance", "interpret",
                          "key issues", "main issues", "issues in", "claims", "allegations",
                          "why did", "explain", "analyze", "analysis"]
    comparative_keywords = ["compare", "difference", "versus", "vs", "between", "contrast", "how does"]
    evaluative_keywords = ["valid", "strengths", "weaknesses", "assess", "evaluate", "review",
                          "strengths and weaknesses", "merits", "credibility"]
    procedural_keywords = ["how to", "steps", "process", "procedure", "timeline", "sequence"]

    query_type = QueryType.UNKNOWN
    if any(kw in query_lower for kw in evaluative_keywords):
        query_type = QueryType.EVALUATIVE
    elif any(kw in query_lower for kw in comparative_keywords):
        query_type = QueryType.COMPARATIVE
    elif any(kw in query_lower for kw in analytical_keywords):
        query_type = QueryType.ANALYTICAL
    elif any(kw in query_lower for kw in procedural_keywords):
        query_type = QueryType.PROCEDURAL
    elif any(kw in query_lower for kw in factual_keywords):
        query_type = QueryType.FACTUAL

    word_count = len(query.split())
    complexity = 1
    if word_count > 10:
        complexity = 2
    if word_count > 20:
        complexity = 3
    if word_count > 30:
        complexity = 4
    query_words = set(query_lower.split())
    conjunctions = {"and", "or", "but", "however"}
    if query_words & conjunctions:
        complexity = min(complexity + 1, 5)

    potential_entities = []
    words = query.split()
    for i, word in enumerate(words):
        if word[0].isupper() and i > 0 and len(word) > 2:
            potential_entities.append(word.strip(",.?!"))

    return {
        "type": query_type.value,
        "complexity": complexity,
        "word_count": word_count,
        "potential_entities": potential_entities[:5],
    }


@dataclass
class Citation:
    """A citation/reference found during investigation."""
    id: str
    document: str
    page: Optional[int]
    text: str
    context: str
    relevance: str
    timestamp: datetime = field(default_factory=datetime.now)
    url: Optional[str] = None   # Document URL if available
    mime: Optional[str] = None  # Document MIME type if available
    # Source type: "document" | "case_law" | "web"
    # Drives separate inline-citation injection passes in InlineCitationService
    source_type: str = "document"

    @classmethod
    def create(
        cls,
        document: str,
        page: Optional[int],
        text: str,
        context: str,
        relevance: str,
        url: Optional[str] = None,
        mime: Optional[str] = None,
        source_type: str = "document",
    ) -> "Citation":
        return cls(
            id=str(uuid.uuid4())[:8],
            document=document,
            page=page,
            text=text,
            context=context,
            relevance=relevance,
            url=url,
            mime=mime,
            source_type=source_type,
        )


@dataclass
class ThinkingStep:
    """A single step in the investigation process."""
    id: str
    step_type: StepType
    content: str
    details: Optional[dict] = None
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: Optional[int] = None
    depth: int = 0

    @classmethod
    def create(
        cls,
        step_type: StepType,
        content: str,
        details: Optional[dict] = None,
        depth: int = 0,
    ) -> "ThinkingStep":
        return cls(
            id=str(uuid.uuid4())[:8],
            step_type=step_type,
            content=content,
            details=details,
            depth=depth,
        )

    @property
    def display(self) -> str:
        """Human-readable display."""
        indent = "  " * self.depth
        prefix = {
            StepType.THINKING: "[T]",
            StepType.SEARCH: "[S]",
            StepType.READING: "[R]",
            StepType.FINDING: "[F]",
            StepType.REPLAN: "[P]",
            StepType.VERIFY: "[V]",
            StepType.SYNTHESIS: "[Y]",
            StepType.ERROR: "[!]",
            StepType.INVESTIGATION_STARTED: "[>>]",
            StepType.PLAN: "[PLAN]",
            StepType.LEAD_STARTED: "[L>]",
            StepType.LEAD_UPDATE: "[L~]",
            StepType.LEAD_DONE: "[L.]",
            StepType.LEAD_ERROR: "[L!]",
            StepType.CHECKPOINT: "[CK]",
            StepType.SYNTHESIS_STARTED: "[Y>]",
            StepType.SYNTHESIS_COMPLETE: "[Y.]",
        }.get(self.step_type, "*")
        return f"{indent}{prefix} {self.content}"


@dataclass
class Entity:
    """An entity found during investigation."""
    name: str
    entity_type: str  # "person", "company", "date", "amount", "location", "other"
    sources: list[str] = field(default_factory=list)
    mentions: int = 1
    context: Optional[str] = None

    def add_mention(self, source: str):
        """Add a mention of this entity."""
        if source not in self.sources:
            self.sources.append(source)
        self.mentions += 1


@dataclass
class CrossReference:
    """A reference from one document to another."""
    source_doc: str
    target_doc: str
    reference_text: str
    page: Optional[int] = None


@dataclass
class TimelineEvent:
    """An event on the investigation timeline."""
    date_str: str
    date_parsed: Optional[datetime] = None
    description: str = ""
    source_doc: str = ""
    page: Optional[int] = None
    event_type: str = "general"

    def __post_init__(self):
        """Try to parse the date string."""
        if self.date_parsed is None and self.date_str:
            self.date_parsed = self._parse_date(self.date_str)

    @staticmethod
    def _parse_date(date_str: str) -> Optional[datetime]:
        """Try to parse various date formats."""
        import re

        patterns = [
            (r"(\d{1,2})/(\d{1,2})/(\d{4})", "%m/%d/%Y"),
            (r"(\d{1,2})/(\d{1,2})/(\d{2})", "%m/%d/%y"),
            (r"(\d{4})-(\d{2})-(\d{2})", "%Y-%m-%d"),
            (r"(\w+)\s+(\d{1,2}),?\s+(\d{4})", None),
        ]

        for pattern, fmt in patterns:
            match = re.search(pattern, date_str)
            if match:
                try:
                    if fmt:
                        return datetime.strptime(match.group(), fmt)
                    else:
                        months = {
                            "january": 1, "february": 2, "march": 3, "april": 4,
                            "may": 5, "june": 6, "july": 7, "august": 8,
                            "september": 9, "october": 10, "november": 11, "december": 12
                        }
                        month_name = match.group(1).lower()
                        if month_name in months:
                            return datetime(
                                int(match.group(3)),
                                months[month_name],
                                int(match.group(2))
                            )
                except (ValueError, KeyError):
                    pass

        return None


@dataclass
class Contradiction:
    """A potential contradiction found during investigation."""
    id: str
    statement1: str
    source1: str
    statement2: str
    source2: str
    contradiction_type: str  # "factual", "date", "amount", "claim"
    severity: str  # "high", "medium", "low"
    notes: str = ""

    @classmethod
    def create(
        cls,
        statement1: str,
        source1: str,
        statement2: str,
        source2: str,
        contradiction_type: str = "factual",
        severity: str = "medium",
        notes: str = "",
    ) -> "Contradiction":
        return cls(
            id=str(uuid.uuid4())[:8],
            statement1=statement1,
            source1=source1,
            statement2=statement2,
            source2=source2,
            contradiction_type=contradiction_type,
            severity=severity,
            notes=notes,
        )


def _derive_lead_type(description: str) -> str:
    """Derive lead type from description prefix."""
    desc_lower = description.lower().strip()
    if desc_lower.startswith("read document:") or desc_lower.startswith("read:"):
        return "read"
    if desc_lower.startswith("search for:") or desc_lower.startswith("search:"):
        return "search"
    if desc_lower.startswith("caselaw:") or "case law" in desc_lower:
        return "caselaw"
    if desc_lower.startswith("web:") or "web search" in desc_lower:
        return "web"
    return "search"


@dataclass
class Lead:
    """A lead to investigate further."""
    id: str
    description: str
    source: str
    params: dict[str, Any] = field(default_factory=dict)
    investigated: bool = False
    findings: Optional[str] = None
    lead_type: str = "search"
    parent_lead_id: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    @classmethod
    def create(
        cls,
        description: str,
        source: str,
        parent_lead_id: Optional[str] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> "Lead":
        return cls(
            id=str(uuid.uuid4())[:8],
            description=description,
            source=source,
            params=params or {},
            lead_type=_derive_lead_type(description),
            parent_lead_id=parent_lead_id,
        )


@dataclass
class InvestigationState:
    """
    Complete state of an RLM investigation.

    This is the "working memory" that persists across recursive calls.
    Simplified: No scoring - LLM decides what's relevant.
    """
    id: str
    query: str
    repository_path: str

    # Progress tracking
    thinking_steps: list[ThinkingStep] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    leads: list[Lead] = field(default_factory=list)
    entities: dict[str, Entity] = field(default_factory=dict)
    cross_references: list[CrossReference] = field(default_factory=list)
    timeline: list[TimelineEvent] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)

    # Accumulated knowledge
    findings: dict[str, Any] = field(default_factory=dict)
    hypothesis: Optional[str] = None
    query_classification: Optional[dict] = None

    # External research triggers (accumulated from document analysis)
    external_triggers: dict[str, set] = field(default_factory=lambda: {
        "jurisdictions": set(),
        "regulations_statutes": set(),
        "legal_doctrines": set(),
        "industry_standards": set(),
        "case_references": set(),
    })

    # Metrics
    documents_read: int = 0
    searches_performed: int = 0
    recursion_depth: int = 0
    max_depth_reached: int = 0
    api_calls: int = 0
    estimated_tokens: int = 0

    # Status
    status: str = "initialized"
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Telemetry summary (populated by engine after finalize())
    telemetry_summary: Optional[dict] = None

    def __post_init__(self):
        # O(1) dedup set — stores normalized fact strings for fast lookup
        self._fact_hashes: set[str] = set()

    @classmethod
    def create(cls, query: str, repository_path: str) -> "InvestigationState":
        return cls(
            id=str(uuid.uuid4())[:8],
            query=query,
            repository_path=repository_path,
            started_at=datetime.now(),
        )

    def add_step(
        self,
        step_type: StepType,
        content: str,
        details: Optional[dict] = None,
    ) -> ThinkingStep:
        """Add a thinking step."""
        step = ThinkingStep.create(
            step_type=step_type,
            content=content,
            details=details,
            depth=self.recursion_depth,
        )
        self.thinking_steps.append(step)
        return step

    def add_citation(
        self,
        document: str,
        page: Optional[int],
        text: str,
        context: str,
        relevance: str,
        url: Optional[str] = None,
        mime: Optional[str] = None,
        source_type: str = "document",
    ) -> Optional[Citation]:
        """Add a citation if not duplicate."""
        text_normalized = " ".join(text.lower().split())[:100]
        for existing in self.citations:
            existing_normalized = " ".join(existing.text.lower().split())[:100]
            if (existing.document == document and
                existing.page == page and
                existing_normalized == text_normalized):
                return None

        citation = Citation.create(
            document=document,
            page=page,
            text=text,
            context=context,
            relevance=relevance,
            url=url,
            mime=mime,
            source_type=source_type,
        )
        self.citations.append(citation)
        return citation

    def add_lead(
        self,
        description: str,
        source: str,
        parent_lead_id: Optional[str] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> Optional[Lead]:
        """Add a lead to investigate if not duplicate."""
        params = params or {}
        desc_normalized = " ".join(description.lower().split())
        scope_normalized = " ".join(f"{k}={v}" for k, v in sorted(params.items()))

        for existing in self.leads:
            existing_normalized = " ".join(existing.description.lower().split())
            existing_scope = " ".join(
                f"{k}={v}" for k, v in sorted((existing.params or {}).items())
            )
            if (
                self._word_overlap(desc_normalized, existing_normalized) > 0.8
                and scope_normalized == existing_scope
            ):
                return None

        lead = Lead.create(description, source, parent_lead_id=parent_lead_id, params=params)
        self.leads.append(lead)
        return lead

    @staticmethod
    def _word_overlap(s1: str, s2: str) -> float:
        """Calculate simple word overlap ratio."""
        if not s1 or not s2:
            return 0.0
        if s1 == s2:
            return 1.0

        words1 = set(s1.split())
        words2 = set(s2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)

    def add_fact(self, fact: str) -> bool:
        """Add a fact if not duplicate (O(1) hash-based dedup). Returns True if added."""
        if "accumulated_facts" not in self.findings:
            self.findings["accumulated_facts"] = []

        fact_normalized = " ".join(fact.lower().split())

        if fact_normalized in self._fact_hashes:
            return False

        self._fact_hashes.add(fact_normalized)
        self.findings["accumulated_facts"].append(fact)
        return True

    def add_facts(self, facts: list[str]) -> int:
        """Add multiple facts with deduplication. Returns count of added facts."""
        added = 0
        for fact in facts:
            if self.add_fact(fact):
                added += 1
        return added

    def add_triggers(self, triggers: dict[str, list]) -> int:
        """Accumulate external research triggers from document extraction.

        Args:
            triggers: Dict with trigger categories as keys and lists of values

        Returns:
            Number of new triggers added
        """
        added = 0
        for key, values in triggers.items():
            if key in self.external_triggers and values:
                for v in values:
                    if v and v not in self.external_triggers[key]:
                        self.external_triggers[key].add(v)
                        added += 1
        return added

    def has_external_triggers(self, min_triggers: int = 2) -> bool:
        """Check if we have enough triggers to warrant external search.

        Args:
            min_triggers: Minimum number of triggers required

        Returns:
            True if total triggers >= min_triggers
        """
        total = sum(len(v) for v in self.external_triggers.values())
        return total >= min_triggers

    def get_trigger_summary(self) -> str:
        """Format accumulated triggers for external search query generation.

        Returns:
            Formatted string of triggers by category
        """
        parts = []
        for key, values in self.external_triggers.items():
            if values:
                # Convert set to sorted list for consistent output
                formatted_key = key.replace("_", " ").title()
                parts.append(f"{formatted_key}: {', '.join(sorted(values)[:5])}")
        return "\n".join(parts) if parts else "None identified"

    def get_triggers_for_queries(self) -> dict[str, list]:
        """Get triggers formatted for query generation.

        Returns:
            Dict with trigger categories and their values as lists
        """
        return {k: list(v) for k, v in self.external_triggers.items() if v}

    def add_entity(
        self,
        name: str,
        entity_type: str,
        source: str,
        context: Optional[str] = None,
    ) -> Entity:
        """Add or update an entity."""
        key = name.lower().strip()

        if key in self.entities:
            self.entities[key].add_mention(source)
            return self.entities[key]

        entity = Entity(
            name=name.strip(),
            entity_type=entity_type,
            sources=[source],
            mentions=1,
            context=context,
        )
        self.entities[key] = entity
        return entity

    def add_entities_from_analysis(self, entities_dict: dict, source: str):
        """Add entities from LLM analysis output."""
        for entity_type, names in entities_dict.items():
            if isinstance(names, list):
                for name in names:
                    if isinstance(name, str) and name.strip():
                        self.add_entity(name, entity_type, source)

    def get_entities_by_type(self, entity_type: str) -> list[Entity]:
        """Get all entities of a specific type."""
        return [e for e in self.entities.values() if e.entity_type == entity_type]

    def get_top_entities(self, n: int = 10) -> list[Entity]:
        """Get top N entities by mention count."""
        return sorted(self.entities.values(), key=lambda e: e.mentions, reverse=True)[:n]

    def get_entities_formatted(self) -> str:
        """Get formatted entity summary."""
        lines = []
        by_type: dict[str, list[Entity]] = {}

        for entity in self.entities.values():
            if entity.entity_type not in by_type:
                by_type[entity.entity_type] = []
            by_type[entity.entity_type].append(entity)

        for entity_type, entities in sorted(by_type.items()):
            lines.append(f"\n{entity_type.upper()}:")
            for e in sorted(entities, key=lambda x: x.mentions, reverse=True)[:10]:
                lines.append(f"  - {e.name} ({e.mentions} mentions)")

        return "\n".join(lines) if lines else "No entities extracted"

    def get_pending_leads(self) -> list[Lead]:
        """Get uninvestigated leads (FIFO order - LLM picks which to pursue)."""
        return [l for l in self.leads if not l.investigated]

    def get_lead_statistics(self) -> dict[str, Any]:
        """Get statistics about leads."""
        total = len(self.leads)
        investigated = sum(1 for l in self.leads if l.investigated)
        pending = total - investigated

        return {
            "total": total,
            "investigated": investigated,
            "pending": pending,
        }

    def get_confidence_score(self) -> dict[str, Any]:
        """Compute confidence score based on evidence gathered."""
        citations = len(self.citations)
        facts = len(self.findings.get("accumulated_facts", []))
        docs = self.documents_read

        score = min(100, citations * 5 + facts * 2 + docs * 3)
        level = "high" if score >= 70 else "medium" if score >= 40 else "low"

        return {"score": score, "level": level}

    def get_progress(self) -> dict[str, Any]:
        """Get investigation progress metrics."""
        lead_stats = self.get_lead_statistics()

        elapsed = None
        if self.started_at:
            if self.completed_at:
                elapsed = (self.completed_at - self.started_at).total_seconds()
            else:
                elapsed = (datetime.now() - self.started_at).total_seconds()

        return {
            "status": self.status,
            "elapsed_seconds": elapsed,
            "documents_read": self.documents_read,
            "searches_performed": self.searches_performed,
            "citations": len(self.citations),
            "leads_investigated": lead_stats["investigated"],
            "leads_pending": lead_stats["pending"],
            "facts_accumulated": len(self.findings.get("accumulated_facts", [])),
            "entities_found": len(self.entities),
            "api_calls": self.api_calls,
            "estimated_tokens": self.estimated_tokens,
        }

    def increment_api_calls(self, tokens_estimate: int = 0):
        """Increment API call counter and token estimate."""
        self.api_calls += 1
        self.estimated_tokens += tokens_estimate

    def add_cross_reference(
        self,
        source_doc: str,
        target_doc: str,
        reference_text: str,
        page: Optional[int] = None,
    ) -> Optional[CrossReference]:
        """Add a cross-reference if not duplicate."""
        for existing in self.cross_references:
            if (existing.source_doc == source_doc and
                existing.target_doc == target_doc and
                existing.reference_text[:50] == reference_text[:50]):
                return None

        ref = CrossReference(
            source_doc=source_doc,
            target_doc=target_doc,
            reference_text=reference_text,
            page=page,
        )
        self.cross_references.append(ref)
        return ref

    def get_cross_reference_graph(self) -> dict[str, list[str]]:
        """Get document cross-reference graph."""
        graph: dict[str, list[str]] = {}
        for ref in self.cross_references:
            if ref.source_doc not in graph:
                graph[ref.source_doc] = []
            if ref.target_doc not in graph[ref.source_doc]:
                graph[ref.source_doc].append(ref.target_doc)
        return graph

    def add_timeline_event(
        self,
        date_str: str,
        description: str,
        source_doc: str,
        page: Optional[int] = None,
        event_type: str = "general",
    ) -> Optional[TimelineEvent]:
        """Add an event to the timeline if not duplicate."""
        for existing in self.timeline:
            if (existing.date_str == date_str and
                existing.description[:50] == description[:50]):
                return None

        event = TimelineEvent(
            date_str=date_str,
            description=description,
            source_doc=source_doc,
            page=page,
            event_type=event_type,
        )
        self.timeline.append(event)
        return event

    def get_timeline_sorted(self) -> list[TimelineEvent]:
        """Get timeline events sorted by date."""
        def sort_key(e: TimelineEvent):
            if e.date_parsed:
                return (0, e.date_parsed)
            return (1, datetime.max)

        return sorted(self.timeline, key=sort_key)

    def get_timeline_formatted(self) -> str:
        """Get formatted timeline."""
        sorted_events = self.get_timeline_sorted()
        if not sorted_events:
            return "No timeline events"

        lines = ["Timeline:", "=" * 40]
        for event in sorted_events:
            date_display = event.date_str
            if event.date_parsed:
                date_display = event.date_parsed.strftime("%Y-%m-%d")
            lines.append(f"  {date_display}: {event.description[:60]}...")
            lines.append(f"    Source: {event.source_doc}, p.{event.page}")

        return "\n".join(lines)

    def add_contradiction(
        self,
        statement1: str,
        source1: str,
        statement2: str,
        source2: str,
        contradiction_type: str = "factual",
        severity: str = "medium",
        notes: str = "",
    ) -> Contradiction:
        """Add a potential contradiction."""
        contradiction = Contradiction.create(
            statement1=statement1,
            source1=source1,
            statement2=statement2,
            source2=source2,
            contradiction_type=contradiction_type,
            severity=severity,
            notes=notes,
        )
        self.contradictions.append(contradiction)
        return contradiction

    def get_contradictions_by_severity(self, severity: str) -> list[Contradiction]:
        """Get contradictions filtered by severity."""
        return [c for c in self.contradictions if c.severity == severity]

    def get_contradictions_formatted(self) -> str:
        """Get formatted contradictions list."""
        if not self.contradictions:
            return "No contradictions detected"

        lines = ["Potential Contradictions:", "=" * 40]

        for severity in ["high", "medium", "low"]:
            contradictions = self.get_contradictions_by_severity(severity)
            if contradictions:
                lines.append(f"\n[{severity.upper()}]")
                for c in contradictions:
                    lines.append(f"  Type: {c.contradiction_type}")
                    lines.append(f"  Statement 1 ({c.source1}): {c.statement1[:80]}...")
                    lines.append(f"  Statement 2 ({c.source2}): {c.statement2[:80]}...")
                    if c.notes:
                        lines.append(f"  Notes: {c.notes}")
                    lines.append("")

        return "\n".join(lines)

    def mark_lead_investigated(self, lead_id: str, findings: Optional[str] = None):
        """Mark a lead as investigated."""
        for lead in self.leads:
            if lead.id == lead_id:
                lead.investigated = True
                lead.findings = findings
                break

    def get_thinking_trace(self) -> str:
        """Get full thinking trace as text."""
        return "\n".join(step.display for step in self.thinking_steps)

    def get_citations_formatted(self) -> str:
        """Get formatted citations."""
        lines = []
        for i, c in enumerate(self.citations, 1):
            page_str = f", p. {c.page}" if c.page else ""
            lines.append(f"[{i}] {c.document}{page_str}")
            lines.append(f"    \"{c.text[:100]}...\"")
            lines.append(f"    Relevance: {c.relevance}")
            lines.append("")
        return "\n".join(lines)

    def complete(self, final_output: Optional[str] = None):
        """Mark investigation as complete."""
        self.status = "completed"
        self.completed_at = datetime.now()
        if final_output:
            self.findings["final_output"] = final_output

    def fail(self, error: str):
        """Mark investigation as failed."""
        self.status = "failed"
        self.error = error
        self.completed_at = datetime.now()

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get investigation duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive investigation summary."""
        lead_stats = self.get_lead_statistics()

        return {
            "id": self.id,
            "query": self.query,
            "status": self.status,
            "duration_seconds": self.duration_seconds,
            "metrics": {
                "documents_read": self.documents_read,
                "searches_performed": self.searches_performed,
                "citations": len(self.citations),
                "leads_investigated": lead_stats["investigated"],
                "leads_pending": lead_stats["pending"],
                "entities_found": len(self.entities),
                "facts_accumulated": len(self.findings.get("accumulated_facts", [])),
                "max_depth": self.max_depth_reached,
            },
            "hypothesis": self.hypothesis,
            "top_entities": [
                {"name": e.name, "type": e.entity_type, "mentions": e.mentions}
                for e in self.get_top_entities(5)
            ],
            "key_facts": self.findings.get("accumulated_facts", [])[:10],
        }

    def get_summary_text(self) -> str:
        """Get human-readable investigation summary."""
        summary = self.get_summary()
        lines = [
            f"Investigation Summary",
            f"=" * 50,
            f"Query: {summary['query']}",
            f"Status: {summary['status']}",
            f"Duration: {summary['duration_seconds']:.1f}s" if summary['duration_seconds'] else "Duration: In progress",
            f"",
            f"Metrics:",
            f"  Documents read: {summary['metrics']['documents_read']}",
            f"  Searches: {summary['metrics']['searches_performed']}",
            f"  Citations: {summary['metrics']['citations']}",
            f"  Leads: {summary['metrics']['leads_investigated']} investigated, {summary['metrics']['leads_pending']} pending",
            f"  Entities: {summary['metrics']['entities_found']}",
            f"  Facts: {summary['metrics']['facts_accumulated']}",
            f"",
            f"Hypothesis: {summary['hypothesis'] or 'None'}",
            f"",
            f"Top Entities:",
        ]

        for entity in summary["top_entities"]:
            lines.append(f"  - {entity['name']} ({entity['type']}, {entity['mentions']} mentions)")

        if not summary["top_entities"]:
            lines.append("  (none)")

        lines.append("")
        lines.append("Key Facts:")
        for i, fact in enumerate(summary["key_facts"], 1):
            lines.append(f"  {i}. {fact[:100]}...")

        if not summary["key_facts"]:
            lines.append("  (none)")

        return "\n".join(lines)

    def get_state_for_llm(self) -> dict[str, Any]:
        """Get state summary for LLM decision-making."""
        return {
            "files_searched": [s.details.get("files", []) for s in self.thinking_steps
                              if s.step_type == StepType.SEARCH and s.details],
            "docs_read": [s.details.get("file") for s in self.thinking_steps
                         if s.step_type == StepType.READING and s.details],
            "num_facts": len(self.findings.get("accumulated_facts", [])),
            "findings_summary": "; ".join(self.findings.get("accumulated_facts", [])[:5]) or "None yet",
            "citations_count": len(self.citations),
            "leads_pending": len(self.get_pending_leads()),
        }

    def to_dict(self) -> dict:
        """Serialize state to dictionary for checkpointing."""
        return {
            "id": self.id,
            "query": self.query,
            "repository_path": self.repository_path,
            "thinking_steps": [
                {
                    "id": s.id,
                    "step_type": s.step_type.value,
                    "content": s.content,
                    "details": s.details,
                    "timestamp": s.timestamp.isoformat(),
                    "duration_ms": s.duration_ms,
                    "depth": s.depth,
                }
                for s in self.thinking_steps
            ],
            "citations": [
                {
                    "id": c.id,
                    "document": c.document,
                    "page": c.page,
                    "text": c.text,
                    "context": c.context,
                    "relevance": c.relevance,
                    "timestamp": c.timestamp.isoformat(),
                    "url": c.url,
                    "mime": c.mime,
                    "source_type": c.source_type,
                }
                for c in self.citations
            ],
            "leads": [
                {
                    "id": l.id,
                    "description": l.description,
                    "source": l.source,
                    "params": l.params,
                    "lead_type": l.lead_type,
                    "investigated": l.investigated,
                    "findings": l.findings,
                    "started_at": l.started_at.isoformat() if l.started_at else None,
                    "finished_at": l.finished_at.isoformat() if l.finished_at else None,
                }
                for l in self.leads
            ],
            "entities": {
                key: {
                    "name": e.name,
                    "entity_type": e.entity_type,
                    "sources": e.sources,
                    "mentions": e.mentions,
                    "context": e.context,
                }
                for key, e in self.entities.items()
            },
            "cross_references": [
                {
                    "source_doc": ref.source_doc,
                    "target_doc": ref.target_doc,
                    "reference_text": ref.reference_text,
                    "page": ref.page,
                }
                for ref in self.cross_references
            ],
            "timeline": [
                {
                    "date_str": ev.date_str,
                    "date_parsed": ev.date_parsed.isoformat() if ev.date_parsed else None,
                    "description": ev.description,
                    "source_doc": ev.source_doc,
                    "page": ev.page,
                    "event_type": ev.event_type,
                }
                for ev in self.timeline
            ],
            "contradictions": [
                {
                    "id": c.id,
                    "statement1": c.statement1,
                    "source1": c.source1,
                    "statement2": c.statement2,
                    "source2": c.source2,
                    "contradiction_type": c.contradiction_type,
                    "severity": c.severity,
                    "notes": c.notes,
                }
                for c in self.contradictions
            ],
            "findings": self.findings,
            "hypothesis": self.hypothesis,
            "query_classification": self.query_classification,
            "external_triggers": {k: list(v) for k, v in self.external_triggers.items()},
            "documents_read": self.documents_read,
            "searches_performed": self.searches_performed,
            "recursion_depth": self.recursion_depth,
            "max_depth_reached": self.max_depth_reached,
            "api_calls": self.api_calls,
            "estimated_tokens": self.estimated_tokens,
            "status": self.status,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InvestigationState":
        """Deserialize state from dictionary."""
        state = cls(
            id=data["id"],
            query=data["query"],
            repository_path=data["repository_path"],
        )

        # Restore thinking steps
        for s in data.get("thinking_steps", []):
            step = ThinkingStep(
                id=s["id"],
                step_type=StepType(s["step_type"]),
                content=s["content"],
                details=s.get("details"),
                timestamp=datetime.fromisoformat(s["timestamp"]),
                duration_ms=s.get("duration_ms"),
                depth=s.get("depth", 0),
            )
            state.thinking_steps.append(step)

        # Restore citations
        for c in data.get("citations", []):
            citation = Citation(
                id=c["id"],
                document=c["document"],
                page=c.get("page"),
                text=c["text"],
                context=c["context"],
                relevance=c["relevance"],
                timestamp=datetime.fromisoformat(c["timestamp"]),
                url=c.get("url"),
                mime=c.get("mime"),
                source_type=c.get("source_type", "document"),
            )
            state.citations.append(citation)

        # Restore leads
        for l in data.get("leads", []):
            lead = Lead(
                id=l["id"],
                description=l["description"],
                source=l["source"],
                params=l.get("params") or {},
                lead_type=l.get("lead_type", "search"),
                investigated=l.get("investigated", False),
                findings=l.get("findings"),
                started_at=datetime.fromisoformat(l["started_at"]) if l.get("started_at") else None,
                finished_at=datetime.fromisoformat(l["finished_at"]) if l.get("finished_at") else None,
            )
            state.leads.append(lead)

        # Restore entities
        for key, e_data in data.get("entities", {}).items():
            entity = Entity(
                name=e_data["name"],
                entity_type=e_data["entity_type"],
                sources=e_data.get("sources", []),
                mentions=e_data.get("mentions", 1),
                context=e_data.get("context"),
            )
            state.entities[key] = entity

        # Restore cross-references
        for ref_data in data.get("cross_references", []):
            ref = CrossReference(
                source_doc=ref_data["source_doc"],
                target_doc=ref_data["target_doc"],
                reference_text=ref_data["reference_text"],
                page=ref_data.get("page"),
            )
            state.cross_references.append(ref)

        # Restore timeline
        for ev_data in data.get("timeline", []):
            event = TimelineEvent(
                date_str=ev_data["date_str"],
                date_parsed=datetime.fromisoformat(ev_data["date_parsed"]) if ev_data.get("date_parsed") else None,
                description=ev_data["description"],
                source_doc=ev_data["source_doc"],
                page=ev_data.get("page"),
                event_type=ev_data.get("event_type", "general"),
            )
            state.timeline.append(event)

        # Restore contradictions
        for c_data in data.get("contradictions", []):
            contradiction = Contradiction(
                id=c_data["id"],
                statement1=c_data["statement1"],
                source1=c_data["source1"],
                statement2=c_data["statement2"],
                source2=c_data["source2"],
                contradiction_type=c_data["contradiction_type"],
                severity=c_data["severity"],
                notes=c_data.get("notes", ""),
            )
            state.contradictions.append(contradiction)

        # Restore other fields
        state.findings = data.get("findings", {})
        # Rebuild _fact_hashes from restored facts
        for fact in state.findings.get("accumulated_facts", []):
            state._fact_hashes.add(" ".join(fact.lower().split()))
        state.hypothesis = data.get("hypothesis")
        state.query_classification = data.get("query_classification")

        # Restore external triggers (convert lists back to sets)
        if "external_triggers" in data:
            for key, values in data["external_triggers"].items():
                if key in state.external_triggers:
                    state.external_triggers[key] = set(values) if values else set()

        state.documents_read = data.get("documents_read", 0)
        state.searches_performed = data.get("searches_performed", 0)
        state.recursion_depth = data.get("recursion_depth", 0)
        state.max_depth_reached = data.get("max_depth_reached", 0)
        state.api_calls = data.get("api_calls", 0)
        state.estimated_tokens = data.get("estimated_tokens", 0)
        state.status = data.get("status", "initialized")
        state.error = data.get("error")
        state.started_at = datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
        state.completed_at = datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None

        return state

    def save_checkpoint(self, path: str | Path):
        """Save state to checkpoint file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> "InvestigationState":
        """Load state from checkpoint file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_checkpoint_to_s3(self, s3_client, bucket: str, key: str):
        """Save state to S3 as a checkpoint."""
        body = json.dumps(self.to_dict(), indent=2).encode("utf-8")
        s3_client.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")

    @classmethod
    def load_checkpoint_from_s3(cls, s3_client, bucket: str, key: str) -> "InvestigationState":
        """Load state from S3 checkpoint."""
        response = s3_client.get_object(Bucket=bucket, Key=key)
        data = json.loads(response["Body"].read().decode("utf-8"))
        return cls.from_dict(data)
