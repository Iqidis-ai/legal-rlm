"""State management for RLM investigation.

Tracks thinking steps, citations, and investigation progress.
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
    THINKING = "thinking"
    SEARCH = "search"
    READING = "reading"
    FINDING = "finding"
    REPLAN = "replan"
    VERIFY = "verify"
    SYNTHESIS = "synthesis"
    ERROR = "error"


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

    # Detect query type based on keywords
    # Priority: evaluative > comparative > analytical > procedural > factual
    # This ensures complex query types are detected before falling back to simple factual

    factual_keywords = ["what happened", "when did", "who was", "where did", "what is the", "what was the"]
    analytical_keywords = ["what does", "means", "implications", "significance", "interpret",
                          "key issues", "main issues", "issues in", "claims", "allegations",
                          "why did", "explain", "analyze", "analysis"]
    comparative_keywords = ["compare", "difference", "versus", "vs", "between", "contrast", "how does"]
    evaluative_keywords = ["valid", "strengths", "weaknesses", "assess", "evaluate", "review",
                          "strengths and weaknesses", "merits", "credibility"]
    procedural_keywords = ["how to", "steps", "process", "procedure", "timeline", "sequence"]

    # Check in priority order (most complex first)
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

    # Estimate complexity (1-5)
    word_count = len(query.split())
    complexity = 1
    if word_count > 10:
        complexity = 2
    if word_count > 20:
        complexity = 3
    if word_count > 30:
        complexity = 4
    if any(c in query for c in ["and", "or", "but", "however"]):
        complexity = min(complexity + 1, 5)

    # Extract potential entity names (capitalized words)
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
    relevance: str  # Why this was cited
    timestamp: datetime = field(default_factory=datetime.now)
    verified: Optional[bool] = None  # None=unchecked, True=verified, False=not found
    verification_note: Optional[str] = None

    @classmethod
    def create(
        cls,
        document: str,
        page: Optional[int],
        text: str,
        context: str,
        relevance: str,
    ) -> "Citation":
        return cls(
            id=str(uuid.uuid4())[:8],
            document=document,
            page=page,
            text=text,
            context=context,
            relevance=relevance,
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
    depth: int = 0  # Recursion depth

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
        }.get(self.step_type, "*")
        return f"{indent}{prefix} {self.content}"


@dataclass
class Entity:
    """An entity found during investigation."""
    name: str
    entity_type: str  # "person", "company", "date", "amount", "location", "other"
    sources: list[str] = field(default_factory=list)  # Documents where found
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
    confidence: float = 1.0  # How confident we are this is a real reference


@dataclass
class TimelineEvent:
    """An event on the investigation timeline."""
    date_str: str  # Original date string from document
    date_parsed: Optional[datetime] = None  # Parsed date if possible
    description: str = ""
    source_doc: str = ""
    page: Optional[int] = None
    event_type: str = "general"  # "filing", "agreement", "correspondence", "deadline", "general"

    def __post_init__(self):
        """Try to parse the date string."""
        if self.date_parsed is None and self.date_str:
            self.date_parsed = self._parse_date(self.date_str)

    @staticmethod
    def _parse_date(date_str: str) -> Optional[datetime]:
        """Try to parse various date formats."""
        import re

        # Common date patterns
        patterns = [
            (r"(\d{1,2})/(\d{1,2})/(\d{4})", "%m/%d/%Y"),
            (r"(\d{1,2})/(\d{1,2})/(\d{2})", "%m/%d/%y"),
            (r"(\d{4})-(\d{2})-(\d{2})", "%Y-%m-%d"),
            (r"(\w+)\s+(\d{1,2}),?\s+(\d{4})", None),  # "January 15, 2024"
        ]

        for pattern, fmt in patterns:
            match = re.search(pattern, date_str)
            if match:
                try:
                    if fmt:
                        return datetime.strptime(match.group(), fmt)
                    else:
                        # Handle month name format
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


# Evidence strength weights by source type
EVIDENCE_SOURCE_WEIGHTS = {
    # Legal documents - highest weight
    "contract": 1.0,
    "agreement": 1.0,
    "judgment": 1.0,
    "order": 0.95,
    "declaration": 0.9,
    "affidavit": 0.9,
    "complaint": 0.85,
    "motion": 0.8,
    "exhibit": 0.85,
    "amendment": 0.9,
    # Supporting documents
    "report": 0.7,
    "memo": 0.65,
    "memorandum": 0.65,
    "analysis": 0.7,
    "certificate": 0.75,
    # Correspondence - lower weight
    "letter": 0.5,
    "email": 0.4,
    "correspondence": 0.45,
    "note": 0.35,
    "draft": 0.3,
}


def get_source_weight(filename: str) -> float:
    """Get evidence weight based on document type."""
    filename_lower = filename.lower()
    for doc_type, weight in EVIDENCE_SOURCE_WEIGHTS.items():
        if doc_type in filename_lower:
            return weight
    return 0.5  # Default weight


@dataclass
class AnswerQualityAssessment:
    """Assessment of answer quality."""
    overall_score: float = 0.0  # 0-100
    quality_level: str = "unknown"  # excellent, good, adequate, poor
    factors: dict = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    @classmethod
    def assess(
        cls,
        answer: str,
        query: str,
        citations: list,
        verified_count: int,
        entities_found: int,
        facts_count: int,
        documents_read: int,
    ) -> "AnswerQualityAssessment":
        """Assess quality of a generated answer."""
        assessment = cls()
        factors = {}
        issues = []
        recommendations = []

        # Factor 1: Answer length appropriateness (0-15)
        answer_len = len(answer)
        if answer_len < 200:
            factors["length"] = 5
            issues.append("Answer may be too brief")
            recommendations.append("Provide more detailed analysis")
        elif answer_len < 500:
            factors["length"] = 10
        elif answer_len < 2000:
            factors["length"] = 15
        else:
            factors["length"] = 12
            issues.append("Answer may be overly verbose")

        # Factor 2: Citation coverage (0-25)
        citation_count = len(citations)
        if citation_count >= 10:
            factors["citations"] = 25
        elif citation_count >= 5:
            factors["citations"] = 20
        elif citation_count >= 2:
            factors["citations"] = 12
        else:
            factors["citations"] = 5
            issues.append("Limited citation support")
            recommendations.append("Find more supporting evidence")

        # Factor 3: Verification rate (0-20)
        if citation_count > 0:
            verified_rate = verified_count / citation_count
            factors["verification"] = verified_rate * 20
            if verified_rate < 0.5:
                issues.append("Many citations unverified")
                recommendations.append("Verify more citations against source documents")
        else:
            factors["verification"] = 0

        # Factor 4: Query term coverage (0-15)
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        query_coverage = len(query_words & answer_words) / len(query_words) if query_words else 0
        factors["query_coverage"] = query_coverage * 15
        if query_coverage < 0.5:
            issues.append("Answer may not fully address the query")

        # Factor 5: Entity support (0-10)
        if entities_found >= 5:
            factors["entities"] = 10
        elif entities_found >= 2:
            factors["entities"] = 7
        else:
            factors["entities"] = 3
            recommendations.append("Identify key entities in the documents")

        # Factor 6: Fact density (0-10)
        if facts_count >= 10:
            factors["facts"] = 10
        elif facts_count >= 5:
            factors["facts"] = 7
        else:
            factors["facts"] = 3

        # Factor 7: Document coverage (0-5)
        if documents_read >= 5:
            factors["document_coverage"] = 5
        elif documents_read >= 2:
            factors["document_coverage"] = 3
        else:
            factors["document_coverage"] = 1
            issues.append("Limited document coverage")
            recommendations.append("Review more source documents")

        # Calculate total
        total_score = sum(factors.values())

        # Determine quality level
        if total_score >= 80:
            quality_level = "excellent"
        elif total_score >= 60:
            quality_level = "good"
        elif total_score >= 40:
            quality_level = "adequate"
        else:
            quality_level = "poor"

        assessment.overall_score = round(total_score, 1)
        assessment.quality_level = quality_level
        assessment.factors = {k: round(v, 1) for k, v in factors.items()}
        assessment.issues = issues
        assessment.recommendations = recommendations

        return assessment

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "overall_score": self.overall_score,
            "quality_level": self.quality_level,
            "factors": self.factors,
            "issues": self.issues,
            "recommendations": self.recommendations,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AnswerQualityAssessment":
        """Deserialize from dictionary."""
        return cls(
            overall_score=data.get("overall_score", 0.0),
            quality_level=data.get("quality_level", "unknown"),
            factors=data.get("factors", {}),
            issues=data.get("issues", []),
            recommendations=data.get("recommendations", []),
        )

    def get_formatted(self) -> str:
        """Get formatted assessment report."""
        lines = [
            "Answer Quality Assessment",
            "=" * 40,
            f"Overall Score: {self.overall_score}/100 ({self.quality_level.upper()})",
            "",
            "Factor Breakdown:",
        ]

        for factor, score in sorted(self.factors.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  {factor}: {score}")

        if self.issues:
            lines.append("")
            lines.append("Issues Identified:")
            for issue in self.issues:
                lines.append(f"  - {issue}")

        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")

        return "\n".join(lines)


class FeedbackType(Enum):
    """Types of user feedback."""
    RELEVANT = "relevant"
    NOT_RELEVANT = "not_relevant"
    PARTIALLY_RELEVANT = "partially_relevant"
    HELPFUL = "helpful"
    NOT_HELPFUL = "not_helpful"


@dataclass
class RelevanceFeedback:
    """User feedback on a search result or finding."""
    id: str
    item_type: str  # "citation", "lead", "fact", "evidence"
    item_id: str  # ID of the item being rated
    feedback: FeedbackType
    query: str  # The query context when feedback was given
    timestamp: datetime = field(default_factory=datetime.now)
    notes: str = ""
    terms_to_boost: list[str] = field(default_factory=list)
    terms_to_demote: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        item_type: str,
        item_id: str,
        feedback: FeedbackType,
        query: str,
        notes: str = "",
    ) -> "RelevanceFeedback":
        return cls(
            id=str(uuid.uuid4())[:8],
            item_type=item_type,
            item_id=item_id,
            feedback=feedback,
            query=query,
            notes=notes,
        )


@dataclass
class EvidenceItem:
    """An item of evidence with strength scoring."""
    id: str
    claim: str
    source_doc: str
    page: Optional[int]
    quote: str
    strength_score: float = 0.0  # 0-100
    strength_level: str = "unknown"  # "strong", "moderate", "weak", "insufficient"
    factors: dict = field(default_factory=dict)
    corroborating_sources: list[str] = field(default_factory=list)
    contradicting_sources: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        claim: str,
        source_doc: str,
        quote: str,
        page: Optional[int] = None,
    ) -> "EvidenceItem":
        return cls(
            id=str(uuid.uuid4())[:8],
            claim=claim,
            source_doc=source_doc,
            page=page,
            quote=quote,
        )

    def calculate_strength(
        self,
        verified: bool = False,
        corroboration_count: int = 0,
        contradiction_count: int = 0,
        specificity: float = 0.5,  # 0-1, how specific is the quote
    ):
        """Calculate evidence strength score."""
        factors = {}

        # Factor 1: Source document type (0-30)
        source_weight = get_source_weight(self.source_doc)
        factors["source_type"] = source_weight * 30

        # Factor 2: Verification status (0-25)
        if verified:
            factors["verification"] = 25
        else:
            factors["verification"] = 5  # Unverified gets minimal score

        # Factor 3: Corroboration (0-25)
        if corroboration_count >= 3:
            factors["corroboration"] = 25
        elif corroboration_count >= 2:
            factors["corroboration"] = 20
        elif corroboration_count >= 1:
            factors["corroboration"] = 12
        else:
            factors["corroboration"] = 0

        # Factor 4: Contradiction penalty (-15 to 0)
        contradiction_penalty = min(contradiction_count * 5, 15)
        factors["contradictions"] = -contradiction_penalty

        # Factor 5: Specificity (0-20)
        factors["specificity"] = specificity * 20

        # Calculate total
        total_score = sum(factors.values())
        total_score = max(0, min(100, total_score))  # Clamp to 0-100

        # Determine level
        if total_score >= 75:
            level = "strong"
        elif total_score >= 50:
            level = "moderate"
        elif total_score >= 25:
            level = "weak"
        else:
            level = "insufficient"

        self.strength_score = round(total_score, 1)
        self.strength_level = level
        self.factors = {k: round(v, 1) for k, v in factors.items()}

    def add_corroboration(self, source_doc: str):
        """Add a corroborating source."""
        if source_doc not in self.corroborating_sources:
            self.corroborating_sources.append(source_doc)

    def add_contradiction(self, source_doc: str):
        """Add a contradicting source."""
        if source_doc not in self.contradicting_sources:
            self.contradicting_sources.append(source_doc)


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


@dataclass
class Lead:
    """A lead to investigate further."""
    id: str
    description: str
    source: str  # Where this lead came from
    priority: float = 0.5  # 0-1
    investigated: bool = False
    findings: Optional[str] = None
    search_term: Optional[str] = None  # Verbatim search term; bypasses _extract_search_term()
    focus_issue_id: Optional[str] = None  # Issue this lead targets, for coverage tracking

    @classmethod
    def create(
        cls,
        description: str,
        source: str,
        priority: float = 0.5,
        search_term: Optional[str] = None,
        focus_issue_id: Optional[str] = None,
    ) -> "Lead":
        return cls(
            id=str(uuid.uuid4())[:8],
            description=description,
            source=source,
            priority=priority,
            search_term=search_term,
            focus_issue_id=focus_issue_id,
        )


@dataclass
class InvestigationState:
    """
    Complete state of an RLM investigation.

    This is the "working memory" that persists across recursive calls.
    """
    id: str
    query: str
    repository_path: str

    # Progress tracking
    thinking_steps: list[ThinkingStep] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    leads: list[Lead] = field(default_factory=list)
    entities: dict[str, Entity] = field(default_factory=dict)  # key: normalized name
    cross_references: list[CrossReference] = field(default_factory=list)
    timeline: list[TimelineEvent] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    feedback: list[RelevanceFeedback] = field(default_factory=list)

    # Accumulated knowledge
    findings: dict[str, Any] = field(default_factory=dict)
    hypothesis: Optional[str] = None
    query_classification: Optional[dict] = None  # Result of classify_query()

    # In-flight dedup: tracks repo-relative paths currently on the cold path in this run.
    # Prevents the same document from being LLM-analyzed multiple times within a single
    # investigate() call when multiple parallel leads surface the same top-ranked file.
    # asyncio is single-threaded so a plain set is safe (check+add is atomic between awaits).
    _reading_in_progress: set = field(default_factory=set)

    # Metrics
    documents_read: int = 0
    searches_performed: int = 0
    recursion_depth: int = 0
    max_depth_reached: int = 0
    api_calls: int = 0
    estimated_tokens: int = 0
    facts_per_iteration: list[int] = field(default_factory=list)  # Track facts added per iteration for diminishing returns

    # Status
    status: str = "initialized"
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

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
    ) -> Optional[Citation]:
        """Add a citation if not duplicate."""
        # Check for duplicates
        text_normalized = " ".join(text.lower().split())[:100]
        for existing in self.citations:
            existing_normalized = " ".join(existing.text.lower().split())[:100]
            if (existing.document == document and
                existing.page == page and
                existing_normalized == text_normalized):
                return None  # Duplicate

        citation = Citation.create(
            document=document,
            page=page,
            text=text,
            context=context,
            relevance=relevance,
        )
        self.citations.append(citation)
        return citation

    def add_lead(
        self,
        description: str,
        source: str,
        priority: float = 0.5,
        search_term: Optional[str] = None,
        focus_issue_id: Optional[str] = None,
    ) -> Optional[Lead]:
        """Add a lead to investigate if not duplicate."""
        # Normalize description for comparison
        desc_normalized = " ".join(description.lower().split())

        for existing in self.leads:
            existing_normalized = " ".join(existing.description.lower().split())
            # Check for similar leads (80% overlap considered duplicate)
            if self._string_similarity(desc_normalized, existing_normalized) > 0.8:
                # Update priority if new lead has higher priority
                if priority > existing.priority:
                    existing.priority = priority
                # Upgrade focus_issue_id from None → non-None so dedup never discards linkage
                if existing.focus_issue_id is None and focus_issue_id is not None:
                    existing.focus_issue_id = focus_issue_id
                return None  # Duplicate

        lead = Lead.create(description, source, priority,
                           search_term=search_term, focus_issue_id=focus_issue_id)
        self.leads.append(lead)
        return lead

    @staticmethod
    def _string_similarity(s1: str, s2: str) -> float:
        """Calculate simple string similarity ratio."""
        if not s1 or not s2:
            return 0.0
        if s1 == s2:
            return 1.0

        # Simple word overlap similarity
        words1 = set(s1.split())
        words2 = set(s2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)

    def add_fact(self, fact: str) -> bool:
        """Add a fact if not duplicate. Returns True if added."""
        if "accumulated_facts" not in self.findings:
            self.findings["accumulated_facts"] = []

        fact_normalized = " ".join(fact.lower().split())

        for existing in self.findings["accumulated_facts"]:
            existing_normalized = " ".join(existing.lower().split())
            if self._string_similarity(fact_normalized, existing_normalized) > 0.7:
                return False  # Duplicate

        self.findings["accumulated_facts"].append(fact)
        return True

    def add_facts(self, facts: list[str]) -> int:
        """Add multiple facts with deduplication. Returns count of added facts."""
        added = 0
        for fact in facts:
            if self.add_fact(fact):
                added += 1
        return added

    def add_entity(
        self,
        name: str,
        entity_type: str,
        source: str,
        context: Optional[str] = None,
    ) -> Entity:
        """Add or update an entity."""
        # Normalize name for lookup
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
        """Get uninvestigated leads sorted by priority."""
        pending = [l for l in self.leads if not l.investigated]
        return sorted(pending, key=lambda l: l.priority, reverse=True)

    def reprioritize_leads(self):
        """Reprioritize leads based on current investigation context."""
        if not self.leads:
            return

        # Boost leads related to frequently mentioned entities
        top_entities = {e.name.lower() for e in self.get_top_entities(5)}

        # Boost leads related to hypothesis
        hypothesis_words = set()
        if self.hypothesis:
            hypothesis_words = set(self.hypothesis.lower().split())

        for lead in self.leads:
            if lead.investigated:
                continue

            boost = 0.0
            desc_lower = lead.description.lower()

            # Boost issue-targeted and predicate-driven leads (SO-4 coverage accuracy).
            # These leads were generated specifically to fill gaps in the issue predicate
            # tree, so they should be investigated before generic exploration leads.
            if lead.focus_issue_id is not None:
                boost += 0.2
            if lead.source == "predicate":
                boost += 0.1

            # Boost if mentions top entity
            for entity_name in top_entities:
                if entity_name in desc_lower:
                    boost += 0.15
                    break

            # Boost if related to hypothesis
            desc_words = set(desc_lower.split())
            overlap = len(desc_words & hypothesis_words)
            if overlap >= 2:
                boost += 0.1

            # Apply boost (cap at 1.0)
            lead.priority = min(1.0, lead.priority + boost)

    def get_lead_statistics(self) -> dict[str, Any]:
        """Get statistics about leads."""
        total = len(self.leads)
        investigated = sum(1 for l in self.leads if l.investigated)
        pending = total - investigated
        avg_priority = sum(l.priority for l in self.leads) / total if total > 0 else 0

        return {
            "total": total,
            "investigated": investigated,
            "pending": pending,
            "avg_priority": round(avg_priority, 2),
        }

    def get_progress(self) -> dict[str, Any]:
        """Get investigation progress metrics."""
        lead_stats = self.get_lead_statistics()

        # Estimate progress (0-100)
        progress = 0
        if self.status == "completed":
            progress = 100
        elif self.status == "failed":
            progress = 0
        else:
            # Weight different factors
            if lead_stats["total"] > 0:
                lead_progress = (lead_stats["investigated"] / lead_stats["total"]) * 40
            else:
                lead_progress = 0

            citation_progress = min(len(self.citations) / 10, 1.0) * 30
            depth_progress = min(self.max_depth_reached / 3, 1.0) * 20
            doc_progress = min(self.documents_read / 5, 1.0) * 10

            progress = int(lead_progress + citation_progress + depth_progress + doc_progress)

        elapsed = None
        if self.started_at:
            if self.completed_at:
                elapsed = (self.completed_at - self.started_at).total_seconds()
            else:
                elapsed = (datetime.now() - self.started_at).total_seconds()

        return {
            "progress_percent": min(progress, 100),
            "status": self.status,
            "elapsed_seconds": elapsed,
            "documents_read": self.documents_read,
            "searches_performed": self.searches_performed,
            "citations": len(self.citations),
            "leads_investigated": lead_stats["investigated"],
            "leads_pending": lead_stats["pending"],
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
        confidence: float = 1.0,
    ) -> Optional[CrossReference]:
        """Add a cross-reference if not duplicate."""
        # Check for duplicates
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
            confidence=confidence,
        )
        self.cross_references.append(ref)
        return ref

    def detect_cross_references(self, text: str, source_doc: str, known_docs: list[str], page: Optional[int] = None):
        """Detect references to other documents in text."""
        text_lower = text.lower()

        # Reference patterns to look for
        ref_patterns = [
            "see exhibit", "per exhibit", "attached as exhibit",
            "referenced in", "as stated in", "according to",
            "see attached", "per the", "pursuant to",
        ]

        for doc_name in known_docs:
            if doc_name == source_doc:
                continue

            # Check if document name appears in text
            doc_name_lower = doc_name.lower()
            doc_base = doc_name_lower.rsplit('.', 1)[0]  # Remove extension

            if doc_base in text_lower or doc_name_lower in text_lower:
                # Find the context around the reference
                idx = text_lower.find(doc_base)
                if idx == -1:
                    idx = text_lower.find(doc_name_lower)

                start = max(0, idx - 50)
                end = min(len(text), idx + len(doc_base) + 50)
                context = text[start:end]

                # Check if it's a real reference (has reference pattern nearby)
                confidence = 0.5  # Base confidence
                for pattern in ref_patterns:
                    if pattern in text_lower[max(0, idx-100):idx+100]:
                        confidence = 0.9
                        break

                self.add_cross_reference(
                    source_doc=source_doc,
                    target_doc=doc_name,
                    reference_text=context,
                    page=page,
                    confidence=confidence,
                )

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
        # Check for duplicates
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
        # Sort by parsed date (None dates go to end)
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

        # Group by severity
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

    def add_evidence(
        self,
        claim: str,
        source_doc: str,
        quote: str,
        page: Optional[int] = None,
        verified: bool = False,
        specificity: float = 0.5,
    ) -> Optional[EvidenceItem]:
        """Add an evidence item with strength calculation."""
        # Check for duplicates
        claim_normalized = " ".join(claim.lower().split())[:100]
        for existing in self.evidence:
            existing_normalized = " ".join(existing.claim.lower().split())[:100]
            if claim_normalized == existing_normalized:
                return None

        evidence = EvidenceItem.create(
            claim=claim,
            source_doc=source_doc,
            quote=quote,
            page=page,
        )

        # Calculate initial strength
        evidence.calculate_strength(
            verified=verified,
            corroboration_count=0,
            contradiction_count=0,
            specificity=specificity,
        )

        self.evidence.append(evidence)
        return evidence

    def get_evidence_by_strength(self, level: str) -> list[EvidenceItem]:
        """Get evidence filtered by strength level."""
        return [e for e in self.evidence if e.strength_level == level]

    def get_strong_evidence(self) -> list[EvidenceItem]:
        """Get all strong evidence items."""
        return [e for e in self.evidence if e.strength_level in ["strong", "moderate"]]

    def recalculate_evidence_strength(self):
        """Recalculate all evidence strength based on corroboration/contradictions."""
        # Find corroborating claims
        for i, ev1 in enumerate(self.evidence):
            corroboration_count = 0
            contradiction_count = 0

            for j, ev2 in enumerate(self.evidence):
                if i == j:
                    continue

                # Simple check: same claim from different sources
                claim1_words = set(ev1.claim.lower().split())
                claim2_words = set(ev2.claim.lower().split())
                overlap = len(claim1_words & claim2_words) / max(len(claim1_words), 1)

                if overlap > 0.5 and ev1.source_doc != ev2.source_doc:
                    ev1.add_corroboration(ev2.source_doc)
                    corroboration_count += 1

            # Check contradictions
            for contradiction in self.contradictions:
                if (ev1.source_doc in [contradiction.source1, contradiction.source2]):
                    contradiction_count += 1

            # Recalculate with corroboration
            ev1.calculate_strength(
                verified=any(c.verified for c in self.citations if c.document == ev1.source_doc),
                corroboration_count=corroboration_count,
                contradiction_count=contradiction_count,
                specificity=len(ev1.quote) / 500,  # Rough specificity based on quote length
            )

    def get_evidence_summary(self) -> dict[str, Any]:
        """Get evidence strength summary."""
        total = len(self.evidence)
        strong = len([e for e in self.evidence if e.strength_level == "strong"])
        moderate = len([e for e in self.evidence if e.strength_level == "moderate"])
        weak = len([e for e in self.evidence if e.strength_level == "weak"])
        insufficient = len([e for e in self.evidence if e.strength_level == "insufficient"])

        avg_score = sum(e.strength_score for e in self.evidence) / total if total > 0 else 0

        return {
            "total": total,
            "strong": strong,
            "moderate": moderate,
            "weak": weak,
            "insufficient": insufficient,
            "average_score": round(avg_score, 1),
        }

    def get_evidence_formatted(self) -> str:
        """Get formatted evidence list by strength."""
        if not self.evidence:
            return "No evidence collected"

        lines = ["Evidence Summary:", "=" * 40]

        for level in ["strong", "moderate", "weak", "insufficient"]:
            items = self.get_evidence_by_strength(level)
            if items:
                lines.append(f"\n[{level.upper()}] ({len(items)} items)")
                for ev in sorted(items, key=lambda x: x.strength_score, reverse=True)[:5]:
                    lines.append(f"  Score: {ev.strength_score}")
                    lines.append(f"  Claim: {ev.claim[:80]}...")
                    lines.append(f"  Source: {ev.source_doc}, p.{ev.page}")
                    if ev.corroborating_sources:
                        lines.append(f"  Corroborated by: {', '.join(ev.corroborating_sources[:3])}")
                    lines.append("")

        return "\n".join(lines)

    def add_feedback(
        self,
        item_type: str,
        item_id: str,
        feedback_type: FeedbackType,
        notes: str = "",
    ) -> RelevanceFeedback:
        """Add user feedback on a finding."""
        fb = RelevanceFeedback.create(
            item_type=item_type,
            item_id=item_id,
            feedback=feedback_type,
            query=self.query,
            notes=notes,
        )

        # Extract terms from the item for boosting/demotion
        item_text = self._get_item_text(item_type, item_id)
        if item_text:
            terms = self._extract_key_terms(item_text)
            if feedback_type in (FeedbackType.RELEVANT, FeedbackType.HELPFUL):
                fb.terms_to_boost = terms
            elif feedback_type in (FeedbackType.NOT_RELEVANT, FeedbackType.NOT_HELPFUL):
                fb.terms_to_demote = terms

        self.feedback.append(fb)
        return fb

    def _get_item_text(self, item_type: str, item_id: str) -> Optional[str]:
        """Get text content of an item by type and ID."""
        if item_type == "citation":
            for c in self.citations:
                if c.id == item_id:
                    return c.text
        elif item_type == "lead":
            for l in self.leads:
                if l.id == item_id:
                    return l.description
        elif item_type == "evidence":
            for e in self.evidence:
                if e.id == item_id:
                    return e.claim + " " + e.quote
        elif item_type == "fact":
            facts = self.findings.get("accumulated_facts", [])
            if item_id.isdigit() and int(item_id) < len(facts):
                return facts[int(item_id)]
        return None

    def _extract_key_terms(self, text: str, n: int = 5) -> list[str]:
        """Extract key terms from text."""
        import re
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        # Simple frequency-based extraction
        from collections import Counter
        stop_words = {"that", "this", "with", "from", "have", "been", "their", "there", "which", "what", "when", "where"}
        words = [w for w in words if w not in stop_words]
        counts = Counter(words)
        return [term for term, _ in counts.most_common(n)]

    def get_feedback_by_type(self, feedback_type: FeedbackType) -> list[RelevanceFeedback]:
        """Get all feedback of a specific type."""
        return [f for f in self.feedback if f.feedback == feedback_type]

    def get_boosted_terms(self) -> list[str]:
        """Get terms that should be boosted based on positive feedback."""
        terms: list[str] = []
        for fb in self.feedback:
            if fb.feedback in (FeedbackType.RELEVANT, FeedbackType.HELPFUL):
                terms.extend(fb.terms_to_boost)
        # Deduplicate while preserving order
        seen = set()
        result = []
        for t in terms:
            if t not in seen:
                seen.add(t)
                result.append(t)
        return result

    def get_demoted_terms(self) -> list[str]:
        """Get terms that should be demoted based on negative feedback."""
        terms: list[str] = []
        for fb in self.feedback:
            if fb.feedback in (FeedbackType.NOT_RELEVANT, FeedbackType.NOT_HELPFUL):
                terms.extend(fb.terms_to_demote)
        seen = set()
        result = []
        for t in terms:
            if t not in seen:
                seen.add(t)
                result.append(t)
        return result

    def get_feedback_summary(self) -> dict[str, Any]:
        """Get feedback statistics."""
        total = len(self.feedback)
        positive = sum(1 for f in self.feedback if f.feedback in (FeedbackType.RELEVANT, FeedbackType.HELPFUL))
        negative = sum(1 for f in self.feedback if f.feedback in (FeedbackType.NOT_RELEVANT, FeedbackType.NOT_HELPFUL))
        partial = sum(1 for f in self.feedback if f.feedback == FeedbackType.PARTIALLY_RELEVANT)

        return {
            "total": total,
            "positive": positive,
            "negative": negative,
            "partial": partial,
            "boosted_terms": self.get_boosted_terms()[:10],
            "demoted_terms": self.get_demoted_terms()[:10],
        }

    def apply_feedback_to_leads(self):
        """Adjust lead priorities based on feedback."""
        boosted = set(self.get_boosted_terms())
        demoted = set(self.get_demoted_terms())

        for lead in self.leads:
            if lead.investigated:
                continue

            desc_words = set(lead.description.lower().split())

            # Boost if contains boosted terms
            boost_count = len(desc_words & boosted)
            if boost_count > 0:
                lead.priority = min(1.0, lead.priority + boost_count * 0.1)

            # Demote if contains demoted terms
            demote_count = len(desc_words & demoted)
            if demote_count > 0:
                lead.priority = max(0.0, lead.priority - demote_count * 0.1)

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
            verify_str = ""
            if c.verified is True:
                verify_str = " [VERIFIED]"
            elif c.verified is False:
                verify_str = " [UNVERIFIED]"
            lines.append(f"[{i}] {c.document}{page_str}{verify_str}")
            lines.append(f"    \"{c.text[:100]}...\"")
            lines.append(f"    Relevance: {c.relevance}")
            if c.verification_note:
                lines.append(f"    Note: {c.verification_note}")
            lines.append("")
        return "\n".join(lines)

    def get_unverified_citations(self) -> list[Citation]:
        """Get citations that haven't been verified yet."""
        return [c for c in self.citations if c.verified is None]

    def get_verification_stats(self) -> dict[str, int]:
        """Get verification statistics."""
        verified = sum(1 for c in self.citations if c.verified is True)
        unverified = sum(1 for c in self.citations if c.verified is False)
        unchecked = sum(1 for c in self.citations if c.verified is None)
        return {
            "verified": verified,
            "unverified": unverified,
            "unchecked": unchecked,
            "total": len(self.citations),
        }

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

    def interrupt(self):
        """Mark investigation as interrupted by user stop. Partial state preserved."""
        self.status = "interrupted"
        self.completed_at = datetime.now()

    def assess_answer_quality(self) -> AnswerQualityAssessment:
        """Assess quality of the final answer."""
        answer = self.findings.get("final_output", "")
        verification_stats = self.get_verification_stats()

        return AnswerQualityAssessment.assess(
            answer=answer,
            query=self.query,
            citations=self.citations,
            verified_count=verification_stats["verified"],
            entities_found=len(self.entities),
            facts_count=len(self.findings.get("accumulated_facts", [])),
            documents_read=self.documents_read,
        )

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get investigation duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive investigation summary."""
        lead_stats = self.get_lead_statistics()
        verification_stats = self.get_verification_stats()
        confidence = self.get_confidence_score()

        return {
            "id": self.id,
            "query": self.query,
            "status": self.status,
            "duration_seconds": self.duration_seconds,
            "confidence": confidence,
            "metrics": {
                "documents_read": self.documents_read,
                "searches_performed": self.searches_performed,
                "citations": len(self.citations),
                "verified_citations": verification_stats["verified"],
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

    def get_confidence_score(self) -> dict[str, Any]:
        """Calculate investigation confidence score (0-100)."""
        factors = {}

        # Factor 1: Citation coverage (0-25)
        citation_count = len(self.citations)
        factors["citations"] = min(citation_count * 2.5, 25)

        # Factor 2: Verification rate (0-25)
        verification_stats = self.get_verification_stats()
        if verification_stats["total"] > 0:
            verified_rate = verification_stats["verified"] / verification_stats["total"]
            factors["verification"] = verified_rate * 25
        else:
            factors["verification"] = 0

        # Factor 3: Document coverage (0-20)
        if self.documents_read >= 5:
            factors["documents"] = 20
        else:
            factors["documents"] = self.documents_read * 4

        # Factor 4: Entity discovery (0-15)
        entity_count = len(self.entities)
        factors["entities"] = min(entity_count * 3, 15)

        # Factor 5: Fact accumulation (0-15)
        fact_count = len(self.findings.get("accumulated_facts", []))
        factors["facts"] = min(fact_count * 1.5, 15)

        total_score = sum(factors.values())

        # Determine confidence level
        if total_score >= 80:
            level = "high"
        elif total_score >= 50:
            level = "medium"
        elif total_score >= 25:
            level = "low"
        else:
            level = "insufficient"

        return {
            "score": round(total_score, 1),
            "level": level,
            "factors": {k: round(v, 1) for k, v in factors.items()},
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
            f"  Citations: {summary['metrics']['citations']} ({summary['metrics']['verified_citations']} verified)",
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
                    "verified": c.verified,
                    "verification_note": c.verification_note,
                }
                for c in self.citations
            ],
            "leads": [
                {
                    "id": l.id,
                    "description": l.description,
                    "source": l.source,
                    "priority": l.priority,
                    "investigated": l.investigated,
                    "findings": l.findings,
                    "search_term": l.search_term,
                    "focus_issue_id": l.focus_issue_id,
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
            "evidence": [
                {
                    "id": ev.id,
                    "claim": ev.claim,
                    "source_doc": ev.source_doc,
                    "page": ev.page,
                    "quote": ev.quote,
                    "strength_score": ev.strength_score,
                    "strength_level": ev.strength_level,
                    "factors": ev.factors,
                    "corroborating_sources": ev.corroborating_sources,
                    "contradicting_sources": ev.contradicting_sources,
                }
                for ev in self.evidence
            ],
            "feedback": [
                {
                    "id": fb.id,
                    "item_type": fb.item_type,
                    "item_id": fb.item_id,
                    "feedback": fb.feedback.value,
                    "query": fb.query,
                    "timestamp": fb.timestamp.isoformat(),
                    "notes": fb.notes,
                    "terms_to_boost": fb.terms_to_boost,
                    "terms_to_demote": fb.terms_to_demote,
                }
                for fb in self.feedback
            ],
            "findings": self.findings,
            "hypothesis": self.hypothesis,
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
                verified=c.get("verified"),
                verification_note=c.get("verification_note"),
            )
            state.citations.append(citation)

        # Restore leads
        for l in data.get("leads", []):
            lead = Lead(
                id=l["id"],
                description=l["description"],
                source=l["source"],
                priority=l.get("priority", 0.5),
                investigated=l.get("investigated", False),
                findings=l.get("findings"),
                search_term=l.get("search_term"),
                focus_issue_id=l.get("focus_issue_id"),
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

        # Restore evidence
        for ev_data in data.get("evidence", []):
            evidence = EvidenceItem(
                id=ev_data["id"],
                claim=ev_data["claim"],
                source_doc=ev_data["source_doc"],
                page=ev_data.get("page"),
                quote=ev_data["quote"],
                strength_score=ev_data.get("strength_score", 0.0),
                strength_level=ev_data.get("strength_level", "unknown"),
                factors=ev_data.get("factors", {}),
                corroborating_sources=ev_data.get("corroborating_sources", []),
                contradicting_sources=ev_data.get("contradicting_sources", []),
            )
            state.evidence.append(evidence)

        # Restore feedback
        for fb_data in data.get("feedback", []):
            feedback = RelevanceFeedback(
                id=fb_data["id"],
                item_type=fb_data["item_type"],
                item_id=fb_data["item_id"],
                feedback=FeedbackType(fb_data["feedback"]),
                query=fb_data["query"],
                timestamp=datetime.fromisoformat(fb_data["timestamp"]),
                notes=fb_data.get("notes", ""),
                terms_to_boost=fb_data.get("terms_to_boost", []),
                terms_to_demote=fb_data.get("terms_to_demote", []),
            )
            state.feedback.append(feedback)

        # Restore other fields
        state.findings = data.get("findings", {})
        state.hypothesis = data.get("hypothesis")
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
