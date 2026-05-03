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


class ResearchMode(Enum):
    """User-selected investigation depth/budget profiles."""
    SIMPLE = "simple"
    DEEP = "deep"
    SEBIH_SPECIAL = "sebih_special"


class WorkflowKind(Enum):
    """High-level shape of work the engine is performing."""
    ANALYSIS = "analysis"
    DRAFTING = "drafting"
    SOLUTION = "solution"
    LOOKUP = "lookup"
    TRACE = "trace"
    STEERING = "steering"
    COMPARISON = "comparison"
    CLARIFICATION = "clarification"


def normalize_research_mode(
    value: Any,
    *,
    default: "str | ResearchMode" = ResearchMode.DEEP,
    strict: bool = False,
) -> str:
    """Normalize a research mode string/enum to a stable lowercase value."""
    default_value = default.value if isinstance(default, ResearchMode) else str(default)
    if value is None:
        return default_value
    if isinstance(value, ResearchMode):
        return value.value

    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "simple": ResearchMode.SIMPLE.value,
        "deep": ResearchMode.DEEP.value,
        "sebih_special": ResearchMode.SEBIH_SPECIAL.value,
        "sebihspecial": ResearchMode.SEBIH_SPECIAL.value,
        "special": ResearchMode.SEBIH_SPECIAL.value,
    }
    resolved = aliases.get(normalized)
    if resolved is not None:
        return resolved
    if strict:
        valid = ", ".join(mode.value for mode in ResearchMode)
        raise ValueError(f"Invalid research_mode '{value}'. Expected one of: {valid}")
    return default_value


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
    # MVI-5 per-lead EV gating. Coarse cost class and expected
    # coverage gain so the termination controller can spend where
    # it advances answerability per dollar. Both default to 0 so
    # old code paths that don't populate EV fall back to the
    # priority-threshold gate — the engine's _viable_leads handles
    # that fallback. The lead planner populates these when it knows
    # the action class and target-issue weakness.
    expected_cost_usd: float = 0.0
    expected_coverage_gain: float = 0.0

    @property
    def ev_score(self) -> float:
        """Return expected answerability advance per dollar. Zero when
        expected_cost is zero (e.g. stale leads with no cost set)."""
        if self.expected_cost_usd <= 0:
            return 0.0
        return self.expected_coverage_gain / self.expected_cost_usd

    @classmethod
    def create(
        cls,
        description: str,
        source: str,
        priority: float = 0.5,
        search_term: Optional[str] = None,
        focus_issue_id: Optional[str] = None,
        expected_cost_usd: Optional[float] = None,
        expected_coverage_gain: Optional[float] = None,
    ) -> "Lead":
        kwargs = dict(
            id=str(uuid.uuid4())[:8],
            description=description,
            source=source,
            priority=priority,
            search_term=search_term,
            focus_issue_id=focus_issue_id,
        )
        if expected_cost_usd is not None:
            kwargs["expected_cost_usd"] = expected_cost_usd
        if expected_coverage_gain is not None:
            kwargs["expected_coverage_gain"] = expected_coverage_gain
        return cls(**kwargs)


@dataclass
class RunObjective:
    """The user-visible objective a run is trying to satisfy."""
    id: str
    workflow_kind: str
    user_goal: str
    output_shape: str
    audience: str = "internal"
    policy_audience: str = "clean"
    success_criteria: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    source_query: Optional[str] = None

    @classmethod
    def create(
        cls,
        *,
        user_goal: str,
        output_shape: str,
        workflow_kind: str = WorkflowKind.ANALYSIS.value,
        audience: str = "internal",
        policy_audience: str = "clean",
        success_criteria: Optional[list[str]] = None,
        constraints: Optional[list[str]] = None,
        source_query: Optional[str] = None,
    ) -> "RunObjective":
        return cls(
            id=str(uuid.uuid4())[:8],
            workflow_kind=str(workflow_kind or WorkflowKind.ANALYSIS.value),
            user_goal=user_goal,
            output_shape=output_shape,
            audience=audience,
            policy_audience=policy_audience,
            success_criteria=list(success_criteria or []),
            constraints=list(constraints or []),
            source_query=source_query,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "workflow_kind": self.workflow_kind,
            "user_goal": self.user_goal,
            "output_shape": self.output_shape,
            "audience": self.audience,
            "policy_audience": self.policy_audience,
            "success_criteria": self.success_criteria,
            "constraints": self.constraints,
            "source_query": self.source_query,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunObjective":
        return cls(
            id=str(data.get("id") or str(uuid.uuid4())[:8]),
            workflow_kind=str(data.get("workflow_kind") or WorkflowKind.ANALYSIS.value),
            user_goal=str(data.get("user_goal") or ""),
            output_shape=str(data.get("output_shape") or ""),
            audience=str(data.get("audience") or "internal"),
            policy_audience=str(data.get("policy_audience") or "clean"),
            success_criteria=list(data.get("success_criteria") or []),
            constraints=list(data.get("constraints") or []),
            source_query=data.get("source_query"),
        )


@dataclass
class Obligation:
    """A condition that must be satisfied for an output to be acceptable."""
    id: str
    description: str
    obligation_type: str
    required: bool = True
    blocking: bool = True
    satisfied: bool = False
    source_refs: list[str] = field(default_factory=list)
    validator: Optional[str] = None
    status_note: Optional[str] = None

    @classmethod
    def create(
        cls,
        *,
        description: str,
        obligation_type: str,
        required: bool = True,
        blocking: bool = True,
        source_refs: Optional[list[str]] = None,
        validator: Optional[str] = None,
    ) -> "Obligation":
        return cls(
            id=str(uuid.uuid4())[:8],
            description=description,
            obligation_type=obligation_type,
            required=required,
            blocking=blocking,
            source_refs=list(source_refs or []),
            validator=validator,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "obligation_type": self.obligation_type,
            "required": self.required,
            "blocking": self.blocking,
            "satisfied": self.satisfied,
            "source_refs": self.source_refs,
            "validator": self.validator,
            "status_note": self.status_note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Obligation":
        return cls(
            id=str(data.get("id") or str(uuid.uuid4())[:8]),
            description=str(data.get("description") or ""),
            obligation_type=str(data.get("obligation_type") or "general"),
            required=bool(data.get("required", True)),
            blocking=bool(data.get("blocking", True)),
            satisfied=bool(data.get("satisfied", False)),
            source_refs=list(data.get("source_refs") or []),
            validator=data.get("validator"),
            status_note=data.get("status_note"),
        )


@dataclass
class WorkingSet:
    """Object ids and dependency metadata a workflow is allowed to reason over."""
    verified_assertion_ids: list[str] = field(default_factory=list)
    candidate_assertion_ids: list[str] = field(default_factory=list)
    issue_ids: list[str] = field(default_factory=list)
    gap_ids: list[str] = field(default_factory=list)
    document_ids: list[str] = field(default_factory=list)
    authority_ids: list[str] = field(default_factory=list)
    assumption_ids: list[str] = field(default_factory=list)
    dependency_manifest_hash: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified_assertion_ids": self.verified_assertion_ids,
            "candidate_assertion_ids": self.candidate_assertion_ids,
            "issue_ids": self.issue_ids,
            "gap_ids": self.gap_ids,
            "document_ids": self.document_ids,
            "authority_ids": self.authority_ids,
            "assumption_ids": self.assumption_ids,
            "dependency_manifest_hash": self.dependency_manifest_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkingSet":
        return cls(
            verified_assertion_ids=list(data.get("verified_assertion_ids") or []),
            candidate_assertion_ids=list(data.get("candidate_assertion_ids") or []),
            issue_ids=list(data.get("issue_ids") or []),
            gap_ids=list(data.get("gap_ids") or []),
            document_ids=list(data.get("document_ids") or []),
            authority_ids=list(data.get("authority_ids") or []),
            assumption_ids=list(data.get("assumption_ids") or []),
            dependency_manifest_hash=data.get("dependency_manifest_hash"),
        )


@dataclass
class PlanAction:
    """One planned operator step for satisfying workflow obligations."""
    id: str
    action_type: str
    description: str
    target_obligation_ids: list[str] = field(default_factory=list)
    input_refs: list[str] = field(default_factory=list)
    expected_output: Optional[str] = None
    status: str = "planned"
    result_refs: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        action_type: str,
        description: str,
        target_obligation_ids: Optional[list[str]] = None,
        input_refs: Optional[list[str]] = None,
        expected_output: Optional[str] = None,
    ) -> "PlanAction":
        return cls(
            id=str(uuid.uuid4())[:8],
            action_type=action_type,
            description=description,
            target_obligation_ids=list(target_obligation_ids or []),
            input_refs=list(input_refs or []),
            expected_output=expected_output,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "action_type": self.action_type,
            "description": self.description,
            "target_obligation_ids": self.target_obligation_ids,
            "input_refs": self.input_refs,
            "expected_output": self.expected_output,
            "status": self.status,
            "result_refs": self.result_refs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanAction":
        return cls(
            id=str(data.get("id") or str(uuid.uuid4())[:8]),
            action_type=str(data.get("action_type") or "unknown"),
            description=str(data.get("description") or ""),
            target_obligation_ids=list(data.get("target_obligation_ids") or []),
            input_refs=list(data.get("input_refs") or []),
            expected_output=data.get("expected_output"),
            status=str(data.get("status") or "planned"),
            result_refs=list(data.get("result_refs") or []),
        )


@dataclass
class ValidationResult:
    """Result of checking an output or working set against obligations."""
    validator: str
    passed: bool
    score: float = 0.0
    blocking_issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    obligation_status: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "validator": self.validator,
            "passed": self.passed,
            "score": self.score,
            "blocking_issues": self.blocking_issues,
            "warnings": self.warnings,
            "obligation_status": self.obligation_status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidationResult":
        return cls(
            validator=str(data.get("validator") or "unknown"),
            passed=bool(data.get("passed", False)),
            score=float(data.get("score", 0.0) or 0.0),
            blocking_issues=list(data.get("blocking_issues") or []),
            warnings=list(data.get("warnings") or []),
            obligation_status=dict(data.get("obligation_status") or {}),
        )


@dataclass
class OutputEnvelope:
    """Auditable wrapper around user-facing output text."""
    id: str
    output_text: str
    workflow_kind: str
    output_shape: str
    emitter: str
    objective_id: Optional[str] = None
    working_set_hash: Optional[str] = None
    dependency_manifest_hash: Optional[str] = None
    validation_results: list[ValidationResult] = field(default_factory=list)
    blocking_issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    review_required: bool = False
    created_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def create(
        cls,
        *,
        output_text: str,
        workflow_kind: str,
        output_shape: str,
        emitter: str,
        objective_id: Optional[str] = None,
        working_set_hash: Optional[str] = None,
        dependency_manifest_hash: Optional[str] = None,
        validation_results: Optional[list[ValidationResult]] = None,
        review_required: bool = False,
    ) -> "OutputEnvelope":
        results = list(validation_results or [])
        return cls(
            id=str(uuid.uuid4())[:8],
            output_text=output_text,
            workflow_kind=workflow_kind,
            output_shape=output_shape,
            emitter=emitter,
            objective_id=objective_id,
            working_set_hash=working_set_hash,
            dependency_manifest_hash=dependency_manifest_hash,
            validation_results=results,
            blocking_issues=[
                issue for result in results for issue in result.blocking_issues
            ],
            warnings=[warning for result in results for warning in result.warnings],
            review_required=review_required,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "output_text": self.output_text,
            "workflow_kind": self.workflow_kind,
            "output_shape": self.output_shape,
            "emitter": self.emitter,
            "objective_id": self.objective_id,
            "working_set_hash": self.working_set_hash,
            "dependency_manifest_hash": self.dependency_manifest_hash,
            "validation_results": [
                result.to_dict() for result in self.validation_results
            ],
            "blocking_issues": self.blocking_issues,
            "warnings": self.warnings,
            "review_required": self.review_required,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OutputEnvelope":
        return cls(
            id=str(data.get("id") or str(uuid.uuid4())[:8]),
            output_text=str(data.get("output_text") or ""),
            workflow_kind=str(data.get("workflow_kind") or WorkflowKind.ANALYSIS.value),
            output_shape=str(data.get("output_shape") or "answer"),
            emitter=str(data.get("emitter") or "unknown"),
            objective_id=data.get("objective_id"),
            working_set_hash=data.get("working_set_hash"),
            dependency_manifest_hash=data.get("dependency_manifest_hash"),
            validation_results=[
                ValidationResult.from_dict(item)
                for item in data.get("validation_results", [])
            ],
            blocking_issues=list(data.get("blocking_issues") or []),
            warnings=list(data.get("warnings") or []),
            review_required=bool(data.get("review_required", False)),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if data.get("created_at") else datetime.now()
            ),
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
    conversation_history: list[dict[str, str]] = field(default_factory=list)

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
    # MVI-3: ExecutionContract from the cascade governor. Engine
    # termination checks read from this; investigate family uses the
    # default contract when the caller didn't thread one through.
    execution_contract: Optional[Any] = None
    research_mode: str = ResearchMode.DEEP.value
    query_classification: Optional[dict] = None  # Result of classify_query()
    run_objective: Optional[RunObjective] = None
    workflow_obligations: list[Obligation] = field(default_factory=list)
    working_set: Optional[WorkingSet] = None
    plan_actions: list[PlanAction] = field(default_factory=list)
    validation_results: list[ValidationResult] = field(default_factory=list)
    output_envelope: Optional[OutputEnvelope] = None

    # In-flight dedup: tracks repo-relative paths currently on the cold path in this run.
    # Prevents the same document from being LLM-analyzed multiple times within a single
    # investigate() call when multiple parallel leads surface the same top-ranked file.
    # asyncio is single-threaded so a plain set is safe (check+add is atomic between awaits).
    _reading_in_progress: set = field(default_factory=set)
    _cached_domain: Optional[str] = None

    # Metrics
    documents_read: int = 0
    documents_from_cache: int = 0  # SO-1: hot-path hits (already-ingested docs skipped)
    searches_performed: int = 0
    recursion_depth: int = 0
    max_depth_reached: int = 0
    api_calls: int = 0
    estimated_tokens: int = 0
    facts_per_iteration: list[int] = field(default_factory=list)  # Track facts added per iteration for diminishing returns
    # MVI-3 governed-progress telemetry. For each iteration, record the
    # matter-wide (sum of issue coverage_fraction) BEFORE the iteration
    # ran. A "material answerability delta" is any iteration where the
    # sum advanced by >= _COVERAGE_DELTA_EPSILON (or a proof gap
    # closed). This replaces the old count-based diminishing-returns
    # signal.
    coverage_sum_per_iteration: list[float] = field(default_factory=list)
    open_gap_count_per_iteration: list[int] = field(default_factory=list)
    # P0.7: coverage-driven lead planner. Counts how many leads the
    # planner has produced THIS RUN so the per-run cap (default 6)
    # can be enforced across iterations.
    planner_leads_added: int = 0
    # Sufficiency probe (plan A): when the interim probe says "we can
    # answer now at ≥medium+≥1 citation", the engine sets this to
    # a short string describing the trigger, sets the probe's answer
    # as final_output, and exits the loop — overriding contract
    # min_iter. Unset (None) means no early-terminate decision has
    # been made.
    early_terminate_reason: Optional[str] = None
    # SO-1 real reuse telemetry: count LLM calls avoided (cache hits, inventory skips)
    # vs. required (cache misses, cold calls). True reuse rate = avoided / (avoided + required).
    llm_calls_avoided: int = 0
    llm_calls_required: int = 0
    llm_usage: dict[str, Any] = field(default_factory=dict)
    cache_manifest_hash: Optional[str] = None

    # Status
    status: str = "initialized"
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Durable reasoning trail (SO-3) — populated at the end of investigate() from the
    # DB ledger. Callers see the full structured trace without a separate API call.
    reasoning_trail: list[dict] = field(default_factory=list)

    # Pending clarification questions (SO-7) — questions generated from high-materiality
    # gaps at the end of the run. Included here so callers receive them in the default
    # workflow without a separate API call to /matter/{id}/clarifications.
    pending_clarifications: list[dict] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        query: str,
        repository_path: str,
        research_mode: "str | ResearchMode | None" = None,
        conversation_history: Optional[list[dict[str, str]]] = None,
    ) -> "InvestigationState":
        return cls(
            id=str(uuid.uuid4())[:8],
            query=query,
            repository_path=repository_path,
            conversation_history=[
                {"query": str(turn.get("query") or "").strip(), "answer": str(turn.get("answer") or "").strip()}
                for turn in (conversation_history or [])
                if str(turn.get("query") or "").strip() or str(turn.get("answer") or "").strip()
            ],
            started_at=datetime.now(),
            research_mode=normalize_research_mode(research_mode),
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
        expected_cost_usd: Optional[float] = None,
        expected_coverage_gain: Optional[float] = None,
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
                # MVI-5: adopt the higher expected_coverage_gain when a
                # duplicate lead is surfaced for a higher-weakness issue.
                if (
                    expected_coverage_gain is not None
                    and expected_coverage_gain > existing.expected_coverage_gain
                ):
                    existing.expected_coverage_gain = expected_coverage_gain
                return None  # Duplicate

        lead = Lead.create(
            description, source, priority,
            search_term=search_term, focus_issue_id=focus_issue_id,
            expected_cost_usd=expected_cost_usd,
            expected_coverage_gain=expected_coverage_gain,
        )
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
            "research_mode": self.research_mode,
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
            "research_mode": self.research_mode,
            "duration_seconds": self.duration_seconds,
            "confidence": confidence,
            "metrics": {
                "documents_read": self.documents_read,
                "documents_from_cache": self.documents_from_cache,  # SO-1: hot-path reuse
                "reuse_rate": round(
                    self.documents_from_cache / self.documents_read, 3
                ) if self.documents_read > 0 else 0.0,
                # SO-1 real LLM reuse telemetry (true_reuse_rate = avoided / total LLM opportunities)
                "llm_calls_avoided": self.llm_calls_avoided,
                "llm_calls_required": self.llm_calls_required,
                "true_reuse_rate": round(
                    self.llm_calls_avoided / (self.llm_calls_avoided + self.llm_calls_required), 3
                ) if (self.llm_calls_avoided + self.llm_calls_required) > 0 else None,
                "llm_request_count": self.llm_usage.get("request_count", 0),
                "llm_input_tokens": self.llm_usage.get("input_tokens", 0),
                "llm_cache_read_tokens": self.llm_usage.get("cache_read_tokens", 0),
                "llm_tool_use_prompt_tokens": self.llm_usage.get("tool_use_prompt_tokens", 0),
                "llm_thinking_tokens": self.llm_usage.get("thinking_tokens", 0),
                "llm_output_tokens": self.llm_usage.get("output_tokens", 0),
                "llm_total_processed_tokens": self.llm_usage.get("total_processed_tokens", 0),
                "llm_estimated_cost_usd": self.llm_usage.get("estimated_cost_usd", 0.0),
                "llm_by_tier": self.llm_usage.get("by_tier", {}),
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
        """Calculate an evidence-quality confidence score (0-100)."""
        factors = {}

        facts = self.findings.get("accumulated_facts", [])
        citation_count = len(self.citations)
        verification_stats = self.get_verification_stats()

        role_weights = {
            "AUTHORITATIVE": 1.0,
            "OPERATIVE": 1.0,
            "PROCEDURAL": 0.75,
            "INFORMAL": 0.5,
            "UNKNOWN": 0.4,
            "DRAFT": 0.3,
            "POST_HOC": 0.3,
            "ADVOCACY": 0.2,
        }

        fact_weights: list[float] = []
        high_trust_count = 0
        for fact in facts:
            label = "UNKNOWN"
            if isinstance(fact, str) and fact.startswith("[") and "]" in fact:
                label = fact[1:fact.index("]")]
            weight = role_weights.get(label.upper(), 0.4)
            fact_weights.append(weight)
            if weight >= 0.75:
                high_trust_count += 1

        # Factor 1: Source quality (0-25)
        if fact_weights:
            avg_trust = sum(fact_weights) / len(fact_weights)
            factors["source_quality"] = avg_trust * 25
        else:
            factors["source_quality"] = 0

        # Factor 2: Weighted evidence volume (0-20)
        factors["evidence_volume"] = min(sum(fact_weights) * 2.0, 20)

        # Factor 3: High-trust support ratio (0-15)
        if fact_weights:
            factors["high_trust_support"] = (high_trust_count / len(fact_weights)) * 15
        else:
            factors["high_trust_support"] = 0

        # Factor 4: Citation support (0-15)
        factors["citations"] = min(citation_count * 1.5, 15)

        # Factor 5: Verification rate (0-15)
        if verification_stats["total"] > 0:
            verified_rate = verification_stats["verified"] / verification_stats["total"]
            factors["verification"] = verified_rate * 15
        else:
            factors["verification"] = 0

        # Factor 6: Corroboration across sources (0-5)
        unique_citation_docs = len({
            citation.document for citation in self.citations if getattr(citation, "document", None)
        })
        factors["corroboration"] = min(float(unique_citation_docs), 5.0)

        # Factor 7: Consistency of the current record (0-5)
        contradiction_penalty = min(len(self.contradictions), 5)
        factors["consistency"] = max(0.0, 5.0 - float(contradiction_penalty))

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
            "conversation_history": self.conversation_history,
            # Identity fields for cross-matter validation on resume
            "_matter_id": getattr(self, "_matter_id", None),
            "_run_id": getattr(self, "_run_id", None),
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
            "research_mode": self.research_mode,
            "query_classification": self.query_classification,
            "run_objective": (
                self.run_objective.to_dict() if self.run_objective else None
            ),
            "workflow_obligations": [
                obligation.to_dict() for obligation in self.workflow_obligations
            ],
            "working_set": self.working_set.to_dict() if self.working_set else None,
            "plan_actions": [action.to_dict() for action in self.plan_actions],
            "validation_results": [
                result.to_dict() for result in self.validation_results
            ],
            "output_envelope": (
                self.output_envelope.to_dict() if self.output_envelope else None
            ),
            "facts_per_iteration": self.facts_per_iteration,
            # P0.7 (adv#11 review fix #3): persist planner_leads_added so a
            # checkpoint/resume doesn't reset the per-run cap and allow
            # another 6 planner leads in the same logical run.
            "planner_leads_added": self.planner_leads_added,
            "early_terminate_reason": self.early_terminate_reason,
            "documents_read": self.documents_read,
            "documents_from_cache": self.documents_from_cache,
            "searches_performed": self.searches_performed,
            "recursion_depth": self.recursion_depth,
            "max_depth_reached": self.max_depth_reached,
            "api_calls": self.api_calls,
            "estimated_tokens": self.estimated_tokens,
            "llm_usage": self.llm_usage,
            "status": self.status,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "reasoning_trail": self.reasoning_trail,
            "pending_clarifications": self.pending_clarifications,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InvestigationState":
        """Deserialize state from dictionary."""
        state = cls(
            id=data["id"],
            query=data["query"],
            repository_path=data["repository_path"],
            conversation_history=data.get("conversation_history", []),
        )
        # Restore identity fields for cross-matter validation
        if data.get("_matter_id"):
            state._matter_id = data["_matter_id"]
        if data.get("_run_id"):
            state._run_id = data["_run_id"]

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
        state.research_mode = normalize_research_mode(data.get("research_mode"))
        state.query_classification = data.get("query_classification")
        if data.get("run_objective"):
            state.run_objective = RunObjective.from_dict(data["run_objective"])
        state.workflow_obligations = [
            Obligation.from_dict(item)
            for item in data.get("workflow_obligations", [])
        ]
        if data.get("working_set"):
            state.working_set = WorkingSet.from_dict(data["working_set"])
        state.plan_actions = [
            PlanAction.from_dict(item)
            for item in data.get("plan_actions", [])
        ]
        state.validation_results = [
            ValidationResult.from_dict(item)
            for item in data.get("validation_results", [])
        ]
        if data.get("output_envelope"):
            state.output_envelope = OutputEnvelope.from_dict(data["output_envelope"])
        state.facts_per_iteration = data.get("facts_per_iteration", [])
        # Restore planner counter; absent in pre-P0.7 checkpoints.
        state.planner_leads_added = int(data.get("planner_leads_added", 0) or 0)
        state.early_terminate_reason = data.get("early_terminate_reason")
        state.documents_read = data.get("documents_read", 0)
        state.documents_from_cache = data.get("documents_from_cache", 0)
        state.searches_performed = data.get("searches_performed", 0)
        state.recursion_depth = data.get("recursion_depth", 0)
        state.max_depth_reached = data.get("max_depth_reached", 0)
        state.api_calls = data.get("api_calls", 0)
        state.estimated_tokens = data.get("estimated_tokens", 0)
        state.llm_usage = data.get("llm_usage", {})
        state.status = data.get("status", "initialized")
        state.error = data.get("error")
        state.started_at = datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
        state.completed_at = datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
        state.reasoning_trail = data.get("reasoning_trail", [])
        state.pending_clarifications = data.get("pending_clarifications", [])

        return state

    def save_checkpoint(self, path: str | Path):
        """Save state to checkpoint file (atomic: write temp then rename)."""
        import os
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            # Atomic replace: no reader sees a half-written file
            os.replace(tmp, path)
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            raise

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> "InvestigationState":
        """Load state from checkpoint file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
