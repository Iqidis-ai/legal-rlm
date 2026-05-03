"""Deterministic domain detection from structured signals.

This module produces evidence-backed domain facet candidates from text,
metadata, and structural signals. It does NOT use LLM classification —
all detection is pattern-based and deterministic.

Detection runs at ingest/extraction time. build_query_context() reads
already-recorded facets; it never runs fresh detection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from .domain_profiles import DOMAIN_PROFILE_IDS

DETECTOR_VERSION = "v1"

CONFIDENCE_ACTIVE = 0.70
CONFIDENCE_CANDIDATE = 0.40


@dataclass(frozen=True)
class DetectionSignals:
    lexical: tuple[str, ...] = ()
    structural: tuple[str, ...] = ()
    entity: tuple[str, ...] = ()
    source_role: tuple[str, ...] = ()
    citation: tuple[str, ...] = ()
    artifact_type: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, list[str]]:
        d: dict[str, list[str]] = {}
        if self.lexical:
            d["lexical"] = list(self.lexical)
        if self.structural:
            d["structural"] = list(self.structural)
        if self.entity:
            d["entity"] = list(self.entity)
        if self.source_role:
            d["source_role"] = list(self.source_role)
        if self.citation:
            d["citation"] = list(self.citation)
        if self.artifact_type:
            d["artifact_type"] = list(self.artifact_type)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> DetectionSignals:
        return cls(
            lexical=tuple(d.get("lexical", ())),
            structural=tuple(d.get("structural", ())),
            entity=tuple(d.get("entity", ())),
            source_role=tuple(d.get("source_role", ())),
            citation=tuple(d.get("citation", ())),
            artifact_type=tuple(d.get("artifact_type", ())),
        )

    def total_count(self) -> int:
        return (
            len(self.lexical) + len(self.structural) + len(self.entity)
            + len(self.source_role) + len(self.citation) + len(self.artifact_type)
        )


@dataclass(frozen=True)
class DetectionCandidate:
    profile_id: str
    confidence: float
    signals: DetectionSignals
    evidence_refs: tuple[str, ...] = ()

    @property
    def is_active(self) -> bool:
        return self.confidence >= CONFIDENCE_ACTIVE

    @property
    def is_candidate(self) -> bool:
        return CONFIDENCE_CANDIDATE <= self.confidence < CONFIDENCE_ACTIVE


# ---------------------------------------------------------------------------
# Signal patterns per domain
# ---------------------------------------------------------------------------

_LEGAL_LEXICAL = re.compile(
    r"\b(?:"
    r"plaintiff|defendant|court|judge|jury|verdict|statute|regulation|"
    r"stipulat|deposition|interrogator|subpoena|affidavit|"
    r"motion\s+to|summary\s+judgment|habeas\s+corpus|amicus\s+curiae|"
    r"negligence|breach\s+of\s+contract|tort|damages|injunction|"
    r"discovery|pleading|indictment|arraignment|"
    r"Rule\s+\d+[a-z]?(?:-\d+)?|"
    r"U\.?S\.?C\.?\s*§?\s*\d+|"
    r"(?:18|26|28|42)\s+U\.S\.C|"
    r"attorney.client\s+privilege|work\s+product|"
    r"privileged|confidential.*attorney|"
    r"pursuant\s+to|hereinafter|notwithstanding|"
    r"complaint|answer|counterclaim|cross.claim"
    r")\b",
    re.IGNORECASE,
)

_LEGAL_CITATION = re.compile(
    r"\b\d+\s+(?:U\.S\.|F\.\s*(?:2d|3d|4th)|S\.\s*Ct\.|L\.\s*Ed|"
    r"F\.\s*Supp\.\s*(?:2d|3d)?|So\.\s*(?:2d|3d)?|"
    r"N\.E\.\s*(?:2d|3d)?|A\.\s*(?:2d|3d)?|"
    r"P\.\s*(?:2d|3d)?|N\.W\.\s*(?:2d)?|S\.E\.\s*(?:2d)?|"
    r"Cal\.\s*(?:2d|3d|4th|5th)|N\.Y\.\s*(?:2d|3d)?"
    r")\s+\d+",
    re.IGNORECASE,
)

_LEGAL_STRUCTURAL = re.compile(
    r"(?:UNITED\s+STATES\s+DISTRICT\s+COURT|"
    r"SUPERIOR\s+COURT|COURT\s+OF\s+APPEALS|"
    r"IN\s+THE\s+MATTER\s+OF|"
    r"Case\s+No\.\s*\d|"
    r"Docket\s+No\.\s*\d|"
    r"v\.\s+[A-Z])",
    re.IGNORECASE,
)

_FINANCE_LEXICAL = re.compile(
    r"\b(?:"
    r"revenue|EBITDA|earnings\s+per\s+share|EPS|"
    r"net\s+income|gross\s+margin|operating\s+(?:income|margin|expense)|"
    r"10-K|10-Q|8-K|20-F|S-1|DEF\s+14A|"
    r"ASC\s+\d+|IFRS\s+\d+|GAAP|"
    r"balance\s+sheet|income\s+statement|cash\s+flow\s+statement|"
    r"shareholder|dividend|stock\s+(?:option|split|buyback)|"
    r"market\s+cap|P/?E\s+ratio|"
    r"material\s+nonpublic|insider\s+trading|"
    r"Rule\s+10b-5|Reg\s+FD|"
    r"fiscal\s+(?:year|quarter)|year.over.year|quarter.over.quarter|"
    r"guidance|forecast|analyst\s+estimate|"
    r"underwriter|IPO|secondary\s+offering|"
    r"derivative|swap|option|forward|"
    r"credit\s+rating|Moody|S&P|Fitch"
    r")\b",
    re.IGNORECASE,
)

_FINANCE_STRUCTURAL = re.compile(
    r"(?:CONSOLIDATED\s+(?:BALANCE|INCOME|STATEMENT)|"
    r"NOTES\s+TO\s+(?:CONSOLIDATED\s+)?FINANCIAL\s+STATEMENTS|"
    r"MANAGEMENT.S\s+DISCUSSION|"
    r"REPORT\s+OF\s+INDEPENDENT\s+(?:REGISTERED\s+)?(?:PUBLIC\s+)?ACCOUNTING|"
    r"SECURITIES\s+AND\s+EXCHANGE\s+COMMISSION)",
    re.IGNORECASE,
)

_FINANCE_ENTITY = re.compile(
    r"\b(?:SEC|FINRA|NYSE|NASDAQ|CFTC|FCA|ESMA|"
    r"Federal\s+Reserve|Treasury|OCC|FDIC)\b",
)

_CODING_LEXICAL = re.compile(
    r"\b(?:"
    r"function|def\s|class\s|import\s|from\s.*import|"
    r"return\s|if\s.*:|for\s.*:|while\s.*:|"
    r"async\s+(?:def|function)|await\s|"
    r"try:|except\s|catch\s*\(|throw\s|raise\s|"
    r"nullptr|NULL|nil|undefined|"
    r"git\s+(?:commit|push|pull|merge|rebase|checkout)|"
    r"CI/?CD|pipeline|workflow|"
    r"Dockerfile|docker-compose|Kubernetes|kubectl|"
    r"package\.json|requirements\.txt|Cargo\.toml|go\.mod|"
    r"npm\s+(?:install|run|test)|pip\s+install|"
    r"assert\s|assertEquals|expect\(|describe\(|it\(|test\(|"
    r"API\s+endpoint|REST|gRPC|GraphQL|"
    r"CVE-\d{4}-\d+|"
    r"segfault|null\s+pointer|buffer\s+overflow|"
    r"race\s+condition|deadlock|memory\s+leak"
    r")\b",
    re.IGNORECASE,
)

_CODING_STRUCTURAL = re.compile(
    r"(?:```(?:python|javascript|typescript|go|rust|java|c\+\+|c#|ruby|bash|yaml|json|toml)|"
    r"^\s*(?:func|fn|pub\s+fn|package\s+main|#include\s|using\s+namespace|"
    r"@Override|@Test|@pytest)|"
    r"(?:\.py|\.js|\.ts|\.go|\.rs|\.java|\.cpp|\.c|\.rb)(?:\s|$|:))",
    re.IGNORECASE | re.MULTILINE,
)

_RESEARCH_LEXICAL = re.compile(
    r"\b(?:"
    r"hypothesis|methodology|abstract|"
    r"peer.review|literature\s+review|meta.analysis|"
    r"systematic\s+review|randomized|controlled\s+trial|"
    r"p[<>]\s*0\.\d+|p\s*=\s*0\.\d+|"
    r"confidence\s+interval|standard\s+deviation|"
    r"chi.square|ANOVA|regression\s+(?:analysis|coefficient)|"
    r"effect\s+size|Cohen.s\s+d|"
    r"reproducib|replicat|"
    r"arXiv|doi:\s*10\.\d+|"
    r"et\s+al\.|ibid\.|op\.\s*cit\.|"
    r"preprint|working\s+paper|"
    r"IRB|institutional\s+review\s+board|informed\s+consent|"
    r"sample\s+size|n\s*=\s*\d+|"
    r"null\s+hypothesis|alternative\s+hypothesis|"
    r"double.blind|placebo.controlled"
    r")\b",
    re.IGNORECASE,
)

_RESEARCH_CITATION = re.compile(
    r"(?:"
    r"\([A-Z][a-z]+(?:\s+(?:et\s+al\.?|&\s+[A-Z][a-z]+))?,?\s*\d{4}[a-z]?\)|"
    r"\[\d+(?:,\s*\d+)*\]|"
    r"doi:\s*10\.\d{4,}/\S+"
    r")",
    re.IGNORECASE,
)

_BIOMEDICAL_LEXICAL = re.compile(
    r"\b(?:"
    r"patient|clinical\s+trial|Phase\s+[I]{1,3}[Vv]?(?:\s+trial)?|"
    r"diagnosis|prognosis|etiology|pathogen|"
    r"ICD-\d+|CPT\s+\d+|SNOMED|LOINC|"
    r"mg/?(?:kg|dL|mL|day)|"
    r"FDA|EMA|WHO|CDC|NIH|"
    r"adverse\s+(?:event|effect|reaction)|"
    r"contraindic|dose.response|pharmacokinetic|"
    r"in\s+vitro|in\s+vivo|ex\s+vivo|"
    r"biomarker|genotype|phenotype|allele|polymorphism|"
    r"CRISPR|mRNA|siRNA|monoclonal\s+antibody|"
    r"endpoint|primary\s+outcome|secondary\s+outcome|"
    r"hazard\s+ratio|odds\s+ratio|relative\s+risk|"
    r"intention.to.treat|per.protocol|"
    r"PHI|HIPAA|protected\s+health\s+information"
    r")\b",
    re.IGNORECASE,
)

_BIOMEDICAL_STRUCTURAL = re.compile(
    r"(?:CLINICAL\s+TRIAL\s+REGISTRATION|"
    r"ClinicalTrials\.gov|NCT\d{8}|"
    r"CONSORT|STROBE|PRISMA|"
    r"Table\s+\d+\.\s*(?:Patient|Baseline|Endpoint|Outcome))",
    re.IGNORECASE,
)


def _count_matches(pattern: re.Pattern, text: str) -> list[str]:
    return [m.group() for m in pattern.finditer(text)]


_DOMAIN_DETECTORS: dict[str, list[tuple[str, str, re.Pattern]]] = {
    "legal": [
        ("lexical", "legal_term", _LEGAL_LEXICAL),
        ("citation", "case_law", _LEGAL_CITATION),
        ("structural", "court_caption", _LEGAL_STRUCTURAL),
    ],
    "finance": [
        ("lexical", "financial_term", _FINANCE_LEXICAL),
        ("structural", "financial_statement", _FINANCE_STRUCTURAL),
        ("entity", "financial_regulator", _FINANCE_ENTITY),
    ],
    "coding": [
        ("lexical", "code_term", _CODING_LEXICAL),
        ("structural", "code_block", _CODING_STRUCTURAL),
    ],
    "academic_research": [
        ("lexical", "research_term", _RESEARCH_LEXICAL),
        ("citation", "academic_reference", _RESEARCH_CITATION),
    ],
    "biomedical": [
        ("lexical", "clinical_term", _BIOMEDICAL_LEXICAL),
        ("structural", "clinical_structure", _BIOMEDICAL_STRUCTURAL),
    ],
}

# Base score per match hit in each category (diminishing returns via log-ish curve)
_CATEGORY_BASE: dict[str, float] = {
    "lexical": 0.10,
    "structural": 0.25,
    "entity": 0.18,
    "citation": 0.22,
    "source_role": 0.20,
    "artifact_type": 0.15,
}

# Category diversity bonus: each distinct category adds this flat bonus
_CATEGORY_DIVERSITY_BONUS = 0.12

# Density bonus: if a single category has many matches, signal strength is higher
_DENSITY_THRESHOLD = 5
_DENSITY_BONUS = 0.15

# Per-category score cap: prevents single-category repetition from reaching active threshold
_CATEGORY_SCORE_CAP = 0.45

# Metadata signal bonus per hit
_METADATA_BONUS = 0.15


def detect_domain_signals(
    text: str,
    *,
    source_type: str = "",
    filename: str = "",
    metadata: dict | None = None,
) -> list[DetectionCandidate]:
    """Run deterministic domain detection on text and metadata.

    Returns candidates for all profiles that score above CONFIDENCE_CANDIDATE,
    sorted by confidence descending.
    """
    metadata = metadata or {}
    candidates: list[DetectionCandidate] = []

    for profile_id in DOMAIN_PROFILE_IDS:
        detectors = _DOMAIN_DETECTORS.get(profile_id, [])
        if not detectors:
            continue

        signals_by_cat: dict[str, list[str]] = {}
        evidence_refs: list[str] = []
        raw_score = 0.0
        active_categories: set[str] = set()
        category_scores: dict[str, float] = {}

        for category, label, pattern in detectors:
            matches = _count_matches(pattern, text)
            if matches:
                signals_by_cat.setdefault(category, []).append(label)
                active_categories.add(category)
                base = _CATEGORY_BASE.get(category, 0.08)
                n = len(matches)
                score = base * min(n, 3) + base * 0.5 * max(min(n, 10) - 3, 0)
                if n >= _DENSITY_THRESHOLD:
                    score += _DENSITY_BONUS
                prev = category_scores.get(category, 0.0)
                capped = min(prev + score, _CATEGORY_SCORE_CAP) - prev
                category_scores[category] = prev + capped
                raw_score += capped
                evidence_refs.append(f"{category}:{label}:{n}")

        # Metadata-based signals
        meta_signals = _metadata_signals(profile_id, source_type, filename, metadata)
        for cat, label in meta_signals:
            signals_by_cat.setdefault(cat, []).append(label)
            active_categories.add(cat)
            raw_score += _METADATA_BONUS
            evidence_refs.append(f"{cat}:{label}:metadata")

        if raw_score == 0.0:
            continue

        # Category diversity bonus
        raw_score += max(len(active_categories) - 1, 0) * _CATEGORY_DIVERSITY_BONUS

        confidence = min(raw_score, 1.0)
        if confidence < CONFIDENCE_CANDIDATE:
            continue

        signals = DetectionSignals(
            lexical=tuple(sorted(set(signals_by_cat.get("lexical", [])))),
            structural=tuple(sorted(set(signals_by_cat.get("structural", [])))),
            entity=tuple(sorted(set(signals_by_cat.get("entity", [])))),
            source_role=tuple(sorted(set(signals_by_cat.get("source_role", [])))),
            citation=tuple(sorted(set(signals_by_cat.get("citation", [])))),
            artifact_type=tuple(sorted(set(signals_by_cat.get("artifact_type", [])))),
        )
        candidates.append(DetectionCandidate(
            profile_id=profile_id,
            confidence=round(confidence, 4),
            signals=signals,
            evidence_refs=tuple(evidence_refs),
        ))

    candidates.sort(key=lambda c: (-c.confidence, c.profile_id))
    return candidates


def _metadata_signals(
    profile_id: str,
    source_type: str,
    filename: str,
    metadata: dict,
) -> list[tuple[str, str]]:
    signals: list[tuple[str, str]] = []
    st = source_type.lower()
    fn = filename.lower()

    if profile_id == "legal":
        if st in ("complaint", "motion", "brief", "deposition", "pleading", "contract"):
            signals.append(("artifact_type", st))
        if any(fn.endswith(ext) for ext in (".legal", ".case")):
            signals.append(("artifact_type", "legal_file"))

    elif profile_id == "finance":
        if st in ("10-k", "10-q", "8-k", "20-f", "s-1", "proxy", "filing"):
            signals.append(("artifact_type", st))
        if any(fn.endswith(ext) for ext in (".xbrl", ".xlsx")):
            signals.append(("artifact_type", "financial_data"))

    elif profile_id == "coding":
        code_exts = (
            ".py", ".js", ".ts", ".go", ".rs", ".java", ".cpp", ".c",
            ".h", ".rb", ".sh", ".yaml", ".yml", ".json", ".toml",
            ".dockerfile", ".tf", ".hcl",
        )
        if any(fn.endswith(ext) for ext in code_exts):
            signals.append(("artifact_type", "source_code"))
        if st in ("source_code", "test", "config", "build_log", "ci_output"):
            signals.append(("artifact_type", st))

    elif profile_id == "academic_research":
        if st in ("paper", "preprint", "thesis", "dissertation", "review_article"):
            signals.append(("artifact_type", st))
        if fn.endswith(".bib") or fn.endswith(".tex"):
            signals.append(("artifact_type", "academic_file"))

    elif profile_id == "biomedical":
        if st in ("clinical_trial", "lab_report", "patient_record", "drug_label"):
            signals.append(("artifact_type", st))
        if any(fn.endswith(ext) for ext in (".hl7", ".fhir", ".dicom")):
            signals.append(("artifact_type", "clinical_data"))

    return signals


def pick_primary_profile(candidates: Sequence[DetectionCandidate]) -> str | None:
    """Select the primary profile from detection candidates.

    Returns the highest-confidence active profile, or None if no active candidates.
    """
    active = [c for c in candidates if c.is_active]
    if not active:
        return None
    active.sort(key=lambda c: (-c.confidence, c.profile_id))
    return active[0].profile_id
