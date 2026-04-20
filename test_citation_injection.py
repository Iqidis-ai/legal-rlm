"""Test script: citation injection service in isolation.

Simulates the PM's failure cases — answers that mention case names (exact and
paraphrased) paired with Citation objects.  Calls InlineCitationService.inject()
and reports which citations were placed vs dropped.

USAGE:
    python test_citation_injection.py

Requires GEMINI_API_KEY in .env (uses Gemini Lite for the LLM call).
"""

import json
import sys
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("test_citation_injection")

from src.irys.rlm.state import Citation
from src.irys.service.inline_citation_service import InlineCitationService
from src.irys.api import IrysConfig

# ── Test scenarios ────────────────────────────────────────────────────────────
# Each scenario has: name, answer text, list of citations, expected case names
# that MUST appear inline.

SCENARIOS = [
    {
        "name": "v4 — all 5 cases cited, Kroger must not drop",
        "answer": (
            "The Texas Supreme Court has established clear standards for evidence review. "
            "In Trevino v. State, the court addressed procedural requirements for criminal appeals, "
            "establishing important precedent for appellate review. "
            "The Formosa Plastics decision fundamentally changed how Texas courts handle "
            "challenges to legal and factual sufficiency. "
            "Kroger's liability in the Persley case demonstrated the duty of care owed by "
            "commercial establishments to their customers. "
            "City of Keller v. Wilson remains the definitive authority on legal sufficiency "
            "review in Texas civil cases. "
            "The Halliburton mandamus proceeding clarified the standards for extraordinary relief "
            "in complex litigation."
        ),
        "citations": [
            Citation.create(
                document="[Case Law] Trevino v. State",
                page=None,
                text="Trevino v. State addresses procedural requirements for criminal appeals in Texas.",
                context="Citation: 991 S.W.2d 849 | Court: Tex. Crim. App. 1999",
                relevance="high",
                source_type="case_law",
            ),
            Citation.create(
                document="[Case Law] Formosa Plastics Corp. USA v. Presidio Engineers & Contractors, Inc.",
                page=None,
                text="Formosa Plastics changed how Texas courts handle challenges to legal and factual sufficiency of evidence.",
                context="Citation: 960 S.W.2d 41 | Court: Tex. 1998",
                relevance="high",
                source_type="case_law",
            ),
            Citation.create(
                document="[Case Law] Kroger Co. v. Persley",
                page=None,
                text="Kroger Co. v. Persley addressed duty of care owed by commercial establishments.",
                context="Citation: 261 S.W.3d 316 | Court: Tex. App.—Houston [1st Dist.] 2008, no pet.",
                relevance="high",
                source_type="case_law",
            ),
            Citation.create(
                document="[Case Law] City of Keller v. Wilson",
                page=None,
                text="City of Keller v. Wilson is the definitive authority on legal sufficiency review.",
                context="Citation: 168 S.W.3d 802 | Court: Tex. 2005",
                relevance="high",
                source_type="case_law",
            ),
            Citation.create(
                document="[Case Law] In re Halliburton Co.",
                page=None,
                text="In re Halliburton Co. clarified standards for mandamus relief in complex litigation.",
                context="Citation: 80 S.W.3d 566 | Court: Tex. 2002",
                relevance="high",
                source_type="case_law",
            ),
        ],
        "must_match": ["Trevino", "Formosa", "Kroger", "Keller", "Halliburton"],
    },
    {
        "name": "v1 — contract for deed with document + case law mix",
        "answer": (
            "A contract for deed, also known as an executory contract in Texas, is an agreement "
            "where the buyer takes possession of property but the seller retains legal title until "
            "the purchase price is fully paid. Texas Property Code Chapter 5 governs these transactions. "
            "The landmark case of Johnson v. Cherry established that sellers must provide annual "
            "accounting statements to buyers under executory contracts. "
            "According to the inspection report, the property condition was documented thoroughly "
            "before the contract was executed."
        ),
        "citations": [
            Citation.create(
                document="[Case Law] Johnson v. Cherry",
                page=None,
                text="Johnson v. Cherry established annual accounting requirements for executory contracts.",
                context="Citation: 726 S.W.2d 586 | Court: Tex. App. 1987",
                relevance="high",
                source_type="case_law",
            ),
            Citation.create(
                document="property_inspection_report.pdf",
                page=3,
                text="The property condition was documented with photographs and detailed notes prior to contract execution.",
                context="Inspection performed on 2023-05-15",
                relevance="medium",
                source_type="document",
            ),
        ],
        "must_match": ["Johnson", "inspection"],
    },
]


def _check_keyword_cited(keyword: str, text: str) -> bool:
    """Check if a keyword appears near a [[cite:N]] marker in the text.

    Uses a sliding window around each cite marker instead of naive sentence
    splitting (which breaks on 'v. State', 'v. Wilson', etc.).
    """
    import re
    lower = text.lower()
    kw = keyword.lower()
    # Find every [[cite:N]] position, check if keyword is within 200 chars before it
    for m in re.finditer(r'\[\[cite:[0-9]+\]\]', lower):
        window_start = max(0, m.start() - 200)
        window = lower[window_start:m.end()]
        if kw in window:
            return True
    return False


def run_scenario(scenario: dict) -> dict:
    """Run one test scenario and return results."""
    config = IrysConfig(enable_inline_citations=True)

    answer = scenario["answer"]
    citations = scenario["citations"]

    annotated, reordered, diag = InlineCitationService.inject(
        answer=answer,
        citations=citations,
        config=config,
    )

    results = {"name": scenario["name"], "passed": [], "failed": [], "diag": diag}
    for keyword in scenario["must_match"]:
        if _check_keyword_cited(keyword, annotated):
            results["passed"].append(keyword)
        else:
            results["failed"].append(keyword)

    return results, annotated, answer, citations


def print_separator(label: str = ""):
    width = 70
    if label:
        pad = (width - len(label) - 2) // 2
        print(f"\n{'─' * pad} {label} {'─' * pad}")
    else:
        print("─" * width)


def main():
    all_passed = True

    for scenario in SCENARIOS:
        results, annotated, original, citations = run_scenario(scenario)
        diag = results["diag"]

        print_separator(results["name"])

        # 1. Original answer
        print("\n[INPUT TEXT]")
        print(original)

        # 2. Citation list
        print("\n[CITATIONS FED]")
        for i, c in enumerate(citations, 1):
            stype = getattr(c, 'source_type', 'document')
            doc = getattr(c, 'document', '?')
            text_preview = (getattr(c, 'text', '') or '')[:80]
            print(f"  {i}. [{stype}] {doc}")
            print(f"     Text: {text_preview}...")

        # 3. Final annotated output
        print("\n[OUTPUT TEXT]")
        print(annotated)

        # 4. Diagnostics
        print("\n[DIAGNOSTICS]")
        print(f"  Fed to LLM:  {diag.get('citations_fed_to_llm', '?')}")
        print(f"  Matched:     {diag.get('citations_matched_inline', '?')}")
        print(f"  Unmatched:   {diag.get('citations_unmatched', '?')}")
        print(f"  Latency:     {diag.get('llm_latency_ms', '?')}ms")
        print(f"  Validation:  {diag.get('validation_passed', '?')}")

        if diag.get("unmatched_details"):
            print("  Dropped citations:")
            for d in diag["unmatched_details"]:
                print(f"    ✗ {d['name']} ({d['source_type']})")

        # 5. Keyword match results
        print("\n[KEYWORD CHECK]")
        for kw in results["passed"]:
            print(f"  ✓ {kw}")
        for kw in results["failed"]:
            print(f"  ✗ {kw} — NOT cited inline")
            all_passed = False

    print_separator("RESULT")
    if all_passed:
        print("  ALL SCENARIOS PASSED\n")
    else:
        print("  SOME SCENARIOS FAILED — see ✗ above\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
