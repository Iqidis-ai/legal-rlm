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


async def _run_scenario_async(scenario: dict) -> dict:
    """Run one test scenario and return results."""
    config = IrysConfig(enable_inline_citations=True)

    answer = scenario["answer"]
    citations = scenario["citations"]

    annotated, reordered, diag = await InlineCitationService.inject(
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


def run_scenario(scenario: dict) -> dict:
    """Sync wrapper for test compatibility."""
    import asyncio
    return asyncio.run(_run_scenario_async(scenario))


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



# ── Trace-replay test (no LLM call) ──────────────────────────────────────────
#
# Reproduces the exact failure captured in:
#   llm_traces/CITE_INJECT_ERROR-20260528_173316_776000/
#                        016_citation_injection_20260528_173754_856000.json
#
# The LLM wrote [8c5bd97d] but the valid ID is 8e5bd97d (one char off: c→e).
# Old behaviour: _validate_response detected the unknown ID and discarded the
#   entire injected answer → frontend received zero citation markers.
# New behaviour: _strip_invalid_ids removes [8c5bd97d] silently; the remaining
#   valid markers are kept and renumbered as [[cite:N]].
#
# The mock returns the exact substring of the real LLM response that triggered
# the bug (trimmed for readability while preserving all relevant IDs).

# Three citations present in the real trace that we exercise here.
# IDs match exactly what the trace used.
_TRACE_VALID_IDS = {
    "8e5bd97d",   # WitnessStatement_Kim – attendance tracking admission
    "e34143a9",   # Torres v. El Paso Electric Co. (case law)
    "5fe8234e",   # Complaint_and_Answer – breach of duty paragraph
}

# Minimal answer excerpt — preserves both the correct-placement sentences and
# the sentence where the LLM hallucinated the bad ID.
_TRACE_ANSWER = """\
### 2. BREACH OF DUTY
Luminos breached its duty of care through a systemic failure to enforce its own safety policies.

*   **Failure to Enforce Rigging Safety Orientation:** Luminos had a 42% compliance rate \
(6 of 14 artists) and no verification system. Sondra Kim admitted to imperfect implementation. \
Under New Mexico law, adopting a safety policy but systemically failing to enforce it is strong \
evidence of a breach of the standard of care.

### 3. CAUSATION (PROXIMATE AND SUPERSEDING CAUSE)
**Legal Standard:** A superseding cause is an intervening act of a third party that is so \
extraordinary and unforeseeable that it breaks the chain of causation. \
*Torres v. El Paso Electric Co.*, 127 N.M. 729 (1999). The critical inquiry is foreseeability.
"""

# Exact LLM response from the trace — has [8c5bd97d] (hallucinated) alongside
# valid markers [8e5bd97d would have been correct], [e34143a9], [5fe8234e].
# We inject [5fe8234e] on the first sentence so there is at least one match
# that is unambiguously correct, giving the length-ratio check room to pass.
_TRACE_LLM_RESPONSE = """\
### 2. BREACH OF DUTY
Luminos breached its duty of care through a systemic failure to enforce its own safety policies [5fe8234e].

*   **Failure to Enforce Rigging Safety Orientation:** Luminos had a 42% compliance rate \
(6 of 14 artists) and no verification system [8c5bd97d]. Sondra Kim admitted to imperfect implementation. \
Under New Mexico law, adopting a safety policy but systemically failing to enforce it is strong \
evidence of a breach of the standard of care.

### 3. CAUSATION (PROXIMATE AND SUPERSEDING CAUSE)
**Legal Standard:** A superseding cause is an intervening act of a third party that is so \
extraordinary and unforeseeable that it breaks the chain of causation. \
*Torres v. El Paso Electric Co.*, 127 N.M. 729 (1999) [e34143a9]. The critical inquiry is foreseeability.
"""


def _make_trace_citations():
    """Build the three Citation objects used in the trace replay.

    We use direct dataclass construction (not Citation.create) so we can pin
    the IDs to the exact values that appeared in the real trace.
    """
    from datetime import datetime
    return [
        Citation(
            id="8e5bd97d",   # Valid ID — LLM hallucinated 8c5bd97d (c vs e)
            document="/tmp/irys/stream_7b006655/06c_WitnessStatement_Kim.docx",
            page=1,
            text="I will acknowledge that no individual tracking system existed to confirm "
                 "attendance prior to granting ongoing rigging access.",
            context="",
            relevance="high",
            source_type="document",
            timestamp=datetime.now(),
        ),
        Citation(
            id="e34143a9",
            document="/tmp/irys/stream_856076a9/10_Defendant_Luminos_Motion_to_Dismiss.docx",
            page=1,
            text="Torres v. El Paso Electric Co., 1999-NMSC-029, ¶ 27, 127 N.M. 729.",
            context="Citation: 987 P.2d 386 | Court: New Mexico Supreme Court",
            relevance="high",
            source_type="case_law",
            timestamp=datetime.now(),
        ),
        Citation(
            id="5fe8234e",
            document="/tmp/irys/stream_856076a9/05_Complaint_and_Answer.docx",
            page=1,
            text="Luminos breached its duty of care through the acts and omissions described herein.",
            context="",
            relevance="high",
            source_type="document",
            timestamp=datetime.now(),
        ),
    ]


async def _run_trace_replay_async() -> bool:
    """Run the trace-replay scenario without a real LLM call.

    Patches _call_gemini_lite to return the exact (buggy) LLM response from
    the real trace, then verifies the fix behaves correctly.
    """
    from unittest.mock import patch, AsyncMock

    config = IrysConfig(enable_inline_citations=True)
    citations = _make_trace_citations()

    with patch.object(
        InlineCitationService,
        "_call_gemini_lite",
        new=AsyncMock(return_value=_TRACE_LLM_RESPONSE),
    ):
        annotated, reordered, diag = await InlineCitationService.inject(
            answer=_TRACE_ANSWER,
            citations=citations,
            config=config,
        )

    passed = True

    checks = [
        # Fix: invalid ID must be stripped, not used to reject everything
        ("validation_passed is True",
         diag.get("validation_passed") is True),

        # Fix: exactly one bad ID was stripped
        ("1 invalid ID stripped",
         diag.get("invalid_ids_stripped") == 1),

        # Fix: output contains [[cite: markers (injection was not discarded)
        ("output contains [[cite: markers",
         "[[cite:" in annotated),

        # Fix: the hallucinated ID must not appear in the final output
        ("hallucinated ID 8c5bd97d absent",
         "8c5bd97d" not in annotated),

        # Sanity: at least one valid marker appeared (Torres case law)
        ("Torres case law marker present",
         _check_keyword_cited("Torres", annotated)),

        # Sanity: breach-of-duty anchor marker present
        ("breach-of-duty anchor marker present",
         _check_keyword_cited("systemic failure", annotated)),
    ]

    print_separator("TRACE REPLAY — CITE_INJECT_ERROR (no LLM call)")
    print("\n[MOCK LLM RESPONSE fed to pipeline]")
    print(_TRACE_LLM_RESPONSE)
    print("\n[OUTPUT after strip + renumber]")
    print(annotated)
    print("\n[DIAGNOSTICS]")
    print(f"  validation_passed:    {diag.get('validation_passed')}")
    print(f"  invalid_ids_stripped: {diag.get('invalid_ids_stripped', 0)}")
    print(f"  citations_matched:    {diag.get('citations_matched_inline')}")
    print(f"  citations_unmatched:  {diag.get('citations_unmatched')}")
    print("\n[ASSERTIONS]")
    for label, result in checks:
        icon = "✓" if result else "✗"
        print(f"  {icon} {label}")
        if not result:
            passed = False

    return passed


def run_trace_replay() -> bool:
    import asyncio
    return asyncio.run(_run_trace_replay_async())


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

    # ── Trace-replay test (deterministic, no LLM call) ───────────────────────
    replay_passed = run_trace_replay()
    if not replay_passed:
        all_passed = False

    print_separator("RESULT")
    if all_passed:
        print("  ALL SCENARIOS PASSED\n")
    else:
        print("  SOME SCENARIOS FAILED — see ✗ above\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    # --replay-only: run only the deterministic trace-replay test (no LLM call / no API key needed)
    if "--replay-only" in sys.argv or "--replay" in sys.argv:
        ok = run_trace_replay()
        print_separator("RESULT")
        print("  TRACE REPLAY PASSED\n" if ok else "  TRACE REPLAY FAILED\n")
        sys.exit(0 if ok else 1)
    sys.exit(main())
