"""Tests for deterministic domain detection from structured signals."""

from irys.matter.domain_detection import (
    CONFIDENCE_ACTIVE,
    CONFIDENCE_CANDIDATE,
    DETECTOR_VERSION,
    DetectionCandidate,
    DetectionSignals,
    detect_domain_signals,
    pick_primary_profile,
)


def test_detector_version_is_v1():
    assert DETECTOR_VERSION == "v1"


def test_detection_signals_round_trip():
    s = DetectionSignals(
        lexical=("term1",),
        structural=("struct1",),
        entity=("ent1",),
        citation=("cite1",),
    )
    d = s.to_dict()
    s2 = DetectionSignals.from_dict(d)
    assert s2.lexical == ("term1",)
    assert s2.citation == ("cite1",)
    assert s.total_count() == 4


def test_detection_signals_empty():
    s = DetectionSignals()
    assert s.to_dict() == {}
    assert s.total_count() == 0


def test_legal_text_detected():
    text = (
        "The plaintiff filed a motion to compel discovery. The court ruled that "
        "pursuant to Rule 37, the defendant must produce the deposition transcripts. "
        "The verdict was in favor of the plaintiff, damages awarded."
    )
    candidates = detect_domain_signals(text)
    legal = [c for c in candidates if c.profile_id == "legal"]
    assert len(legal) == 1
    assert legal[0].confidence >= CONFIDENCE_ACTIVE
    assert legal[0].is_active


def test_finance_text_detected():
    text = (
        "Revenue for Q3 increased 12% year-over-year. EBITDA margin improved to 25%. "
        "The company filed its 10-K with the SEC. Earnings per share beat analyst estimates. "
        "Management provided forward guidance for fiscal year 2026."
    )
    candidates = detect_domain_signals(text)
    finance = [c for c in candidates if c.profile_id == "finance"]
    assert len(finance) == 1
    assert finance[0].confidence >= CONFIDENCE_ACTIVE


def test_coding_text_detected():
    text = (
        "```python\n"
        "def handle_request(req):\n"
        "    try:\n"
        "        return await process(req)\n"
        "    except Exception as e:\n"
        "        raise HTTPError(500)\n"
        "```\n"
        "The race condition in the connection pool causes a deadlock under load. "
        "Fix the async function to use proper locking. Run npm test to verify."
    )
    candidates = detect_domain_signals(text)
    coding = [c for c in candidates if c.profile_id == "coding"]
    assert len(coding) == 1
    assert coding[0].confidence >= CONFIDENCE_CANDIDATE


def test_research_text_detected():
    text = (
        "Our hypothesis was tested using a randomized controlled trial (n=500). "
        "Results show a statistically significant effect (p < 0.001, 95% confidence interval "
        "[0.12, 0.45]). This is consistent with findings from a recent meta-analysis "
        "(Smith et al., 2024). Effect size was Cohen's d = 0.38. "
        "doi: 10.1234/example.2024"
    )
    candidates = detect_domain_signals(text)
    research = [c for c in candidates if c.profile_id == "academic_research"]
    assert len(research) == 1
    assert research[0].confidence >= CONFIDENCE_ACTIVE


def test_biomedical_text_detected():
    text = (
        "In this Phase III clinical trial (NCT01234567), 240 patients received "
        "the monoclonal antibody at 5 mg/kg. The primary endpoint showed a hazard ratio "
        "of 0.65 (p=0.003). Adverse events were consistent with the known safety profile. "
        "FDA approved the drug for the new indication."
    )
    candidates = detect_domain_signals(text)
    biomed = [c for c in candidates if c.profile_id == "biomedical"]
    assert len(biomed) == 1
    assert biomed[0].confidence >= CONFIDENCE_ACTIVE


def test_mixed_legal_finance_text():
    text = (
        "The SEC filed a complaint alleging that the defendant violated Rule 10b-5 "
        "by making material misstatements in the company's 10-K filing. Revenue was "
        "overstated by $50M. The court issued a preliminary injunction. "
        "EBITDA was inflated through improper revenue recognition under ASC 606."
    )
    candidates = detect_domain_signals(text)
    assert len(candidates) >= 2
    profile_ids = {c.profile_id for c in candidates}
    assert "legal" in profile_ids
    assert "finance" in profile_ids


def test_empty_text_no_candidates():
    candidates = detect_domain_signals("")
    assert candidates == []


def test_generic_text_low_confidence():
    text = "The weather is nice today. Let's go for a walk in the park."
    candidates = detect_domain_signals(text)
    assert all(c.confidence < CONFIDENCE_ACTIVE for c in candidates)


def test_metadata_boosts_detection():
    text = "Revenue grew 15% and EBITDA improved."
    candidates_no_meta = detect_domain_signals(text)
    candidates_with_meta = detect_domain_signals(
        text, source_type="10-K", filename="annual_report.xbrl"
    )
    fin_no_meta = [c for c in candidates_no_meta if c.profile_id == "finance"]
    fin_with_meta = [c for c in candidates_with_meta if c.profile_id == "finance"]
    assert len(fin_with_meta) >= 1
    if fin_no_meta:
        assert fin_with_meta[0].confidence > fin_no_meta[0].confidence


def test_coding_metadata_detection():
    text = "This module handles authentication using async def verify()."
    candidates = detect_domain_signals(text, filename="auth_handler.py", source_type="source_code")
    coding = [c for c in candidates if c.profile_id == "coding"]
    assert len(coding) >= 1


def test_candidates_sorted_by_confidence_desc():
    text = (
        "The court found that the defendant's 10-K filing contained material "
        "misstatements about revenue per ASC 606. The plaintiff alleged fraud "
        "under Rule 10b-5. EBITDA was overstated."
    )
    candidates = detect_domain_signals(text)
    for i in range(len(candidates) - 1):
        assert candidates[i].confidence >= candidates[i + 1].confidence


def test_pick_primary_profile_selects_highest():
    c1 = DetectionCandidate("legal", 0.85, DetectionSignals())
    c2 = DetectionCandidate("finance", 0.72, DetectionSignals())
    assert pick_primary_profile([c1, c2]) == "legal"
    assert pick_primary_profile([c2, c1]) == "legal"


def test_pick_primary_profile_none_when_no_active():
    c1 = DetectionCandidate("legal", 0.55, DetectionSignals())
    c2 = DetectionCandidate("finance", 0.45, DetectionSignals())
    assert pick_primary_profile([c1, c2]) is None


def test_pick_primary_profile_empty():
    assert pick_primary_profile([]) is None


def test_evidence_refs_populated():
    text = (
        "The plaintiff alleged negligence and breach of contract. "
        "The court found in 123 F.3d 456. Summary judgment was granted."
    )
    candidates = detect_domain_signals(text)
    legal = [c for c in candidates if c.profile_id == "legal"]
    assert len(legal) == 1
    assert len(legal[0].evidence_refs) > 0
    assert any("lexical" in ref for ref in legal[0].evidence_refs)


def test_legal_citation_signals():
    text = "The holding in 500 U.S. 123 controls. See also 300 F.3d 789."
    candidates = detect_domain_signals(text)
    legal = [c for c in candidates if c.profile_id == "legal"]
    assert len(legal) >= 1
    assert any("citation" in ref for ref in legal[0].evidence_refs)
