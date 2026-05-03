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
        "The plaintiff filed a motion to compel discovery in the "
        "UNITED STATES DISTRICT COURT for the Southern District. "
        "Pursuant to 500 U.S. 123, the court ruled that "
        "the defendant must produce the deposition transcripts. "
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


# --- Adversarial generic-text false-positive tests (PR Gate 4 #5) ---


def test_generic_prose_no_false_positive():
    """Generic English prose should not trigger any domain detection."""
    text = (
        "The weather was pleasant today and the children played in the park. "
        "We had dinner at the restaurant and then watched a movie at home. "
        "The flowers in the garden are blooming beautifully this spring."
    )
    candidates = detect_domain_signals(text)
    active = [c for c in candidates if c.is_active]
    assert len(active) == 0, f"Generic prose triggered: {[(c.profile_id, c.confidence) for c in active]}"


def test_repeated_option_does_not_false_positive_finance():
    """Repeating broad terms like 'option' should not push finance above active."""
    text = " ".join(["We have the option to choose from various options."] * 10)
    candidates = detect_domain_signals(text)
    finance = [c for c in candidates if c.profile_id == "finance" and c.is_active]
    assert len(finance) == 0, f"Repeated 'option' false positive: conf={finance[0].confidence if finance else 0}"


def test_repeated_function_does_not_false_positive_coding():
    """Repeating 'function' in prose shouldn't push coding above active."""
    text = " ".join(["The function of this function is to function properly."] * 10)
    candidates = detect_domain_signals(text)
    coding = [c for c in candidates if c.profile_id == "coding" and c.is_active]
    assert len(coding) == 0, f"Repeated 'function' false positive: conf={coding[0].confidence if coding else 0}"


def test_repeated_patient_does_not_false_positive_biomedical():
    """Repeating 'patient' in non-medical context shouldn't push biomedical above active."""
    text = " ".join(["The patient teacher was very patient with the patient students."] * 10)
    candidates = detect_domain_signals(text)
    bio = [c for c in candidates if c.profile_id == "biomedical" and c.is_active]
    assert len(bio) == 0, f"Repeated 'patient' false positive: conf={bio[0].confidence if bio else 0}"


def test_numbered_list_does_not_false_positive_research():
    """Numbered items like [1], [2] should not push research above active."""
    text = " ".join([f"[{i}] Some numbered item about general topics." for i in range(20)])
    candidates = detect_domain_signals(text)
    research = [c for c in candidates if c.profile_id == "research" and c.is_active]
    assert len(research) == 0, f"Numbered list false positive: conf={research[0].confidence if research else 0}"


def test_single_category_score_cap_prevents_over_promotion():
    """Even with massive lexical hits, single-category cap should limit score."""
    text = " ".join(["court judgment plaintiff defendant"] * 50)
    candidates = detect_domain_signals(text)
    legal = [c for c in candidates if c.profile_id == "legal"]
    assert len(legal) >= 1
    # Should be capped — single category (lexical) can't reach 1.0 alone
    assert legal[0].confidence < 0.70, (
        f"Single-category should be below active: conf={legal[0].confidence}"
    )


# --- Integration: detection wired into document ingest (Phase 4) ---


def test_document_profile_triggers_domain_detection():
    """upsert_document_profile with finance metadata creates workspace facets."""
    from irys.matter import MatterModel

    model = MatterModel.open_in_memory()
    model.inventory.upsert(
        relative_path="financials/10K_2025.pdf",
        sha256="sha256:abc123",
    )
    model.upsert_document_profile(
        relative_path="financials/10K_2025.pdf",
        analysis={
            "title": "Annual Report 10-K SEC Filing",
            "doc_type": "10-K",
            "purpose": "Revenue disclosure and EBITDA margin analysis for fiscal year 2025. "
                       "Earnings per share improved. Management provided forward guidance.",
        },
    )
    broker = model.memory_broker
    facets = broker.get_object_domain_facets(
        "workspace", model.matter_id, status="active",
    )
    finance_facets = [f for f in facets if f["domain_profile_id"] == "finance"]
    assert len(finance_facets) >= 1, f"Expected finance facet, got: {[f['domain_profile_id'] for f in facets]}"


def test_document_profile_legal_text_creates_facets():
    """upsert_document_profile with legal metadata creates workspace facets."""
    from irys.matter import MatterModel

    model = MatterModel.open_in_memory()
    model.inventory.upsert(
        relative_path="pleadings/complaint.pdf",
        sha256="sha256:def456",
    )
    model.upsert_document_profile(
        relative_path="pleadings/complaint.pdf",
        analysis={
            "title": "Complaint for Breach of Contract filed in UNITED STATES DISTRICT COURT",
            "doc_type": "complaint",
            "purpose": "Plaintiff alleges defendant breached pursuant to 500 U.S. 123. "
                       "Motion for summary judgment on damages.",
        },
    )
    broker = model.memory_broker
    facets = broker.get_object_domain_facets(
        "workspace", model.matter_id, status="active",
    )
    legal_facets = [f for f in facets if f["domain_profile_id"] == "legal"]
    assert len(legal_facets) >= 1


def test_legal_profile_trust_weights_fallback():
    """get_profile_trust_weights for legal returns SOURCE_TRUST_WEIGHTS fallback."""
    from irys.matter import MatterModel
    from irys.matter.enums import SOURCE_TRUST_WEIGHTS

    model = MatterModel.open_in_memory()
    tw = model.memory_broker.get_profile_trust_weights("legal")
    assert len(tw) > 0
    for role, val in SOURCE_TRUST_WEIGHTS.items():
        assert tw[role] == val


# --- End-to-end multi-domain pipeline validation ---


def _ingest_finance_matter():
    """Set up a matter with finance documents and return the model."""
    from irys.matter import MatterModel

    model = MatterModel.open_in_memory()
    model.inventory.upsert(
        relative_path="financials/10K_2025.pdf",
        sha256="sha256:fin001",
    )
    model.upsert_document_profile(
        relative_path="financials/10K_2025.pdf",
        analysis={
            "title": "Annual Report 10-K SEC Filing FY2025",
            "doc_type": "10-K",
            "purpose": "Revenue disclosure and EBITDA margin analysis. "
                       "Earnings per share beat analyst estimates. "
                       "Management provided forward guidance for fiscal year 2026.",
            "key_facts": [
                "Revenue increased 12% year-over-year to $2.4B",
                "EBITDA margin improved to 25%",
                "Earnings per share of $3.42 vs $3.10 estimate",
            ],
            "numeric_facts": [
                {"label": "Revenue", "value": "2.4B", "currency": "USD"},
                {"label": "EBITDA margin", "value": "25%"},
            ],
        },
    )
    return model


def test_e2e_finance_document_produces_domain_composition():
    """Full pipeline: ingest finance doc → detection → facets → composed trust weights."""
    model = _ingest_finance_matter()
    facets, tw, primary = model._read_matter_domain_composition()

    assert primary == "finance"
    assert len(tw) > 0
    assert "auditor" in tw or "regulator" in tw


def test_e2e_finance_trust_weights_differ_from_legal():
    """Composed trust weights for a finance matter should include finance-specific roles."""
    model = _ingest_finance_matter()
    _, tw, _ = model._read_matter_domain_composition()

    finance_roles = {"auditor", "regulator", "issuer_management", "analyst", "rating_agency"}
    assert finance_roles & set(tw), f"Expected finance roles, got: {set(tw)}"


def test_e2e_mixed_domain_produces_multi_profile_composition():
    """Ingest both legal and finance docs → composition should include both profiles."""
    from irys.matter import MatterModel

    model = MatterModel.open_in_memory()
    model.inventory.upsert(
        relative_path="pleadings/complaint.pdf",
        sha256="sha256:leg001",
    )
    model.upsert_document_profile(
        relative_path="pleadings/complaint.pdf",
        analysis={
            "title": "Complaint for Breach of Contract filed in UNITED STATES DISTRICT COURT",
            "doc_type": "complaint",
            "purpose": "Plaintiff alleges defendant breached pursuant to 500 U.S. 123. "
                       "Motion for summary judgment on damages.",
        },
    )
    model.inventory.upsert(
        relative_path="financials/10K_2025.pdf",
        sha256="sha256:fin002",
    )
    model.upsert_document_profile(
        relative_path="financials/10K_2025.pdf",
        analysis={
            "title": "Annual Report 10-K SEC Filing FY2025",
            "doc_type": "10-K",
            "purpose": "Revenue disclosure and EBITDA margin analysis. "
                       "EBITDA margin improved. Earnings per share beat estimates. "
                       "Management provided forward guidance.",
        },
    )
    facets, tw, primary = model._read_matter_domain_composition()
    facet_profiles = {f["domain_profile_id"] for f in facets}
    assert "legal" in facet_profiles
    assert "finance" in facet_profiles
    assert len(tw) > 0
    assert "operative" in tw or "auditor" in tw


def test_e2e_belief_engine_receives_composed_weights():
    """After ingest, belief engine trust_weights should be populated from composition."""
    from irys.matter.models import AssertionCandidate
    from irys.matter.enums import (
        AssertionKind, BeliefState, ModelLayer, OriginKind, SourceRole, SpeechAct,
    )

    model = _ingest_finance_matter()
    run_id = model.start_run("test_finance")
    a_id, _ = model.record_assertion(
        AssertionCandidate(
            proposition_text="Revenue increased 12% year-over-year.",
            model_layer=ModelLayer.RECORD,
            assertion_kind=AssertionKind.QUANTITATIVE,
            speech_act=SpeechAct.ALLEGED,
            source_role=SourceRole.OPERATIVE,
            origin_kind=OriginKind.EXTRACTED,
            document_id="financials/10K_2025.pdf",
        ),
        run_id=run_id,
    )
    model._ensure_belief_trust_weights()
    assert model.belief.trust_weights is not None
    assert len(model.belief.trust_weights) > 0


def test_e2e_brokered_correction_on_finance_matter():
    """CAS-protected correction works on a finance-domain matter."""
    from irys.matter.models import AssertionCandidate
    from irys.matter.enums import (
        AssertionKind, BeliefState, ModelLayer, OriginKind, SourceRole, SpeechAct,
    )

    model = _ingest_finance_matter()
    run_id = model.start_run("test_finance")
    a_id, _ = model.record_assertion(
        AssertionCandidate(
            proposition_text="EBITDA margin improved to 25%.",
            model_layer=ModelLayer.RECORD,
            assertion_kind=AssertionKind.QUANTITATIVE,
            speech_act=SpeechAct.ALLEGED,
            source_role=SourceRole.OPERATIVE,
            origin_kind=OriginKind.EXTRACTED,
            document_id="financials/10K_2025.pdf",
        ),
        run_id=run_id,
    )
    model.complete_run(run_id)

    revisions = model.correct_assertion_revision_keys(a_id)
    result = model.correct_assertion(
        assertion_id=a_id,
        new_state=BeliefState.OPERATIVE,
        note="Confirmed by auditor",
        expected_revisions=revisions,
    )
    assert result.assertion_id == a_id

    row = model.db.execute(
        "SELECT belief_state FROM assertion WHERE id=?", (a_id,)
    ).fetchone()
    assert row["belief_state"] == "operative"


def test_e2e_coding_domain_detection_and_composition():
    """Ingest a code artifact → detection → coding profile facets."""
    from irys.matter import MatterModel

    model = MatterModel.open_in_memory()
    model.inventory.upsert(
        relative_path="src/auth_handler.py",
        sha256="sha256:code001",
    )
    model.upsert_document_profile(
        relative_path="src/auth_handler.py",
        analysis={
            "title": "Authentication Handler Module",
            "doc_type": "source_code",
            "purpose": "Handles JWT token validation and session management. "
                       "Uses async def verify_token() with proper locking. "
                       "Race condition fix in connection pool. "
                       "```python\ndef authenticate(request):\n    pass\n```",
            "key_facts": [
                "Implements OAuth2 bearer token flow",
                "Uses asyncio.Lock for thread safety",
                "API endpoint /api/v2/auth handles refresh tokens",
            ],
        },
    )
    broker = model.memory_broker
    facets = broker.get_object_domain_facets(
        "workspace", model.matter_id, status="active",
    )
    coding_facets = [f for f in facets if f["domain_profile_id"] == "coding"]
    assert len(coding_facets) >= 1, (
        f"Expected coding facet, got: {[f['domain_profile_id'] for f in facets]}"
    )


def test_e2e_academic_research_detection_and_composition():
    """Ingest a research paper → detection → academic_research profile facets + trust weights."""
    from irys.matter import MatterModel

    model = MatterModel.open_in_memory()
    model.inventory.upsert(
        relative_path="papers/rct_cognition_2024.pdf",
        sha256="sha256:res001",
    )
    model.upsert_document_profile(
        relative_path="papers/rct_cognition_2024.pdf",
        analysis={
            "title": "Randomized Controlled Trial of Cognitive Enhancement",
            "doc_type": "peer_reviewed_paper",
            "purpose": "Results from RCT (n=500) show statistically significant effect "
                       "(p < 0.001, 95% CI [0.12, 0.45]). Meta-analysis confirms effect "
                       "size Cohen's d = 0.38. doi: 10.1234/cognition.2024",
            "key_facts": [
                "Hypothesis tested via randomized controlled trial",
                "Effect replicated in independent cohort (Smith et al., 2024)",
                "Methodology: double-blind placebo-controlled",
            ],
            "numeric_facts": [
                {"label": "sample_size", "value": "500"},
                {"label": "p_value", "value": "<0.001"},
                {"label": "effect_size", "value": "0.38"},
            ],
        },
    )
    facets, tw, primary = model._read_matter_domain_composition()
    assert primary == "academic_research", f"Expected academic_research primary, got {primary}"
    research_roles = {"peer_reviewed_paper", "replication_study", "dataset", "preprint"}
    assert research_roles & set(tw), f"Expected research roles, got: {set(tw)}"


def test_e2e_biomedical_detection_and_composition():
    """Ingest a clinical trial report → detection → biomedical profile facets + trust weights."""
    from irys.matter import MatterModel

    model = MatterModel.open_in_memory()
    model.inventory.upsert(
        relative_path="trials/phase3_mab_2025.pdf",
        sha256="sha256:bio001",
    )
    model.upsert_document_profile(
        relative_path="trials/phase3_mab_2025.pdf",
        analysis={
            "title": "Phase III Clinical Trial of Monoclonal Antibody NCT01234567",
            "doc_type": "clinical_trial_report",
            "purpose": "Phase III trial (NCT01234567) enrolled 240 patients receiving "
                       "monoclonal antibody at 5 mg/kg. Primary endpoint: hazard ratio 0.65 "
                       "(p=0.003). Adverse events consistent with known safety profile. "
                       "FDA approved for new indication.",
            "key_facts": [
                "240 patients enrolled across 12 sites",
                "Hazard ratio 0.65 for primary endpoint",
                "FDA approval granted for expanded indication",
            ],
            "numeric_facts": [
                {"label": "patients", "value": "240"},
                {"label": "hazard_ratio", "value": "0.65"},
                {"label": "p_value", "value": "0.003"},
            ],
        },
    )
    facets, tw, primary = model._read_matter_domain_composition()
    assert primary == "biomedical", f"Expected biomedical primary, got {primary}"
    biomed_roles = {"phase_iii_trial", "regulator", "clinical_guideline", "lab_result"}
    assert biomed_roles & set(tw), f"Expected biomedical roles, got: {set(tw)}"


def test_e2e_all_five_domains_compose_together():
    """Ingest one doc from each of 5 domains → all detected, composition has multiple facets."""
    from irys.matter import MatterModel

    model = MatterModel.open_in_memory()
    docs = [
        ("pleadings/complaint.pdf", "sha256:d1", {
            "title": "Complaint filed in UNITED STATES DISTRICT COURT",
            "doc_type": "complaint",
            "purpose": "Plaintiff alleges breach pursuant to 500 U.S. 123. Summary judgment motion.",
        }),
        ("financials/10K.pdf", "sha256:d2", {
            "title": "10-K SEC Filing",
            "doc_type": "10-K",
            "purpose": "Revenue disclosure. EBITDA margin 25%. Earnings per share beat estimates. Forward guidance.",
        }),
        ("src/handler.py", "sha256:d3", {
            "title": "Request Handler",
            "doc_type": "source_code",
            "purpose": "```python\nasync def handle(req):\n    pass\n```\nFix race condition in connection pool. Run npm test.",
            "key_facts": ["OAuth2 bearer token flow", "asyncio.Lock for thread safety"],
        }),
        ("papers/rct.pdf", "sha256:d4", {
            "title": "Randomized Controlled Trial",
            "doc_type": "peer_reviewed_paper",
            "purpose": "RCT (n=500) p < 0.001 CI [0.12, 0.45]. Meta-analysis. doi: 10.1234/x.2024",
        }),
        ("trials/phase3.pdf", "sha256:d5", {
            "title": "Phase III Clinical Trial NCT01234567",
            "doc_type": "clinical_trial_report",
            "purpose": "Phase III trial NCT01234567 240 patients monoclonal antibody 5 mg/kg. "
                       "Hazard ratio 0.65 p=0.003. FDA approved.",
        }),
    ]
    for path, sha, analysis in docs:
        model.inventory.upsert(relative_path=path, sha256=sha)
        model.upsert_document_profile(relative_path=path, analysis=analysis)

    broker = model.memory_broker
    facets = broker.get_object_domain_facets(
        "workspace", model.matter_id, status="active",
    )
    detected_profiles = {f["domain_profile_id"] for f in facets}
    assert len(detected_profiles) >= 3, (
        f"Expected at least 3 domain profiles detected, got: {detected_profiles}"
    )
    _, tw, primary = model._read_matter_domain_composition()
    assert len(tw) > 0
    assert primary is not None
