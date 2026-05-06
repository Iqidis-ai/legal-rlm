from irys.matter import AbsenceStatus, AbsenceStatusRecord


def test_absence_status_record_distinguishes_negative_answer_from_missing_source():
    record = AbsenceStatusRecord(
        target="Section 19.7 ESG covenants",
        status=AbsenceStatus.FALSE_PREMISE_LIKELY,
        searched_documents=3,
        searched_terms=("Section 19.7", "ESG covenants"),
        confidence=0.82,
        rationale="No matching provision in operative agreements.",
    )

    assert record.is_negative_answer is True
    assert record.requires_more_source is False
    prompt = record.to_prompt_line()
    assert "status=false_premise_likely" in prompt
    assert "searched_documents=3" in prompt
    assert "Section 19.7" in prompt


def test_absence_status_source_missing_is_not_a_negative_answer():
    record = AbsenceStatusRecord(
        target="unproduced amendment",
        status=AbsenceStatus.SOURCE_MISSING,
    )

    assert record.is_negative_answer is False
    assert record.requires_more_source is True
