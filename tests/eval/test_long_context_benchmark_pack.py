from tests.eval.benchmark_loader import (
    check_forbidden_output_patterns,
    ensure_unique_ids,
    load_benchmark_pack,
    load_corpus_benchmark,
    run_pack_contract_checks,
)


def test_long_context_benchmark_pack_has_unique_query_ids():
    pack = load_benchmark_pack("long_context_mixed_v1")

    assert pack.name == "long_context_mixed_v1"
    ensure_unique_ids(pack.queries)
    assert pack.run_policy.cheap_contract_checks_first is True


def test_long_context_benchmark_pack_matches_task_ontology():
    pack = load_benchmark_pack("long_context_mixed_v1")

    results = run_pack_contract_checks(pack)

    assert all(result.passed for result in results), [
        (result.query_id, result.failures, result.observed)
        for result in results
        if not result.passed
    ]


def test_long_context_benchmark_pack_covers_failure_families():
    pack = load_benchmark_pack("long_context_mixed_v1")
    families = pack.families()
    styles = pack.styles()

    assert "transactional_long_context" in families
    assert "litigation_long_context" in families
    assert "wilson_style_synthesis" in families
    assert "quantitative_analysis" in families
    assert "adversarial_absence" in families
    assert "scoped_quant_reconciliation" in styles
    assert "identity_with_conflicts" in styles


def test_legal_tester_regression_benchmark_jsonl_is_loadable():
    rows = load_corpus_benchmark("legal_tester_regression_eval.jsonl")

    assert len(rows) >= 10
    ensure_unique_ids(rows)
    for row in rows:
        assert row.query
        assert row.category
        assert row.gold_answer_sketch
        assert row.required_capabilities
        assert row.ontology_stress


def test_forbidden_output_patterns_classify_known_bad_artifacts():
    pack = load_benchmark_pack("long_context_mixed_v1")
    by_id = {item.id: item for item in pack.queries}

    compare_result = check_forbidden_output_patterns(
        by_id["DELE-002"],
        "## What changed\n\nAssertions 75 -> 75 (+0).",
    )
    lookup_result = check_forbidden_output_patterns(
        by_id["DELE-008"],
        "## List Documents\n\n- Lion.pdf (32 pending)",
    )
    clean_result = check_forbidden_output_patterns(
        by_id["DELE-015"],
        "The biggest risk is the BSR force-majeure carve-out.",
    )

    assert compare_result.passed is False
    assert "## What changed" in compare_result.failures[0]
    assert lookup_result.passed is False
    assert "## List Documents" in lookup_result.failures[0]
    assert clean_result.passed is True
