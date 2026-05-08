"""Tests for the slot-profile registry, dispatcher, and built-in market_row profile."""

from __future__ import annotations

import pytest

from irys.rlm.slot_profiles import (
    LegalCpGapProfile,
    LegalMarketRowProfile,
    LegalQoeLineItemProfile,
    ProfileMatch,
    SlotProfile,
    SlotProfileContext,
    SlotProfileDispatch,
    SlotProfileRegistry,
    SlotRegistration,
    TypedEvidenceWrite,
    default_registry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubProfile:
    """Tiny profile for dispatcher tests — match decided by the query terms."""

    def __init__(
        self,
        *,
        profile_id: str,
        score: float = 1.0,
        priority: int = 100,
        exclusive_group=None,
        version: int = 1,
        enabled: bool = True,
        match_term: str = "",
    ) -> None:
        self.profile_id = profile_id
        self.profile_version = version
        self.domain_profile_ids = ("legal:1",)
        self.schema_ref = profile_id
        self.record_kind = "stub"
        self.slot_kind = "stub"
        self.priority = priority
        self.exclusive_group = exclusive_group
        self.supports_mixed_corpus = True
        self.enabled = enabled
        self._score = score
        self._match_term = match_term

    def match(self, context):
        if self._match_term and self._match_term not in (context.query or "").lower():
            return None
        return ProfileMatch(profile_id=self.profile_id, score=self._score)

    def scout(self, context, runtime):
        return ()

    def prompt_addendum(self, context):
        return ""

    def parse_evidence(self, *, analysis, document, context):
        return ()

    def render_answer_rows(self, rows):
        return ""


# ---------------------------------------------------------------------------
# Registry basics
# ---------------------------------------------------------------------------


def test_registry_register_duplicate_raises():
    reg = SlotProfileRegistry()
    p = _StubProfile(profile_id="x")
    reg.register(p)
    with pytest.raises(ValueError):
        reg.register(_StubProfile(profile_id="x"))


def test_registry_get_returns_none_for_unknown():
    reg = SlotProfileRegistry()
    assert reg.get("not.a.profile") is None


def test_default_registry_includes_market_row():
    reg = default_registry()
    p = reg.get("legal.market_row.v1")
    assert p is not None
    assert p.schema_ref == "legal.market_row.v1"


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def test_dispatch_no_match_when_query_doesnt_match():
    reg = default_registry()
    disp = reg.dispatch(SlotProfileContext.empty(query="review change of control"))
    assert disp.selected == ()
    assert disp.candidates == ()


def test_dispatch_market_row_matches_antitrust_query():
    reg = default_registry()
    disp = reg.dispatch(SlotProfileContext.empty(query="antitrust HSR strategy"))
    assert len(disp.selected) == 1
    assert disp.selected[0].profile_id == "legal.market_row.v1"


def test_dispatch_disabled_profile_skipped():
    reg = SlotProfileRegistry()
    reg.register(_StubProfile(profile_id="enabled.v1", match_term="x", score=0.9))
    reg.register(_StubProfile(
        profile_id="disabled.v1", match_term="x", score=0.95, enabled=False,
    ))
    disp = reg.dispatch(SlotProfileContext.empty(query="x"))
    assert [p.profile_id for p in disp.selected] == ["enabled.v1"]


def test_dispatch_orders_by_score_then_priority():
    reg = SlotProfileRegistry()
    reg.register(_StubProfile(profile_id="hi.v1", match_term="q", score=0.9, priority=50))
    reg.register(_StubProfile(profile_id="best.v1", match_term="q", score=0.95, priority=10))
    reg.register(_StubProfile(profile_id="med.v1", match_term="q", score=0.7, priority=200))
    disp = reg.dispatch(SlotProfileContext.empty(query="q"))
    assert [p.profile_id for p in disp.selected] == ["best.v1", "hi.v1", "med.v1"]


def test_dispatch_exclusive_group_suppresses_lower_rank():
    reg = SlotProfileRegistry()
    reg.register(_StubProfile(
        profile_id="a.v1", match_term="q", score=0.9, exclusive_group="g",
    ))
    reg.register(_StubProfile(
        profile_id="b.v1", match_term="q", score=0.95, exclusive_group="g",
    ))
    disp = reg.dispatch(SlotProfileContext.empty(query="q"))
    assert [p.profile_id for p in disp.selected] == ["b.v1"]
    assert any(s["profile_id"] == "a.v1" for s in disp.suppressed)


def test_dispatch_prompt_cap_drops_overflow():
    reg = SlotProfileRegistry()
    for i, score in enumerate([0.95, 0.9, 0.85, 0.8]):
        reg.register(_StubProfile(profile_id=f"p{i}.v1", match_term="q", score=score))
    disp = reg.dispatch(SlotProfileContext.empty(query="q"), prompt_profile_cap=2)
    assert len(disp.prompt_profiles) == 2
    assert disp.prompt_cap_drops == ("p2.v1", "p3.v1")


def test_dispatch_resilient_to_broken_match():
    class _Broken(_StubProfile):
        def match(self, context):
            raise RuntimeError("boom")

    reg = SlotProfileRegistry()
    reg.register(_Broken(profile_id="bad.v1"))
    reg.register(_StubProfile(profile_id="ok.v1", match_term="q", score=0.7))
    disp = reg.dispatch(SlotProfileContext.empty(query="q"))
    assert [p.profile_id for p in disp.selected] == ["ok.v1"]


# ---------------------------------------------------------------------------
# Built-in: legal.market_row.v1
# ---------------------------------------------------------------------------


def test_market_row_match_terms():
    p = LegalMarketRowProfile()
    for q in ["antitrust", "HSR strategy", "market share", "HHI analysis"]:
        m = p.match(SlotProfileContext.empty(query=q))
        assert m is not None, f"failed to match: {q}"
        assert m.profile_id == "legal.market_row.v1"


def test_market_row_skips_inventory_lookup_family():
    p = LegalMarketRowProfile()
    ctx = SlotProfileContext(
        matter_id="m", query="antitrust HHI",
        task_spec={"family": "inventory_lookup"},
        domain_composition_hash="", domain_facets=(), corpus_signals={},
    )
    assert p.match(ctx) is None


def test_market_row_parse_evidence_returns_one_per_row():
    p = LegalMarketRowProfile()
    analysis = {"market_rows": [
        {
            "market_name": "Atlanta MSA",
            "post_merger_hhi": 2681,
            "delta_hhi": 836,
            "structural_presumption": True,
        },
        {
            "market_name": "Birmingham MSA",
            "acquirer_share": "28%",
            "target_share": "18%",
        },
        {"market_name": ""},  # empty — should be skipped
        {"market_name": "Empty MSA"},  # no signals — should be skipped
    ]}
    class _Doc:
        filename = "memo.docx"
    writes = p.parse_evidence(
        analysis=analysis, document=_Doc(), context=SlotProfileContext.empty(),
    )
    assert len(writes) == 2
    assert all(w.record_kind == "market_row" for w in writes)
    assert writes[0].slot_key.endswith(":msa:atlanta")


def test_market_row_render_orders_presumption_first():
    p = LegalMarketRowProfile()
    rows = [
        {"payload": {"market_name": "X", "delta_hhi": 200,
                     "structural_presumption": False}},
        {"payload": {"market_name": "Y", "delta_hhi": 800,
                     "structural_presumption": True}},
        {"payload": {"market_name": "Z", "delta_hhi": 500,
                     "structural_presumption": False}},
    ]
    out = p.render_answer_rows(rows)
    # Y should appear first (presumption)
    assert out.index("| Y |") < out.index("| Z |")
    assert out.index("| Z |") < out.index("| X |")


def test_market_row_render_empty_returns_empty():
    p = LegalMarketRowProfile()
    assert p.render_answer_rows([]) == ""


# ---------------------------------------------------------------------------
# Built-in: legal.cp_gap.v1
# ---------------------------------------------------------------------------


def test_cp_gap_match_banking_compare_terms():
    p = LegalCpGapProfile()
    matched_queries = [
        "compare credit agreement against term sheet",
        "compare closing documents against conditions precedent",
        "compare borrower disclosures against due diligence findings",
        "compliance certificate review",
    ]
    for q in matched_queries:
        m = p.match(SlotProfileContext.empty(query=q))
        assert m is not None, f"failed: {q}"
        assert m.profile_id == "legal.cp_gap.v1"


def test_cp_gap_skips_non_compare_queries():
    p = LegalCpGapProfile()
    for q in ["antitrust hsr", "review the contract", "extract entities"]:
        assert p.match(SlotProfileContext.empty(query=q)) is None


def test_cp_gap_parse_returns_one_row_per_requirement():
    p = LegalCpGapProfile()
    class _Doc:
        filename = "credit_agreement.pdf"
    analysis = {"cp_gaps": [
        {"cp_id": "4.01(a)", "requirement_text": "secretary cert",
         "status": "missing", "severity": "critical",
         "required_by_document": "Credit Agreement.pdf"},
        {"cp_id": "4.01(b)", "requirement_text": "opinion letter",
         "status": "met"},
        {"cp_id": "", "requirement_text": ""},  # empty — skipped
    ]}
    writes = p.parse_evidence(
        analysis=analysis, document=_Doc(), context=SlotProfileContext.empty(),
    )
    assert len(writes) == 2
    # Slot key uses requirement identity, not status (so two revisions of
    # the same CP collapse to the same slot)
    assert all(w.slot_key.startswith("obligation:legal.cp_gap.v1:cp:") for w in writes)


def test_cp_gap_slot_key_dedupes_by_requirement_only():
    """R5 patch: drop status from slot_key — two revisions of same CP fill same slot."""
    p = LegalCpGapProfile()
    class _Doc:
        filename = "credit_agreement.pdf"
    analysis = {"cp_gaps": [
        {"cp_id": "4.01(a)", "requirement_text": "secretary cert",
         "status": "missing", "required_by_document": "Credit Agreement.pdf"},
        {"cp_id": "4.01(a)", "requirement_text": "secretary cert",
         "status": "met", "required_by_document": "Credit Agreement.pdf"},
    ]}
    writes = p.parse_evidence(
        analysis=analysis, document=_Doc(), context=SlotProfileContext.empty(),
    )
    assert writes[0].slot_key == writes[1].slot_key
    # Typed evidence keys differ (different generations)
    assert writes[0].record_key != writes[1].record_key


def test_cp_gap_render_orders_missing_first():
    p = LegalCpGapProfile()
    rows = [
        {"payload": {"cp_id": "M", "requirement_text": "met item",
                     "status": "met", "severity": "high"}},
        {"payload": {"cp_id": "X", "requirement_text": "missing item",
                     "status": "missing", "severity": "critical"}},
        {"payload": {"cp_id": "P", "requirement_text": "partial item",
                     "status": "partial", "severity": "high"}},
    ]
    out = p.render_answer_rows(rows)
    assert out.index("| X |") < out.index("| P |") < out.index("| M |")


def test_cp_gap_issue_link_relation_attacks_when_missing():
    p = LegalCpGapProfile()
    class _Doc:
        filename = "ca.pdf"
    writes = p.parse_evidence(
        analysis={"cp_gaps": [
            {"cp_id": "1", "requirement_text": "x", "status": "missing"},
            {"cp_id": "2", "requirement_text": "y", "status": "met"},
        ]},
        document=_Doc(), context=SlotProfileContext.empty(),
    )
    assert writes[0].issue_link.relation == "attacks"
    assert writes[1].issue_link.relation == "supports"


# ---------------------------------------------------------------------------
# Built-in: legal.qoe_line_item.v1
# ---------------------------------------------------------------------------


def test_qoe_match_qoe_terms():
    p = LegalQoeLineItemProfile()
    for q in ["analyze qoe reconciliation", "ebitda bridge analysis",
              "purchase price allocation review"]:
        m = p.match(SlotProfileContext.empty(query=q))
        assert m is not None, f"failed: {q}"


def test_qoe_skips_non_qoe_queries():
    p = LegalQoeLineItemProfile()
    for q in ["antitrust", "review credit agreement", "compare cp"]:
        assert p.match(SlotProfileContext.empty(query=q)) is None


def test_qoe_parse_returns_one_row_per_line_item():
    p = LegalQoeLineItemProfile()
    class _Doc:
        filename = "qoe.pdf"
    analysis = {"qoe_line_items": [
        {"category": "EBITDA bridge", "line_item_label": "Owner comp",
         "period": "FY2024", "schedule": "EBITDA Bridge",
         "currency": "USD", "seller_value": "$0.8M", "buyer_value": "$1.3M"},
        {"category": "NWC", "line_item_label": "Inventory reserve",
         "period": "LTM Mar 2025", "schedule": "NWC schedule"},
    ]}
    writes = p.parse_evidence(
        analysis=analysis, document=_Doc(), context=SlotProfileContext.empty(),
    )
    assert len(writes) == 2
    # R5 patch: slot key includes artifact_family/schedule/currency/category/period/label
    assert all("measurement:legal.qoe_line_item.v1:" in w.slot_key for w in writes)


def test_qoe_slot_key_distinguishes_periods():
    """Same label, different periods should produce different slot keys."""
    p = LegalQoeLineItemProfile()
    class _Doc:
        filename = "qoe.pdf"
    writes = p.parse_evidence(
        analysis={"qoe_line_items": [
            {"category": "EBITDA bridge", "line_item_label": "Owner comp",
             "period": "FY2023"},
            {"category": "EBITDA bridge", "line_item_label": "Owner comp",
             "period": "FY2024"},
        ]},
        document=_Doc(), context=SlotProfileContext.empty(),
    )
    assert writes[0].slot_key != writes[1].slot_key


def test_default_registry_includes_three_profiles():
    reg = default_registry()
    ids = {p.profile_id for p in reg.all()}
    assert ids == {
        "legal.market_row.v1",
        "legal.cp_gap.v1",
        "legal.qoe_line_item.v1",
    }


def test_dispatcher_routes_to_correct_profile():
    reg = default_registry()
    cases = [
        ("compare expert market share data", "legal.market_row.v1"),
        ("compare credit agreement against term sheet", "legal.cp_gap.v1"),
        ("analyze qoe reconciliation", "legal.qoe_line_item.v1"),
    ]
    for query, expected_profile in cases:
        disp = reg.dispatch(SlotProfileContext.empty(query=query))
        ids = [p.profile_id for p in disp.selected]
        assert expected_profile in ids, f"{query} -> {ids}"
