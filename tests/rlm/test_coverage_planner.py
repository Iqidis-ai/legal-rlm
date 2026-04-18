"""P0.7 Coverage-Driven Lead Planner tests.

The planner proactively injects issue-targeted leads based on the
matter model's coverage state, not the user's utterance. Covered:
  - weakest gapped issue becomes a coverage_planner lead with the
    first open predicate as search_term and focus_issue_id set
  - no lead when the issue lane is full or a duplicate already exists
  - per-iteration cap (2) and per-run cap (6) enforced
  - already-strong, no-gap issues are skipped
  - low-materiality issues are skipped
"""

import pytest

from irys.matter import MatterModel, IssueType
from irys.rlm.engine import RLMEngine, RLMConfig
from irys.rlm.governance import ExecutionContract
from irys.rlm.state import InvestigationState, Lead


class _StubClient:
    pass


@pytest.fixture
def model():
    return MatterModel.open_in_memory()


@pytest.fixture
def engine(model):
    e = RLMEngine(gemini_client=_StubClient(), config=RLMConfig())
    e._matter_model = model
    return e


def _state(**overrides) -> InvestigationState:
    s = InvestigationState.create("test query", "/tmp/repo", research_mode="deep")
    s.execution_contract = ExecutionContract(
        family="investigate", lead_ev_floor=0.0,
    )
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def _seed_weak_issue(model, title, materiality=0.8, with_predicate=True):
    """Create a weak issue (no supporting assertions) with an optional
    predicate so the planner has a search_term to use."""
    iid, _ = model.issues.upsert_issue(
        title=title, issue_type=IssueType.CLAIM,
        materiality=materiality,
    )
    if with_predicate:
        model.issues.add_predicate(
            iid, description=f"Plaintiff must show {title.lower()}",
            burden_side="plaintiff",
        )
    return iid


def test_planner_injects_weakest_issue_lead(engine, model):
    """Weakest material issue with a predicate gets a coverage_planner
    lead carrying that predicate as search_term."""
    iid = _seed_weak_issue(model, "Breach of contract")
    s = _state()
    cov_map = engine._get_issue_coverage_map()
    assert iid in cov_map

    added = engine._coverage_planner(s, cov_map)
    assert added == 1

    planner_leads = [l for l in s.leads if l.source == "coverage_planner"]
    assert len(planner_leads) == 1
    lead = planner_leads[0]
    assert lead.focus_issue_id == iid
    assert "plaintiff must show" in (lead.search_term or "").lower()
    # EV enrichment must have run — cost and gain stamped.
    assert lead.expected_cost_usd > 0
    assert lead.expected_coverage_gain > 0


def test_planner_respects_per_iteration_cap(engine, model):
    """With 5 weak issues in the queue, a single call still caps at
    the _PLANNER_LEADS_PER_ITER=2 limit."""
    for i in range(5):
        _seed_weak_issue(model, f"Issue {i}")
    s = _state()
    cov_map = engine._get_issue_coverage_map()

    added = engine._coverage_planner(s, cov_map)
    assert added == engine._PLANNER_LEADS_PER_ITER


def test_planner_respects_per_run_cap(engine, model):
    """Across many iterations the planner must never exceed
    _PLANNER_LEADS_PER_RUN=6 per run. Between iterations we simulate
    the engine consuming leads (marking them investigated) so the
    issue-quota deficit reopens — mirrors the real loop.
    """
    for i in range(10):
        _seed_weak_issue(model, f"Issue {i}")
    s = _state()
    cov_map = engine._get_issue_coverage_map()

    total = 0
    for _ in range(10):
        added = engine._coverage_planner(s, cov_map)
        total += added
        if added == 0:
            continue
        # Simulate engine dispatch — mark every pending planner lead
        # as investigated so the deficit reopens for the next call.
        for l in s.get_pending_leads():
            s.mark_lead_investigated(l.id, "test dispatch")
    assert total == engine._PLANNER_LEADS_PER_RUN
    assert s.planner_leads_added == engine._PLANNER_LEADS_PER_RUN
    # After hitting the per-run cap, further calls must be no-ops
    # even if we keep consuming leads.
    assert engine._coverage_planner(s, cov_map) == 0


def test_planner_skips_already_issue_focused_leads(engine, model):
    """When an issue already has a pending issue-targeted lead in the
    state, the planner must not emit a duplicate for it. Codex design
    requirement — avoid crowding out reactive work."""
    iid = _seed_weak_issue(model, "Breach")
    s = _state()
    # Pre-existing user/reactive lead already focused on this issue.
    s.add_lead(
        description="User asked: did we get notice?",
        source="user",
        priority=0.8,
        focus_issue_id=iid,
    )
    cov_map = engine._get_issue_coverage_map()
    added = engine._coverage_planner(s, cov_map)
    assert added == 0


def test_planner_skips_low_materiality_issues(engine, model):
    """Issues below 0.4 materiality are not worth a lead — the
    clarification end-of-run pass handles marginal questions."""
    _seed_weak_issue(model, "Marginal wording nit", materiality=0.2)
    s = _state()
    cov_map = engine._get_issue_coverage_map()
    added = engine._coverage_planner(s, cov_map)
    assert added == 0


def test_planner_no_ops_when_family_is_not_investigate(engine, model):
    """A read / query / deliverable contract should not trigger the
    planner — it's an investigate-family tool."""
    _seed_weak_issue(model, "Breach")
    s = _state()
    s.execution_contract = ExecutionContract(family="read", lead_ev_floor=0.0)
    cov_map = engine._get_issue_coverage_map()
    added = engine._coverage_planner(s, cov_map)
    assert added == 0


def test_historical_reactive_lead_does_not_permanently_block_planner(engine, model):
    """adv#11 P0.7.1 review fix #1: the planner used to dedup against
    EVERY issue-focused lead in state.leads — so a reactive lead that
    was already investigated but didn't move coverage would block the
    planner forever on that issue. Now dedup scope is:
      - pending issue-focused leads (any source) block THIS iter
      - historical reactive leads do NOT block planner
      - historical coverage_planner leads block only same (iid, term)
    """
    iid = _seed_weak_issue(model, "Notice timing dispute")
    s = _state()
    # A prior reactive lead for this issue, already consumed.
    prior = s.add_lead(
        description="Search for notice correspondence",
        source="recursive",
        priority=0.6,
        focus_issue_id=iid,
    )
    s.mark_lead_investigated(prior.id, "consumed last iter")

    # Coverage unchanged — still weak. Planner must now fill the gap.
    cov_map = engine._get_issue_coverage_map()
    added = engine._coverage_planner(s, cov_map)
    assert added == 1
    planner_lead = next(l for l in s.leads if l.source == "coverage_planner")
    assert planner_lead.focus_issue_id == iid


def test_same_issue_same_term_planner_dedup(engine, model):
    """adv#11 P0.7.1 review fix #1b: two planner runs on the same
    (issue_id, normalized_term) must dedup — don't burn budget on the
    identical planner lead twice. The first call adds it; the second
    (after consuming it to reopen deficit) sees the historical term
    and skips."""
    iid = _seed_weak_issue(model, "Exclusivity clause")
    s = _state()
    cov_map = engine._get_issue_coverage_map()
    assert engine._coverage_planner(s, cov_map) == 1
    # Consume it so deficit reopens, but the historical lead stays.
    for l in s.get_pending_leads():
        s.mark_lead_investigated(l.id, "consumed")
    # Second pass: same issue, same predicate → dedup.
    assert engine._coverage_planner(s, cov_map) == 0


def test_planner_leads_added_survives_checkpoint(engine, model):
    """adv#11 P0.7.1 review fix #3: the per-run cap counter must
    survive to_dict/from_dict so a checkpoint/resume doesn't reset
    the cap and allow a fresh 6 planner leads."""
    _seed_weak_issue(model, "Breach")
    s = _state()
    cov_map = engine._get_issue_coverage_map()
    engine._coverage_planner(s, cov_map)
    assert s.planner_leads_added > 0

    dumped = s.to_dict()
    from irys.rlm.state import InvestigationState
    restored = InvestigationState.from_dict(dumped)
    assert restored.planner_leads_added == s.planner_leads_added


def test_planner_lead_passes_ev_gate(engine, model):
    """The planner lead must be enriched AND viable under a normal
    lead_ev_floor. Weak material issues produce enough coverage gain
    that ev_score clears 0.5."""
    _seed_weak_issue(model, "Breach")
    s = _state()
    cov_map = engine._get_issue_coverage_map()
    engine._coverage_planner(s, cov_map)
    planner_lead = next(l for l in s.leads if l.source == "coverage_planner")

    realistic = ExecutionContract(family="investigate", lead_ev_floor=0.5)
    assert engine._viable_leads([planner_lead], contract=realistic) == [planner_lead]
