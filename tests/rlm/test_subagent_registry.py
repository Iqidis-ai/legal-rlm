"""Tests for sub-agent registry, dispatcher, persona policy, and budget."""

from __future__ import annotations

import asyncio

import pytest

from irys.rlm.agents import (
    AgentArtifact,
    AgentInvocation,
    AgentInvocationResult,
    AgentMatch,
    AgentRequirement,
    AgentTaskView,
    OperatorBudget,
    OperatorBudgetExceeded,
    Persona,
    PersonaPolicy,
    PersonaRegistry,
    SubAgentDispatcher,
    SubAgentRegistry,
)


# ---------------------------------------------------------------------------
# Test stubs
# ---------------------------------------------------------------------------


class _Agent:
    """Tiny configurable agent for dispatcher/policy tests."""

    def __init__(
        self,
        *,
        agent_id: str,
        score: float = 1.0,
        priority: int = 100,
        capability_tags=("compute",),
        phases=("pre_synthesis",),
        version: int = 1,
        enabled: bool = True,
        exclusive_group=None,
        deterministic: bool = True,
        invoke_status: str = "success",
        elapsed_ms: int = 10,
    ) -> None:
        self.agent_id = agent_id
        self.version = version
        self.enabled = enabled
        self.priority = priority
        self.capability_tags = tuple(capability_tags)
        self.supported_domain_profiles = ("legal:1",)
        self.phases = tuple(phases)
        self.exclusive_group = exclusive_group
        self.deterministic = deterministic
        self._score = score
        self._invoke_status = invoke_status
        self._elapsed_ms = elapsed_ms

    def match(self, invocation):
        return AgentMatch(agent_id=self.agent_id, score=self._score)

    async def invoke(self, invocation, runtime):
        return AgentInvocationResult(
            status=self._invoke_status,
            elapsed_ms=self._elapsed_ms,
            artifacts=(
                AgentArtifact(
                    artifact_kind="stub",
                    artifact_key=f"k:{self.agent_id}",
                    payload={"agent": self.agent_id},
                ),
            ) if self._invoke_status == "success" else (),
        )

    def verify_output(self, invocation, result):
        return result


def _make_invocation(*, requirement=AgentRequirement.OPTIONAL, budget=None,
                     matter_id="m"):
    return AgentInvocation(
        matter_id=matter_id,
        run_id="r",
        agent_id="dispatcher",
        phase="pre_synthesis",
        persona_id=None,
        requirement=requirement,
        task=AgentTaskView(),
        execution_family="investigate",
        workflow_kind="default",
        budget=budget or OperatorBudget(),
        input_refs=(),
        input_hash="h",
    )


# ---------------------------------------------------------------------------
# Registry basics
# ---------------------------------------------------------------------------


def test_registry_register_duplicate_raises():
    reg = SubAgentRegistry()
    reg.register(_Agent(agent_id="a"))
    with pytest.raises(ValueError):
        reg.register(_Agent(agent_id="a"))


def test_registry_dispatch_picks_matched_agents():
    reg = SubAgentRegistry()
    reg.register(_Agent(agent_id="a", score=0.9))
    reg.register(_Agent(agent_id="b", score=0.8))
    inv = _make_invocation()
    disp = reg.dispatch(inv, phase="pre_synthesis", phase_cap=5)
    assert [a.agent_id for a in disp.selected] == ["a", "b"]


def test_registry_dispatch_orders_by_score():
    reg = SubAgentRegistry()
    reg.register(_Agent(agent_id="lo", score=0.5))
    reg.register(_Agent(agent_id="hi", score=0.95))
    reg.register(_Agent(agent_id="mid", score=0.7))
    inv = _make_invocation()
    disp = reg.dispatch(inv, phase="pre_synthesis", phase_cap=5)
    assert [a.agent_id for a in disp.selected] == ["hi", "mid", "lo"]


def test_registry_dispatch_phase_cap():
    reg = SubAgentRegistry()
    for i in range(5):
        reg.register(_Agent(agent_id=f"a{i}", score=1.0 - i * 0.05))
    inv = _make_invocation()
    disp = reg.dispatch(inv, phase="pre_synthesis", phase_cap=2)
    assert len(disp.selected) == 2
    assert any(s.get("reason") == "phase_cap" for s in disp.suppressed)


def test_registry_dispatch_phase_mismatch():
    reg = SubAgentRegistry()
    reg.register(_Agent(agent_id="post", score=1.0, phases=("post_extract",)))
    inv = _make_invocation()
    disp = reg.dispatch(inv, phase="pre_synthesis", phase_cap=5)
    assert disp.selected == ()
    assert any(s.get("reason") == "phase_mismatch" for s in disp.suppressed)


def test_registry_dispatch_disabled_skipped():
    reg = SubAgentRegistry()
    reg.register(_Agent(agent_id="off", score=1.0, enabled=False))
    reg.register(_Agent(agent_id="on", score=0.5))
    inv = _make_invocation()
    disp = reg.dispatch(inv, phase="pre_synthesis", phase_cap=5)
    assert [a.agent_id for a in disp.selected] == ["on"]


def test_registry_dispatch_resilient_to_match_exception():
    class _Broken(_Agent):
        def match(self, invocation):
            raise RuntimeError("boom")
    reg = SubAgentRegistry()
    reg.register(_Broken(agent_id="bad"))
    reg.register(_Agent(agent_id="good", score=0.7))
    inv = _make_invocation()
    disp = reg.dispatch(inv, phase="pre_synthesis", phase_cap=5)
    assert [a.agent_id for a in disp.selected] == ["good"]


def test_registry_dispatch_exclusive_group():
    reg = SubAgentRegistry()
    reg.register(_Agent(agent_id="a", score=0.9, exclusive_group="g"))
    reg.register(_Agent(agent_id="b", score=0.95, exclusive_group="g"))
    inv = _make_invocation()
    disp = reg.dispatch(inv, phase="pre_synthesis", phase_cap=5)
    assert [a.agent_id for a in disp.selected] == ["b"]


# ---------------------------------------------------------------------------
# PersonaPolicy: open by default, restrictions explicit opt-in
# ---------------------------------------------------------------------------


def test_persona_policy_default_allows_all():
    p = PersonaPolicy()
    a = _Agent(agent_id="x", capability_tags=("anything", "goes"))
    d = p.evaluate(a)
    assert d.allowed
    assert d.reason == "no_policy_default_allow"


def test_persona_policy_wildcard_allows_all():
    p = PersonaPolicy(allowed_tags=("*",))
    assert p.evaluate(_Agent(agent_id="x")).allowed


def test_persona_policy_denied_tag_blocks():
    p = PersonaPolicy(denied_tags=("compute.numerical",))
    a = _Agent(agent_id="x", capability_tags=("compute.numerical",))
    d = p.evaluate(a)
    assert not d.allowed
    assert d.reason.startswith("denied_tag")


def test_persona_policy_explicit_override_overrides_tags():
    p = PersonaPolicy(
        denied_tags=("compute.numerical",),
        explicit_agent_overrides={"x": True},
    )
    a = _Agent(agent_id="x", capability_tags=("compute.numerical",))
    d = p.evaluate(a)
    assert d.allowed  # explicit override beats tag denial


def test_persona_policy_allowed_tag_required_when_set():
    p = PersonaPolicy(allowed_tags=("calculate.hhi",))
    a = _Agent(agent_id="x", capability_tags=("compute.numerical",))
    d = p.evaluate(a)
    assert not d.allowed


# ---------------------------------------------------------------------------
# PersonaRegistry selection
# ---------------------------------------------------------------------------


def test_persona_registry_selects_by_task_type():
    reg = PersonaRegistry()
    reg.register(Persona(persona_id="ma", task_types=("corporate-ma",)))
    reg.register(Persona(persona_id="anti", task_types=("antitrust-competition",)))
    sel = reg.select(task_type="antitrust-competition")
    assert sel is not None
    assert sel.persona_id == "anti"


def test_persona_registry_falls_back_to_generic():
    reg = PersonaRegistry()
    reg.register(Persona(persona_id="generic"))  # no constraints
    reg.register(Persona(persona_id="ma", task_types=("corporate-ma",)))
    sel = reg.select(task_type="other-area")
    assert sel is not None
    assert sel.persona_id == "generic"


# ---------------------------------------------------------------------------
# OperatorBudget
# ---------------------------------------------------------------------------


def test_operator_budget_simple_mode_constraints():
    """Simple mode keeps LLM-call/token/cost budget tight; per-phase
    agent count is intentionally generous because built-in operators
    are deterministic + cheap."""
    s = OperatorBudget.for_mode("simple")
    d = OperatorBudget.for_mode("deep")
    # The cost-bearing dimensions stay tighter
    assert s.max_llm_calls_total <= d.max_llm_calls_total
    assert s.max_tokens_total <= d.max_tokens_total
    assert s.max_cost_estimate_usd <= d.max_cost_estimate_usd
    # Agent count cap is generous in both modes
    assert s.max_agents_per_phase >= 5
    assert s.max_agents_total >= s.max_agents_per_phase


def test_operator_budget_unknown_keys_ignored():
    class _C:
        output_contract = {"operator_budget": {
            "max_agents_total": 9,
            "made_up_field": "ignored",
        }}
    b = OperatorBudget.from_execution_contract(_C())
    assert b.max_agents_total == 9


# ---------------------------------------------------------------------------
# Dispatcher (smoke without real LLM)
# ---------------------------------------------------------------------------


def test_dispatcher_runs_selected_agents_and_writes_artifacts(tmp_path):
    from irys.matter import MatterModel
    m = MatterModel.open_in_memory()

    reg = SubAgentRegistry()
    reg.register(_Agent(agent_id="alpha", score=0.9))
    reg.register(_Agent(agent_id="beta", score=0.8))

    disp = SubAgentDispatcher(registry=reg, matter_model=m)
    inv = _make_invocation(matter_id=m.matter_id, budget=OperatorBudget(max_agents_per_phase=5))
    result = asyncio.run(disp.run_phase(inv, phase="pre_synthesis"))
    assert len(result.invocations) == 2
    assert all(r.status == "success" for _, r in result.invocations)
    # Invocation rows persisted
    n_inv = m.db.execute(
        "SELECT COUNT(*) FROM sub_agent_invocation"
    ).fetchone()[0]
    assert n_inv == 2
    n_art = m.db.execute("SELECT COUNT(*) FROM agent_artifact").fetchone()[0]
    assert n_art == 2


def test_dispatcher_breach_budget_total_agents(tmp_path):
    from irys.matter import MatterModel
    m = MatterModel.open_in_memory()
    reg = SubAgentRegistry()
    for i in range(5):
        reg.register(_Agent(agent_id=f"a{i}", score=1.0 - i * 0.05))
    disp = SubAgentDispatcher(registry=reg, matter_model=m)
    # total budget = 2 agents only
    inv = _make_invocation(matter_id=m.matter_id, budget=OperatorBudget(
        max_agents_total=2,
        max_agents_per_phase=10,
    ))
    result = asyncio.run(disp.run_phase(inv, phase="pre_synthesis"))
    successes = [r for _, r in result.invocations if r.status == "success"]
    assert len(successes) == 2
    assert "max_agents_total" in result.breached


def test_dispatcher_blocking_validator_raises_on_failure():
    from irys.matter import MatterModel
    m = MatterModel.open_in_memory()
    reg = SubAgentRegistry()
    reg.register(_Agent(agent_id="bad", score=1.0, invoke_status="error"))
    disp = SubAgentDispatcher(registry=reg, matter_model=m)
    inv = _make_invocation(requirement=AgentRequirement.BLOCKING_VALIDATOR)
    with pytest.raises(RuntimeError):
        asyncio.run(disp.run_phase(inv, phase="pre_synthesis"))


def test_dispatcher_records_capability_tags():
    from irys.matter import MatterModel
    m = MatterModel.open_in_memory()
    reg = SubAgentRegistry()
    reg.register(_Agent(agent_id="x", score=1.0, capability_tags=("foo", "bar")))
    disp = SubAgentDispatcher(registry=reg, matter_model=m)
    asyncio.run(disp.run_phase(_make_invocation(matter_id=m.matter_id), phase="pre_synthesis"))
    row = m.db.execute(
        "SELECT capability_tags_json FROM sub_agent_invocation"
    ).fetchone()
    import json
    assert json.loads(row["capability_tags_json"]) == ["foo", "bar"]
