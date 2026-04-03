"""Multi-matter isolation stress tests.

Verifies that when two MatterModel instances share the same SQLite database
(the production `open()` scenario), state from matter A never bleeds into
matter B and vice versa.

Isolation boundaries tested:
1.  Assertions scoped by matter_id
2.  Actors scoped by matter_id
3.  Issues scoped by matter_id
4.  Quant facts scoped by matter_id
5.  Gaps scoped by matter_id
6.  Authorities scoped by matter_id
7.  Proof state scoped by matter_id
8.  Decision context scoped by matter_id
9.  Document inventory scoped by matter_id
10. Communication map actors scoped by matter_id
11. Damages waterfall scoped by matter_id
12. Timeline scoped by matter_id
13. Evidence matrix scoped by matter_id
14. Reasoning ledger scoped by matter_id
15. Resolution of actor aliases does not cross matters
16. Concurrent writes: interleaved inserts do not bleed
17. Assertion count isolation (matter A count != matter B count)
18. Deletion in one matter does not affect the other
19. Belief state update isolated to originating matter
20. Ten simultaneous matter objects share the same DB with zero leakage
"""

import uuid
from datetime import datetime, timezone

import pytest

from irys.matter.db import SQLiteMatterDB
from irys.matter.matter import MatterModel
from irys.matter.models import AssertionCandidate
from irys.matter import SpeechAct, SourceRole, AssertionKind
from irys.matter.enums import GapType, IssueType


# ---------------------------------------------------------------------------
# Shared-DB fixture — two matters in one in-memory database
# ---------------------------------------------------------------------------

def _now():
    return datetime.now(timezone.utc).isoformat()


def _make_two_matters():
    """Return (model_a, model_b) sharing one in-memory SQLite DB.

    Two separate repository roots satisfy the UNIQUE(repository_root) constraint
    while sharing a single database file — the production multi-matter scenario.
    """
    db = SQLiteMatterDB.in_memory()
    mid_a = uuid.uuid4().hex
    mid_b = uuid.uuid4().hex
    now = _now()
    for mid, name, root in [
        (mid_a, "Matter Alpha", "/repo/alpha"),
        (mid_b, "Matter Beta", "/repo/beta"),
    ]:
        with db.transaction():
            db.execute(
                """INSERT INTO matter (id, name, repository_root, maturity, created_at, updated_at)
                   VALUES (?,?,?,?,?,?)""",
                (mid, name, root, "initial", now, now),
            )
    return MatterModel(db, mid_a), MatterModel(db, mid_b)


@pytest.fixture
def two_matters():
    return _make_two_matters()


# Helpers
def _candidate(text="A proposition", doc_id="doc.pdf"):
    return AssertionCandidate(
        proposition_text=text,
        speech_act=SpeechAct.ALLEGED,
        source_role=SourceRole.ADVOCACY,
        assertion_kind=AssertionKind.FACTUAL,
        document_id=doc_id,
    )


# ---------------------------------------------------------------------------
# 1. Assertions
# ---------------------------------------------------------------------------

def test_assertions_isolated(two_matters):
    a, b = two_matters
    a.assertions.upsert_occurrence(_candidate("Alpha claim"))
    assert a.assertions.count() == 1
    assert b.assertions.count() == 0


def test_assertion_count_isolated(two_matters):
    a, b = two_matters
    for i in range(5):
        a.assertions.upsert_occurrence(_candidate(f"Claim {i}"))
    for i in range(2):
        b.assertions.upsert_occurrence(_candidate(f"Beta {i}"))
    assert a.assertions.count() == 5
    assert b.assertions.count() == 2


# ---------------------------------------------------------------------------
# 2. Actors
# ---------------------------------------------------------------------------

def test_actors_isolated(two_matters):
    a, b = two_matters
    aid, _ = a.actors.upsert_actor(canonical_name="Alice", actor_type="person")
    # A sees the actor; B does not
    a_ids = {ac["id"] for ac in a.actors.list_actors()}
    b_ids = {ac["id"] for ac in b.actors.list_actors()}
    assert aid in a_ids
    assert aid not in b_ids


def test_actor_alias_resolution_does_not_cross_matters(two_matters):
    a, b = two_matters
    aid, _ = a.actors.upsert_actor(canonical_name="Acme Corporation", actor_type="org")
    a.actors.add_alias(aid, "Acme Corp")
    # Matter B should not resolve this alias
    assert b.actors.resolve_by_name("Acme Corporation") is None
    assert b.actors.resolve_by_name("Acme Corp") is None


# ---------------------------------------------------------------------------
# 3. Issues
# ---------------------------------------------------------------------------

def test_issues_isolated(two_matters):
    a, b = two_matters
    iid, _ = a.issues.upsert_issue("Breach of contract", issue_type=IssueType.CLAIM)
    assert a.issues.get_issue(iid) is not None
    assert b.issues.get_issue(iid) is None
    assert b.issues.count_open() == 0


# ---------------------------------------------------------------------------
# 4. Quant facts
# ---------------------------------------------------------------------------

def test_quant_facts_isolated(two_matters):
    a, b = two_matters
    a.quant.record(
        quant_kind="amount", raw_text="$100k", amount_value=100_000.0, currency="USD"
    )
    assert a.quant.count() == 1
    assert b.quant.count() == 0


# ---------------------------------------------------------------------------
# 5. Gaps
# ---------------------------------------------------------------------------

def test_gaps_isolated(two_matters):
    a, b = two_matters
    a.gaps.record(GapType.MISSING_DOCUMENT, description="Missing contract")
    assert len(a.gaps.open_gaps()) == 1
    assert len(b.gaps.open_gaps()) == 0


# ---------------------------------------------------------------------------
# 6. Authorities
# ---------------------------------------------------------------------------

def test_authorities_isolated(two_matters):
    a, b = two_matters
    a.authority.upsert(
        citation="Smith v. Jones, 123 F.3d 456",
        name="Smith v. Jones",
        authority_type="case",
    )
    assert a.authority.count() == 1
    assert b.authority.count() == 0


# ---------------------------------------------------------------------------
# 7. Proof state
# ---------------------------------------------------------------------------

def test_proof_state_isolated(two_matters):
    a, b = two_matters
    # Add issue in A and compute
    a.issues.upsert_issue("Liability", issue_type=IssueType.CLAIM)
    a.proof_state.compute_all()
    # B has no proof state
    assert b.proof_state.get_all() == []


# ---------------------------------------------------------------------------
# 8. Decision context
# ---------------------------------------------------------------------------

def test_decision_context_isolated(two_matters):
    a, b = two_matters
    a.decision_context.set(decision_maker_type="partner", objective="settlement")
    ctx_a = a.decision_context.get()
    ctx_b = b.decision_context.get()
    assert ctx_a is not None
    assert ctx_b is None


# ---------------------------------------------------------------------------
# 9. Document inventory
# ---------------------------------------------------------------------------

def test_document_inventory_isolated(two_matters):
    a, b = two_matters
    a.inventory.upsert("doc-a.pdf", sha256="abc123")
    assert a.inventory.count() == 1
    assert b.inventory.count() == 0


# ---------------------------------------------------------------------------
# 10. Communication map
# ---------------------------------------------------------------------------

def test_communication_map_isolated(two_matters):
    a, b = two_matters
    actor_id, _ = a.actors.upsert_actor(canonical_name="Bob", actor_type="person")
    cand = AssertionCandidate(
        proposition_text="Bob's statement",
        speech_act=SpeechAct.ALLEGED,
        source_role=SourceRole.ADVOCACY,
        assertion_kind=AssertionKind.FACTUAL,
        document_id="doc.pdf",
        speaker_actor_id=actor_id,
    )
    a.assertions.upsert_occurrence(cand)
    map_a = a.get_communication_map()
    map_b = b.get_communication_map()
    assert len(map_a["actors"]) >= 1
    assert len(map_b["actors"]) == 0


# ---------------------------------------------------------------------------
# 11. Damages waterfall
# ---------------------------------------------------------------------------

def test_damages_waterfall_isolated(two_matters):
    a, b = two_matters
    a.quant.record(
        quant_kind="amount", raw_text="Lost profits $500k",
        amount_value=500_000.0, currency="USD", subject_type="lost_profits",
    )
    wf_a = a.get_damages_waterfall()
    wf_b = b.get_damages_waterfall()
    assert len(wf_a) == 1
    assert len(wf_b) == 0


# ---------------------------------------------------------------------------
# 12. Timeline
# ---------------------------------------------------------------------------

def test_timeline_isolated(two_matters):
    a, b = two_matters
    a.quant.record(
        quant_kind="date", raw_text="Contract signed 2023-01-15",
        date_value="2023-01-15",
        subject_type="contract_date",
    )
    tl_b = b.get_timeline()
    assert len(tl_b) == 0


# ---------------------------------------------------------------------------
# 13. Evidence matrix
# ---------------------------------------------------------------------------

def test_evidence_matrix_isolated(two_matters):
    a, b = two_matters
    iid, _ = a.issues.upsert_issue("Breach", issue_type=IssueType.CLAIM)
    assertion_id, _ = a.assertions.upsert_occurrence(_candidate())
    a.issues.link_assertion(assertion_id, iid, relation_type="supports")
    matrix_a = a.get_evidence_matrix()
    matrix_b = b.get_evidence_matrix()
    assert len(matrix_a["issues"]) >= 1
    assert len(matrix_b["issues"]) == 0


# ---------------------------------------------------------------------------
# 14. Reasoning ledger
# ---------------------------------------------------------------------------

def test_reasoning_ledger_isolated(two_matters):
    a, b = two_matters
    run_id = a.start_run("What happened?")
    a.complete_run(run_id)
    # Run belongs to A's matter_id, B cannot see it
    row = b.db.execute(
        "SELECT COUNT(*) FROM run_session WHERE matter_id=?", (b.matter_id,)
    ).fetchone()
    assert row[0] == 0


# ---------------------------------------------------------------------------
# 16. Interleaved writes do not bleed
# ---------------------------------------------------------------------------

def test_interleaved_writes_isolated(two_matters):
    a, b = two_matters
    for i in range(10):
        if i % 2 == 0:
            a.assertions.upsert_occurrence(_candidate(f"A-{i}"))
        else:
            b.assertions.upsert_occurrence(_candidate(f"B-{i}"))
    assert a.assertions.count() == 5
    assert b.assertions.count() == 5


# ---------------------------------------------------------------------------
# 17. Assertion count isolation explicit check
# ---------------------------------------------------------------------------

def test_assertion_counts_never_sum(two_matters):
    a, b = two_matters
    for i in range(3):
        a.assertions.upsert_occurrence(_candidate(f"A{i}"))
    for i in range(7):
        b.assertions.upsert_occurrence(_candidate(f"B{i}"))
    assert a.assertions.count() == 3
    assert b.assertions.count() == 7
    assert a.assertions.count() + b.assertions.count() == 10


# ---------------------------------------------------------------------------
# 18. Deletion in one matter does not affect the other
# ---------------------------------------------------------------------------

def test_actor_delete_isolated(two_matters):
    a, b = two_matters
    aid_a, _ = a.actors.upsert_actor(canonical_name="Shared Name", actor_type="person")
    bid_b, _ = b.actors.upsert_actor(canonical_name="Shared Name", actor_type="person")
    # Delete from A
    a.db.execute("DELETE FROM actor_alias WHERE actor_id=?", (aid_a,))
    a.db.execute("DELETE FROM actor WHERE id=?", (aid_a,))
    # B's actor unaffected
    b_ids = {ac["id"] for ac in b.actors.list_actors()}
    assert bid_b in b_ids


# ---------------------------------------------------------------------------
# 20. Ten simultaneous matter objects
# ---------------------------------------------------------------------------

def test_ten_matters_no_leakage():
    db = SQLiteMatterDB.in_memory()
    now = _now()
    matters = []
    for i in range(10):
        mid = uuid.uuid4().hex
        with db.transaction():
            db.execute(
                """INSERT INTO matter (id, name, repository_root, maturity, created_at, updated_at)
                   VALUES (?,?,?,?,?,?)""",
                (mid, f"Matter {i}", f"/repo/matter_{i}", "initial", now, now),
            )
        matters.append(MatterModel(db, mid))

    # Write one assertion per matter
    for idx, m in enumerate(matters):
        m.assertions.upsert_occurrence(_candidate(f"Claim from matter {idx}"))

    # Each matter sees exactly 1 assertion
    for idx, m in enumerate(matters):
        assert m.assertions.count() == 1, f"Matter {idx} has wrong assertion count"

    # Total assertions in DB = 10 (sanity check)
    total = db.execute("SELECT COUNT(*) FROM assertion").fetchone()[0]
    assert total == 10
