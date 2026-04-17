"""Two fixture runners — store mode and engine-stub mode.

Store mode: seed a fresh in-memory MatterModel from the FixtureSpec, run any
declared maintenance steps, and expose the seeded state via HarnessResult.
Invariants run against that state. No LLM calls, no engine loop.

Engine-stub mode: the same seeding, plus an RLMEngine wired to a
ScriptedGeminiClient. Lets tests exercise real engine reads like
_assemble_context_packet, _build_issue_focus_block, or _synthesize without a
live LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from irys.matter import (
    AssertionKind,
    MatterModel,
    ModelLayer,
    SourceRole,
    SpeechAct,
)
from irys.matter.enums import (
    GapType,
    IssueType,
    OriginKind,
    ReviewScope,
    ReviewedByKind,
    VerificationTargetKind,
)
from irys.matter.models import AssertionCandidate

from .schema import FixtureSpec


@dataclass
class HarnessResult:
    fixture: FixtureSpec
    model: MatterModel
    alias_to_id: dict[str, str] = field(default_factory=dict)
    engine: Optional[Any] = None  # RLMEngine in engine_stub mode
    scripted_client: Optional[Any] = None


_SPEECH_ACT = {v.value: v for v in SpeechAct}
_SOURCE_ROLE = {v.value: v for v in SourceRole}
_MODEL_LAYER = {v.value: v for v in ModelLayer}
_ASSERTION_KIND = {v.value: v for v in AssertionKind}
_ORIGIN_KIND = {v.value: v for v in OriginKind}
_ISSUE_TYPE = {v.value: v for v in IssueType}


def _seed_model(fixture: FixtureSpec) -> HarnessResult:
    """Build and seed an in-memory MatterModel from the fixture spec.

    Raw UUIDs never appear in fixtures. Alias-to-id mapping is populated as
    rows are inserted so invariants can look up real ids by fixture alias.
    """
    model = MatterModel.open_in_memory()
    result = HarnessResult(fixture=fixture, model=model)

    # Documents — seed the inventory row first (document_card.doc_id has an
    # FK onto it) and then the card so fixtures can flip privilege and
    # source_role without driving the full repo ingestion pipeline.
    for doc in fixture.documents:
        privilege_flag = (doc.privilege_status or "").lower() in {
            "privileged",
            "candidate_privileged",
            "partially_privileged",
        }
        # Deterministic fake sha — content-hash is irrelevant for fixture data.
        fake_sha = f"eval_{doc.alias}".ljust(64, "0")[:64]
        inv_id, _ = model.inventory.upsert(
            relative_path=doc.document_id,
            sha256=fake_sha,
            size_bytes=len(doc.text or "") or 1,
            file_type=doc.doc_type,
        )
        model.document_cards.upsert(
            doc_id=inv_id,
            title=doc.filename,
            doc_type=doc.doc_type,
            source_side=doc.source_side,
            source_role=doc.source_role,
            privilege_flag=privilege_flag,
        )
        result.alias_to_id[f"doc:{doc.alias}"] = doc.document_id
        result.alias_to_id[f"doc_inv:{doc.alias}"] = inv_id

    # Assertions via the public upsert path to match production writes.
    for a in fixture.assertions:
        doc_row = next(
            (d for d in fixture.documents if d.alias == a.document_alias),
            None,
        )
        if doc_row is None:
            raise ValueError(
                f"assertion {a.alias!r} references unknown document "
                f"alias {a.document_alias!r}"
            )
        cand = AssertionCandidate(
            proposition_text=a.proposition_text,
            model_layer=_MODEL_LAYER.get(a.model_layer, ModelLayer.RECORD),
            assertion_kind=_ASSERTION_KIND.get(
                a.assertion_kind, AssertionKind.FACTUAL
            ),
            document_id=doc_row.document_id,
            speech_act=_SPEECH_ACT.get(a.speech_act, SpeechAct.EXTRACTED),
            source_role=_SOURCE_ROLE.get(a.source_role, SourceRole.UNKNOWN),
            origin_kind=_ORIGIN_KIND.get(a.origin_kind, OriginKind.EXTRACTED),
        )
        aid, _ = model.assertions.upsert_occurrence(cand)
        result.alias_to_id[f"assertion:{a.alias}"] = aid
        # MVP.2: fixtures can mark an assertion as pre-verified by an
        # attorney by setting "verified": true. upsert_occurrence already
        # created a candidate row; promote it here so fixtures that drive
        # verification-gate behavior have a real verified baseline.
        if a.verified:
            model.verification.verify(
                VerificationTargetKind.ASSERTION,
                aid,
                reviewed_by_kind=ReviewedByKind.ATTORNEY,
                review_scope=ReviewScope.EXTRACTION_CORRECT,
                cause="fixture_seed",
            )

    # Issues + predicates.
    for issue in fixture.issues:
        iid, _ = model.issues.upsert_issue(
            issue.title,
            _ISSUE_TYPE.get(issue.issue_type, IssueType.CLAIM),
            materiality=issue.materiality,
            salience=issue.salience,
            burden_side=issue.burden_side,
        )
        result.alias_to_id[f"issue:{issue.alias}"] = iid
        for pred in issue.predicates:
            model.issues.add_predicate(iid, pred)

    # Issue links wire assertions to issues.
    for link in fixture.issue_links:
        aid = result.alias_to_id.get(f"assertion:{link.assertion_alias}")
        iid = result.alias_to_id.get(f"issue:{link.issue_alias}")
        if aid is None or iid is None:
            raise ValueError(
                f"issue_link references unknown alias(es): "
                f"assertion={link.assertion_alias} issue={link.issue_alias}"
            )
        if link.legacy_only:
            # MVP.3 backfill fixture path: insert only the legacy row so
            # the evidence_edge store starts empty for the backfill
            # invariant to exercise.
            import uuid as _uuid
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc).isoformat()
            model.db.execute(
                """INSERT OR IGNORE INTO assertion_issue_link
                   (id, assertion_id, issue_id, relation_type, created_at)
                   VALUES (?,?,?,?,?)""",
                (_uuid.uuid4().hex, aid, iid, link.relation_type, now),
            )
        else:
            model.issues.link_assertion(aid, iid, relation_type=link.relation_type)

    # Gap seeding — direct gap store is preferred over engine-triggered so
    # fixtures can assert the invariant independently of maintenance behavior.
    for gap in fixture.gaps:
        affected_id = None
        if gap.affected_type and gap.affected_alias:
            affected_id = result.alias_to_id.get(
                f"{gap.affected_type}:{gap.affected_alias}"
            )
        try:
            gap_type_enum = GapType(gap.gap_type)
        except ValueError as exc:
            raise ValueError(
                f"fixture gap {gap.alias!r} has unknown gap_type "
                f"{gap.gap_type!r}"
            ) from exc
        gap_id = model.gaps.record(
            gap_type=gap_type_enum,
            description=gap.description,
            materiality=gap.materiality_score,
            affected_type=gap.affected_type,
            affected_id=affected_id,
        )
        result.alias_to_id[f"gap:{gap.alias}"] = gap_id

    return result


def _run_maintenance(result: HarnessResult) -> None:
    """Execute declared maintenance steps in order.

    Supported step names are the minimal set MVP.1 needs. Unknown steps raise
    so fixtures can't silently skip a maintenance phase they depend on.
    """
    model = result.model
    for step in result.fixture.maintenance:
        name = step.step
        if name == "proof_state.compute_all":
            for issue_id in [
                v for k, v in result.alias_to_id.items() if k.startswith("issue:")
            ]:
                model.proof_state.compute_and_store(issue_id)
        elif name == "engine._detect_proof_gaps":
            # Import here to avoid engine import at module load time for store
            # mode fixtures that never touch the engine.
            from irys.rlm.engine import RLMEngine
            engine = RLMEngine.__new__(RLMEngine)
            engine._matter_model = model
            engine._detect_proof_gaps()
        else:
            raise ValueError(f"Unknown maintenance step: {name!r}")


def run_store_mode(fixture: FixtureSpec) -> HarnessResult:
    result = _seed_model(fixture)
    _run_maintenance(result)
    return result


def run_engine_stub_mode(fixture: FixtureSpec) -> HarnessResult:
    """Same seeding as store mode, plus an RLMEngine wired to a canned client.

    The engine is not driven end-to-end here — tests call specific engine
    methods like _assemble_context_packet directly. The engine is constructed
    just enough for those methods to work; full config defaults are fine.
    """
    from irys.rlm.engine import RLMConfig, RLMEngine

    from .stubs import ScriptedGeminiClient

    result = _seed_model(fixture)
    _run_maintenance(result)

    client = ScriptedGeminiClient(responses=dict(fixture.scripted_responses))
    engine = RLMEngine(
        gemini_client=client,
        config=RLMConfig(),
        matter_model=result.model,
    )
    result.engine = engine
    result.scripted_client = client
    return result
