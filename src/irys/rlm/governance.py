"""Answerability-Governed Cost Cascade — front-door governance.

MVI-1 of the cascade design (see `codex_master_plan.txt` and the
project CLAUDE.md). Decides whether a user query should:

  - enter the full recursive investigate loop (expensive, minutes)
  - be answered from existing matter state + conversation (cheap,
    one synth call)
  - be bounced back to the user as a clarification question (no
    LLM spend on the answer)

The core primitive is an `ExecutionContract` — the classifier emits
one, and every downstream gate (termination, per-lead EV, family
handlers) reads from it. Mode selection and stopping rules are the
same decision at different levels of the cascade.

Later MVIs will add more families (query, trace, steer, compare,
scenario, deliverable). MVI-1 ships only the three routes required
to stop wasting the full loop on questions the matter can already
answer: investigate | read | clarify.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from ..core.models import GeminiClient, ModelTier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Contract shapes
# ---------------------------------------------------------------------------


@dataclass
class ExecutionContract:
    """What a mode handler promises about its spend and stopping rules.

    Emitted by the classifier and carried through the cascade so every
    downstream gate reads from one source of truth. Later MVIs extend
    this with coverage_goal, freshness_floor, etc.
    """
    family: str                    # investigate | read | query | trace | clarify
    min_iter: int = 0              # 0 for read/clarify, floor for investigate
    max_iter: int = 20             # cap even on investigate
    citation_floor: int = 0        # minimum citations before a read can answer
    answer_confidence_floor: float = 0.5  # read escalates below this
    escalation_allowed: bool = True       # can a handler escalate to investigate?
    # MVI-3 per-lead viability floor — the cold-loop terminator uses
    # this instead of a hardcoded priority threshold. MVI-5 will
    # upgrade this to a real expected-value signal (coverage gain
    # per expected cost).
    lead_ev_floor: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnswerabilitySnapshot:
    """Compact state signal fed to the classifier.

    Deliberately small — the classifier is NANO and its input budget
    matters. Fields here are the ones Codex called out as actually
    relevant to routing (entity resolution confidence, coverage,
    verification status, freshness, policy/audience). Global counts
    like `assertion_count` are kept as coarse hints but the classifier
    is instructed to weigh slice-relevant signals more heavily.
    """
    matter_id: Optional[str]
    assertion_count: int
    verified_assertion_count: int
    open_issue_count: int
    open_gap_count: int
    actor_count: int
    has_any_facts: bool
    has_any_verified: bool
    trust_revision: int
    policy_audience: str = "clean"
    # Last-turn hints so deixis ("that contract", "the prior point")
    # can resolve without re-sending the full conversation.
    recent_turn_count: int = 0
    last_turn_summary: Optional[str] = None

    def to_prompt_block(self) -> str:
        """Render as compact key:value lines for the classifier prompt."""
        lines = [
            f"- has_any_facts: {self.has_any_facts}",
            f"- has_any_verified: {self.has_any_verified}",
            f"- assertion_count: {self.assertion_count}",
            f"- verified_assertion_count: {self.verified_assertion_count}",
            f"- open_issue_count: {self.open_issue_count}",
            f"- open_gap_count: {self.open_gap_count}",
            f"- actor_count: {self.actor_count}",
            f"- trust_revision: {self.trust_revision}",
            f"- policy_audience: {self.policy_audience}",
            f"- recent_turn_count: {self.recent_turn_count}",
        ]
        if self.last_turn_summary:
            lines.append(f"- last_turn_summary: {self.last_turn_summary[:160]}")
        return "\n".join(lines)


@dataclass
class CascadeDecision:
    """One routing decision, persisted on run_session for auditability.

    `escalation_reason` is set when a read-family handler bounces to
    investigate because coverage was insufficient — we need that to
    tune the classifier later.
    """
    family: str
    confidence: float
    rationale: str
    contract: ExecutionContract
    classifier_version: str
    snapshot: AnswerabilitySnapshot
    escalation_reason: Optional[str] = None

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "classifier_version": self.classifier_version,
            "contract": self.contract.to_dict(),
            "escalation_reason": self.escalation_reason,
        }


# ---------------------------------------------------------------------------
# Classifier prompt (NANO tier)
# ---------------------------------------------------------------------------


# Bump whenever the prompt or schema changes — included in the
# classifier_version so old cached decisions miss cleanly.
CLASSIFIER_SCHEMA_VERSION = "mvi4.0"


VALID_FAMILIES = {
    "investigate", "read", "query", "trace", "steer", "clarify",
}


INTENT_CLASSIFIER_PROMPT = """You are a routing classifier for a legal intelligence platform. For each user query you pick ONE route that matches how much work the system should actually do.

Six routes are available:

1. `investigate` — the user is asking a novel question about this legal matter that probably needs new evidence extraction, document search, or synthesis of findings the matter model does not yet contain. Examples: "What's our damages exposure?", "Did the opposing party breach the agreement?", "Find me evidence of intent to deceive." Route here if the matter is fresh (no facts yet), OR if the question targets material that probably hasn't been extracted, OR if the user explicitly asks for an investigation.

2. `read` — the user is asking for a summary, recap, restatement, reformat, or substantive answer synthesized from facts the matter model already contains. Examples: "Summarize our session for my team", "Give me that analysis as bullet points", "Draft a client email explaining our conclusions", "What have we found about the MSA?". Route here if the matter has content AND the query asks about existing findings as a narrative answer, not a plain enumeration.

3. `query` — the user is asking for a plain enumeration or lookup from matter model tables: "list all quants", "show me every actor", "what gaps are open", "give me the full timeline", "list every contradiction". These are DB reads — no synthesis or reasoning needed. Route here when the request is structurally "give me the list of X" or "show me the data in store Y".

4. `trace` — the user is asking where a specific prior conclusion came from: "why did you say X", "show me the source for claim Y", "what's the provenance of the damages figure", "how did you derive that timeline". Route here when the request targets the reasoning ledger / provenance of an existing finding.

5. `steer` — the user is CORRECTING a prior fact, OVERRIDING a belief, annotating, editing assumptions, or otherwise mutating matter state. Examples: "Actually the date was April, not March", "That assertion is wrong", "Mark the MSA as the operative contract", "Change the damages figure to 50000", "Ignore the email from March 3rd — it's drafts". The user is not asking a question; they're correcting or directing the matter model. Route here even when the phrasing is indirect ("no, the payment was 30 days after").

6. `clarify` — the user's referent is ambiguous or the query is so vague that proceeding would produce a wrong cheap answer. Examples: "Tell me about Smith" when there are two Smiths. Route here SPARINGLY — only when a specific ambiguity makes routing unsafe.

Guidance:
- Default to `investigate` on a fresh matter (has_any_facts=False).
- On a warm matter: `query` for "list X" / "show X" / "which X", `read` for "summarize" / "draft" / "explain" / "what does X mean", `trace` for "why" / "how did you" / "show the source", `steer` for "correct" / "actually X" / "no, it was Y" / "change" / "ignore".
- Never route to `read`, `query`, `trace`, or `steer` if has_any_facts=False — there's nothing to read or correct.
- Your job is cost governance, not content judgment. Keep the decision fast.

Matter state snapshot:
{snapshot_block}

Recent conversation turns (for deixis only — do not rely on them as a knowledge source):
{conversation_block}

User query: {query}

Respond ONLY with a single JSON object:
{{
  "family": "investigate" | "read" | "query" | "trace" | "steer" | "clarify",
  "confidence": 0.0-1.0,
  "rationale": "one short sentence — why this route"
}}
"""


# ---------------------------------------------------------------------------
# Governor
# ---------------------------------------------------------------------------


class CascadeGovernor:
    """Produces a `CascadeDecision` for a user query.

    Thin wrapper around a NANO LLM call plus a compact snapshot build.
    State-aware — the snapshot is the mechanism that keeps the router
    out of the prompt-as-memory anti-pattern Codex flagged.
    """

    def __init__(
        self,
        client: GeminiClient,
        matter_model: Any = None,  # MatterModel or None for fresh matters
    ) -> None:
        self.client = client
        self.matter_model = matter_model

    async def decide(
        self,
        query: str,
        conversation_history: Optional[list[dict[str, str]]] = None,
    ) -> CascadeDecision:
        """Classify and derive a contract. Falls back to `investigate`
        on any classifier error — we'd rather over-spend than answer
        a question we can't route. This matches Codex's directive
        that silent cheap-wrong answers are worse than latency."""
        snapshot = self._build_snapshot(conversation_history)

        # Cold-start shortcut: if there are literally no facts in the
        # matter, `read` is impossible by definition. Skip the NANO
        # call and hard-route to investigate. Saves a round trip on
        # every fresh-matter query.
        if not snapshot.has_any_facts:
            return CascadeDecision(
                family="investigate",
                confidence=1.0,
                rationale="cold-start: matter has no facts yet",
                contract=self._contract_for("investigate"),
                classifier_version=CLASSIFIER_SCHEMA_VERSION,
                snapshot=snapshot,
            )

        family, confidence, rationale = await self._classify(
            query, snapshot, conversation_history,
        )
        return CascadeDecision(
            family=family,
            confidence=confidence,
            rationale=rationale,
            contract=self._contract_for(family),
            classifier_version=CLASSIFIER_SCHEMA_VERSION,
            snapshot=snapshot,
        )

    def _build_snapshot(
        self,
        conversation_history: Optional[list[dict[str, str]]],
    ) -> AnswerabilitySnapshot:
        """Read a compact answerability snapshot from the matter model."""
        mm = self.matter_model
        if mm is None:
            return AnswerabilitySnapshot(
                matter_id=None,
                assertion_count=0,
                verified_assertion_count=0,
                open_issue_count=0,
                open_gap_count=0,
                actor_count=0,
                has_any_facts=False,
                has_any_verified=False,
                trust_revision=0,
                recent_turn_count=len(conversation_history or []),
                last_turn_summary=_tail_turn_summary(conversation_history),
            )
        try:
            assertion_count = mm.assertions.count()
            verified_count = self._count_verified_assertions(mm)
            open_issues = mm.issues.count_open()
            open_gaps = mm.gaps.count_open()
            actor_count = mm.actors.count()
            trust_rev = mm.cache.current_trust_revision()
        except Exception as exc:  # noqa: BLE001
            logger.warning("snapshot build failed, defaulting to empty: %s", exc)
            return AnswerabilitySnapshot(
                matter_id=getattr(mm, "matter_id", None),
                assertion_count=0,
                verified_assertion_count=0,
                open_issue_count=0,
                open_gap_count=0,
                actor_count=0,
                has_any_facts=False,
                has_any_verified=False,
                trust_revision=0,
                recent_turn_count=len(conversation_history or []),
                last_turn_summary=_tail_turn_summary(conversation_history),
            )
        return AnswerabilitySnapshot(
            matter_id=mm.matter_id,
            assertion_count=assertion_count,
            verified_assertion_count=verified_count,
            open_issue_count=open_issues,
            open_gap_count=open_gaps,
            actor_count=actor_count,
            has_any_facts=assertion_count > 0,
            has_any_verified=verified_count > 0,
            trust_revision=trust_rev,
            recent_turn_count=len(conversation_history or []),
            last_turn_summary=_tail_turn_summary(conversation_history),
        )

    @staticmethod
    def _count_verified_assertions(mm: Any) -> int:
        """Counts assertions with verification_state.status = verified.
        Tolerates schemas where that table is empty or missing."""
        try:
            row = mm.db.execute(
                """SELECT COUNT(*) AS n FROM verification_state
                   WHERE matter_id=? AND status='verified'
                     AND target_kind='assertion'""",
                (mm.matter_id,),
            ).fetchone()
            return int(row["n"] or 0) if row else 0
        except Exception:
            return 0

    async def _classify(
        self,
        query: str,
        snapshot: AnswerabilitySnapshot,
        conversation_history: Optional[list[dict[str, str]]],
    ) -> tuple[str, float, str]:
        """Single NANO call. Returns (family, confidence, rationale).
        Defaults to ('investigate', 0.0, 'classifier error: <msg>') on
        any failure — fail safe, not fail silent."""
        prompt = INTENT_CLASSIFIER_PROMPT.format(
            snapshot_block=snapshot.to_prompt_block(),
            conversation_block=_render_conversation(conversation_history),
            query=query,
        )
        try:
            response = await self.client.complete(
                prompt,
                tier=ModelTier.NANO,
                json_mode=True,
                usage_label="intent_classifier",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("classifier NANO call failed: %s", exc)
            return ("investigate", 0.0, f"classifier error: {exc}")
        try:
            parsed = json.loads(response or "{}")
        except (TypeError, ValueError) as exc:
            logger.warning("classifier response not JSON: %s", exc)
            return ("investigate", 0.0, f"parse error: {exc}")
        family = str(parsed.get("family") or "").strip().lower()
        if family not in VALID_FAMILIES:
            return (
                "investigate", 0.0,
                f"unknown family '{family}', defaulting to investigate",
            )
        if (
            family in {"read", "query", "trace"}
            and not snapshot.has_any_facts
        ):
            # Belt-and-suspenders: if the classifier routes to a warm-
            # matter family but the matter is empty, override. Can't
            # read / query / trace what isn't there.
            return (
                "investigate", 0.0,
                f"{family} requested but matter has no facts",
            )
        try:
            confidence = float(parsed.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        rationale = str(parsed.get("rationale") or "").strip()[:240]
        return (family, confidence, rationale)

    @staticmethod
    def _contract_for(family: str) -> ExecutionContract:
        """Family → contract mapping. Per Codex: mode selection and
        stopping rules are the same decision at different levels."""
        if family == "read":
            return ExecutionContract(
                family="read",
                min_iter=0,
                max_iter=1,
                citation_floor=1,
                answer_confidence_floor=0.5,
                escalation_allowed=True,
            )
        if family == "query":
            # MVI-2: zero-iteration, zero-LLM plain enumeration
            return ExecutionContract(
                family="query",
                min_iter=0,
                max_iter=0,
                citation_floor=0,
                answer_confidence_floor=0.0,
                escalation_allowed=True,
            )
        if family == "trace":
            # MVI-2: DB-read against provenance + reasoning ledger.
            # NANO fallback only if the user phrasing is ambiguous.
            return ExecutionContract(
                family="trace",
                min_iter=0,
                max_iter=1,
                citation_floor=0,
                answer_confidence_floor=0.0,
                escalation_allowed=True,
            )
        if family == "steer":
            # MVI-4: NANO parse + best-match target, no auto-apply.
            # Returns a preview the user confirms via existing UI.
            return ExecutionContract(
                family="steer",
                min_iter=0,
                max_iter=1,
                citation_floor=0,
                answer_confidence_floor=0.0,
                escalation_allowed=True,
            )
        if family == "clarify":
            return ExecutionContract(
                family="clarify",
                min_iter=0,
                max_iter=0,
                citation_floor=0,
                answer_confidence_floor=0.0,
                escalation_allowed=False,
            )
        # investigate — existing loop contract
        return ExecutionContract(
            family="investigate",
            min_iter=1,
            max_iter=20,
            citation_floor=1,
            answer_confidence_floor=0.6,
            escalation_allowed=False,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _render_conversation(
    conversation_history: Optional[list[dict[str, str]]],
    tail: int = 3,
) -> str:
    if not conversation_history:
        return "(no prior turns)"
    turns = conversation_history[-tail:]
    lines = []
    for i, t in enumerate(turns, start=1):
        q = str(t.get("query") or "").strip()[:200]
        a = str(t.get("answer") or "").strip()[:200]
        if q:
            lines.append(f"Turn {i} user: {q}")
        if a:
            lines.append(f"Turn {i} system: {a}")
    return "\n".join(lines) or "(no prior turns)"


def _tail_turn_summary(
    conversation_history: Optional[list[dict[str, str]]],
) -> Optional[str]:
    if not conversation_history:
        return None
    last = conversation_history[-1]
    q = str(last.get("query") or "").strip()
    a = str(last.get("answer") or "").strip()
    if not (q or a):
        return None
    return f"Q: {q[:100]} A: {a[:100]}"


def decision_cache_key(
    query: str,
    snapshot: AnswerabilitySnapshot,
    classifier_version: str = CLASSIFIER_SCHEMA_VERSION,
) -> str:
    """Stable key for caching route decisions. Per Codex master plan:
    key on normalized query signature + snapshot fingerprint + policy
    + classifier schema version. Do NOT include raw conversation turns
    — that crushes hit rate."""
    payload = {
        "q": query.strip().lower(),
        "snap": {
            "m": snapshot.matter_id,
            "h": snapshot.has_any_facts,
            "v": snapshot.has_any_verified,
            "tr": snapshot.trust_revision,
            "oi": snapshot.open_issue_count,
            "og": snapshot.open_gap_count,
            "p": snapshot.policy_audience,
        },
        "ver": classifier_version,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
# Read-family handler (MVI-1)
# ---------------------------------------------------------------------------


READ_FAMILY_PROMPT = """You are answering a follow-up question about a legal matter that has already been investigated. You may ONLY use the facts, issues, conversation, and other context below — you have NOT searched any documents on this turn. Do not invent findings, do not claim a new investigation, and do not speculate beyond what the matter model contains.

If the existing state contains a direct answer, give it concisely and cite source documents from the facts below.

If the existing state does NOT contain a sufficient answer, say so plainly and set answer_confidence to "low". The caller may escalate to a fresh investigation.

Matter snapshot:
{matter_summary}

Verified facts (these have been human-reviewed — weight highest):
{verified_block}

Candidate facts (extracted but unreviewed — weight lower, flag if material):
{candidate_block}

Open issues:
{issues_block}

Known gaps:
{gaps_block}

Recent conversation (for context; do not treat as authoritative knowledge):
{conversation_block}

User question: {query}

Respond in JSON ONLY, no prose outside the JSON:
{{
  "answer": "your concise answer, or a short explanation of why the matter model can't answer",
  "answer_confidence": "low" | "medium" | "high",
  "citations": ["doc1.pdf", "doc2.pdf"],
  "used_existing_state_only": true,
  "escalation_hint": "if confidence is low, what a fresh investigation would need to look for; otherwise empty"
}}
"""


@dataclass
class ReadFamilyResult:
    """Outcome of one read-family call."""
    answer: str
    confidence_label: str         # "low" | "medium" | "high"
    confidence_score: float       # 0.0-1.0 numeric mapping
    citations: list[str]
    escalation_needed: bool
    escalation_reason: Optional[str]
    raw_response: str


class ReadFamilyHandler:
    """Answers queries from existing matter state with one synth call.
    No loop, no search, no extraction. If the matter state doesn't
    contain enough to answer at the contract's floor, escalates to
    investigate via the caller."""

    # Mapping from the LLM's coarse label to a numeric score so the
    # contract's `answer_confidence_floor` can gate escalation.
    _CONFIDENCE_MAP = {"low": 0.25, "medium": 0.65, "high": 0.9}

    def __init__(
        self,
        client: GeminiClient,
        matter_model: Any,
    ) -> None:
        self.client = client
        self.matter_model = matter_model

    async def run(
        self,
        query: str,
        contract: ExecutionContract,
        conversation_history: Optional[list[dict[str, str]]] = None,
    ) -> ReadFamilyResult:
        if self.matter_model is None:
            return ReadFamilyResult(
                answer="",
                confidence_label="low",
                confidence_score=0.0,
                citations=[],
                escalation_needed=True,
                escalation_reason="no matter model available",
                raw_response="",
            )

        context = self._assemble_read_context()
        prompt = READ_FAMILY_PROMPT.format(
            matter_summary=context["matter_summary"],
            verified_block=context["verified_block"],
            candidate_block=context["candidate_block"],
            issues_block=context["issues_block"],
            gaps_block=context["gaps_block"],
            conversation_block=_render_conversation(
                conversation_history, tail=5,
            ),
            query=query,
        )

        # FLASH for the synth call — quality matters more than cost on
        # the single answer turn. If this proves too expensive we can
        # drop to LITE once the eval harness (P0.8) lands and the
        # quality gap is measured rather than guessed.
        try:
            response = await self.client.complete(
                prompt,
                tier=ModelTier.FLASH,
                json_mode=True,
                usage_label="read_synth",
                conversation_history=conversation_history,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("read_synth call failed: %s", exc)
            return ReadFamilyResult(
                answer="",
                confidence_label="low",
                confidence_score=0.0,
                citations=[],
                escalation_needed=True,
                escalation_reason=f"read call failed: {exc}",
                raw_response="",
            )

        parsed = self._parse_read_json(response)
        label = str(parsed.get("answer_confidence") or "low").lower()
        if label not in self._CONFIDENCE_MAP:
            label = "low"
        score = self._CONFIDENCE_MAP[label]

        answer = str(parsed.get("answer") or "").strip()
        citations = [
            str(c) for c in (parsed.get("citations") or [])
            if isinstance(c, (str, int, float))
        ]
        escalation_hint = str(parsed.get("escalation_hint") or "").strip()

        escalation_needed = (
            contract.escalation_allowed
            and score < contract.answer_confidence_floor
        )
        escalation_reason = (
            escalation_hint or
            f"read confidence {label} below floor {contract.answer_confidence_floor}"
        ) if escalation_needed else None

        return ReadFamilyResult(
            answer=answer,
            confidence_label=label,
            confidence_score=score,
            citations=citations,
            escalation_needed=escalation_needed,
            escalation_reason=escalation_reason,
            raw_response=response or "",
        )

    def _assemble_read_context(self) -> dict[str, str]:
        mm = self.matter_model
        stats = mm.stats() if hasattr(mm, "stats") else {}
        matter_summary = (
            f"matter_id: {mm.matter_id}\n"
            f"assertion_count: {stats.get('assertion_count', 0)}\n"
            f"open_issue_count: {stats.get('open_issue_count', 0)}\n"
            f"open_gap_count: {stats.get('open_gap_count', 0)}\n"
            f"actor_count: {stats.get('actor_count', 0)}\n"
            f"quant_fact_count: {stats.get('quant_fact_count', 0)}"
        )

        verified_block, candidate_block = self._render_assertions(mm)
        issues_block = self._render_issues(mm)
        gaps_block = self._render_gaps(mm)

        return {
            "matter_summary": matter_summary,
            "verified_block": verified_block or "(no verified facts yet)",
            "candidate_block": candidate_block or "(no candidate facts yet)",
            "issues_block": issues_block or "(no open issues)",
            "gaps_block": gaps_block or "(no known gaps)",
        }

    def _render_assertions(self, mm: Any) -> tuple[str, str]:
        """Render up to ~40 recent assertions split into verified /
        candidate lanes. Trust-aware so privileged/rejected content
        never leaks into the read prompt (content policy stays
        enforced even here)."""
        try:
            rows = mm.assertions.list_recent(limit=60)
        except Exception:
            return "", ""
        verified_lines: list[str] = []
        candidate_lines: list[str] = []
        for r in rows:
            prop = str(r.get("proposition_text") or "").strip()
            if not prop:
                continue
            belief = str(r.get("belief_state") or "").lower()
            if belief in {"withdrawn", "superseded", "rejected"}:
                continue
            role = str(r.get("primary_source_role") or "unknown").upper()
            doc = str(r.get("primary_document_id") or "").strip() or "unknown"
            line = f"- [{role}] {prop[:200]}  (doc: {doc})"
            if self._assertion_is_verified(mm, str(r.get("id"))):
                if len(verified_lines) < 25:
                    verified_lines.append(line)
            else:
                if len(candidate_lines) < 15:
                    candidate_lines.append(line)
        return "\n".join(verified_lines), "\n".join(candidate_lines)

    @staticmethod
    def _assertion_is_verified(mm: Any, assertion_id: str) -> bool:
        if not assertion_id:
            return False
        try:
            row = mm.db.execute(
                """SELECT status FROM verification_state
                   WHERE matter_id=? AND target_kind='assertion'
                     AND target_id=?""",
                (mm.matter_id, assertion_id),
            ).fetchone()
        except Exception:
            return False
        return bool(row and row["status"] == "verified")

    @staticmethod
    def _render_issues(mm: Any) -> str:
        try:
            issues = mm.issues.get_open_issues()[:20]
        except Exception:
            return ""
        lines: list[str] = []
        for i in issues:
            title = str(i.get("title") or "").strip()
            mat = i.get("materiality", 0) or 0
            status = i.get("status", "open")
            if title:
                lines.append(f"- {title} (materiality={mat}, status={status})")
        return "\n".join(lines)

    @staticmethod
    def _render_gaps(mm: Any) -> str:
        try:
            gaps = mm.gaps.open_gaps(limit=10)
        except Exception:
            return ""
        lines: list[str] = []
        for g in gaps:
            desc = str(g.get("description") or "").strip()
            gtype = g.get("gap_type", "unknown")
            if desc:
                lines.append(f"- [{gtype}] {desc[:180]}")
        return "\n".join(lines)

    @staticmethod
    def _parse_read_json(response: str) -> dict[str, Any]:
        if not response:
            return {}
        text = response.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}


# ---------------------------------------------------------------------------
# Query-family handler (MVI-2) — zero-LLM plain enumeration
# ---------------------------------------------------------------------------


@dataclass
class QueryFamilyResult:
    """Outcome of one query-family request. Zero LLM spend."""
    intent: str                    # which canned intent fired
    rows: list[dict]               # raw rows returned
    rendered_answer: str           # markdown-ish rendering for display
    escalation_needed: bool        # True if the query didn't match any intent
    escalation_reason: Optional[str] = None


QUERY_INTENTS = [
    ("list_quants", "enumerate quantitative facts — amounts, payments, damages, dollar figures"),
    ("list_actors", "enumerate actors — parties, counsel, witnesses, people, companies"),
    ("list_documents", "enumerate documents in the matter — files, contracts, pleadings, their review status"),
    ("list_gaps", "enumerate open gaps — missing docs, unresolved questions, unknowns"),
    ("list_issues", "enumerate open legal issues — claims, defenses, damages components"),
    ("list_contradictions", "enumerate contradictions — conflicting assertions, disputed facts"),
    ("list_recent_facts", "enumerate recent facts/assertions in the matter model"),
    ("list_authorities", "enumerate legal authorities cited in the matter — cases, statutes"),
]


SUB_INTENT_PROMPT = """Classify which enumeration the user wants from a matter's stored data. Pick EXACTLY one intent from the list, or return "none" if the query doesn't fit any enumeration.

Available intents:
{intent_list}

Rules:
- Match the user's intent, even if they phrase it unusually. "Break down the money" → list_quants. "Who's on the other side" → list_actors. "What's still missing" → list_gaps.
- Return "none" ONLY when the query truly doesn't map — e.g. they want a synthesis ("summarize") or a specific fact lookup. Don't stretch to fit.

User query: {query}

Respond ONLY with JSON: {{"intent": "name_from_list_or_none"}}
"""


class QueryFamilyHandler:
    """Answers plain enumeration queries (list, show, count, who, what)
    directly from matter-model tables. One cheap NANO call to resolve
    the sub-intent, then zero LLM work on the actual data fetch.
    Returns rows + a pre-rendered markdown answer.

    MVI-2 scope: a hand-curated set of canned intents with NANO-driven
    resolution for robustness. A keyword pre-filter lets unambiguous
    queries skip the NANO round-trip entirely.
    """

    # Fast-path keyword pre-filter. If a query unambiguously matches
    # exactly one intent, skip NANO. Otherwise NANO decides.
    _KEYWORD_HINTS: list[tuple[str, set[str]]] = [
        ("list_quants", {"quant", "dollar", "money"}),
        ("list_actors", {"actor", "counsel", "witness", "opposing"}),
        ("list_documents", {"document", "files", "pdfs"}),
        ("list_gaps", {"gap", "missing"}),
        ("list_issues", {"issues", "claims"}),
        ("list_contradictions", {"contradict", "conflict", "dispute"}),
        ("list_recent_facts", {"fact list", "all facts"}),
        ("list_authorities", {"citation", "case law", "statute"}),
    ]

    def __init__(
        self,
        matter_model: Any,
        client: Optional[GeminiClient] = None,
    ) -> None:
        self.matter_model = matter_model
        self.client = client

    async def run(
        self,
        query: str,
        contract: ExecutionContract,
    ) -> QueryFamilyResult:
        if self.matter_model is None:
            return QueryFamilyResult(
                intent="",
                rows=[],
                rendered_answer="",
                escalation_needed=True,
                escalation_reason="no matter model available",
            )
        intent = await self._resolve_intent(query)
        if intent is None:
            return QueryFamilyResult(
                intent="",
                rows=[],
                rendered_answer="",
                escalation_needed=contract.escalation_allowed,
                escalation_reason="no enumeration intent matched",
            )
        handler = getattr(self, f"_intent_{intent}", None)
        if handler is None:
            return QueryFamilyResult(
                intent=intent,
                rows=[],
                rendered_answer="",
                escalation_needed=True,
                escalation_reason=f"missing handler for intent '{intent}'",
            )
        try:
            rows = handler()
        except Exception as exc:  # noqa: BLE001
            logger.warning("query intent %s failed: %s", intent, exc)
            return QueryFamilyResult(
                intent=intent,
                rows=[],
                rendered_answer="",
                escalation_needed=True,
                escalation_reason=f"intent '{intent}' raised: {exc}",
            )
        rendered = self._render(intent, rows)
        return QueryFamilyResult(
            intent=intent,
            rows=rows,
            rendered_answer=rendered,
            escalation_needed=False,
        )

    async def _resolve_intent(self, query: str) -> Optional[str]:
        # Fast path: unambiguous keyword match skips NANO.
        q = query.lower()
        matches = {
            intent for intent, keywords in self._KEYWORD_HINTS
            if any(k in q for k in keywords)
        }
        if len(matches) == 1:
            return matches.pop()

        # Ambiguous / no-match / client unavailable → NANO.
        if self.client is None:
            # Degrade gracefully — return the single keyword match if
            # one exists, otherwise None.
            if len(matches) == 1:
                return matches.pop()
            return None

        intent_list = "\n".join(
            f"- {name}: {desc}" for name, desc in QUERY_INTENTS
        )
        prompt = SUB_INTENT_PROMPT.format(
            intent_list=intent_list, query=query,
        )
        try:
            response = await self.client.complete(
                prompt,
                tier=ModelTier.NANO,
                json_mode=True,
                usage_label="query_sub_intent",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("sub-intent NANO failed: %s", exc)
            return None
        try:
            parsed = json.loads(response or "{}")
        except (TypeError, ValueError):
            return None
        intent = str(parsed.get("intent") or "").strip()
        if intent == "none" or not intent:
            return None
        valid = {name for name, _ in QUERY_INTENTS}
        if intent not in valid:
            return None
        return intent

    # ---- canned intents (zero-LLM) -----------------------------------------

    def _intent_list_quants(self) -> list[dict]:
        try:
            return list(self.matter_model.quant.list_all())[:50]
        except Exception:
            return []

    def _intent_list_actors(self) -> list[dict]:
        try:
            return list(self.matter_model.actors.list_actors(limit=100))
        except Exception:
            return []

    def _intent_list_documents(self) -> list[dict]:
        try:
            return list(self.matter_model.list_reviewable_documents())[:100]
        except Exception:
            return []

    def _intent_list_gaps(self) -> list[dict]:
        try:
            return list(self.matter_model.gaps.open_gaps(limit=50))
        except Exception:
            return []

    def _intent_list_issues(self) -> list[dict]:
        try:
            return list(self.matter_model.issues.get_open_issues())[:50]
        except Exception:
            return []

    def _intent_list_contradictions(self) -> list[dict]:
        """Surface assertion-link rows of type attacks/contradicts."""
        try:
            rows = self.matter_model.db.execute(
                """SELECT al.src_assertion_id, al.dst_assertion_id,
                          al.link_type,
                          s.proposition_text AS src_text,
                          d.proposition_text AS dst_text
                   FROM assertion_link al
                   JOIN assertion s ON s.id=al.src_assertion_id
                   JOIN assertion d ON d.id=al.dst_assertion_id
                   WHERE s.matter_id=?
                     AND al.link_type IN ('attacks','contradicts')
                   LIMIT 50""",
                (self.matter_model.matter_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def _intent_list_recent_facts(self) -> list[dict]:
        try:
            return list(self.matter_model.assertions.list_recent(limit=50))
        except Exception:
            return []

    def _intent_list_authorities(self) -> list[dict]:
        try:
            rows = self.matter_model.db.execute(
                """SELECT id, citation, authority_type, COALESCE(weight, 0.0) AS weight
                   FROM authority WHERE matter_id=?
                   ORDER BY weight DESC, citation
                   LIMIT 50""",
                (self.matter_model.matter_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def _render(self, intent: str, rows: list[dict]) -> str:
        """Deterministic markdown rendering per intent. No LLM."""
        if not rows:
            return f"No results for `{intent}`."
        title = intent.replace("_", " ").title()
        lines = [f"## {title}", ""]
        if intent == "list_quants":
            for r in rows:
                lines.append(
                    f"- {r.get('raw_text') or r.get('quant_kind','quant')}  "
                    f"(amount={r.get('amount_value')} "
                    f"{r.get('currency') or ''})"
                )
        elif intent == "list_actors":
            for r in rows:
                lines.append(
                    f"- {r.get('canonical_name', 'actor')}  "
                    f"(role={r.get('role') or 'unknown'})"
                )
        elif intent == "list_documents":
            for r in rows:
                pending = r.get("pending", 0)
                verified = r.get("verified", 0)
                tag = (
                    f"{pending} pending" if pending
                    else f"{verified} reviewed ✓" if verified
                    else "—"
                )
                lines.append(f"- {r.get('path')}  ({tag})")
        elif intent == "list_gaps":
            for r in rows:
                lines.append(
                    f"- [{r.get('gap_type')}] {r.get('description','')[:200]}"
                )
        elif intent == "list_issues":
            for r in rows:
                lines.append(
                    f"- **{r.get('title','')}**  "
                    f"(materiality={r.get('materiality')})"
                )
        elif intent == "list_contradictions":
            for r in rows:
                lines.append(
                    f"- {r.get('link_type').upper()}: "
                    f"_{(r.get('src_text') or '')[:120]}_  vs  "
                    f"_{(r.get('dst_text') or '')[:120]}_"
                )
        elif intent == "list_recent_facts":
            for r in rows:
                lines.append(
                    f"- {(r.get('proposition_text') or '')[:200]}"
                )
        elif intent == "list_authorities":
            for r in rows:
                lines.append(
                    f"- {r.get('citation')} "
                    f"({r.get('authority_type','')}, weight={r.get('weight',0):.2f})"
                )
        else:
            for r in rows:
                lines.append(f"- {r}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Trace-family handler (MVI-2) — provenance + ledger lookup
# ---------------------------------------------------------------------------


@dataclass
class TraceFamilyResult:
    """Outcome of one trace-family request."""
    target_kind: Optional[str]
    target_id: Optional[str]
    events: list[dict]
    provenance: list[dict]
    rendered_answer: str
    escalation_needed: bool
    escalation_reason: Optional[str] = None


class TraceFamilyHandler:
    """Answers provenance / "why did you say X" queries by reading
    the reasoning ledger and provenance_event tables. No LLM in the
    common path — MVI-2 surfaces the raw chain of events, which is
    what attorneys actually want for audit.

    For MVI-2, the default target is the most recent completed run —
    "why did you say X" usually means "explain the last answer." A
    future MVI can resolve target_kind/target_id from text when the
    user names a specific assertion.
    """

    def __init__(self, matter_model: Any) -> None:
        self.matter_model = matter_model

    def run(
        self,
        query: str,
        contract: ExecutionContract,
    ) -> TraceFamilyResult:
        if self.matter_model is None:
            return TraceFamilyResult(
                target_kind=None, target_id=None,
                events=[], provenance=[],
                rendered_answer="",
                escalation_needed=True,
                escalation_reason="no matter model available",
            )
        # Find the most recent completed/running run — "most recent
        # answer" is the implicit referent for MVI-2 trace queries.
        try:
            row = self.matter_model.db.execute(
                """SELECT id, query, status, operation_type, started_at
                   FROM run_session
                   WHERE matter_id=?
                     AND (operation_type IS NULL
                          OR operation_type NOT IN ('manual_flush','background_flush'))
                   ORDER BY started_at DESC LIMIT 1""",
                (self.matter_model.matter_id,),
            ).fetchone()
        except Exception as exc:  # noqa: BLE001
            return TraceFamilyResult(
                target_kind=None, target_id=None,
                events=[], provenance=[],
                rendered_answer="",
                escalation_needed=True,
                escalation_reason=f"run lookup failed: {exc}",
            )
        if not row:
            return TraceFamilyResult(
                target_kind=None, target_id=None,
                events=[], provenance=[],
                rendered_answer="No prior runs to trace.",
                escalation_needed=False,
            )
        run_id = row["id"]
        events = self._fetch_ledger(run_id)
        provenance = self._fetch_run_provenance(run_id)
        rendered = self._render_trace(dict(row), events, provenance)
        return TraceFamilyResult(
            target_kind="run",
            target_id=run_id,
            events=events,
            provenance=provenance,
            rendered_answer=rendered,
            escalation_needed=False,
        )

    def _fetch_ledger(self, run_id: str) -> list[dict]:
        try:
            rows = self.matter_model.db.execute(
                """SELECT event_type, summary, why, created_at
                   FROM ledger_event
                   WHERE run_id=?
                   ORDER BY seq_no ASC
                   LIMIT 120""",
                (run_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def _fetch_run_provenance(self, run_id: str) -> list[dict]:
        """Read provenance_event rows tied to this run (P0.1 table).
        Best-effort — tolerates missing table on older schemas."""
        try:
            rows = self.matter_model.db.execute(
                """SELECT target_kind, target_id, event_kind, model_tier,
                          usage_label, created_at
                   FROM provenance_event
                   WHERE run_id=?
                   ORDER BY created_at ASC
                   LIMIT 80""",
                (run_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []

    @staticmethod
    def _render_trace(run_row: dict, events: list[dict], provenance: list[dict]) -> str:
        lines = [
            f"## Trace — run {run_row.get('id','?')}",
            "",
            f"**Query:** {run_row.get('query','(unknown)')[:240]}",
            f"**Status:** {run_row.get('status','unknown')}",
            f"**Type:** {run_row.get('operation_type') or 'query'}",
            f"**Started:** {run_row.get('started_at','?')}",
            "",
            "### Reasoning ledger",
            "",
        ]
        if events:
            for e in events:
                lines.append(
                    f"- **{e.get('event_type')}** — {(e.get('summary') or '')[:180]}"
                )
                if e.get("why"):
                    lines.append(f"  _why:_ {str(e['why'])[:180]}")
        else:
            lines.append("- (no ledger events recorded for this run)")
        lines.append("")
        lines.append("### LLM provenance")
        lines.append("")
        if provenance:
            for p in provenance[:40]:
                lines.append(
                    f"- {p.get('event_kind','?')} · tier={p.get('model_tier','?')} "
                    f"· stage={p.get('usage_label','?')} · "
                    f"target={p.get('target_kind','?')}:{p.get('target_id','?')}"
                )
        else:
            lines.append("- (no provenance events recorded)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Steer-family handler (MVI-4) — preview-only correction parser
# ---------------------------------------------------------------------------


STEER_PARSE_PROMPT = """You are parsing a user's matter-mutation intent from a short utterance. The user is NOT asking a question — they're correcting, overriding, or annotating something in the matter model.

Classify exactly one action:
- `correct_assertion` — a specific fact is wrong and needs a different value. E.g. "the date was April, not March", "no, the payment was 30 days, not 15".
- `reject_target` — a specific finding should be marked as rejected/withdrawn. E.g. "that assertion is wrong", "ignore that email".
- `set_source_role` — reclassify how a document should be weighted. E.g. "treat the MSA as operative", "mark the complaint as advocacy".
- `add_assumption` — add or change a working assumption. E.g. "assume the contract is valid", "for now treat jurisdiction as California".
- `other` — the intent doesn't map to a clear mutation pattern.

Extract a concise `target_hint` from the utterance (1–80 chars) that lets a matching step find the referenced fact/document/target. Examples: "payment date", "MSA", "the email from March 3".

If the action is `correct_assertion`, extract `old_value` and `new_value` verbatim when the user stated them. Otherwise leave them null.

User utterance: {query}

Respond ONLY in JSON:
{{
  "action": "correct_assertion" | "reject_target" | "set_source_role" | "add_assumption" | "other",
  "target_hint": "short phrase",
  "old_value": "string or null",
  "new_value": "string or null",
  "rationale": "one-line explanation"
}}
"""


@dataclass
class SteerFamilyResult:
    """Outcome of a steer-family parse. MVI-4 never auto-applies — it
    returns a preview the user confirms via the existing UI correction
    / annotation surfaces. Legal state must not mutate silently."""
    action: str                       # correct_assertion | reject_target | set_source_role | add_assumption | other
    target_hint: str
    old_value: Optional[str]
    new_value: Optional[str]
    candidates: list[dict]            # best-match assertions/documents for the hint
    rendered_answer: str              # markdown preview shown to the user
    escalation_needed: bool
    escalation_reason: Optional[str] = None


class SteerFamilyHandler:
    """Parses a correction/mutation utterance and returns a preview of
    the proposed change — never applies it. The user confirms via the
    existing UI (correction form, rejection control, etc.).

    This is deliberate: auto-applying mutations on ambiguous natural-
    language parses is the kind of failure that blows up legal work.
    MVI-4 does the heavy lifting of INTENT PARSING + TARGET MATCHING;
    the human does the final click.
    """

    def __init__(
        self,
        matter_model: Any,
        client: Optional[GeminiClient] = None,
    ) -> None:
        self.matter_model = matter_model
        self.client = client

    async def run(
        self,
        query: str,
        contract: ExecutionContract,
    ) -> SteerFamilyResult:
        if self.matter_model is None:
            return SteerFamilyResult(
                action="other",
                target_hint="",
                old_value=None, new_value=None,
                candidates=[],
                rendered_answer="",
                escalation_needed=True,
                escalation_reason="no matter model available",
            )
        parsed = await self._parse(query)
        action = parsed.get("action") or "other"
        target_hint = (parsed.get("target_hint") or "").strip()
        old_value = parsed.get("old_value")
        new_value = parsed.get("new_value")

        if action == "other" or not target_hint:
            return SteerFamilyResult(
                action=action,
                target_hint=target_hint,
                old_value=old_value, new_value=new_value,
                candidates=[],
                rendered_answer="",
                escalation_needed=True,
                escalation_reason="parse failed or ambiguous mutation intent",
            )

        candidates = self._find_candidates(action, target_hint)
        rendered = self._render_preview(
            action=action,
            target_hint=target_hint,
            old_value=old_value,
            new_value=new_value,
            candidates=candidates,
        )
        return SteerFamilyResult(
            action=action,
            target_hint=target_hint,
            old_value=old_value,
            new_value=new_value,
            candidates=candidates,
            rendered_answer=rendered,
            escalation_needed=False,
        )

    async def _parse(self, query: str) -> dict:
        if self.client is None:
            return {"action": "other"}
        try:
            response = await self.client.complete(
                STEER_PARSE_PROMPT.format(query=query),
                tier=ModelTier.NANO,
                json_mode=True,
                usage_label="steer_parse",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("steer_parse NANO failed: %s", exc)
            return {"action": "other"}
        try:
            parsed = json.loads(response or "{}")
        except (TypeError, ValueError):
            return {"action": "other"}
        return parsed if isinstance(parsed, dict) else {"action": "other"}

    def _find_candidates(self, action: str, target_hint: str) -> list[dict]:
        """Find up to 5 best-match rows to preview for the user.
        Different actions search different stores."""
        if not target_hint:
            return []
        needle = target_hint.lower()
        try:
            if action in {"correct_assertion", "reject_target"}:
                # Match against recent assertions by substring.
                rows = self.matter_model.assertions.list_recent(limit=120)
                scored = [
                    (self._score(r.get("proposition_text", ""), needle), r)
                    for r in rows
                ]
                scored = [s for s in scored if s[0] > 0]
                scored.sort(key=lambda t: -t[0])
                return [r for _, r in scored[:5]]
            if action == "set_source_role":
                rows = self.matter_model.list_reviewable_documents()
                scored = [
                    (self._score(r.get("path", ""), needle), r)
                    for r in rows
                ]
                scored = [s for s in scored if s[0] > 0]
                scored.sort(key=lambda t: -t[0])
                return [r for _, r in scored[:5]]
            # add_assumption has no pre-existing target to match; leave
            # candidates empty so the UI just echoes the hint + values.
            return []
        except Exception:
            return []

    @staticmethod
    def _score(text: str, needle: str) -> int:
        """Dumb substring scorer — count occurrences of needle words
        in text. Good enough for the preview; not a retrieval layer."""
        if not text or not needle:
            return 0
        text_l = text.lower()
        return sum(
            1 for word in needle.split()
            if len(word) > 2 and word in text_l
        )

    @staticmethod
    def _render_preview(
        action: str,
        target_hint: str,
        old_value: Optional[str],
        new_value: Optional[str],
        candidates: list[dict],
    ) -> str:
        """Deterministic markdown preview. The user confirms via UI."""
        lines = [f"## Proposed change — `{action}`", ""]
        lines.append(f"**Target hint:** {target_hint}")
        if old_value or new_value:
            lines.append(f"**Old value:** {old_value or '—'}")
            lines.append(f"**New value:** {new_value or '—'}")
        lines.append("")
        if not candidates:
            lines.append(
                "_No matching target found in the matter. Confirm or "
                "narrow the phrasing, or use the UI correction form "
                "to target a specific item._"
            )
            return "\n".join(lines)
        lines.append("**Candidates (pick one in the UI to apply):**")
        lines.append("")
        for i, c in enumerate(candidates, start=1):
            if action in {"correct_assertion", "reject_target"}:
                aid = c.get("id", "?")
                prop = (c.get("proposition_text") or "").strip()[:180]
                lines.append(f"{i}. `{aid}` — {prop}")
            elif action == "set_source_role":
                path = c.get("path", "?")
                pending = c.get("pending", 0)
                verified = c.get("verified", 0)
                lines.append(
                    f"{i}. `{path}`  ({pending} pending / {verified} reviewed)"
                )
            else:
                lines.append(f"{i}. {c}")
        lines.append("")
        lines.append(
            "_Irys will not auto-apply this mutation — confirm the "
            "target and value in the correction / annotation UI._"
        )
        return "\n".join(lines)
