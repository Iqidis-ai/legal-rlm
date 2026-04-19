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

    def to_audit_dict(self, terminal_family: Optional[str] = None) -> dict[str, Any]:
        """Audit dict for state.findings['route'] and ledger payloads.

        `family` remains for back-compat. adv#12 Finding #2: always
        emits `classifier_family` (the NANO-chosen route) and
        `terminal_family` (what actually executed after any
        escalation). Callers that know they escalated pass the
        terminal_family explicitly; otherwise both fields equal
        `family`. FastAPI clients read these fields to differentiate
        cascade behavior without parsing prose.
        """
        term = terminal_family or self.family
        return {
            "family": self.family,
            "classifier_family": self.family,
            "terminal_family": term,
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
CLASSIFIER_SCHEMA_VERSION = "mvi7.0"


VALID_FAMILIES = {
    "investigate", "read", "query", "trace", "steer",
    "compare", "scenario", "deliverable", "clarify",
}


INTENT_CLASSIFIER_PROMPT = """You are a routing classifier for a legal intelligence platform. For each user query you pick ONE route that matches how much work the system should actually do.

Six routes are available:

1. `investigate` — the user is asking a novel question about this legal matter that probably needs new evidence extraction, document search, or synthesis of findings the matter model does not yet contain. Examples: "What's our damages exposure?", "Did the opposing party breach the agreement?", "Find me evidence of intent to deceive." Route here if the matter is fresh (no facts yet), OR if the question targets material that probably hasn't been extracted, OR if the user explicitly asks for an investigation.

2. `read` — the user is asking for a summary, recap, restatement, reformat, or substantive answer synthesized from facts the matter model already contains. Examples: "Summarize our session for my team", "Give me that analysis as bullet points", "Draft a client email explaining our conclusions", "What have we found about the MSA?". Route here if the matter has content AND the query asks about existing findings as a narrative answer, not a plain enumeration.

3. `query` — the user is asking for a plain enumeration or lookup from matter model tables: "list all quants", "show me every actor", "what gaps are open", "give me the full timeline", "list every contradiction". These are DB reads — no synthesis or reasoning needed. Route here when the request is structurally "give me the list of X" or "show me the data in store Y".

4. `trace` — the user is asking where a specific prior conclusion came from: "why did you say X", "show me the source for claim Y", "what's the provenance of the damages figure", "how did you derive that timeline". Route here when the request targets the reasoning ledger / provenance of an existing finding.

5. `steer` — the user is CORRECTING a prior fact, OVERRIDING a belief, annotating, editing assumptions, or otherwise mutating matter state. Examples: "Actually the date was April, not March", "That assertion is wrong", "Mark the MSA as the operative contract", "Change the damages figure to 50000", "Ignore the email from March 3rd — it's drafts". The user is not asking a question; they're correcting or directing the matter model. Route here even when the phrasing is indirect ("no, the payment was 30 days after").

6. `compare` — the user is asking for a DIFF across time or across alternatives. Examples: "What changed since yesterday's production?", "What's new since the last investigation?", "How does this version differ from the previous one?", "What did we learn in the latest run?". Route here when the request is explicitly about changes, deltas, or comparisons across matter states.

7. `scenario` — the user is asking a HYPOTHETICAL or counterfactual — "what if X were true". Examples: "Redo the analysis assuming the contract is void", "What if we concede jurisdiction?", "Treat the waiver as valid and recompute damages", "Imagine the statute of limitations hasn't run". The user is not correcting state; they're asking for an alternative computation with an overridden assumption.

8. `deliverable` — the user is asking for a STRUCTURED WORK PRODUCT — a privilege log, a Rule 26 disclosure, a deposition outline, a meet-and-confer letter, etc. These have specific legal templates and the strictest verification/policy floor. Examples: "Generate a privilege log", "Draft the Rule 26(a)(1) disclosure", "Outline my deposition of Smith", "Prepare a production letter".

9. `clarify` — the user's referent is ambiguous or the query is so vague that proceeding would produce a wrong cheap answer. Examples: "Tell me about Smith" when there are two Smiths. Route here SPARINGLY — only when a specific ambiguity makes routing unsafe.

Guidance:
- Default to `investigate` on a fresh matter (has_any_facts=False).
- On a warm matter: `query` for "list X" / "show X" / "which X", `read` for "summarize" / "draft" / "explain" / "what does X mean", `trace` for "why" / "how did you" / "show the source", `steer` for "correct" / "actually X" / "no, it was Y" / "change" / "ignore", `compare` for "what changed" / "diff" / "since [X]", `scenario` for "what if" / "assume X" / "redo assuming", `deliverable` for named work products ("privilege log", "Rule 26", "deposition outline", "production letter").
- Never route to read/query/trace/steer/compare/scenario if has_any_facts=False — there's nothing to read or compare.
- Your job is cost governance, not content judgment. Keep the decision fast.

Matter state snapshot:
{snapshot_block}

Recent conversation turns (for deixis only — do not rely on them as a knowledge source):
{conversation_block}

User query: {query}

Respond ONLY with a single JSON object:
{{
  "family": "investigate" | "read" | "query" | "trace" | "steer" | "compare" | "scenario" | "deliverable" | "clarify",
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

    # MVI-2b (Fix D): decision-cache stage name for reasoning_cache.
    _CACHE_STAGE = "cascade_decision"

    async def decide(
        self,
        query: str,
        conversation_history: Optional[list[dict[str, str]]] = None,
    ) -> CascadeDecision:
        """Classify and derive a contract. Falls back to `investigate`
        on any classifier error — we'd rather over-spend than answer
        a question we can't route. This matches Codex's directive
        that silent cheap-wrong answers are worse than latency.

        Adversarial #10 Fix D: route cache. Reuses existing routing
        decisions keyed on (normalized_query, snapshot_fingerprint,
        classifier_version). Does NOT hash raw conversation turns
        (per Codex master plan — kills hit rate for no gain). On a
        NANO rate-limit event, the cache catches repeats so warm
        queries don't thundering-herd into the full AR loop.
        """
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

        # Route cache lookup. On hit the full decision (family +
        # confidence + rationale) is reused verbatim — no NANO call.
        cache_key = decision_cache_key(query, snapshot)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return CascadeDecision(
                family=cached["family"],
                confidence=float(cached.get("confidence", 0.0)),
                rationale=cached.get("rationale", "cache hit"),
                contract=self._contract_for(cached["family"]),
                classifier_version=CLASSIFIER_SCHEMA_VERSION,
                snapshot=snapshot,
                escalation_reason="cache_hit",
            )

        family, confidence, rationale = await self._classify(
            query, snapshot, conversation_history,
        )

        # Only cache REAL decisions (not classifier-error fallbacks).
        # Classifier-error rationale starts with "classifier error",
        # "parse error", or "unknown family" — these indicate
        # provider issues, not stable routing calls.
        fallback_markers = (
            "classifier error", "parse error", "unknown family",
        )
        is_fallback = any(
            rationale.startswith(m) for m in fallback_markers
        )
        if not is_fallback:
            self._cache_put(cache_key, family, confidence, rationale)
        elif cached is None:
            # Classifier failed AND cache is cold — last-resort attempt
            # against a cross-version cache lookup so repeated queries
            # during a sustained outage don't thundering-herd into AR.
            # Codex fallout R2: stale-cache fallback must NOT
            # misrepresent the reused route as the classifier's call.
            # We set classifier_family="_stale_cache_fallback" on the
            # audit payload (via to_audit_dict override) so the
            # ledger clearly distinguishes reused-during-outage from
            # an actual NANO emission.
            stale = self._cache_get_any_version(query, snapshot)
            if stale is not None:
                return CascadeDecision(
                    family=stale["family"],
                    confidence=float(stale.get("confidence", 0.0)),
                    rationale=(
                        f"classifier unavailable — "
                        f"reused stale cached route ({rationale})"
                    ),
                    contract=self._contract_for(stale["family"]),
                    classifier_version="_stale_cache_fallback",
                    snapshot=snapshot,
                    escalation_reason="stale_cache_fallback",
                )

        return CascadeDecision(
            family=family,
            confidence=confidence,
            rationale=rationale,
            contract=self._contract_for(family),
            classifier_version=CLASSIFIER_SCHEMA_VERSION,
            snapshot=snapshot,
        )

    def _cache_get(self, cache_key: str) -> Optional[dict]:
        """Read a cached routing decision from reasoning_cache. Silent
        miss on any error — cache failures never block the classifier."""
        mm = self.matter_model
        if mm is None or not hasattr(mm, "cache"):
            return None
        try:
            return mm.cache.get(self._CACHE_STAGE, cache_key)
        except Exception:
            return None

    def _cache_put(
        self,
        cache_key: str,
        family: str,
        confidence: float,
        rationale: str,
    ) -> None:
        mm = self.matter_model
        if mm is None or not hasattr(mm, "cache"):
            return
        try:
            mm.cache.put(self._CACHE_STAGE, cache_key, {
                "family": family,
                "confidence": float(confidence),
                "rationale": rationale,
                "schema_version": CLASSIFIER_SCHEMA_VERSION,
            })
        except Exception:
            pass

    def _cache_get_any_version(
        self, query: str, snapshot: AnswerabilitySnapshot,
    ) -> Optional[dict]:
        """Stale-cache escape hatch for sustained classifier outage.

        Codex fallout review round 2: the previous implementation
        returned "the first cached family it sees" — ignoring query,
        snapshot, and trust revision. That would resurrect
        pre-correction routes after a trust_revision bump from
        reject_target. The fix: match STRICTLY on the query +
        snapshot fingerprint (only the classifier schema version is
        allowed to differ), and preserve the existing trust-revision
        prefix that reasoning_cache already applies so a trust bump
        still invalidates stale routes.
        """
        mm = self.matter_model
        if mm is None or not hasattr(mm, "db"):
            return None
        try:
            import json as _json
            # Build a set of cache keys that would match the current
            # query + snapshot under ANY classifier schema version.
            # We iterate known schema versions the caller could have
            # cached under — today that's just the current
            # CLASSIFIER_SCHEMA_VERSION plus the next one back — but
            # the scan is bounded by the cache_key match so extra
            # entries never bleed in.
            candidate_versions = [CLASSIFIER_SCHEMA_VERSION]
            # Historical version prefixes that may still exist in the
            # cache table. Add new entries here when the schema bumps.
            # The list is intentionally explicit, not a wildcard —
            # we never match "any" route regardless of query.
            for past_ver in ("mvi6.0", "mvi4.0", "mvi2.0", "mvi1.0"):
                if past_ver != CLASSIFIER_SCHEMA_VERSION:
                    candidate_versions.append(past_ver)
            # The real reasoning_cache key is trust-prefixed by the
            # store (tr{N}:{hash}), so we look up by `cache.get()`
            # directly — which applies the prefix and thus respects
            # trust_revision bumps. That's the key property: a
            # rejection → trust_rev bump → stale routes DON'T resurface.
            for ver in candidate_versions:
                key = decision_cache_key(query, snapshot, ver)
                payload = self._cache_get(key)
                if payload is not None and payload.get("family") in VALID_FAMILIES:
                    return payload
            return None
        except Exception:
            return None

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
            family in {
                "read", "query", "trace", "steer",
                "compare", "scenario", "deliverable",
            }
            and not snapshot.has_any_facts
        ):
            # Belt-and-suspenders: if the classifier routes to a warm-
            # matter family but the matter is empty, override. Can't
            # read / query / trace / steer / compare / scenario an
            # empty matter.
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
        if family == "compare":
            # MVI-6: diff matter state across named snapshots.
            return ExecutionContract(
                family="compare",
                min_iter=0,
                max_iter=1,
                citation_floor=0,
                answer_confidence_floor=0.0,
                escalation_allowed=True,
            )
        if family == "scenario":
            # MVI-6: override assumption + re-read impacted state via
            # the read-family pipeline.
            return ExecutionContract(
                family="scenario",
                min_iter=0,
                max_iter=1,
                citation_floor=0,
                answer_confidence_floor=0.5,
                escalation_allowed=True,
            )
        if family == "deliverable":
            # MVI-7: strictest verification + policy floor. Render a
            # named legal work product from verified matter state.
            # Contract citation_floor >= 1 so a deliverable with
            # nothing to render escalates rather than silently
            # producing an empty template.
            return ExecutionContract(
                family="deliverable",
                min_iter=0,
                max_iter=1,
                citation_floor=1,
                answer_confidence_floor=0.7,
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

Attorney guidance (internal, work-product — frame + prioritize your answer around these, but NEVER quote these as evidence, NEVER cite them as document sources, NEVER reveal them verbatim in the answer text):
{attorney_guidance_block}

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
    """Outcome of one read-family call.

    `failure_kind` distinguishes a real routing signal (state
    insufficient → escalate to investigate) from an infra failure
    (LLM call itself failed → surface to user, do NOT silently kick
    off a multi-minute AR loop during an outage). Adversarial #10
    finding #6.
    """
    answer: str
    confidence_label: str         # "low" | "medium" | "high"
    confidence_score: float       # 0.0-1.0 numeric mapping
    citations: list[str]
    escalation_needed: bool
    escalation_reason: Optional[str]
    raw_response: str
    failure_kind: Optional[str] = None  # None | "state_insufficient" | "infra"


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
        *,
        include_attorney_guidance: bool = False,
    ) -> ReadFamilyResult:
        """Answer a query against existing matter state.

        `include_attorney_guidance` gates whether verification_state
        review notes + document annotations are rendered into the
        prompt as attorney work-product. DEFAULT IS FALSE — every
        external / export / deliverable caller MUST stay false so
        strategic notes never leak. Only the internal UI/API path
        (Irys.investigate → _run_read_family) opts in. Per the P0
        notes→reasoning design.
        """
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

        context = self._assemble_read_context(
            include_attorney_guidance=include_attorney_guidance,
        )
        prompt = READ_FAMILY_PROMPT.format(
            matter_summary=context["matter_summary"],
            verified_block=context["verified_block"],
            attorney_guidance_block=context["attorney_guidance_block"],
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
            # Adversarial #10 fix: infra failure must NOT silently
            # fall through to a multi-minute AR loop. Surface it.
            logger.warning("read_synth call failed: %s", exc)
            return ReadFamilyResult(
                answer="",
                confidence_label="low",
                confidence_score=0.0,
                citations=[],
                escalation_needed=False,  # do NOT auto-escalate on infra
                escalation_reason=f"read call failed: {exc}",
                raw_response="",
                failure_kind="infra",
            )

        parsed = self._parse_read_json(response)
        label = str(parsed.get("answer_confidence") or "low").lower()
        if label not in self._CONFIDENCE_MAP:
            label = "low"
        score = self._CONFIDENCE_MAP[label]

        answer = str(parsed.get("answer") or "").strip()
        # Codex fallout R3: citations must be non-empty STRINGS. The
        # R2 fix still accepted int/float/bool, so `[0]` or `[False]`
        # trivially satisfied the floor. Legal citations are document
        # identifiers (paths, filenames, bates numbers) — always
        # strings. Anything else is LLM hallucination and gets filtered.
        raw_citations = parsed.get("citations") or []
        citations = [
            c.strip() for c in raw_citations
            if isinstance(c, str) and not isinstance(c, bool) and c.strip()
        ]
        escalation_hint = str(parsed.get("escalation_hint") or "").strip()

        # Adversarial #10 finding #2: citation_floor was declared on
        # the contract but never enforced. A high-confidence zero-
        # citation answer used to ship. Now an answer below the floor
        # forces escalation to investigate (if the contract allows)
        # regardless of the confidence label.
        citation_shortfall = len(citations) < max(0, contract.citation_floor)
        confidence_shortfall = score < contract.answer_confidence_floor

        escalation_needed = (
            contract.escalation_allowed
            and (citation_shortfall or confidence_shortfall)
        )
        if escalation_needed:
            reasons = []
            if citation_shortfall:
                reasons.append(
                    f"citations {len(citations)} < floor {contract.citation_floor}"
                )
            if confidence_shortfall:
                reasons.append(
                    f"confidence {label} < floor {contract.answer_confidence_floor}"
                )
            escalation_reason = (
                f"{escalation_hint} (" + "; ".join(reasons) + ")"
                if escalation_hint else "; ".join(reasons)
            )
            failure_kind = "state_insufficient"
        else:
            escalation_reason = None
            failure_kind = None

        return ReadFamilyResult(
            answer=answer,
            confidence_label=label,
            confidence_score=score,
            citations=citations,
            escalation_needed=escalation_needed,
            escalation_reason=escalation_reason,
            raw_response=response or "",
            failure_kind=failure_kind,
        )

    def _assemble_read_context(
        self,
        *,
        include_attorney_guidance: bool = False,
    ) -> dict[str, str]:
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

        verified_block, candidate_block, visible_verified_ids, visible_docs = (
            self._render_assertions(mm)
        )
        issues_block = self._render_issues(mm)
        gaps_block = self._render_gaps(mm)

        # P0 notes→reasoning: render attorney guidance only when the
        # caller explicitly opts in. Fail-closed. Scoped to notes on
        # prompt-visible verified assertions + annotations matching
        # their source documents — never a global recent-note dump.
        if include_attorney_guidance:
            guidance_block = self._render_attorney_guidance(
                mm, visible_verified_ids, visible_docs,
            )
        else:
            guidance_block = ""

        return {
            "matter_summary": matter_summary,
            "verified_block": verified_block or "(no verified facts yet)",
            "attorney_guidance_block": (
                guidance_block or "(no attorney guidance on these facts)"
            ),
            "candidate_block": candidate_block or "(no candidate facts yet)",
            "issues_block": issues_block or "(no open issues)",
            "gaps_block": gaps_block or "(no known gaps)",
        }

    def _render_assertions(
        self, mm: Any,
    ) -> tuple[str, str, list[str], list[str]]:
        """Render up to ~40 recent assertions split into verified /
        candidate lanes. Trust-aware so privileged/rejected content
        never leaks into the read prompt (content policy stays
        enforced even here).

        Returns (verified_block, candidate_block, visible_verified_ids,
        visible_docs). The last two are the assertion ids and source
        documents actually rendered into the verified block — used by
        `_render_attorney_guidance` to scope note lookup so a random
        recent note can't dominate an unrelated read query.
        """
        try:
            rows = mm.assertions.list_recent(limit=60)
        except Exception:
            return "", "", [], []
        verified_lines: list[str] = []
        candidate_lines: list[str] = []
        visible_verified_ids: list[str] = []
        visible_docs_set: set[str] = set()
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
                    visible_verified_ids.append(str(r.get("id")))
                    if doc and doc != "unknown":
                        visible_docs_set.add(doc)
            else:
                if len(candidate_lines) < 15:
                    candidate_lines.append(line)
        return (
            "\n".join(verified_lines),
            "\n".join(candidate_lines),
            visible_verified_ids,
            sorted(visible_docs_set),
        )

    # P0 notes→reasoning caps per Codex design.
    _GUIDANCE_MAX_ENTRIES = 5
    _GUIDANCE_MAX_VERIFY_NOTE_CHARS = 220
    _GUIDANCE_MAX_ANNOTATION_CHARS = 180
    _GUIDANCE_MAX_TOTAL_CHARS = 900

    def _render_attorney_guidance(
        self,
        mm: Any,
        visible_verified_ids: list[str],
        visible_docs: list[str],
    ) -> str:
        """Build the internal attorney-guidance block. Only verified,
        human-authored review notes on prompt-visible assertions plus
        annotations matching their source docs. Never reveals:
          - stale_reason (mark_stale reuses the review_note column;
            explicit status='verified' filter protects us)
          - system-authored notes (reviewed_by_kind filter)
          - rejected / candidate notes
          - notes on assertions not in the current verified_block

        Returns "" when nothing qualifies — the caller then emits
        "(no attorney guidance on these facts)" so the LLM knows the
        absence is intentional rather than a missing placeholder.
        """
        entries: list[str] = []
        total_chars = 0

        # 1. Verification notes on prompt-visible verified assertions,
        # ordered by the same visibility order (matches verified_block).
        if visible_verified_ids:
            _BIND_LIMIT = 900
            note_rows_by_id: dict[str, dict] = {}
            for _chunk_start in range(0, len(visible_verified_ids), _BIND_LIMIT):
                _chunk = visible_verified_ids[
                    _chunk_start : _chunk_start + _BIND_LIMIT
                ]
                try:
                    _rows = mm.db.execute(
                        """SELECT target_id, review_note, reviewed_by_kind
                           FROM verification_state
                           WHERE matter_id=?
                             AND target_kind='assertion'
                             AND status='verified'
                             AND reviewed_by_kind IN ('user', 'attorney')
                             AND review_note IS NOT NULL
                             AND TRIM(review_note) <> ''
                             AND target_id IN ({})""".format(
                            ",".join("?" * len(_chunk))
                        ),
                        (mm.matter_id, *_chunk),
                    ).fetchall()
                except Exception as _exc:
                    logger.warning("attorney guidance — note query failed: %s", _exc)
                    _rows = []
                for _r in _rows:
                    note_rows_by_id[str(_r["target_id"])] = dict(_r)
            for aid in visible_verified_ids:
                if len(entries) >= self._GUIDANCE_MAX_ENTRIES:
                    break
                row = note_rows_by_id.get(aid)
                if not row:
                    continue
                raw = " ".join(str(row.get("review_note") or "").split())
                if not raw:
                    continue
                truncated = raw[: self._GUIDANCE_MAX_VERIFY_NOTE_CHARS]
                if len(raw) > self._GUIDANCE_MAX_VERIFY_NOTE_CHARS:
                    truncated = truncated.rstrip() + "…"
                line = f"- Attorney note, not evidence: \"{truncated}\""
                if total_chars + len(line) > self._GUIDANCE_MAX_TOTAL_CHARS:
                    break
                entries.append(line)
                total_chars += len(line) + 1  # +1 for the newline joiner

        # 2. Document annotations matching visible source docs.
        # Current annotation schema key is `document_pattern` — same
        # matching rules as trust overrides (full path or basename).
        if (
            visible_docs
            and len(entries) < self._GUIDANCE_MAX_ENTRIES
            and total_chars < self._GUIDANCE_MAX_TOTAL_CHARS
        ):
            try:
                ann_rows = mm.db.execute(
                    """SELECT document_pattern, annotation_text, annotation_type
                       FROM document_annotation
                       WHERE matter_id=?
                         AND annotation_text IS NOT NULL
                         AND TRIM(annotation_text) <> ''""",
                    (mm.matter_id,),
                ).fetchall()
            except Exception as _exc:
                logger.warning("attorney guidance — annotation query failed: %s", _exc)
                ann_rows = []
            # Normalize visible_docs for match.
            docs_norm = {d.replace("\\", "/"): d for d in visible_docs}
            bases_norm = {d.split("/")[-1]: d for d in docs_norm}
            for ann in ann_rows:
                if len(entries) >= self._GUIDANCE_MAX_ENTRIES:
                    break
                pat = str(ann["document_pattern"] or "").replace("\\", "/")
                if not pat:
                    continue
                pat_base = pat.split("/")[-1]
                matched = None
                if pat in docs_norm:
                    matched = docs_norm[pat]
                elif pat_base in bases_norm:
                    matched = bases_norm[pat_base]
                if not matched:
                    continue
                raw = " ".join(str(ann["annotation_text"] or "").split())
                if not raw:
                    continue
                truncated = raw[: self._GUIDANCE_MAX_ANNOTATION_CHARS]
                if len(raw) > self._GUIDANCE_MAX_ANNOTATION_CHARS:
                    truncated = truncated.rstrip() + "…"
                kind_label = str(ann.get("annotation_type") or "").strip().upper()
                label_part = f" [{kind_label}]" if kind_label else ""
                line = (
                    f"- Annotation on {matched}{label_part}: \"{truncated}\""
                )
                if total_chars + len(line) > self._GUIDANCE_MAX_TOTAL_CHARS:
                    break
                entries.append(line)
                total_chars += len(line) + 1

        return "\n".join(entries)

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

        candidates = self._find_candidates(
            action, target_hint, old_value, new_value,
        )
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

    def _find_candidates(
        self,
        action: str,
        target_hint: str,
        old_value: Optional[str] = None,
        new_value: Optional[str] = None,
    ) -> list[dict]:
        """Find up to 5 best-match rows to preview for the user.
        Different actions search different stores.

        Adversarial #10 Fix F: scoring factors in old_value / new_value
        and specific tokens (dates, numeric amounts) so that
        'April 15' / '$50' style corrections rank the correct
        assertion above siblings that happen to share generic words.
        """
        if not target_hint and not old_value and not new_value:
            return []
        try:
            if action in {"correct_assertion", "reject_target"}:
                rows = self.matter_model.assertions.list_recent(limit=120)
                scored = [
                    (
                        self._score_assertion(
                            r.get("proposition_text", ""),
                            target_hint, old_value, new_value,
                        ),
                        r,
                    )
                    for r in rows
                ]
                scored = [s for s in scored if s[0] > 0]
                scored.sort(key=lambda t: -t[0])
                return [r for _, r in scored[:5]]
            if action == "set_source_role":
                rows = self.matter_model.list_reviewable_documents()
                scored = [
                    (
                        self._score_assertion(
                            r.get("path", ""),
                            target_hint, old_value, new_value,
                        ),
                        r,
                    )
                    for r in rows
                ]
                scored = [s for s in scored if s[0] > 0]
                scored.sort(key=lambda t: -t[0])
                return [r for _, r in scored[:5]]
            # add_assumption has no pre-existing target to match.
            return []
        except Exception:
            return []

    # Token kinds used in specific-match scoring. Dates and numeric
    # amounts get higher weights because they're what distinguishes
    # "April 15" from "April 20" on otherwise-identical sibling
    # assertions.
    #
    # Codex fallout R7+R8: unified ISO regex. Rules:
    #   - Year 1900-2099 (plausible legal-document range)
    #   - Neither side may touch digit, hyphen, letter, or underscore.
    #     Lookbehind rejects digit-shifted (x20260-13-45y),
    #     hyphen-bounded identifiers (Case-2026-13-45-A),
    #     letter-embedded IDs (Ex2026-04-15A), and underscore
    #     identifiers (case_2026-04-15_a). Valid embedded dates
    #     like `ref=2026-04-15/paper` still match because `=` / `/`
    #     are delimiters, not word chars.
    # Calendar validation in the match loop decides whether to also
    # emit the date token — implausible shapes consume-only.
    #
    # Known over-match: `2026-04-15-2026-04-20` (date-range shape).
    # The trailing hyphen from the first date blocks the second's
    # lookbehind, so both dates drop and fragments flow. This is
    # acceptable for a preview-only handler (the user confirms the
    # candidate anyway); a future enhancement could add an explicit
    # date-range pattern alongside.
    _ISO_DATE_REGEX = (
        r"(?<![\w\d-])((?:19|20)\d{2})-(\d{1,2})-(\d{1,2})(?![\w\d-])"
    )
    _DATE_PATTERNS = [
        # English month forms remain separate since they never
        # collide with hyphenated identifiers.
        # "April 15" / "Apr 15" / "March 3"
        r"\b(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December|Jan|Feb|Mar|Apr|"
        r"Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+\d{1,2}(?:,?\s+\d{4})?\b",
        # "15 April" / "15th April"
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|"
        r"May|June|July|August|September|October|November|December|"
        r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\b",
    ]
    _NUMBER_PATTERN = r"\$?\d[\d,]*(?:\.\d+)?"

    @classmethod
    def _specific_tokens(cls, *fragments: Optional[str]) -> set[str]:
        """Extract date and numeric tokens from any of the fragments.
        Returns a set of lowercased tokens worth matching on.

        Codex fallout R3: ISO dates are validated as real calendar
        dates via datetime.date (rejects Feb 31, Apr 31, etc.). And
        number extraction STRIPS spans we already captured as valid
        dates so the "13" and "45" inside a rejected "2026-13-45"
        don't leak through as independent numeric tokens.
        """
        import re as _re
        from datetime import date as _date
        out: set[str] = set()
        for frag in fragments:
            if not frag:
                continue
            text = str(frag)
            # Extract date tokens first. ISO dates get an extra
            # calendar-validation pass via datetime.date.
            consumed_spans: list[tuple[int, int]] = []
            # ISO YYYY-MM-DD is _DATE_PATTERNS[0]
            # Codex fallout R7: single unified pass on the ISO regex.
            # Every match (which is already year-range + boundary
            # filtered) gets its span consumed so fragments can't
            # leak. Then calendar validation decides whether to
            # EMIT the date token. Consistent consumption for both
            # valid dates (consume + emit) and calendar-invalid
            # dates (consume only).
            for m in _re.finditer(cls._ISO_DATE_REGEX, text):
                consumed_spans.append(m.span())
                try:
                    yr = int(m.group(1))
                    mo = int(m.group(2))
                    dy = int(m.group(3))
                    _date(yr, mo, dy)
                except (ValueError, TypeError):
                    continue
                out.add(m.group(0).lower())
            # Other date patterns (English month + day).
            for pat in cls._DATE_PATTERNS[1:]:
                for m in _re.finditer(pat, text, flags=_re.IGNORECASE):
                    out.add(m.group().lower().strip())
                    consumed_spans.append(m.span())
            # Number extraction — skip any match whose span overlaps
            # a consumed date span, so ISO fragments ("2026", "13",
            # "45") don't leak through as fake numeric tokens.
            for m in _re.finditer(cls._NUMBER_PATTERN, text):
                s, e = m.span()
                if any(not (e <= cs or s >= ce)
                       for cs, ce in consumed_spans):
                    continue
                token = m.group().strip()
                if len(token) >= 2:
                    out.add(token.lower())
        return out

    @classmethod
    def _score_assertion(
        cls,
        text: str,
        target_hint: Optional[str],
        old_value: Optional[str] = None,
        new_value: Optional[str] = None,
    ) -> int:
        """Rank assertions by weighted token overlap.

        Codex fallout R2: old_value GENERIC words (the descriptive
        part, e.g. "invoice due date") must NOT boost score — the
        user is saying the assertion is MISCHARACTERIZED. Matching
        on those words just ranks the wrong-labeled sibling higher.
        Only SPECIFIC tokens inside old_value (dates, amounts) are
        discriminating signals and still score.

        Weights:
          - word overlap with target_hint: 1 point per generic word
          - specific tokens from target_hint + old_value (dates +
            amounts): 5 points per token — heavy weight because
            they're what distinguishes sibling assertions. Extracted
            from old_value specifically so "April 15" in a
            "payment was April 15" correction still steers ranking.
          - word overlap with old_value (generic words): 0 points.
            The user is saying "the text that mentions this is
            wrong"; don't preferentially rank it.
        new_value is NOT used for scoring: the user is saying the
        text SHOULD become new_value; it doesn't currently contain it.
        """
        if not text:
            return 0
        text_l = text.lower()

        score = 0
        # Generic-word overlap with target hint only.
        for word in (target_hint or "").split():
            if len(word) > 2 and word.lower() in text_l:
                score += 1
        # Specific tokens (dates + amounts) — from target_hint AND
        # old_value. These are the discriminating signals.
        tokens = cls._specific_tokens(target_hint, old_value)
        for tok in tokens:
            if tok in text_l:
                score += 5
        return score

    # Back-compat alias — kept so external callers that imported the
    # old `_score` still work.
    @classmethod
    def _score(cls, text: str, needle: str) -> int:
        return cls._score_assertion(text, needle)

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


# ---------------------------------------------------------------------------
# Compare-family handler (MVI-6) — state diff across runs
# ---------------------------------------------------------------------------


@dataclass
class CompareFamilyResult:
    """Outcome of a compare-family request."""
    baseline_run_id: Optional[str]
    current_assertion_count: int
    baseline_assertion_count: int
    new_documents_read: int
    coverage_delta: float
    open_gap_delta: int
    rendered_answer: str
    escalation_needed: bool
    escalation_reason: Optional[str] = None


class CompareFamilyHandler:
    """Diffs the current matter state against a prior checkpoint and
    surfaces what's changed. MVI-6 uses run_session rows as natural
    snapshot boundaries — "since the last completed run" is the
    default comparison axis. Zero LLM in the common path.
    """

    def __init__(self, matter_model: Any) -> None:
        self.matter_model = matter_model

    def run(
        self,
        query: str,
        contract: ExecutionContract,
    ) -> CompareFamilyResult:
        if self.matter_model is None:
            return CompareFamilyResult(
                baseline_run_id=None,
                current_assertion_count=0,
                baseline_assertion_count=0,
                new_documents_read=0,
                coverage_delta=0.0,
                open_gap_delta=0,
                rendered_answer="",
                escalation_needed=True,
                escalation_reason="no matter model available",
            )
        try:
            # Baseline = the run BEFORE the most recent one. If only
            # one run exists, compare against zero state.
            rows = self.matter_model.db.execute(
                """SELECT id, assertions_at_start, started_at
                   FROM run_session
                   WHERE matter_id=?
                     AND status IN ('completed','failed','interrupted')
                     AND (operation_type IS NULL
                          OR operation_type NOT IN ('manual_flush','background_flush'))
                   ORDER BY started_at DESC
                   LIMIT 2""",
                (self.matter_model.matter_id,),
            ).fetchall()
        except Exception as exc:  # noqa: BLE001
            return CompareFamilyResult(
                baseline_run_id=None,
                current_assertion_count=0,
                baseline_assertion_count=0,
                new_documents_read=0,
                coverage_delta=0.0,
                open_gap_delta=0,
                rendered_answer="",
                escalation_needed=True,
                escalation_reason=f"run lookup failed: {exc}",
            )
        current = self.matter_model.assertions.count()
        if not rows:
            baseline = None
            baseline_count = 0
        elif len(rows) == 1:
            # Only one run — compare against assertions_at_start of
            # that run (the "empty matter" baseline).
            baseline = dict(rows[0])
            baseline_count = int(baseline.get("assertions_at_start") or 0)
        else:
            # Most recent run's assertions_at_start is how many
            # assertions existed BEFORE it ran — that's our baseline.
            baseline = dict(rows[0])
            baseline_count = int(baseline.get("assertions_at_start") or 0)

        assertion_delta = current - baseline_count
        # Other deltas — documents read, open gaps, coverage sum —
        # are computed from current state vs nothing for MVI-6. A
        # richer snapshot layer lives in MVI-7+.
        try:
            coverage_rows = self.matter_model.get_issue_coverage_report(
                policy_audience="internal",
            )
            coverage_sum = sum(
                float(r.get("coverage_fraction") or 0.0)
                for r in coverage_rows
            )
        except Exception:
            coverage_sum = 0.0
        try:
            open_gaps = int(self.matter_model.gaps.count_open())
        except Exception:
            open_gaps = 0

        rendered = self._render(
            baseline_run_id=baseline.get("id") if baseline else None,
            baseline_count=baseline_count,
            current=current,
            assertion_delta=assertion_delta,
            coverage_sum=coverage_sum,
            open_gaps=open_gaps,
        )
        return CompareFamilyResult(
            baseline_run_id=(baseline.get("id") if baseline else None),
            current_assertion_count=current,
            baseline_assertion_count=baseline_count,
            new_documents_read=0,  # MVI-6 scope; richer in later MVI
            coverage_delta=coverage_sum,
            open_gap_delta=open_gaps,
            rendered_answer=rendered,
            escalation_needed=False,
        )

    @staticmethod
    def _render(
        baseline_run_id: Optional[str],
        baseline_count: int,
        current: int,
        assertion_delta: int,
        coverage_sum: float,
        open_gaps: int,
    ) -> str:
        lines = ["## What changed", ""]
        if baseline_run_id:
            lines.append(f"**Baseline:** run `{baseline_run_id}`")
        else:
            lines.append("**Baseline:** empty matter (no prior runs)")
        lines.append("")
        sign = "+" if assertion_delta >= 0 else ""
        lines.append(
            f"- **Assertions:** {baseline_count} → {current} "
            f"({sign}{assertion_delta})"
        )
        lines.append(f"- **Current issue coverage sum:** {coverage_sum:.2f}")
        lines.append(f"- **Open gaps right now:** {open_gaps}")
        lines.append("")
        if assertion_delta == 0 and coverage_sum == 0 and open_gaps == 0:
            lines.append(
                "_No material changes. If you expected new findings, "
                "the run may not have produced them, or a later run "
                "hasn't landed yet._"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scenario-family handler (MVI-6) — assumption override + re-read
# ---------------------------------------------------------------------------


SCENARIO_PARSE_PROMPT = """You are parsing a hypothetical legal scenario. The user wants the system to re-answer as if a specific assumption were true. Extract a concise, one-sentence assumption statement that can be treated as temporarily true for reasoning.

Examples:
- "Redo assuming the contract is void" → "The contract is void"
- "What if jurisdiction is California" → "Jurisdiction is California"
- "Treat the waiver as valid" → "The waiver is valid"
- "Imagine the statute of limitations hasn't run" → "The statute of limitations has not run"

User utterance: {query}

Respond ONLY in JSON:
{{
  "assumption": "a concise, one-sentence assumption",
  "core_question": "what the user actually wants answered under this assumption, if stated"
}}
"""


@dataclass
class ScenarioFamilyResult:
    """Outcome of a scenario-family request."""
    assumption: str
    core_question: str
    answer: str
    confidence_label: str
    confidence_score: float
    escalation_needed: bool
    escalation_reason: Optional[str] = None


class ScenarioFamilyHandler:
    """Parses a hypothetical assumption, then re-runs the read family
    with the assumption injected into the synthesis prompt. MVI-6 is
    the narrow slice — the assumption is not persisted to the matter
    (no state mutation), it's just used as a one-turn override. A
    future MVI can persist scenario_only assumptions for multi-turn
    counterfactual sessions."""

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
    ) -> ScenarioFamilyResult:
        if self.matter_model is None:
            return ScenarioFamilyResult(
                assumption="", core_question="",
                answer="", confidence_label="low",
                confidence_score=0.0,
                escalation_needed=True,
                escalation_reason="no matter model available",
            )
        parsed = await self._parse(query)
        assumption = (parsed.get("assumption") or "").strip()
        core_question = (parsed.get("core_question") or query).strip()
        if not assumption:
            return ScenarioFamilyResult(
                assumption="", core_question=core_question,
                answer="", confidence_label="low",
                confidence_score=0.0,
                escalation_needed=True,
                escalation_reason="could not parse assumption",
            )
        # Inline the assumption as a prefix on the user question so the
        # read handler's synth call treats it as a temporary override.
        # No state mutation.
        overridden_query = (
            f"TEMPORARY HYPOTHETICAL — treat the following as true for "
            f"the purpose of this question only: \"{assumption}\"\n\n"
            f"Under that assumption: {core_question}"
        )
        read_handler = ReadFamilyHandler(
            client=self.client, matter_model=self.matter_model,
        )
        read_contract = ExecutionContract(
            family="scenario",
            min_iter=0, max_iter=1,
            citation_floor=contract.citation_floor,
            answer_confidence_floor=contract.answer_confidence_floor,
            escalation_allowed=contract.escalation_allowed,
        )
        read_result = await read_handler.run(
            query=overridden_query,
            contract=read_contract,
            conversation_history=conversation_history,
        )
        return ScenarioFamilyResult(
            assumption=assumption,
            core_question=core_question,
            answer=read_result.answer,
            confidence_label=read_result.confidence_label,
            confidence_score=read_result.confidence_score,
            escalation_needed=read_result.escalation_needed,
            escalation_reason=read_result.escalation_reason,
        )

    async def _parse(self, query: str) -> dict:
        try:
            response = await self.client.complete(
                SCENARIO_PARSE_PROMPT.format(query=query),
                tier=ModelTier.NANO,
                json_mode=True,
                usage_label="scenario_parse",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("scenario_parse NANO failed: %s", exc)
            return {}
        try:
            parsed = json.loads(response or "{}")
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}


# ---------------------------------------------------------------------------
# Deliverable-family handler (MVI-7) — first renderer: privilege log
# ---------------------------------------------------------------------------


DELIVERABLE_SUB_INTENTS = [
    ("privilege_log", "privilege log / privilege review / claw-back list"),
    ("dep_outline", "deposition outline / cross-examination prep"),
    ("rule_26", "Rule 26(a)(1) initial disclosures"),
    ("production_letter", "production cover letter"),
    ("other", "any other named work product"),
]


DELIVERABLE_SUB_INTENT_PROMPT = """Classify which named legal work product the user is asking for. Pick EXACTLY one.

Options:
{intent_list}

User request: {query}

Respond ONLY with JSON: {{"intent": "name_from_list"}}
"""


@dataclass
class DeliverableFamilyResult:
    """Outcome of a deliverable-family request."""
    intent: str                     # privilege_log | dep_outline | rule_26 | production_letter | other
    rendered_answer: str
    row_count: int
    escalation_needed: bool
    escalation_reason: Optional[str] = None


class DeliverableFamilyHandler:
    """Renders a named legal work product from the verified matter
    state. MVI-7 ships ONE renderer (privilege log) to prove the
    deliverable pattern — other intents escalate to read for now.
    Future MVIs add renderers one at a time; the classifier already
    carries the sub-intent so downstream expansion is a per-template
    add, not a factory rewrite.
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
    ) -> DeliverableFamilyResult:
        if self.matter_model is None:
            return DeliverableFamilyResult(
                intent="",
                rendered_answer="",
                row_count=0,
                escalation_needed=True,
                escalation_reason="no matter model available",
            )
        intent = await self._resolve_sub_intent(query)
        if intent == "privilege_log":
            return self._render_privilege_log()
        # Other intents are not yet shipped — escalate to read so the
        # user still gets an answer. This matches Codex's directive:
        # ship ONE renderer first, add the rest one-by-one.
        return DeliverableFamilyResult(
            intent=intent,
            rendered_answer="",
            row_count=0,
            escalation_needed=True,
            escalation_reason=(
                f"renderer '{intent}' not yet implemented — "
                f"escalating to read handler for a narrative response"
            ),
        )

    async def _resolve_sub_intent(self, query: str) -> str:
        """NANO decides which named deliverable the user wants. If the
        client is absent, return 'other' so we escalate rather than
        silently default to privilege_log."""
        if self.client is None:
            return "other"
        intent_list = "\n".join(
            f"- {name}: {desc}" for name, desc in DELIVERABLE_SUB_INTENTS
        )
        prompt = DELIVERABLE_SUB_INTENT_PROMPT.format(
            intent_list=intent_list, query=query,
        )
        try:
            response = await self.client.complete(
                prompt,
                tier=ModelTier.NANO,
                json_mode=True,
                usage_label="deliverable_sub_intent",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("deliverable sub-intent NANO failed: %s", exc)
            return "other"
        try:
            parsed = json.loads(response or "{}")
        except (TypeError, ValueError):
            return "other"
        intent = str(parsed.get("intent") or "other").strip()
        valid = {name for name, _ in DELIVERABLE_SUB_INTENTS}
        return intent if intent in valid else "other"

    def _render_privilege_log(self) -> DeliverableFamilyResult:
        """Render a privilege log from document_card rows where
        privilege_flag = 1 (stored as INTEGER; MVP.4's fail-closed
        mapping puts both 'true' and 'unknown' LLM outputs into this
        bucket). Columns align with the standard Rule 26(b)(5)(A)(ii)
        pattern: entry no., date, author, recipient, type, privilege
        basis, description.

        Descriptions are limited to the `purpose` field's one-sentence
        classification summary — privileged body text is NOT quoted.
        """
        try:
            rows = self.matter_model.db.execute(
                """SELECT dc.id, dc.doc_id, di.relative_path AS path,
                          dc.doc_type, dc.doc_subtype, dc.title,
                          dc.author, dc.sender, dc.recipient,
                          dc.creation_date, dc.effective_date,
                          dc.privilege_flag, dc.unresolved_flags,
                          dc.purpose
                   FROM document_card dc
                   JOIN document_inventory di ON di.id = dc.doc_id
                   WHERE di.matter_id = ?
                     AND dc.privilege_flag = 1
                   ORDER BY
                      COALESCE(dc.creation_date, dc.effective_date, ''),
                      dc.id""",
                (self.matter_model.matter_id,),
            ).fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.warning("privilege log query failed: %s", exc)
            return DeliverableFamilyResult(
                intent="privilege_log",
                rendered_answer="",
                row_count=0,
                escalation_needed=True,
                escalation_reason=f"privilege log query failed: {exc}",
            )

        if not rows:
            return DeliverableFamilyResult(
                intent="privilege_log",
                rendered_answer=(
                    "## Privilege log\n\n"
                    "_No documents currently classified as privileged or "
                    "privilege-unknown. If documents are missing, confirm "
                    "ingestion has completed for this matter before "
                    "generating the log._"
                ),
                row_count=0,
                escalation_needed=False,
            )

        header = (
            "| # | Doc ID | Date | Author | Recipient | Type | Basis | Description |\n"
            "|---|--------|------|--------|-----------|------|-------|-------------|"
        )
        body_lines = []
        tbd_count = 0
        # Adversarial #10 Fix A — NEVER render `purpose` or `title` as
        # the description. Those are free-form LLM-authored fields and
        # can (and in real demos DO) contain privileged substance.
        # Rule 26(b)(5)(A)(ii) requires a description sufficient to
        # assess the claim without revealing protected information —
        # which means we need a separately-reviewed description field.
        # Until that field exists on document_card (scheduled as part
        # of the MVI-7 hardening pass), every row renders a locked-
        # down placeholder and carries a TBD basis so no one serves
        # this log as-is. That is the fail-closed posture legal work
        # requires. Codex adversarial #10 finding #1 / demo-breaker #10.
        for i, r in enumerate(rows, start=1):
            date = r["creation_date"] or r["effective_date"] or "—"
            author = r["author"] or r["sender"] or "—"
            recipient = r["recipient"] or "—"
            doc_type = r["doc_subtype"] or r["doc_type"] or "—"
            doc_id = (r["path"] or r["doc_id"] or "—")
            # Every row is TBD until a reviewed description field
            # exists. Counts are honest about that.
            tbd_count += 1
            basis_label = "TBD — attorney review required"
            description = (
                "[withheld — awaiting reviewed privilege description]"
            )
            body_lines.append(
                f"| {i} | {_md_cell(doc_id)} | {_md_cell(date)} "
                f"| {_md_cell(author)} | {_md_cell(recipient)} "
                f"| {_md_cell(doc_type)} | {basis_label} "
                f"| {description} |"
            )
        footer_lines = [
            "",
            f"**Total entries:** {len(rows)} "
            f"(all marked TBD — attorney review required before "
            f"service).",
            "",
            "> **⚠️ Do not serve this log without attorney review of "
            "every row.** Descriptions are intentionally withheld "
            "because the underlying `document_card.purpose` and "
            "`title` fields are LLM-authored and can contain "
            "privileged substance. A reviewed, privileged-content-"
            "free description field is on the roadmap; until it "
            "lands, every entry must be manually described before "
            "the log is served under Fed. R. Civ. P. 26(b)(5)(A)(ii).",
        ]
        rendered = (
            "## Privilege log\n\n"
            + header + "\n"
            + "\n".join(body_lines) + "\n"
            + "\n".join(footer_lines)
        )
        return DeliverableFamilyResult(
            intent="privilege_log",
            rendered_answer=rendered,
            row_count=len(rows),
            escalation_needed=False,
        )


def _md_cell(value: Any) -> str:
    """Sanitize a cell value for markdown table rendering — collapse
    pipes and newlines so a rogue doc title doesn't break the table."""
    text = str(value or "—").replace("|", "/").replace("\n", " ").strip()
    return text[:80] if len(text) > 80 else text
