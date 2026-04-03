"""MatterModel — the central facade for all matter intelligence stores.

Usage:
    model = MatterModel.open("path/to/repository", matter_name="Acme v TechServices")
    run_id = model.start_run("What are the key obligations?")
    assertion_id, is_new = model.record_assertion(candidate)
    model.ledger.append_event(run_id, LedgerEventType.ASSERTION_ADDED, ...)
    model.complete_run(run_id)
"""

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .db import SQLiteMatterDB
from .graph import (
    AssertionStore, GapStore, ActorStore, IssueStore, ClarificationStore, QuantStore,
    DocumentInventoryStore, ReasoningCacheStore, TrustOverrideStore, DocumentAnnotationStore,
)
from .reasoning import ReasoningLedgerStore
from .belief_revision import BeliefRevisionEngine
from .enums import (
    BeliefState, AssertionLinkType, RevisionCause,
    LedgerEventType, GapType,
)
from .models import (
    AssertionCandidate, AssertionRecord, RevisionResult,
    QueryMatterContext, RunSessionRecord,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id() -> str:
    return uuid.uuid4().hex


class MatterModel:
    """
    Central facade providing access to all matter intelligence stores.

    One MatterModel per matter, one SQLite DB per repository.

    The engine interacts exclusively through this facade — stores are
    implementation details. The facade is also the integration seam
    that the MatterRuntimeAdapter wraps to keep engine.py clean.
    """

    def __init__(self, db: SQLiteMatterDB, matter_id: str):
        self.db = db
        self.matter_id = matter_id
        # Sub-stores
        self.assertions = AssertionStore(db, matter_id)
        self.gaps = GapStore(db, matter_id)
        self.actors = ActorStore(db, matter_id)
        self.issues = IssueStore(db, matter_id)
        self.clarifications = ClarificationStore(db, matter_id)
        self.quant = QuantStore(db, matter_id)
        self.ledger = ReasoningLedgerStore(db, matter_id)
        self.belief = BeliefRevisionEngine(db, self.assertions)
        self.inventory = DocumentInventoryStore(db, matter_id)
        self.cache = ReasoningCacheStore(db, matter_id)
        self.trust_overrides = TrustOverrideStore(db, matter_id)
        self.annotations = DocumentAnnotationStore(db, matter_id)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def open(
        cls,
        repository_path: str | Path,
        matter_name: Optional[str] = None,
    ) -> "MatterModel":
        """
        Open (or create) a MatterModel for a repository path.

        If no matter exists for this repository, creates one.
        """
        db = SQLiteMatterDB.for_repository(repository_path)
        repo_str = str(Path(repository_path).resolve())

        # Look up or create matter record
        row = db.execute(
            "SELECT id FROM matter WHERE repository_root=?", (repo_str,)
        ).fetchone()

        if row is None:
            matter_id = _id()
            name = matter_name or Path(repository_path).name
            now = _now()
            with db.transaction():
                db.execute(
                    """INSERT INTO matter (id, name, repository_root, maturity, created_at, updated_at)
                       VALUES (?,?,?,?,?,?)""",
                    (matter_id, name, repo_str, "initial", now, now),
                )
        else:
            matter_id = row["id"]
            if matter_name:
                db.execute(
                    "UPDATE matter SET name=?, updated_at=? WHERE id=?",
                    (matter_name, _now(), matter_id),
                )

        return cls(db, matter_id)

    @classmethod
    def open_in_memory(cls, matter_name: str = "test_matter") -> "MatterModel":
        """Open an in-memory MatterModel for testing."""
        db = SQLiteMatterDB.in_memory()
        matter_id = _id()
        now = _now()
        with db.transaction():
            db.execute(
                """INSERT INTO matter (id, name, repository_root, maturity, created_at, updated_at)
                   VALUES (?,?,?,?,?,?)""",
                (matter_id, matter_name, ":memory:", "initial", now, now),
            )
        return cls(db, matter_id)

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(self, query: str, objective: Optional[str] = None) -> str:
        """Start a new investigation run. Returns run_id."""
        return self.ledger.start_run(query, objective)

    def complete_run(self, run_id: str, summary: Optional[str] = None) -> None:
        self.ledger.complete_run(run_id, summary)

    def fail_run(self, run_id: str, reason: str) -> None:
        self.ledger.fail_run(run_id, reason)

    def interrupt_run(self, run_id: str) -> None:
        self.ledger.interrupt_run(run_id)

    # ------------------------------------------------------------------
    # Assertion management
    # ------------------------------------------------------------------

    def record_assertion(
        self, candidate: AssertionCandidate
    ) -> tuple[str, bool]:
        """
        Upsert a canonical assertion and record an occurrence.

        Returns (assertion_id, is_new_assertion).
        """
        return self.assertions.upsert_occurrence(candidate)

    def link_assertions(
        self,
        src_id: str,
        dst_id: str,
        link_type: AssertionLinkType,
        weight: float = 1.0,
    ) -> str:
        return self.assertions.link(src_id, dst_id, link_type, weight)

    def apply_revision(
        self,
        seed_assertion_ids: list[str],
        cause: RevisionCause,
        run_id: Optional[str] = None,
        note: Optional[str] = None,
    ) -> list[RevisionResult]:
        """Trigger belief revision from seed assertions."""
        return self.belief.apply(seed_assertion_ids, cause, run_id, note)

    def correct_assertion(
        self,
        assertion_id: str,
        new_state: BeliefState,
        run_id: Optional[str] = None,
        note: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> RevisionResult:
        """Apply a user correction to an assertion and propagate."""
        if confidence is None:
            confidence_map = {
                BeliefState.OPERATIVE: 0.95,
                BeliefState.ADMITTED: 0.90,
                BeliefState.SUPERSEDED: 0.10,
                BeliefState.WITHDRAWN: 0.0,
                BeliefState.DISPUTED: 0.3,
            }
            confidence = confidence_map.get(new_state, 0.5)
        return self.belief.force_state(
            assertion_id=assertion_id,
            new_state=new_state,
            new_confidence=confidence,
            cause=RevisionCause.USER_CORRECTION,
            run_id=run_id,
            note=note,
        )

    # ------------------------------------------------------------------
    # Gap management
    # ------------------------------------------------------------------

    def record_gap(
        self,
        gap_type: GapType,
        description: str,
        expected_artifact: Optional[str] = None,
        materiality: float = 0.5,
        affected_type: Optional[str] = None,
        affected_id: Optional[str] = None,
    ) -> str:
        return self.gaps.record(
            gap_type, description, expected_artifact, materiality,
            affected_type=affected_type, affected_id=affected_id,
        )

    # ------------------------------------------------------------------
    # Query context (read at start of each run)
    # ------------------------------------------------------------------

    def build_query_context(self) -> QueryMatterContext:
        """
        Build a QueryMatterContext from current matter state.

        Called at the start of each investigation run to give the engine
        a snapshot of what is already known, enabling targeted retrieval.
        """
        row = self.db.execute(
            "SELECT id, name FROM matter WHERE id=?", (self.matter_id,)
        ).fetchone()
        matter_name = row["name"] if row else "unknown"

        assertion_count = self.assertions.count()
        open_gaps = self.gaps.open_gaps(min_materiality=0.3)
        open_issues = self.issues.get_open_issues(min_materiality=0.3)
        actor_count = self.actors.count()

        # Top actors by canonical name (limit 10 to keep context brief)
        known_actors = [
            a["canonical_name"]
            for a in self.actors.list_actors()[:10]
        ]

        # Documents already indexed in the assertion store
        rows = self.db.execute(
            """SELECT DISTINCT document_id FROM assertion_occurrence
               WHERE assertion_id IN (SELECT id FROM assertion WHERE matter_id=?)
               ORDER BY document_id LIMIT 20""",
            (self.matter_id,),
        ).fetchall()
        known_document_ids = [r["document_id"] for r in rows]

        # Find the weakest issue: the highest-priority issue with the least evidentiary support.
        # "Weakest" means most important AND least covered — where work will have most impact.
        # Priority = materiality × salience × (1 - coverage_fraction).
        # coverage_fraction = supporting_count / (supporting_count + 1) to avoid zero-division.
        weakest_issue_id = None
        if open_issues:
            # Get supporting-assertion count per open issue via JOIN — avoids IN-list
            # variable-count limits. Exclude non-active belief states so DISPUTED/
            # WITHDRAWN assertions don't overstate issue coverage (SO-2 correctness).
            support_rows = self.db.execute(
                """SELECT ail.issue_id, COUNT(*) AS cnt
                   FROM assertion_issue_link ail
                   JOIN issue i ON i.id=ail.issue_id
                   JOIN assertion a ON a.id=ail.assertion_id
                   WHERE i.matter_id=? AND i.status='open'
                     AND ail.relation_type IN ('supports','establishes')
                     AND a.belief_state NOT IN ('disputed','withdrawn','superseded')
                   GROUP BY ail.issue_id""",
                (self.matter_id,),
            ).fetchall()
            support_counts = {r["issue_id"]: r["cnt"] for r in support_rows}

            def _weakness(issue: dict) -> tuple:
                cnt = support_counts.get(issue["id"], 0)
                coverage = cnt / (cnt + 1.0)
                priority = issue["materiality"] * issue["salience"] * (1.0 - coverage)
                # Higher priority = higher weakness; negate for min()
                return (-priority, issue["id"])

            weakest = min(open_issues, key=_weakness)
            weakest_issue_id = weakest["id"]

        # Answered clarifications: inject user context into orientation
        answered_clarifications = self.clarifications.get_answered()

        # Document annotations: strategic notes from user (SO-3 annotation)
        document_annotations = self.annotations.list_recent(limit=10)

        return QueryMatterContext(
            matter_id=self.matter_id,
            matter_name=matter_name,
            open_gaps=open_gaps,
            open_issues=open_issues,
            existing_assertion_count=assertion_count,
            existing_actor_count=actor_count,
            known_actors=known_actors,
            known_document_ids=known_document_ids,
            answered_clarifications=answered_clarifications,
            document_annotations=document_annotations,
            weakest_issue_id=weakest_issue_id,
        )

    def get_issue_coverage_report(self) -> list[dict]:
        """Return per-issue evidence coverage for all open issues (SO-4).

        Each entry contains:
          - id, title, issue_type, materiality, salience
          - supporting_count: number of supporting/establishing assertion links
          - coverage_fraction: supporting_count / (supporting_count + 1), [0, 1)
          - has_proof_gap: True if an open MISSING_ISSUE_PREDICATE gap is linked
          - gap_id: id of that gap, or None

        Ordered by coverage_fraction ascending (weakest coverage first).
        """
        open_issues = self.issues.get_open_issues(min_materiality=0.0)
        if not open_issues:
            return []

        mid = self.matter_id

        # Use JOIN instead of IN-list to avoid SQLite variable-count limits (SO-4 scale).
        # Exclude non-active belief states so DISPUTED/WITHDRAWN assertions don't
        # inflate coverage (SO-2 correctness: revised beliefs must flow into coverage).
        support_rows = self.db.execute(
            """SELECT ail.issue_id, COUNT(*) AS cnt
               FROM assertion_issue_link ail
               JOIN issue i ON i.id = ail.issue_id
               JOIN assertion a ON a.id = ail.assertion_id
               WHERE i.matter_id=? AND i.status='open'
                 AND ail.relation_type IN ('supports','establishes')
                 AND a.belief_state NOT IN ('disputed','withdrawn','superseded')
               GROUP BY ail.issue_id""",
            (mid,),
        ).fetchall()
        support_counts = {r["issue_id"]: r["cnt"] for r in support_rows}

        proof_gap_rows = self.db.execute(
            """SELECT gl.affected_id AS issue_id, g.id AS gap_id
               FROM gap g
               JOIN gap_link gl ON gl.gap_id=g.id
               JOIN issue i ON i.id=gl.affected_id
               WHERE g.matter_id=? AND g.status='open'
                 AND g.gap_type='missing_issue_predicate'
                 AND gl.affected_type='issue'
                 AND i.matter_id=? AND i.status='open'""",
            (mid, mid),
        ).fetchall()
        proof_gaps = {r["issue_id"]: r["gap_id"] for r in proof_gap_rows}

        report = []
        for issue in open_issues:
            cnt = support_counts.get(issue["id"], 0)
            coverage = cnt / (cnt + 1.0)
            report.append({
                "id": issue["id"],
                "title": issue.get("title", ""),
                "issue_type": issue.get("issue_type", ""),
                "materiality": issue.get("materiality", 0.0),
                "salience": issue.get("salience", 0.0),
                "supporting_count": cnt,
                "coverage_fraction": round(coverage, 4),
                "has_proof_gap": issue["id"] in proof_gaps,
                "gap_id": proof_gaps.get(issue["id"]),
            })

        report.sort(key=lambda x: x["coverage_fraction"])
        return report

    # ------------------------------------------------------------------
    # Clarification engine (SO-7, SO-3)
    # ------------------------------------------------------------------

    def generate_clarifications_from_gaps(
        self,
        run_id: Optional[str] = None,
        top_n: int = 3,
        min_materiality: float = 0.5,
    ) -> list[str]:
        """
        Generate clarification questions for the highest-materiality open gaps.

        Called at the end of an investigation run. Returns list of new question_ids.
        Only generates questions for gaps that don't already have a pending question.
        """
        gaps = self.gaps.open_gaps(min_materiality=min_materiality)
        # Sort by materiality descending and take top N
        gaps = sorted(gaps, key=lambda g: g.get("materiality_score", 0), reverse=True)[:top_n]

        question_ids = []
        for gap in gaps:
            description = gap.get("description", "")
            if not description:
                continue

            # Look up what this gap is linked to (issue or assertion) so the impact
            # statement is specific rather than generic (SO-7).
            gap_id = gap.get("id")
            link_rows = self.gaps.db.execute(
                "SELECT affected_type, affected_id FROM gap_link WHERE gap_id=? LIMIT 1",
                (gap_id,),
            ).fetchall()
            link = link_rows[0] if link_rows else None
            materiality = gap.get("materiality_score", 0.5)
            materiality_label = "high" if materiality >= 0.7 else ("medium" if materiality >= 0.4 else "low")

            # Format question based on gap type
            gap_type = gap.get("gap_type", "")
            if gap_type == "missing_issue_predicate":
                # Proof gap: issue has zero supporting assertions — ask for evidence, not a doc
                question = f"{description}. Can you provide documents, testimony, or other evidence relevant to this issue?"
                why = "No supporting evidence was found for this legal issue in the current document set."
                impact = (
                    f"Providing supporting evidence ({materiality_label} materiality) will advance "
                    f"proof coverage for this claim element and may alter case strength assessment."
                )
            elif "document" in gap_type:
                question = f"We could not find the following in the repository: {description}. Do you have access to this document or information?"
                why = "This document was referenced in the matter but is not present in the repository."
                if link and link["affected_type"] == "issue":
                    impact = (
                        f"This document directly affects a tracked issue "
                        f"(materiality: {materiality_label}). "
                        f"Providing it will update the evidence coverage for that claim element."
                    )
                elif link and link["affected_type"] == "assertion":
                    impact = (
                        f"This document was cited by an existing extracted fact "
                        f"(materiality: {materiality_label}). "
                        f"Providing it may corroborate, contradict, or supersede that assertion."
                    )
                else:
                    impact = (
                        f"This document has {materiality_label} materiality to the current matter. "
                        f"If available, it could alter relevant factual findings or conclusions."
                    )
            else:
                question = f"We identified a gap: {description}. Can you provide any additional context or documentation?"
                why = "This information is needed to complete the analysis."
                if link:
                    impact = (
                        f"Filling this gap (materiality: {materiality_label}) will update "
                        f"the {link['affected_type']} it is linked to in the matter model."
                    )
                else:
                    impact = f"Providing this information ({materiality_label} materiality) will improve matter coverage."

            q_id = self.clarifications.add_question(
                question_text=question,
                why_it_matters=why,
                expected_impact=impact,
                gap_id=gap.get("id"),
                run_id=run_id,
            )
            question_ids.append(q_id)

        return question_ids

    # ------------------------------------------------------------------
    # Quantitative intelligence (SO-6 + SO-7)
    # ------------------------------------------------------------------

    def detect_quant_conflicts(self) -> list[str]:
        """
        Detect numeric conflicts: same subject_type+currency with divergent amounts.

        For each conflict group found, records an UNRESOLVED_CONTRADICTION gap with
        materiality 0.8 (high — amount conflicts are almost always significant).
        Returns list of new gap_ids created. Idempotent: skips conflicts whose gap
        description is already in the open gap store.
        """
        conflicts = self.quant.get_conflicts()
        if not conflicts:
            return []

        existing_descriptions = {
            g.get("description", "").lower()
            for g in self.gaps.open_gaps(min_materiality=0.0)
        }

        gap_ids = []
        for conflict in conflicts:
            subject = conflict.get("subject_type") or "unknown"
            currency = conflict.get("currency") or ""
            values = conflict.get("values", [])
            value_str = ", ".join(f"{v:,.2f}" for v in values[:5])
            desc = f"Conflicting {subject} amounts ({currency}): {value_str}"
            if desc.lower() in existing_descriptions:
                continue
            gap_id = self.record_gap(
                description=desc,
                gap_type=GapType.UNRESOLVED_CONTRADICTION,
                materiality=0.8,
            )
            gap_ids.append(gap_id)

        return gap_ids

    def reconcile(self, currency: str = "USD") -> dict:
        """Return reconciliation summary grouped by subject_type for a currency."""
        return self.quant.reconcile_by_subject(currency)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return a summary of matter model state."""
        return {
            "matter_id": self.matter_id,
            "assertion_count": self.assertions.count(),
            "open_gap_count": self.gaps.count_open(),
            "open_issue_count": self.issues.count_open(),
            "actor_count": self.actors.count(),
            "quant_fact_count": self.quant.count(),
            "pending_clarifications": self.clarifications.count_pending(),
            "recent_runs": len(self.ledger.recent_runs(limit=5)),
        }

    def __repr__(self) -> str:
        return f"MatterModel(matter_id={self.matter_id[:8]}..., db={self.db})"
