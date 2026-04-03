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
    DecisionContextStore, AuthorityStore, ProofStateStore,
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
        self.decision_context = DecisionContextStore(db, matter_id)
        self.authority = AuthorityStore(db, matter_id)
        self.proof_state = ProofStateStore(db, matter_id)

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

        # SO-2: top predicate_key values from the typed assertion graph so orientation
        # can generate SPO-aware search leads targeting known relationship types.
        pred_rows = self.db.execute(
            """SELECT predicate_key, COUNT(*) AS cnt
               FROM assertion
               WHERE matter_id=? AND predicate_key IS NOT NULL
               GROUP BY predicate_key
               ORDER BY cnt DESC
               LIMIT 10""",
            (self.matter_id,),
        ).fetchall()
        key_predicates = [r["predicate_key"] for r in pred_rows]

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
            key_predicates=key_predicates,
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

        For each conflict group found:
        1. Records an UNRESOLVED_CONTRADICTION gap (materiality 0.8).
        2. Wires bidirectional 'contradicts' assertion_links between the assertions
           that carry the conflicting quant facts (SO-6 → SO-2 propagation).
        3. Runs BeliefRevisionEngine on those assertions so they are marked DISPUTED
           in the assertion graph (truth maintenance).

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
        all_conflict_assertion_ids: list[str] = []

        for conflict in conflicts:
            subject = conflict.get("subject_type") or "unknown"
            currency = conflict.get("currency") or ""
            values = conflict.get("values", [])
            value_str = ", ".join(f"{v:,.2f}" for v in values[:5])
            desc = f"Conflicting {subject} amounts ({currency}): {value_str}"

            # Record gap (idempotent)
            if desc.lower() not in existing_descriptions:
                gap_id = self.record_gap(
                    description=desc,
                    gap_type=GapType.UNRESOLVED_CONTRADICTION,
                    materiality=0.8,
                )
                gap_ids.append(gap_id)

            # Collect assertion_ids for quant facts in this conflict group (SO-6→SO-2)
            subject_id = conflict.get("subject_id")
            rows = self.db.execute(
                """SELECT DISTINCT assertion_id
                   FROM quant_fact
                   WHERE matter_id=? AND quant_kind='amount'
                     AND subject_type=?
                     AND COALESCE(subject_id, '')=?
                     AND (currency=? OR (currency IS NULL AND ?=''))
                     AND assertion_id IS NOT NULL""",
                (self.matter_id, subject, subject_id or "", currency, currency),
            ).fetchall()
            aids = [r["assertion_id"] for r in rows]
            if len(aids) >= 2:
                # Wire bidirectional contradicts links (idempotent via UNIQUE index)
                for i, a1 in enumerate(aids):
                    for a2 in aids[i + 1:]:
                        self.assertions.link(a1, a2, AssertionLinkType.CONTRADICTS)
                        self.assertions.link(a2, a1, AssertionLinkType.CONTRADICTS)
                all_conflict_assertion_ids.extend(aids)

        # Run belief revision on all conflicting assertions so they become DISPUTED
        if all_conflict_assertion_ids:
            self.belief.apply(
                seed_assertion_ids=list(dict.fromkeys(all_conflict_assertion_ids)),
                cause=RevisionCause.CONFLICT_DETECTION,
                note="Automatic: conflicting amount values detected for same subject",
            )

        return gap_ids

    def reconcile(self, currency: str = "USD") -> dict:
        """Return reconciliation summary grouped by subject_type for a currency."""
        return self.quant.reconcile_by_subject(currency)

    def reconcile_payment_chain(self, currency: str = "USD") -> dict:
        """Return structured payment reconciliation: invoiced, paid, disputed, exposure.

        Satisfies SO-6: shows what was invoiced, paid, disputed, and the claimed
        exposure, grounded in source spans from the quant_fact store.
        """
        return self.quant.reconcile_payment_chain(currency)

    def reconcile_invoice_chain(self, currency: str = "USD") -> list:
        """Return per-invoice reconciliation rows (invoice_id, invoiced, paid, outstanding).

        Satisfies SO-6 per-invoice tracking requirement.
        """
        return self.quant.reconcile_invoice_chain(currency)

    # ------------------------------------------------------------------
    # Timeline view (SO-6, Priority 2 visual work product)
    # ------------------------------------------------------------------

    def get_timeline(self, limit: int = 200) -> list[dict]:
        """Return a chronological event list derived from quant dates and assertions.

        Each event has:
          date         — ISO date string or raw text when not parseable
          event        — human-readable description
          source_doc   — document the event came from
          quant_id     — quant_fact.id when sourced from quant store, else None
          assertion_id — assertion.id when sourced from assertion store, else None
          subject      — subject_type/subject_id when available
          kind         — 'date' | 'date_range' | 'temporal_assertion'

        Ordered by date ascending (None dates last).
        """
        events: list[dict] = []

        # Pull date-type quant facts.
        date_facts = self.quant.get_by_kind("date", limit=limit)
        date_range_facts = self.quant.get_by_kind("date_range", limit=limit)

        for qf in date_facts + date_range_facts:
            date_val = qf.get("date_value") or qf.get("date_end_value") or qf.get("raw_text", "")[:60]
            events.append({
                "date": date_val,
                "event": qf.get("raw_text", "")[:200],
                "source_doc": qf.get("span_id"),
                "quant_id": qf.get("id"),
                "assertion_id": qf.get("assertion_id"),
                "subject": qf.get("subject_id") or qf.get("subject_type"),
                "kind": qf.get("quant_kind", "date"),
            })

        # Pull assertions that have a temporal scope, with their document source.
        temporal_rows = self.db.execute(
            """SELECT a.id, a.proposition_text, a.temporal_scope_start,
                      MIN(ao.document_id) AS doc_id
               FROM assertion a
               LEFT JOIN assertion_occurrence ao ON ao.assertion_id = a.id
               WHERE a.matter_id=? AND a.temporal_scope_start IS NOT NULL
               GROUP BY a.id
               ORDER BY a.temporal_scope_start
               LIMIT ?""",
            (self.matter_id, limit),
        ).fetchall()

        for row in temporal_rows:
            date_val = row["temporal_scope_start"]
            desc = (row["proposition_text"] or "")[:200]
            events.append({
                "date": date_val,
                "event": desc,
                "source_doc": row["doc_id"],
                "quant_id": None,
                "assertion_id": row["id"],
                "subject": None,
                "kind": "temporal_assertion",
            })

        # Sort: events with parseable dates first, None/empty last.
        def _sort_key(e):
            d = e.get("date") or ""
            # ISO dates sort correctly as strings (YYYY-MM-DD).
            # Prefix None-like with "~" so they sort last.
            return d if d else "~"

        events.sort(key=_sort_key)
        return events[:limit]

    # ------------------------------------------------------------------
    # Evidence matrix (Priority 2 visual work product)
    # ------------------------------------------------------------------

    def get_evidence_matrix(self) -> dict:
        """Return a coverage matrix: issues × source documents.

        Structure:
        {
          "issues": [{"id": ..., "title": ..., "issue_type": ...}, ...],
          "sources": ["doc_a.pdf", "doc_b.pdf", ...],
          "cells": {
            issue_id: {
              doc_id: {"supporting": N, "attacking": N, "total": N}
            }
          },
          "issue_totals": {issue_id: {"supporting": N, "attacking": N}},
          "source_totals": {doc_id: {"supporting": N, "attacking": N}},
        }

        Only open issues with at least one assertion link are included.
        Only document sources with at least one assertion link are included.
        """
        open_issues = self.issues.get_open_issues(min_materiality=0.0)
        if not open_issues:
            return {"issues": [], "sources": [], "cells": {}, "issue_totals": {}, "source_totals": {}}

        # Query: assertion-issue links joined to occurrences to get document source.
        rows = self.db.execute(
            """SELECT ail.issue_id, ao.document_id, ail.relation_type, COUNT(*) AS cnt
               FROM assertion_issue_link ail
               JOIN issue i ON i.id = ail.issue_id
               JOIN assertion a ON a.id = ail.assertion_id
               JOIN assertion_occurrence ao ON ao.assertion_id = a.id
               WHERE i.matter_id=? AND i.status='open'
                 AND a.belief_state NOT IN ('superseded', 'withdrawn')
               GROUP BY ail.issue_id, ao.document_id, ail.relation_type""",
            (self.matter_id,),
        ).fetchall()

        # Build matrix.
        cells: dict = {}
        issue_totals: dict = {}
        source_totals: dict = {}
        sources_seen: set = set()
        issues_seen: set = set()

        for row in rows:
            iid = row["issue_id"]
            doc = row["document_id"] or "(unknown)"
            rel = row["relation_type"]
            cnt = row["cnt"]
            is_support = rel in ("supports", "establishes")
            is_attack = rel in ("attacks", "negates")

            issues_seen.add(iid)
            sources_seen.add(doc)

            if iid not in cells:
                cells[iid] = {}
            if doc not in cells[iid]:
                cells[iid][doc] = {"supporting": 0, "attacking": 0, "total": 0}
            if is_support:
                cells[iid][doc]["supporting"] += cnt
            elif is_attack:
                cells[iid][doc]["attacking"] += cnt
            cells[iid][doc]["total"] += cnt

            # Issue totals
            if iid not in issue_totals:
                issue_totals[iid] = {"supporting": 0, "attacking": 0}
            if is_support:
                issue_totals[iid]["supporting"] += cnt
            elif is_attack:
                issue_totals[iid]["attacking"] += cnt

            # Source totals
            if doc not in source_totals:
                source_totals[doc] = {"supporting": 0, "attacking": 0}
            if is_support:
                source_totals[doc]["supporting"] += cnt
            elif is_attack:
                source_totals[doc]["attacking"] += cnt

        issues_out = [
            {
                "id": i["id"],
                "title": i.get("title", ""),
                "issue_type": i.get("issue_type", ""),
            }
            for i in open_issues
            if i["id"] in issues_seen
        ]
        sources_out = sorted(sources_seen)

        return {
            "issues": issues_out,
            "sources": sources_out,
            "cells": cells,
            "issue_totals": issue_totals,
            "source_totals": source_totals,
        }

    # ------------------------------------------------------------------
    # Communication map (Priority 2 visual work product)
    # ------------------------------------------------------------------

    def get_communication_map(self) -> dict:
        """Return the actor-document interaction graph.

        Structure:
        {
          "actors": [{"id": ..., "name": ..., "actor_type": ..., "home_side": ...}],
          "documents": ["doc_a.pdf", "doc_b.pdf", ...],
          "actor_document_edges": [
            {"actor_id": ..., "actor_name": ..., "document_id": ..., "occurrence_count": N}
          ],
          "actor_actor_edges": [
            {"actor_a_id": ..., "actor_b_id": ..., "shared_documents": N, "documents": [...]}
          ],
        }

        actor_document_edges: actor X appeared in document Y in N assertion occurrences.
        actor_actor_edges: actors A and B both appeared in at least one common document.
        Only actors with at least one occurrence are included.
        """
        # Actor-document occurrence counts.
        occ_rows = self.db.execute(
            """SELECT ao.speaker_actor_id AS actor_id,
                      ao.document_id,
                      COUNT(*) AS cnt
               FROM assertion_occurrence ao
               JOIN actor ac ON ac.id = ao.speaker_actor_id
               WHERE ac.matter_id=? AND ao.speaker_actor_id IS NOT NULL
               GROUP BY ao.speaker_actor_id, ao.document_id""",
            (self.matter_id,),
        ).fetchall()

        actor_ids_seen: set = set()
        docs_seen: set = set()
        actor_doc_edges: list = []
        # actor_id → set of documents
        actor_docs: dict = {}

        # Build a lookup for actor names.
        actor_rows = self.db.execute(
            "SELECT id, canonical_name, actor_type, home_side FROM actor WHERE matter_id=?",
            (self.matter_id,),
        ).fetchall()
        actor_map = {r["id"]: dict(r) for r in actor_rows}

        for row in occ_rows:
            aid = row["actor_id"]
            doc = row["document_id"] or "(unknown)"
            actor_ids_seen.add(aid)
            docs_seen.add(doc)
            if aid not in actor_docs:
                actor_docs[aid] = set()
            actor_docs[aid].add(doc)
            actor_doc_edges.append({
                "actor_id": aid,
                "actor_name": actor_map.get(aid, {}).get("canonical_name", aid),
                "document_id": doc,
                "occurrence_count": row["cnt"],
            })

        # Build actor-actor co-appearance edges.
        actor_actor_edges: list = []
        actor_list = sorted(actor_ids_seen)
        for i, a1 in enumerate(actor_list):
            for a2 in actor_list[i + 1:]:
                shared = actor_docs.get(a1, set()) & actor_docs.get(a2, set())
                if shared:
                    actor_actor_edges.append({
                        "actor_a_id": a1,
                        "actor_b_id": a2,
                        "shared_documents": len(shared),
                        "documents": sorted(shared),
                    })

        actors_out = [
            {
                "id": aid,
                "name": actor_map[aid]["canonical_name"],
                "actor_type": actor_map[aid]["actor_type"],
                "home_side": actor_map[aid]["home_side"],
            }
            for aid in actor_ids_seen
            if aid in actor_map
        ]

        return {
            "actors": actors_out,
            "documents": sorted(docs_seen),
            "actor_document_edges": actor_doc_edges,
            "actor_actor_edges": actor_actor_edges,
        }

    # ------------------------------------------------------------------
    # Damages waterfall (Priority 2 visual work product)
    # ------------------------------------------------------------------

    def get_damages_waterfall(self, currency: str = "USD") -> list[dict]:
        """Return a structured damages breakdown by category (subject_type).

        Each entry:
          component       — subject_type label (e.g. "damages", "invoice", "fee")
          claimed_amount  — total of all amounts in this category
          source_count    — number of distinct quant_fact entries
          amounts         — list of {raw_text, amount_value, subject_id, assertion_id}
          conflicts       — list of conflicting amounts when multiple values exist
                            and the range is > 20% of the max value

        Ordered by claimed_amount descending (largest exposure first).
        Only 'amount'-kind quant facts for the given currency are included.
        Categories with subject_type IS NULL are grouped under "(uncategorised)".
        """
        rows = self.db.execute(
            """SELECT id, subject_type, subject_id, amount_value, raw_text, assertion_id
               FROM quant_fact
               WHERE matter_id=? AND quant_kind='amount'
                 AND (currency=? OR (currency IS NULL AND ?='USD'))
               ORDER BY subject_type, amount_value DESC""",
            (self.matter_id, currency, currency),
        ).fetchall()

        if not rows:
            return []

        # Group by subject_type.
        from collections import defaultdict
        groups: dict = defaultdict(list)
        for row in rows:
            key = row["subject_type"] or "(uncategorised)"
            groups[key].append({
                "raw_text": (row["raw_text"] or "")[:200],
                "amount_value": row["amount_value"],
                "subject_id": row["subject_id"],
                "assertion_id": row["assertion_id"],
            })

        waterfall = []
        for component, entries in groups.items():
            values = [e["amount_value"] for e in entries if e["amount_value"] is not None]
            total = sum(values) if values else 0.0
            max_val = max(values) if values else 0.0

            # Detect conflicts: multiple distinct amounts where spread > 20% of max.
            unique_vals = sorted(set(values), reverse=True)
            conflicts = []
            if len(unique_vals) >= 2 and max_val > 0:
                spread = unique_vals[0] - unique_vals[-1]
                if spread / max_val > 0.20:
                    conflicts = [f"${v:,.2f}" for v in unique_vals[:5]]

            waterfall.append({
                "component": component,
                "claimed_amount": round(total, 2),
                "source_count": len(entries),
                "currency": currency,
                "amounts": entries[:20],  # cap for readability
                "conflicts": conflicts,
            })

        waterfall.sort(key=lambda x: x["claimed_amount"], reverse=True)
        return waterfall

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

    def get_so_metrics(self) -> dict:
        """Compute measurable Sacred Outcome success criteria from stored state.

        Returns a snapshot of how well the current matter model satisfies the
        quantitative success criteria defined in the project CLAUDE.md.  Metrics
        that require ground truth or run telemetry (reuse_rate, gap_detection_recall,
        numeric_extraction_rate) are reported as None.

        Targets:
          assertion_structure_rate >= 1.0   (SO-2: 100% typed, full metadata)
          source_role_known_rate   >= 0.9   (SO-5: advocacy vs operative calibration)
          issue_coverage_avg       >= 0.8   (SO-4: issue-driven retrieval coverage)
          steerability             == True  (SO-3: user can interrupt mid-run)
          belief_revision          == True  (SO-2: corrections propagate)
        """
        # --- assertion structure: speech_act + source_role both set in occurrence ---
        ao_total_row = self.db.execute(
            "SELECT COUNT(*) AS n FROM assertion_occurrence WHERE assertion_id IN "
            "(SELECT id FROM assertion WHERE matter_id=?)",
            (self.matter_id,),
        ).fetchone()
        ao_total = int(ao_total_row["n"]) if ao_total_row else 0

        ao_typed_row = self.db.execute(
            "SELECT COUNT(*) AS n FROM assertion_occurrence WHERE assertion_id IN "
            "(SELECT id FROM assertion WHERE matter_id=?)"
            " AND speech_act IS NOT NULL AND speech_act != ''"
            " AND source_role IS NOT NULL AND source_role != ''",
            (self.matter_id,),
        ).fetchone()
        ao_typed = int(ao_typed_row["n"]) if ao_typed_row else 0

        assertion_structure_rate = (ao_typed / ao_total) if ao_total > 0 else None

        # --- source calibration: % occurrences with source_role != 'unknown' ---
        ao_known_row = self.db.execute(
            "SELECT COUNT(*) AS n FROM assertion_occurrence WHERE assertion_id IN "
            "(SELECT id FROM assertion WHERE matter_id=?)"
            " AND source_role IS NOT NULL AND source_role != '' AND source_role != 'unknown'",
            (self.matter_id,),
        ).fetchone()
        ao_known = int(ao_known_row["n"]) if ao_known_row else 0

        source_role_known_rate = (ao_known / ao_total) if ao_total > 0 else None

        # --- issue coverage ---
        issue_coverage_avg: "float | None" = None
        issues_with_proof_gap = 0
        try:
            coverage_report = self.get_issue_coverage_report()
            if coverage_report:
                fracs = [float(r.get("coverage_fraction", 0.0)) for r in coverage_report]
                issue_coverage_avg = round(sum(fracs) / len(fracs), 4) if fracs else None
                issues_with_proof_gap = sum(1 for r in coverage_report if r.get("has_proof_gap"))
        except Exception:
            pass

        assertion_count = self.assertions.count()
        issue_count = self.issues.count_open()
        open_gap_count = self.gaps.count_open()
        quant_fact_count = self.quant.count()
        actor_count = self.actors.count()

        targets = {
            "assertion_structure_rate": 1.0,
            "source_role_known_rate": 0.9,
            "issue_coverage_avg": 0.8,
            "steerability": True,
            "belief_revision": True,
        }

        def _pass(metric: str, value: "float | bool | None") -> "bool | None":
            if value is None:
                return None
            target = targets[metric]
            if isinstance(target, bool):
                return bool(value) == target
            return float(value) >= float(target)  # type: ignore[arg-type]

        return {
            "matter_id": self.matter_id,
            # Measurable SO metrics
            "assertion_structure_rate": assertion_structure_rate,
            "source_role_known_rate": source_role_known_rate,
            "issue_coverage_avg": issue_coverage_avg,
            "issues_with_proof_gap": issues_with_proof_gap,
            # SO-3/SO-2: requires run telemetry — not measurable from stored state alone
            "steerability": None,
            "belief_revision": None,
            # Requires run telemetry or ground truth — not yet measured
            "reuse_rate": None,
            "gap_detection_recall": None,
            "numeric_extraction_rate": None,
            # Raw counts
            "counts": {
                "assertions": assertion_count,
                "assertion_occurrences": ao_total,
                "issues_open": issue_count,
                "open_gaps": open_gap_count,
                "quant_facts": quant_fact_count,
                "actors": actor_count,
            },
            # Targets from CLAUDE.md
            "targets": targets,
            # Pass/fail per metric (None = not enough data to evaluate)
            "targets_met": {
                "assertion_structure_rate": _pass("assertion_structure_rate", assertion_structure_rate),
                "source_role_known_rate": _pass("source_role_known_rate", source_role_known_rate),
                "issue_coverage_avg": _pass("issue_coverage_avg", issue_coverage_avg),
                "steerability": None,   # requires run telemetry
                "belief_revision": None,  # requires run telemetry
            },
        }

    def __repr__(self) -> str:
        return f"MatterModel(matter_id={self.matter_id[:8]}..., db={self.db})"
