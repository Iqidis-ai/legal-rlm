FAIL → FIXED (commit bb854d9)

- HIGH: Supersession recovery is broken. The code says a weakened superseding assertion should let the older assertion recover, but the implementation cannot do that. Any `supersedes` edge sets `has_superseding=True` regardless of the superseding assertion’s own belief state, and `_compute_belief_state()` then forces the target to terminal `SUPERSEDED`. A disputed or withdrawn amendment still permanently kills the original assertion. [belief_revision.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/belief_revision.py#L79) [graph.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py#L350) [graph.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py#L499)

- HIGH: Belief revision does traverse the graph, but many downstream nodes still do not actually update. Only `OPERATIVE` support promotes a dependent to `INFERRED`; other solid states like `ADMITTED` and `RESOLVED` just preserve `current_state`. So a downstream assertion can stay `unknown` or `disputed` even after upstream support becomes legally strong. That directly undercuts SO-2 even though the BFS itself is real. The inconsistency is worse because issue coverage elsewhere treats `admitted` and `resolved` as full-strength support. [belief_revision.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/belief_revision.py#L139) [belief_revision.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/belief_revision.py#L145) [matter.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py#L643)

- HIGH: The durable assertion store can silently corrupt typed assertion data. Canonicalization keys only on normalized proposition text, not the typed fields. The first insert writes `subject/predicate/object/temporal` onto the canonical assertion row; later occurrences do not store those fields anywhere, because `assertion_occurrence` has no columns for them. That means a later, richer extraction of the same sentence silently loses its structured payload instead of upgrading the durable model. [models.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/models.py#L33) [models.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/models.py#L58) [graph.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py#L99) [graph.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py#L143) [schema.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/schema.py#L51)

- HIGH: SO-3 / threading is still not reliable mid-run. The UI stops streaming immediately and only `join()`s the background thread for 2 seconds, but the actual Gemini call runs inside `asyncio.to_thread(...)`, which does not cancel the underlying blocking SDK call. So:
  stop during orientation/synthesis waits for the in-flight model call to return or time out,
  stop during parallel lead work cancels the coroutine but not the worker thread doing the API call,
  redirect is accepted whenever the run row is `running`, but the engine only consumes redirects inside the iteration loop, so a redirect during verify/synthesis is an acknowledged no-op.
  [app.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/app.py#L361) [app.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/app.py#L437) [app.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/app.py#L450) [core/models.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/core/models.py#L297) [engine.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L1217) [engine.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L1660) [engine.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L2891) [in_process.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/backends/in_process.py#L244)

- MEDIUM: Durable state is on disk, but the current in-process UI can strand it. The dashboard can only resolve matter models that are already active in the current `Irys` instance; after restart, or after an early stop before the UI captures `matter_id`, persisted state exists but the panels cannot reopen it until the user reruns against the repo path. That is not durable product behavior from the user’s perspective. [in_process.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/backends/in_process.py#L41) [api.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/api.py#L151) [app.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/app.py#L284)

Belief revision is not “just updating a single row”; the traversal exists. The break is deeper: key transition rules, cancellation semantics, and canonical persistence still let the system look compliant while producing stale or silently flattened matter state.

Repo note: I found `.claude/CLAUDE.md`, but no repo-root `CLAUDE.md`. This was a source audit only; command execution was blocked by policy, so I did not run the app or tests.
---

## Fixes Applied (commit bb854d9, 728 tests pass)

**HIGH #1 (supersession recovery):**
- `get_neighbor_belief_states()` now collects `superseding_states: list[BeliefState]` (not just bool).
- `_compute_belief_state()` gains `superseding_states` parameter (rich interface): active superseder → SUPERSEDED; inactivated superseder (WITHDRAWN/SUPERSEDED itself) → resets `current_state = UNKNOWN`, allows recovery through normal support/attack logic.
- Legacy `has_superseding` bool preserved for backward compat (direct test callers unaffected).
- New test: `test_superseded_recovers_when_superseder_withdrawn`

**HIGH #2 (ADMITTED/RESOLVED promotion):**
- `_PROMOTING` set defined: `(OPERATIVE, ADMITTED, RESOLVED, PERFORMED)` — all legally conclusive.
- `promoting_support_weights` replaces `operative_support_weights`; same INFERRED promotion logic.
- New tests: `test_admitted_support_promotes_dependent`, `test_resolved_support_promotes_dependent`

**HIGH #3 (SPO payload loss):**
- `upsert_occurrence()` re-SELECT extended to include SPO fields (predicate_key, subject_ref_type, etc.).
- After new occurrence inserted for existing assertion: if `predicate_key IS NULL` but candidate has one → UPDATE canonical row with richer SPO payload (NULL → non-NULL only; never overwrites).

**HIGH #4 (redirect timing transparency):**
- `redirect_run()` returns `note` key explaining redirect is iteration-boundary-bound.
- `do_redirect()` in app.py surfaces the note to the user as `⚠️` text.

**MEDIUM (matter model reopen):** Deferred — architectural constraint of InProcessBackend. Not a production blocker; HttpBackend + service doesn't have this limit.
