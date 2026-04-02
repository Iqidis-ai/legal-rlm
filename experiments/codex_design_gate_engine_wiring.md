**1. SO-1**

`build_query_context()` is explicitly a start-of-run snapshot in [matter.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py#L211), and `get_context()` exposes it in [runtime.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/runtime.py#L108). The clean place to call it is in `investigate()` immediately after the adapter is attached, before `_orient()` runs, at [engine.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L653), lines 653-665.

Minimal call-site change:
```py
state._matter_adapter = matter_adapter
matter_context = matter_adapter.get_context()

# Phase 1: Orientation
await self._orient(state, repo, matter_context)
```

Then inject the persisted-state snapshot into `ORIENTATION_PROMPT` between current lines 51 and 53 in [engine.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L51):
```py
Known Matter Context:
- Known facts: {known_facts_count}
- Open issues: {open_issues}
- Weakest issue: {weakest_issue_id}
- Known gaps: {known_gaps_count}
```

Use `open_issues` as a compact `id: title` string, not raw objects. Complexity: `M`.

**2. SO-4**

`weakest_issue_id` should not just be present in the context block; it should also change ranking behavior inside the `PRIORITIZE:` section at current lines 61-65 in [engine.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L61).

Prompt addition:
```py
- If `weakest_issue_id` is present, bias the first `initial_searches` toward resolving that issue before broad exploration.
```

That is the exact place where lead generation is shaped, so it is the right steering hook. Complexity: `S`.

**3. SO-3**

For `_investigate_lead()` at [engine.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L845), insert the stop check at lines 851-853, before semaphore acquisition.

Before:
```py
"""Investigate a single lead - may spawn sub-investigations."""
# Acquire semaphore to limit concurrent heavy operations
async with self._get_semaphore():
```

After:
```py
"""Investigate a single lead - may spawn sub-investigations."""
if (adapter := getattr(state, "_matter_adapter", None)) and adapter.is_stop_requested():
    state.mark_lead_investigated(lead.id, "Skipped - user requested stop")
    return
async with self._get_semaphore():
```

Complexity: `S`.

For `_deep_read_document()` at [engine.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L1006), insert the stop check at lines 1012-1015, before emit/read.

Before:
```py
"""Perform deep analysis of a document."""
self._emit_step(state, StepType.READING, f"Deep reading: {Path(file_path).name}")
try:
```

After:
```py
"""Perform deep analysis of a document."""
if (adapter := getattr(state, "_matter_adapter", None)) and adapter.is_stop_requested():
    return
self._emit_step(state, StepType.READING, f"Deep reading: {Path(file_path).name}")
try:
```

Complexity: `S`.

For post-loop behavior in `investigate()` at [engine.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L667), gate the expensive tail after lines 667-674.

Before:
```py
await self._investigate_loop(state, repo)

# Phase 2.5: Verify citations
await self._verify_citations(state, repo)
```

After:
```py
await self._investigate_loop(state, repo)
if matter_adapter is None or not matter_adapter.is_stop_requested():
    await self._verify_citations(state, repo)
    await self._synthesize(state)
```

Complexity: `M`.

Skipping synthesis entirely is the correct default. Partial synthesis on already-collected citations is useful only as an explicit follow-up action; doing it automatically on stop weakens SO-3 because the system keeps spending time after the user asked it to stop. I would skip both verification and synthesis on stop, preserve the accumulated state, and let a later explicit action synthesize from that saved state if needed.