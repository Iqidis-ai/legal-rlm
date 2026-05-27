# Design: context-store-v2 deferred cleanup + merge unblock

**Date:** 2026-05-27  
**Branch:** `feat/context-store-v2`  
**Goal:** Resolve the two items blocking a clean merge into `feat/rlm-improvements`, then address all five deferred items called out in CONTEXT.md §7, so the branch is fully production-ready before the PR and quality analysis.

---

## Scope

### Merge blockers
1. `src/irys/service/api.py` — 3 call sites have the `accumulated_fact_texts()` fix in the working tree but unstaged.
2. `CONTEXT.md` — staged deletions (Section 2 extra branch rows, Section 5 weight table, Section 8 workstream rows) not yet committed.

### Deferred items (CONTEXT.md §7)
3. `_TIER_PROMOTE` / `_TIER_DEMOTE` class dicts defined on `FactStore` but `_check_tier` uses hard-coded literals.
4. `tick_decay` is an O(n) Python loop (full table scan + one UPDATE + one `_check_tier` per row).
5. `_build_synopsis` uses 3 facts at 80 chars — degrades BM25 q2 recall on large documents.
6. `on_re_extraction` is defined but never called — intent was to wire into `_read_document` when a scope is re-visited.
7. `EvidencePacker` leaves token budget unused when a source is sparse.

---

## Approach

Six commits, risk-ordered: blockers first, then pure refactors (tier dict, tick_decay), then behavioral changes (synopsis, on_re_extraction wiring, EvidencePacker top-up). Each commit is independently bisectable. Integration tests run after each behavioral commit.

---

## Commit 1 — Merge blockers

**Files:** `src/irys/service/api.py`, `CONTEXT.md`

Stage the three unstaged `accumulated_fact_texts()` call sites in `api.py` (lines 1024, 1312, 1538) and commit together with the already-staged `CONTEXT.md` deletions. No logic changes — purely committing existing working-tree state.

---

## Commit 2 — Tier dict cleanup (`fact_store.py`)

**Files:** `src/irys/core/fact_store.py`

Replace the 8 hard-coded threshold literals in `_check_tier` with lookups from the existing `_TIER_PROMOTE` / `_TIER_DEMOTE` class dicts. Pure refactor — identical logic, no behavior change.

```python
# _check_tier — after
if tier == "draft"      and importance >= self._TIER_PROMOTE["draft"]:      new_tier = "validated"
elif tier == "validated" and importance >= self._TIER_PROMOTE["validated"]:  new_tier = "core"
elif tier == "core"      and importance <  self._TIER_DEMOTE["core"]:        new_tier = "validated"
elif tier == "validated" and importance <  self._TIER_DEMOTE["validated"]:   new_tier = "draft"
```

**Testing:** No new tests needed — existing `test_fact_store_v2.py` covers tier transitions.

---

## Commit 3 — `tick_decay` SQL optimization (`fact_store.py`)

**Files:** `src/irys/core/fact_store.py`

Replace the O(n) Python loop with a single SQL `UPDATE` over all idle rows, then a second SELECT to collect affected hashes for tier checks. The tier-check loop is now bounded by rows with `recency_updated < today` rather than the full table.

```python
def tick_decay(self) -> int:
    self._conn.execute("""
        UPDATE facts
        SET importance = importance * POWER(0.995, CAST(
              julianday('now') - julianday(recency_updated) AS INTEGER))
        WHERE CAST(julianday('now') - julianday(recency_updated) AS INTEGER) > 0
    """)
    updated = self._conn.execute("SELECT changes()").fetchone()[0]
    if updated:
        changed = self._conn.execute("""
            SELECT content_hash FROM facts
            WHERE CAST(julianday('now') - julianday(recency_updated) AS INTEGER) > 0
        """).fetchall()
        for row in changed:
            self._check_tier(row["content_hash"])
    self._conn.commit()
    return updated
```

**Testing:** Existing decay tests in `test_fact_store_v2.py` verify correct importance values; no new tests needed unless the test harness explicitly checks loop behavior.

---

## Commit 4 — Synopsis expansion (`fact_store.py`)

**Files:** `src/irys/core/fact_store.py`

Two changes:

**4a. Expand `_build_synopsis`** from 3 facts × 80 chars to 10 facts × 120 chars, controlled by two class-level constants:

```python
_SYNOPSIS_FACT_COUNT = 10
_SYNOPSIS_FACT_CHARS = 120
```

**4b. Add `_refresh_synopsis(source)`** — called from `add_facts_from_extraction` after the `INSERT OR IGNORE` path. Counts existing `\n  - ` lines in the stored synopsis; if below `_SYNOPSIS_FACT_COUNT`, queries the top-importance facts for that source and rewrites the synopsis row via `UPDATE`.

```python
def _refresh_synopsis(self, source: str) -> None:
    existing = self.get_synopsis(source)
    if not existing:
        return
    fact_count = existing.count("\n  - ")
    if fact_count >= self._SYNOPSIS_FACT_COUNT:
        return
    rows = self._conn.execute(
        "SELECT fact FROM facts WHERE source = ? ORDER BY importance DESC LIMIT ?",
        (source, self._SYNOPSIS_FACT_COUNT)
    ).fetchall()
    if len(rows) <= fact_count:
        return
    facts = [r["fact"] for r in rows]
    sample_lines = "\n".join(f"  - {f[:self._SYNOPSIS_FACT_CHARS]}" for f in facts)
    synopsis = f"Source: {source}\nSample facts:\n{sample_lines}"
    token_count = len(synopsis.split())
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    self._conn.execute(
        "UPDATE source_synopses SET synopsis=?, token_count=?, updated_at=? WHERE source=?",
        (synopsis, token_count, now, source),
    )
    self._conn.commit()
```

`add_facts_from_extraction` block becomes:

```python
if facts:
    if not self.get_synopsis(source):
        self._build_synopsis(source, facts)
    else:
        self._refresh_synopsis(source)
```

**Testing:** Add test asserting that a synopsis built from 3 initial facts is refreshed to 10 after subsequent `add_facts_from_extraction` calls.

---

## Commit 5 — `on_re_extraction` wiring (`fact_store.py` + `engine.py`)

**Files:** `src/irys/core/fact_store.py`, `src/irys/rlm/engine.py`

**5a. Add `on_source_revisited(source)` to `FactStore`** — bulk UPDATE for all facts from a source (+5 importance, capped at 100), then tier checks on affected hashes. Returns count of updated rows.

```python
def on_source_revisited(self, source: str) -> int:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    self._conn.execute(
        """UPDATE facts
           SET importance = MIN(importance + 5, 100.0), recency_updated = ?
           WHERE source = ?""",
        (now, source),
    )
    updated = self._conn.execute("SELECT changes()").fetchone()[0]
    if updated:
        hashes = self._conn.execute(
            "SELECT content_hash FROM facts WHERE source = ?", (source,)
        ).fetchall()
        for row in hashes:
            self._check_tier(row["content_hash"])
    self._conn.commit()
    return updated
```

The existing per-hash `on_re_extraction(content_hash)` method is left in place — it remains available for future use.

**5b. Wire into `engine.py` `_read_document` early-return path** (currently line 2575):

```python
if cache.has_extracted(file_path, scope_key):
    if self.fact_store:
        filename = Path(file_path).name
        await asyncio.to_thread(self.fact_store.on_source_revisited, filename)
    return True
```

**Testing:** Add test asserting that calling `_read_document` on an already-extracted scope bumps importance for stored facts from that source.

---

## Commit 6 — EvidencePacker top-up pass (`evidence_packer.py`)

**Files:** `src/irys/core/evidence_packer.py`

After Pass 1 fills per-source budgets, if `total_chars < char_budget * 0.85`, run Pass 2: collect all unchosen facts across all sources, sort by density descending, and fill remaining budget without the per-source cap.

The 0.85 threshold ensures the top-up fires only when ≥15% of the budget is unused — sparse-source matters benefit, normal matters are unaffected.

```python
# Pass 2: top-up if budget has room
if total_chars < char_budget * 0.85:
    chosen_ids = {id(fact) for _, fact in selected}
    remaining = [
        (quick_score(f), f)
        for source_facts in by_source.values()
        for f in source_facts
        if id(f) not in chosen_ids
    ]
    remaining.sort(key=lambda x: _density(x[1], x[0]), reverse=True)
    for score, fact in remaining:
        if total_chars >= char_budget:
            break
        line = EvidencePacker._format_fact(fact)
        if total_chars + len(line) > char_budget:
            break
        selected.append((score, fact))
        total_chars += len(line)
```

**Testing:** Add test with a single-source fact set that triggers sparse budget and verify Pass 2 fills more facts than Pass 1 alone.

---

## File change summary

| File | Commits | Nature |
|---|---|---|
| `src/irys/service/api.py` | 1 | Stage existing fix |
| `CONTEXT.md` | 1 | Commit staged deletions |
| `src/irys/core/fact_store.py` | 2, 3, 4, 5 | Refactor + behavioral |
| `src/irys/rlm/engine.py` | 5 | Wiring |
| `src/irys/core/evidence_packer.py` | 6 | Behavioral |
| `tests/test_fact_store_v2.py` | 4, 5 | New test cases |
| `tests/test_*.py` (evidence packer) | 6 | New test case |

---

## Out of scope

- `on_re_extraction(content_hash)` per-hash method: kept, not deleted.
- Citation provenance verification (DELE-013): separate workstream.
- Classifier integration: downstream of Step 9 merge.
- OR-fallback stopword filtering, `_CHARS_PER_PAGE` calibration: separate workstreams.
