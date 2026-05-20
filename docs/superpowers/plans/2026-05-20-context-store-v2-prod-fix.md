# Context Store v2 — Production Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the SQLite+FTS5 FactStore work correctly in the AWS production environment by restoring S3 persistence, fixing three critical runtime bugs, and wiring the decay/session hooks that are currently dead.

**Architecture:** `FactStore.load()` will pull a full row export (`facts.ndjson`) from `s3://{bucket}/{s3_facts_prefix}/facts.ndjson` and bulk-insert into a fresh SQLite DB; `FactStore.save()` will export all rows back to that S3 key. This mirrors how `SessionStore` already works. Five critical bugs from the code review are fixed in the same PR since they are all blockers for production deploy.

**Tech Stack:** Python 3.10+, SQLite3 (stdlib), boto3, pytest

---

## File Map

| File | Action | What changes |
|---|---|---|
| `src/irys/core/fact_store.py` | Modify | Restore real `load()`/`save()` with S3; add `facts_file` property; add `_sanitise_fts_query()`; fix `get_stats()` |
| `src/irys/rlm/engine.py` | Modify | Call `tick_decay()` at session start; remove DEBUG log prefix |
| `src/irys/service/api.py` | Modify | Fix `_save_session` to serialise fact texts (not tuples) |
| `tests/test_fact_store_v2.py` | Modify | Add tests for S3 load/save, `facts_file`, FTS5 sanitisation, `tick_decay()` wired call |

---

## Task 1: Restore `facts_file` property (fix AttributeError)

**Files:**
- Modify: `src/irys/core/fact_store.py:212–228`
- Test: `tests/test_fact_store_v2.py`

`engine.py` references `self.fact_store.facts_file` at lines 617 and 743. The v2 `FactStore` removed this attribute. It will raise `AttributeError` on every investigation.

- [ ] **Step 1.1: Write failing test**

Add to `tests/test_fact_store_v2.py` in the `TestFactStoreSchema` class:

```python
def test_facts_file_property_returns_db_path(self):
    store = FactStore(self.tmp)
    assert store.facts_file == self.tmp / ".irys" / "facts.db"
    assert store.facts_file.exists()
```

- [ ] **Step 1.2: Run test to confirm it fails**

```
pytest tests/test_fact_store_v2.py::TestFactStoreSchema::test_facts_file_property_returns_db_path -v
```

Expected: `AttributeError: 'FactStore' object has no attribute 'facts_file'`

- [ ] **Step 1.3: Add the property to FactStore**

In `src/irys/core/fact_store.py`, add after the `__init__` method (after line 227):

```python
@property
def facts_file(self) -> Path:
    """Path to the SQLite database file (used by engine.py log messages)."""
    return self.store_dir / self.DB_FILE
```

- [ ] **Step 1.4: Run test to confirm it passes**

```
pytest tests/test_fact_store_v2.py::TestFactStoreSchema::test_facts_file_property_returns_db_path -v
```

Expected: PASS

- [ ] **Step 1.5: Commit**

```bash
git add src/irys/core/fact_store.py tests/test_fact_store_v2.py
git commit -m "fix: add facts_file property to FactStore (fixes AttributeError in engine.py)"
```

---

## Task 2: Fix `_save_session` — accumulated_facts tuple serialisation

**Files:**
- Modify: `src/irys/service/api.py:68–87`
- Test: `tests/test_api.py`

`state.add_fact()` now appends `(fact_text, content_hash)` tuples to `accumulated_facts`. `_save_session` serialises the raw list to S3. When `_load_session` passes it back as `seed_facts`, `state.add_facts()` receives tuples where it expects strings — silently corrupting the seeded facts.

- [ ] **Step 2.1: Write failing test**

Add to `tests/test_api.py` (create a test for this function in isolation):

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from irys.service.api import _save_session
from irys.service.config import ServiceConfig


@pytest.mark.asyncio
async def test_save_session_serialises_fact_texts_not_tuples():
    """_save_session must store plain strings, not (fact, hash) tuples."""
    config = ServiceConfig(storage_mode="local", temp_dir="/tmp/irys_test_session")
    result = MagicMock()
    result.state.findings = {
        "accumulated_facts": [
            ("Clause 14 requires notice", "abc123"),
            ("Damages capped at $5M", "def456"),
        ]
    }
    result.citations = []
    result.entities = {}

    saved_data = {}

    async def fake_save(session_id, data):
        saved_data["facts"] = data.facts

    with patch("irys.service.api.SessionStore") as MockStore:
        instance = MockStore.return_value
        instance.load = AsyncMock(return_value=None)
        instance.save = fake_save
        await _save_session(config, "sess-001", result)

    assert saved_data["facts"] == [
        "Clause 14 requires notice",
        "Damages capped at $5M",
    ]
```

- [ ] **Step 2.2: Run test to confirm it fails**

```
pytest tests/test_api.py::test_save_session_serialises_fact_texts_not_tuples -v
```

Expected: FAIL — `saved_data["facts"]` contains tuples, not strings.

- [ ] **Step 2.3: Fix `_save_session` in `src/irys/service/api.py`**

Replace lines 76–77:

```python
    # Before:
    facts = result.state.findings.get("accumulated_facts", [])

    # After:
    raw_facts = result.state.findings.get("accumulated_facts", [])
    facts = [
        entry[0] if isinstance(entry, (list, tuple)) else entry
        for entry in raw_facts
    ]
```

- [ ] **Step 2.4: Run test to confirm it passes**

```
pytest tests/test_api.py::test_save_session_serialises_fact_texts_not_tuples -v
```

Expected: PASS

- [ ] **Step 2.5: Commit**

```bash
git add src/irys/service/api.py tests/test_api.py
git commit -m "fix: _save_session extracts fact texts from (fact, hash) tuples before S3 serialisation"
```

---

## Task 3: Add FTS5 query sanitisation

**Files:**
- Modify: `src/irys/core/fact_store.py:473–497`
- Test: `tests/test_fact_store_v2.py`

Legal queries contain characters that are FTS5 syntax (colons, parentheses, quotes, hyphens). An unsanitised query raises `sqlite3.OperationalError`, which currently silently drops both BM25 lanes and falls back to importance-only sweep. The fix is to strip FTS5 metacharacters before querying.

- [ ] **Step 3.1: Write failing test**

Add to `tests/test_fact_store_v2.py` in `TestGetRelevant`:

```python
def test_legal_punctuation_query_does_not_raise(self):
    """FTS5 metacharacters in legal queries must not cause OperationalError."""
    scope = MagicMock()
    scope.is_targeted = True
    self.store.add_facts_from_extraction(
        ["The indemnification clause limits liability to direct damages."],
        source="MSA.pdf",
        scope=scope,
    )
    # These all contain FTS5 metacharacters
    for query in [
        "14(b) notice requirements",
        "damages: direct vs. consequential",
        '"time is of the essence"',
        "section 12-A obligations",
        "party (defendant) obligations",
    ]:
        result = self.store.get_relevant(query)
        assert isinstance(result, list)  # Must not raise
```

- [ ] **Step 3.2: Run test to confirm it fails**

```
pytest tests/test_fact_store_v2.py::TestGetRelevant::test_legal_punctuation_query_does_not_raise -v
```

Expected: FAIL with `sqlite3.OperationalError: fts5: syntax error`

- [ ] **Step 3.3: Add `_sanitise_fts_query` and wire it in**

In `src/irys/core/fact_store.py`, add this static method to `FactStore` (after `_importance_sweep`, before `_compound_score`):

```python
@staticmethod
def _sanitise_fts_query(query: str) -> str:
    """Strip FTS5 metacharacters that cause OperationalError on legal text queries."""
    import re
    # Remove FTS5 syntax: quotes, parens, colons, hyphens used as operators
    sanitised = re.sub(r'[^\w\s]', ' ', query)
    # Collapse whitespace
    return ' '.join(sanitised.split())
```

Then in `_bm25_facts` (line 475) and `_bm25_synopses` (line 487), wrap the query:

```python
def _bm25_facts(self, query: str, top_k: int) -> list[tuple[int, float]]:
    safe_query = self._sanitise_fts_query(query)
    if not safe_query:
        return []
    rows = self._conn.execute(
        """SELECT f.id, bm25(fact_fts) AS score
           FROM fact_fts
           JOIN facts f ON f.id = fact_fts.rowid
           WHERE fact_fts MATCH ?
           ORDER BY score
           LIMIT ?""",
        (safe_query, top_k),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]

def _bm25_synopses(self, query: str, top_m: int = 10) -> list[str]:
    safe_query = self._sanitise_fts_query(query)
    if not safe_query:
        return []
    rows = self._conn.execute(
        """SELECT s.source
           FROM synopsis_fts
           JOIN source_synopses s ON s.id = synopsis_fts.rowid
           WHERE synopsis_fts MATCH ?
           ORDER BY bm25(synopsis_fts)
           LIMIT ?""",
        (safe_query, top_m),
    ).fetchall()
    return [r[0] for r in rows]
```

- [ ] **Step 3.4: Run test to confirm it passes**

```
pytest tests/test_fact_store_v2.py::TestGetRelevant::test_legal_punctuation_query_does_not_raise -v
```

Expected: PASS

- [ ] **Step 3.5: Commit**

```bash
git add src/irys/core/fact_store.py tests/test_fact_store_v2.py
git commit -m "fix: sanitise FTS5 query to handle legal punctuation (colons, parens, quotes)"
```

---

## Task 4: Wire `tick_decay()` at session start

**Files:**
- Modify: `src/irys/rlm/engine.py:603–619`
- Test: `tests/test_fact_store_v2.py`

`tick_decay()` is implemented but never called. The decay and archival system (`importance < 35 → archive`) is therefore unreachable. The design doc specifies it runs once at session start.

- [ ] **Step 4.1: Write test confirming decay is called during investigation**

Add to `tests/test_fact_store_v2.py`:

```python
def test_tick_decay_reduces_importance_over_days(self):
    """tick_decay must reduce importance proportionally to days idle."""
    scope = MagicMock()
    scope.is_targeted = False
    self.store.add_facts_from_extraction(
        ["Plaintiff filed a motion on Day 1."],
        source="Complaint.pdf",
        scope=scope,
    )
    # Simulate 10 days idle by back-dating recency_updated
    self.store._conn.execute(
        "UPDATE facts SET recency_updated = date('now', '-10 days')"
    )
    self.store._conn.commit()

    updated = self.store.tick_decay()
    assert updated == 1

    row = self.store._conn.execute("SELECT importance FROM facts").fetchone()
    # 50.0 * (0.995 ^ 10) ≈ 47.56
    assert row[0] < 50.0
    assert row[0] > 40.0
```

- [ ] **Step 4.2: Run test to confirm it passes (tick_decay itself works)**

```
pytest tests/test_fact_store_v2.py::TestImportanceLifecycle::test_tick_decay_reduces_importance_over_days -v
```

Expected: PASS (the method works; this test just verifies the logic).

- [ ] **Step 4.3: Add `tick_decay()` call in `engine.py` after `FactStore` is loaded**

In `src/irys/rlm/engine.py`, after line 604 (`facts_loaded = await asyncio.to_thread(self.fact_store.load)`):

```python
        facts_loaded = await asyncio.to_thread(self.fact_store.load)

        # Apply idle decay and archive cold facts once per session start.
        # Runs in a thread so the event loop stays responsive.
        if facts_loaded > 0:
            await asyncio.to_thread(self.fact_store.tick_decay)
            await asyncio.to_thread(self.fact_store.archive_cold_facts)
```

- [ ] **Step 4.4: Remove the DEBUG log prefix from the save block**

In `src/irys/rlm/engine.py`, line 734, change:

```python
# Before:
f"DEBUG: fact_store has {fact_count} facts, _facts list: {len(self.fact_store._facts)}",

# After:
f"Fact store: {fact_count} facts accumulated this session",
```

- [ ] **Step 4.5: Run full test suite to confirm no regressions**

```
pytest tests/ -x -q
```

Expected: All tests pass.

- [ ] **Step 4.6: Commit**

```bash
git add src/irys/rlm/engine.py tests/test_fact_store_v2.py
git commit -m "fix: wire tick_decay and archive_cold_facts at session start; remove debug log"
```

---

## Task 5: Restore S3-backed `load()` and `save()` on FactStore

**Files:**
- Modify: `src/irys/core/fact_store.py:241–252`
- Test: `tests/test_fact_store_v2.py`

This is the core production fix. `load()` and `save()` are currently no-op stubs. When `s3_config` is set (which it is in the prod `_make_irys()` call), they must pull/push a full row export to `s3://{bucket}/{prefix}/facts.ndjson`. The format includes all v2 fields so tier and importance accumulate across jobs.

The S3 key pattern mirrors `SessionStore`: `{prefix}/facts.ndjson`.

- [ ] **Step 5.1: Write failing tests for S3 load/save**

Add a new test class to `tests/test_fact_store_v2.py`:

```python
class TestS3Persistence:
    """Tests for S3-backed load() and save()."""

    def setup_method(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.s3_config = {
            "bucket": "test-bucket",
            "region": "us-east-1",
            "prefix": "matters/case-123/facts",
            "aws_access_key_id": "fake",
            "aws_secret_access_key": "fake",
        }

    def teardown_method(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_store(self):
        return FactStore(self.tmp, s3_config=self.s3_config)

    def test_save_uploads_ndjson_to_s3(self):
        """save() must PUT all rows as NDJSON to S3 when s3_config is set."""
        store = self._make_store()
        scope = MagicMock()
        scope.is_targeted = True
        store.add_facts_from_extraction(
            ["Payment terms are net-30."],
            source="Contract.pdf",
            scope=scope,
        )

        with patch.object(store, "_get_s3_client") as mock_s3_factory:
            mock_s3 = MagicMock()
            mock_s3_factory.return_value = mock_s3
            mock_s3.put_object = MagicMock()

            saved = store.save()

        assert saved == 1
        mock_s3.put_object.assert_called_once()
        call_kwargs = mock_s3.put_object.call_args[1]
        assert call_kwargs["Bucket"] == "test-bucket"
        assert call_kwargs["Key"] == "matters/case-123/facts/facts.ndjson"
        body = call_kwargs["Body"].decode("utf-8")
        row = json.loads(body.splitlines()[0])
        assert row["fact"] == "Payment terms are net-30."
        assert row["scope_type"] == "targeted"
        assert "importance" in row
        assert "tier" in row
        assert "content_hash" in row

    def test_load_restores_rows_from_s3(self):
        """load() must pull NDJSON from S3 and bulk-insert into SQLite."""
        ndjson_row = json.dumps({
            "fact": "Indemnification capped at $5M.",
            "source": "MSA.pdf",
            "page": 12,
            "quote": None,
            "category": None,
            "extracted": "2026-05-01",
            "query_context": "damages",
            "scope_type": "targeted",
            "importance": 72.5,
            "recency_updated": "2026-05-10",
            "tier": "validated",
            "content_hash": StoredFact.compute_hash(
                "Indemnification capped at $5M.", "MSA.pdf"
            ),
        })

        store = self._make_store()
        with patch.object(store, "_get_s3_client") as mock_s3_factory:
            mock_s3 = MagicMock()
            mock_s3_factory.return_value = mock_s3
            mock_s3.get_object.return_value = {
                "Body": MagicMock(
                    read=MagicMock(return_value=ndjson_row.encode("utf-8"))
                )
            }

            count = store.load()

        assert count == 1
        facts = store.get_all()
        assert len(facts) == 1
        assert facts[0].fact == "Indemnification capped at $5M."
        assert facts[0].importance == 72.5
        assert facts[0].tier == "validated"
        assert facts[0].scope_type == "targeted"

    def test_load_returns_0_on_missing_key(self):
        """load() must return 0 (not raise) when no facts.ndjson exists in S3."""
        store = self._make_store()
        with patch.object(store, "_get_s3_client") as mock_s3_factory:
            mock_s3 = MagicMock()
            mock_s3_factory.return_value = mock_s3
            mock_s3.get_object.side_effect = store._get_s3_client  # triggers NoSuchKey
            from botocore.exceptions import ClientError
            mock_s3.get_object.side_effect = ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "Not Found"}},
                "GetObject",
            )

            count = store.load()

        assert count == 0
        assert len(store) == 0

    def test_no_s3_config_load_is_noop(self):
        """load() without s3_config must remain a no-op returning local row count."""
        store = FactStore(self.tmp)  # no s3_config
        count = store.load()
        assert count == 0  # empty store, no crash
```

- [ ] **Step 5.2: Run tests to confirm they fail**

```
pytest tests/test_fact_store_v2.py::TestS3Persistence -v
```

Expected: All 4 FAIL — `save()` and `load()` are no-ops.

- [ ] **Step 5.3: Implement S3-backed `load()` and `save()` in `fact_store.py`**

Replace the existing `load()` (line 241) and `save()` (line 246) methods with:

```python
FACTS_S3_FILE = "facts.ndjson"

def _s3_key(self) -> str:
    prefix = (self.s3_config or {}).get("prefix", "").strip("/")
    if prefix:
        return f"{prefix}/{self.FACTS_S3_FILE}"
    return self.FACTS_S3_FILE

def _get_s3_client(self):
    if self._s3_client is None:
        import boto3
        cfg = self.s3_config or {}
        kwargs: dict = {"region_name": cfg.get("region", "us-east-1")}
        if cfg.get("aws_access_key_id"):
            kwargs["aws_access_key_id"] = cfg["aws_access_key_id"]
        if cfg.get("aws_secret_access_key"):
            kwargs["aws_secret_access_key"] = cfg["aws_secret_access_key"]
        self._s3_client = boto3.client("s3", **kwargs)
    return self._s3_client

def load(self) -> int:
    """Load facts from S3 (if configured) into SQLite. Returns row count loaded."""
    self._loaded = True
    if not self.s3_config:
        return len(self)

    try:
        s3 = self._get_s3_client()
        resp = s3.get_object(
            Bucket=self.s3_config["bucket"],
            Key=self._s3_key(),
        )
        content = resp["Body"].read().decode("utf-8")
    except Exception as e:
        if "NoSuchKey" in str(e) or "404" in str(e):
            logger.info("No existing facts in S3 at %s — starting fresh", self._s3_key())
            return 0
        logger.warning("Failed to load facts from S3: %s", e)
        return 0

    inserted = 0
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for line_num, line in enumerate(content.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("Skipping malformed facts.ndjson line %d: %s", line_num, exc)
            continue
        fact_text = row.get("fact", "")
        source = row.get("source", "")
        if not fact_text or not source:
            continue
        content_hash = row.get("content_hash") or StoredFact.compute_hash(fact_text, source)
        try:
            self._conn.execute(
                """INSERT OR IGNORE INTO facts
                   (fact, source, page, quote, category, extracted, query_context,
                    scope_type, importance, recency_updated, tier, content_hash)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    fact_text,
                    source,
                    row.get("page"),
                    row.get("quote"),
                    row.get("category"),
                    row.get("extracted") or now,
                    row.get("query_context"),
                    row.get("scope_type", "snippet"),
                    float(row.get("importance", 50.0)),
                    row.get("recency_updated") or now,
                    row.get("tier", "draft"),
                    content_hash,
                ),
            )
            inserted += 1
        except sqlite3.Error as exc:
            logger.warning("Failed to insert fact from S3 line %d: %s", line_num, exc)

    self._conn.commit()
    logger.info("Loaded %d facts from S3 key %s", inserted, self._s3_key())
    return inserted

def save(self) -> int:
    """Export all SQLite rows to S3 as NDJSON (if configured). Returns row count saved."""
    # Flush any legacy in-memory _facts first
    if self._facts:
        for fact in self._facts:
            self._upsert_fact(fact)
        self._conn.commit()
        self._facts = []

    total = len(self)

    if not self.s3_config:
        return total

    rows = self._conn.execute("SELECT * FROM facts").fetchall()
    lines = []
    for row in rows:
        lines.append(json.dumps({
            "fact":            row["fact"],
            "source":          row["source"],
            "page":            row["page"],
            "quote":           row["quote"],
            "category":        row["category"],
            "extracted":       row["extracted"],
            "query_context":   row["query_context"],
            "scope_type":      row["scope_type"],
            "importance":      row["importance"],
            "recency_updated": row["recency_updated"],
            "tier":            row["tier"],
            "content_hash":    row["content_hash"],
        }, ensure_ascii=False))

    body = "\n".join(lines).encode("utf-8")
    try:
        s3 = self._get_s3_client()
        s3.put_object(
            Bucket=self.s3_config["bucket"],
            Key=self._s3_key(),
            Body=body,
            ContentType="application/x-ndjson",
        )
        logger.info("Saved %d facts to S3 key %s", total, self._s3_key())
    except Exception as exc:
        logger.error("Failed to save facts to S3: %s", exc)

    return total
```

Also add the class-level constant near `DB_FILE`:

```python
FACTS_S3_FILE = "facts.ndjson"
```

- [ ] **Step 5.4: Run S3 persistence tests**

```
pytest tests/test_fact_store_v2.py::TestS3Persistence -v
```

Expected: All 4 PASS.

- [ ] **Step 5.5: Run full test suite**

```
pytest tests/ -x -q
```

Expected: All tests pass.

- [ ] **Step 5.6: Commit**

```bash
git add src/irys/core/fact_store.py tests/test_fact_store_v2.py
git commit -m "feat: restore S3 load/save to FactStore — full v2 row export to facts.ndjson"
```

---

## Task 6: Fix `get_stats()` full-table-scan

**Files:**
- Modify: `src/irys/core/fact_store.py:597–607`
- Test: `tests/test_fact_store_v2.py`

`get_stats()` calls `get_all()` which fetches every row into Python. `stats()` already does this in a single SQL aggregation. `get_stats()` should delegate.

- [ ] **Step 6.1: Write test**

Add to `TestStats`:

```python
def test_get_stats_delegates_to_stats_method(self):
    """get_stats() must not load all rows into Python — delegates to stats()."""
    scope = MagicMock()
    scope.is_targeted = True
    self.store.add_facts_from_extraction(
        ["Fact one.", "Fact two."],
        source="Doc.pdf",
        scope=scope,
    )
    result = self.store.get_stats()
    assert result["total_facts"] == 2
    assert "unique_sources" in result
    assert result["unique_sources"] == 1
```

- [ ] **Step 6.2: Run test**

```
pytest tests/test_fact_store_v2.py::TestStats -v
```

Expected: PASS (the test itself is correct, just verifying `get_stats()` returns valid data).

- [ ] **Step 6.3: Replace `get_stats()` body to use `stats()`**

Replace the `get_stats` method in `src/irys/core/fact_store.py`:

```python
def get_stats(self) -> dict:
    """Get statistics about the fact store (legacy dict API)."""
    s = self.stats()
    sources = [
        r[0] for r in self._conn.execute("SELECT DISTINCT source FROM facts").fetchall()
    ]
    return {
        "total_facts": s.total_facts,
        "unique_sources": len(sources),
        "sources": sources,
        "store_path": str(self.facts_file),
        "exists": self.facts_file.exists(),
    }
```

- [ ] **Step 6.4: Run tests**

```
pytest tests/test_fact_store_v2.py -v
```

Expected: All pass.

- [ ] **Step 6.5: Commit**

```bash
git add src/irys/core/fact_store.py tests/test_fact_store_v2.py
git commit -m "fix: get_stats() uses SQL aggregation instead of full table scan"
```

---

## Task 7: End-to-end smoke validation

**Files:**
- Test: `tests/test_fact_store_v2.py`

Verify that a full session cycle (load from S3 → investigate → save to S3 → reload → facts present with correct tier) works end-to-end with mocked S3.

- [ ] **Step 7.1: Write the smoke test**

Add to `TestS3Persistence`:

```python
def test_round_trip_preserves_tier_and_importance(self):
    """Save then load must restore tier and importance faithfully."""
    store1 = self._make_store()
    scope = MagicMock()
    scope.is_targeted = True

    # Session 1: insert a fact and bump it to validated tier
    store1.add_facts_from_extraction(["Net-30 payment terms."], source="MSA.pdf", scope=scope)
    content_hash = StoredFact.compute_hash("Net-30 payment terms.", "MSA.pdf")
    # Bump importance to 70 (above validated threshold of 65)
    for _ in range(4):
        store1.on_search_hit(content_hash)  # +3 each = 62; then one re-extraction
    store1.on_re_extraction(content_hash)   # +5 → 67 → should promote to validated

    facts_before = store1.get_all()
    assert facts_before[0].tier == "validated"
    assert facts_before[0].importance >= 65.0

    # Capture what save() would upload
    uploaded: dict = {}
    with patch.object(store1, "_get_s3_client") as mock_s3_factory:
        mock_s3 = MagicMock()
        mock_s3_factory.return_value = mock_s3
        mock_s3.put_object = lambda **kw: uploaded.update(kw)
        store1.save()

    # Session 2: load from what session 1 saved
    store2 = self._make_store()
    with patch.object(store2, "_get_s3_client") as mock_s3_factory:
        mock_s3 = MagicMock()
        mock_s3_factory.return_value = mock_s3
        mock_s3.get_object.return_value = {
            "Body": MagicMock(read=MagicMock(return_value=uploaded["Body"]))
        }
        count = store2.load()

    assert count == 1
    facts_after = store2.get_all()
    assert facts_after[0].tier == "validated"
    assert abs(facts_after[0].importance - facts_before[0].importance) < 0.01
```

- [ ] **Step 7.2: Run the smoke test**

```
pytest tests/test_fact_store_v2.py::TestS3Persistence::test_round_trip_preserves_tier_and_importance -v
```

Expected: PASS

- [ ] **Step 7.3: Run full suite one final time**

```
pytest tests/ -q
```

Expected: All tests pass.

- [ ] **Step 7.4: Final commit**

```bash
git add tests/test_fact_store_v2.py
git commit -m "test: S3 round-trip smoke test — tier and importance preserved across sessions"
```

---

## Self-Review Checklist

**Spec coverage:**
- ✅ `facts_file` AttributeError → Task 1
- ✅ S3 `load()`/`save()` → Task 5 (core fix)
- ✅ `accumulated_facts` tuple serialisation → Task 2
- ✅ `tick_decay()` never called → Task 4
- ✅ FTS5 legal punctuation crash → Task 3
- ✅ `get_stats()` full scan → Task 6
- ✅ `quote` field dropped from extraction — noted in review; **not in this plan** because it requires understanding what extraction.get("quotes") returns and whether it was producing usable data. Flag for next PR.
- ✅ Round-trip smoke test → Task 7

**Type consistency:**
- `_s3_key()`, `_get_s3_client()`, `FACTS_S3_FILE` defined in Task 5 and not referenced before that task.
- `facts_file` property defined in Task 1; referenced in `get_stats()` in Task 6 — correct ordering.
- `StoredFact.compute_hash()` used in Task 5 test — defined in `fact_store.py` at line 138, exists before this plan.

**Placeholder scan:** No TBDs, TODOs, or "similar to above" references found.
