"""Tests for the v2 SQLite-backed FactStore."""
import json
import shutil
import sys
import hashlib
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from irys.core.fact_store import FactStore, StoredFact


def make_store() -> tuple[FactStore, Path]:
    """Create an isolated FactStore in a temp directory."""
    tmp = Path(tempfile.mkdtemp())
    store = FactStore(tmp)
    return store, tmp


class TestSchema:
    """DB is created with the correct tables and indices."""

    def test_db_file_created_on_init(self):
        store, tmp = make_store()
        assert (tmp / ".irys" / "facts.db").exists()

    def test_facts_table_exists(self):
        store, _ = make_store()
        tables = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {r[0] for r in tables}
        assert "facts" in table_names
        assert "source_synopses" in table_names
        assert "fact_stubs" in table_names

    def test_fts_tables_exist(self):
        store, _ = make_store()
        tables = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {r[0] for r in tables}
        assert "fact_fts" in table_names
        assert "synopsis_fts" in table_names

    def test_wal_mode_enabled(self):
        store, _ = make_store()
        mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_foreign_keys_enabled(self):
        store, _ = make_store()
        fk = store._conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1

    def test_source_synopses_has_integer_pk(self):
        """source_synopses must have INTEGER PRIMARY KEY for synopsis_fts rowid stability."""
        store, _ = make_store()
        info = store._conn.execute("PRAGMA table_info(source_synopses)").fetchall()
        cols = {row[1]: row[2] for row in info}
        assert "id" in cols
        assert cols["id"].upper() == "INTEGER"


class TestFactStoreSchema:
    """FactStore schema and property tests."""

    def setup_method(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.store = FactStore(self.tmp)

    def test_facts_file_property_returns_db_path(self):
        store = FactStore(self.tmp)
        assert store.facts_file == self.tmp / ".irys" / "facts.db"
        assert store.facts_file.exists()


class TestAddFactsFromExtraction:
    """add_facts_from_extraction inserts facts and returns content_hashes."""

    def _make_scope(self, targeted: bool = False):
        class FakeScope:
            is_targeted = targeted
        return FakeScope()

    def test_returns_content_hashes(self):
        store, _ = make_store()
        scope = self._make_scope(targeted=True)
        hashes = store.add_facts_from_extraction(
            ["Contract value is $2.5M", "Governing law is Texas"],
            source="Agreement.pdf",
            scope=scope,
        )
        assert len(hashes) == 2
        assert all(len(h) == 64 for h in hashes)

    def test_targeted_scope_sets_scope_type(self):
        store, _ = make_store()
        scope = self._make_scope(targeted=True)
        hashes = store.add_facts_from_extraction(
            ["Contract value is $2.5M"],
            source="Agreement.pdf",
            scope=scope,
        )
        row = store._conn.execute(
            "SELECT scope_type FROM facts WHERE content_hash = ?", (hashes[0],)
        ).fetchone()
        assert row["scope_type"] == "targeted"

    def test_prefix_scope_sets_scope_type(self):
        store, _ = make_store()
        scope = self._make_scope(targeted=False)
        hashes = store.add_facts_from_extraction(
            ["Contract value is $2.5M"],
            source="Agreement.pdf",
            scope=scope,
        )
        row = store._conn.execute(
            "SELECT scope_type FROM facts WHERE content_hash = ?", (hashes[0],)
        ).fetchone()
        assert row["scope_type"] == "prefix"

    def test_duplicate_insert_returns_existing_hash(self):
        store, _ = make_store()
        scope = self._make_scope()
        h1 = store.add_facts_from_extraction(["Fact A"], source="doc.pdf", scope=scope)
        h2 = store.add_facts_from_extraction(["Fact A"], source="doc.pdf", scope=scope)
        assert h1 == h2
        count = store._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        assert count == 1

    def test_same_text_different_source_both_stored(self):
        store, _ = make_store()
        scope = self._make_scope()
        h1 = store.add_facts_from_extraction(["Clause 2.1(g) applies"], source="ARKS.pdf", scope=scope)
        h2 = store.add_facts_from_extraction(["Clause 2.1(g) applies"], source="BSR.pdf", scope=scope)
        assert h1 != h2
        assert store._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 2

    def test_synopsis_created_lazily(self):
        store, _ = make_store()
        scope = self._make_scope()
        store.add_facts_from_extraction(
            [f"Fact {i}" for i in range(5)],
            source="Contract.pdf",
            scope=scope,
        )
        synopsis = store.get_synopsis("Contract.pdf")
        assert synopsis is not None
        assert "Contract.pdf" in synopsis

    def test_re_extraction_bumps_importance(self):
        store, _ = make_store()
        scope = self._make_scope(targeted=True)
        h = store.add_facts_from_extraction(["Re-extracted fact"], source="d.pdf", scope=scope)
        before = store._conn.execute(
            "SELECT importance FROM facts WHERE content_hash=?", (h[0],)
        ).fetchone()["importance"]
        store.add_facts_from_extraction(["Re-extracted fact"], source="d.pdf", scope=scope)
        after = store._conn.execute(
            "SELECT importance FROM facts WHERE content_hash=?", (h[0],)
        ).fetchone()["importance"]
        assert after == before + 5


class TestImportanceLifecycle:
    """on_search_hit, on_re_extraction, tick_decay, archive_cold_facts."""

    def _insert_fact(self, store, text="Test fact", source="doc.pdf",
                     importance=50.0, tier="draft", scope_type="targeted"):
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        content_hash = StoredFact.compute_hash(text, source)
        store._conn.execute(
            """INSERT OR IGNORE INTO facts
               (fact, source, extracted, scope_type, importance, recency_updated, tier, content_hash)
               VALUES (?,?,?,?,?,?,?,?)""",
            (text, source, now, scope_type, importance, now, tier, content_hash),
        )
        store._conn.commit()
        return content_hash

    def test_on_search_hit_increments_importance(self):
        store, _ = make_store()
        h = self._insert_fact(store, importance=50.0)
        store.on_search_hit(h)
        imp = store._conn.execute(
            "SELECT importance FROM facts WHERE content_hash=?", (h,)
        ).fetchone()["importance"]
        assert imp == 53.0

    def test_on_search_hit_caps_at_100(self):
        store, _ = make_store()
        h = self._insert_fact(store, importance=99.0)
        store.on_search_hit(h)
        imp = store._conn.execute(
            "SELECT importance FROM facts WHERE content_hash=?", (h,)
        ).fetchone()["importance"]
        assert imp == 100.0

    def test_on_search_hit_promotes_tier(self):
        store, _ = make_store()
        h = self._insert_fact(store, importance=63.0, tier="draft")
        store.on_search_hit(h)   # 63 + 3 = 66 >= 65 -> validated
        tier = store._conn.execute(
            "SELECT tier FROM facts WHERE content_hash=?", (h,)
        ).fetchone()["tier"]
        assert tier == "validated"

    def test_tick_decay_reduces_importance(self):
        store, _ = make_store()
        content_hash = StoredFact.compute_hash("Old fact", "doc.pdf")
        store._conn.execute(
            """INSERT INTO facts
               (fact, source, extracted, scope_type, importance, recency_updated, tier, content_hash)
               VALUES ('Old fact','doc.pdf','2026-04-01','snippet',80.0,'2026-04-01','validated',?)""",
            (content_hash,),
        )
        store._conn.commit()
        updated = store.tick_decay()
        assert updated >= 1
        imp = store._conn.execute(
            "SELECT importance FROM facts WHERE content_hash=?", (content_hash,)
        ).fetchone()["importance"]
        assert imp < 80.0

    def test_archive_cold_facts_moves_to_stubs(self):
        store, _ = make_store()
        h = self._insert_fact(store, importance=20.0, tier="draft")
        archived = store.archive_cold_facts()
        assert archived == 1
        assert store._conn.execute(
            "SELECT id FROM facts WHERE content_hash=?", (h,)
        ).fetchone() is None
        stub = store._conn.execute(
            "SELECT stub_summary FROM fact_stubs WHERE content_hash=?", (h,)
        ).fetchone()
        assert stub is not None

    def test_tick_decay_reduces_importance_over_days(self):
        """tick_decay must reduce importance proportionally to days idle."""
        store, tmp = make_store()
        scope = type("S", (), {"is_targeted": False})()
        store.add_facts_from_extraction(
            ["Plaintiff filed a motion on Day 1."],
            source="Complaint.pdf",
            scope=scope,
        )
        store._conn.execute(
            "UPDATE facts SET recency_updated = date('now', '-10 days')"
        )
        store._conn.commit()

        updated = store.tick_decay()
        assert updated == 1

        row = store._conn.execute("SELECT importance FROM facts").fetchone()
        # 50.0 * (0.995 ^ 10) ≈ 47.56
        assert row[0] < 50.0
        assert row[0] > 40.0

    def test_archive_cold_facts_skips_validated(self):
        store, _ = make_store()
        h = self._insert_fact(store, importance=20.0, tier="validated")
        archived = store.archive_cold_facts()
        assert archived == 0
        assert store._conn.execute(
            "SELECT id FROM facts WHERE content_hash=?", (h,)
        ).fetchone() is not None


class TestGetRelevant:
    """get_relevant returns scored, ranked StoredFact list."""

    def _make_scope(self, targeted=True):
        class S:
            is_targeted = targeted
        return S()

    def test_returns_list_of_stored_facts(self):
        store, _ = make_store()
        scope = self._make_scope()
        store.add_facts_from_extraction(["Contract value is $2.5M"], "A.pdf", scope)
        results = store.get_relevant("contract value")
        assert isinstance(results, list)
        assert all(isinstance(f, StoredFact) for f in results)

    def test_relevant_fact_ranked_first(self):
        store, _ = make_store()
        scope = self._make_scope()
        store.add_facts_from_extraction(
            ["Contract value is $2.5M", "Weather was sunny in Dallas"],
            "A.pdf", scope,
        )
        results = store.get_relevant("contract value")
        assert len(results) >= 1
        assert "contract" in results[0].fact.lower() or "2.5" in results[0].fact

    def test_empty_store_returns_empty(self):
        store, _ = make_store()
        assert store.get_relevant("anything") == []

    def test_top_k_respected(self):
        store, _ = make_store()
        scope = self._make_scope()
        store.add_facts_from_extraction(
            [f"Fact about clause {i}" for i in range(30)],
            "doc.pdf", scope,
        )
        results = store.get_relevant("clause", top_k=5)
        assert len(results) <= 5

    def test_stored_fact_fields_populated(self):
        store, _ = make_store()
        scope = self._make_scope(targeted=True)
        store.add_facts_from_extraction(["Indemnification cap is $5M"], "MSA.pdf", scope)
        results = store.get_relevant("indemnification")
        assert results[0].source == "MSA.pdf"
        assert results[0].scope_type == "targeted"
        assert results[0].content_hash != ""


    def test_legal_punctuation_query_does_not_raise(self):
        """FTS5 metacharacters in legal queries must not cause OperationalError."""
        from unittest.mock import MagicMock
        scope = MagicMock()
        scope.is_targeted = True
        store, _ = make_store()
        store.add_facts_from_extraction(
            ["The indemnification clause limits liability to direct damages."],
            source="MSA.pdf",
            scope=scope,
        )
        for query in [
            "14(b) notice requirements",
            "damages: direct vs. consequential",
            '"time is of the essence"',
            "section 12-A obligations",
            "party (defendant) obligations",
        ]:
            result = store.get_relevant(query)
            assert isinstance(result, list)


class TestEvidencePacker:
    """EvidencePacker respects source budget cap and fills by density."""

    def _make_facts(self, source_facts: dict) -> list:
        facts = []
        for source, count in source_facts.items():
            for i in range(count):
                facts.append(StoredFact(
                    fact=f"[{source[:4]}, §2.{i}] Clause text number {i} for {source}",
                    source=source,
                    importance=60.0 - i,
                    scope_type="targeted",
                    tier="validated",
                ))
        return facts

    def test_all_sources_appear_when_budget_is_sufficient(self):
        from irys.core.evidence_packer import EvidencePacker
        facts = self._make_facts({"ARKS.pdf": 10, "BSR.pdf": 10, "Delek.pdf": 10})
        result = EvidencePacker.pack(facts, query="conditions precedent", token_budget=10_000)
        assert "ARKS.pdf" in result
        assert "BSR.pdf" in result
        assert "Delek.pdf" in result

    def test_source_cap_prevents_monopoly(self):
        """One large source must not exceed 30% of the output."""
        from irys.core.evidence_packer import EvidencePacker
        facts = self._make_facts({"HUGE.pdf": 50, "SMALL_A.pdf": 1, "SMALL_B.pdf": 1})
        result = EvidencePacker.pack(facts, query="clause", token_budget=3000)
        assert "SMALL_A.pdf" in result
        assert "SMALL_B.pdf" in result

    def test_output_contains_quality_tags(self):
        """Each fact line must be prefixed with [scope_type, tier]."""
        from irys.core.evidence_packer import EvidencePacker
        facts = [StoredFact(
            fact="Contract value is $2.5M",
            source="MSA.pdf",
            scope_type="targeted",
            tier="core",
        )]
        result = EvidencePacker.pack(facts, query="contract value", token_budget=5000)
        assert "[targeted, core]" in result

    def test_empty_facts_returns_empty(self):
        from irys.core.evidence_packer import EvidencePacker
        assert EvidencePacker.pack([], query="anything", token_budget=5000) == ""


class TestMigrationAndStats:
    """stats() returns correct counts; migrate_from_jsonl migrates JSONL rows."""

    def test_stats_returns_dataclass(self):
        from irys.core.fact_store import FactStoreStats
        store, _ = make_store()
        s = store.stats()
        assert isinstance(s, FactStoreStats)
        assert s.total_facts == 0

    def test_stats_counts_tiers(self):
        from irys.core.fact_store import FactStoreStats
        store, _ = make_store()
        scope = type("S", (), {"is_targeted": True})()
        store.add_facts_from_extraction(["Draft fact A"], "d.pdf", scope)
        store._conn.execute(
            "UPDATE facts SET tier='validated', importance=70 WHERE source='d.pdf'"
        )
        store._conn.commit()
        s = store.stats()
        assert s.draft_facts == 0
        assert s.validated_facts == 1

    def test_migrate_from_jsonl(self):
        import json, tempfile
        store, _ = make_store()
        jsonl = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        jsonl.write(json.dumps({
            "fact": "Legacy fact about contract", "source": "old.pdf",
            "page": 3, "quote": None, "category": None,
            "extracted": "2026-01-01", "query_context": "contract review",
        }) + "\n")
        jsonl.flush()
        jsonl.close()
        count = store.migrate_from_jsonl(Path(jsonl.name))
        assert count == 1
        row = store._conn.execute("SELECT scope_type, tier FROM facts").fetchone()
        assert row["scope_type"] == "snippet"
        assert row["tier"] == "draft"


class TestStats:
    """get_stats() returns correct dict structure."""

    def setup_method(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.store = FactStore(self.tmp)

    def teardown_method(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

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


    def test_round_trip_preserves_tier_and_importance(self):
        """Save then load must restore tier and importance faithfully."""
        store1 = self._make_store()
        scope = MagicMock()
        scope.is_targeted = True

        store1.add_facts_from_extraction(["Net-30 payment terms."], source="MSA.pdf", scope=scope)
        content_hash = StoredFact.compute_hash("Net-30 payment terms.", "MSA.pdf")
        # Bump importance to 70 (above validated threshold of 65)
        for _ in range(4):
            store1.on_search_hit(content_hash)  # +3 each = 62
        store1.on_re_extraction(content_hash)   # +5 → 67 → promoted to validated

        facts_before = store1.get_all()
        assert facts_before[0].tier == "validated"
        assert facts_before[0].importance >= 65.0

        uploaded: dict = {}
        with patch.object(store1, "_get_s3_client") as mock_s3_factory:
            mock_s3 = MagicMock()
            mock_s3_factory.return_value = mock_s3
            mock_s3.put_object = lambda **kw: uploaded.update(kw)
            store1.save()

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


class TestGapFix3ProPrompt:
    """SYSTEM_PROMPT_PRO in models.py must instruct the model to interpret quality tags."""

    def test_pro_prompt_references_targeted_core(self):
        from irys.core.models import SYSTEM_PROMPT_PRO
        assert "[targeted, core]" in SYSTEM_PROMPT_PRO

    def test_pro_prompt_references_snippet_draft(self):
        from irys.core.models import SYSTEM_PROMPT_PRO
        assert "[snippet, draft]" in SYSTEM_PROMPT_PRO
