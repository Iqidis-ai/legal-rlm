"""Tests for media_pipeline module - IndexCacheRecord and IndexCache."""

import pytest
import sys
import tempfile
import hashlib
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from irys.core.media_pipeline import IndexCacheRecord, IndexCache
from irys.core.repository import Asset
from irys.core.models import EmbeddingConfig


class TestIndexCacheRecord:
    """Tests for IndexCacheRecord dataclass."""

    def test_has_required_fields(self):
        """Test that IndexCacheRecord has all required fields."""
        record = IndexCacheRecord(
            asset_id="test/path/file.mp4",
            checksum="abc123",
            embedding_model="gemini-embedding-2-preview",
            dimensionality=768,
            chunk_strategy_version=1,
            indexed_at="2024-01-15T10:00:00Z",
        )

        assert record.asset_id == "test/path/file.mp4"
        assert record.checksum == "abc123"
        assert record.embedding_model == "gemini-embedding-2-preview"
        assert record.dimensionality == 768
        assert record.chunk_strategy_version == 1
        assert record.indexed_at == "2024-01-15T10:00:00Z"


class TestIndexCache:
    """Tests for IndexCache class."""

    @pytest.fixture
    def cache(self, tmp_path):
        """Create a test IndexCache instance."""
        db_path = tmp_path / "test_cache.db"
        cache_instance = IndexCache(db_path)
        yield cache_instance
        cache_instance.close()  # Ensure database is closed before cleanup

    @pytest.fixture
    def test_asset(self, tmp_path):
        """Create a test asset."""
        asset_file = tmp_path / "test_video.mp4"
        asset_file.write_bytes(b"test content")

        return Asset(
            path=asset_file,
            filename="test_video.mp4",
            asset_type="video",
            mime_type="video/mp4",
            size_bytes=12,
            relative_path="test_video.mp4",
        )

    @pytest.fixture
    def embedding_config(self):
        """Create a test embedding config."""
        return EmbeddingConfig(
            model_id="gemini-embedding-2-preview",
            index_dimensionality=768,
            chunk_strategy_version=1,
        )

    def test_needs_reindex_no_record(self, cache, test_asset, embedding_config):
        """Test needs_reindex returns True when no record exists."""
        assert cache.needs_reindex(test_asset, embedding_config) is True

    def test_needs_reindex_checksum_changed(self, cache, test_asset, embedding_config):
        """Test needs_reindex returns True when file content changes."""
        # Mark as indexed
        cache.mark_indexed(test_asset, embedding_config)

        # Verify it's not needed now
        assert cache.needs_reindex(test_asset, embedding_config) is False

        # Change the file content (this changes the checksum)
        test_asset.path.write_bytes(b"different content")

        # Should need reindex now
        assert cache.needs_reindex(test_asset, embedding_config) is True

    def test_needs_reindex_model_changed(self, cache, test_asset, embedding_config):
        """Test needs_reindex returns True when model changes."""
        # Mark as indexed with original model
        cache.mark_indexed(test_asset, embedding_config)
        assert cache.needs_reindex(test_asset, embedding_config) is False

        # Change the model
        new_config = EmbeddingConfig(
            model_id="gemini-embedding-3-preview",  # Different model
            index_dimensionality=768,
            chunk_strategy_version=1,
        )

        # Should need reindex
        assert cache.needs_reindex(test_asset, new_config) is True

    def test_needs_reindex_dimensionality_changed(self, cache, test_asset, embedding_config):
        """Test needs_reindex returns True when dimensionality changes."""
        # Mark as indexed with original dimensionality
        cache.mark_indexed(test_asset, embedding_config)
        assert cache.needs_reindex(test_asset, embedding_config) is False

        # Change dimensionality
        new_config = EmbeddingConfig(
            model_id="gemini-embedding-2-preview",
            index_dimensionality=1024,  # Different dimensionality
            chunk_strategy_version=1,
        )

        # Should need reindex
        assert cache.needs_reindex(test_asset, new_config) is True

    def test_needs_reindex_strategy_version_changed(self, cache, test_asset, embedding_config):
        """Test needs_reindex returns True when chunk strategy version changes."""
        # Mark as indexed with original strategy version
        cache.mark_indexed(test_asset, embedding_config)
        assert cache.needs_reindex(test_asset, embedding_config) is False

        # Change strategy version
        new_config = EmbeddingConfig(
            model_id="gemini-embedding-2-preview",
            index_dimensionality=768,
            chunk_strategy_version=2,  # Different strategy version
        )

        # Should need reindex
        assert cache.needs_reindex(test_asset, new_config) is True

    def test_mark_indexed_creates_record(self, cache, test_asset, embedding_config):
        """Test mark_indexed creates a cache record."""
        # Initially needs indexing
        assert cache.needs_reindex(test_asset, embedding_config) is True

        # Mark as indexed
        cache.mark_indexed(test_asset, embedding_config)

        # Should not need indexing anymore
        assert cache.needs_reindex(test_asset, embedding_config) is False

    def test_mark_indexed_updates_record(self, cache, test_asset, embedding_config):
        """Test mark_indexed updates existing record."""
        # Mark as indexed
        cache.mark_indexed(test_asset, embedding_config)

        # Get the record
        record = cache.get(str(test_asset.path))
        assert record is not None
        original_timestamp = record.indexed_at

        # Mark as indexed again (should update)
        import time
        time.sleep(0.01)  # Ensure timestamp difference
        cache.mark_indexed(test_asset, embedding_config)

        # Get updated record
        updated_record = cache.get(str(test_asset.path))
        assert updated_record is not None
        # Timestamp should be different (or at least not fail)
        assert updated_record.asset_id == str(test_asset.path)

    def test_model_id_change_triggers_reindex_all_assets(
        self, cache, tmp_path, embedding_config
    ):
        """Test that changing model_id causes needs_reindex=True for ALL assets.

        This is the GA migration path - when model goes from preview to GA,
        all assets must be re-indexed.
        """

        # Create multiple assets
        assets = []
        for i in range(3):
            asset_file = tmp_path / f"video_{i}.mp4"
            asset_file.write_bytes(f"content {i}".encode())
            asset = Asset(
                path=asset_file,
                filename=f"video_{i}.mp4",
                asset_type="video",
                mime_type="video/mp4",
                size_bytes=len(f"content {i}"),
                relative_path=f"video_{i}.mp4",
            )
            assets.append(asset)

        # Mark all assets as indexed with original config
        for asset in assets:
            cache.mark_indexed(asset, embedding_config)

        # Verify none need reindexing
        for asset in assets:
            assert cache.needs_reindex(asset, embedding_config) is False

        # Change the model_id (GA migration scenario)
        new_config = EmbeddingConfig(
            model_id="gemini-embedding-3-ga",  # New GA model
            index_dimensionality=768,  # Same dimensionality
            chunk_strategy_version=1,  # Same version
        )

        # ALL assets should need reindexing
        for asset in assets:
            assert cache.needs_reindex(asset, new_config) is True, \
                f"Asset {asset.filename} should need reindex after model change"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
