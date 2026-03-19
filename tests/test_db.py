"""Tests for the database integration module."""

import sys
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from irys.db.base import Base
from irys.db.config import (
    DatabaseConfig,
    get_database_config,
    is_database_configured,
    try_get_database_config,
)
from irys.db.models import StoredDocument
from irys.db.repositories import DocumentRepository, DocumentUpsert
from irys.db.tools import run_document_smoke_test


class TestDatabaseConfig:
    """Tests for env-based database configuration."""

    def test_preview_config_from_env(self, monkeypatch):
        monkeypatch.setenv("IRYS_ENV", "preview")
        monkeypatch.setenv("IRYS_PREVIEW_DATABASE_URL", "postgres://preview-db/test")
        monkeypatch.setenv("IRYS_DB_ECHO", "true")

        config = DatabaseConfig.from_env()

        assert config.environment == "preview"
        assert config.url == "postgresql+psycopg://preview-db/test"
        assert config.echo is True

    def test_default_environment_is_production(self, monkeypatch):
        monkeypatch.delenv("IRYS_ENV", raising=False)
        monkeypatch.setenv("IRYS_PRODUCTION_DATABASE_URL", "postgres://prod-db/test")

        config = DatabaseConfig.from_env()

        assert config.environment == "production"
        assert config.url == "postgresql+psycopg://prod-db/test"

    def test_invalid_environment_raises(self, monkeypatch):
        monkeypatch.setenv("IRYS_ENV", "staging")

        try:
            DatabaseConfig.from_env()
        except ValueError as exc:
            assert "IRYS_ENV" in str(exc)
        else:
            raise AssertionError("Expected ValueError for invalid IRYS_ENV")

    def test_optional_config_helpers_do_not_raise_when_missing(self, monkeypatch):
        monkeypatch.delenv("IRYS_ENV", raising=False)
        monkeypatch.delenv("IRYS_PREVIEW_DATABASE_URL", raising=False)
        monkeypatch.delenv("IRYS_PRODUCTION_DATABASE_URL", raising=False)
        get_database_config.cache_clear()

        assert try_get_database_config() is None
        assert is_database_configured() is False

        get_database_config.cache_clear()


class TestDocumentRepository:
    """Tests for document persistence helpers."""

    def test_documents_table_uses_irys_prefix(self):
        assert StoredDocument.__tablename__ == "irys_rlm_documents"

    def test_updated_at_column_is_indexed(self):
        assert StoredDocument.__table__.c.updated_at.index is True

    def test_extracted_at_column_is_indexed(self):
        assert StoredDocument.__table__.c.extracted_at.index is True

    def test_upsert_inserts_and_updates_document(self):
        engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        repo = DocumentRepository()

        with session_factory() as session:
            created = repo.upsert(
                session,
                DocumentUpsert(
                    canonical_url="https://example.com/documents/contract.pdf",
                    source_url=(
                        "https://signed.example.com/documents/contract.pdf?token=abc"
                    ),
                    file_name="contract.pdf",
                    content_type="application/pdf",
                    source="upload",
                    page_count=2,
                    total_chars=22,
                    pages_json=[
                        {"page_num": 1, "text": "Initial page one"},
                        {"page_num": 2, "text": "Initial page two"},
                    ],
                    full_text="Initial extracted text",
                    extraction_status="completed",
                    extraction_version="v1",
                    metadata_json={"pages": 12},
                ),
            )
            session.commit()

        with session_factory() as session:
            updated = repo.upsert(
                session,
                DocumentUpsert(
                    canonical_url="https://example.com/documents/contract.pdf",
                    full_text="Updated extracted text",
                    checksum="abc123",
                ),
            )
            session.commit()

            assert updated.id == created.id
            assert updated.file_name == "contract.pdf"
            assert updated.canonical_url == "https://example.com/documents/contract.pdf"
            assert updated.source_url == "https://signed.example.com/documents/contract.pdf?token=abc"
            assert updated.full_text == "Updated extracted text"
            assert updated.checksum == "abc123"
            assert updated.page_count == 2

    def test_list_documents_returns_latest_updates_first(self):
        engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        repo = DocumentRepository()

        with session_factory() as session:
            repo.upsert(
                session,
                DocumentUpsert(canonical_url="https://example.com/a", file_name="a.pdf"),
            )
            session.commit()
            repo.upsert(
                session,
                DocumentUpsert(canonical_url="https://example.com/b", file_name="b.pdf"),
            )
            session.commit()

            documents = repo.list_documents(session)

            assert [document.canonical_url for document in documents] == [
                "https://example.com/b",
                "https://example.com/a",
            ]

    def test_upsert_defaults_source_url_to_canonical_url(self):
        engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        repo = DocumentRepository()

        with session_factory() as session:
            created = repo.upsert(
                session,
                DocumentUpsert(canonical_url="https://example.com/default-source"),
            )
            session.commit()

            assert created.source_url == "https://example.com/default-source"

    def test_run_document_smoke_test(self):
        engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)

        result = run_document_smoke_test(
            session_factory=session_factory,
            canonical_url="https://example.com/smoke-test",
            file_name="smoke-test.txt",
        )

        assert result.canonical_url == "https://example.com/smoke-test"
        assert result.source_url == "https://example.com/smoke-test"
        assert result.file_name == "smoke-test.txt"
        assert result.checksum == "smoke-test-checksum"
        assert result.extraction_status == "completed"
        assert "https://example.com/smoke-test" in result.listed_canonical_urls