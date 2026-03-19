"""Repository helpers for the documents table."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import StoredDocument


@dataclass(slots=True)
class DocumentUpsert:
    """Input payload for creating or updating a document row."""

    canonical_url: str
    source_url: str | None = None
    file_name: str | None = None
    content_type: str | None = None
    source: str | None = None
    checksum: str | None = None
    byte_size: int | None = None
    page_count: int | None = None
    total_chars: int | None = None
    pages_json: list[dict[str, Any]] | None = None
    full_text: str | None = None
    extraction_status: str | None = None
    extraction_version: str | None = None
    extracted_at: datetime | None = None
    fetch_verified_at: datetime | None = None
    metadata_json: dict[str, Any] | None = None

    def insert_values(self) -> dict[str, Any]:
        """Return values for inserting a new row."""
        return {
            "canonical_url": self.canonical_url,
            "source_url": self.source_url or self.canonical_url,
            "file_name": self.file_name,
            "content_type": self.content_type,
            "source": self.source,
            "checksum": self.checksum,
            "byte_size": self.byte_size,
            "page_count": self.page_count,
            "total_chars": self.total_chars,
            "pages_json": self.pages_json,
            "full_text": self.full_text,
            "extraction_status": self.extraction_status,
            "extraction_version": self.extraction_version,
            "extracted_at": self.extracted_at,
            "fetch_verified_at": self.fetch_verified_at,
            "metadata_json": self.metadata_json,
        }

    def update_values(self) -> dict[str, Any]:
        """Return explicitly provided values for updating an existing row."""
        return {
            "canonical_url": self.canonical_url,
            "source_url": self.source_url,
            "file_name": self.file_name,
            "content_type": self.content_type,
            "source": self.source,
            "checksum": self.checksum,
            "byte_size": self.byte_size,
            "page_count": self.page_count,
            "total_chars": self.total_chars,
            "pages_json": self.pages_json,
            "full_text": self.full_text,
            "extraction_status": self.extraction_status,
            "extraction_version": self.extraction_version,
            "extracted_at": self.extracted_at,
            "fetch_verified_at": self.fetch_verified_at,
            "metadata_json": self.metadata_json,
        }


class DocumentRepository:
    """CRUD helpers for persisted documents."""

    def get_by_canonical_url(self, session: Session, canonical_url: str) -> StoredDocument | None:
        """Return a document by its stable canonical URL."""
        return session.execute(
            select(StoredDocument).where(StoredDocument.canonical_url == canonical_url)
        ).scalar_one_or_none()

    def get_by_url(self, session: Session, url: str) -> StoredDocument | None:
        """Backward-compatible alias for canonical URL lookups."""
        return self.get_by_canonical_url(session, url)

    def list_documents(
        self,
        session: Session,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[StoredDocument]:
        """Return documents ordered by most recently updated first."""
        statement = (
            select(StoredDocument)
            .order_by(StoredDocument.updated_at.desc())
            .offset(offset)
            .limit(limit)
        )
        return list(session.execute(statement).scalars().all())

    def upsert(self, session: Session, payload: DocumentUpsert) -> StoredDocument:
        """Insert a document or update non-null fields on an existing row."""
        existing = self.get_by_canonical_url(session, payload.canonical_url)
        if existing is None:
            document = StoredDocument(**payload.insert_values())
            session.add(document)
            session.flush()
            return document

        updates = payload.update_values()
        for field_name, value in updates.items():
            if field_name == "canonical_url" or value is None:
                continue
            setattr(existing, field_name, value)

        session.add(existing)
        session.flush()
        return existing