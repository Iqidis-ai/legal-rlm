"""Repository helpers for the documents table."""

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import StoredDocument


@dataclass(slots=True)
class DocumentUpsert:
    """Input payload for creating or updating a document row."""

    url: str
    file_name: str | None = None
    content_type: str | None = None
    source: str | None = None
    checksum: str | None = None
    extracted_text: str | None = None
    metadata_json: dict[str, Any] | None = None

    def create_values(self) -> dict[str, Any]:
        """Return values for inserting a new row."""
        return {
            "url": self.url,
            "file_name": self.file_name,
            "content_type": self.content_type,
            "source": self.source,
            "checksum": self.checksum,
            "extracted_text": self.extracted_text,
            "metadata_json": self.metadata_json,
        }


class DocumentRepository:
    """CRUD helpers for persisted documents."""

    def get_by_url(self, session: Session, url: str) -> StoredDocument | None:
        """Return a document by its unique URL."""
        return session.execute(
            select(StoredDocument).where(StoredDocument.url == url)
        ).scalar_one_or_none()

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
        existing = self.get_by_url(session, payload.url)
        if existing is None:
            document = StoredDocument(**payload.create_values())
            session.add(document)
            session.flush()
            return document

        updates = payload.create_values()
        for field_name, value in updates.items():
            if field_name == "url" or value is None:
                continue
            setattr(existing, field_name, value)

        session.add(existing)
        session.flush()
        return existing