"""Small DB utilities for smoke testing and verification."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from sqlalchemy.orm import Session, sessionmaker

from .repositories import DocumentRepository, DocumentUpsert
from .session import get_session_factory, session_scope


@dataclass(slots=True)
class DocumentSmokeTestResult:
    """Result summary from a document persistence smoke test."""

    document_id: str
    canonical_url: str
    source_url: str
    file_name: str | None
    checksum: str | None
    extraction_status: str | None
    listed_canonical_urls: list[str]

    def to_dict(self) -> dict[str, str | None | list[str]]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def run_document_smoke_test(
    *,
    session_factory: sessionmaker[Session] | None = None,
    canonical_url: str = "https://example.com/irys-db-smoke-test.txt",
    source_url: str | None = None,
    file_name: str = "irys-db-smoke-test.txt",
    content_type: str = "text/plain",
    source: str = "db-smoke-test",
    list_limit: int = 5,
) -> DocumentSmokeTestResult:
    """Insert, update, fetch, and list a document to verify DB wiring."""
    factory = session_factory or get_session_factory()
    repo = DocumentRepository()
    resolved_source_url = source_url or canonical_url
    extracted_at = datetime.now(timezone.utc)

    with session_scope(factory) as session:
        repo.upsert(
            session,
            DocumentUpsert(
                canonical_url=canonical_url,
                source_url=resolved_source_url,
                file_name=file_name,
                content_type=content_type,
                source=source,
                byte_size=17,
                page_count=1,
                total_chars=17,
                pages_json=[{"page_num": 1, "text": "smoke test insert"}],
                full_text="smoke test insert",
                extraction_status="completed",
                extraction_version="v1",
                extracted_at=extracted_at,
                fetch_verified_at=extracted_at,
                metadata_json={"kind": "smoke-test", "step": "create"},
            ),
        )

    with session_scope(factory) as session:
        updated = repo.upsert(
            session,
            DocumentUpsert(
                canonical_url=canonical_url,
                checksum="smoke-test-checksum",
                total_chars=17,
                full_text="smoke test update",
                metadata_json={"kind": "smoke-test", "step": "update"},
            ),
        )
        fetched = repo.get_by_canonical_url(session, canonical_url)
        documents = repo.list_documents(session, limit=list_limit)

        if fetched is None:
            raise RuntimeError(
                f"Smoke test failed to fetch document by canonical url: {canonical_url}"
            )

        return DocumentSmokeTestResult(
            document_id=updated.id,
            canonical_url=fetched.canonical_url,
            source_url=fetched.source_url,
            file_name=fetched.file_name,
            checksum=fetched.checksum,
            extraction_status=fetched.extraction_status,
            listed_canonical_urls=[document.canonical_url for document in documents],
        )