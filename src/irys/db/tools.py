"""Small DB utilities for smoke testing and verification."""

from dataclasses import asdict, dataclass

from sqlalchemy.orm import Session, sessionmaker

from .repositories import DocumentRepository, DocumentUpsert
from .session import get_session_factory, session_scope


@dataclass(slots=True)
class DocumentSmokeTestResult:
    """Result summary from a document persistence smoke test."""

    document_id: str
    url: str
    file_name: str | None
    checksum: str | None
    listed_urls: list[str]

    def to_dict(self) -> dict[str, str | None | list[str]]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def run_document_smoke_test(
    *,
    session_factory: sessionmaker[Session] | None = None,
    url: str = "https://example.com/irys-db-smoke-test.txt",
    file_name: str = "irys-db-smoke-test.txt",
    content_type: str = "text/plain",
    source: str = "db-smoke-test",
    list_limit: int = 5,
) -> DocumentSmokeTestResult:
    """Insert, update, fetch, and list a document to verify DB wiring."""
    factory = session_factory or get_session_factory()
    repo = DocumentRepository()

    with session_scope(factory) as session:
        repo.upsert(
            session,
            DocumentUpsert(
                url=url,
                file_name=file_name,
                content_type=content_type,
                source=source,
                extracted_text="smoke test insert",
                metadata_json={"kind": "smoke-test", "step": "create"},
            ),
        )

    with session_scope(factory) as session:
        updated = repo.upsert(
            session,
            DocumentUpsert(
                url=url,
                checksum="smoke-test-checksum",
                extracted_text="smoke test update",
                metadata_json={"kind": "smoke-test", "step": "update"},
            ),
        )
        fetched = repo.get_by_url(session, url)
        documents = repo.list_documents(session, limit=list_limit)

        if fetched is None:
            raise RuntimeError(f"Smoke test failed to fetch document by url: {url}")

        return DocumentSmokeTestResult(
            document_id=updated.id,
            url=fetched.url,
            file_name=fetched.file_name,
            checksum=fetched.checksum,
            listed_urls=[document.url for document in documents],
        )