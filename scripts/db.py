"""Database helper for migrations and document table smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

load_dotenv(ROOT / ".env")

from alembic import command
from alembic.config import Config

from irys.db import DocumentRepository, get_session_factory, run_document_smoke_test, session_scope


def _build_alembic_config() -> Config:
    return Config(str(ROOT / "alembic.ini"))


def cmd_upgrade(args: argparse.Namespace) -> None:
    command.upgrade(_build_alembic_config(), args.revision)


def cmd_downgrade(args: argparse.Namespace) -> None:
    command.downgrade(_build_alembic_config(), args.revision)


def cmd_current(_: argparse.Namespace) -> None:
    command.current(_build_alembic_config())


def cmd_history(_: argparse.Namespace) -> None:
    command.history(_build_alembic_config())


def cmd_revision(args: argparse.Namespace) -> None:
    command.revision(
        _build_alembic_config(),
        message=args.message,
        autogenerate=args.autogenerate,
    )


def cmd_smoke_test(args: argparse.Namespace) -> None:
    result = run_document_smoke_test(
        session_factory=get_session_factory(),
        canonical_url=args.url,
        source_url=args.source_url,
        file_name=args.file_name,
        content_type=args.content_type,
        source=args.source,
        list_limit=args.limit,
    )
    print(json.dumps(result.to_dict(), indent=2))


def cmd_get(args: argparse.Namespace) -> None:
    repo = DocumentRepository()
    with session_scope(get_session_factory()) as session:
        document = repo.get_by_canonical_url(session, args.url)
        if document is None:
            print(json.dumps({"found": False, "url": args.url}, indent=2))
            return

        print(
            json.dumps(
                {
                    "found": True,
                    "id": document.id,
                    "canonical_url": document.canonical_url,
                    "source_url": document.source_url,
                    "file_name": document.file_name,
                    "content_type": document.content_type,
                    "source": document.source,
                    "checksum": document.checksum,
                    "extraction_status": document.extraction_status,
                    "extraction_version": document.extraction_version,
                },
                indent=2,
            )
        )


def cmd_list(args: argparse.Namespace) -> None:
    repo = DocumentRepository()
    with session_scope(get_session_factory()) as session:
        documents = repo.list_documents(session, limit=args.limit)
        print(
            json.dumps(
                [
                    {
                        "id": document.id,
                        "canonical_url": document.canonical_url,
                        "source_url": document.source_url,
                        "file_name": document.file_name,
                        "source": document.source,
                        "extraction_status": document.extraction_status,
                    }
                    for document in documents
                ],
                indent=2,
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    upgrade_parser = subparsers.add_parser("upgrade", help="Apply Alembic migrations")
    upgrade_parser.add_argument("revision", nargs="?", default="head")
    upgrade_parser.set_defaults(func=cmd_upgrade)

    downgrade_parser = subparsers.add_parser("downgrade", help="Rollback Alembic migrations")
    downgrade_parser.add_argument("revision", nargs="?", default="-1")
    downgrade_parser.set_defaults(func=cmd_downgrade)

    current_parser = subparsers.add_parser("current", help="Show current DB revision")
    current_parser.set_defaults(func=cmd_current)

    history_parser = subparsers.add_parser("history", help="Show migration history")
    history_parser.set_defaults(func=cmd_history)

    revision_parser = subparsers.add_parser("revision", help="Create a new Alembic revision")
    revision_parser.add_argument("-m", "--message", required=True)
    revision_parser.add_argument("--autogenerate", action="store_true")
    revision_parser.set_defaults(func=cmd_revision)

    smoke_parser = subparsers.add_parser(
        "smoke-test",
        help="Insert/update/fetch/list a document row against the configured DB",
    )
    smoke_parser.add_argument(
        "--url",
        default="https://example.com/irys-db-smoke-test.txt",
    )
    smoke_parser.add_argument("--source-url")
    smoke_parser.add_argument("--file-name", default="irys-db-smoke-test.txt")
    smoke_parser.add_argument("--content-type", default="text/plain")
    smoke_parser.add_argument("--source", default="db-smoke-test")
    smoke_parser.add_argument("--limit", type=int, default=5)
    smoke_parser.set_defaults(func=cmd_smoke_test)

    get_parser = subparsers.add_parser("get", help="Fetch a document by url")
    get_parser.add_argument("--url", required=True)
    get_parser.set_defaults(func=cmd_get)

    list_parser = subparsers.add_parser("list", help="List recent documents")
    list_parser.add_argument("--limit", type=int, default=10)
    list_parser.set_defaults(func=cmd_list)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()