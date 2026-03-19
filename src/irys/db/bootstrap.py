"""Helpers for creating and dropping database tables."""

from sqlalchemy.engine import Engine

from .base import Base
from .models import StoredDocument
from .session import get_engine


def create_all_tables(engine: Engine | None = None) -> None:
    """Create all registered database tables."""
    _ = StoredDocument
    Base.metadata.create_all(bind=engine or get_engine())


def drop_all_tables(engine: Engine | None = None) -> None:
    """Drop all registered database tables."""
    _ = StoredDocument
    Base.metadata.drop_all(bind=engine or get_engine())
