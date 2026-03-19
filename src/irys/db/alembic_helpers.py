"""Shared helpers for Alembic migration setup."""

from sqlalchemy.sql.schema import MetaData

from .base import Base
from .config import get_database_config
from .models import StoredDocument


def get_target_metadata() -> MetaData:
    """Return SQLAlchemy metadata for all registered DB models."""
    _ = StoredDocument
    return Base.metadata


def get_migration_database_url() -> str:
    """Return the configured database URL for Alembic migrations."""
    return get_database_config().url