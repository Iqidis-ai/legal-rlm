"""Engine and session factory helpers."""

from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .config import DatabaseConfig, get_database_config


def create_db_engine(config: DatabaseConfig | None = None) -> Engine:
    """Create a SQLAlchemy engine for the configured database."""
    resolved_config = config or get_database_config()
    return create_engine(
        resolved_config.url,
        echo=resolved_config.echo,
        pool_pre_ping=True,
    )


@lru_cache()
def get_engine() -> Engine:
    """Return a cached database engine."""
    return create_db_engine()


@lru_cache()
def get_session_factory() -> sessionmaker[Session]:
    """Return a cached SQLAlchemy session factory."""
    return sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)


@contextmanager
def session_scope(
    session_factory: sessionmaker[Session] | None = None,
) -> Iterator[Session]:
    """Provide a transactional session scope."""
    factory = session_factory or get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()