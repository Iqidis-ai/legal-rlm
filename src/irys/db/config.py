"""Database configuration and environment-based URL resolution."""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, cast

DatabaseEnvironment = Literal["preview", "production"]


def normalize_database_url(url: str) -> str:
    """Normalize a Postgres URL into a SQLAlchemy-compatible psycopg URL."""
    normalized = url.strip()
    if normalized.startswith("postgres://"):
        return "postgresql+psycopg://" + normalized[len("postgres://"):]
    if normalized.startswith("postgresql://"):
        return "postgresql+psycopg://" + normalized[len("postgresql://"):]
    return normalized


@dataclass(frozen=True, slots=True)
class DatabaseConfig:
    """Minimal database configuration for preview/production environments."""

    environment: DatabaseEnvironment
    url: str
    echo: bool = False

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Load configuration from environment variables."""
        environment_raw = os.getenv("IRYS_ENV", "production").strip().lower()
        if environment_raw not in {"preview", "production"}:
            raise ValueError(
                "IRYS_ENV must be either 'preview' or 'production'"
            )
        environment = cast(DatabaseEnvironment, environment_raw)

        url_env_key = (
            "IRYS_PREVIEW_DATABASE_URL"
            if environment == "preview"
            else "IRYS_PRODUCTION_DATABASE_URL"
        )
        database_url = os.getenv(url_env_key, "").strip()
        if not database_url:
            raise ValueError(
                f"Database URL not configured. Please set {url_env_key}."
            )

        return cls(
            environment=environment,
            url=normalize_database_url(database_url),
            echo=os.getenv("IRYS_DB_ECHO", "false").strip().lower() == "true",
        )


@lru_cache()
def get_database_config() -> DatabaseConfig:
    """Return the cached database configuration."""
    return DatabaseConfig.from_env()


def try_get_database_config() -> DatabaseConfig | None:
    """Return database config if available, otherwise None.

    This is useful for future service integration where database support
    should remain optional and must not crash startup when env vars are absent.
    """
    try:
        return get_database_config()
    except ValueError:
        return None


def is_database_configured() -> bool:
    """Return whether a valid database configuration is available."""
    return try_get_database_config() is not None