"""Irys RLM Production Service.

FastAPI-based REST API for document investigation with S3 integration.
Designed for lightweight deployment on small instances.
"""

# Lazy imports to avoid circular import issues
# (api.py imports chat_app which imports service.config)
__all__ = [
    "app",
    "create_app",
    "ServiceConfig",
    "get_config",
    "S3Repository",
]

# Direct imports that don't cause cycles
from .config import ServiceConfig, get_config
from .s3_repository import S3Repository


def __getattr__(name):
    """Lazy load api module to avoid circular imports."""
    if name == "app":
        from .api import app
        return app
    elif name == "create_app":
        from .api import create_app
        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

