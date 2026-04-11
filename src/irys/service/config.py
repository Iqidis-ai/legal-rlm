"""Service configuration with environment-based settings."""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


@dataclass
class ServiceConfig:
    """Production service configuration."""

    # API Settings
    port: int = 8000
    debug: bool = False

    # API Keys
    gemini_api_key: str = ""

    # S3 Settings
    s3_bucket: str = ""
    s3_region: str = "us-east-1"
    s3_prefix: str = ""  # Default prefix for documents
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None

    # Storage Settings (for small instances)
    temp_dir: str = "/tmp/irys"
    matter_db_dir: str = "/tmp/irys/matters"  # Persistent matter model DBs
    checkpoint_dir: str = "/tmp/irys/checkpoints"  # Investigation checkpoint files (SO-3 resume)
    cleanup_after_seconds: int = 300  # 5 minutes
    max_concurrent_jobs: int = 3

    # Processing Settings
    max_documents_per_job: int = 50
    max_document_size_mb: int = 10
    # Logging
    log_level: str = "INFO"

    # Storage mode: "local" for dev (files on disk), "s3" for production (stream to S3)
    storage_mode: str = "s3"

    # Matter model: all investigations build a persistent SQLite matter model (default on)
    enable_matter_model: bool = True

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        """Load configuration from environment variables."""
        return cls(
            # API Settings
            port=int(os.getenv("IRYS_PORT", "8000")),
            debug=os.getenv("IRYS_DEBUG", "false").lower() == "true",
            # API Keys
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            # S3 Settings
            s3_bucket=os.getenv("S3_BUCKET", ""),
            s3_region=os.getenv("S3_REGION", "us-east-1"),
            s3_prefix=os.getenv("S3_PREFIX", ""),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            # Storage Settings
            temp_dir=os.getenv("IRYS_TEMP_DIR", "/tmp/irys"),
            matter_db_dir=os.getenv("IRYS_MATTER_DB_DIR", "/tmp/irys/matters"),
            checkpoint_dir=os.getenv("IRYS_CHECKPOINT_DIR", "/tmp/irys/checkpoints"),
            cleanup_after_seconds=int(os.getenv("IRYS_CLEANUP_SECONDS", "300")),
            max_concurrent_jobs=int(os.getenv("IRYS_MAX_CONCURRENT_JOBS", "3")),
            # Processing Settings
            max_documents_per_job=int(os.getenv("IRYS_MAX_DOCS_PER_JOB", "50")),
            max_document_size_mb=int(os.getenv("IRYS_MAX_DOC_SIZE_MB", "10")),
            # Logging
            log_level=os.getenv("IRYS_LOG_LEVEL", "INFO"),
            # Storage mode
            storage_mode=os.getenv("IRYS_STORAGE_MODE", "s3"),
            # Matter model
            enable_matter_model=os.getenv("IRYS_ENABLE_MATTER_MODEL", "true").lower() == "true",
        )

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if not self.gemini_api_key:
            errors.append("GEMINI_API_KEY is required")

        if not self.s3_bucket:
            errors.append("S3_BUCKET is required for production")

        if self.enable_matter_model and not self.matter_db_dir.strip():
            errors.append("IRYS_MATTER_DB_DIR must not be empty when enable_matter_model=True")

        return errors


@lru_cache()
def get_config() -> ServiceConfig:
    """Get cached configuration instance."""
    return ServiceConfig.from_env()

