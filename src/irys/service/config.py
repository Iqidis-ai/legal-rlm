"""Service configuration with environment-based settings."""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional


@dataclass
class ServiceConfig:
    """Production service configuration for Irys RLM."""

    # API Settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    debug: bool = False

    # API Keys
    gemini_api_key: str = ""
    api_secret_key: str = ""  # For JWT/API auth (optional)

    # S3 Settings
    s3_bucket: str = ""
    s3_region: str = "us-east-1"
    s3_prefix: str = ""  # Default prefix for documents
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None

    # Storage Settings
    temp_dir: str = "/tmp/irys"
    max_temp_size_mb: int = 5000  # 5GB max temp storage before cleanup
    cleanup_after_seconds: int = 600  # 10 minutes
    max_concurrent_jobs: int = 5

    # Processing Settings
    max_documents_per_job: int = 500
    max_document_size_mb: int = 200
    request_timeout_seconds: int = 600  # 10 minutes

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # json or text

    # Storage mode: "local" for dev (files on disk), "s3" for production (stream to S3)
    storage_mode: str = "s3"

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        """Load configuration from environment variables."""
        return cls(
            # API Settings
            host=os.getenv("IRYS_HOST", "0.0.0.0"),
            port=int(os.getenv("IRYS_PORT", "8000")),
            workers=int(os.getenv("IRYS_WORKERS", "1")),
            debug=os.getenv("IRYS_DEBUG", "false").lower() == "true",
            # API Keys
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            api_secret_key=os.getenv("API_SECRET_KEY", ""),
            # S3 Settings
            s3_bucket=os.getenv("S3_BUCKET", ""),
            s3_region=os.getenv("S3_REGION", "us-east-1"),
            s3_prefix=os.getenv("S3_PREFIX", ""),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            # Storage Settings
            temp_dir=os.getenv("IRYS_TEMP_DIR", "/tmp/irys"),
            max_temp_size_mb=int(os.getenv("IRYS_MAX_TEMP_SIZE_MB", "5000")),
            cleanup_after_seconds=int(os.getenv("IRYS_CLEANUP_SECONDS", "600")),
            max_concurrent_jobs=int(os.getenv("IRYS_MAX_CONCURRENT_JOBS", "5")),
            # Processing Settings
            max_documents_per_job=int(os.getenv("IRYS_MAX_DOCS_PER_JOB", "500")),
            max_document_size_mb=int(os.getenv("IRYS_MAX_DOC_SIZE_MB", "200")),
            request_timeout_seconds=int(os.getenv("IRYS_TIMEOUT_SECONDS", "600")),
            # Logging
            log_level=os.getenv("IRYS_LOG_LEVEL", "INFO"),
            log_format=os.getenv("IRYS_LOG_FORMAT", "json"),
            # Storage mode
            storage_mode=os.getenv("IRYS_STORAGE_MODE", "s3"),
        )

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if not self.gemini_api_key:
            errors.append("GEMINI_API_KEY is required")

        if not self.s3_bucket:
            errors.append("S3_BUCKET is required for production")

        return errors


@lru_cache()
def get_config() -> ServiceConfig:
    """Get cached configuration instance."""
    return ServiceConfig.from_env()

