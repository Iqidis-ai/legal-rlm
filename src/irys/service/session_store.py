"""Session-based fact/citation persistence.

Stores accumulated facts and citations from investigations in S3 (production)
or local disk (development), keyed by session_id.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import ServiceConfig

logger = logging.getLogger(__name__)

SESSION_S3_PREFIX = "sessions"


class SessionData:
    """Lightweight session data container."""

    def __init__(
        self,
        facts: Optional[list[str]] = None,
        citations: Optional[list[dict]] = None,
        updated_at: Optional[str] = None,
        investigation_count: int = 0,
    ):
        self.facts = facts or []
        self.citations = citations or []
        self.updated_at = updated_at or datetime.now().isoformat()
        self.investigation_count = investigation_count

    def to_dict(self) -> dict:
        return {
            "facts": self.facts,
            "citations": self.citations,
            "updated_at": self.updated_at,
            "investigation_count": self.investigation_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionData":
        return cls(
            facts=data.get("facts", []),
            citations=data.get("citations", []),
            updated_at=data.get("updated_at"),
            investigation_count=data.get("investigation_count", 0),
        )


class SessionStore:
    """Load/save session data from S3 or local disk."""

    def __init__(self, config: ServiceConfig):
        self.config = config
        self._s3 = None

    def _get_s3_client(self):
        if self._s3 is None:
            import boto3
            kwargs = {"region_name": self.config.s3_region}
            if self.config.aws_access_key_id:
                kwargs["aws_access_key_id"] = self.config.aws_access_key_id
            if self.config.aws_secret_access_key:
                kwargs["aws_secret_access_key"] = self.config.aws_secret_access_key
            self._s3 = boto3.client("s3", **kwargs)
        return self._s3

    def _s3_key(self, session_id: str) -> str:
        prefix = self.config.s3_prefix.strip("/")
        if prefix:
            return f"{prefix}/{SESSION_S3_PREFIX}/{session_id}.json"
        return f"{SESSION_S3_PREFIX}/{session_id}.json"

    def _local_path(self, session_id: str) -> Path:
        return Path(self.config.temp_dir) / "sessions" / f"{session_id}.json"

    async def load(self, session_id: str) -> Optional[SessionData]:
        """Load session data. Returns None if session doesn't exist."""
        try:
            if self.config.storage_mode == "local":
                return await self._load_local(session_id)
            else:
                return await self._load_s3(session_id)
        except Exception as e:
            logger.warning(f"Failed to load session {session_id}: {e}")
            return None

    async def save(self, session_id: str, data: SessionData) -> None:
        """Save session data."""
        data.updated_at = datetime.now().isoformat()
        try:
            if self.config.storage_mode == "local":
                await self._save_local(session_id, data)
            else:
                await self._save_s3(session_id, data)
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")

    async def _load_local(self, session_id: str) -> Optional[SessionData]:
        path = self._local_path(session_id)
        if not path.exists():
            return None

        def _read():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        raw = await asyncio.to_thread(_read)
        return SessionData.from_dict(raw)

    async def _save_local(self, session_id: str, data: SessionData) -> None:
        path = self._local_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _write():
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data.to_dict(), f)

        await asyncio.to_thread(_write)

    async def _load_s3(self, session_id: str) -> Optional[SessionData]:
        key = self._s3_key(session_id)

        def _get():
            try:
                s3 = self._get_s3_client()
                resp = s3.get_object(Bucket=self.config.s3_bucket, Key=key)
                return json.loads(resp["Body"].read().decode("utf-8"))
            except Exception as e:
                if "NoSuchKey" in str(e):
                    return None
                raise

        raw = await asyncio.to_thread(_get)
        if raw is None:
            return None
        return SessionData.from_dict(raw)

    async def _save_s3(self, session_id: str, data: SessionData) -> None:
        key = self._s3_key(session_id)
        body = json.dumps(data.to_dict())

        def _put():
            s3 = self._get_s3_client()
            s3.put_object(
                Bucket=self.config.s3_bucket,
                Key=key,
                Body=body.encode("utf-8"),
                ContentType="application/json",
            )

        await asyncio.to_thread(_put)
