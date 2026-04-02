"""SQLite connection lifecycle for the matter model.

One DB per repository at repository/.irys/matter.sqlite3.
WAL mode, foreign_keys=ON, STRICT tables.
"""

import sqlite3
import threading
from pathlib import Path
from typing import Optional

from .schema import apply_schema


class SQLiteMatterDB:
    """
    Manages the SQLite connection for a single matter database.

    Thread-safety: one connection per thread via threading.local().
    All writes happen inside explicit transactions; reads use
    auto-commit (isolation_level=None with explicit BEGIN where needed).
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._local = threading.local()
        # Ensure parent directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # Initialize / migrate schema on first open
        conn = self._conn()
        apply_schema(conn)

    def _conn(self) -> sqlite3.Connection:
        """Get (or create) a thread-local connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                isolation_level=None,  # manual transaction control
            )
            conn.row_factory = sqlite3.Row
            # Pragmas
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA temp_store=MEMORY")
            self._local.conn = conn
        return self._local.conn

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn()

    def execute(self, sql: str, params=()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def executemany(self, sql: str, params_seq) -> sqlite3.Cursor:
        return self.conn.executemany(sql, params_seq)

    def begin(self):
        self.conn.execute("BEGIN")

    def commit(self):
        self.conn.execute("COMMIT")

    def rollback(self):
        self.conn.execute("ROLLBACK")

    def transaction(self):
        """Context manager for explicit transactions."""
        return _Transaction(self)

    def close(self):
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    @classmethod
    def for_repository(cls, repository_path: str | Path) -> "SQLiteMatterDB":
        """Open (or create) the matter DB for a repository path."""
        repo = Path(repository_path)
        db_dir = repo / ".irys"
        db_path = db_dir / "matter.sqlite3"
        return cls(db_path)

    @classmethod
    def in_memory(cls) -> "SQLiteMatterDB":
        """Open an in-memory DB for testing."""
        return cls(Path(":memory:"))

    def __repr__(self) -> str:
        return f"SQLiteMatterDB({self.db_path})"


class _Transaction:
    """Context manager for explicit transactions with savepoint support for nesting."""

    def __init__(self, db: SQLiteMatterDB):
        self._db = db
        self._savepoint: Optional[str] = None

    def __enter__(self):
        if self._db.conn.in_transaction:
            # Already inside a transaction — use a savepoint instead of BEGIN
            import uuid
            self._savepoint = f"sp_{uuid.uuid4().hex[:8]}"
            self._db.conn.execute(f"SAVEPOINT {self._savepoint}")
        else:
            self._db.begin()
        return self._db

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._savepoint is not None:
            if exc_type is None:
                self._db.conn.execute(f"RELEASE SAVEPOINT {self._savepoint}")
            else:
                self._db.conn.execute(f"ROLLBACK TO SAVEPOINT {self._savepoint}")
                self._db.conn.execute(f"RELEASE SAVEPOINT {self._savepoint}")
        else:
            if exc_type is None:
                self._db.commit()
            else:
                self._db.rollback()
        return False
