"""
Iqidis PostgreSQL data service.

Fetches matters and documents from the Iqidis database for Knowledge Graph extraction.
Uses POSTGRES_URL from config. Read-only - does not write to Iqidis.

Document flow: matter_documents (matter_id, doc_id) -> document -> artifact
"""
from typing import List, Optional, Dict, Any

from .config import POSTGRES_URL

_psycopg2 = None


def _get_connection():
    """Get psycopg2 connection."""
    global _psycopg2
    if _psycopg2 is None:
        try:
            import psycopg2
            _psycopg2 = psycopg2
        except ImportError:
            raise ImportError(
                "psycopg2 is required for Iqidis database connection. "
                "Install with: pip install psycopg2-binary"
            )
    if not POSTGRES_URL:
        raise ValueError(
            "POSTGRES_URL is not set. Add it to your .env to connect to Iqidis database."
        )
    return _psycopg2.connect(POSTGRES_URL)


def get_matters(user_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """Fetch matters from Iqidis database."""
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            if user_id:
                cur.execute(
                    """
                    SELECT id, matter_name, description, status, user_id
                    FROM matters
                    WHERE user_id = %s AND status = 'Active'
                    ORDER BY updated_at DESC
                    LIMIT %s
                    """,
                    (user_id, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT id, matter_name, description, status, user_id
                    FROM matters
                    WHERE status = 'Active'
                    ORDER BY updated_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
            cols = [c.name for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def get_matter_by_id(matter_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Fetch a single matter by ID."""
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            if user_id:
                cur.execute(
                    """
                    SELECT id, matter_name, description, status, user_id
                    FROM matters
                    WHERE id = %s AND user_id = %s
                    """,
                    (matter_id, user_id),
                )
            else:
                cur.execute(
                    "SELECT id, matter_name, description, status, user_id FROM matters WHERE id = %s",
                    (matter_id,),
                )
            row = cur.fetchone()
            if not row:
                return None
            cols = [c.name for c in cur.description]
            return dict(zip(cols, row))
    finally:
        conn.close()


def get_matter_documents(
    matter_id: str,
    include_folder_docs: bool = True,
) -> List[Dict[str, Any]]:
    """
    Fetch documents for a matter from Iqidis database.

    Flow: matter_documents (matter_id, doc_id) -> document -> artifact
    - matter_documents links matter to documents
    - document has original_name, mime, artifact_id, artifact_status
    - artifact has storage_key for S3 download

    Returns documents with: doc_id, original_name, mime, storage_key, artifact_id, size_byte
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            # From matter_documents -> document -> artifact
            cur.execute(
                """
                SELECT
                    d.id AS doc_id,
                    d.original_name,
                    d.mime,
                    a.storage_key,
                    a.id AS artifact_id,
                    COALESCE(d.size_byte, a.size_byte) AS size_byte
                FROM matter_documents md
                INNER JOIN document d ON md.doc_id = d.id
                LEFT JOIN artifact a ON d.artifact_id = a.id
                WHERE md.matter_id = %s
                  AND d.artifact_status = 'AVAILABLE'
                  AND a.storage_key IS NOT NULL
                """,
                (matter_id,),
            )
            cols = [c.name for c in cur.description]
            docs = [dict(zip(cols, row)) for row in cur.fetchall()]
            seen_ids = {d["doc_id"] for d in docs}

            if include_folder_docs:
                # Also from document_folder (folders linked to matter)
                cur.execute(
                    """
                    SELECT
                        d.id AS doc_id,
                        d.original_name,
                        d.mime,
                        a.storage_key,
                        a.id AS artifact_id,
                        COALESCE(d.size_byte, a.size_byte) AS size_byte
                    FROM document_folder df
                    INNER JOIN document d ON d.folder_id = df.id
                    LEFT JOIN artifact a ON d.artifact_id = a.id
                    WHERE df.matter_id = %s
                      AND d.artifact_status = 'AVAILABLE'
                      AND a.storage_key IS NOT NULL
                    """,
                    (matter_id,),
                )
                for row in cur.fetchall():
                    rec = dict(zip(cols, row))
                    if rec["doc_id"] not in seen_ids:
                        seen_ids.add(rec["doc_id"])
                        docs.append(rec)

            return docs
    finally:
        conn.close()
