"""
PostgreSQL database layer for the Knowledge Graph system.
Replaces SQLite with PostgreSQL for multi-user/multi-matter support.
"""
import psycopg2
import psycopg2.extras
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from .models import Entity, Edge, Mention, Document, Event, Alias


class PostgreSQLDatabase:
    """PostgreSQL database wrapper for knowledge graph storage."""

    def __init__(self, connection_string: str, matter_id: str):
        """
        Initialize PostgreSQL connection for a specific matter.

        Args:
            connection_string: PostgreSQL connection string
            matter_id: UUID of the matter this KG belongs to
        """
        self.connection_string = connection_string
        self.matter_id = matter_id
        self.conn = self._new_connection()
        psycopg2.extras.register_uuid()
        self._ensure_doc_id_column()
        self._ensure_document_chunks_table()
        self._ensure_embeddings_table()

    def _new_connection(self):
        """Create a new PostgreSQL connection with TCP keepalives."""
        conn = psycopg2.connect(
            self.connection_string,
            keepalives=1,
            keepalives_idle=60,
            keepalives_interval=15,
            keepalives_count=3,
        )
        conn.autocommit = False
        return conn

    def _ensure_connection(self):
        """Reconnect if the connection has been closed or dropped."""
        try:
            if self.conn.closed:
                raise psycopg2.InterfaceError("connection already closed")
            cur = self.conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
            self.conn.rollback()
        except (psycopg2.OperationalError, psycopg2.InterfaceError, psycopg2.DatabaseError):
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = self._new_connection()

    def _get_cursor(self):
        """Get a cursor with dict-like row factory.

        Automatically recovers from aborted transaction state (e.g. after a
        constraint violation).  Only rolls back when the connection is in the
        INERROR state — normal in-progress transactions are left untouched.
        Also reconnects if the connection was dropped by the server.
        """
        self._ensure_connection()
        if self.conn.get_transaction_status() == psycopg2.extensions.TRANSACTION_STATUS_INERROR:
            try:
                self.conn.rollback()
            except Exception:
                pass
        return self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    def _ensure_doc_id_column(self):
        """Add doc_id column to kg_documents if it doesn't exist (one-time migration).

        Uses information_schema to check first — avoids ACCESS EXCLUSIVE lock
        from ALTER TABLE when the column already exists.  This prevents deadlocks
        when multiple connections initialise concurrently.
        """
        try:
            cursor = self._get_cursor()
            cursor.execute("""
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'kg_documents' AND column_name = 'doc_id'
            """)
            if cursor.fetchone() is not None:
                # Column already exists — nothing to do
                self.conn.commit()
                return
            cursor.execute(
                "ALTER TABLE kg_documents ADD COLUMN doc_id UUID")
            self.conn.commit()
        except Exception:
            # Table may not exist yet or column already exists
            try:
                self.conn.rollback()
            except Exception:
                pass

    def _ensure_document_chunks_table(self):
        """Create kg_document_chunks table if it doesn't exist.

        Stores pre-computed (Voyage AI) or backend-generated chunk embeddings
        for document-level semantic search.
        """
        try:
            cursor = self._get_cursor()
            cursor.execute("""
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'kg_document_chunks'
            """)
            if cursor.fetchone() is not None:
                self.conn.commit()
                return
            cursor.execute("""
                CREATE TABLE kg_document_chunks (
                    id UUID PRIMARY KEY,
                    matter_id UUID NOT NULL,
                    kg_doc_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding BYTEA,
                    dimension INTEGER,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_doc_chunks_matter
                ON kg_document_chunks (matter_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_doc_chunks_doc
                ON kg_document_chunks (kg_doc_id)
            """)
            self.conn.commit()
        except Exception:
            try:
                self.conn.rollback()
            except Exception:
                pass

    def _ensure_embeddings_table(self):
        """Create kg_embeddings table if it doesn't exist."""
        try:
            cursor = self._get_cursor()
            cursor.execute("""
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'kg_embeddings'
            """)
            if cursor.fetchone() is not None:
                self.conn.commit()
                return
            cursor.execute("""
                CREATE TABLE kg_embeddings (
                    entity_id UUID PRIMARY KEY,
                    vector BYTEA NOT NULL,
                    dimension INTEGER,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            self.conn.commit()
        except Exception:
            try:
                self.conn.rollback()
            except Exception:
                pass

    # ==================== Entity Operations ====================

    def add_entity(self, entity: Entity) -> str:
        """Add a new entity to the database."""
        cursor = self._get_cursor()
        cursor.execute("""
            INSERT INTO kg_entities (id, matter_id, type, canonical_name, properties, confidence, status, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            entity.id, self.matter_id, entity.type, entity.canonical_name,
            json.dumps(entity.properties), entity.confidence, entity.status,
            entity.created_at, entity.updated_at
        ))
        self.conn.commit()
        self._log_event("create_entity", {
                        "entity_id": entity.id, "type": entity.type, "name": entity.canonical_name})
        return entity.id

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        cursor = self._get_cursor()
        cursor.execute(
            "SELECT * FROM kg_entities WHERE id = %s AND matter_id = %s", (entity_id, self.matter_id))
        row = cursor.fetchone()
        if row:
            return self._row_to_entity(row)
        return None

    def get_entities_by_type(self, entity_type: str, limit: int = 100) -> List[Entity]:
        """Get entities by type."""
        cursor = self._get_cursor()
        cursor.execute("""
            SELECT * FROM kg_entities 
            WHERE type = %s AND matter_id = %s AND status = 'active' 
            LIMIT %s
        """, (entity_type, self.matter_id, limit))
        return [self._row_to_entity(row) for row in cursor.fetchall()]

    def get_all_entities(self, limit: int = 1000) -> List[Entity]:
        """Get all active entities."""
        cursor = self._get_cursor()
        cursor.execute("""
            SELECT * FROM kg_entities 
            WHERE matter_id = %s AND status = 'active' 
            LIMIT %s
        """, (self.matter_id, limit))
        return [self._row_to_entity(row) for row in cursor.fetchall()]

    def search_entities_by_name(self, name: str, limit: int = 20) -> List[Entity]:
        """Search entities by name (fuzzy match)."""
        cursor = self._get_cursor()
        search_pattern = f"%{name}%"
        cursor.execute("""
            SELECT DISTINCT e.* FROM kg_entities e
            LEFT JOIN kg_aliases a ON e.id = a.entity_id
            WHERE (e.canonical_name ILIKE %s OR a.alias_text ILIKE %s)
            AND e.matter_id = %s AND e.status = 'active'
            LIMIT %s
        """, (search_pattern, search_pattern, self.matter_id, limit))
        return [self._row_to_entity(row) for row in cursor.fetchall()]

    def update_entity(self, entity: Entity):
        """Update an existing entity."""
        entity.updated_at = datetime.now()
        cursor = self._get_cursor()
        cursor.execute("""
            UPDATE kg_entities SET
                type = %s, canonical_name = %s, properties = %s, confidence = %s,
                status = %s, updated_at = %s
            WHERE id = %s AND matter_id = %s
        """, (
            entity.type, entity.canonical_name, json.dumps(entity.properties),
            entity.confidence, entity.status, entity.updated_at, entity.id, self.matter_id
        ))
        self.conn.commit()
        self._log_event("update_entity", {"entity_id": entity.id})

    def delete_entity(self, entity_id: str):
        """Soft delete an entity (mark as tombstone)."""
        cursor = self._get_cursor()
        cursor.execute("""
            UPDATE kg_entities SET status = 'tombstone', updated_at = %s 
            WHERE id = %s AND matter_id = %s
        """, (datetime.now(), entity_id, self.matter_id))
        self.conn.commit()
        self._log_event("delete_entity", {"entity_id": entity_id})

    def merge_entities(self, keep_id: str, merge_id: str):
        """Merge two entities, keeping one and tombstoning the other."""
        cursor = self._get_cursor()

        # Move all mentions from merge to keep
        cursor.execute(
            "UPDATE kg_mentions SET entity_id = %s WHERE entity_id = %s", (keep_id, merge_id))

        # Move all aliases from merge to keep
        cursor.execute(
            "UPDATE kg_aliases SET entity_id = %s WHERE entity_id = %s", (keep_id, merge_id))

        # Update edges - source
        cursor.execute("UPDATE kg_edges SET source_entity_id = %s WHERE source_entity_id = %s AND matter_id = %s",
                       (keep_id, merge_id, self.matter_id))

        # Update edges - target
        cursor.execute("UPDATE kg_edges SET target_entity_id = %s WHERE target_entity_id = %s AND matter_id = %s",
                       (keep_id, merge_id, self.matter_id))

        # Tombstone the merged entity
        cursor.execute("""
            UPDATE kg_entities SET status = 'tombstone', updated_at = %s 
            WHERE id = %s AND matter_id = %s
        """, (datetime.now(), merge_id, self.matter_id))

        self.conn.commit()
        self._log_event("merge_entities", {
                        "keep_id": keep_id, "merge_id": merge_id}, user_initiated=True)

    def _row_to_entity(self, row: Dict) -> Entity:
        """Convert a database row to an Entity object."""
        return Entity(
            id=str(row["id"]),
            type=row["type"],
            canonical_name=row["canonical_name"],
            properties=row["properties"] if isinstance(
                row["properties"], dict) else {},
            confidence=row["confidence"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        )

    # ==================== Alias Operations ====================

    def add_alias(self, alias: Alias):
        """Add an alias for an entity."""
        cursor = self._get_cursor()
        try:
            cursor.execute("""
                INSERT INTO kg_aliases (id, entity_id, alias_text, source)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (entity_id, alias_text) DO NOTHING
            """, (alias.id, alias.entity_id, alias.alias_text, alias.source))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            # Ignore duplicate aliases
            pass

    def get_aliases(self, entity_id: str) -> List[Alias]:
        """Get all aliases for an entity."""
        cursor = self._get_cursor()
        cursor.execute(
            "SELECT * FROM kg_aliases WHERE entity_id = %s", (entity_id,))
        return [Alias(
            id=str(row["id"]),
            entity_id=str(row["entity_id"]),
            alias_text=row["alias_text"],
            source=row["source"]
        ) for row in cursor.fetchall()]

    def get_all_aliases_for_matter(self) -> List[Alias]:
        """Get all aliases for entities in this matter (single query)."""
        cursor = self._get_cursor()
        cursor.execute("""
            SELECT a.* FROM kg_aliases a
            JOIN kg_entities e ON a.entity_id = e.id
            WHERE e.matter_id = %s AND e.status = 'active'
        """, (self.matter_id,))
        return [Alias(
            id=str(row["id"]),
            entity_id=str(row["entity_id"]),
            alias_text=row["alias_text"],
            source=row["source"]
        ) for row in cursor.fetchall()]

    # ==================== Edge Operations ====================

    def add_edge(self, edge: Edge) -> str:
        """Add a new edge to the database."""
        cursor = self._get_cursor()
        cursor.execute("""
            INSERT INTO kg_edges (id, matter_id, source_entity_id, target_entity_id, relation_type,
                                 properties, confidence, provenance_doc_id, provenance_span, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            edge.id, self.matter_id, edge.source_entity_id, edge.target_entity_id, edge.relation_type,
            json.dumps(
                edge.properties), edge.confidence, edge.provenance_doc_id,
            edge.provenance_span, edge.created_at
        ))
        self.conn.commit()
        self._log_event("create_edge", {
            "edge_id": edge.id,
            "source": edge.source_entity_id,
            "target": edge.target_entity_id,
            "relation": edge.relation_type
        })
        return edge.id

    def get_edges_from(self, entity_id: str) -> List[Edge]:
        """Get all outgoing edges from an entity."""
        cursor = self._get_cursor()
        cursor.execute("SELECT * FROM kg_edges WHERE source_entity_id = %s AND matter_id = %s",
                       (entity_id, self.matter_id))
        return [self._row_to_edge(row) for row in cursor.fetchall()]

    def get_edges_to(self, entity_id: str) -> List[Edge]:
        """Get all incoming edges to an entity."""
        cursor = self._get_cursor()
        cursor.execute("SELECT * FROM kg_edges WHERE target_entity_id = %s AND matter_id = %s",
                       (entity_id, self.matter_id))
        return [self._row_to_edge(row) for row in cursor.fetchall()]

    def get_all_edges(self, limit: int = 1000) -> List[Edge]:
        """Get all edges."""
        cursor = self._get_cursor()
        cursor.execute(
            "SELECT * FROM kg_edges WHERE matter_id = %s LIMIT %s", (self.matter_id, limit))
        return [self._row_to_edge(row) for row in cursor.fetchall()]

    def get_entity_neighbors(self, entity_id: str, max_hops: int = 1) -> Dict[str, Any]:
        """Get neighboring entities within N hops."""
        visited = set()
        result = {"entities": [], "edges": []}

        def traverse(current_id: str, depth: int):
            if depth > max_hops or current_id in visited:
                return
            visited.add(current_id)

            entity = self.get_entity(current_id)
            if entity and entity.status == "active":
                result["entities"].append(entity)

            # Get outgoing edges
            for edge in self.get_edges_from(current_id):
                result["edges"].append(edge)
                traverse(edge.target_entity_id, depth + 1)

            # Get incoming edges
            for edge in self.get_edges_to(current_id):
                result["edges"].append(edge)
                traverse(edge.source_entity_id, depth + 1)

        traverse(entity_id, 0)
        return result

    def delete_edge(self, edge_id: str):
        """Delete an edge."""
        cursor = self._get_cursor()
        cursor.execute(
            "DELETE FROM kg_edges WHERE id = %s AND matter_id = %s", (edge_id, self.matter_id))
        self.conn.commit()
        self._log_event("delete_edge", {"edge_id": edge_id})

    def _row_to_edge(self, row: Dict) -> Edge:
        """Convert a database row to an Edge object."""
        return Edge(
            id=str(row["id"]),
            source_entity_id=str(row["source_entity_id"]),
            target_entity_id=str(row["target_entity_id"]),
            relation_type=row["relation_type"],
            properties=row["properties"] if isinstance(
                row["properties"], dict) else {},
            confidence=row["confidence"],
            provenance_doc_id=str(
                row["provenance_doc_id"]) if row["provenance_doc_id"] else None,
            provenance_span=row["provenance_span"],
            created_at=row["created_at"]
        )

    # ==================== Mention Operations ====================

    def add_mention(self, mention: Mention) -> str:
        """Add a new mention."""
        cursor = self._get_cursor()
        cursor.execute("""
            INSERT INTO kg_mentions (id, entity_id, doc_id, span_start, span_end, surface_text, context_snippet)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            mention.id, mention.entity_id, mention.doc_id,
            mention.span_start, mention.span_end, mention.surface_text, mention.context_snippet
        ))
        self.conn.commit()
        return mention.id

    def get_mentions_for_entity(self, entity_id: str) -> List[Mention]:
        """Get all mentions of an entity."""
        cursor = self._get_cursor()
        cursor.execute(
            "SELECT * FROM kg_mentions WHERE entity_id = %s", (entity_id,))
        return [self._row_to_mention(row) for row in cursor.fetchall()]

    def get_mentions_in_doc(self, doc_id: str) -> List[Mention]:
        """Get all mentions in a document."""
        cursor = self._get_cursor()
        cursor.execute(
            "SELECT * FROM kg_mentions WHERE doc_id = %s", (doc_id,))
        return [self._row_to_mention(row) for row in cursor.fetchall()]

    def _row_to_mention(self, row: Dict) -> Mention:
        """Convert a database row to a Mention object."""
        return Mention(
            id=str(row["id"]),
            entity_id=str(row["entity_id"]),
            doc_id=str(row["doc_id"]),
            span_start=row["span_start"],
            span_end=row["span_end"],
            surface_text=row["surface_text"],
            context_snippet=row["context_snippet"]
        )

    # ==================== Document Operations ====================

    def add_document(self, document: Document) -> str:
        """Add a new document. Returns existing document ID if hash already exists."""
        cursor = self._get_cursor()
        cursor.execute("""
            INSERT INTO kg_documents (id, matter_id, filename, filepath, file_hash, added_at, processed_at, doc_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (matter_id, file_hash) DO UPDATE SET
                filename = EXCLUDED.filename,
                doc_id = COALESCE(EXCLUDED.doc_id, kg_documents.doc_id)
            RETURNING id
        """, (
            document.id, self.matter_id, document.filename, document.filepath, document.file_hash,
            document.added_at,
            document.processed_at if document.processed_at else None,
            document.doc_id
        ))
        row = cursor.fetchone()
        self.conn.commit()
        # Return the actual ID (may differ from document.id if conflict occurred)
        return str(row["id"]) if row else document.id

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        cursor = self._get_cursor()
        cursor.execute(
            "SELECT * FROM kg_documents WHERE id = %s AND matter_id = %s", (doc_id, self.matter_id))
        row = cursor.fetchone()
        if row:
            return self._row_to_document(row)
        return None

    def get_document_by_hash(self, file_hash: str) -> Optional[Document]:
        """Get a document by file hash."""
        cursor = self._get_cursor()
        cursor.execute("SELECT * FROM kg_documents WHERE file_hash = %s AND matter_id = %s",
                       (file_hash, self.matter_id))
        row = cursor.fetchone()
        if row:
            return self._row_to_document(row)
        return None

    def get_all_documents(self) -> List[Document]:
        """Get all documents."""
        cursor = self._get_cursor()
        cursor.execute(
            "SELECT * FROM kg_documents WHERE matter_id = %s", (self.matter_id,))
        return [self._row_to_document(row) for row in cursor.fetchall()]

    def mark_document_processed(self, doc_id: str):
        """Mark a document as processed."""
        cursor = self._get_cursor()
        cursor.execute("UPDATE kg_documents SET processed_at = %s WHERE id = %s AND matter_id = %s",
                       (datetime.now(), doc_id, self.matter_id))
        self.conn.commit()

    def delete_document(self, doc_id: str):
        """Delete a document and all related data (cascade handled by FK)."""
        cursor = self._get_cursor()

        # Get entities that only exist because of this document
        cursor.execute("""
            SELECT entity_id FROM kg_mentions WHERE doc_id = %s
            GROUP BY entity_id
        """, (doc_id,))
        entity_ids = [str(row["entity_id"]) for row in cursor.fetchall()]

        # Delete mentions for this document
        cursor.execute("DELETE FROM kg_mentions WHERE doc_id = %s", (doc_id,))

        # Delete edges with provenance from this document
        cursor.execute("DELETE FROM kg_edges WHERE provenance_doc_id = %s AND matter_id = %s",
                       (doc_id, self.matter_id))

        # Check if any entities became orphaned (no remaining mentions)
        for entity_id in entity_ids:
            cursor.execute(
                "SELECT COUNT(*) as cnt FROM kg_mentions WHERE entity_id = %s", (entity_id,))
            if cursor.fetchone()["cnt"] == 0:
                # Orphaned - tombstone it
                cursor.execute("""
                    UPDATE kg_entities SET status = 'tombstone', updated_at = %s 
                    WHERE id = %s AND matter_id = %s
                """, (datetime.now(), entity_id, self.matter_id))

        # Delete the document
        cursor.execute(
            "DELETE FROM kg_documents WHERE id = %s AND matter_id = %s", (doc_id, self.matter_id))
        self.conn.commit()
        self._log_event("delete_document", {"doc_id": doc_id})

    def _row_to_document(self, row: Dict) -> Document:
        """Convert a database row to a Document object."""
        return Document(
            id=str(row["id"]),
            filename=row["filename"],
            filepath=row["filepath"],
            file_hash=row["file_hash"],
            added_at=row["added_at"],
            processed_at=row["processed_at"] if row["processed_at"] else None,
            doc_id=str(row["doc_id"]) if row.get("doc_id") else None
        )

    # ==================== Resolution Queue Operations ====================

    def add_to_resolution_queue(self, surface_text: str, context: str, doc_id: str,
                                span_start: int, span_end: int, candidates: List[Dict]):
        """Add an unresolved mention to the queue."""
        cursor = self._get_cursor()
        from .models import generate_id
        cursor.execute("""
            INSERT INTO kg_resolution_queue (id, matter_id, mention_surface_text, mention_context, doc_id,
                                            span_start, span_end, candidate_entities, status, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'pending', %s)
        """, (
            generate_id(), self.matter_id, surface_text, context, doc_id, span_start, span_end,
            json.dumps(candidates), datetime.now()
        ))
        self.conn.commit()

    def get_pending_resolutions(self, limit: int = 50) -> List[Dict]:
        """Get pending resolution items."""
        cursor = self._get_cursor()
        cursor.execute("""
            SELECT * FROM kg_resolution_queue WHERE status = 'pending' AND matter_id = %s
            ORDER BY created_at LIMIT %s
        """, (self.matter_id, limit))
        return [{
            "id": str(row["id"]),
            "surface_text": row["mention_surface_text"],
            "context": row["mention_context"],
            "doc_id": str(row["doc_id"]),
            "span_start": row["span_start"],
            "span_end": row["span_end"],
            "candidates": row["candidate_entities"] if isinstance(row["candidate_entities"], list) else [],
            "created_at": row["created_at"]
        } for row in cursor.fetchall()]

    def resolve_queue_item(self, queue_id: str, entity_id: str):
        """Resolve a queue item by linking to an entity."""
        cursor = self._get_cursor()

        # Get queue item details
        cursor.execute("SELECT * FROM kg_resolution_queue WHERE id = %s AND matter_id = %s",
                       (queue_id, self.matter_id))
        row = cursor.fetchone()
        if not row:
            return

        # Create mention
        mention = Mention.create(
            entity_id=entity_id,
            doc_id=str(row["doc_id"]),
            span_start=row["span_start"],
            span_end=row["span_end"],
            surface_text=row["mention_surface_text"],
            context_snippet=row["mention_context"]
        )
        self.add_mention(mention)

        # Mark as resolved
        cursor.execute(
            "UPDATE kg_resolution_queue SET status = 'resolved' WHERE id = %s", (queue_id,))
        self.conn.commit()

    # ==================== Event Logging ====================

    def _log_event(self, operation: str, payload: Dict, user_initiated: bool = False):
        """Log an event for audit trail."""
        event = Event.create(operation, payload, user_initiated)
        cursor = self._get_cursor()
        cursor.execute("""
            INSERT INTO kg_events (id, matter_id, timestamp, operation, payload, user_initiated)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            event.id, self.matter_id, event.timestamp, event.operation,
            json.dumps(event.payload), user_initiated
        ))
        self.conn.commit()

    def get_events(self, limit: int = 100) -> List[Event]:
        """Get recent events."""
        cursor = self._get_cursor()
        cursor.execute("""
            SELECT * FROM kg_events WHERE matter_id = %s 
            ORDER BY timestamp DESC LIMIT %s
        """, (self.matter_id, limit))
        return [Event(
            id=str(row["id"]),
            timestamp=row["timestamp"],
            operation=row["operation"],
            payload=row["payload"] if isinstance(row["payload"], dict) else {},
            user_initiated=row["user_initiated"]
        ) for row in cursor.fetchall()]

    # ==================== Embedding Operations ====================

    def store_embedding(self, entity_id: str, vector: bytes):
        """Store an embedding vector for an entity."""
        cursor = self._get_cursor()
        dimension = len(vector) // 4  # Assuming float32
        cursor.execute("""
            INSERT INTO kg_embeddings (entity_id, vector, dimension, created_at)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (entity_id) DO UPDATE SET vector = EXCLUDED.vector, created_at = EXCLUDED.created_at
        """, (entity_id, vector, dimension, datetime.now()))
        self.conn.commit()

    def get_embedding(self, entity_id: str) -> Optional[bytes]:
        """Get embedding vector for an entity."""
        cursor = self._get_cursor()
        cursor.execute(
            "SELECT vector FROM kg_embeddings WHERE entity_id = %s", (entity_id,))
        row = cursor.fetchone()
        return bytes(row["vector"]) if row else None

    def get_all_embeddings(self) -> List[Tuple[str, bytes]]:
        """Get all embeddings for entities in this matter."""
        cursor = self._get_cursor()
        cursor.execute("""
            SELECT e.entity_id, e.vector 
            FROM kg_embeddings e
            JOIN kg_entities ent ON e.entity_id = ent.id
            WHERE ent.matter_id = %s
        """, (self.matter_id,))
        return [(str(row["entity_id"]), bytes(row["vector"])) for row in cursor.fetchall()]

    # ==================== Stats ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        cursor = self._get_cursor()

        cursor.execute("SELECT COUNT(*) as cnt FROM kg_entities WHERE status = 'active' AND matter_id = %s",
                       (self.matter_id,))
        entity_count = cursor.fetchone()["cnt"]

        cursor.execute(
            "SELECT COUNT(*) as cnt FROM kg_edges WHERE matter_id = %s", (self.matter_id,))
        edge_count = cursor.fetchone()["cnt"]

        cursor.execute(
            "SELECT COUNT(*) as cnt FROM kg_documents WHERE matter_id = %s", (self.matter_id,))
        doc_count = cursor.fetchone()["cnt"]

        cursor.execute("""
            SELECT COUNT(*) as cnt FROM kg_mentions m
            JOIN kg_entities e ON m.entity_id = e.id
            WHERE e.matter_id = %s
        """, (self.matter_id,))
        mention_count = cursor.fetchone()["cnt"]

        cursor.execute("""
            SELECT COUNT(*) as cnt FROM kg_resolution_queue 
            WHERE status = 'pending' AND matter_id = %s
        """, (self.matter_id,))
        pending_count = cursor.fetchone()["cnt"]

        cursor.execute("""
            SELECT type, COUNT(*) as cnt FROM kg_entities 
            WHERE status = 'active' AND matter_id = %s 
            GROUP BY type
        """, (self.matter_id,))
        type_counts = {row["type"]: row["cnt"] for row in cursor.fetchall()}

        return {
            "entities": entity_count,
            "edges": edge_count,
            "documents": doc_count,
            "mentions": mention_count,
            "pending_resolutions": pending_count,
            "entities_by_type": type_counts
        }

    # ==================== Batch Operations (Performance) ====================

    def add_entities_batch(self, entities: List[Entity]) -> List[str]:
        """Add multiple entities in a single transaction using bulk insert."""
        if not entities:
            return []
        cursor = self._get_cursor()
        values = [
            (e.id, self.matter_id, e.type, e.canonical_name,
             json.dumps(e.properties), e.confidence, e.status,
             e.created_at, e.updated_at)
            for e in entities
        ]
        psycopg2.extras.execute_values(
            cursor,
            """INSERT INTO kg_entities (id, matter_id, type, canonical_name, properties, confidence, status, created_at, updated_at)
               VALUES %s ON CONFLICT (id) DO NOTHING""",
            values,
            page_size=500
        )
        self.conn.commit()
        # Batch log events
        self._log_events_batch([
            ("create_entity", {"entity_id": e.id,
             "type": e.type, "name": e.canonical_name})
            for e in entities
        ])
        return [e.id for e in entities]

    def add_edges_batch(self, edges: List[Edge]) -> List[str]:
        """Add multiple edges in a single transaction using bulk insert."""
        if not edges:
            return []
        cursor = self._get_cursor()
        values = [
            (e.id, self.matter_id, e.source_entity_id, e.target_entity_id, e.relation_type,
             json.dumps(e.properties), e.confidence, e.provenance_doc_id,
             e.provenance_span, e.created_at)
            for e in edges
        ]
        psycopg2.extras.execute_values(
            cursor,
            """INSERT INTO kg_edges (id, matter_id, source_entity_id, target_entity_id, relation_type,
                                     properties, confidence, provenance_doc_id, provenance_span, created_at)
               VALUES %s ON CONFLICT (id) DO NOTHING""",
            values,
            page_size=500
        )
        self.conn.commit()
        return [e.id for e in edges]

    def add_mentions_batch(self, mentions: List[Mention]) -> List[str]:
        """Add multiple mentions in a single transaction using bulk insert."""
        if not mentions:
            return []
        cursor = self._get_cursor()
        values = [
            (m.id, m.entity_id, m.doc_id, m.span_start, m.span_end,
             m.surface_text, m.context_snippet)
            for m in mentions
        ]
        psycopg2.extras.execute_values(
            cursor,
            """INSERT INTO kg_mentions (id, entity_id, doc_id, span_start, span_end, surface_text, context_snippet)
               VALUES %s ON CONFLICT (id) DO NOTHING""",
            values,
            page_size=500
        )
        self.conn.commit()
        return [m.id for m in mentions]

    def add_aliases_batch(self, aliases: List[Alias]):
        """Add multiple aliases in a single transaction using bulk insert."""
        if not aliases:
            return
        cursor = self._get_cursor()
        values = [
            (a.id, a.entity_id, a.alias_text, a.source)
            for a in aliases
        ]
        try:
            psycopg2.extras.execute_values(
                cursor,
                """INSERT INTO kg_aliases (id, entity_id, alias_text, source)
                   VALUES %s ON CONFLICT (entity_id, alias_text) DO NOTHING""",
                values,
                page_size=500
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()

    def store_embeddings_batch(self, embeddings: List[Tuple[str, bytes]]):
        """Store multiple embedding vectors in a single transaction."""
        if not embeddings:
            return
        cursor = self._get_cursor()
        now = datetime.now()
        values = [
            (entity_id, vector, len(vector) // 4, now)
            for entity_id, vector in embeddings
        ]
        psycopg2.extras.execute_values(
            cursor,
            """INSERT INTO kg_embeddings (entity_id, vector, dimension, created_at)
               VALUES %s
               ON CONFLICT (entity_id) DO UPDATE SET vector = EXCLUDED.vector, created_at = EXCLUDED.created_at""",
            values,
            page_size=500
        )
        self.conn.commit()

    def _log_events_batch(self, events_data: List[Tuple[str, Dict]]):
        """Log multiple events in a single transaction."""
        if not events_data:
            return
        from .models import generate_id
        cursor = self._get_cursor()
        now = datetime.now()
        values = [
            (generate_id(), self.matter_id, now,
             operation, json.dumps(payload), False)
            for operation, payload in events_data
        ]
        try:
            psycopg2.extras.execute_values(
                cursor,
                """INSERT INTO kg_events (id, matter_id, timestamp, operation, payload, user_initiated)
                   VALUES %s""",
                values,
                page_size=500
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()

    def store_document_chunks_batch(self, kg_doc_id: str, chunks: list):
        """Store pre-computed document chunks with embeddings.

        Args:
            kg_doc_id: The KG document ID (kg_documents.id)
            chunks: List of dicts with keys: chunk_index, content, embedding (list[float] | None)
        """
        if not chunks:
            return
        import numpy as np
        from .models import generate_id
        cursor = self._get_cursor()
        now = datetime.now()
        values = []
        for chunk in chunks:
            embedding = chunk.get("embedding")
            if embedding is not None:
                vec = np.array(embedding, dtype=np.float32)
                vec_bytes = vec.tobytes()
                dim = len(embedding)
            else:
                vec_bytes = None
                dim = None
            values.append((
                generate_id(), self.matter_id, kg_doc_id,
                chunk.get("chunk_index", 0), chunk.get("content", ""),
                vec_bytes, dim, now
            ))
        psycopg2.extras.execute_values(
            cursor,
            """INSERT INTO kg_document_chunks
               (id, matter_id, kg_doc_id, chunk_index, content, embedding, dimension, created_at)
               VALUES %s
               ON CONFLICT DO NOTHING""",
            values,
            page_size=500
        )
        self.conn.commit()

    def close(self):
        """Close database connection."""
        self.conn.close()


# Alias for backwards compatibility
Database = PostgreSQLDatabase
