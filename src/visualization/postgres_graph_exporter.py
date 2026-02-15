"""
Export knowledge graph data for visualization from PostgreSQL.
"""
import json
import psycopg2
import psycopg2.extras
from typing import Dict, List, Optional, Set
from collections import defaultdict


class PostgreSQLGraphExporter:
    """Export graph data for web visualization from PostgreSQL."""

    # Color palette for entity types
    TYPE_COLORS = {
        'Organization': '#4285f4',  # Blue
        'Person': '#ea4335',        # Red
        'Document': '#fbbc04',      # Yellow
        'Date': '#34a853',          # Green
        'Money': '#ff6d01',         # Orange
        'Location': '#46bdc6',      # Teal
        'Reference': '#9c27b0',     # Purple
        'Fact': '#607d8b',          # Gray
    }

    def __init__(self, connection_string: str, matter_id: str):
        """
        Initialize PostgreSQL exporter.

        Args:
            connection_string: PostgreSQL connection string
            matter_id: UUID of the matter
        """
        self.connection_string = connection_string
        self.matter_id = matter_id
        self.conn = self._new_connection()
        psycopg2.extras.register_uuid()

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
        """Get a cursor with dict-like row factory, reconnecting if needed."""
        self._ensure_connection()
        return self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    def close(self):
        self.conn.close()

    def get_graph_data(
        self,
        entity_types: Optional[List[str]] = None,
        exclude_types: Optional[List[str]] = None,
        min_connections: int = 0,
        limit_nodes: int = 500,
        include_facts: bool = False
    ) -> Dict:
        """
        Export graph data in D3.js compatible format.

        Args:
            entity_types: Only include these entity types
            exclude_types: Exclude these entity types
            min_connections: Only include nodes with at least this many connections
            limit_nodes: Maximum number of nodes to include
            include_facts: Whether to include Fact entities (can be many)

        Returns:
            Dict with 'nodes' and 'links' for D3.js
        """
        cursor = self._get_cursor()

        # Build entity filter
        type_filter = ""
        if entity_types:
            types_str = ", ".join(f"'{t}'" for t in entity_types)
            type_filter = f"AND type IN ({types_str})"
        elif exclude_types:
            types_str = ", ".join(f"'{t}'" for t in exclude_types)
            type_filter = f"AND type NOT IN ({types_str})"
        elif not include_facts:
            type_filter = "AND type != 'Fact'"

        # Get entities with connection counts
        cursor.execute(f'''
            WITH edge_counts AS (
                SELECT entity_id, SUM(cnt) as connections FROM (
                    SELECT source_entity_id as entity_id, COUNT(*) as cnt 
                    FROM kg_edges WHERE matter_id = %s GROUP BY source_entity_id
                    UNION ALL
                    SELECT target_entity_id as entity_id, COUNT(*) as cnt 
                    FROM kg_edges WHERE matter_id = %s GROUP BY target_entity_id
                ) sub GROUP BY entity_id
            )
            SELECT e.id, e.canonical_name, e.type, e.properties, e.confidence,
                   COALESCE(ec.connections, 0) as connections
            FROM kg_entities e
            LEFT JOIN edge_counts ec ON e.id = ec.entity_id
            WHERE e.status = 'active' AND e.matter_id = %s
            {type_filter}
            AND COALESCE(ec.connections, 0) >= %s
            ORDER BY connections DESC
            LIMIT %s
        ''', (self.matter_id, self.matter_id, self.matter_id, min_connections, limit_nodes))

        entities = cursor.fetchall()
        entity_ids = {str(e['id']) for e in entities}

        # Build nodes
        nodes = []
        id_to_index = {}
        for i, e in enumerate(entities):
            entity_id_str = str(e['id'])
            id_to_index[entity_id_str] = i
            nodes.append({
                'id': i,
                'entity_id': entity_id_str,
                'name': e['canonical_name'][:50],  # Truncate long names
                'full_name': e['canonical_name'],
                'type': e['type'],
                'color': self.TYPE_COLORS.get(e['type'], '#999999'),
                'connections': e['connections'],
                'confidence': e['confidence'],
                'properties': e['properties'] if isinstance(e['properties'], dict) else {}
            })

        # Get edges between included entities
        if entity_ids:
            entity_ids_list = list(entity_ids)
            cursor.execute('''
                SELECT source_entity_id, target_entity_id, relation_type, confidence, properties
                FROM kg_edges
                WHERE matter_id = %s 
                  AND source_entity_id = ANY(%s::uuid[]) 
                  AND target_entity_id = ANY(%s::uuid[])
            ''', (self.matter_id, entity_ids_list, entity_ids_list))

            edges = cursor.fetchall()
        else:
            edges = []

        # Build links (deduplicate)
        seen_links = set()
        links = []
        for e in edges:
            source_id_str = str(e['source_entity_id'])
            target_id_str = str(e['target_entity_id'])
            source_idx = id_to_index.get(source_id_str)
            target_idx = id_to_index.get(target_id_str)

            if source_idx is not None and target_idx is not None:
                link_key = (source_idx, target_idx, e['relation_type'])
                if link_key not in seen_links:
                    seen_links.add(link_key)
                    links.append({
                        'source': source_idx,
                        'target': target_idx,
                        'relation': e['relation_type'],
                        'confidence': e['confidence']
                    })

        return {
            'nodes': nodes,
            'links': links,
            'stats': {
                'total_nodes': len(nodes),
                'total_links': len(links),
                'entity_types': list(set(n['type'] for n in nodes))
            }
        }

    def get_entity_neighborhood(
        self,
        entity_id: str,
        depth: int = 2,
        max_nodes: int = 50
    ) -> Dict:
        """Get entities within N hops of a starting entity."""
        cursor = self._get_cursor()
        
        # Use recursive CTE for graph traversal
        cursor.execute('''
            WITH RECURSIVE neighborhood AS (
                -- Start node
                SELECT id, 0 as depth
                FROM kg_entities
                WHERE id = %s AND matter_id = %s
                
                UNION
                
                -- Connected nodes
                SELECT DISTINCT
                    CASE 
                        WHEN e.source_entity_id = n.id THEN e.target_entity_id
                        ELSE e.source_entity_id
                    END as id,
                    n.depth + 1 as depth
                FROM neighborhood n
                JOIN kg_edges e ON (e.source_entity_id = n.id OR e.target_entity_id = n.id)
                WHERE n.depth < %s AND e.matter_id = %s
                LIMIT %s
            )
            SELECT DISTINCT e.id, e.canonical_name, e.type, e.properties, e.confidence
            FROM neighborhood n
            JOIN kg_entities e ON e.id = n.id
            WHERE e.status = 'active'
        ''', (entity_id, self.matter_id, depth, self.matter_id, max_nodes))
        
        entities = cursor.fetchall()
        entity_ids = {str(e['id']) for e in entities}
        
        # Build nodes
        nodes = []
        id_to_index = {}
        for i, e in enumerate(entities):
            entity_id_str = str(e['id'])
            id_to_index[entity_id_str] = i
            nodes.append({
                'id': i,
                'entity_id': entity_id_str,
                'name': e['canonical_name'][:50],
                'full_name': e['canonical_name'],
                'type': e['type'],
                'color': self.TYPE_COLORS.get(e['type'], '#999999'),
                'properties': e['properties'] if isinstance(e['properties'], dict) else {}
            })
        
        # Get edges
        if entity_ids:
            entity_ids_list = list(entity_ids)
            cursor.execute('''
                SELECT source_entity_id, target_entity_id, relation_type, confidence
                FROM kg_edges
                WHERE matter_id = %s
                  AND source_entity_id = ANY(%s::uuid[])
                  AND target_entity_id = ANY(%s::uuid[])
            ''', (self.matter_id, entity_ids_list, entity_ids_list))
            
            edges = cursor.fetchall()
        else:
            edges = []
        
        # Build links
        links = []
        for e in edges:
            source_idx = id_to_index.get(str(e['source_entity_id']))
            target_idx = id_to_index.get(str(e['target_entity_id']))
            if source_idx is not None and target_idx is not None:
                links.append({
                    'source': source_idx,
                    'target': target_idx,
                    'relation': e['relation_type'],
                    'confidence': e['confidence']
                })
        
        return {'nodes': nodes, 'links': links}

    def search_entities(self, query: str, limit: int = 20) -> List[Dict]:
        """Search entities by name."""
        cursor = self._get_cursor()
        search_pattern = f"%{query}%"
        
        cursor.execute('''
            SELECT DISTINCT e.id, e.canonical_name, e.type, e.properties
            FROM kg_entities e
            LEFT JOIN kg_aliases a ON e.id = a.entity_id
            WHERE e.matter_id = %s
              AND e.status = 'active'
              AND (e.canonical_name ILIKE %s OR a.alias_text ILIKE %s)
            LIMIT %s
        ''', (self.matter_id, search_pattern, search_pattern, limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': str(row['id']),
                'name': row['canonical_name'],
                'type': row['type'],
                'properties': row['properties'] if isinstance(row['properties'], dict) else {}
            })
        
        return results

    def get_stats(self) -> Dict:
        """Get graph statistics."""
        cursor = self._get_cursor()
        
        # Entity count
        cursor.execute('''
            SELECT COUNT(*) as cnt FROM kg_entities 
            WHERE matter_id = %s AND status = 'active'
        ''', (self.matter_id,))
        entity_count = cursor.fetchone()['cnt']
        
        # Edge count
        cursor.execute('''
            SELECT COUNT(*) as cnt FROM kg_edges 
            WHERE matter_id = %s
        ''', (self.matter_id,))
        edge_count = cursor.fetchone()['cnt']
        
        # Document count
        cursor.execute('''
            SELECT COUNT(*) as cnt FROM kg_documents 
            WHERE matter_id = %s
        ''', (self.matter_id,))
        doc_count = cursor.fetchone()['cnt']
        
        # Entity types
        cursor.execute('''
            SELECT type, COUNT(*) as cnt FROM kg_entities 
            WHERE matter_id = %s AND status = 'active' 
            GROUP BY type
        ''', (self.matter_id,))
        
        type_counts = {}
        for row in cursor.fetchall():
            type_counts[row['type']] = row['cnt']
        
        return {
            'total_entities': entity_count,
            'total_edges': edge_count,
            'total_documents': doc_count,
            'entities_by_type': type_counts,  # Added for API compatibility
            'type_colors': self.TYPE_COLORS    # Added for API compatibility
        }


# Alias for backwards compatibility
GraphExporter = PostgreSQLGraphExporter
