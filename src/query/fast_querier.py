"""
Fast Knowledge Graph Querier Module
⚡ Optimized for sub-second query performance using Neo4j fulltext indexes

Based on logic from single_file.py with inverted index optimization
"""

import logging
import time
import hashlib
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


class FastKGQuerier:
    """
    Optimized Knowledge Graph Querier for sub-second responses
    
    Features:
    - Fulltext search using Neo4j inverted indexes
    - Query result caching
    - Confidence-based filtering
    - Neighborhood exploration
    - Performance monitoring
    """
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, 
                 config: Optional[Dict] = None):
        """Initialize FastKGQuerier"""
        self.config = config or {}
        
        # Connection settings
        max_lifetime = self.config.get('max_connection_lifetime', 3600)
        pool_size = self.config.get('max_connection_pool_size', 50)
        
        self.driver = GraphDatabase.driver(
            neo4j_uri, 
            auth=(neo4j_user, neo4j_password),
            max_connection_lifetime=max_lifetime,
            max_connection_pool_size=pool_size
        )
        
        # Cache settings
        self.enable_cache = self.config.get('enable_cache', True)
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_cache_size = self.config.get('cache_size', 1000)
        
        # Query settings
        self.min_confidence = self.config.get('min_confidence', 0.6)
        self.default_top_k = self.config.get('default_top_k', 15)
        self.max_neighbors = self.config.get('max_neighbors', 5)
        
        logger.info("✅ FastKGQuerier initialized")
    
    def ensure_indexes(self):
        """Create all necessary indexes for fast queries"""
        with self.driver.session() as session:
            indexes = [
                # Regular indexes
                "CREATE INDEX entity_text_idx IF NOT EXISTS FOR (e:Entity) ON (e.text)",
                "CREATE INDEX entity_label_idx IF NOT EXISTS FOR (e:Entity) ON (e.label)",
                "CREATE INDEX entity_block_idx IF NOT EXISTS FOR (e:Entity) ON (e.block_id)",
                
                # ⚡ Fulltext index for semantic search (CRITICAL for performance)
                """CREATE FULLTEXT INDEX entity_fulltext_idx IF NOT EXISTS 
                   FOR (e:Entity) ON EACH [e.text, e.label]"""
            ]
            
            for idx_query in indexes:
                try:
                    session.run(idx_query)
                    logger.info(f"✓ Index created/verified")
                except Exception as e:
                    logger.warning(f"Index creation warning: {e}")
            
            logger.info("✅ All Neo4j indexes created/verified")
    
    def _get_cache_key(self, prefix: str, *args) -> str:
        """Generate cache key from query parameters"""
        key_str = f"{prefix}_{'_'.join(map(str, args))}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def semantic_query(self, query_text: str, top_k: Optional[int] = None, 
                      include_neighbors: bool = True, use_cache: Optional[bool] = None,
                      min_confidence: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Fast semantic query optimized for chatbot responses
        Target: <500ms response time
        """
        # Apply defaults
        if top_k is None:
            top_k = self.default_top_k
        if use_cache is None:
            use_cache = self.enable_cache
        if min_confidence is None:
            min_confidence = self.min_confidence
        
        # Check cache
        cache_key = self._get_cache_key("semantic", query_text, top_k, include_neighbors, min_confidence)
        
        if use_cache and cache_key in self.cache:
            self.cache_hits += 1
            logger.debug(f"⚡ Cache hit for query: {query_text[:50]}")
            return self.cache[cache_key]
        
        self.cache_misses += 1
        start = time.time()
        
        with self.driver.session() as session:
            if include_neighbors:
                # Query with neighbors using fulltext index
                query = """
                CALL db.index.fulltext.queryNodes('entity_fulltext_idx', $query_text)
                YIELD node, score
                WHERE node.confidence >= $min_confidence
                WITH node, score
                ORDER BY score DESC
                LIMIT $top_k
                
                OPTIONAL MATCH (node)-[r:RELATED]-(connected)
                WHERE r.confidence > $min_confidence
                WITH node, score, 
                     collect(DISTINCT {
                         text: connected.text,
                         type: r.type,
                         confidence: r.confidence
                     })[..$max_neighbors] as neighbors
                
                RETURN node.id as id,
                       node.text as entity,
                       node.label as label,
                       node.confidence as entity_confidence,
                       score,
                       neighbors
                ORDER BY score DESC
                """
            else:
                # Query without neighbors
                query = """
                CALL db.index.fulltext.queryNodes('entity_fulltext_idx', $query_text)
                YIELD node, score
                WHERE node.confidence >= $min_confidence
                WITH node, score
                ORDER BY score DESC
                LIMIT $top_k
                RETURN node.id as id,
                    node.text as entity,
                    node.label as label,
                    node.confidence as entity_confidence,
                    score,
                    [] as neighbors
                """
            
            result = session.run(
                query, 
                query_text=query_text, 
                top_k=top_k,
                min_confidence=min_confidence,
                max_neighbors=self.max_neighbors
            )
            data = result.data()
        
        elapsed = time.time() - start
        logger.info(f"⚡ Query completed in {elapsed*1000:.0f}ms - Found {len(data)} results")
        
        # Cache result if enabled
        if use_cache:
            # Implement LRU-style cache eviction if needed
            if len(self.cache) >= self.max_cache_size:
                # Remove oldest entries
                keys_to_remove = list(self.cache.keys())[:len(self.cache) // 4]
                for key in keys_to_remove:
                    del self.cache[key]
            
            self.cache[cache_key] = data
        
        return data
    
    def get_entity_neighborhood(self, entity_id: str, limit: int = 20,
                               min_confidence: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Get immediate neighbors of an entity (depth=1 for speed)"""
        if min_confidence is None:
            min_confidence = self.min_confidence
        
        start = time.time()
        
        with self.driver.session() as session:
            query = """
            MATCH (e:Entity {id: $entity_id})
            OPTIONAL MATCH (e)-[r:RELATED]-(connected)
            WHERE r.confidence > $min_confidence
            WITH e, connected, r
            ORDER BY r.confidence DESC
            LIMIT $limit
            RETURN e.text as source,
                   e.label as source_label,
                   collect({
                       text: connected.text,
                       label: connected.label,
                       relationship: r.type,
                       confidence: r.confidence
                   }) as neighbors
            """
            
            result = session.run(
                query, 
                entity_id=entity_id, 
                limit=limit,
                min_confidence=min_confidence
            )
            data = result.single()
        
        elapsed = time.time() - start
        logger.info(f"⚡ Neighborhood query completed in {elapsed*1000:.0f}ms")
        
        return data
    
    def find_path(self, entity1_id: str, entity2_id: str, max_depth: int = 3) -> List[Dict[str, Any]]:
        """Find shortest path between two entities"""
        start = time.time()
        
        with self.driver.session() as session:
            query = """
            MATCH path = shortestPath(
                (e1:Entity {id: $entity1_id})-[*1..$max_depth]-(e2:Entity {id: $entity2_id})
            )
            RETURN [node in nodes(path) | {
                id: node.id,
                text: node.text,
                label: node.label
            }] as nodes,
            [rel in relationships(path) | {
                type: rel.type,
                confidence: rel.confidence
            }] as relationships,
            length(path) as path_length
            LIMIT 5
            """
            
            result = session.run(
                query,
                entity1_id=entity1_id,
                entity2_id=entity2_id,
                max_depth=max_depth
            )
            paths = result.data()
        
        elapsed = time.time() - start
        logger.info(f"⚡ Path finding completed in {elapsed*1000:.0f}ms - Found {len(paths)} paths")
        
        return paths
    
    def format_for_llm(self, query_results: List[Dict[str, Any]], 
                       max_length: int = 2000) -> str:
        """Format query results for LLM consumption"""
        context_parts = []
        char_count = 0
        
        for item in query_results:
            entity = item['entity']
            label = item.get('label', 'Entity')
            neighbors = item.get('neighbors', [])
            
            if neighbors:
                neighbor_texts = [f"{n['text']} (via {n['type']})" for n in neighbors[:3]]
                line = f"• {entity} ({label}): connected to {', '.join(neighbor_texts)}"
            else:
                line = f"• {entity} ({label})"
            
            if char_count + len(line) > max_length:
                break
            
            context_parts.append(line)
            char_count += len(line) + 1  # +1 for newline
        
        return "\n".join(context_parts)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get query performance and cache statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        
        return {
            'cache_size': len(self.cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': f"{hit_rate:.1f}%",
            'total_queries': total
        }
    
    def clear_cache(self):
        """Clear query cache and reset statistics"""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Cache cleared")
    
    def test_connection(self) -> bool:
        """Test Neo4j connection"""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
            logger.info("✅ Neo4j connection test successful")
            return True
        except Exception as e:
            logger.error(f"❌ Neo4j connection test failed: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        with self.driver.session() as session:
            stats = {}
            
            # Total entities
            result = session.run("MATCH (n:Entity) RETURN count(n) as count")
            stats['total_entities'] = result.single()['count']
            
            # Total relationships
            result = session.run("MATCH ()-[r:RELATED]->() RETURN count(r) as count")
            stats['total_relationships'] = result.single()['count']
            
            # Entity types distribution
            result = session.run("""
                MATCH (n:Entity)
                RETURN n.label as type, count(*) as count
                ORDER BY count DESC
                LIMIT 10
            """)
            stats['top_entity_types'] = result.data()
            
            # Relationship types distribution
            result = session.run("""
                MATCH ()-[r:RELATED]->()
                RETURN r.type as type, count(*) as count
                ORDER BY count DESC
                LIMIT 10
            """)
            stats['top_relationship_types'] = result.data()
            
        return stats
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()