"""
Optimized Neo4j Connector Module
⚡ Fast batch import with index optimization
Based on single_file.py logic
"""

import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


class OptimizedNeo4jConnector:
    """
    Optimized Neo4j connector with fast batch import
    
    Features:
    - Batch processing for large datasets
    - Index creation after data import (faster)
    - Confidence-based filtering
    - Progress tracking
    - Connection pooling
    """
    
    def __init__(self, uri: str, username: str, password: str, config: Dict = None):
        """Initialize OptimizedNeo4jConnector"""
        self.uri = uri
        self.username = username
        self.password = password
        self.config = config or {}
        self.driver = None
        self.logger = logger
        
        # ⚡ Optimization settings
        self.entity_batch_size = self.config.get('entity_batch_size', 5000)
        self.relationship_batch_size = self.config.get('relationship_batch_size', 2000)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.clear_on_import = self.config.get('clear_database', True)
        self.create_indexes = self.config.get('create_indexes', True)
        
        # Connection pooling settings
        max_lifetime = self.config.get('max_connection_lifetime', 3600)
        pool_size = self.config.get('max_connection_pool_size', 50)
        
        self.connection_config = {
            'max_connection_lifetime': max_lifetime,
            'max_connection_pool_size': pool_size
        }
    
    def connect(self):
        """Establish connection to Neo4j with optimized settings"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password),
                **self.connection_config
            )
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
            self.logger.info(f"✅ Connected to Neo4j at {self.uri}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Neo4j: {e}")
            return False
    
    def disconnect(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            self.logger.info("Disconnected from Neo4j")
    
    def export_knowledge_graph(self, entities: List[Dict[str, Any]],
                               relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        ⚡ Optimized export with batch processing
        
        Strategy:
        1. Clear database (optional)
        2. Create ID constraint only
        3. Batch import entities
        4. Batch import relationships
        5. Create remaining indexes (faster after import)
        """
        if not self.driver:
            if not self.connect():
                raise ConnectionError("Failed to connect to Neo4j")
        
        try:
            with self.driver.session() as session:
                # Step 1: Clear database if configured
                if self.clear_on_import:
                    self.logger.info("Clearing existing data...")
                    session.run("MATCH (n) DETACH DELETE n")
                    self.logger.info("✅ Database cleared")
                
                # Step 2: Create ID constraint ONLY (before import)
                self.logger.info("Creating entity ID constraint...")
                try:
                    session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
                    self.logger.info("✅ ID constraint created")
                except Exception as e:
                    self.logger.warning(f"Constraint warning: {e}")
                
                # # Step 3: Filter by confidence
                # filtered_entities = [e for e in entities if e.get('confidence', 0) >= self.confidence_threshold]
                # filtered_relationships = [r for r in relationships if r.get('confidence', 0) >= self.confidence_threshold]
                
                # self.logger.info(f"Filtered to {len(filtered_entities):,} entities (threshold: {self.confidence_threshold})")
                # self.logger.info(f"Filtered to {len(filtered_relationships):,} relationships")
                
                # # Step 3: Filter by confidence
                # filtered_entities = [e for e in entities if e.get('confidence', 0) >= self.confidence_threshold]

                # # ⚡ AGGRESSIVE FILTER: Use higher threshold for relationships
                # RELATIONSHIP_THRESHOLD = 0.75  # Higher than general threshold
                # filtered_relationships = [r for r in relationships if r.get('confidence', 0) >= RELATIONSHIP_THRESHOLD]

                # self.logger.info(f"Filtered to {len(filtered_entities):,} entities (threshold: {self.confidence_threshold})")
                # self.logger.info(f"🎯 Filtered to {len(filtered_relationships):,} relationships (threshold: {RELATIONSHIP_THRESHOLD})")
                # self.logger.info(f"   Reduction: {len(relationships):,} → {len(filtered_relationships):,} ({(1-len(filtered_relationships)/len(relationships))*100:.1f}% fewer)")
                
                # Step 3: NO FILTERING - Import everything
                filtered_entities = entities  # Keep all
                filtered_relationships = relationships  # Keep all

                self.logger.info(f"⚡ NO FILTERING - Importing ALL data")
                self.logger.info(f"   Entities: {len(filtered_entities):,}")
                self.logger.info(f"   Relationships: {len(filtered_relationships):,}")
                                
                # Step 4: Batch import entities
                entity_count = self._import_entities_batch(session, filtered_entities)
                
                # Step 5: Batch import relationships
                rel_count = self._import_relationships_batch(session, filtered_relationships)
                
                # Step 6: Create remaining indexes AFTER import (much faster)
                if self.create_indexes:
                    self._create_indexes_optimized(session)
                
                # Get statistics
                stats = self._get_statistics(session)
                stats['entities_imported'] = entity_count
                stats['relationships_imported'] = rel_count
                
                self.logger.info("=" * 60)
                self.logger.info("✅ NEO4J EXPORT COMPLETE")
                self.logger.info("=" * 60)
                self.logger.info(f"Entities: {stats['total_nodes']:,}")
                self.logger.info(f"Relationships: {stats['total_relationships']:,}")
                self.logger.info("=" * 60)
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Error exporting to Neo4j: {e}")
            raise
    
    def _import_entities_batch(self, session, entities: List[Dict[str, Any]]) -> int:
        """⚡ Fast batch import for entities using UNWIND"""
        self.logger.info(f"Importing {len(entities):,} entities in batches of {self.entity_batch_size:,}...")
        
        total_imported = 0
        
        # Prepare batches
        entity_batches = []
        for i in range(0, len(entities), self.entity_batch_size):
            batch = entities[i:i + self.entity_batch_size]
            
            # Prepare batch data
            batch_data = []
            for ent in batch:
                batch_data.append({
                    'id': str(ent.get('id')),
                    'text': str(ent.get('text', '')),
                    'label': str(ent.get('label', 'ENTITY')),
                    'confidence': float(ent.get('confidence', 0)),
                    'source': str(ent.get('source', '')),
                    'block_id': str(ent.get('block_id', '')),
                    'lemma': str(ent.get('lemma', ''))
                })
            
            entity_batches.append(batch_data)
        
        # Import batches with progress bar
        for batch_data in tqdm(entity_batches, desc="Creating entities"):
            query = """
            UNWIND $entities AS entity
            CREATE (e:Entity {
                id: entity.id,
                text: entity.text,
                label: entity.label,
                confidence: entity.confidence,
                source: entity.source,
                block_id: entity.block_id,
                lemma: entity.lemma
            })
            """
            
            session.run(query, entities=batch_data)
            total_imported += len(batch_data)
        
        self.logger.info(f"✅ Imported {total_imported:,} entities")
        return total_imported
    
    def _import_relationships_batch(self, session, relationships: List[Dict[str, Any]]) -> int:
        """⚡ Fast batch import for relationships using UNWIND"""
        self.logger.info(f"Importing {len(relationships):,} relationships in batches of {self.relationship_batch_size:,}...")
        
        total_imported = 0
        
        # Prepare batches
        rel_batches = []
        for i in range(0, len(relationships), self.relationship_batch_size):
            batch = relationships[i:i + self.relationship_batch_size]
            
            # Prepare batch data
            batch_data = []
            for rel in batch:
                subject_id = rel.get('subject', {}).get('id')
                object_id = rel.get('object', {}).get('id')
                
                if subject_id and object_id:
                    rel_type = rel.get('discovered_type', 'RELATED').upper().replace(' ', '_').replace('-', '_')
                    batch_data.append({
                        'subject_id': str(subject_id),
                        'object_id': str(object_id),
                        'rel_type': rel_type,
                        'confidence': float(rel.get('confidence', 0)),
                        'sources': ', '.join(rel.get('sources', [])),
                        'block_id': str(rel.get('block_id', ''))
                    })
            
            if batch_data:
                rel_batches.append(batch_data)
        
        # Import batches with progress bar
        for batch_data in tqdm(rel_batches, desc="Creating relationships"):
            query = """
            UNWIND $batch as rel
            MATCH (s:Entity {id: rel.subject_id})
            MATCH (o:Entity {id: rel.object_id})
            CREATE (s)-[r:RELATED {
                type: rel.rel_type,
                confidence: rel.confidence,
                sources: rel.sources,
                block_id: rel.block_id
            }]->(o)
            """
            
            session.run(query, batch=batch_data)
            total_imported += len(batch_data)
        
        self.logger.info(f"✅ Imported {total_imported:,} relationships")
        return total_imported
    
    def _create_indexes_optimized(self, session):
        """⚡ Create indexes AFTER data import (much faster)"""
        self.logger.info("Creating search indexes (this may take 1-2 minutes)...")
        
        indexes = [
            # Regular indexes
            ("CREATE INDEX entity_text_idx IF NOT EXISTS FOR (e:Entity) ON (e.text)", "text"),
            ("CREATE INDEX entity_label_idx IF NOT EXISTS FOR (e:Entity) ON (e.label)", "label"),
            ("CREATE INDEX entity_block_idx IF NOT EXISTS FOR (e:Entity) ON (e.block_id)", "block_id"),
            ("CREATE INDEX entity_confidence_idx IF NOT EXISTS FOR (e:Entity) ON (e.confidence)", "confidence"),
            
            # ⚡ Fulltext index (CRITICAL for fast semantic search)
            ("""CREATE FULLTEXT INDEX entity_fulltext_idx IF NOT EXISTS 
               FOR (e:Entity) ON EACH [e.text, e.label]""", "fulltext")
        ]
        
        for idx_query, name in indexes:
            try:
                session.run(idx_query)
                self.logger.info(f"  ✓ Created {name} index")
            except Exception as e:
                self.logger.warning(f"Index {name} creation: {e}")
        
        self.logger.info("✅ All indexes created!")
    
    def _get_statistics(self, session) -> Dict[str, Any]:
        """Get database statistics"""
        stats = {}
        
        # Total nodes
        result = session.run("MATCH (n) RETURN count(n) as count")
        stats['total_nodes'] = result.single()['count']
        
        # Total relationships
        result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
        stats['total_relationships'] = result.single()['count']
        
        # Entity types distribution
        result = session.run("""
            MATCH (n:Entity)
            RETURN n.label as type, count(*) as count
            ORDER BY count DESC
            LIMIT 10
        """)
        stats['entity_types'] = result.data()
        
        # Relationship types distribution
        result = session.run("""
            MATCH ()-[r:RELATED]->()
            RETURN r.type as type, count(*) as count
            ORDER BY count DESC
            LIMIT 10
        """)
        stats['relationship_types'] = result.data()
        
        return stats
    
    def test_connection(self) -> bool:
        """Test Neo4j connection"""
        try:
            if not self.driver:
                if not self.connect():
                    return False
            
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
            
            self.logger.info("✅ Neo4j connection test successful")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Neo4j connection test failed: {e}")
            return False
    
    def clear_database(self):
        """Clear all data from Neo4j"""
        if not self.driver:
            if not self.connect():
                raise ConnectionError("Failed to connect to Neo4j")
        
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            self.logger.info("✅ Cleared Neo4j database")
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()