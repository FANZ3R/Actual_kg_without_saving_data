"""
Index Builder Module
Utilities for creating and managing Neo4j indexes for fast queries
"""

import logging
from typing import Dict, Any, List
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


class IndexBuilder:
    """Manages Neo4j index creation and optimization"""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        """Initialize IndexBuilder"""
        self.driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password)
        )
        logger.info("IndexBuilder initialized")
    
    def create_all_indexes(self, include_fulltext: bool = True):
        """Create all recommended indexes for optimal query performance"""
        logger.info("Creating indexes...")
        
        with self.driver.session() as session:
            # Constraint for unique entity IDs
            try:
                session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
                logger.info("✓ Created entity ID constraint")
            except Exception as e:
                logger.warning(f"Constraint creation: {e}")
            
            # Regular indexes
            indexes = [
                ("CREATE INDEX entity_text_idx IF NOT EXISTS FOR (e:Entity) ON (e.text)", "text"),
                ("CREATE INDEX entity_label_idx IF NOT EXISTS FOR (e:Entity) ON (e.label)", "label"),
                ("CREATE INDEX entity_block_idx IF NOT EXISTS FOR (e:Entity) ON (e.block_id)", "block_id"),
                ("CREATE INDEX entity_confidence_idx IF NOT EXISTS FOR (e:Entity) ON (e.confidence)", "confidence"),
            ]
            
            for idx_query, name in indexes:
                try:
                    session.run(idx_query)
                    logger.info(f"✓ Created {name} index")
                except Exception as e:
                    logger.warning(f"Index {name} creation: {e}")
            
            # Fulltext index (critical for fast semantic search)
            if include_fulltext:
                try:
                    session.run("""
                        CREATE FULLTEXT INDEX entity_fulltext_idx IF NOT EXISTS 
                        FOR (e:Entity) ON EACH [e.text, e.label]
                    """)
                    logger.info("✓ Created fulltext search index")
                except Exception as e:
                    logger.warning(f"Fulltext index creation: {e}")
        
        logger.info("✅ All indexes created successfully")
    
    def list_indexes(self) -> List[Dict[str, Any]]:
        """List all indexes in the database"""
        with self.driver.session() as session:
            result = session.run("SHOW INDEXES")
            indexes = result.data()
        
        logger.info(f"Found {len(indexes)} indexes")
        return indexes
    
    def drop_all_indexes(self):
        """Drop all indexes (use with caution)"""
        logger.warning("Dropping all indexes...")
        
        with self.driver.session() as session:
            # Get all index names
            result = session.run("SHOW INDEXES")
            indexes = result.data()
            
            for idx in indexes:
                idx_name = idx.get('name')
                if idx_name:
                    try:
                        session.run(f"DROP INDEX {idx_name} IF EXISTS")
                        logger.info(f"✓ Dropped index: {idx_name}")
                    except Exception as e:
                        logger.warning(f"Failed to drop {idx_name}: {e}")
        
        logger.info("✅ All indexes dropped")
    
    def analyze_index_usage(self) -> Dict[str, Any]:
        """Analyze index usage statistics"""
        with self.driver.session() as session:
            result = session.run("""
                SHOW INDEXES
                YIELD name, type, state, populationPercent
                RETURN name, type, state, populationPercent
            """)
            indexes = result.data()
        
        analysis = {
            'total_indexes': len(indexes),
            'online_indexes': sum(1 for idx in indexes if idx.get('state') == 'ONLINE'),
            'indexes': indexes
        }
        
        return analysis
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("IndexBuilder connection closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()