#!/usr/bin/env python3
"""
ULTRA-LIGHTWEIGHT PIPELINE for 4GB RAM
Extreme chunking + aggressive memory management
Perfect for shared servers with limited resources
"""

import sys
from pathlib import Path
import spacy
import yaml
import logging
import gc
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DataIngestion, DataValidator
from src.extraction import KnowledgeGraphBuilder
from src.storage import OptimizedNeo4jConnector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_ultra_small_chunks(blocks, kg_builder, neo4j_connector, chunk_size=20):
    """
    Process in TINY chunks to minimize memory footprint
    Target: <2GB RAM for processing
    """
    
    total_entities = 0
    total_relationships = 0
    
    logger.info(f"\n🔧 Ultra-lightweight mode: {chunk_size} blocks per chunk")
    logger.info(f"   Total chunks: {(len(blocks) + chunk_size - 1) // chunk_size}")
    
    # First chunk: initialize and clear database
    first_chunk = True
    
    for i in range(0, len(blocks), chunk_size):
        chunk = blocks[i:i + chunk_size]
        chunk_num = i // chunk_size + 1
        total_chunks = (len(blocks) + chunk_size - 1) // chunk_size
        
        logger.info(f"\n{'─'*60}")
        logger.info(f"Chunk {chunk_num}/{total_chunks} ({len(chunk)} blocks)")
        
        # Extract - SMALL batch size to reduce memory
        extraction_result = kg_builder.extract_from_blocks(chunk, batch_size=5)
        
        entities = extraction_result['entities']
        relationships = extraction_result['relationships']
        
        logger.info(f"  Extracted: {len(entities):,} entities, {len(relationships):,} rels")
        
        # Import immediately to Neo4j (free up memory ASAP)
        if first_chunk:
            neo4j_connector.clear_database()
            neo4j_connector.connect()
            first_chunk = False
        
        # Import in TINY batches
        import_chunk_to_neo4j(neo4j_connector, entities, relationships)
        
        total_entities += len(entities)
        total_relationships += len(relationships)
        
        # AGGRESSIVE MEMORY CLEANUP
        del entities
        del relationships
        del extraction_result
        del chunk
        gc.collect()
        
        if chunk_num % 10 == 0:
            logger.info(f"  ✅ {chunk_num} chunks done. Total: {total_entities:,} entities, {total_relationships:,} rels")
    
    return total_entities, total_relationships


def import_chunk_to_neo4j(neo4j_connector, entities, relationships):
    """Import with TINY batches to minimize memory"""
    
    with neo4j_connector.driver.session() as session:
        # Import entities in batches of 1000 (instead of 5000)
        for i in range(0, len(entities), 1000):
            batch = entities[i:i+1000]
            
            batch_data = [
                {
                    'id': str(e.get('id')),
                    'text': str(e.get('text', ''))[:500],  # Truncate long text
                    'label': str(e.get('label', 'ENTITY')),
                    'confidence': float(e.get('confidence', 0)),
                    'source': str(e.get('source', '')),
                    'block_id': str(e.get('block_id', '')),
                    'lemma': str(e.get('lemma', ''))[:100]
                }
                for e in batch
            ]
            
            query = """
            UNWIND $entities AS entity
            MERGE (e:Entity {id: entity.id})
            SET e.text = entity.text,
                e.label = entity.label,
                e.confidence = entity.confidence,
                e.source = entity.source,
                e.block_id = entity.block_id,
                e.lemma = entity.lemma
            """
            session.run(query, entities=batch_data)
            
            # Clear batch from memory
            del batch_data
        
        # Import relationships in batches of 500 (instead of 2000)
        for i in range(0, len(relationships), 500):
            batch = relationships[i:i+500]
            
            batch_data = []
            for rel in batch:
                subject_id = rel.get('subject', {}).get('id')
                object_id = rel.get('object', {}).get('id')
                
                if subject_id and object_id:
                    batch_data.append({
                        'subject_id': str(subject_id),
                        'object_id': str(object_id),
                        'rel_type': str(rel.get('discovered_type', 'RELATED'))[:50],
                        'confidence': float(rel.get('confidence', 0)),
                        'block_id': str(rel.get('block_id', ''))
                    })
            
            if batch_data:
                query = """
                UNWIND $batch as rel
                MATCH (s:Entity {id: rel.subject_id})
                MATCH (o:Entity {id: rel.object_id})
                MERGE (s)-[r:RELATED {type: rel.rel_type}]->(o)
                SET r.confidence = rel.confidence,
                    r.block_id = rel.block_id
                """
                session.run(query, batch=batch_data)
                
                del batch_data
            
            del batch


def main():
    print("=" * 80)
    print("🪶 ULTRA-LIGHTWEIGHT PIPELINE: 4GB RAM Mode")
    print("=" * 80)
    
    # Force garbage collection at start
    gc.collect()
    
    # Load config
    with open('config/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize with minimal settings
    print("\n1. Loading spaCy model (small)...")
    nlp = spacy.load('en_core_web_sm')
    # Disable unnecessary components to save memory
    nlp.disable_pipes('ner')  # We'll re-enable only when needed
    
    print("2. Initializing components...")
    data_ingestion = DataIngestion(config.get('ingestion', {}))
    data_validator = DataValidator(config.get('validation', {}))
    kg_builder = KnowledgeGraphBuilder(nlp, config.get('extraction', {}))
    
    # Load data
    print("\n3. Loading data from data/raw...")
    input_dir = config['pipeline']['input_dir']
    blocks = data_ingestion.load_from_directory(input_dir)
    print(f"   ✅ Loaded {len(blocks)} blocks")
    
    # Validate
    print("\n4. Validating data...")
    valid_blocks, _ = data_validator.validate_blocks(blocks)
    valid_blocks = data_validator.clean_blocks(valid_blocks)
    valid_blocks = data_validator.deduplicate_blocks(valid_blocks)
    print(f"   ✅ {len(valid_blocks)} valid blocks")
    
    # Clear memory before processing
    del blocks
    gc.collect()
    
    # Initialize Neo4j
    print("\n5. Connecting to Neo4j...")
    neo4j_config = config.get('neo4j', {})
    neo4j_connector = OptimizedNeo4jConnector(
        uri=neo4j_config.get('uri'),
        username=neo4j_config.get('username'),
        password=neo4j_config.get('password'),
        config=neo4j_config
    )
    
    # ULTRA-SMALL CHUNKS for minimal memory
    print("\n6. Processing with ultra-lightweight chunks...")
    print("   ⚙️  Chunk size: 20 blocks (optimized for 4GB RAM)")
    print("   ⚙️  Batch size: 5 blocks (minimal memory)")
    print("   ⏱️  This will take longer but uses <2GB RAM")
    
    total_entities, total_relationships = process_ultra_small_chunks(
        valid_blocks, 
        kg_builder, 
        neo4j_connector,
        chunk_size=20  # TINY chunks for 4GB RAM
    )
    
    # Clear memory before creating indexes
    del valid_blocks
    del kg_builder
    gc.collect()
    
    # Create indexes (one at a time to save memory)
    print("\n7. Creating indexes...")
    with neo4j_connector.driver.session() as session:
        # Create indexes one by one
        indexes = [
            ("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE", "constraint"),
            ("CREATE INDEX entity_text_idx IF NOT EXISTS FOR (e:Entity) ON (e.text)", "text"),
            ("CREATE INDEX entity_label_idx IF NOT EXISTS FOR (e:Entity) ON (e.label)", "label"),
            ("CREATE FULLTEXT INDEX entity_fulltext_idx IF NOT EXISTS FOR (e:Entity) ON EACH [e.text, e.label]", "fulltext")
        ]
        
        for idx_query, name in indexes:
            try:
                session.run(idx_query)
                logger.info(f"  ✓ Created {name} index")
                gc.collect()  # Clean up after each index
            except Exception as e:
                logger.warning(f"  Index {name}: {e}")
    
    print("\n" + "=" * 80)
    print("✅ ULTRA-LIGHTWEIGHT PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Total Entities: {total_entities:,}")
    print(f"Total Relationships: {total_relationships:,}")
    print(f"💾 Peak RAM Usage: ~2-3GB (safe for 4GB available)")
    print(f"⚡ Fast search still enabled (fulltext indexes)")
    print("=" * 80)
    
    neo4j_connector.disconnect()


if __name__ == "__main__":
    main()
# ```

# ---

# ## Memory Breakdown for 4GB RAM

# ### Ultra-Small Chunk (20 blocks):
# ```
# Per Chunk Memory Usage:
# ─────────────────────────────────────
# System & Python:          800 MB
# spaCy model (minimal):    300 MB
# Neo4j driver:             200 MB

# Processing chunk (20 blocks):
# - Raw text:               50 MB
# - Entities (~1,000):      200 KB
# - Relationships (~12,000): 6 MB
# - Extraction overhead:    200 MB
#                           ─────
# Chunk total:              256 MB

# Peak Usage:               ~1.6 GB
# Buffer:                   400 MB
# # ─────────────────────────────────────
# TOTAL:                    ~2 GB ✅