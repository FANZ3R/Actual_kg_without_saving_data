#!/usr/bin/env python3
"""
Fast CSV to Neo4j Importer
⚡ Optimized batch import with fulltext indexes

Usage:
    python scripts/import_to_neo4j.py \
      --entities data/output/entities/entities.csv \
      --relationships data/output/relationships/relationships.csv
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from neo4j import GraphDatabase
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class FastCSVImporter:
    """Fast CSV to Neo4j importer using optimized batch processing"""

    def __init__(self, uri, username, password, batch_size=5000):
        self.uri = uri
        self.username = username
        self.password = password
        self.batch_size = batch_size
        self.driver = None

    def connect(self):
        """Connect to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            with self.driver.session() as session:
                session.run("RETURN 1")
            print(f"✅ Connected to Neo4j at {self.uri}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            return False

    def disconnect(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            print("✅ Disconnected from Neo4j")

    def clear_database(self):
        """Clear all nodes and relationships"""
        print("⚠️  Clearing existing data...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("✅ Database cleared")

    def create_constraints(self):
        """Create database constraints"""
        print("Creating constraints...")
        with self.driver.session() as session:
            try:
                session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
                print("✅ Constraints created")
            except Exception as e:
                print(f"⚠️  Constraint warning: {e}")

    def create_indexes(self):
        """⚡ Create optimized indexes for fast queries"""
        print("Creating indexes for fast queries...")
        with self.driver.session() as session:
            indexes = [
                ("CREATE INDEX entity_text_idx IF NOT EXISTS FOR (e:Entity) ON (e.text)", "text"),
                ("CREATE INDEX entity_label_idx IF NOT EXISTS FOR (e:Entity) ON (e.label)", "label"),
                ("CREATE INDEX entity_block_idx IF NOT EXISTS FOR (e:Entity) ON (e.block_id)", "block_id"),
                ("""CREATE FULLTEXT INDEX entity_fulltext_idx IF NOT EXISTS 
                   FOR (e:Entity) ON EACH [e.text, e.label]""", "fulltext")
            ]
            
            for idx_query, name in indexes:
                try:
                    session.run(idx_query)
                    print(f"  ✓ Created {name} index")
                except Exception as e:
                    print(f"  ⚠️  {name} index: {e}")
        
        print("✅ All indexes created!")

    def import_entities(self, csv_path, confidence_threshold=0.5):
        """Import entities from CSV file"""
        print(f"\n{'='*80}")
        print(f"IMPORTING ENTITIES FROM: {csv_path}")
        print(f"{'='*80}")

        # Read CSV
        print("📖 Reading CSV file...")
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} entities")

        # Filter by confidence
        if confidence_threshold > 0:
            df = df[df['confidence'] >= confidence_threshold]
            print(f"✅ Filtered to {len(df)} entities (confidence >= {confidence_threshold})")

        if len(df) == 0:
            print("⚠️  No entities to import after filtering")
            return 0

        # Import in batches
        print(f"\n🚀 Importing entities in batches of {self.batch_size}...")
        total_imported = 0

        with self.driver.session() as session:
            for i in tqdm(range(0, len(df), self.batch_size), desc="Importing entities"):
                batch = df.iloc[i:i + self.batch_size]

                # Prepare batch data
                entities_batch = []
                for _, row in batch.iterrows():
                    entities_batch.append({
                        'id': str(row['id']),
                        'text': str(row['text']),
                        'label': str(row['label']),
                        'confidence': float(row['confidence']),
                        'source': str(row['source']),
                        'block_id': str(row.get('block_id', 'unknown')),
                        'lemma': str(row.get('lemma', ''))
                    })

                # Batch insert using UNWIND
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

                session.run(query, entities=entities_batch)
                total_imported += len(entities_batch)

        print(f"\n✅ Imported {total_imported} entities")
        return total_imported

    def import_relationships(self, csv_path, confidence_threshold=0.5):
        """Import relationships from CSV file"""
        print(f"\n{'='*80}")
        print(f"IMPORTING RELATIONSHIPS FROM: {csv_path}")
        print(f"{'='*80}")

        # Read CSV
        print("📖 Reading CSV file...")
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} relationships")

        # Filter by confidence
        if confidence_threshold > 0:
            df = df[df['confidence'] >= confidence_threshold]
            print(f"✅ Filtered to {len(df)} relationships (confidence >= {confidence_threshold})")

        if len(df) == 0:
            print("⚠️  No relationships to import after filtering")
            return 0

        # Import in batches
        print(f"\n🚀 Importing relationships in batches of {self.batch_size}...")
        total_imported = 0

        with self.driver.session() as session:
            for i in tqdm(range(0, len(df), self.batch_size), desc="Importing relationships"):
                batch = df.iloc[i:i + self.batch_size]

                # Prepare batch data
                rels_batch = []
                for _, row in batch.iterrows():
                    rels_batch.append({
                        'subject_id': str(row['subject_id']),
                        'object_id': str(row['object_id']),
                        'type': str(row['discovered_type']),
                        'confidence': float(row['confidence']),
                        'sources': str(row.get('sources', '')),
                        'block_id': str(row.get('block_id', 'unknown'))
                    })

                # Batch insert using UNWIND
                query = """
                UNWIND $relationships AS rel
                MATCH (s:Entity {id: rel.subject_id})
                MATCH (o:Entity {id: rel.object_id})
                CREATE (s)-[r:RELATED {
                    type: rel.type,
                    confidence: rel.confidence,
                    sources: rel.sources,
                    block_id: rel.block_id
                }]->(o)
                """

                session.run(query, relationships=rels_batch)
                total_imported += len(rels_batch)

        print(f"\n✅ Imported {total_imported} relationships")
        return total_imported

    def get_statistics(self):
        """Get database statistics"""
        print(f"\n{'='*80}")
        print("DATABASE STATISTICS")
        print(f"{'='*80}")

        with self.driver.session() as session:
            # Count nodes
            result = session.run("MATCH (n:Entity) RETURN count(n) as count")
            node_count = result.single()['count']
            print(f"Total Entities: {node_count:,}")

            # Count relationships
            result = session.run("MATCH ()-[r:RELATED]->() RETURN count(r) as count")
            rel_count = result.single()['count']
            print(f"Total Relationships: {rel_count:,}")

            # Top entity labels
            result = session.run("""
                MATCH (n:Entity)
                RETURN n.label as label, count(*) as count
                ORDER BY count DESC
                LIMIT 10
            """)
            print("\nTop Entity Types:")
            for record in result:
                print(f"  {record['label']}: {record['count']:,}")

            # Top relationship types
            result = session.run("""
                MATCH ()-[r:RELATED]->()
                RETURN r.type as type, count(*) as count
                ORDER BY count DESC
                LIMIT 10
            """)
            print("\nTop Relationship Types:")
            for record in result:
                print(f"  {record['type']}: {record['count']:,}")

        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Fast CSV to Neo4j Importer with Optimized Indexes",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--entities', required=True, help='Path to entities CSV file')
    parser.add_argument('--relationships', required=True, help='Path to relationships CSV file')
    parser.add_argument('--uri', default='bolt://localhost:7687', help='Neo4j URI')
    parser.add_argument('--username', default='neo4j', help='Neo4j username')
    parser.add_argument('--password', default='password', help='Neo4j password')
    parser.add_argument('--confidence', type=float, default=0.6, help='Minimum confidence threshold')
    parser.add_argument('--batch-size', type=int, default=5000, help='Batch size for imports')
    parser.add_argument('--no-clear', action='store_true', help='Do not clear existing data')
    parser.add_argument('--create-indexes', action='store_true', default=True, help='Create optimized indexes')

    args = parser.parse_args()

    # Verify files exist
    if not Path(args.entities).exists():
        print(f"❌ Entities file not found: {args.entities}")
        return 1

    if not Path(args.relationships).exists():
        print(f"❌ Relationships file not found: {args.relationships}")
        return 1

    # Initialize importer
    print("="*80)
    print("⚡ FAST CSV TO NEO4J IMPORTER")
    print("="*80)
    print(f"Entities CSV: {args.entities}")
    print(f"Relationships CSV: {args.relationships}")
    print(f"Neo4j URI: {args.uri}")
    print(f"Confidence Threshold: {args.confidence}")
    print(f"Batch Size: {args.batch_size}")
    print("="*80 + "\n")

    importer = FastCSVImporter(
        uri=args.uri,
        username=args.username,
        password=args.password,
        batch_size=args.batch_size
    )

    try:
        # Connect
        if not importer.connect():
            return 1

        # Clear database if requested
        if not args.no_clear:
            importer.clear_database()

        # Create constraints
        importer.create_constraints()

        # Import entities
        start_time = time.time()
        entities_imported = importer.import_entities(args.entities, args.confidence)
        entities_time = time.time() - start_time

        # Import relationships
        start_time = time.time()
        rels_imported = importer.import_relationships(args.relationships, args.confidence)
        rels_time = time.time() - start_time

        # Create indexes for fast queries
        if args.create_indexes:
            importer.create_indexes()

        # Get statistics
        importer.get_statistics()

        # Print summary
        print("="*80)
        print("✅ IMPORT COMPLETE")
        print("="*80)
        print(f"Entities Imported: {entities_imported:,} (in {entities_time:.2f}s)")
        print(f"Relationships Imported: {rels_imported:,} (in {rels_time:.2f}s)")
        print(f"\n✅ Success! View your graph at: http://localhost:7474")
        print("\n⚡ Test fast queries:")
        print("  python scripts/test_query.py --query 'your search'")
        print("="*80)

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        importer.disconnect()


if __name__ == "__main__":
    sys.exit(main())