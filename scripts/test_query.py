#!/usr/bin/env python3
"""
Test Fast Query System
Demonstrates sub-second query performance with the FastKGQuerier

Usage:
    python scripts/test_query.py
    python scripts/test_query.py --query "your search query"
    python scripts/test_query.py --benchmark
"""

import sys
import argparse
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.query import FastKGQuerier, IndexBuilder


def test_connection(querier: FastKGQuerier):
    """Test Neo4j connection"""
    print("\n" + "="*60)
    print("TESTING CONNECTION")
    print("="*60)
    
    if querier.test_connection():
        print("✅ Connection successful!")
        
        # Get database stats
        stats = querier.get_database_stats()
        print(f"\nDatabase Statistics:")
        print(f"  Total Entities: {stats.get('total_entities', 0):,}")
        print(f"  Total Relationships: {stats.get('total_relationships', 0):,}")
        
        if stats.get('top_entity_types'):
            print(f"\nTop Entity Types:")
            for item in stats['top_entity_types'][:5]:
                print(f"  • {item['type']}: {item['count']:,}")
        
        return True
    else:
        print("❌ Connection failed!")
        return False


def test_semantic_query(querier: FastKGQuerier, query_text: str, top_k: int = 10):
    """Test semantic search query"""
    print("\n" + "="*60)
    print(f"SEMANTIC QUERY: {query_text}")
    print("="*60)
    
    start = time.time()
    results = querier.semantic_query(
        query_text=query_text,
        top_k=top_k,
        include_neighbors=True
    )
    elapsed = (time.time() - start) * 1000
    
    print(f"\n⚡ Query completed in {elapsed:.0f}ms")
    print(f"Found {len(results)} results:\n")
    
    for idx, item in enumerate(results[:5], 1):
        print(f"{idx}. {item['entity']} ({item['label']})")
        print(f"   Score: {item['score']:.3f}, Confidence: {item.get('entity_confidence', 0):.2f}")
        
        neighbors = item.get('neighbors', [])
        if neighbors:
            print(f"   Connected to:")
            for neighbor in neighbors[:3]:
                print(f"     → {neighbor['text']} via {neighbor['type']} (conf: {neighbor['confidence']:.2f})")
        print()


def test_cache_performance(querier: FastKGQuerier, query_text: str):
    """Test cache performance"""
    print("\n" + "="*60)
    print("TESTING CACHE PERFORMANCE")
    print("="*60)
    
    # Clear cache first
    querier.clear_cache()
    print("\nCache cleared")
    
    # First query - cache miss
    print(f"\nQuery 1 (cache miss): {query_text}")
    start = time.time()
    results1 = querier.semantic_query(query_text, top_k=10, use_cache=True)
    elapsed1 = (time.time() - start) * 1000
    print(f"⚡ Time: {elapsed1:.0f}ms")
    
    # Second query - cache hit
    print(f"\nQuery 2 (cache hit): {query_text}")
    start = time.time()
    results2 = querier.semantic_query(query_text, top_k=10, use_cache=True)
    elapsed2 = (time.time() - start) * 1000
    print(f"⚡ Time: {elapsed2:.0f}ms")
    
    # Statistics
    stats = querier.get_statistics()
    print(f"\nCache Statistics:")
    print(f"  Cache Size: {stats['cache_size']}")
    print(f"  Hit Rate: {stats['hit_rate']}")
    print(f"  Speedup: {elapsed1/elapsed2:.1f}x faster")


def run_benchmark(querier: FastKGQuerier):
    """Run comprehensive benchmark"""
    print("\n" + "="*60)
    print("RUNNING BENCHMARK")
    print("="*60)
    
    test_queries = [
        "risk management",
        "supplier relationship",
        "compliance procedures",
        "business process",
        "quality control"
    ]
    
    times = []
    
    # Warm up
    print("\nWarming up...")
    querier.semantic_query("test", top_k=5)
    
    # Run benchmark
    print("\nRunning queries...")
    for query in test_queries:
        start = time.time()
        results = querier.semantic_query(query, top_k=10, use_cache=False)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        print(f"  '{query}': {elapsed:.0f}ms ({len(results)} results)")
    
    # Statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\nBenchmark Results:")
    print(f"  Average: {avg_time:.0f}ms")
    print(f"  Min: {min_time:.0f}ms")
    print(f"  Max: {max_time:.0f}ms")
    
    if avg_time < 500:
        print(f"\n✅ Excellent! Average query time is under 500ms target")
    else:
        print(f"\n⚠️  Warning: Average query time exceeds 500ms target")


def ensure_indexes(uri: str, user: str, password: str):
    """Ensure all indexes are created"""
    print("\n" + "="*60)
    print("ENSURING INDEXES")
    print("="*60)
    
    with IndexBuilder(uri, user, password) as builder:
        builder.create_all_indexes(include_fulltext=True)
        
        # List indexes
        indexes = builder.list_indexes()
        print(f"\nTotal indexes: {len(indexes)}")
        for idx in indexes:
            print(f"  • {idx.get('name', 'unknown')}: {idx.get('state', 'unknown')}")


def main():
    parser = argparse.ArgumentParser(
        description="Test Fast Query System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--uri', default='bolt://localhost:7687', help='Neo4j URI')
    parser.add_argument('--user', default='neo4j', help='Neo4j username')
    parser.add_argument('--password', default='password', help='Neo4j password')
    parser.add_argument('--query', type=str, help='Test query text')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    parser.add_argument('--ensure-indexes', action='store_true', help='Ensure indexes are created')
    parser.add_argument('--top-k', type=int, default=10, help='Number of results')
    
    args = parser.parse_args()
    
    print("="*60)
    print("FAST QUERY SYSTEM TEST")
    print("="*60)
    print(f"Neo4j URI: {args.uri}")
    print("="*60)
    
    try:
        # Ensure indexes first if requested
        if args.ensure_indexes:
            ensure_indexes(args.uri, args.user, args.password)
        
        # Initialize querier
        print("\nInitializing FastKGQuerier...")
        querier = FastKGQuerier(
            neo4j_uri=args.uri,
            neo4j_user=args.user,
            neo4j_password=args.password,
            config={
                'enable_cache': True,
                'min_confidence': 0.6,
                'default_top_k': args.top_k
            }
        )
        
        # Test connection
        if not test_connection(querier):
            print("\n❌ Cannot proceed without database connection")
            return 1
        
        # Ensure indexes
        print("\nEnsuring indexes are created...")
        querier.ensure_indexes()
        
        # Run tests based on arguments
        if args.benchmark:
            run_benchmark(querier)
        elif args.query:
            test_semantic_query(querier, args.query, args.top_k)
            test_cache_performance(querier, args.query)
        else:
            # Default test suite
            print("\nRunning default test suite...")
            
            # Test semantic query
            test_semantic_query(querier, "supplier risk management", top_k=10)
            
            # Test cache performance
            test_cache_performance(querier, "compliance procedures")
        
        # Final statistics
        print("\n" + "="*60)
        print("FINAL STATISTICS")
        print("="*60)
        stats = querier.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Close connection
        querier.close()
        
        print("\n✅ All tests completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return 130
        
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())