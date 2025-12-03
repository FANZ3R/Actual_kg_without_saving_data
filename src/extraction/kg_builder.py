"""
Knowledge Graph Builder Module
Main orchestrator for knowledge graph extraction
"""

import logging
from typing import List, Dict, Any
from collections import Counter
from tqdm import tqdm

from .entity_extractor import EntityExtractor
from .relationship_extractor import RelationshipExtractor

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """Main knowledge graph construction class"""

    def __init__(self, nlp_model, config: Dict = None):
        """Initialize KnowledgeGraphBuilder"""
        self.nlp = nlp_model
        self.config = config or {}
        self.entity_extractor = EntityExtractor(config)
        self.relationship_extractor = RelationshipExtractor(config)
        self.logger = logger

    def extract_from_text(self, text: str, block_id: Any = None) -> Dict[str, Any]:
        """Extract knowledge graph from a single text block"""
        # Process text with spaCy
        doc = self.nlp(text)

        # Extract entities
        entities = self.entity_extractor.extract_entities(doc)

        # Extract relationships
        relationships = self.relationship_extractor.extract_relationships(doc, entities)

        # Add block_id to entities and relationships
        if block_id is not None:
            for entity in entities:
                entity['block_id'] = block_id
            for rel in relationships:
                rel['block_id'] = block_id

        # Generate statistics
        stats = self._generate_statistics(entities, relationships, doc)

        return {
            "text": text,
            "entities": entities,
            "relationships": relationships,
            "statistics": stats,
            "linguistic_features": {
                "token_count": len(doc),
                "sentence_count": len(list(doc.sents)),
                "noun_chunks": len(list(doc.noun_chunks))
            }
        }

    def extract_from_blocks(self, blocks: List[Dict[str, Any]], batch_size: int = 50) -> Dict[str, Any]:
        """Extract knowledge graphs from multiple text blocks"""
        all_entities = []
        all_relationships = []
        block_metadata = []

        self.logger.info(f"Processing {len(blocks)} blocks in batches of {batch_size}")

        for i in tqdm(range(0, len(blocks), batch_size), desc="Processing batches"):
            batch = blocks[i:i + batch_size]

            for block_idx, block in enumerate(batch):
                actual_idx = i + block_idx

                # Extract text from block
                if isinstance(block, dict):
                    text = block.get('text', '')
                    block_id = block.get('block_id', f"block_{actual_idx}")
                else:
                    text = str(block)
                    block_id = f"block_{actual_idx}"

                if not text or len(text.strip()) < 10:
                    continue

                try:
                    # Extract knowledge graph
                    kg = self.extract_from_text(text, block_id)

                    # Collect entities and relationships
                    all_entities.extend(kg['entities'])
                    all_relationships.extend(kg['relationships'])

                    # Store metadata
                    block_metadata.append({
                        'block_id': block_id,
                        'statistics': kg['statistics'],
                        'linguistic_features': kg['linguistic_features']
                    })

                except Exception as e:
                    self.logger.error(f"Error processing block {block_id}: {str(e)}")
                    continue

        # Generate overall statistics
        overall_stats = self._generate_overall_statistics(all_entities, all_relationships, block_metadata)

        self.logger.info(f"Extracted {len(all_entities)} entities and {len(all_relationships)} relationships")

        return {
            'entities': all_entities,
            'relationships': all_relationships,
            'block_metadata': block_metadata,
            'overall_statistics': overall_stats
        }

    def _generate_statistics(self, entities: List[Dict], relationships: List[Dict], doc) -> Dict[str, Any]:
        """Generate statistics for a single extraction"""
        stats = {
            "total_entities": len(entities),
            "total_relationships": len(relationships),
            "unique_relationship_types": len(set(r.get("discovered_type", "unknown") for r in relationships)),
            "entity_types": dict(Counter(ent["label"] for ent in entities)),
            "relationship_types": dict(Counter(rel.get("discovered_type", "unknown") for rel in relationships)),
            "discovery_methods": dict(Counter(source for rel in relationships for source in rel.get("sources", []))),
            "avg_confidence": {
                "entities": sum(e["confidence"] for e in entities) / len(entities) if entities else 0,
                "relationships": sum(r["confidence"] for r in relationships) / len(relationships) if relationships else 0
            }
        }

        return stats

    def _generate_overall_statistics(self, entities: List[Dict], relationships: List[Dict],
                                      metadata: List[Dict]) -> Dict[str, Any]:
        """Generate overall statistics for multiple blocks"""
        stats = {
            "total_blocks_processed": len(metadata),
            "total_entities": len(entities),
            "total_relationships": len(relationships),
            "entity_type_distribution": dict(Counter(ent["label"] for ent in entities)),
            "relationship_type_distribution": dict(Counter(rel.get("discovered_type", "unknown") for rel in relationships)),
            "source_distribution": dict(Counter(ent["source"] for ent in entities)),
            "discovery_method_distribution": dict(Counter(source for rel in relationships for source in rel.get("sources", []))),
            "confidence_statistics": {
                "entity_avg": sum(e["confidence"] for e in entities) / len(entities) if entities else 0,
                "entity_min": min((e["confidence"] for e in entities), default=0),
                "entity_max": max((e["confidence"] for e in entities), default=0),
                "relationship_avg": sum(r["confidence"] for r in relationships) / len(relationships) if relationships else 0,
                "relationship_min": min((r["confidence"] for r in relationships), default=0),
                "relationship_max": max((r["confidence"] for r in relationships), default=0)
            }
        }

        # Top entity types
        entity_counter = Counter(ent["label"] for ent in entities)
        stats["top_entity_types"] = dict(entity_counter.most_common(10))

        # Top relationship types
        rel_counter = Counter(rel.get("discovered_type", "unknown") for rel in relationships)
        stats["top_relationship_types"] = dict(rel_counter.most_common(10))

        return stats

    def reset_counters(self):
        """Reset all ID counters"""
        self.entity_extractor.reset_counter()
        self.relationship_extractor.reset_counter()

    def get_entity_counter(self) -> int:
        """Get current entity ID counter"""
        return self.entity_extractor.entity_id_counter

    def get_relationship_counter(self) -> int:
        """Get current relationship ID counter"""
        return self.relationship_extractor.relationship_id_counter