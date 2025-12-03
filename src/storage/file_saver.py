"""
File Saver Module
Saves extraction results to JSON and CSV formats
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class FileSaver:
    """Handles saving knowledge graph extraction results to files"""

    def __init__(self, output_dir: str, config: Dict = None):
        """Initialize FileSaver"""
        self.output_dir = Path(output_dir)
        self.config = config or {}
        self.logger = logger

        # Create output subdirectories
        self.entities_dir = self.output_dir / 'entities'
        self.relationships_dir = self.output_dir / 'relationships'
        self.reports_dir = self.output_dir / 'reports'

        for dir_path in [self.entities_dir, self.relationships_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    # def save_all(self, extraction_result: Dict[str, Any]) -> Dict[str, str]:
    #     """Save all extraction results to files"""
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    #     created_files = {}

    #     # Save entities
    #     created_files['entities_json'] = self.save_entities_json(
    #         extraction_result['entities'],
    #         f"entities_{timestamp}.json"
    #     )

    #     created_files['entities_csv'] = self.save_entities_csv(
    #         extraction_result['entities'],
    #         f"entities_{timestamp}.csv"
    #     )

    #     # Save relationships
    #     created_files['relationships_json'] = self.save_relationships_json(
    #         extraction_result['relationships'],
    #         f"relationships_{timestamp}.json"
    #     )

    #     created_files['relationships_csv'] = self.save_relationships_csv(
    #         extraction_result['relationships'],
    #         f"relationships_{timestamp}.csv"
    #     )

    #     # Save statistics and metadata
    #     created_files['report'] = self.save_report(
    #         extraction_result,
    #         f"extraction_report_{timestamp}.json"
    #     )

    #     self.logger.info(f"Saved all extraction results with timestamp {timestamp}")
    #     return created_files
    
    def save_all(self, extraction_result: Dict[str, Any]) -> Dict[str, str]:
        """⚡ SKIP FILE SAVING - Direct Neo4j import mode"""
        self.logger.info("⚡ Skipping file creation (direct import mode)")
        return {}  # Return empty dict, no files created
    

    def save_entities_json(self, entities: List[Dict[str, Any]], filename: str) -> str:
        """Save entities to JSON file"""
        filepath = self.entities_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(entities, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved {len(entities)} entities to {filepath}")
        return str(filepath)

    def save_entities_csv(self, entities: List[Dict[str, Any]], filename: str) -> str:
        """Save entities to CSV file"""
        filepath = self.entities_dir / filename

        # Flatten entity structure for CSV
        flattened = []
        for ent in entities:
            flattened.append({
                'id': ent.get('id'),
                'text': ent.get('text'),
                'label': ent.get('label'),
                'confidence': ent.get('confidence'),
                'source': ent.get('source'),
                'block_id': ent.get('block_id'),
                'lemma': ent.get('lemma'),
                'root_dep': ent.get('root_dep'),
                'root_pos': ent.get('root_pos'),
                'start': ent.get('start'),
                'end': ent.get('end'),
                'start_token': ent.get('start_token'),
                'end_token': ent.get('end_token')
            })

        df = pd.DataFrame(flattened)
        df.to_csv(filepath, index=False, encoding='utf-8')

        self.logger.info(f"Saved {len(entities)} entities to CSV: {filepath}")
        return str(filepath)

    # def save_relationships_json(self, relationships: List[Dict[str, Any]], filename: str) -> str:
    #     """Save relationships to JSON file"""
    #     filepath = self.relationships_dir / filename

    #     with open(filepath, 'w', encoding='utf-8') as f:
    #         json.dump(relationships, f, indent=2, ensure_ascii=False)

    #     self.logger.info(f"Saved {len(relationships)} relationships to {filepath}")
    #     return str(filepath)
    
    def save_relationships_json(self, relationships: List[Dict[str, Any]], filename: str) -> str:
        """Save relationships to JSON file"""
        filepath = self.relationships_dir / filename

        # ⚡ FILTER: Only save high-confidence relationships (>= 0.75)
        HIGH_CONFIDENCE_THRESHOLD = 0.75
        original_count = len(relationships)
        relationships = [r for r in relationships if r.get('confidence', 0) >= HIGH_CONFIDENCE_THRESHOLD]
        
        self.logger.info(f"Filtered relationships: {original_count:,} → {len(relationships):,} ({len(relationships)/original_count*100:.1f}%)")

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(relationships, f, indent=2, ensure_ascii=False)

    def save_relationships_csv(self, relationships: List[Dict[str, Any]], filename: str) -> str:
        # """Save relationships to CSV file"""
        # filepath = self.relationships_dir / filename

        # # Flatten relationship structure for CSV
        # flattened = []
        """Save relationships to CSV file"""
        filepath = self.relationships_dir / filename

        # ⚡ FILTER: Only save high-confidence relationships (>= 0.75)
        HIGH_CONFIDENCE_THRESHOLD = 0.75
        relationships = [r for r in relationships if r.get('confidence', 0) >= HIGH_CONFIDENCE_THRESHOLD]
        
        self.logger.info(f"Filtered to {len(relationships):,} high-confidence relationships (>= {HIGH_CONFIDENCE_THRESHOLD})")

        # Flatten relationship structure for CSV
        flattened = []
        for rel in relationships:
            flattened.append({
                'id': rel.get('id'),
                'subject_id': rel.get('subject', {}).get('id', ''),
                'subject_text': rel.get('subject', {}).get('text', ''),
                'subject_label': rel.get('subject', {}).get('label', ''),
                'discovered_type': rel.get('discovered_type', 'unknown'),
                'object_id': rel.get('object', {}).get('id', ''),
                'object_text': rel.get('object', {}).get('text', ''),
                'object_label': rel.get('object', {}).get('label', ''),
                'confidence': rel.get('confidence', 0),
                'sources': ', '.join(rel.get('sources', [])),
                'block_id': rel.get('block_id'),
                'sentence': rel.get('sentence', '')
            })

        df = pd.DataFrame(flattened)
        df.to_csv(filepath, index=False, encoding='utf-8')

        self.logger.info(f"Saved {len(relationships)} relationships to CSV: {filepath}")
        return str(filepath)

    def save_report(self, extraction_result: Dict[str, Any], filename: str) -> str:
        """Save extraction report with statistics and metadata"""
        filepath = self.reports_dir / filename

        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_entities': len(extraction_result.get('entities', [])),
                'total_relationships': len(extraction_result.get('relationships', [])),
                'blocks_processed': len(extraction_result.get('block_metadata', []))
            },
            'statistics': extraction_result.get('overall_statistics', {}),
            'block_metadata': extraction_result.get('block_metadata', [])
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved extraction report to {filepath}")
        return str(filepath)

    def save_state(self, kg_builder, filename: str = None) -> str:
        """Save the state of the KG builder"""
        if filename is None:
            filename = "extractor_state.json"

        filepath = self.output_dir / filename

        state = {
            'entity_id_counter': kg_builder.get_entity_counter(),
            'relationship_id_counter': kg_builder.get_relationship_counter(),
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)

        self.logger.info(f"Saved extractor state to {filepath}")
        return str(filepath)

    def create_summary_report(self, extraction_result: Dict[str, Any]) -> str:
        """Create a human-readable summary report"""
        stats = extraction_result.get('overall_statistics', {})

        summary = []
        summary.append("=" * 80)
        summary.append("KNOWLEDGE GRAPH EXTRACTION SUMMARY")
        summary.append("=" * 80)
        summary.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append("")
        summary.append(f"Total Blocks Processed: {stats.get('total_blocks_processed', 0)}")
        summary.append(f"Total Entities Extracted: {stats.get('total_entities', 0)}")
        summary.append(f"Total Relationships Extracted: {stats.get('total_relationships', 0)}")
        summary.append("")

        # Top entity types
        summary.append("Top Entity Types:")
        top_entities = stats.get('top_entity_types', {})
        for entity_type, count in list(top_entities.items())[:10]:
            summary.append(f"  {entity_type}: {count}")

        summary.append("")

        # Top relationship types
        summary.append("Top Relationship Types:")
        top_rels = stats.get('top_relationship_types', {})
        for rel_type, count in list(top_rels.items())[:10]:
            summary.append(f"  {rel_type}: {count}")

        summary.append("")

        # Confidence statistics
        conf_stats = stats.get('confidence_statistics', {})
        summary.append("Confidence Statistics:")
        summary.append(f"  Entities - Avg: {conf_stats.get('entity_avg', 0):.3f}, "
                       f"Min: {conf_stats.get('entity_min', 0):.3f}, "
                       f"Max: {conf_stats.get('entity_max', 0):.3f}")
        summary.append(f"  Relationships - Avg: {conf_stats.get('relationship_avg', 0):.3f}, "
                       f"Min: {conf_stats.get('relationship_min', 0):.3f}, "
                       f"Max: {conf_stats.get('relationship_max', 0):.3f}")

        summary.append("=" * 80)

        summary_text = "\n".join(summary)

        # Save to file
        filepath = self.reports_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(summary_text)

        self.logger.info(f"Created summary report: {filepath}")

        return summary_text