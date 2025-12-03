"""
Data Converter Module
Converts all ingested data blocks into standardized JSON format
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class DataConverter:
    """Converts data blocks into standardized JSON format"""

    def __init__(self, config: Dict = None):
        """Initialize DataConverter"""
        self.config = config or {}
        self.logger = logger

    def convert_to_json(self, blocks: List[Dict[str, Any]], output_path: str) -> str:
        """Convert data blocks to standardized JSON format"""
        if not blocks:
            self.logger.warning("No blocks to convert")
            return None

        standardized_data = {
            'metadata': {
                'total_blocks': len(blocks),
                'conversion_timestamp': datetime.now().isoformat(),
                'source_types': self._get_source_types(blocks),
                'source_files': self._get_source_files(blocks)
            },
            'blocks': []
        }

        for idx, block in enumerate(blocks):
            standardized_block = {
                'block_id': f"block_{idx}",
                'text': block.get('text', ''),
                'source': {
                    'file': block.get('source_file', 'unknown'),
                    'type': block.get('source_type', 'unknown'),
                    'index': block.get('index', idx)
                },
                'metadata': block.get('metadata', {}),
                'stats': {
                    'char_count': len(block.get('text', '')),
                    'word_count': len(block.get('text', '').split())
                }
            }
            standardized_data['blocks'].append(standardized_block)

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"converted_data_{timestamp}.json"
        filepath = output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(standardized_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Converted {len(blocks)} blocks to {filepath}")
        return str(filepath)

    def _get_source_types(self, blocks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get count of blocks by source type"""
        source_types = {}
        for block in blocks:
            source_type = block.get('source_type', 'unknown')
            source_types[source_type] = source_types.get(source_type, 0) + 1
        return source_types

    def _get_source_files(self, blocks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get count of blocks by source file"""
        source_files = {}
        for block in blocks:
            source_file = block.get('source_file', 'unknown')
            source_files[source_file] = source_files.get(source_file, 0) + 1
        return source_files