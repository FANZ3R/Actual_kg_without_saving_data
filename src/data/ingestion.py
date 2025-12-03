"""
Data Ingestion Module
Handles loading data from multiple file formats (JSON, CSV, Excel, PDF, DOCX, TXT)
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class DataIngestion:
    """
    Universal data ingestion class that handles multiple file formats
    """

    SUPPORTED_FORMATS = {
        'json': ['.json'],
        'excel': ['.xlsx', '.xls'],
        'csv': ['.csv'],
        'pdf': ['.pdf'],
        'docx': ['.docx'],
        'text': ['.txt']
    }

    def __init__(self, config: Optional[Dict] = None):
        """Initialize DataIngestion"""
        self.config = config or {}
        self.min_text_length = self.config.get('min_text_length', 10)
        self.logger = logger

    def load_from_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """Load all supported files from a directory"""
        directory = Path(directory_path)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        all_blocks = []
        file_count = 0

        for file_path in directory.iterdir():
            if file_path.is_file() and self._is_supported_format(file_path):
                try:
                    blocks = self.load_file(str(file_path))
                    all_blocks.extend(blocks)
                    file_count += 1
                    self.logger.info(f"Loaded {len(blocks)} blocks from {file_path.name}")
                except Exception as e:
                    self.logger.error(f"Failed to load {file_path.name}: {e}")

        self.logger.info(f"Total: Loaded {len(all_blocks)} blocks from {file_count} files")
        return all_blocks

    def load_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from a single file"""
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == '.json':
            return self._load_json(path)
        elif ext in ['.xlsx', '.xls']:
            return self._load_excel(path)
        elif ext == '.csv':
            return self._load_csv(path)
        elif ext == '.pdf':
            return self._load_pdf(path)
        elif ext == '.docx':
            return self._load_docx(path)
        elif ext == '.txt':
            return self._load_text(path)
        else:
            return self._load_text(path)

    def _load_json(self, path: Path) -> List[Dict[str, Any]]:
        """Load data from JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        blocks = []
        if isinstance(data, list):
            for idx, item in enumerate(data):
                text = self._extract_text_from_dict(item)
                if text and len(text.strip()) >= self.min_text_length:
                    blocks.append({
                        'text': text,
                        'source_file': path.name,
                        'source_type': 'json',
                        'index': idx,
                        'metadata': item if isinstance(item, dict) else {}
                    })
        elif isinstance(data, dict):
            text = self._extract_text_from_dict(data)
            if text and len(text.strip()) >= self.min_text_length:
                blocks.append({
                    'text': text,
                    'source_file': path.name,
                    'source_type': 'json',
                    'index': 0,
                    'metadata': data
                })

        return blocks

    def _load_excel(self, path: Path) -> List[Dict[str, Any]]:
        """Load data from Excel file"""
        try:
            df = pd.read_excel(path)
            return self._process_dataframe(df, path.name, 'excel')
        except Exception as e:
            self.logger.error(f"Error loading Excel file {path}: {e}")
            return []

    def _load_csv(self, path: Path) -> List[Dict[str, Any]]:
        """Load data from CSV file"""
        try:
            df = pd.read_csv(path)
            return self._process_dataframe(df, path.name, 'csv')
        except Exception as e:
            self.logger.error(f"Error loading CSV file {path}: {e}")
            return []

    def _load_pdf(self, path: Path) -> List[Dict[str, Any]]:
        """Load data from PDF file"""
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(str(path))
            blocks = []

            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and len(text.strip()) >= self.min_text_length:
                    blocks.append({
                        'text': text.strip(),
                        'source_file': path.name,
                        'source_type': 'pdf',
                        'index': page_num,
                        'metadata': {'page': page_num + 1}
                    })

            return blocks
        except ImportError:
            self.logger.warning("PyPDF2 not installed. Cannot load PDF files.")
            return []
        except Exception as e:
            self.logger.error(f"Error loading PDF file {path}: {e}")
            return []

    def _load_docx(self, path: Path) -> List[Dict[str, Any]]:
        """Load data from DOCX file"""
        try:
            from docx import Document

            doc = Document(str(path))
            blocks = []

            for idx, paragraph in enumerate(doc.paragraphs):
                text = paragraph.text.strip()
                if text and len(text) >= self.min_text_length:
                    blocks.append({
                        'text': text,
                        'source_file': path.name,
                        'source_type': 'docx',
                        'index': idx,
                        'metadata': {'paragraph': idx + 1}
                    })

            return blocks
        except ImportError:
            self.logger.warning("python-docx not installed. Cannot load DOCX files.")
            return []
        except Exception as e:
            self.logger.error(f"Error loading DOCX file {path}: {e}")
            return []

    def _load_text(self, path: Path) -> List[Dict[str, Any]]:
        """Load data from plain text file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()

            paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) >= self.min_text_length]

            blocks = []
            for idx, text in enumerate(paragraphs):
                blocks.append({
                    'text': text,
                    'source_file': path.name,
                    'source_type': 'text',
                    'index': idx,
                    'metadata': {'paragraph': idx + 1}
                })

            return blocks
        except Exception as e:
            self.logger.error(f"Error loading text file {path}: {e}")
            return []

    def _process_dataframe(self, df: pd.DataFrame, filename: str, source_type: str) -> List[Dict[str, Any]]:
        """Process pandas DataFrame into text blocks"""
        blocks = []

        text_cols = [col for col in df.columns
                    if any(keyword in col.lower() for keyword in ['text', 'content', 'description', 'body', 'message'])]

        if not text_cols:
            text_cols = [col for col in df.columns if df[col].dtype == 'object']

        for idx, row in df.iterrows():
            text_parts = [str(row[col]) for col in text_cols if pd.notna(row[col]) and str(row[col]).strip()]
            text = ' '.join(text_parts)

            if text and len(text.strip()) >= self.min_text_length:
                blocks.append({
                    'text': text.strip(),
                    'source_file': filename,
                    'source_type': source_type,
                    'index': idx,
                    'metadata': row.to_dict()
                })

        return blocks

    def _extract_text_from_dict(self, data: Dict) -> str:
        """Extract text content from dictionary"""
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            for key in ['text', 'content', 'description', 'body', 'message']:
                if key in data:
                    return str(data[key])
            text_values = [str(v) for v in data.values() if isinstance(v, (str, int, float))]
            return ' '.join(text_values)
        else:
            return str(data)

    def _is_supported_format(self, path: Path) -> bool:
        """Check if file format is supported"""
        ext = path.suffix.lower()
        for formats in self.SUPPORTED_FORMATS.values():
            if ext in formats:
                return True
        return False

    def get_statistics(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate statistics about loaded data"""
        if not blocks:
            return {'total_blocks': 0}

        stats = {
            'total_blocks': len(blocks),
            'total_characters': sum(len(block['text']) for block in blocks),
            'avg_text_length': sum(len(block['text']) for block in blocks) / len(blocks),
            'source_types': {},
            'source_files': {}
        }

        for block in blocks:
            source_type = block.get('source_type', 'unknown')
            source_file = block.get('source_file', 'unknown')

            stats['source_types'][source_type] = stats['source_types'].get(source_type, 0) + 1
            stats['source_files'][source_file] = stats['source_files'].get(source_file, 0) + 1

        return stats