"""
Data Validator Module
Validates data quality and ensures data meets requirements for KG extraction
"""

import logging
import re
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data quality and filters out low-quality blocks"""

    def __init__(self, config: Dict = None):
        """Initialize DataValidator"""
        self.config = config or {}
        self.min_text_length = self.config.get('min_text_length', 10)
        self.max_text_length = self.config.get('max_text_length', 100000)
        self.min_alpha_ratio = self.config.get('min_alpha_ratio', 0.5)
        self.remove_urls = self.config.get('remove_urls', True)
        self.remove_emails = self.config.get('remove_emails', True)
        self.logger = logger

    def validate_blocks(self, blocks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Validate all blocks and return valid ones with statistics"""
        valid_blocks = []
        stats = {
            'total_input': len(blocks),
            'valid': 0,
            'rejected': 0,
            'rejection_reasons': {
                'too_short': 0,
                'too_long': 0,
                'low_alpha_ratio': 0,
                'empty_text': 0,
                'other': 0
            }
        }

        for block in blocks:
            is_valid, reason = self.validate_block(block)

            if is_valid:
                valid_blocks.append(block)
                stats['valid'] += 1
            else:
                stats['rejected'] += 1
                if reason in stats['rejection_reasons']:
                    stats['rejection_reasons'][reason] += 1
                else:
                    stats['rejection_reasons']['other'] += 1

        self.logger.info(f"Validation: {stats['valid']} valid, {stats['rejected']} rejected")
        return valid_blocks, stats

    def validate_block(self, block: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate a single text block"""
        text = block.get('text', '').strip()

        if not text:
            return False, 'empty_text'

        if len(text) < self.min_text_length:
            return False, 'too_short'

        if len(text) > self.max_text_length:
            return False, 'too_long'

        alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / len(text)
        if alpha_ratio < self.min_alpha_ratio:
            return False, 'low_alpha_ratio'

        return True, None

    def clean_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean text in blocks by removing unwanted content"""
        cleaned_blocks = []

        for block in blocks:
            text = block.get('text', '')
            
            if self.remove_urls:
                text = self._remove_urls(text)

            if self.remove_emails:
                text = self._remove_emails(text)

            text = self._normalize_whitespace(text)

            cleaned_block = block.copy()
            cleaned_block['text'] = text
            cleaned_blocks.append(cleaned_block)

        return cleaned_blocks

    def deduplicate_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate text blocks"""
        seen_texts = set()
        unique_blocks = []

        for block in blocks:
            text = block.get('text', '').strip().lower()

            if text not in seen_texts:
                seen_texts.add(text)
                unique_blocks.append(block)

        removed = len(blocks) - len(unique_blocks)
        self.logger.info(f"Removed {removed} duplicates, {len(unique_blocks)} unique blocks")
        return unique_blocks

    def filter_by_quality(self, blocks: List[Dict[str, Any]], min_quality: float = 0.5) -> List[Dict[str, Any]]:
        """Filter blocks by minimum quality score"""
        filtered_blocks = []

        for block in blocks:
            quality = self.get_quality_score(block)
            block['quality_score'] = quality

            if quality >= min_quality:
                filtered_blocks.append(block)

        removed = len(blocks) - len(filtered_blocks)
        self.logger.info(f"Quality filter: Kept {len(filtered_blocks)}, removed {removed}")
        return filtered_blocks

    def get_quality_score(self, block: Dict[str, Any]) -> float:
        """Calculate quality score for a text block (0-1)"""
        text = block.get('text', '')
        if not text:
            return 0.0

        score = 0.0
        weights = {
            'length': 0.3,
            'alpha_ratio': 0.3,
            'sentence_structure': 0.2,
            'vocabulary_diversity': 0.2
        }

        # Length score
        length_score = min(len(text) / 1000, 1.0) if len(text) < 1000 else max(1.0 - (len(text) - 1000) / 10000, 0.3)
        score += length_score * weights['length']

        # Alpha ratio score
        alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / len(text)
        score += alpha_ratio * weights['alpha_ratio']

        # Sentence structure score
        has_sentence_end = any(p in text for p in ['.', '!', '?'])
        sentence_score = 1.0 if has_sentence_end else 0.5
        score += sentence_score * weights['sentence_structure']

        # Vocabulary diversity
        words = text.split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            score += unique_ratio * weights['vocabulary_diversity']

        return min(score, 1.0)

    @staticmethod
    def _remove_urls(text: str) -> str:
        """Remove URLs from text"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)

    @staticmethod
    def _remove_emails(text: str) -> str:
        """Remove email addresses from text"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, '', text)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Normalize whitespace in text"""
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n\n+', '\n\n', text)
        return text.strip()