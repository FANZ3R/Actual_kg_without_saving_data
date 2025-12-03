"""
Entity Extraction Module
Extracts entities from text using multiple methods
"""

import logging
from typing import List, Dict, Any, Set, Tuple

logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Extracts entities from spaCy documents using multiple methods:
    1. spaCy NER (Named Entity Recognition)
    2. Noun phrase extraction
    3. Single token extraction (nouns, proper nouns)
    """

    def __init__(self, config: Dict = None):
        """Initialize EntityExtractor"""
        self.config = config or {}
        self.entity_id_counter = 0
        self.logger = logger

    def extract_entities(self, doc) -> List[Dict[str, Any]]:
        """Extract entities using multiple methods"""
        entities = []
        seen_spans: Set[Tuple[int, int]] = set()

        # Method 1: Standard spaCy NER entities
        entities.extend(self._extract_ner_entities(doc, seen_spans))

        # Method 2: Extract noun phrases as potential entities
        entities.extend(self._extract_noun_phrases(doc, seen_spans))

        # Method 3: Extract significant single tokens
        entities.extend(self._extract_single_tokens(doc, seen_spans))

        return entities

    def _extract_ner_entities(self, doc, seen_spans: Set[Tuple[int, int]]) -> List[Dict[str, Any]]:
        """Extract entities using spaCy's NER"""
        entities = []

        for ent in doc.ents:
            span_key = (ent.start, ent.end)
            if span_key not in seen_spans:
                entity = {
                    "id": f"ent_{self.entity_id_counter}",
                    "text": ent.text.strip(),
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "start_token": ent.start,
                    "end_token": ent.end,
                    "confidence": 1.0,
                    "source": "spacy_ner",
                    "lemma": ent.lemma_,
                    "root_dep": ent.root.dep_,
                    "root_pos": ent.root.pos_
                }
                entities.append(entity)
                seen_spans.add(span_key)
                self.entity_id_counter += 1

        return entities

    def _extract_noun_phrases(self, doc, seen_spans: Set[Tuple[int, int]]) -> List[Dict[str, Any]]:
        """Extract noun phrases as entities"""
        entities = []

        for np in doc.noun_chunks:
            span_key = (np.start, np.end)
            if span_key not in seen_spans and len(np.text.strip()) > 2:
                entity = {
                    "id": f"ent_{self.entity_id_counter}",
                    "text": np.text.strip(),
                    "label": "NOUN_PHRASE",
                    "start": np.start_char,
                    "end": np.end_char,
                    "start_token": np.start,
                    "end_token": np.end,
                    "confidence": 0.7,
                    "source": "noun_chunk",
                    "lemma": np.lemma_,
                    "root_dep": np.root.dep_,
                    "root_pos": np.root.pos_
                }
                entities.append(entity)
                seen_spans.add(span_key)
                self.entity_id_counter += 1

        return entities

    def _extract_single_tokens(self, doc, seen_spans: Set[Tuple[int, int]]) -> List[Dict[str, Any]]:
        """Extract significant single tokens as entities"""
        entities = []

        for token in doc:
            if (token.pos_ in ["NOUN", "PROPN"] and
                    not token.is_stop and
                    not token.is_punct and
                    len(token.text) > 2):

                span_key = (token.i, token.i + 1)
                if span_key not in seen_spans:
                    entity = {
                        "id": f"ent_{self.entity_id_counter}",
                        "text": token.text,
                        "label": f"{token.pos_}",
                        "start": token.idx,
                        "end": token.idx + len(token.text),
                        "start_token": token.i,
                        "end_token": token.i + 1,
                        "confidence": 0.5,
                        "source": "single_token",
                        "lemma": token.lemma_,
                        "root_dep": token.dep_,
                        "root_pos": token.pos_
                    }
                    entities.append(entity)
                    seen_spans.add(span_key)
                    self.entity_id_counter += 1

        return entities

    def reset_counter(self):
        """Reset entity ID counter"""
        self.entity_id_counter = 0