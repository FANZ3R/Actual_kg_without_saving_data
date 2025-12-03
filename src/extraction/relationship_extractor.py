"""
Relationship Extraction Module
Discovers relationships between entities using multiple methods
"""

import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict
from itertools import combinations

logger = logging.getLogger(__name__)


class RelationshipExtractor:
    """
    Discovers relationships between entities using multiple methods:
    1. Verb-based relationships
    2. Preposition-based relationships
    3. Dependency path analysis
    4. Syntactic pattern matching
    5. Semantic proximity
    """

    def __init__(self, config: Dict = None):
        """Initialize RelationshipExtractor"""
        self.config = config or {}
        self.relationship_id_counter = 0
        self.discovered_patterns = defaultdict(list)
        self.logger = logger

    def extract_relationships(self, doc, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relationships using all discovery methods"""
        all_relationships = []

        # Apply all relationship discovery methods
        all_relationships.extend(self._discover_verb_relationships(doc, entities))
        all_relationships.extend(self._discover_preposition_relationships(doc, entities))
        all_relationships.extend(self._discover_dependency_path_relationships(doc, entities))
        all_relationships.extend(self._discover_pattern_based_relationships(doc, entities))
        all_relationships.extend(self._discover_semantic_relationships(doc, entities))

        # Merge duplicate relationships
        merged_relationships = self._merge_duplicate_relationships(all_relationships)

        return merged_relationships

    def _discover_verb_relationships(self, doc, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Discover relationships based on verbs connecting entities"""
        relationships = []

        for token in doc:
            if token.pos_ == "VERB":
                verb = token

                # Find all entities connected to this verb
                connected_entities = []

                # Look at all children and ancestors of the verb
                for child in verb.children:
                    entity = self._find_entity_for_token(child, entities)
                    if entity:
                        connected_entities.append({
                            "entity": entity,
                            "role": child.dep_,
                            "position": "child"
                        })

                # Check verb's head if it's an entity
                if verb.head != verb:
                    entity = self._find_entity_for_token(verb.head, entities)
                    if entity:
                        connected_entities.append({
                            "entity": entity,
                            "role": verb.dep_,
                            "position": "parent"
                        })

                # Create relationships between connected entities
                if len(connected_entities) >= 2:
                    for i, conn1 in enumerate(connected_entities):
                        for conn2 in connected_entities[i + 1:]:
                            relationship = {
                                "id": f"rel_{self.relationship_id_counter}",
                                "discovered_type": f"{verb.lemma_}",
                                "verb_text": verb.text,
                                "verb_lemma": verb.lemma_,
                                "subject": conn1["entity"],
                                "subject_role": conn1["role"],
                                "object": conn2["entity"],
                                "object_role": conn2["role"],
                                "confidence": self._calculate_confidence(verb, conn1, conn2),
                                "source": "verb_discovery",
                                "sentence": str(verb.sent),
                                "context_window": doc[max(0, verb.i - 5):min(len(doc), verb.i + 6)].text
                            }
                            relationships.append(relationship)
                            self.relationship_id_counter += 1

        return relationships

    def _discover_preposition_relationships(self, doc, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Discover relationships through prepositions"""
        relationships = []

        for token in doc:
            if token.pos_ == "ADP":  # ADP is the POS tag for prepositions
                prep = token

                # Find entities before and after the preposition
                before_entity = None
                after_entity = None

                # Look for entity that this preposition modifies
                if prep.head != prep:
                    before_entity = self._find_entity_for_token(prep.head, entities)

                # Look for entity that is the object of the preposition
                for child in prep.children:
                    if child.dep_ in ["pobj", "dobj", "obj"]:
                        after_entity = self._find_entity_for_token(child, entities)
                        break

                if before_entity and after_entity:
                    relationship = {
                        "id": f"rel_{self.relationship_id_counter}",
                        "discovered_type": f"{prep.text}_relation",
                        "preposition": prep.text,
                        "preposition_lemma": prep.lemma_,
                        "subject": before_entity,
                        "object": after_entity,
                        "confidence": 0.7,
                        "source": "preposition_discovery",
                        "sentence": str(prep.sent),
                        "context_window": doc[max(0, prep.i - 5):min(len(doc), prep.i + 6)].text
                    }
                    relationships.append(relationship)
                    self.relationship_id_counter += 1

        return relationships

    def _discover_dependency_path_relationships(self, doc, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Discover relationships by analyzing dependency paths between entities"""
        relationships = []

        # For each pair of entities in the same sentence
        for sent in doc.sents:
            sent_entities = [
                ent for ent in entities
                if ent["start_token"] >= sent.start and ent["end_token"] <= sent.end
            ]

            for ent1, ent2 in combinations(sent_entities, 2):
                # Find the dependency path between entities
                path = self._find_dependency_path(doc, ent1, ent2)

                if path and len(path) > 0:
                    # Extract the relationship from the path
                    path_text = self._extract_path_text(doc, path)
                    path_pattern = self._extract_path_pattern(doc, path)

                    relationship = {
                        "id": f"rel_{self.relationship_id_counter}",
                        "discovered_type": path_pattern,
                        "path_text": path_text,
                        "dependency_path": [doc[i].dep_ for i in path],
                        "subject": ent1,
                        "object": ent2,
                        "confidence": self._calculate_path_confidence(path),
                        "source": "dependency_path",
                        "sentence": str(sent),
                        "path_length": len(path)
                    }
                    relationships.append(relationship)
                    self.relationship_id_counter += 1

        return relationships

    def _discover_pattern_based_relationships(self, doc, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Discover relationships using syntactic patterns"""
        relationships = []

        # Pattern 1: Entity-Verb-Entity
        for i in range(len(doc) - 2):
            if (self._is_entity_token(doc[i], entities) and
                    doc[i + 1].pos_ == "VERB" and
                    self._is_entity_token(doc[i + 2], entities)):

                ent1 = self._find_entity_for_token(doc[i], entities)
                ent2 = self._find_entity_for_token(doc[i + 2], entities)

                if ent1 and ent2:
                    relationship = {
                        "id": f"rel_{self.relationship_id_counter}",
                        "discovered_type": f"{doc[i + 1].lemma_}_pattern",
                        "connecting_word": doc[i + 1].text,
                        "connecting_lemma": doc[i + 1].lemma_,
                        "subject": ent1,
                        "object": ent2,
                        "confidence": 0.6,
                        "source": "syntactic_pattern",
                        "pattern": "E-V-E",
                        "sentence": str(doc[i].sent)
                    }
                    relationships.append(relationship)
                    self.relationship_id_counter += 1

        # Pattern 2: Entity's Entity (possessive)
        for i in range(len(doc) - 2):
            if (self._is_entity_token(doc[i], entities) and
                    doc[i + 1].text in ["'s", "'s", "of"] and
                    self._is_entity_token(doc[i + 2], entities)):

                ent1 = self._find_entity_for_token(doc[i], entities)
                ent2 = self._find_entity_for_token(doc[i + 2], entities)

                if ent1 and ent2:
                    relationship = {
                        "id": f"rel_{self.relationship_id_counter}",
                        "discovered_type": "possessive_relation",
                        "connecting_word": doc[i + 1].text,
                        "subject": ent1,
                        "object": ent2,
                        "confidence": 0.7,
                        "source": "syntactic_pattern",
                        "pattern": "E-POSS-E",
                        "sentence": str(doc[i].sent)
                    }
                    relationships.append(relationship)
                    self.relationship_id_counter += 1

        return relationships

    def _discover_semantic_relationships(self, doc, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Discover relationships based on semantic similarity and context"""
        relationships = []

        # For entities that appear close to each other
        for sent in doc.sents:
            sent_entities = [
                ent for ent in entities
                if ent["start_token"] >= sent.start and ent["end_token"] <= sent.end
            ]

            for ent1, ent2 in combinations(sent_entities, 2):
                # Calculate token distance
                distance = abs(ent1["start_token"] - ent2["start_token"])

                if distance <= 10:  # Within 10 tokens
                    # Find connecting words between entities
                    start = min(ent1["end_token"], ent2["end_token"])
                    end = max(ent1["start_token"], ent2["start_token"])

                    if start < end:
                        connecting_tokens = doc[start:end]
                        connecting_text = " ".join([t.text for t in connecting_tokens])

                        # Identify key connecting words
                        key_words = [t for t in connecting_tokens
                                     if t.pos_ in ["VERB", "ADP", "CCONJ"]]

                        if key_words:
                            relationship = {
                                "id": f"rel_{self.relationship_id_counter}",
                                "discovered_type": "_".join([w.lemma_ for w in key_words[:2]]),
                                "connecting_text": connecting_text,
                                "key_words": [w.lemma_ for w in key_words],
                                "subject": ent1,
                                "object": ent2,
                                "confidence": 0.5 / (1 + distance / 10),
                                "source": "semantic_proximity",
                                "distance": distance,
                                "sentence": str(sent)
                            }
                            relationships.append(relationship)
                            self.relationship_id_counter += 1

        return relationships

    # Helper methods
    def _find_entity_for_token(self, token, entities: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find entity that contains the given token"""
        for entity in entities:
            if entity["start_token"] <= token.i < entity["end_token"]:
                return entity
        return None

    def _is_entity_token(self, token, entities: List[Dict[str, Any]]) -> bool:
        """Check if token is part of an entity"""
        return self._find_entity_for_token(token, entities) is not None

    def _find_dependency_path(self, doc, ent1: Dict[str, Any], ent2: Dict[str, Any]) -> Optional[List[int]]:
        """Find shortest dependency path between two entities using BFS"""
        token1 = doc[ent1["start_token"]]
        token2 = doc[ent2["start_token"]]

        visited = set()
        queue = [(token1, [token1.i])]

        while queue:
            current, path = queue.pop(0)

            if current == token2:
                return path

            if current in visited:
                continue

            visited.add(current)

            # Add head and children to queue
            if current.head != current and len(path) < 10:
                queue.append((current.head, path + [current.head.i]))

            for child in current.children:
                if len(path) < 10:
                    queue.append((child, path + [child.i]))

        return None

    @staticmethod
    def _extract_path_text(doc, path: List[int]) -> str:
        """Extract text from dependency path"""
        if not path:
            return ""
        return " ".join([doc[i].text for i in path])

    @staticmethod
    def _extract_path_pattern(doc, path: List[int]) -> str:
        """Extract pattern from dependency path"""
        if not path:
            return "unknown"

        pattern_parts = []
        for i in path:
            token = doc[i]
            if token.pos_ in ["VERB", "ADP", "CCONJ"]:
                pattern_parts.append(token.lemma_)

        return "_".join(pattern_parts) if pattern_parts else "direct_connection"

    @staticmethod
    def _calculate_confidence(verb, conn1: Dict, conn2: Dict) -> float:
        """Calculate confidence score for verb-based relationships"""
        base_confidence = 0.5

        # Boost for certain dependency relations
        important_deps = ["nsubj", "dobj", "nsubjpass", "agent", "attr"]
        if conn1["role"] in important_deps:
            base_confidence += 0.15
        if conn2["role"] in important_deps:
            base_confidence += 0.15

        # Boost for active voice
        if verb.tag_ in ["VB", "VBZ", "VBP", "VBD"]:
            base_confidence += 0.1

        return min(base_confidence, 1.0)

    @staticmethod
    def _calculate_path_confidence(path: List[int]) -> float:
        """Calculate confidence based on dependency path length"""
        if not path:
            return 0.1

        # Shorter paths are more confident
        length_penalty = len(path) / 10.0
        return max(0.3, 1.0 - length_penalty)

    def _merge_duplicate_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge duplicate relationships and adjust confidence"""
        merged = {}

        for rel in relationships:
            # Create a key for grouping similar relationships
            key = (
                rel.get("subject", {}).get("id"),
                rel.get("object", {}).get("id"),
                rel.get("discovered_type", "unknown")
            )

            if key in merged:
                # Increase confidence when multiple methods find the same relationship
                merged[key]["confidence"] = min(
                    1.0,
                    merged[key]["confidence"] + rel["confidence"] * 0.2
                )
                merged[key]["sources"].append(rel["source"])
            else:
                rel["sources"] = [rel["source"]]
                merged[key] = rel

        return list(merged.values())

    def reset_counter(self):
        """Reset relationship ID counter"""
        self.relationship_id_counter = 0
        self.discovered_patterns.clear()