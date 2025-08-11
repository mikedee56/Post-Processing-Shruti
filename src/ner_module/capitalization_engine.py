"""
Capitalization Engine for Yoga Vedanta Proper Nouns.

This module provides intelligent capitalization functionality for proper nouns
identified by the NER system, applying context-aware rules and patterns.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum
import logging

from utils.logger_config import get_logger
from .entity_classifier import EntityCategory, EntityClassifier
from .yoga_vedanta_ner import NamedEntity, YogaVedantaNER
from sanskrit_hindi_identifier.lexicon_manager import LexiconManager


class CapitalizationRule(Enum):
    """Capitalization rule types."""
    PROPER_CASE = "proper_case"      # "Krishna"
    TITLE_CASE = "title_case"        # "Bhagavad Gita"  
    ALL_CAPS = "all_caps"            # "OM"
    SENTENCE_CASE = "sentence_case"  # "dharma"
    MIXED_CASE = "mixed_case"        # "McArthur"


@dataclass
class CapitalizationResult:
    """Result of capitalization operation."""
    original_text: str
    capitalized_text: str
    rule_applied: CapitalizationRule
    confidence: float
    entities_processed: int
    changes_made: List[Dict[str, Any]]
    validation_passed: bool


@dataclass
class CapitalizationChange:
    """Represents a single capitalization change."""
    original: str
    corrected: str
    start_pos: int
    end_pos: int
    entity_category: EntityCategory
    rule_applied: CapitalizationRule
    confidence: float


class CapitalizationEngine:
    """
    Engine for applying proper capitalization to Yoga Vedanta proper nouns.
    
    Integrates with NER system to identify proper nouns and applies appropriate
    capitalization rules based on category, context, and lexicon data.
    """
    
    def __init__(self, ner_model: YogaVedantaNER = None, lexicon_manager: LexiconManager = None):
        """
        Initialize the capitalization engine.
        
        Args:
            ner_model: Optional YogaVedantaNER instance
            lexicon_manager: Optional LexiconManager instance
        """
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.ner_model = ner_model or YogaVedantaNER()
        self.lexicon_manager = lexicon_manager or LexiconManager()
        self.entity_classifier = EntityClassifier()
        
        # Capitalization rules mapping
        self.category_rules = self._initialize_category_rules()
        
        # Special case patterns
        self.special_patterns = self._initialize_special_patterns()
        
        # Context-sensitive rules
        self.context_rules = self._initialize_context_rules()
        
        self.logger.info("CapitalizationEngine initialized")
    
    def _initialize_category_rules(self) -> Dict[EntityCategory, CapitalizationRule]:
        """Initialize capitalization rules for each entity category."""
        return {
            EntityCategory.SCRIPTURE: CapitalizationRule.TITLE_CASE,
            EntityCategory.DEITY: CapitalizationRule.PROPER_CASE,
            EntityCategory.CHARACTER: CapitalizationRule.PROPER_CASE,
            EntityCategory.TEACHER: CapitalizationRule.PROPER_CASE,
            EntityCategory.PLACE: CapitalizationRule.PROPER_CASE,
            EntityCategory.PHILOSOPHY: CapitalizationRule.PROPER_CASE,
            EntityCategory.UNKNOWN: CapitalizationRule.PROPER_CASE
        }
    
    def _initialize_special_patterns(self) -> Dict[str, str]:
        """Initialize special case capitalization patterns."""
        return {
            # Multi-word scriptures
            r'\bbhagavad\s+gita\b': 'Bhagavad Gita',
            r'\byoga\s+sutras?\b': 'Yoga Sutras',
            r'\bbrahma\s+sutras?\b': 'Brahma Sutras',
            
            # Compound names
            r'\bswami\s+vivekananda\b': 'Swami Vivekananda',
            r'\bsri\s+ramana\s+maharshi\b': 'Sri Ramana Maharshi',
            r'\bparamahansa\s+yogananda\b': 'Paramahansa Yogananda',
            
            # Special honorifics
            r'\bsri\s+krishna\b': 'Sri Krishna',
            r'\blord\s+shiva\b': 'Lord Shiva',
            r'\bshri\s+rama\b': 'Shri Rama',
            
            # Philosophical terms with specific capitalization
            r'\badvaita\s+vedanta\b': 'Advaita Vedanta',
            r'\bkarma\s+yoga\b': 'Karma Yoga',
            r'\bbhakti\s+yoga\b': 'Bhakti Yoga',
            r'\braja\s+yoga\b': 'Raja Yoga',
            r'\bjnana\s+yoga\b': 'Jnana Yoga'
        }
    
    def _initialize_context_rules(self) -> Dict[str, CapitalizationRule]:
        """Initialize context-sensitive capitalization rules."""
        return {
            'sentence_start': CapitalizationRule.PROPER_CASE,
            'after_period': CapitalizationRule.PROPER_CASE,
            'after_colon': CapitalizationRule.PROPER_CASE,
            'in_title': CapitalizationRule.TITLE_CASE,
            'mid_sentence': CapitalizationRule.PROPER_CASE
        }
    
    def capitalize_text(self, text: str) -> CapitalizationResult:
        """
        Apply capitalization to all proper nouns in the text.
        
        Args:
            text: Input text to capitalize
            
        Returns:
            CapitalizationResult with capitalized text and metadata
        """
        original_text = text
        capitalized_text = text
        changes_made = []
        
        # Step 1: Identify entities using NER
        ner_result = self.ner_model.identify_entities(text)
        entities = ner_result.entities
        
        # Step 2: Apply special patterns first
        capitalized_text, pattern_changes = self._apply_special_patterns(capitalized_text)
        changes_made.extend(pattern_changes)
        
        # Step 3: Process identified entities
        entity_changes = self._process_entities(capitalized_text, entities)
        changes_made.extend(entity_changes)
        
        # Apply entity changes to text
        for change in entity_changes:
            # Use regex to replace while preserving case sensitivity
            pattern = re.compile(re.escape(change.original), re.IGNORECASE)
            capitalized_text = pattern.sub(change.corrected, capitalized_text, count=1)
        
        # Step 4: Apply context-sensitive rules
        capitalized_text, context_changes = self._apply_context_rules(capitalized_text, entities)
        changes_made.extend(context_changes)
        
        # Step 5: Validate results
        validation_passed = self._validate_capitalization(original_text, capitalized_text)
        
        return CapitalizationResult(
            original_text=original_text,
            capitalized_text=capitalized_text,
            rule_applied=CapitalizationRule.MIXED_CASE,  # Multiple rules applied
            confidence=self._calculate_overall_confidence(changes_made),
            entities_processed=len(entities),
            changes_made=[{
                'original': change.original,
                'corrected': change.corrected,
                'position': change.start_pos,
                'category': change.entity_category.value,
                'rule': change.rule_applied.value,
                'confidence': change.confidence
            } for change in changes_made if isinstance(change, CapitalizationChange)],
            validation_passed=validation_passed
        )
    
    def _apply_special_patterns(self, text: str) -> Tuple[str, List[CapitalizationChange]]:
        """Apply special capitalization patterns."""
        result_text = text
        changes = []
        
        for pattern, replacement in self.special_patterns.items():
            matches = list(re.finditer(pattern, result_text, re.IGNORECASE))
            for match in reversed(matches):  # Reverse to maintain positions
                original = match.group()
                if original != replacement:  # Only if change needed
                    change = CapitalizationChange(
                        original=original,
                        corrected=replacement,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        entity_category=EntityCategory.UNKNOWN,  # Will be updated
                        rule_applied=CapitalizationRule.TITLE_CASE,
                        confidence=0.95
                    )
                    changes.append(change)
                    
                    # Apply the change
                    result_text = (result_text[:match.start()] + 
                                 replacement + 
                                 result_text[match.end():])
        
        return result_text, changes
    
    def _process_entities(self, text: str, entities: List[NamedEntity]) -> List[CapitalizationChange]:
        """Process individual entities for capitalization."""
        changes = []
        
        for entity in entities:
            # Skip if already processed by special patterns
            if self._is_already_capitalized(entity.text):
                continue
            
            # Get appropriate capitalization rule
            rule = self.category_rules.get(entity.category, CapitalizationRule.PROPER_CASE)
            
            # Apply the rule
            capitalized = self._apply_capitalization_rule(entity.text, rule)
            
            if capitalized != entity.original_text:
                change = CapitalizationChange(
                    original=entity.original_text,
                    corrected=capitalized,
                    start_pos=entity.start_pos,
                    end_pos=entity.end_pos,
                    entity_category=entity.category,
                    rule_applied=rule,
                    confidence=entity.confidence * 0.9  # Slight confidence reduction for capitalization
                )
                changes.append(change)
        
        return changes
    
    def _apply_context_rules(self, text: str, entities: List[NamedEntity]) -> Tuple[str, List[CapitalizationChange]]:
        """Apply context-sensitive capitalization rules."""
        result_text = text
        changes = []
        
        # Check for sentence beginnings
        sentences = re.split(r'[.!?]+\s+', text)
        current_pos = 0
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            # Find first word of sentence
            first_word_match = re.search(r'\b\w+', sentence)
            if first_word_match:
                first_word = first_word_match.group()
                
                # Check if it's an entity that needs capitalization
                for entity in entities:
                    if (entity.start_pos >= current_pos and 
                        entity.start_pos <= current_pos + len(sentence) and
                        entity.original_text.lower() == first_word.lower()):
                        
                        # Apply sentence start capitalization
                        capitalized = first_word.capitalize()
                        if capitalized != first_word:
                            change = CapitalizationChange(
                                original=first_word,
                                corrected=capitalized,
                                start_pos=entity.start_pos,
                                end_pos=entity.start_pos + len(first_word),
                                entity_category=entity.category,
                                rule_applied=CapitalizationRule.PROPER_CASE,
                                confidence=0.8
                            )
                            changes.append(change)
            
            current_pos += len(sentence) + 2  # Account for delimiter
        
        return result_text, changes
    
    def _apply_capitalization_rule(self, text: str, rule: CapitalizationRule) -> str:
        """Apply specific capitalization rule to text."""
        if rule == CapitalizationRule.PROPER_CASE:
            return text.title()
        
        elif rule == CapitalizationRule.TITLE_CASE:
            # Proper title case with articles/prepositions lowercase
            words = text.split()
            result = []
            
            for i, word in enumerate(words):
                if i == 0 or word.lower() not in ['a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with']:
                    result.append(word.capitalize())
                else:
                    result.append(word.lower())
            
            return ' '.join(result)
        
        elif rule == CapitalizationRule.ALL_CAPS:
            return text.upper()
        
        elif rule == CapitalizationRule.SENTENCE_CASE:
            return text.lower()
        
        elif rule == CapitalizationRule.MIXED_CASE:
            # Keep original case for mixed case items
            return text
        
        return text
    
    def _is_already_capitalized(self, text: str) -> bool:
        """Check if text is already properly capitalized."""
        # Simple heuristic: check if first letter is uppercase
        return text and text[0].isupper()
    
    def _validate_capitalization(self, original: str, capitalized: str) -> bool:
        """Validate the capitalization result."""
        # Basic validation checks
        if len(original) != len(capitalized):
            # Allow for small differences due to corrections
            if abs(len(original) - len(capitalized)) > 10:
                return False
        
        # Check that we didn't break sentence structure
        original_sentences = len(re.split(r'[.!?]+', original))
        capitalized_sentences = len(re.split(r'[.!?]+', capitalized))
        
        if original_sentences != capitalized_sentences:
            return False
        
        return True
    
    def _calculate_overall_confidence(self, changes: List[CapitalizationChange]) -> float:
        """Calculate overall confidence for the capitalization result."""
        if not changes:
            return 1.0  # No changes needed = high confidence
        
        total_confidence = sum(change.confidence for change in changes if isinstance(change, CapitalizationChange))
        return min(1.0, total_confidence / len([c for c in changes if isinstance(c, CapitalizationChange)]))
    
    def capitalize_entity(self, entity_text: str, category: EntityCategory) -> str:
        """
        Capitalize a single entity based on its category.
        
        Args:
            entity_text: Text of the entity to capitalize
            category: EntityCategory of the entity
            
        Returns:
            Capitalized entity text
        """
        rule = self.category_rules.get(category, CapitalizationRule.PROPER_CASE)
        return self._apply_capitalization_rule(entity_text, rule)
    
    def add_special_pattern(self, pattern: str, replacement: str) -> bool:
        """
        Add a new special capitalization pattern.
        
        Args:
            pattern: Regex pattern to match
            replacement: Replacement text with proper capitalization
            
        Returns:
            True if successfully added
        """
        try:
            # Test the pattern
            re.compile(pattern)
            self.special_patterns[pattern] = replacement
            self.logger.info(f"Added special pattern: {pattern} -> {replacement}")
            return True
        except re.error as e:
            self.logger.error(f"Invalid regex pattern '{pattern}': {e}")
            return False
    
    def get_capitalization_statistics(self) -> Dict[str, Any]:
        """Get statistics about the capitalization engine."""
        return {
            'category_rules': {cat.value: rule.value for cat, rule in self.category_rules.items()},
            'special_patterns_count': len(self.special_patterns),
            'context_rules_count': len(self.context_rules),
            'supported_categories': [cat.value for cat in self.category_rules.keys()]
        }
    
    def validate_capitalization_rules(self) -> Dict[str, Any]:
        """Validate the capitalization rules configuration."""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check that all entity categories have rules
        for category in EntityCategory:
            if category not in self.category_rules:
                validation_result['warnings'].append(f"No capitalization rule for category: {category.value}")
        
        # Validate special patterns
        for pattern in self.special_patterns.keys():
            try:
                re.compile(pattern)
            except re.error as e:
                validation_result['errors'].append(f"Invalid regex pattern '{pattern}': {e}")
                validation_result['is_valid'] = False
        
        return validation_result