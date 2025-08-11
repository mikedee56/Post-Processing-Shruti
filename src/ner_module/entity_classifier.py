"""
Entity Classification System for Yoga Vedanta NER.

This module provides classification functionality for different types of proper nouns
found in Yoga Vedanta texts, organizing them into meaningful categories.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

from utils.logger_config import get_logger


class EntityCategory(Enum):
    """Entity categories for Yoga Vedanta proper nouns."""
    SCRIPTURE = "SCRIPTURE"     # Sacred texts and books
    DEITY = "DEITY"             # Divine beings and deities
    CHARACTER = "CHARACTER"     # Mythological and epic characters
    TEACHER = "TEACHER"         # Spiritual teachers and gurus
    PLACE = "PLACE"             # Sacred places and locations
    PHILOSOPHY = "PHILOSOPHY"   # Philosophical concepts and systems
    UNKNOWN = "UNKNOWN"         # Unclassified entities


@dataclass
class CategoryMetadata:
    """Metadata for entity categories."""
    id: int
    name: str
    description: str
    capitalization_rule: str
    confidence_threshold: float
    examples: List[str]
    patterns: List[str]


@dataclass
class ClassificationResult:
    """Result of entity classification."""
    entity_text: str
    predicted_category: EntityCategory
    confidence: float
    alternative_categories: List[tuple]  # (category, confidence) pairs
    reasoning: str
    source: str  # 'lexicon', 'pattern', 'heuristic'


class EntityClassifier:
    """
    Classifier for categorizing Yoga Vedanta proper nouns.
    
    Uses multiple approaches:
    1. Direct lexicon lookup
    2. Pattern matching
    3. Heuristic classification based on context and structure
    """
    
    def __init__(self, config_dir: Path = None):
        """
        Initialize the entity classifier.
        
        Args:
            config_dir: Directory containing classification configuration
        """
        self.logger = get_logger(__name__)
        self.config_dir = config_dir or Path("data/ner_training")
        
        # Category metadata and patterns
        self.categories: Dict[EntityCategory, CategoryMetadata] = {}
        self.category_patterns = {}
        self.classification_rules = {}
        
        # Load classification data
        self._load_classification_config()
        
        self.logger.info(f"EntityClassifier initialized with {len(self.categories)} categories")
    
    def _load_classification_config(self) -> None:
        """Load classification configuration from YAML."""
        try:
            config_file = self.config_dir / "entity_categories.yaml"
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                # Load category definitions
                categories_data = data.get('categories', {})
                for category_name, category_info in categories_data.items():
                    try:
                        category_enum = EntityCategory[category_name]
                        metadata = CategoryMetadata(
                            id=category_info.get('id', 0),
                            name=category_info.get('name', category_name),
                            description=category_info.get('description', ''),
                            capitalization_rule=category_info.get('capitalization_rule', 'proper_case'),
                            confidence_threshold=category_info.get('confidence_threshold', 0.8),
                            examples=category_info.get('examples', []),
                            patterns=category_info.get('patterns', [])
                        )
                        self.categories[category_enum] = metadata
                        
                    except KeyError:
                        self.logger.warning(f"Unknown category: {category_name}")
                
                # Load classification rules if present
                self.classification_rules = data.get('classification_rules', {})
                
                self.logger.info(f"Loaded configuration for {len(self.categories)} categories")
                
            else:
                self.logger.warning(f"Classification config not found: {config_file}")
                self._initialize_default_categories()
                
        except Exception as e:
            self.logger.error(f"Error loading classification config: {e}")
            self._initialize_default_categories()
    
    def _initialize_default_categories(self) -> None:
        """Initialize default categories if config loading fails."""
        default_categories = {
            EntityCategory.SCRIPTURE: CategoryMetadata(
                id=0, name="Scripture", description="Sacred texts",
                capitalization_rule="title_case", confidence_threshold=0.8,
                examples=["Bhagavad Gita", "Upanishads"], patterns=[]
            ),
            EntityCategory.DEITY: CategoryMetadata(
                id=1, name="Deity", description="Divine beings",
                capitalization_rule="proper_case", confidence_threshold=0.9,
                examples=["Krishna", "Shiva"], patterns=[]
            ),
            EntityCategory.CHARACTER: CategoryMetadata(
                id=2, name="Character", description="Epic characters",
                capitalization_rule="proper_case", confidence_threshold=0.85,
                examples=["Arjuna", "Rama"], patterns=[]
            ),
            EntityCategory.TEACHER: CategoryMetadata(
                id=3, name="Teacher", description="Spiritual teachers",
                capitalization_rule="proper_case", confidence_threshold=0.85,
                examples=["Patanjali", "Shankaracharya"], patterns=[]
            ),
            EntityCategory.PLACE: CategoryMetadata(
                id=4, name="Place", description="Sacred places",
                capitalization_rule="proper_case", confidence_threshold=0.8,
                examples=["Rishikesh", "Varanasi"], patterns=[]
            ),
            EntityCategory.PHILOSOPHY: CategoryMetadata(
                id=5, name="Philosophy", description="Philosophical concepts",
                capitalization_rule="proper_case", confidence_threshold=0.75,
                examples=["Vedanta", "Advaita"], patterns=[]
            )
        }
        
        self.categories = default_categories
        self.logger.info("Initialized with default categories")
    
    def classify_entity(self, entity_text: str, lexicon_category: str = None) -> EntityCategory:
        """
        Classify an entity into the most appropriate category.
        
        Args:
            entity_text: The entity text to classify
            lexicon_category: Optional category from lexicon
            
        Returns:
            EntityCategory enum value
        """
        result = self.classify_with_confidence(entity_text, lexicon_category)
        return result.predicted_category
    
    def classify_with_confidence(self, entity_text: str, lexicon_category: str = None) -> ClassificationResult:
        """
        Classify an entity with detailed confidence scoring.
        
        Args:
            entity_text: The entity text to classify
            lexicon_category: Optional category from lexicon
            
        Returns:
            ClassificationResult with detailed classification information
        """
        entity_lower = entity_text.lower().strip()
        
        # Method 1: Direct lexicon category mapping
        if lexicon_category:
            mapped_category = self._map_lexicon_category(lexicon_category)
            if mapped_category != EntityCategory.UNKNOWN:
                return ClassificationResult(
                    entity_text=entity_text,
                    predicted_category=mapped_category,
                    confidence=0.95,
                    alternative_categories=[],
                    reasoning=f"Direct lexicon mapping from '{lexicon_category}'",
                    source='lexicon'
                )
        
        # Method 2: Example-based classification
        for category, metadata in self.categories.items():
            for example in metadata.examples:
                if entity_lower == example.lower() or entity_lower in example.lower():
                    return ClassificationResult(
                        entity_text=entity_text,
                        predicted_category=category,
                        confidence=0.90,
                        alternative_categories=[],
                        reasoning=f"Matches example '{example}' for category {category.value}",
                        source='example'
                    )
        
        # Method 3: Heuristic classification
        heuristic_result = self._classify_by_heuristics(entity_text)
        if heuristic_result:
            return heuristic_result
        
        # Method 4: Default to UNKNOWN with low confidence
        return ClassificationResult(
            entity_text=entity_text,
            predicted_category=EntityCategory.UNKNOWN,
            confidence=0.1,
            alternative_categories=[],
            reasoning="No classification method succeeded",
            source='default'
        )
    
    def _map_lexicon_category(self, lexicon_category: str) -> EntityCategory:
        """Map lexicon category strings to EntityCategory enums."""
        category_mappings = {
            'deity': EntityCategory.DEITY,
            'scripture': EntityCategory.SCRIPTURE,
            'character': EntityCategory.CHARACTER,
            'teacher': EntityCategory.TEACHER,
            'place': EntityCategory.PLACE,
            'philosophy': EntityCategory.PHILOSOPHY,
            'concept': EntityCategory.PHILOSOPHY,  # Map concept to philosophy
            'practice': EntityCategory.PHILOSOPHY,  # Map practice to philosophy
            'temporal': EntityCategory.PHILOSOPHY,  # Map temporal to philosophy
            'reference': EntityCategory.SCRIPTURE   # Map reference to scripture
        }
        
        return category_mappings.get(lexicon_category.lower(), EntityCategory.UNKNOWN)
    
    def _classify_by_heuristics(self, entity_text: str) -> Optional[ClassificationResult]:
        """Classify entity using heuristic rules."""
        entity_lower = entity_text.lower().strip()
        
        # Scripture heuristics
        scripture_indicators = ['gita', 'sutra', 'upanishad', 'veda', 'purana', 'tantra', 'shastra']
        if any(indicator in entity_lower for indicator in scripture_indicators):
            return ClassificationResult(
                entity_text=entity_text,
                predicted_category=EntityCategory.SCRIPTURE,
                confidence=0.75,
                alternative_categories=[],
                reasoning=f"Contains scripture indicators: {[i for i in scripture_indicators if i in entity_lower]}",
                source='heuristic'
            )
        
        # Philosophy heuristics
        philosophy_indicators = ['yoga', 'vedanta', 'advaita', 'dharma', 'karma', 'moksha', 'samadhi']
        if any(indicator in entity_lower for indicator in philosophy_indicators):
            return ClassificationResult(
                entity_text=entity_text,
                predicted_category=EntityCategory.PHILOSOPHY,
                confidence=0.70,
                alternative_categories=[],
                reasoning=f"Contains philosophy indicators: {[i for i in philosophy_indicators if i in entity_lower]}",
                source='heuristic'
            )
        
        # Teacher heuristics (common prefixes/suffixes)
        teacher_indicators = ['swami', 'guru', 'acharya', 'maharshi', 'rishi']
        if any(indicator in entity_lower for indicator in teacher_indicators):
            return ClassificationResult(
                entity_text=entity_text,
                predicted_category=EntityCategory.TEACHER,
                confidence=0.80,
                alternative_categories=[],
                reasoning=f"Contains teacher indicators: {[i for i in teacher_indicators if i in entity_lower]}",
                source='heuristic'
            )
        
        # Place heuristics (common suffixes)
        place_indicators = ['pur', 'giri', 'puram', 'nagar', 'kshetra']
        if any(entity_lower.endswith(indicator) for indicator in place_indicators):
            return ClassificationResult(
                entity_text=entity_text,
                predicted_category=EntityCategory.PLACE,
                confidence=0.65,
                alternative_categories=[],
                reasoning=f"Ends with place indicators: {[i for i in place_indicators if entity_lower.endswith(i)]}",
                source='heuristic'
            )
        
        return None
    
    def get_category_by_name(self, category_name: str) -> EntityCategory:
        """Get EntityCategory by string name."""
        try:
            return EntityCategory[category_name.upper()]
        except KeyError:
            self.logger.warning(f"Unknown category name: {category_name}")
            return EntityCategory.UNKNOWN
    
    def get_category_metadata(self, category: EntityCategory) -> Optional[CategoryMetadata]:
        """Get metadata for a specific category."""
        return self.categories.get(category)
    
    def get_capitalization_rule(self, category: EntityCategory) -> str:
        """Get the capitalization rule for a category."""
        metadata = self.categories.get(category)
        if metadata:
            return metadata.capitalization_rule
        return "proper_case"  # default
    
    def get_confidence_threshold(self, category: EntityCategory) -> float:
        """Get the confidence threshold for a category."""
        metadata = self.categories.get(category)
        if metadata:
            return metadata.confidence_threshold
        return 0.8  # default
    
    def get_all_categories(self) -> List[EntityCategory]:
        """Get all available entity categories."""
        return list(self.categories.keys())
    
    def get_category_examples(self, category: EntityCategory) -> List[str]:
        """Get examples for a specific category."""
        metadata = self.categories.get(category)
        if metadata:
            return metadata.examples
        return []
    
    def validate_classification(self, entity_text: str, predicted_category: EntityCategory, 
                              confidence: float) -> bool:
        """
        Validate a classification result.
        
        Args:
            entity_text: The entity text
            predicted_category: Predicted category
            confidence: Confidence score
            
        Returns:
            True if classification passes validation
        """
        # Check if confidence meets threshold
        threshold = self.get_confidence_threshold(predicted_category)
        if confidence < threshold:
            return False
        
        # Additional validation logic can be added here
        
        return True
    
    def get_classification_statistics(self) -> Dict[str, Any]:
        """Get statistics about the classification system."""
        return {
            'total_categories': len(self.categories),
            'categories': {cat.value: {
                'examples_count': len(metadata.examples),
                'confidence_threshold': metadata.confidence_threshold,
                'capitalization_rule': metadata.capitalization_rule
            } for cat, metadata in self.categories.items()},
            'heuristic_rules_available': True
        }