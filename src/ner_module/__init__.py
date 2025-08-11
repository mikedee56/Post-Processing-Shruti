"""
Named Entity Recognition (NER) Module for Yoga Vedanta Texts.

This module provides comprehensive NER functionality for identifying and
correctly capitalizing proper nouns specific to Yoga Vedanta domain.

Components:
- YogaVedantaNER: Core NER model for entity identification
- EntityClassifier: Classification system for different entity types  
- CapitalizationEngine: Intelligent capitalization based on entity types
- NERModelManager: Management system for model versioning and expansion

Usage:
    from ner_module import YogaVedantaNER, CapitalizationEngine
    
    ner_model = YogaVedantaNER()
    cap_engine = CapitalizationEngine(ner_model)
    
    entities = ner_model.identify_entities("Today we study Krishna and dharma")
    capitalized = cap_engine.capitalize_text("today we study krishna")
"""

from .yoga_vedanta_ner import YogaVedantaNER, NamedEntity, NERResult, NERConfidenceLevel
from .entity_classifier import EntityClassifier, EntityCategory, ClassificationResult
from .capitalization_engine import CapitalizationEngine, CapitalizationResult, CapitalizationRule
from .ner_model_manager import NERModelManager, ProperNounSuggestion, SuggestionSource, ModelStatus

__all__ = [
    # Core NER functionality
    'YogaVedantaNER',
    'NamedEntity', 
    'NERResult',
    'NERConfidenceLevel',
    
    # Entity classification
    'EntityClassifier',
    'EntityCategory',
    'ClassificationResult',
    
    # Capitalization
    'CapitalizationEngine',
    'CapitalizationResult',
    'CapitalizationRule',
    
    # Model management
    'NERModelManager',
    'ProperNounSuggestion',
    'SuggestionSource',
    'ModelStatus'
]

__version__ = "1.0.0"
__author__ = "Yoga Vedanta Post-Processing System"