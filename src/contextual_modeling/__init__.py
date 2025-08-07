"""
Contextual Modeling Module for Story 2.2 and Story 2.4.2

This module provides advanced contextual modeling capabilities for Sanskrit/Hindi
text processing, including n-gram language models, phonetic encoding, contextual
rule engines, spelling normalization, and semantic similarity computation.

Story 2.4.2 Semantic Similarity Components:
- SemanticSimilarityCalculator: Core iNLTK-based semantic similarity computation
- SemanticCacheManager: File-based embedding cache management
- BatchSemanticProcessor: High-performance batch processing capabilities
- SemanticContextualIntegrator: Integration with existing contextual modeling
"""

__version__ = "2.4.2"

# Core Story 2.2 exports (maintained for backward compatibility)
from .ngram_language_model import NGramLanguageModel, NGramModelConfig, ContextPrediction
from .phonetic_encoder import PhoneticEncoder
from .contextual_rule_engine import ContextualRuleEngine, ContextualRule, RuleType
from .spelling_normalizer import SpellingNormalizer

# Story 2.4.2 Semantic Similarity exports
from .semantic_similarity_calculator import (
    SemanticSimilarityCalculator,
    SemanticSimilarityResult,
    SemanticVectorCache,
    LanguageModel
)
from .semantic_cache_manager import SemanticCacheManager, CacheStatistics
from .batch_semantic_processor import (
    BatchSemanticProcessor,
    BatchProcessingConfig,
    BatchProcessingResult
)
from .semantic_contextual_integration import (
    SemanticContextualIntegrator,
    SemanticValidationMode,
    EnhancedContextualMatch,
    EnhancedContextPrediction
)

__all__ = [
    # Story 2.2 components
    "NGramLanguageModel",
    "NGramModelConfig", 
    "ContextPrediction",
    "PhoneticEncoder",
    "ContextualRuleEngine",
    "ContextualRule",
    "RuleType",
    "SpellingNormalizer",
    
    # Story 2.4.2 components
    "SemanticSimilarityCalculator",
    "SemanticSimilarityResult",
    "SemanticVectorCache",
    "LanguageModel",
    "SemanticCacheManager",
    "CacheStatistics",
    "BatchSemanticProcessor",
    "BatchProcessingConfig",
    "BatchProcessingResult",
    "SemanticContextualIntegrator",
    "SemanticValidationMode",
    "EnhancedContextualMatch",
    "EnhancedContextPrediction"
]