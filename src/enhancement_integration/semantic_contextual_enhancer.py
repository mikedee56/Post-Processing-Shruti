"""
Semantic Contextual Enhancement for Story 2.4.4

This module enhances Story 2.2 contextual modeling with semantic similarity validation
and phonetic contextual matching capabilities.

Key Features:
- Semantic similarity validation for n-gram language models
- Higher confidence when syntactic and semantic models agree
- Phonetic pattern matching across spelling variations
- Integration with existing contextual rule engine
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import time
import logging

from contextual_modeling.ngram_language_model import (
    NGramLanguageModel, 
    ContextPrediction, 
    NGramModelConfig
)
from contextual_modeling.contextual_rule_engine import ContextualRuleEngine
from contextual_modeling.phonetic_encoder import PhoneticEncoder
from contextual_modeling.semantic_similarity_calculator import SemanticSimilarityCalculator
from enhancement_integration.unified_confidence_scorer import (
    UnifiedConfidenceScorer, 
    ConfidenceSource, 
    ConfidenceScore
)
from utils.logger_config import get_logger


@dataclass
class EnhancedContextPrediction:
    """Enhanced context prediction with semantic validation."""
    base_prediction: ContextPrediction
    semantic_similarity_score: float
    phonetic_variations: List[str]
    agreement_level: str  # "high", "moderate", "low"
    enhanced_confidence: float
    validation_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticValidationResult:
    """Result of semantic validation against context."""
    is_semantically_valid: bool
    semantic_score: float
    context_coherence: float
    validation_method: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PhoneticContextMatch:
    """Result of phonetic contextual matching."""
    original_text: str
    phonetic_variations: List[str]
    context_matches: List[str]
    phonetic_confidence: float
    context_support: float


class SemanticContextualEnhancer:
    """
    Semantic enhancement for Story 2.2 contextual modeling.
    
    This component implements AC3 and AC4 of Story 2.4.4, providing:
    - Semantic similarity validation for n-gram predictions (AC3)
    - Phonetic contextual matching across spelling variations (AC4)
    - Higher confidence when syntactic and semantic models agree
    - Integration with existing Story 2.2 components
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize semantic contextual enhancer.
        
        Args:
            config: Configuration parameters for enhancement
        """
        self.logger = get_logger(__name__)
        self.config = config or {}
        
        # Initialize components
        self.semantic_calculator = SemanticSimilarityCalculator()
        self.phonetic_encoder = PhoneticEncoder()
        self.confidence_scorer = UnifiedConfidenceScorer()
        
        # Configuration parameters
        self.semantic_threshold = self.config.get('semantic_threshold', 0.7)
        self.agreement_threshold = self.config.get('agreement_threshold', 0.8)
        self.phonetic_threshold = self.config.get('phonetic_threshold', 0.75)
        self.context_window_size = self.config.get('context_window_size', 5)
        
        # Performance tracking
        self.enhancement_stats = {
            'total_predictions_enhanced': 0,
            'semantic_validations': 0,
            'phonetic_matches': 0,
            'agreement_boosts': 0,
            'total_processing_time': 0.0
        }
        
        self.logger.info("Semantic contextual enhancer initialized")
    
    def enhance_ngram_predictions(
        self, 
        predictions: List[ContextPrediction],
        context_text: str,
        target_domain: str = "spiritual"
    ) -> List[EnhancedContextPrediction]:
        """
        Enhance n-gram predictions with semantic validation.
        
        Args:
            predictions: Original n-gram predictions
            context_text: Full context text for semantic analysis
            target_domain: Domain for semantic validation (e.g., "spiritual", "academic")
            
        Returns:
            List of enhanced predictions with semantic validation
        """
        start_time = time.time()
        enhanced_predictions = []
        
        for prediction in predictions:
            # Perform semantic validation
            semantic_validation = self._validate_semantic_coherence(
                prediction, context_text, target_domain
            )
            
            # Find phonetic variations
            phonetic_variations = self._find_phonetic_variations(
                prediction.word, context_text
            )
            
            # Determine agreement level
            agreement_level = self._assess_agreement_level(
                prediction, semantic_validation, phonetic_variations
            )
            
            # Calculate enhanced confidence
            enhanced_confidence = self._calculate_enhanced_confidence(
                prediction, semantic_validation, agreement_level
            )
            
            # Create enhanced prediction
            enhanced_pred = EnhancedContextPrediction(
                base_prediction=prediction,
                semantic_similarity_score=semantic_validation.semantic_score,
                phonetic_variations=phonetic_variations,
                agreement_level=agreement_level,
                enhanced_confidence=enhanced_confidence,
                validation_metadata={
                    'semantic_validation': semantic_validation,
                    'context_length': len(context_text),
                    'target_domain': target_domain
                }
            )
            enhanced_predictions.append(enhanced_pred)
            
            self.enhancement_stats['total_predictions_enhanced'] += 1
            if semantic_validation.is_semantically_valid:
                self.enhancement_stats['semantic_validations'] += 1
            if phonetic_variations:
                self.enhancement_stats['phonetic_matches'] += 1
            if agreement_level == "high":
                self.enhancement_stats['agreement_boosts'] += 1
        
        # Sort by enhanced confidence
        enhanced_predictions.sort(key=lambda p: p.enhanced_confidence, reverse=True)
        
        processing_time = time.time() - start_time
        self.enhancement_stats['total_processing_time'] += processing_time
        
        return enhanced_predictions
    
    def _validate_semantic_coherence(
        self, 
        prediction: ContextPrediction,
        context_text: str,
        target_domain: str
    ) -> SemanticValidationResult:
        """Validate semantic coherence of prediction with context."""
        # Extract relevant context window
        context_window = self._extract_context_window(context_text, prediction.context)
        
        # Build candidate sentence with prediction
        candidate_text = " ".join(prediction.context + [prediction.word])
        
        # Calculate semantic similarity with context
        similarity_result = self.semantic_calculator.compute_semantic_similarity(
            candidate_text, context_window
        )
        
        semantic_score = similarity_result.similarity_score
        is_valid = semantic_score >= self.semantic_threshold
        
        # Calculate context coherence (how well the prediction fits the domain)
        coherence_score = self._calculate_domain_coherence(
            prediction.word, target_domain, context_window
        )
        
        return SemanticValidationResult(
            is_semantically_valid=is_valid,
            semantic_score=semantic_score,
            context_coherence=coherence_score,
            validation_method="semantic_similarity",
            metadata={
                'context_window': context_window,
                'candidate_text': candidate_text,
                'similarity_result': similarity_result.to_dict()
            }
        )
    
    def _extract_context_window(self, full_context: str, immediate_context: List[str]) -> str:
        """Extract relevant context window for semantic analysis."""
        # Use immediate context first
        if immediate_context:
            context_window = " ".join(immediate_context)
        else:
            # Fallback to last N words from full context
            words = full_context.split()
            context_window = " ".join(words[-self.context_window_size:])
        
        return context_window
    
    def _calculate_domain_coherence(
        self, 
        word: str, 
        target_domain: str, 
        context: str
    ) -> float:
        """Calculate how well a word fits the target domain context."""
        # Domain-specific vocabulary lists (can be expanded)
        domain_vocabularies = {
            "spiritual": [
                "dharma", "karma", "yoga", "meditation", "consciousness", "soul", 
                "divine", "sacred", "enlightenment", "wisdom", "truth", "peace",
                "bhagavad", "gita", "vedanta", "upanishad", "sanskrit", "vedas"
            ],
            "academic": [
                "study", "research", "analysis", "theory", "concept", "principle",
                "methodology", "framework", "literature", "evidence", "hypothesis"
            ],
            "general": [
                "today", "discuss", "understand", "explain", "learn", "practice"
            ]
        }
        
        domain_vocab = domain_vocabularies.get(target_domain, domain_vocabularies["general"])
        
        # Check if word or its root is in domain vocabulary
        word_lower = word.lower()
        direct_match = any(vocab_word in word_lower or word_lower in vocab_word 
                          for vocab_word in domain_vocab)
        
        if direct_match:
            return 1.0
        
        # Check context for domain vocabulary
        context_lower = context.lower()
        domain_words_in_context = sum(1 for vocab_word in domain_vocab 
                                     if vocab_word in context_lower)
        
        # Calculate coherence based on domain presence in context
        coherence = min(1.0, domain_words_in_context / max(len(domain_vocab) * 0.1, 1))
        
        return coherence
    
    def _find_phonetic_variations(self, word: str, context_text: str) -> List[str]:
        """Find phonetic variations of word that appear in context."""
        variations = []
        
        # Generate phonetic variations using phonetic encoder
        similarity_results = self.phonetic_encoder.find_phonetically_similar_words(
            word, context_text.split(), threshold=self.phonetic_threshold
        )
        
        for result in similarity_results:
            if result.similarity_score >= self.phonetic_threshold:
                variations.append(result.similar_word)
        
        return variations
    
    def _assess_agreement_level(
        self, 
        prediction: ContextPrediction,
        semantic_validation: SemanticValidationResult,
        phonetic_variations: List[str]
    ) -> str:
        """Assess agreement level between different validation methods."""
        # Score components
        syntactic_score = prediction.confidence_score
        semantic_score = semantic_validation.semantic_score if semantic_validation.is_semantically_valid else 0.0
        phonetic_support = min(1.0, len(phonetic_variations) * 0.3)  # Max 1.0
        
        # Calculate overall agreement
        scores = [syntactic_score, semantic_score, phonetic_support]
        avg_score = sum(scores) / len(scores)
        score_variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
        
        # Determine agreement level
        if avg_score >= self.agreement_threshold and score_variance < 0.05:
            return "high"
        elif avg_score >= 0.6 and score_variance < 0.1:
            return "moderate"
        else:
            return "low"
    
    def _calculate_enhanced_confidence(
        self,
        prediction: ContextPrediction,
        semantic_validation: SemanticValidationResult,
        agreement_level: str
    ) -> float:
        """Calculate enhanced confidence using unified confidence scorer."""
        confidence_scores = []
        
        # N-gram contextual confidence
        contextual_confidence = ConfidenceScore(
            value=prediction.confidence_score,
            source=ConfidenceSource.CONTEXTUAL_MODELING,
            weight=1.0,
            metadata={
                'ngram_order': prediction.ngram_order,
                'log_probability': prediction.log_probability
            }
        )
        confidence_scores.append(contextual_confidence)
        
        # Semantic similarity confidence
        if semantic_validation.is_semantically_valid:
            semantic_confidence = ConfidenceScore(
                value=semantic_validation.semantic_score,
                source=ConfidenceSource.SEMANTIC_SIMILARITY,
                weight=0.9,
                metadata={
                    'context_coherence': semantic_validation.context_coherence,
                    'validation_method': semantic_validation.validation_method
                }
            )
            confidence_scores.append(semantic_confidence)
        
        # Combine confidences
        combination_method = "weighted_average" if agreement_level == "high" else "harmonic_mean"
        result = self.confidence_scorer.combine_confidence_scores(
            confidence_scores, method=combination_method
        )
        
        # Apply agreement boost
        final_confidence = result.final_confidence
        if agreement_level == "high" and final_confidence >= 0.8:
            boost = min(0.1, 1.0 - final_confidence)
            final_confidence += boost
        
        return final_confidence
    
    def enhance_contextual_rules(
        self,
        rule_engine: ContextualRuleEngine,
        text: str,
        context_words: List[str]
    ) -> Dict[str, Any]:
        """
        Enhance contextual rule matching with phonetic pattern support.
        
        Args:
            rule_engine: Existing contextual rule engine
            text: Text to apply rules to
            context_words: Context words for rule application
            
        Returns:
            Enhanced rule matching results
        """
        start_time = time.time()
        
        # Apply standard contextual rules first
        standard_matches = rule_engine.apply_contextual_rules(text, context_words)
        
        # Enhance with phonetic pattern matching
        phonetic_matches = self._find_phonetic_contextual_matches(text, context_words)
        
        # Combine and validate matches
        all_matches = standard_matches + [m.original_text for m in phonetic_matches]
        unique_matches = list(set(all_matches))
        
        # Calculate enhanced confidence for each match
        enhanced_results = {}
        for match in unique_matches:
            # Check if match was found by standard rules
            in_standard = match in [m.original_text for m in standard_matches]
            
            # Check if match was found by phonetic rules
            phonetic_match = next(
                (m for m in phonetic_matches if m.original_text == match), None
            )
            in_phonetic = phonetic_match is not None
            
            # Calculate confidence based on multiple sources
            confidence_scores = []
            
            if in_standard:
                rule_confidence = ConfidenceScore(
                    value=0.9,  # High confidence for rule-based matches
                    source=ConfidenceSource.CONTEXTUAL_MODELING,
                    weight=1.0,
                    metadata={'match_type': 'rule_based'}
                )
                confidence_scores.append(rule_confidence)
            
            if in_phonetic:
                phonetic_confidence = ConfidenceScore(
                    value=phonetic_match.phonetic_confidence,
                    source=ConfidenceSource.PHONETIC_HASHING,
                    weight=0.8,
                    metadata={'phonetic_variations': phonetic_match.phonetic_variations}
                )
                confidence_scores.append(phonetic_confidence)
            
            # Combine confidences
            if confidence_scores:
                result = self.confidence_scorer.combine_confidence_scores(
                    confidence_scores, method="weighted_average"
                )
                enhanced_confidence = result.final_confidence
            else:
                enhanced_confidence = 0.5  # Default confidence
            
            enhanced_results[match] = {
                'confidence': enhanced_confidence,
                'found_by_standard_rules': in_standard,
                'found_by_phonetic_rules': in_phonetic,
                'phonetic_variations': phonetic_match.phonetic_variations if phonetic_match else [],
                'context_support': phonetic_match.context_support if phonetic_match else 0.0
            }
        
        processing_time = time.time() - start_time
        
        return {
            'enhanced_matches': enhanced_results,
            'total_matches': len(unique_matches),
            'standard_matches': len(standard_matches),
            'phonetic_matches': len(phonetic_matches),
            'processing_time': processing_time
        }
    
    def _find_phonetic_contextual_matches(
        self, 
        text: str, 
        context_words: List[str]
    ) -> List[PhoneticContextMatch]:
        """Find contextual matches using phonetic pattern matching."""
        matches = []
        words = text.split()
        
        for word in words:
            if len(word) >= 3:  # Only process words with minimum length
                # Find phonetic variations of the word
                phonetic_variations = []
                phonetic_codes = self.phonetic_encoder.encode_text_batch([word])
                
                if word in phonetic_codes:
                    word_code = phonetic_codes[word]
                    
                    # Find words in context with similar phonetic codes
                    context_codes = self.phonetic_encoder.encode_text_batch(context_words)
                    
                    for context_word, context_code in context_codes.items():
                        similarity = self.phonetic_encoder.calculate_phonetic_similarity(
                            word, context_word
                        )
                        
                        if similarity.similarity_score >= self.phonetic_threshold:
                            phonetic_variations.append(context_word)
                
                # Calculate context support
                context_support = len(phonetic_variations) / max(len(context_words), 1)
                
                # Calculate phonetic confidence
                phonetic_confidence = min(1.0, len(phonetic_variations) * 0.4)
                
                if phonetic_variations and phonetic_confidence >= 0.3:
                    match = PhoneticContextMatch(
                        original_text=word,
                        phonetic_variations=phonetic_variations,
                        context_matches=context_words,
                        phonetic_confidence=phonetic_confidence,
                        context_support=context_support
                    )
                    matches.append(match)
        
        return matches
    
    def get_enhancement_statistics(self) -> Dict[str, Any]:
        """Get enhancement statistics and performance metrics."""
        total_predictions = self.enhancement_stats['total_predictions_enhanced']
        
        return {
            'total_predictions_enhanced': total_predictions,
            'semantic_validation_rate': f"{(self.enhancement_stats['semantic_validations'] / max(total_predictions, 1)) * 100:.1f}%",
            'phonetic_match_rate': f"{(self.enhancement_stats['phonetic_matches'] / max(total_predictions, 1)) * 100:.1f}%",
            'agreement_boost_rate': f"{(self.enhancement_stats['agreement_boosts'] / max(total_predictions, 1)) * 100:.1f}%",
            'average_processing_time': f"{self.enhancement_stats['total_processing_time'] / max(total_predictions, 1):.4f}s",
            'configuration': {
                'semantic_threshold': self.semantic_threshold,
                'agreement_threshold': self.agreement_threshold,
                'phonetic_threshold': self.phonetic_threshold,
                'context_window_size': self.context_window_size
            }
        }
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate enhancer configuration."""
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check thresholds
        thresholds = [
            ('semantic_threshold', self.semantic_threshold),
            ('agreement_threshold', self.agreement_threshold),
            ('phonetic_threshold', self.phonetic_threshold)
        ]
        
        for name, value in thresholds:
            if not 0.0 <= value <= 1.0:
                validation['errors'].append(f"Invalid {name}: {value} (must be 0.0-1.0)")
                validation['is_valid'] = False
        
        # Check component initialization
        components = [
            ('semantic_calculator', self.semantic_calculator),
            ('phonetic_encoder', self.phonetic_encoder),
            ('confidence_scorer', self.confidence_scorer)
        ]
        
        for name, component in components:
            if component is None:
                validation['errors'].append(f"{name} not initialized")
                validation['is_valid'] = False
        
        # Check context window size
        if self.context_window_size <= 0:
            validation['errors'].append(f"Invalid context_window_size: {self.context_window_size}")
            validation['is_valid'] = False
        
        return validation