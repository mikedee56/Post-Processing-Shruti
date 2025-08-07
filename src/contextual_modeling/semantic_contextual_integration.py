"""
Semantic Contextual Integration for Story 2.4.2 

This module provides integration between the semantic similarity calculator and 
existing Story 2.2 contextual modeling components, enhancing contextual rules
and n-gram models with semantic validation capabilities.

Architecture Integration:
- Enhances existing contextual rules with semantic validation (AC6)
- Adds semantic context validation to existing n-gram models
- Ensures backward compatibility with existing Story 2.2 functionality (AC8)
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from utils.logger_config import get_logger
from .semantic_similarity_calculator import SemanticSimilarityCalculator, SemanticSimilarityResult
from .contextual_rule_engine import ContextualRuleEngine, ContextualRule, RuleType
from .ngram_language_model import NGramLanguageModel, ContextPrediction


class SemanticValidationMode(Enum):
    """Modes for semantic validation integration."""
    DISABLED = "disabled"
    ADVISORY = "advisory"  # Semantic scores provided but don't override decisions
    HYBRID = "hybrid"      # Semantic scores influence final confidence
    SEMANTIC_FIRST = "semantic_first"  # Semantic scores take precedence


@dataclass
class EnhancedContextualMatch:
    """Enhanced contextual match with semantic validation."""
    original_text: str
    corrected_text: str
    rule_confidence: float
    semantic_similarity: Optional[float] = None
    combined_confidence: float = 0.0
    validation_mode: str = "disabled"
    semantic_validation_passed: bool = True
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'original_text': self.original_text,
            'corrected_text': self.corrected_text,
            'rule_confidence': self.rule_confidence,
            'semantic_similarity': self.semantic_similarity,
            'combined_confidence': self.combined_confidence,
            'validation_mode': self.validation_mode,
            'semantic_validation_passed': self.semantic_validation_passed,
            'metadata': self.metadata or {}
        }


@dataclass
class EnhancedContextPrediction:
    """Enhanced context prediction with semantic validation."""
    word: str
    ngram_probability: float
    semantic_coherence: Optional[float] = None
    combined_score: float = 0.0
    validation_mode: str = "disabled"
    semantic_validation_passed: bool = True
    supporting_context: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'word': self.word,
            'ngram_probability': self.ngram_probability,
            'semantic_coherence': self.semantic_coherence,
            'combined_score': self.combined_score,
            'validation_mode': self.validation_mode,
            'semantic_validation_passed': self.semantic_validation_passed,
            'supporting_context': self.supporting_context or []
        }


class SemanticContextualIntegrator:
    """
    Integration layer between semantic similarity and contextual modeling.
    
    This component provides:
    1. Enhanced contextual rule validation with semantic coherence (AC6)
    2. N-gram model enhancement with semantic context validation (AC6)
    3. Backward compatibility with existing Story 2.2 functionality (AC8)
    4. Configurable semantic validation modes
    5. Performance optimization with selective semantic validation
    """
    
    def __init__(
        self,
        semantic_calculator: SemanticSimilarityCalculator,
        validation_mode: SemanticValidationMode = SemanticValidationMode.ADVISORY,
        semantic_threshold: float = 0.7,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the semantic contextual integrator.
        
        Args:
            semantic_calculator: SemanticSimilarityCalculator instance
            validation_mode: Mode for semantic validation integration
            semantic_threshold: Minimum semantic similarity for validation
            config: Additional configuration options
        """
        self.logger = get_logger(__name__)
        self.semantic_calculator = semantic_calculator
        self.validation_mode = validation_mode
        self.semantic_threshold = semantic_threshold
        self.config = config or {}
        
        # Performance settings
        self.enable_caching = self.config.get('enable_caching', True)
        self.max_validation_pairs = self.config.get('max_validation_pairs', 50)
        
        # Statistics
        self.stats = {
            'validations_performed': 0,
            'semantic_validations_passed': 0,
            'semantic_validations_failed': 0,
            'cache_hits': 0,
            'processing_time_total': 0.0
        }
        
        self.logger.info(
            f"Initialized SemanticContextualIntegrator with mode: {validation_mode.value}, "
            f"threshold: {semantic_threshold}"
        )
    
    def enhance_contextual_rule_matching(
        self,
        rule_engine: ContextualRuleEngine,
        text: str,
        context_words: List[str],
        apply_semantic_validation: bool = True
    ) -> List[EnhancedContextualMatch]:
        """
        Enhance contextual rule matching with semantic validation (AC6).
        
        Args:
            rule_engine: ContextualRuleEngine instance from Story 2.2
            text: Text to apply rules to
            context_words: Context words for rule application
            apply_semantic_validation: Whether to apply semantic validation
            
        Returns:
            List of EnhancedContextualMatch objects with semantic validation
        """
        # Get original rule matches (backward compatibility)
        original_matches = rule_engine.apply_contextual_rules(text, context_words)
        
        enhanced_matches = []
        
        for match in original_matches:
            enhanced_match = EnhancedContextualMatch(
                original_text=match.original_text,
                corrected_text=match.corrected_text,
                rule_confidence=match.confidence_score,
                validation_mode=self.validation_mode.value,
                metadata={'rule_type': match.rule_type.value if hasattr(match, 'rule_type') else 'unknown'}
            )
            
            # Apply semantic validation if enabled
            if (apply_semantic_validation and 
                self.validation_mode != SemanticValidationMode.DISABLED and
                len(enhanced_matches) < self.max_validation_pairs):
                
                enhanced_match = self._apply_semantic_validation(
                    enhanced_match, context_words
                )
            else:
                # No semantic validation - use original confidence
                enhanced_match.combined_confidence = enhanced_match.rule_confidence
                enhanced_match.semantic_validation_passed = True
            
            enhanced_matches.append(enhanced_match)
        
        self.logger.debug(
            f"Enhanced {len(enhanced_matches)} contextual matches with semantic validation"
        )
        
        return enhanced_matches
    
    def enhance_ngram_predictions(
        self,
        ngram_model: NGramLanguageModel,
        context: List[str],
        top_k: int = 5,
        apply_semantic_validation: bool = True
    ) -> List[EnhancedContextPrediction]:
        """
        Enhance n-gram predictions with semantic coherence validation.
        
        Args:
            ngram_model: NGramLanguageModel instance from Story 2.2
            context: Context words for prediction
            top_k: Number of top predictions to return
            apply_semantic_validation: Whether to apply semantic validation
            
        Returns:
            List of EnhancedContextPrediction objects with semantic validation
        """
        # Get original n-gram predictions (backward compatibility)
        original_predictions = ngram_model.predict_next_words(context, top_k)
        
        enhanced_predictions = []
        
        # Create context string for semantic validation
        context_text = " ".join(context) if context else ""
        
        for prediction in original_predictions:
            enhanced_pred = EnhancedContextPrediction(
                word=prediction.word,
                ngram_probability=prediction.probability,
                validation_mode=self.validation_mode.value,
                supporting_context=context.copy()
            )
            
            # Apply semantic validation if enabled
            if (apply_semantic_validation and 
                self.validation_mode != SemanticValidationMode.DISABLED and
                context_text and
                len(enhanced_predictions) < self.max_validation_pairs):
                
                enhanced_pred = self._apply_semantic_coherence_validation(
                    enhanced_pred, context_text
                )
            else:
                # No semantic validation - use original probability
                enhanced_pred.combined_score = enhanced_pred.ngram_probability
                enhanced_pred.semantic_validation_passed = True
            
            enhanced_predictions.append(enhanced_pred)
        
        # Sort by combined score if semantic validation was applied
        if apply_semantic_validation and self.validation_mode != SemanticValidationMode.DISABLED:
            enhanced_predictions.sort(key=lambda x: x.combined_score, reverse=True)
        
        self.logger.debug(
            f"Enhanced {len(enhanced_predictions)} n-gram predictions with semantic validation"
        )
        
        return enhanced_predictions
    
    def _apply_semantic_validation(
        self,
        match: EnhancedContextualMatch,
        context_words: List[str]
    ) -> EnhancedContextualMatch:
        """Apply semantic validation to contextual match."""
        self.stats['validations_performed'] += 1
        
        try:
            # Create context string
            context_text = " ".join(context_words) if context_words else ""
            
            if not context_text:
                # No context available - skip semantic validation
                match.combined_confidence = match.rule_confidence
                match.semantic_validation_passed = True
                return match
            
            # Compute semantic similarity between original and corrected text in context
            original_in_context = f"{context_text} {match.original_text}"
            corrected_in_context = f"{context_text} {match.corrected_text}"
            
            similarity_result = self.semantic_calculator.compute_semantic_similarity(
                original_in_context, corrected_in_context
            )
            
            match.semantic_similarity = similarity_result.similarity_score
            
            # Apply validation logic based on mode
            if self.validation_mode == SemanticValidationMode.ADVISORY:
                # Advisory mode - provide semantic score but don't change decision
                match.combined_confidence = match.rule_confidence
                match.semantic_validation_passed = True
                
            elif self.validation_mode == SemanticValidationMode.HYBRID:
                # Hybrid mode - combine rule confidence and semantic similarity
                semantic_weight = 0.3  # Configurable
                rule_weight = 0.7
                
                match.combined_confidence = (
                    rule_weight * match.rule_confidence +
                    semantic_weight * match.semantic_similarity
                )
                match.semantic_validation_passed = match.semantic_similarity >= self.semantic_threshold
                
            elif self.validation_mode == SemanticValidationMode.SEMANTIC_FIRST:
                # Semantic-first mode - semantic similarity takes precedence
                if match.semantic_similarity >= self.semantic_threshold:
                    match.combined_confidence = max(match.rule_confidence, match.semantic_similarity)
                    match.semantic_validation_passed = True
                else:
                    match.combined_confidence = match.rule_confidence * 0.5  # Penalize low semantic similarity
                    match.semantic_validation_passed = False
            
            # Update statistics
            if match.semantic_validation_passed:
                self.stats['semantic_validations_passed'] += 1
            else:
                self.stats['semantic_validations_failed'] += 1
            
            # Add cache hit information
            if similarity_result.cache_hit:
                self.stats['cache_hits'] += 1
            
            self.stats['processing_time_total'] += similarity_result.computation_time
            
        except Exception as e:
            self.logger.error(f"Error in semantic validation: {e}")
            # Fallback to original confidence on error
            match.combined_confidence = match.rule_confidence
            match.semantic_validation_passed = True
            match.metadata = match.metadata or {}
            match.metadata['semantic_validation_error'] = str(e)
        
        return match
    
    def _apply_semantic_coherence_validation(
        self,
        prediction: EnhancedContextPrediction,
        context_text: str
    ) -> EnhancedContextPrediction:
        """Apply semantic coherence validation to n-gram prediction."""
        try:
            # Create text with prediction
            predicted_text = f"{context_text} {prediction.word}"
            
            # Compute semantic coherence (similarity to context)
            similarity_result = self.semantic_calculator.compute_semantic_similarity(
                context_text, predicted_text
            )
            
            prediction.semantic_coherence = similarity_result.similarity_score
            
            # Apply validation logic based on mode
            if self.validation_mode == SemanticValidationMode.ADVISORY:
                prediction.combined_score = prediction.ngram_probability
                prediction.semantic_validation_passed = True
                
            elif self.validation_mode == SemanticValidationMode.HYBRID:
                # Combine n-gram probability and semantic coherence
                semantic_weight = 0.4
                ngram_weight = 0.6
                
                prediction.combined_score = (
                    ngram_weight * prediction.ngram_probability +
                    semantic_weight * prediction.semantic_coherence
                )
                prediction.semantic_validation_passed = prediction.semantic_coherence >= self.semantic_threshold
                
            elif self.validation_mode == SemanticValidationMode.SEMANTIC_FIRST:
                # Prioritize semantic coherence
                if prediction.semantic_coherence >= self.semantic_threshold:
                    prediction.combined_score = max(prediction.ngram_probability, prediction.semantic_coherence)
                    prediction.semantic_validation_passed = True
                else:
                    prediction.combined_score = prediction.ngram_probability * 0.7
                    prediction.semantic_validation_passed = False
            
        except Exception as e:
            self.logger.error(f"Error in semantic coherence validation: {e}")
            prediction.combined_score = prediction.ngram_probability
            prediction.semantic_validation_passed = True
        
        return prediction
    
    def validate_contextual_consistency(
        self,
        text_segments: List[str],
        max_pairs: int = 20
    ) -> Dict[str, Any]:
        """
        Validate contextual consistency across multiple text segments using semantic analysis.
        
        Args:
            text_segments: List of text segments to validate for consistency
            max_pairs: Maximum number of segment pairs to validate
            
        Returns:
            Validation results with consistency scores
        """
        if len(text_segments) < 2:
            return {
                'consistency_score': 1.0,
                'segments_analyzed': len(text_segments),
                'pairwise_comparisons': 0,
                'average_similarity': 1.0,
                'consistency_level': 'HIGH'
            }
        
        # Generate pairs for comparison (limited by max_pairs)
        segment_pairs = []
        for i in range(len(text_segments)):
            for j in range(i + 1, min(len(text_segments), i + max_pairs // len(text_segments) + 1)):
                segment_pairs.append((text_segments[i], text_segments[j]))
                if len(segment_pairs) >= max_pairs:
                    break
            if len(segment_pairs) >= max_pairs:
                break
        
        # Compute semantic similarities
        similarities = []
        for seg1, seg2 in segment_pairs:
            result = self.semantic_calculator.compute_semantic_similarity(seg1, seg2)
            similarities.append(result.similarity_score)
        
        # Calculate consistency metrics
        average_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        min_similarity = min(similarities) if similarities else 0.0
        max_similarity = max(similarities) if similarities else 0.0
        
        # Determine consistency level
        if average_similarity >= 0.8:
            consistency_level = 'HIGH'
        elif average_similarity >= 0.6:
            consistency_level = 'MEDIUM'
        else:
            consistency_level = 'LOW'
        
        return {
            'consistency_score': average_similarity,
            'segments_analyzed': len(text_segments),
            'pairwise_comparisons': len(similarities),
            'average_similarity': average_similarity,
            'min_similarity': min_similarity,
            'max_similarity': max_similarity,
            'consistency_level': consistency_level,
            'similarities': similarities
        }
    
    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get integration performance statistics."""
        success_rate = (
            self.stats['semantic_validations_passed'] / 
            max(self.stats['validations_performed'], 1) * 100
        )
        
        avg_processing_time = (
            self.stats['processing_time_total'] / 
            max(self.stats['validations_performed'], 1)
        )
        
        cache_hit_rate = (
            self.stats['cache_hits'] / 
            max(self.stats['validations_performed'], 1) * 100
        )
        
        return {
            'validation_mode': self.validation_mode.value,
            'semantic_threshold': self.semantic_threshold,
            'total_validations': self.stats['validations_performed'],
            'validations_passed': self.stats['semantic_validations_passed'],
            'validations_failed': self.stats['semantic_validations_failed'],
            'success_rate': f"{success_rate:.1f}%",
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'average_processing_time': f"{avg_processing_time:.4f}s",
            'total_processing_time': f"{self.stats['processing_time_total']:.2f}s"
        }
    
    def set_validation_mode(self, mode: SemanticValidationMode) -> None:
        """Change the validation mode at runtime."""
        old_mode = self.validation_mode
        self.validation_mode = mode
        self.logger.info(f"Changed validation mode from {old_mode.value} to {mode.value}")
    
    def reset_statistics(self) -> None:
        """Reset integration statistics."""
        self.stats = {
            'validations_performed': 0,
            'semantic_validations_passed': 0,
            'semantic_validations_failed': 0,
            'cache_hits': 0,
            'processing_time_total': 0.0
        }
        self.logger.info("Reset integration statistics")