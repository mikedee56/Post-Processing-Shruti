"""
Unified Confidence Scoring System for Story 2.4.4

This module provides system-wide confidence scoring with 0.0-1.0 normalization
across all enhanced components in the ASR post-processing pipeline.

Key Features:
- Normalized confidence scoring (0.0-1.0 range)
- Weighted confidence combination across multiple components
- Confidence score provenance tracking
- Integration with existing Story 2.1-2.3 confidence systems
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import math
import logging
from pathlib import Path

from utils.logger_config import get_logger


class ConfidenceSource(Enum):
    """Sources of confidence scores in the system."""
    LEXICON_MATCH = "lexicon_match"
    SANDHI_PREPROCESSING = "sandhi_preprocessing"
    PHONETIC_HASHING = "phonetic_hashing"
    FUZZY_MATCHING = "fuzzy_matching"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    CONTEXTUAL_MODELING = "contextual_modeling"
    SCRIPTURE_IDENTIFICATION = "scripture_identification"
    HYBRID_MATCHING = "hybrid_matching"
    MANUAL_VALIDATION = "manual_validation"


@dataclass
class ConfidenceScore:
    """Individual confidence score with metadata."""
    value: float
    source: ConfidenceSource
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate confidence score on creation."""
        self.value = max(0.0, min(1.0, self.value))
        if self.weight < 0.0:
            self.weight = 0.0


@dataclass
class UnifiedConfidenceResult:
    """Result of unified confidence calculation."""
    final_confidence: float
    individual_scores: List[ConfidenceScore]
    composite_method: str
    confidence_provenance: Dict[str, Any]
    normalization_applied: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedConfidenceScorer:
    """
    System-wide confidence scoring with normalization and weighted combination.
    
    This component implements AC5 of Story 2.4.4, providing:
    - Normalized confidence scores (0.0-1.0) across all components
    - Weighted combination of multiple confidence sources
    - Confidence score provenance tracking
    - Integration with existing confidence systems
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the unified confidence scorer.
        
        Args:
            config: Configuration parameters for confidence scoring
        """
        self.logger = get_logger(__name__)
        self.config = config or {}
        
        # Default confidence weights for different sources
        self.default_weights = {
            ConfidenceSource.LEXICON_MATCH: 1.0,
            ConfidenceSource.SANDHI_PREPROCESSING: 0.8,
            ConfidenceSource.PHONETIC_HASHING: 0.7,
            ConfidenceSource.FUZZY_MATCHING: 0.6,
            ConfidenceSource.SEMANTIC_SIMILARITY: 0.9,
            ConfidenceSource.CONTEXTUAL_MODELING: 0.8,
            ConfidenceSource.SCRIPTURE_IDENTIFICATION: 1.0,
            ConfidenceSource.HYBRID_MATCHING: 1.0,
            ConfidenceSource.MANUAL_VALIDATION: 1.0,
        }
        
        # Update weights from config
        config_weights = self.config.get('confidence_weights', {})
        for source_str, weight in config_weights.items():
            try:
                source = ConfidenceSource(source_str)
                self.default_weights[source] = weight
            except ValueError:
                self.logger.warning(f"Unknown confidence source in config: {source_str}")
        
        # Scoring parameters
        self.min_scores_for_composite = self.config.get('min_scores_for_composite', 2)
        self.confidence_boost_threshold = self.config.get('confidence_boost_threshold', 0.8)
        self.agreement_boost_factor = self.config.get('agreement_boost_factor', 0.1)
        
        self.logger.info("Unified confidence scorer initialized")
    
    def normalize_confidence_score(
        self, 
        raw_score: Union[float, int], 
        source: ConfidenceSource,
        original_range: Optional[tuple] = None
    ) -> float:
        """
        Normalize confidence score to 0.0-1.0 range.
        
        Args:
            raw_score: Raw confidence score
            source: Source of the confidence score
            original_range: Original range (min, max) for normalization
            
        Returns:
            Normalized confidence score (0.0-1.0)
        """
        if original_range:
            min_val, max_val = original_range
            if max_val == min_val:
                return 1.0 if raw_score >= max_val else 0.0
            normalized = (raw_score - min_val) / (max_val - min_val)
        else:
            # Assume already in 0.0-1.0 range but clamp to be safe
            normalized = float(raw_score)
        
        # Apply source-specific normalization adjustments
        if source == ConfidenceSource.FUZZY_MATCHING:
            # Fuzzy matching often gives high scores, apply slight dampening
            normalized = normalized * 0.9
        elif source == ConfidenceSource.PHONETIC_HASHING:
            # Phonetic hashing can be noisy, apply confidence adjustment
            normalized = normalized * 0.8 if normalized < 0.5 else normalized
        elif source == ConfidenceSource.SEMANTIC_SIMILARITY:
            # Semantic similarity is often well-calibrated, minimal adjustment
            pass
        
        return max(0.0, min(1.0, normalized))
    
    def create_confidence_score(
        self,
        value: Union[float, int],
        source: ConfidenceSource,
        weight: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        original_range: Optional[tuple] = None
    ) -> ConfidenceScore:
        """
        Create a normalized confidence score.
        
        Args:
            value: Raw confidence value
            source: Source of the confidence
            weight: Custom weight (uses default if None)
            metadata: Additional metadata
            original_range: Original range for normalization
            
        Returns:
            ConfidenceScore object with normalized value
        """
        normalized_value = self.normalize_confidence_score(value, source, original_range)
        
        return ConfidenceScore(
            value=normalized_value,
            source=source,
            weight=weight or self.default_weights.get(source, 1.0),
            metadata=metadata or {}
        )
    
    def compute_weighted_average(self, scores: List[ConfidenceScore]) -> float:
        """
        Compute weighted average of confidence scores.
        
        Args:
            scores: List of confidence scores
            
        Returns:
            Weighted average confidence
        """
        if not scores:
            return 0.0
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for score in scores:
            total_weighted_score += score.value * score.weight
            total_weight += score.weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def compute_harmonic_mean(self, scores: List[ConfidenceScore]) -> float:
        """
        Compute weighted harmonic mean (conservative approach).
        
        Args:
            scores: List of confidence scores
            
        Returns:
            Harmonic mean confidence
        """
        if not scores:
            return 0.0
        
        # Filter out zero scores to avoid division by zero
        non_zero_scores = [s for s in scores if s.value > 0]
        if not non_zero_scores:
            return 0.0
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for score in non_zero_scores:
            weighted_sum += score.weight / score.value
            total_weight += score.weight
        
        return total_weight / weighted_sum if weighted_sum > 0 else 0.0
    
    def compute_geometric_mean(self, scores: List[ConfidenceScore]) -> float:
        """
        Compute weighted geometric mean (moderate approach).
        
        Args:
            scores: List of confidence scores
            
        Returns:
            Geometric mean confidence
        """
        if not scores:
            return 0.0
        
        # Filter out zero scores
        non_zero_scores = [s for s in scores if s.value > 0]
        if not non_zero_scores:
            return 0.0
        
        log_sum = 0.0
        total_weight = 0.0
        
        for score in non_zero_scores:
            log_sum += score.weight * math.log(score.value)
            total_weight += score.weight
        
        return math.exp(log_sum / total_weight) if total_weight > 0 else 0.0
    
    def detect_score_agreement(self, scores: List[ConfidenceScore]) -> Dict[str, Any]:
        """
        Detect agreement between different confidence sources.
        
        Args:
            scores: List of confidence scores
            
        Returns:
            Agreement analysis metadata
        """
        if len(scores) < 2:
            return {"agreement_level": "insufficient_data", "variance": 0.0}
        
        values = [s.value for s in scores]
        mean_value = sum(values) / len(values)
        variance = sum((v - mean_value) ** 2 for v in values) / len(values)
        std_dev = math.sqrt(variance)
        
        # Determine agreement level
        if std_dev < 0.1:
            agreement_level = "high"
        elif std_dev < 0.2:
            agreement_level = "moderate"
        else:
            agreement_level = "low"
        
        return {
            "agreement_level": agreement_level,
            "variance": variance,
            "standard_deviation": std_dev,
            "mean_confidence": mean_value,
            "score_count": len(scores)
        }
    
    def apply_agreement_boost(
        self, 
        base_confidence: float, 
        agreement_analysis: Dict[str, Any]
    ) -> float:
        """
        Apply confidence boost for high agreement between sources.
        
        Args:
            base_confidence: Base confidence score
            agreement_analysis: Agreement analysis from detect_score_agreement
            
        Returns:
            Boosted confidence score
        """
        agreement_level = agreement_analysis.get("agreement_level", "low")
        
        if (agreement_level == "high" and 
            base_confidence >= self.confidence_boost_threshold):
            
            boost = min(self.agreement_boost_factor, 1.0 - base_confidence)
            boosted_confidence = base_confidence + boost
            
            self.logger.debug(
                f"Applied agreement boost: {base_confidence:.3f} -> {boosted_confidence:.3f}"
            )
            return boosted_confidence
        
        return base_confidence
    
    def combine_confidence_scores(
        self, 
        scores: List[ConfidenceScore],
        method: str = "weighted_average"
    ) -> UnifiedConfidenceResult:
        """
        Combine multiple confidence scores into a unified score.
        
        Args:
            scores: List of confidence scores to combine
            method: Combination method ("weighted_average", "harmonic_mean", "geometric_mean", "adaptive")
            
        Returns:
            UnifiedConfidenceResult with combined score and metadata
        """
        if not scores:
            return UnifiedConfidenceResult(
                final_confidence=0.0,
                individual_scores=[],
                composite_method="empty_input",
                confidence_provenance={},
                normalization_applied=False
            )
        
        # Normalize all scores if needed
        normalized_scores = []
        normalization_applied = False
        
        for score in scores:
            if 0.0 <= score.value <= 1.0:
                normalized_scores.append(score)
            else:
                # Re-normalize if score is out of range
                normalized_value = max(0.0, min(1.0, score.value))
                normalized_score = ConfidenceScore(
                    value=normalized_value,
                    source=score.source,
                    weight=score.weight,
                    metadata=score.metadata
                )
                normalized_scores.append(normalized_score)
                normalization_applied = True
        
        # Compute base confidence using specified method
        if method == "weighted_average":
            base_confidence = self.compute_weighted_average(normalized_scores)
        elif method == "harmonic_mean":
            base_confidence = self.compute_harmonic_mean(normalized_scores)
        elif method == "geometric_mean":
            base_confidence = self.compute_geometric_mean(normalized_scores)
        elif method == "adaptive":
            # Use different methods based on score count and agreement
            agreement_analysis = self.detect_score_agreement(normalized_scores)
            if len(normalized_scores) >= 3 and agreement_analysis["agreement_level"] == "high":
                base_confidence = self.compute_weighted_average(normalized_scores)
            elif agreement_analysis["agreement_level"] == "low":
                base_confidence = self.compute_harmonic_mean(normalized_scores)  # Conservative
            else:
                base_confidence = self.compute_geometric_mean(normalized_scores)
        else:
            self.logger.warning(f"Unknown combination method: {method}, using weighted_average")
            base_confidence = self.compute_weighted_average(normalized_scores)
        
        # Analyze agreement and apply boost if applicable
        agreement_analysis = self.detect_score_agreement(normalized_scores)
        final_confidence = self.apply_agreement_boost(base_confidence, agreement_analysis)
        
        # Create provenance information
        confidence_provenance = {
            "source_breakdown": {
                score.source.value: {
                    "confidence": score.value,
                    "weight": score.weight,
                    "metadata": score.metadata
                }
                for score in normalized_scores
            },
            "agreement_analysis": agreement_analysis,
            "combination_method": method,
            "agreement_boost_applied": final_confidence > base_confidence
        }
        
        return UnifiedConfidenceResult(
            final_confidence=final_confidence,
            individual_scores=normalized_scores,
            composite_method=method,
            confidence_provenance=confidence_provenance,
            normalization_applied=normalization_applied,
            metadata={
                "score_count": len(normalized_scores),
                "base_confidence": base_confidence,
                "final_confidence": final_confidence
            }
        )
    
    def get_confidence_statistics(self) -> Dict[str, Any]:
        """Get statistics about confidence scoring operations."""
        return {
            "scorer_info": {
                "version": "2.4.4",
                "configured_sources": len(self.default_weights),
                "min_scores_for_composite": self.min_scores_for_composite,
                "confidence_boost_threshold": self.confidence_boost_threshold,
                "agreement_boost_factor": self.agreement_boost_factor
            },
            "source_weights": {
                source.value: weight 
                for source, weight in self.default_weights.items()
            }
        }
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate confidence scorer configuration."""
        validation = {
            "is_valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Check weights
        for source, weight in self.default_weights.items():
            if weight < 0:
                validation["errors"].append(f"Negative weight for {source.value}: {weight}")
                validation["is_valid"] = False
            elif weight == 0:
                validation["warnings"].append(f"Zero weight for {source.value} - will be ignored")
        
        # Check parameters
        if self.confidence_boost_threshold < 0 or self.confidence_boost_threshold > 1:
            validation["errors"].append(f"Invalid boost threshold: {self.confidence_boost_threshold}")
            validation["is_valid"] = False
        
        if self.agreement_boost_factor < 0 or self.agreement_boost_factor > 1:
            validation["errors"].append(f"Invalid agreement boost factor: {self.agreement_boost_factor}")
            validation["is_valid"] = False
        
        return validation