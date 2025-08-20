"""
Hybrid Matching Engine for Story 2.4.3

This module implements a comprehensive 3-stage matching pipeline for scripture verse identification:
1. Stage 1: Sanskrit Phonetic Hashing for fast candidate filtering
2. Stage 2: Smith-Waterman sequence alignment for precision matching of noisy ASR text  
3. Stage 3: Semantic similarity using iNLTK embeddings

Integrates seamlessly with existing Story 2.3 ScriptureProcessor while providing
research-grade accuracy for matching ASR transcripts to canonical scriptural verses.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging
import time

from utils.logger_config import get_logger
from utils.sanskrit_phonetic_hasher import SanskritPhoneticHasher, PhoneticCandidateMatch
from utils.sequence_alignment_engine import SequenceAlignmentEngine, AlignmentResult
from scripture_processing.canonical_text_manager import CanonicalTextManager, VerseCandidate, ScriptureSource
from scripture_processing.scripture_identifier import ScriptureMatch
from contextual_modeling.semantic_similarity_calculator import SemanticSimilarityCalculator


class SourceProvenance(Enum):
    """Source provenance classification for scripture verses."""
    GOLD = "gold"      # Authoritative sources (traditional commentaries, verified editions)
    SILVER = "silver"  # Well-regarded sources (academic editions, scholarly works)
    BRONZE = "bronze"  # General sources (popular editions, digital collections)


class MatchingStage(Enum):
    """Stages of the hybrid matching pipeline."""
    PHONETIC_HASH = "phonetic_hash"
    SEQUENCE_ALIGNMENT = "sequence_alignment" 
    SEMANTIC_SIMILARITY = "semantic_similarity"


@dataclass
class PhoneticMatchResult:
    """Result from Stage 1: Phonetic hashing."""
    candidate_verse_id: str
    phonetic_score: float
    phonetic_hash_original: str
    phonetic_hash_candidate: str
    hash_distance: int


@dataclass
class SequenceAlignmentResult:
    """Result from Stage 2: Smith-Waterman alignment."""
    candidate_verse_id: str
    alignment_score: float
    normalized_score: float
    alignment_length: int
    identity_percentage: float
    gaps: int
    alignment_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticMatchResult:
    """Result from Stage 3: Semantic similarity."""
    candidate_verse_id: str
    semantic_score: float
    embedding_model_used: str
    computation_time: float
    cache_hit: bool
    language_detected: str


@dataclass
class HybridMatchingResult:
    """
    Comprehensive result from the 3-stage hybrid matching pipeline.
    
    This is the primary output that integrates with existing Story 2.3 systems
    while providing enhanced matching capabilities.
    """
    # Input information
    original_passage: str
    matched_verse: Optional[VerseCandidate] = None
    
    # Stage results
    phonetic_result: Optional[PhoneticMatchResult] = None
    sequence_result: Optional[SequenceAlignmentResult] = None
    semantic_result: Optional[SemanticMatchResult] = None
    
    # Composite scoring
    composite_confidence: float = 0.0
    weighted_scores: Dict[str, float] = field(default_factory=dict)
    
    # Source and metadata
    source_provenance: SourceProvenance = SourceProvenance.BRONZE
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Integration with Story 2.3
    traditional_match: Optional[ScriptureMatch] = None
    enhanced_match: bool = False
    
    # Pipeline performance
    total_processing_time: float = 0.0
    stages_completed: List[MatchingStage] = field(default_factory=list)
    pipeline_success: bool = False
    fallback_used: bool = False


@dataclass
class HybridPipelineConfig:
    """Configuration for the hybrid matching pipeline."""
    
    # Stage weights for composite scoring
    phonetic_weight: float = 0.3
    sequence_weight: float = 0.4
    semantic_weight: float = 0.3
    
    # Minimum thresholds for each stage
    phonetic_min_score: float = 0.6
    sequence_min_score: float = 0.5
    semantic_min_score: float = 0.4
    
    # Source provenance weights
    gold_source_multiplier: float = 1.0
    silver_source_multiplier: float = 0.9
    bronze_source_multiplier: float = 0.8
    
    # Pipeline behavior
    enable_phonetic_stage: bool = True
    enable_sequence_stage: bool = True
    enable_semantic_stage: bool = True
    
    # Performance optimization
    max_candidates_per_stage: int = 50
    phonetic_candidates_limit: int = 100
    sequence_candidates_limit: int = 20
    
    # Graceful fallback
    enable_fallback_to_traditional: bool = True
    fallback_threshold: float = 0.3


class HybridMatchingEngine:
    """
    Core 3-stage hybrid matching engine for scripture verse identification.
    
    This engine implements the research-grade matching pipeline while maintaining
    full backward compatibility with existing Story 2.3 functionality.
    
    Architecture Integration:
    - Enhances existing ScriptureProcessor without breaking changes
    - Uses file-based approach consistent with current YAML architecture  
    - Integrates with Story 2.4.2 semantic components
    - Provides graceful fallback to Story 2.3 traditional matching
    """
    
    def __init__(
        self, 
        canonical_manager: CanonicalTextManager,
        semantic_calculator: Optional[SemanticSimilarityCalculator] = None,
        config: Optional[HybridPipelineConfig] = None,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize the Hybrid Matching Engine.
        
        Args:
            canonical_manager: Existing canonical text manager from Story 2.3
            semantic_calculator: Semantic similarity calculator from Story 2.4.2
            config: Pipeline configuration
            cache_dir: Directory for caching intermediate results
        """
        self.logger = get_logger(__name__)
        self.canonical_manager = canonical_manager
        self.config = config or HybridPipelineConfig()
        
        # Initialize semantic component
        if semantic_calculator is None:
            self.semantic_calculator = SemanticSimilarityCalculator(cache_dir=cache_dir)
        else:
            self.semantic_calculator = semantic_calculator
        
        # Initialize stage components
        self.phonetic_hasher = SanskritPhoneticHasher(
            config=self.config.__dict__ if hasattr(self.config, '__dict__') else {},
            cache_dir=cache_dir
        )
        self.sequence_aligner = SequenceAlignmentEngine()
        
        # Performance tracking
        self.stats = {
            'total_matches_processed': 0,
            'pipeline_successes': 0,
            'fallback_usage': 0,
            'stage_performance': {
                stage.value: {'count': 0, 'total_time': 0.0, 'avg_time': 0.0}
                for stage in MatchingStage
            }
        }
        
        # Validate configuration
        self._validate_configuration()
        
        self.logger.info("HybridMatchingEngine initialized with 3-stage pipeline")
    
    def _validate_configuration(self) -> None:
        """Validate pipeline configuration."""
        # Check weight normalization
        total_weight = (
            self.config.phonetic_weight + 
            self.config.sequence_weight + 
            self.config.semantic_weight
        )
        
        if abs(total_weight - 1.0) > 0.01:
            self.logger.warning(
                f"Stage weights do not sum to 1.0 (sum={total_weight:.3f}). "
                "Normalizing weights..."
            )
            # Normalize weights
            self.config.phonetic_weight /= total_weight
            self.config.sequence_weight /= total_weight
            self.config.semantic_weight /= total_weight
    
    def match_verse_passage(
        self, 
        passage: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> HybridMatchingResult:
        """
        Main entry point for hybrid verse matching using 3-stage pipeline.
        
        This method coordinates all three stages and produces a comprehensive
        matching result that can be used by existing Story 2.3 components.
        
        Args:
            passage: ASR transcript passage to match against scripture
            context: Additional context for matching
            
        Returns:
            HybridMatchingResult with complete pipeline analysis
        """
        start_time = time.time()
        self.stats['total_matches_processed'] += 1
        
        context = context or {}
        
        # Initialize result
        result = HybridMatchingResult(
            original_passage=passage,
            processing_metadata={
                'input_length': len(passage),
                'context_provided': bool(context),
                'timestamp': time.time(),
                'pipeline_version': '2.4.3'
            }
        )
        
        try:
            # Get initial candidate pool from canonical manager
            initial_candidates = self._get_initial_candidates(passage, context)
            
            if not initial_candidates:
                self.logger.info(f"No initial candidates found for passage: {passage[:50]}...")
                result.processing_metadata['no_candidates_reason'] = 'empty_candidate_pool'
                return self._apply_fallback_matching(result, passage, context)
            
            # Stage 1: Phonetic Hashing
            if self.config.enable_phonetic_stage:
                result = self._execute_phonetic_stage(result, passage, initial_candidates)
            
            # Stage 2: Sequence Alignment  
            if self.config.enable_sequence_stage:
                result = self._execute_sequence_stage(result, passage)
            
            # Stage 3: Semantic Similarity
            if self.config.enable_semantic_stage:
                result = self._execute_semantic_stage(result, passage)
            
            # Composite scoring
            result = self._calculate_composite_confidence(result)
            
            # Source provenance weighting
            result = self._apply_source_provenance_weighting(result)
            
            # Final validation
            result.pipeline_success = self._validate_pipeline_result(result)
            
            if result.pipeline_success:
                self.stats['pipeline_successes'] += 1
            else:
                result = self._apply_fallback_matching(result, passage, context)
            
        except Exception as e:
            self.logger.error(f"Error in hybrid matching pipeline: {e}")
            result.processing_metadata['error'] = str(e)
            result = self._apply_fallback_matching(result, passage, context)
        
        # Record performance
        result.total_processing_time = time.time() - start_time
        self._update_performance_stats(result)
        
        return result
    
    def _get_initial_candidates(
        self, 
        passage: str, 
        context: Dict[str, Any]
    ) -> List[VerseCandidate]:
        """
        Get initial candidate verses using existing Story 2.3 functionality.
        
        Args:
            passage: Input passage
            context: Matching context
            
        Returns:
            List of potential verse candidates
        """
        try:
            # Use existing canonical manager to get candidates
            candidates = self.canonical_manager.get_verse_candidates(
                passage, 
                max_candidates=self.config.phonetic_candidates_limit
            )
            
            self.logger.debug(f"Retrieved {len(candidates)} initial candidates")
            return candidates
            
        except Exception as e:
            self.logger.error(f"Error retrieving initial candidates: {e}")
            return []
    
    def _execute_phonetic_stage(
        self, 
        result: HybridMatchingResult, 
        passage: str,
        candidates: List[VerseCandidate]
    ) -> HybridMatchingResult:
        """
        Execute Stage 1: Phonetic hashing for fast candidate filtering.
        
        Args:
            result: Current matching result
            passage: Input passage
            candidates: List of candidate verses
            
        Returns:
            Updated result with phonetic matching data
        """
        stage_start = time.time()
        
        try:
            # Build hash index if needed
            if not self.phonetic_hasher.hash_index:
                self.logger.info("Building phonetic hash index from candidates...")
                index_stats = self.phonetic_hasher.build_hash_index(candidates)
                self.logger.info(f"Hash index built: {index_stats['verses_indexed']} verses, {index_stats['unique_hashes']} unique hashes")
            
            # Get phonetic candidates
            phonetic_matches = self.phonetic_hasher.get_phonetic_candidates(
                passage, 
                max_candidates=self.config.phonetic_candidates_limit,
                max_distance=2
            )
            
            if phonetic_matches:
                # Take the best phonetic match
                best_match = phonetic_matches[0]
                
                result.phonetic_result = PhoneticMatchResult(
                    candidate_verse_id=f"{best_match.verse_candidate.source.value}_{best_match.verse_candidate.chapter}_{best_match.verse_candidate.verse}",
                    phonetic_score=best_match.phonetic_score,
                    phonetic_hash_original=best_match.original_hash,
                    phonetic_hash_candidate=best_match.candidate_hash,
                    hash_distance=best_match.hash_distance
                )
                
                result.matched_verse = best_match.verse_candidate
                
                self.logger.debug(
                    f"Phonetic match: {best_match.phonetic_score:.3f} score, "
                    f"distance: {best_match.hash_distance}"
                )
            
            result.stages_completed.append(MatchingStage.PHONETIC_HASH)
            
        except Exception as e:
            self.logger.error(f"Error in phonetic stage: {e}")
            result.processing_metadata['phonetic_stage_error'] = str(e)
        
        # Update performance tracking
        stage_time = time.time() - stage_start
        self.stats['stage_performance'][MatchingStage.PHONETIC_HASH.value]['count'] += 1
        self.stats['stage_performance'][MatchingStage.PHONETIC_HASH.value]['total_time'] += stage_time
        
        return result
    
    def _execute_sequence_stage(
        self, 
        result: HybridMatchingResult, 
        passage: str
    ) -> HybridMatchingResult:
        """
        Execute Stage 2: Smith-Waterman sequence alignment.
        
        Args:
            result: Current matching result
            passage: Input passage
            
        Returns:
            Updated result with sequence alignment data
        """
        stage_start = time.time()
        
        try:
            if result.matched_verse is None:
                self.logger.debug("No matched verse from phonetic stage for sequence alignment")
                return result
            
            # Perform sequence alignment between passage and matched verse
            alignment_result = self.sequence_aligner.calculate_sequence_alignment(
                passage, 
                result.matched_verse.canonical_text,
                local=True  # Use local alignment (Smith-Waterman)
            )
            
            # Store sequence alignment result
            result.sequence_result = SequenceAlignmentResult(
                candidate_verse_id=f"{result.matched_verse.source.value}_{result.matched_verse.chapter}_{result.matched_verse.verse}",
                alignment_score=alignment_result.alignment_score,
                normalized_score=alignment_result.normalized_score,
                alignment_length=alignment_result.alignment_length,
                identity_percentage=alignment_result.identity_percentage,
                gaps=alignment_result.gaps,
                alignment_metadata={
                    'query_start': alignment_result.query_start,
                    'query_end': alignment_result.query_end,
                    'target_start': alignment_result.target_start,
                    'target_end': alignment_result.target_end,
                    'aligned_query': alignment_result.aligned_query,
                    'aligned_target': alignment_result.aligned_target,
                    'alignment_string': alignment_result.alignment_string
                }
            )
            
            self.logger.debug(
                f"Sequence alignment: {alignment_result.normalized_score:.3f} score, "
                f"{alignment_result.identity_percentage:.1f}% identity"
            )
            
            result.stages_completed.append(MatchingStage.SEQUENCE_ALIGNMENT)
            
        except Exception as e:
            self.logger.error(f"Error in sequence alignment stage: {e}")
            result.processing_metadata['sequence_stage_error'] = str(e)
        
        # Update performance tracking  
        stage_time = time.time() - stage_start
        self.stats['stage_performance'][MatchingStage.SEQUENCE_ALIGNMENT.value]['count'] += 1
        self.stats['stage_performance'][MatchingStage.SEQUENCE_ALIGNMENT.value]['total_time'] += stage_time
        
        return result
    
    def _execute_semantic_stage(
        self, 
        result: HybridMatchingResult, 
        passage: str
    ) -> HybridMatchingResult:
        """
        Execute Stage 3: Semantic similarity computation.
        
        Args:
            result: Current matching result
            passage: Input passage
            
        Returns:
            Updated result with semantic similarity data
        """
        stage_start = time.time()
        
        try:
            # Use semantic calculator from Story 2.4.2
            if result.matched_verse is None:
                self.logger.debug("No matched verse for semantic comparison")
                return result
            
            # Compute semantic similarity
            semantic_result = self.semantic_calculator.compute_semantic_similarity(
                passage, 
                result.matched_verse.canonical_text
            )
            
            # Store semantic match result
            result.semantic_result = SemanticMatchResult(
                candidate_verse_id=f"{result.matched_verse.source.value}_{result.matched_verse.chapter}_{result.matched_verse.verse}",
                semantic_score=semantic_result.similarity_score,
                embedding_model_used=semantic_result.embedding_model,
                computation_time=semantic_result.computation_time,
                cache_hit=semantic_result.cache_hit,
                language_detected=semantic_result.language_used
            )
            
            result.stages_completed.append(MatchingStage.SEMANTIC_SIMILARITY)
            
            self.logger.debug(
                f"Semantic similarity: {semantic_result.similarity_score:.3f} "
                f"using {semantic_result.embedding_model}"
            )
            
        except Exception as e:
            self.logger.error(f"Error in semantic stage: {e}")
            result.processing_metadata['semantic_stage_error'] = str(e)
        
        # Update performance tracking
        stage_time = time.time() - stage_start
        self.stats['stage_performance'][MatchingStage.SEMANTIC_SIMILARITY.value]['count'] += 1
        self.stats['stage_performance'][MatchingStage.SEMANTIC_SIMILARITY.value]['total_time'] += stage_time
        
        return result
    
    def _calculate_composite_confidence(self, result: HybridMatchingResult) -> HybridMatchingResult:
        """
        Calculate weighted composite confidence from all completed stages.
        
        Args:
            result: Matching result with stage data
            
        Returns:
            Updated result with composite confidence
        """
        scores = {}
        total_weight = 0.0
        
        # Collect stage scores
        if result.phonetic_result:
            scores['phonetic'] = result.phonetic_result.phonetic_score
            total_weight += self.config.phonetic_weight
        
        if result.sequence_result:
            scores['sequence'] = result.sequence_result.normalized_score  
            total_weight += self.config.sequence_weight
        
        if result.semantic_result:
            scores['semantic'] = result.semantic_result.semantic_score
            total_weight += self.config.semantic_weight
        
        # Calculate weighted average
        if total_weight > 0:
            weighted_sum = 0.0
            
            if 'phonetic' in scores:
                weighted_sum += scores['phonetic'] * self.config.phonetic_weight
            if 'sequence' in scores:
                weighted_sum += scores['sequence'] * self.config.sequence_weight
            if 'semantic' in scores:
                weighted_sum += scores['semantic'] * self.config.semantic_weight
            
            result.composite_confidence = weighted_sum / total_weight
        else:
            result.composite_confidence = 0.0
        
        result.weighted_scores = scores
        
        return result
    
    def _apply_source_provenance_weighting(self, result: HybridMatchingResult) -> HybridMatchingResult:
        """
        Apply source provenance weighting to composite confidence.
        
        Args:
            result: Matching result
            
        Returns:
            Updated result with provenance-weighted confidence  
        """
        if result.matched_verse is None:
            return result
        
        # Determine source provenance from verse metadata
        provenance_str = "silver"  # default
        
        # Check if verse has provenance metadata
        if hasattr(result.matched_verse, 'metadata') and result.matched_verse.metadata:
            provenance_str = result.matched_verse.metadata.get('source_provenance', 'silver')
        
        # Convert to enum
        try:
            result.source_provenance = SourceProvenance(provenance_str.lower())
        except ValueError:
            result.source_provenance = SourceProvenance.SILVER  # fallback
        
        # Apply provenance multiplier
        multiplier_map = {
            SourceProvenance.GOLD: self.config.gold_source_multiplier,
            SourceProvenance.SILVER: self.config.silver_source_multiplier,
            SourceProvenance.BRONZE: self.config.bronze_source_multiplier
        }
        
        multiplier = multiplier_map[result.source_provenance]
        original_confidence = result.composite_confidence
        result.composite_confidence *= multiplier
        
        # Ensure confidence stays in valid range
        result.composite_confidence = min(1.0, result.composite_confidence)
        
        # Record provenance weighting in metadata
        result.processing_metadata['source_provenance'] = {
            'classification': result.source_provenance.value,
            'multiplier_applied': multiplier,
            'confidence_before': original_confidence,
            'confidence_after': result.composite_confidence
        }
        
        return result
    
    def _validate_pipeline_result(self, result: HybridMatchingResult) -> bool:
        """
        Validate that pipeline result meets quality thresholds.
        
        Args:
            result: Matching result to validate
            
        Returns:
            True if result meets quality standards
        """
        # Check composite confidence threshold
        if result.composite_confidence < self.config.fallback_threshold:
            return False
        
        # Check individual stage thresholds (if stages were completed)
        if (result.phonetic_result and 
            result.phonetic_result.phonetic_score < self.config.phonetic_min_score):
            return False
        
        if (result.sequence_result and 
            result.sequence_result.normalized_score < self.config.sequence_min_score):
            return False
        
        if (result.semantic_result and 
            result.semantic_result.semantic_score < self.config.semantic_min_score):
            return False
        
        # Require at least one stage to be completed
        if not result.stages_completed:
            return False
        
        return True
    
    def _apply_fallback_matching(
        self, 
        result: HybridMatchingResult, 
        passage: str,
        context: Dict[str, Any]
    ) -> HybridMatchingResult:
        """
        Apply graceful fallback to traditional Story 2.3 matching.
        
        Args:
            result: Current result (may be partially complete)
            passage: Original passage
            context: Matching context
            
        Returns:
            Result with fallback matching applied
        """
        if not self.config.enable_fallback_to_traditional:
            return result
        
        self.stats['fallback_usage'] += 1
        result.fallback_used = True
        
        try:
            # Use existing canonical manager for basic matching
            candidates = self.canonical_manager.get_verse_candidates(passage, max_candidates=1)
            
            if candidates:
                result.matched_verse = candidates[0]
                # Use a basic confidence score for fallback
                result.composite_confidence = 0.5  # Moderate confidence for fallback
                result.pipeline_success = True
                
                self.logger.info(f"Applied fallback matching for passage: {passage[:50]}...")
            
        except Exception as e:
            self.logger.error(f"Error in fallback matching: {e}")
            result.processing_metadata['fallback_error'] = str(e)
        
        return result
    
    def _update_performance_stats(self, result: HybridMatchingResult) -> None:
        """Update performance statistics from completed result."""
        # Update average computation times for completed stages
        for stage in result.stages_completed:
            stage_stats = self.stats['stage_performance'][stage.value]
            if stage_stats['count'] > 0:
                stage_stats['avg_time'] = stage_stats['total_time'] / stage_stats['count']
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance statistics for monitoring.
        
        Returns:
            Performance statistics dictionary
        """
        total_processed = max(self.stats['total_matches_processed'], 1)
        
        return {
            'pipeline_overview': {
                'total_matches_processed': total_processed,
                'pipeline_success_rate': f"{(self.stats['pipeline_successes'] / total_processed) * 100:.1f}%",
                'fallback_usage_rate': f"{(self.stats['fallback_usage'] / total_processed) * 100:.1f}%"
            },
            'stage_performance': {
                stage: {
                    'executions': stats['count'],
                    'average_time': f"{stats['avg_time']:.4f}s",
                    'total_time': f"{stats['total_time']:.2f}s"
                }
                for stage, stats in self.stats['stage_performance'].items()
            },
            'configuration': {
                'phonetic_enabled': self.config.enable_phonetic_stage,
                'sequence_enabled': self.config.enable_sequence_stage,
                'semantic_enabled': self.config.enable_semantic_stage,
                'fallback_enabled': self.config.enable_fallback_to_traditional,
                'stage_weights': {
                    'phonetic': self.config.phonetic_weight,
                    'sequence': self.config.sequence_weight,
                    'semantic': self.config.semantic_weight
                }
            }
        }
    
    def validate_system_integration(self) -> Dict[str, Any]:
        """
        Validate integration with existing Story 2.3 systems.
        
        Returns:
            Integration validation results
        """
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'component_status': {}
        }
        
        # Check canonical manager integration
        try:
            test_candidates = self.canonical_manager.get_verse_candidates("test", max_candidates=1)
            validation['component_status']['canonical_manager'] = 'operational'
        except Exception as e:
            validation['errors'].append(f"Canonical manager integration failed: {e}")
            validation['is_valid'] = False
            validation['component_status']['canonical_manager'] = 'failed'
        
        # Check semantic calculator integration
        try:
            test_result = self.semantic_calculator.compute_semantic_similarity("test1", "test2")
            validation['component_status']['semantic_calculator'] = 'operational'
        except Exception as e:
            validation['warnings'].append(f"Semantic calculator issues: {e}")
            validation['component_status']['semantic_calculator'] = 'degraded'
        
        return validation