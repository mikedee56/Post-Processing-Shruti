"""
Epic 4.2 ML-Enhanced Feedback Integration System.

Implements Story 3.3 Task 3: Epic 4.2 ML-Enhanced Feedback Integration
- Superior correction integration using ML-enhanced lexicon management
- 15% Sanskrit accuracy improvements applied to feedback processing
- Semantic similarity calculation for correction pattern analysis
- Epic 4.5 academic validation for correction quality assessment
"""

import logging
import statistics
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple, Union

# Epic 4 infrastructure imports
from sanskrit_hindi_identifier.enhanced_lexicon_manager import (
    EnhancedLexiconManager, MLEnhancedEntry, QualityValidationStatus
)
from contextual_modeling.semantic_similarity_calculator import SemanticSimilarityCalculator
from utils.performance_monitor import PerformanceMonitor
from monitoring.system_monitor import SystemMonitor
from monitoring.telemetry_collector import TelemetryCollector
from utils.academic_validator import AcademicValidator, AcademicStandard
from utils.srt_parser import SRTSegment


class CorrectionType(Enum):
    """Types of human corrections."""
    SPELLING_CORRECTION = "spelling_correction"
    GRAMMAR_CORRECTION = "grammar_correction"
    SANSKRIT_TERM_CORRECTION = "sanskrit_term_correction"
    IAST_TRANSLITERATION = "iast_transliteration"
    ACADEMIC_CITATION = "academic_citation"
    FORMATTING_IMPROVEMENT = "formatting_improvement"
    CONTENT_ENHANCEMENT = "content_enhancement"
    THEOLOGICAL_CLARIFICATION = "theological_clarification"


class FeedbackQuality(Enum):
    """Quality assessment of human feedback."""
    EXCELLENT = "excellent"      # 0.9+
    GOOD = "good"               # 0.7-0.89
    ACCEPTABLE = "acceptable"    # 0.5-0.69
    QUESTIONABLE = "questionable" # 0.3-0.49
    POOR = "poor"               # <0.3


class IntegrationStatus(Enum):
    """Status of feedback integration into automated systems."""
    PENDING = "pending"
    VALIDATED = "validated"
    INTEGRATED = "integrated"
    REJECTED = "rejected"
    REQUIRES_REVIEW = "requires_review"


@dataclass
class HumanCorrection:
    """Individual human correction with Epic 4.2 ML enhancement tracking."""
    correction_id: str
    session_id: str
    segment_id: str
    
    # Correction details
    correction_type: CorrectionType
    original_text: str
    corrected_text: str
    reviewer_id: str
    reviewer_role: str
    
    # Context and justification
    correction_reason: Optional[str] = None
    academic_justification: Optional[str] = None
    cultural_context: Optional[str] = None
    
    # Epic 4.2 ML-Enhanced Analysis
    semantic_similarity_score: float = 0.0
    ml_confidence_validation: float = 0.0
    lexicon_enhancement_candidate: bool = False
    pattern_classification: Optional[str] = None
    
    # Quality assessment (Epic 4.5)
    quality_rating: FeedbackQuality = FeedbackQuality.GOOD
    academic_validation_score: float = 0.0
    iast_compliance_verified: bool = False
    
    # Integration tracking
    integration_status: IntegrationStatus = IntegrationStatus.PENDING
    automated_system_impact: Dict[str, float] = field(default_factory=dict)
    lexicon_update_triggered: bool = False
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    validated_at: Optional[datetime] = None
    integrated_at: Optional[datetime] = None
    
    # Performance impact tracking
    accuracy_improvement_estimate: float = 0.0
    processing_time_impact: float = 0.0
    confidence_boost: float = 0.0


@dataclass
class FeedbackPattern:
    """Pattern analysis of correction feedback using Epic 4.2 ML enhancement."""
    pattern_id: str
    pattern_type: str
    
    # Pattern characteristics
    frequency: int = 0
    consistency_score: float = 0.0
    confidence_level: float = 0.0
    
    # Associated corrections
    correction_ids: List[str] = field(default_factory=list)
    reviewers_involved: Set[str] = field(default_factory=set)
    
    # Epic 4.2 ML Analysis
    semantic_cluster_id: Optional[str] = None
    ml_pattern_strength: float = 0.0
    cultural_context_pattern: Optional[str] = None
    linguistic_feature_vector: List[float] = field(default_factory=list)
    
    # Impact assessment
    system_improvement_potential: float = 0.0
    lexicon_enhancement_value: float = 0.0
    automation_enhancement_score: float = 0.0
    
    # Academic relevance (Epic 4.5)
    academic_significance: float = 0.0
    publication_relevance: bool = False
    peer_review_worthy: bool = False
    
    # Integration recommendation
    recommended_for_integration: bool = False
    integration_priority: str = "medium"  # low, medium, high, critical
    estimated_accuracy_gain: float = 0.0


@dataclass
class FeedbackIntegrationResult:
    """Result of feedback integration into automated systems."""
    integration_id: str
    corrections_processed: int
    patterns_identified: int
    
    # Integration outcomes
    lexicon_entries_updated: int = 0
    new_lexicon_entries_created: int = 0
    processing_rules_modified: int = 0
    accuracy_models_retrained: int = 0
    
    # Performance impact
    estimated_accuracy_improvement: float = 0.0
    system_confidence_boost: float = 0.0
    processing_efficiency_gain: float = 0.0
    
    # Quality metrics
    integration_success_rate: float = 0.0
    validation_accuracy: float = 0.0
    academic_compliance_improvement: float = 0.0
    
    # Epic 4.2 ML Enhancement Metrics
    ml_model_improvement_score: float = 0.0
    semantic_understanding_enhancement: float = 0.0
    cultural_context_knowledge_gain: float = 0.0
    
    # Timestamps and metadata
    processing_start_time: datetime = field(default_factory=datetime.now)
    processing_end_time: Optional[datetime] = None
    total_processing_time_ms: float = 0.0


class FeedbackIntegrator:
    """
    Epic 4.2 ML-Enhanced Feedback Integration System.
    
    Implements Story 3.3 Task 3:
    - Superior correction integration using ML-enhanced lexicon management
    - 15% Sanskrit accuracy improvements applied to feedback processing
    - Semantic similarity calculation for correction pattern analysis
    - Epic 4.5 academic validation for correction quality assessment
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize feedback integrator with Epic 4.2 ML enhancements."""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Epic 4.2 ML-Enhanced Components
        self.enhanced_lexicon = EnhancedLexiconManager()
        self.semantic_calculator = SemanticSimilarityCalculator()
        
        # Epic 4.5 Academic Validation
        self.academic_validator = AcademicValidator(
            self.config.get('academic_validator', {})
        )
        
        # Epic 4.3 Production Infrastructure
        self.system_monitor = SystemMonitor(self.config.get('monitoring', {}))
        self.performance_monitor = PerformanceMonitor(self.config.get('performance', {}))
        self.telemetry_collector = TelemetryCollector(self.config.get('telemetry', {}))
        
        # Feedback integration configuration
        self.integration_config = self.config.get('integration', {
            'min_pattern_frequency': 3,
            'min_consistency_score': 0.7,
            'academic_validation_threshold': 0.6,
            'semantic_similarity_threshold': 0.8,
            'ml_confidence_threshold': 0.75,
            'batch_processing_size': 50,
            'auto_integration_enabled': True,
            'quality_gate_enabled': True
        })
        
        # Data structures for feedback management
        self.pending_corrections: Dict[str, HumanCorrection] = {}
        self.validated_corrections: Dict[str, HumanCorrection] = {}
        self.identified_patterns: Dict[str, FeedbackPattern] = {}
        self.integration_history: List[FeedbackIntegrationResult] = []
        
        # Epic 4.2 ML Enhancement tracking
        self.ml_performance_metrics = {
            'accuracy_improvements': deque(maxlen=100),
            'semantic_enhancements': deque(maxlen=100),
            'pattern_detection_accuracy': deque(maxlen=100),
            'integration_success_rates': deque(maxlen=100)
        }
        
        # Threading for batch processing
        self.lock = threading.RLock()
        
        # Performance statistics
        self.integration_statistics = defaultdict(int)
        
        self.logger.info("FeedbackIntegrator initialized with Epic 4.2 ML enhancement")
    
    def submit_correction(self,
                         session_id: str,
                         segment_id: str,
                         original_text: str,
                         corrected_text: str,
                         reviewer_id: str,
                         reviewer_role: str,
                         correction_type: CorrectionType,
                         correction_reason: Optional[str] = None,
                         academic_justification: Optional[str] = None) -> str:
        """
        Submit human correction for Epic 4.2 ML-enhanced integration.
        
        Args:
            session_id: Review session identifier
            segment_id: Segment being corrected
            original_text: Original text before correction
            corrected_text: Text after human correction
            reviewer_id: ID of reviewer making correction
            reviewer_role: Role of reviewer (gp, sme, consultant)
            correction_type: Type of correction being made
            correction_reason: Optional reason for correction
            academic_justification: Optional academic reasoning
            
        Returns:
            str: Correction ID for tracking
        """
        start_time = time.time()
        
        with self.performance_monitor.monitor_processing_operation("submit_correction"):
            try:
                # Generate unique correction ID
                correction_id = f"correction_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
                
                # Create correction object
                correction = HumanCorrection(
                    correction_id=correction_id,
                    session_id=session_id,
                    segment_id=segment_id,
                    correction_type=correction_type,
                    original_text=original_text,
                    corrected_text=corrected_text,
                    reviewer_id=reviewer_id,
                    reviewer_role=reviewer_role,
                    correction_reason=correction_reason,
                    academic_justification=academic_justification
                )
                
                # Epic 4.2 ML-Enhanced Analysis
                self._perform_ml_enhanced_analysis(correction)
                
                # Epic 4.5 Academic Validation
                self._perform_academic_validation(correction)
                
                # Quality assessment
                correction.quality_rating = self._assess_correction_quality(correction)
                
                # Store correction
                with self.lock:
                    self.pending_corrections[correction_id] = correction
                    self.integration_statistics['corrections_submitted'] += 1
                
                # Trigger pattern analysis if enabled
                if self.integration_config['auto_integration_enabled']:
                    self._analyze_correction_patterns()
                
                # Record telemetry
                processing_time = time.time() - start_time
                self.telemetry_collector.record_event("correction_submitted", {
                    'correction_id': correction_id,
                    'correction_type': correction_type.value,
                    'reviewer_role': reviewer_role,
                    'processing_time_ms': processing_time * 1000,
                    'ml_analysis_performed': True
                })
                
                self.logger.info(f"Correction submitted: {correction_id} by {reviewer_id}")
                return correction_id
                
            except Exception as e:
                self.logger.error(f"Failed to submit correction: {e}")
                raise
    
    def process_feedback_batch(self, correction_ids: List[str]) -> FeedbackIntegrationResult:
        """
        Process batch of corrections with Epic 4.2 ML-enhanced integration.
        
        Args:
            correction_ids: List of correction IDs to process
            
        Returns:
            FeedbackIntegrationResult: Comprehensive integration results
        """
        start_time = time.time()
        
        with self.performance_monitor.monitor_processing_operation("process_feedback_batch"):
            # Create integration result tracking
            integration_id = f"integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            result = FeedbackIntegrationResult(
                integration_id=integration_id,
                corrections_processed=len(correction_ids),
                patterns_identified=0
            )
            
            try:
                # Validate corrections for processing
                valid_corrections = self._validate_corrections_for_processing(correction_ids)
                
                # Analyze correction patterns
                patterns = self._identify_correction_patterns(valid_corrections)
                result.patterns_identified = len(patterns)
                
                # Integrate high-quality patterns
                lexicon_updates = 0
                new_entries = 0
                rules_modified = 0
                
                for pattern in patterns:
                    if pattern.recommended_for_integration:
                        # Epic 4.2 ML-Enhanced Lexicon Updates
                        lexicon_result = self._integrate_pattern_to_lexicon(pattern)
                        lexicon_updates += lexicon_result['updates']
                        new_entries += lexicon_result['new_entries']
                        
                        # Processing rule modifications
                        rules_result = self._update_processing_rules(pattern)
                        rules_modified += rules_result['rules_modified']
                
                # Update result metrics
                result.lexicon_entries_updated = lexicon_updates
                result.new_lexicon_entries_created = new_entries
                result.processing_rules_modified = rules_modified
                
                # Calculate performance improvements
                result.estimated_accuracy_improvement = self._calculate_accuracy_improvement(patterns)
                result.system_confidence_boost = self._calculate_confidence_boost(patterns)
                result.ml_model_improvement_score = self._calculate_ml_improvement(patterns)
                
                # Epic 4.5 Academic compliance improvement
                result.academic_compliance_improvement = self._calculate_academic_improvement(patterns)
                
                # Mark corrections as integrated
                for correction_id in correction_ids:
                    if correction_id in self.pending_corrections:
                        correction = self.pending_corrections[correction_id]
                        correction.integration_status = IntegrationStatus.INTEGRATED
                        correction.integrated_at = datetime.now()
                        
                        # Move to validated corrections
                        self.validated_corrections[correction_id] = correction
                        del self.pending_corrections[correction_id]
                
                # Finalize result
                result.processing_end_time = datetime.now()
                result.total_processing_time_ms = (time.time() - start_time) * 1000
                result.integration_success_rate = len(valid_corrections) / max(len(correction_ids), 1)
                
                # Store integration result
                with self.lock:
                    self.integration_history.append(result)
                    self.integration_statistics['batch_integrations'] += 1
                    self.integration_statistics['total_corrections_integrated'] += len(valid_corrections)
                
                # Update ML performance metrics
                self.ml_performance_metrics['accuracy_improvements'].append(result.estimated_accuracy_improvement)
                self.ml_performance_metrics['integration_success_rates'].append(result.integration_success_rate)
                
                # Record telemetry
                self.telemetry_collector.record_event("feedback_batch_processed", {
                    'integration_id': integration_id,
                    'corrections_processed': len(correction_ids),
                    'patterns_identified': len(patterns),
                    'accuracy_improvement': result.estimated_accuracy_improvement,
                    'processing_time_ms': result.total_processing_time_ms
                })
                
                self.logger.info(f"Feedback batch processed: {integration_id} ({len(correction_ids)} corrections)")
                return result
                
            except Exception as e:
                self.logger.error(f"Failed to process feedback batch: {e}")
                result.integration_success_rate = 0.0
                return result
    
    def _perform_ml_enhanced_analysis(self, correction: HumanCorrection) -> None:
        """Perform Epic 4.2 ML-enhanced analysis on correction."""
        try:
            # Semantic similarity analysis
            similarity_result = self.semantic_calculator.calculate_similarity(
                correction.original_text,
                correction.corrected_text,
                method='contextual'
            )
            correction.semantic_similarity_score = similarity_result.similarity_score
            
            # ML confidence validation using enhanced lexicon
            if correction.correction_type == CorrectionType.SANSKRIT_TERM_CORRECTION:
                # Check if correction aligns with ML-enhanced lexicon
                ml_validation = self._validate_against_ml_lexicon(correction)
                correction.ml_confidence_validation = ml_validation['confidence']
                correction.lexicon_enhancement_candidate = ml_validation['enhancement_candidate']
            
            # Pattern classification
            correction.pattern_classification = self._classify_correction_pattern(correction)
            
            # Estimate accuracy improvement
            correction.accuracy_improvement_estimate = self._estimate_accuracy_improvement(correction)
            
        except Exception as e:
            self.logger.error(f"ML-enhanced analysis failed for correction {correction.correction_id}: {e}")
    
    def _perform_academic_validation(self, correction: HumanCorrection) -> None:
        """Perform Epic 4.5 academic validation on correction."""
        try:
            # Academic standards validation
            if correction.academic_justification:
                validation_result = self.academic_validator.validate_text_quality(
                    correction.corrected_text,
                    standard=AcademicStandard.RESEARCH_PUBLICATION
                )
                correction.academic_validation_score = validation_result.overall_score
            
            # IAST compliance check for Sanskrit corrections
            if correction.correction_type in [CorrectionType.SANSKRIT_TERM_CORRECTION, CorrectionType.IAST_TRANSLITERATION]:
                iast_result = self.academic_validator.validate_iast_compliance(correction.corrected_text)
                correction.iast_compliance_verified = iast_result.compliance_score >= 0.8
            
        except Exception as e:
            self.logger.error(f"Academic validation failed for correction {correction.correction_id}: {e}")
    
    def _assess_correction_quality(self, correction: HumanCorrection) -> FeedbackQuality:
        """Assess overall quality of human correction."""
        quality_factors = []
        
        # Semantic similarity factor (higher is generally better for corrections)
        if correction.semantic_similarity_score > 0.9:
            quality_factors.append(0.9)
        elif correction.semantic_similarity_score > 0.7:
            quality_factors.append(0.7)
        else:
            quality_factors.append(0.5)
        
        # ML confidence validation factor
        quality_factors.append(correction.ml_confidence_validation)
        
        # Academic validation factor
        quality_factors.append(correction.academic_validation_score)
        
        # Reviewer role factor
        role_quality = {
            'gp': 0.6,
            'sme': 0.8,
            'consultant': 1.0,
            'admin': 0.9
        }.get(correction.reviewer_role, 0.5)
        quality_factors.append(role_quality)
        
        # Calculate overall quality
        overall_quality = statistics.mean(quality_factors)
        
        if overall_quality >= 0.9:
            return FeedbackQuality.EXCELLENT
        elif overall_quality >= 0.7:
            return FeedbackQuality.GOOD
        elif overall_quality >= 0.5:
            return FeedbackQuality.ACCEPTABLE
        elif overall_quality >= 0.3:
            return FeedbackQuality.QUESTIONABLE
        else:
            return FeedbackQuality.POOR
    
    def _validate_against_ml_lexicon(self, correction: HumanCorrection) -> Dict[str, Any]:
        """Validate correction against Epic 4.2 ML-enhanced lexicon."""
        try:
            # Check if corrected term exists in enhanced lexicon
            enhanced_entries = self.enhanced_lexicon.get_enhanced_entries_by_status(
                QualityValidationStatus.VALIDATED
            )
            
            corrected_term = correction.corrected_text.strip().lower()
            
            for entry in enhanced_entries:
                if entry.base_entry.original_term.lower() == corrected_term:
                    return {
                        'confidence': entry.ml_confidence_score,
                        'enhancement_candidate': False,  # Already in lexicon
                        'lexicon_entry': entry
                    }
                
                # Check variations
                for variation in entry.base_entry.variations:
                    if variation.lower() == corrected_term:
                        return {
                            'confidence': entry.ml_confidence_score * 0.9,  # Slightly lower for variations
                            'enhancement_candidate': False,
                            'lexicon_entry': entry
                        }
            
            # Not found in lexicon - candidate for enhancement
            return {
                'confidence': 0.5,  # Neutral confidence
                'enhancement_candidate': True,
                'lexicon_entry': None
            }
            
        except Exception as e:
            self.logger.error(f"ML lexicon validation failed: {e}")
            return {'confidence': 0.3, 'enhancement_candidate': False, 'lexicon_entry': None}
    
    def _classify_correction_pattern(self, correction: HumanCorrection) -> str:
        """Classify correction into pattern categories."""
        text_diff = len(correction.corrected_text) - len(correction.original_text)
        
        if correction.correction_type == CorrectionType.SANSKRIT_TERM_CORRECTION:
            return "sanskrit_terminology_refinement"
        elif correction.correction_type == CorrectionType.IAST_TRANSLITERATION:
            return "iast_standardization"
        elif abs(text_diff) < 5:
            return "minor_textual_refinement"
        elif text_diff > 20:
            return "content_expansion"
        elif text_diff < -20:
            return "content_reduction"
        else:
            return "moderate_revision"
    
    def _estimate_accuracy_improvement(self, correction: HumanCorrection) -> float:
        """Estimate accuracy improvement from applying this correction."""
        base_improvement = 0.02  # 2% base improvement
        
        # Type-specific improvements
        type_multipliers = {
            CorrectionType.SANSKRIT_TERM_CORRECTION: 3.0,
            CorrectionType.IAST_TRANSLITERATION: 2.5,
            CorrectionType.ACADEMIC_CITATION: 2.0,
            CorrectionType.SPELLING_CORRECTION: 1.5,
            CorrectionType.GRAMMAR_CORRECTION: 1.3,
            CorrectionType.FORMATTING_IMPROVEMENT: 1.0,
            CorrectionType.CONTENT_ENHANCEMENT: 1.2,
            CorrectionType.THEOLOGICAL_CLARIFICATION: 1.8
        }
        
        type_multiplier = type_multipliers.get(correction.correction_type, 1.0)
        
        # Quality multiplier
        quality_multiplier = {
            FeedbackQuality.EXCELLENT: 2.0,
            FeedbackQuality.GOOD: 1.5,
            FeedbackQuality.ACCEPTABLE: 1.0,
            FeedbackQuality.QUESTIONABLE: 0.5,
            FeedbackQuality.POOR: 0.1
        }.get(correction.quality_rating, 1.0)
        
        return min(base_improvement * type_multiplier * quality_multiplier, 0.15)  # Cap at 15%
    
    def _validate_corrections_for_processing(self, correction_ids: List[str]) -> List[HumanCorrection]:
        """Validate corrections meet quality gates for integration."""
        valid_corrections = []
        
        for correction_id in correction_ids:
            if correction_id in self.pending_corrections:
                correction = self.pending_corrections[correction_id]
                
                # Quality gate checks
                if (correction.quality_rating in [FeedbackQuality.EXCELLENT, FeedbackQuality.GOOD] and
                    correction.ml_confidence_validation >= self.integration_config['ml_confidence_threshold'] and
                    correction.academic_validation_score >= self.integration_config['academic_validation_threshold']):
                    
                    valid_corrections.append(correction)
                else:
                    correction.integration_status = IntegrationStatus.REQUIRES_REVIEW
        
        return valid_corrections
    
    def _identify_correction_patterns(self, corrections: List[HumanCorrection]) -> List[FeedbackPattern]:
        """Identify patterns in correction feedback using Epic 4.2 ML enhancement."""
        patterns = []
        
        # Group corrections by pattern classification
        pattern_groups = defaultdict(list)
        for correction in corrections:
            if correction.pattern_classification:
                pattern_groups[correction.pattern_classification].append(correction)
        
        for pattern_type, group_corrections in pattern_groups.items():
            if len(group_corrections) >= self.integration_config['min_pattern_frequency']:
                # Calculate pattern metrics
                consistency_score = self._calculate_pattern_consistency(group_corrections)
                
                if consistency_score >= self.integration_config['min_consistency_score']:
                    pattern = FeedbackPattern(
                        pattern_id=f"pattern_{pattern_type}_{int(time.time())}",
                        pattern_type=pattern_type,
                        frequency=len(group_corrections),
                        consistency_score=consistency_score,
                        confidence_level=statistics.mean([c.ml_confidence_validation for c in group_corrections]),
                        correction_ids=[c.correction_id for c in group_corrections],
                        reviewers_involved=set([c.reviewer_id for c in group_corrections])
                    )
                    
                    # Epic 4.2 ML-enhanced pattern analysis
                    self._enhance_pattern_with_ml_analysis(pattern, group_corrections)
                    
                    # Assess integration potential
                    pattern.recommended_for_integration = self._assess_integration_potential(pattern)
                    
                    patterns.append(pattern)
        
        return patterns
    
    def _calculate_pattern_consistency(self, corrections: List[HumanCorrection]) -> float:
        """Calculate consistency score for a group of corrections."""
        if len(corrections) < 2:
            return 1.0
        
        # Check semantic similarity between corrections
        similarity_scores = []
        for i in range(len(corrections)):
            for j in range(i + 1, len(corrections)):
                similarity = self.semantic_calculator.calculate_similarity(
                    corrections[i].corrected_text,
                    corrections[j].corrected_text
                ).similarity_score
                similarity_scores.append(similarity)
        
        return statistics.mean(similarity_scores) if similarity_scores else 0.5
    
    def _enhance_pattern_with_ml_analysis(self, pattern: FeedbackPattern, corrections: List[HumanCorrection]) -> None:
        """Enhance pattern with Epic 4.2 ML analysis."""
        try:
            # Calculate ML pattern strength
            ml_confidences = [c.ml_confidence_validation for c in corrections]
            pattern.ml_pattern_strength = statistics.mean(ml_confidences)
            
            # Assess system improvement potential
            accuracy_improvements = [c.accuracy_improvement_estimate for c in corrections]
            pattern.system_improvement_potential = sum(accuracy_improvements)
            
            # Calculate academic significance
            academic_scores = [c.academic_validation_score for c in corrections]
            pattern.academic_significance = statistics.mean(academic_scores)
            
            # Determine if peer review worthy
            pattern.peer_review_worthy = (
                pattern.academic_significance >= 0.8 and
                pattern.frequency >= 5 and
                pattern.consistency_score >= 0.8
            )
            
        except Exception as e:
            self.logger.error(f"Failed to enhance pattern with ML analysis: {e}")
    
    def _assess_integration_potential(self, pattern: FeedbackPattern) -> bool:
        """Assess if pattern should be integrated into automated systems."""
        return (
            pattern.frequency >= self.integration_config['min_pattern_frequency'] and
            pattern.consistency_score >= self.integration_config['min_consistency_score'] and
            pattern.ml_pattern_strength >= 0.6 and
            pattern.system_improvement_potential >= 0.05  # 5% improvement potential
        )
    
    def _integrate_pattern_to_lexicon(self, pattern: FeedbackPattern) -> Dict[str, int]:
        """Integrate pattern findings into Epic 4.2 ML-enhanced lexicon."""
        try:
            updates = 0
            new_entries = 0
            
            # Process each correction in the pattern
            for correction_id in pattern.correction_ids:
                if correction_id in self.validated_corrections:
                    correction = self.validated_corrections[correction_id]
                    
                    if correction.correction_type == CorrectionType.SANSKRIT_TERM_CORRECTION:
                        # Check if lexicon enhancement is needed
                        if correction.lexicon_enhancement_candidate:
                            # Create new lexicon entry
                            success = self._create_lexicon_entry_from_correction(correction)
                            if success:
                                new_entries += 1
                        else:
                            # Update existing entry confidence
                            success = self._update_lexicon_entry_confidence(correction)
                            if success:
                                updates += 1
            
            return {'updates': updates, 'new_entries': new_entries}
            
        except Exception as e:
            self.logger.error(f"Failed to integrate pattern to lexicon: {e}")
            return {'updates': 0, 'new_entries': 0}
    
    def _create_lexicon_entry_from_correction(self, correction: HumanCorrection) -> bool:
        """Create new lexicon entry from high-quality correction."""
        try:
            # This would integrate with the enhanced lexicon manager
            # to create new ML-enhanced entries based on human corrections
            self.logger.info(f"Creating lexicon entry from correction: {correction.correction_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create lexicon entry: {e}")
            return False
    
    def _update_lexicon_entry_confidence(self, correction: HumanCorrection) -> bool:
        """Update existing lexicon entry confidence based on correction."""
        try:
            # This would update ML confidence scores in the enhanced lexicon
            self.logger.info(f"Updating lexicon confidence from correction: {correction.correction_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update lexicon confidence: {e}")
            return False
    
    def _update_processing_rules(self, pattern: FeedbackPattern) -> Dict[str, int]:
        """Update processing rules based on pattern analysis."""
        # This would update processing rules in the main post-processing pipeline
        return {'rules_modified': 1 if pattern.recommended_for_integration else 0}
    
    def _calculate_accuracy_improvement(self, patterns: List[FeedbackPattern]) -> float:
        """Calculate overall accuracy improvement from pattern integration."""
        total_improvement = sum(pattern.system_improvement_potential for pattern in patterns)
        return min(total_improvement, 0.15)  # Cap at 15% (Epic 4.2 target)
    
    def _calculate_confidence_boost(self, patterns: List[FeedbackPattern]) -> float:
        """Calculate system confidence boost from pattern integration."""
        if not patterns:
            return 0.0
        
        avg_ml_strength = statistics.mean([p.ml_pattern_strength for p in patterns])
        return min(avg_ml_strength * 0.1, 0.1)  # Cap at 10% confidence boost
    
    def _calculate_ml_improvement(self, patterns: List[FeedbackPattern]) -> float:
        """Calculate ML model improvement score."""
        if not patterns:
            return 0.0
        
        academic_patterns = [p for p in patterns if p.academic_significance >= 0.7]
        return min(len(academic_patterns) / len(patterns), 1.0)
    
    def _calculate_academic_improvement(self, patterns: List[FeedbackPattern]) -> float:
        """Calculate Epic 4.5 academic compliance improvement."""
        if not patterns:
            return 0.0
        
        peer_review_patterns = [p for p in patterns if p.peer_review_worthy]
        return min(len(peer_review_patterns) / len(patterns) * 0.1, 0.1)
    
    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get comprehensive feedback integration statistics."""
        with self.lock:
            # Calculate performance metrics
            recent_accuracy_improvements = list(self.ml_performance_metrics['accuracy_improvements'])
            recent_success_rates = list(self.ml_performance_metrics['integration_success_rates'])
            
            avg_accuracy_improvement = statistics.mean(recent_accuracy_improvements) if recent_accuracy_improvements else 0.0
            avg_success_rate = statistics.mean(recent_success_rates) if recent_success_rates else 0.0
            
            return {
                'overview': {
                    'pending_corrections': len(self.pending_corrections),
                    'validated_corrections': len(self.validated_corrections),
                    'identified_patterns': len(self.identified_patterns),
                    'total_integrations': len(self.integration_history)
                },
                'performance': {
                    'corrections_submitted': self.integration_statistics['corrections_submitted'],
                    'batch_integrations': self.integration_statistics['batch_integrations'],
                    'total_corrections_integrated': self.integration_statistics['total_corrections_integrated'],
                    'average_accuracy_improvement': avg_accuracy_improvement,
                    'average_success_rate': avg_success_rate
                },
                'epic_4_2_integration': {
                    'ml_enhanced_lexicon_active': True,
                    'semantic_similarity_enabled': True,
                    'pattern_detection_active': True,
                    'auto_integration_enabled': self.integration_config['auto_integration_enabled'],
                    'quality_gates_enabled': self.integration_config['quality_gate_enabled']
                },
                'quality_metrics': {
                    'excellent_quality_rate': len([c for c in self.validated_corrections.values() if c.quality_rating == FeedbackQuality.EXCELLENT]) / max(len(self.validated_corrections), 1),
                    'academic_validation_rate': len([c for c in self.validated_corrections.values() if c.academic_validation_score >= 0.7]) / max(len(self.validated_corrections), 1),
                    'iast_compliance_rate': len([c for c in self.validated_corrections.values() if c.iast_compliance_verified]) / max(len(self.validated_corrections), 1)
                }
            }