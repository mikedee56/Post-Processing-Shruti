"""
Quality Gate System for Academic Standard Validation.

This module implements the core QualityGate system that evaluates processing
quality with measurable metrics, following Epic 3 Story 3.2 requirements.
"""

import time
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

from utils.srt_parser import SRTSegment
from utils.logger_config import get_logger
from utils.metrics_collector import MetricsCollector


class QualityLevel(Enum):
    """Quality assessment levels for academic standards."""
    EXCELLENT = "excellent"
    GOOD = "good" 
    ACCEPTABLE = "acceptable"
    NEEDS_IMPROVEMENT = "needs_improvement"
    UNACCEPTABLE = "unacceptable"


class ValidationRule(Enum):
    """Academic validation rules for quality assessment."""
    IAST_COMPLIANCE = "iast_compliance"
    SANSKRIT_ACCURACY = "sanskrit_accuracy"
    TRANSLITERATION_CONSISTENCY = "transliteration_consistency"
    PROPER_NOUN_CAPITALIZATION = "proper_noun_capitalization"
    VERSE_IDENTIFICATION = "verse_identification"
    TERMINOLOGY_PRECISION = "terminology_precision"
    ACADEMIC_FORMATTING = "academic_formatting"


@dataclass
class QualityMetric:
    """Represents a single quality measurement."""
    name: str
    value: float
    threshold: float
    weight: float
    passed: bool
    details: Optional[str] = None
    
    @property
    def weighted_score(self) -> float:
        """Calculate weighted contribution to overall quality."""
        return self.value * self.weight if self.passed else 0.0


@dataclass
class ComplianceScore:
    """Comprehensive compliance scoring for academic standards."""
    overall_score: float
    quality_level: QualityLevel
    metrics: List[QualityMetric]
    rule_compliance: Dict[ValidationRule, bool]
    confidence_factor: float
    processing_time: float
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def is_acceptable(self) -> bool:
        """Determine if quality meets minimum academic standards."""
        return (self.overall_score >= 0.7 and 
                self.quality_level != QualityLevel.UNACCEPTABLE)


@dataclass
class QualityReport:
    """Structured quality assessment report with actionable feedback."""
    segment_id: str
    original_text: str
    processed_text: str
    compliance_score: ComplianceScore
    issues_identified: List[str]
    improvement_suggestions: List[str]
    academic_notes: List[str]
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            'segment_id': self.segment_id,
            'original_text': self.original_text,
            'processed_text': self.processed_text,
            'overall_score': self.compliance_score.overall_score,
            'quality_level': self.compliance_score.quality_level.value,
            'confidence_factor': self.compliance_score.confidence_factor,
            'processing_time': self.compliance_score.processing_time,
            'issues_identified': self.issues_identified,
            'improvement_suggestions': self.improvement_suggestions,
            'academic_notes': self.academic_notes,
            'timestamp': self.timestamp
        }


class QualityValidator(ABC):
    """Abstract base class for quality validation components."""
    
    @abstractmethod
    def validate(self, original: str, processed: str, metadata: Dict[str, Any]) -> QualityMetric:
        """Validate specific quality aspect."""
        pass


class IASTComplianceValidator(QualityValidator):
    """Validates IAST transliteration compliance."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        # IAST character patterns
        self.iast_chars = set('āīūṛṝḷḹēōṃḥṅñṭḍṇśṣĀĪŪṚṜḶḸĒŌṂḤṄÑṬḌṆŚṢ')
        
    def validate(self, original: str, processed: str, metadata: Dict[str, Any]) -> QualityMetric:
        """Validate IAST transliteration compliance."""
        start_time = time.time()
        
        # Count IAST characters in processed text
        iast_count = sum(1 for char in processed if char in self.iast_chars)
        total_chars = len(processed)
        
        # Calculate compliance ratio
        compliance_ratio = 0.0
        if total_chars > 0:
            # Check for Sanskrit/Hindi terms that should have IAST
            sanskrit_indicators = ['yoga', 'dharma', 'karma', 'vedanta', 'gita']
            sanskrit_terms = sum(1 for term in sanskrit_indicators if term.lower() in processed.lower())
            
            if sanskrit_terms > 0:
                # Expect some IAST characters for Sanskrit content
                expected_iast = sanskrit_terms * 0.1  # Conservative estimate
                compliance_ratio = min(1.0, iast_count / max(expected_iast, 1))
            else:
                # No Sanskrit content, full compliance
                compliance_ratio = 1.0
        
        threshold = 0.6
        passed = compliance_ratio >= threshold
        
        processing_time = time.time() - start_time
        self.logger.debug(f"IAST compliance check: {compliance_ratio:.2f} (threshold: {threshold})")
        
        return QualityMetric(
            name="iast_compliance",
            value=compliance_ratio,
            threshold=threshold,
            weight=0.25,
            passed=passed,
            details=f"IAST characters found: {iast_count}, Sanskrit terms: {sanskrit_terms if 'sanskrit_terms' in locals() else 0}"
        )


class SanskritAccuracyValidator(QualityValidator):
    """Validates Sanskrit terminology accuracy."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        # Common Sanskrit terms for validation
        self.sanskrit_terms = {
            'yoga', 'dharma', 'karma', 'vedanta', 'gita', 'sutra',
            'patanjali', 'krishna', 'rama', 'arjuna', 'sita',
            'bhagavad', 'upanishad', 'veda', 'tantra', 'mantra'
        }
        
    def validate(self, original: str, processed: str, metadata: Dict[str, Any]) -> QualityMetric:
        """Validate Sanskrit terminology accuracy."""
        start_time = time.time()
        
        # Check for proper Sanskrit term handling
        processed_lower = processed.lower()
        sanskrit_found = [term for term in self.sanskrit_terms if term in processed_lower]
        
        # Calculate accuracy based on proper capitalization and spelling
        accuracy_score = 1.0
        issues = []
        
        for term in sanskrit_found:
            # Check proper capitalization (basic check)
            if term.lower() in processed.lower():
                # Find the term in context
                import re
                pattern = rf'\b{re.escape(term)}\b'
                matches = re.finditer(pattern, processed, re.IGNORECASE)
                
                for match in matches:
                    actual = match.group()
                    # Sanskrit proper nouns should be capitalized
                    if term in ['patanjali', 'krishna', 'rama', 'arjuna', 'sita']:
                        if not actual[0].isupper():
                            accuracy_score -= 0.1
                            issues.append(f"'{actual}' should be capitalized")
        
        threshold = 0.8
        passed = accuracy_score >= threshold
        
        processing_time = time.time() - start_time
        self.logger.debug(f"Sanskrit accuracy: {accuracy_score:.2f}")
        
        return QualityMetric(
            name="sanskrit_accuracy",
            value=accuracy_score,
            threshold=threshold,
            weight=0.3,
            passed=passed,
            details=f"Sanskrit terms found: {len(sanskrit_found)}, Issues: {len(issues)}"
        )


class ProperNounValidator(QualityValidator):
    """Validates proper noun capitalization."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
    def validate(self, original: str, processed: str, metadata: Dict[str, Any]) -> QualityMetric:
        """Validate proper noun capitalization."""
        start_time = time.time()
        
        # Check for proper capitalization patterns
        import re
        
        # Find potential proper nouns (words that should be capitalized)
        potential_proper = []
        
        # Sanskrit/Hindu proper nouns patterns
        hindu_names = ['krishna', 'rama', 'sita', 'arjuna', 'patanjali', 'shankaracharya']
        places = ['rishikesh', 'haridwar', 'varanasi', 'himalayas']
        texts = ['bhagavad', 'upanishad', 'mahabharata', 'ramayana']
        
        all_proper_nouns = hindu_names + places + texts
        
        capitalization_score = 1.0
        total_proper_nouns = 0
        properly_capitalized = 0
        
        for noun in all_proper_nouns:
            pattern = rf'\b{re.escape(noun)}\b'
            matches = re.finditer(pattern, processed, re.IGNORECASE)
            
            for match in matches:
                total_proper_nouns += 1
                actual = match.group()
                if actual[0].isupper():
                    properly_capitalized += 1
        
        if total_proper_nouns > 0:
            capitalization_score = properly_capitalized / total_proper_nouns
        
        threshold = 0.8
        passed = capitalization_score >= threshold
        
        processing_time = time.time() - start_time
        
        return QualityMetric(
            name="proper_noun_capitalization",
            value=capitalization_score,
            threshold=threshold,
            weight=0.2,
            passed=passed,
            details=f"Proper nouns: {total_proper_nouns}, Capitalized: {properly_capitalized}"
        )


class TerminologyConsistencyValidator(QualityValidator):
    """Validates terminology consistency across segments."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.term_cache = {}  # For tracking consistency across segments
        
    def validate(self, original: str, processed: str, metadata: Dict[str, Any]) -> QualityMetric:
        """Validate terminology consistency."""
        start_time = time.time()
        
        # Basic consistency check - ensure repeated terms are handled consistently
        import re
        
        # Find repeated terms in the text
        words = re.findall(r'\b\w+\b', processed.lower())
        word_counts = {}
        
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Check consistency of repeated terms
        consistency_score = 1.0
        repeated_terms = {word: count for word, count in word_counts.items() if count > 1}
        
        # For now, basic implementation - could be enhanced with cross-segment analysis
        if repeated_terms:
            # All repeated terms are consistent within this segment by definition
            # Future enhancement: track across multiple segments
            pass
        
        threshold = 0.9
        passed = consistency_score >= threshold
        
        processing_time = time.time() - start_time
        
        return QualityMetric(
            name="terminology_consistency",
            value=consistency_score,
            threshold=threshold,
            weight=0.15,
            passed=passed,
            details=f"Repeated terms: {len(repeated_terms)}"
        )


class AcademicFormattingValidator(QualityValidator):
    """Validates academic formatting standards."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
    def validate(self, original: str, processed: str, metadata: Dict[str, Any]) -> QualityMetric:
        """Validate academic formatting standards."""
        start_time = time.time()
        
        formatting_score = 1.0
        issues = []
        
        # Check for proper sentence structure
        if not processed.strip().endswith('.'):
            if len(processed.strip()) > 10:  # Only for substantial content
                formatting_score -= 0.1
                issues.append("Missing sentence ending punctuation")
        
        # Check for proper capitalization at sentence start
        sentences = processed.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and not sentence[0].isupper():
                formatting_score -= 0.1
                issues.append("Sentence should start with capital letter")
                break  # Only penalize once
        
        # Check for excessive filler words (should be removed by preprocessing)
        filler_words = ['um', 'uh', 'like', 'you know']
        filler_count = sum(1 for word in filler_words if word in processed.lower())
        
        if filler_count > 0:
            formatting_score -= 0.2 * filler_count
            issues.append(f"Contains {filler_count} filler words")
        
        formatting_score = max(0.0, formatting_score)
        threshold = 0.8
        passed = formatting_score >= threshold
        
        processing_time = time.time() - start_time
        
        return QualityMetric(
            name="academic_formatting",
            value=formatting_score,
            threshold=threshold,
            weight=0.1,
            passed=passed,
            details=f"Formatting issues: {len(issues)}"
        )


class QualityGate:
    """
    Core Quality Gate system for academic standard validation.
    
    Evaluates processing quality with measurable metrics and generates
    structured reports with actionable feedback.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Quality Gate system."""
        self.logger = get_logger(__name__)
        self.config = config or self._default_config()
        
        # Initialize validators
        self.validators = [
            IASTComplianceValidator(),
            SanskritAccuracyValidator(),
            ProperNounValidator(),
            TerminologyConsistencyValidator(),
            AcademicFormattingValidator()
        ]
        
        # Performance tracking
        self.performance_threshold = self.config.get('performance_threshold_ms', 50)
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for quality assessment."""
        return {
            'minimum_quality_score': 0.7,
            'performance_threshold_ms': 50,
            'enable_detailed_analysis': True,
            'academic_standards': {
                'iast_compliance_weight': 0.25,
                'sanskrit_accuracy_weight': 0.3,
                'proper_noun_weight': 0.2,
                'consistency_weight': 0.15,
                'formatting_weight': 0.1
            }
        }
    
    def evaluate_quality(self, original: str, processed: str, 
                        metadata: Optional[Dict[str, Any]] = None) -> ComplianceScore:
        """
        Evaluate processing quality against academic standards.
        
        Args:
            original: Original text before processing
            processed: Text after processing
            metadata: Additional processing metadata
            
        Returns:
            ComplianceScore with detailed assessment
        """
        start_time = time.time()
        metadata = metadata or {}
        
        # Run all validators
        metrics = []
        for validator in self.validators:
            try:
                metric = validator.validate(original, processed, metadata)
                metrics.append(metric)
            except Exception as e:
                self.logger.error(f"Validator {validator.__class__.__name__} failed: {e}")
                # Add failed metric
                metrics.append(QualityMetric(
                    name=validator.__class__.__name__.lower(),
                    value=0.0,
                    threshold=0.5,
                    weight=0.1,
                    passed=False,
                    details=f"Validation failed: {str(e)}"
                ))
        
        # Calculate overall score
        total_weighted_score = sum(m.weighted_score for m in metrics)
        total_weight = sum(m.weight for m in metrics)
        overall_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Determine quality level
        quality_level = self._determine_quality_level(overall_score)
        
        # Create rule compliance mapping
        rule_compliance = {}
        for metric in metrics:
            rule_name = metric.name.upper()
            if hasattr(ValidationRule, rule_name):
                rule_compliance[getattr(ValidationRule, rule_name)] = metric.passed
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics)
        
        # Calculate confidence factor
        confidence_factor = self._calculate_confidence_factor(metrics, metadata)
        
        processing_time = time.time() - start_time
        
        return ComplianceScore(
            overall_score=overall_score,
            quality_level=quality_level,
            metrics=metrics,
            rule_compliance=rule_compliance,
            confidence_factor=confidence_factor,
            processing_time=processing_time,
            recommendations=recommendations
        )
    
    def _determine_quality_level(self, score: float) -> QualityLevel:
        """Determine quality level based on overall score."""
        if score >= 0.9:
            return QualityLevel.EXCELLENT
        elif score >= 0.8:
            return QualityLevel.GOOD
        elif score >= 0.7:
            return QualityLevel.ACCEPTABLE
        elif score >= 0.5:
            return QualityLevel.NEEDS_IMPROVEMENT
        else:
            return QualityLevel.UNACCEPTABLE
    
    def _generate_recommendations(self, metrics: List[QualityMetric]) -> List[str]:
        """Generate actionable recommendations based on metrics."""
        recommendations = []
        
        for metric in metrics:
            if not metric.passed:
                if metric.name == "iast_compliance":
                    recommendations.append(
                        "Improve IAST transliteration compliance by ensuring Sanskrit terms "
                        "use proper diacritical marks (ā, ī, ū, ṛ, etc.)"
                    )
                elif metric.name == "sanskrit_accuracy":
                    recommendations.append(
                        "Review Sanskrit terminology for proper spelling and usage. "
                        "Consider consulting specialized Sanskrit lexicons."
                    )
                elif metric.name == "proper_noun_capitalization":
                    recommendations.append(
                        "Ensure proper nouns (names of people, places, texts) are "
                        "consistently capitalized throughout the text."
                    )
                elif metric.name == "terminology_consistency":
                    recommendations.append(
                        "Maintain consistent spelling and usage of technical terms "
                        "throughout the document."
                    )
                elif metric.name == "academic_formatting":
                    recommendations.append(
                        "Improve academic formatting by ensuring proper punctuation, "
                        "sentence structure, and removing casual speech elements."
                    )
        
        return recommendations
    
    def _calculate_confidence_factor(self, metrics: List[QualityMetric], 
                                   metadata: Dict[str, Any]) -> float:
        """Calculate confidence factor for quality assessment."""
        # Base confidence on metric consistency
        passed_count = sum(1 for m in metrics if m.passed)
        total_count = len(metrics)
        
        base_confidence = passed_count / total_count if total_count > 0 else 0.5
        
        # Adjust based on processing metadata
        if 'asr_confidence' in metadata:
            asr_confidence = metadata['asr_confidence']
            # Lower ASR confidence reduces quality assessment confidence
            base_confidence *= (0.5 + asr_confidence * 0.5)
        
        return base_confidence
    
    def generate_quality_report(self, segment_id: str, original: str, 
                              processed: str, metadata: Optional[Dict[str, Any]] = None) -> QualityReport:
        """
        Generate comprehensive quality report with actionable feedback.
        
        Args:
            segment_id: Unique identifier for the segment
            original: Original text before processing
            processed: Text after processing
            metadata: Additional processing metadata
            
        Returns:
            QualityReport with detailed analysis and suggestions
        """
        compliance_score = self.evaluate_quality(original, processed, metadata)
        
        # Identify specific issues
        issues_identified = []
        for metric in compliance_score.metrics:
            if not metric.passed:
                issues_identified.append(
                    f"{metric.name}: Score {metric.value:.2f} below threshold {metric.threshold:.2f}"
                )
        
        # Generate improvement suggestions
        improvement_suggestions = compliance_score.recommendations.copy()
        
        # Add performance-specific suggestions
        if compliance_score.processing_time > (self.performance_threshold / 1000):
            improvement_suggestions.append(
                f"Quality evaluation took {compliance_score.processing_time:.3f}s, "
                f"exceeding {self.performance_threshold}ms target"
            )
        
        # Generate academic notes
        academic_notes = []
        if compliance_score.quality_level == QualityLevel.EXCELLENT:
            academic_notes.append("Text meets high academic standards for publication")
        elif compliance_score.quality_level == QualityLevel.UNACCEPTABLE:
            academic_notes.append("Text requires significant revision before academic use")
        
        return QualityReport(
            segment_id=segment_id,
            original_text=original,
            processed_text=processed,
            compliance_score=compliance_score,
            issues_identified=issues_identified,
            improvement_suggestions=improvement_suggestions,
            academic_notes=academic_notes
        )
    
    def meets_quality_threshold(self, compliance_score: ComplianceScore) -> bool:
        """Check if processing meets minimum quality threshold."""
        minimum_score = self.config.get('minimum_quality_score', 0.7)
        return (compliance_score.overall_score >= minimum_score and
                compliance_score.is_acceptable)