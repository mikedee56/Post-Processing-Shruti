"""
Processing Quality Validator for timestamp integrity and correction impact assessment.

This module provides quality validation capabilities to ensure that text processing
maintains timestamp integrity and semantic preservation while measuring the impact
of corrections on transcript quality.
"""

import re
import logging
from typing import Dict, List, Set, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
import statistics
from .srt_parser import SRTSegment


class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"


class ValidationResult(Enum):
    """Validation result status."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


@dataclass
class ValidationIssue:
    """Represents a validation issue found during processing."""
    issue_type: str
    severity: ValidationResult
    description: str
    segment_index: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    original_text: Optional[str] = None
    processed_text: Optional[str] = None
    confidence_score: float = 0.0


@dataclass
class QualityReport:
    """Comprehensive quality assessment report."""
    timestamp_integrity_score: float
    semantic_preservation_score: float
    correction_impact_score: float
    overall_quality_score: float
    validation_issues: List[ValidationIssue]
    processing_metrics: Dict[str, float]
    recommendations: List[str]
    passed_validation: bool


@dataclass
class TimestampValidationResult:
    """Result of timestamp integrity validation."""
    is_valid: bool
    issues: List[ValidationIssue]
    integrity_score: float
    total_segments: int
    problematic_segments: int


@dataclass
class SemanticValidationResult:
    """Result of semantic preservation validation."""
    preservation_score: float
    drift_segments: List[Tuple[int, float]]  # (segment_index, drift_score)
    average_drift: float
    max_drift: float
    acceptable_drift: bool


class ProcessingQualityValidator:
    """
    Validates processing quality including timestamp integrity and semantic preservation.
    
    This validator ensures that text processing operations maintain the integrity
    of SRT timestamps and preserve the original semantic meaning of the content.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the processing quality validator.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Validation thresholds
        self.validation_level = ValidationLevel(self.config.get('validation_level', 'moderate'))
        self.max_semantic_drift = self.config.get('max_semantic_drift', 0.3)
        self.max_timestamp_deviation = self.config.get('max_timestamp_deviation', 0.001)
        self.min_quality_score = self.config.get('min_quality_score', 0.7)
        
        # Setup validation rules based on level
        self._setup_validation_thresholds()
    
    def validate_processing_quality(
        self, 
        original_segments: List[SRTSegment], 
        processed_segments: List[SRTSegment],
        corrections_applied: List[str] = None
    ) -> QualityReport:
        """
        Perform comprehensive quality validation of processed segments.
        
        Args:
            original_segments: Original SRT segments
            processed_segments: Processed SRT segments
            corrections_applied: List of correction types applied
            
        Returns:
            QualityReport with comprehensive assessment
        """
        validation_issues = []
        processing_metrics = {}
        recommendations = []
        
        # Validate timestamp integrity
        timestamp_result = self.validate_timestamp_integrity(original_segments, processed_segments)
        validation_issues.extend(timestamp_result.issues)
        processing_metrics['timestamp_integrity'] = timestamp_result.integrity_score
        
        # Validate semantic preservation
        semantic_result = self.validate_semantic_preservation(original_segments, processed_segments)
        validation_issues.extend(self._convert_semantic_issues(semantic_result))
        processing_metrics['semantic_preservation'] = semantic_result.preservation_score
        
        # Assess correction impact
        correction_impact = self.assess_correction_impact(original_segments, processed_segments, corrections_applied or [])
        processing_metrics['correction_impact'] = correction_impact
        
        # Calculate overall scores
        overall_quality_score = self._calculate_overall_quality_score(
            timestamp_result.integrity_score,
            semantic_result.preservation_score,
            correction_impact
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            timestamp_result, semantic_result, correction_impact, validation_issues
        )
        
        # Determine if validation passed
        passed_validation = self._determine_validation_status(validation_issues, overall_quality_score)
        
        return QualityReport(
            timestamp_integrity_score=timestamp_result.integrity_score,
            semantic_preservation_score=semantic_result.preservation_score,
            correction_impact_score=correction_impact,
            overall_quality_score=overall_quality_score,
            validation_issues=validation_issues,
            processing_metrics=processing_metrics,
            recommendations=recommendations,
            passed_validation=passed_validation
        )
    
    def validate_timestamp_integrity(
        self, 
        original_segments: List[SRTSegment], 
        processed_segments: List[SRTSegment]
    ) -> TimestampValidationResult:
        """
        Validate that timestamp integrity is maintained after processing.
        
        Args:
            original_segments: Original SRT segments
            processed_segments: Processed SRT segments
            
        Returns:
            TimestampValidationResult with detailed validation results
        """
        issues = []
        
        # Check segment count consistency
        if len(original_segments) != len(processed_segments):
            issues.append(ValidationIssue(
                issue_type="segment_count_mismatch",
                severity=ValidationResult.FAILED,
                description=f"Segment count changed from {len(original_segments)} to {len(processed_segments)}",
                confidence_score=1.0
            ))
        
        # Validate individual segments
        problematic_segments = 0
        max_segments = min(len(original_segments), len(processed_segments))
        
        for i in range(max_segments):
            original = original_segments[i]
            processed = processed_segments[i]
            
            segment_issues = self._validate_segment_timestamps(original, processed, i)
            if segment_issues:
                issues.extend(segment_issues)
                problematic_segments += 1
        
        # Validate segment sequence integrity
        sequence_issues = self._validate_segment_sequence(processed_segments)
        issues.extend(sequence_issues)
        
        # Calculate integrity score
        integrity_score = self._calculate_timestamp_integrity_score(issues, max_segments)
        
        return TimestampValidationResult(
            is_valid=len([i for i in issues if i.severity == ValidationResult.FAILED]) == 0,
            issues=issues,
            integrity_score=integrity_score,
            total_segments=max_segments,
            problematic_segments=problematic_segments
        )
    
    def validate_semantic_preservation(
        self, 
        original_segments: List[SRTSegment], 
        processed_segments: List[SRTSegment]
    ) -> SemanticValidationResult:
        """
        Validate that semantic meaning is preserved after processing.
        
        Args:
            original_segments: Original SRT segments
            processed_segments: Processed SRT segments
            
        Returns:
            SemanticValidationResult with semantic drift analysis
        """
        drift_segments = []
        drift_scores = []
        
        max_segments = min(len(original_segments), len(processed_segments))
        
        for i in range(max_segments):
            original_text = original_segments[i].text
            processed_text = processed_segments[i].text
            
            drift_score = self._calculate_semantic_drift(original_text, processed_text)
            drift_scores.append(drift_score)
            
            if drift_score > self.max_semantic_drift:
                drift_segments.append((i, drift_score))
        
        # Calculate statistics
        average_drift = statistics.mean(drift_scores) if drift_scores else 0.0
        max_drift = max(drift_scores) if drift_scores else 0.0
        preservation_score = max(0.0, 1.0 - average_drift)
        acceptable_drift = max_drift <= self.max_semantic_drift
        
        return SemanticValidationResult(
            preservation_score=preservation_score,
            drift_segments=drift_segments,
            average_drift=average_drift,
            max_drift=max_drift,
            acceptable_drift=acceptable_drift
        )
    
    def assess_correction_impact(
        self, 
        original_segments: List[SRTSegment], 
        processed_segments: List[SRTSegment], 
        corrections_applied: List[str]
    ) -> float:
        """
        Assess the impact and effectiveness of applied corrections.
        
        Args:
            original_segments: Original SRT segments
            processed_segments: Processed SRT segments
            corrections_applied: List of correction types applied
            
        Returns:
            Correction impact score (0.0 = poor, 1.0 = excellent)
        """
        if not corrections_applied:
            return 1.0  # No corrections applied, no negative impact
        
        impact_factors = []
        
        # Calculate text length changes
        length_changes = []
        for i in range(min(len(original_segments), len(processed_segments))):
            original_len = len(original_segments[i].text)
            processed_len = len(processed_segments[i].text)
            if original_len > 0:
                length_change = abs(processed_len - original_len) / original_len
                length_changes.append(length_change)
        
        # Length change impact (smaller changes are better)
        avg_length_change = statistics.mean(length_changes) if length_changes else 0.0
        length_impact = max(0.0, 1.0 - avg_length_change)
        impact_factors.append(length_impact)
        
        # Correction type impact assessment
        correction_weights = {
            'removed_filler_words': 0.9,  # Generally good
            'converted_numbers': 0.95,    # Usually beneficial
            'standardized_punctuation': 0.9,
            'fixed_capitalization': 0.85,
            'handled_conversational_nuances': 0.8,  # More complex, potentially riskier
        }
        
        correction_impact = 1.0
        for correction_type in corrections_applied:
            weight = correction_weights.get(correction_type, 0.7)  # Default for unknown corrections
            correction_impact *= weight
        
        impact_factors.append(correction_impact)
        
        # Calculate overall impact score
        overall_impact = statistics.mean(impact_factors)
        
        return overall_impact
    
    def check_segment_boundary_preservation(self, segments: List[SRTSegment]) -> List[ValidationIssue]:
        """
        Check that segment boundaries are properly preserved.
        
        Args:
            segments: SRT segments to validate
            
        Returns:
            List of validation issues found
        """
        issues = []
        
        for i, segment in enumerate(segments):
            # Check for valid timestamp format
            if segment.start_time < 0 or segment.end_time < 0:
                issues.append(ValidationIssue(
                    issue_type="negative_timestamp",
                    severity=ValidationResult.FAILED,
                    description=f"Segment {i} has negative timestamp",
                    segment_index=i,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    confidence_score=1.0
                ))
            
            # Check for valid duration
            if segment.end_time <= segment.start_time:
                issues.append(ValidationIssue(
                    issue_type="invalid_duration",
                    severity=ValidationResult.FAILED,
                    description=f"Segment {i} has invalid duration",
                    segment_index=i,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    confidence_score=1.0
                ))
            
            # Check for reasonable segment length
            duration = segment.end_time - segment.start_time
            if duration > 30.0:  # Segments longer than 30 seconds are suspicious
                issues.append(ValidationIssue(
                    issue_type="unusually_long_segment",
                    severity=ValidationResult.WARNING,
                    description=f"Segment {i} is unusually long ({duration:.1f}s)",
                    segment_index=i,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    confidence_score=0.8
                ))
            
            # Check for overlap with next segment
            if i < len(segments) - 1:
                next_segment = segments[i + 1]
                if segment.end_time > next_segment.start_time:
                    overlap = segment.end_time - next_segment.start_time
                    issues.append(ValidationIssue(
                        issue_type="segment_overlap",
                        severity=ValidationResult.WARNING if overlap < 1.0 else ValidationResult.FAILED,
                        description=f"Segment {i} overlaps with segment {i+1} by {overlap:.3f}s",
                        segment_index=i,
                        start_time=segment.start_time,
                        end_time=segment.end_time,
                        confidence_score=1.0
                    ))
        
        return issues
    
    def _validate_segment_timestamps(self, original: SRTSegment, processed: SRTSegment, index: int) -> List[ValidationIssue]:
        """Validate timestamps for a single segment."""
        issues = []
        
        # Check start time preservation
        start_deviation = abs(processed.start_time - original.start_time)
        if start_deviation > self.max_timestamp_deviation:
            issues.append(ValidationIssue(
                issue_type="start_time_drift",
                severity=ValidationResult.FAILED if start_deviation > 0.1 else ValidationResult.WARNING,
                description=f"Start time changed by {start_deviation:.3f}s",
                segment_index=index,
                start_time=original.start_time,
                end_time=original.end_time,
                confidence_score=1.0
            ))
        
        # Check end time preservation
        end_deviation = abs(processed.end_time - original.end_time)
        if end_deviation > self.max_timestamp_deviation:
            issues.append(ValidationIssue(
                issue_type="end_time_drift",
                severity=ValidationResult.FAILED if end_deviation > 0.1 else ValidationResult.WARNING,
                description=f"End time changed by {end_deviation:.3f}s",
                segment_index=index,
                start_time=original.start_time,
                end_time=original.end_time,
                confidence_score=1.0
            ))
        
        return issues
    
    def _validate_segment_sequence(self, segments: List[SRTSegment]) -> List[ValidationIssue]:
        """Validate the sequence integrity of segments."""
        issues = []
        
        for i in range(len(segments) - 1):
            current = segments[i]
            next_segment = segments[i + 1]
            
            # Check chronological order
            if current.start_time >= next_segment.start_time:
                issues.append(ValidationIssue(
                    issue_type="chronological_disorder",
                    severity=ValidationResult.FAILED,
                    description=f"Segments {i} and {i+1} are not in chronological order",
                    segment_index=i,
                    start_time=current.start_time,
                    end_time=current.end_time,
                    confidence_score=1.0
                ))
            
            # Check for unreasonable gaps
            gap = next_segment.start_time - current.end_time
            if gap > 10.0:  # Gaps longer than 10 seconds might be suspicious
                issues.append(ValidationIssue(
                    issue_type="large_gap",
                    severity=ValidationResult.WARNING,
                    description=f"Large gap ({gap:.1f}s) between segments {i} and {i+1}",
                    segment_index=i,
                    start_time=current.start_time,
                    end_time=current.end_time,
                    confidence_score=0.6
                ))
        
        return issues
    
    def _calculate_semantic_drift(self, original_text: str, processed_text: str) -> float:
        """Calculate semantic drift between original and processed text."""
        if not original_text.strip():
            return 0.0 if not processed_text.strip() else 1.0
        
        # Tokenize texts
        original_words = set(re.findall(r'\b\w+\b', original_text.lower()))
        processed_words = set(re.findall(r'\b\w+\b', processed_text.lower()))
        
        if not original_words:
            return 0.0 if not processed_words else 1.0
        
        # Calculate Jaccard similarity
        intersection = original_words.intersection(processed_words)
        union = original_words.union(processed_words)
        jaccard_similarity = len(intersection) / len(union) if union else 1.0
        
        # Calculate length-based similarity
        len_original = len(original_text)
        len_processed = len(processed_text)
        length_ratio = min(len_original, len_processed) / max(len_original, len_processed) if max(len_original, len_processed) > 0 else 1.0
        
        # Combine metrics (drift = 1 - similarity)
        combined_similarity = (jaccard_similarity * 0.7) + (length_ratio * 0.3)
        semantic_drift = 1.0 - combined_similarity
        
        return min(semantic_drift, 1.0)
    
    def _calculate_timestamp_integrity_score(self, issues: List[ValidationIssue], total_segments: int) -> float:
        """Calculate timestamp integrity score based on issues found."""
        if total_segments == 0:
            return 1.0
        
        # Weight different issue types
        issue_weights = {
            'start_time_drift': 0.3,
            'end_time_drift': 0.3,
            'segment_count_mismatch': 1.0,
            'chronological_disorder': 0.8,
            'segment_overlap': 0.5,
            'negative_timestamp': 1.0,
            'invalid_duration': 1.0
        }
        
        penalty = 0.0
        for issue in issues:
            weight = issue_weights.get(issue.issue_type, 0.5)
            severity_multiplier = {
                ValidationResult.WARNING: 0.5,
                ValidationResult.FAILED: 1.0
            }.get(issue.severity, 1.0)
            
            penalty += weight * severity_multiplier
        
        # Normalize penalty by total segments
        normalized_penalty = penalty / max(total_segments, 1)
        
        # Calculate score (1.0 = perfect, 0.0 = completely broken)
        integrity_score = max(0.0, 1.0 - normalized_penalty)
        
        return integrity_score
    
    def _calculate_overall_quality_score(
        self, 
        timestamp_score: float, 
        semantic_score: float, 
        correction_impact: float
    ) -> float:
        """Calculate overall quality score from component scores."""
        # Weight the different components
        weights = {
            'timestamp': 0.4,
            'semantic': 0.4,
            'correction': 0.2
        }
        
        overall_score = (
            timestamp_score * weights['timestamp'] +
            semantic_score * weights['semantic'] +
            correction_impact * weights['correction']
        )
        
        return overall_score
    
    def _convert_semantic_issues(self, semantic_result: SemanticValidationResult) -> List[ValidationIssue]:
        """Convert semantic validation results to validation issues."""
        issues = []
        
        for segment_index, drift_score in semantic_result.drift_segments:
            severity = ValidationResult.WARNING if drift_score < self.max_semantic_drift * 1.5 else ValidationResult.FAILED
            
            issues.append(ValidationIssue(
                issue_type="semantic_drift",
                severity=severity,
                description=f"High semantic drift ({drift_score:.3f}) in segment {segment_index}",
                segment_index=segment_index,
                confidence_score=min(drift_score, 1.0)
            ))
        
        return issues
    
    def _generate_recommendations(
        self, 
        timestamp_result: TimestampValidationResult,
        semantic_result: SemanticValidationResult,
        correction_impact: float,
        issues: List[ValidationIssue]
    ) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Timestamp-based recommendations
        if timestamp_result.integrity_score < 0.9:
            recommendations.append("Review timestamp preservation - some segments show timing drift")
        
        if timestamp_result.problematic_segments > 0:
            recommendations.append(f"Investigate {timestamp_result.problematic_segments} segments with timestamp issues")
        
        # Semantic preservation recommendations
        if semantic_result.average_drift > self.max_semantic_drift:
            recommendations.append("Consider reducing processing aggressiveness to preserve semantic meaning")
        
        if len(semantic_result.drift_segments) > 0:
            recommendations.append(f"Review {len(semantic_result.drift_segments)} segments with high semantic drift")
        
        # Correction impact recommendations
        if correction_impact < 0.8:
            recommendations.append("Review correction strategies - current approach may be too aggressive")
        
        # Issue-specific recommendations
        failed_issues = [i for i in issues if i.severity == ValidationResult.FAILED]
        if failed_issues:
            recommendations.append(f"Address {len(failed_issues)} critical validation failures before proceeding")
        
        return recommendations
    
    def _determine_validation_status(self, issues: List[ValidationIssue], overall_score: float) -> bool:
        """Determine if validation passed based on issues and scores."""
        # Check for critical failures
        critical_failures = [i for i in issues if i.severity == ValidationResult.FAILED]
        if critical_failures:
            return False
        
        # Check overall quality score
        if overall_score < self.min_quality_score:
            return False
        
        return True
    
    def _setup_validation_thresholds(self):
        """Setup validation thresholds based on validation level."""
        if self.validation_level == ValidationLevel.STRICT:
            self.max_semantic_drift = 0.2
            self.max_timestamp_deviation = 0.0001
            self.min_quality_score = 0.9
        elif self.validation_level == ValidationLevel.LENIENT:
            self.max_semantic_drift = 0.5
            self.max_timestamp_deviation = 0.01
            self.min_quality_score = 0.6
        # MODERATE is the default set in __init__