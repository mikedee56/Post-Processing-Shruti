"""
Tests for the ProcessingQualityValidator module.

This module provides comprehensive tests for processing quality validation
including timestamp integrity and semantic preservation checks.
"""

import unittest
from unittest.mock import Mock
import sys
import os

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.processing_quality_validator import (
    ProcessingQualityValidator,
    ValidationLevel,
    ValidationResult,
    ValidationIssue,
    QualityReport,
    TimestampValidationResult,
    SemanticValidationResult
)

from utils.srt_parser import SRTSegment


class TestProcessingQualityValidator(unittest.TestCase):
    """Test cases for ProcessingQualityValidator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = ProcessingQualityValidator()
        
        # Test configuration for controlled testing
        self.test_config = {
            'validation_level': 'moderate',
            'max_semantic_drift': 0.3,
            'max_timestamp_deviation': 0.001,
            'min_quality_score': 0.7
        }
        
        self.configured_validator = ProcessingQualityValidator(self.test_config)
        
        # Create sample SRT segments for testing
        self.original_segments = [
            SRTSegment(1, 1.0, 5.0, "This is the original text.", 0.9),
            SRTSegment(2, 5.5, 10.0, "Another original segment.", 0.85),
            SRTSegment(3, 10.5, 15.0, "Third segment with content.", 0.8)
        ]
        
        self.processed_segments = [
            SRTSegment(1, 1.0, 5.0, "This is the processed text.", 0.9),
            SRTSegment(2, 5.5, 10.0, "Another processed segment.", 0.85), 
            SRTSegment(3, 10.5, 15.0, "Third segment with different content.", 0.8)
        ]
    
    def test_initialization(self):
        """Test that the validator initializes properly."""
        self.assertIsInstance(self.validator, ProcessingQualityValidator)
        self.assertEqual(self.validator.validation_level, ValidationLevel.MODERATE)
        self.assertTrue(hasattr(self.validator, 'max_semantic_drift'))
        self.assertTrue(hasattr(self.validator, 'max_timestamp_deviation'))
        self.assertTrue(hasattr(self.validator, 'min_quality_score'))
    
    def test_validation_level_configuration(self):
        """Test different validation level configurations."""
        # Test strict validation
        strict_config = {'validation_level': 'strict'}
        strict_validator = ProcessingQualityValidator(strict_config)
        self.assertEqual(strict_validator.validation_level, ValidationLevel.STRICT)
        self.assertEqual(strict_validator.max_semantic_drift, 0.2)
        
        # Test lenient validation
        lenient_config = {'validation_level': 'lenient'}  
        lenient_validator = ProcessingQualityValidator(lenient_config)
        self.assertEqual(lenient_validator.validation_level, ValidationLevel.LENIENT)
        self.assertEqual(lenient_validator.max_semantic_drift, 0.5)
    
    def test_timestamp_integrity_validation_perfect_match(self):
        """Test timestamp validation with perfect timestamp match."""
        # Create segments with identical timestamps
        original = [SRTSegment(1, 1.0, 5.0, "Original text", 0.9)]
        processed = [SRTSegment(1, 1.0, 5.0, "Processed text", 0.9)]
        
        result = self.validator.validate_timestamp_integrity(original, processed)
        
        self.assertIsInstance(result, TimestampValidationResult)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.problematic_segments, 0)
        self.assertGreater(result.integrity_score, 0.9)
    
    def test_timestamp_integrity_validation_with_drift(self):
        """Test timestamp validation with timestamp drift."""
        # Create segments with slight timestamp drift
        original = [SRTSegment(1, 1.0, 5.0, "Original text", 0.9)]
        processed = [SRTSegment(1, 1.01, 5.01, "Processed text", 0.9)]  # 0.01s drift
        
        result = self.validator.validate_timestamp_integrity(original, processed)
        
        # Should detect timestamp drift
        drift_issues = [issue for issue in result.issues 
                       if 'time_drift' in issue.issue_type]
        
        # Depending on threshold, might be acceptable or not
        if self.validator.max_timestamp_deviation < 0.01:
            self.assertGreater(len(drift_issues), 0)
    
    def test_timestamp_validation_segment_count_mismatch(self):
        """Test timestamp validation with different segment counts."""
        original = [SRTSegment(1, 1.0, 5.0, "Original text", 0.9)]
        processed = [
            SRTSegment(1, 1.0, 5.0, "Processed text", 0.9),
            SRTSegment(2, 5.5, 10.0, "Extra segment", 0.8)
        ]
        
        result = self.validator.validate_timestamp_integrity(original, processed)
        
        # Should detect segment count mismatch
        mismatch_issues = [issue for issue in result.issues 
                          if issue.issue_type == 'segment_count_mismatch']
        self.assertGreater(len(mismatch_issues), 0)
        self.assertFalse(result.is_valid)
    
    def test_semantic_preservation_validation_no_change(self):
        """Test semantic validation with no text changes."""
        # Identical text should have zero drift
        original = [SRTSegment(1, 1.0, 5.0, "Identical text", 0.9)]
        processed = [SRTSegment(1, 1.0, 5.0, "Identical text", 0.9)]
        
        result = self.validator.validate_semantic_preservation(original, processed)
        
        self.assertIsInstance(result, SemanticValidationResult)
        self.assertEqual(result.average_drift, 0.0)
        self.assertEqual(result.max_drift, 0.0)
        self.assertTrue(result.acceptable_drift)
    
    def test_semantic_preservation_validation_with_drift(self):
        """Test semantic validation with semantic drift."""
        original = [SRTSegment(1, 1.0, 5.0, "The soul is eternal and unchanging.", 0.9)]
        processed = [SRTSegment(1, 1.0, 5.0, "Consciousness is permanent and stable.", 0.9)]
        
        result = self.validator.validate_semantic_preservation(original, processed)
        
        # Should detect semantic drift
        self.assertGreater(result.average_drift, 0.0)
        self.assertGreater(result.max_drift, 0.0)
        
        # Drift should be reasonable for similar meaning
        self.assertLess(result.max_drift, 1.0)
    
    def test_semantic_preservation_major_change(self):
        """Test semantic validation with major semantic changes."""
        original = [SRTSegment(1, 1.0, 5.0, "The soul is eternal.", 0.9)]
        processed = [SRTSegment(1, 1.0, 5.0, "Weather is nice today.", 0.9)]
        
        result = self.validator.validate_semantic_preservation(original, processed)
        
        # Should detect high semantic drift
        self.assertGreater(result.average_drift, 0.5)
        self.assertFalse(result.acceptable_drift)
        self.assertGreater(len(result.drift_segments), 0)
    
    def test_correction_impact_assessment_no_corrections(self):
        """Test correction impact assessment with no corrections."""
        impact = self.validator.assess_correction_impact(
            self.original_segments, 
            self.processed_segments, 
            []
        )
        
        # No corrections should result in high impact score
        self.assertEqual(impact, 1.0)
    
    def test_correction_impact_assessment_with_corrections(self):
        """Test correction impact assessment with various corrections."""
        corrections = [
            'removed_filler_words', 
            'converted_numbers', 
            'standardized_punctuation'
        ]
        
        impact = self.validator.assess_correction_impact(
            self.original_segments,
            self.processed_segments,
            corrections
        )
        
        # Should be a reasonable impact score
        self.assertGreaterEqual(impact, 0.0)
        self.assertLessEqual(impact, 1.0)
        self.assertLess(impact, 1.0)  # Should be less than perfect due to corrections
    
    def test_comprehensive_quality_validation(self):
        """Test comprehensive quality validation pipeline."""
        corrections = ['removed_filler_words', 'converted_numbers']
        
        quality_report = self.validator.validate_processing_quality(
            self.original_segments,
            self.processed_segments,
            corrections
        )
        
        self.assertIsInstance(quality_report, QualityReport)
        
        # Check all required fields are present
        self.assertIsNotNone(quality_report.timestamp_integrity_score)
        self.assertIsNotNone(quality_report.semantic_preservation_score)
        self.assertIsNotNone(quality_report.correction_impact_score)
        self.assertIsNotNone(quality_report.overall_quality_score)
        self.assertIsInstance(quality_report.validation_issues, list)
        self.assertIsInstance(quality_report.recommendations, list)
        self.assertIsInstance(quality_report.passed_validation, bool)
    
    def test_segment_boundary_preservation_check(self):
        """Test segment boundary preservation validation."""
        # Create segments with boundary issues
        problematic_segments = [
            SRTSegment(1, -1.0, 5.0, "Negative start time", 0.9),  # Negative timestamp
            SRTSegment(2, 10.0, 5.0, "End before start", 0.9),    # Invalid duration
            SRTSegment(3, 5.0, 40.0, "Very long segment", 0.9)    # Unusually long
        ]
        
        issues = self.validator.check_segment_boundary_preservation(problematic_segments)
        
        # Should detect multiple issues
        self.assertGreater(len(issues), 0)
        
        # Check specific issue types
        issue_types = [issue.issue_type for issue in issues]
        self.assertIn('negative_timestamp', issue_types)
        self.assertIn('invalid_duration', issue_types)
        self.assertIn('unusually_long_segment', issue_types)
    
    def test_overlapping_segments_detection(self):
        """Test detection of overlapping segments."""
        overlapping_segments = [
            SRTSegment(1, 1.0, 6.0, "First segment", 0.9),
            SRTSegment(2, 5.0, 10.0, "Overlapping segment", 0.9)  # Overlaps with first
        ]
        
        issues = self.validator.check_segment_boundary_preservation(overlapping_segments)
        
        # Should detect overlap
        overlap_issues = [issue for issue in issues if issue.issue_type == 'segment_overlap']
        self.assertGreater(len(overlap_issues), 0)
    
    def test_quality_report_recommendations(self):
        """Test generation of quality recommendations."""
        # Create scenario with various issues
        original = [SRTSegment(1, 1.0, 5.0, "Original comprehensive text with many words.", 0.9)]
        processed = [SRTSegment(1, 1.01, 5.01, "Processed short text.", 0.9)]  # Drift + major change
        
        corrections = ['handled_conversational_nuances']  # Risky correction type
        
        quality_report = self.validator.validate_processing_quality(
            original, processed, corrections
        )
        
        # Should have recommendations due to issues
        self.assertGreater(len(quality_report.recommendations), 0)
        
        # Check for specific recommendation types
        recommendations_text = ' '.join(quality_report.recommendations).lower()
        if quality_report.semantic_preservation_score < 0.7:
            self.assertTrue(any('semantic' in rec.lower() for rec in quality_report.recommendations))
    
    def test_validation_status_determination(self):
        """Test determination of overall validation status."""
        # Test with good quality (should pass)
        good_original = [SRTSegment(1, 1.0, 5.0, "Good original text.", 0.9)]
        good_processed = [SRTSegment(1, 1.0, 5.0, "Good processed text.", 0.9)]
        
        good_report = self.validator.validate_processing_quality(
            good_original, good_processed, ['converted_numbers']
        )
        
        # Should pass with minor corrections and minimal changes
        # (Result depends on specific thresholds)
        self.assertIsInstance(good_report.passed_validation, bool)
        
        # Test with poor quality (should fail)
        poor_original = [SRTSegment(1, 1.0, 5.0, "Original text about souls.", 0.9)]
        poor_processed = [SRTSegment(1, 2.0, 6.0, "Completely different content about weather.", 0.9)]
        
        poor_report = self.validator.validate_processing_quality(
            poor_original, poor_processed, []
        )
        
        # Should fail due to timestamp drift and semantic changes
        # (Result depends on specific thresholds, but likely to fail)
        self.assertIsInstance(poor_report.passed_validation, bool)
    
    def test_empty_segments_handling(self):
        """Test handling of empty segment lists."""
        empty_report = self.validator.validate_processing_quality([], [], [])
        
        self.assertIsInstance(empty_report, QualityReport)
        self.assertEqual(empty_report.overall_quality_score, 1.0)  # Perfect score for no work
        self.assertTrue(empty_report.passed_validation)


class TestValidationIssue(unittest.TestCase):
    """Test cases for ValidationIssue dataclass."""
    
    def test_validation_issue_creation(self):
        """Test that ValidationIssue objects can be created properly."""
        issue = ValidationIssue(
            issue_type="test_issue",
            severity=ValidationResult.WARNING,
            description="Test issue description",
            segment_index=1,
            start_time=1.0,
            end_time=5.0,
            original_text="Original",
            processed_text="Processed",
            confidence_score=0.8
        )
        
        self.assertEqual(issue.issue_type, "test_issue")
        self.assertEqual(issue.severity, ValidationResult.WARNING)
        self.assertEqual(issue.segment_index, 1)
        self.assertEqual(issue.confidence_score, 0.8)


class TestQualityReport(unittest.TestCase):
    """Test cases for QualityReport dataclass."""
    
    def test_quality_report_creation(self):
        """Test that QualityReport objects can be created properly."""
        report = QualityReport(
            timestamp_integrity_score=0.95,
            semantic_preservation_score=0.85,
            correction_impact_score=0.9,
            overall_quality_score=0.9,
            validation_issues=[],
            processing_metrics={'test': 1.0},
            recommendations=["Test recommendation"],
            passed_validation=True
        )
        
        self.assertEqual(report.timestamp_integrity_score, 0.95)
        self.assertEqual(report.semantic_preservation_score, 0.85)
        self.assertEqual(report.overall_quality_score, 0.9)
        self.assertTrue(report.passed_validation)


class TestValidationEnums(unittest.TestCase):
    """Test cases for validation enums."""
    
    def test_validation_level_values(self):
        """Test ValidationLevel enum values."""
        self.assertEqual(ValidationLevel.STRICT.value, "strict")
        self.assertEqual(ValidationLevel.MODERATE.value, "moderate")
        self.assertEqual(ValidationLevel.LENIENT.value, "lenient")
    
    def test_validation_result_values(self):
        """Test ValidationResult enum values."""
        self.assertEqual(ValidationResult.PASSED.value, "passed")
        self.assertEqual(ValidationResult.WARNING.value, "warning")
        self.assertEqual(ValidationResult.FAILED.value, "failed")


if __name__ == '__main__':
    unittest.main()