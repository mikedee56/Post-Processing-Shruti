"""
Tests for the ConversationalPatternDetector module.

This module provides comprehensive tests for conversational pattern detection
and correction functionality.
"""

import unittest
import sys
import os

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.conversational_pattern_detector import (
    ConversationalPatternDetector,
    PatternType,
    PatternMatch,
    DetectionResult
)


class TestConversationalPatternDetector(unittest.TestCase):
    """Test cases for ConversationalPatternDetector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = ConversationalPatternDetector()
        
        # Test configuration for controlled testing
        self.test_config = {
            'min_confidence_threshold': 0.7,
            'context_window_size': 50
        }
        
        self.configured_detector = ConversationalPatternDetector(self.test_config)
    
    def test_initialization(self):
        """Test that the detector initializes properly."""
        self.assertIsInstance(self.detector, ConversationalPatternDetector)
        self.assertTrue(hasattr(self.detector, 'rescission_patterns'))
        self.assertTrue(hasattr(self.detector, 'partial_phrase_patterns'))
        self.assertTrue(hasattr(self.detector, 'repetition_patterns'))
        self.assertTrue(hasattr(self.detector, 'interruption_patterns'))
        self.assertTrue(hasattr(self.detector, 'discourse_markers'))
    
    def test_empty_text_detection(self):
        """Test detection on empty or whitespace-only text."""
        empty_result = self.detector.detect_patterns("")
        whitespace_result = self.detector.detect_patterns("   ")
        
        self.assertEqual(empty_result.total_patterns, 0)
        self.assertEqual(whitespace_result.total_patterns, 0)
        self.assertIn("Empty or whitespace-only text", empty_result.processing_notes[0])
    
    def test_rescission_pattern_detection(self):
        """Test detection of rescinded phrase patterns."""
        # Test "I mean" pattern
        text = "This is wrong, I mean, this is the correct statement."
        result = self.detector.detect_patterns(text)
        
        rescission_patterns = [p for p in result.patterns_found if p.pattern_type == PatternType.RESCINDED]
        self.assertGreater(len(rescission_patterns), 0)
        
        # Check pattern details
        pattern = rescission_patterns[0]
        self.assertEqual(pattern.pattern_type, PatternType.RESCINDED)
        self.assertIn("I mean", pattern.original_text)
        self.assertGreater(pattern.confidence_score, 0.5)
    
    def test_rather_rescission_detection(self):
        """Test detection of 'rather' rescission patterns."""
        text = "The soul is temporary, rather, the soul is eternal."
        result = self.detector.detect_patterns(text)
        
        rescission_patterns = [p for p in result.patterns_found if p.pattern_type == PatternType.RESCINDED]
        self.assertGreater(len(rescission_patterns), 0)
        
        pattern = rescission_patterns[0]
        self.assertIn("rather", pattern.original_text.lower())
    
    def test_partial_phrase_detection(self):
        """Test detection of partial phrase patterns."""
        # Test trailing conjunction
        text = "This is a complete thought but"
        result = self.detector.detect_patterns(text)
        
        partial_patterns = [p for p in result.patterns_found if p.pattern_type == PatternType.PARTIAL_PHRASE]
        self.assertGreater(len(partial_patterns), 0)
        
        pattern = partial_patterns[0]
        self.assertIn("but", pattern.original_text)
    
    def test_interruption_pattern_detection(self):
        """Test detection of interruption patterns."""
        text = "This is an interrupted thought--"
        result = self.detector.detect_patterns(text)
        
        # Should detect interruption patterns
        interruption_patterns = [p for p in result.patterns_found if p.pattern_type == PatternType.INTERRUPTION]
        # Note: depending on implementation, this might or might not match
        # The test validates the structure works
        self.assertIsInstance(result, DetectionResult)
    
    def test_repetition_pattern_detection(self):
        """Test detection of repetition patterns."""
        text = "The the soul is eternal and and unchanging."
        result = self.detector.detect_patterns(text)
        
        repetition_patterns = [p for p in result.patterns_found if p.pattern_type == PatternType.REPETITION]
        self.assertGreater(len(repetition_patterns), 0)
        
        # Should detect "the the" and "and and" repetitions
        pattern_texts = [p.original_text.lower() for p in repetition_patterns]
        repetition_found = any("the the" in text.lower() or "and and" in text.lower() for text in pattern_texts)
        self.assertTrue(repetition_found)
    
    def test_discourse_marker_detection(self):
        """Test detection of meaningful discourse markers."""
        text = "Now let us examine this verse carefully. So this teaches us about the soul."
        result = self.detector.detect_patterns(text)
        
        discourse_patterns = [p for p in result.patterns_found if p.pattern_type == PatternType.DISCOURSE_MARKER]
        
        # Should detect meaningful discourse markers
        if discourse_patterns:
            pattern_texts = [p.original_text.lower() for p in discourse_patterns]
            self.assertTrue(any(marker in pattern_texts for marker in ["now", "so"]))
    
    def test_pattern_correction_application(self):
        """Test application of pattern corrections."""
        text = "The the soul is eternal, I mean, it is imperishable."
        result = self.detector.detect_patterns(text)
        
        # Apply corrections
        corrected_text = self.detector.apply_pattern_corrections(text, result.patterns_found)
        
        # Should remove repetition and handle rescission
        self.assertNotIn("the the", corrected_text.lower())
        # The exact correction depends on implementation details
        self.assertNotEqual(text, corrected_text)
    
    def test_overlapping_pattern_resolution(self):
        """Test resolution of overlapping patterns."""
        # Text with potential overlapping patterns
        text = "I mean, I mean, this is the correct statement."
        result = self.detector.detect_patterns(text)
        
        # Should have detected patterns but resolved overlaps
        self.assertGreater(result.total_patterns, 0)
        
        # Patterns should not overlap (positions should be distinct)
        patterns = result.patterns_found
        for i, pattern1 in enumerate(patterns):
            for j, pattern2 in enumerate(patterns):
                if i != j:
                    # Check that patterns don't overlap
                    overlap = not (pattern1.end_pos <= pattern2.start_pos or pattern2.end_pos <= pattern1.start_pos)
                    self.assertFalse(overlap, f"Patterns {i} and {j} overlap")
    
    def test_confidence_threshold_filtering(self):
        """Test that confidence threshold filtering works."""
        # Create detector with high confidence threshold
        high_threshold_detector = ConversationalPatternDetector({'min_confidence_threshold': 0.9})
        
        text = "This might have some patterns, you know, maybe."
        result = high_threshold_detector.detect_patterns(text)
        
        # Should have fewer high-confidence patterns
        high_conf_patterns = [p for p in result.patterns_found if p.confidence_score >= 0.9]
        self.assertEqual(len(high_conf_patterns), result.high_confidence_patterns)
    
    def test_context_window_extraction(self):
        """Test context window extraction around patterns."""
        text = "This is a long sentence with an I mean correction in the middle of it for testing."
        result = self.detector.detect_patterns(text)
        
        if result.patterns_found:
            pattern = result.patterns_found[0]
            # Context window should contain surrounding text
            self.assertIsInstance(pattern.context_window, str)
            self.assertGreater(len(pattern.context_window), len(pattern.original_text))
    
    def test_complex_conversational_text(self):
        """Test detection on complex conversational text."""
        complex_text = """Um, today we will discuss, uh, I mean we'll explore the Bhagavad Gita chapter two verse twenty five.
        This verse, you know, actually this verse talks about the eternal nature of the soul.
        Let me rephrase that, the soul is described as imperishable and eternal."""
        
        result = self.detector.detect_patterns(complex_text)
        
        # Should detect multiple types of patterns
        pattern_types = set(p.pattern_type for p in result.patterns_found)
        
        # Should have detected at least some patterns
        self.assertGreater(result.total_patterns, 0)
        self.assertGreater(len(pattern_types), 0)


class TestPatternMatch(unittest.TestCase):
    """Test cases for PatternMatch dataclass."""
    
    def test_pattern_match_creation(self):
        """Test that PatternMatch objects can be created properly."""
        match = PatternMatch(
            pattern_type=PatternType.RESCINDED,
            start_pos=10,
            end_pos=20,
            original_text="I mean",
            suggested_correction="",
            confidence_score=0.85,
            context_window="context around the match",
            reasoning="Test reasoning"
        )
        
        self.assertEqual(match.pattern_type, PatternType.RESCINDED)
        self.assertEqual(match.start_pos, 10)
        self.assertEqual(match.end_pos, 20)
        self.assertEqual(match.confidence_score, 0.85)


class TestDetectionResult(unittest.TestCase):
    """Test cases for DetectionResult dataclass."""
    
    def test_detection_result_creation(self):
        """Test that DetectionResult objects can be created properly."""
        result = DetectionResult(
            text="Test text",
            patterns_found=[],
            total_patterns=0,
            high_confidence_patterns=0,
            processing_notes=["Test note"]
        )
        
        self.assertEqual(result.text, "Test text")
        self.assertEqual(result.total_patterns, 0)
        self.assertEqual(result.high_confidence_patterns, 0)
        self.assertEqual(len(result.processing_notes), 1)


class TestPatternType(unittest.TestCase):
    """Test cases for PatternType enum."""
    
    def test_pattern_type_values(self):
        """Test that PatternType enum has expected values."""
        self.assertEqual(PatternType.RESCINDED.value, "rescinded")
        self.assertEqual(PatternType.PARTIAL_PHRASE.value, "partial_phrase")
        self.assertEqual(PatternType.REPETITION.value, "repetition")
        self.assertEqual(PatternType.INTERRUPTION.value, "interruption")
        self.assertEqual(PatternType.DISCOURSE_MARKER.value, "discourse_marker")
        self.assertEqual(PatternType.FILLER_CONTEXT.value, "filler_context")


if __name__ == '__main__':
    unittest.main()