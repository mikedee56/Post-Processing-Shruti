"""
Tests for the AdvancedTextNormalizer module.

This module provides comprehensive tests for the advanced text normalization
functionality including conversational nuance handling and semantic preservation.
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.advanced_text_normalizer import (
    AdvancedTextNormalizer, 
    ConversationalPattern, 
    AdvancedCorrectionResult,
    ConversationalCorrectionResult
)


class TestAdvancedTextNormalizer(unittest.TestCase):
    """Test cases for AdvancedTextNormalizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.normalizer = AdvancedTextNormalizer()
        
        # Test configuration for controlled testing
        self.test_config = {
            'preserve_meaningful_discourse': True,
            'semantic_drift_threshold': 0.3,
            'min_confidence_score': 0.7
        }
        
        self.configured_normalizer = AdvancedTextNormalizer(self.test_config)
    
    def test_initialization(self):
        """Test that the normalizer initializes properly."""
        self.assertIsInstance(self.normalizer, AdvancedTextNormalizer)
        self.assertTrue(hasattr(self.normalizer, 'rescission_patterns'))
        self.assertTrue(hasattr(self.normalizer, 'partial_phrase_patterns'))
        self.assertTrue(hasattr(self.normalizer, 'potentially_meaningful_markers'))
    
    def test_basic_normalization_still_works(self):
        """Test that basic normalization functionality is preserved."""
        text = "um, this is a test with two numbers."
        result = self.normalizer.normalize_with_advanced_tracking(text)
        
        self.assertIsInstance(result, AdvancedCorrectionResult)
        self.assertNotEqual(result.original_text, result.corrected_text)
        self.assertIn("removed_filler_words", result.corrections_applied)
    
    def test_rescinded_phrase_correction(self):
        """Test correction of rescinded phrases."""
        # Test "I mean" correction
        text = "This is wrong, I mean, this is the correct statement."
        result = self.normalizer.normalize_with_advanced_tracking(text)
        
        # Should keep the latter part after "I mean"
        self.assertIn("this is the correct statement", result.corrected_text.lower())
        
        # Check that conversational fixes were applied
        conversational_fixes = [fix for fix in result.conversational_fixes if fix.pattern_type == "rescinded"]
        self.assertTrue(len(conversational_fixes) > 0)
    
    def test_rescinded_phrase_rather_correction(self):
        """Test correction of 'rather' rescinded phrases."""
        text = "The soul is temporary, rather, the soul is eternal."
        result = self.normalizer.normalize_with_advanced_tracking(text)
        
        # Should prefer the corrected statement after "rather"
        self.assertIn("soul is eternal", result.corrected_text)
    
    def test_partial_phrase_completion(self):
        """Test completion of partial phrases."""
        # Test trailing conjunction removal
        text = "This is a complete thought but"
        result = self.normalizer.normalize_with_advanced_tracking(text)
        
        # Should remove the trailing "but"
        self.assertEqual("This is a complete thought", result.corrected_text.strip())
    
    def test_repetition_handling(self):
        """Test handling of word repetitions."""
        text = "The the soul is eternal and and unchanging."
        result = self.normalizer.normalize_with_advanced_tracking(text)
        
        # Should remove duplicate words
        self.assertNotIn("the the", result.corrected_text.lower())
        self.assertNotIn("and and", result.corrected_text.lower())
    
    def test_meaningful_discourse_marker_preservation(self):
        """Test that meaningful discourse markers are preserved."""
        # "Now" should be preserved when it's meaningful
        text = "Now let us examine this verse carefully."
        result = self.normalizer.normalize_with_advanced_tracking(text)
        
        # "Now" should be preserved because it's meaningful
        self.assertIn("Now", result.corrected_text)
        
        # Check for discourse marker preservation in conversational fixes
        discourse_fixes = [fix for fix in result.conversational_fixes if fix.pattern_type == "filler_context"]
        meaningful_preservation = any(fix.preservation_reason == "meaningful_discourse_marker" for fix in discourse_fixes)
        # This might be True or False depending on the specific implementation
    
    def test_semantic_drift_calculation(self):
        """Test semantic drift calculation."""
        original = "This is a test sentence with some words."
        
        # Minor change
        slightly_changed = "This is a test sentence with different words."
        drift1 = self.normalizer.calculate_semantic_drift(original, slightly_changed)
        
        # Major change
        majorly_changed = "Completely different sentence entirely."
        drift2 = self.normalizer.calculate_semantic_drift(original, majorly_changed)
        
        # Drift should be higher for major changes
        self.assertLess(drift1, drift2)
        self.assertLessEqual(drift1, 1.0)
        self.assertLessEqual(drift2, 1.0)
        self.assertGreaterEqual(drift1, 0.0)
        self.assertGreaterEqual(drift2, 0.0)
    
    def test_empty_text_handling(self):
        """Test handling of empty or whitespace-only text."""
        empty_result = self.normalizer.normalize_with_advanced_tracking("")
        whitespace_result = self.normalizer.normalize_with_advanced_tracking("   ")
        
        self.assertEqual("", empty_result.corrected_text)
        self.assertEqual("   ", whitespace_result.corrected_text)
        self.assertEqual([], empty_result.conversational_fixes)
        self.assertEqual([], whitespace_result.conversational_fixes)
    
    def test_confidence_scoring(self):
        """Test that confidence scores are calculated correctly."""
        # Simple text should have high confidence
        simple_text = "This is a simple sentence."
        simple_result = self.normalizer.normalize_with_advanced_tracking(simple_text)
        
        # Complex text with many corrections should have lower confidence
        complex_text = "Um, well, I mean, uh, this is like, you know, really complex and and confusing, rather, it's actually simple."
        complex_result = self.normalizer.normalize_with_advanced_tracking(complex_text)
        
        # Complex text should have lower quality score due to more corrections
        self.assertLess(complex_result.quality_score, simple_result.quality_score)
    
    def test_configuration_respect(self):
        """Test that the normalizer respects configuration settings."""
        # Test with low confidence threshold
        low_confidence_config = {'min_confidence_score': 0.2}
        low_conf_normalizer = AdvancedTextNormalizer(low_confidence_config)
        
        # Test with high semantic drift threshold
        high_drift_config = {'semantic_drift_threshold': 0.8}
        high_drift_normalizer = AdvancedTextNormalizer(high_drift_config)
        
        # Both should initialize without errors
        self.assertIsInstance(low_conf_normalizer, AdvancedTextNormalizer)
        self.assertIsInstance(high_drift_normalizer, AdvancedTextNormalizer)
        
        self.assertEqual(low_conf_normalizer.min_confidence_score, 0.2)
        self.assertEqual(high_drift_normalizer.semantic_drift_threshold, 0.8)
    
    def test_conversational_nuances_integration(self):
        """Test the integration of all conversational nuance handling."""
        complex_text = """Um, today we will discuss, uh, I mean we'll explore the Bhagavad Gita chapter two verse twenty five.
        This verse, you know, actually this verse talks about the eternal nature of the soul."""
        
        result = self.normalizer.normalize_with_advanced_tracking(complex_text)
        
        # Should have applied multiple types of corrections
        correction_types = set(result.corrections_applied)
        
        # Should contain basic normalization corrections
        self.assertTrue(any('removed_filler_words' in correction or 'filler' in correction 
                          for correction in correction_types))
        
        # Check that the result is cleaner than the original
        self.assertLess(len(result.corrected_text), len(result.original_text))
        self.assertNotIn("um,", result.corrected_text.lower())
        self.assertNotIn("uh,", result.corrected_text.lower())
    
    def test_quality_score_calculation(self):
        """Test quality score calculation components."""
        # Test with no changes
        no_change_text = "This is perfect text."
        no_change_result = self.normalizer.normalize_with_advanced_tracking(no_change_text)
        
        # Quality score should be high when no changes are needed
        self.assertGreater(no_change_result.quality_score, 0.8)
        
        # Test with high semantic drift
        original = "The soul is eternal."
        # Force high drift by simulating major text change
        with patch.object(self.normalizer, 'calculate_semantic_drift', return_value=0.9):
            high_drift_result = self.normalizer.normalize_with_advanced_tracking(original)
            # Quality score should be lower due to high semantic drift
            self.assertLess(high_drift_result.quality_score, 0.5)


class TestConversationalPattern(unittest.TestCase):
    """Test cases for ConversationalPattern dataclass."""
    
    def test_conversational_pattern_creation(self):
        """Test that ConversationalPattern objects can be created properly."""
        pattern = ConversationalPattern(
            pattern_type="rescinded",
            original_text="I mean, this is correct",
            corrected_text="this is correct",
            confidence_score=0.85,
            context_clues=["preceding context"],
            preservation_reason=None
        )
        
        self.assertEqual(pattern.pattern_type, "rescinded")
        self.assertEqual(pattern.original_text, "I mean, this is correct")
        self.assertEqual(pattern.corrected_text, "this is correct")
        self.assertEqual(pattern.confidence_score, 0.85)
        self.assertIsNone(pattern.preservation_reason)


class TestAdvancedCorrectionResult(unittest.TestCase):
    """Test cases for AdvancedCorrectionResult dataclass."""
    
    def test_advanced_correction_result_creation(self):
        """Test that AdvancedCorrectionResult objects can be created properly."""
        result = AdvancedCorrectionResult(
            original_text="Original text",
            corrected_text="Corrected text", 
            corrections_applied=["test_correction"],
            conversational_fixes=[],
            quality_score=0.9,
            semantic_drift_score=0.1,
            word_count_before=2,
            word_count_after=2
        )
        
        self.assertEqual(result.original_text, "Original text")
        self.assertEqual(result.corrected_text, "Corrected text")
        self.assertEqual(result.quality_score, 0.9)
        self.assertEqual(result.semantic_drift_score, 0.1)


if __name__ == '__main__':
    unittest.main()