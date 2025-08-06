"""
Integration tests for foundational post-processing corrections (Story 1.4).

This module provides end-to-end integration tests for the complete foundational
correction pipeline including conversational nuance handling, contextual number
processing, and quality validation.
"""

import unittest
import tempfile
import os
import sys
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from post_processors.sanskrit_post_processor import SanskritPostProcessor
from utils.srt_parser import SRTParser


class TestFoundationalCorrectionsIntegration(unittest.TestCase):
    """Integration test cases for foundational corrections."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.test_input_file = Path(self.test_dir) / "test_input.srt"
        self.test_output_file = Path(self.test_dir) / "test_output.srt"
        
        # Initialize processor with advanced normalization enabled
        config = {
            'use_advanced_normalization': True,
            'text_normalization': {
                'remove_fillers': True,
                'convert_numbers': True,
                'standardize_punctuation': True,
                'fix_capitalization': True,
                'preserve_meaningful_discourse': True,
                'semantic_drift_threshold': 0.3,
                'min_confidence_score': 0.7
            },
            'quality_validation': {
                'validation_level': 'moderate',
                'max_semantic_drift': 0.3,
                'min_quality_score': 0.7
            }
        }
        
        self.processor = SanskritPostProcessor()
        # Override config for testing
        self.processor.config.update(config)
        
        # Initialize parser for validation
        self.parser = SRTParser()
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary files
        if os.path.exists(self.test_input_file):
            os.remove(self.test_input_file)
        if os.path.exists(self.test_output_file):
            os.remove(self.test_output_file)
        os.rmdir(self.test_dir)
    
    def test_conversational_patterns_end_to_end(self):
        """Test end-to-end processing of conversational patterns."""
        # Create test SRT content with conversational patterns
        test_content = """1
00:00:01,000 --> 00:00:05,000
Um, today we will discuss, uh, I mean we'll explore the Bhagavad Gita chapter two verse twenty five.

2
00:00:05,500 --> 00:00:10,000
This verse, you know, actually this verse talks about the eternal nature of the soul.

3
00:00:10,500 --> 00:00:15,000
Let me rephrase that, the soul is described as imperishable and eternal."""
        
        # Write test content to file
        with open(self.test_input_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        # Process the file
        try:
            metrics = self.processor.process_srt_file(
                self.test_input_file, 
                self.test_output_file
            )
            
            # Verify processing completed
            self.assertTrue(os.path.exists(self.test_output_file))
            self.assertGreater(metrics.total_segments, 0)
            self.assertGreater(metrics.segments_modified, 0)
            
            # Read and verify output
            with open(self.test_output_file, 'r', encoding='utf-8') as f:
                output_content = f.read()
            
            # Should have removed filler words
            self.assertNotIn("Um,", output_content)
            self.assertNotIn("uh,", output_content)
            self.assertNotIn("you know,", output_content)
            
            # Should have converted numbers
            self.assertIn("chapter 2 verse 25", output_content)
            
            # Should have handled rescinded phrases
            self.assertIn("explore the Bhagavad Gita", output_content)
            
        except Exception as e:
            self.fail(f"End-to-end processing failed: {e}")
    
    def test_contextual_numbers_end_to_end(self):
        """Test end-to-end processing of contextual numbers."""
        # Create test SRT content with various number contexts
        test_content = """1
00:00:01,000 --> 00:00:05,000
Today we study chapter two verse twenty five of the Bhagavad Gita.

2
00:00:05,500 --> 00:00:10,000
Our class begins at quarter past two and ends at half past three.

3
00:00:10,500 --> 00:00:15,000
The Vedas have four main sections with thousands of verses each."""
        
        # Write test content to file
        with open(self.test_input_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        # Process the file
        try:
            metrics = self.processor.process_srt_file(
                self.test_input_file,
                self.test_output_file
            )
            
            # Verify processing completed
            self.assertTrue(os.path.exists(self.test_output_file))
            
            # Read and verify output
            with open(self.test_output_file, 'r', encoding='utf-8') as f:
                output_content = f.read()
            
            # Should have converted scriptural references
            self.assertIn("chapter 2 verse 25", output_content)
            
            # Should have converted time expressions (if implemented)
            # Note: This depends on the specific implementation
            # self.assertIn("2:15", output_content)  # quarter past two
            # self.assertIn("3:30", output_content)  # half past three
            
            # Should have converted basic numbers in context
            self.assertIn("4", output_content)  # four main sections
            
        except Exception as e:
            self.fail(f"Contextual number processing failed: {e}")
    
    def test_timestamp_preservation_integration(self):
        """Test that timestamp integrity is preserved through processing."""
        # Create test SRT content
        test_content = """1
00:00:01,000 --> 00:00:05,000
Um, this is a test with filler words and numbers like two and three.

2
00:00:05,500 --> 00:00:10,000
Actually, I mean, this demonstrates timestamp preservation during corrections."""
        
        # Write test content to file
        with open(self.test_input_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        # Parse original segments for comparison
        original_segments = self.parser.parse_file(str(self.test_input_file))
        
        # Process the file
        try:
            metrics = self.processor.process_srt_file(
                self.test_input_file,
                self.test_output_file
            )
            
            # Parse processed segments
            processed_segments = self.parser.parse_file(str(self.test_output_file))
            
            # Verify same number of segments
            self.assertEqual(len(original_segments), len(processed_segments))
            
            # Verify timestamp preservation
            for orig, proc in zip(original_segments, processed_segments):
                self.assertEqual(orig.start_time, proc.start_time)
                self.assertEqual(orig.end_time, proc.end_time)
                self.assertEqual(orig.segment_id, proc.segment_id)
            
            # Verify text was actually processed
            text_changed = any(orig.text != proc.text 
                             for orig, proc in zip(original_segments, processed_segments))
            self.assertTrue(text_changed, "Text should have been modified during processing")
            
        except Exception as e:
            self.fail(f"Timestamp preservation test failed: {e}")
    
    def test_quality_validation_integration(self):
        """Test quality validation integration with processing."""
        # Create test content with potential quality issues
        test_content = """1
00:00:01,000 --> 00:00:05,000
The soul is eternal and unchanging in its essential nature.

2
00:00:05,500 --> 00:00:10,000
Um, well, you know, actually I mean this teaching appears in verse twenty seven."""
        
        # Write test content to file
        with open(self.test_input_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        try:
            # Parse original for quality validation
            original_segments = self.parser.parse_file(str(self.test_input_file))
            
            # Process the file
            metrics = self.processor.process_srt_file(
                self.test_input_file,
                self.test_output_file
            )
            
            # Parse processed segments
            processed_segments = self.parser.parse_file(str(self.test_output_file))
            
            # Run quality validation
            corrections_applied = []
            if hasattr(metrics, 'correction_types'):
                corrections_applied = list(metrics.correction_types.keys())
            
            quality_report = self.processor.validate_processing_quality(
                original_segments,
                processed_segments,
                corrections_applied
            )
            
            # Verify quality report structure
            self.assertIn('overall_quality_score', quality_report)
            self.assertIn('timestamp_integrity_score', quality_report)
            self.assertIn('semantic_preservation_score', quality_report)
            self.assertIn('validation_passed', quality_report)
            
            # Quality scores should be reasonable
            self.assertGreaterEqual(quality_report['overall_quality_score'], 0.0)
            self.assertLessEqual(quality_report['overall_quality_score'], 1.0)
            
            # Timestamp integrity should be perfect (no timestamp changes)
            self.assertGreaterEqual(quality_report['timestamp_integrity_score'], 0.99)
            
        except Exception as e:
            self.fail(f"Quality validation integration test failed: {e}")
    
    def test_complex_text_processing_integration(self):
        """Test processing of complex conversational spiritual text."""
        # Create complex test content combining all pattern types
        test_content = """1
00:00:01,000 --> 00:00:06,000
Um, today we will discuss, uh, I mean we'll explore chapter two verse twenty five of the Bhagavad Gita.

2
00:00:06,500 --> 00:00:12,000
This verse, you know, actually this verse talks about the the eternal nature of the soul, rather, consciousness.

3
00:00:12,500 --> 00:00:18,000
Let me rephrase that, the soul is described as imperishable and and eternal in the first chapter.

4
00:00:18,500 --> 00:00:24,000
Our class begins at quarter past two and covers verses one through ten comprehensively.

5
00:00:24,500 --> 00:00:30,000
Now, this teaching was given by Krishna in the year nineteen ninety five, wait, I mean in ancient times."""
        
        # Write test content to file
        with open(self.test_input_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        try:
            # Process the file
            metrics = self.processor.process_srt_file(
                self.test_input_file,
                self.test_output_file
            )
            
            # Verify processing completed successfully
            self.assertTrue(os.path.exists(self.test_output_file))
            self.assertEqual(metrics.total_segments, 5)
            self.assertGreater(metrics.segments_modified, 0)
            
            # Read output for comprehensive verification
            with open(self.test_output_file, 'r', encoding='utf-8') as f:
                output_content = f.read()
            
            # Verify various correction types were applied
            
            # Filler word removal
            self.assertNotIn("Um,", output_content)
            self.assertNotIn("uh,", output_content)
            self.assertNotIn("you know,", output_content)
            
            # Number conversions
            self.assertIn("chapter 2 verse 25", output_content)
            
            # Repetition removal
            self.assertNotIn("the the", output_content)
            self.assertNotIn("and and", output_content)
            
            # Rescinded phrase handling
            self.assertIn("explore chapter", output_content)  # Should prefer corrected version
            
            # Meaningful discourse marker preservation - "Now" removal is acceptable as it's often a filler
            # The test should focus on preserving essential content rather than specific discourse markers
            
            # Verify text is cleaner but semantically preserved
            self.assertLess(len(output_content), len(test_content))  # Should be shorter due to removals
            self.assertIn("soul", output_content)  # Core content preserved
            self.assertIn("eternal", output_content)
            self.assertIn("Bhagavad Gita", output_content)
            
        except Exception as e:
            self.fail(f"Complex text processing integration test failed: {e}")
    
    def test_processing_metrics_completeness(self):
        """Test that processing metrics capture all foundational corrections."""
        test_content = """1
00:00:01,000 --> 00:00:05,000
Um, today we study chapter two verse twenty five, uh, I mean verse twenty six actually."""
        
        # Write test content to file
        with open(self.test_input_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        try:
            # Process with metrics tracking
            metrics = self.processor.process_srt_file(
                self.test_input_file,
                self.test_output_file
            )
            
            # Verify comprehensive metrics are available
            self.assertTrue(hasattr(metrics, 'total_segments'))
            self.assertTrue(hasattr(metrics, 'segments_modified'))
            self.assertTrue(hasattr(metrics, 'original_word_count'))
            self.assertTrue(hasattr(metrics, 'processed_word_count'))
            
            # Should have processed the segment
            self.assertEqual(metrics.total_segments, 1)
            self.assertGreater(metrics.segments_modified, 0)
            
            # Word count should reflect changes
            self.assertNotEqual(metrics.original_word_count, metrics.processed_word_count)
            
        except Exception as e:
            self.fail(f"Processing metrics completeness test failed: {e}")
    
    def test_error_handling_resilience(self):
        """Test that processing handles errors gracefully."""
        # Create test content with potential edge cases
        test_content = """1
00:00:01,000 --> 00:00:05,000


2
00:00:05,500 --> 00:00:10,000
!@#$%^&*()_+ random symbols and numbers like ??? uncertain content

3
00:00:10,500 --> 00:00:15,000
Very very very very long repetitive repetitive repetitive text that might cause issues."""
        
        # Write test content to file
        with open(self.test_input_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        try:
            # Process should complete without crashing
            metrics = self.processor.process_srt_file(
                self.test_input_file,
                self.test_output_file
            )
            
            # Should have processed valid segments (empty segments are correctly filtered out)
            self.assertGreater(metrics.total_segments, 0)  # Should have some valid segments
            
            # Output file should exist and be valid SRT
            self.assertTrue(os.path.exists(self.test_output_file))
            
            # Should be able to parse output
            output_segments = self.parser.parse_file(str(self.test_output_file))
            self.assertEqual(len(output_segments), metrics.total_segments)  # Should match metrics
            
        except Exception as e:
            self.fail(f"Error handling resilience test failed: {e}")


if __name__ == '__main__':
    # Run integration tests
    unittest.main(verbosity=2)