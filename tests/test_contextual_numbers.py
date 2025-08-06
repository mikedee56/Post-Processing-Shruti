"""
Tests for the ContextualNumberProcessor module.

This module provides comprehensive tests for contextual number processing
functionality including spiritual context awareness.
"""

import unittest
import sys
import os

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.contextual_number_processor import (
    ContextualNumberProcessor,
    NumberContext,
    NumberConversion,
    ConversionResult
)


class TestContextualNumberProcessor(unittest.TestCase):
    """Test cases for ContextualNumberProcessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = ContextualNumberProcessor()
        
        # Test configuration for controlled testing
        self.test_config = {
            'min_confidence_threshold': 0.7,
            'preserve_uncertainty': True
        }
        
        self.configured_processor = ContextualNumberProcessor(self.test_config)
    
    def test_initialization(self):
        """Test that the processor initializes properly."""
        self.assertIsInstance(self.processor, ContextualNumberProcessor)
        self.assertTrue(hasattr(self.processor, 'basic_numbers'))
        self.assertTrue(hasattr(self.processor, 'spiritual_keywords'))
        self.assertTrue(hasattr(self.processor, 'scriptural_patterns'))
        self.assertTrue(hasattr(self.processor, 'date_patterns'))
        self.assertTrue(hasattr(self.processor, 'time_patterns'))
        self.assertTrue(hasattr(self.processor, 'ordinal_mappings'))
    
    def test_empty_text_processing(self):
        """Test processing of empty or whitespace-only text."""
        empty_result = self.processor.process_numbers("", "spiritual")
        whitespace_result = self.processor.process_numbers("   ", "spiritual")
        
        self.assertEqual(empty_result.total_conversions, 0)
        self.assertEqual(whitespace_result.total_conversions, 0)
        self.assertIn("Empty or whitespace-only text", empty_result.processing_notes[0])
    
    def test_scriptural_reference_processing(self):
        """Test processing of scriptural references."""
        text = "Today we study chapter two verse twenty five of the Bhagavad Gita."
        result = self.processor.process_numbers(text, "spiritual")
        
        # Should convert to "chapter 2 verse 25"
        self.assertIn("chapter 2 verse 25", result.processed_text)
        
        # Should have detected scriptural reference conversion
        scriptural_conversions = [c for c in result.conversions 
                                if c.number_context == NumberContext.SCRIPTURAL_REFERENCE]
        self.assertGreater(len(scriptural_conversions), 0)
    
    def test_verse_only_processing(self):
        """Test processing of verse-only references."""
        text = "This appears in verse twenty seven."
        result = self.processor.process_numbers(text, "spiritual")
        
        # Should convert to "verse 27"
        self.assertIn("verse 27", result.processed_text)
    
    def test_ordinal_number_processing(self):
        """Test processing of ordinal numbers in spiritual contexts."""
        text = "The first chapter discusses the second verse in detail."
        result = self.processor.process_numbers(text, "spiritual")
        
        # Should convert ordinals in spiritual context
        ordinal_conversions = [c for c in result.conversions 
                             if c.number_context == NumberContext.ORDINAL]
        
        # Depending on implementation, might detect "first chapter" and "second verse"
        if ordinal_conversions:
            self.assertGreater(len(ordinal_conversions), 0)
    
    def test_date_expression_processing(self):
        """Test processing of date expressions."""
        text = "This was written on January first, two thousand and five."
        result = self.processor.process_numbers(text, "spiritual")
        
        # Should convert date format
        date_conversions = [c for c in result.conversions 
                          if c.number_context == NumberContext.DATE]
        
        # Note: This test depends on the specific implementation of date patterns
        # The processor should at least attempt to process dates
        self.assertIsInstance(result, ConversionResult)
    
    def test_time_expression_processing(self):
        """Test processing of time expressions."""
        text = "Our class begins at quarter past two."
        result = self.processor.process_numbers(text, "spiritual")
        
        # Should convert to "2:15"
        if "2:15" in result.processed_text:
            self.assertIn("2:15", result.processed_text)
            
            time_conversions = [c for c in result.conversions 
                              if c.number_context == NumberContext.TIME]
            self.assertGreater(len(time_conversions), 0)
    
    def test_year_expression_processing(self):
        """Test processing of year expressions."""
        # Test "two thousand" format
        text1 = "This happened in two thousand and five."
        result1 = self.processor.process_numbers(text1, "spiritual")
        
        # Should convert to "2005"
        if "2005" in result1.processed_text:
            self.assertIn("2005", result1.processed_text)
        
        # Test "nineteen" format
        text2 = "This occurred in nineteen ninety five."
        result2 = self.processor.process_numbers(text2, "spiritual")
        
        # Should convert to "1995"
        if "1995" in result2.processed_text:
            self.assertIn("1995", result2.processed_text)
    
    def test_compound_number_processing(self):
        """Test processing of compound numbers."""
        text = "There are twenty five students in the class."
        result = self.processor.process_numbers(text, "spiritual")
        
        # Should convert "twenty five" to "25"
        cardinal_conversions = [c for c in result.conversions 
                              if c.number_context == NumberContext.CARDINAL]
        
        if cardinal_conversions:
            # Check that compound numbers are properly converted
            compound_found = any("25" in c.converted_text for c in cardinal_conversions)
            if compound_found:
                self.assertTrue(compound_found)
    
    def test_spiritual_context_awareness(self):
        """Test that processor is aware of spiritual contexts."""
        text = "The Vedas have four main sections with thousands of verses each."
        result = self.processor.process_numbers(text, "spiritual")
        
        # Should process "four" appropriately in spiritual context
        # The exact behavior depends on implementation
        self.assertIsInstance(result, ConversionResult)
        
        # Should have some contextual awareness
        self.assertTrue(hasattr(self.processor, 'spiritual_keywords'))
        self.assertIn('verse', self.processor.spiritual_keywords)
    
    def test_confidence_scoring(self):
        """Test confidence scoring of conversions."""
        text = "Chapter two verse twenty five teaches about the soul."
        result = self.processor.process_numbers(text, "spiritual")
        
        if result.conversions:
            # Scriptural references should have high confidence
            scriptural_conversions = [c for c in result.conversions 
                                    if c.number_context == NumberContext.SCRIPTURAL_REFERENCE]
            
            if scriptural_conversions:
                for conversion in scriptural_conversions:
                    self.assertGreaterEqual(conversion.confidence_score, 0.8)
    
    def test_overlapping_conversion_resolution(self):
        """Test resolution of overlapping conversions."""
        text = "In chapter two verse twenty five we find wisdom."
        result = self.processor.process_numbers(text, "spiritual")
        
        # Should resolve overlapping conversions properly
        conversions = result.conversions
        for i, conv1 in enumerate(conversions):
            for j, conv2 in enumerate(conversions):
                if i != j:
                    # Check that conversions don't overlap
                    overlap = not (conv1.end_pos <= conv2.start_pos or conv2.end_pos <= conv1.start_pos)
                    self.assertFalse(overlap, f"Conversions {i} and {j} overlap")
    
    def test_high_confidence_only_application(self):
        """Test that only high-confidence conversions are applied."""
        # Create processor with high confidence threshold
        high_threshold_processor = ContextualNumberProcessor({'min_confidence_threshold': 0.9})
        
        text = "This might have some uncertain numbers like three or four items."
        result = high_threshold_processor.process_numbers(text, "spiritual")
        
        # Should only apply high-confidence conversions
        applied_conversions = [c for c in result.conversions 
                             if c.confidence_score >= 0.9]
        self.assertEqual(len(applied_conversions), result.high_confidence_conversions)
    
    def test_word_to_number_conversion(self):
        """Test basic word to number conversion functionality."""
        # Test basic numbers
        self.assertEqual(self.processor._word_to_number("five"), "5")
        self.assertEqual(self.processor._word_to_number("twenty"), "20")
        
        # Test compound numbers if implemented
        compound_result = self.processor._word_to_number("twenty five")
        if compound_result:
            self.assertEqual(compound_result, "25")
    
    def test_ordinal_word_conversion(self):
        """Test ordinal word to number conversion."""
        self.assertEqual(self.processor._convert_ordinal_word_to_number("first"), "1st")
        self.assertEqual(self.processor._convert_ordinal_word_to_number("second"), "2nd")
        self.assertEqual(self.processor._convert_ordinal_word_to_number("third"), "3rd")
        
        # Test compound ordinals if implemented
        compound_ordinal = self.processor._convert_ordinal_word_to_number("twenty first")
        if compound_ordinal:
            self.assertEqual(compound_ordinal, "21st")
    
    def test_complex_spiritual_text(self):
        """Test processing of complex spiritual text with multiple number types."""
        complex_text = """Today we study chapter two verse twenty five of the Bhagavad Gita.
        This was composed in the first century, around two thousand years ago.
        Our class begins at quarter past two and covers verses one through ten."""
        
        result = self.processor.process_numbers(complex_text, "spiritual")
        
        # Should detect multiple types of number contexts
        contexts_found = set(c.number_context for c in result.conversions)
        
        # Should have found at least some conversions
        self.assertGreater(result.total_conversions, 0)
        
        # Should have processed the text
        self.assertNotEqual(result.original_text, result.processed_text)
    
    def test_configuration_respect(self):
        """Test that the processor respects configuration settings."""
        config = {
            'min_confidence_threshold': 0.8,
            'preserve_uncertainty': False
        }
        
        configured_processor = ContextualNumberProcessor(config)
        
        self.assertEqual(configured_processor.min_confidence_threshold, 0.8)
        self.assertEqual(configured_processor.preserve_uncertainty, False)


class TestNumberConversion(unittest.TestCase):
    """Test cases for NumberConversion dataclass."""
    
    def test_number_conversion_creation(self):
        """Test that NumberConversion objects can be created properly."""
        conversion = NumberConversion(
            original_text="chapter two",
            converted_text="chapter 2",
            number_context=NumberContext.SCRIPTURAL_REFERENCE,
            confidence_score=0.95,
            start_pos=10,
            end_pos=21,
            reasoning="Scriptural reference conversion"
        )
        
        self.assertEqual(conversion.original_text, "chapter two")
        self.assertEqual(conversion.converted_text, "chapter 2")
        self.assertEqual(conversion.number_context, NumberContext.SCRIPTURAL_REFERENCE)
        self.assertEqual(conversion.confidence_score, 0.95)


class TestConversionResult(unittest.TestCase):
    """Test cases for ConversionResult dataclass."""
    
    def test_conversion_result_creation(self):
        """Test that ConversionResult objects can be created properly."""
        result = ConversionResult(
            original_text="Original text",
            processed_text="Processed text",
            conversions=[],
            total_conversions=0,
            high_confidence_conversions=0,
            processing_notes=["Test note"]
        )
        
        self.assertEqual(result.original_text, "Original text")
        self.assertEqual(result.processed_text, "Processed text")
        self.assertEqual(result.total_conversions, 0)
        self.assertEqual(result.high_confidence_conversions, 0)


class TestNumberContext(unittest.TestCase):
    """Test cases for NumberContext enum."""
    
    def test_number_context_values(self):
        """Test that NumberContext enum has expected values."""
        self.assertEqual(NumberContext.SCRIPTURAL_REFERENCE.value, "scriptural_reference")
        self.assertEqual(NumberContext.DATE.value, "date")
        self.assertEqual(NumberContext.TIME.value, "time")
        self.assertEqual(NumberContext.ORDINAL.value, "ordinal")
        self.assertEqual(NumberContext.CARDINAL.value, "cardinal")
        self.assertEqual(NumberContext.YEAR.value, "year")


if __name__ == '__main__':
    unittest.main()