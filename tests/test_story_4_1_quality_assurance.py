"""
Comprehensive Quality Assurance Tests for Story 4.1: MCP Infrastructure Foundation

This test suite provides comprehensive regression testing for "one by one" patterns
and quality gate validation for context classification as specified in AC3.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.advanced_text_normalizer import AdvancedTextNormalizer, NumberContextType


class TestIdomaticExpressionRegression:
    """
    AC3 Requirement: Comprehensive regression testing for "one by one" patterns
    
    This class ensures that critical idiomatic expressions are NEVER converted
    to digits, preventing regression of the primary quality issue.
    """
    
    @pytest.fixture
    def normalizer(self):
        """Create AdvancedTextNormalizer with MCP enabled for testing."""
        config = {'enable_mcp_processing': True, 'enable_fallback': True}
        return AdvancedTextNormalizer(config)
    
    def test_primary_one_by_one_pattern(self, normalizer):
        """Test the primary 'one by one' pattern that caused the original issue."""
        test_cases = [
            "And one by one, he killed six of their children.",
            "One by one, the students entered the classroom.",
            "We examined the evidence one by one.",
            "The flowers bloomed one by one in spring.",
            "One by one, they shared their stories.",
        ]
        
        for text in test_cases:
            result = normalizer.convert_numbers_with_context(text)
            assert "one by one" in result.lower(), f"REGRESSION: 'one by one' was incorrectly converted in: {text}"
            assert "1 by 1" not in result, f"REGRESSION: Found '1 by 1' conversion in: {text}"
    
    def test_similar_idiomatic_patterns(self, normalizer):
        """Test similar idiomatic patterns that should also be preserved."""
        idiomatic_test_cases = [
            ("Two by two, they marched into the ark.", "two by two"),
            ("Step by step, we learned the process.", "step by step"),
            ("Day by day, we grew stronger.", "day by day"),
            ("Hand in hand, they walked together.", "hand in hand"),
            ("Side by side, we stood united.", "side by side"),
            ("Piece by piece, he assembled the puzzle.", "piece by piece"),
        ]
        
        for text, expected_phrase in idiomatic_test_cases:
            result = normalizer.convert_numbers_with_context(text)
            assert expected_phrase.lower() in result.lower(), f"REGRESSION: Idiomatic '{expected_phrase}' was converted in: {text}"
    
    def test_complex_idiomatic_expressions(self, normalizer):
        """Test complex idiomatic expressions that should be preserved."""
        complex_idioms = [
            "It takes two to tango in this relationship.",
            "Kill two birds with one stone for efficiency.",
            "A picture is worth a thousand words in this case.",
            "One man's trash is another man's treasure.",
        ]
        
        for text in complex_idioms:
            result = normalizer.convert_numbers_with_context(text)
            # These should remain largely unchanged
            original_numbers = ["two", "one", "thousand"]
            for num in original_numbers:
                if num in text.lower():
                    assert num in result.lower(), f"REGRESSION: Complex idiom altered in: {text}"


class TestContextClassificationQualityGates:
    """
    AC3 Requirement: Quality gate validation for context classification confidence
    
    This class ensures context classification accuracy and prevents misclassification
    that could lead to inappropriate number conversion.
    """
    
    @pytest.fixture
    def normalizer(self):
        """Create AdvancedTextNormalizer for testing."""
        config = {'enable_mcp_processing': True, 'enable_fallback': True}
        return AdvancedTextNormalizer(config)
    
    def test_context_classification_confidence_thresholds(self, normalizer):
        """Test that context classification meets minimum confidence thresholds."""
        test_cases = [
            ("And one by one, he killed six of their children.", NumberContextType.IDIOMATIC, 0.9),
            ("Chapter two verse twenty five of the Gita.", NumberContextType.SCRIPTURAL, 0.85),
            ("Year two thousand five was significant.", NumberContextType.TEMPORAL, 0.9),
            ("Two plus two equals four exactly.", NumberContextType.MATHEMATICAL, 0.85),
            ("Lesson two covers basic concepts.", NumberContextType.EDUCATIONAL, 0.8),
            ("The first time we met was special.", NumberContextType.ORDINAL, 0.8),
            ("Once upon a time, there lived a king.", NumberContextType.NARRATIVE, 0.85),
        ]
        
        for text, expected_type, min_confidence in test_cases:
            context_type, confidence, segments = normalizer._classify_number_context_enhanced(text)
            
            assert context_type == expected_type, f"Context misclassified for: {text}. Expected {expected_type}, got {context_type}"
            assert confidence >= min_confidence, f"Confidence too low for: {text}. Expected >{min_confidence}, got {confidence}"
    
    def test_idiomatic_detection_accuracy(self, normalizer):
        """Test that idiomatic expressions are consistently detected with high confidence."""
        idiomatic_texts = [
            "And one by one, he killed six of their children.",
            "Two by two, they entered the building.",
            "Step by step, we solved the problem.",
            "Day by day, progress was made.",
            "Piece by piece, the truth emerged.",
        ]
        
        for text in idiomatic_texts:
            context_type, confidence, segments = normalizer._classify_number_context_enhanced(text)
            
            assert context_type == NumberContextType.IDIOMATIC, f"Failed to detect idiomatic context in: {text}"
            assert confidence >= 0.85, f"Idiomatic confidence too low for: {text}. Got {confidence}"
    
    def test_scriptural_vs_educational_distinction(self, normalizer):
        """Test distinction between scriptural and educational contexts."""
        test_cases = [
            # Should be SCRIPTURAL
            ("Chapter two verse twenty five of Bhagavad Gita.", NumberContextType.SCRIPTURAL),
            ("Verse three of chapter four explains dharma.", NumberContextType.SCRIPTURAL),
            ("Bhagavad Gita chapter two teaches karma yoga.", NumberContextType.SCRIPTURAL),
            
            # Should be EDUCATIONAL
            ("Chapter two of the textbook covers basics.", NumberContextType.EDUCATIONAL),
            ("Lesson three explains the methodology.", NumberContextType.EDUCATIONAL),
            ("Page twenty five has the answer.", NumberContextType.EDUCATIONAL),
        ]
        
        for text, expected_type in test_cases:
            context_type, confidence, segments = normalizer._classify_number_context_enhanced(text)
            assert context_type == expected_type, f"Context misclassified for: {text}. Expected {expected_type}, got {context_type}"


class TestEnhancedContextProcessing:
    """
    AC2 Requirement: Enhanced context-aware number processing
    
    Tests the new advanced context types and processing methods.
    """
    
    @pytest.fixture
    def normalizer(self):
        """Create AdvancedTextNormalizer for testing."""
        config = {'enable_mcp_processing': True, 'enable_fallback': True}
        return AdvancedTextNormalizer(config)
    
    def test_advanced_temporal_processing(self, normalizer):
        """Test enhanced temporal number processing patterns (AC2)."""
        temporal_test_cases = [
            ("Year two thousand five was significant.", "Year 2005 was significant."),
            ("In two thousand six, we started this.", "In 2006, we started this."),
            ("January two thousand seven brought changes.", "January 2007 brought changes."),
            ("Spring of two thousand eight was beautiful.", "Spring of 2008 was beautiful."),
            ("Born in two thousand nine.", "Born in 2009."),
        ]
        
        for input_text, expected_output in temporal_test_cases:
            result = normalizer.convert_numbers_with_context(input_text)
            assert result == expected_output, f"Temporal conversion failed. Input: {input_text}, Expected: {expected_output}, Got: {result}"
    
    def test_enhanced_scriptural_processing(self, normalizer):
        """Test enhanced scriptural number processing (AC2)."""
        scriptural_test_cases = [
            ("Chapter two verse twenty five", "Chapter 2 verse 25"),
            ("Bhagavad Gita chapter three verse ten", "Bhagavad Gita chapter 3 verse 10"),
            ("Yoga Sutras chapter two sutra fifteen", "Yoga Sutras chapter 2 sutra 15"),
            ("Ramayana book four chapter twenty", "Ramayana book 4 chapter 20"),
        ]
        
        for input_text, expected_output in scriptural_test_cases:
            result = normalizer.convert_numbers_with_context(input_text)
            assert result == expected_output, f"Scriptural conversion failed. Input: {input_text}, Expected: {expected_output}, Got: {result}"
    
    def test_mathematical_expression_processing(self, normalizer):
        """Test mathematical context processing (AC2)."""
        mathematical_test_cases = [
            ("Two plus two equals four", "2 plus 2 equals 4"),
            ("Three times five is fifteen", "3 times 5 is 15"),
            ("Ten divided by two equals five", "10 divided by 2 equals 5"),
            ("The sum of three and seven", "The sum of 3 and 7"),
        ]
        
        for input_text, expected_output in mathematical_test_cases:
            result = normalizer.convert_numbers_with_context(input_text)
            assert result == expected_output, f"Mathematical conversion failed. Input: {input_text}, Expected: {expected_output}, Got: {result}"
    
    def test_educational_context_processing(self, normalizer):
        """Test educational context processing (Story 4.1 enhancement)."""
        educational_test_cases = [
            ("Lesson two covers basic concepts", "Lesson 2 covers basic concepts"),
            ("Page twenty five has the answer", "Page 25 has the answer"),
            ("Question three is difficult", "Question 3 is difficult"),
            ("Grade four students learn this", "Grade 4 students learn this"),
        ]
        
        for input_text, expected_output in educational_test_cases:
            result = normalizer.convert_numbers_with_context(input_text)
            assert result == expected_output, f"Educational conversion failed. Input: {input_text}, Expected: {expected_output}, Got: {result}"
    
    def test_ordinal_context_processing(self, normalizer):
        """Test ordinal context processing (Story 4.1 enhancement)."""
        ordinal_test_cases = [
            ("The first time we met", "The 1st time we met"),
            ("Second attempt was successful", "2nd attempt was successful"),
            ("Third try worked perfectly", "3rd try worked perfectly"),
        ]
        
        for input_text, expected_output in ordinal_test_cases:
            result = normalizer.convert_numbers_with_context(input_text)
            assert result == expected_output, f"Ordinal conversion failed. Input: {input_text}, Expected: {expected_output}, Got: {result}"
    
    def test_narrative_context_processing(self, normalizer):
        """Test narrative context processing (Story 4.1 enhancement)."""
        narrative_test_cases = [
            # Should be PRESERVED
            ("Once upon a time, there lived a king.", "Once upon a time, there lived a king."),
            ("Twice upon a time in the forest.", "Twice upon a time in the forest."),
            ("In the beginning, there was light.", "In the beginning, there was light."),
            
            # Should be CONVERTED (structural elements)
            ("First act begins now.", "1st act begins now."),
            ("Second scene was dramatic.", "2nd scene was dramatic."),
        ]
        
        for input_text, expected_output in narrative_test_cases:
            result = normalizer.convert_numbers_with_context(input_text)
            assert result == expected_output, f"Narrative processing failed. Input: {input_text}, Expected: {expected_output}, Got: {result}"


class TestQualityGateValidation:
    """
    AC3 Requirement: Quality gate validation preventing regression
    
    This class implements automated quality gates that prevent regression
    of critical quality patterns.
    """
    
    @pytest.fixture
    def normalizer(self):
        """Create AdvancedTextNormalizer for testing."""
        config = {'enable_mcp_processing': True, 'enable_fallback': True}
        return AdvancedTextNormalizer(config)
    
    def test_critical_pattern_quality_gate(self, normalizer):
        """
        QUALITY GATE: Ensure critical patterns are never regressed.
        
        This test acts as a quality gate that must ALWAYS pass.
        """
        critical_patterns = [
            # The original critical issue
            ("And one by one, he killed six of their children.", "one by one"),
            
            # Additional critical patterns
            ("Two by two, animals entered the ark.", "two by two"),
            ("Step by step, we climbed the mountain.", "step by step"),
            ("Day by day, we grew stronger.", "day by day"),
            
            # Complex cases
            ("Hand in hand, they walked together.", "hand in hand"),
            ("Side by side, we fought the battle.", "side by side"),
        ]
        
        for text, critical_phrase in critical_patterns:
            result = normalizer.convert_numbers_with_context(text)
            
            # QUALITY GATE: Critical phrase MUST be preserved
            assert critical_phrase in result.lower(), f"QUALITY GATE FAILURE: Critical phrase '{critical_phrase}' was lost in: {text}"
            
            # QUALITY GATE: Classification MUST be IDIOMATIC
            context_type, confidence, segments = normalizer._classify_number_context_enhanced(text)
            assert context_type == NumberContextType.IDIOMATIC, f"QUALITY GATE FAILURE: Critical text misclassified as {context_type}: {text}"
            assert confidence >= 0.85, f"QUALITY GATE FAILURE: Confidence too low ({confidence}) for critical text: {text}"
    
    def test_context_classification_confidence_gates(self, normalizer):
        """
        QUALITY GATE: Context classification confidence validation.
        
        Ensures all context classifications meet minimum confidence thresholds.
        """
        confidence_test_cases = [
            # High confidence required for IDIOMATIC (critical)
            ("One by one, they departed.", NumberContextType.IDIOMATIC, 0.90),
            ("Two by two, they arrived.", NumberContextType.IDIOMATIC, 0.90),
            
            # Good confidence for SCRIPTURAL
            ("Chapter two verse twenty five", NumberContextType.SCRIPTURAL, 0.85),
            ("Bhagavad Gita chapter three", NumberContextType.SCRIPTURAL, 0.85),
            
            # Good confidence for TEMPORAL
            ("Year two thousand five", NumberContextType.TEMPORAL, 0.90),
            ("In two thousand six", NumberContextType.TEMPORAL, 0.85),
            
            # Acceptable confidence for MATHEMATICAL
            ("Two plus two equals four", NumberContextType.MATHEMATICAL, 0.80),
            ("Three times five is fifteen", NumberContextType.MATHEMATICAL, 0.80),
        ]
        
        for text, expected_type, min_confidence in confidence_test_cases:
            context_type, confidence, segments = normalizer._classify_number_context_enhanced(text)
            
            assert context_type == expected_type, f"QUALITY GATE: Context type mismatch for '{text}'"
            assert confidence >= min_confidence, f"QUALITY GATE: Confidence {confidence:.3f} below threshold {min_confidence} for '{text}'"
    
    def test_processing_consistency_gate(self, normalizer):
        """
        QUALITY GATE: Ensure processing consistency across similar patterns.
        
        Similar patterns should be processed consistently.
        """
        consistency_groups = [
            # Group 1: Idiomatic expressions
            [
                "One by one, they left.",
                "Two by two, they arrived.",
                "Three by three, they organized.",
            ],
            
            # Group 2: Scriptural references
            [
                "Chapter two verse three",
                "Chapter four verse seven",
                "Chapter six verse twelve",
            ],
            
            # Group 3: Temporal references
            [
                "Year two thousand five",
                "Year two thousand six",
                "Year two thousand seven",
            ],
        ]
        
        for group in consistency_groups:
            context_types = []
            confidences = []
            
            for text in group:
                context_type, confidence, segments = normalizer._classify_number_context_enhanced(text)
                context_types.append(context_type)
                confidences.append(confidence)
            
            # All items in group should have same context type
            assert len(set(context_types)) == 1, f"QUALITY GATE: Inconsistent context classification in group: {group}"
            
            # Confidence variance should be reasonable (within 0.1)
            if len(confidences) > 1:
                confidence_range = max(confidences) - min(confidences)
                assert confidence_range <= 0.1, f"QUALITY GATE: High confidence variance ({confidence_range:.3f}) in group: {group}"


class TestRegressionPreventionFramework:
    """
    AC3 Requirement: Automated testing for all critical quality patterns
    
    This framework prevents regression by testing all critical patterns
    that could impact quality.
    """
    
    @pytest.fixture
    def normalizer(self):
        """Create AdvancedTextNormalizer for testing."""
        config = {'enable_mcp_processing': True, 'enable_fallback': True}
        return AdvancedTextNormalizer(config)
    
    def test_comprehensive_idiomatic_preservation(self, normalizer):
        """Test comprehensive preservation of all idiomatic expressions."""
        # Comprehensive list of idiomatic expressions that should NEVER be converted
        idiomatic_expressions = [
            "one by one", "two by two", "three by three",
            "step by step", "day by day", "week by week",
            "hand in hand", "side by side", "face to face",
            "piece by piece", "bit by bit", "inch by inch",
            "one after another", "one at a time", "two at a time",
        ]
        
        for expression in idiomatic_expressions:
            # Test in various sentence contexts
            test_contexts = [
                f"And {expression}, they proceeded.",
                f"We moved {expression} through the process.",
                f"{expression.capitalize()}, we addressed each issue.",
            ]
            
            for context in test_contexts:
                result = normalizer.convert_numbers_with_context(context)
                assert expression in result.lower(), f"REGRESSION PREVENTION: Expression '{expression}' was converted in: {context}"
    
    def test_mixed_context_handling(self, normalizer):
        """Test handling of text with mixed context types."""
        mixed_context_cases = [
            # Idiomatic should take precedence
            ("One by one, we studied chapter two verse three.", "one by one"),
            
            # Complex sentences with multiple numbers
            ("In year two thousand five, lesson three taught us about one by one processing.", ["2005", "lesson 3", "one by one"]),
        ]
        
        for text, expected_elements in mixed_context_cases:
            result = normalizer.convert_numbers_with_context(text)
            
            if isinstance(expected_elements, str):
                assert expected_elements in result.lower(), f"Expected element '{expected_elements}' not found in: {result}"
            else:
                for element in expected_elements:
                    assert element in result.lower(), f"Expected element '{element}' not found in: {result}"
    
    def test_edge_case_patterns(self, normalizer):
        """Test edge cases that could cause regression."""
        edge_cases = [
            # Numbers at sentence boundaries
            ("One by one.", "One by one."),
            ("Two by two!", "Two by two!"),
            
            # Numbers with punctuation
            ("One by one, carefully and slowly,", "One by one, carefully and slowly,"),
            ("Two by two: they advanced forward.", "Two by two: they advanced forward."),
            
            # Case variations
            ("ONE BY ONE, THEY SHOUTED.", "ONE BY ONE, THEY SHOUTED."),
            ("One By One, They Whispered.", "One By One, They Whispered."),
        ]
        
        for input_text, expected_output in edge_cases:
            result = normalizer.convert_numbers_with_context(input_text)
            assert result == expected_output, f"Edge case failed. Input: {input_text}, Expected: {expected_output}, Got: {result}"


class TestPerformanceQualityGates:
    """
    AC4 Requirement: Performance validation with quality gates
    
    Ensures performance targets are met while maintaining quality.
    """
    
    @pytest.fixture
    def normalizer(self):
        """Create AdvancedTextNormalizer for testing."""
        config = {'enable_mcp_processing': True, 'enable_fallback': True}
        return AdvancedTextNormalizer(config)
    
    def test_processing_time_quality_gate(self, normalizer):
        """Test that processing time meets <1 second target (AC4)."""
        import time
        
        test_texts = [
            "And one by one, he killed six of their children.",
            "Chapter two verse twenty five of the Bhagavad Gita teaches us about karma.",
            "In the year two thousand five, we started our spiritual journey step by step.",
        ]
        
        for text in test_texts:
            start_time = time.time()
            result = normalizer.convert_numbers_with_context(text)
            processing_time = time.time() - start_time
            
            assert processing_time < 1.0, f"QUALITY GATE: Processing time {processing_time:.4f}s exceeds 1s target for: {text}"
            assert result is not None, f"Processing failed for: {text}"
    
    def test_fallback_system_quality_gate(self, normalizer):
        """Test that fallback system maintains quality when MCP fails."""
        # Test with potentially problematic input
        problematic_texts = [
            "One by one, two by two, three by three, they came.",
            "Chapter two verse twenty five and lesson three combine.",
            "Year two thousand five, first time, once upon a time.",
        ]
        
        for text in problematic_texts:
            result = normalizer.convert_numbers_with_context(text)
            
            # Basic quality checks
            assert result is not None, f"Fallback failed to produce result for: {text}"
            assert len(result.strip()) > 0, f"Fallback produced empty result for: {text}"
            
            # Critical pattern preservation
            if "one by one" in text.lower():
                assert "one by one" in result.lower(), f"Fallback failed to preserve 'one by one' in: {text}"


if __name__ == "__main__":
    # Run the quality assurance tests
    pytest.main([__file__, "-v", "--tb=short"])