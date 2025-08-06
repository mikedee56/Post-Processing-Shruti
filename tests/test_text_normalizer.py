"""
Unit tests for Text Normalizer functionality.

Tests comprehensive text normalization including number conversion,
filler word removal, punctuation standardization, and capitalization.
"""

import pytest
from typing import Dict, List

from src.utils.text_normalizer import TextNormalizer, NormalizationResult


class TestTextNormalizer:
    """Test suite for TextNormalizer class."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.normalizer = TextNormalizer()
    
    def test_normalize_simple_text(self):
        """Test basic text normalization."""
        text = "Hello world"
        result = self.normalizer.normalize_text(text)
        assert result == "Hello world"
    
    def test_normalize_empty_text(self):
        """Test normalizing empty text."""
        assert self.normalizer.normalize_text("") == ""
        assert self.normalizer.normalize_text("   ") == "   "
        assert self.normalizer.normalize_text(None) == None
    
    def test_normalize_with_tracking(self):
        """Test normalization with change tracking."""
        text = "Um, hello world you know"
        result = self.normalizer.normalize_with_tracking(text)
        
        assert isinstance(result, NormalizationResult)
        assert result.original_text == text
        assert "um" not in result.normalized_text.lower()
        assert "you know" not in result.normalized_text.lower()
        assert "removed_filler_words" in result.changes_applied
        assert result.word_count_before > result.word_count_after
    
    def test_convert_basic_numbers(self):
        """Test converting basic number words to digits."""
        test_cases = [
            ("one", "1"),
            ("two", "2"),
            ("ten", "10"),
            ("twenty", "20"),
            ("I have one apple", "I have 1 apple"),
            ("chapter two verse three", "chapter 2 verse 3"),
            ("The first ten items", "The 1st 10 items")
        ]
        
        for input_text, expected in test_cases:
            result = self.normalizer.convert_numbers(input_text)
            assert result == expected, f"Failed for '{input_text}' -> expected '{expected}', got '{result}'"
    
    def test_convert_compound_numbers(self):
        """Test converting compound numbers."""
        test_cases = [
            ("twenty one", "21"),
            ("thirty five", "35"),
            ("forty two", "42"),
            ("ninety nine", "99"),
            ("I am twenty five years old", "I am 25 years old")
        ]
        
        for input_text, expected in test_cases:
            result = self.normalizer.convert_numbers(input_text)
            assert result == expected, f"Failed for '{input_text}'"
    
    def test_convert_ordinals(self):
        """Test converting ordinal numbers."""
        test_cases = [
            ("first", "1st"),
            ("second", "2nd"),
            ("third", "3rd"),
            ("fourth", "4th"),
            ("tenth", "10th"),
            ("the first chapter", "the 1st chapter"),
            ("in the second verse", "in the 2nd verse")
        ]
        
        for input_text, expected in test_cases:
            result = self.normalizer.convert_numbers(input_text)
            assert result == expected, f"Failed for '{input_text}'"
    
    def test_convert_year_patterns(self):
        """Test converting year patterns."""
        test_cases = [
            ("two thousand", "2000"),
            ("In the year two thousand five", "In the year 2000 5"),  # Simplified pattern
        ]
        
        for input_text, expected in test_cases:
            result = self.normalizer.convert_numbers(input_text)
            assert result == expected, f"Failed for '{input_text}'"
    
    def test_remove_single_filler_words(self):
        """Test removing single-word fillers."""
        test_cases = [
            ("um hello", "hello"),
            ("Hello uh world", "Hello world"),
            ("Actually, this is good", "this is good"),
            ("This is, like, amazing", "This is amazing"),
            ("I mean, you know, it's great", "it's great")
        ]
        
        for input_text, expected in test_cases:
            result = self.normalizer.remove_filler_words(input_text)
            assert result == expected, f"Failed for '{input_text}'"
    
    def test_remove_multiword_fillers(self):
        """Test removing multi-word filler phrases."""
        test_cases = [
            ("This is you know great", "This is great"),
            ("I mean this is good", "this is good"),
            ("It's sort of difficult", "It's difficult"),
            ("This is kind of important", "This is important"),
            ("What I mean is this", "this")
        ]
        
        for input_text, expected in test_cases:
            result = self.normalizer.remove_filler_words(input_text)
            assert result == expected, f"Failed for '{input_text}'"
    
    def test_preserve_important_words(self):
        """Test that important words are not removed as fillers."""
        # Words that might be fillers in some contexts but not others
        test_cases = [
            ("Actually, the actual result", "the actual result"),  # "actual" should be preserved
            ("This is really real", "This is real"),  # "real" should be preserved
            ("Like this example", "this example"),  # "like" as filler vs "like" as verb
        ]
        
        for input_text, expected in test_cases:
            result = self.normalizer.remove_filler_words(input_text)
            assert result == expected, f"Failed for '{input_text}'"
    
    def test_standardize_punctuation_spacing(self):
        """Test punctuation spacing standardization."""
        test_cases = [
            ("Hello , world", "Hello, world"),
            ("Hello ,world", "Hello, world"),
            ("Hello,world", "Hello, world"),
            ("What ?Really", "What? Really"),
            ("Yes !Exactly", "Yes! Exactly"),
            ("End.Start", "End. Start"),
            ("Multiple   spaces", "Multiple spaces")
        ]
        
        for input_text, expected in test_cases:
            result = self.normalizer.standardize_punctuation(input_text)
            assert result == expected, f"Failed for '{input_text}'"
    
    def test_standardize_quotation_marks(self):
        """Test quotation mark standardization."""
        test_cases = [
            ('"Hello"', '"Hello"'),
            ("'Hello'", "'Hello'"),
            ('"Mixed quotes"', '"Mixed quotes"'),
            ("Don't", "Don't")
        ]
        
        for input_text, expected in test_cases:
            result = self.normalizer.standardize_punctuation(input_text)
            assert result == expected, f"Failed for '{input_text}'"
    
    def test_standardize_multiple_punctuation(self):
        """Test handling multiple punctuation marks."""
        test_cases = [
            ("Hello...", "Hello..."),
            ("Hello....", "Hello..."),
            ("What??", "What?"),
            ("Amazing!!!", "Amazing!"),
            ("Wait......", "Wait...")
        ]
        
        for input_text, expected in test_cases:
            result = self.normalizer.standardize_punctuation(input_text)
            assert result == expected, f"Failed for '{input_text}'"
    
    def test_fix_capitalization_sentences(self):
        """Test sentence capitalization fixes."""
        test_cases = [
            ("hello world", "Hello world"),
            ("hello. world", "Hello. World"),
            ("hello! how are you?", "Hello! How are you?"),
            ("yes. i am fine.", "Yes. I am fine."),
            ("what? really! amazing.", "What? Really! Amazing.")
        ]
        
        for input_text, expected in test_cases:
            result = self.normalizer.fix_capitalization(input_text)
            assert result == expected, f"Failed for '{input_text}'"
    
    def test_fix_capitalization_i_pronoun(self):
        """Test 'i' pronoun capitalization."""
        test_cases = [
            ("i am here", "I am here"),
            ("Yes, i think so", "Yes, I think so"),
            ("i think i know", "I think I know"),
            ("india is great", "India is great")  # Should capitalize as proper noun
        ]
        
        for input_text, expected in test_cases:
            result = self.normalizer.fix_capitalization(input_text)
            assert result == expected, f"Failed for '{input_text}'"
    
    def test_fix_proper_nouns(self):
        """Test proper noun capitalization."""
        test_cases = [
            ("english is hard", "English is hard"),
            ("i study sanskrit", "I study Sanskrit"),
            ("the bhagavad gita", "The Bhagavad Gita"),
            ("yoga and vedanta", "Yoga and Vedanta"),
            ("hindi and english", "Hindi and English")
        ]
        
        for input_text, expected in test_cases:
            result = self.normalizer.fix_capitalization(input_text)
            assert result == expected, f"Failed for '{input_text}'"
    
    def test_complete_normalization_pipeline(self):
        """Test the complete normalization pipeline."""
        input_text = "um, today we will discuss the, uh, bhagavad gita chapter two verse twenty five. this verse talks about the eternal nature of the soul, you know."
        expected_text = "Today we will discuss the Bhagavad Gita chapter 2 verse 25. This verse talks about the eternal nature of the soul."
        
        result = self.normalizer.normalize_text(input_text)
        assert result == expected_text
    
    def test_normalization_with_custom_config(self):
        """Test normalization with custom configuration."""
        config = {
            'remove_fillers': False,  # Disable filler removal
            'convert_numbers': True,
            'standardize_punctuation': True,
            'fix_capitalization': True
        }
        
        normalizer = TextNormalizer(config)
        input_text = "um, today we discuss chapter two"
        result = normalizer.normalize_text(input_text)
        
        # Should still contain "um" since filler removal is disabled
        assert "um" in result.lower()
        assert "2" in result  # Numbers should still be converted
    
    def test_normalization_edge_cases(self):
        """Test normalization edge cases."""
        edge_cases = [
            ("", ""),  # Empty string
            ("   ", ""),  # Whitespace only
            ("A", "A"),  # Single character
            ("123", "123"),  # Numbers only
            ("!!!", "!!!"),  # Punctuation only
            ("one two three four five", "1 2 3 4 5"),  # All numbers
            ("Um uh er ah oh", ""),  # All fillers
        ]
        
        for input_text, expected in edge_cases:
            result = self.normalizer.normalize_text(input_text)
            assert result == expected, f"Failed for '{input_text}'"
    
    def test_filler_word_boundaries(self):
        """Test that filler words respect word boundaries."""
        test_cases = [
            ("summarize this", "summarize this"),  # "um" in middle shouldn't be removed
            ("aluminum can", "aluminum can"),  # "um" in middle shouldn't be removed
            ("actually factorial", "factorial"),  # "actually" should be removed, "actual" preserved
        ]
        
        for input_text, expected in test_cases:
            result = self.normalizer.remove_filler_words(input_text)
            assert result == expected, f"Failed for '{input_text}'"
    
    def test_number_word_boundaries(self):
        """Test that number conversion respects word boundaries."""
        test_cases = [
            ("someone", "someone"),  # "one" in middle shouldn't be converted
            ("everything", "everything"),  # "everything" shouldn't become "every9"
            ("twenty-one", "twenty-1"),  # Hyphenated should work partially
        ]
        
        for input_text, expected in test_cases:
            result = self.normalizer.convert_numbers(input_text)
            assert result == expected, f"Failed for '{input_text}'"
    
    def test_case_insensitive_processing(self):
        """Test that processing works with different cases."""
        test_cases = [
            ("UM hello", "hello"),
            ("ACTUALLY this", "this"),
            ("ONE apple", "1 apple"),
            ("FIRST chapter", "1st chapter"),
        ]
        
        for input_text, expected in test_cases:
            result = self.normalizer.normalize_text(input_text)
            assert result.lower() == expected.lower(), f"Failed for '{input_text}'"


class TestNormalizationResult:
    """Test suite for NormalizationResult dataclass."""
    
    def test_normalization_result_creation(self):
        """Test creating NormalizationResult instances."""
        result = NormalizationResult(
            original_text="um hello world",
            normalized_text="hello world",
            changes_applied=["removed_filler_words"],
            word_count_before=3,
            word_count_after=2
        )
        
        assert result.original_text == "um hello world"
        assert result.normalized_text == "hello world"
        assert result.changes_applied == ["removed_filler_words"]
        assert result.word_count_before == 3
        assert result.word_count_after == 2
    
    def test_normalization_result_no_changes(self):
        """Test NormalizationResult when no changes are made."""
        result = NormalizationResult(
            original_text="hello world",
            normalized_text="hello world",
            changes_applied=[],
            word_count_before=2,
            word_count_after=2
        )
        
        assert result.original_text == result.normalized_text
        assert len(result.changes_applied) == 0
        assert result.word_count_before == result.word_count_after


class TestTextNormalizerConfiguration:
    """Test suite for TextNormalizer configuration options."""
    
    def test_custom_filler_words(self):
        """Test custom filler word configuration."""
        config = {
            'additional_fillers': ['custom_filler'],
            'excluded_fillers': ['um']  # Don't remove 'um'
        }
        
        normalizer = TextNormalizer(config)
        
        # Should remove custom filler but not 'um'
        result = normalizer.remove_filler_words("um custom_filler hello")
        assert "um" in result
        assert "custom_filler" not in result
        assert "hello" in result
    
    def test_disabled_features(self):
        """Test disabling specific normalization features."""
        config = {
            'remove_fillers': False,
            'convert_numbers': False,
            'standardize_punctuation': False,
            'fix_capitalization': False
        }
        
        normalizer = TextNormalizer(config)
        input_text = "um, chapter one is  good ."
        result = normalizer.normalize_text(input_text)
        
        # Should be unchanged since all features are disabled
        assert result == input_text
    
    def test_selective_feature_enabling(self):
        """Test enabling only specific features."""
        config = {
            'remove_fillers': True,
            'convert_numbers': False,
            'standardize_punctuation': False,
            'fix_capitalization': False
        }
        
        normalizer = TextNormalizer(config)
        input_text = "um, chapter one is  good ."
        result = normalizer.normalize_text(input_text)
        
        # Should only remove fillers
        assert "um" not in result
        assert "one" in result  # Numbers not converted
        assert "  " in result  # Spacing not fixed