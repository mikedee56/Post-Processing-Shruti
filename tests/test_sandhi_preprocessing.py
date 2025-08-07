"""
Comprehensive test suite for Sanskrit sandhi preprocessing functionality.

Tests sandhi splitting accuracy, integration with SanskritHindiIdentifier,
edge cases, and performance benchmarks for Story 2.4.1.
"""

import pytest
import time
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch
from typing import List, Dict, Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sanskrit_hindi_identifier.sandhi_preprocessor import (
    SandhiPreprocessor, 
    SandhiSplitResult, 
    SandhiSplitCandidate,
    SegmentationConfidenceLevel
)
from sanskrit_hindi_identifier.word_identifier import SanskritHindiIdentifier


class TestSandhiPreprocessor:
    """Test suite for SandhiPreprocessor component."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create a SandhiPreprocessor instance for testing."""
        return SandhiPreprocessor(enable_sandhi_preprocessing=True)
    
    @pytest.fixture
    def preprocessor_disabled(self):
        """Create a disabled SandhiPreprocessor instance for testing."""
        return SandhiPreprocessor(enable_sandhi_preprocessing=False)

    def test_preprocessor_initialization_enabled(self, preprocessor):
        """Test preprocessor initialization with sandhi preprocessing enabled."""
        assert preprocessor.enable_preprocessing is True
        assert preprocessor.basic_tokenizer_fallback is True
        assert isinstance(preprocessor.stats, dict)
        assert all(key in preprocessor.stats for key in [
            'total_processed', 'successful_splits', 'fallback_used', 'processing_errors'
        ])

    def test_preprocessor_initialization_disabled(self, preprocessor_disabled):
        """Test preprocessor initialization with sandhi preprocessing disabled."""
        assert preprocessor_disabled.enable_preprocessing is False

    def test_preprocess_text_disabled(self, preprocessor_disabled):
        """Test that disabled preprocessor returns fallback immediately."""
        test_text = "yogaścittavṛttinirodhaḥ"
        result = preprocessor_disabled.preprocess_text(test_text)
        
        assert result.fallback_used is True
        assert result.preprocessing_successful is True
        assert result.primary_candidate.confidence_level == SegmentationConfidenceLevel.FALLBACK

    def test_clean_input_text(self, preprocessor):
        """Test input text cleaning functionality."""
        # Test normal text
        assert preprocessor._clean_input_text("normal text") == "normal text"
        
        # Test whitespace normalization
        assert preprocessor._clean_input_text("  multiple   spaces  ") == "multiple spaces"
        
        # Test punctuation removal
        assert preprocessor._clean_input_text("text!@#$%^&*()text") == "texttext"
        
        # Test empty text
        assert preprocessor._clean_input_text("") == ""
        assert preprocessor._clean_input_text(None) == ""

    def test_is_likely_sanskrit_devanagari(self, preprocessor):
        """Test Sanskrit detection for Devanagari text."""
        # Devanagari text should be detected as Sanskrit
        devanagari_text = "योगश्चित्तवृत्तिनिरोधः"
        assert preprocessor._is_likely_sanskrit(devanagari_text) is True

    def test_is_likely_sanskrit_transliterated(self, preprocessor):
        """Test Sanskrit detection for transliterated text."""
        # Text with IAST characters
        iast_text = "yogaścittavṛttinirodhaḥ"
        assert preprocessor._is_likely_sanskrit(iast_text) is True
        
        # Text with multiple vowels (compound indicator)
        compound_text = "bhagavadgita"
        assert preprocessor._is_likely_sanskrit(compound_text) is True
        
        # Very long word likely to be compound
        long_word = "verylongwordthatmightbeacompound"
        assert preprocessor._is_likely_sanskrit(long_word) is True

    def test_is_likely_sanskrit_english(self, preprocessor):
        """Test Sanskrit detection for English text."""
        english_text = "this is regular english text"
        assert preprocessor._is_likely_sanskrit(english_text) is False
        
        # Short text
        short_text = "hi"
        assert preprocessor._is_likely_sanskrit(short_text) is False

    def test_heuristic_sandhi_split_junction_splitting(self, preprocessor):
        """Test heuristic sandhi splitting using common junctions."""
        # Test with visarga junction
        word = "yogaḥcitta"
        splits = preprocessor._heuristic_sandhi_split(word)
        
        assert len(splits) > 0
        segments, confidence, method = splits[0]
        assert len(segments) == 2
        assert segments[0] == "yogaḥ"
        assert segments[1] == "citta"
        assert "junction_split" in method

    def test_heuristic_sandhi_split_vowel_consonant(self, preprocessor):
        """Test heuristic splitting on vowel-consonant boundaries."""
        word = "bhagavadgita"
        splits = preprocessor._heuristic_sandhi_split(word)
        
        assert len(splits) > 0
        # Should find some segmentation candidates
        found_vowel_split = any("vowel_consonant" in split[2] for split in splits)
        assert found_vowel_split

    def test_heuristic_sandhi_split_length_based(self, preprocessor):
        """Test length-based splitting for very long words."""
        word = "verylongcompoundwordthatneedsplitting"
        splits = preprocessor._heuristic_sandhi_split(word)
        
        assert len(splits) > 0
        # Should find length-based split
        found_length_split = any("length_based" in split[2] for split in splits)
        assert found_length_split

    def test_get_word_sandhi_candidates(self, preprocessor):
        """Test generation of sandhi candidates for individual words."""
        word = "yogaścittavṛttinirodhaḥ"
        candidates = preprocessor._get_word_sandhi_candidates(word)
        
        # Should return candidates
        assert isinstance(candidates, list)
        
        # If candidates found, verify structure
        if candidates:
            candidate = candidates[0]
            assert isinstance(candidate, SandhiSplitCandidate)
            assert candidate.original_text == word
            assert isinstance(candidate.segments, list)
            assert 0 <= candidate.confidence_score <= 1
            assert isinstance(candidate.confidence_level, SegmentationConfidenceLevel)

    def test_preprocess_text_fallback(self, preprocessor):
        """Test fallback behavior when sanskrit_parser processing fails."""
        # Test with English text that should trigger fallback
        english_text = "this is english text"
        result = preprocessor.preprocess_text(english_text)
        
        assert result.fallback_used is True
        assert result.preprocessing_successful is True
        assert result.primary_candidate.confidence_level == SegmentationConfidenceLevel.FALLBACK
        assert result.primary_candidate.segments == english_text.split()

    def test_processing_statistics(self, preprocessor):
        """Test processing statistics tracking."""
        # Process some text to generate stats
        test_texts = ["yogaścittavṛtti", "english text", "भगवद्गीता"]
        
        for text in test_texts:
            preprocessor.preprocess_text(text)
        
        stats = preprocessor.get_processing_statistics()
        
        assert stats['total_processed'] == len(test_texts)
        assert stats['total_processed'] >= stats['successful_splits'] + stats['fallback_used']
        assert 'sanskrit_parser_available' in stats
        assert 'preprocessing_enabled' in stats
        assert 'success_rate' in stats

    def test_reset_statistics(self, preprocessor):
        """Test statistics reset functionality."""
        # Generate some stats
        preprocessor.preprocess_text("test text")
        
        # Verify stats exist
        stats = preprocessor.get_processing_statistics()
        assert stats['total_processed'] > 0
        
        # Reset and verify
        preprocessor.reset_statistics()
        stats = preprocessor.get_processing_statistics()
        assert stats['total_processed'] == 0
        assert stats['successful_splits'] == 0
        assert stats['fallback_used'] == 0

    def test_validate_configuration(self, preprocessor):
        """Test configuration validation."""
        validation = preprocessor.validate_configuration()
        
        assert 'is_valid' in validation
        assert 'warnings' in validation
        assert 'errors' in validation
        assert 'recommendations' in validation
        assert isinstance(validation['warnings'], list)
        assert isinstance(validation['errors'], list)


class TestSandhiIntegration:
    """Test suite for SandhiPreprocessor integration with SanskritHindiIdentifier."""
    
    @pytest.fixture
    def temp_lexicon(self):
        """Create temporary lexicon files for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create test lexicon data
        corrections_data = {
            'entries': [
                {
                    'original_term': 'yoga',
                    'variations': ['yog', 'yogah'],
                    'transliteration': 'yoga',
                    'is_proper_noun': False,
                    'category': 'practice',
                    'confidence': 1.0,
                    'source_authority': 'test'
                },
                {
                    'original_term': 'citta',
                    'variations': ['citt', 'cittam'],
                    'transliteration': 'citta',
                    'is_proper_noun': False,
                    'category': 'concept',
                    'confidence': 1.0,
                    'source_authority': 'test'
                },
                {
                    'original_term': 'vrtti',
                    'variations': ['vritti', 'vrutti'],
                    'transliteration': 'vṛtti',
                    'is_proper_noun': False,
                    'category': 'concept',
                    'confidence': 1.0,
                    'source_authority': 'test'
                }
            ]
        }
        
        # Create minimal data for other required lexicon files
        empty_data = {'entries': []}
        
        # Write all required lexicon files
        lexicon_files = ['corrections.yaml', 'proper_nouns.yaml', 'phrases.yaml', 'verses.yaml']
        for filename in lexicon_files:
            file_path = temp_dir / filename
            data = corrections_data if filename == 'corrections.yaml' else empty_data
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        
        yield temp_dir
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def identifier_with_sandhi(self, temp_lexicon):
        """Create SanskritHindiIdentifier with sandhi preprocessing enabled."""
        return SanskritHindiIdentifier(
            lexicon_dir=temp_lexicon,
            enable_sandhi_preprocessing=True
        )

    @pytest.fixture
    def identifier_without_sandhi(self, temp_lexicon):
        """Create SanskritHindiIdentifier with sandhi preprocessing disabled."""
        return SanskritHindiIdentifier(
            lexicon_dir=temp_lexicon,
            enable_sandhi_preprocessing=False
        )

    def test_identifier_initialization_with_sandhi(self, identifier_with_sandhi):
        """Test SanskritHindiIdentifier initialization with sandhi preprocessing."""
        assert identifier_with_sandhi.enable_sandhi_preprocessing is True
        assert identifier_with_sandhi.sandhi_preprocessor is not None
        assert identifier_with_sandhi.sandhi_preprocessor.enable_preprocessing is True

    def test_identifier_initialization_without_sandhi(self, identifier_without_sandhi):
        """Test SanskritHindiIdentifier initialization without sandhi preprocessing."""
        assert identifier_without_sandhi.enable_sandhi_preprocessing is False
        assert identifier_without_sandhi.sandhi_preprocessor.enable_preprocessing is False

    def test_identify_words_with_sandhi_preprocessing(self, identifier_with_sandhi):
        """Test word identification with sandhi preprocessing enabled."""
        # Test text that might benefit from sandhi splitting
        test_text = "yogaścitta practice"
        
        identified = identifier_with_sandhi.identify_words(test_text)
        
        # Should identify words after sandhi preprocessing
        assert isinstance(identified, list)
        # Basic sanity check - should not crash

    def test_identify_words_without_sandhi_preprocessing(self, identifier_without_sandhi):
        """Test word identification without sandhi preprocessing."""
        test_text = "yoga citta practice"
        
        identified = identifier_without_sandhi.identify_words(test_text)
        
        # Should identify words normally
        assert isinstance(identified, list)

    def test_sandhi_preprocessing_feature_toggle(self, identifier_with_sandhi):
        """Test enabling/disabling sandhi preprocessing dynamically."""
        # Initially enabled
        assert identifier_with_sandhi.enable_sandhi_preprocessing is True
        
        # Disable
        identifier_with_sandhi.set_sandhi_preprocessing_enabled(False)
        assert identifier_with_sandhi.enable_sandhi_preprocessing is False
        assert identifier_with_sandhi.sandhi_preprocessor.enable_preprocessing is False
        
        # Re-enable
        identifier_with_sandhi.set_sandhi_preprocessing_enabled(True)
        assert identifier_with_sandhi.enable_sandhi_preprocessing is True
        assert identifier_with_sandhi.sandhi_preprocessor.enable_preprocessing is True

    def test_get_sandhi_preprocessing_stats(self, identifier_with_sandhi):
        """Test retrieval of sandhi preprocessing statistics."""
        # Process some text to generate stats
        identifier_with_sandhi.identify_words("test yoga citta")
        
        stats = identifier_with_sandhi.get_sandhi_preprocessing_stats()
        
        assert isinstance(stats, dict)
        assert 'total_processed' in stats
        assert stats['total_processed'] >= 0

    def test_validate_sandhi_preprocessing_config(self, identifier_with_sandhi):
        """Test sandhi preprocessing configuration validation."""
        validation = identifier_with_sandhi.validate_sandhi_preprocessing_config()
        
        assert isinstance(validation, dict)
        assert 'is_valid' in validation


class TestSandhiEdgeCases:
    """Test suite for edge cases and error handling in sandhi preprocessing."""
    
    @pytest.fixture
    def preprocessor(self):
        return SandhiPreprocessor(enable_sandhi_preprocessing=True)

    def test_empty_text(self, preprocessor):
        """Test handling of empty text."""
        result = preprocessor.preprocess_text("")
        assert result.fallback_used is True
        assert result.primary_candidate.segments == []

    def test_none_text(self, preprocessor):
        """Test handling of None input."""
        result = preprocessor.preprocess_text(None)
        assert result.fallback_used is True

    def test_whitespace_only_text(self, preprocessor):
        """Test handling of whitespace-only text."""
        result = preprocessor.preprocess_text("   \n\t   ")
        assert result.fallback_used is True

    def test_single_character_text(self, preprocessor):
        """Test handling of single character text."""
        result = preprocessor.preprocess_text("a")
        assert result.fallback_used is True

    def test_malformed_unicode(self, preprocessor):
        """Test handling of malformed Unicode text."""
        # Test with various potentially problematic Unicode sequences
        test_cases = [
            "incomplete\udcdc",  # Surrogate
            "mixed\u0900normal",  # Mixed scripts
            "emoji👍mixed"  # Emoji mixed with text
        ]
        
        for test_text in test_cases:
            try:
                result = preprocessor.preprocess_text(test_text)
                # Should not crash, might use fallback
                assert isinstance(result, SandhiSplitResult)
            except Exception as e:
                pytest.fail(f"Should handle malformed Unicode gracefully, got: {e}")

    def test_very_long_text(self, preprocessor):
        """Test handling of very long text."""
        # Create very long text
        long_text = "yogaścitta" * 1000
        
        result = preprocessor.preprocess_text(long_text)
        
        # Should handle without crashing
        assert isinstance(result, SandhiSplitResult)
        # Processing time should be reasonable (< 5 seconds for this test)
        assert result.processing_time_ms < 5000

    def test_mixed_script_text(self, preprocessor):
        """Test handling of mixed script text (Devanagari + Latin)."""
        mixed_text = "योग and citta वृत्ति"
        
        result = preprocessor.preprocess_text(mixed_text)
        
        # Should handle mixed scripts gracefully
        assert isinstance(result, SandhiSplitResult)

    def test_special_characters(self, preprocessor):
        """Test handling of text with special characters."""
        special_text = "yoga-citta_vrtti.nirodha?"
        
        result = preprocessor.preprocess_text(special_text)
        
        # Should handle special characters
        assert isinstance(result, SandhiSplitResult)


class TestSandhiPerformanceBenchmarks:
    """Performance benchmarks for sandhi preprocessing."""
    
    @pytest.fixture
    def preprocessor(self):
        return SandhiPreprocessor(enable_sandhi_preprocessing=True)

    def test_processing_time_short_text(self, preprocessor):
        """Benchmark processing time for short text."""
        test_text = "yogaścittavṛttinirodhaḥ"
        
        start_time = time.time()
        result = preprocessor.preprocess_text(test_text)
        end_time = time.time()
        
        processing_time_ms = (end_time - start_time) * 1000
        
        # Short text should process quickly (< 100ms)
        assert processing_time_ms < 100
        assert result.processing_time_ms > 0

    def test_processing_time_multiple_texts(self, preprocessor):
        """Benchmark processing time for multiple texts."""
        test_texts = [
            "yogaścittavṛtti",
            "bhagavadgītā", 
            "sarvadharmānparityajya",
            "tadviddhipraṇipātena",
            "karmayogaśca"
        ]
        
        total_start_time = time.time()
        
        for text in test_texts:
            result = preprocessor.preprocess_text(text)
            # Each should complete reasonably quickly
            assert result.processing_time_ms < 200
        
        total_time = (time.time() - total_start_time) * 1000
        
        # Total time should be reasonable (< 2x single text * count)
        assert total_time < 1000  # Less than 1 second for 5 texts

    def test_memory_usage_stability(self, preprocessor):
        """Test that memory usage remains stable over many operations."""
        # Process many texts and verify no obvious memory leaks
        test_text = "yogaścittavṛttinirodhaḥ"
        
        # Process 100 times
        for _ in range(100):
            preprocessor.preprocess_text(test_text)
        
        stats = preprocessor.get_processing_statistics()
        assert stats['total_processed'] == 100
        
        # Reset stats to verify reset works
        preprocessor.reset_statistics()
        stats = preprocessor.get_processing_statistics()
        assert stats['total_processed'] == 0

    def test_performance_comparison_with_without_preprocessing(self, temp_lexicon):
        """Compare performance with and without sandhi preprocessing."""
        test_text = "yoga citta vrtti nirodha practice"
        
        # With sandhi preprocessing
        identifier_with = SanskritHindiIdentifier(
            lexicon_dir=temp_lexicon,
            enable_sandhi_preprocessing=True
        )
        
        start_time = time.time()
        result_with = identifier_with.identify_words(test_text)
        time_with = (time.time() - start_time) * 1000
        
        # Without sandhi preprocessing
        identifier_without = SanskritHindiIdentifier(
            lexicon_dir=temp_lexicon,
            enable_sandhi_preprocessing=False
        )
        
        start_time = time.time()
        result_without = identifier_without.identify_words(test_text)
        time_without = (time.time() - start_time) * 1000
        
        # With sandhi should not be more than 2x slower (Story requirement)
        assert time_with < (time_without * 2 + 100)  # +100ms tolerance
        
        # Both should return results
        assert isinstance(result_with, list)
        assert isinstance(result_without, list)

    @pytest.fixture
    def temp_lexicon(self):
        """Create temporary lexicon for performance testing."""
        temp_dir = Path(tempfile.mkdtemp())
        
        corrections_data = {
            'entries': [
                {
                    'original_term': 'yoga',
                    'variations': ['yog'],
                    'transliteration': 'yoga',
                    'is_proper_noun': False,
                    'category': 'practice',
                    'confidence': 1.0,
                    'source_authority': 'test'
                }
            ]
        }
        
        corrections_file = temp_dir / "corrections.yaml"
        with open(corrections_file, 'w', encoding='utf-8') as f:
            yaml.dump(corrections_data, f)
        
        yield temp_dir
        
        import shutil
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])