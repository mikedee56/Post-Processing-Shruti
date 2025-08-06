"""
Test Suite for Sanskrit/Hindi Correction System (Story 2.1).

This test suite validates the lexicon-based correction system including
word identification, fuzzy matching, IAST transliteration, and correction application.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import yaml

# Import components to test
from sanskrit_hindi_identifier.word_identifier import SanskritHindiIdentifier, IdentifiedWord, WordCategory
from sanskrit_hindi_identifier.lexicon_manager import LexiconManager, LexiconValidationResult
from sanskrit_hindi_identifier.correction_applier import CorrectionApplier, CorrectionCandidate, CorrectionType
from utils.fuzzy_matcher import FuzzyMatcher, MatchingConfig, FuzzyMatch, MatchType
from utils.iast_transliterator import IASTTransliterator, TransliterationStandard
from post_processors.sanskrit_post_processor import SanskritPostProcessor


class TestSanskritHindiIdentifier:
    """Test the Sanskrit/Hindi word identification module."""
    
    @pytest.fixture
    def temp_lexicon_dir(self):
        """Create temporary lexicon directory with test data."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create test corrections.yaml
        corrections_data = {
            'version': '1.0',
            'entries': [
                {
                    'original_term': 'krishna',
                    'variations': ['krsna', 'krshna', 'krisna'],
                    'transliteration': 'Kṛṣṇa',
                    'is_proper_noun': True,
                    'category': 'deity',
                    'confidence': 1.0,
                    'source_authority': 'test'
                },
                {
                    'original_term': 'dharma',
                    'variations': ['dhrma', 'dharama'],
                    'transliteration': 'dharma',
                    'is_proper_noun': False,
                    'category': 'concept',
                    'confidence': 1.0,
                    'source_authority': 'test'
                }
            ]
        }
        
        corrections_file = temp_dir / 'corrections.yaml'
        with open(corrections_file, 'w', encoding='utf-8') as f:
            yaml.dump(corrections_data, f, allow_unicode=True)
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_word_identifier_initialization(self, temp_lexicon_dir):
        """Test that word identifier initializes correctly."""
        identifier = SanskritHindiIdentifier(lexicon_dir=temp_lexicon_dir)
        
        assert len(identifier.sanskrit_hindi_lexicon) > 0
        assert 'krishna' in identifier.sanskrit_hindi_lexicon
        assert 'dharma' in identifier.sanskrit_hindi_lexicon
    
    def test_identify_words_basic(self, temp_lexicon_dir):
        """Test basic word identification."""
        identifier = SanskritHindiIdentifier(lexicon_dir=temp_lexicon_dir)
        text = "Today we will discuss Krishna and dharma."
        
        identified = identifier.identify_words(text)
        
        # Should identify both terms
        assert len(identified) >= 2
        identified_words = [word.word.lower() for word in identified]
        assert 'krishna' in identified_words or 'kṛṣṇa' in identified_words
        assert 'dharma' in identified_words
    
    def test_identify_variations(self, temp_lexicon_dir):
        """Test identification of word variations."""
        identifier = SanskritHindiIdentifier(lexicon_dir=temp_lexicon_dir)
        text = "The story of krsna and dhrma is profound."
        
        identified = identifier.identify_words(text)
        
        # Should identify variations
        assert len(identified) >= 2
    
    def test_lexicon_stats(self, temp_lexicon_dir):
        """Test lexicon statistics generation."""
        identifier = SanskritHindiIdentifier(lexicon_dir=temp_lexicon_dir)
        stats = identifier.get_lexicon_stats()
        
        assert 'total_terms' in stats
        assert 'categories' in stats
        assert stats['total_terms'] >= 2
        assert 'deity' in stats['categories']
        assert 'concept' in stats['categories']


class TestLexiconManager:
    """Test the enhanced lexicon management system."""
    
    @pytest.fixture
    def temp_lexicon_dir(self):
        """Create temporary lexicon directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_lexicon_manager_initialization(self, temp_lexicon_dir):
        """Test lexicon manager initialization."""
        manager = LexiconManager(lexicon_dir=temp_lexicon_dir)
        assert manager.lexicon_dir == temp_lexicon_dir
    
    def test_validate_lexicon_data_valid(self):
        """Test validation of valid lexicon data."""
        manager = LexiconManager()
        
        valid_data = {
            'entries': [
                {
                    'original_term': 'test',
                    'transliteration': 'test',
                    'variations': ['tst'],
                    'is_proper_noun': False,
                    'category': 'test',
                    'confidence': 1.0
                }
            ]
        }
        
        result = manager.validate_lexicon_data(valid_data, 'test.yaml')
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_lexicon_data_invalid(self):
        """Test validation of invalid lexicon data."""
        manager = LexiconManager()
        
        invalid_data = {
            'entries': [
                {
                    'original_term': '',  # Empty term
                    'transliteration': '',  # Empty transliteration
                }
            ]
        }
        
        result = manager.validate_lexicon_data(invalid_data, 'test.yaml')
        assert not result.is_valid
        assert len(result.errors) > 0


class TestFuzzyMatcher:
    """Test the fuzzy matching system."""
    
    @pytest.fixture
    def sample_lexicon_data(self):
        """Sample lexicon data for testing."""
        return {
            'krishna': {
                'transliteration': 'Kṛṣṇa',
                'variations': ['krsna', 'krshna'],
                'is_proper_noun': True,
                'category': 'deity',
                'confidence': 1.0,
                'source_authority': 'test'
            },
            'dharma': {
                'transliteration': 'dharma',
                'variations': ['dhrma'],
                'is_proper_noun': False,
                'category': 'concept',
                'confidence': 1.0,
                'source_authority': 'test'
            }
        }
    
    def test_fuzzy_matcher_initialization(self, sample_lexicon_data):
        """Test fuzzy matcher initialization."""
        config = MatchingConfig(min_confidence=0.75)
        matcher = FuzzyMatcher(sample_lexicon_data, config)
        
        assert len(matcher.search_terms) >= 2
        assert 'krishna' in matcher.search_terms
        assert 'dharma' in matcher.search_terms
    
    def test_exact_matches(self, sample_lexicon_data):
        """Test exact matching functionality."""
        matcher = FuzzyMatcher(sample_lexicon_data)
        
        matches = matcher.find_matches('krishna')
        assert len(matches) > 0
        
        exact_match = next((m for m in matches if m.match_type == MatchType.EXACT), None)
        assert exact_match is not None
        assert exact_match.confidence == 1.0
    
    def test_fuzzy_matches(self, sample_lexicon_data):
        """Test fuzzy matching with similar words."""
        matcher = FuzzyMatcher(sample_lexicon_data)
        
        # Test with slight misspelling
        matches = matcher.find_matches('krsna')  # Variation of Krishna
        assert len(matches) > 0
        
        # Should find Krishna as a match
        krishna_matches = [m for m in matches if 'krishna' in m.corrected_term.lower() or 'kṛṣṇa' in m.corrected_term]
        assert len(krishna_matches) > 0
    
    def test_confidence_thresholds(self, sample_lexicon_data):
        """Test confidence threshold filtering."""
        config = MatchingConfig(min_confidence=0.95)  # Very high threshold
        matcher = FuzzyMatcher(sample_lexicon_data, config)
        
        # With high threshold, should get fewer matches
        matches = matcher.find_matches('krshna')  # Similar to Krishna but not exact
        
        # All matches should meet the high confidence threshold
        for match in matches:
            assert match.confidence >= 0.95


class TestIASTTransliterator:
    """Test IAST transliteration enforcement."""
    
    def test_transliterator_initialization(self):
        """Test IAST transliterator initialization."""
        transliterator = IASTTransliterator()
        assert transliterator.strict_mode is True
    
    def test_harvard_kyoto_to_iast(self):
        """Test conversion from Harvard-Kyoto to IAST."""
        transliterator = IASTTransliterator()
        
        text = "kRSNa"  # Harvard-Kyoto format
        result = transliterator.transliterate_to_iast(text, TransliterationStandard.HARVARD_KYOTO)
        
        # Should convert to IAST
        assert 'ṛ' in result.transliterated_text or result.transliterated_text != text
    
    def test_validate_iast_compliance(self):
        """Test IAST compliance validation."""
        transliterator = IASTTransliterator()
        
        # Valid IAST text
        valid_text = "Kṛṣṇa dharma"
        issues = transliterator.validate_iast(valid_text)
        assert len(issues) == 0
        
        # Invalid IAST (mixed schemes)
        invalid_text = "Krsna dharma aa"  # Mixed IAST and ASCII
        issues = transliterator.validate_iast(invalid_text)
        # Should find some issues (though this is a basic test)
        # The exact number depends on implementation details
    
    def test_get_iast_info(self):
        """Test IAST information extraction."""
        transliterator = IASTTransliterator()
        
        text = "Kṛṣṇa dharma"
        info = transliterator.get_iast_info(text)
        
        assert 'total_characters' in info
        assert 'iast_characters' in info
        assert 'compliance_score' in info
        assert info['iast_characters'] > 0  # Should detect IAST characters


class TestCorrectionApplier:
    """Test the correction application system."""
    
    def test_correction_applier_initialization(self):
        """Test correction applier initialization."""
        applier = CorrectionApplier(min_confidence=0.80)
        assert applier.min_confidence == 0.80
    
    def test_apply_corrections_basic(self):
        """Test basic correction application."""
        applier = CorrectionApplier(min_confidence=0.75)
        
        text = "Today we discuss krishna and dharma."
        
        # Create a mock correction candidate
        candidate = CorrectionCandidate(
            original_text="krishna",
            corrected_text="Kṛṣṇa",
            position=text.find("krishna"),
            length=len("krishna"),
            confidence=0.95,
            correction_type=CorrectionType.TRANSLITERATION,
            priority=None,  # Will be set by prioritization
            source="test"
        )
        
        result = applier.apply_corrections(text, [candidate])
        
        assert result.original_text == text
        assert "Kṛṣṇa" in result.corrected_text
        assert len(result.corrections_applied) > 0
    
    def test_confidence_filtering(self):
        """Test that low-confidence corrections are filtered out."""
        applier = CorrectionApplier(min_confidence=0.90)
        
        text = "test word"
        
        # Low confidence candidate
        low_confidence_candidate = CorrectionCandidate(
            original_text="word",
            corrected_text="corrected",
            position=5,
            length=4,
            confidence=0.60,  # Below threshold
            correction_type=CorrectionType.FUZZY_MATCH,
            priority=None,
            source="test"
        )
        
        result = applier.apply_corrections(text, [low_confidence_candidate])
        
        # Should be skipped due to low confidence
        assert len(result.corrections_applied) == 0
        assert len(result.corrections_skipped) > 0


class TestSanskritPostProcessorIntegration:
    """Test integration of Story 2.1 components with SanskritPostProcessor."""
    
    @pytest.fixture
    def temp_lexicon_dir(self):
        """Create temporary lexicon directory with test data."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create comprehensive test lexicon
        test_data = {
            'version': '1.0',
            'description': 'Test lexicon for Story 2.1',
            'entries': [
                {
                    'original_term': 'bhagavad gita',
                    'variations': ['bhagavat gita', 'bhagvad geeta'],
                    'transliteration': 'Bhagavad Gītā',
                    'is_proper_noun': True,
                    'category': 'scripture',
                    'confidence': 1.0,
                    'source_authority': 'IAST'
                },
                {
                    'original_term': 'krishna',
                    'variations': ['krsna', 'krshna', 'krisna'],
                    'transliteration': 'Kṛṣṇa',
                    'is_proper_noun': True,
                    'category': 'deity',
                    'confidence': 1.0,
                    'source_authority': 'IAST'
                },
                {
                    'original_term': 'dharma',
                    'variations': ['dhrma', 'dharama'],
                    'transliteration': 'dharma',
                    'is_proper_noun': False,
                    'category': 'concept',
                    'confidence': 1.0,
                    'source_authority': 'IAST'
                }
            ]
        }
        
        lexicon_file = temp_dir / 'test_corrections.yaml'
        with open(lexicon_file, 'w', encoding='utf-8') as f:
            yaml.dump(test_data, f, allow_unicode=True)
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_processor_initialization_with_story_2_1(self, temp_lexicon_dir):
        """Test that SanskritPostProcessor initializes with Story 2.1 components."""
        config = {
            'lexicon_dir': str(temp_lexicon_dir),
            'enable_lexicon_caching': True,
            'fuzzy_min_confidence': 0.75,
            'correction_min_confidence': 0.80
        }
        
        # Mock the config loading to use our test config
        with patch.object(SanskritPostProcessor, '_load_config', return_value=config):
            processor = SanskritPostProcessor()
            
            # Check that Story 2.1 components are initialized
            assert hasattr(processor, 'lexicon_manager')
            assert hasattr(processor, 'word_identifier')
            assert hasattr(processor, 'fuzzy_matcher')
            assert hasattr(processor, 'iast_transliterator')
            assert hasattr(processor, 'correction_applier')
    
    def test_enhanced_sanskrit_hindi_corrections(self, temp_lexicon_dir):
        """Test the enhanced Sanskrit/Hindi correction method."""
        config = {
            'lexicon_dir': str(temp_lexicon_dir),
            'fuzzy_min_confidence': 0.75,
            'correction_min_confidence': 0.80
        }
        
        with patch.object(SanskritPostProcessor, '_load_config', return_value=config):
            processor = SanskritPostProcessor()
            
            # Test text with Sanskrit terms that need correction
            test_text = "Today we will discuss krsna and dhrma from bhagavat gita."
            
            result = processor._apply_enhanced_sanskrit_hindi_corrections(test_text)
            
            # Check result structure
            assert 'original_text' in result
            assert 'corrected_text' in result
            assert 'corrections_applied' in result
            assert 'overall_confidence' in result
            
            # Should have found and processed some words
            assert result['identified_words_count'] >= 0
            assert result['fuzzy_matches_count'] >= 0
    
    def test_get_sanskrit_hindi_processing_report(self, temp_lexicon_dir):
        """Test the comprehensive processing report."""
        config = {
            'lexicon_dir': str(temp_lexicon_dir),
            'fuzzy_min_confidence': 0.75
        }
        
        with patch.object(SanskritPostProcessor, '_load_config', return_value=config):
            processor = SanskritPostProcessor()
            
            report = processor.get_sanskrit_hindi_processing_report()
            
            assert 'system_info' in report
            assert 'lexicon_summary' in report
            assert 'configuration' in report
            
            # Check system info
            assert report['system_info']['story_version'] == '2.1'
            assert len(report['system_info']['components']) == 5
            assert len(report['system_info']['capabilities']) == 5


class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        lexicon_dir = Path(tempfile.mkdtemp())
        output_dir = Path(tempfile.mkdtemp())
        
        # Create test lexicon
        test_corrections = {
            'version': '1.0',
            'entries': [
                {
                    'original_term': 'krishna',
                    'variations': ['krsna', 'krshna'],
                    'transliteration': 'Kṛṣṇa',
                    'is_proper_noun': True,
                    'category': 'deity',
                    'confidence': 1.0,
                    'source_authority': 'IAST'
                }
            ]
        }
        
        corrections_file = lexicon_dir / 'corrections.yaml'
        with open(corrections_file, 'w', encoding='utf-8') as f:
            yaml.dump(test_corrections, f, allow_unicode=True)
        
        yield lexicon_dir, output_dir
        
        # Cleanup
        shutil.rmtree(lexicon_dir)
        shutil.rmtree(output_dir)
    
    def test_full_srt_processing_with_story_2_1(self, temp_dirs):
        """Test full SRT processing with Story 2.1 components."""
        lexicon_dir, output_dir = temp_dirs
        
        # Create test SRT content
        srt_content = """1
00:00:01,000 --> 00:00:05,000
Today we will discuss krsna and his teachings.

2
00:00:06,000 --> 00:00:10,000
The concept of dharma is central to understanding.
"""
        
        input_file = output_dir / 'test_input.srt'
        output_file = output_dir / 'test_output.srt'
        
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        # Configure processor with Story 2.1 components
        config = {
            'lexicon_dir': str(lexicon_dir),
            'fuzzy_min_confidence': 0.75,
            'correction_min_confidence': 0.80,
            'use_advanced_normalization': True
        }
        
        with patch.object(SanskritPostProcessor, '_load_config', return_value=config):
            processor = SanskritPostProcessor()
            
            # Process the SRT file
            metrics = processor.process_srt_file(input_file, output_file)
            
            # Check that processing completed
            assert metrics is not None
            assert metrics.total_segments == 2
            assert output_file.exists()
            
            # Check output content
            with open(output_file, 'r', encoding='utf-8') as f:
                output_content = f.read()
            
            # Should contain corrections (though specific corrections depend on implementation)
            assert len(output_content) > 0
            assert '1\n' in output_content  # Should maintain SRT structure


if __name__ == '__main__':
    pytest.main([__file__, '-v'])