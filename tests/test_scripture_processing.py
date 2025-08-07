"""
Test Suite for Scripture Processing Module.

Comprehensive tests for verse identification, canonical text substitution,
IAST formatting, and verse selection capabilities.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scripture_processing.scripture_identifier import ScriptureIdentifier, VerseMatch, PassageType
from scripture_processing.canonical_text_manager import CanonicalTextManager, CanonicalVerse, ScriptureSource
from scripture_processing.verse_substitution_engine import VerseSubstitutionEngine, SubstitutionResult
from scripture_processing.scripture_validator import ScriptureValidator, ValidationResult
from scripture_processing.scripture_iast_formatter import ScriptureIASTFormatter, VerseFormatting
from scripture_processing.verse_selection_system import VerseSelectionSystem, SelectionStrategy
from scripture_processing.scripture_processor import ScriptureProcessor
from utils.srt_parser import SRTSegment


class TestScriptureIdentifier:
    """Test scripture passage identification functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.identifier = ScriptureIdentifier()
    
    def test_scripture_identifier_initialization(self):
        """Test proper initialization of scripture identifier."""
        assert self.identifier is not None
        assert hasattr(self.identifier, 'verse_entries')
        assert hasattr(self.identifier, 'canonical_texts')
    
    def test_identify_verse_passages(self):
        """Test identification of verse passages."""
        test_text = "Today we study avyakto 'yam acintyo 'yam avikāryo 'yam ucyate from Bhagavad Gita"
        
        matches = self.identifier.identify_scripture_passages(test_text)
        
        # Should identify at least some potential matches
        assert isinstance(matches, list)
        # Each match should have required attributes
        for match in matches:
            assert hasattr(match, 'original_text')
            assert hasattr(match, 'confidence_score')
            assert hasattr(match, 'passage_type')
    
    def test_verse_segmentation(self):
        """Test verse segmentation logic."""
        test_text = "First part of verse | second part of verse || End of verse"
        
        segments = self.identifier._segment_text_for_verses(test_text)
        
        assert len(segments) > 0
        # Should have proper segment structure
        for segment in segments:
            assert hasattr(segment, 'text')
            assert hasattr(segment, 'start_pos')
            assert hasattr(segment, 'end_pos')
            assert hasattr(segment, 'sanskrit_word_density')
    
    def test_sanskrit_density_calculation(self):
        """Test Sanskrit word density calculation."""
        sanskrit_text = "karma dharma yoga gita"
        english_text = "today we will discuss the topic"
        
        sanskrit_density = self.identifier._calculate_sanskrit_density(sanskrit_text)
        english_density = self.identifier._calculate_sanskrit_density(english_text)
        
        # Sanskrit text should have higher density
        assert sanskrit_density >= english_density
    
    def test_confidence_scoring(self):
        """Test confidence score calculation."""
        # Create mock segment and entry
        mock_segment = Mock()
        mock_segment.text = "avyakto 'yam acintyo 'yam"
        mock_segment.word_count = 4
        mock_segment.sanskrit_word_density = 0.8
        
        mock_entry = Mock()
        mock_entry.canonical_text = "avyakto 'yam acintyo 'yam avikāryo 'yam ucyate"
        
        similarity_scores = {'overall': 0.8, 'word_matches': 3, 'total_words': 4}
        
        confidence = self.identifier._calculate_confidence_score(
            mock_segment, mock_entry, similarity_scores
        )
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be reasonably high for good match


class TestCanonicalTextManager:
    """Test canonical text management functionality."""
    
    def setup_method(self):
        """Set up test fixtures with temporary scripture directory."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.scripture_dir = self.temp_dir / "scriptures"
        self.scripture_dir.mkdir(parents=True)
        
        # Create test scripture file
        test_scripture = {
            'metadata': {
                'source': 'bhagavad_gita',
                'version': '1.0',
                'authority': 'IAST'
            },
            'verses': [
                {
                    'chapter': 2,
                    'verse': 25,
                    'canonical_text': "avyakto 'yam acintyo 'yam avikāryo 'yam ucyate",
                    'transliteration': "avyakto 'yam acintyo 'yam avikāryo 'yam ucyate",
                    'source_authority': 'IAST',
                    'tags': ['soul', 'unchanging'],
                    'variations': ['bhagavad gita 2.25', 'gita chapter 2 verse 25']
                }
            ]
        }
        
        with open(self.scripture_dir / "bhagavad_gita.yaml", 'w') as f:
            yaml.dump(test_scripture, f)
        
        self.manager = CanonicalTextManager(scripture_dir=self.scripture_dir)
    
    def test_canonical_manager_initialization(self):
        """Test proper initialization of canonical text manager."""
        assert self.manager is not None
        assert len(self.manager.canonical_verses) > 0
        assert ScriptureSource.BHAGAVAD_GITA in self.manager.source_indexes
    
    def test_verse_lookup_by_text(self):
        """Test verse lookup by text content."""
        search_text = "avyakto yam acintyo"
        
        matches = self.manager.lookup_verse_by_text(search_text)
        
        assert isinstance(matches, list)
        # Should find the test verse
        if matches:
            assert matches[0].canonical_text is not None
            assert "avyakto" in matches[0].canonical_text
    
    def test_verse_candidate_generation(self):
        """Test verse candidate generation."""
        test_text = "avyakto yam"
        
        candidates = self.manager.get_verse_candidates(
            test_text, max_candidates=3
        )
        
        assert isinstance(candidates, list)
        assert len(candidates) <= 3
        # Each candidate should be a valid verse
        for candidate in candidates:
            assert hasattr(candidate, 'canonical_text')
            assert hasattr(candidate, 'source')
            assert hasattr(candidate, 'chapter')
    
    def test_statistics_generation(self):
        """Test statistics generation."""
        stats = self.manager.get_statistics()
        
        assert 'total_verses' in stats
        assert 'sources' in stats
        assert stats['total_verses'] >= 0
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)


class TestVerseSubstitutionEngine:
    """Test verse substitution functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock the dependencies
        self.mock_identifier = Mock(spec=ScriptureIdentifier)
        self.mock_manager = Mock(spec=CanonicalTextManager)
        self.mock_validator = Mock(spec=ScriptureValidator)
        
        self.engine = VerseSubstitutionEngine(
            scripture_identifier=self.mock_identifier,
            canonical_manager=self.mock_manager,
            validator=self.mock_validator
        )
    
    def test_substitution_engine_initialization(self):
        """Test proper initialization of substitution engine."""
        assert self.engine is not None
        assert self.engine.scripture_identifier is not None
        assert self.engine.canonical_manager is not None
    
    def test_verse_substitution_process(self):
        """Test basic verse substitution process."""
        test_text = "Today we study the verse about the unchanging soul"
        
        # Mock verse identification
        mock_match = Mock(spec=VerseMatch)
        mock_match.original_text = "verse about the unchanging soul"
        mock_match.confidence_score = 0.85
        mock_match.match_start = 20
        mock_match.match_end = 50
        mock_match.passage_type = PassageType.VERSE
        
        # Mock canonical entry
        mock_entry = Mock()
        mock_entry.canonical_text = "avyakto 'yam acintyo 'yam"
        mock_match.canonical_entry = mock_entry
        
        self.mock_identifier.identify_scripture_passages.return_value = [mock_match]
        
        # Mock canonical lookup
        mock_verse = Mock(spec=CanonicalVerse)
        mock_verse.canonical_text = "avyakto 'yam acintyo 'yam avikāryo 'yam ucyate"
        mock_verse.id = "bhagavad_gita_2_25"
        self.mock_manager.lookup_verse_by_reference.return_value = mock_verse
        
        result = self.engine.substitute_verses_in_text(test_text)
        
        assert isinstance(result, SubstitutionResult)
        assert result.original_text == test_text
        assert result.substituted_text is not None
    
    def test_substitution_preview(self):
        """Test substitution preview functionality."""
        test_text = "Sample verse text"
        
        # Mock the dependencies to return empty results for preview
        self.mock_identifier.identify_scripture_passages.return_value = []
        
        previews = self.engine.get_substitution_preview(test_text, max_operations=3)
        
        assert isinstance(previews, list)
        assert len(previews) <= 3


class TestScriptureValidator:
    """Test scripture validation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ScriptureValidator()
    
    def test_validator_initialization(self):
        """Test proper initialization of scripture validator."""
        assert self.validator is not None
        assert hasattr(self.validator, 'min_validation_confidence')
        assert hasattr(self.validator, 'consistency_patterns')
    
    def test_verse_identification_validation(self):
        """Test validation of verse identification."""
        # Create mock verse match
        mock_match = Mock(spec=VerseMatch)
        mock_match.original_text = "avyakto 'yam acintyo 'yam"
        mock_match.confidence_score = 0.85
        
        mock_entry = Mock()
        mock_entry.canonical_text = "avyakto 'yam acintyo 'yam avikāryo 'yam ucyate"
        mock_match.canonical_entry = mock_entry
        
        result = self.validator.validate_verse_identification(mock_match, "test context")
        
        assert isinstance(result, ValidationResult)
        assert hasattr(result, 'is_valid')
        assert hasattr(result, 'confidence_score')
        assert hasattr(result, 'issues')
    
    def test_substitution_validation(self):
        """Test validation of verse substitution."""
        original = "verse about unchanging soul"
        canonical = "avyakto 'yam acintyo 'yam avikāryo 'yam ucyate"
        
        mock_match = Mock(spec=VerseMatch)
        mock_match.confidence_score = 0.8
        
        result = self.validator.validate_substitution(original, canonical, mock_match)
        
        assert isinstance(result, ValidationResult)
        assert result.confidence_score >= 0.0
        assert result.confidence_score <= 1.0
    
    def test_word_overlap_calculation(self):
        """Test word overlap calculation."""
        text1 = "avyakto yam acintyo"
        text2 = "avyakto yam acintyo yam ucyate"
        
        overlap = self.validator._calculate_word_overlap(text1, text2)
        
        assert 0.0 <= overlap <= 1.0
        assert overlap > 0  # Should have some overlap
    
    def test_iast_compliance_validation(self):
        """Test IAST compliance validation."""
        iast_text = "avyakto 'yam acintyo 'yam avikāryo"
        non_iast_text = "avyakto yam acintyo yam avikāryo"
        
        iast_result = self.validator._validate_iast_compliance(iast_text)
        non_iast_result = self.validator._validate_iast_compliance(non_iast_text)
        
        assert isinstance(iast_result, dict)
        assert 'is_compliant' in iast_result
        assert 'has_iast_chars' in iast_result


class TestScriptureIASTFormatter:
    """Test IAST formatting functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = ScriptureIASTFormatter()
    
    def test_formatter_initialization(self):
        """Test proper initialization of IAST formatter."""
        assert self.formatter is not None
        assert self.formatter.iast_transliterator is not None
        assert hasattr(self.formatter, 'verse_formatting_rules')
    
    def test_verse_formatting_with_metadata(self):
        """Test verse formatting with metadata."""
        from scripture_processing.scripture_iast_formatter import VerseMetadata
        from scripture_processing.canonical_text_manager import ScriptureSource
        
        verse_text = "karma dharma yoga practice"
        metadata = VerseMetadata(
            source=ScriptureSource.BHAGAVAD_GITA,
            chapter=2,
            verse=47
        )
        
        result = self.formatter.format_verse_with_metadata(
            verse_text, metadata, VerseFormatting.ACADEMIC
        )
        
        assert result is not None
        assert hasattr(result, 'formatted_text')
        assert hasattr(result, 'academic_compliance')
        assert result.verse_metadata == metadata
    
    def test_consistency_validation(self):
        """Test cross-verse consistency validation."""
        # Create mock verse results
        mock_results = []
        for i in range(3):
            mock_result = Mock()
            mock_result.formatted_text = f"verse text {i} with dharma"
            mock_result.formatting_style = VerseFormatting.ACADEMIC
            
            mock_metadata = Mock()
            mock_metadata.source = ScriptureSource.BHAGAVAD_GITA
            mock_result.verse_metadata = mock_metadata
            
            mock_results.append(mock_result)
        
        consistency_report = self.formatter.validate_cross_verse_consistency(mock_results)
        
        assert isinstance(consistency_report, dict)
        assert 'is_consistent' in consistency_report
        assert 'total_verses' in consistency_report


class TestVerseSelectionSystem:
    """Test verse selection functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_manager = Mock(spec=CanonicalTextManager)
        self.mock_identifier = Mock(spec=ScriptureIdentifier)
        self.mock_validator = Mock(spec=ScriptureValidator)
        
        self.selection_system = VerseSelectionSystem(
            canonical_manager=self.mock_manager,
            scripture_identifier=self.mock_identifier,
            validator=self.mock_validator
        )
    
    def test_selection_system_initialization(self):
        """Test proper initialization of selection system."""
        assert self.selection_system is not None
        assert self.selection_system.selection_strategy is not None
        assert hasattr(self.selection_system, 'confidence_thresholds')
    
    def test_verse_candidate_ranking(self):
        """Test verse candidate ranking."""
        test_text = "sample verse text"
        
        # Create mock candidates
        mock_candidates = []
        for i in range(3):
            mock_verse = Mock(spec=CanonicalVerse)
            mock_verse.canonical_text = f"canonical text {i}"
            mock_verse.source_authority = "IAST"
            mock_verse.tags = ["tag1", "tag2"]
            mock_candidates.append(mock_verse)
        
        ranked_candidates = self.selection_system.rank_verse_candidates(
            test_text, mock_candidates
        )
        
        assert isinstance(ranked_candidates, list)
        assert len(ranked_candidates) <= len(mock_candidates)
        
        # Check ranking structure
        for candidate in ranked_candidates:
            assert hasattr(candidate, 'overall_score')
            assert hasattr(candidate, 'ranking')
            assert hasattr(candidate, 'selection_reasons')
    
    def test_best_verse_selection(self):
        """Test best verse selection process."""
        test_text = "sample verse text"
        
        # Mock the identification process
        self.mock_identifier.identify_scripture_passages.return_value = []
        self.mock_manager.get_verse_candidates.return_value = []
        
        result = self.selection_system.select_best_verse(test_text)
        
        assert result is not None
        assert hasattr(result, 'selected_verse')
        assert hasattr(result, 'confidence_level')
        assert hasattr(result, 'requires_human_review')
    
    def test_source_attribution(self):
        """Test source attribution generation."""
        mock_verse = Mock(spec=CanonicalVerse)
        mock_verse.source = Mock()
        mock_verse.source.value = "bhagavad_gita"
        mock_verse.chapter = 2
        mock_verse.verse = 47
        mock_verse.source_authority = "IAST"
        
        attribution = self.selection_system.add_source_attribution(
            mock_verse, format_style="academic"
        )
        
        assert isinstance(attribution, str)
        assert "2.47" in attribution
        assert len(attribution) > 0


class TestScriptureProcessor:
    """Test integrated scripture processing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = ScriptureProcessor()
    
    def test_processor_initialization(self):
        """Test proper initialization of scripture processor."""
        assert self.processor is not None
        assert self.processor.canonical_manager is not None
        assert self.processor.scripture_identifier is not None
        assert self.processor.substitution_engine is not None
        assert self.processor.validator is not None
        assert self.processor.iast_formatter is not None
        assert self.processor.selection_system is not None
    
    def test_text_processing_pipeline(self):
        """Test complete text processing pipeline."""
        test_text = "Today we study karma and dharma from the sacred texts"
        
        result = self.processor.process_text(test_text)
        
        assert result is not None
        assert hasattr(result, 'original_text')
        assert hasattr(result, 'processed_text')
        assert hasattr(result, 'verses_identified')
        assert hasattr(result, 'substitutions_made')
        assert hasattr(result, 'validation_passed')
        assert result.original_text == test_text
    
    def test_srt_segment_processing(self):
        """Test SRT segment processing."""
        test_segment = SRTSegment(
            index=1,
            start_time="00:00:01,000",
            end_time="00:00:05,000",
            text="Today we study the Bhagavad Gita verses about karma"
        )
        
        processed_segment, result = self.processor.process_srt_segment(test_segment)
        
        assert processed_segment is not None
        assert processed_segment.index == test_segment.index
        assert processed_segment.start_time == test_segment.start_time
        assert processed_segment.end_time == test_segment.end_time
        assert result is not None
    
    def test_processing_statistics(self):
        """Test processing statistics generation."""
        stats = self.processor.get_processing_statistics()
        
        assert isinstance(stats, dict)
        assert 'canonical_texts' in stats
        assert 'selection_system_config' in stats
        assert 'processing_config' in stats
    
    def test_preview_functionality(self):
        """Test verse substitution preview."""
        test_text = "Sample text with potential verses"
        
        previews = self.processor.preview_verse_substitutions(test_text, max_previews=3)
        
        assert isinstance(previews, list)
        assert len(previews) <= 3
    
    def test_configuration_updates(self):
        """Test configuration updates."""
        original_strategy = self.processor.selection_system.selection_strategy
        
        self.processor.configure_selection_strategy(
            SelectionStrategy.AUTOMATIC,
            {'auto_select': 0.95, 'human_review': 0.75}
        )
        
        assert self.processor.selection_system.selection_strategy == SelectionStrategy.AUTOMATIC
        assert self.processor.selection_system.auto_select_threshold == 0.95
        assert self.processor.selection_system.human_review_threshold == 0.75


class TestIntegrationScenarios:
    """Test integration scenarios across components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary scripture directory
        self.temp_dir = Path(tempfile.mkdtemp())
        self.scripture_dir = self.temp_dir / "scriptures"
        self.scripture_dir.mkdir(parents=True)
        
        # Create comprehensive test data
        self._create_test_scripture_data()
        
        # Initialize processor with test data
        self.processor = ScriptureProcessor(scripture_dir=self.scripture_dir)
    
    def _create_test_scripture_data(self):
        """Create comprehensive test scripture data."""
        bhagavad_gita_data = {
            'metadata': {
                'source': 'bhagavad_gita',
                'version': '1.0',
                'authority': 'IAST'
            },
            'verses': [
                {
                    'chapter': 2,
                    'verse': 47,
                    'canonical_text': "karmaṇy evādhikāras te mā phaleṣu kadācana",
                    'transliteration': "karmaṇy evādhikāras te mā phaleṣu kadācana",
                    'translation': "You have a right to action, but not to the fruits",
                    'source_authority': 'IAST',
                    'tags': ['karma', 'duty', 'action'],
                    'variations': ['karma yoga verse', 'bhagavad gita 2.47']
                }
            ]
        }
        
        with open(self.scripture_dir / "bhagavad_gita.yaml", 'w') as f:
            yaml.dump(bhagavad_gita_data, f)
    
    def test_end_to_end_verse_processing(self):
        """Test complete end-to-end verse processing."""
        # Test text that should trigger verse identification and substitution
        test_text = "Today we discuss the teaching about karma and duty from Chapter 2 verse 47 of the Gita"
        
        result = self.processor.process_text(test_text)
        
        # Verify processing occurred
        assert result is not None
        assert result.processing_metadata['steps_completed'] is not None
        
        # Should have attempted identification
        assert 'identification' in result.processing_metadata.get('steps_completed', [])
        
        # Text should be processed (may or may not be changed depending on matches)
        assert result.processed_text is not None
    
    def test_multiple_segment_consistency(self):
        """Test consistency across multiple segments."""
        test_segments = [
            SRTSegment(1, "00:00:01,000", "00:00:05,000", "First verse about karma"),
            SRTSegment(2, "00:00:06,000", "00:00:10,000", "Second verse about dharma"),
            SRTSegment(3, "00:00:11,000", "00:00:15,000", "Third verse about yoga")
        ]
        
        processed_segments = []
        processing_results = []
        
        for segment in test_segments:
            proc_segment, result = self.processor.process_srt_segment(segment)
            processed_segments.append(proc_segment)
            processing_results.append(result)
        
        # All segments should be processed
        assert len(processed_segments) == len(test_segments)
        assert len(processing_results) == len(test_segments)
        
        # Each result should have consistent structure
        for result in processing_results:
            assert hasattr(result, 'verses_identified')
            assert hasattr(result, 'validation_passed')
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        # Test with problematic input
        problematic_texts = [
            "",  # Empty text
            "   ",  # Whitespace only
            "Normal English text with no Sanskrit content",  # No verses
            "Text with very long" + " word" * 1000,  # Very long text
        ]
        
        for text in problematic_texts:
            try:
                result = self.processor.process_text(text)
                
                # Should handle gracefully
                assert result is not None
                assert hasattr(result, 'validation_passed')
                # For problematic input, might not pass validation but shouldn't crash
                
            except Exception as e:
                pytest.fail(f"Processing should handle problematic input gracefully: {e}")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)


# Pytest configuration and fixtures
@pytest.fixture
def sample_verse_data():
    """Provide sample verse data for tests."""
    return {
        'bhagavad_gita_2_47': {
            'text': "karmaṇy evādhikāras te mā phaleṣu kadācana",
            'translation': "You have a right to action, but not to the fruits",
            'tags': ['karma', 'duty', 'action']
        },
        'bhagavad_gita_2_25': {
            'text': "avyakto 'yam acintyo 'yam avikāryo 'yam ucyate",
            'translation': "This soul is said to be unmanifest, unthinkable and unchanging",
            'tags': ['soul', 'unchanging', 'eternal']
        }
    }


@pytest.fixture
def temp_scripture_dir():
    """Provide temporary scripture directory for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    import shutil
    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])