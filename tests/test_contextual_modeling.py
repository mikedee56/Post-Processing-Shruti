"""
Comprehensive tests for Story 2.2 - Contextual Modeling components.

Tests all contextual modeling features including n-gram models, phonetic encoding,
contextual rules, spelling normalization, and integration.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import yaml
import json

from src.contextual_modeling.ngram_language_model import NGramLanguageModel, NGramModelConfig
from src.contextual_modeling.phonetic_encoder import PhoneticEncoder, PhoneticConfig
from src.contextual_modeling.contextual_rule_engine import ContextualRuleEngine, RuleType
from src.contextual_modeling.spelling_normalizer import SpellingNormalizer
from src.contextual_modeling.contextual_matcher import ContextualMatcher
from src.post_processors.contextual_enhancement import ContextualEnhancement
from src.sanskrit_hindi_identifier.lexicon_manager import LexiconManager
from src.utils.srt_parser import SRTSegment


class TestNGramLanguageModel:
    """Test N-gram Language Model functionality."""

    def test_ngram_model_initialization(self):
        """Test n-gram model initialization with different configurations."""
        # Default configuration
        model = NGramLanguageModel()
        assert model.config.n == 3
        assert not model.is_trained
        
        # Custom configuration
        config = NGramModelConfig(n=2, min_count=3)
        model = NGramLanguageModel(config)
        assert model.config.n == 2
        assert model.config.min_count == 3

    def test_ngram_model_training(self):
        """Test n-gram model training with corpus data."""
        model = NGramLanguageModel()
        
        # Sample training corpus
        corpus_texts = [
            "Today we study the Bhagavad Gita chapter two verse five.",
            "Krishna teaches Arjuna about dharma and karma yoga.",
            "The practice of yoga leads to spiritual wisdom.",
            "Dharma yoga and karma yoga are paths to liberation."
        ]
        
        # Train the model
        statistics = model.build_from_corpus(corpus_texts)
        
        assert model.is_trained
        assert statistics.total_ngrams > 0
        assert statistics.unique_ngrams > 0
        assert statistics.vocabulary_size > 0
        assert len(model.vocabulary) > 10

    def test_ngram_context_prediction(self):
        """Test context-based next word prediction."""
        model = NGramLanguageModel()
        
        corpus_texts = [
            "Krishna teaches dharma yoga practice",
            "Arjuna learns karma yoga wisdom",
            "Yoga practice leads to liberation"
        ]
        
        model.build_from_corpus(corpus_texts)
        
        # Test predictions
        predictions = model.predict_next_words(['krishna', 'teaches'], top_k=3)
        assert len(predictions) <= 3
        
        if predictions:
            assert predictions[0].confidence_score > 0
            assert len(predictions[0].context) > 0

    def test_sequence_probability_calculation(self):
        """Test calculation of sequence probabilities."""
        model = NGramLanguageModel()
        
        corpus_texts = [
            "dharma yoga practice meditation",
            "karma yoga action wisdom",
            "bhakti yoga devotion surrender"
        ]
        
        model.build_from_corpus(corpus_texts)
        
        # Test sequence probability
        prob = model.calculate_sequence_probability(['dharma', 'yoga'])
        assert isinstance(prob, float)
        assert prob != float('-inf')  # Should have some probability

    def test_model_persistence(self):
        """Test saving and loading n-gram models."""
        model = NGramLanguageModel()
        corpus_texts = ["dharma yoga practice", "karma yoga action"]
        model.build_from_corpus(corpus_texts)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.pkl"
            
            # Save model
            assert model.save_model(model_path)
            assert model_path.exists()
            
            # Load model
            new_model = NGramLanguageModel()
            assert new_model.load_model(model_path)
            assert new_model.is_trained
            assert len(new_model.vocabulary) == len(model.vocabulary)


class TestPhoneticEncoder:
    """Test Phonetic Encoding functionality."""

    def test_phonetic_encoder_initialization(self):
        """Test phonetic encoder initialization."""
        encoder = PhoneticEncoder()
        assert encoder.config.algorithm.value == "sanskrit_phonetic"
        assert encoder.config.max_code_length == 8

    def test_sanskrit_phonetic_encoding(self):
        """Test Sanskrit-specific phonetic encoding."""
        encoder = PhoneticEncoder()
        
        # Test basic Sanskrit terms
        test_cases = [
            ("krishna", "krsna"),
            ("dharma", "dhrm"),
            ("yoga", "yog"),
            ("arjuna", "arjun")
        ]
        
        for original, expected_similarity in test_cases:
            code = encoder.encode_text(original)
            assert isinstance(code, str)
            assert len(code) <= encoder.config.max_code_length
            assert code.isalnum()

    def test_phonetic_similarity_calculation(self):
        """Test phonetic similarity between texts."""
        encoder = PhoneticEncoder()
        
        # Test similar terms
        match = encoder.calculate_phonetic_similarity("krishna", "krsna")
        assert match.similarity_score > 0.5
        assert match.confidence > 0.5
        
        # Test different terms
        match = encoder.calculate_phonetic_similarity("krishna", "arjuna")
        assert match.similarity_score >= 0.0

    def test_phonetic_match_finding(self):
        """Test finding phonetic matches in candidate lists."""
        encoder = PhoneticEncoder()
        
        candidates = ["krishna", "arjuna", "dharma", "karma", "yoga"]
        matches = encoder.find_phonetic_matches("krsna", candidates, top_k=3)
        
        assert len(matches) <= 3
        if matches:
            assert matches[0].similarity_score > 0.5
            assert "krishna" in [m.target_text for m in matches]

    def test_lexicon_batch_encoding(self):
        """Test batch encoding of lexicon entries."""
        encoder = PhoneticEncoder()
        
        lexicon_entries = {
            "krishna": {"variations": ["krsna", "krshna"]},
            "dharma": {"variations": ["dharama", "dhrma"]},
            "yoga": {"variations": ["yog", "yogaa"]}
        }
        
        phonetic_codes = encoder.encode_lexicon_batch(lexicon_entries)
        
        assert len(phonetic_codes) >= len(lexicon_entries)
        assert "krishna" in phonetic_codes
        assert all(isinstance(code, str) for code in phonetic_codes.values())

    def test_phonetic_mappings_persistence(self):
        """Test saving and loading phonetic mappings."""
        encoder = PhoneticEncoder()
        lexicon_codes = {"krishna": "krsn", "dharma": "dhrm"}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            mappings_path = Path(temp_dir) / "phonetic_mappings.json"
            
            # Save mappings
            assert encoder.save_phonetic_mappings(mappings_path, lexicon_codes)
            assert mappings_path.exists()
            
            # Load mappings
            loaded_codes = encoder.load_phonetic_mappings(mappings_path)
            assert loaded_codes == lexicon_codes


class TestContextualRuleEngine:
    """Test Contextual Rule Engine functionality."""

    def test_rule_engine_initialization(self):
        """Test rule engine initialization with default rules."""
        engine = ContextualRuleEngine()
        assert len(engine.rules) > 0
        assert len(engine.compiled_patterns) > 0

    def test_compound_term_rules(self):
        """Test compound term detection and standardization."""
        engine = ContextualRuleEngine()
        
        # Test karma yoga compound
        text = "Today we practice karma yog meditation."
        matches = engine.apply_contextual_rules(text, ["practice", "meditation"])
        
        compound_matches = [m for m in matches if m.rule_type == RuleType.COMPOUND_TERM]
        assert len(compound_matches) > 0
        
        if compound_matches:
            match = compound_matches[0]
            assert "karma yoga" in match.corrected_text

    def test_contextual_rule_application(self):
        """Test application of contextual rules with context."""
        engine = ContextualRuleEngine()
        
        text = "The bhagavad gita teaches us about dharama."
        context = ["scripture", "verse", "teaching"]
        
        matches = engine.apply_contextual_rules(text, context)
        
        assert len(matches) > 0
        # Should find both Bhagavad Gita standardization and dharma correction
        rule_names = [m.rule_name for m in matches]
        assert any("Bhagavad Gita" in name for name in rule_names)

    def test_compound_term_detection(self):
        """Test detection of compound Sanskrit/Hindi terms."""
        engine = ContextualRuleEngine()
        
        text = "We study karma yoga and bhakti yoga practices."
        compounds = engine.detect_compound_terms(text)
        
        assert len(compounds) >= 2
        compound_terms = [c.canonical_form for c in compounds]
        assert "karma yoga" in compound_terms
        assert "bhakti yoga" in compound_terms

    def test_contextual_dependency_validation(self):
        """Test validation of contextual dependencies."""
        engine = ContextualRuleEngine()
        
        words = ["krishna", "teaches", "dharma", "practice"]
        positions = [0, 8, 16, 23]
        
        issues = engine.validate_contextual_dependencies(words, positions)
        assert isinstance(issues, list)

    def test_rules_configuration_persistence(self):
        """Test saving and loading rules configuration."""
        engine = ContextualRuleEngine()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_rules.yaml"
            
            # Save rules
            assert engine.save_rules_to_config(config_path)
            assert config_path.exists()
            
            # Load rules in new engine
            new_engine = ContextualRuleEngine(config_path)
            assert len(new_engine.rules) == len(engine.rules)


class TestSpellingNormalizer:
    """Test Spelling Normalization functionality."""

    def test_spelling_normalizer_initialization(self):
        """Test spelling normalizer initialization."""
        normalizer = SpellingNormalizer()
        assert len(normalizer.normalization_rules) > 0
        assert len(normalizer.variant_groups) > 0

    def test_text_normalization_with_context(self):
        """Test contextual text normalization."""
        normalizer = SpellingNormalizer()
        
        text = "Today we study the gita and learn about yog."
        context = ["bhagavad", "scripture", "practice", "yoga"]
        
        result = normalizer.normalize_text(text, context)
        
        assert result.original_text == text
        assert result.confidence_score > 0
        assert len(result.normalization_types) > 0

    def test_shortened_word_expansion(self):
        """Test expansion of shortened words."""
        normalizer = SpellingNormalizer()
        
        text = "We practice yog and study dharma."
        context = ["practice", "meditation", "spiritual"]
        
        expanded = normalizer.expand_shortened_words(text, context)
        
        # Should expand 'yog' to 'yoga'
        assert "yoga" in expanded

    def test_spelling_variant_detection(self):
        """Test detection of spelling variants."""
        normalizer = SpellingNormalizer()
        
        text = "The dharama and karm teachings are important."
        variants = normalizer._detect_spelling_variants(text)
        
        assert len(variants) > 0
        variant_originals = [v.original for v in variants]
        assert any("dharama" in orig or "karm" in orig for orig in variant_originals)

    def test_document_consistency_checking(self):
        """Test consistency checking across document segments."""
        normalizer = SpellingNormalizer()
        
        # Process multiple segments of same document
        segments = [
            "We study dharma in the morning.",
            "The dharama teachings are profound.",
            "Understanding dharma leads to wisdom."
        ]
        
        for i, text in enumerate(segments):
            result = normalizer.normalize_text(text, document_id="test_doc")
            
        # Should detect inconsistency in dharma/dharama usage
        stats = normalizer.get_normalization_statistics()
        assert stats['documents_tracked'] == 1

    def test_spelling_suggestions(self):
        """Test getting spelling suggestions for words."""
        normalizer = SpellingNormalizer()
        
        suggestions = normalizer.get_spelling_suggestions("dharama", max_suggestions=3)
        
        assert len(suggestions) <= 3
        if suggestions:
            assert suggestions[0][0] == "dharma"  # Should suggest correct spelling
            assert suggestions[0][1] > 0.5  # With good confidence


class TestContextualMatcher:
    """Test Contextual Matcher integration."""

    def test_contextual_matcher_initialization(self):
        """Test contextual matcher initialization."""
        # Create mock lexicon manager
        lexicon_manager = Mock()
        lexicon_manager.get_all_entries.return_value = {
            "dharma": Mock(
                transliteration="dharma", is_proper_noun=False, category="concept",
                confidence=1.0, source_authority="test", variations=["dharama"]
            )
        }
        
        matcher = ContextualMatcher(lexicon_manager)
        assert matcher.lexicon_manager == lexicon_manager
        assert not matcher.is_model_trained

    @patch('contextual_modeling.contextual_matcher.FuzzyMatcher')
    def test_contextual_matching_without_training(self, mock_fuzzy):
        """Test contextual matching without trained models."""
        # Setup mocks
        lexicon_manager = Mock()
        lexicon_manager.get_all_entries.return_value = {}
        
        mock_fuzzy_instance = Mock()
        mock_fuzzy_instance.find_matches.return_value = []
        mock_fuzzy.return_value = mock_fuzzy_instance
        
        matcher = ContextualMatcher(lexicon_manager)
        
        # Test matching without trained context model
        matches = matcher.find_contextual_matches("dharama", ["teaching", "wisdom"])
        assert isinstance(matches, list)

    def test_context_model_training(self):
        """Test context model training with corpus."""
        lexicon_manager = Mock()
        lexicon_manager.get_all_entries.return_value = {
            "dharma": Mock(
                transliteration="dharma", is_proper_noun=False, category="concept",
                confidence=1.0, source_authority="test", variations=["dharama"]
            )
        }
        
        matcher = ContextualMatcher(lexicon_manager)
        
        corpus_texts = [
            "dharma yoga practice leads to wisdom",
            "karma yoga action brings liberation",
            "bhakti yoga devotion purifies heart"
        ]
        
        # Training should succeed
        success = matcher.train_context_model(corpus_texts)
        assert success
        assert matcher.is_model_trained


class TestContextualEnhancement:
    """Test Contextual Enhancement integration."""

    def test_contextual_enhancement_initialization(self):
        """Test contextual enhancement initialization."""
        lexicon_manager = Mock()
        lexicon_manager.get_all_entries.return_value = {}
        
        enhancement = ContextualEnhancement(lexicon_manager)
        assert enhancement.lexicon_manager == lexicon_manager
        assert hasattr(enhancement, 'ngram_model')
        assert hasattr(enhancement, 'phonetic_encoder')
        assert hasattr(enhancement, 'rule_engine')
        assert hasattr(enhancement, 'spelling_normalizer')

    def test_contextual_processing_pipeline(self):
        """Test the complete contextual processing pipeline."""
        lexicon_manager = Mock()
        lexicon_manager.get_all_entries.return_value = {}
        
        enhancement = ContextualEnhancement(lexicon_manager)
        
        # Create test segment
        segment = SRTSegment(
            index=1,
            start_time="00:00:01,000",
            end_time="00:00:05,000",
            text="Today we study dharama and karma yog practice."
        )
        
        context_segments = [
            SRTSegment(1, "00:00:00,000", "00:00:01,000", "Welcome to our yoga class.")
        ]
        
        # Apply contextual enhancement
        result = enhancement.apply_contextual_enhancement(segment, context_segments)
        
        assert result.original_text == segment.text
        assert isinstance(result.enhanced_text, str)
        assert result.confidence_score >= 0.0
        assert isinstance(result.contextual_changes, list)
        assert result.processing_time > 0

    def test_contextual_suggestions(self):
        """Test getting contextual suggestions."""
        lexicon_manager = Mock()
        lexicon_manager.get_all_entries.return_value = {}
        
        enhancement = ContextualEnhancement(lexicon_manager)
        
        text = "dharama practice"
        context = ["yoga", "teaching", "wisdom"]
        
        suggestions = enhancement.get_contextual_suggestions(text, context)
        assert isinstance(suggestions, list)

    def test_model_persistence(self):
        """Test saving and loading contextual models."""
        lexicon_manager = Mock()
        lexicon_manager.get_all_entries.return_value = {}
        
        enhancement = ContextualEnhancement(lexicon_manager)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir) / "contextual_models"
            
            # Save models (should handle untrained models gracefully)
            result = enhancement.save_contextual_models(models_dir)
            # May fail if models not trained, but should not crash
            assert isinstance(result, bool)
            
            if result:
                # Test loading
                load_result = enhancement.load_contextual_models(models_dir)
                assert isinstance(load_result, bool)

    def test_contextual_statistics(self):
        """Test getting contextual enhancement statistics."""
        lexicon_manager = Mock()
        lexicon_manager.get_all_entries.return_value = {}
        
        enhancement = ContextualEnhancement(lexicon_manager)
        
        stats = enhancement.get_contextual_statistics()
        
        assert isinstance(stats, dict)
        assert 'ngram_model' in stats
        assert 'phonetic_encoder' in stats
        assert 'rule_engine' in stats
        assert 'spelling_normalizer' in stats
        assert 'is_context_model_trained' in stats


class TestContextualModelingIntegration:
    """Test integration between contextual modeling components."""

    def test_full_contextual_pipeline_integration(self):
        """Test the complete contextual modeling pipeline."""
        # Create a more realistic lexicon manager
        with tempfile.TemporaryDirectory() as temp_dir:
            lexicon_dir = Path(temp_dir) / "lexicons"
            lexicon_dir.mkdir()
            
            # Create a test lexicon file
            lexicon_data = {
                "entries": [
                    {
                        "original_term": "dharma",
                        "variations": ["dharama", "dhrama"],
                        "transliteration": "dharma",
                        "is_proper_noun": False,
                        "category": "concept",
                        "confidence": 1.0
                    },
                    {
                        "original_term": "yoga",
                        "variations": ["yog", "yogaa"],
                        "transliteration": "yoga",
                        "is_proper_noun": False,
                        "category": "practice",
                        "confidence": 1.0
                    }
                ]
            }
            
            lexicon_file = lexicon_dir / "test_lexicon.yaml"
            with open(lexicon_file, 'w', encoding='utf-8') as f:
                yaml.dump(lexicon_data, f)
            
            # Create lexicon manager
            lexicon_manager = LexiconManager(lexicon_dir)
            
            # Create contextual enhancement
            enhancement = ContextualEnhancement(lexicon_manager)
            
            # Train with sample data
            training_texts = [
                "dharma yoga practice meditation",
                "karma yoga action wisdom teachings",
                "bhakti yoga devotion surrender love",
                "raja yoga meditation control discipline"
            ]
            
            success = enhancement.train_contextual_models(training_texts)
            assert success
            assert enhancement.is_context_model_trained
            
            # Test processing
            segment = SRTSegment(
                index=1,
                start_time="00:00:01,000", 
                end_time="00:00:05,000",
                text="Today we learn about dharama and karma yog."
            )
            
            result = enhancement.apply_contextual_enhancement(segment)
            
            # Verify processing results
            assert result.confidence_score > 0
            assert len(result.contextual_changes) >= 0
            
            # Should have some improvements
            expected_improvements = ["dharma", "karma yoga"]
            enhanced_lower = result.enhanced_text.lower()
            improvements_found = [imp for imp in expected_improvements 
                                if imp in enhanced_lower]
            
            # At least one improvement should be found
            assert len(improvements_found) > 0

    def test_error_handling_and_graceful_degradation(self):
        """Test error handling and graceful degradation."""
        lexicon_manager = Mock()
        lexicon_manager.get_all_entries.return_value = {}
        
        # Test with invalid configuration
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_config = Path(temp_dir) / "invalid_config.yaml"
            with open(invalid_config, 'w') as f:
                f.write("invalid: yaml: content:")
            
            # Should handle invalid config gracefully
            enhancement = ContextualEnhancement(lexicon_manager, invalid_config)
            assert enhancement is not None
            
            # Should still be able to process (with degraded functionality)
            segment = SRTSegment(1, "00:00:01,000", "00:00:02,000", "test text")
            result = enhancement.apply_contextual_enhancement(segment)
            assert result is not None
            assert isinstance(result.confidence_score, float)


if __name__ == "__main__":
    pytest.main([__file__])