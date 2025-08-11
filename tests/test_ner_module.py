"""
Comprehensive test suite for the NER Module.

Tests all components of the Yoga Vedanta Named Entity Recognition system
including entity identification, classification, capitalization, and model management.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, 'src')

from ner_module.yoga_vedanta_ner import YogaVedantaNER, NamedEntity, NERResult, NERConfidenceLevel
from ner_module.entity_classifier import EntityClassifier, EntityCategory, ClassificationResult
from ner_module.capitalization_engine import CapitalizationEngine, CapitalizationResult, CapitalizationRule
from ner_module.ner_model_manager import NERModelManager, ProperNounSuggestion, SuggestionSource, ModelStatus
from sanskrit_hindi_identifier.lexicon_manager import LexiconManager, LexiconEntry


class TestYogaVedantaNER(unittest.TestCase):
    """Test cases for YogaVedantaNER model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.ner_model = YogaVedantaNER(training_data_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_ner_model_initialization(self):
        """Test NER model initialization."""
        self.assertIsInstance(self.ner_model, YogaVedantaNER)
        self.assertEqual(self.ner_model.model_version, "2.0-PRD-Compliant")
        self.assertIsNotNone(self.ner_model.lexicon_manager)
        self.assertIsNotNone(self.ner_model.entity_classifier)
    
    def test_identify_entities_basic(self):
        """Test basic entity identification."""
        text = "In the Bhagavad Gita, Krishna teaches Arjuna about dharma."
        result = self.ner_model.identify_entities(text)
        
        self.assertIsInstance(result, NERResult)
        self.assertEqual(result.original_text, text)
        self.assertIsInstance(result.entities, list)
        self.assertGreaterEqual(result.processing_time, 0)
    
    def test_entity_confidence_levels(self):
        """Test entity confidence level assignment."""
        entity_high = NamedEntity(
            text="Krishna", start_pos=0, end_pos=7,
            category=EntityCategory.DEITY, confidence=0.95,
            confidence_level=NERConfidenceLevel.HIGH, source_type="lexicon"
        )
        
        entity_medium = NamedEntity(
            text="Patanjali", start_pos=0, end_pos=9,
            category=EntityCategory.TEACHER, confidence=0.75,
            confidence_level=NERConfidenceLevel.MEDIUM, source_type="pattern"
        )
        
        self.assertEqual(entity_high.confidence_level, NERConfidenceLevel.HIGH)
        self.assertEqual(entity_medium.confidence_level, NERConfidenceLevel.MEDIUM)
    
    def test_model_statistics(self):
        """Test model statistics generation."""
        stats = self.ner_model.get_model_statistics()
        
        self.assertIn('model_version', stats)
        self.assertIn('training_examples_count', stats)
        self.assertIn('categories_supported', stats)
        self.assertIsInstance(stats['categories_supported'], int)
    
    def test_update_model_with_example(self):
        """Test updating model with new training example."""
        text = "Shankaracharya established Advaita Vedanta philosophy."
        entities = [
            {"text": "Shankaracharya", "start": 0, "end": 14, "label": "TEACHER"},
            {"text": "Advaita Vedanta", "start": 27, "end": 42, "label": "PHILOSOPHY"}
        ]
        
        success = self.ner_model.update_model_with_example(text, entities)
        self.assertTrue(success)


class TestEntityClassifier(unittest.TestCase):
    """Test cases for EntityClassifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.classifier = EntityClassifier(config_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_classifier_initialization(self):
        """Test classifier initialization."""
        self.assertIsInstance(self.classifier, EntityClassifier)
        self.assertGreater(len(self.classifier.categories), 0)
    
    def test_classify_entity_basic(self):
        """Test basic entity classification."""
        category = self.classifier.classify_entity("Krishna", "deity")
        self.assertEqual(category, EntityCategory.DEITY)
    
    def test_classify_with_confidence(self):
        """Test classification with confidence scoring."""
        result = self.classifier.classify_with_confidence("Bhagavad Gita", "scripture")
        
        self.assertIsInstance(result, ClassificationResult)
        self.assertEqual(result.predicted_category, EntityCategory.SCRIPTURE)
        self.assertGreater(result.confidence, 0.5)
        self.assertIsInstance(result.reasoning, str)
    
    def test_heuristic_classification(self):
        """Test heuristic-based classification."""
        # Test scripture heuristics
        result = self.classifier.classify_with_confidence("Yoga Sutras")
        self.assertEqual(result.predicted_category, EntityCategory.SCRIPTURE)
        
        # Test teacher heuristics
        result = self.classifier.classify_with_confidence("Swami Sivananda")
        self.assertEqual(result.predicted_category, EntityCategory.TEACHER)
    
    def test_get_category_metadata(self):
        """Test category metadata retrieval."""
        metadata = self.classifier.get_category_metadata(EntityCategory.DEITY)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.name, "Deity")
    
    def test_validation(self):
        """Test classification validation."""
        # Valid classification
        valid = self.classifier.validate_classification("Krishna", EntityCategory.DEITY, 0.95)
        self.assertTrue(valid)
        
        # Invalid low confidence
        invalid = self.classifier.validate_classification("Unknown", EntityCategory.DEITY, 0.3)
        self.assertFalse(invalid)


class TestCapitalizationEngine(unittest.TestCase):
    """Test cases for CapitalizationEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock dependencies
        self.mock_ner = Mock(spec=YogaVedantaNER)
        self.mock_lexicon = Mock(spec=LexiconManager)
        
        self.engine = CapitalizationEngine(
            ner_model=self.mock_ner,
            lexicon_manager=self.mock_lexicon
        )
    
    def test_engine_initialization(self):
        """Test capitalization engine initialization."""
        self.assertIsInstance(self.engine, CapitalizationEngine)
        self.assertGreater(len(self.engine.category_rules), 0)
        self.assertGreater(len(self.engine.special_patterns), 0)
    
    def test_apply_capitalization_rule(self):
        """Test capitalization rule application."""
        # Test proper case
        result = self.engine._apply_capitalization_rule("krishna", CapitalizationRule.PROPER_CASE)
        self.assertEqual(result, "Krishna")
        
        # Test title case
        result = self.engine._apply_capitalization_rule("bhagavad gita", CapitalizationRule.TITLE_CASE)
        self.assertEqual(result, "Bhagavad Gita")
        
        # Test all caps
        result = self.engine._apply_capitalization_rule("om", CapitalizationRule.ALL_CAPS)
        self.assertEqual(result, "OM")
    
    def test_special_patterns(self):
        """Test special capitalization patterns."""
        text = "today we study the bhagavad gita and yoga sutras."
        
        # Mock NER result
        mock_entities = [
            NamedEntity(
                text="Bhagavad Gita", start_pos=19, end_pos=32,
                category=EntityCategory.SCRIPTURE, confidence=0.9,
                confidence_level=NERConfidenceLevel.HIGH, source_type="lexicon"
            )
        ]
        
        mock_ner_result = Mock()
        mock_ner_result.entities = mock_entities
        self.mock_ner.identify_entities.return_value = mock_ner_result
        
        result = self.engine.capitalize_text(text)
        
        self.assertIsInstance(result, CapitalizationResult)
        self.assertNotEqual(result.original_text, result.capitalized_text)
        self.assertGreater(result.confidence, 0)
    
    def test_capitalize_single_entity(self):
        """Test capitalizing a single entity."""
        result = self.engine.capitalize_entity("krishna", EntityCategory.DEITY)
        self.assertEqual(result, "Krishna")
        
        result = self.engine.capitalize_entity("bhagavad gita", EntityCategory.SCRIPTURE)
        self.assertEqual(result, "Bhagavad Gita")
    
    def test_add_special_pattern(self):
        """Test adding new special patterns."""
        pattern = r'\btest\s+pattern\b'
        replacement = "Test Pattern"
        
        success = self.engine.add_special_pattern(pattern, replacement)
        self.assertTrue(success)
        self.assertIn(pattern, self.engine.special_patterns)
        
        # Test invalid pattern
        invalid_success = self.engine.add_special_pattern("[invalid", "replacement")
        self.assertFalse(invalid_success)
    
    def test_validation(self):
        """Test capitalization validation."""
        validation = self.engine.validate_capitalization_rules()
        
        self.assertIn('is_valid', validation)
        self.assertIn('errors', validation)
        self.assertIn('warnings', validation)
        self.assertIsInstance(validation['is_valid'], bool)


class TestNERModelManager(unittest.TestCase):
    """Test cases for NERModelManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.manager = NERModelManager(models_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_manager_initialization(self):
        """Test model manager initialization."""
        self.assertIsInstance(self.manager, NERModelManager)
        self.assertTrue(self.temp_dir.exists())
        self.assertIsNotNone(self.manager.active_model)
    
    def test_add_proper_noun_suggestion(self):
        """Test adding proper noun suggestions."""
        success = self.manager.add_proper_noun_suggestion(
            text="Ramanuja",
            category=EntityCategory.TEACHER,
            source=SuggestionSource.USER_INPUT,
            context="Ramanuja was a great Vedanta teacher."
        )
        
        self.assertTrue(success)
        self.assertEqual(len(self.manager.suggestions), 1)
        
        suggestion = self.manager.suggestions[0]
        self.assertEqual(suggestion.text, "Ramanuja")
        self.assertEqual(suggestion.suggested_category, EntityCategory.TEACHER)
        self.assertEqual(suggestion.frequency, 1)
    
    def test_suggestion_frequency_update(self):
        """Test updating suggestion frequency."""
        # Add same suggestion twice
        self.manager.add_proper_noun_suggestion("Ramanuja", EntityCategory.TEACHER)
        self.manager.add_proper_noun_suggestion("Ramanuja", EntityCategory.TEACHER)
        
        # Should have only one suggestion with frequency 2
        self.assertEqual(len(self.manager.suggestions), 1)
        self.assertEqual(self.manager.suggestions[0].frequency, 2)
    
    def test_get_suggestions_for_review(self):
        """Test getting suggestions ready for review."""
        # Add suggestions with different frequencies
        for i in range(5):
            self.manager.add_proper_noun_suggestion(f"TestEntity{i}", EntityCategory.TEACHER)
        
        # Add one with high frequency
        for i in range(4):
            self.manager.add_proper_noun_suggestion("HighFrequency", EntityCategory.DEITY)
        
        suggestions = self.manager.get_suggestions_for_review(min_frequency=3)
        self.assertEqual(len(suggestions), 1)
        self.assertEqual(suggestions[0].text, "HighFrequency")
    
    def test_approve_suggestion(self):
        """Test approving a suggestion."""
        # Add a suggestion
        self.manager.add_proper_noun_suggestion("TestApproval", EntityCategory.DEITY)
        
        # Mock lexicon manager
        with patch.object(self.manager.lexicon_manager, 'add_entry', return_value=True), \
             patch.object(self.manager.lexicon_manager, 'save_lexicon', return_value=True):
            
            success = self.manager.approve_suggestion("TestApproval")
            self.assertTrue(success)
            
            # Check that suggestion was marked as approved
            suggestion = next(s for s in self.manager.suggestions if s.text == "TestApproval")
            self.assertEqual(suggestion.user_feedback, "approved")
    
    def test_reject_suggestion(self):
        """Test rejecting a suggestion."""
        self.manager.add_proper_noun_suggestion("TestReject", EntityCategory.TEACHER)
        
        success = self.manager.reject_suggestion("TestReject", "Not a valid entity")
        self.assertTrue(success)
        
        suggestion = next(s for s in self.manager.suggestions if s.text == "TestReject")
        self.assertIn("rejected", suggestion.user_feedback)
    
    def test_create_new_model_version(self):
        """Test creating new model versions."""
        description = "Test model with new entities"
        changes = ["Added 5 new proper nouns", "Improved classification accuracy"]
        
        new_version = self.manager.create_new_model_version(description, changes)
        
        self.assertIsNotNone(new_version)
        self.assertIn(new_version, self.manager.model_versions)
        
        version_info = self.manager.model_versions[new_version]
        self.assertEqual(version_info.description, description)
        self.assertEqual(version_info.changes, changes)
    
    def test_model_version_activation(self):
        """Test model version activation."""
        # Create a new version
        new_version = self.manager.create_new_model_version("Test version", ["Test change"])
        
        # Activate it
        success = self.manager.activate_model_version(new_version)
        self.assertTrue(success)
        self.assertEqual(self.manager.active_version, new_version)
        
        # Check status changes
        version_info = self.manager.model_versions[new_version]
        self.assertEqual(version_info.status, ModelStatus.ACTIVE)
    
    def test_model_statistics(self):
        """Test model statistics generation."""
        stats = self.manager.get_model_statistics()
        
        self.assertIn('total_versions', stats)
        self.assertIn('active_version', stats)
        self.assertIn('total_suggestions', stats)
        self.assertIn('pending_suggestions', stats)
        self.assertIsInstance(stats['total_versions'], int)


class TestNERIntegration(unittest.TestCase):
    """Integration tests for the complete NER system."""
    
    def setUp(self):
        """Set up test fixtures for integration testing."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create integrated system components with proper lexicon access
        from sanskrit_hindi_identifier.lexicon_manager import LexiconManager
        lexicon_manager = LexiconManager()  # This loads the real lexicons
        
        self.ner_model = YogaVedantaNER(training_data_dir=self.temp_dir, lexicon_manager=lexicon_manager)
        self.capitalization_engine = CapitalizationEngine(ner_model=self.ner_model)
        self.model_manager = NERModelManager(models_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_processing(self):
        """Test complete end-to-end NER processing."""
        text = "today we study krishna in the bhagavad gita with guidance from shankaracharya."
        
        # Step 1: Identify entities
        ner_result = self.ner_model.identify_entities(text)
        self.assertGreater(len(ner_result.entities), 0)
        
        # Step 2: Apply capitalization
        cap_result = self.capitalization_engine.capitalize_text(text)
        self.assertIsInstance(cap_result, CapitalizationResult)
        self.assertNotEqual(cap_result.original_text, cap_result.capitalized_text)
        
        # Step 3: Generate suggestions for unknown entities
        for entity in ner_result.entities:
            if entity.confidence < 0.8:  # Low confidence entities
                self.model_manager.add_proper_noun_suggestion(
                    text=entity.text,
                    category=entity.category,
                    source=SuggestionSource.AUTO_DISCOVERY,
                    context=text
                )
        
        # Verify suggestions were created
        suggestions = self.model_manager.get_suggestions_for_review(min_frequency=1)
        self.assertIsInstance(suggestions, list)
    
    def test_performance_with_large_text(self):
        """Test NER performance with larger text."""
        # Create a longer text with multiple entities
        text = """
        In the ancient tradition of Yoga Vedanta, great teachers like Patanjali,
        Shankaracharya, and Swami Vivekananda have guided seekers through the
        profound teachings found in scriptures such as the Bhagavad Gita,
        Upanishads, and Yoga Sutras. The philosophy of Advaita Vedanta,
        developed in sacred places like Rishikesh and Varanasi, teaches us
        about the divine nature represented by Krishna, Shiva, and Vishnu.
        Characters like Arjuna and Rama exemplify the dharmic path of
        spiritual evolution through Karma Yoga, Bhakti Yoga, and Jnana Yoga.
        """
        
        # Process the text
        result = self.ner_model.identify_entities(text)
        
        # Verify processing completed successfully
        self.assertIsInstance(result, NERResult)
        self.assertGreater(len(result.entities), 5)  # Should find multiple entities
        self.assertLess(result.processing_time, 5.0)  # Should complete within reasonable time
        
        # Verify entity quality
        high_confidence_entities = result.get_high_confidence_entities()
        self.assertGreater(len(high_confidence_entities), 0)
    
    def test_suggestion_workflow(self):
        """Test complete suggestion workflow."""
        # Simulate discovering a new entity during processing
        new_entity = "Madhvacharya"
        
        # Add suggestion
        success = self.model_manager.add_proper_noun_suggestion(
            text=new_entity,
            category=EntityCategory.TEACHER,
            source=SuggestionSource.AUTO_DISCOVERY,
            context="Madhvacharya founded Dvaita Vedanta school."
        )
        
        self.assertTrue(success)
        
        # Simulate multiple occurrences
        for i in range(3):
            self.model_manager.add_proper_noun_suggestion(new_entity, EntityCategory.TEACHER)
        
        # Get suggestions for review
        suggestions = self.model_manager.get_suggestions_for_review()
        self.assertEqual(len(suggestions), 1)
        self.assertEqual(suggestions[0].frequency, 4)  # 1 initial + 3 additional
        
        # Approve the suggestion
        with patch.object(self.model_manager.lexicon_manager, 'add_entry', return_value=True), \
             patch.object(self.model_manager.lexicon_manager, 'save_lexicon', return_value=True):
            
            approval_success = self.model_manager.approve_suggestion(
                new_entity,
                transliteration="Madhvācārya",
                variations=["Madhva", "Madhavacharya"]
            )
            
            self.assertTrue(approval_success)


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestYogaVedantaNER))
    suite.addTest(unittest.makeSuite(TestEntityClassifier))
    suite.addTest(unittest.makeSuite(TestCapitalizationEngine))
    suite.addTest(unittest.makeSuite(TestNERModelManager))
    suite.addTest(unittest.makeSuite(TestNERIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    if result.wasSuccessful():
        print("\n✅ All NER module tests passed successfully!")
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        
    exit(0 if result.wasSuccessful() else 1)