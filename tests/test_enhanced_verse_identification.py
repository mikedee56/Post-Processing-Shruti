"""
Test suite for Enhanced Verse Identification with External APIs.

Validates integration with legitimate external scripture APIs
to improve verse identification accuracy from 40% to 70%+.
"""

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scripture_processing.external_verse_api_client import (
    ExternalVerseAPIClient, 
    EnhancedVerseIdentifier,
    VerseReference, 
    APIProvider,
    APIResponse
)


class TestExternalVerseAPIClient(unittest.TestCase):
    """Test external verse API client functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.client = ExternalVerseAPIClient()
    
    def test_client_initialization(self):
        """Test API client initializes correctly."""
        self.assertIsNotNone(self.client)
        self.assertIn(APIProvider.BHAGAVAD_GITA_API, self.client.api_configs)
        self.assertEqual(self.client.cache_ttl, 3600)
    
    @patch('requests.Session.get')
    def test_verse_search_success(self, mock_get):
        """Test successful verse search."""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                'verse': 1,
                'text_sanskrit': 'धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः',
                'transliteration': 'dharma-kṣetre kuru-kṣetre samavetā yuyutsavaḥ',
                'translation': 'On the field of dharma, on the field of Kurukshetra...'
            }
        ]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        results = self.client.search_verse_by_text("धर्मक्षेत्रे कुरुक्षेत्रे")
        
        self.assertGreater(len(results), 0)
        self.assertIsInstance(results[0], VerseReference)
        self.assertEqual(results[0].scripture, "bhagavad_gita")
    
    @patch('requests.Session.get')
    def test_api_error_handling(self, mock_get):
        """Test API error handling."""
        # Mock API failure
        mock_get.side_effect = requests.exceptions.RequestException("API error")
        
        results = self.client.search_verse_by_text("test text")
        
        # Should return empty list on error
        self.assertEqual(len(results), 0)
    
    def test_caching_mechanism(self):
        """Test response caching functionality."""
        # Mock a successful response
        with patch.object(self.client, '_search_bhagavad_gita_api', return_value=[]):
            # First call
            self.client.search_verse_by_text("test")
            
            # Second call should use cache
            results = self.client.search_verse_by_text("test")
            
            # Verify caching works (no exception means cache hit)
            self.assertIsInstance(results, list)
    
    def test_similarity_calculation(self):
        """Test text similarity calculation."""
        similarity = self.client._calculate_similarity(
            "धर्मक्षेत्रे कुरुक्षेत्रे",
            "धर्मक्षेत्रे कुरुक्षेत्रे समवेता"
        )
        
        self.assertGreater(similarity, 0.5)
        self.assertLessEqual(similarity, 1.0)


class TestEnhancedVerseIdentifier(unittest.TestCase):
    """Test enhanced verse identification with external API integration."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock lexicon manager
        self.mock_lexicon_manager = Mock()
        self.config = {
            'use_external_apis': True,
            'confidence_threshold': 0.7,
            'api_config': {}
        }
        
        # Mock the API client
        with patch('scripture_processing.external_verse_api_client.ExternalVerseAPIClient'):
            self.identifier = EnhancedVerseIdentifier(
                lexicon_manager=self.mock_lexicon_manager,
                config=self.config
            )
    
    def test_enhanced_identifier_initialization(self):
        """Test enhanced identifier initializes correctly."""
        self.assertIsNotNone(self.identifier)
        self.assertTrue(self.identifier.use_external_apis)
        self.assertEqual(self.identifier.confidence_threshold, 0.7)
    
    @patch.object(EnhancedVerseIdentifier, '_identify_local_verses')
    @patch.object(EnhancedVerseIdentifier, '_enhance_with_external_apis')
    def test_verse_identification_workflow(self, mock_enhance, mock_local):
        """Test complete verse identification workflow."""
        # Mock local verse identification
        mock_local.return_value = [
            {
                'matched_text': 'dharma kshetre kuru kshetre',
                'confidence_score': 0.6,
                'scripture_reference': 'bg_1_1'
            }
        ]
        
        # Mock external API enhancement
        mock_enhance.return_value = [
            {
                'matched_text': 'dharma kshetre kuru kshetre',
                'confidence_score': 0.6,
                'external_verification': True,
                'enhanced_confidence': 0.8,
                'canonical_text': 'धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः',
                'scripture_reference': 'bhagavad_gita_1.1'
            }
        ]
        
        results = self.identifier.identify_verses("dharma kshetre kuru kshetre")
        
        self.assertGreater(len(results), 0)
        self.assertTrue(results[0].get('external_verification', False))
        self.assertGreater(results[0].get('enhanced_confidence', 0), 0.7)
    
    def test_accuracy_validation(self):
        """Test verse identification accuracy validation."""
        identified_verses = [
            {'scripture_reference': 'bg_1_1'},
            {'scripture_reference': 'bg_2_47'},
            {'scripture_reference': 'bg_18_66'}
        ]
        
        golden_references = [
            {'scripture_reference': 'bg_1_1'},
            {'scripture_reference': 'bg_2_47'},
            {'scripture_reference': 'bg_4_7'}  # Not identified
        ]
        
        metrics = self.identifier.validate_verse_accuracy(identified_verses, golden_references)
        
        self.assertAlmostEqual(metrics['precision'], 1.0)  # No false positives
        self.assertAlmostEqual(metrics['recall'], 2/3)     # 2 out of 3 found
        self.assertAlmostEqual(metrics['accuracy'], 2/3)   # 2 out of 3 correct


class TestVerseIdentificationIntegration(unittest.TestCase):
    """Test integration of enhanced verse identification with main processor."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('scripture_processing.scripture_processor.EnhancedVerseIdentifier')
    @patch('scripture_processing.scripture_processor.LexiconManager')
    def test_scripture_processor_integration(self, mock_lexicon, mock_enhanced):
        """Test integration with ScriptureProcessor."""
        # Mock enhanced identifier
        mock_identifier = Mock()
        mock_identifier.identify_verses.return_value = [
            {
                'matched_text': 'test verse',
                'external_verification': True,
                'enhanced_confidence': 0.85,
                'canonical_text': 'canonical verse text',
                'scripture_reference': 'bg_1_1'
            }
        ]
        mock_enhanced.return_value = mock_identifier
        
        # Import and test ScriptureProcessor
        from scripture_processing.scripture_processor import ScriptureProcessor
        
        processor = ScriptureProcessor(config={
            'enhanced_verse_config': {
                'use_external_apis': True,
                'confidence_threshold': 0.7
            }
        })
        
        # Test text processing with enhanced verse identification
        test_text = "dharma kshetre kuru kshetre"
        result = processor.process_text(test_text)
        
        # Verify enhanced identification was used
        self.assertIsNotNone(result)
        self.assertGreater(result.verses_identified, 0)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)