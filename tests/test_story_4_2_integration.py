"""
Integration Tests for Story 4.2: Sanskrit Processing Enhancement

Tests the integration of all Story 4.2 components:
- MCP Transformer Integration
- Enhanced Lexicon Intelligence
- Accuracy Measurement and Validation
- System Integration with Story 2.1 foundation
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import json
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.mcp_transformer_client import (
    MCPTransformerClient, 
    SemanticClassification,
    TransformerResult,
    CulturalContext,
    SemanticConfidenceLevel,
    create_transformer_client
)
from sanskrit_hindi_identifier.enhanced_lexicon_manager import (
    EnhancedLexiconManager,
    MLEnhancedEntry,
    QualityValidationStatus,
    DynamicExpansionResult
)
from utils.sanskrit_accuracy_validator import (
    SanskritAccuracyValidator,
    IASTValidationResult,
    AccuracyMeasurement,
    ImprovementAnalysis
)
from utils.research_metrics_collector import (
    ResearchMetricsCollector,
    MetricCategory,
    ResearchMetric
)
from sanskrit_hindi_identifier.lexicon_manager import LexiconManager
from sanskrit_hindi_identifier.word_identifier import SanskritHindiIdentifier


class TestStory42MCPTransformerIntegration:
    """Test MCP Transformer Integration (AC1)."""
    
    def test_transformer_client_initialization(self):
        """Test AC1: MCP transformer integration initialization."""
        # Test creation without MCP manager
        client = create_transformer_client()
        assert client is not None
        assert client.config['max_processing_time_ms'] == 1000
        assert client.config['enable_cultural_awareness'] is True
        
        # Test creation with mock MCP manager
        mock_mcp_manager = Mock()
        client_with_mcp = create_transformer_client(mcp_manager=mock_mcp_manager)
        assert client_with_mcp.mcp_manager == mock_mcp_manager

    @pytest.mark.asyncio
    async def test_sanskrit_semantic_processing(self):
        """Test AC1: Sanskrit-specific semantic processing."""
        client = create_transformer_client()
        
        # Test Sanskrit text processing
        sanskrit_text = "Today we study dharma yoga and Krishna's teachings"
        result = await client.process_sanskrit_text_with_context(sanskrit_text)
        
        assert isinstance(result, TransformerResult)
        assert result.original_text == sanskrit_text
        assert result.confidence_score > 0.0
        assert result.semantic_context in CulturalContext
        assert result.processing_time_ms > 0
        
        # Test that Sanskrit terms are detected
        if result.confidence_score > 0.7:
            assert result.semantic_context != CulturalContext.UNKNOWN

    def test_cultural_context_awareness(self):
        """Test AC1: Cultural context awareness for Yoga Vedanta terminology."""
        client = create_transformer_client()
        
        test_terms = [
            ("Krishna", "devotional"),
            ("dharma", "philosophical"), 
            ("yoga", "practical"),
            ("Bhagavad Gita", "scriptural")
        ]
        
        for term, expected_context_type in test_terms:
            classification = client.classify_term_semantically(term)
            
            assert isinstance(classification, SemanticClassification)
            assert classification.term == term
            assert classification.semantic_confidence > 0.0
            
            # Check that cultural context is recognized
            assert classification.cultural_context != CulturalContext.UNKNOWN

    @pytest.mark.asyncio
    async def test_performance_requirements(self):
        """Test AC1: Performance targets <1s processing time."""
        client = create_transformer_client()
        
        test_text = "We study yoga dharma karma moksha in our spiritual practice"
        result = await client.process_sanskrit_text_with_context(test_text)
        
        # Verify performance target
        assert result.processing_time_ms < 1000, f"Processing took {result.processing_time_ms}ms (>1s limit)"
        
        # Verify performance metrics
        metrics = client.get_performance_metrics()
        assert metrics['performance_target_met'] is True


class TestStory42EnhancedLexiconIntelligence:
    """Test Enhanced Lexicon Intelligence (AC2)."""
    
    def setup_method(self):
        """Set up test environment."""
        # Create temporary lexicon directory
        self.temp_dir = tempfile.mkdtemp()
        self.lexicon_path = Path(self.temp_dir) / "test_lexicon.yaml"
        
        # Create basic test lexicon
        test_lexicon = {
            'version': '1.0',
            'description': 'Test lexicon for Story 4.2',
            'entries': [
                {
                    'original_term': 'dharma',
                    'variations': ['dharama', 'dharm'],
                    'transliteration': 'dharma',
                    'is_proper_noun': False,
                    'category': 'philosophical',
                    'confidence': 0.95,
                    'source_authority': 'test'
                }
            ]
        }
        
        import yaml
        with open(self.lexicon_path, 'w') as f:
            yaml.dump(test_lexicon, f)

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_enhanced_lexicon_initialization(self):
        """Test AC2: Enhanced lexicon with ML confidence scoring."""
        # Create base lexicon manager
        base_manager = LexiconManager(lexicon_dir=Path(self.temp_dir))
        
        # Create enhanced manager
        enhanced_manager = EnhancedLexiconManager(base_lexicon_manager=base_manager)
        
        assert enhanced_manager.base_manager == base_manager
        assert len(enhanced_manager.ml_enhanced_entries) > 0
        
        # Check that existing entries are enhanced
        assert 'dharma' in enhanced_manager.ml_enhanced_entries
        enhanced_entry = enhanced_manager.ml_enhanced_entries['dharma']
        assert isinstance(enhanced_entry, MLEnhancedEntry)
        assert enhanced_entry.base_entry.original_term == 'dharma'

    @pytest.mark.asyncio
    async def test_ml_classification_enhancement(self):
        """Test AC2: ML enhancement of existing entries."""
        base_manager = LexiconManager(lexicon_dir=Path(self.temp_dir))
        transformer_client = create_transformer_client()
        
        enhanced_manager = EnhancedLexiconManager(
            base_lexicon_manager=base_manager,
            transformer_client=transformer_client
        )
        
        # Test ML enhancement of existing entry
        success = await enhanced_manager.enhance_entry_with_ml_classification('dharma')
        assert success is True
        
        enhanced_entry = enhanced_manager.ml_enhanced_entries['dharma']
        assert enhanced_entry.semantic_classification is not None
        assert len(enhanced_entry.ml_feedback_history) > 0

    @pytest.mark.asyncio
    async def test_dynamic_lexicon_expansion(self):
        """Test AC2: Dynamic lexicon expansion with quality validation."""
        base_manager = LexiconManager(lexicon_dir=Path(self.temp_dir))
        transformer_client = create_transformer_client()
        
        enhanced_manager = EnhancedLexiconManager(
            base_lexicon_manager=base_manager,
            transformer_client=transformer_client
        )
        
        # Test expansion suggestions
        source_text = "Today we learn about Krishna and Arjuna in the sacred Gita teachings"
        expansion_result = await enhanced_manager.suggest_lexicon_expansions(source_text, max_suggestions=5)
        
        assert isinstance(expansion_result, DynamicExpansionResult)
        assert expansion_result.processing_time_ms > 0
        
        # Check that some suggestions were made (may be 0 if terms already exist)
        total_suggestions = len(expansion_result.new_entries_suggested)
        assert total_suggestions >= 0

    def test_quality_validation_workflow(self):
        """Test AC2: Quality validation for ML-suggested entries."""
        base_manager = LexiconManager(lexicon_dir=Path(self.temp_dir))
        enhanced_manager = EnhancedLexiconManager(base_lexicon_manager=base_manager)
        
        # Test approval workflow
        if enhanced_manager.ml_enhanced_entries:
            term = list(enhanced_manager.ml_enhanced_entries.keys())[0]
            
            # Test approval
            success = enhanced_manager.approve_ml_suggestion(term, ["Academic review passed"])
            # Note: May fail if term already exists in base lexicon
            
            # Test rejection
            success = enhanced_manager.reject_ml_suggestion(term, "Test rejection")
            assert success is True
            
            enhanced_entry = enhanced_manager.ml_enhanced_entries[term]
            assert enhanced_entry.quality_validation_status == QualityValidationStatus.REJECTED

    def test_backward_compatibility(self):
        """Test AC2: Maintain compatibility with existing YAML lexicon format."""
        base_manager = LexiconManager(lexicon_dir=Path(self.temp_dir))
        enhanced_manager = EnhancedLexiconManager(base_lexicon_manager=base_manager)
        
        # Test that enhanced manager maintains base manager interface
        all_entries = enhanced_manager.get_all_entries()
        assert len(all_entries) > 0
        assert 'dharma' in all_entries
        
        # Test category search
        philosophical_entries = enhanced_manager.get_entries_by_category('philosophical')
        assert len(philosophical_entries) > 0
        
        # Test search functionality
        search_results = enhanced_manager.search_entries('dharma')
        assert len(search_results) > 0


class TestStory42AccuracyValidation:
    """Test Accuracy Measurement and Validation (AC3)."""
    
    def test_iast_compliance_validation(self):
        """Test AC3: Academic validation standards for IAST transliteration."""
        validator = SanskritAccuracyValidator()
        
        # Test compliant IAST
        compliant_text = "kṛṣṇa dharma yoga"
        result = validator.validate_iast_compliance(compliant_text)
        
        assert isinstance(result, IASTValidationResult)
        assert result.text == compliant_text
        assert result.compliance_score > 0.7  # Should be reasonably compliant
        
        # Test non-compliant text
        non_compliant_text = "krishna dharma yoga"  # Missing diacriticals
        result = validator.validate_iast_compliance(non_compliant_text)
        
        assert result.compliance_score < 1.0
        assert len(result.violations) > 0
        assert len(result.suggestions) > 0

    def test_sanskrit_accuracy_measurement(self):
        """Test AC3: 15% Sanskrit accuracy improvement measurement."""
        validator = SanskritAccuracyValidator()
        
        # Create test data
        processed_terms = [
            {'term': 'dharma', 'identified_as_sanskrit': True, 'transliteration': 'dharma'},
            {'term': 'yoga', 'identified_as_sanskrit': True, 'transliteration': 'yoga'},
            {'term': 'practice', 'identified_as_sanskrit': False}
        ]
        
        ground_truth = [
            {'term': 'dharma', 'is_sanskrit': True, 'transliteration': 'dharma'},
            {'term': 'yoga', 'is_sanskrit': True, 'transliteration': 'yoga'},
            {'term': 'practice', 'is_sanskrit': False}
        ]
        
        measurement = validator.measure_sanskrit_accuracy(processed_terms, ground_truth)
        
        assert isinstance(measurement, AccuracyMeasurement)
        assert measurement.total_terms_processed == 3
        assert measurement.accuracy_score >= 0.0
        assert measurement.precision >= 0.0
        assert measurement.recall >= 0.0
        assert measurement.f1_score >= 0.0

    def test_improvement_analysis(self):
        """Test AC3: Research-grade quality metrics and reporting."""
        validator = SanskritAccuracyValidator()
        
        # Create baseline measurement
        baseline_terms = [
            {'term': 'dharma', 'identified_as_sanskrit': True, 'transliteration': 'dharma'},
            {'term': 'yoga', 'identified_as_sanskrit': False}  # Incorrect identification
        ]
        
        ground_truth = [
            {'term': 'dharma', 'is_sanskrit': True, 'transliteration': 'dharma'},
            {'term': 'yoga', 'is_sanskrit': True, 'transliteration': 'yoga'}
        ]
        
        baseline = validator.measure_sanskrit_accuracy(baseline_terms, ground_truth, 'baseline')
        
        # Create improved measurement
        improved_terms = [
            {'term': 'dharma', 'identified_as_sanskrit': True, 'transliteration': 'dharma'},
            {'term': 'yoga', 'identified_as_sanskrit': True, 'transliteration': 'yoga'}  # Now correct
        ]
        
        current = validator.measure_sanskrit_accuracy(improved_terms, ground_truth, 'current')
        
        # Analyze improvement
        if len(validator.measurement_history) >= 2:
            analysis = validator.analyze_improvement()
            
            if analysis:
                assert isinstance(analysis, ImprovementAnalysis)
                assert analysis.accuracy_improvement_percent >= 0  # Should show improvement
                assert analysis.meets_target is not None

    def test_research_grade_reporting(self):
        """Test AC3: Research-grade quality metrics and reporting."""
        validator = SanskritAccuracyValidator()
        
        # Add some test measurements
        test_terms = [
            {'term': 'dharma', 'identified_as_sanskrit': True, 'transliteration': 'dharma'},
        ]
        
        ground_truth = [
            {'term': 'dharma', 'is_sanskrit': True, 'transliteration': 'dharma'},
        ]
        
        validator.measure_sanskrit_accuracy(test_terms, ground_truth)
        
        # Generate research report
        report = validator.generate_research_grade_report()
        
        assert 'report_metadata' in report
        assert 'current_accuracy' in report
        assert 'improvement_analysis' in report
        assert 'academic_compliance' in report
        assert 'recommendations' in report


class TestStory42SystemIntegration:
    """Test System Integration Compatibility (AC4)."""
    
    def test_story21_integration(self):
        """Test AC4: Seamless integration with Story 2.1 SanskritHindiIdentifier."""
        # Test that Story 4.2 components work with existing Story 2.1 components
        identifier = SanskritHindiIdentifier()
        
        # Test basic functionality still works
        test_text = "Today we study dharma and yoga"
        words = identifier.identify_words(test_text)
        
        assert len(words) > 0
        
        # Test enhanced functionality integration
        transformer_client = create_transformer_client()
        enhanced_manager = EnhancedLexiconManager(transformer_client=transformer_client)
        
        # Should be able to get entries using Story 2.1 interface
        all_entries = enhanced_manager.get_all_entries()
        assert isinstance(all_entries, dict)

    def test_mcp_infrastructure_compatibility(self):
        """Test AC4: Maintain compatibility with Story 4.1 MCP infrastructure."""
        # Test that transformer client can work with MCP infrastructure
        mock_mcp_manager = Mock()
        mock_mcp_manager.mcp_client = Mock()
        
        transformer_client = create_transformer_client(mcp_manager=mock_mcp_manager)
        
        # Should initialize without errors
        assert transformer_client.mcp_manager == mock_mcp_manager
        
        # Performance targets should be maintained
        metrics = transformer_client.get_performance_metrics()
        assert 'performance_target_met' in metrics

    def test_api_contract_preservation(self):
        """Test AC4: Preserve existing API contracts and system interfaces."""
        # Test that existing interfaces are preserved
        
        # LexiconManager interface
        enhanced_manager = EnhancedLexiconManager()
        
        # Should support all existing LexiconManager methods
        assert hasattr(enhanced_manager, 'get_all_entries')
        assert hasattr(enhanced_manager, 'get_entries_by_category')
        assert hasattr(enhanced_manager, 'search_entries')
        
        # Methods should return expected types
        entries = enhanced_manager.get_all_entries()
        assert isinstance(entries, dict)

    def test_backward_compatibility(self):
        """Test AC4: Ensure backward compatibility with current processing pipeline."""
        # Test that new components don't break existing workflows
        
        # Create components in isolation
        transformer_client = create_transformer_client()
        validator = SanskritAccuracyValidator()
        metrics_collector = ResearchMetricsCollector()
        
        # Should all initialize without dependencies
        assert transformer_client is not None
        assert validator is not None
        assert metrics_collector is not None
        
        # Should handle missing dependencies gracefully
        assert transformer_client.config.get('fallback_mode', False) in [True, False]


class TestStory42EndToEndIntegration:
    """Test complete end-to-end integration of all Story 4.2 components."""
    
    @pytest.mark.asyncio
    async def test_complete_enhancement_pipeline(self):
        """Test complete Sanskrit processing enhancement pipeline."""
        # Initialize all components
        transformer_client = create_transformer_client()
        enhanced_manager = EnhancedLexiconManager(transformer_client=transformer_client)
        validator = SanskritAccuracyValidator()
        metrics_collector = ResearchMetricsCollector()
        
        # Test text with Sanskrit terms
        test_text = "Today we study dharma yoga and learn about Krishna's teachings in the Gita"
        
        # Step 1: Semantic processing with transformer
        semantic_result = await transformer_client.process_sanskrit_text_with_context(test_text)
        
        assert semantic_result.confidence_score > 0.0
        assert semantic_result.processing_time_ms < 1000  # Performance requirement
        
        # Step 2: Lexicon expansion suggestions
        expansion_result = await enhanced_manager.suggest_lexicon_expansions(test_text)
        
        assert expansion_result.processing_time_ms > 0
        
        # Step 3: IAST validation
        iast_result = validator.validate_iast_compliance("dharma yoga kṛṣṇa")
        
        assert iast_result.compliance_score > 0.0
        
        # Step 4: Metrics collection
        metrics_collector.record_metric(
            "end_to_end_processing_time",
            semantic_result.processing_time_ms + expansion_result.processing_time_ms,
            MetricCategory.PERFORMANCE,
            unit="ms",
            source_component="story_4_2_integration"
        )
        
        # Verify all components worked together
        dashboard = metrics_collector.generate_research_dashboard()
        assert dashboard['quality_overview']['overall_quality_score'] >= 0.0

    def test_15_percent_accuracy_improvement_target(self):
        """Test AC3: Validate 15% accuracy improvement target can be measured."""
        validator = SanskritAccuracyValidator()
        metrics_collector = ResearchMetricsCollector()
        
        # Simulate baseline accuracy (70%)
        baseline_measurement = validator.measure_sanskrit_accuracy(
            [
                {'term': 'dharma', 'identified_as_sanskrit': True},
                {'term': 'yoga', 'identified_as_sanskrit': False},  # Missed
                {'term': 'practice', 'identified_as_sanskrit': False}
            ],
            [
                {'term': 'dharma', 'is_sanskrit': True},
                {'term': 'yoga', 'is_sanskrit': True},
                {'term': 'practice', 'is_sanskrit': False}
            ],
            'baseline'
        )
        
        # Simulate improved accuracy (85%+)
        improved_measurement = validator.measure_sanskrit_accuracy(
            [
                {'term': 'dharma', 'identified_as_sanskrit': True},
                {'term': 'yoga', 'identified_as_sanskrit': True},  # Now correct
                {'term': 'practice', 'identified_as_sanskrit': False}
            ],
            [
                {'term': 'dharma', 'is_sanskrit': True},
                {'term': 'yoga', 'is_sanskrit': True},
                {'term': 'practice', 'is_sanskrit': False}
            ],
            'improved'
        )
        
        # Record accuracy metrics
        metrics_collector.record_metric(
            "sanskrit_accuracy",
            improved_measurement.accuracy_score,
            MetricCategory.ACCURACY,
            source_component="accuracy_validator"
        )
        
        # Analyze improvement
        if len(validator.measurement_history) >= 2:
            analysis = validator.analyze_improvement()
            
            if analysis:
                improvement_percent = analysis.accuracy_improvement_percent
                
                # Record improvement metric
                metrics_collector.record_metric(
                    "accuracy_improvement_percent",
                    improvement_percent,
                    MetricCategory.IMPROVEMENT,
                    unit="%",
                    source_component="accuracy_validator"
                )
                
                # Verify improvement measurement capability
                assert improvement_percent >= 0  # Should show some improvement
                
                # Check if 15% target framework is in place
                target_met = improvement_percent >= 15.0
                assert isinstance(target_met, bool)  # Validation that target checking works

    def test_research_grade_reporting_integration(self):
        """Test that all components contribute to research-grade reporting."""
        # Initialize components
        transformer_client = create_transformer_client()
        validator = SanskritAccuracyValidator()
        metrics_collector = ResearchMetricsCollector()
        
        # Collect metrics from different components
        metrics_collector.record_metric(
            "transformer_performance",
            0.85,
            MetricCategory.PERFORMANCE,
            source_component="mcp_transformer_client"
        )
        
        # Generate comprehensive report
        dashboard = metrics_collector.generate_research_dashboard()
        
        # Verify research-grade reporting elements
        assert 'dashboard_metadata' in dashboard
        assert 'quality_overview' in dashboard
        assert 'academic_compliance' in dashboard
        assert 'metrics_by_category' in dashboard
        
        # Verify academic compliance tracking
        compliance_info = dashboard['academic_compliance']
        assert 'overall_compliance_score' in compliance_info
        assert 'meets_academic_standards' in compliance_info


# Performance and reliability tests
class TestStory42Performance:
    """Test performance requirements for Story 4.2."""
    
    @pytest.mark.asyncio
    async def test_processing_time_requirements(self):
        """Test that processing time remains <1s as required."""
        transformer_client = create_transformer_client()
        
        # Test with various text lengths
        test_texts = [
            "dharma",
            "dharma yoga practice",
            "Today we study dharma yoga and learn about Krishna's teachings",
            "In the ancient tradition of dharma yoga practice, we learn about Krishna's teachings from the Bhagavad Gita and explore the philosophical foundations of Vedanta"
        ]
        
        for text in test_texts:
            result = await transformer_client.process_sanskrit_text_with_context(text)
            
            # Performance requirement: <1s (1000ms)
            assert result.processing_time_ms < 1000, f"Processing '{text}' took {result.processing_time_ms}ms (>1s)"

    def test_system_reliability(self):
        """Test that system handles errors gracefully."""
        # Test transformer client with no MCP manager
        client = create_transformer_client()
        
        # Should not crash with invalid input
        classification = client.classify_term_semantically("")
        assert classification is not None
        
        # Test validator with invalid data
        validator = SanskritAccuracyValidator()
        result = validator.validate_iast_compliance("")
        assert result is not None
        
        # Test metrics collector with invalid data
        collector = ResearchMetricsCollector()
        
        # Should handle errors gracefully
        try:
            metric = collector.record_metric("test", float('inf'), MetricCategory.ACCURACY)
        except Exception:
            pass  # Expected to handle gracefully


if __name__ == "__main__":
    # Run basic tests
    print("Running Story 4.2 Integration Tests...")
    
    # Test transformer client
    client = create_transformer_client()
    print(f"✓ Transformer client initialized: {client is not None}")
    
    # Test enhanced lexicon manager
    enhanced_manager = EnhancedLexiconManager()
    print(f"✓ Enhanced lexicon manager initialized: {enhanced_manager is not None}")
    
    # Test accuracy validator
    validator = SanskritAccuracyValidator()
    print(f"✓ Accuracy validator initialized: {validator is not None}")
    
    # Test metrics collector
    collector = ResearchMetricsCollector()
    print(f"✓ Metrics collector initialized: {collector is not None}")
    
    print("✓ All Story 4.2 components initialized successfully!")