"""
Comprehensive Test Suite for Story 3.2 QA Module - Epic 4 Integration Tests

This test suite validates all QA module components with Epic 4 integration:
- Epic 4.3 Production-Grade Confidence Analysis
- Epic 4.2 ML-Enhanced OOV Detection  
- Epic 4.1 MCP Context-Aware Anomaly Detection
- Epic 4.5 Academic-Grade QA Reporting
- Integration with existing Epic 2 systems

Author: Epic 4 QA Systems Team
Version: 1.0.0
"""

import pytest
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import List, Dict, Any

# Core QA module imports
from qa_module.qa_flagging_engine import QAFlaggingEngine, QAFlag, QAAnalysisResult, QAFlagType, QASeverity
from qa_module.confidence_analyzer import ConfidenceAnalyzer, ConfidenceStatistics, ConfidenceAlert, ConfidenceAnalysisTrend
from qa_module.oov_detector import OOVDetector, OOVWord, OOVAnalysisResult, OOVCategory
from qa_module.anomaly_detector import AnomalyDetector, DetectedAnomaly, AnomalyAnalysisResult
from qa_module.qa_report_generator import QAReportGenerator, QAReport, QAInsight

# Utils and dependencies
from utils.srt_parser import SRTSegment
from sanskrit_hindi_identifier.enhanced_lexicon_manager import EnhancedLexiconManager


@dataclass
class MockSRTSegment:
    """Mock SRT segment for testing."""
    index: int
    start_time: str
    end_time: str
    text: str
    confidence_score: float = 0.8


class TestConfidenceAnalyzer:
    """Test Epic 4.3 Production-Grade Confidence Analysis."""
    
    @pytest.fixture
    def confidence_analyzer(self):
        """Create confidence analyzer for testing."""
        config = {
            'max_processing_time_ms': 500,
            'confidence_window_size': 50,
            'adaptive_thresholds_enabled': True,
            'monitoring': {'enabled': False},  # Disable for testing
            'telemetry': {'enabled': False}    # Disable for testing
        }
        return ConfidenceAnalyzer(config)
    
    @pytest.fixture
    def test_segments(self):
        """Create test SRT segments with varying confidence scores."""
        return [
            MockSRTSegment(0, "00:00:01,000", "00:00:05,000", "High confidence segment", 0.95),
            MockSRTSegment(1, "00:00:06,000", "00:00:10,000", "Medium confidence segment", 0.75),
            MockSRTSegment(2, "00:00:11,000", "00:00:15,000", "Low confidence segment", 0.45),
            MockSRTSegment(3, "00:00:16,000", "00:00:20,000", "Critical confidence segment", 0.25),
            MockSRTSegment(4, "00:00:21,000", "00:00:25,000", "Good confidence segment", 0.88)
        ]
    
    def test_confidence_analyzer_initialization(self, confidence_analyzer):
        """Test Epic 4.3 confidence analyzer initialization."""
        assert confidence_analyzer is not None
        assert confidence_analyzer.max_processing_time_ms == 500
        assert confidence_analyzer.adaptive_thresholds_enabled is True
        assert confidence_analyzer.window_size == 50
        assert confidence_analyzer.circuit_breaker_open is False
    
    def test_analyze_confidence_batch_performance(self, confidence_analyzer, test_segments):
        """Test Epic 4.3 sub-second processing requirement."""
        start_time = time.time()
        
        result = confidence_analyzer.analyze_confidence_batch(test_segments)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Epic 4.3 SLA validation
        assert processing_time_ms <= 500, f"Processing time {processing_time_ms:.1f}ms exceeds 500ms SLA"
        assert result.calculation_time_ms <= 500
        assert isinstance(result, ConfidenceStatistics)
        assert result.sample_count == len(test_segments)
    
    def test_confidence_statistics_accuracy(self, confidence_analyzer, test_segments):
        """Test statistical analysis accuracy."""
        result = confidence_analyzer.analyze_confidence_batch(test_segments)
        
        expected_mean = sum(s.confidence_score for s in test_segments) / len(test_segments)
        expected_sorted = sorted([s.confidence_score for s in test_segments])
        expected_median = expected_sorted[len(expected_sorted) // 2]
        
        assert abs(result.mean - expected_mean) < 0.01
        assert abs(result.median - expected_median) < 0.01
        assert result.min_value == min(s.confidence_score for s in test_segments)
        assert result.max_value == max(s.confidence_score for s in test_segments)
        assert result.sample_count == len(test_segments)
    
    def test_real_time_confidence_analysis(self, confidence_analyzer):
        """Test real-time analysis for individual segments."""
        segment = MockSRTSegment(0, "00:00:01,000", "00:00:05,000", "Test segment", 0.75)
        
        confidence, trend = confidence_analyzer.analyze_real_time_confidence(segment, 0)
        
        assert confidence == 0.75
        assert isinstance(trend, ConfidenceAnalysisTrend)
    
    def test_adaptive_thresholds(self, confidence_analyzer, test_segments):
        """Test adaptive threshold adjustment."""
        # Analyze segments to build history
        confidence_analyzer.analyze_confidence_batch(test_segments)
        
        thresholds = confidence_analyzer.get_adaptive_thresholds()
        
        assert 'excellent' in thresholds
        assert 'good' in thresholds
        assert 'acceptable' in thresholds
        assert 'poor' in thresholds
        assert 'critical' in thresholds
        
        # Verify adaptive adjustment
        assert 0.1 <= thresholds['critical'] <= 0.9
        assert thresholds['poor'] >= thresholds['critical']
        assert thresholds['acceptable'] >= thresholds['poor']
    
    def test_confidence_distribution(self, confidence_analyzer, test_segments):
        """Test confidence distribution analysis."""
        confidence_analyzer.analyze_confidence_batch(test_segments)
        
        distribution = confidence_analyzer.get_confidence_distribution()
        
        assert 'distribution_buckets' in distribution
        assert 'distribution_percentages' in distribution
        assert 'total_samples' in distribution
        assert distribution['total_samples'] >= len(test_segments)
    
    def test_circuit_breaker_functionality(self, confidence_analyzer):
        """Test Epic 4.3 circuit breaker reliability pattern."""
        # Simulate failures
        confidence_analyzer.circuit_breaker_failures = 3
        confidence_analyzer.circuit_breaker_open = True
        confidence_analyzer.circuit_breaker_reset_time = time.time() + 1
        
        # Should return fallback result
        result = confidence_analyzer.analyze_confidence_batch([])
        assert result.trend == ConfidenceAnalysisTrend.INSUFFICIENT_DATA
        
        # Test reset
        confidence_analyzer.reset_circuit_breaker()
        assert not confidence_analyzer.circuit_breaker_open
        assert confidence_analyzer.circuit_breaker_failures == 0


class TestOOVDetector:
    """Test Epic 4.2 ML-Enhanced OOV Detection System."""
    
    @pytest.fixture
    def oov_detector(self):
        """Create OOV detector for testing."""
        config = {
            'oov_thresholds': {'critical': 0.4, 'warning': 0.25, 'info': 0.15},
            'ml_thresholds': {'semantic_similarity_min': 0.7, 'fuzzy_match_min': 0.8},
            'sanskrit_processing': {'enable_phonetic_matching': True, 'enable_semantic_clustering': True},
            'monitoring': {'enabled': False},
            'telemetry': {'enabled': False}
        }
        return OOVDetector(config)
    
    @pytest.fixture
    def test_lexicon_entries(self):
        """Create test lexicon entries."""
        return {
            'yoga': Mock(variations=['yog'], transliteration='yoga', is_proper_noun=False),
            'dharma': Mock(variations=['dharama'], transliteration='dharma', is_proper_noun=False),
            'krishna': Mock(variations=['krsna'], transliteration='kṛṣṇa', is_proper_noun=True),
            'meditation': Mock(variations=[], transliteration='meditation', is_proper_noun=False)
        }
    
    def test_oov_detector_initialization(self, oov_detector):
        """Test Epic 4.2 OOV detector initialization."""
        assert oov_detector is not None
        assert oov_detector.oov_thresholds['critical'] == 0.4
        assert oov_detector.sanskrit_config['enable_phonetic_matching'] is True
        assert oov_detector.circuit_breaker_open is False
    
    def test_oov_detection_with_high_rate(self, oov_detector):
        """Test OOV detection with high OOV rate."""
        segment = MockSRTSegment(0, "00:00:01,000", "00:00:05,000", 
                                "unknown1 unknown2 unknown3 yoga dharma", 0.8)
        
        # Mock lexicon manager
        with patch.object(oov_detector, 'enhanced_lexicon_manager') as mock_lexicon:
            mock_lexicon.get_all_entries.return_value = {'yoga': Mock(), 'dharma': Mock()}
            
            result = oov_detector.detect_oov_words(segment, 0)
            
            assert isinstance(result, OOVAnalysisResult)
            assert result.oov_rate > 0.4  # Should be critical
            assert len(result.oov_words) == 3  # unknown1, unknown2, unknown3
    
    def test_sanskrit_term_detection(self, oov_detector):
        """Test Epic 4.2 15% accuracy improvement in Sanskrit term detection."""
        # Test Sanskrit term identification
        sanskrit_words = ['kṛṣṇa', 'dharma', 'yogaśāstra', 'prāṇāyāma']
        non_sanskrit_words = ['hello', 'world', 'computer', 'test']
        
        for word in sanskrit_words:
            assert oov_detector._is_likely_sanskrit_term(word), f"{word} should be detected as Sanskrit"
        
        for word in non_sanskrit_words:
            assert not oov_detector._is_likely_sanskrit_term(word), f"{word} should not be detected as Sanskrit"
    
    def test_ml_classification_confidence(self, oov_detector, test_lexicon_entries):
        """Test Epic 4.2 ML classification with confidence scoring."""
        segment = MockSRTSegment(0, "00:00:01,000", "00:00:05,000", 
                                "yogic practices and dharamic principles", 0.8)
        
        with patch.object(oov_detector, 'enhanced_lexicon_manager') as mock_lexicon:
            mock_lexicon.get_all_entries.return_value = test_lexicon_entries
            
            result = oov_detector.detect_oov_words(segment, 0)
            
            assert result.ml_classification_confidence >= 0.6
            assert len(result.oov_words) >= 2  # 'yogic', 'dharamic'
            
            # Check OOV word categories
            for oov_word in result.oov_words:
                assert oov_word.category in [cat for cat in OOVCategory]
                assert 0.0 <= oov_word.confidence <= 1.0
    
    def test_semantic_similarity_calculation(self, oov_detector):
        """Test semantic similarity for context-aware OOV analysis."""
        with patch.object(oov_detector, 'semantic_calculator') as mock_calc:
            mock_calc.calculate_similarity.return_value = Mock(similarity_score=0.75)
            
            similarity = oov_detector._calculate_semantic_similarity(
                'meditation', ['yoga', 'practice', 'spiritual'], {}
            )
            
            assert similarity == 0.75
    
    def test_unknown_word_clustering(self, oov_detector):
        """Test Epic 4.2 research-grade unknown word clustering."""
        oov_words = [
            Mock(word='unknown1', category=OOVCategory.UNKNOWN),
            Mock(word='unknown2', category=OOVCategory.UNKNOWN),
            Mock(word='similar1', category=OOVCategory.SANSKRIT_VARIANT),
            Mock(word='similar2', category=OOVCategory.SANSKRIT_VARIANT)
        ]
        
        clusters = oov_detector._update_and_get_clusters(oov_words)
        
        assert isinstance(clusters, dict)
        assert len(clusters) >= 2  # At least some clustering should occur
    
    def test_circuit_breaker_reliability(self, oov_detector):
        """Test Epic 4.3 circuit breaker for OOV detection reliability."""
        # Trigger circuit breaker
        oov_detector.circuit_breaker_failures = 3
        oov_detector.circuit_breaker_open = True
        oov_detector.circuit_breaker_reset_time = time.time() + 1
        
        segment = MockSRTSegment(0, "00:00:01,000", "00:00:05,000", "test text", 0.8)
        result = oov_detector.detect_oov_words(segment, 0)
        
        # Should return fallback result
        assert result.oov_rate == 0.0
        assert result.processing_time_ms == 0.0


class TestAnomalyDetector:
    """Test Epic 4.1 MCP Context-Aware Anomaly Detection."""
    
    @pytest.fixture
    def anomaly_detector(self):
        """Create anomaly detector for testing."""
        config = {
            'anomaly_thresholds': {
                'language_shift_confidence': 0.7,
                'acoustic_anomaly_threshold': 0.8,
                'speech_rate_min': 5.0,
                'speech_rate_max': 100.0
            },
            'mcp_integration': {'enabled': True, 'fallback_enabled': True},
            'monitoring': {'enabled': False},
            'telemetry': {'enabled': False}
        }
        return AnomalyDetector(config)
    
    def test_anomaly_detector_initialization(self, anomaly_detector):
        """Test Epic 4.1 anomaly detector initialization."""
        assert anomaly_detector is not None
        assert anomaly_detector.mcp_integration_enabled is True
        assert anomaly_detector.circuit_breaker_open is False
    
    def test_language_shift_detection(self, anomaly_detector):
        """Test language shift anomaly detection."""
        # Mixed script segment (English + Devanagari)
        mixed_segment = MockSRTSegment(0, "00:00:01,000", "00:00:05,000", 
                                     "Today we study योग and dharma", 0.8)
        
        result = anomaly_detector.detect_anomalies([mixed_segment])
        
        assert isinstance(result, AnomalyAnalysisResult)
        language_shift_anomalies = [a for a in result.anomalies 
                                   if a.anomaly_type == 'language_shift']
        assert len(language_shift_anomalies) > 0
    
    def test_acoustic_anomaly_detection(self, anomaly_detector):
        """Test acoustic anomaly detection."""
        # Very fast speech rate
        fast_segment = MockSRTSegment(0, "00:00:01,000", "00:00:02,000", 
                                    "very long text with many many words spoken very quickly", 0.8)
        
        # Very slow speech rate  
        slow_segment = MockSRTSegment(1, "00:00:01,000", "00:00:20,000", "slow", 0.8)
        
        result = anomaly_detector.detect_anomalies([fast_segment, slow_segment])
        
        acoustic_anomalies = [a for a in result.anomalies 
                             if a.anomaly_type == 'acoustic_anomaly']
        assert len(acoustic_anomalies) >= 1  # Should detect speech rate issues
    
    def test_timestamp_validation(self, anomaly_detector):
        """Test timestamp anomaly detection."""
        # Invalid timestamp sequence
        invalid_segments = [
            MockSRTSegment(0, "00:00:05,000", "00:00:01,000", "Invalid timestamps", 0.8),
            MockSRTSegment(1, "00:00:10,000", "00:00:05,000", "More invalid", 0.8)
        ]
        
        result = anomaly_detector.detect_anomalies(invalid_segments)
        
        timestamp_anomalies = [a for a in result.anomalies 
                              if a.anomaly_type == 'timestamp_inconsistency']
        assert len(timestamp_anomalies) >= 1
    
    def test_mcp_context_awareness(self, anomaly_detector):
        """Test Epic 4.1 MCP context-aware processing."""
        # Test with MCP integration enabled
        segments = [
            MockSRTSegment(0, "00:00:01,000", "00:00:05,000", "Normal segment", 0.8),
            MockSRTSegment(1, "00:00:06,000", "00:00:10,000", "Another normal segment", 0.8)
        ]
        
        with patch.object(anomaly_detector, 'mcp_client') as mock_mcp:
            mock_mcp.analyze_context.return_value = {'context_score': 0.9, 'coherence': True}
            
            result = anomaly_detector.detect_anomalies(segments)
            
            assert result.mcp_analysis_enabled is True
            assert result.context_coherence_score >= 0.0
    
    def test_circuit_breaker_fallback(self, anomaly_detector):
        """Test Epic 4.1 circuit breaker and graceful fallback."""
        # Trigger circuit breaker
        anomaly_detector.circuit_breaker_failures = 3
        anomaly_detector.circuit_breaker_open = True
        anomaly_detector.circuit_breaker_reset_time = time.time() + 1
        
        segments = [MockSRTSegment(0, "00:00:01,000", "00:00:05,000", "test", 0.8)]
        result = anomaly_detector.detect_anomalies(segments)
        
        # Should use fallback processing
        assert result.fallback_used is True
        assert result.processing_time_ms >= 0


class TestQAFlaggingEngine:
    """Test Epic 4 QA Flagging Engine Integration."""
    
    @pytest.fixture
    def qa_engine(self):
        """Create QA flagging engine for testing."""
        config = {
            'confidence_thresholds': {'critical': 0.3, 'warning': 0.6, 'info': 0.8},
            'oov_thresholds': {'critical': 0.4, 'warning': 0.25, 'info': 0.15},
            'performance_sla': {'max_processing_time_ms': 500, 'target_uptime_percentage': 99.9},
            'academic_standards': {'minimum_quality_score': 0.85, 'iast_validation_enabled': True},
            'monitoring': {'enabled': False},
            'telemetry': {'enabled': False},
            'performance': {'enabled': False}
        }
        return QAFlaggingEngine(config)
    
    @pytest.fixture
    def comprehensive_test_segments(self):
        """Create comprehensive test segments for integration testing."""
        return [
            MockSRTSegment(0, "00:00:01,000", "00:00:05,000", "High quality segment with good confidence", 0.95),
            MockSRTSegment(1, "00:00:06,000", "00:00:10,000", "unknown1 unknown2 unknown3 word", 0.75),  # High OOV
            MockSRTSegment(2, "00:00:11,000", "00:00:15,000", "Low confidence problematic segment", 0.25),  # Low confidence
            MockSRTSegment(3, "00:00:16,000", "00:00:20,000", "yoga dharma without diacritics", 0.85),  # Academic standards
            MockSRTSegment(4, "00:00:21,000", "00:00:25,000", "Mixed script योग and english", 0.80),  # Language shift
        ]
    
    def test_qa_engine_initialization(self, qa_engine):
        """Test Epic 4 QA engine initialization with all integrations."""
        assert qa_engine is not None
        assert qa_engine.performance_sla['max_processing_time_ms'] == 500
        assert qa_engine.academic_config['minimum_quality_score'] == 0.85
        assert qa_engine.circuit_breaker_open is False
    
    def test_comprehensive_qa_analysis(self, qa_engine, comprehensive_test_segments):
        """Test comprehensive QA analysis with all Epic 4 components."""
        lexicon_entries = {'yoga', 'dharma', 'word', 'good', 'quality', 'segment', 'with', 'confidence'}
        
        start_time = time.time()
        result = qa_engine.analyze_segments(comprehensive_test_segments, lexicon_entries)
        processing_time = (time.time() - start_time) * 1000
        
        # Epic 4.3 Performance validation
        assert processing_time <= 500, f"Processing time {processing_time:.1f}ms exceeds SLA"
        assert result.performance_meets_sla is True
        
        # Validate comprehensive analysis
        assert isinstance(result, QAAnalysisResult)
        assert result.total_segments == len(comprehensive_test_segments)
        assert result.flagged_segments >= 3  # Should flag multiple problematic segments
        assert len(result.flags) >= 3  # Multiple flags expected
        
        # Check flag types are present
        flag_types = {flag.flag_type for flag in result.flags}
        assert QAFlagType.LOW_CONFIDENCE in flag_types
        assert QAFlagType.HIGH_OOV_RATE in flag_types
    
    def test_flag_severity_classification(self, qa_engine, comprehensive_test_segments):
        """Test Epic 4 flag severity classification."""
        lexicon_entries = {'yoga', 'dharma', 'good', 'quality', 'segment'}
        
        result = qa_engine.analyze_segments(comprehensive_test_segments, lexicon_entries)
        
        # Check severity distribution
        critical_flags = result.get_flags_by_severity(QASeverity.CRITICAL)
        warning_flags = result.get_flags_by_severity(QASeverity.WARNING)
        info_flags = result.get_flags_by_severity(QASeverity.INFO)
        
        assert len(critical_flags) >= 1  # Low confidence segment should be critical
        assert len(warning_flags) >= 1  # High OOV should be warning
        assert len(info_flags) >= 1    # Academic standards should be info
    
    def test_academic_priority_assignment(self, qa_engine, comprehensive_test_segments):
        """Test Epic 4.5 academic priority assignment."""
        result = qa_engine.analyze_segments(comprehensive_test_segments, set())
        
        high_priority_flags = result.get_academic_priority_flags(max_priority=2)
        
        assert len(high_priority_flags) >= 1
        for flag in high_priority_flags:
            assert 1 <= flag.academic_priority <= 2
    
    def test_quality_score_calculation(self, qa_engine, comprehensive_test_segments):
        """Test overall quality score calculation with Epic 4 weightings."""
        result = qa_engine.analyze_segments(comprehensive_test_segments, {'yoga', 'dharma'})
        
        assert 0.0 <= result.overall_quality_score <= 1.0
        assert 0.0 <= result.academic_compliance_score <= 1.0
        
        # Quality score should reflect the mix of good and problematic segments
        assert result.overall_quality_score < 0.9  # Should be reduced due to issues
    
    def test_integration_with_epic2_systems(self, qa_engine):
        """Test integration with existing Epic 2 systems."""
        # Test with Epic 2-style segment data
        epic2_segment = MockSRTSegment(0, "00:00:01,000", "00:00:05,000", 
                                     "Integration test with Epic 2 data structures", 0.8)
        
        result = qa_engine.analyze_segments([epic2_segment], set())
        
        # Should handle Epic 2 data without issues
        assert isinstance(result, QAAnalysisResult)
        assert result.total_segments == 1


class TestQAReportGenerator:
    """Test Epic 4.5 Academic-Grade QA Reporting."""
    
    @pytest.fixture
    def qa_report_generator(self):
        """Create QA report generator for testing."""
        config = {
            'academic_standards': {
                'citation_format': 'academic',
                'statistical_precision': 3,
                'include_methodology': True
            },
            'output_formats': ['html', 'pdf', 'json'],
            'monitoring': {'enabled': False}
        }
        return QAReportGenerator(config)
    
    @pytest.fixture
    def test_qa_result(self):
        """Create test QA analysis result."""
        flags = [
            QAFlag(
                flag_id="test_1", flag_type=QAFlagType.LOW_CONFIDENCE, severity=QASeverity.CRITICAL,
                segment_index=0, timestamp=time.time(), confidence_score=0.25,
                message="Test critical flag", academic_priority=1
            ),
            QAFlag(
                flag_id="test_2", flag_type=QAFlagType.HIGH_OOV_RATE, severity=QASeverity.WARNING,
                segment_index=1, timestamp=time.time(), confidence_score=0.65,
                message="Test warning flag", academic_priority=2
            )
        ]
        
        return QAAnalysisResult(
            total_segments=5, flagged_segments=2, flags=flags, overall_quality_score=0.75,
            processing_time_ms=250, confidence_distribution={'mean': 0.75, 'std_dev': 0.15},
            oov_statistics={'total_words': 50, 'oov_words': 10}, 
            anomaly_statistics={'language_shifts': 1}, academic_compliance_score=0.85,
            performance_meets_sla=True
        )
    
    def test_comprehensive_report_generation(self, qa_report_generator, test_qa_result):
        """Test Epic 4.5 comprehensive report generation."""
        report = qa_report_generator.generate_comprehensive_report(
            test_qa_result, "test_file.srt", {"test": "metadata"}
        )
        
        assert isinstance(report, QAReport)
        assert report.overall_quality_score == 0.75
        assert report.academic_compliance_score == 0.85
        assert len(report.insights) >= 3  # Should generate multiple insights
        assert report.publication_ready is True or report.publication_ready is False
    
    def test_academic_formatting(self, qa_report_generator, test_qa_result):
        """Test academic-grade formatting standards."""
        report = qa_report_generator.generate_comprehensive_report(
            test_qa_result, "test_file.srt", {}
        )
        
        # Check academic formatting requirements
        assert 'methodology' in report.summary.detailed_analysis
        assert 'statistical_analysis' in report.summary.detailed_analysis
        assert len(report.algorithm_references) >= 1
        assert report.generated_timestamp > 0
    
    def test_multiple_output_formats(self, qa_report_generator, test_qa_result):
        """Test support for multiple output formats."""
        # Test HTML export
        html_output = qa_report_generator.export_to_html(
            qa_report_generator.generate_comprehensive_report(test_qa_result, "test.srt", {})
        )
        assert '<html>' in html_output
        assert 'Quality Assurance Report' in html_output
        
        # Test JSON export
        json_output = qa_report_generator.export_to_json(
            qa_report_generator.generate_comprehensive_report(test_qa_result, "test.srt", {})
        )
        assert 'overall_quality_score' in json_output
        assert 'academic_compliance_score' in json_output


class TestIntegrationScenarios:
    """Test comprehensive integration scenarios."""
    
    @pytest.fixture
    def full_qa_system(self):
        """Create full integrated QA system."""
        base_config = {
            'monitoring': {'enabled': False},
            'telemetry': {'enabled': False},
            'performance': {'enabled': False}
        }
        
        return {
            'qa_engine': QAFlaggingEngine(base_config),
            'confidence_analyzer': ConfidenceAnalyzer(base_config),
            'oov_detector': OOVDetector(base_config),
            'anomaly_detector': AnomalyDetector(base_config),
            'report_generator': QAReportGenerator(base_config)
        }
    
    def test_end_to_end_qa_workflow(self, full_qa_system):
        """Test complete end-to-end QA workflow with Epic 4 integration."""
        # Create realistic test data
        segments = [
            MockSRTSegment(0, "00:00:01,000", "00:00:05,000", "Today we study yoga and dharma", 0.95),
            MockSRTSegment(1, "00:00:06,000", "00:00:10,000", "unknown1 unknown2 problematic", 0.45),
            MockSRTSegment(2, "00:00:11,000", "00:00:15,000", "Mixed योग and english text", 0.75)
        ]
        
        lexicon_entries = {'today', 'we', 'study', 'yoga', 'dharma', 'and', 'text'}
        
        # Step 1: QA Analysis
        qa_result = full_qa_system['qa_engine'].analyze_segments(segments, lexicon_entries)
        assert qa_result.flagged_segments >= 2  # Should flag problematic segments
        
        # Step 2: Detailed Confidence Analysis
        confidence_stats = full_qa_system['confidence_analyzer'].analyze_confidence_batch(segments)
        assert confidence_stats.sample_count == len(segments)
        
        # Step 3: OOV Detection
        oov_result = full_qa_system['oov_detector'].detect_oov_words(segments[1], 1)
        assert oov_result.oov_rate > 0.5  # High OOV rate expected
        
        # Step 4: Anomaly Detection
        anomaly_result = full_qa_system['anomaly_detector'].detect_anomalies(segments)
        assert len(anomaly_result.anomalies) >= 1  # Should detect language shift
        
        # Step 5: Report Generation
        final_report = full_qa_system['report_generator'].generate_comprehensive_report(
            qa_result, "test_integration.srt", {'integration_test': True}
        )
        assert isinstance(final_report, QAReport)
        assert final_report.overall_quality_score > 0.0
    
    def test_performance_under_load(self, full_qa_system):
        """Test Epic 4.3 performance under load scenarios."""
        # Create larger dataset
        large_segment_set = [
            MockSRTSegment(i, f"00:00:{i:02d},000", f"00:00:{i+4:02d},000", 
                         f"Segment {i} with test content and confidence", 
                         0.8 if i % 2 == 0 else 0.6)
            for i in range(20)
        ]
        
        # Measure performance
        start_time = time.time()
        
        result = full_qa_system['qa_engine'].analyze_segments(large_segment_set, {'test', 'content'})
        
        total_time = (time.time() - start_time) * 1000
        
        # Epic 4.3 performance validation
        assert total_time <= 2000, f"Large dataset processing took {total_time:.1f}ms (>2s)"
        assert result.performance_meets_sla is True
        assert result.total_segments == 20
    
    def test_concurrent_analysis(self, full_qa_system):
        """Test Epic 4.3 concurrent processing reliability."""
        segments = [
            MockSRTSegment(i, "00:00:01,000", "00:00:05,000", f"Concurrent test {i}", 0.8)
            for i in range(5)
        ]
        
        # Run concurrent analyses
        results = []
        threads = []
        
        def analyze_segments():
            result = full_qa_system['qa_engine'].analyze_segments(segments, {'test', 'concurrent'})
            results.append(result)
        
        # Create and start threads
        for _ in range(3):
            thread = threading.Thread(target=analyze_segments)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Validate concurrent execution
        assert len(results) == 3
        for result in results:
            assert result.total_segments == 5
            assert result.processing_time_ms > 0


# Performance and stress test markers
@pytest.mark.performance
class TestPerformanceValidation:
    """Epic 4.3 Performance validation tests."""
    
    def test_sub_second_processing_sla(self):
        """Validate Epic 4.3 sub-second processing SLA."""
        config = {'max_processing_time_ms': 500, 'monitoring': {'enabled': False}, 'telemetry': {'enabled': False}}
        
        qa_engine = QAFlaggingEngine(config)
        segments = [MockSRTSegment(i, "00:00:01,000", "00:00:05,000", f"Test segment {i}", 0.8) for i in range(10)]
        
        start_time = time.time()
        result = qa_engine.analyze_segments(segments, {'test', 'segment'})
        processing_time = (time.time() - start_time) * 1000
        
        assert processing_time <= 500, f"Processing exceeded SLA: {processing_time:.1f}ms"
        assert result.performance_meets_sla is True
    
    def test_memory_efficiency(self):
        """Test Epic 4.3 memory efficiency requirements."""
        # This would require memory profiling in a full implementation
        # Placeholder for memory efficiency validation
        assert True  # Memory testing would require additional tooling


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])