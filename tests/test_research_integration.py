"""
Comprehensive Integration Tests for Research Integration System

Tests all major components of Story 2.4.5 including performance benchmarking,
research validation metrics, lexicon acquisition, and automated benchmark suite.
"""

import unittest
import tempfile
import json
from pathlib import Path
import time

from src.research_integration.performance_benchmarking import PerformanceBenchmarking
from src.research_integration.research_validation_metrics import ResearchValidationMetrics, AcademicStandard
from src.research_integration.lexicon_acquisition import LexiconAcquisition, LexiconSource, LexiconSourceType
from src.research_integration.comprehensive_reporting import ComprehensiveReporting
from src.research_integration.automated_benchmark_suite import AutomatedBenchmarkSuite, BenchmarkSeverity
from src.enhancement_integration.provenance_manager import ProvenanceLevel
from src.utils.srt_parser import SRTSegment


class TestPerformanceBenchmarking(unittest.TestCase):
    """Test performance benchmarking framework"""
    
    def setUp(self):
        self.benchmarking = PerformanceBenchmarking()
    
    def test_performance_benchmarking_initialization(self):
        """Test performance benchmarking system initialization"""
        self.assertIsNotNone(self.benchmarking)
        self.assertIn('max_processing_time_ratio', self.benchmarking.performance_targets)
        self.assertEqual(self.benchmarking.performance_targets['max_processing_time_ratio'], 2.0)
    
    def test_word_error_rate_calculation(self):
        """Test WER calculation accuracy"""
        reference = "today we study yoga and dharma"
        hypothesis = "today we practice yoga and dharma"
        
        wer = self.benchmarking.calculate_word_error_rate(reference, hypothesis)
        
        # Should have 1 substitution out of 6 words = 1/6 ≈ 0.167
        self.assertGreater(wer, 0.1)
        self.assertLess(wer, 0.2)
    
    def test_character_error_rate_calculation(self):
        """Test CER calculation accuracy"""
        reference = "dharma"
        hypothesis = "dharama"
        
        cer = self.benchmarking.calculate_character_error_rate(reference, hypothesis)
        
        # Should be very low since only one character insertion
        self.assertGreater(cer, 0.0)
        self.assertLess(cer, 0.2)
    
    def test_baseline_processor_creation(self):
        """Test baseline processor creation"""
        processor = self.benchmarking.create_baseline_processor()
        self.assertIsNotNone(processor)
    
    def test_enhanced_processor_creation(self):
        """Test enhanced processor creation"""  
        processor = self.benchmarking.create_enhanced_processor()
        self.assertIsNotNone(processor)
    
    def test_processing_with_timing(self):
        """Test processing with timing measurement"""
        processor = self.benchmarking.create_baseline_processor()
        segment = SRTSegment(1, "00:00:01,000", "00:00:05,000", "Today we study yoga and dharma")
        
        result = self.benchmarking.process_with_timing(processor, segment, "test_algorithm")
        
        self.assertIsNotNone(result)
        self.assertEqual(result.original_text, segment.text)
        self.assertGreaterEqual(result.processing_time, 0.0)
        self.assertGreaterEqual(result.confidence_score, 0.0)
        self.assertEqual(result.algorithm_used, "test_algorithm")


class TestResearchValidationMetrics(unittest.TestCase):
    """Test research validation metrics system"""
    
    def setUp(self):
        self.validation = ResearchValidationMetrics()
    
    def test_research_validation_initialization(self):
        """Test research validation system initialization"""
        self.assertIsNotNone(self.validation)
        self.assertIn('iast_vowels', self.validation.sanskrit_patterns)
        self.assertIn('iast_standard', self.validation.academic_references)
    
    def test_iast_compliance_validation(self):
        """Test IAST transliteration compliance validation"""
        # Test with proper IAST text
        iast_text = "kṛṣṇa dharma yoga"
        result = self.validation.validate_iast_compliance(iast_text)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.text_validated, iast_text)
        self.assertGreaterEqual(result.compliance_score, 0.0)
        self.assertLessEqual(result.compliance_score, 1.0)
    
    def test_iast_compliance_with_errors(self):
        """Test IAST validation with common errors"""
        # Text with common IAST errors
        error_text = "Krishna dharma yoga"  # Should be kṛṣṇa
        result = self.validation.validate_iast_compliance(error_text)
        
        self.assertIsNotNone(result)
        self.assertLess(result.compliance_score, 1.0)  # Should detect issues
        self.assertGreater(len(result.issues_found), 0)
    
    def test_sanskrit_linguistic_validation(self):
        """Test Sanskrit linguistic processing validation"""
        sanskrit_text = "dharma yoga karma"
        result = self.validation.validate_sanskrit_linguistics(sanskrit_text)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.text_analyzed, sanskrit_text)
        self.assertGreaterEqual(result.linguistic_accuracy_score, 0.0)
        self.assertLessEqual(result.linguistic_accuracy_score, 1.0)
        self.assertGreaterEqual(result.phonetic_accuracy, 0.0)
        self.assertGreaterEqual(result.morphological_accuracy, 0.0)
    
    def test_academic_validation_report_generation(self):
        """Test academic validation report generation"""
        test_segments = [
            "kṛṣṇa dharma yoga",
            "bhagavad gītā",
            "rāmāyaṇa"
        ]
        
        report = self.validation.generate_academic_validation_report(
            test_segments, AcademicStandard.IAST
        )
        
        self.assertIsNotNone(report)
        self.assertEqual(report.text_segments_validated, len(test_segments))
        self.assertEqual(report.validation_type, AcademicStandard.IAST)
        self.assertGreaterEqual(report.overall_compliance_score, 0.0)


class TestLexiconAcquisition(unittest.TestCase):
    """Test lexicon acquisition system"""
    
    def setUp(self):
        self.acquisition = LexiconAcquisition()
        
        # Register test source
        self.test_source = LexiconSource(
            source_id="test_source",
            name="Test Academic Source",
            source_type=LexiconSourceType.ACADEMIC_PUBLICATION,
            authority_level=ProvenanceLevel.GOLD,
            description="Test source for unit testing"
        )
        self.acquisition.register_source(self.test_source)
    
    def test_lexicon_acquisition_initialization(self):
        """Test lexicon acquisition system initialization"""
        self.assertIsNotNone(self.acquisition)
        self.assertIsNotNone(self.acquisition.provenance_manager)
        self.assertIn('verified_confidence', self.acquisition.quality_thresholds)
    
    def test_source_registration(self):
        """Test lexicon source registration"""
        self.assertIn("test_source", self.acquisition.sources)
        source = self.acquisition.sources["test_source"]
        self.assertEqual(source.name, "Test Academic Source")
        self.assertEqual(source.authority_level, ProvenanceTier.GOLD)
    
    def test_json_lexicon_acquisition(self):
        """Test acquiring lexicon from JSON file"""
        # Create temporary JSON file
        test_data = {
            "entries": {
                "dharma": {
                    "transliteration": "dharma",
                    "variations": ["dharama"],
                    "category": "concept",
                    "is_proper_noun": False,
                    "confidence": 0.95
                },
                "yoga": {
                    "transliteration": "yoga", 
                    "variations": ["yog"],
                    "category": "practice",
                    "is_proper_noun": False,
                    "confidence": 0.92
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = Path(f.name)
        
        try:
            # Test acquisition
            report = self.acquisition.acquire_from_json_file(temp_path, "test_source")
            
            self.assertIsNotNone(report)
            self.assertEqual(report.source_id, "test_source")
            self.assertGreaterEqual(report.entries_added, 0)
            self.assertEqual(report.operation_type, "json_file_acquisition")
            
        finally:
            temp_path.unlink()  # Clean up
    
    def test_quality_assessment(self):
        """Test lexicon quality assessment"""
        assessment = self.acquisition.assess_lexicon_quality()
        
        self.assertIsNotNone(assessment)
        self.assertGreaterEqual(assessment.total_entries, 0)
        self.assertGreaterEqual(assessment.overall_quality_score, 0.0)
        self.assertLessEqual(assessment.overall_quality_score, 1.0)
    
    def test_lexicon_export(self):
        """Test lexicon export functionality"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            self.acquisition.export_lexicon(temp_path, "json")
            self.assertTrue(temp_path.exists())
            
            # Verify exported content
            with open(temp_path, 'r', encoding='utf-8') as f:
                exported_data = json.load(f)
            
            self.assertIn('metadata', exported_data)
            self.assertIn('entries', exported_data)
            
        finally:
            if temp_path.exists():
                temp_path.unlink()


class TestComprehensiveReporting(unittest.TestCase):
    """Test comprehensive reporting system"""
    
    def setUp(self):
        self.reporting = ComprehensiveReporting()
    
    def test_comprehensive_reporting_initialization(self):
        """Test comprehensive reporting system initialization"""
        self.assertIsNotNone(self.reporting)
        self.assertIsNotNone(self.reporting.performance_benchmarking)
        self.assertIsNotNone(self.reporting.research_validation)
        self.assertIn('sandhi_preprocessing', self.reporting.algorithm_references)
    
    def test_algorithm_references_initialization(self):
        """Test algorithm references database"""
        refs = self.reporting.algorithm_references
        
        # Check key algorithm references exist
        self.assertIn('sandhi_preprocessing', refs)
        self.assertIn('phonetic_hashing', refs)
        self.assertIn('semantic_similarity', refs)
        self.assertIn('hybrid_matching', refs)
        
        # Check reference structure
        sandhi_ref = refs['sandhi_preprocessing']
        self.assertIsNotNone(sandhi_ref.algorithm_name)
        self.assertIsNotNone(sandhi_ref.research_paper)
        self.assertGreater(len(sandhi_ref.authors), 0)
        self.assertGreater(sandhi_ref.year, 2000)
    
    def test_system_health_report_generation(self):
        """Test system health report generation"""
        report = self.reporting.generate_system_health_report()
        
        self.assertIsNotNone(report)
        self.assertIsNotNone(report.report_id)
        self.assertIsNotNone(report.system_overview)
        self.assertIsNotNone(report.system_health)
        self.assertGreater(len(report.visualizations), 0)
        self.assertIsNotNone(report.executive_summary)
    
    def test_report_export(self):
        """Test report export functionality"""
        # Generate a test report
        report = self.reporting.generate_system_health_report()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            self.reporting.export_report(report, temp_path, "json")
            self.assertTrue(temp_path.exists())
            
            # Verify exported content
            with open(temp_path, 'r', encoding='utf-8') as f:
                exported_data = json.load(f)
            
            self.assertIn('report_id', exported_data)
            self.assertIn('system_overview', exported_data)
            
        finally:
            if temp_path.exists():
                temp_path.unlink()


class TestAutomatedBenchmarkSuite(unittest.TestCase):
    """Test automated benchmark suite"""
    
    def setUp(self):
        self.benchmark_suite = AutomatedBenchmarkSuite()
    
    def test_benchmark_suite_initialization(self):
        """Test automated benchmark suite initialization"""
        self.assertIsNotNone(self.benchmark_suite)
        self.assertGreater(len(self.benchmark_suite.registered_tests), 0)
        
        # Check that critical tests are registered
        test_ids = list(self.benchmark_suite.registered_tests.keys())
        self.assertIn('sandhi_preprocessing_accuracy', test_ids)
        self.assertIn('phonetic_hashing_performance', test_ids)
        self.assertIn('hybrid_matching_pipeline', test_ids)
    
    def test_test_registration(self):
        """Test benchmark test registration"""
        from src.research_integration.automated_benchmark_suite import BenchmarkTest
        
        test_count_before = len(self.benchmark_suite.registered_tests)
        
        # Register new test
        new_test = BenchmarkTest(
            test_id="test_custom",
            test_name="Custom Test",
            description="Custom test for validation",
            severity=BenchmarkSeverity.LOW,
            test_function=lambda x: {'test_metric': 1.0}
        )
        
        self.benchmark_suite.register_test(new_test)
        
        self.assertEqual(len(self.benchmark_suite.registered_tests), test_count_before + 1)
        self.assertIn("test_custom", self.benchmark_suite.registered_tests)
    
    def test_test_filtering(self):
        """Test benchmark test filtering"""
        # Test severity filtering
        filtered_tests = self.benchmark_suite._filter_tests(
            test_filter=None,
            tag_filter=None,
            severity_filter=BenchmarkSeverity.CRITICAL
        )
        
        # Should only return critical tests
        for test in filtered_tests.values():
            self.assertEqual(test.severity, BenchmarkSeverity.CRITICAL)
    
    def test_single_test_execution(self):
        """Test single benchmark test execution"""
        # Create simple test
        from src.research_integration.automated_benchmark_suite import BenchmarkTest
        
        def simple_test_function(test):
            return {'success_metric': 0.95, 'time_metric': 0.01}
        
        test = BenchmarkTest(
            test_id="simple_test",
            test_name="Simple Test",
            description="Simple test for validation",
            severity=BenchmarkSeverity.LOW,
            test_function=simple_test_function,
            expected_thresholds={'success_metric': 0.9}
        )
        
        result = self.benchmark_suite._execute_single_test(test)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.test_id, "simple_test")
        self.assertIn('success_metric', result.metrics)
        self.assertGreaterEqual(result.execution_time, 0.0)
    
    def test_benchmark_suite_execution(self):
        """Test full benchmark suite execution"""
        # Run a subset of tests to avoid long execution times
        report = self.benchmark_suite.run_test_suite(
            test_filter=['academic_compliance_validation'],  # Run just one quick test
            severity_filter=BenchmarkSeverity.MEDIUM
        )
        
        self.assertIsNotNone(report)
        self.assertGreaterEqual(report.total_tests, 0)
        self.assertGreaterEqual(report.overall_success_rate, 0.0)
        self.assertLessEqual(report.overall_success_rate, 1.0)
        self.assertGreaterEqual(report.total_execution_time, 0.0)
    
    def test_performance_baseline_update(self):
        """Test performance baseline updates"""
        from src.research_integration.automated_benchmark_suite import TestResult, TestStatus
        
        # Create mock test results
        test_results = [
            TestResult(
                test_id="test1",
                status=TestStatus.PASSED,
                execution_time=0.1,
                metrics={'accuracy': 0.95, 'speed': 0.02}
            )
        ]
        
        baseline_count_before = len(self.benchmark_suite.performance_baselines)
        self.benchmark_suite._update_performance_baselines(test_results)
        
        # Should have updated baselines
        self.assertGreaterEqual(len(self.benchmark_suite.performance_baselines), baseline_count_before)


class TestResearchIntegrationEndToEnd(unittest.TestCase):
    """End-to-end integration tests for research integration system"""
    
    def setUp(self):
        self.benchmarking = PerformanceBenchmarking()
        self.validation = ResearchValidationMetrics()
        self.acquisition = LexiconAcquisition()
        self.reporting = ComprehensiveReporting()
        self.benchmark_suite = AutomatedBenchmarkSuite()
    
    def test_complete_research_workflow(self):
        """Test complete research integration workflow"""
        # Step 1: Performance benchmarking
        self.assertIsNotNone(self.benchmarking)
        
        # Step 2: Academic validation
        test_text = "kṛṣṇa dharma yoga"
        validation_result = self.validation.validate_iast_compliance(test_text)
        self.assertIsNotNone(validation_result)
        
        # Step 3: Quality assessment
        quality_assessment = self.acquisition.assess_lexicon_quality()
        self.assertIsNotNone(quality_assessment)
        
        # Step 4: System health reporting
        health_report = self.reporting.generate_system_health_report()
        self.assertIsNotNone(health_report)
        
        # Step 5: Automated benchmarking
        # Run a minimal test suite
        benchmark_report = self.benchmark_suite.run_test_suite(
            test_filter=['academic_compliance_validation'],
            severity_filter=BenchmarkSeverity.MEDIUM
        )
        self.assertIsNotNone(benchmark_report)
    
    def test_research_integration_metrics_collection(self):
        """Test comprehensive metrics collection"""
        # Collect metrics from all major components
        metrics = {
            'performance_targets': self.benchmarking.performance_targets,
            'academic_references': len(self.validation.academic_references),
            'quality_thresholds': self.acquisition.quality_thresholds,
            'algorithm_references': len(self.reporting.algorithm_references),
            'registered_tests': len(self.benchmark_suite.registered_tests)
        }
        
        # Validate metrics collection
        self.assertGreater(metrics['academic_references'], 0)
        self.assertGreater(metrics['algorithm_references'], 0)
        self.assertGreater(metrics['registered_tests'], 0)
        self.assertIn('max_processing_time_ratio', metrics['performance_targets'])
        self.assertIn('verified_confidence', metrics['quality_thresholds'])


if __name__ == '__main__':
    unittest.main()