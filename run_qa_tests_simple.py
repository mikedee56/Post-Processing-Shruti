#!/usr/bin/env python3
"""
Simple test runner for QA module tests without pytest dependency.
This will identify which specific tests are failing.
"""

import sys
import time
import traceback
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def run_test_method(test_class_instance, method_name, *args):
    """Run a single test method and return result."""
    try:
        method = getattr(test_class_instance, method_name)
        method(*args) if args else method()
        return True, None
    except Exception as e:
        return False, str(e)

def run_qa_tests():
    """Run QA module tests and identify failures."""
    
    print("=== QA Module Test Suite Validation ===")
    print()
    
    # Import required modules
    try:
        from utils.srt_parser import SRTSegment
        from qa_module.confidence_analyzer import ConfidenceAnalyzer, ConfidenceAnalysisTrend, ConfidenceStatistics
        from qa_module.anomaly_detector import AnomalyDetector
        from qa_module.oov_detector import OOVDetector
        from qa_module.qa_flagging_engine import QAFlaggingEngine
        from qa_module.qa_report_generator import QAReportGenerator
        print("✅ All QA module imports successful")
    except Exception as e:
        print(f"❌ Import failure: {e}")
        traceback.print_exc()
        return False

    # Create mock SRT segment for testing
    class MockSRTSegment:
        def __init__(self, index, start_time, end_time, text, confidence_score=0.8):
            self.index = index
            self.start_time = start_time
            self.end_time = end_time
            self.text = text
            self.confidence_score = confidence_score

    test_results = {}
    failed_tests = []

    # Test 1: ConfidenceAnalyzer
    print("Testing ConfidenceAnalyzer...")
    try:
        config = {
            'max_processing_time_ms': 500,
            'confidence_window_size': 50,
            'adaptive_thresholds_enabled': True,
            'monitoring': {'enabled': False},
            'telemetry': {'enabled': False}
        }
        analyzer = ConfidenceAnalyzer(config)
        
        # Test segments
        test_segments = [
            MockSRTSegment(0, "00:00:01,000", "00:00:05,000", "High confidence segment", 0.95),
            MockSRTSegment(1, "00:00:06,000", "00:00:10,000", "Medium confidence segment", 0.75),
            MockSRTSegment(2, "00:00:11,000", "00:00:15,000", "Low confidence segment", 0.45),
        ]
        
        # Test batch analysis
        start_time = time.time()
        result = analyzer.analyze_confidence_batch(test_segments)
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Validate results
        assert isinstance(result, ConfidenceStatistics)
        assert result.sample_count == len(test_segments)
        assert processing_time_ms <= 500, f"Processing time {processing_time_ms:.1f}ms exceeds 500ms SLA"
        
        print("  ✅ ConfidenceAnalyzer batch analysis: PASS")
        test_results['confidence_analyzer_batch'] = True
        
        # Test real-time analysis
        segment = MockSRTSegment(0, "00:00:01,000", "00:00:05,000", "Test segment", 0.75)
        confidence, trend = analyzer.analyze_real_time_confidence(segment, 0)
        
        assert confidence == 0.75
        assert isinstance(trend, ConfidenceAnalysisTrend)
        
        print("  ✅ ConfidenceAnalyzer real-time analysis: PASS")
        test_results['confidence_analyzer_realtime'] = True
        
    except Exception as e:
        print(f"  ❌ ConfidenceAnalyzer tests failed: {e}")
        failed_tests.append(f"ConfidenceAnalyzer: {e}")
        test_results['confidence_analyzer'] = False

    # Test 2: AnomalyDetector
    print("Testing AnomalyDetector...")
    try:
        config = {
            'anomaly_types': ['statistical_outlier', 'semantic_inconsistency'],
            'statistical_threshold': 2.0,
            'semantic_threshold': 0.3,
            'telemetry': {'enabled': False}
        }
        detector = AnomalyDetector(config)
        
        # Test detection
        test_segment = MockSRTSegment(0, "00:00:01,000", "00:00:05,000", "Normal text segment", 0.8)
        anomalies = detector.detect_anomalies([test_segment])
        
        # Should return list (may be empty)
        assert isinstance(anomalies, list)
        
        print("  ✅ AnomalyDetector detection: PASS")
        test_results['anomaly_detector'] = True
        
    except Exception as e:
        print(f"  ❌ AnomalyDetector tests failed: {e}")
        failed_tests.append(f"AnomalyDetector: {e}")
        test_results['anomaly_detector'] = False

    # Test 3: OOVDetector  
    print("Testing OOVDetector...")
    try:
        config = {
            'oov_threshold': 0.1,
            'telemetry': {'enabled': False}
        }
        oov_detector = OOVDetector(config)
        
        # Test detection
        test_segments = [MockSRTSegment(0, "00:00:01,000", "00:00:05,000", "yoga dharma meditation", 0.8)]
        oov_results = oov_detector.detect_oov_terms(test_segments)
        
        # Should return results structure
        assert isinstance(oov_results, (list, dict))
        
        print("  ✅ OOVDetector detection: PASS")
        test_results['oov_detector'] = True
        
    except Exception as e:
        print(f"  ❌ OOVDetector tests failed: {e}")
        failed_tests.append(f"OOVDetector: {e}")
        test_results['oov_detector'] = False

    # Test 4: QAFlaggingEngine
    print("Testing QAFlaggingEngine...")
    try:
        config = {
            'confidence_threshold': 0.7,
            'anomaly_threshold': 0.3,
            'oov_threshold': 0.1,
            'telemetry': {'enabled': False}
        }
        flagging_engine = QAFlaggingEngine(config)
        
        # Test flagging
        test_segments = [MockSRTSegment(0, "00:00:01,000", "00:00:05,000", "test segment", 0.6)]
        flags = flagging_engine.flag_segments(test_segments)
        
        # Should return flagging results
        assert isinstance(flags, (list, dict))
        
        print("  ✅ QAFlaggingEngine flagging: PASS")
        test_results['qa_flagging_engine'] = True
        
    except Exception as e:
        print(f"  ❌ QAFlaggingEngine tests failed: {e}")
        failed_tests.append(f"QAFlaggingEngine: {e}")
        test_results['qa_flagging_engine'] = False

    # Test 5: QAReportGenerator
    print("Testing QAReportGenerator...")
    try:
        config = {'telemetry': {'enabled': False}}
        report_generator = QAReportGenerator(config)
        
        # Test report generation
        test_data = {
            'segments_processed': 10,
            'confidence_stats': {'mean': 0.8, 'std': 0.1},
            'anomalies_detected': 2,
            'oov_terms_found': 1
        }
        
        report = report_generator.generate_qa_report(test_data)
        
        # Should return report structure
        assert isinstance(report, dict)
        assert 'report_id' in report or 'summary' in report
        
        print("  ✅ QAReportGenerator generation: PASS")
        test_results['qa_report_generator'] = True
        
    except Exception as e:
        print(f"  ❌ QAReportGenerator tests failed: {e}")
        failed_tests.append(f"QAReportGenerator: {e}")
        test_results['qa_report_generator'] = False

    # Summary
    print()
    print("=== Test Results Summary ===")
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    failed_count = total_tests - passed_tests
    
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_count}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_tests:
        print()
        print("Failed tests:")
        for failure in failed_tests:
            print(f"  - {failure}")
    
    print()
    if failed_count == 0:
        print("🎉 ALL QA TESTS PASSED!")
        return True
    else:
        print(f"❌ {failed_count} tests failed - need investigation")
        return False

if __name__ == "__main__":
    success = run_qa_tests()
    sys.exit(0 if success else 1)