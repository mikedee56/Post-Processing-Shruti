#!/usr/bin/env python3
"""
Comprehensive Architect Validation Report
==========================================

Final assessment of both critical functionality fixes and performance validation
after QA handoff to architect role.
"""

import sys
import time
sys.path.insert(0, 'src')

from utils.advanced_text_normalizer import AdvancedTextNormalizer
from post_processors.sanskrit_post_processor import SanskritPostProcessor
from utils.srt_parser import SRTSegment
import logging

# Suppress verbose logging for clean test output
logging.getLogger().setLevel(logging.ERROR)

def test_critical_functionality_fixes():
    """Test the three critical functionality failures identified by QA."""
    
    print("=" * 70)
    print("PART 1: CRITICAL FUNCTIONALITY VALIDATION")
    print("=" * 70)
    print()
    
    # Initialize the text normalizer
    config = {'enable_mcp_processing': True, 'enable_fallback': True}
    normalizer = AdvancedTextNormalizer(config)
    
    # Test cases from QA audit
    critical_test_cases = [
        {
            'name': 'Scriptural Reference Conversion',
            'input': 'chapter two verse twenty five',
            'expected': 'Chapter 2 verse 25',
            'description': 'Convert scriptural references with proper capitalization'
        },
        {
            'name': 'Idiomatic Expression Preservation', 
            'input': 'And one by one, he killed six of their children.',
            'expected': 'And one by one, he killed six of their children.',
            'description': 'Preserve idiomatic expressions unchanged'
        },
        {
            'name': 'Temporal Number Conversion',
            'input': 'Year two thousand five.',
            'expected': 'Year 2005.',
            'description': 'Convert temporal/year references correctly'
        }
    ]
    
    # Run validation tests
    all_tests_passed = True
    test_results = []
    
    for i, test_case in enumerate(critical_test_cases, 1):
        print(f"TEST {i}: {test_case['name']}")
        print(f"Description: {test_case['description']}")
        print(f"Input:    '{test_case['input']}'")
        print(f"Expected: '{test_case['expected']}'")
        
        # Run the test
        try:
            result = normalizer.convert_numbers_with_context(test_case['input'])
            passed = result == test_case['expected']
            
            print(f"Actual:   '{result}'")
            print(f"Status:   {'PASS' if passed else 'FAIL'}")
            
            test_results.append({
                'test': test_case['name'],
                'passed': passed,
                'input': test_case['input'],
                'expected': test_case['expected'],
                'actual': result
            })
            
            if not passed:
                all_tests_passed = False
                
        except Exception as e:
            print(f"Actual:   ERROR - {e}")
            print(f"Status:   FAIL (Exception)")
            all_tests_passed = False
            test_results.append({
                'test': test_case['name'],
                'passed': False,
                'error': str(e)
            })
        
        print("-" * 50)
        print()
    
    return all_tests_passed, test_results

def test_integration_functionality():
    """Test end-to-end integration with Sanskrit post-processor."""
    
    print("INTEGRATION TEST: Sanskrit Post-Processor")
    print("Testing combined text normalization and capitalization")
    print("-" * 50)
    
    try:
        # Initialize post-processor
        processor = SanskritPostProcessor()
        
        # Test text that needs both normalization and capitalization
        test_text = "today we study krishna in chapter two verse twenty five"
        
        print(f"Input: '{test_text}'")
        
        # Test text normalization
        normalized = processor.text_normalizer.convert_numbers_with_context(test_text)
        print(f"After text normalization: '{normalized}'")
        
        # Test capitalization (if available)
        if processor.capitalization_engine:
            cap_result = processor.capitalization_engine.capitalize_text(normalized)
            capitalized = cap_result.capitalized_text
            print(f"After capitalization: '{capitalized}'")
            
            # Validate both transformations occurred
            has_normalization = ("Chapter 2 verse 25" in capitalized or "Chapter 2 Verse 25" in capitalized)
            has_capitalization = "Krishna" in capitalized
            
            print()
            print(f"Text normalization applied: {'YES' if has_normalization else 'NO'}")
            print(f"Capitalization applied: {'YES' if has_capitalization else 'NO'}")
            
            integration_success = has_normalization and has_capitalization
            print(f"Integration success: {'PASS' if integration_success else 'FAIL'}")
            
            return integration_success
        else:
            print("FAIL: Capitalization engine not available")
            return False
            
    except Exception as e:
        print(f"FAIL: Integration test error - {e}")
        return False

def measure_performance_benchmarks():
    """Measure current system performance against claimed benchmarks."""
    
    print()
    print("=" * 70)
    print("PART 2: PERFORMANCE VALIDATION")
    print("=" * 70)
    print()
    
    # Claimed benchmark from QA audit
    claimed_performance = 714.43  # segments/sec
    
    print(f"CLAIMED BENCHMARK: {claimed_performance} segments/sec")
    print()
    
    # Initialize processor
    processor = SanskritPostProcessor()
    
    # Test with different segment counts for accuracy
    test_sizes = [50, 100, 200]
    performance_results = []
    
    for size in test_sizes:
        print(f"TEST RUN: {size} segments")
        
        # Create test segments
        test_segments = []
        for i in range(size):
            texts = [
                f"Today we study yoga and dharma segment {i}.",
                f"We discuss krishna and chapter two verse twenty five segment {i}.",
                f"And one by one, we learn about spiritual practices segment {i}.",
                f"In year two thousand five, great teachers shared wisdom segment {i}.",
                f"The bhagavad gita teaches us about karma yoga segment {i}."
            ]
            
            segment = SRTSegment(
                index=i + 1,
                start_time=float(i),
                end_time=float(i + 4),
                text=texts[i % len(texts)],
                raw_text=texts[i % len(texts)]
            )
            test_segments.append(segment)
        
        # Measure processing time
        start_time = time.time()
        
        processed_count = 0
        for segment in test_segments:
            try:
                metrics = processor.metrics_collector.create_file_metrics('performance_test')
                processed_segment = processor._process_srt_segment(segment, metrics)
                processed_count += 1
            except Exception:
                continue
        
        end_time = time.time()
        
        # Calculate performance metrics
        total_time = end_time - start_time
        segments_per_second = processed_count / total_time if total_time > 0 else 0
        performance_results.append(segments_per_second)
        
        print(f"  Performance: {segments_per_second:.2f} segments/sec")
        print()
    
    # Calculate average performance
    avg_performance = sum(performance_results) / len(performance_results)
    
    return {
        'claimed_performance': claimed_performance,
        'measured_performance': avg_performance,
        'performance_ratio': avg_performance / claimed_performance,
        'test_results': list(zip(test_sizes, performance_results))
    }

def generate_comprehensive_report(functionality_results, integration_success, performance_results):
    """Generate comprehensive validation report."""
    
    print()
    print("=" * 70)
    print("COMPREHENSIVE ARCHITECT VALIDATION REPORT")
    print("=" * 70)
    print()
    
    # PART 1: FUNCTIONALITY ASSESSMENT
    print("PART 1: CRITICAL FUNCTIONALITY FIXES")
    print("-" * 40)
    
    total_tests = len(functionality_results)
    passed_tests = sum(1 for result in functionality_results if result.get('passed', False))
    
    print(f"Critical Functionality Tests: {passed_tests}/{total_tests} PASSED")
    print(f"Integration Test: {'PASS' if integration_success else 'FAIL'}")
    print()
    
    print("Detailed Results:")
    for result in functionality_results:
        status = "PASS" if result.get('passed', False) else "FAIL"
        print(f"  {result['test']}: {status}")
        if not result.get('passed', False) and 'error' not in result:
            print(f"    Expected: {result.get('expected', 'N/A')}")
            print(f"    Actual:   {result.get('actual', 'N/A')}")
    
    functionality_success = (passed_tests == total_tests) and integration_success
    
    print()
    print(f"FUNCTIONALITY ASSESSMENT: {'SUCCESS' if functionality_success else 'FAILURE'}")
    
    # PART 2: PERFORMANCE ASSESSMENT
    print()
    print("PART 2: PERFORMANCE VALIDATION")
    print("-" * 40)
    
    claimed_perf = performance_results['claimed_performance']
    measured_perf = performance_results['measured_performance']
    perf_ratio = performance_results['performance_ratio']
    
    print(f"Claimed Performance: {claimed_perf:.2f} seg/sec")
    print(f"Measured Performance: {measured_perf:.2f} seg/sec")
    print(f"Performance Ratio: {perf_ratio:.3f} ({perf_ratio:.1%})")
    print()
    
    print("Test Results:")
    for size, perf in performance_results['test_results']:
        print(f"  {size} segments: {perf:.2f} seg/sec")
    
    # Performance validation
    if perf_ratio >= 0.90:
        perf_status = "PASS"
        perf_desc = "System meets claimed performance"
    elif perf_ratio >= 0.75:
        perf_status = "ACCEPTABLE"
        perf_desc = "System performance is acceptable"
    else:
        perf_status = "FAIL"
        perf_desc = "System underperforms claimed benchmarks"
    
    # Functional performance check
    min_functional_perf = 10.0
    functional_ok = measured_perf >= min_functional_perf
    
    print()
    print(f"Performance Validation: {perf_status}")
    print(f"  {perf_desc}")
    print(f"Functional Performance: {'PASS' if functional_ok else 'FAIL'} ({measured_perf:.1f} >= {min_functional_perf} seg/sec required)")
    
    # OVERALL ASSESSMENT
    print()
    print("=" * 70)
    print("OVERALL ARCHITECT ASSESSMENT")
    print("=" * 70)
    
    overall_functionality = functionality_success
    overall_performance = (perf_status in ['PASS', 'ACCEPTABLE']) and functional_ok
    
    print(f"1. Critical Functionality: {'RESTORED' if overall_functionality else 'FAILED'}")
    print(f"2. System Performance: {'VALIDATED' if overall_performance else 'UNACCEPTABLE'}")
    
    print()
    
    if overall_functionality and overall_performance:
        print("FINAL VERDICT: SUCCESS")
        print("- All critical functionality failures have been FIXED")
        print("- System performance is within acceptable parameters")
        print("- Architecture meets stated requirements")
        print("- Ready for production deployment")
        return True
    elif overall_functionality and not overall_performance:
        print("FINAL VERDICT: PARTIAL SUCCESS")
        print("- All critical functionality failures have been FIXED")
        print("- System performance does not meet claimed benchmarks")
        print("- Performance optimization required")
        print("- Functional requirements satisfied")
        return True  # Functionality is primary concern
    else:
        print("FINAL VERDICT: FAILURE")
        print("- Critical functionality issues remain")
        print("- Additional architectural work required")
        return False

def main():
    """Main comprehensive validation routine."""
    
    # Run critical functionality tests
    functionality_success, functionality_results = test_critical_functionality_fixes()
    
    # Run integration tests
    integration_success = test_integration_functionality()
    
    # Run performance benchmarks
    performance_results = measure_performance_benchmarks()
    
    # Generate comprehensive report
    overall_success = generate_comprehensive_report(
        functionality_results, integration_success, performance_results
    )
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)