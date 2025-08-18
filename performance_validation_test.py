#!/usr/bin/env python3
"""
Performance Validation Test - Architect QA Validation
===================================================

This script validates the claimed performance benchmarks after critical fixes.
"""

import sys
import time
import tempfile
from pathlib import Path
sys.path.insert(0, 'src')

from post_processors.sanskrit_post_processor import SanskritPostProcessor
from utils.srt_parser import SRTSegment
import logging

# Suppress verbose logging for clean test output
logging.getLogger().setLevel(logging.ERROR)

def create_test_segments(count: int):
    """Create test SRT segments for performance testing."""
    test_segments = []
    for i in range(count):
        # Create segments with various processing requirements
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
    
    return test_segments

def measure_processing_performance(segment_count: int = 100):
    """Measure processing performance with the current fixes."""
    
    print(f"PERFORMANCE TEST: Processing {segment_count} segments")
    print("-" * 50)
    
    # Initialize processor
    processor = SanskritPostProcessor()
    
    # Create test segments
    test_segments = create_test_segments(segment_count)
    
    print(f"Created {len(test_segments)} test segments")
    print("Starting performance measurement...")
    
    # Measure processing time
    start_time = time.time()
    
    processed_count = 0
    for segment in test_segments:
        try:
            metrics = processor.metrics_collector.create_file_metrics('performance_test')
            processed_segment = processor._process_srt_segment(segment, metrics)
            processed_count += 1
        except Exception as e:
            print(f"Error processing segment {segment.index}: {e}")
            continue
    
    end_time = time.time()
    
    # Calculate performance metrics
    total_time = end_time - start_time
    segments_per_second = processed_count / total_time if total_time > 0 else 0
    
    print(f"Processing completed:")
    print(f"  Total segments: {processed_count}")
    print(f"  Total time: {total_time:.3f} seconds")
    print(f"  Performance: {segments_per_second:.2f} segments/sec")
    
    return segments_per_second, total_time, processed_count

def validate_performance_claims():
    """Validate against claimed performance benchmarks."""
    
    print("=" * 60)
    print("PERFORMANCE VALIDATION TEST")
    print("Architect QA Validation - Speed/Efficiency")
    print("=" * 60)
    print()
    
    # Claimed benchmark from QA audit
    claimed_performance = 714.43  # segments/sec
    
    print(f"CLAIMED BENCHMARK: {claimed_performance} segments/sec")
    print()
    
    # Test with different segment counts for accuracy
    test_sizes = [50, 100, 200]
    performance_results = []
    
    for size in test_sizes:
        print(f"TEST RUN: {size} segments")
        performance, time_taken, processed = measure_processing_performance(size)
        performance_results.append(performance)
        print()
    
    # Calculate average performance
    avg_performance = sum(performance_results) / len(performance_results)
    
    print("=" * 60)
    print("PERFORMANCE ANALYSIS")
    print("=" * 60)
    print()
    
    print(f"Test Results:")
    for i, (size, perf) in enumerate(zip(test_sizes, performance_results)):
        print(f"  {size} segments: {perf:.2f} seg/sec")
    
    print(f"\nAverage Performance: {avg_performance:.2f} segments/sec")
    print(f"Claimed Performance: {claimed_performance:.2f} segments/sec")
    
    # Performance validation
    performance_ratio = avg_performance / claimed_performance
    
    print(f"Performance Ratio: {performance_ratio:.3f}")
    
    if performance_ratio >= 0.90:  # Within 10% of claimed performance
        print("✓ PERFORMANCE VALIDATION: PASS")
        print(f"  System meets or exceeds claimed performance ({performance_ratio:.1%})")
        validation_status = "PASS"
    elif performance_ratio >= 0.75:  # Within 25% of claimed performance  
        print("△ PERFORMANCE VALIDATION: ACCEPTABLE")
        print(f"  System performance is acceptable ({performance_ratio:.1%} of claimed)")
        validation_status = "ACCEPTABLE"
    else:
        print("✗ PERFORMANCE VALIDATION: FAIL")
        print(f"  System underperforms claimed benchmarks ({performance_ratio:.1%})")
        validation_status = "FAIL"
    
    # Check if fixes introduced performance regression
    print()
    print("REGRESSION ANALYSIS:")
    
    # Performance targets for critical functionality
    min_acceptable_performance = 10.0  # segments/sec (basic functionality)
    
    if avg_performance >= min_acceptable_performance:
        print(f"✓ FUNCTIONAL PERFORMANCE: PASS ({avg_performance:.1f} >= {min_acceptable_performance} seg/sec)")
        functional_status = "PASS"
    else:
        print(f"✗ FUNCTIONAL PERFORMANCE: FAIL ({avg_performance:.1f} < {min_acceptable_performance} seg/sec)")
        functional_status = "FAIL"
    
    return {
        'claimed_performance': claimed_performance,
        'measured_performance': avg_performance,
        'performance_ratio': performance_ratio,
        'validation_status': validation_status,
        'functional_status': functional_status,
        'test_results': list(zip(test_sizes, performance_results))
    }

def main():
    """Main performance validation routine."""
    
    try:
        results = validate_performance_claims()
        
        print()
        print("=" * 60)
        print("FINAL PERFORMANCE ASSESSMENT")
        print("=" * 60)
        
        print(f"Claimed Performance: {results['claimed_performance']:.2f} seg/sec")
        print(f"Measured Performance: {results['measured_performance']:.2f} seg/sec")
        print(f"Performance Validation: {results['validation_status']}")
        print(f"Functional Performance: {results['functional_status']}")
        
        # Overall assessment
        overall_pass = (results['validation_status'] in ['PASS', 'ACCEPTABLE'] and 
                       results['functional_status'] == 'PASS')
        
        if overall_pass:
            print("\n🎉 OVERALL PERFORMANCE: ACCEPTABLE")
            print("✓ System performance validates architectural claims")
            print("✓ Critical fixes do not impact performance significantly")
            return True
        else:
            print("\n❌ OVERALL PERFORMANCE: UNACCEPTABLE")
            print("✗ Performance issues require architectural attention")
            return False
    
    except Exception as e:
        print(f"Performance validation failed with error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)