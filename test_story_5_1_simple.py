#!/usr/bin/env python3
"""
Story 5.1 Performance Optimization Test Script (Simplified)

This script applies comprehensive performance optimizations to SanskritPostProcessor
and validates that the 305% variance issue is eliminated.
"""

import sys
import time
import statistics
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def main():
    """Main test execution function."""
    print("Story 5.1 Performance Optimization Test")
    print("=" * 50)
    
    try:
        from post_processors.sanskrit_post_processor import SanskritPostProcessor
        from utils.performance_optimizer import optimize_processor_for_story_5_1, get_global_optimizer
        from utils.srt_parser import SRTSegment
        print("SUCCESS: All required modules imported")
    except ImportError as e:
        print(f"ERROR: Import failed - {e}")
        return False
    
    # Initialize processor
    print("Initializing SanskritPostProcessor...")
    processor = SanskritPostProcessor()
    
    # Create test segments
    print("Creating test segments...")
    test_segments = []
    test_texts = [
        "Today we study yoga and dharma from ancient scriptures.",
        "Krishna teaches us about chapter two verse twenty five.",
        "We learn from the Bhagavad Gita and practice meditation.",
        "The ancient wisdom guides us in spiritual development.",
        "Meditation and mindfulness are important yogic practices.",
    ]
    
    for i in range(20):
        text = test_texts[i % len(test_texts)]
        segment = SRTSegment(
            index=i + 1,
            start_time=f"00:00:{i:02d},000",
            end_time=f"00:00:{i+4:02d},000",
            text=text + f" Segment {i+1}.",
            raw_text=text + f" Segment {i+1}."
        )
        test_segments.append(segment)
    
    print(f"Created {len(test_segments)} test segments")
    
    # Measure baseline performance
    print("Measuring baseline performance...")
    processing_times = []
    
    for segment in test_segments:
        start_time = time.perf_counter()
        try:
            file_metrics = processor.metrics_collector.create_file_metrics("baseline_test")
            processor._process_srt_segment(segment, file_metrics)
            processing_time = time.perf_counter() - start_time
            processing_times.append(processing_time)
        except Exception as e:
            print(f"Warning: Failed to process baseline segment: {e}")
    
    if not processing_times:
        print("ERROR: No successful baseline processing")
        return False
    
    # Calculate baseline statistics
    avg_time = statistics.mean(processing_times)
    total_time = sum(processing_times)
    segments_per_second = len(test_segments) / total_time
    
    if len(processing_times) > 1:
        stdev_time = statistics.stdev(processing_times)
        variance_percentage = (stdev_time / avg_time * 100) if avg_time > 0 else 0
    else:
        variance_percentage = 0
    
    print(f"Baseline Performance:")
    print(f"   Average time per segment: {avg_time:.4f}s")
    print(f"   Segments per second: {segments_per_second:.2f}")
    print(f"   Variance: {variance_percentage:.1f}%")
    
    # Apply optimizations
    print("Applying Story 5.1 performance optimizations...")
    optimization_results = optimize_processor_for_story_5_1(processor)
    
    print(f"Optimizations applied:")
    for optimization in optimization_results['optimizations_applied']:
        print(f"   - {optimization}")
    
    print(f"Pre-loaded models: {optimization_results['preloaded_models']}")
    print(f"Optimization time: {optimization_results['optimization_time']:.4f}s")
    
    # Measure optimized performance
    print("Measuring optimized performance...")
    optimized_times = []
    
    for segment in test_segments:
        start_time = time.perf_counter()
        try:
            file_metrics = processor.metrics_collector.create_file_metrics("optimized_test")
            processor._process_srt_segment(segment, file_metrics)
            processing_time = time.perf_counter() - start_time
            optimized_times.append(processing_time)
        except Exception as e:
            print(f"Warning: Failed to process optimized segment: {e}")
    
    if not optimized_times:
        print("ERROR: No successful optimized processing")
        return False
    
    # Calculate optimized statistics
    opt_avg_time = statistics.mean(optimized_times)
    opt_total_time = sum(optimized_times)
    opt_segments_per_second = len(test_segments) / opt_total_time
    
    if len(optimized_times) > 1:
        opt_stdev_time = statistics.stdev(optimized_times)
        opt_variance_percentage = (opt_stdev_time / opt_avg_time * 100) if opt_avg_time > 0 else 0
    else:
        opt_variance_percentage = 0
    
    print(f"Optimized Performance:")
    print(f"   Average time per segment: {opt_avg_time:.4f}s")
    print(f"   Segments per second: {opt_segments_per_second:.2f}")
    print(f"   Variance: {opt_variance_percentage:.1f}%")
    
    # Analyze improvements
    speed_improvement = opt_segments_per_second / segments_per_second
    variance_improvement = variance_percentage - opt_variance_percentage
    time_improvement = avg_time / opt_avg_time
    
    # Check target achievement
    variance_target_met = opt_variance_percentage <= 10.0
    throughput_target_met = opt_segments_per_second >= 10.0
    
    print("")
    print("=" * 50)
    print("STORY 5.1 PERFORMANCE TEST RESULTS")
    print("=" * 50)
    
    print(f"Performance Analysis:")
    print(f"   Speed improvement: {speed_improvement:.2f}x")
    print(f"   Variance reduction: {variance_improvement:.1f} percentage points")
    print(f"   Time improvement: {time_improvement:.2f}x")
    print(f"   Variance target (<10%): {'MET' if variance_target_met else 'NOT MET'}")
    print(f"   Throughput target (10+ seg/sec): {'MET' if throughput_target_met else 'NOT MET'}")
    
    # Get cache statistics
    optimizer = get_global_optimizer()
    cache_stats = optimizer._get_cache_statistics()
    
    if 'overall' in cache_stats:
        overall_stats = cache_stats['overall']
        print(f"Cache Performance:")
        print(f"   Hit rate: {overall_stats['overall_hit_rate_percent']:.1f}%")
        print(f"   Total hits: {overall_stats['total_hits']}")
        print(f"   Active caches: {overall_stats['active_caches']}")
    
    overall_success = variance_target_met and throughput_target_met
    
    if overall_success:
        print("SUCCESS: All Story 5.1 performance targets achieved!")
        print(f"   Variance reduced to {opt_variance_percentage:.1f}% (target: <10%)")
        print(f"   Throughput: {opt_segments_per_second:.2f} segments/sec (target: 10+)")
    else:
        print("INCOMPLETE: Some Story 5.1 targets not fully achieved")
        if not variance_target_met:
            print(f"   Variance: {opt_variance_percentage:.1f}% (target: <10%)")
        if not throughput_target_met:
            print(f"   Throughput: {opt_segments_per_second:.2f} seg/sec (target: 10+)")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)