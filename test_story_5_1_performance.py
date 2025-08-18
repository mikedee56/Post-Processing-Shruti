#!/usr/bin/env python3
"""
Story 5.1 Performance Optimization Test Script

This script applies comprehensive performance optimizations to SanskritPostProcessor
and validates that the 305% variance issue is eliminated.

Key Performance Targets:
- Reduce variance from 305.4% to <10%
- Achieve consistent 10+ segments/second processing
- Eliminate Word2Vec loading bottleneck
"""

import sys
import time
import statistics
import tempfile
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from post_processors.sanskrit_post_processor import SanskritPostProcessor
    from utils.performance_optimizer import optimize_processor_for_story_5_1, get_global_optimizer
    from utils.performance_profiler import PerformanceProfiler
    from utils.srt_parser import SRTSegment
    print("SUCCESS: All required modules imported successfully")
except ImportError as e:
    print(f"ERROR: Import error: {e}")
    sys.exit(1)

def create_test_segments(count: int = 20) -> list:
    """Create test SRT segments for performance testing."""
    segments = []
    test_texts = [
        "Today we study yoga and dharma from ancient scriptures.",
        "Krishna teaches us about chapter two verse twenty five.",
        "We learn from the Bhagavad Gita and practice meditation.",
        "The ancient wisdom guides us in spiritual development.",
        "Meditation and mindfulness are important yogic practices.",
    ]
    
    for i in range(count):
        text = test_texts[i % len(test_texts)]
        segment = SRTSegment(
            index=i + 1,
            start_time=f"00:00:{i:02d},000",
            end_time=f"00:00:{i+4:02d},000",
            text=text + f" Segment {i+1}.",
            raw_text=text + f" Segment {i+1}."
        )
        segments.append(segment)
    
    return segments

def measure_baseline_performance(processor, test_segments) -> dict:
    """Measure baseline performance before optimization."""
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
            print(f"WARNING: Failed to process baseline segment: {e}")
    
    if not processing_times:
        return {"error": "No successful baseline processing"}
    
    # Calculate baseline statistics
    avg_time = statistics.mean(processing_times)
    total_time = sum(processing_times)
    segments_per_second = len(test_segments) / total_time
    
    if len(processing_times) > 1:
        stdev_time = statistics.stdev(processing_times)
        variance_percentage = (stdev_time / avg_time * 100) if avg_time > 0 else 0
    else:
        variance_percentage = 0
    
    baseline_results = {
        'average_processing_time': avg_time,
        'total_processing_time': total_time,
        'segments_per_second': segments_per_second,
        'variance_percentage': variance_percentage,
        'processing_times': processing_times,
        'segments_processed': len(test_segments)
    }
    
    print(f"Baseline Performance:")
    print(f"   Average time per segment: {avg_time:.4f}s")
    print(f"   Segments per second: {segments_per_second:.2f}")
    print(f"   Variance: {variance_percentage:.1f}%")
    
    return baseline_results

def apply_story_5_1_optimizations(processor) -> dict:
    """Apply Story 5.1 performance optimizations."""
    print("🚀 Applying Story 5.1 performance optimizations...")
    
    optimization_results = optimize_processor_for_story_5_1(processor)
    
    print(f"⚡ Optimizations applied:")
    for optimization in optimization_results['optimizations_applied']:
        print(f"   ✓ {optimization}")
    
    print(f"📦 Pre-loaded models: {optimization_results['preloaded_models']}")
    print(f"⏱️  Optimization time: {optimization_results['optimization_time']:.4f}s")
    
    return optimization_results

def measure_optimized_performance(processor, test_segments) -> dict:
    """Measure performance after optimization."""
    print("📊 Measuring optimized performance...")
    
    processing_times = []
    
    for segment in test_segments:
        start_time = time.perf_counter()
        try:
            file_metrics = processor.metrics_collector.create_file_metrics("optimized_test")
            processor._process_srt_segment(segment, file_metrics)
            processing_time = time.perf_counter() - start_time
            processing_times.append(processing_time)
        except Exception as e:
            print(f"⚠️  Failed to process optimized segment: {e}")
    
    if not processing_times:
        return {"error": "No successful optimized processing"}
    
    # Calculate optimized statistics
    avg_time = statistics.mean(processing_times)
    total_time = sum(processing_times)
    segments_per_second = len(test_segments) / total_time
    
    if len(processing_times) > 1:
        stdev_time = statistics.stdev(processing_times)
        variance_percentage = (stdev_time / avg_time * 100) if avg_time > 0 else 0
    else:
        variance_percentage = 0
    
    optimized_results = {
        'average_processing_time': avg_time,
        'total_processing_time': total_time,
        'segments_per_second': segments_per_second,
        'variance_percentage': variance_percentage,
        'processing_times': processing_times,
        'segments_processed': len(test_segments)
    }
    
    print(f"📈 Optimized Performance:")
    print(f"   Average time per segment: {avg_time:.4f}s")
    print(f"   Segments per second: {segments_per_second:.2f}")
    print(f"   Variance: {variance_percentage:.1f}%")
    
    return optimized_results

def analyze_performance_improvement(baseline, optimized) -> dict:
    """Analyze the performance improvement achieved."""
    print("🔍 Analyzing performance improvements...")
    
    # Calculate improvements
    speed_improvement = optimized['segments_per_second'] / baseline['segments_per_second']
    variance_improvement = baseline['variance_percentage'] - optimized['variance_percentage']
    time_improvement = baseline['average_processing_time'] / optimized['average_processing_time']
    
    # Check target achievement
    variance_target_met = optimized['variance_percentage'] <= 10.0
    throughput_target_met = optimized['segments_per_second'] >= 10.0
    
    analysis = {
        'speed_improvement_factor': speed_improvement,
        'variance_reduction_percentage': variance_improvement,
        'time_improvement_factor': time_improvement,
        'variance_target_met': variance_target_met,
        'throughput_target_met': throughput_target_met,
        'baseline_variance': baseline['variance_percentage'],
        'optimized_variance': optimized['variance_percentage'],
        'baseline_throughput': baseline['segments_per_second'],
        'optimized_throughput': optimized['segments_per_second']
    }
    
    print(f"📊 Performance Analysis:")
    print(f"   Speed improvement: {speed_improvement:.2f}x")
    print(f"   Variance reduction: {variance_improvement:.1f} percentage points")
    print(f"   Time improvement: {time_improvement:.2f}x")
    print(f"   Variance target (<10%): {'✅ MET' if variance_target_met else '❌ NOT MET'}")
    print(f"   Throughput target (10+ seg/sec): {'✅ MET' if throughput_target_met else '❌ NOT MET'}")
    
    return analysis

def generate_performance_report(baseline, optimized, analysis, cache_stats) -> dict:
    """Generate comprehensive performance report."""
    report = {
        'story_version': '5.1',
        'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'performance_target': 'Eliminate 305% variance, achieve consistent 10+ segments/sec',
        'baseline_performance': baseline,
        'optimized_performance': optimized,
        'improvement_analysis': analysis,
        'cache_statistics': cache_stats,
        'targets_achievement': {
            'variance_target': {
                'target': '<10%',
                'achieved': optimized['variance_percentage'],
                'met': analysis['variance_target_met']
            },
            'throughput_target': {
                'target': '10+ segments/sec',
                'achieved': optimized['segments_per_second'],
                'met': analysis['throughput_target_met']
            }
        },
        'overall_success': analysis['variance_target_met'] and analysis['throughput_target_met']
    }
    
    return report

def main():
    """Main test execution function."""
    print("🎯 Story 5.1 Performance Optimization Test")
    print("=" * 50)
    
    # Initialize processor
    print("🔧 Initializing SanskritPostProcessor...")
    processor = SanskritPostProcessor()
    
    # Create test segments
    test_segments = create_test_segments(20)
    print(f"📝 Created {len(test_segments)} test segments")
    
    # Measure baseline performance
    baseline_results = measure_baseline_performance(processor, test_segments)
    if 'error' in baseline_results:
        print(f"❌ Baseline measurement failed: {baseline_results['error']}")
        return False
    
    # Apply optimizations
    optimization_results = apply_story_5_1_optimizations(processor)
    
    # Measure optimized performance
    optimized_results = measure_optimized_performance(processor, test_segments)
    if 'error' in optimized_results:
        print(f"❌ Optimized measurement failed: {optimized_results['error']}")
        return False
    
    # Analyze improvements
    analysis = analyze_performance_improvement(baseline_results, optimized_results)
    
    # Get cache statistics
    optimizer = get_global_optimizer()
    cache_stats = optimizer._get_cache_statistics()
    
    # Generate comprehensive report
    report = generate_performance_report(baseline_results, optimized_results, analysis, cache_stats)
    
    print("\n" + "=" * 50)
    print("🎯 STORY 5.1 PERFORMANCE TEST RESULTS")
    print("=" * 50)
    
    if report['overall_success']:
        print("✅ SUCCESS: All Story 5.1 performance targets achieved!")
        print(f"   ✓ Variance reduced to {optimized_results['variance_percentage']:.1f}% (target: <10%)")
        print(f"   ✓ Throughput: {optimized_results['segments_per_second']:.2f} segments/sec (target: 10+)")
        print(f"   ✓ Speed improvement: {analysis['speed_improvement_factor']:.2f}x")
        print(f"   ✓ Variance reduction: {analysis['variance_reduction_percentage']:.1f} percentage points")
    else:
        print("❌ INCOMPLETE: Some Story 5.1 targets not fully achieved")
        if not analysis['variance_target_met']:
            print(f"   ❌ Variance: {optimized_results['variance_percentage']:.1f}% (target: <10%)")
        if not analysis['throughput_target_met']:
            print(f"   ❌ Throughput: {optimized_results['segments_per_second']:.2f} seg/sec (target: 10+)")
    
    # Cache performance summary
    if 'overall' in cache_stats:
        overall_stats = cache_stats['overall']
        print(f"\n📦 Cache Performance:")
        print(f"   Hit rate: {overall_stats['overall_hit_rate_percent']:.1f}%")
        print(f"   Total hits: {overall_stats['total_hits']}")
        print(f"   Active caches: {overall_stats['active_caches']}")
    
    print(f"\n📋 Full report available in memory")
    return report['overall_success']

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)