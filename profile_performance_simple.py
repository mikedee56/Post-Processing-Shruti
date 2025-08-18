#!/usr/bin/env python3
"""
Simple Performance Profiling Script for Sanskrit Post-Processing System

This script profiles the performance bottleneck causing 4.04 segments/sec instead of 10+ segments/sec.
"""

import sys
import time
import cProfile
import pstats
import io
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def profile_performance():
    """Profile the Sanskrit post-processor performance."""
    from post_processors.sanskrit_post_processor import SanskritPostProcessor
    from utils.srt_parser import SRTSegment
    
    print("=== Sanskrit Post-Processor Performance Profiling ===")
    print()
    
    # Initialize processor
    processor = SanskritPostProcessor()
    
    # Create test segments (representative workload)
    test_segments = [
        SRTSegment(1, "00:00:01,000", "00:00:05,000", "Today we study yoga and dharma from ancient scriptures."),
        SRTSegment(2, "00:00:06,000", "00:00:10,000", "Um, the bhagavad gita teaches us about, uh, spiritual wisdom."),
        SRTSegment(3, "00:00:11,000", "00:00:15,000", "Krishna dharma yoga meditation practice brings inner peace."),
        SRTSegment(4, "00:00:16,000", "00:00:20,000", "In chapter two verse twenty five we learn about the eternal soul."),
        SRTSegment(5, "00:00:21,000", "00:00:25,000", "The teacher explains that in the year two thousand five many began their journey."),
        SRTSegment(6, "00:00:26,000", "00:00:30,000", "Actually, let me correct that - rather, it speaks about dharma."),
        SRTSegment(7, "00:00:31,000", "00:00:35,000", "The Sanskrit terms need proper transliteration with diacritics."),
        SRTSegment(8, "00:00:36,000", "00:00:40,000", "I mean, we should understand the deeper meaning of these teachings."),
        SRTSegment(9, "00:00:41,000", "00:00:45,000", "You know, this practice of meditation has been used for centuries."),
        SRTSegment(10, "00:00:46,000", "00:00:50,000", "Well, the ancient wisdom continues to guide us today."),
    ]
    
    print(f"Testing with {len(test_segments)} segments...")
    
    # 1. Profile overall performance
    print("1. Overall Performance Test...")
    start_time = time.time()
    
    for segment in test_segments:
        processor._process_srt_segment(segment, processor.metrics_collector.create_file_metrics("test"))
    
    end_time = time.time()
    total_time = end_time - start_time
    segments_per_second = len(test_segments) / total_time
    
    print(f"   Processed {len(test_segments)} segments in {total_time:.2f} seconds")
    print(f"   Current performance: {segments_per_second:.2f} segments/sec")
    print(f"   Target performance: 10.0 segments/sec")
    print(f"   Performance gap: {10.0 - segments_per_second:.2f} segments/sec")
    print(f"   Need {10.0 / segments_per_second:.1f}x improvement")
    print()
    
    # 2. Profile individual components
    print("2. Component Performance Breakdown...")
    test_segment = test_segments[0]
    
    # Component timing tests
    components = {
        'text_normalizer': None,
        'number_processor': None,
        'sanskrit_corrections': None,
        'lexicon_corrections': None,
        'ner_processing': None,
        'confidence_calc': None
    }
    
    # Text normalizer
    times = []
    for _ in range(20):
        start = time.perf_counter()
        try:
            if hasattr(processor.text_normalizer, 'normalize_with_advanced_tracking'):
                processor.text_normalizer.normalize_with_advanced_tracking(test_segment.text)
            else:
                processor.text_normalizer.normalize_with_tracking(test_segment.text)
        except Exception as e:
            print(f"   Text normalizer error: {e}")
            break
        end = time.perf_counter()
        times.append((end - start) * 1000)
    if times:
        components['text_normalizer'] = sum(times) / len(times)
    
    # Number processor
    times = []
    for _ in range(20):
        start = time.perf_counter()
        try:
            processor.number_processor.process_numbers(test_segment.text, context="spiritual")
        except Exception as e:
            print(f"   Number processor error: {e}")
            break
        end = time.perf_counter()
        times.append((end - start) * 1000)
    if times:
        components['number_processor'] = sum(times) / len(times)
    
    # Sanskrit corrections
    times = []
    for _ in range(20):
        start = time.perf_counter()
        try:
            processor._apply_enhanced_sanskrit_hindi_corrections(test_segment.text)
        except Exception as e:
            print(f"   Sanskrit corrections error: {e}")
            break
        end = time.perf_counter()
        times.append((end - start) * 1000)
    if times:
        components['sanskrit_corrections'] = sum(times) / len(times)
    
    # Lexicon corrections
    times = []
    for _ in range(20):
        start = time.perf_counter()
        try:
            processor._apply_lexicon_corrections(test_segment.text)
        except Exception as e:
            print(f"   Lexicon corrections error: {e}")
            break
        end = time.perf_counter()
        times.append((end - start) * 1000)
    if times:
        components['lexicon_corrections'] = sum(times) / len(times)
    
    # NER processing (if enabled)
    if processor.enable_ner and processor.ner_model:
        times = []
        for _ in range(20):
            start = time.perf_counter()
            try:
                processor.ner_model.identify_entities(test_segment.text)
                if processor.capitalization_engine:
                    processor.capitalization_engine.capitalize_text(test_segment.text)
            except Exception as e:
                print(f"   NER processing error: {e}")
                break
            end = time.perf_counter()
            times.append((end - start) * 1000)
        if times:
            components['ner_processing'] = sum(times) / len(times)
    
    # Confidence calculation
    times = []
    for _ in range(20):
        start = time.perf_counter()
        try:
            processor._calculate_enhanced_confidence(test_segment.text, [], test_segment.text)
        except Exception as e:
            print(f"   Confidence calculation error: {e}")
            break
        end = time.perf_counter()
        times.append((end - start) * 1000)
    if times:
        components['confidence_calc'] = sum(times) / len(times)
    
    # Display results
    valid_components = {k: v for k, v in components.items() if v is not None}
    if valid_components:
        total_component_time = sum(valid_components.values())
        sorted_components = sorted(valid_components.items(), key=lambda x: x[1], reverse=True)
        
        print("   Component timing (average over 20 runs):")
        for component, time_ms in sorted_components:
            percentage = (time_ms / total_component_time) * 100 if total_component_time > 0 else 0
            print(f"   {component:<25}: {time_ms:6.2f}ms ({percentage:5.1f}%)")
        
        # Identify bottleneck
        if sorted_components:
            bottleneck = sorted_components[0]
            print(f"\n   🚨 BOTTLENECK: {bottleneck[0]} ({bottleneck[1]:.2f}ms, {(bottleneck[1]/total_component_time)*100:.1f}%)")
    
    print()
    
    # 3. cProfile analysis
    print("3. Detailed cProfile Analysis...")
    profiler = cProfile.Profile()
    
    profiler.enable()
    for segment in test_segments[:5]:  # Profile subset for detail
        processor._process_srt_segment(segment, processor.metrics_collector.create_file_metrics("test"))
    profiler.disable()
    
    # Get top functions by total time
    stats = pstats.Stats(profiler)
    stats.sort_stats('tottime')
    
    print("   Top time-consuming functions:")
    stats_data = stats.get_stats()
    function_times = []
    
    for func_key, (tottime, cumtime, ncalls, _, _) in stats_data.items():
        filename, line_num, func_name = func_key
        if tottime > 0.001:  # Only functions taking more than 1ms
            function_times.append((tottime, func_name, filename, ncalls, tottime/ncalls if ncalls > 0 else 0))
    
    function_times.sort(reverse=True)
    for i, (tottime, func_name, filename, ncalls, per_call) in enumerate(function_times[:10]):
        if 'src/' in filename or any(keyword in func_name for keyword in ['normalize', 'process', 'correct', 'apply']):
            print(f"   {i+1}. {func_name}")
            print(f"      File: {filename}")
            print(f"      Total: {tottime:.4f}s, Calls: {ncalls}, Per-call: {per_call*1000:.2f}ms")
    
    print()
    
    # 4. Generate optimization recommendations
    print("4. OPTIMIZATION RECOMMENDATIONS:")
    print("="*50)
    
    if segments_per_second < 5.0:
        print("🚨 CRITICAL: Performance is severely below target")
    elif segments_per_second < 8.0:
        print("⚠️  WARNING: Performance is below target")
    else:
        print("✅ GOOD: Performance is close to target")
    
    print(f"Current: {segments_per_second:.2f} seg/sec | Target: 10.0 seg/sec | Gap: {10.0-segments_per_second:.2f}")
    print()
    
    if valid_components:
        bottleneck_component = sorted_components[0][0]
        
        if 'text_normalizer' in bottleneck_component:
            print("PRIMARY BOTTLENECK: Text Normalizer")
            print("- Implement regex pattern caching")
            print("- Optimize MCP client calls")
            print("- Cache repeated normalization results")
            
        elif 'sanskrit_corrections' in bottleneck_component:
            print("PRIMARY BOTTLENECK: Sanskrit Corrections")
            print("- Cache lexicon lookups")
            print("- Optimize fuzzy matching algorithm")
            print("- Implement early termination for matches")
            
        elif 'number_processor' in bottleneck_component:
            print("PRIMARY BOTTLENECK: Number Processor") 
            print("- Pre-compile regex patterns")
            print("- Cache contextual analysis")
            print("- Optimize MCP integration")
            
        elif 'ner_processing' in bottleneck_component:
            print("PRIMARY BOTTLENECK: NER Processing")
            print("- Implement model caching")
            print("- Batch process segments")
            print("- Optimize entity recognition")
        
        print()
        print("GENERAL OPTIMIZATIONS:")
        print("- Enable parallel processing for independent segments")
        print("- Implement configuration caching")
        print("- Optimize memory usage patterns")
        print("- Add performance monitoring")
    
    return {
        'segments_per_second': segments_per_second,
        'target': 10.0,
        'gap': 10.0 - segments_per_second,
        'components': valid_components,
        'bottleneck': sorted_components[0] if sorted_components else None
    }

if __name__ == "__main__":
    try:
        results = profile_performance()
        print(f"\nProfile complete. Results: {results['segments_per_second']:.2f} seg/sec")
    except Exception as e:
        print(f"Profiling failed: {e}")
        import traceback
        traceback.print_exc()