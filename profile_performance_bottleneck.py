#!/usr/bin/env python3
"""
Performance Profiling Script for Sanskrit Post-Processing System

This script profiles the performance bottleneck causing 4.04 segments/sec instead of 10+ segments/sec.
It identifies which components are consuming the most time in the processing pipeline.
"""

import sys
import time
import cProfile
import pstats
import io
from pathlib import Path
from typing import Dict, List, Tuple, Any
import psutil
import threading
from dataclasses import dataclass

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

@dataclass
class PerformanceMeasurement:
    """Performance measurement data."""
    component: str
    time_ms: float
    memory_mb: float
    calls_count: int
    percentage_of_total: float

class PerformanceProfiler:
    """Performance profiler for the Sanskrit processing system."""
    
    def __init__(self):
        self.measurements: List[PerformanceMeasurement] = []
        self.total_time = 0
        self.process = psutil.Process()
        
    def profile_sanskrit_processor(self) -> Dict[str, Any]:
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
        ]
        
        # Profile processing components
        results = {}
        
        # 1. Profile overall segment processing
        print("1. Profiling overall segment processing...")
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024
        
        for i, segment in enumerate(test_segments):
            processor._process_srt_segment(segment, processor.metrics_collector.create_file_metrics("test"))
        
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024
        
        total_time = end_time - start_time
        segments_per_second = len(test_segments) / total_time
        
        results['overall'] = {
            'total_time_sec': total_time,
            'segments_per_second': segments_per_second,
            'memory_used_mb': end_memory - start_memory,
            'target_segments_per_second': 10.0,
            'performance_ratio': segments_per_second / 10.0
        }
        
        print(f"   Current performance: {segments_per_second:.2f} segments/sec")
        print(f"   Target performance: 10.0 segments/sec")
        print(f"   Performance gap: {10.0 - segments_per_second:.2f} segments/sec")
        print()
        
        # 2. Profile individual components
        self._profile_component_performance(processor, test_segments[0], results)
        
        # 3. Profile cProfile for detailed analysis
        self._profile_with_cprofile(processor, test_segments, results)
        
        return results
    
    def _profile_component_performance(self, processor, test_segment, results):
        """Profile individual component performance."""
        print("2. Profiling individual components...")
        
        components = {
            'text_normalizer': lambda: processor.text_normalizer.normalize_with_advanced_tracking(test_segment.text),
            'number_processor': lambda: processor.number_processor.process_numbers(test_segment.text, context="spiritual"),
            'sanskrit_hindi_corrections': lambda: processor._apply_enhanced_sanskrit_hindi_corrections(test_segment.text),
            'lexicon_corrections': lambda: processor._apply_lexicon_corrections(test_segment.text),
            'ner_processing': lambda: self._profile_ner_if_enabled(processor, test_segment),
            'confidence_calculation': lambda: processor._calculate_enhanced_confidence(test_segment.text, [], test_segment.text)
        }
        
        component_times = {}
        
        for component_name, component_func in components.items():
            # Warm up
            try:
                component_func()
            except:
                pass
            
            # Profile with multiple runs for accuracy
            times = []
            for _ in range(10):
                start_time = time.perf_counter()
                try:
                    component_func()
                except Exception as e:
                    print(f"   Warning: {component_name} failed: {e}")
                    continue
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms
            
            if times:
                avg_time = sum(times) / len(times)
                component_times[component_name] = avg_time
                print(f"   {component_name}: {avg_time:.2f}ms")
        
        results['component_times'] = component_times
        
        # Identify bottleneck
        if component_times:
            slowest_component = max(component_times.items(), key=lambda x: x[1])
            total_component_time = sum(component_times.values())
            
            results['bottleneck'] = {
                'component': slowest_component[0],
                'time_ms': slowest_component[1],
                'percentage_of_total': (slowest_component[1] / total_component_time) * 100
            }
            
            print(f"   BOTTLENECK: {slowest_component[0]} ({slowest_component[1]:.2f}ms, {results['bottleneck']['percentage_of_total']:.1f}%)")
        print()
    
    def _profile_ner_if_enabled(self, processor, test_segment):
        """Profile NER processing if enabled."""
        if processor.enable_ner and processor.ner_model:
            ner_result = processor.ner_model.identify_entities(test_segment.text)
            if processor.capitalization_engine:
                processor.capitalization_engine.capitalize_text(test_segment.text)
            return ner_result
        return None
    
    def _profile_with_cprofile(self, processor, test_segments, results):
        """Profile with cProfile for detailed function-level analysis."""
        print("3. Running detailed cProfile analysis...")
        
        # Create profiler
        profiler = cProfile.Profile()
        
        # Profile the processing
        profiler.enable()
        for segment in test_segments[:3]:  # Use subset for detailed analysis
            processor._process_srt_segment(segment, processor.metrics_collector.create_file_metrics("test"))
        profiler.disable()
        
        # Capture stats
        stats_stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions
        
        profile_output = stats_stream.getvalue()
        results['detailed_profile'] = profile_output
        
        # Extract key function times
        stats.sort_stats('tottime')
        key_functions = []
        for func_info in stats.get_stats().items():
            filename, line_num, func_name = func_info[0]
            tottime, cumtime, ncalls = func_info[1][:3]
            
            # Focus on our code (skip standard library)
            if 'src/' in filename or func_name in ['normalize_with_advanced_tracking', 'process_numbers', '_apply_enhanced_sanskrit_hindi_corrections']:
                key_functions.append({
                    'function': f"{filename}:{line_num}({func_name})",
                    'total_time': tottime,
                    'cumulative_time': cumtime,
                    'calls': ncalls,
                    'time_per_call': tottime / ncalls if ncalls > 0 else 0
                })
        
        # Sort by total time and take top 10
        key_functions.sort(key=lambda x: x['total_time'], reverse=True)
        results['top_functions'] = key_functions[:10]
        
        print("   Top time-consuming functions:")
        for i, func in enumerate(key_functions[:5], 1):
            print(f"   {i}. {func['function']}")
            print(f"      Total time: {func['total_time']:.4f}s, Calls: {func['calls']}, Per call: {func['time_per_call']*1000:.2f}ms")
        print()
    
    def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive performance report."""
        report = []
        report.append("=" * 80)
        report.append("PERFORMANCE BOTTLENECK ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall performance
        overall = results['overall']
        report.append("CURRENT PERFORMANCE STATUS:")
        report.append(f"  Current Speed: {overall['segments_per_second']:.2f} segments/sec")
        report.append(f"  Target Speed:  {overall['target_segments_per_second']:.2f} segments/sec")
        report.append(f"  Performance Gap: {overall['target_segments_per_second'] - overall['segments_per_second']:.2f} segments/sec")
        report.append(f"  Performance Ratio: {overall['performance_ratio']:.2f}x (need {1/overall['performance_ratio']:.1f}x improvement)")
        report.append("")
        
        # Component analysis
        if 'component_times' in results:
            report.append("COMPONENT PERFORMANCE BREAKDOWN:")
            component_times = results['component_times']
            sorted_components = sorted(component_times.items(), key=lambda x: x[1], reverse=True)
            
            total_time = sum(component_times.values())
            for component, time_ms in sorted_components:
                percentage = (time_ms / total_time) * 100
                report.append(f"  {component:<30}: {time_ms:6.2f}ms ({percentage:5.1f}%)")
            report.append("")
        
        # Bottleneck identification
        if 'bottleneck' in results:
            bottleneck = results['bottleneck']
            report.append("IDENTIFIED BOTTLENECK:")
            report.append(f"  Component: {bottleneck['component']}")
            report.append(f"  Time: {bottleneck['time_ms']:.2f}ms ({bottleneck['percentage_of_total']:.1f}% of total)")
            report.append("")
        
        # Optimization recommendations
        report.append("OPTIMIZATION RECOMMENDATIONS:")
        
        if 'bottleneck' in results:
            component = results['bottleneck']['component']
            
            if 'text_normalizer' in component:
                report.append("  1. TEXT NORMALIZER OPTIMIZATION:")
                report.append("     - Implement caching for repeated normalization patterns")
                report.append("     - Optimize regex compilation (compile once, reuse)")
                report.append("     - Consider parallel processing for independent operations")
                
            elif 'sanskrit_hindi_corrections' in component:
                report.append("  1. SANSKRIT/HINDI CORRECTIONS OPTIMIZATION:")
                report.append("     - Cache lexicon lookups")
                report.append("     - Optimize fuzzy matching algorithm")
                report.append("     - Implement early termination for high-confidence matches")
                
            elif 'number_processor' in component:
                report.append("  1. NUMBER PROCESSOR OPTIMIZATION:")
                report.append("     - Pre-compile regex patterns")
                report.append("     - Cache contextual analysis results")
                report.append("     - Optimize MCP client calls")
                
            elif 'ner_processing' in component:
                report.append("  1. NER PROCESSING OPTIMIZATION:")
                report.append("     - Implement model prediction caching")
                report.append("     - Optimize entity recognition patterns")
                report.append("     - Batch process multiple segments")
        
        report.append("  2. GENERAL OPTIMIZATIONS:")
        report.append("     - Implement multi-threading for parallel segment processing")
        report.append("     - Add memory pooling to reduce garbage collection")
        report.append("     - Cache frequently accessed configuration")
        report.append("     - Optimize I/O operations")
        report.append("")
        
        # Detailed function analysis
        if 'top_functions' in results:
            report.append("TOP TIME-CONSUMING FUNCTIONS:")
            for i, func in enumerate(results['top_functions'][:5], 1):
                report.append(f"  {i}. {func['function']}")
                report.append(f"     Total: {func['total_time']:.4f}s, Calls: {func['calls']}, Per-call: {func['time_per_call']*1000:.2f}ms")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)

def main():
    """Main profiling function."""
    profiler = PerformanceProfiler()
    
    try:
        results = profiler.profile_sanskrit_processor()
        report = profiler.generate_performance_report(results)
        
        # Print report to console
        print(report)
        
        # Save report to file
        report_path = Path("performance_bottleneck_analysis.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\nDetailed report saved to: {report_path}")
        
        # Return key findings
        overall = results['overall']
        if overall['performance_ratio'] < 0.5:  # Less than 50% of target
            print("\n🚨 CRITICAL PERFORMANCE ISSUE DETECTED")
            print(f"Current performance is {overall['performance_ratio']:.1%} of target")
            if 'bottleneck' in results:
                print(f"Primary bottleneck: {results['bottleneck']['component']}")
        
        return results
        
    except Exception as e:
        print(f"Error during profiling: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()