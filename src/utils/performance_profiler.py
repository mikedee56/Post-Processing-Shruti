# Performance Profiler for Story 5.1 Performance Optimization
# Comprehensive profiling framework for identifying bottlenecks

import time
import cProfile
import pstats
import io
import gc
import tracemalloc
import statistics
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for a single operation."""
    operation_name: str
    execution_time: float  # seconds
    memory_peak: Optional[int] = None  # bytes
    memory_diff: Optional[int] = None  # bytes
    call_count: int = 1
    cpu_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProfilerResults:
    """Results from performance profiling session."""
    session_name: str
    total_time: float
    metrics: List[PerformanceMetrics]
    variance_percentage: float
    bottlenecks: List[str]
    memory_usage: Dict[str, int]
    recommendations: List[str] = field(default_factory=list)

class PerformanceProfiler:
    """
    Comprehensive performance profiler for Story 5.1 optimization.
    
    This profiler identifies bottlenecks in the SRT processing pipeline
    and provides recommendations for optimization.
    """
    
    def __init__(self, enable_memory_tracking: bool = True):
        self.enable_memory_tracking = enable_memory_tracking
        self.metrics: List[PerformanceMetrics] = []
        self.profiler_stack: List[str] = []
        self.start_times: Dict[str, float] = {}
        self.memory_snapshots: Dict[str, int] = {}
        
        if self.enable_memory_tracking:
            tracemalloc.start()
    
    @contextmanager
    def profile_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for profiling individual operations.
        
        Args:
            operation_name: Name of the operation being profiled
            metadata: Additional metadata to store with metrics
        """
        if metadata is None:
            metadata = {}
            
        # Start profiling
        start_time = time.perf_counter()
        memory_start = None
        
        if self.enable_memory_tracking:
            current, peak = tracemalloc.get_traced_memory()
            memory_start = current
        
        # Setup CPU profiling
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            yield
        finally:
            # Stop profiling
            profiler.disable()
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            # Memory measurements
            memory_peak = None
            memory_diff = None
            if self.enable_memory_tracking and memory_start is not None:
                current, peak = tracemalloc.get_traced_memory()
                memory_peak = peak
                memory_diff = current - memory_start
            
            # CPU time from profiler
            s = io.StringIO()
            stats = pstats.Stats(profiler, stream=s)
            cpu_time = stats.total_tt
            
            # Create metrics
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                execution_time=execution_time,
                memory_peak=memory_peak,
                memory_diff=memory_diff,
                cpu_time=cpu_time,
                metadata=metadata
            )
            
            self.metrics.append(metrics)
            logger.debug(f"Profiled {operation_name}: {execution_time:.4f}s")
    
    def profile_function(self, func: Callable, *args, operation_name: str = None, **kwargs) -> Any:
        """
        Profile a function call and return its result.
        
        Args:
            func: Function to profile
            *args: Function arguments
            operation_name: Name for the operation (defaults to function name)
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        if operation_name is None:
            operation_name = getattr(func, '__name__', 'anonymous_function')
        
        with self.profile_operation(operation_name):
            return func(*args, **kwargs)
    
    def analyze_bottlenecks(self, threshold_ms: float = 10.0) -> List[str]:
        """
        Analyze metrics to identify performance bottlenecks.
        
        Args:
            threshold_ms: Operations taking longer than this are considered bottlenecks
            
        Returns:
            List of bottleneck descriptions
        """
        bottlenecks = []
        threshold_seconds = threshold_ms / 1000.0
        
        # Group metrics by operation name
        operation_groups = {}
        for metric in self.metrics:
            if metric.operation_name not in operation_groups:
                operation_groups[metric.operation_name] = []
            operation_groups[metric.operation_name].append(metric)
        
        # Analyze each operation
        for operation_name, operation_metrics in operation_groups.items():
            times = [m.execution_time for m in operation_metrics]
            avg_time = statistics.mean(times)
            
            if avg_time > threshold_seconds:
                if len(times) > 1:
                    stdev = statistics.stdev(times)
                    variance_pct = (stdev / avg_time * 100) if avg_time > 0 else 0
                    bottlenecks.append(f"{operation_name}: {avg_time:.4f}s avg, {variance_pct:.1f}% variance")
                else:
                    bottlenecks.append(f"{operation_name}: {avg_time:.4f}s")
        
        return bottlenecks
    
    def calculate_variance(self) -> float:
        """Calculate overall performance variance percentage."""
        if len(self.metrics) < 2:
            return 0.0
        
        times = [m.execution_time for m in self.metrics]
        mean_time = statistics.mean(times)
        stdev_time = statistics.stdev(times)
        
        return (stdev_time / mean_time * 100) if mean_time > 0 else 0.0
    
    def generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on profiling results."""
        recommendations = []
        
        # Analyze bottlenecks
        bottlenecks = self.analyze_bottlenecks()
        
        # Memory-related recommendations
        memory_intensive = [m for m in self.metrics if m.memory_diff and m.memory_diff > 1024*1024]  # > 1MB
        if memory_intensive:
            recommendations.append("Consider caching for memory-intensive operations")
            recommendations.append("Implement object pooling for frequently created objects")
        
        # Variance-related recommendations
        variance = self.calculate_variance()
        if variance > 50:
            recommendations.append("High variance detected - implement consistent caching")
            recommendations.append("Optimize lazy loading to prevent initialization variance")
        
        # Sanskrit parser specific recommendations
        sanskrit_operations = [m for m in self.metrics if 'sanskrit' in m.operation_name.lower()]
        if sanskrit_operations:
            avg_sanskrit_time = statistics.mean([m.execution_time for m in sanskrit_operations])
            if avg_sanskrit_time > 0.010:  # > 10ms
                recommendations.append("Sanskrit parser loading detected - implement Word2Vec caching")
                recommendations.append("Pre-load sanskrit_parser models at startup")
        
        return recommendations
    
    def generate_report(self, session_name: str = "Performance Analysis") -> ProfilerResults:
        """Generate a comprehensive performance report."""
        total_time = sum(m.execution_time for m in self.metrics)
        variance = self.calculate_variance()
        bottlenecks = self.analyze_bottlenecks()
        
        # Memory usage summary
        memory_usage = {}
        if self.enable_memory_tracking:
            current, peak = tracemalloc.get_traced_memory()
            memory_usage = {
                'current_bytes': current,
                'peak_bytes': peak,
                'total_allocated': sum(m.memory_diff for m in self.metrics if m.memory_diff)
            }
        
        recommendations = self.generate_recommendations()
        
        return ProfilerResults(
            session_name=session_name,
            total_time=total_time,
            metrics=self.metrics.copy(),
            variance_percentage=variance,
            bottlenecks=bottlenecks,
            memory_usage=memory_usage,
            recommendations=recommendations
        )
    
    def save_report(self, report: ProfilerResults, output_path: Path):
        """Save profiling report to JSON file."""
        report_data = {
            'session_name': report.session_name,
            'total_time': report.total_time,
            'variance_percentage': report.variance_percentage,
            'bottlenecks': report.bottlenecks,
            'memory_usage': report.memory_usage,
            'recommendations': report.recommendations,
            'metrics': [
                {
                    'operation_name': m.operation_name,
                    'execution_time': m.execution_time,
                    'memory_peak': m.memory_peak,
                    'memory_diff': m.memory_diff,
                    'call_count': m.call_count,
                    'cpu_time': m.cpu_time,
                    'metadata': m.metadata
                }
                for m in report.metrics
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Performance report saved to {output_path}")
    
    def reset(self):
        """Reset profiler state for new profiling session."""
        self.metrics.clear()
        self.profiler_stack.clear()
        self.start_times.clear()
        self.memory_snapshots.clear()
        
        if self.enable_memory_tracking:
            tracemalloc.clear_traces()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if self.enable_memory_tracking:
            tracemalloc.stop()

class SRTProcessingProfiler:
    """
    Specialized profiler for SRT processing pipeline.
    
    This profiler instruments the main SRT processing components
    to identify performance bottlenecks specific to Story 5.1.
    """
    
    def __init__(self, processor):
        self.processor = processor
        self.profiler = PerformanceProfiler()
        self.original_methods = {}
        
    def instrument_processor(self):
        """Instrument SanskritPostProcessor methods for profiling."""
        # Store original methods
        self.original_methods['_process_srt_segment'] = self.processor._process_srt_segment
        self.original_methods['_apply_text_normalization'] = self.processor._apply_text_normalization
        self.original_methods['_apply_lexicon_corrections'] = self.processor._apply_lexicon_corrections
        self.original_methods['_apply_ner_processing'] = self.processor._apply_ner_processing
        
        # Wrap methods with profiling
        self.processor._process_srt_segment = self._profile_segment_processing
        self.processor._apply_text_normalization = self._profile_text_normalization
        self.processor._apply_lexicon_corrections = self._profile_lexicon_corrections
        self.processor._apply_ner_processing = self._profile_ner_processing
    
    def _profile_segment_processing(self, segment, file_metrics):
        """Profiled version of _process_srt_segment."""
        with self.profiler.profile_operation("srt_segment_processing", 
                                           {"text_length": len(segment.text)}):
            return self.original_methods['_process_srt_segment'](segment, file_metrics)
    
    def _profile_text_normalization(self, text):
        """Profiled version of _apply_text_normalization."""
        with self.profiler.profile_operation("text_normalization", 
                                           {"text_length": len(text)}):
            return self.original_methods['_apply_text_normalization'](text)
    
    def _profile_lexicon_corrections(self, text):
        """Profiled version of _apply_lexicon_corrections."""
        with self.profiler.profile_operation("lexicon_corrections", 
                                           {"text_length": len(text)}):
            return self.original_methods['_apply_lexicon_corrections'](text)
    
    def _profile_ner_processing(self, text):
        """Profiled version of _apply_ner_processing."""
        with self.profiler.profile_operation("ner_processing", 
                                           {"text_length": len(text)}):
            return self.original_methods['_apply_ner_processing'](text)
    
    def restore_processor(self):
        """Restore original processor methods."""
        for method_name, original_method in self.original_methods.items():
            setattr(self.processor, method_name, original_method)
    
    def profile_batch(self, segments, session_name: str = "SRT Batch Processing"):
        """Profile processing of a batch of segments."""
        self.instrument_processor()
        
        try:
            for segment in segments:
                file_metrics = self.processor.metrics_collector.create_file_metrics("profiling")
                self.processor._process_srt_segment(segment, file_metrics)
        finally:
            self.restore_processor()
        
        return self.profiler.generate_report(session_name)