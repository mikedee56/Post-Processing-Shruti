#!/usr/bin/env python3
"""
Performance Regression Testing Suite for ASR Post-Processing System
Validates system performance and detects performance regressions.

Part of Story 5.5: Testing & Quality Assurance Framework
Target: Maintain 10+ segments/sec processing with <10% variance
"""

import sys
import os
import pytest
import time
import statistics
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import resource
import psutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@dataclass
class PerformanceMetrics:
    """Performance measurement results."""
    operation: str
    duration: float
    throughput: float  # operations per second
    memory_usage: int  # bytes
    cpu_usage: float  # percentage
    variance: float  # coefficient of variation
    samples: int
    timestamp: datetime


@dataclass
class PerformanceBenchmark:
    """Performance benchmark configuration."""
    name: str
    target_throughput: float
    max_variance: float  # percentage
    max_memory_mb: int
    max_cpu_percent: float
    timeout_seconds: int = 300


class PerformanceMonitor:
    """Monitor system performance during tests."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = False
        self.metrics = []
        self._stop_event = threading.Event()
        self._monitor_thread = None
    
    def start_monitoring(self, interval: float = 0.1):
        """Start performance monitoring."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return metrics."""
        if not self.monitoring:
            return {}
        
        self.monitoring = False
        self._stop_event.set()
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        
        if not self.metrics:
            return {}
        
        # Calculate aggregated metrics
        cpu_values = [m['cpu_percent'] for m in self.metrics]
        memory_values = [m['memory_mb'] for m in self.metrics]
        
        return {
            'avg_cpu_percent': statistics.mean(cpu_values),
            'max_cpu_percent': max(cpu_values),
            'avg_memory_mb': statistics.mean(memory_values),
            'max_memory_mb': max(memory_values),
            'sample_count': len(self.metrics)
        }
    
    def _monitor_loop(self, interval: float):
        """Monitor loop running in separate thread."""
        while not self._stop_event.wait(interval):
            try:
                cpu_percent = self.process.cpu_percent()
                memory_info = self.process.memory_info()
                
                self.metrics.append({
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory_info.rss / (1024 * 1024)
                })
            except Exception:
                # Ignore monitoring errors
                pass


class PerformanceTester:
    """Performance testing framework."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.benchmarks = self._load_benchmarks()
        self.monitor = PerformanceMonitor()
        self.baseline_file = Path(config.get("baseline_file", "performance_baseline.json"))
        self.baseline_data = self._load_baseline()
    
    def _load_benchmarks(self) -> Dict[str, PerformanceBenchmark]:
        """Load performance benchmarks."""
        return {
            "segment_processing": PerformanceBenchmark(
                name="Single Segment Processing",
                target_throughput=10.0,  # segments/sec
                max_variance=10.0,  # 10%
                max_memory_mb=512,
                max_cpu_percent=80.0,
                timeout_seconds=60
            ),
            "batch_processing": PerformanceBenchmark(
                name="Batch SRT Processing",
                target_throughput=5.0,  # files/sec
                max_variance=15.0,  # 15%
                max_memory_mb=1024,
                max_cpu_percent=90.0,
                timeout_seconds=300
            ),
            "concurrent_processing": PerformanceBenchmark(
                name="Concurrent Processing",
                target_throughput=8.0,  # segments/sec per worker
                max_variance=20.0,  # 20%
                max_memory_mb=2048,
                max_cpu_percent=95.0,
                timeout_seconds=300
            ),
            "memory_efficiency": PerformanceBenchmark(
                name="Memory Efficiency",
                target_throughput=10.0,  # segments/sec
                max_variance=10.0,  # 10%
                max_memory_mb=256,  # Strict memory limit
                max_cpu_percent=70.0,
                timeout_seconds=120
            )
        }
    
    def _load_baseline(self) -> Dict[str, Any]:
        """Load performance baseline data."""
        if self.baseline_file.exists():
            try:
                with open(self.baseline_file) as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load baseline data: {e}")
        
        return {}
    
    def _save_baseline(self, metrics: Dict[str, PerformanceMetrics]):
        """Save performance baseline data."""
        baseline_data = {}
        
        for name, metric in metrics.items():
            baseline_data[name] = {
                "operation": metric.operation,
                "duration": metric.duration,
                "throughput": metric.throughput,
                "memory_usage": metric.memory_usage,
                "cpu_usage": metric.cpu_usage,
                "variance": metric.variance,
                "samples": metric.samples,
                "timestamp": metric.timestamp.isoformat()
            }
        
        try:
            with open(self.baseline_file, 'w') as f:
                json.dump(baseline_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save baseline data: {e}")
    
    def measure_operation_performance(self,
                                     operation_func: callable,
                                     operation_name: str,
                                     iterations: int = 10,
                                     warmup_iterations: int = 3) -> PerformanceMetrics:
        """Measure performance of a specific operation."""
        # Warmup runs
        for _ in range(warmup_iterations):
            try:
                operation_func()
            except Exception:
                pass
        
        # Actual measurement
        durations = []
        memory_usage = []
        cpu_usage = []
        
        for i in range(iterations):
            # Start monitoring
            self.monitor.start_monitoring()
            
            # Measure operation
            start_time = time.perf_counter()
            start_memory = self.monitor.process.memory_info().rss
            
            try:
                operation_func()
            except Exception as e:
                print(f"Warning: Operation failed in iteration {i}: {e}")
                continue
            
            end_time = time.perf_counter()
            end_memory = self.monitor.process.memory_info().rss
            
            # Stop monitoring and get metrics
            system_metrics = self.monitor.stop_monitoring()
            
            # Record measurements
            duration = end_time - start_time
            durations.append(duration)
            memory_usage.append(max(end_memory, start_memory))
            
            if system_metrics:
                cpu_usage.append(system_metrics.get('avg_cpu_percent', 0))
        
        if not durations:
            raise RuntimeError(f"No successful measurements for {operation_name}")
        
        # Calculate statistics
        avg_duration = statistics.mean(durations)
        variance = (statistics.stdev(durations) / avg_duration * 100) if len(durations) > 1 else 0
        throughput = 1.0 / avg_duration if avg_duration > 0 else 0
        avg_memory = statistics.mean(memory_usage) if memory_usage else 0
        avg_cpu = statistics.mean(cpu_usage) if cpu_usage else 0
        
        return PerformanceMetrics(
            operation=operation_name,
            duration=avg_duration,
            throughput=throughput,
            memory_usage=int(avg_memory),
            cpu_usage=avg_cpu,
            variance=variance,
            samples=len(durations),
            timestamp=datetime.now()
        )
    
    def run_performance_benchmark(self, benchmark_name: str) -> Dict[str, Any]:
        """Run a specific performance benchmark."""
        if benchmark_name not in self.benchmarks:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
        benchmark = self.benchmarks[benchmark_name]
        
        # Run benchmark based on type
        if benchmark_name == "segment_processing":
            return self._benchmark_segment_processing(benchmark)
        elif benchmark_name == "batch_processing":
            return self._benchmark_batch_processing(benchmark)
        elif benchmark_name == "concurrent_processing":
            return self._benchmark_concurrent_processing(benchmark)
        elif benchmark_name == "memory_efficiency":
            return self._benchmark_memory_efficiency(benchmark)
        else:
            raise NotImplementedError(f"Benchmark {benchmark_name} not implemented")
    
    def _benchmark_segment_processing(self, benchmark: PerformanceBenchmark) -> Dict[str, Any]:
        """Benchmark single segment processing performance."""
        try:
            from post_processors.sanskrit_post_processor import SanskritPostProcessor
            from utils.srt_parser import SRTSegment
            
            processor = SanskritPostProcessor()
            test_segment = SRTSegment(
                index=1,
                start_time="00:00:01,000",
                end_time="00:00:05,000",
                text="Today we study yoga and dharma from ancient scriptures.",
                raw_text="Today we study yoga and dharma from ancient scriptures."
            )
            
            def process_segment():
                file_metrics = processor.metrics_collector.create_file_metrics("perf_test")
                return processor._process_srt_segment(test_segment, file_metrics)
            
            metrics = self.measure_operation_performance(
                process_segment,
                "segment_processing",
                iterations=20
            )
            
            # Validate against benchmark
            results = {
                "benchmark": benchmark.name,
                "metrics": metrics,
                "passed": True,
                "issues": []
            }
            
            # Check throughput
            if metrics.throughput < benchmark.target_throughput:
                results["passed"] = False
                results["issues"].append(
                    f"Throughput {metrics.throughput:.2f} < target {benchmark.target_throughput}"
                )
            
            # Check variance
            if metrics.variance > benchmark.max_variance:
                results["passed"] = False
                results["issues"].append(
                    f"Variance {metrics.variance:.1f}% > max {benchmark.max_variance}%"
                )
            
            # Check memory usage
            memory_mb = metrics.memory_usage / (1024 * 1024)
            if memory_mb > benchmark.max_memory_mb:
                results["passed"] = False
                results["issues"].append(
                    f"Memory usage {memory_mb:.1f} MB > max {benchmark.max_memory_mb} MB"
                )
            
            return results
            
        except ImportError:
            pytest.skip("SanskritPostProcessor not available for performance testing")
        except Exception as e:
            return {
                "benchmark": benchmark.name,
                "error": str(e),
                "passed": False
            }
    
    def _benchmark_batch_processing(self, benchmark: PerformanceBenchmark) -> Dict[str, Any]:
        """Benchmark batch SRT file processing performance."""
        try:
            from post_processors.sanskrit_post_processor import SanskritPostProcessor
            
            processor = SanskritPostProcessor()
            
            # Create test SRT files
            test_files = []
            for i in range(5):
                content = f"""1
00:00:01,000 --> 00:00:05,000
Today we study yoga and dharma segment {i}.

2
00:00:06,000 --> 00:00:10,000
This practice brings peace and wisdom {i}."""

                with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
                    f.write(content)
                    test_files.append(Path(f.name))
            
            def process_batch():
                processed_files = []
                for input_file in test_files:
                    output_file = input_file.with_suffix('.processed.srt')
                    processor.process_srt_file(input_file, output_file)
                    processed_files.append(output_file)
                
                # Cleanup output files
                for output_file in processed_files:
                    if output_file.exists():
                        output_file.unlink()
                
                return len(test_files)
            
            metrics = self.measure_operation_performance(
                process_batch,
                "batch_processing",
                iterations=5
            )
            
            # Cleanup test files
            for test_file in test_files:
                if test_file.exists():
                    test_file.unlink()
            
            return self._validate_benchmark_results(metrics, benchmark)
            
        except ImportError:
            pytest.skip("Components not available for batch processing performance testing")
        except Exception as e:
            return {
                "benchmark": benchmark.name,
                "error": str(e),
                "passed": False
            }
    
    def _benchmark_concurrent_processing(self, benchmark: PerformanceBenchmark) -> Dict[str, Any]:
        """Benchmark concurrent processing performance."""
        try:
            from post_processors.sanskrit_post_processor import SanskritPostProcessor
            from utils.srt_parser import SRTSegment
            
            processor = SanskritPostProcessor()
            num_workers = min(4, multiprocessing.cpu_count())
            
            def create_test_segment(index: int):
                return SRTSegment(
                    index=index,
                    start_time="00:00:01,000",
                    end_time="00:00:05,000",
                    text=f"Today we study yoga and dharma segment {index}.",
                    raw_text=f"Today we study yoga and dharma segment {index}."
                )
            
            def process_concurrent():
                segments = [create_test_segment(i) for i in range(20)]
                
                def process_segment(segment):
                    file_metrics = processor.metrics_collector.create_file_metrics(f"perf_test_{segment.index}")
                    return processor._process_srt_segment(segment, file_metrics)
                
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    results = list(executor.map(process_segment, segments))
                
                return len(results)
            
            metrics = self.measure_operation_performance(
                process_concurrent,
                "concurrent_processing",
                iterations=3
            )
            
            return self._validate_benchmark_results(metrics, benchmark)
            
        except ImportError:
            pytest.skip("Components not available for concurrent processing performance testing")
        except Exception as e:
            return {
                "benchmark": benchmark.name,
                "error": str(e),
                "passed": False
            }
    
    def _benchmark_memory_efficiency(self, benchmark: PerformanceBenchmark) -> Dict[str, Any]:
        """Benchmark memory efficiency performance."""
        try:
            from post_processors.sanskrit_post_processor import SanskritPostProcessor
            from utils.srt_parser import SRTSegment
            
            # Use configuration optimized for memory efficiency
            config = {
                "enable_performance_monitoring": False,
                "enable_logging": False,
                "text_normalization": {
                    "enable_mcp_processing": False,  # Disable heavy MCP processing
                    "enable_monitoring": False
                }
            }
            
            processor = SanskritPostProcessor(config)
            
            def process_memory_efficient():
                # Process multiple segments to test memory usage
                segments = []
                for i in range(50):
                    segment = SRTSegment(
                        index=i,
                        start_time="00:00:01,000",
                        end_time="00:00:05,000",
                        text=f"Simple text segment {i}.",
                        raw_text=f"Simple text segment {i}."
                    )
                    segments.append(segment)
                
                file_metrics = processor.metrics_collector.create_file_metrics("memory_test")
                processed = []
                
                for segment in segments:
                    result = processor._process_srt_segment(segment, file_metrics)
                    processed.append(result)
                
                return len(processed)
            
            metrics = self.measure_operation_performance(
                process_memory_efficient,
                "memory_efficiency",
                iterations=5
            )
            
            return self._validate_benchmark_results(metrics, benchmark)
            
        except ImportError:
            pytest.skip("Components not available for memory efficiency performance testing")
        except Exception as e:
            return {
                "benchmark": benchmark.name,
                "error": str(e),
                "passed": False
            }
    
    def _validate_benchmark_results(self, metrics: PerformanceMetrics, benchmark: PerformanceBenchmark) -> Dict[str, Any]:
        """Validate benchmark results against targets."""
        results = {
            "benchmark": benchmark.name,
            "metrics": metrics,
            "passed": True,
            "issues": []
        }
        
        # Check throughput
        if metrics.throughput < benchmark.target_throughput:
            results["passed"] = False
            results["issues"].append(
                f"Throughput {metrics.throughput:.2f} < target {benchmark.target_throughput}"
            )
        
        # Check variance
        if metrics.variance > benchmark.max_variance:
            results["passed"] = False
            results["issues"].append(
                f"Variance {metrics.variance:.1f}% > max {benchmark.max_variance}%"
            )
        
        # Check memory usage
        memory_mb = metrics.memory_usage / (1024 * 1024)
        if memory_mb > benchmark.max_memory_mb:
            results["passed"] = False
            results["issues"].append(
                f"Memory usage {memory_mb:.1f} MB > max {benchmark.max_memory_mb} MB"
            )
        
        # Check CPU usage
        if metrics.cpu_usage > benchmark.max_cpu_percent:
            results["passed"] = False
            results["issues"].append(
                f"CPU usage {metrics.cpu_usage:.1f}% > max {benchmark.max_cpu_percent}%"
            )
        
        return results
    
    def check_performance_regression(self, current_metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Check for performance regression against baseline."""
        if not self.baseline_data or current_metrics.operation not in self.baseline_data:
            return {"regression_detected": False, "reason": "No baseline data available"}
        
        baseline = self.baseline_data[current_metrics.operation]
        baseline_throughput = baseline["throughput"]
        
        # Calculate regression threshold (10% decrease)
        regression_threshold = 0.10
        throughput_decrease = (baseline_throughput - current_metrics.throughput) / baseline_throughput
        
        regression_result = {
            "regression_detected": False,
            "baseline_throughput": baseline_throughput,
            "current_throughput": current_metrics.throughput,
            "throughput_change_percent": -throughput_decrease * 100,
            "threshold_percent": regression_threshold * 100
        }
        
        if throughput_decrease > regression_threshold:
            regression_result["regression_detected"] = True
            regression_result["severity"] = "major" if throughput_decrease > 0.20 else "minor"
        
        return regression_result


class TestPerformanceRegression:
    """Performance regression tests."""
    
    @pytest.fixture(autouse=True)
    def setup(self, test_config):
        """Set up performance testing environment."""
        self.test_config = test_config
        self.performance_tester = PerformanceTester(test_config)
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_segment_processing_performance(self):
        """Test single segment processing performance."""
        results = self.performance_tester.run_performance_benchmark("segment_processing")
        
        if "error" in results:
            pytest.fail(f"Performance benchmark failed: {results['error']}")
        
        assert results["passed"], f"Performance benchmark failed: {results['issues']}"
        
        metrics = results["metrics"]
        print(f"Segment processing performance: {metrics.throughput:.2f} segments/sec "
              f"(variance: {metrics.variance:.1f}%)")
        
        # Check for regression
        regression = self.performance_tester.check_performance_regression(metrics)
        if regression["regression_detected"]:
            pytest.fail(f"Performance regression detected: "
                       f"{regression['throughput_change_percent']:.1f}% decrease")
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_batch_processing_performance(self):
        """Test batch SRT processing performance."""
        results = self.performance_tester.run_performance_benchmark("batch_processing")
        
        if "error" in results:
            pytest.fail(f"Batch performance benchmark failed: {results['error']}")
        
        assert results["passed"], f"Batch performance benchmark failed: {results['issues']}"
        
        metrics = results["metrics"]
        print(f"Batch processing performance: {metrics.throughput:.2f} files/sec "
              f"(variance: {metrics.variance:.1f}%)")
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_concurrent_processing_performance(self):
        """Test concurrent processing performance."""
        results = self.performance_tester.run_performance_benchmark("concurrent_processing")
        
        if "error" in results:
            pytest.fail(f"Concurrent performance benchmark failed: {results['error']}")
        
        assert results["passed"], f"Concurrent performance benchmark failed: {results['issues']}"
        
        metrics = results["metrics"]
        print(f"Concurrent processing performance: {metrics.throughput:.2f} ops/sec "
              f"(variance: {metrics.variance:.1f}%)")
    
    @pytest.mark.performance
    def test_memory_efficiency_performance(self):
        """Test memory efficiency performance."""
        results = self.performance_tester.run_performance_benchmark("memory_efficiency")
        
        if "error" in results:
            pytest.fail(f"Memory efficiency benchmark failed: {results['error']}")
        
        assert results["passed"], f"Memory efficiency benchmark failed: {results['issues']}"
        
        metrics = results["metrics"]
        memory_mb = metrics.memory_usage / (1024 * 1024)
        print(f"Memory efficiency: {metrics.throughput:.2f} ops/sec "
              f"(memory: {memory_mb:.1f} MB, variance: {metrics.variance:.1f}%)")
    
    @pytest.mark.performance
    def test_performance_baseline_update(self):
        """Update performance baseline with current measurements."""
        baseline_metrics = {}
        
        for benchmark_name in ["segment_processing", "memory_efficiency"]:
            try:
                results = self.performance_tester.run_performance_benchmark(benchmark_name)
                if not "error" in results and results["passed"]:
                    baseline_metrics[benchmark_name] = results["metrics"]
            except Exception as e:
                print(f"Warning: Failed to measure {benchmark_name}: {e}")
        
        if baseline_metrics:
            self.performance_tester._save_baseline(baseline_metrics)
            print(f"Updated performance baseline with {len(baseline_metrics)} benchmarks")
        else:
            pytest.skip("No successful benchmarks to update baseline")


# Test markers for categorization
pytestmark = [
    pytest.mark.performance,
    pytest.mark.regression,
    pytest.mark.slow
]


if __name__ == "__main__":
    # Run performance tests directly
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "performance",
        "-x"  # Stop on first failure
    ])