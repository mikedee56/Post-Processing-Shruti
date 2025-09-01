"""
Performance monitoring and metrics collection system for QA architecture components.

This module provides comprehensive performance monitoring capabilities for the 
TechnicalQualityGate, OptimizedASRScriptureMatcher, and RobustWisdomLibraryIntegrator
components, ensuring they meet specified performance requirements.
"""

import time
import psutil
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict, deque
import json
from pathlib import Path
import logging
from contextlib import contextmanager
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger(__name__)


@dataclass
class PerformanceThresholds:
    """Defines performance thresholds for different components."""
    search_latency_p95_ms: float = 100.0
    memory_usage_max_mb: float = 2048.0
    cache_hit_ratio_min: float = 0.85
    concurrent_requests: int = 100
    cpu_usage_max_percent: float = 80.0
    integration_latency_p95_ms: float = 500.0
    quality_gate_latency_p95_ms: float = 50.0
    error_rate_max_percent: float = 1.0


@dataclass
class MetricSnapshot:
    """Represents a single performance measurement."""
    timestamp: datetime
    component: str
    operation: str
    latency_ms: float
    memory_mb: float
    cpu_percent: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceReport:
    """Comprehensive performance analysis report."""
    component: str
    time_period: timedelta
    total_operations: int
    success_rate: float
    error_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    avg_memory_mb: float
    peak_memory_mb: float
    avg_cpu_percent: float
    peak_cpu_percent: float
    cache_hit_ratio: Optional[float] = None
    threshold_violations: List[str] = field(default_factory=list)
    performance_grade: str = "A"


class MetricsCollector:
    """Collects and aggregates performance metrics."""
    
    def __init__(self, max_snapshots: int = 10000):
        self.max_snapshots = max_snapshots
        self._snapshots: deque = deque(maxlen=max_snapshots)
        self._lock = threading.RLock()
        self._cache_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {'hits': 0, 'misses': 0})
        
    def record_metric(self, snapshot: MetricSnapshot) -> None:
        """Record a performance metric snapshot."""
        with self._lock:
            self._snapshots.append(snapshot)
    
    def record_cache_hit(self, component: str) -> None:
        """Record a cache hit for the specified component."""
        with self._lock:
            self._cache_stats[component]['hits'] += 1
    
    def record_cache_miss(self, component: str) -> None:
        """Record a cache miss for the specified component."""
        with self._lock:
            self._cache_stats[component]['misses'] += 1
    
    def get_cache_hit_ratio(self, component: str) -> float:
        """Calculate cache hit ratio for a component."""
        with self._lock:
            stats = self._cache_stats[component]
            total = stats['hits'] + stats['misses']
            return stats['hits'] / total if total > 0 else 0.0
    
    def generate_report(
        self, 
        component: str, 
        time_period: timedelta = timedelta(hours=1),
        thresholds: Optional[PerformanceThresholds] = None
    ) -> PerformanceReport:
        """Generate a comprehensive performance report."""
        if thresholds is None:
            thresholds = PerformanceThresholds()
            
        cutoff_time = datetime.now() - time_period
        
        with self._lock:
            # Filter snapshots for the component and time period
            relevant_snapshots = [
                s for s in self._snapshots 
                if s.component == component and s.timestamp >= cutoff_time
            ]
        
        if not relevant_snapshots:
            return PerformanceReport(
                component=component,
                time_period=time_period,
                total_operations=0,
                success_rate=0.0,
                error_rate=0.0,
                latency_p50_ms=0.0,
                latency_p95_ms=0.0,
                latency_p99_ms=0.0,
                avg_memory_mb=0.0,
                peak_memory_mb=0.0,
                avg_cpu_percent=0.0,
                peak_cpu_percent=0.0,
                performance_grade="F"
            )
        
        # Calculate metrics
        total_operations = len(relevant_snapshots)
        successful_ops = sum(1 for s in relevant_snapshots if s.success)
        success_rate = successful_ops / total_operations
        error_rate = 1.0 - success_rate
        
        latencies = [s.latency_ms for s in relevant_snapshots]
        memory_usage = [s.memory_mb for s in relevant_snapshots]
        cpu_usage = [s.cpu_percent for s in relevant_snapshots]
        
        latency_p50 = statistics.median(latencies) if latencies else 0.0
        latency_p95 = self._percentile(latencies, 0.95) if latencies else 0.0
        latency_p99 = self._percentile(latencies, 0.99) if latencies else 0.0
        
        avg_memory = statistics.mean(memory_usage) if memory_usage else 0.0
        peak_memory = max(memory_usage) if memory_usage else 0.0
        
        avg_cpu = statistics.mean(cpu_usage) if cpu_usage else 0.0
        peak_cpu = max(cpu_usage) if cpu_usage else 0.0
        
        cache_hit_ratio = self.get_cache_hit_ratio(component)
        
        # Check threshold violations
        violations = []
        if latency_p95 > thresholds.search_latency_p95_ms:
            violations.append(f"Latency P95 ({latency_p95:.1f}ms) exceeds threshold ({thresholds.search_latency_p95_ms}ms)")
        if peak_memory > thresholds.memory_usage_max_mb:
            violations.append(f"Memory usage ({peak_memory:.1f}MB) exceeds threshold ({thresholds.memory_usage_max_mb}MB)")
        if cache_hit_ratio < thresholds.cache_hit_ratio_min:
            violations.append(f"Cache hit ratio ({cache_hit_ratio:.2f}) below threshold ({thresholds.cache_hit_ratio_min})")
        if error_rate * 100 > thresholds.error_rate_max_percent:
            violations.append(f"Error rate ({error_rate*100:.1f}%) exceeds threshold ({thresholds.error_rate_max_percent}%)")
        
        # Calculate performance grade
        grade = self._calculate_performance_grade(violations, success_rate, latency_p95, thresholds)
        
        return PerformanceReport(
            component=component,
            time_period=time_period,
            total_operations=total_operations,
            success_rate=success_rate,
            error_rate=error_rate,
            latency_p50_ms=latency_p50,
            latency_p95_ms=latency_p95,
            latency_p99_ms=latency_p99,
            avg_memory_mb=avg_memory,
            peak_memory_mb=peak_memory,
            avg_cpu_percent=avg_cpu,
            peak_cpu_percent=peak_cpu,
            cache_hit_ratio=cache_hit_ratio,
            threshold_violations=violations,
            performance_grade=grade
        )
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate the specified percentile of the data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(percentile * len(sorted_data))
        if index >= len(sorted_data):
            return sorted_data[-1]
        return sorted_data[index]
    
    def _calculate_performance_grade(
        self, 
        violations: List[str], 
        success_rate: float, 
        latency_p95: float,
        thresholds: PerformanceThresholds
    ) -> str:
        """Calculate performance grade based on metrics."""
        if len(violations) == 0 and success_rate > 0.99:
            return "A"
        elif len(violations) <= 1 and success_rate > 0.95:
            return "B" 
        elif len(violations) <= 2 and success_rate > 0.90:
            return "C"
        elif success_rate > 0.80:
            return "D"
        else:
            return "F"


class PerformanceMonitor:
    """Context manager for monitoring performance of operations."""
    
    def __init__(self, collector: MetricsCollector, component: str, operation: str):
        self.collector = collector
        self.component = component
        self.operation = operation
        self.start_time: Optional[float] = None
        self.start_memory: Optional[float] = None
        self.process = psutil.Process()
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        self.start_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        end_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
        cpu_percent = self.process.cpu_percent()
        
        latency_ms = (end_time - self.start_time) * 1000
        memory_mb = max(self.start_memory, end_memory)
        success = exc_type is None
        
        metadata = {}
        if exc_type:
            metadata['error_type'] = exc_type.__name__
            metadata['error_message'] = str(exc_val)
        
        snapshot = MetricSnapshot(
            timestamp=datetime.now(),
            component=self.component,
            operation=self.operation,
            latency_ms=latency_ms,
            memory_mb=memory_mb,
            cpu_percent=cpu_percent,
            success=success,
            metadata=metadata
        )
        
        self.collector.record_metric(snapshot)


class PerformanceRegistry:
    """Central registry for performance monitoring across all components."""
    
    _instance: Optional['PerformanceRegistry'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.collector = MetricsCollector()
        self.thresholds = PerformanceThresholds()
        self._alert_handlers: List[Callable[[PerformanceReport], None]] = []
        self._initialized = True
        
        logger.info("Performance monitoring system initialized")
    
    def monitor(self, component: str, operation: str = "default") -> PerformanceMonitor:
        """Create a performance monitor for the specified component and operation."""
        return PerformanceMonitor(self.collector, component, operation)
    
    def record_cache_hit(self, component: str) -> None:
        """Record a cache hit."""
        self.collector.record_cache_hit(component)
    
    def record_cache_miss(self, component: str) -> None:
        """Record a cache miss."""
        self.collector.record_cache_miss(component)
    
    def get_performance_report(self, component: str, time_period: timedelta = timedelta(hours=1)) -> PerformanceReport:
        """Get a performance report for the specified component."""
        return self.collector.generate_report(component, time_period, self.thresholds)
    
    def add_alert_handler(self, handler: Callable[[PerformanceReport], None]) -> None:
        """Add an alert handler for performance violations."""
        self._alert_handlers.append(handler)
    
    def check_performance_alerts(self) -> None:
        """Check all components for performance violations and trigger alerts."""
        components = ["TechnicalQualityGate", "OptimizedASRScriptureMatcher", "RobustWisdomLibraryIntegrator"]
        
        for component in components:
            report = self.get_performance_report(component)
            
            if report.threshold_violations or report.performance_grade in ['D', 'F']:
                logger.warning(f"Performance issues detected in {component}: Grade {report.performance_grade}")
                for handler in self._alert_handlers:
                    try:
                        handler(report)
                    except Exception as e:
                        logger.error(f"Alert handler failed: {e}")
    
    def export_metrics(self, output_path: Path, time_period: timedelta = timedelta(hours=24)) -> None:
        """Export performance metrics to a JSON file."""
        components = ["TechnicalQualityGate", "OptimizedASRScriptureMatcher", "RobustWisdomLibraryIntegrator"]
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'time_period_hours': time_period.total_seconds() / 3600,
            'thresholds': {
                'search_latency_p95_ms': self.thresholds.search_latency_p95_ms,
                'memory_usage_max_mb': self.thresholds.memory_usage_max_mb,
                'cache_hit_ratio_min': self.thresholds.cache_hit_ratio_min,
                'error_rate_max_percent': self.thresholds.error_rate_max_percent
            },
            'reports': {}
        }
        
        for component in components:
            report = self.get_performance_report(component, time_period)
            export_data['reports'][component] = {
                'total_operations': report.total_operations,
                'success_rate': report.success_rate,
                'error_rate': report.error_rate,
                'latency_p95_ms': report.latency_p95_ms,
                'peak_memory_mb': report.peak_memory_mb,
                'cache_hit_ratio': report.cache_hit_ratio,
                'performance_grade': report.performance_grade,
                'threshold_violations': report.threshold_violations
            }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Performance metrics exported to {output_path}")


# Global performance registry instance
performance_registry = PerformanceRegistry()


# Convenience functions for easy integration
def monitor_performance(component: str, operation: str = "default"):
    """Decorator for monitoring function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with performance_registry.monitor(component, operation):
                return func(*args, **kwargs)
        return wrapper
    return decorator


@contextmanager
def performance_context(component: str, operation: str = "default"):
    """Context manager for monitoring code block performance."""
    with performance_registry.monitor(component, operation):
        yield


def record_cache_event(component: str, hit: bool) -> None:
    """Record a cache hit or miss event."""
    if hit:
        performance_registry.record_cache_hit(component)
    else:
        performance_registry.record_cache_miss(component)