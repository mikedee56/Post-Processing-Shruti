"""
Semantic Performance Monitor - Story 3.4 Performance Optimization and Monitoring

This module provides comprehensive performance monitoring for semantic processing
components, ensuring compliance with Story 3.4 acceptance criteria:
- <5% overhead for semantic processing
- >95% cache hit ratio for semantic embeddings
- <50ms quality gate evaluation time
- Bounded and predictable memory usage
- Integration with existing performance monitoring
- Graceful degradation monitoring

Author: Development Team
Date: 2025-01-30
Epic: 3 - Semantic Refinement & QA Framework
Story: 3.4 - Performance Optimization and Monitoring
"""

import time
import psutil
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

from ..utils.logger_config import get_logger
from .performance_metrics_collector import PerformanceMetricsCollector, MetricType, MetricUnit
from ..utils.config_manager import ConfigManager


@dataclass
class SemanticPerformanceThresholds:
    """Performance thresholds for Story 3.4 compliance."""
    max_overhead_percentage: float = 5.0
    min_cache_hit_ratio: float = 0.95
    max_quality_gate_time_ms: float = 50.0
    max_memory_increase_mb: float = 512.0
    max_processing_time_per_segment_ms: float = 100.0


@dataclass
class SemanticPerformanceMetrics:
    """Comprehensive semantic performance metrics."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Processing Performance
    processing_time_ms: float = 0.0
    baseline_time_ms: float = 0.0
    overhead_percentage: float = 0.0
    throughput_segments_per_second: float = 0.0
    
    # Cache Performance
    cache_hit_ratio: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_size_mb: float = 0.0
    
    # Quality Gate Performance
    quality_gate_evaluations: int = 0
    avg_quality_gate_time_ms: float = 0.0
    quality_gate_failures: int = 0
    
    # Memory Performance
    memory_usage_mb: float = 0.0
    memory_increase_mb: float = 0.0
    peak_memory_mb: float = 0.0
    
    # System Health
    cpu_usage_percentage: float = 0.0
    system_load_average: float = 0.0
    
    # Compliance Status
    meets_overhead_requirement: bool = True
    meets_cache_requirement: bool = True
    meets_quality_gate_requirement: bool = True
    meets_memory_requirement: bool = True


class SemanticPerformanceMonitor:
    """
    Advanced performance monitor for semantic processing components.
    
    This monitor ensures Story 3.4 acceptance criteria are met and provides
    real-time performance tracking with alerting capabilities.
    """
    
    def __init__(
        self,
        thresholds: Optional[SemanticPerformanceThresholds] = None,
        metrics_collector: Optional[PerformanceMetricsCollector] = None,
        config_manager: Optional[ConfigManager] = None
    ):
        """
        Initialize semantic performance monitor.
        
        Args:
            thresholds: Performance thresholds for compliance checking
            metrics_collector: Existing metrics collector integration
            config_manager: Configuration manager for settings
        """
        self.logger = get_logger(__name__)
        self.thresholds = thresholds or SemanticPerformanceThresholds()
        self.metrics_collector = metrics_collector or PerformanceMetricsCollector()
        self.config_manager = config_manager or ConfigManager()
        
        # Performance tracking
        self.baseline_performance = {}
        self.current_metrics = SemanticPerformanceMetrics()
        self.metrics_history = []
        self.max_history_size = 1000
        
        # System monitoring
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Performance alerts
        self.alert_callbacks = []
        self.violation_counts = {
            'overhead': 0,
            'cache_hit_ratio': 0,
            'quality_gate_time': 0,
            'memory_usage': 0
        }
        
        self.logger.info("Semantic performance monitor initialized for Story 3.4 compliance")
    
    def set_baseline_performance(self, component: str, baseline_time_ms: float) -> None:
        """
        Set performance baseline for overhead calculation.
        
        Args:
            component: Component name (e.g., 'semantic_processing', 'quality_gates')
            baseline_time_ms: Baseline processing time in milliseconds
        """
        self.baseline_performance[component] = baseline_time_ms
        self.logger.info(f"Performance baseline set for {component}: {baseline_time_ms:.2f}ms")
    
    def start_monitoring(self, monitoring_interval_seconds: float = 1.0) -> None:
        """
        Start continuous performance monitoring.
        
        Args:
            monitoring_interval_seconds: Interval between monitoring samples
        """
        if self.monitoring_active:
            self.logger.warning("Performance monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(monitoring_interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("Started continuous performance monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop continuous performance monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        self.logger.info("Stopped performance monitoring")
    
    def record_processing_performance(
        self,
        component: str,
        processing_time_ms: float,
        segments_processed: int = 1,
        cache_hits: int = 0,
        cache_misses: int = 0
    ) -> SemanticPerformanceMetrics:
        """
        Record processing performance for a semantic component.
        
        Args:
            component: Component name
            processing_time_ms: Processing time in milliseconds
            segments_processed: Number of segments processed
            cache_hits: Number of cache hits
            cache_misses: Number of cache misses
            
        Returns:
            Current performance metrics
        """
        # Calculate overhead
        baseline_time = self.baseline_performance.get(component, processing_time_ms)
        overhead_percentage = ((processing_time_ms - baseline_time) / baseline_time) * 100 if baseline_time > 0 else 0.0
        
        # Calculate cache hit ratio
        total_cache_accesses = cache_hits + cache_misses
        cache_hit_ratio = cache_hits / total_cache_accesses if total_cache_accesses > 0 else 0.0
        
        # Calculate throughput
        throughput = (segments_processed / processing_time_ms) * 1000 if processing_time_ms > 0 else 0.0
        
        # Update current metrics
        self.current_metrics.processing_time_ms = processing_time_ms
        self.current_metrics.baseline_time_ms = baseline_time
        self.current_metrics.overhead_percentage = overhead_percentage
        self.current_metrics.throughput_segments_per_second = throughput
        self.current_metrics.cache_hit_ratio = cache_hit_ratio
        self.current_metrics.cache_hits = cache_hits
        self.current_metrics.cache_misses = cache_misses
        
        # System metrics
        self._update_system_metrics()
        
        # Compliance checking
        self._check_compliance()
        
        # Record to metrics collector
        self._record_to_metrics_collector(component)
        
        # Add to history
        self._add_to_history()
        
        return self.current_metrics
    
    def record_quality_gate_performance(
        self,
        evaluations: int,
        total_time_ms: float,
        failures: int = 0
    ) -> None:
        """
        Record quality gate performance metrics.
        
        Args:
            evaluations: Number of quality gate evaluations
            total_time_ms: Total time for all evaluations
            failures: Number of failed evaluations
        """
        avg_time_ms = total_time_ms / evaluations if evaluations > 0 else 0.0
        
        self.current_metrics.quality_gate_evaluations = evaluations
        self.current_metrics.avg_quality_gate_time_ms = avg_time_ms
        self.current_metrics.quality_gate_failures = failures
        
        # Check quality gate compliance
        self.current_metrics.meets_quality_gate_requirement = (
            avg_time_ms <= self.thresholds.max_quality_gate_time_ms
        )
        
        if not self.current_metrics.meets_quality_gate_requirement:
            self.violation_counts['quality_gate_time'] += 1
            self._trigger_performance_alert(
                'quality_gate_performance',
                f"Quality gate time {avg_time_ms:.1f}ms exceeds {self.thresholds.max_quality_gate_time_ms}ms threshold"
            )
    
    def record_cache_performance(
        self,
        cache_size_mb: float,
        hit_ratio: float,
        hits: int,
        misses: int
    ) -> None:
        """
        Record cache performance metrics.
        
        Args:
            cache_size_mb: Cache size in megabytes
            hit_ratio: Cache hit ratio (0.0 to 1.0)
            hits: Number of cache hits
            misses: Number of cache misses
        """
        self.current_metrics.cache_size_mb = cache_size_mb
        self.current_metrics.cache_hit_ratio = hit_ratio
        self.current_metrics.cache_hits = hits
        self.current_metrics.cache_misses = misses
        
        # Check cache compliance
        self.current_metrics.meets_cache_requirement = (
            hit_ratio >= self.thresholds.min_cache_hit_ratio
        )
        
        if not self.current_metrics.meets_cache_requirement:
            self.violation_counts['cache_hit_ratio'] += 1
            self._trigger_performance_alert(
                'cache_performance',
                f"Cache hit ratio {hit_ratio:.1%} below {self.thresholds.min_cache_hit_ratio:.1%} threshold"
            )
    
    def _update_system_metrics(self) -> None:
        """Update system-level metrics."""
        try:
            # Memory metrics
            current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            self.current_metrics.memory_usage_mb = current_memory
            self.current_metrics.memory_increase_mb = current_memory - self.initial_memory
            
            # Update peak memory
            if current_memory > self.current_metrics.peak_memory_mb:
                self.current_metrics.peak_memory_mb = current_memory
            
            # CPU and system load
            self.current_metrics.cpu_usage_percentage = self.process.cpu_percent()
            self.current_metrics.system_load_average = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
            
        except Exception as e:
            self.logger.debug(f"Error updating system metrics: {e}")
    
    def _check_compliance(self) -> None:
        """Check compliance against Story 3.4 requirements."""
        metrics = self.current_metrics
        
        # Overhead compliance
        metrics.meets_overhead_requirement = (
            metrics.overhead_percentage <= self.thresholds.max_overhead_percentage
        )
        
        # Memory compliance
        metrics.meets_memory_requirement = (
            metrics.memory_increase_mb <= self.thresholds.max_memory_increase_mb
        )
        
        # Trigger alerts for violations
        if not metrics.meets_overhead_requirement:
            self.violation_counts['overhead'] += 1
            self._trigger_performance_alert(
                'overhead_violation',
                f"Processing overhead {metrics.overhead_percentage:.2f}% exceeds {self.thresholds.max_overhead_percentage}% threshold"
            )
        
        if not metrics.meets_memory_requirement:
            self.violation_counts['memory_usage'] += 1
            self._trigger_performance_alert(
                'memory_violation',
                f"Memory increase {metrics.memory_increase_mb:.1f}MB exceeds {self.thresholds.max_memory_increase_mb}MB threshold"
            )
    
    def _record_to_metrics_collector(self, component: str) -> None:
        """Record metrics to the existing metrics collector."""
        try:
            metrics = self.current_metrics
            
            # Processing metrics
            self.metrics_collector.record_metric(
                f"semantic.{component}.processing_time_ms",
                metrics.processing_time_ms,
                MetricType.GAUGE,
                MetricUnit.MILLISECONDS
            )
            
            self.metrics_collector.record_metric(
                f"semantic.{component}.overhead_percentage",
                metrics.overhead_percentage,
                MetricType.GAUGE,
                MetricUnit.PERCENTAGE
            )
            
            # Cache metrics
            self.metrics_collector.record_metric(
                f"semantic.{component}.cache_hit_ratio",
                metrics.cache_hit_ratio,
                MetricType.GAUGE,
                MetricUnit.RATIO
            )
            
            # Memory metrics
            self.metrics_collector.record_metric(
                f"semantic.{component}.memory_usage_mb",
                metrics.memory_usage_mb,
                MetricType.GAUGE,
                MetricUnit.BYTES
            )
            
            # Quality gate metrics
            if metrics.quality_gate_evaluations > 0:
                self.metrics_collector.record_metric(
                    f"semantic.{component}.quality_gate_time_ms",
                    metrics.avg_quality_gate_time_ms,
                    MetricType.GAUGE,
                    MetricUnit.MILLISECONDS
                )
            
        except Exception as e:
            self.logger.error(f"Error recording to metrics collector: {e}")
    
    def _add_to_history(self) -> None:
        """Add current metrics to history."""
        # Create a copy of current metrics
        metrics_copy = SemanticPerformanceMetrics(
            timestamp=datetime.now(timezone.utc),
            processing_time_ms=self.current_metrics.processing_time_ms,
            baseline_time_ms=self.current_metrics.baseline_time_ms,
            overhead_percentage=self.current_metrics.overhead_percentage,
            throughput_segments_per_second=self.current_metrics.throughput_segments_per_second,
            cache_hit_ratio=self.current_metrics.cache_hit_ratio,
            cache_hits=self.current_metrics.cache_hits,
            cache_misses=self.current_metrics.cache_misses,
            cache_size_mb=self.current_metrics.cache_size_mb,
            quality_gate_evaluations=self.current_metrics.quality_gate_evaluations,
            avg_quality_gate_time_ms=self.current_metrics.avg_quality_gate_time_ms,
            quality_gate_failures=self.current_metrics.quality_gate_failures,
            memory_usage_mb=self.current_metrics.memory_usage_mb,
            memory_increase_mb=self.current_metrics.memory_increase_mb,
            peak_memory_mb=self.current_metrics.peak_memory_mb,
            cpu_usage_percentage=self.current_metrics.cpu_usage_percentage,
            system_load_average=self.current_metrics.system_load_average,
            meets_overhead_requirement=self.current_metrics.meets_overhead_requirement,
            meets_cache_requirement=self.current_metrics.meets_cache_requirement,
            meets_quality_gate_requirement=self.current_metrics.meets_quality_gate_requirement,
            meets_memory_requirement=self.current_metrics.meets_memory_requirement
        )
        
        self.metrics_history.append(metrics_copy)
        
        # Limit history size
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]
    
    def _monitoring_loop(self, interval_seconds: float) -> None:
        """Continuous monitoring loop."""
        while self.monitoring_active:
            try:
                self._update_system_metrics()
                self._check_compliance()
                self._add_to_history()
                time.sleep(interval_seconds)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval_seconds)
    
    def add_performance_alert_callback(self, callback: Callable[[str, str], None]) -> None:
        """
        Add callback for performance alerts.
        
        Args:
            callback: Callback function that takes (alert_type, message)
        """
        self.alert_callbacks.append(callback)
    
    def _trigger_performance_alert(self, alert_type: str, message: str) -> None:
        """Trigger performance alert to all registered callbacks."""
        self.logger.warning(f"Performance Alert [{alert_type}]: {message}")
        
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, message)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        metrics = self.current_metrics
        
        # Calculate averages from history
        if self.metrics_history:
            recent_metrics = self.metrics_history[-10:]  # Last 10 samples
            avg_overhead = sum(m.overhead_percentage for m in recent_metrics) / len(recent_metrics)
            avg_cache_hit_ratio = sum(m.cache_hit_ratio for m in recent_metrics) / len(recent_metrics)
            avg_quality_gate_time = sum(m.avg_quality_gate_time_ms for m in recent_metrics) / len(recent_metrics)
        else:
            avg_overhead = metrics.overhead_percentage
            avg_cache_hit_ratio = metrics.cache_hit_ratio
            avg_quality_gate_time = metrics.avg_quality_gate_time_ms
        
        return {
            'story_3_4_compliance': {
                'overhead_requirement': {
                    'threshold': f"{self.thresholds.max_overhead_percentage}%",
                    'current': f"{avg_overhead:.2f}%",
                    'compliant': avg_overhead <= self.thresholds.max_overhead_percentage,
                    'violations': self.violation_counts['overhead']
                },
                'cache_hit_ratio_requirement': {
                    'threshold': f"{self.thresholds.min_cache_hit_ratio:.1%}",
                    'current': f"{avg_cache_hit_ratio:.1%}",
                    'compliant': avg_cache_hit_ratio >= self.thresholds.min_cache_hit_ratio,
                    'violations': self.violation_counts['cache_hit_ratio']
                },
                'quality_gate_time_requirement': {
                    'threshold': f"{self.thresholds.max_quality_gate_time_ms}ms",
                    'current': f"{avg_quality_gate_time:.1f}ms",
                    'compliant': avg_quality_gate_time <= self.thresholds.max_quality_gate_time_ms,
                    'violations': self.violation_counts['quality_gate_time']
                },
                'memory_usage_requirement': {
                    'threshold': f"{self.thresholds.max_memory_increase_mb}MB",
                    'current': f"{metrics.memory_increase_mb:.1f}MB",
                    'compliant': metrics.memory_increase_mb <= self.thresholds.max_memory_increase_mb,
                    'violations': self.violation_counts['memory_usage']
                }
            },
            'current_performance': {
                'processing_time_ms': metrics.processing_time_ms,
                'throughput_segments_per_second': metrics.throughput_segments_per_second,
                'memory_usage_mb': metrics.memory_usage_mb,
                'cpu_usage_percentage': metrics.cpu_usage_percentage
            },
            'monitoring_status': {
                'active': self.monitoring_active,
                'samples_collected': len(self.metrics_history),
                'alert_callbacks_registered': len(self.alert_callbacks)
            }
        }
    
    def export_performance_report(self, output_file: Path) -> None:
        """
        Export comprehensive performance report.
        
        Args:
            output_file: Path to output report file
        """
        try:
            report = {
                'report_metadata': {
                    'generated_at': datetime.now(timezone.utc).isoformat(),
                    'story': '3.4 - Performance Optimization and Monitoring',
                    'monitoring_duration_hours': len(self.metrics_history) * 1.0 / 3600,  # Assuming 1s intervals
                    'total_samples': len(self.metrics_history)
                },
                'performance_summary': self.get_performance_summary(),
                'detailed_metrics': {
                    'current': {
                        'timestamp': self.current_metrics.timestamp.isoformat(),
                        'processing_time_ms': self.current_metrics.processing_time_ms,
                        'overhead_percentage': self.current_metrics.overhead_percentage,
                        'cache_hit_ratio': self.current_metrics.cache_hit_ratio,
                        'quality_gate_time_ms': self.current_metrics.avg_quality_gate_time_ms,
                        'memory_usage_mb': self.current_metrics.memory_usage_mb
                    }
                },
                'violation_summary': self.violation_counts,
                'baseline_performance': self.baseline_performance
            }
            
            # Add recent metrics history
            if len(self.metrics_history) > 0:
                report['recent_history'] = [
                    {
                        'timestamp': m.timestamp.isoformat(),
                        'overhead_percentage': m.overhead_percentage,
                        'cache_hit_ratio': m.cache_hit_ratio,
                        'memory_usage_mb': m.memory_usage_mb
                    }
                    for m in self.metrics_history[-50:]  # Last 50 samples
                ]
            
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Performance report exported to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error exporting performance report: {e}")
            raise
    
    def reset_metrics(self) -> None:
        """Reset performance metrics and history."""
        self.current_metrics = SemanticPerformanceMetrics()
        self.metrics_history = []
        self.violation_counts = {
            'overhead': 0,
            'cache_hit_ratio': 0,
            'quality_gate_time': 0,
            'memory_usage': 0
        }
        self.logger.info("Performance metrics reset")


# Global instance for easy access
_global_semantic_monitor = None


def initialize_semantic_monitoring(
    thresholds: Optional[SemanticPerformanceThresholds] = None,
    metrics_collector: Optional[PerformanceMetricsCollector] = None,
    config_manager: Optional[ConfigManager] = None
) -> SemanticPerformanceMonitor:
    """
    Initialize global semantic performance monitor.
    
    Args:
        thresholds: Performance thresholds
        metrics_collector: Existing metrics collector
        config_manager: Configuration manager
        
    Returns:
        Initialized performance monitor
    """
    global _global_semantic_monitor
    _global_semantic_monitor = SemanticPerformanceMonitor(
        thresholds=thresholds,
        metrics_collector=metrics_collector,
        config_manager=config_manager
    )
    return _global_semantic_monitor


def get_semantic_monitor() -> Optional[SemanticPerformanceMonitor]:
    """Get global semantic performance monitor."""
    return _global_semantic_monitor


def record_semantic_performance(
    component: str,
    processing_time_ms: float,
    segments_processed: int = 1,
    cache_hits: int = 0,
    cache_misses: int = 0
) -> Optional[SemanticPerformanceMetrics]:
    """
    Record semantic performance metrics using global monitor.
    
    Args:
        component: Component name
        processing_time_ms: Processing time in milliseconds
        segments_processed: Number of segments processed
        cache_hits: Number of cache hits
        cache_misses: Number of cache misses
        
    Returns:
        Current performance metrics if monitor is available
    """
    if _global_semantic_monitor:
        return _global_semantic_monitor.record_processing_performance(
            component, processing_time_ms, segments_processed, cache_hits, cache_misses
        )
    return None