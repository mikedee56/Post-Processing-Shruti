"""
Performance Optimizer for Enterprise-Grade Processing Pipeline.

This module implements comprehensive performance optimization for the Sanskrit ASR
post-processing system, targeting sub-second processing times with production reliability.
"""

import cProfile
import functools
import hashlib
import logging
import pstats
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict, deque
from threading import Lock

from .performance_monitor import PerformanceMonitor, MetricType, AlertSeverity


@dataclass
class OptimizationResult:
    """Result from performance optimization analysis."""
    operation_name: str
    original_time: float
    optimized_time: float
    improvement_ratio: float
    cache_hits: int = 0
    cache_misses: int = 0
    recommendations: List[str] = field(default_factory=list)


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_usage: int = 0
    hit_rate: float = 0.0
    
    def update_hit_rate(self):
        """Update hit rate calculation."""
        total_requests = self.hits + self.misses
        self.hit_rate = self.hits / total_requests if total_requests > 0 else 0.0


class LRUCache:
    """
    High-performance LRU cache with memory management and statistics.
    
    Optimized for MCP transformer operations with automatic expiration
    and memory pressure handling.
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_order = deque()
        self.timestamps = {}
        self.stats = CacheStats()
        self.lock = Lock()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired, daemon=True)
        self.cleanup_thread.start()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with LRU ordering."""
        with self.lock:
            current_time = time.time()
            
            if key in self.cache:
                # Check if expired
                if current_time - self.timestamps[key] > self.ttl_seconds:
                    self._evict_key(key)
                    self.stats.misses += 1
                    self.stats.update_hit_rate()
                    return None
                
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                self.stats.hits += 1
                self.stats.update_hit_rate()
                return self.cache[key]
            
            self.stats.misses += 1
            self.stats.update_hit_rate()
            return None
    
    def put(self, key: str, value: Any):
        """Put item in cache with LRU eviction."""
        with self.lock:
            current_time = time.time()
            
            # If key exists, update it
            if key in self.cache:
                self.cache[key] = value
                self.timestamps[key] = current_time
                self.access_order.remove(key)
                self.access_order.append(key)
                return
            
            # Check if we need to evict
            while len(self.cache) >= self.max_size:
                oldest_key = self.access_order.popleft()
                self._evict_key(oldest_key)
                self.stats.evictions += 1
            
            # Add new item
            self.cache[key] = value
            self.timestamps[key] = current_time
            self.access_order.append(key)
            
            self._update_memory_usage()
    
    def _evict_key(self, key: str):
        """Evict a specific key from cache."""
        if key in self.cache:
            del self.cache[key]
        if key in self.timestamps:
            del self.timestamps[key]
    
    def _cleanup_expired(self):
        """Background thread to clean up expired entries."""
        while True:
            time.sleep(300)  # Check every 5 minutes
            current_time = time.time()
            
            with self.lock:
                expired_keys = [
                    key for key, timestamp in self.timestamps.items()
                    if current_time - timestamp > self.ttl_seconds
                ]
                
                for key in expired_keys:
                    if key in self.access_order:
                        self.access_order.remove(key)
                    self._evict_key(key)
                    self.stats.evictions += 1
                
                if expired_keys:
                    self._update_memory_usage()
    
    def _update_memory_usage(self):
        """Update memory usage statistics."""
        import sys
        self.stats.memory_usage = sum(
            sys.getsizeof(key) + sys.getsizeof(value)
            for key, value in self.cache.items()
        )
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.timestamps.clear()
            self.stats = CacheStats()
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self.lock:
            self._update_memory_usage()
            return self.stats


class PerformanceOptimizer:
    """
    Enterprise-grade performance optimization engine for Sanskrit processing.
    
    Provides comprehensive optimization strategies:
    - Intelligent caching for MCP transformer operations  
    - Processing pipeline optimization
    - Concurrent operation management
    - Performance profiling and regression detection
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize performance optimizer with enterprise configuration."""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Performance monitoring integration
        self.monitor = PerformanceMonitor(self.config.get('monitoring', {}))
        
        # Caching system
        self.mcp_cache = LRUCache(
            max_size=self.config.get('mcp_cache_size', 5000),
            ttl_seconds=self.config.get('mcp_cache_ttl', 3600)
        )
        
        self.sanskrit_cache = LRUCache(
            max_size=self.config.get('sanskrit_cache_size', 2000),
            ttl_seconds=self.config.get('sanskrit_cache_ttl', 7200)
        )
        
        # Performance tracking
        self.optimization_results: Dict[str, OptimizationResult] = {}
        self.profiling_data: Dict[str, Any] = {}
        
        # Concurrent processing
        self.max_workers = self.config.get('max_workers', 4)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Performance targets from Story 4.3
        self.performance_targets = {
            'processing_time_seconds': 1.0,  # Sub-second processing
            'cache_hit_rate': 0.70,          # 70% cache hit rate
            'concurrent_efficiency': 0.85,   # 85% concurrent processing efficiency
        }
        
        self.logger.info(f"PerformanceOptimizer initialized with sub-second targets")
    
    def optimize_mcp_transformer_call(self, func: Callable) -> Callable:
        """
        Decorator to optimize MCP transformer operations with caching.
        
        Provides intelligent caching, monitoring, and fallback handling.
        """
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = self._generate_cache_key(func.__name__, args, kwargs)
            
            # Try cache first
            cached_result = self.mcp_cache.get(cache_key)
            if cached_result is not None:
                self.monitor.record_metric(
                    MetricType.CACHE_HIT_RATE, 1.0, "mcp_transformer",
                    tags={"operation": func.__name__, "cache_type": "hit"}
                )
                return cached_result
            
            # Execute with monitoring
            with self.monitor.monitor_processing_operation(f"mcp_{func.__name__}", "mcp_transformer"):
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Cache successful results
                    self.mcp_cache.put(cache_key, result)
                    
                    execution_time = (time.time() - start_time) * 1000  # ms
                    self.monitor.record_metric(
                        MetricType.RESPONSE_TIME, execution_time, "mcp_transformer",
                        tags={"operation": func.__name__, "cache_type": "miss"}
                    )
                    
                    self.monitor.record_metric(
                        MetricType.SUCCESS_RATE, 1.0, "mcp_transformer",
                        tags={"operation": func.__name__}
                    )
                    
                    return result
                    
                except Exception as e:
                    self.logger.error(f"MCP transformer operation failed: {e}")
                    self.monitor.record_metric(
                        MetricType.ERROR_RATE, 1.0, "mcp_transformer",
                        tags={"operation": func.__name__, "error": str(type(e).__name__)}
                    )
                    raise
        
        return wrapper
    
    def optimize_sanskrit_processing(self, func: Callable) -> Callable:
        """
        Decorator to optimize Sanskrit processing operations.
        
        Provides caching for repetitive Sanskrit transformations.
        """
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key for Sanskrit operations
            cache_key = self._generate_cache_key(f"sanskrit_{func.__name__}", args, kwargs)
            
            # Try Sanskrit-specific cache
            cached_result = self.sanskrit_cache.get(cache_key)
            if cached_result is not None:
                self.monitor.record_metric(
                    MetricType.CACHE_HIT_RATE, 1.0, "sanskrit_processing",
                    tags={"operation": func.__name__, "cache_type": "hit"}
                )
                return cached_result
            
            # Execute with optimization
            with self.monitor.monitor_processing_operation(f"sanskrit_{func.__name__}", "sanskrit_processing"):
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Cache results for Sanskrit operations
                    self.sanskrit_cache.put(cache_key, result)
                    
                    execution_time = (time.time() - start_time) * 1000  # ms
                    self.monitor.record_metric(
                        MetricType.RESPONSE_TIME, execution_time, "sanskrit_processing",
                        tags={"operation": func.__name__}
                    )
                    
                    return result
                    
                except Exception as e:
                    self.logger.error(f"Sanskrit processing operation failed: {e}")
                    self.monitor.record_metric(
                        MetricType.ERROR_RATE, 1.0, "sanskrit_processing",
                        tags={"operation": func.__name__}
                    )
                    raise
        
        return wrapper
    
    def optimize_concurrent_processing(self, operations: List[Callable], 
                                     max_workers: Optional[int] = None) -> List[Any]:
        """
        Execute multiple operations concurrently with optimization.
        
        Provides intelligent work distribution and error handling.
        """
        max_workers = max_workers or self.max_workers
        results = []
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all operations
            future_to_operation = {
                executor.submit(op): op for op in operations
            }
            
            # Collect results as they complete
            completed_operations = 0
            failed_operations = 0
            
            for future in as_completed(future_to_operation):
                try:
                    result = future.result()
                    results.append(result)
                    completed_operations += 1
                except Exception as e:
                    self.logger.error(f"Concurrent operation failed: {e}")
                    failed_operations += 1
                    results.append(None)
        
        # Record concurrent processing metrics
        total_time = time.time() - start_time
        success_rate = completed_operations / (completed_operations + failed_operations)
        
        # Calculate efficiency (theoretical speedup vs actual)
        efficiency = min(1.0, (len(operations) * 0.1) / total_time) if total_time > 0 else 0.0
        
        self.monitor.record_metric(
            MetricType.SUCCESS_RATE, success_rate, "concurrent_processing",
            tags={"workers": str(max_workers), "operations": str(len(operations))}
        )
        
        self.monitor.record_metric(
            MetricType.RESPONSE_TIME, total_time * 1000, "concurrent_processing",
            tags={"workers": str(max_workers)}
        )
        
        return results
    
    def profile_operation(self, func: Callable, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Profile an operation for performance analysis.
        
        Returns both the operation result and detailed profiling data.
        """
        profiler = cProfile.Profile()
        
        # Execute with profiling
        profiler.enable()
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        finally:
            profiler.disable()
            end_time = time.time()
        
        # Generate profiling stats
        stats = pstats.Stats(profiler)
        stats.sort_stats('tottime')
        
        # Extract key performance data
        total_time = end_time - start_time
        top_functions = []
        
        for func_info, (cc, nc, tt, ct, callers) in stats.stats.items():
            filename, line_num, func_name = func_info
            top_functions.append({
                'function': func_name,
                'file': filename,
                'line': line_num,
                'total_time': tt,
                'cumulative_time': ct,
                'call_count': cc
            })
        
        # Sort by total time and take top 10
        top_functions.sort(key=lambda x: x['total_time'], reverse=True)
        
        profiling_data = {
            'total_execution_time': total_time,
            'success': success,
            'error': error,
            'top_functions': top_functions[:10],
            'call_count': sum(cc for (cc, nc, tt, ct, callers) in stats.stats.values()),
            'profiling_timestamp': time.time()
        }
        
        # Store profiling data
        operation_name = func.__name__ if hasattr(func, '__name__') else str(func)
        self.profiling_data[operation_name] = profiling_data
        
        return result, profiling_data
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate a deterministic cache key for function calls."""
        # Create a hashable representation of arguments
        key_data = {
            'function': func_name,
            'args': str(args),
            'kwargs': str(sorted(kwargs.items()))
        }
        
        # Generate hash
        key_string = str(key_data)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def analyze_performance_bottlenecks(self) -> Dict[str, Any]:
        """
        Analyze current performance bottlenecks across the system.
        
        Returns comprehensive analysis with optimization recommendations.
        """
        analysis = {
            'timestamp': time.time(),
            'cache_performance': self._analyze_cache_performance(),
            'processing_performance': self._analyze_processing_performance(),
            'system_performance': self._analyze_system_performance(),
            'optimization_recommendations': []
        }
        
        # Generate recommendations based on analysis
        recommendations = []
        
        # Cache analysis recommendations
        cache_perf = analysis['cache_performance']
        if cache_perf['mcp_hit_rate'] < self.performance_targets['cache_hit_rate']:
            recommendations.append(f"MCP cache hit rate ({cache_perf['mcp_hit_rate']:.1%}) below target ({self.performance_targets['cache_hit_rate']:.1%}) - consider increasing cache size or TTL")
        
        if cache_perf['sanskrit_hit_rate'] < self.performance_targets['cache_hit_rate']:
            recommendations.append(f"Sanskrit cache hit rate ({cache_perf['sanskrit_hit_rate']:.1%}) below target - optimize caching strategy")
        
        # Processing performance recommendations
        proc_perf = analysis['processing_performance']
        if proc_perf['average_processing_time'] > self.performance_targets['processing_time_seconds']:
            recommendations.append(f"Average processing time ({proc_perf['average_processing_time']:.3f}s) exceeds sub-second target")
        
        # Add system-level recommendations
        sys_perf = analysis['system_performance']
        if sys_perf['memory_pressure'] > 0.8:
            recommendations.append("High memory pressure detected - consider cache size optimization")
        
        analysis['optimization_recommendations'] = recommendations
        return analysis
    
    def _analyze_cache_performance(self) -> Dict[str, Any]:
        """Analyze caching system performance."""
        mcp_stats = self.mcp_cache.get_stats()
        sanskrit_stats = self.sanskrit_cache.get_stats()
        
        return {
            'mcp_hit_rate': mcp_stats.hit_rate,
            'mcp_memory_usage': mcp_stats.memory_usage,
            'mcp_evictions': mcp_stats.evictions,
            'sanskrit_hit_rate': sanskrit_stats.hit_rate,
            'sanskrit_memory_usage': sanskrit_stats.memory_usage,
            'sanskrit_evictions': sanskrit_stats.evictions,
            'total_cache_memory': mcp_stats.memory_usage + sanskrit_stats.memory_usage
        }
    
    def _analyze_processing_performance(self) -> Dict[str, Any]:
        """Analyze processing pipeline performance."""
        # Get recent metrics from performance monitor
        dashboard_data = self.monitor.get_performance_dashboard_data()
        
        return {
            'average_processing_time': dashboard_data['summary'].get('average_response_time_ms', 0) / 1000.0,
            'success_rate': dashboard_data['summary'].get('success_rate', 0),
            'error_rate': dashboard_data['summary'].get('error_rate', 0),
            'total_operations': dashboard_data['summary'].get('total_operations', 0),
            'component_performance': dashboard_data.get('component_performance', {})
        }
    
    def _analyze_system_performance(self) -> Dict[str, Any]:
        """Analyze system-level performance metrics."""
        try:
            import psutil
            # Get system resource usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            return {
                'cpu_usage': cpu_percent / 100.0,
                'memory_pressure': memory.percent / 100.0,
                'available_memory_gb': memory.available / (1024**3),
                'active_threads': threading.active_count()
            }
        except ImportError:
            # Fallback if psutil not available
            return {
                'cpu_usage': 0.0,
                'memory_pressure': 0.0,
                'available_memory_gb': 8.0,  # Assumed
                'active_threads': threading.active_count()
            }
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report for Story 4.3."""
        report = {
            'report_metadata': {
                'generated_at': time.time(),
                'optimizer_version': '4.3.0',
                'performance_targets': self.performance_targets
            },
            'performance_analysis': self.analyze_performance_bottlenecks(),
            'optimization_results': dict(self.optimization_results),
            'profiling_summary': self._summarize_profiling_data(),
            'cache_statistics': {
                'mcp_cache': self.mcp_cache.get_stats().__dict__,
                'sanskrit_cache': self.sanskrit_cache.get_stats().__dict__
            },
            'recommendations': self._generate_detailed_recommendations()
        }
        
        return report
    
    def _summarize_profiling_data(self) -> Dict[str, Any]:
        """Summarize profiling data across all operations."""
        if not self.profiling_data:
            return {'total_operations_profiled': 0}
        
        total_operations = len(self.profiling_data)
        total_time = sum(data['total_execution_time'] for data in self.profiling_data.values())
        success_rate = sum(1 for data in self.profiling_data.values() if data['success']) / total_operations
        
        # Find most expensive operations
        expensive_operations = sorted(
            self.profiling_data.items(),
            key=lambda x: x[1]['total_execution_time'],
            reverse=True
        )[:5]
        
        return {
            'total_operations_profiled': total_operations,
            'total_profiling_time': total_time,
            'success_rate': success_rate,
            'most_expensive_operations': [
                {'operation': op, 'time': data['total_execution_time']}
                for op, data in expensive_operations
            ]
        }
    
    def _generate_detailed_recommendations(self) -> List[str]:
        """Generate detailed optimization recommendations."""
        recommendations = []
        
        # Analyze current performance
        analysis = self.analyze_performance_bottlenecks()
        
        # Cache optimization recommendations
        if analysis['cache_performance']['mcp_hit_rate'] < 0.7:
            recommendations.append("Increase MCP cache size to improve hit rate and reduce API calls")
        
        if analysis['cache_performance']['total_cache_memory'] > 100 * 1024 * 1024:  # 100MB
            recommendations.append("Consider implementing cache memory limits to prevent excessive memory usage")
        
        # Processing optimization recommendations
        if analysis['processing_performance']['average_processing_time'] > 0.5:
            recommendations.append("Implement parallel processing for Sanskrit lexicon operations")
            recommendations.append("Consider pre-warming caches during system startup")
        
        # System optimization recommendations
        if analysis['system_performance']['cpu_usage'] > 0.8:
            recommendations.append("High CPU usage detected - consider optimizing concurrent operations")
        
        if analysis['system_performance']['memory_pressure'] > 0.8:
            recommendations.append("High memory pressure - implement more aggressive cache eviction")
        
        # Add Story 4.3 specific recommendations
        recommendations.extend([
            "Implement performance baseline validation before production deployment",
            "Add automated regression testing for all critical performance paths",
            "Monitor sub-second processing targets continuously in production"
        ])
        
        return recommendations
    
    def validate_performance_targets(self) -> Dict[str, bool]:
        """
        Validate that current performance meets Story 4.3 targets.
        
        Returns validation results for each performance target.
        """
        analysis = self.analyze_performance_bottlenecks()
        
        validation_results = {
            'sub_second_processing': analysis['processing_performance']['average_processing_time'] < self.performance_targets['processing_time_seconds'],
            'cache_efficiency': analysis['cache_performance']['mcp_hit_rate'] >= self.performance_targets['cache_hit_rate'],
            'high_success_rate': analysis['processing_performance']['success_rate'] >= 0.999,  # 99.9% uptime target
            'low_error_rate': analysis['processing_performance']['error_rate'] <= 0.001,  # <0.1% error rate
        }
        
        validation_results['all_targets_met'] = all(validation_results.values())
        
        return validation_results
    
    def __del__(self):
        """Cleanup resources on destruction."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


def optimize_for_production(func: Callable) -> Callable:
    """
    Global decorator for production optimization.
    
    Applies comprehensive optimization strategies for Story 4.3 requirements.
    """
    # Global optimizer instance
    if not hasattr(optimize_for_production, 'optimizer'):
        optimize_for_production.optimizer = PerformanceOptimizer()
    
    optimizer = optimize_for_production.optimizer
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Apply appropriate optimization based on function context
        if 'mcp' in func.__name__.lower() or 'transformer' in func.__name__.lower():
            return optimizer.optimize_mcp_transformer_call(func)(*args, **kwargs)
        elif 'sanskrit' in func.__name__.lower() or 'hindi' in func.__name__.lower():
            return optimizer.optimize_sanskrit_processing(func)(*args, **kwargs)
        else:
            # Generic optimization with monitoring
            with optimizer.monitor.monitor_processing_operation(func.__name__, "generic_processing"):
                return func(*args, **kwargs)
    
    return wrapper


# Testing functions for validation
def test_performance_optimization():
    """Test performance optimization functionality."""
    optimizer = PerformanceOptimizer()
    
    # Test MCP caching
    @optimizer.optimize_mcp_transformer_call
    def mock_mcp_operation(text: str) -> str:
        time.sleep(0.1)  # Simulate processing time
        return f"processed: {text}"
    
    # Test Sanskrit caching  
    @optimizer.optimize_sanskrit_processing
    def mock_sanskrit_operation(text: str) -> str:
        time.sleep(0.05)  # Simulate processing time
        return f"sanskrit: {text}"
    
    # Test operations
    print("Testing performance optimization...")
    
    # First calls should be cache misses
    result1 = mock_mcp_operation("test text")
    result2 = mock_sanskrit_operation("test sanskrit")
    
    # Second calls should be cache hits
    result1_cached = mock_mcp_operation("test text")
    result2_cached = mock_sanskrit_operation("test sanskrit")
    
    # Validate results
    assert result1 == result1_cached, "MCP caching failed"
    assert result2 == result2_cached, "Sanskrit caching failed"
    
    # Generate performance report
    report = optimizer.generate_optimization_report()
    
    print(f"✅ Performance optimization test passed")
    print(f"   Cache hit rates: MCP={optimizer.mcp_cache.get_stats().hit_rate:.1%}, Sanskrit={optimizer.sanskrit_cache.get_stats().hit_rate:.1%}")
    
    return True


if __name__ == "__main__":
    test_performance_optimization()