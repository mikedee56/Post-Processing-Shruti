"""
Performance Metrics Collection System for Production Observability
Provides comprehensive metrics collection, analysis, and reporting capabilities
"""

import time
import psutil
import threading
import statistics
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import logging

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics collected"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"


class MetricUnit(Enum):
    """Units for metrics"""
    BYTES = "bytes"
    SECONDS = "seconds"
    MILLISECONDS = "milliseconds"
    PERCENT = "percent"
    COUNT = "count"
    OPERATIONS_PER_SECOND = "ops/sec"


@dataclass
class MetricSample:
    """Individual metric sample"""
    timestamp: datetime
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceMetric:
    """Performance metric definition and data"""
    name: str
    metric_type: MetricType
    unit: MetricUnit
    description: str
    samples: deque = field(default_factory=lambda: deque(maxlen=10000))
    labels: Dict[str, str] = field(default_factory=dict)
    
    def add_sample(self, value: Union[int, float], labels: Optional[Dict[str, str]] = None):
        """Add a metric sample"""
        sample = MetricSample(
            timestamp=datetime.now(timezone.utc),
            value=value,
            labels=labels or {}
        )
        self.samples.append(sample)
    
    def get_current_value(self) -> Optional[Union[int, float]]:
        """Get the most recent value"""
        if self.samples:
            return self.samples[-1].value
        return None
    
    def get_statistics(self, window_minutes: int = 5) -> Dict[str, Any]:
        """Get statistical summary for a time window"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        recent_samples = [
            sample.value for sample in self.samples
            if sample.timestamp >= cutoff_time
        ]
        
        if not recent_samples:
            return {}
        
        return {
            'count': len(recent_samples),
            'min': min(recent_samples),
            'max': max(recent_samples),
            'mean': statistics.mean(recent_samples),
            'median': statistics.median(recent_samples),
            'std_dev': statistics.stdev(recent_samples) if len(recent_samples) > 1 else 0,
            'percentiles': {
                '50': statistics.quantiles(recent_samples, n=2)[0] if len(recent_samples) >= 2 else recent_samples[0],
                '95': statistics.quantiles(recent_samples, n=20)[18] if len(recent_samples) >= 20 else max(recent_samples),
                '99': statistics.quantiles(recent_samples, n=100)[98] if len(recent_samples) >= 100 else max(recent_samples)
            }
        }


class SystemMetricsCollector:
    """Collects system-level performance metrics"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.boot_time = psutil.boot_time()
        
    def collect_cpu_metrics(self) -> Dict[str, float]:
        """Collect CPU metrics"""
        return {
            'cpu_percent_total': psutil.cpu_percent(interval=1),
            'cpu_percent_per_core': psutil.cpu_percent(interval=1, percpu=True),
            'cpu_count_logical': psutil.cpu_count(),
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'load_average_1min': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0,
            'load_average_5min': psutil.getloadavg()[1] if hasattr(psutil, 'getloadavg') else 0,
            'load_average_15min': psutil.getloadavg()[2] if hasattr(psutil, 'getloadavg') else 0,
        }
    
    def collect_memory_metrics(self) -> Dict[str, float]:
        """Collect memory metrics"""
        virtual_mem = psutil.virtual_memory()
        swap_mem = psutil.swap_memory()
        process_mem = self.process.memory_info()
        
        return {
            'memory_total_bytes': virtual_mem.total,
            'memory_available_bytes': virtual_mem.available,
            'memory_used_bytes': virtual_mem.used,
            'memory_percent': virtual_mem.percent,
            'memory_cached_bytes': getattr(virtual_mem, 'cached', 0),
            'memory_buffers_bytes': getattr(virtual_mem, 'buffers', 0),
            'swap_total_bytes': swap_mem.total,
            'swap_used_bytes': swap_mem.used,
            'swap_percent': swap_mem.percent,
            'process_memory_rss_bytes': process_mem.rss,
            'process_memory_vms_bytes': process_mem.vms,
        }
    
    def collect_disk_metrics(self) -> Dict[str, float]:
        """Collect disk metrics"""
        metrics = {}
        
        # Disk usage for key directories
        key_paths = ['/app/data', '/app/logs', '/tmp']
        for path in key_paths:
            try:
                usage = psutil.disk_usage(path)
                safe_path = path.replace('/', '_').strip('_')
                metrics.update({
                    f'disk_{safe_path}_total_bytes': usage.total,
                    f'disk_{safe_path}_used_bytes': usage.used,
                    f'disk_{safe_path}_free_bytes': usage.free,
                    f'disk_{safe_path}_percent': (usage.used / usage.total) * 100,
                })
            except (OSError, FileNotFoundError):
                pass
        
        # Disk I/O
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                metrics.update({
                    'disk_read_bytes_total': disk_io.read_bytes,
                    'disk_write_bytes_total': disk_io.write_bytes,
                    'disk_read_count_total': disk_io.read_count,
                    'disk_write_count_total': disk_io.write_count,
                })
        except Exception:
            pass
        
        return metrics
    
    def collect_network_metrics(self) -> Dict[str, float]:
        """Collect network metrics"""
        try:
            net_io = psutil.net_io_counters()
            return {
                'network_bytes_sent_total': net_io.bytes_sent,
                'network_bytes_recv_total': net_io.bytes_recv,
                'network_packets_sent_total': net_io.packets_sent,
                'network_packets_recv_total': net_io.packets_recv,
                'network_errors_in_total': net_io.errin,
                'network_errors_out_total': net_io.errout,
                'network_drops_in_total': net_io.dropin,
                'network_drops_out_total': net_io.dropout,
            }
        except Exception:
            return {}


class ApplicationMetricsCollector:
    """Collects application-specific performance metrics"""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.processing_times = deque(maxlen=1000)
        self.lock = threading.Lock()
    
    def record_request(self):
        """Record a request"""
        with self.lock:
            self.request_count += 1
    
    def record_error(self):
        """Record an error"""
        with self.lock:
            self.error_count += 1
    
    def record_processing_time(self, duration_ms: float):
        """Record processing time"""
        with self.lock:
            self.processing_times.append(duration_ms)
    
    def collect_application_metrics(self) -> Dict[str, float]:
        """Collect application metrics"""
        with self.lock:
            uptime = time.time() - self.start_time
            requests_per_second = self.request_count / uptime if uptime > 0 else 0
            error_rate = self.error_count / self.request_count if self.request_count > 0 else 0
            
            # Processing time statistics
            avg_processing_time = statistics.mean(self.processing_times) if self.processing_times else 0
            p95_processing_time = (
                statistics.quantiles(list(self.processing_times), n=20)[18]
                if len(self.processing_times) >= 20 else avg_processing_time
            )
            
            return {
                'app_uptime_seconds': uptime,
                'app_requests_total': self.request_count,
                'app_errors_total': self.error_count,
                'app_requests_per_second': requests_per_second,
                'app_error_rate': error_rate,
                'app_processing_time_avg_ms': avg_processing_time,
                'app_processing_time_p95_ms': p95_processing_time,
            }


class PerformanceMetricsCollector:
    """
    Main performance metrics collection system
    Aggregates system, application, and custom metrics
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Collectors
        self.system_collector = SystemMetricsCollector()
        self.app_collector = ApplicationMetricsCollector()
        
        # Metrics storage
        self.metrics: Dict[str, PerformanceMetric] = {}
        self.custom_metrics: Dict[str, Any] = {}
        self.lock = threading.Lock()
        
        # Collection settings
        self.collection_interval = self.config.get('collection_interval_seconds', 60)
        self.retention_hours = self.config.get('retention_hours', 24)
        
        # Background collection thread
        self.collection_thread = None
        self.stop_collection = threading.Event()
        
        self._initialize_core_metrics()
        logger.info("PerformanceMetricsCollector initialized")
    
    def _initialize_core_metrics(self):
        """Initialize core performance metrics"""
        core_metrics = [
            # System metrics
            ('cpu_usage_percent', MetricType.GAUGE, MetricUnit.PERCENT, 'CPU usage percentage'),
            ('memory_usage_percent', MetricType.GAUGE, MetricUnit.PERCENT, 'Memory usage percentage'),
            ('disk_usage_percent', MetricType.GAUGE, MetricUnit.PERCENT, 'Disk usage percentage'),
            
            # Application metrics
            ('request_rate', MetricType.GAUGE, MetricUnit.OPERATIONS_PER_SECOND, 'Request rate'),
            ('error_rate', MetricType.GAUGE, MetricUnit.PERCENT, 'Error rate'),
            ('response_time', MetricType.HISTOGRAM, MetricUnit.MILLISECONDS, 'Response time'),
            ('processing_time', MetricType.HISTOGRAM, MetricUnit.MILLISECONDS, 'Processing time'),
            
            # Business metrics
            ('files_processed_total', MetricType.COUNTER, MetricUnit.COUNT, 'Total files processed'),
            ('processing_queue_size', MetricType.GAUGE, MetricUnit.COUNT, 'Processing queue size'),
            ('active_sessions', MetricType.GAUGE, MetricUnit.COUNT, 'Active user sessions'),
        ]
        
        for name, metric_type, unit, description in core_metrics:
            self.metrics[name] = PerformanceMetric(
                name=name,
                metric_type=metric_type,
                unit=unit,
                description=description
            )
    
    def start_collection(self):
        """Start background metrics collection"""
        if self.collection_thread and self.collection_thread.is_alive():
            logger.warning("Metrics collection already running")
            return
        
        self.stop_collection.clear()
        self.collection_thread = threading.Thread(target=self._collect_metrics_loop, daemon=True)
        self.collection_thread.start()
        logger.info("Started background metrics collection")
    
    def stop_collection_thread(self):
        """Stop background metrics collection"""
        if self.collection_thread:
            self.stop_collection.set()
            self.collection_thread.join(timeout=10)
            logger.info("Stopped background metrics collection")
    
    def _collect_metrics_loop(self):
        """Background thread for periodic metrics collection"""
        while not self.stop_collection.wait(self.collection_interval):
            try:
                self._collect_all_metrics()
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
    
    def _collect_all_metrics(self):
        """Collect all metrics from various sources"""
        timestamp = datetime.now(timezone.utc)
        
        # Collect system metrics
        try:
            cpu_metrics = self.system_collector.collect_cpu_metrics()
            self.record_metric('cpu_usage_percent', cpu_metrics['cpu_percent_total'])
            
            memory_metrics = self.system_collector.collect_memory_metrics()
            self.record_metric('memory_usage_percent', memory_metrics['memory_percent'])
            
            disk_metrics = self.system_collector.collect_disk_metrics()
            # Use root disk usage as primary metric
            for key, value in disk_metrics.items():
                if 'app_data_percent' in key:
                    self.record_metric('disk_usage_percent', value)
                    break
                    
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
        
        # Collect application metrics
        try:
            app_metrics = self.app_collector.collect_application_metrics()
            self.record_metric('request_rate', app_metrics['app_requests_per_second'])
            self.record_metric('error_rate', app_metrics['app_error_rate'] * 100)  # Convert to percentage
            self.record_metric('response_time', app_metrics['app_processing_time_avg_ms'])
            
        except Exception as e:
            logger.warning(f"Failed to collect application metrics: {e}")
    
    def record_metric(self, name: str, value: Union[int, float], 
                     labels: Optional[Dict[str, str]] = None):
        """Record a metric value"""
        with self.lock:
            if name not in self.metrics:
                # Auto-create metric if it doesn't exist
                self.metrics[name] = PerformanceMetric(
                    name=name,
                    metric_type=MetricType.GAUGE,
                    unit=MetricUnit.COUNT,
                    description=f"Auto-created metric: {name}"
                )
            
            self.metrics[name].add_sample(value, labels)
    
    def record_request(self):
        """Record an incoming request"""
        self.app_collector.record_request()
    
    def record_error(self):
        """Record an error"""
        self.app_collector.record_error()
    
    def record_processing_time(self, duration_ms: float):
        """Record processing time"""
        self.app_collector.record_processing_time(duration_ms)
        self.record_metric('processing_time', duration_ms)
    
    def increment_counter(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        current_value = self.get_metric_value(name) or 0
        self.record_metric(name, current_value + 1, labels)
    
    def set_gauge(self, name: str, value: Union[int, float], 
                  labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric value"""
        self.record_metric(name, value, labels)
    
    def get_metric_value(self, name: str) -> Optional[Union[int, float]]:
        """Get current value of a metric"""
        with self.lock:
            metric = self.metrics.get(name)
            return metric.get_current_value() if metric else None
    
    def get_metric_statistics(self, name: str, window_minutes: int = 5) -> Dict[str, Any]:
        """Get statistical summary for a metric"""
        with self.lock:
            metric = self.metrics.get(name)
            return metric.get_statistics(window_minutes) if metric else {}
    
    def get_all_metrics_snapshot(self) -> Dict[str, Any]:
        """Get snapshot of all current metrics"""
        snapshot = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'metrics': {}
        }
        
        with self.lock:
            for name, metric in self.metrics.items():
                current_value = metric.get_current_value()
                stats = metric.get_statistics(window_minutes=5)
                
                snapshot['metrics'][name] = {
                    'current_value': current_value,
                    'type': metric.metric_type.value,
                    'unit': metric.unit.value,
                    'description': metric.description,
                    'statistics': stats
                }
        
        return snapshot
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get high-level performance summary"""
        return {
            'system': {
                'cpu_usage': self.get_metric_value('cpu_usage_percent'),
                'memory_usage': self.get_metric_value('memory_usage_percent'),
                'disk_usage': self.get_metric_value('disk_usage_percent'),
            },
            'application': {
                'request_rate': self.get_metric_value('request_rate'),
                'error_rate': self.get_metric_value('error_rate'),
                'avg_response_time': self.get_metric_value('response_time'),
            },
            'health_status': self._calculate_health_status()
        }
    
    def _calculate_health_status(self) -> str:
        """Calculate overall health status based on metrics"""
        try:
            cpu_usage = self.get_metric_value('cpu_usage_percent') or 0
            memory_usage = self.get_metric_value('memory_usage_percent') or 0
            error_rate = self.get_metric_value('error_rate') or 0
            
            # Health thresholds
            if cpu_usage > 90 or memory_usage > 95 or error_rate > 5:
                return 'critical'
            elif cpu_usage > 75 or memory_usage > 85 or error_rate > 1:
                return 'warning'
            else:
                return 'healthy'
                
        except Exception:
            return 'unknown'
    
    def export_metrics_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        with self.lock:
            for name, metric in self.metrics.items():
                current_value = metric.get_current_value()
                if current_value is not None:
                    # Convert metric name to Prometheus format
                    prom_name = name.replace('-', '_').lower()
                    
                    # Add help text
                    lines.append(f"# HELP {prom_name} {metric.description}")
                    lines.append(f"# TYPE {prom_name} {metric.metric_type.value}")
                    
                    # Add metric value
                    if metric.labels:
                        label_str = ','.join([f'{k}="{v}"' for k, v in metric.labels.items()])
                        lines.append(f"{prom_name}{{{label_str}}} {current_value}")
                    else:
                        lines.append(f"{prom_name} {current_value}")
        
        return '\n'.join(lines)
    
    def cleanup_old_metrics(self):
        """Clean up old metric samples"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.retention_hours)
        
        with self.lock:
            for metric in self.metrics.values():
                # Remove old samples
                while metric.samples and metric.samples[0].timestamp < cutoff_time:
                    metric.samples.popleft()


# Decorator for timing function execution
def timed_metric(metrics_collector: PerformanceMetricsCollector, metric_name: str):
    """Decorator to time function execution and record as metric"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                metrics_collector.record_metric(metric_name, duration_ms)
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                metrics_collector.record_metric(f"{metric_name}_error", duration_ms)
                raise
        return wrapper
    return decorator


# Global metrics collector
_global_metrics_collector: Optional[PerformanceMetricsCollector] = None


def initialize_metrics_collection(config: Optional[Dict] = None) -> PerformanceMetricsCollector:
    """Initialize global metrics collector"""
    global _global_metrics_collector
    _global_metrics_collector = PerformanceMetricsCollector(config)
    _global_metrics_collector.start_collection()
    return _global_metrics_collector


def get_metrics_collector() -> Optional[PerformanceMetricsCollector]:
    """Get global metrics collector"""
    return _global_metrics_collector


def record_metric(name: str, value: Union[int, float], 
                 labels: Optional[Dict[str, str]] = None):
    """Record metric using global collector"""
    if _global_metrics_collector:
        _global_metrics_collector.record_metric(name, value, labels)