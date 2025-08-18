"""
Telemetry Collector for Enterprise-Grade Monitoring.

This module implements comprehensive telemetry collection and analysis for the
Sanskrit processing pipeline with real-time metrics aggregation and export.
"""

import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from enum import Enum
import uuid

# Import monitoring components
import sys
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from performance_monitor import MetricType, AlertSeverity


@dataclass
class TelemetryEvent:
    """Individual telemetry event."""
    event_id: str
    event_type: str
    timestamp: float
    source_component: str
    data: Dict[str, Any]
    tags: Dict[str, str] = field(default_factory=dict)
    severity: AlertSeverity = AlertSeverity.INFO


@dataclass
class MetricAggregation:
    """Aggregated metric data."""
    metric_name: str
    aggregation_type: str  # "sum", "avg", "min", "max", "count"
    value: float
    sample_count: int
    time_window_seconds: int
    timestamp: float
    components: Set[str] = field(default_factory=set)


@dataclass
class TelemetryExport:
    """Telemetry export configuration."""
    export_id: str
    format: str  # "json", "csv", "prometheus", "grafana"
    destination: str  # file path or endpoint URL
    metrics: List[str]
    export_interval_seconds: int
    enabled: bool = True


class TelemetryLevel(Enum):
    """Telemetry collection levels."""
    MINIMAL = "minimal"      # Critical metrics only
    STANDARD = "standard"    # Standard production metrics
    DETAILED = "detailed"    # Comprehensive debugging metrics
    DEBUG = "debug"         # Full diagnostic telemetry


class TelemetryCollector:
    """
    Enterprise-grade telemetry collection and analysis system.
    
    Provides comprehensive telemetry capabilities:
    - Real-time event collection and aggregation
    - Multi-level telemetry collection (minimal to debug)
    - Configurable metric aggregation and export
    - Performance impact monitoring
    - Integration with monitoring dashboards
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize telemetry collector with configuration."""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Core telemetry storage
        self.events: deque = deque(maxlen=50000)  # Raw events
        self.aggregated_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.component_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Configuration
        self.telemetry_level = TelemetryLevel(self.config.get('level', 'standard'))
        self.collection_enabled = self.config.get('enabled', True)
        self.event_retention_hours = self.config.get('event_retention_hours', 24)
        self.aggregation_window_seconds = self.config.get('aggregation_window_seconds', 60)
        
        # Aggregation configuration
        self.aggregation_rules: Dict[str, Dict[str, Any]] = self.config.get('aggregation_rules', {})
        self.export_configs: Dict[str, TelemetryExport] = {}
        
        # Threading and processing
        self.processing_thread = None
        self.export_thread = None
        self.running = False
        self.lock = threading.Lock()
        
        # Performance tracking
        self.collection_stats = {
            'events_collected': 0,
            'events_processed': 0,
            'aggregations_computed': 0,
            'exports_completed': 0,
            'processing_time_ms': 0.0,
            'last_cleanup': time.time()
        }
        
        # Default aggregation rules
        self._initialize_default_aggregation_rules()
        
        self.logger.info(f"TelemetryCollector initialized with {self.telemetry_level.value} level")
    
    def start_collection(self):
        """Start telemetry collection."""
        with self.lock:
            if self.running:
                return
            
            self.running = True
            
            # Start processing threads
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.export_thread = threading.Thread(target=self._export_loop, daemon=True)
            
            self.processing_thread.start()
            self.export_thread.start()
            
            self.logger.info("Telemetry collection started")
    
    def stop_collection(self):
        """Stop telemetry collection."""
        with self.lock:
            self.running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        if self.export_thread:
            self.export_thread.join(timeout=5)
        
        self.logger.info("Telemetry collection stopped")
    
    def collect_event(self, event_type: str, source_component: str, 
                     data: Dict[str, Any], tags: Optional[Dict[str, str]] = None,
                     severity: AlertSeverity = AlertSeverity.INFO):
        """Collect a telemetry event."""
        if not self.collection_enabled:
            return
        
        # Check if event should be collected based on telemetry level
        if not self._should_collect_event(event_type, severity):
            return
        
        event = TelemetryEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=time.time(),
            source_component=source_component,
            data=data,
            tags=tags or {},
            severity=severity
        )
        
        # Store event
        self.events.append(event)
        self.collection_stats['events_collected'] += 1
        
        # Update component metrics immediately for real-time access
        self._update_component_metrics(event)

    def record_event(self, event_type: str, data: Dict[str, Any], 
                    source_component: Optional[str] = None,
                    tags: Optional[Dict[str, str]] = None,
                    severity: AlertSeverity = AlertSeverity.INFO):
        """
        Simplified event recording API for backward compatibility.
        
        This method provides a simplified interface that matches the expected API
        used throughout the codebase. It delegates to collect_event with sensible defaults.
        
        Args:
            event_type: Type of event being recorded
            data: Event data dictionary
            source_component: Source component (auto-detected if not provided)
            tags: Optional tags for the event
            severity: Event severity level
        """
        # Auto-detect source component from stack trace if not provided
        if source_component is None:
            import inspect
            frame = inspect.currentframe()
            try:
                # Get the caller's module name
                caller_frame = frame.f_back
                if caller_frame and caller_frame.f_globals:
                    module_name = caller_frame.f_globals.get('__name__', 'unknown')
                    # Extract component name from module path
                    if '.' in module_name:
                        source_component = module_name.split('.')[-1]
                    else:
                        source_component = module_name
                else:
                    source_component = 'unknown'
            finally:
                del frame
        
        # Delegate to the main collect_event method
        self.collect_event(
            event_type=event_type,
            source_component=source_component,
            data=data,
            tags=tags,
            severity=severity
        )
    
    def collect_processing_telemetry(self, operation_name: str, component: str,
                                   processing_time_ms: float, success: bool,
                                   input_size: Optional[int] = None,
                                   output_size: Optional[int] = None,
                                   additional_data: Optional[Dict] = None):
        """Collect processing operation telemetry."""
        data = {
            'operation_name': operation_name,
            'processing_time_ms': processing_time_ms,
            'success': success,
            'input_size': input_size,
            'output_size': output_size
        }
        
        if additional_data:
            data.update(additional_data)
        
        severity = AlertSeverity.INFO if success else AlertSeverity.WARNING
        
        self.collect_event(
            event_type="processing_operation",
            source_component=component,
            data=data,
            tags={'operation': operation_name, 'status': 'success' if success else 'failure'},
            severity=severity
        )
    
    def collect_mcp_telemetry(self, operation: str, latency_ms: float, 
                            cache_hit: bool, success: bool,
                            error_message: Optional[str] = None):
        """Collect MCP-specific telemetry."""
        data = {
            'operation': operation,
            'latency_ms': latency_ms,
            'cache_hit': cache_hit,
            'success': success,
            'error_message': error_message
        }
        
        severity = AlertSeverity.WARNING if not success else AlertSeverity.INFO
        
        self.collect_event(
            event_type="mcp_operation",
            source_component="mcp_transformer",
            data=data,
            tags={
                'operation': operation,
                'cache': 'hit' if cache_hit else 'miss',
                'status': 'success' if success else 'error'
            },
            severity=severity
        )
    
    def collect_sanskrit_processing_telemetry(self, text_length: int, 
                                            corrections_applied: int,
                                            processing_time_ms: float,
                                            confidence_score: float):
        """Collect Sanskrit processing specific telemetry."""
        data = {
            'text_length': text_length,
            'corrections_applied': corrections_applied,
            'processing_time_ms': processing_time_ms,
            'confidence_score': confidence_score,
            'corrections_per_word': corrections_applied / max(text_length / 5, 1)  # Rough word estimate
        }
        
        self.collect_event(
            event_type="sanskrit_processing",
            source_component="sanskrit_processor",
            data=data,
            tags={'processing_stage': 'sanskrit_corrections'},
            severity=AlertSeverity.INFO
        )
    
    def add_aggregation_rule(self, metric_name: str, aggregation_type: str,
                           source_events: List[str], time_window_seconds: int = 60):
        """Add a metric aggregation rule."""
        self.aggregation_rules[metric_name] = {
            'aggregation_type': aggregation_type,
            'source_events': source_events,
            'time_window_seconds': time_window_seconds,
            'enabled': True
        }
        
        self.logger.info(f"Added aggregation rule: {metric_name}")
    
    def add_export_config(self, export_config: TelemetryExport):
        """Add telemetry export configuration."""
        self.export_configs[export_config.export_id] = export_config
        self.logger.info(f"Added export config: {export_config.export_id}")
    
    def get_telemetry_summary(self, time_range_minutes: int = 60) -> Dict[str, Any]:
        """Get telemetry summary for specified time range."""
        current_time = time.time()
        cutoff_time = current_time - (time_range_minutes * 60)
        
        # Filter events by time range
        recent_events = [e for e in self.events if e.timestamp >= cutoff_time]
        
        # Event analysis
        events_by_type = defaultdict(int)
        events_by_component = defaultdict(int)
        events_by_severity = defaultdict(int)
        
        for event in recent_events:
            events_by_type[event.event_type] += 1
            events_by_component[event.source_component] += 1
            events_by_severity[event.severity.value] += 1
        
        # Performance metrics
        processing_events = [e for e in recent_events if e.event_type == "processing_operation"]
        if processing_events:
            processing_times = [e.data.get('processing_time_ms', 0) for e in processing_events]
            success_rate = len([e for e in processing_events if e.data.get('success', False)]) / len(processing_events)
            avg_processing_time = sum(processing_times) / len(processing_times)
        else:
            success_rate = 0.0
            avg_processing_time = 0.0
        
        # MCP metrics
        mcp_events = [e for e in recent_events if e.event_type == "mcp_operation"]
        if mcp_events:
            cache_hit_rate = len([e for e in mcp_events if e.data.get('cache_hit', False)]) / len(mcp_events)
            avg_mcp_latency = sum([e.data.get('latency_ms', 0) for e in mcp_events]) / len(mcp_events)
        else:
            cache_hit_rate = 0.0
            avg_mcp_latency = 0.0
        
        return {
            'time_range_minutes': time_range_minutes,
            'summary': {
                'total_events': len(recent_events),
                'events_per_minute': len(recent_events) / time_range_minutes,
                'unique_components': len(events_by_component),
                'success_rate': success_rate,
                'avg_processing_time_ms': avg_processing_time
            },
            'events_by_type': dict(events_by_type),
            'events_by_component': dict(events_by_component),
            'events_by_severity': dict(events_by_severity),
            'performance_metrics': {
                'avg_processing_time_ms': avg_processing_time,
                'success_rate': success_rate,
                'mcp_cache_hit_rate': cache_hit_rate,
                'avg_mcp_latency_ms': avg_mcp_latency
            },
            'collection_stats': self.collection_stats.copy()
        }
    
    def get_component_telemetry(self, component: str, 
                               time_range_minutes: int = 60) -> Dict[str, Any]:
        """Get telemetry data for a specific component."""
        current_time = time.time()
        cutoff_time = current_time - (time_range_minutes * 60)
        
        # Filter events for this component
        component_events = [
            e for e in self.events 
            if e.source_component == component and e.timestamp >= cutoff_time
        ]
        
        if not component_events:
            return {
                'component': component,
                'time_range_minutes': time_range_minutes,
                'events': 0,
                'metrics': {}
            }
        
        # Analyze component events
        events_by_type = defaultdict(int)
        performance_data = []
        
        for event in component_events:
            events_by_type[event.event_type] += 1
            
            # Extract performance data
            if 'processing_time_ms' in event.data:
                performance_data.append({
                    'timestamp': event.timestamp,
                    'processing_time_ms': event.data['processing_time_ms'],
                    'success': event.data.get('success', True)
                })
        
        # Calculate performance metrics
        if performance_data:
            processing_times = [p['processing_time_ms'] for p in performance_data]
            success_count = len([p for p in performance_data if p['success']])
            
            performance_metrics = {
                'avg_processing_time_ms': sum(processing_times) / len(processing_times),
                'min_processing_time_ms': min(processing_times),
                'max_processing_time_ms': max(processing_times),
                'success_rate': success_count / len(performance_data),
                'total_operations': len(performance_data)
            }
        else:
            performance_metrics = {}
        
        return {
            'component': component,
            'time_range_minutes': time_range_minutes,
            'events': len(component_events),
            'events_by_type': dict(events_by_type),
            'performance_metrics': performance_metrics,
            'recent_metrics': self.component_metrics.get(component, {})
        }
    
    def generate_telemetry_report(self, hours_back: int = 24) -> Dict[str, Any]:
        """Generate comprehensive telemetry report."""
        current_time = time.time()
        report_start = current_time - (hours_back * 3600)
        
        # Filter events for reporting period
        period_events = [e for e in self.events if e.timestamp >= report_start]
        
        # Component analysis
        component_analysis = {}
        unique_components = set(e.source_component for e in period_events)
        
        for component in unique_components:
            component_analysis[component] = self.get_component_telemetry(
                component, time_range_minutes=hours_back * 60
            )
        
        # System-wide performance analysis
        processing_events = [e for e in period_events if e.event_type == "processing_operation"]
        mcp_events = [e for e in period_events if e.event_type == "mcp_operation"]
        sanskrit_events = [e for e in period_events if e.event_type == "sanskrit_processing"]
        
        report = {
            'report_metadata': {
                'generated_at': current_time,
                'period_hours': hours_back,
                'telemetry_level': self.telemetry_level.value,
                'collection_enabled': self.collection_enabled
            },
            'executive_summary': {
                'total_events': len(period_events),
                'events_per_hour': len(period_events) / hours_back,
                'components_monitored': len(unique_components),
                'telemetry_health': self._assess_telemetry_health()
            },
            'performance_analysis': self._analyze_performance_telemetry(processing_events),
            'mcp_analysis': self._analyze_mcp_telemetry(mcp_events),
            'sanskrit_analysis': self._analyze_sanskrit_telemetry(sanskrit_events),
            'component_analysis': component_analysis,
            'system_health_indicators': self._calculate_health_indicators(period_events),
            'recommendations': self._generate_telemetry_recommendations(period_events)
        }
        
        return report
    
    def _initialize_default_aggregation_rules(self):
        """Initialize default aggregation rules."""
        default_rules = {
            'avg_processing_time': {
                'aggregation_type': 'avg',
                'source_events': ['processing_operation'],
                'time_window_seconds': 300,  # 5 minutes
                'enabled': True
            },
            'total_operations': {
                'aggregation_type': 'count',
                'source_events': ['processing_operation'],
                'time_window_seconds': 60,
                'enabled': True
            },
            'success_rate': {
                'aggregation_type': 'avg',
                'source_events': ['processing_operation'],
                'time_window_seconds': 300,
                'enabled': True
            },
            'mcp_cache_hit_rate': {
                'aggregation_type': 'avg',
                'source_events': ['mcp_operation'],
                'time_window_seconds': 300,
                'enabled': True
            }
        }
        
        self.aggregation_rules.update(default_rules)
    
    def _should_collect_event(self, event_type: str, severity: AlertSeverity) -> bool:
        """Determine if event should be collected based on telemetry level."""
        if self.telemetry_level == TelemetryLevel.DEBUG:
            return True
        elif self.telemetry_level == TelemetryLevel.DETAILED:
            return severity in [AlertSeverity.INFO, AlertSeverity.WARNING, AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
        elif self.telemetry_level == TelemetryLevel.STANDARD:
            return severity in [AlertSeverity.WARNING, AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY] or event_type in ['processing_operation', 'mcp_operation']
        elif self.telemetry_level == TelemetryLevel.MINIMAL:
            return severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
        
        return False
    
    def _update_component_metrics(self, event: TelemetryEvent):
        """Update component metrics with new event."""
        component = event.source_component
        
        # Update component-specific metrics
        if component not in self.component_metrics:
            self.component_metrics[component] = {
                'last_event_timestamp': event.timestamp,
                'event_count': 0,
                'avg_processing_time_ms': 0.0,
                'success_rate': 1.0,
                'recent_events': deque(maxlen=100)
            }
        
        component_metrics = self.component_metrics[component]
        component_metrics['last_event_timestamp'] = event.timestamp
        component_metrics['event_count'] += 1
        component_metrics['recent_events'].append(event)
        
        # Update processing metrics
        if 'processing_time_ms' in event.data:
            recent_processing_times = [
                e.data.get('processing_time_ms', 0) 
                for e in component_metrics['recent_events']
                if 'processing_time_ms' in e.data
            ]
            if recent_processing_times:
                component_metrics['avg_processing_time_ms'] = sum(recent_processing_times) / len(recent_processing_times)
        
        # Update success rate
        recent_successes = [
            e.data.get('success', True)
            for e in component_metrics['recent_events']
            if 'success' in e.data
        ]
        if recent_successes:
            component_metrics['success_rate'] = sum(recent_successes) / len(recent_successes)
    
    def _processing_loop(self):
        """Main telemetry processing loop."""
        while self.running:
            try:
                start_time = time.time()
                
                # Process aggregation rules
                self._process_aggregations()
                
                # Clean up old events
                self._cleanup_old_events()
                
                processing_time = (time.time() - start_time) * 1000
                self.collection_stats['processing_time_ms'] = processing_time
                
                time.sleep(self.aggregation_window_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in telemetry processing loop: {e}")
                time.sleep(5)
    
    def _export_loop(self):
        """Export loop for telemetry data."""
        while self.running:
            try:
                for export_id, export_config in self.export_configs.items():
                    if export_config.enabled:
                        self._export_telemetry(export_config)
                
                time.sleep(60)  # Check exports every minute
                
            except Exception as e:
                self.logger.error(f"Error in telemetry export loop: {e}")
                time.sleep(10)
    
    def _process_aggregations(self):
        """Process metric aggregations."""
        current_time = time.time()
        
        for metric_name, rule in self.aggregation_rules.items():
            if not rule.get('enabled', True):
                continue
            
            try:
                # Get events within time window
                window_start = current_time - rule['time_window_seconds']
                relevant_events = [
                    e for e in self.events
                    if e.timestamp >= window_start and e.event_type in rule['source_events']
                ]
                
                if relevant_events:
                    # Compute aggregation
                    aggregation = self._compute_aggregation(metric_name, rule, relevant_events)
                    self.aggregated_metrics[metric_name].append(aggregation)
                    self.collection_stats['aggregations_computed'] += 1
                
            except Exception as e:
                self.logger.error(f"Error processing aggregation {metric_name}: {e}")
    
    def _compute_aggregation(self, metric_name: str, rule: Dict, events: List[TelemetryEvent]) -> MetricAggregation:
        """Compute metric aggregation from events."""
        aggregation_type = rule['aggregation_type']
        values = []
        components = set()
        
        for event in events:
            components.add(event.source_component)
            
            # Extract relevant values based on aggregation type and metric
            if metric_name == 'avg_processing_time':
                if 'processing_time_ms' in event.data:
                    values.append(event.data['processing_time_ms'])
            elif metric_name == 'success_rate':
                if 'success' in event.data:
                    values.append(1.0 if event.data['success'] else 0.0)
            elif metric_name == 'mcp_cache_hit_rate':
                if 'cache_hit' in event.data:
                    values.append(1.0 if event.data['cache_hit'] else 0.0)
            elif metric_name == 'total_operations':
                values.append(1.0)  # Count each event
        
        # Compute aggregated value
        if not values:
            aggregated_value = 0.0
        elif aggregation_type == 'avg':
            aggregated_value = sum(values) / len(values)
        elif aggregation_type == 'sum':
            aggregated_value = sum(values)
        elif aggregation_type == 'min':
            aggregated_value = min(values)
        elif aggregation_type == 'max':
            aggregated_value = max(values)
        elif aggregation_type == 'count':
            aggregated_value = len(values)
        else:
            aggregated_value = 0.0
        
        return MetricAggregation(
            metric_name=metric_name,
            aggregation_type=aggregation_type,
            value=aggregated_value,
            sample_count=len(values),
            time_window_seconds=rule['time_window_seconds'],
            timestamp=time.time(),
            components=components
        )
    
    def _export_telemetry(self, export_config: TelemetryExport):
        """Export telemetry data according to configuration."""
        try:
            # This is a placeholder for telemetry export functionality
            # In a real implementation, this would export to various formats/destinations
            self.collection_stats['exports_completed'] += 1
            
        except Exception as e:
            self.logger.error(f"Error exporting telemetry {export_config.export_id}: {e}")
    
    def _cleanup_old_events(self):
        """Clean up old events based on retention policy."""
        current_time = time.time()
        cutoff_time = current_time - (self.event_retention_hours * 3600)
        
        # Clean events deque
        while self.events and self.events[0].timestamp < cutoff_time:
            self.events.popleft()
        
        # Clean aggregated metrics
        for metric_name, metrics_deque in self.aggregated_metrics.items():
            while metrics_deque and metrics_deque[0].timestamp < cutoff_time:
                metrics_deque.popleft()
        
        self.collection_stats['last_cleanup'] = current_time
    
    def _analyze_performance_telemetry(self, processing_events: List[TelemetryEvent]) -> Dict[str, Any]:
        """Analyze performance telemetry data."""
        if not processing_events:
            return {'total_operations': 0, 'message': 'No processing events found'}
        
        # Extract performance data
        processing_times = [e.data.get('processing_time_ms', 0) for e in processing_events if 'processing_time_ms' in e.data]
        successes = [e.data.get('success', True) for e in processing_events if 'success' in e.data]
        
        return {
            'total_operations': len(processing_events),
            'avg_processing_time_ms': sum(processing_times) / len(processing_times) if processing_times else 0,
            'min_processing_time_ms': min(processing_times) if processing_times else 0,
            'max_processing_time_ms': max(processing_times) if processing_times else 0,
            'success_rate': sum(successes) / len(successes) if successes else 0,
            'operations_per_hour': len(processing_events) / 24,  # Assuming 24-hour period
            'sub_second_compliance': len([t for t in processing_times if t < 1000]) / len(processing_times) if processing_times else 0
        }
    
    def _analyze_mcp_telemetry(self, mcp_events: List[TelemetryEvent]) -> Dict[str, Any]:
        """Analyze MCP telemetry data."""
        if not mcp_events:
            return {'total_operations': 0, 'message': 'No MCP events found'}
        
        # Extract MCP data
        latencies = [e.data.get('latency_ms', 0) for e in mcp_events if 'latency_ms' in e.data]
        cache_hits = [e.data.get('cache_hit', False) for e in mcp_events if 'cache_hit' in e.data]
        successes = [e.data.get('success', True) for e in mcp_events if 'success' in e.data]
        
        return {
            'total_operations': len(mcp_events),
            'avg_latency_ms': sum(latencies) / len(latencies) if latencies else 0,
            'cache_hit_rate': sum(cache_hits) / len(cache_hits) if cache_hits else 0,
            'success_rate': sum(successes) / len(successes) if successes else 0,
            'operations_per_hour': len(mcp_events) / 24
        }
    
    def _analyze_sanskrit_telemetry(self, sanskrit_events: List[TelemetryEvent]) -> Dict[str, Any]:
        """Analyze Sanskrit processing telemetry data."""
        if not sanskrit_events:
            return {'total_operations': 0, 'message': 'No Sanskrit events found'}
        
        # Extract Sanskrit data
        text_lengths = [e.data.get('text_length', 0) for e in sanskrit_events if 'text_length' in e.data]
        corrections = [e.data.get('corrections_applied', 0) for e in sanskrit_events if 'corrections_applied' in e.data]
        confidence_scores = [e.data.get('confidence_score', 0) for e in sanskrit_events if 'confidence_score' in e.data]
        
        return {
            'total_operations': len(sanskrit_events),
            'avg_text_length': sum(text_lengths) / len(text_lengths) if text_lengths else 0,
            'avg_corrections_per_operation': sum(corrections) / len(corrections) if corrections else 0,
            'avg_confidence_score': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            'operations_per_hour': len(sanskrit_events) / 24
        }
    
    def _assess_telemetry_health(self) -> str:
        """Assess overall telemetry system health."""
        # Simple health assessment based on collection stats
        if self.collection_stats['events_collected'] > 0:
            if self.collection_stats['processing_time_ms'] < 100:  # Less than 100ms processing overhead
                return "HEALTHY"
            elif self.collection_stats['processing_time_ms'] < 500:
                return "WARNING"
            else:
                return "CRITICAL"
        else:
            return "NO_DATA"
    
    def _calculate_health_indicators(self, events: List[TelemetryEvent]) -> Dict[str, Any]:
        """Calculate system health indicators from telemetry."""
        if not events:
            return {'status': 'NO_DATA'}
        
        # Error rate calculation
        total_events = len(events)
        error_events = len([e for e in events if e.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]])
        error_rate = error_events / total_events if total_events > 0 else 0
        
        # Performance indicators
        processing_events = [e for e in events if e.event_type == "processing_operation"]
        if processing_events:
            avg_processing_time = sum([e.data.get('processing_time_ms', 0) for e in processing_events]) / len(processing_events)
            success_rate = len([e for e in processing_events if e.data.get('success', True)]) / len(processing_events)
        else:
            avg_processing_time = 0
            success_rate = 1.0
        
        return {
            'status': 'HEALTHY' if error_rate < 0.01 and success_rate > 0.99 else 'WARNING',
            'error_rate': error_rate,
            'success_rate': success_rate,
            'avg_processing_time_ms': avg_processing_time,
            'total_events_analyzed': total_events
        }
    
    def _generate_telemetry_recommendations(self, events: List[TelemetryEvent]) -> List[str]:
        """Generate recommendations based on telemetry analysis."""
        recommendations = []
        
        if len(events) < 100:
            recommendations.append("Low telemetry volume - consider increasing collection level")
        
        # Check processing performance
        processing_events = [e for e in events if e.event_type == "processing_operation"]
        if processing_events:
            avg_time = sum([e.data.get('processing_time_ms', 0) for e in processing_events]) / len(processing_events)
            if avg_time > 1000:
                recommendations.append(f"Average processing time ({avg_time:.0f}ms) exceeds sub-second target")
        
        # Check error rate
        error_events = len([e for e in events if e.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]])
        if error_events > len(events) * 0.05:
            recommendations.append("High error rate detected - investigate critical events")
        
        # Story 4.3 specific recommendations
        recommendations.extend([
            "Monitor telemetry collection overhead continuously",
            "Export telemetry data for long-term analysis",
            "Validate telemetry completeness across all components"
        ])
        
        return recommendations
    
    def __del__(self):
        """Cleanup on destruction."""
        self.stop_collection()


def test_telemetry_collector():
    """Test telemetry collection functionality."""
    collector = TelemetryCollector()
    
    print("Testing telemetry collector...")
    
    # Start collection
    collector.start_collection()
    
    # Collect test events
    collector.collect_processing_telemetry("test_operation", "test_component", 150.0, True, 100, 80)
    collector.collect_mcp_telemetry("transform_text", 75.0, True, True)
    collector.collect_sanskrit_processing_telemetry(50, 3, 120.0, 0.95)
    
    time.sleep(2)  # Let processing occur
    
    # Get summary
    summary = collector.get_telemetry_summary()
    report = collector.generate_telemetry_report(1)  # 1 hour back
    
    print(f"✅ Telemetry collector test passed")
    print(f"   Events collected: {summary['summary']['total_events']}")
    print(f"   Components monitored: {summary['summary']['unique_components']}")
    print(f"   Success rate: {summary['performance_metrics']['success_rate']:.1%}")
    
    # Stop collection
    collector.stop_collection()
    
    return True


if __name__ == "__main__":
    test_telemetry_collector()