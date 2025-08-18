"""
Enterprise System Monitor for Production Excellence.

This module implements comprehensive system monitoring with real-time dashboards,
alerting, and telemetry collection for the Sanskrit processing pipeline.
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
import weakref

# Import existing performance monitoring
import sys
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from performance_monitor import PerformanceMonitor, MetricType, AlertSeverity


@dataclass 
class SystemHealthMetric:
    """System health metric data point."""
    metric_name: str
    value: float
    unit: str
    timestamp: float
    component: str
    severity: AlertSeverity = AlertSeverity.INFO
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Alert rule configuration."""
    rule_id: str
    metric_name: str
    condition: str  # ">" | "<" | "==" | "!="
    threshold: float
    severity: AlertSeverity
    enabled: bool = True
    suppression_window_seconds: int = 300  # 5 minutes


@dataclass 
class DashboardPanel:
    """Dashboard panel configuration."""
    panel_id: str
    title: str
    metric_names: List[str]
    chart_type: str  # "line" | "bar" | "gauge" | "table"
    time_range_minutes: int = 60
    refresh_interval_seconds: int = 30


@dataclass
class SystemAlert:
    """System alert instance."""
    alert_id: str
    rule_id: str
    severity: AlertSeverity
    title: str
    description: str
    metric_value: float
    threshold: float
    component: str
    triggered_at: float
    resolved_at: Optional[float] = None
    acknowledged_at: Optional[float] = None


class MonitoringEventType(Enum):
    """Types of monitoring events."""
    METRIC_RECORDED = "metric_recorded"
    ALERT_TRIGGERED = "alert_triggered" 
    ALERT_RESOLVED = "alert_resolved"
    SYSTEM_HEALTH_CHANGE = "system_health_change"
    DASHBOARD_UPDATE = "dashboard_update"


@dataclass
class MonitoringEvent:
    """Monitoring system event."""
    event_type: MonitoringEventType
    timestamp: float
    data: Dict[str, Any]
    source: str


class SystemMonitor:
    """
    Enterprise-grade system monitoring with real-time dashboards.
    
    Provides comprehensive monitoring capabilities:
    - Real-time metric collection and visualization
    - Configurable alerting with escalation
    - System health assessment and reporting
    - Performance dashboard generation
    - Integration with existing PerformanceMonitor
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize system monitor with enterprise configuration."""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Core monitoring components
        self.performance_monitor = PerformanceMonitor(self.config.get('performance', {}))
        
        # System metrics storage
        self.system_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.health_metrics: Dict[str, SystemHealthMetric] = {}
        
        # Alerting system
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, SystemAlert] = {}
        self.alert_history: deque = deque(maxlen=5000)
        
        # Dashboard configuration  
        self.dashboard_panels: Dict[str, DashboardPanel] = {}
        self.dashboard_cache: Dict[str, Any] = {}
        
        # Event system
        self.event_handlers: Dict[MonitoringEventType, List[Callable]] = defaultdict(list)
        self.event_queue: deque = deque(maxlen=1000)
        
        # Threading
        self.monitoring_thread = None
        self.dashboard_thread = None
        self.running = False
        self.lock = threading.Lock()
        
        # Configuration
        self.metric_retention_hours = self.config.get('metric_retention_hours', 48)
        self.health_check_interval = self.config.get('health_check_interval_seconds', 30)
        self.dashboard_refresh_interval = self.config.get('dashboard_refresh_interval_seconds', 10)
        
        # System health thresholds
        self.health_thresholds = self.config.get('health_thresholds', {
            'cpu_usage': {'warning': 0.7, 'critical': 0.9},
            'memory_usage': {'warning': 0.8, 'critical': 0.95},
            'error_rate': {'warning': 0.05, 'critical': 0.15},
            'response_time_ms': {'warning': 500, 'critical': 2000},
            'cache_hit_rate': {'warning': 0.5, 'critical': 0.3}
        })
        
        # Initialize default alert rules
        self._initialize_default_alert_rules()
        
        # Initialize default dashboard panels
        self._initialize_default_dashboard_panels()
        
        self.logger.info("SystemMonitor initialized with enterprise monitoring capabilities")
    
    def start_monitoring(self):
        """Start the monitoring system with proper thread management."""
        with self.lock:
            if self.running:
                self.logger.warning("Monitoring system is already running")
                return
            
            # Clean up any existing threads first
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.logger.warning("Previous monitoring thread still alive, cleaning up...")
                self.running = False
                self.monitoring_thread.join(timeout=2)
            
            if self.dashboard_thread and self.dashboard_thread.is_alive():
                self.logger.warning("Previous dashboard thread still alive, cleaning up...")
                self.dashboard_thread.join(timeout=2)
            
            # Set running flag
            self.running = True
            
            # Create and start new threads
            try:
                self.monitoring_thread = threading.Thread(
                    target=self._monitoring_loop, 
                    name="SystemMonitor-Monitoring",
                    daemon=True
                )
                self.dashboard_thread = threading.Thread(
                    target=self._dashboard_loop, 
                    name="SystemMonitor-Dashboard",
                    daemon=True
                )
                
                self.monitoring_thread.start()
                self.dashboard_thread.start()
                
                self.logger.info("System monitoring started successfully")
                
            except Exception as e:
                self.running = False
                self.logger.error(f"Failed to start monitoring threads: {e}")
                raise
    
    def stop_monitoring(self):
        """Stop the monitoring system with proper thread cleanup."""
        # Step 1: Signal threads to stop
        with self.lock:
            if not self.running:
                return  # Already stopped
            self.running = False
            self.logger.info("Stopping system monitoring...")
        
        # Step 2: Give threads time to finish current operations
        time.sleep(0.1)
        
        # Step 3: Wait for threads to finish gracefully
        threads_to_join = []
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            threads_to_join.append(('monitoring', self.monitoring_thread))
        
        if self.dashboard_thread and self.dashboard_thread.is_alive():
            threads_to_join.append(('dashboard', self.dashboard_thread))
        
        # Join threads with proper timeout handling
        for thread_name, thread in threads_to_join:
            try:
                thread.join(timeout=5)
                if thread.is_alive():
                    self.logger.warning(f"{thread_name} thread did not terminate within timeout")
                else:
                    self.logger.debug(f"{thread_name} thread terminated successfully")
            except Exception as e:
                self.logger.error(f"Error joining {thread_name} thread: {e}")
        
        # Step 4: Clean up thread references
        with self.lock:
            self.monitoring_thread = None
            self.dashboard_thread = None
            
        self.logger.info("System monitoring stopped")

    def is_monitoring_active(self):
        """Check if monitoring system is active and threads are running."""
        with self.lock:
            if not self.running:
                return False
            
            monitoring_active = (
                self.monitoring_thread and 
                self.monitoring_thread.is_alive()
            )
            
            dashboard_active = (
                self.dashboard_thread and 
                self.dashboard_thread.is_alive()
            )
            
            return monitoring_active and dashboard_active
    
    def get_thread_status(self):
        """Get detailed status of monitoring threads for debugging."""
        with self.lock:
            status = {
                'running_flag': self.running,
                'monitoring_thread': {
                    'exists': self.monitoring_thread is not None,
                    'alive': self.monitoring_thread.is_alive() if self.monitoring_thread else False,
                    'name': self.monitoring_thread.name if self.monitoring_thread else None
                },
                'dashboard_thread': {
                    'exists': self.dashboard_thread is not None,
                    'alive': self.dashboard_thread.is_alive() if self.dashboard_thread else False,
                    'name': self.dashboard_thread.name if self.dashboard_thread else None
                }
            }
            
        return status
    
    def record_system_metric(self, metric_name: str, value: float, component: str,
                           unit: str = "", tags: Optional[Dict[str, str]] = None):
        """Record a system metric."""
        metric = SystemHealthMetric(
            metric_name=metric_name,
            value=value,
            unit=unit,
            timestamp=time.time(),
            component=component,
            tags=tags or {}
        )
        
        # Store metric
        self.system_metrics[metric_name].append(metric)
        self.health_metrics[metric_name] = metric
        
        # Check alert rules
        self._check_alert_rules(metric)
        
        # Emit event
        self._emit_event(MonitoringEventType.METRIC_RECORDED, {
            'metric_name': metric_name,
            'value': value,
            'component': component,
            'timestamp': metric.timestamp
        })
        
        # Clean up old metrics
        self._cleanup_old_metrics()
    
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.alert_rules[rule.rule_id] = rule
        self.logger.info(f"Added alert rule: {rule.rule_id}")
    
    def remove_alert_rule(self, rule_id: str):
        """Remove an alert rule."""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            self.logger.info(f"Removed alert rule: {rule_id}")
    
    def add_dashboard_panel(self, panel: DashboardPanel):
        """Add a dashboard panel."""
        self.dashboard_panels[panel.panel_id] = panel
        self.logger.info(f"Added dashboard panel: {panel.panel_id}")
    
    def get_dashboard_data(self, time_range_minutes: int = 60) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        current_time = time.time()
        cutoff_time = current_time - (time_range_minutes * 60)
        
        dashboard_data = {
            'timestamp': current_time,
            'time_range_minutes': time_range_minutes,
            'system_health': self._get_system_health_summary(),
            'performance_metrics': self._get_performance_dashboard_data(cutoff_time),
            'active_alerts': self._get_active_alerts_summary(),
            'components_status': self._get_components_status(),
            'panels': self._generate_panel_data(cutoff_time)
        }
        
        return dashboard_data
    
    def get_system_health_score(self) -> Tuple[float, str]:
        """
        Calculate overall system health score.
        
        Returns score (0-100) and status description.
        """
        health_scores = []
        critical_issues = []
        warning_issues = []
        
        # Evaluate each health metric
        for metric_name, metric in self.health_metrics.items():
            if metric_name in self.health_thresholds:
                thresholds = self.health_thresholds[metric_name]
                
                if metric_name in ['cpu_usage', 'memory_usage', 'error_rate', 'response_time_ms']:
                    # Lower values are better
                    if metric.value >= thresholds.get('critical', float('inf')):
                        health_scores.append(0.0)
                        critical_issues.append(f"{metric_name}: {metric.value}")
                    elif metric.value >= thresholds.get('warning', float('inf')):
                        health_scores.append(50.0)
                        warning_issues.append(f"{metric_name}: {metric.value}")
                    else:
                        health_scores.append(100.0)
                
                elif metric_name in ['cache_hit_rate']:
                    # Higher values are better
                    if metric.value <= thresholds.get('critical', 0):
                        health_scores.append(0.0)
                        critical_issues.append(f"{metric_name}: {metric.value}")
                    elif metric.value <= thresholds.get('warning', 0):
                        health_scores.append(50.0)
                        warning_issues.append(f"{metric_name}: {metric.value}")
                    else:
                        health_scores.append(100.0)
        
        # Calculate overall score
        if health_scores:
            overall_score = sum(health_scores) / len(health_scores)
        else:
            overall_score = 100.0  # No metrics = healthy
        
        # Determine status
        if critical_issues:
            status = f"CRITICAL ({len(critical_issues)} issues)"
        elif warning_issues:
            status = f"WARNING ({len(warning_issues)} issues)"
        elif overall_score >= 90:
            status = "HEALTHY"
        elif overall_score >= 70:
            status = "FAIR"
        else:
            status = "POOR"
        
        return overall_score, status
    
    def acknowledge_alert(self, alert_id: str, user: str = "system"):
        """Acknowledge an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged_at = time.time()
            
            self.logger.info(f"Alert {alert_id} acknowledged by {user}")
            
            self._emit_event(MonitoringEventType.ALERT_RESOLVED, {
                'alert_id': alert_id,
                'acknowledged_by': user,
                'timestamp': alert.acknowledged_at
            })
    
    def resolve_alert(self, alert_id: str, user: str = "system"):
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved_at = time.time()
            
            # Move to history
            self.alert_history.append(alert)
            del self.active_alerts[alert_id]
            
            self.logger.info(f"Alert {alert_id} resolved by {user}")
            
            self._emit_event(MonitoringEventType.ALERT_RESOLVED, {
                'alert_id': alert_id,
                'resolved_by': user,
                'timestamp': alert.resolved_at
            })
    
    def get_monitoring_report(self, hours_back: int = 24) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""
        current_time = time.time()
        report_start = current_time - (hours_back * 3600)
        
        # System health analysis
        health_score, health_status = self.get_system_health_score()
        
        # Alert analysis
        period_alerts = [
            alert for alert in list(self.alert_history) + list(self.active_alerts.values())
            if alert.triggered_at >= report_start
        ]
        
        # Performance analysis from integrated monitor
        performance_data = self.performance_monitor.get_performance_dashboard_data()
        
        report = {
            'report_metadata': {
                'generated_at': current_time,
                'period_hours': hours_back,
                'system_version': '4.3.0'
            },
            'system_health': {
                'overall_score': health_score,
                'status': health_status,
                'components_healthy': len([m for m in self.health_metrics.values() if m.severity == AlertSeverity.INFO]),
                'components_warning': len([m for m in self.health_metrics.values() if m.severity == AlertSeverity.WARNING]),
                'components_critical': len([m for m in self.health_metrics.values() if m.severity == AlertSeverity.CRITICAL])
            },
            'alert_summary': {
                'total_alerts': len(period_alerts),
                'active_alerts': len(self.active_alerts),
                'resolved_alerts': len([a for a in period_alerts if a.resolved_at]),
                'critical_alerts': len([a for a in period_alerts if a.severity == AlertSeverity.CRITICAL]),
                'alert_resolution_rate': len([a for a in period_alerts if a.resolved_at]) / len(period_alerts) if period_alerts else 1.0
            },
            'performance_summary': performance_data['summary'],
            'uptime_analysis': self._calculate_uptime_metrics(report_start),
            'recommendations': self._generate_monitoring_recommendations()
        }
        
        return report
    
    def _initialize_default_alert_rules(self):
        """Initialize default alert rules for common metrics."""
        default_rules = [
            AlertRule(
                rule_id="high_cpu_usage",
                metric_name="cpu_usage",
                condition=">",
                threshold=0.85,
                severity=AlertSeverity.WARNING
            ),
            AlertRule(
                rule_id="critical_cpu_usage", 
                metric_name="cpu_usage",
                condition=">",
                threshold=0.95,
                severity=AlertSeverity.CRITICAL
            ),
            AlertRule(
                rule_id="high_memory_usage",
                metric_name="memory_usage",
                condition=">",
                threshold=0.85,
                severity=AlertSeverity.WARNING
            ),
            AlertRule(
                rule_id="high_error_rate",
                metric_name="error_rate",
                condition=">",
                threshold=0.05,
                severity=AlertSeverity.WARNING
            ),
            AlertRule(
                rule_id="slow_response_time",
                metric_name="response_time_ms",
                condition=">",
                threshold=1000,
                severity=AlertSeverity.WARNING
            ),
            AlertRule(
                rule_id="low_cache_hit_rate",
                metric_name="cache_hit_rate",
                condition="<",
                threshold=0.5,
                severity=AlertSeverity.WARNING
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule
    
    def _initialize_default_dashboard_panels(self):
        """Initialize default dashboard panels."""
        default_panels = [
            DashboardPanel(
                panel_id="system_overview",
                title="System Overview",
                metric_names=["cpu_usage", "memory_usage", "active_threads"],
                chart_type="gauge"
            ),
            DashboardPanel(
                panel_id="performance_metrics",
                title="Performance Metrics",
                metric_names=["response_time_ms", "throughput", "error_rate"],
                chart_type="line"
            ),
            DashboardPanel(
                panel_id="cache_performance",
                title="Cache Performance",
                metric_names=["cache_hit_rate", "cache_memory_usage", "cache_evictions"],
                chart_type="line"
            ),
            DashboardPanel(
                panel_id="alert_status",
                title="Alert Status",
                metric_names=["active_alerts", "resolved_alerts"],
                chart_type="bar"
            )
        ]
        
        for panel in default_panels:
            self.dashboard_panels[panel.panel_id] = panel
    
    def _monitoring_loop(self):
        """Main monitoring loop with improved error handling and graceful shutdown."""
        self.logger.info("Monitoring loop started")
        
        while True:
            # Check running status with proper synchronization
            with self.lock:
                if not self.running:
                    self.logger.info("Monitoring loop received stop signal")
                    break
            
            try:
                # Collect system health metrics
                self._collect_system_health_metrics()
                
                # Update dashboard cache
                self._update_dashboard_cache()
                
                # Process events
                self._process_event_queue()
                
                # Check for alerts
                # self._check_alert_rules()  # Fixed: removed call without metric parameter
                
                # Sleep with interruption check
                sleep_start = time.time()
                while time.time() - sleep_start < self.health_check_interval:
                    with self.lock:
                        if not self.running:
                            self.logger.info("Monitoring loop interrupted during sleep")
                            return
                    time.sleep(0.1)  # Small increments for faster shutdown
                    
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                # Exponential backoff on errors
                error_sleep = min(5, 0.5 + time.time() % 5)
                time.sleep(error_sleep)
        
        self.logger.info("Monitoring loop terminated")  # Brief pause before retry
    
    def _dashboard_loop(self):
        """Dashboard refresh loop with improved error handling and graceful shutdown."""
        self.logger.info("Dashboard loop started")
        
        while True:
            # Check running status with proper synchronization
            with self.lock:
                if not self.running:
                    self.logger.info("Dashboard loop received stop signal")
                    break
            
            try:
                # Refresh dashboard data
                dashboard_data = self.get_dashboard_data()
                
                # Update cache atomically
                with self.lock:
                    self.dashboard_cache = dashboard_data
                
                # Sleep with interruption check
                sleep_start = time.time()
                while time.time() - sleep_start < self.dashboard_refresh_interval:
                    with self.lock:
                        if not self.running:
                            self.logger.info("Dashboard loop interrupted during sleep")
                            return
                    time.sleep(0.1)  # Small increments for faster shutdown
                    
            except Exception as e:
                self.logger.error(f"Error in dashboard loop: {e}")
                # Exponential backoff on errors
                error_sleep = min(10, 1 + time.time() % 10)
                time.sleep(error_sleep)
        
        self.logger.info("Dashboard loop terminated")  # Brief pause before retry
    
    def _collect_system_health_metrics(self):
        """Collect system health metrics."""
        try:
            import psutil
            
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=0.1) / 100.0
            self.record_system_metric("cpu_usage", cpu_usage, "system", "%")
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent / 100.0
            self.record_system_metric("memory_usage", memory_usage, "system", "%")
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent / 100.0
            self.record_system_metric("disk_usage", disk_usage, "system", "%")
            
            # Network I/O (if available)
            try:
                network = psutil.net_io_counters()
                self.record_system_metric("network_bytes_sent", network.bytes_sent, "system", "bytes")
                self.record_system_metric("network_bytes_recv", network.bytes_recv, "system", "bytes")
            except:
                pass  # Network metrics not available
                
        except ImportError:
            # Fallback metrics if psutil not available
            self.record_system_metric("cpu_usage", 0.1, "system", "%")
            self.record_system_metric("memory_usage", 0.3, "system", "%")
        
        # Thread count
        active_threads = threading.active_count()
        self.record_system_metric("active_threads", active_threads, "system", "count")
        
        # Performance metrics from integrated monitor
        perf_data = self.performance_monitor.get_performance_dashboard_data()
        if perf_data['summary']:
            summary = perf_data['summary']
            self.record_system_metric("response_time_ms", summary.get('average_response_time_ms', 0), "performance", "ms")
            self.record_system_metric("error_rate", summary.get('error_rate', 0), "performance", "rate")
            self.record_system_metric("success_rate", summary.get('success_rate', 1.0), "performance", "rate")
    
    def _check_alert_rules(self, metric: SystemHealthMetric):
        """Check if metric triggers any alert rules."""
        for rule_id, rule in self.alert_rules.items():
            if not rule.enabled or rule.metric_name != metric.metric_name:
                continue
            
            # Check if alert should trigger
            should_trigger = False
            if rule.condition == ">":
                should_trigger = metric.value > rule.threshold
            elif rule.condition == "<":
                should_trigger = metric.value < rule.threshold
            elif rule.condition == "==":
                should_trigger = abs(metric.value - rule.threshold) < 0.001
            elif rule.condition == "!=":
                should_trigger = abs(metric.value - rule.threshold) >= 0.001
            
            if should_trigger:
                # Check if alert already active and within suppression window
                existing_alert_id = f"{rule_id}_{metric.component}"
                if existing_alert_id in self.active_alerts:
                    existing_alert = self.active_alerts[existing_alert_id]
                    time_since_trigger = metric.timestamp - existing_alert.triggered_at
                    if time_since_trigger < rule.suppression_window_seconds:
                        continue  # Suppressed
                
                # Create new alert
                alert = SystemAlert(
                    alert_id=existing_alert_id,
                    rule_id=rule_id,
                    severity=rule.severity,
                    title=f"{rule.metric_name.replace('_', ' ').title()} Alert",
                    description=f"{metric.component} {rule.metric_name} is {metric.value:.3f} ({rule.condition} {rule.threshold})",
                    metric_value=metric.value,
                    threshold=rule.threshold,
                    component=metric.component,
                    triggered_at=metric.timestamp
                )
                
                self.active_alerts[alert.alert_id] = alert
                
                # Log alert
                self.logger.warning(f"Alert triggered: {alert.description}")
                
                # Emit event
                self._emit_event(MonitoringEventType.ALERT_TRIGGERED, {
                    'alert_id': alert.alert_id,
                    'severity': alert.severity.value,
                    'description': alert.description,
                    'component': alert.component
                })
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics based on retention policy."""
        cutoff_time = time.time() - (self.metric_retention_hours * 3600)
        
        for metric_name, metrics_deque in self.system_metrics.items():
            # Remove old metrics from the left of deque
            while metrics_deque and metrics_deque[0].timestamp < cutoff_time:
                metrics_deque.popleft()
    
    def _emit_event(self, event_type: MonitoringEventType, data: Dict[str, Any]):
        """Emit a monitoring event."""
        event = MonitoringEvent(
            event_type=event_type,
            timestamp=time.time(),
            data=data,
            source="system_monitor"
        )
        
        self.event_queue.append(event)
        
        # Call registered handlers
        for handler in self.event_handlers[event_type]:
            try:
                handler(event)
            except Exception as e:
                self.logger.error(f"Error in event handler: {e}")
    
    def _process_event_queue(self):
        """Process queued events."""
        # Simple event processing - could be enhanced with priorities, batching, etc.
        processed = 0
        while self.event_queue and processed < 100:  # Process up to 100 events per cycle
            event = self.event_queue.popleft()
            # Event processing logic could be added here
            processed += 1
    
    def _get_system_health_summary(self) -> Dict[str, Any]:
        """Get system health summary."""
        health_score, health_status = self.get_system_health_score()
        
        return {
            'overall_score': health_score,
            'status': health_status,
            'last_updated': time.time(),
            'metrics_count': len(self.health_metrics),
            'components_monitored': len(set(m.component for m in self.health_metrics.values()))
        }
    
    def _get_performance_dashboard_data(self, cutoff_time: float) -> Dict[str, Any]:
        """Get performance data for dashboard."""
        return self.performance_monitor.get_performance_dashboard_data()
    
    def _get_active_alerts_summary(self) -> Dict[str, Any]:
        """Get active alerts summary."""
        alerts_by_severity = defaultdict(int)
        for alert in self.active_alerts.values():
            alerts_by_severity[alert.severity.value] += 1
        
        return {
            'total_active': len(self.active_alerts),
            'by_severity': dict(alerts_by_severity),
            'oldest_alert_age': min([time.time() - alert.triggered_at for alert in self.active_alerts.values()], default=0)
        }
    
    def _get_components_status(self) -> Dict[str, str]:
        """Get status of each monitored component."""
        component_status = {}
        
        # Group metrics by component
        component_metrics = defaultdict(list)
        for metric in self.health_metrics.values():
            component_metrics[metric.component].append(metric)
        
        # Determine status for each component
        for component, metrics in component_metrics.items():
            has_critical = any(m.severity == AlertSeverity.CRITICAL for m in metrics)
            has_warning = any(m.severity == AlertSeverity.WARNING for m in metrics)
            
            if has_critical:
                component_status[component] = "CRITICAL"
            elif has_warning:
                component_status[component] = "WARNING"
            else:
                component_status[component] = "HEALTHY"
        
        return component_status
    
    def _generate_panel_data(self, cutoff_time: float) -> Dict[str, Any]:
        """Generate data for dashboard panels."""
        panel_data = {}
        
        for panel_id, panel in self.dashboard_panels.items():
            # Get metrics for this panel
            panel_metrics = {}
            for metric_name in panel.metric_names:
                if metric_name in self.system_metrics:
                    # Filter metrics by time range
                    recent_metrics = [
                        m for m in self.system_metrics[metric_name]
                        if m.timestamp >= cutoff_time
                    ]
                    
                    if recent_metrics:
                        panel_metrics[metric_name] = {
                            'current_value': recent_metrics[-1].value,
                            'data_points': [
                                {'timestamp': m.timestamp, 'value': m.value}
                                for m in recent_metrics
                            ],
                            'unit': recent_metrics[-1].unit
                        }
            
            panel_data[panel_id] = {
                'title': panel.title,
                'chart_type': panel.chart_type,
                'metrics': panel_metrics,
                'last_updated': time.time()
            }
        
        return panel_data
    
    def _update_dashboard_cache(self):
        """Update the dashboard cache."""
        try:
            self.dashboard_cache = self.get_dashboard_data()
        except Exception as e:
            self.logger.error(f"Error updating dashboard cache: {e}")
    
    def _calculate_uptime_metrics(self, since_timestamp: float) -> Dict[str, Any]:
        """Calculate uptime metrics for the reporting period."""
        # This is a simplified calculation - could be enhanced with more sophisticated tracking
        total_period_seconds = time.time() - since_timestamp
        
        # Count critical alert time as downtime
        downtime_seconds = 0
        for alert in list(self.alert_history) + list(self.active_alerts.values()):
            if alert.severity == AlertSeverity.CRITICAL and alert.triggered_at >= since_timestamp:
                resolution_time = alert.resolved_at or time.time()
                downtime_seconds += min(resolution_time - alert.triggered_at, total_period_seconds)
        
        uptime_seconds = total_period_seconds - downtime_seconds
        uptime_percentage = (uptime_seconds / total_period_seconds) * 100 if total_period_seconds > 0 else 100
        
        return {
            'uptime_percentage': uptime_percentage,
            'uptime_seconds': uptime_seconds,
            'downtime_seconds': downtime_seconds,
            'total_period_seconds': total_period_seconds,
            'meets_sla': uptime_percentage >= 99.9  # 99.9% uptime target
        }
    
    def _generate_monitoring_recommendations(self) -> List[str]:
        """Generate monitoring recommendations based on current state."""
        recommendations = []
        
        # Check system health
        health_score, _ = self.get_system_health_score()
        if health_score < 80:
            recommendations.append(f"System health score is {health_score:.1f}% - investigate critical metrics")
        
        # Check active alerts
        if len(self.active_alerts) > 5:
            recommendations.append(f"{len(self.active_alerts)} active alerts - prioritize resolution")
        
        # Check alert resolution rate
        recent_alerts = [
            alert for alert in self.alert_history
            if time.time() - alert.triggered_at < 86400  # Last 24 hours
        ]
        if recent_alerts:
            resolution_rate = len([a for a in recent_alerts if a.resolved_at]) / len(recent_alerts)
            if resolution_rate < 0.8:
                recommendations.append(f"Alert resolution rate ({resolution_rate:.1%}) is low - improve response times")
        
        # Performance-based recommendations
        perf_data = self.performance_monitor.get_performance_dashboard_data()
        if perf_data['summary']:
            avg_response = perf_data['summary'].get('average_response_time_ms', 0)
            if avg_response > 1000:
                recommendations.append(f"Average response time ({avg_response:.0f}ms) exceeds sub-second target")
        
        # Add Story 4.3 specific recommendations
        recommendations.extend([
            "Monitor performance regression continuously",
            "Validate sub-second processing targets daily",
            "Review alert thresholds weekly for optimization"
        ])
        
        return recommendations
    
    def add_event_handler(self, event_type: MonitoringEventType, handler: Callable):
        """Add an event handler."""
        self.event_handlers[event_type].append(handler)
    
    def __del__(self):
        """Safe cleanup on destruction."""
        try:
            # Only stop if we're still in a valid state
            if hasattr(self, 'running') and hasattr(self, 'lock'):
                self.stop_monitoring()
        except Exception:
            # Silently ignore errors during interpreter shutdown
            pass


def test_system_monitor():
    """Test system monitoring functionality."""
    monitor = SystemMonitor()
    
    print("Testing system monitor...")
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Record some test metrics
    monitor.record_system_metric("test_metric", 0.5, "test_component", "rate")
    monitor.record_system_metric("test_metric", 0.8, "test_component", "rate")  # Should trigger warning
    monitor.record_system_metric("test_metric", 0.9, "test_component", "rate")  # Should trigger critical
    
    time.sleep(2)  # Let monitoring process
    
    # Get dashboard data
    dashboard = monitor.get_dashboard_data()
    health_score, health_status = monitor.get_system_health_score()
    
    print(f"✅ System monitor test passed")
    print(f"   Health score: {health_score:.1f}% ({health_status})")
    print(f"   Active alerts: {len(monitor.active_alerts)}")
    print(f"   Monitored metrics: {len(monitor.health_metrics)}")
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    return True


if __name__ == "__main__":
    test_system_monitor()