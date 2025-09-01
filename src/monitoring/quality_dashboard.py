"""
Quality Assurance Dashboard for Story 4.

This module provides real-time quality metrics visualization and monitoring
dashboard functionality for the Sanskrit processing system.
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
import threading
from collections import deque

from .performance_monitor import PerformanceMonitor, PerformanceMetrics
from ..utils.metrics_collector import MetricsCollector, ProcessingMetrics


@dataclass
class DashboardMetrics:
    """Real-time metrics for quality dashboard display."""
    
    # Current session metrics
    active_session_id: Optional[str] = None
    session_start_time: Optional[str] = None
    total_files_processed: int = 0
    files_in_progress: int = 0
    
    # Processing performance
    current_processing_rate: float = 0.0  # files per minute
    average_processing_time: float = 0.0  # seconds per file
    peak_processing_time: float = 0.0
    
    # Quality metrics
    current_quality_score: float = 0.0
    average_confidence: float = 0.0
    total_corrections_made: int = 0
    correction_types: Dict[str, int] = field(default_factory=dict)
    
    # Advanced vs fallback usage
    advanced_pipeline_successes: int = 0
    fallback_usage_count: int = 0
    advanced_usage_rate: float = 0.0
    
    # Alerts and warnings
    active_alerts: List[str] = field(default_factory=list)
    warning_count: int = 0
    error_count: int = 0
    
    # System health
    system_cpu_usage: float = 0.0
    system_memory_usage: float = 0.0
    component_health_status: Dict[str, str] = field(default_factory=dict)
    
    # Historical trends (last 24 hours)
    processing_rate_history: List[float] = field(default_factory=lambda: deque(maxlen=288))  # 5-minute intervals
    quality_score_history: List[float] = field(default_factory=lambda: deque(maxlen=288))
    error_rate_history: List[float] = field(default_factory=lambda: deque(maxlen=288))


@dataclass
class AlertRule:
    """Configuration for dashboard alerts."""
    
    name: str
    metric_path: str  # dot-separated path like 'performance.processing_time_per_subtitle'
    threshold: float
    comparison: str  # 'greater_than', 'less_than', 'equals'
    severity: str  # 'info', 'warning', 'critical'
    message_template: str
    cooldown_minutes: int = 5  # minimum time between same alerts
    
    def __post_init__(self):
        self.last_triggered: Optional[datetime] = None


class QualityDashboard:
    """
    Real-time quality assurance dashboard with metrics visualization.
    
    Provides live monitoring of processing quality, performance metrics,
    and system health for Story 4 requirements (AC4).
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the quality dashboard."""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Initialize components
        self.performance_monitor = PerformanceMonitor(config)
        self.metrics_collector = MetricsCollector(config)
        
        # Dashboard state
        self.dashboard_metrics = DashboardMetrics()
        self.is_running = False
        self.update_thread = None
        self.update_interval = self.config.get('dashboard_update_interval', 5.0)  # seconds
        
        # Metrics history for trends
        self.metrics_history: deque = deque(maxlen=1000)  # Keep last 1000 updates
        
        # Alert system
        self.alert_rules = self._initialize_alert_rules()
        self.active_alerts: Dict[str, datetime] = {}
        
        # Dashboard callbacks for real-time updates
        self.update_callbacks: List[Callable[[DashboardMetrics], None]] = []
        
        # Web interface configuration
        self.web_port = self.config.get('dashboard_port', 8080)
        self.web_host = self.config.get('dashboard_host', '127.0.0.1')
        
    def _initialize_alert_rules(self) -> List[AlertRule]:
        """Initialize alert rules for quality monitoring."""
        default_rules = [
            AlertRule(
                name="Processing Time Exceeded",
                metric_path="average_processing_time",
                threshold=2.0,
                comparison="greater_than",
                severity="warning",
                message_template="Processing time ({value:.2f}s) exceeds target (2.0s)",
                cooldown_minutes=5
            ),
            AlertRule(
                name="Quality Score Low",
                metric_path="current_quality_score",
                threshold=0.95,
                comparison="less_than",
                severity="warning",
                message_template="Quality score ({value:.3f}) below acceptable threshold (0.95)",
                cooldown_minutes=10
            ),
            AlertRule(
                name="Advanced Pipeline Usage Low",
                metric_path="advanced_usage_rate",
                threshold=0.90,
                comparison="less_than",
                severity="critical",
                message_template="Advanced pipeline usage ({value:.1%}) below target (90%)",
                cooldown_minutes=15
            ),
            AlertRule(
                name="High Error Rate",
                metric_path="error_count",
                threshold=10,
                comparison="greater_than",
                severity="critical",
                message_template="Error count ({value}) is high - investigate immediately",
                cooldown_minutes=5
            ),
            AlertRule(
                name="System Memory High",
                metric_path="system_memory_usage",
                threshold=85.0,
                comparison="greater_than",
                severity="warning",
                message_template="System memory usage ({value:.1f}%) is high",
                cooldown_minutes=10
            )
        ]
        
        return default_rules
    
    def start_dashboard(self, session_id: Optional[str] = None) -> str:
        """
        Start the quality dashboard monitoring.
        
        Args:
            session_id: Optional session ID to monitor
            
        Returns:
            Session ID being monitored
        """
        if self.is_running:
            self.logger.warning("Dashboard is already running")
            return self.dashboard_metrics.active_session_id or ""
        
        # Start performance monitoring session
        session_id = self.performance_monitor.start_performance_session(session_id)
        
        # Initialize dashboard metrics
        self.dashboard_metrics.active_session_id = session_id
        self.dashboard_metrics.session_start_time = datetime.now(timezone.utc).isoformat()
        
        # Start update thread
        self.is_running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        self.logger.info(f"Quality dashboard started for session: {session_id}")
        return session_id
    
    def stop_dashboard(self) -> Optional[PerformanceMetrics]:
        """
        Stop the quality dashboard monitoring.
        
        Returns:
            Final performance metrics
        """
        if not self.is_running:
            self.logger.warning("Dashboard is not running")
            return None
        
        self.is_running = False
        
        if self.update_thread:
            self.update_thread.join(timeout=10)
        
        # Get final performance metrics
        final_metrics = self.performance_monitor.end_performance_session()
        
        self.logger.info(f"Quality dashboard stopped for session: {self.dashboard_metrics.active_session_id}")
        
        # Reset dashboard state
        self.dashboard_metrics = DashboardMetrics()
        
        return final_metrics
    
    def add_update_callback(self, callback: Callable[[DashboardMetrics], None]) -> None:
        """Add a callback function for real-time dashboard updates."""
        self.update_callbacks.append(callback)
        self.logger.debug("Added dashboard update callback")
    
    def remove_update_callback(self, callback: Callable[[DashboardMetrics], None]) -> None:
        """Remove a callback function."""
        if callback in self.update_callbacks:
            self.update_callbacks.remove(callback)
            self.logger.debug("Removed dashboard update callback")
    
    def get_current_metrics(self) -> DashboardMetrics:
        """Get current dashboard metrics."""
        return self.dashboard_metrics
    
    def get_metrics_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """
        Get historical metrics for the specified time period.
        
        Args:
            hours: Number of hours of history to return
            
        Returns:
            List of historical metric snapshots
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        history = []
        for metrics_snapshot in self.metrics_history:
            if datetime.fromisoformat(metrics_snapshot['timestamp'].replace('Z', '+00:00')) >= cutoff_time:
                history.append(metrics_snapshot)
        
        return history
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive quality assurance report.
        
        Returns:
            Dictionary containing quality metrics and analysis
        """
        current_metrics = self.dashboard_metrics
        
        # Calculate additional metrics
        total_processed = current_metrics.advanced_pipeline_successes + current_metrics.fallback_usage_count
        success_rate = (current_metrics.advanced_pipeline_successes / total_processed * 100) if total_processed > 0 else 0
        
        # Generate trend analysis
        recent_history = self.get_metrics_history(hours=1)
        trend_analysis = self._analyze_trends(recent_history)
        
        report = {
            'quality_summary': {
                'session_id': current_metrics.active_session_id,
                'session_duration_minutes': self._calculate_session_duration(),
                'total_files_processed': current_metrics.total_files_processed,
                'files_in_progress': current_metrics.files_in_progress,
                'current_quality_score': f"{current_metrics.current_quality_score:.3f}",
                'average_confidence': f"{current_metrics.average_confidence:.3f}",
                'total_corrections_made': current_metrics.total_corrections_made
            },
            'performance_metrics': {
                'current_processing_rate_files_per_minute': f"{current_metrics.current_processing_rate:.2f}",
                'average_processing_time_seconds': f"{current_metrics.average_processing_time:.2f}",
                'peak_processing_time_seconds': f"{current_metrics.peak_processing_time:.2f}",
                'advanced_pipeline_success_rate': f"{success_rate:.1f}%",
                'meets_performance_targets': success_rate >= 90 and current_metrics.average_processing_time <= 2.0
            },
            'correction_analysis': {
                'correction_types_breakdown': current_metrics.correction_types.copy(),
                'most_frequent_correction': max(current_metrics.correction_types.items(), 
                                              key=lambda x: x[1], default=("none", 0))[0],
                'correction_rate_per_file': (current_metrics.total_corrections_made / current_metrics.total_files_processed) 
                                          if current_metrics.total_files_processed > 0 else 0
            },
            'system_health': {
                'active_alerts_count': len(current_metrics.active_alerts),
                'warning_count': current_metrics.warning_count,
                'error_count': current_metrics.error_count,
                'system_cpu_usage_percent': f"{current_metrics.system_cpu_usage:.1f}%",
                'system_memory_usage_percent': f"{current_metrics.system_memory_usage:.1f}%",
                'component_health_summary': current_metrics.component_health_status.copy()
            },
            'trend_analysis': trend_analysis,
            'alert_summary': {
                'active_alerts': current_metrics.active_alerts.copy(),
                'alert_rules_configured': len(self.alert_rules),
                'recent_alert_activity': self._get_recent_alerts()
            },
            'recommendations': self._generate_quality_recommendations(current_metrics),
            'report_metadata': {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'dashboard_uptime_minutes': self._calculate_session_duration(),
                'data_freshness_seconds': self.update_interval
            }
        }
        
        return report
    
    def export_quality_report(self, format_type: str = 'json') -> Path:
        """
        Export quality report to file.
        
        Args:
            format_type: Export format ('json' or 'html')
            
        Returns:
            Path to exported report file
        """
        report = self.generate_quality_report()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format_type.lower() == 'html':
            # Generate HTML report
            report_content = self._generate_html_report(report)
            filename = f"quality_dashboard_report_{timestamp}.html"
        else:
            # Default to JSON
            report_content = json.dumps(report, indent=2, ensure_ascii=False)
            filename = f"quality_dashboard_report_{timestamp}.json"
        
        report_path = Path(f"data/reports/{filename}")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"Quality report exported to: {report_path}")
        return report_path
    
    def _update_loop(self) -> None:
        """Main update loop for dashboard metrics."""
        while self.is_running:
            try:
                self._update_dashboard_metrics()
                self._check_alert_rules()
                self._notify_update_callbacks()
                
                # Store metrics snapshot for history
                metrics_snapshot = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'metrics': asdict(self.dashboard_metrics)
                }
                self.metrics_history.append(metrics_snapshot)
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in dashboard update loop: {e}")
                time.sleep(self.update_interval)
    
    def _update_dashboard_metrics(self) -> None:
        """Update current dashboard metrics from various sources."""
        import psutil
        
        # Get current session from metrics collector
        current_session = self.metrics_collector.current_session
        if current_session:
            self.dashboard_metrics.total_files_processed = current_session.total_files_processed
            self.dashboard_metrics.total_corrections_made = current_session.total_corrections_applied
            
            # Calculate processing rate
            if current_session.start_time:
                session_start = datetime.fromisoformat(current_session.start_time.replace('Z', '+00:00'))
                session_duration = datetime.now(timezone.utc) - session_start
                duration_minutes = session_duration.total_seconds() / 60
                if duration_minutes > 0:
                    self.dashboard_metrics.current_processing_rate = current_session.total_files_processed / duration_minutes
            
            # Update quality metrics
            if current_session.file_metrics:
                all_confidences = []
                all_corrections = {}
                processing_times = []
                
                for fm in current_session.file_metrics:
                    all_confidences.extend(fm.confidence_scores)
                    processing_times.append(fm.processing_time)
                    
                    for correction_type, count in fm.corrections_applied.items():
                        all_corrections[correction_type] = all_corrections.get(correction_type, 0) + count
                
                if all_confidences:
                    self.dashboard_metrics.average_confidence = sum(all_confidences) / len(all_confidences)
                    self.dashboard_metrics.current_quality_score = self.dashboard_metrics.average_confidence
                
                if processing_times:
                    self.dashboard_metrics.average_processing_time = sum(processing_times) / len(processing_times)
                    self.dashboard_metrics.peak_processing_time = max(processing_times)
                
                self.dashboard_metrics.correction_types = all_corrections
        
        # Update system health metrics
        self.dashboard_metrics.system_cpu_usage = psutil.cpu_percent()
        self.dashboard_metrics.system_memory_usage = psutil.virtual_memory().percent
        
        # Update component health status
        self._update_component_health()
    
    def _update_component_health(self) -> None:
        """Update component health status."""
        try:
            # Check various system components
            health_status = {}
            
            # Check if main processing components are available
            try:
                from ..post_processors.sanskrit_post_processor import SanskritPostProcessor
                health_status['sanskrit_processor'] = 'healthy'
            except ImportError:
                health_status['sanskrit_processor'] = 'unavailable'
            
            try:
                from ..utils.metrics_collector import MetricsCollector
                health_status['metrics_collector'] = 'healthy'
            except ImportError:
                health_status['metrics_collector'] = 'unavailable'
            
            try:
                from ..monitoring.performance_monitor import PerformanceMonitor
                health_status['performance_monitor'] = 'healthy'
            except ImportError:
                health_status['performance_monitor'] = 'unavailable'
            
            self.dashboard_metrics.component_health_status = health_status
            
        except Exception as e:
            self.logger.error(f"Error updating component health: {e}")
    
    def _check_alert_rules(self) -> None:
        """Check alert rules and trigger alerts if necessary."""
        current_time = datetime.now(timezone.utc)
        
        for rule in self.alert_rules:
            try:
                # Get metric value using dot notation
                metric_value = self._get_metric_value(rule.metric_path)
                if metric_value is None:
                    continue
                
                # Check if alert condition is met
                should_trigger = False
                if rule.comparison == 'greater_than':
                    should_trigger = metric_value > rule.threshold
                elif rule.comparison == 'less_than':
                    should_trigger = metric_value < rule.threshold
                elif rule.comparison == 'equals':
                    should_trigger = metric_value == rule.threshold
                
                # Check cooldown period
                last_triggered = rule.last_triggered
                if should_trigger and (not last_triggered or 
                                     (current_time - last_triggered).total_seconds() >= rule.cooldown_minutes * 60):
                    # Trigger alert
                    alert_message = rule.message_template.format(value=metric_value)
                    self._trigger_alert(rule, alert_message)
                    rule.last_triggered = current_time
                
            except Exception as e:
                self.logger.error(f"Error checking alert rule '{rule.name}': {e}")
    
    def _get_metric_value(self, metric_path: str) -> Optional[float]:
        """Get metric value using dot notation path."""
        try:
            value = self.dashboard_metrics
            for part in metric_path.split('.'):
                value = getattr(value, part)
            return float(value) if value is not None else None
        except (AttributeError, ValueError, TypeError):
            return None
    
    def _trigger_alert(self, rule: AlertRule, message: str) -> None:
        """Trigger an alert."""
        alert_info = f"[{rule.severity.upper()}] {rule.name}: {message}"
        
        # Add to active alerts
        if alert_info not in self.dashboard_metrics.active_alerts:
            self.dashboard_metrics.active_alerts.append(alert_info)
        
        # Update counters
        if rule.severity in ['warning', 'critical']:
            self.dashboard_metrics.warning_count += 1 if rule.severity == 'warning' else 0
            self.dashboard_metrics.error_count += 1 if rule.severity == 'critical' else 0
        
        # Log alert
        log_method = self.logger.warning if rule.severity == 'warning' else self.logger.critical
        log_method(f"Dashboard Alert: {alert_info}")
    
    def _notify_update_callbacks(self) -> None:
        """Notify all registered update callbacks."""
        for callback in self.update_callbacks:
            try:
                callback(self.dashboard_metrics)
            except Exception as e:
                self.logger.error(f"Error in dashboard update callback: {e}")
    
    def _calculate_session_duration(self) -> float:
        """Calculate session duration in minutes."""
        if not self.dashboard_metrics.session_start_time:
            return 0.0
        
        start_time = datetime.fromisoformat(self.dashboard_metrics.session_start_time.replace('Z', '+00:00'))
        duration = datetime.now(timezone.utc) - start_time
        return duration.total_seconds() / 60
    
    def _analyze_trends(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends from historical data."""
        if len(history) < 2:
            return {'status': 'insufficient_data', 'message': 'Need at least 2 data points for trend analysis'}
        
        # Extract trend data
        processing_rates = [h['metrics'].get('current_processing_rate', 0) for h in history]
        quality_scores = [h['metrics'].get('current_quality_score', 0) for h in history]
        error_counts = [h['metrics'].get('error_count', 0) for h in history]
        
        # Calculate trends
        def calculate_trend(values):
            if len(values) < 2:
                return 'stable'
            recent_avg = sum(values[-5:]) / len(values[-5:]) if len(values) >= 5 else sum(values) / len(values)
            older_avg = sum(values[:5]) / 5 if len(values) >= 10 else sum(values[:-5]) / len(values[:-5]) if len(values) > 5 else recent_avg
            
            if recent_avg > older_avg * 1.1:
                return 'improving'
            elif recent_avg < older_avg * 0.9:
                return 'degrading'
            else:
                return 'stable'
        
        return {
            'processing_rate_trend': calculate_trend(processing_rates),
            'quality_score_trend': calculate_trend(quality_scores),
            'error_rate_trend': calculate_trend(error_counts),
            'data_points_analyzed': len(history),
            'trend_analysis_period_minutes': self._calculate_session_duration()
        }
    
    def _get_recent_alerts(self) -> List[Dict[str, Any]]:
        """Get recent alert activity."""
        # This would typically query a more persistent alert store
        # For now, return current active alerts with timestamps
        return [
            {
                'alert': alert,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'status': 'active'
            }
            for alert in self.dashboard_metrics.active_alerts
        ]
    
    def _generate_quality_recommendations(self, metrics: DashboardMetrics) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        # Performance recommendations
        if metrics.average_processing_time > 2.0:
            recommendations.append("Processing time exceeds target - consider optimizing Sanskrit processing pipeline")
        
        if metrics.advanced_usage_rate < 0.90:
            recommendations.append("Advanced pipeline usage is low - investigate component failures")
        
        # Quality recommendations
        if metrics.current_quality_score < 0.95:
            recommendations.append("Quality score below target - review correction accuracy and confidence scoring")
        
        if metrics.error_count > 5:
            recommendations.append("High error count detected - review error logs and system health")
        
        # System health recommendations
        if metrics.system_memory_usage > 80:
            recommendations.append("High memory usage detected - monitor for memory leaks")
        
        if metrics.system_cpu_usage > 85:
            recommendations.append("High CPU usage - consider load balancing or optimization")
        
        if not recommendations:
            recommendations.append("System is performing well - maintain current configuration")
        
        return recommendations
    
    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML format quality report."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Quality Dashboard Report</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .section {{ margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 5px; }}
        .metric {{ margin: 5px 0; }}
        .good {{ color: #27ae60; }}
        .warning {{ color: #f39c12; }}
        .error {{ color: #e74c3c; }}
        .recommendation {{ background: #e8f4f8; padding: 10px; margin: 5px 0; border-radius: 3px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Quality Dashboard Report</h1>
        <p>Generated: {report['report_metadata']['generated_at']}</p>
        <p>Session: {report['quality_summary']['session_id']}</p>
    </div>

    <div class="section">
        <h2>Quality Summary</h2>
        <div class="metric">Files Processed: <strong>{report['quality_summary']['total_files_processed']}</strong></div>
        <div class="metric">Quality Score: <strong>{report['quality_summary']['current_quality_score']}</strong></div>
        <div class="metric">Total Corrections: <strong>{report['quality_summary']['total_corrections_made']}</strong></div>
    </div>

    <div class="section">
        <h2>Performance Metrics</h2>
        <div class="metric">Processing Rate: <strong>{report['performance_metrics']['current_processing_rate_files_per_minute']} files/min</strong></div>
        <div class="metric">Average Processing Time: <strong>{report['performance_metrics']['average_processing_time_seconds']}s</strong></div>
        <div class="metric">Advanced Pipeline Success Rate: <strong>{report['performance_metrics']['advanced_pipeline_success_rate']}</strong></div>
    </div>

    <div class="section">
        <h2>System Health</h2>
        <div class="metric">Active Alerts: <span class="{'error' if report['system_health']['active_alerts_count'] > 0 else 'good'}">{report['system_health']['active_alerts_count']}</span></div>
        <div class="metric">CPU Usage: <strong>{report['system_health']['system_cpu_usage_percent']}</strong></div>
        <div class="metric">Memory Usage: <strong>{report['system_health']['system_memory_usage_percent']}</strong></div>
    </div>

    <div class="section">
        <h2>Recommendations</h2>
"""
        
        for i, rec in enumerate(report['recommendations'], 1):
            html += f'        <div class="recommendation">{i}. {rec}</div>\n'
        
        html += """    </div>
</body>
</html>"""
        
        return html