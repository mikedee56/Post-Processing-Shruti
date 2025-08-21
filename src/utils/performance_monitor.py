"""
Performance Monitoring and Telemetry System for Story 4.1
Comprehensive performance tracking and regression detection for MCP infrastructure.
"""

import time
import logging
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from enum import Enum
from collections import defaultdict, deque


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricType(Enum):
    """Types of metrics being tracked."""
    RESPONSE_TIME = "response_time"
    SUCCESS_RATE = "success_rate"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    FALLBACK_RATE = "fallback_rate"
    MEMORY_USAGE = "memory_usage"


@dataclass
class PerformanceMetric:
    """Individual performance metric data point."""
    metric_type: MetricType
    value: float
    timestamp: float
    component: str  # Which component generated this metric
    tags: Dict[str, str] = field(default_factory=dict)
    context: Optional[str] = None


@dataclass
class PerformanceAlert:
    """Performance alert with escalation and resolution tracking."""
    alert_id: str
    severity: AlertSeverity
    metric_type: MetricType
    component: str
    threshold_value: float
    current_value: float
    description: str
    triggered_at: float
    resolved_at: Optional[float] = None
    auto_resolved: bool = False
    escalation_count: int = 0


@dataclass
class RegressionDetectionResult:
    """Results from performance regression detection."""
    regression_detected: bool
    affected_metrics: List[MetricType]
    severity_level: AlertSeverity
    baseline_comparison: Dict[str, float]
    recommendations: List[str]
    confidence_score: float


class PerformanceMonitor:
    """
    Enterprise-grade performance monitoring system for MCP infrastructure.
    
    Features:
    - Real-time metric collection and analysis
    - Automated regression detection using statistical methods
    - Configurable alerting with severity levels
    - Performance baseline establishment and tracking
    - Comprehensive reporting and visualization data
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize performance monitor with enterprise configuration."""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Metric storage with time-based retention
        self.metrics: Dict[MetricType, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.alerts: List[PerformanceAlert] = []
        
        # Performance baselines (established over time)
        self.baselines: Dict[str, Dict[MetricType, float]] = {}
        
        # Configuration
        self.retention_hours = self.config.get('metrics_retention_hours', 24)
        self.baseline_window_hours = self.config.get('baseline_window_hours', 1)
        self.regression_detection_window = self.config.get('regression_detection_window_minutes', 30)
        
        # Alert thresholds (customizable per deployment)
        self.alert_thresholds = self.config.get('alert_thresholds', self._get_default_thresholds())
        
        # Performance targets from Story 4.1
        self.performance_targets = {
            'processing_time_seconds': 1.0,  # <1 second target
            'success_rate': 0.999,           # 99.9% uptime target
            'error_rate': 0.001,             # <0.1% error rate
        }
        
        # Alert suppression (prevent spam)
        self.alert_suppression: Dict[str, float] = {}
        self.suppression_window_seconds = self.config.get('alert_suppression_seconds', 300)  # 5 minutes
        
        self.logger.info("PerformanceMonitor initialized with enterprise-grade monitoring")
    
    def _get_default_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Get default alert thresholds for different metrics."""
        return {
            'response_time_ms': {
                'WARNING': 500,
                'CRITICAL': 1000,
                'EMERGENCY': 5000
            },
            'success_rate': {
                'WARNING': 0.95,   # Below 95%
                'CRITICAL': 0.90,  # Below 90%
                'EMERGENCY': 0.80  # Below 80%
            },
            'error_rate': {
                'WARNING': 0.05,   # Above 5%
                'CRITICAL': 0.15,  # Above 15%
                'EMERGENCY': 0.30  # Above 30%
            },
            'fallback_rate': {
                'WARNING': 0.10,   # Above 10%
                'CRITICAL': 0.25,  # Above 25%
                'EMERGENCY': 0.50  # Above 50%
            }
        }
    
    def add_metric_threshold(self, metric_type: MetricType, threshold: float, severity: AlertSeverity):
        """Add or update a metric threshold for alerting.
        
        Args:
            metric_type: The type of metric to monitor
            threshold: The threshold value for triggering alerts
            severity: The severity level for the alert
        """
        # Convert MetricType enum to string key for storage
        metric_key = metric_type.value
        
        # Initialize if not exists
        if metric_key not in self.alert_thresholds:
            self.alert_thresholds[metric_key] = {}
        
        # Store the threshold
        self.alert_thresholds[metric_key][severity.value.upper()] = threshold
        
        self.logger.debug(f"Added {severity.value} threshold for {metric_key}: {threshold}")
    
    def record_metric(self, metric_type: MetricType, value: float, component: str, 
                     tags: Optional[Dict[str, str]] = None, context: Optional[str] = None):
        """Record a performance metric for analysis."""
        metric = PerformanceMetric(
            metric_type=metric_type,
            value=value,
            timestamp=time.time(),
            component=component,
            tags=tags or {},
            context=context
        )
        
        self.metrics[metric_type].append(metric)
        
        # Check for immediate alerts
        self._check_metric_for_alerts(metric)
        
        # Clean up old metrics
        self._cleanup_old_metrics()
    
    def _cleanup_old_metrics(self):
        """Remove metrics older than retention period."""
        cutoff_time = time.time() - (self.retention_hours * 3600)
        
        for metric_type, metrics_deque in self.metrics.items():
            # Remove old metrics from the left of deque
            while metrics_deque and metrics_deque[0].timestamp < cutoff_time:
                metrics_deque.popleft()
    
    def _check_metric_for_alerts(self, metric: PerformanceMetric):
        """Check if a metric triggers any performance alerts."""
        metric_key = metric.metric_type.value
        
        if metric_key not in self.alert_thresholds:
            return
        
        thresholds = self.alert_thresholds[metric_key]
        
        # Determine alert severity
        alert_severity = None
        threshold_value = 0.0
        
        if metric_key in ['response_time_ms']:
            # Higher values are worse
            if metric.value >= thresholds.get('EMERGENCY', float('inf')):
                alert_severity = AlertSeverity.EMERGENCY
                threshold_value = thresholds['EMERGENCY']
            elif metric.value >= thresholds.get('CRITICAL', float('inf')):
                alert_severity = AlertSeverity.CRITICAL
                threshold_value = thresholds['CRITICAL']
            elif metric.value >= thresholds.get('WARNING', float('inf')):
                alert_severity = AlertSeverity.WARNING
                threshold_value = thresholds['WARNING']
        
        elif metric_key in ['success_rate']:
            # Lower values are worse
            if metric.value <= thresholds.get('EMERGENCY', 0):
                alert_severity = AlertSeverity.EMERGENCY
                threshold_value = thresholds['EMERGENCY']
            elif metric.value <= thresholds.get('CRITICAL', 0):
                alert_severity = AlertSeverity.CRITICAL
                threshold_value = thresholds['CRITICAL']
            elif metric.value <= thresholds.get('WARNING', 0):
                alert_severity = AlertSeverity.WARNING
                threshold_value = thresholds['WARNING']
        
        elif metric_key in ['error_rate', 'fallback_rate']:
            # Higher values are worse
            if metric.value >= thresholds.get('EMERGENCY', float('inf')):
                alert_severity = AlertSeverity.EMERGENCY
                threshold_value = thresholds['EMERGENCY']
            elif metric.value >= thresholds.get('CRITICAL', float('inf')):
                alert_severity = AlertSeverity.CRITICAL
                threshold_value = thresholds['CRITICAL']
            elif metric.value >= thresholds.get('WARNING', float('inf')):
                alert_severity = AlertSeverity.WARNING
                threshold_value = thresholds['WARNING']
        
        # Create alert if threshold exceeded and not suppressed
        if alert_severity:
            alert_key = f"{metric.component}_{metric_key}_{alert_severity.value}"
            
            if not self._is_alert_suppressed(alert_key):
                alert = self._create_performance_alert(
                    metric, alert_severity, threshold_value
                )
                self.alerts.append(alert)
                self._suppress_alert(alert_key)
                
                # Log based on severity
                if alert_severity == AlertSeverity.EMERGENCY:
                    self.logger.critical(alert.description)
                elif alert_severity == AlertSeverity.CRITICAL:
                    self.logger.error(alert.description)
                elif alert_severity == AlertSeverity.WARNING:
                    self.logger.warning(alert.description)
    
    def _create_performance_alert(self, metric: PerformanceMetric, severity: AlertSeverity, threshold: float) -> PerformanceAlert:
        """Create a performance alert from a metric threshold violation."""
        alert_id = f"{metric.component}_{metric.metric_type.value}_{int(metric.timestamp)}"
        
        description = f"{metric.component} {metric.metric_type.value} {metric.value:.2f} exceeds {severity.value} threshold {threshold:.2f}"
        
        return PerformanceAlert(
            alert_id=alert_id,
            severity=severity,
            metric_type=metric.metric_type,
            component=metric.component,
            threshold_value=threshold,
            current_value=metric.value,
            description=description,
            triggered_at=metric.timestamp
        )
    
    def _is_alert_suppressed(self, alert_key: str) -> bool:
        """Check if an alert is currently suppressed."""
        if alert_key in self.alert_suppression:
            return time.time() - self.alert_suppression[alert_key] < self.suppression_window_seconds
        return False
    
    def _suppress_alert(self, alert_key: str):
        """Suppress an alert for the configured time window."""
        self.alert_suppression[alert_key] = time.time()
    
    def detect_performance_regression(self, component: str, window_minutes: int = 30) -> RegressionDetectionResult:
        """
        Detect performance regressions using statistical analysis.
        
        Uses sliding window comparison against established baselines.
        """
        current_time = time.time()
        window_start = current_time - (window_minutes * 60)
        
        affected_metrics = []
        baseline_comparison = {}
        recommendations = []
        
        # Analyze each metric type for regression
        for metric_type in MetricType:
            if metric_type not in self.metrics:
                continue
            
            # Get recent metrics for this component
            recent_metrics = [
                m for m in self.metrics[metric_type]
                if m.component == component and m.timestamp >= window_start
            ]
            
            if len(recent_metrics) < 5:  # Need minimum sample size
                continue
            
            # Calculate current average
            current_avg = sum(m.value for m in recent_metrics) / len(recent_metrics)
            
            # Compare against baseline
            baseline_key = f"{component}_{metric_type.value}"
            if baseline_key in self.baselines:
                baseline_avg = self.baselines[baseline_key].get(metric_type, current_avg)
                
                # Calculate regression severity
                if metric_type in [MetricType.RESPONSE_TIME, MetricType.ERROR_RATE, MetricType.FALLBACK_RATE]:
                    # Higher values indicate regression
                    regression_ratio = current_avg / baseline_avg if baseline_avg > 0 else 1.0
                    if regression_ratio > 1.5:  # 50% worse
                        affected_metrics.append(metric_type)
                        baseline_comparison[metric_type.value] = {
                            'baseline': baseline_avg,
                            'current': current_avg,
                            'regression_ratio': regression_ratio
                        }
                
                elif metric_type in [MetricType.SUCCESS_RATE, MetricType.THROUGHPUT]:
                    # Lower values indicate regression
                    regression_ratio = baseline_avg / current_avg if current_avg > 0 else float('inf')
                    if regression_ratio > 1.2:  # 20% worse
                        affected_metrics.append(metric_type)
                        baseline_comparison[metric_type.value] = {
                            'baseline': baseline_avg,
                            'current': current_avg,
                            'regression_ratio': regression_ratio
                        }
        
        # Determine overall severity and recommendations
        regression_detected = len(affected_metrics) > 0
        
        if regression_detected:
            severity_level = self._calculate_regression_severity(baseline_comparison)
            recommendations = self._generate_regression_recommendations(affected_metrics, baseline_comparison)
        else:
            severity_level = AlertSeverity.INFO
        
        # Calculate confidence based on sample size and consistency
        confidence_score = self._calculate_regression_confidence(recent_metrics, baseline_comparison)
        
        return RegressionDetectionResult(
            regression_detected=regression_detected,
            affected_metrics=affected_metrics,
            severity_level=severity_level,
            baseline_comparison=baseline_comparison,
            recommendations=recommendations,
            confidence_score=confidence_score
        )
    
    def _calculate_regression_severity(self, baseline_comparison: Dict[str, Dict]) -> AlertSeverity:
        """Calculate overall regression severity based on affected metrics."""
        max_regression_ratio = 0.0
        
        for metric_data in baseline_comparison.values():
            max_regression_ratio = max(max_regression_ratio, metric_data['regression_ratio'])
        
        if max_regression_ratio > 3.0:  # 300% worse
            return AlertSeverity.EMERGENCY
        elif max_regression_ratio > 2.0:  # 200% worse
            return AlertSeverity.CRITICAL
        elif max_regression_ratio > 1.5:  # 150% worse
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO
    
    def _generate_regression_recommendations(self, affected_metrics: List[MetricType], 
                                           baseline_comparison: Dict[str, Dict]) -> List[str]:
        """Generate actionable recommendations for addressing performance regression."""
        recommendations = []
        
        for metric_type in affected_metrics:
            metric_key = metric_type.value
            if metric_key in baseline_comparison:
                comparison = baseline_comparison[metric_key]
                
                if metric_type == MetricType.RESPONSE_TIME:
                    recommendations.append(f"Investigate response time regression: {comparison['current']:.1f}ms vs baseline {comparison['baseline']:.1f}ms")
                    recommendations.append("Check MCP server load and network connectivity")
                
                elif metric_type == MetricType.ERROR_RATE:
                    recommendations.append(f"Address increased error rate: {comparison['current']:.1%} vs baseline {comparison['baseline']:.1%}")
                    recommendations.append("Review recent code changes and MCP server configurations")
                
                elif metric_type == MetricType.FALLBACK_RATE:
                    recommendations.append(f"Investigate increased fallback usage: {comparison['current']:.1%} vs baseline {comparison['baseline']:.1%}")
                    recommendations.append("Check MCP server availability and circuit breaker states")
                
                elif metric_type == MetricType.SUCCESS_RATE:
                    recommendations.append(f"Address declining success rate: {comparison['current']:.1%} vs baseline {comparison['baseline']:.1%}")
                    recommendations.append("Review error logs and implement additional resilience patterns")
        
        # Add general recommendations
        if len(affected_metrics) > 2:
            recommendations.append("Consider implementing additional circuit breaker patterns")
            recommendations.append("Review MCP server capacity and scaling options")
        
        return recommendations
    
    def _calculate_regression_confidence(self, recent_metrics: List[PerformanceMetric], 
                                       baseline_comparison: Dict[str, Dict]) -> float:
        """Calculate confidence score for regression detection."""
        if not recent_metrics or not baseline_comparison:
            return 0.0
        
        # Base confidence on sample size
        sample_size_score = min(len(recent_metrics) / 20.0, 1.0)  # Ideal sample size is 20+
        
        # Base confidence on consistency of regression signal
        consistency_scores = []
        for metric_data in baseline_comparison.values():
            ratio = metric_data['regression_ratio']
            if ratio > 1.5:  # Clear regression signal
                consistency_scores.append(min((ratio - 1.0) / 2.0, 1.0))
        
        consistency_score = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0
        
        # Combine scores
        confidence = (sample_size_score * 0.4) + (consistency_score * 0.6)
        return min(confidence, 1.0)
    
    def establish_performance_baseline(self, component: str, window_hours: int = 24) -> Dict[MetricType, float]:
        """Establish performance baseline for a component using historical data."""
        current_time = time.time()
        baseline_start = current_time - (window_hours * 3600)
        
        baseline_values = {}
        
        for metric_type, metrics_deque in self.metrics.items():
            # Get metrics for this component within the baseline window
            component_metrics = [
                m for m in metrics_deque
                if m.component == component and m.timestamp >= baseline_start
            ]
            
            if len(component_metrics) >= 10:  # Minimum sample size
                # Calculate baseline (median for robustness against outliers)
                values = sorted([m.value for m in component_metrics])
                median_value = values[len(values) // 2]
                baseline_values[metric_type] = median_value
        
        # Store baseline
        baseline_key = f"{component}_{metric_type.value}"
        self.baselines[baseline_key] = baseline_values
        
        self.logger.info(f"Established baseline for {component} with {len(baseline_values)} metrics")
        return baseline_values
    
    def monitor_processing_operation(self, operation_name: str, component: str = "mcp_processing"):
        """Context manager for monitoring processing operations."""
        return ProcessingOperationMonitor(self, operation_name, component)
    
    def get_performance_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for performance dashboards and visualization."""
        current_time = time.time()
        
        # Calculate recent metrics (last hour)
        recent_window = current_time - 3600
        
        dashboard_data = {
            'timestamp': current_time,
            'summary': self._calculate_summary_metrics(recent_window),
            'metrics_by_type': self._aggregate_metrics_by_type(recent_window),
            'component_performance': self._calculate_component_performance(recent_window),
            'alerts_summary': self._summarize_recent_alerts(),
            'health_indicators': self._calculate_health_indicators(),
            'performance_trends': self._calculate_performance_trends()
        }
        
        return dashboard_data
    
    def _calculate_summary_metrics(self, since_timestamp: float) -> Dict[str, float]:
        """Calculate summary metrics for dashboard overview."""
        total_operations = 0
        successful_operations = 0
        total_response_time = 0.0
        response_time_count = 0
        
        for metric_type, metrics_deque in self.metrics.items():
            recent_metrics = [m for m in metrics_deque if m.timestamp >= since_timestamp]
            
            if metric_type == MetricType.SUCCESS_RATE:
                for metric in recent_metrics:
                    total_operations += 1
                    if metric.value > 0.5:  # Success threshold
                        successful_operations += 1
            
            elif metric_type == MetricType.RESPONSE_TIME:
                for metric in recent_metrics:
                    total_response_time += metric.value
                    response_time_count += 1
        
        success_rate = successful_operations / total_operations if total_operations > 0 else 0.0
        avg_response_time = total_response_time / response_time_count if response_time_count > 0 else 0.0
        
        return {
            'total_operations': total_operations,
            'success_rate': success_rate,
            'average_response_time_ms': avg_response_time,
            'error_rate': 1.0 - success_rate
        }
    
    def _aggregate_metrics_by_type(self, since_timestamp: float) -> Dict[str, List[Dict]]:
        """Aggregate metrics by type for time series visualization."""
        aggregated = {}
        
        for metric_type, metrics_deque in self.metrics.items():
            recent_metrics = [m for m in metrics_deque if m.timestamp >= since_timestamp]
            
            # Create time series data points
            time_series = []
            for metric in recent_metrics:
                time_series.append({
                    'timestamp': metric.timestamp,
                    'value': metric.value,
                    'component': metric.component,
                    'tags': metric.tags
                })
            
            aggregated[metric_type.value] = time_series
        
        return aggregated
    
    def _calculate_component_performance(self, since_timestamp: float) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics grouped by component."""
        component_stats = defaultdict(lambda: defaultdict(list))
        
        # Group metrics by component
        for metric_type, metrics_deque in self.metrics.items():
            recent_metrics = [m for m in metrics_deque if m.timestamp >= since_timestamp]
            
            for metric in recent_metrics:
                component_stats[metric.component][metric_type.value].append(metric.value)
        
        # Calculate aggregated stats per component
        component_performance = {}
        for component, metric_types in component_stats.items():
            component_performance[component] = {}
            
            for metric_type_name, values in metric_types.items():
                if values:
                    component_performance[component][metric_type_name] = {
                        'average': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values),
                        'count': len(values)
                    }
        
        return component_performance
    
    def _summarize_recent_alerts(self) -> Dict[str, Any]:
        """Summarize recent alerts for dashboard display."""
        current_time = time.time()
        recent_window = current_time - 3600  # Last hour
        
        recent_alerts = [a for a in self.alerts if a.triggered_at >= recent_window]
        
        alert_summary = {
            'total_alerts': len(recent_alerts),
            'by_severity': defaultdict(int),
            'by_component': defaultdict(int),
            'active_alerts': [a for a in recent_alerts if not a.resolved_at],
            'resolved_alerts': [a for a in recent_alerts if a.resolved_at]
        }
        
        for alert in recent_alerts:
            alert_summary['by_severity'][alert.severity.value] += 1
            alert_summary['by_component'][alert.component] += 1
        
        return dict(alert_summary)
    
    def _calculate_health_indicators(self) -> Dict[str, str]:
        """Calculate overall system health indicators."""
        # Count recent critical alerts
        current_time = time.time()
        recent_critical = sum(
            1 for a in self.alerts 
            if (current_time - a.triggered_at) < 3600 and a.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
        )
        
        # Determine overall health
        if recent_critical > 5:
            overall_health = "CRITICAL"
        elif recent_critical > 2:
            overall_health = "WARNING"
        else:
            overall_health = "HEALTHY"
        
        return {
            'overall_health': overall_health,
            'recent_critical_alerts': recent_critical,
            'monitoring_status': "ACTIVE",
            'last_baseline_update': max(
                [baseline.get('last_updated', 0) for baseline in self.baselines.values()],
                default=0
            )
        }
    
    def _calculate_performance_trends(self) -> Dict[str, str]:
        """Calculate performance trends over time."""
        # Simple trend analysis - compare last hour vs previous hour
        current_time = time.time()
        hour_1_start = current_time - 3600
        hour_2_start = current_time - 7200
        
        trends = {}
        
        for metric_type in MetricType:
            if metric_type not in self.metrics:
                continue
            
            hour_1_metrics = [m for m in self.metrics[metric_type] if hour_1_start <= m.timestamp < current_time]
            hour_2_metrics = [m for m in self.metrics[metric_type] if hour_2_start <= m.timestamp < hour_1_start]
            
            if len(hour_1_metrics) >= 3 and len(hour_2_metrics) >= 3:
                hour_1_avg = sum(m.value for m in hour_1_metrics) / len(hour_1_metrics)
                hour_2_avg = sum(m.value for m in hour_2_metrics) / len(hour_2_metrics)
                
                if hour_2_avg > 0:
                    change_ratio = hour_1_avg / hour_2_avg
                    if change_ratio > 1.1:
                        trends[metric_type.value] = "IMPROVING" if metric_type in [MetricType.SUCCESS_RATE, MetricType.THROUGHPUT] else "DEGRADING"
                    elif change_ratio < 0.9:
                        trends[metric_type.value] = "DEGRADING" if metric_type in [MetricType.SUCCESS_RATE, MetricType.THROUGHPUT] else "IMPROVING"
                    else:
                        trends[metric_type.value] = "STABLE"
        
        return trends
    
    def generate_performance_report(self, hours_back: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report for specified time period."""
        current_time = time.time()
        report_start = current_time - (hours_back * 3600)
        
        # Collect metrics for the reporting period
        period_metrics = {}
        for metric_type, metrics_deque in self.metrics.items():
            period_metrics[metric_type] = [
                m for m in metrics_deque if m.timestamp >= report_start
            ]
        
        # Generate report sections
        report = {
            'report_metadata': {
                'generated_at': current_time,
                'period_hours': hours_back,
                'period_start': report_start,
                'period_end': current_time
            },
            'executive_summary': self._generate_executive_summary(period_metrics),
            'detailed_metrics': self._generate_detailed_metrics_report(period_metrics),
            'alert_analysis': self._generate_alert_analysis_report(report_start),
            'regression_analysis': self._generate_regression_analysis_report(),
            'recommendations': self._generate_performance_recommendations(period_metrics)
        }
        
        return report
    
    def _generate_executive_summary(self, period_metrics: Dict[MetricType, List]) -> Dict[str, Any]:
        """Generate executive summary of performance for the reporting period."""
        total_operations = sum(len(metrics) for metrics in period_metrics.values())
        
        # Key performance indicators
        if MetricType.SUCCESS_RATE in period_metrics and period_metrics[MetricType.SUCCESS_RATE]:
            success_metrics = period_metrics[MetricType.SUCCESS_RATE]
            avg_success_rate = sum(m.value for m in success_metrics) / len(success_metrics)
        else:
            avg_success_rate = 0.0
        
        if MetricType.RESPONSE_TIME in period_metrics and period_metrics[MetricType.RESPONSE_TIME]:
            response_metrics = period_metrics[MetricType.RESPONSE_TIME]
            avg_response_time = sum(m.value for m in response_metrics) / len(response_metrics)
        else:
            avg_response_time = 0.0
        
        # Performance vs targets
        performance_vs_targets = {
            'processing_time': {
                'current': avg_response_time / 1000.0,  # Convert to seconds
                'target': self.performance_targets['processing_time_seconds'],
                'meets_target': avg_response_time / 1000.0 <= self.performance_targets['processing_time_seconds']
            },
            'success_rate': {
                'current': avg_success_rate,
                'target': self.performance_targets['success_rate'],
                'meets_target': avg_success_rate >= self.performance_targets['success_rate']
            }
        }
        
        return {
            'total_operations': total_operations,
            'average_success_rate': avg_success_rate,
            'average_response_time_ms': avg_response_time,
            'performance_vs_targets': performance_vs_targets,
            'overall_health_grade': self._calculate_health_grade(performance_vs_targets)
        }
    
    def _calculate_health_grade(self, performance_vs_targets: Dict) -> str:
        """Calculate overall health grade based on target achievement."""
        targets_met = sum(1 for target in performance_vs_targets.values() if target['meets_target'])
        total_targets = len(performance_vs_targets)
        
        if targets_met == total_targets:
            return "A"  # All targets met
        elif targets_met >= total_targets * 0.8:
            return "B"  # 80%+ targets met
        elif targets_met >= total_targets * 0.6:
            return "C"  # 60%+ targets met
        else:
            return "F"  # Less than 60% targets met
    
    def _generate_detailed_metrics_report(self, period_metrics: Dict) -> Dict[str, Any]:
        """Generate detailed metrics breakdown."""
        detailed_report = {}
        
        for metric_type, metrics_list in period_metrics.items():
            if not metrics_list:
                continue
            
            values = [m.value for m in metrics_list]
            
            detailed_report[metric_type.value] = {
                'count': len(values),
                'average': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'median': sorted(values)[len(values) // 2],
                'percentile_95': sorted(values)[int(len(values) * 0.95)] if len(values) > 20 else max(values),
                'component_breakdown': self._breakdown_by_component(metrics_list)
            }
        
        return detailed_report
    
    def _breakdown_by_component(self, metrics_list: List[PerformanceMetric]) -> Dict[str, Dict]:
        """Break down metrics by component."""
        component_breakdown = defaultdict(list)
        
        for metric in metrics_list:
            component_breakdown[metric.component].append(metric.value)
        
        breakdown_summary = {}
        for component, values in component_breakdown.items():
            breakdown_summary[component] = {
                'count': len(values),
                'average': sum(values) / len(values),
                'min': min(values),
                'max': max(values)
            }
        
        return breakdown_summary
    
    def _generate_alert_analysis_report(self, since_timestamp: float) -> Dict[str, Any]:
        """Generate analysis of alerts during the reporting period."""
        period_alerts = [a for a in self.alerts if a.triggered_at >= since_timestamp]
        
        return {
            'total_alerts': len(period_alerts),
            'alerts_by_severity': {
                severity.value: sum(1 for a in period_alerts if a.severity == severity)
                for severity in AlertSeverity
            },
            'alerts_by_component': {
                component: sum(1 for a in period_alerts if a.component == component)
                for component in set(a.component for a in period_alerts)
            },
            'most_frequent_alert_types': self._get_most_frequent_alert_types(period_alerts),
            'alert_resolution_rate': sum(1 for a in period_alerts if a.resolved_at) / len(period_alerts) if period_alerts else 0.0
        }
    
    def _get_most_frequent_alert_types(self, alerts: List[PerformanceAlert]) -> List[Dict[str, Any]]:
        """Get most frequent alert types during period."""
        alert_counts = defaultdict(int)
        
        for alert in alerts:
            alert_key = f"{alert.metric_type.value}_{alert.severity.value}"
            alert_counts[alert_key] += 1
        
        # Sort by frequency and return top 5
        frequent_alerts = sorted(alert_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return [
            {'alert_type': alert_type, 'count': count}
            for alert_type, count in frequent_alerts
        ]
    
    def _generate_regression_analysis_report(self) -> Dict[str, Any]:
        """Generate regression analysis for all monitored components."""
        regression_results = {}
        
        # Check each unique component for regression
        all_components = set()
        for metrics_deque in self.metrics.values():
            all_components.update(m.component for m in metrics_deque)
        
        for component in all_components:
            regression_result = self.detect_performance_regression(component)
            regression_results[component] = {
                'regression_detected': regression_result.regression_detected,
                'affected_metrics': [m.value for m in regression_result.affected_metrics],
                'severity': regression_result.severity_level.value,
                'confidence_score': regression_result.confidence_score,
                'recommendations_count': len(regression_result.recommendations)
            }
        
        return regression_results
    
    def _generate_performance_recommendations(self, period_metrics: Dict) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Analyze response time patterns
        if MetricType.RESPONSE_TIME in period_metrics:
            response_times = [m.value for m in period_metrics[MetricType.RESPONSE_TIME]]
            if response_times:
                avg_response = sum(response_times) / len(response_times)
                if avg_response > 500:  # > 500ms
                    recommendations.append(f"Consider optimizing MCP server response times (current avg: {avg_response:.1f}ms)")
        
        # Analyze error rates
        if MetricType.ERROR_RATE in period_metrics:
            error_rates = [m.value for m in period_metrics[MetricType.ERROR_RATE]]
            if error_rates:
                avg_error_rate = sum(error_rates) / len(error_rates)
                if avg_error_rate > 0.02:  # > 2%
                    recommendations.append(f"Investigate and reduce error rate (current: {avg_error_rate:.1%})")
        
        # Analyze fallback usage
        if MetricType.FALLBACK_RATE in period_metrics:
            fallback_rates = [m.value for m in period_metrics[MetricType.FALLBACK_RATE]]
            if fallback_rates:
                avg_fallback = sum(fallback_rates) / len(fallback_rates)
                if avg_fallback > 0.1:  # > 10%
                    recommendations.append(f"High fallback usage detected ({avg_fallback:.1%}) - consider MCP server capacity upgrades")
        
        # Add Story 4.1 specific recommendations
        recommendations.extend([
            "Monitor MCP server performance daily during Week 1-2",
            "Establish production baselines for all critical metrics",
            "Implement automated alerting for regression detection"
        ])
        
        return recommendations


class ProcessingOperationMonitor:
    """Context manager for monitoring individual processing operations."""
    
    def __init__(self, performance_monitor: PerformanceMonitor, operation_name: str, component: str):
        self.performance_monitor = performance_monitor
        self.operation_name = operation_name
        self.component = component
        self.start_time = None
        self.logger = logging.getLogger(__name__)
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            end_time = time.time()
            duration_ms = (end_time - self.start_time) * 1000
            
            # Record response time metric
            self.performance_monitor.record_metric(
                MetricType.RESPONSE_TIME,
                duration_ms,
                self.component,
                tags={'operation': self.operation_name}
            )
            
            # Record success/failure metric
            success_rate = 1.0 if exc_type is None else 0.0
            self.performance_monitor.record_metric(
                MetricType.SUCCESS_RATE,
                success_rate,
                self.component,
                tags={'operation': self.operation_name}
            )
            
            # Log operation completion
            if exc_type is None:
                self.logger.debug(f"Operation {self.operation_name} completed in {duration_ms:.1f}ms")
            else:
                self.logger.warning(f"Operation {self.operation_name} failed after {duration_ms:.1f}ms: {exc_val}")


# Test function for validation
async def test_performance_monitor():
    """Test performance monitor functionality."""
    print("Testing Performance Monitor Enterprise Features...")
    
    config = {
        'metrics_retention_hours': 1,
        'target_response_time_ms': 500,
        'enable_automated_reporting': False
    }
    
    monitor = PerformanceMonitor(config)
    
    # Test metric recording
    monitor.record_metric(MetricType.RESPONSE_TIME, 234.5, "mcp_client", {"operation": "context_analysis"})
    monitor.record_metric(MetricType.SUCCESS_RATE, 0.98, "mcp_client", {"operation": "text_processing"})
    
    print("✅ Metric recording working")
    
    # Test operation monitoring
    with monitor.monitor_processing_operation("test_operation", "test_component"):
        time.sleep(0.1)  # Simulate processing
    
    print("✅ Operation monitoring working")
    
    # Test dashboard data generation
    dashboard_data = monitor.get_performance_dashboard_data()
    print(f"✅ Dashboard data generated with {len(dashboard_data)} sections")
    
    # Test performance report
    report = monitor.generate_performance_report(hours_back=1)
    print(f"✅ Performance report generated with {len(report)} sections")
    
    return monitor


if __name__ == "__main__":
    asyncio.run(test_performance_monitor())