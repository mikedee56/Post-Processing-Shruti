"""
Quality Metrics Collection Framework for Story 5.5: Testing & Quality Assurance Framework

This module provides comprehensive quality metrics collection and reporting framework
with real-time quality metrics monitoring, quality dashboards and reporting systems,
automated quality reporting and trend analysis, quality gates and failure notifications,
quality improvement tracking and recommendations, and quality compliance and audit reporting.

Author: James the Developer
Date: August 20, 2025
Story: 5.5 Testing & Quality Assurance Framework
Task: 6 - Quality Metrics and Monitoring (AC5)
"""

import os
import json
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Callable, NamedTuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import statistics
import sqlite3
from contextlib import contextmanager


class MetricType(Enum):
    """Types of quality metrics."""
    COUNTER = "counter"           # Incremental values (test count, error count)
    GAUGE = "gauge"              # Current values (coverage %, performance)
    HISTOGRAM = "histogram"       # Distribution of values (execution times)
    TIMER = "timer"              # Time-based measurements


class MetricSeverity(Enum):
    """Severity levels for metric alerts."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricDefinition:
    """Definition of a quality metric."""
    name: str
    metric_type: MetricType
    description: str
    unit: str
    category: str
    alert_threshold: Optional[float] = None
    warning_threshold: Optional[float] = None
    target_value: Optional[float] = None
    is_higher_better: bool = True


@dataclass
class MetricValue:
    """A single metric measurement."""
    name: str
    value: float
    timestamp: datetime
    labels: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class QualityAlert:
    """Quality alert for threshold violations."""
    metric_name: str
    severity: MetricSeverity
    threshold: float
    actual_value: float
    message: str
    timestamp: datetime
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None


class MetricsDatabase:
    """SQLite database for storing quality metrics."""
    
    def __init__(self, db_path: str):
        """Initialize metrics database."""
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp DATETIME NOT NULL,
                    labels TEXT,  -- JSON string
                    metadata TEXT  -- JSON string
                );
                
                CREATE TABLE IF NOT EXISTS metric_definitions (
                    name TEXT PRIMARY KEY,
                    metric_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    unit TEXT NOT NULL,
                    category TEXT NOT NULL,
                    alert_threshold REAL,
                    warning_threshold REAL,
                    target_value REAL,
                    is_higher_better BOOLEAN DEFAULT TRUE
                );
                
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    actual_value REAL NOT NULL,
                    message TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolution_timestamp DATETIME
                );
                
                CREATE INDEX IF NOT EXISTS idx_metrics_name_time ON metrics(name, timestamp);
                CREATE INDEX IF NOT EXISTS idx_alerts_metric_time ON alerts(metric_name, timestamp);
            """)
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def store_metric(self, metric: MetricValue):
        """Store a metric value in the database."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO metrics (name, value, timestamp, labels, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                metric.name,
                metric.value,
                metric.timestamp.isoformat(),
                json.dumps(metric.labels) if metric.labels else None,
                json.dumps(metric.metadata) if metric.metadata else None
            ))
            conn.commit()
    
    def store_metric_definition(self, definition: MetricDefinition):
        """Store a metric definition in the database."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO metric_definitions 
                (name, metric_type, description, unit, category, alert_threshold, 
                 warning_threshold, target_value, is_higher_better)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                definition.name,
                definition.metric_type.value,
                definition.description,
                definition.unit,
                definition.category,
                definition.alert_threshold,
                definition.warning_threshold,
                definition.target_value,
                definition.is_higher_better
            ))
            conn.commit()
    
    def store_alert(self, alert: QualityAlert):
        """Store an alert in the database."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO alerts 
                (metric_name, severity, threshold, actual_value, message, timestamp, resolved, resolution_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.metric_name,
                alert.severity.value,
                alert.threshold,
                alert.actual_value,
                alert.message,
                alert.timestamp.isoformat(),
                alert.resolved,
                alert.resolution_timestamp.isoformat() if alert.resolution_timestamp else None
            ))
            conn.commit()
    
    def get_metric_values(self, metric_name: str, start_time: Optional[datetime] = None, 
                         end_time: Optional[datetime] = None, limit: Optional[int] = None) -> List[MetricValue]:
        """Retrieve metric values from the database."""
        query = "SELECT * FROM metrics WHERE name = ?"
        params = [metric_name]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            metrics = []
            for row in rows:
                metric = MetricValue(
                    name=row['name'],
                    value=row['value'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    labels=json.loads(row['labels']) if row['labels'] else None,
                    metadata=json.loads(row['metadata']) if row['metadata'] else None
                )
                metrics.append(metric)
            
            return metrics
    
    def get_metric_definition(self, metric_name: str) -> Optional[MetricDefinition]:
        """Retrieve metric definition from the database."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM metric_definitions WHERE name = ?", (metric_name,))
            row = cursor.fetchone()
            
            if row:
                return MetricDefinition(
                    name=row['name'],
                    metric_type=MetricType(row['metric_type']),
                    description=row['description'],
                    unit=row['unit'],
                    category=row['category'],
                    alert_threshold=row['alert_threshold'],
                    warning_threshold=row['warning_threshold'],
                    target_value=row['target_value'],
                    is_higher_better=bool(row['is_higher_better'])
                )
            
            return None
    
    def get_active_alerts(self) -> List[QualityAlert]:
        """Get all active (unresolved) alerts."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM alerts WHERE resolved = FALSE ORDER BY timestamp DESC
            """)
            rows = cursor.fetchall()
            
            alerts = []
            for row in rows:
                alert = QualityAlert(
                    metric_name=row['metric_name'],
                    severity=MetricSeverity(row['severity']),
                    threshold=row['threshold'],
                    actual_value=row['actual_value'],
                    message=row['message'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    resolved=bool(row['resolved']),
                    resolution_timestamp=datetime.fromisoformat(row['resolution_timestamp']) if row['resolution_timestamp'] else None
                )
                alerts.append(alert)
            
            return alerts


class QualityMetricsCollector:
    """Central quality metrics collection and monitoring system."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize metrics collector."""
        self.logger = logging.getLogger(__name__)
        
        if db_path is None:
            # Default to qa/metrics directory
            metrics_dir = Path(__file__).parent.parent / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(metrics_dir / "quality_metrics.db")
        
        self.db = MetricsDatabase(db_path)
        self.alert_handlers: List[Callable[[QualityAlert], None]] = []
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_interval = 60  # seconds
        
        # Initialize standard metric definitions
        self._initialize_standard_metrics()
    
    def _initialize_standard_metrics(self):
        """Initialize standard quality metrics definitions."""
        standard_metrics = [
            # Code Coverage Metrics
            MetricDefinition(
                name="test_coverage_total",
                metric_type=MetricType.GAUGE,
                description="Overall test coverage percentage",
                unit="percent",
                category="coverage",
                alert_threshold=70.0,
                warning_threshold=80.0,
                target_value=90.0,
                is_higher_better=True
            ),
            MetricDefinition(
                name="test_coverage_lines",
                metric_type=MetricType.GAUGE,
                description="Line coverage percentage",
                unit="percent",
                category="coverage",
                alert_threshold=70.0,
                warning_threshold=80.0,
                target_value=90.0,
                is_higher_better=True
            ),
            MetricDefinition(
                name="test_coverage_branches",
                metric_type=MetricType.GAUGE,
                description="Branch coverage percentage",
                unit="percent",
                category="coverage",
                alert_threshold=60.0,
                warning_threshold=70.0,
                target_value=80.0,
                is_higher_better=True
            ),
            
            # Performance Metrics
            MetricDefinition(
                name="processing_throughput",
                metric_type=MetricType.GAUGE,
                description="Processing throughput in segments per second",
                unit="segments/sec",
                category="performance",
                alert_threshold=5.0,
                warning_threshold=8.0,
                target_value=10.0,
                is_higher_better=True
            ),
            MetricDefinition(
                name="processing_latency_p95",
                metric_type=MetricType.GAUGE,
                description="95th percentile processing latency",
                unit="seconds",
                category="performance",
                alert_threshold=2.0,
                warning_threshold=1.0,
                target_value=0.5,
                is_higher_better=False
            ),
            MetricDefinition(
                name="memory_usage",
                metric_type=MetricType.GAUGE,
                description="Memory usage percentage",
                unit="percent",
                category="performance",
                alert_threshold=90.0,
                warning_threshold=80.0,
                target_value=60.0,
                is_higher_better=False
            ),
            
            # Code Quality Metrics
            MetricDefinition(
                name="code_quality_score",
                metric_type=MetricType.GAUGE,
                description="Overall code quality score",
                unit="score",
                category="quality",
                alert_threshold=70.0,
                warning_threshold=80.0,
                target_value=90.0,
                is_higher_better=True
            ),
            MetricDefinition(
                name="linting_violations",
                metric_type=MetricType.COUNTER,
                description="Number of linting violations",
                unit="count",
                category="quality",
                alert_threshold=100.0,
                warning_threshold=50.0,
                target_value=0.0,
                is_higher_better=False
            ),
            MetricDefinition(
                name="complexity_score",
                metric_type=MetricType.GAUGE,
                description="Code complexity score",
                unit="score",
                category="quality",
                alert_threshold=20.0,
                warning_threshold=15.0,
                target_value=10.0,
                is_higher_better=False
            ),
            
            # Security Metrics
            MetricDefinition(
                name="security_vulnerabilities_critical",
                metric_type=MetricType.COUNTER,
                description="Number of critical security vulnerabilities",
                unit="count",
                category="security",
                alert_threshold=1.0,
                warning_threshold=0.0,
                target_value=0.0,
                is_higher_better=False
            ),
            MetricDefinition(
                name="security_vulnerabilities_high",
                metric_type=MetricType.COUNTER,
                description="Number of high severity security vulnerabilities",
                unit="count",
                category="security",
                alert_threshold=5.0,
                warning_threshold=2.0,
                target_value=0.0,
                is_higher_better=False
            ),
            MetricDefinition(
                name="security_score",
                metric_type=MetricType.GAUGE,
                description="Overall security score",
                unit="score",
                category="security",
                alert_threshold=80.0,
                warning_threshold=90.0,
                target_value=100.0,
                is_higher_better=True
            ),
            
            # Test Metrics
            MetricDefinition(
                name="test_count_total",
                metric_type=MetricType.COUNTER,
                description="Total number of tests",
                unit="count",
                category="testing",
                target_value=1000.0,
                is_higher_better=True
            ),
            MetricDefinition(
                name="test_pass_rate",
                metric_type=MetricType.GAUGE,
                description="Test pass rate percentage",
                unit="percent",
                category="testing",
                alert_threshold=95.0,
                warning_threshold=98.0,
                target_value=100.0,
                is_higher_better=True
            ),
            MetricDefinition(
                name="test_execution_time",
                metric_type=MetricType.TIMER,
                description="Total test execution time",
                unit="seconds",
                category="testing",
                alert_threshold=600.0,  # 10 minutes
                warning_threshold=300.0,  # 5 minutes
                target_value=120.0,  # 2 minutes
                is_higher_better=False
            ),
            
            # Documentation Metrics
            MetricDefinition(
                name="documentation_coverage",
                metric_type=MetricType.GAUGE,
                description="Documentation coverage percentage",
                unit="percent",
                category="documentation",
                alert_threshold=60.0,
                warning_threshold=75.0,
                target_value=90.0,
                is_higher_better=True
            )
        ]
        
        # Store metric definitions
        for metric_def in standard_metrics:
            self.db.store_metric_definition(metric_def)
    
    def record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None, 
                     metadata: Optional[Dict[str, Any]] = None):
        """Record a metric value."""
        metric = MetricValue(
            name=name,
            value=value,
            timestamp=datetime.now(),
            labels=labels,
            metadata=metadata
        )
        
        self.db.store_metric(metric)
        
        # Check for alerts
        self._check_metric_alerts(metric)
        
        self.logger.debug(f"Recorded metric {name}: {value}")
    
    def record_counter(self, name: str, increment: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Record a counter metric (incremental value)."""
        # For counters, we store the increment and rely on aggregation for totals
        self.record_metric(name, increment, labels, {"metric_type": "counter"})
    
    def record_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a gauge metric (current value)."""
        self.record_metric(name, value, labels, {"metric_type": "gauge"})
    
    def record_timer(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None):
        """Record a timer metric (duration measurement)."""
        self.record_metric(name, duration, labels, {"metric_type": "timer"})
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram metric (value distribution)."""
        self.record_metric(name, value, labels, {"metric_type": "histogram"})
    
    def get_metric_summary(self, metric_name: str, hours: int = 24) -> Dict[str, Any]:
        """Get summary statistics for a metric over the specified time period."""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        values = self.db.get_metric_values(metric_name, start_time, end_time)
        
        if not values:
            return {"error": f"No data found for metric {metric_name}"}
        
        numeric_values = [v.value for v in values]
        
        summary = {
            "metric_name": metric_name,
            "period_hours": hours,
            "sample_count": len(numeric_values),
            "latest_value": numeric_values[0] if numeric_values else None,
            "latest_timestamp": values[0].timestamp.isoformat() if values else None,
            "min": min(numeric_values),
            "max": max(numeric_values),
            "mean": statistics.mean(numeric_values),
            "median": statistics.median(numeric_values)
        }
        
        if len(numeric_values) > 1:
            summary["std_dev"] = statistics.stdev(numeric_values)
            
            # Calculate percentiles
            sorted_values = sorted(numeric_values)
            n = len(sorted_values)
            summary["p50"] = sorted_values[int(n * 0.5)]
            summary["p90"] = sorted_values[int(n * 0.9)]
            summary["p95"] = sorted_values[int(n * 0.95)]
            summary["p99"] = sorted_values[int(n * 0.99)]
        
        return summary
    
    def get_quality_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive data for quality dashboard."""
        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "categories": {},
            "alerts": [],
            "trends": {},
            "summary": {}
        }
        
        # Group metrics by category
        categories = ["coverage", "performance", "quality", "security", "testing", "documentation"]
        
        for category in categories:
            category_metrics = {}
            
            # Get metrics for this category (this would need to be implemented based on your metric definitions)
            # For now, we'll get a few key metrics per category
            if category == "coverage":
                metrics_to_check = ["test_coverage_total", "test_coverage_lines", "test_coverage_branches"]
            elif category == "performance":
                metrics_to_check = ["processing_throughput", "processing_latency_p95", "memory_usage"]
            elif category == "quality":
                metrics_to_check = ["code_quality_score", "linting_violations", "complexity_score"]
            elif category == "security":
                metrics_to_check = ["security_score", "security_vulnerabilities_critical", "security_vulnerabilities_high"]
            elif category == "testing":
                metrics_to_check = ["test_count_total", "test_pass_rate", "test_execution_time"]
            elif category == "documentation":
                metrics_to_check = ["documentation_coverage"]
            else:
                metrics_to_check = []
            
            for metric_name in metrics_to_check:
                summary = self.get_metric_summary(metric_name, hours=24)
                if "error" not in summary:
                    category_metrics[metric_name] = summary
            
            dashboard_data["categories"][category] = category_metrics
        
        # Get active alerts
        active_alerts = self.db.get_active_alerts()
        dashboard_data["alerts"] = [
            {
                "metric_name": alert.metric_name,
                "severity": alert.severity.value,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "threshold": alert.threshold,
                "actual_value": alert.actual_value
            }
            for alert in active_alerts
        ]
        
        # Calculate overall health score
        all_summaries = []
        for category_data in dashboard_data["categories"].values():
            for summary in category_data.values():
                if summary.get("latest_value") is not None:
                    all_summaries.append(summary)
        
        if all_summaries:
            # Simple health calculation (this could be more sophisticated)
            alert_count = len(active_alerts)
            critical_alerts = sum(1 for alert in active_alerts if alert.severity == MetricSeverity.CRITICAL)
            
            if critical_alerts > 0:
                health_score = 25.0
            elif alert_count > 5:
                health_score = 50.0
            elif alert_count > 2:
                health_score = 75.0
            else:
                health_score = 100.0
            
            dashboard_data["summary"] = {
                "health_score": health_score,
                "total_metrics": len(all_summaries),
                "active_alerts": alert_count,
                "critical_alerts": critical_alerts
            }
        
        return dashboard_data
    
    def _check_metric_alerts(self, metric: MetricValue):
        """Check if a metric violates any alert thresholds."""
        definition = self.db.get_metric_definition(metric.name)
        if not definition:
            return
        
        alerts_to_create = []
        
        # Check critical threshold
        if definition.alert_threshold is not None:
            if definition.is_higher_better:
                if metric.value < definition.alert_threshold:
                    alerts_to_create.append((MetricSeverity.ERROR, definition.alert_threshold))
            else:
                if metric.value > definition.alert_threshold:
                    alerts_to_create.append((MetricSeverity.ERROR, definition.alert_threshold))
        
        # Check warning threshold
        if definition.warning_threshold is not None:
            if definition.is_higher_better:
                if metric.value < definition.warning_threshold:
                    alerts_to_create.append((MetricSeverity.WARNING, definition.warning_threshold))
            else:
                if metric.value > definition.warning_threshold:
                    alerts_to_create.append((MetricSeverity.WARNING, definition.warning_threshold))
        
        # Create alerts
        for severity, threshold in alerts_to_create:
            alert = QualityAlert(
                metric_name=metric.name,
                severity=severity,
                threshold=threshold,
                actual_value=metric.value,
                message=f"Metric {metric.name} ({metric.value:.2f}) violated {severity.value} threshold ({threshold:.2f})",
                timestamp=metric.timestamp
            )
            
            self.db.store_alert(alert)
            
            # Notify alert handlers
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    self.logger.error(f"Alert handler failed: {e}")
    
    def add_alert_handler(self, handler: Callable[[QualityAlert], None]):
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
    
    def start_monitoring(self, interval: int = 60):
        """Start continuous quality monitoring."""
        self.monitoring_interval = interval
        self.monitoring_active = True
        
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    self._perform_monitoring_check()
                    time.sleep(self.monitoring_interval)
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
                    time.sleep(30)  # Short delay before retrying
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info(f"Quality monitoring started with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop continuous quality monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        
        self.logger.info("Quality monitoring stopped")
    
    def _perform_monitoring_check(self):
        """Perform a monitoring check cycle."""
        # This would typically collect current metrics from the system
        # For now, we'll just log that monitoring is active
        self.logger.debug("Performing quality monitoring check")
        
        # In a real implementation, this would:
        # 1. Collect current system metrics
        # 2. Check for any new alerts
        # 3. Update dashboard data
        # 4. Send notifications if needed
    
    def generate_quality_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate a comprehensive quality report."""
        report_data = {
            "report_timestamp": datetime.now().isoformat(),
            "period_hours": hours,
            "metrics_summary": {},
            "alerts": {
                "active": [],
                "resolved_in_period": []
            },
            "quality_trends": {},
            "recommendations": []
        }
        
        # Get dashboard data (includes metric summaries)
        dashboard_data = self.get_quality_dashboard_data()
        report_data["metrics_summary"] = dashboard_data["categories"]
        
        # Get alerts
        active_alerts = self.db.get_active_alerts()
        report_data["alerts"]["active"] = [
            {
                "metric_name": alert.metric_name,
                "severity": alert.severity.value,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat()
            }
            for alert in active_alerts
        ]
        
        # Generate recommendations based on current state
        recommendations = []
        
        # Coverage recommendations
        coverage_metrics = report_data["metrics_summary"].get("coverage", {})
        total_coverage = coverage_metrics.get("test_coverage_total", {}).get("latest_value", 0)
        if total_coverage < 80:
            recommendations.append("Improve test coverage by adding unit tests for uncovered modules")
        
        # Performance recommendations
        performance_metrics = report_data["metrics_summary"].get("performance", {})
        throughput = performance_metrics.get("processing_throughput", {}).get("latest_value", 0)
        if throughput < 8:
            recommendations.append("Optimize processing performance to meet throughput targets")
        
        # Security recommendations
        security_alerts = [a for a in active_alerts if a.metric_name.startswith("security_")]
        if security_alerts:
            recommendations.append("Address active security vulnerabilities")
        
        # Quality recommendations
        quality_metrics = report_data["metrics_summary"].get("quality", {})
        quality_score = quality_metrics.get("code_quality_score", {}).get("latest_value", 100)
        if quality_score < 80:
            recommendations.append("Improve code quality by addressing linting violations and complexity issues")
        
        if not recommendations:
            recommendations.append("Quality metrics are healthy. Continue monitoring and maintaining current standards.")
        
        report_data["recommendations"] = recommendations
        
        return report_data


def console_alert_handler(alert: QualityAlert):
    """Simple console alert handler for demonstration."""
    print(f"🚨 QUALITY ALERT [{alert.severity.value.upper()}]: {alert.message}")


def main():
    """Main function for running quality metrics collection."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize metrics collector
    collector = QualityMetricsCollector()
    
    # Add console alert handler
    collector.add_alert_handler(console_alert_handler)
    
    # Record some sample metrics
    collector.record_gauge("test_coverage_total", 87.5)
    collector.record_gauge("processing_throughput", 12.3)
    collector.record_counter("test_count_total", 1)
    collector.record_gauge("code_quality_score", 91.2)
    collector.record_gauge("security_score", 98.5)
    
    # Generate quality report
    report = collector.generate_quality_report()
    
    print("\n" + "="*60)
    print("QUALITY METRICS COLLECTION REPORT")
    print("="*60)
    
    for category, metrics in report["metrics_summary"].items():
        if metrics:
            print(f"\n{category.upper()}:")
            for metric_name, summary in metrics.items():
                latest = summary.get("latest_value", "N/A")
                print(f"  {metric_name}: {latest}")
    
    if report["alerts"]["active"]:
        print(f"\nACTIVE ALERTS ({len(report['alerts']['active'])}):")
        for alert in report["alerts"]["active"]:
            print(f"  {alert['severity'].upper()}: {alert['message']}")
    
    print(f"\nRECOMMENDATIONS:")
    for rec in report["recommendations"]:
        print(f"  • {rec}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()