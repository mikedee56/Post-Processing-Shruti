"""
Quality Metrics Collection and Analysis System for Story 5.5
Implements comprehensive quality metrics tracking and reporting
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import statistics
import threading
from contextlib import contextmanager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QualityMetric:
    """Represents a single quality metric measurement"""
    metric_name: str
    value: float
    unit: str
    category: str
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'metric_name': self.metric_name,
            'value': self.value,
            'unit': self.unit,
            'category': self.category,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context
        }


@dataclass
class QualityTrend:
    """Represents quality metric trends over time"""
    metric_name: str
    current_value: float
    previous_value: float
    change_percentage: float
    trend_direction: str  # 'improving', 'degrading', 'stable'
    period_days: int
    data_points: List[float] = field(default_factory=list)


@dataclass
class QualityDashboard:
    """Quality dashboard data structure"""
    timestamp: datetime
    overall_health_score: float
    category_scores: Dict[str, float]
    recent_trends: List[QualityTrend]
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    summary_stats: Dict[str, Any] = field(default_factory=dict)


class QualityMetricsCollector:
    """Comprehensive quality metrics collection and analysis system"""
    
    def __init__(self, db_path: str = "qa/metrics/quality_metrics.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialize_database()
        
        # Quality thresholds
        self.thresholds = {
            'code_quality': {
                'excellent': 0.95,
                'good': 0.85,
                'acceptable': 0.70,
                'poor': 0.50
            },
            'security': {
                'excellent': 0.98,
                'good': 0.90,
                'acceptable': 0.80,
                'poor': 0.60
            },
            'performance': {
                'excellent': 0.90,
                'good': 0.80,
                'acceptable': 0.70,
                'poor': 0.50
            },
            'documentation': {
                'excellent': 0.90,
                'good': 0.75,
                'acceptable': 0.60,
                'poor': 0.40
            },
            'test_coverage': {
                'excellent': 0.95,
                'good': 0.85,
                'acceptable': 0.70,
                'poor': 0.50
            }
        }
        
    def _initialize_database(self):
        """Initialize SQLite database for metrics storage"""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT NOT NULL,
                    category TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    context TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp 
                ON quality_metrics(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_category 
                ON quality_metrics(category)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_name 
                ON quality_metrics(metric_name)
            """)
    
    @contextmanager
    def _get_connection(self):
        """Thread-safe database connection context manager"""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()
    
    def record_metric(self, metric: QualityMetric):
        """Record a single quality metric"""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO quality_metrics 
                (metric_name, value, unit, category, timestamp, context)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                metric.metric_name,
                metric.value,
                metric.unit,
                metric.category,
                metric.timestamp.isoformat(),
                json.dumps(metric.context)
            ))
            conn.commit()
        
        logger.debug(f"Recorded metric: {metric.metric_name} = {metric.value} {metric.unit}")
    
    def record_batch_metrics(self, metrics: List[QualityMetric]):
        """Record multiple quality metrics in batch"""
        with self._get_connection() as conn:
            data = [
                (
                    m.metric_name,
                    m.value,
                    m.unit,
                    m.category,
                    m.timestamp.isoformat(),
                    json.dumps(m.context)
                )
                for m in metrics
            ]
            
            conn.executemany("""
                INSERT INTO quality_metrics 
                (metric_name, value, unit, category, timestamp, context)
                VALUES (?, ?, ?, ?, ?, ?)
            """, data)
            conn.commit()
        
        logger.info(f"Recorded {len(metrics)} quality metrics")
    
    def get_recent_metrics(self, 
                          category: Optional[str] = None,
                          hours: int = 24) -> List[QualityMetric]:
        """Get recent quality metrics"""
        since = datetime.now() - timedelta(hours=hours)
        
        query = """
            SELECT * FROM quality_metrics 
            WHERE timestamp >= ?
        """
        params = [since.isoformat()]
        
        if category:
            query += " AND category = ?"
            params.append(category)
        
        query += " ORDER BY timestamp DESC"
        
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            
            metrics = []
            for row in rows:
                context = json.loads(row['context']) if row['context'] else {}
                metrics.append(QualityMetric(
                    metric_name=row['metric_name'],
                    value=row['value'],
                    unit=row['unit'],
                    category=row['category'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    context=context
                ))
            
            return metrics
    
    def calculate_trend(self, 
                       metric_name: str,
                       days: int = 7) -> Optional[QualityTrend]:
        """Calculate quality trend for a specific metric"""
        since = datetime.now() - timedelta(days=days)
        
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT value, timestamp FROM quality_metrics
                WHERE metric_name = ? AND timestamp >= ?
                ORDER BY timestamp ASC
            """, (metric_name, since.isoformat())).fetchall()
            
            if len(rows) < 2:
                return None
            
            values = [row['value'] for row in rows]
            current_value = values[-1]
            previous_value = values[0]
            
            change_percentage = ((current_value - previous_value) / previous_value * 100) if previous_value != 0 else 0
            
            # Determine trend direction
            if abs(change_percentage) < 2:
                trend_direction = 'stable'
            elif change_percentage > 0:
                trend_direction = 'improving'
            else:
                trend_direction = 'degrading'
            
            return QualityTrend(
                metric_name=metric_name,
                current_value=current_value,
                previous_value=previous_value,
                change_percentage=change_percentage,
                trend_direction=trend_direction,
                period_days=days,
                data_points=values
            )
    
    def get_category_health_score(self, category: str, hours: int = 24) -> float:
        """Calculate overall health score for a quality category"""
        metrics = self.get_recent_metrics(category=category, hours=hours)
        
        if not metrics:
            return 0.0
        
        # Group metrics by name and get latest values
        latest_metrics = {}
        for metric in metrics:
            if metric.metric_name not in latest_metrics or metric.timestamp > latest_metrics[metric.metric_name].timestamp:
                latest_metrics[metric.metric_name] = metric
        
        # Calculate weighted average (normalize all values to 0-1 scale)
        total_score = 0.0
        total_weight = 0.0
        
        for metric in latest_metrics.values():
            # Normalize different metric types to 0-1 scale
            normalized_value = self._normalize_metric_value(metric)
            weight = self._get_metric_weight(metric.metric_name)
            
            total_score += normalized_value * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _normalize_metric_value(self, metric: QualityMetric) -> float:
        """Normalize metric value to 0-1 scale"""
        if metric.unit == 'percentage' or metric.unit == 'ratio':
            return min(1.0, max(0.0, metric.value))
        elif metric.unit == 'count':
            # For counts, lower is usually better (e.g., error count)
            if 'error' in metric.metric_name.lower() or 'issue' in metric.metric_name.lower():
                return max(0.0, 1.0 - (metric.value / 100.0))  # Assume 100+ errors = 0 score
            else:
                return min(1.0, metric.value / 100.0)  # Assume 100+ good things = 1.0 score
        elif metric.unit == 'seconds' or metric.unit == 'milliseconds':
            # For time metrics, lower is usually better
            max_acceptable = 10.0 if metric.unit == 'seconds' else 10000.0
            return max(0.0, 1.0 - (metric.value / max_acceptable))
        else:
            # Default: assume 0-1 scale
            return min(1.0, max(0.0, metric.value))
    
    def _get_metric_weight(self, metric_name: str) -> float:
        """Get weight for different metric types"""
        weights = {
            'code_coverage': 2.0,
            'security_score': 3.0,
            'performance_score': 2.0,
            'documentation_coverage': 1.5,
            'error_rate': 2.5,
            'response_time': 2.0,
            'memory_usage': 1.5,
            'cpu_usage': 1.5,
        }
        
        for pattern, weight in weights.items():
            if pattern in metric_name.lower():
                return weight
        
        return 1.0  # Default weight
    
    def generate_quality_dashboard(self) -> QualityDashboard:
        """Generate comprehensive quality dashboard"""
        timestamp = datetime.now()
        
        # Calculate category scores
        categories = ['code_quality', 'security', 'performance', 'documentation', 'test_coverage']
        category_scores = {}
        
        for category in categories:
            category_scores[category] = self.get_category_health_score(category)
        
        # Calculate overall health score
        overall_health_score = sum(category_scores.values()) / len(category_scores) if category_scores else 0.0
        
        # Get recent trends
        recent_trends = []
        metric_names = self._get_unique_metric_names()
        
        for metric_name in metric_names[:10]:  # Limit to top 10 metrics
            trend = self.calculate_trend(metric_name)
            if trend:
                recent_trends.append(trend)
        
        # Generate alerts
        alerts = self._generate_quality_alerts(category_scores, recent_trends)
        
        # Summary statistics
        summary_stats = {
            'total_metrics': self._count_total_metrics(),
            'metrics_last_24h': len(self.get_recent_metrics(hours=24)),
            'active_categories': len([s for s in category_scores.values() if s > 0]),
            'health_trend': self._calculate_overall_trend(),
            'last_updated': timestamp.isoformat()
        }
        
        return QualityDashboard(
            timestamp=timestamp,
            overall_health_score=overall_health_score,
            category_scores=category_scores,
            recent_trends=recent_trends,
            alerts=alerts,
            summary_stats=summary_stats
        )
    
    def _get_unique_metric_names(self) -> List[str]:
        """Get list of unique metric names"""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT DISTINCT metric_name FROM quality_metrics
                ORDER BY metric_name
            """).fetchall()
            
            return [row['metric_name'] for row in rows]
    
    def _count_total_metrics(self) -> int:
        """Count total number of recorded metrics"""
        with self._get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) as count FROM quality_metrics").fetchone()
            return row['count']
    
    def _calculate_overall_trend(self) -> str:
        """Calculate overall quality trend"""
        # Simple implementation: check if more metrics are improving vs degrading
        metric_names = self._get_unique_metric_names()
        trends = [self.calculate_trend(name) for name in metric_names[:5]]
        trends = [t for t in trends if t is not None]
        
        if not trends:
            return 'unknown'
        
        improving = sum(1 for t in trends if t.trend_direction == 'improving')
        degrading = sum(1 for t in trends if t.trend_direction == 'degrading')
        
        if improving > degrading:
            return 'improving'
        elif degrading > improving:
            return 'degrading'
        else:
            return 'stable'
    
    def _generate_quality_alerts(self, 
                                category_scores: Dict[str, float],
                                trends: List[QualityTrend]) -> List[Dict[str, Any]]:
        """Generate quality alerts based on scores and trends"""
        alerts = []
        
        # Category score alerts
        for category, score in category_scores.items():
            threshold_levels = self.thresholds.get(category, self.thresholds['code_quality'])
            
            if score < threshold_levels['poor']:
                alerts.append({
                    'type': 'critical',
                    'category': category,
                    'message': f"{category.replace('_', ' ').title()} score is critically low: {score:.2f}",
                    'score': score,
                    'threshold': threshold_levels['poor'],
                    'timestamp': datetime.now().isoformat()
                })
            elif score < threshold_levels['acceptable']:
                alerts.append({
                    'type': 'warning',
                    'category': category,
                    'message': f"{category.replace('_', ' ').title()} score is below acceptable: {score:.2f}",
                    'score': score,
                    'threshold': threshold_levels['acceptable'],
                    'timestamp': datetime.now().isoformat()
                })
        
        # Trend alerts
        for trend in trends:
            if trend.trend_direction == 'degrading' and abs(trend.change_percentage) > 10:
                alerts.append({
                    'type': 'warning',
                    'category': 'trend',
                    'message': f"{trend.metric_name} has degraded by {abs(trend.change_percentage):.1f}% over {trend.period_days} days",
                    'metric': trend.metric_name,
                    'change': trend.change_percentage,
                    'timestamp': datetime.now().isoformat()
                })
        
        return sorted(alerts, key=lambda x: ('critical', 'warning', 'info').index(x['type']))
    
    def export_metrics(self, 
                      output_file: str,
                      format: str = 'json',
                      days: int = 30) -> str:
        """Export quality metrics to file"""
        since = datetime.now() - timedelta(days=days)
        metrics = []
        
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM quality_metrics
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            """, (since.isoformat(),)).fetchall()
            
            for row in rows:
                context = json.loads(row['context']) if row['context'] else {}
                metrics.append({
                    'metric_name': row['metric_name'],
                    'value': row['value'],
                    'unit': row['unit'],
                    'category': row['category'],
                    'timestamp': row['timestamp'],
                    'context': context
                })
        
        # Export data
        if format.lower() == 'json':
            with open(output_file, 'w') as f:
                json.dump(metrics, f, indent=2)
        elif format.lower() == 'csv':
            import csv
            with open(output_file, 'w', newline='') as f:
                if metrics:
                    writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
                    writer.writeheader()
                    writer.writerows(metrics)
        
        return f"Exported {len(metrics)} metrics to {output_file}"
    
    def cleanup_old_metrics(self, days_to_keep: int = 90):
        """Clean up old metrics to prevent database bloat"""
        cutoff = datetime.now() - timedelta(days=days_to_keep)
        
        with self._get_connection() as conn:
            result = conn.execute("""
                DELETE FROM quality_metrics
                WHERE timestamp < ?
            """, (cutoff.isoformat(),))
            
            deleted_count = result.rowcount
            conn.commit()
        
        logger.info(f"Cleaned up {deleted_count} old metrics (older than {days_to_keep} days)")
        return deleted_count


# Convenience functions for common quality metrics
def record_code_quality_metrics(collector: QualityMetricsCollector, 
                               coverage: float,
                               complexity: float,
                               lint_issues: int):
    """Record common code quality metrics"""
    timestamp = datetime.now()
    
    metrics = [
        QualityMetric(
            metric_name="code_coverage",
            value=coverage,
            unit="percentage",
            category="code_quality",
            timestamp=timestamp
        ),
        QualityMetric(
            metric_name="code_complexity",
            value=complexity,
            unit="score",
            category="code_quality",
            timestamp=timestamp
        ),
        QualityMetric(
            metric_name="lint_issues",
            value=lint_issues,
            unit="count",
            category="code_quality",
            timestamp=timestamp
        )
    ]
    
    collector.record_batch_metrics(metrics)


def record_performance_metrics(collector: QualityMetricsCollector,
                             response_time: float,
                             throughput: float,
                             error_rate: float):
    """Record common performance metrics"""
    timestamp = datetime.now()
    
    metrics = [
        QualityMetric(
            metric_name="response_time",
            value=response_time,
            unit="seconds",
            category="performance",
            timestamp=timestamp
        ),
        QualityMetric(
            metric_name="throughput",
            value=throughput,
            unit="requests_per_second",
            category="performance",
            timestamp=timestamp
        ),
        QualityMetric(
            metric_name="error_rate",
            value=error_rate,
            unit="percentage",
            category="performance",
            timestamp=timestamp
        )
    ]
    
    collector.record_batch_metrics(metrics)


def record_security_metrics(collector: QualityMetricsCollector,
                           security_score: float,
                           vulnerabilities: int,
                           compliance_score: float):
    """Record common security metrics"""
    timestamp = datetime.now()
    
    metrics = [
        QualityMetric(
            metric_name="security_score",
            value=security_score,
            unit="score",
            category="security",
            timestamp=timestamp
        ),
        QualityMetric(
            metric_name="vulnerabilities",
            value=vulnerabilities,
            unit="count",
            category="security",
            timestamp=timestamp
        ),
        QualityMetric(
            metric_name="compliance_score",
            value=compliance_score,
            unit="score",
            category="security",
            timestamp=timestamp
        )
    ]
    
    collector.record_batch_metrics(metrics)


if __name__ == "__main__":
    # Example usage
    collector = QualityMetricsCollector()
    
    # Record some sample metrics
    record_code_quality_metrics(collector, coverage=0.85, complexity=3.2, lint_issues=5)
    record_performance_metrics(collector, response_time=0.15, throughput=45.2, error_rate=0.01)
    record_security_metrics(collector, security_score=0.92, vulnerabilities=0, compliance_score=0.88)
    
    # Generate dashboard
    dashboard = collector.generate_quality_dashboard()
    
    print("Quality Dashboard:")
    print(f"Overall Health Score: {dashboard.overall_health_score:.2f}")
    print("Category Scores:")
    for category, score in dashboard.category_scores.items():
        print(f"  {category}: {score:.2f}")
    
    print(f"Alerts: {len(dashboard.alerts)}")
    for alert in dashboard.alerts:
        print(f"  {alert['type'].upper()}: {alert['message']}")