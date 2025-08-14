"""
Research Metrics Collector for Story 4.2 Sanskrit Processing Enhancement

Collects and aggregates research-grade quality metrics and comprehensive reporting
for academic validation and publication standards.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
import logging
import json
import statistics
from datetime import datetime, timedelta
from pathlib import Path
import uuid


class MetricCategory(Enum):
    """Categories of research metrics."""
    ACCURACY = "accuracy"
    PERFORMANCE = "performance" 
    QUALITY = "quality"
    COMPLIANCE = "compliance"
    IMPROVEMENT = "improvement"


class MetricSeverity(Enum):
    """Severity levels for metric issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ResearchMetric:
    """Individual research metric with metadata."""
    metric_id: str
    name: str
    category: MetricCategory
    value: float
    unit: str
    timestamp: str
    source_component: str
    metadata: Dict[str, Any]
    severity: MetricSeverity
    target_value: Optional[float]
    meets_target: bool


@dataclass
class MetricCollection:
    """Collection of related research metrics."""
    collection_id: str
    name: str
    description: str
    metrics: List[ResearchMetric]
    collection_timestamp: str
    aggregated_score: float
    quality_gates_passed: int
    total_quality_gates: int
    academic_compliance_score: float


@dataclass
class TrendAnalysis:
    """Trend analysis for metric values over time."""
    metric_name: str
    time_period_days: int
    trend_direction: str  # 'improving', 'declining', 'stable'
    rate_of_change: float
    statistical_significance: bool
    confidence_interval: Tuple[float, float]
    data_points_count: int


class ResearchMetricsCollector:
    """
    Research-grade metrics collector for Sanskrit processing enhancement.
    
    Provides comprehensive metrics collection, aggregation, and analysis
    for academic validation and research publication standards.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the research metrics collector.
        
        Args:
            config: Configuration parameters
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # Metric storage
        self.metrics: List[ResearchMetric] = []
        self.metric_collections: List[MetricCollection] = []
        
        # Target values for quality gates
        self.target_values = {
            'sanskrit_accuracy': 0.85,
            'iast_compliance': 0.90,
            'processing_time_ms': 1000,
            'precision': 0.85,
            'recall': 0.85,
            'f1_score': 0.85,
            'improvement_percentage': 15.0
        }
        
        # Initialize storage directories
        self._initialize_storage()

    def _get_default_config(self) -> Dict:
        """Get default configuration for metrics collector."""
        return {
            'metrics_retention_days': 365,
            'quality_gate_threshold': 0.8,
            'academic_compliance_threshold': 0.9,
            'statistical_significance_threshold': 0.05,
            'trend_analysis_min_points': 5,
            'enable_automated_reporting': True,
            'report_generation_interval_hours': 24
        }

    def _initialize_storage(self):
        """Initialize metrics storage directories."""
        storage_path = Path("data/research_metrics")
        storage_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (storage_path / "raw_metrics").mkdir(exist_ok=True)
        (storage_path / "collections").mkdir(exist_ok=True)
        (storage_path / "reports").mkdir(exist_ok=True)

    def record_metric(
        self,
        name: str,
        value: float,
        category: MetricCategory,
        unit: str = "",
        source_component: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
        target_value: Optional[float] = None
    ) -> ResearchMetric:
        """
        Record a research metric.
        
        Args:
            name: Metric name
            value: Metric value
            category: Metric category
            unit: Unit of measurement
            source_component: Component that generated the metric
            metadata: Additional metadata
            target_value: Optional target value for quality gates
            
        Returns:
            Created ResearchMetric
        """
        try:
            # Generate unique ID
            metric_id = str(uuid.uuid4())
            
            # Determine target value
            if target_value is None:
                target_value = self.target_values.get(name)
            
            # Check if meets target
            meets_target = False
            if target_value is not None:
                meets_target = value >= target_value
            
            # Determine severity based on target compliance
            if meets_target:
                severity = MetricSeverity.INFO
            elif target_value and value >= target_value * 0.8:
                severity = MetricSeverity.LOW
            elif target_value and value >= target_value * 0.6:
                severity = MetricSeverity.MEDIUM
            else:
                severity = MetricSeverity.HIGH
            
            metric = ResearchMetric(
                metric_id=metric_id,
                name=name,
                category=category,
                value=value,
                unit=unit,
                timestamp=datetime.now().isoformat(),
                source_component=source_component,
                metadata=metadata or {},
                severity=severity,
                target_value=target_value,
                meets_target=meets_target
            )
            
            self.metrics.append(metric)
            
            self.logger.debug(f"Recorded metric '{name}': {value} {unit} (meets target: {meets_target})")
            
            return metric
            
        except Exception as e:
            self.logger.error(f"Error recording metric '{name}': {e}")
            raise

    def create_metric_collection(
        self,
        name: str,
        description: str,
        metric_names: List[str],
        collection_id: Optional[str] = None
    ) -> MetricCollection:
        """
        Create a collection of related metrics.
        
        Args:
            name: Collection name
            description: Collection description
            metric_names: Names of metrics to include
            collection_id: Optional collection ID
            
        Returns:
            Created MetricCollection
        """
        try:
            collection_id = collection_id or str(uuid.uuid4())
            
            # Find metrics by name (latest for each)
            collection_metrics = []
            for metric_name in metric_names:
                matching_metrics = [m for m in self.metrics if m.name == metric_name]
                if matching_metrics:
                    # Get latest metric
                    latest_metric = max(matching_metrics, key=lambda m: m.timestamp)
                    collection_metrics.append(latest_metric)
            
            # Calculate aggregated score
            if collection_metrics:
                scores = []
                for metric in collection_metrics:
                    if metric.target_value:
                        # Normalize score against target (0-1)
                        normalized_score = min(metric.value / metric.target_value, 1.0)
                        scores.append(normalized_score)
                
                aggregated_score = statistics.mean(scores) if scores else 0.0
            else:
                aggregated_score = 0.0
            
            # Count quality gates
            quality_gates_passed = sum(1 for m in collection_metrics if m.meets_target)
            total_quality_gates = len([m for m in collection_metrics if m.target_value is not None])
            
            # Calculate academic compliance score
            academic_compliance_score = self._calculate_academic_compliance(collection_metrics)
            
            collection = MetricCollection(
                collection_id=collection_id,
                name=name,
                description=description,
                metrics=collection_metrics,
                collection_timestamp=datetime.now().isoformat(),
                aggregated_score=aggregated_score,
                quality_gates_passed=quality_gates_passed,
                total_quality_gates=total_quality_gates,
                academic_compliance_score=academic_compliance_score
            )
            
            self.metric_collections.append(collection)
            
            self.logger.info(f"Created metric collection '{name}' with {len(collection_metrics)} metrics")
            
            return collection
            
        except Exception as e:
            self.logger.error(f"Error creating metric collection '{name}': {e}")
            raise

    def _calculate_academic_compliance(self, metrics: List[ResearchMetric]) -> float:
        """Calculate academic compliance score for metrics."""
        compliance_factors = []
        
        for metric in metrics:
            if metric.name in ['iast_compliance', 'sanskrit_accuracy', 'precision', 'recall']:
                # These are key academic metrics
                if metric.target_value:
                    compliance = min(metric.value / metric.target_value, 1.0)
                    compliance_factors.append(compliance)
        
        return statistics.mean(compliance_factors) if compliance_factors else 0.0

    def analyze_trends(
        self,
        metric_name: str,
        time_period_days: int = 30
    ) -> Optional[TrendAnalysis]:
        """
        Analyze trends for a specific metric over time.
        
        Args:
            metric_name: Name of metric to analyze
            time_period_days: Time period for analysis
            
        Returns:
            TrendAnalysis if sufficient data available
        """
        try:
            # Filter metrics by name and time period
            cutoff_date = datetime.now() - timedelta(days=time_period_days)
            
            relevant_metrics = [
                m for m in self.metrics 
                if m.name == metric_name and 
                datetime.fromisoformat(m.timestamp) >= cutoff_date
            ]
            
            if len(relevant_metrics) < self.config['trend_analysis_min_points']:
                self.logger.warning(f"Insufficient data points for trend analysis of '{metric_name}'")
                return None
            
            # Sort by timestamp
            relevant_metrics.sort(key=lambda m: m.timestamp)
            
            # Extract values and timestamps
            values = [m.value for m in relevant_metrics]
            timestamps = [datetime.fromisoformat(m.timestamp) for m in relevant_metrics]
            
            # Simple linear trend analysis
            n = len(values)
            x_mean = n / 2  # Simplified time index
            y_mean = statistics.mean(values)
            
            # Calculate slope (rate of change)
            numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
            denominator = sum((i - x_mean) ** 2 for i in range(n))
            
            if denominator == 0:
                rate_of_change = 0.0
            else:
                rate_of_change = numerator / denominator
            
            # Determine trend direction
            if rate_of_change > 0.01:
                trend_direction = 'improving'
            elif rate_of_change < -0.01:
                trend_direction = 'declining'
            else:
                trend_direction = 'stable'
            
            # Statistical significance (simplified)
            statistical_significance = abs(rate_of_change) > 0.05 and n >= 10
            
            # Confidence interval (simplified)
            std_dev = statistics.stdev(values) if n > 1 else 0.0
            margin = 1.96 * std_dev / (n ** 0.5)  # 95% confidence
            confidence_interval = (y_mean - margin, y_mean + margin)
            
            return TrendAnalysis(
                metric_name=metric_name,
                time_period_days=time_period_days,
                trend_direction=trend_direction,
                rate_of_change=rate_of_change,
                statistical_significance=statistical_significance,
                confidence_interval=confidence_interval,
                data_points_count=n
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing trends for '{metric_name}': {e}")
            return None

    def generate_research_dashboard(self) -> Dict[str, Any]:
        """Generate comprehensive research dashboard."""
        try:
            # Get latest metrics by category
            latest_metrics_by_category = {}
            for category in MetricCategory:
                category_metrics = [m for m in self.metrics if m.category == category]
                if category_metrics:
                    # Group by name and get latest for each
                    by_name = {}
                    for metric in category_metrics:
                        if metric.name not in by_name or metric.timestamp > by_name[metric.name].timestamp:
                            by_name[metric.name] = metric
                    latest_metrics_by_category[category.value] = list(by_name.values())
                else:
                    latest_metrics_by_category[category.value] = []
            
            # Calculate overall quality score
            all_latest_metrics = []
            for metrics in latest_metrics_by_category.values():
                all_latest_metrics.extend(metrics)
            
            if all_latest_metrics:
                quality_scores = []
                for metric in all_latest_metrics:
                    if metric.target_value:
                        score = min(metric.value / metric.target_value, 1.0)
                        quality_scores.append(score)
                
                overall_quality_score = statistics.mean(quality_scores) if quality_scores else 0.0
            else:
                overall_quality_score = 0.0
            
            # Quality gates summary
            total_gates = sum(1 for m in all_latest_metrics if m.target_value is not None)
            passed_gates = sum(1 for m in all_latest_metrics if m.meets_target)
            
            # Recent trend analysis
            key_metrics = ['sanskrit_accuracy', 'iast_compliance', 'processing_time_ms']
            trends = {}
            for metric_name in key_metrics:
                trend = self.analyze_trends(metric_name, 7)  # Last week
                if trend:
                    trends[metric_name] = {
                        'direction': trend.trend_direction,
                        'rate_of_change': trend.rate_of_change,
                        'significant': trend.statistical_significance
                    }
            
            dashboard = {
                'dashboard_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'total_metrics': len(self.metrics),
                    'metric_collections': len(self.metric_collections),
                    'data_retention_days': self.config['metrics_retention_days']
                },
                'quality_overview': {
                    'overall_quality_score': overall_quality_score,
                    'quality_gates_passed': passed_gates,
                    'total_quality_gates': total_gates,
                    'quality_gate_success_rate': passed_gates / total_gates if total_gates > 0 else 0.0
                },
                'metrics_by_category': latest_metrics_by_category,
                'trend_analysis': trends,
                'alerts': self._generate_alerts(all_latest_metrics),
                'academic_compliance': {
                    'overall_compliance_score': self._calculate_academic_compliance(all_latest_metrics),
                    'compliance_threshold': self.config['academic_compliance_threshold'],
                    'meets_academic_standards': self._calculate_academic_compliance(all_latest_metrics) >= self.config['academic_compliance_threshold']
                }
            }
            
            return dashboard
            
        except Exception as e:
            self.logger.error(f"Error generating research dashboard: {e}")
            return {"error": str(e)}

    def _generate_alerts(self, metrics: List[ResearchMetric]) -> List[Dict[str, Any]]:
        """Generate alerts based on metric values and targets."""
        alerts = []
        
        for metric in metrics:
            if metric.severity in [MetricSeverity.CRITICAL, MetricSeverity.HIGH]:
                alert_type = "quality_gate_failure"
                message = f"{metric.name} below target: {metric.value} {metric.unit}"
                
                if metric.target_value:
                    shortfall_percent = ((metric.target_value - metric.value) / metric.target_value) * 100
                    message += f" (target: {metric.target_value}, shortfall: {shortfall_percent:.1f}%)"
                
                alerts.append({
                    'type': alert_type,
                    'severity': metric.severity.value,
                    'message': message,
                    'metric_name': metric.name,
                    'timestamp': metric.timestamp,
                    'source_component': metric.source_component
                })
        
        return alerts

    def export_research_data(self, file_path: Optional[Path] = None, format: str = 'json') -> bool:
        """
        Export research metrics data for external analysis.
        
        Args:
            file_path: Optional output file path
            format: Export format ('json' or 'csv')
            
        Returns:
            True if export successful
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if file_path is None:
                file_path = Path(f"data/research_metrics/exports/research_data_{timestamp}.{format}")
            
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == 'json':
                # Export as JSON
                export_data = {
                    'export_metadata': {
                        'generated_at': datetime.now().isoformat(),
                        'total_metrics': len(self.metrics),
                        'metric_collections': len(self.metric_collections),
                        'format': 'json'
                    },
                    'metrics': [asdict(m) for m in self.metrics],
                    'metric_collections': [asdict(c) for c in self.metric_collections],
                    'target_values': self.target_values,
                    'config': self.config
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            elif format.lower() == 'csv':
                # Export as CSV (simplified)
                import csv
                
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    
                    # Header
                    writer.writerow([
                        'metric_id', 'name', 'category', 'value', 'unit', 'timestamp',
                        'source_component', 'severity', 'target_value', 'meets_target'
                    ])
                    
                    # Data rows
                    for metric in self.metrics:
                        writer.writerow([
                            metric.metric_id,
                            metric.name,
                            metric.category.value,
                            metric.value,
                            metric.unit,
                            metric.timestamp,
                            metric.source_component,
                            metric.severity.value,
                            metric.target_value,
                            metric.meets_target
                        ])
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            self.logger.info(f"Exported research data to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting research data: {e}")
            return False

    def cleanup_old_metrics(self, retention_days: Optional[int] = None) -> int:
        """
        Clean up old metrics beyond retention period.
        
        Args:
            retention_days: Optional custom retention period
            
        Returns:
            Number of metrics removed
        """
        try:
            retention_days = retention_days or self.config['metrics_retention_days']
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            original_count = len(self.metrics)
            
            # Filter out old metrics
            self.metrics = [
                m for m in self.metrics
                if datetime.fromisoformat(m.timestamp) >= cutoff_date
            ]
            
            # Also clean up old collections
            self.metric_collections = [
                c for c in self.metric_collections
                if datetime.fromisoformat(c.collection_timestamp) >= cutoff_date
            ]
            
            removed_count = original_count - len(self.metrics)
            
            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} old metrics (retention: {retention_days} days)")
            
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old metrics: {e}")
            return 0


# Test function for development
def test_research_metrics_collector():
    """Test the research metrics collector."""
    collector = ResearchMetricsCollector()
    
    # Record some test metrics
    collector.record_metric(
        "sanskrit_accuracy",
        0.87,
        MetricCategory.ACCURACY,
        unit="",
        source_component="sanskrit_processor"
    )
    
    collector.record_metric(
        "processing_time_ms",
        850,
        MetricCategory.PERFORMANCE,
        unit="ms",
        source_component="mcp_transformer"
    )
    
    collector.record_metric(
        "iast_compliance",
        0.92,
        MetricCategory.COMPLIANCE,
        unit="",
        source_component="accuracy_validator"
    )
    
    # Create a metric collection
    collection = collector.create_metric_collection(
        "Sanskrit Processing Quality",
        "Overall quality metrics for Sanskrit processing",
        ["sanskrit_accuracy", "processing_time_ms", "iast_compliance"]
    )
    
    print(f"Collection quality score: {collection.aggregated_score:.3f}")
    print(f"Quality gates passed: {collection.quality_gates_passed}/{collection.total_quality_gates}")
    
    # Generate dashboard
    dashboard = collector.generate_research_dashboard()
    print(f"Overall quality score: {dashboard['quality_overview']['overall_quality_score']:.3f}")
    print(f"Academic compliance: {dashboard['academic_compliance']['overall_compliance_score']:.3f}")
    
    # Export data
    success = collector.export_research_data()
    print(f"Data export: {'Success' if success else 'Failed'}")


if __name__ == "__main__":
    test_research_metrics_collector()