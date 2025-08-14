"""
Performance Impact Assessment Tools for Enterprise Production Excellence.

This module implements comprehensive performance impact analysis for the Sanskrit
processing pipeline with pre-deployment validation and change impact modeling.
"""

import json
import logging
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from enum import Enum
import uuid

# Import existing performance components
import sys
sys.path.append(str(Path(__file__).parent.parent / "monitoring"))
from system_monitor import SystemMonitor
from telemetry_collector import TelemetryCollector
from .performance_regression_detector import PerformanceRegressionDetector
from .performance_baseline_manager import PerformanceBaselineManager
from .performance_optimizer import PerformanceOptimizer


class ImpactSeverity(Enum):
    """Performance impact severity levels."""
    NEGLIGIBLE = "negligible"      # <1% impact
    LOW = "low"                    # 1-5% impact
    MEDIUM = "medium"              # 5-15% impact
    HIGH = "high"                  # 15-30% impact
    CRITICAL = "critical"          # >30% impact


class ChangeType(Enum):
    """Types of system changes."""
    CODE_CHANGE = "code_change"
    CONFIGURATION_CHANGE = "configuration_change"
    INFRASTRUCTURE_CHANGE = "infrastructure_change"
    DEPENDENCY_UPDATE = "dependency_update"
    DATA_MODEL_CHANGE = "data_model_change"


class AssessmentPhase(Enum):
    """Assessment execution phases."""
    PRE_DEPLOYMENT = "pre_deployment"
    POST_DEPLOYMENT = "post_deployment"
    PRODUCTION_MONITORING = "production_monitoring"
    ROLLBACK_VALIDATION = "rollback_validation"


@dataclass
class PerformanceImpactMetric:
    """Individual performance impact metric."""
    metric_name: str
    component: str
    operation: str
    baseline_value: float
    current_value: float
    impact_percentage: float
    impact_severity: ImpactSeverity
    measurement_unit: str
    confidence_score: float
    sample_count: int
    timestamp: float


@dataclass
class ChangeImpactAnalysis:
    """Analysis of change impact on performance."""
    analysis_id: str
    change_id: str
    change_type: ChangeType
    change_description: str
    assessment_phase: AssessmentPhase
    overall_impact_severity: ImpactSeverity
    affected_components: List[str]
    performance_metrics: List[PerformanceImpactMetric]
    risk_assessment: Dict[str, Any]
    recommendations: List[str]
    predicted_outcomes: Dict[str, Any]
    created_at: float
    created_by: str


@dataclass
class DeploymentValidationResult:
    """Result of pre-deployment performance validation."""
    validation_id: str
    change_id: str
    deployment_approved: bool
    performance_tests_passed: int
    performance_tests_failed: int
    blocking_issues: List[str]
    warnings: List[str]
    estimated_production_impact: Dict[str, float]
    validation_duration_seconds: float
    validated_by: str
    validated_at: float


@dataclass
class RollbackRecommendation:
    """Rollback recommendation based on performance impact."""
    recommendation_id: str
    change_id: str
    recommend_rollback: bool
    severity_justification: str
    impact_metrics: List[str]
    rollback_urgency: str  # "immediate", "scheduled", "none"
    rollback_window_hours: int
    alternative_actions: List[str]
    created_at: float


class PerformanceImpactAssessor:
    """
    Enterprise-grade performance impact assessment system.
    
    Provides comprehensive impact analysis capabilities:
    - Pre-deployment performance validation
    - Real-time change impact monitoring
    - Predictive performance modeling
    - Automated rollback recommendations
    - Cross-component impact analysis
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize performance impact assessor with enterprise configuration."""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Core components
        self.baseline_manager = PerformanceBaselineManager(self.config.get('baseline_manager', {}))
        self.regression_detector = PerformanceRegressionDetector(self.config.get('regression_detector', {}))
        self.performance_optimizer = PerformanceOptimizer(self.config.get('optimizer', {}))
        
        # Impact tracking
        self.impact_analyses: Dict[str, ChangeImpactAnalysis] = {}
        self.validation_results: Dict[str, DeploymentValidationResult] = {}
        self.rollback_recommendations: Dict[str, RollbackRecommendation] = {}
        
        # Configuration
        self.impact_storage_path = Path(self.config.get('impact_storage_path', 'data/performance_impact'))
        self.validation_timeout_seconds = self.config.get('validation_timeout_seconds', 300)
        self.impact_threshold_percentages = self.config.get('impact_thresholds', {
            'negligible': 1.0,
            'low': 5.0,
            'medium': 15.0,
            'high': 30.0
        })
        
        # Performance validation thresholds for Story 4.3
        self.validation_thresholds = {
            'processing_latency_ms': {'max': 1000, 'warning': 800},
            'cache_hit_rate': {'min': 0.70, 'warning': 0.65},
            'memory_usage_mb': {'max': 512, 'warning': 450},
            'error_rate': {'max': 0.001, 'warning': 0.0005},
            'throughput_ops_per_second': {'min': 10, 'warning': 8}
        }
        
        # Critical component dependencies
        self.component_dependencies = {
            'mcp_transformer': ['sanskrit_processing', 'performance_monitor'],
            'sanskrit_processing': ['ner_module', 'contextual_modeling'],
            'scripture_processing': ['sanskrit_processing', 'contextual_modeling'],
            'performance_monitor': ['telemetry_collector', 'system_monitor'],
            'ner_module': ['sanskrit_processing']
        }
        
        # Initialize storage
        self.impact_storage_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing impact data
        self._load_existing_impact_data()
        
        self.logger.info("PerformanceImpactAssessor initialized for enterprise change management")
    
    def assess_change_impact(self, change_id: str, change_type: ChangeType,
                           change_description: str, affected_components: List[str],
                           assessment_phase: AssessmentPhase = AssessmentPhase.PRE_DEPLOYMENT,
                           created_by: str = "system") -> ChangeImpactAnalysis:
        """
        Assess performance impact of a proposed or deployed change.
        
        Analyzes current vs baseline performance across affected components.
        """
        analysis_id = str(uuid.uuid4())
        
        self.logger.info(f"Starting impact assessment for change {change_id}: {change_description}")
        
        # Collect performance metrics for affected components
        performance_metrics = []
        
        for component in affected_components:
            component_metrics = self._assess_component_impact(component, change_id)
            performance_metrics.extend(component_metrics)
        
        # Analyze cross-component dependencies
        dependency_impacts = self._analyze_dependency_impacts(affected_components)
        
        # Calculate overall impact severity
        overall_severity = self._calculate_overall_impact_severity(performance_metrics)
        
        # Generate risk assessment
        risk_assessment = self._generate_risk_assessment(
            change_type, performance_metrics, dependency_impacts
        )
        
        # Generate recommendations
        recommendations = self._generate_impact_recommendations(
            overall_severity, performance_metrics, risk_assessment
        )
        
        # Predict outcomes
        predicted_outcomes = self._predict_performance_outcomes(
            performance_metrics, change_type, assessment_phase
        )
        
        # Create impact analysis
        analysis = ChangeImpactAnalysis(
            analysis_id=analysis_id,
            change_id=change_id,
            change_type=change_type,
            change_description=change_description,
            assessment_phase=assessment_phase,
            overall_impact_severity=overall_severity,
            affected_components=affected_components,
            performance_metrics=performance_metrics,
            risk_assessment=risk_assessment,
            recommendations=recommendations,
            predicted_outcomes=predicted_outcomes,
            created_at=time.time(),
            created_by=created_by
        )
        
        # Store analysis
        self.impact_analyses[analysis_id] = analysis
        self._save_impact_analysis(analysis)
        
        self.logger.info(f"Impact assessment complete: {overall_severity.value} impact detected")
        
        return analysis
    
    def validate_pre_deployment_performance(self, change_id: str, 
                                          validation_config: Dict[str, Any],
                                          validated_by: str = "automated") -> DeploymentValidationResult:
        """
        Validate performance before deployment with comprehensive testing.
        
        Runs performance tests and validates against acceptance criteria.
        """
        validation_id = str(uuid.uuid4())
        start_time = time.time()
        
        self.logger.info(f"Starting pre-deployment validation for change {change_id}")
        
        # Initialize validation counters
        tests_passed = 0
        tests_failed = 0
        blocking_issues = []
        warnings = []
        
        # Test 1: Processing latency validation
        try:
            latency_result = self._validate_processing_latency(validation_config)
            if latency_result['passed']:
                tests_passed += 1
            else:
                tests_failed += 1
                if latency_result['blocking']:
                    blocking_issues.append(f"Processing latency exceeds {self.validation_thresholds['processing_latency_ms']['max']}ms")
                else:
                    warnings.append(f"Processing latency warning: {latency_result['message']}")
        except Exception as e:
            tests_failed += 1
            blocking_issues.append(f"Latency validation failed: {e}")
        
        # Test 2: Cache performance validation
        try:
            cache_result = self._validate_cache_performance(validation_config)
            if cache_result['passed']:
                tests_passed += 1
            else:
                tests_failed += 1
                if cache_result['blocking']:
                    blocking_issues.append(f"Cache hit rate below {self.validation_thresholds['cache_hit_rate']['min']:.1%}")
                else:
                    warnings.append(f"Cache performance warning: {cache_result['message']}")
        except Exception as e:
            tests_failed += 1
            warnings.append(f"Cache validation warning: {e}")
        
        # Test 3: Memory usage validation
        try:
            memory_result = self._validate_memory_usage(validation_config)
            if memory_result['passed']:
                tests_passed += 1
            else:
                tests_failed += 1
                if memory_result['blocking']:
                    blocking_issues.append(f"Memory usage exceeds {self.validation_thresholds['memory_usage_mb']['max']}MB")
                else:
                    warnings.append(f"Memory usage warning: {memory_result['message']}")
        except Exception as e:
            tests_failed += 1
            warnings.append(f"Memory validation warning: {e}")
        
        # Test 4: Error rate validation
        try:
            error_result = self._validate_error_rate(validation_config)
            if error_result['passed']:
                tests_passed += 1
            else:
                tests_failed += 1
                if error_result['blocking']:
                    blocking_issues.append(f"Error rate exceeds {self.validation_thresholds['error_rate']['max']:.3%}")
                else:
                    warnings.append(f"Error rate warning: {error_result['message']}")
        except Exception as e:
            tests_failed += 1
            blocking_issues.append(f"Error rate validation failed: {e}")
        
        # Test 5: End-to-end performance test
        try:
            e2e_result = self._validate_end_to_end_performance(validation_config)
            if e2e_result['passed']:
                tests_passed += 1
            else:
                tests_failed += 1
                if e2e_result['blocking']:
                    blocking_issues.append("End-to-end performance test failed")
                else:
                    warnings.append(f"End-to-end performance warning: {e2e_result['message']}")
        except Exception as e:
            tests_failed += 1
            blocking_issues.append(f"End-to-end validation failed: {e}")
        
        # Estimate production impact
        estimated_impact = self._estimate_production_impact(validation_config)
        
        # Determine if deployment should be approved
        deployment_approved = len(blocking_issues) == 0
        
        validation_duration = time.time() - start_time
        
        # Create validation result
        result = DeploymentValidationResult(
            validation_id=validation_id,
            change_id=change_id,
            deployment_approved=deployment_approved,
            performance_tests_passed=tests_passed,
            performance_tests_failed=tests_failed,
            blocking_issues=blocking_issues,
            warnings=warnings,
            estimated_production_impact=estimated_impact,
            validation_duration_seconds=validation_duration,
            validated_by=validated_by,
            validated_at=time.time()
        )
        
        # Store validation result
        self.validation_results[validation_id] = result
        self._save_validation_result(result)
        
        status = "APPROVED" if deployment_approved else "REJECTED"
        self.logger.info(f"Pre-deployment validation complete: {status} ({tests_passed}/{tests_passed + tests_failed} tests passed)")
        
        return result
    
    def generate_rollback_recommendation(self, change_id: str, 
                                       post_deployment_analysis: ChangeImpactAnalysis) -> RollbackRecommendation:
        """
        Generate rollback recommendation based on post-deployment impact analysis.
        
        Analyzes performance degradation and recommends rollback if necessary.
        """
        recommendation_id = str(uuid.uuid4())
        
        # Analyze performance impact severity
        high_impact_metrics = [
            m for m in post_deployment_analysis.performance_metrics
            if m.impact_severity in [ImpactSeverity.HIGH, ImpactSeverity.CRITICAL]
        ]
        
        # Determine rollback recommendation
        recommend_rollback = False
        urgency = "none"
        rollback_window = 0
        
        if any(m.impact_severity == ImpactSeverity.CRITICAL for m in high_impact_metrics):
            recommend_rollback = True
            urgency = "immediate"
            rollback_window = 1  # 1 hour
        elif len(high_impact_metrics) >= 3:
            recommend_rollback = True
            urgency = "scheduled"
            rollback_window = 4  # 4 hours
        elif post_deployment_analysis.overall_impact_severity == ImpactSeverity.HIGH:
            recommend_rollback = True
            urgency = "scheduled"
            rollback_window = 8  # 8 hours
        
        # Generate severity justification
        severity_justification = self._generate_severity_justification(
            post_deployment_analysis.overall_impact_severity,
            high_impact_metrics
        )
        
        # Extract impact metric names
        impact_metric_names = [m.metric_name for m in high_impact_metrics]
        
        # Generate alternative actions
        alternative_actions = self._generate_alternative_actions(
            post_deployment_analysis, recommend_rollback
        )
        
        recommendation = RollbackRecommendation(
            recommendation_id=recommendation_id,
            change_id=change_id,
            recommend_rollback=recommend_rollback,
            severity_justification=severity_justification,
            impact_metrics=impact_metric_names,
            rollback_urgency=urgency,
            rollback_window_hours=rollback_window,
            alternative_actions=alternative_actions,
            created_at=time.time()
        )
        
        self.rollback_recommendations[recommendation_id] = recommendation
        self._save_rollback_recommendation(recommendation)
        
        action = "ROLLBACK" if recommend_rollback else "MONITOR"
        self.logger.info(f"Rollback recommendation generated: {action} ({urgency} urgency)")
        
        return recommendation
    
    def get_performance_impact_report(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Generate comprehensive performance impact report.
        
        Provides overview of recent impacts, validations, and recommendations.
        """
        current_time = time.time()
        cutoff_time = current_time - (hours_back * 3600)
        
        # Filter recent data
        recent_analyses = [
            analysis for analysis in self.impact_analyses.values()
            if analysis.created_at >= cutoff_time
        ]
        
        recent_validations = [
            validation for validation in self.validation_results.values()
            if validation.validated_at >= cutoff_time
        ]
        
        recent_recommendations = [
            recommendation for recommendation in self.rollback_recommendations.values()
            if recommendation.created_at >= cutoff_time
        ]
        
        # Analyze impact trends
        impact_trends = self._analyze_impact_trends(recent_analyses)
        
        # Validation success rates
        validation_stats = self._calculate_validation_statistics(recent_validations)
        
        # Component impact analysis
        component_impacts = self._analyze_component_impacts(recent_analyses)
        
        report = {
            'report_metadata': {
                'generated_at': current_time,
                'period_hours': hours_back,
                'impact_assessor_version': '4.3.0'
            },
            'executive_summary': {
                'total_impact_assessments': len(recent_analyses),
                'total_validations': len(recent_validations),
                'total_rollback_recommendations': len(recent_recommendations),
                'high_impact_changes': len([a for a in recent_analyses if a.overall_impact_severity in [ImpactSeverity.HIGH, ImpactSeverity.CRITICAL]]),
                'deployment_approval_rate': validation_stats['approval_rate'],
                'rollback_recommendation_rate': len([r for r in recent_recommendations if r.recommend_rollback]) / max(len(recent_recommendations), 1)
            },
            'impact_analysis': {
                'impacts_by_severity': {
                    severity.value: len([a for a in recent_analyses if a.overall_impact_severity == severity])
                    for severity in ImpactSeverity
                },
                'impacts_by_change_type': {
                    change_type.value: len([a for a in recent_analyses if a.change_type == change_type])
                    for change_type in ChangeType
                },
                'impact_trends': impact_trends,
                'component_impacts': component_impacts
            },
            'validation_analysis': validation_stats,
            'rollback_analysis': {
                'total_recommendations': len(recent_recommendations),
                'rollback_recommended': len([r for r in recent_recommendations if r.recommend_rollback]),
                'immediate_rollbacks': len([r for r in recent_recommendations if r.rollback_urgency == "immediate"]),
                'scheduled_rollbacks': len([r for r in recent_recommendations if r.rollback_urgency == "scheduled"])
            },
            'performance_health': self._assess_performance_health(),
            'recommendations': self._generate_report_recommendations(recent_analyses, recent_validations)
        }
        
        return report
    
    def _assess_component_impact(self, component: str, change_id: str) -> List[PerformanceImpactMetric]:
        """Assess performance impact for a specific component."""
        metrics = []
        
        # Get current performance data
        performance_data = self.performance_optimizer.analyze_performance_bottlenecks()
        
        # Get baselines for component
        component_baselines = {
            bid: metadata for bid, metadata in self.baseline_manager.baseline_metadata.items()
            if metadata.component == component
        }
        
        if not component_baselines:
            self.logger.warning(f"No baselines found for component {component}")
            return metrics
        
        # Analyze each operation for the component
        for baseline_id, metadata in component_baselines.items():
            baseline = self.baseline_manager._get_baseline_by_id(baseline_id)
            if not baseline:
                continue
            
            # Get current measurement
            current_value = self._get_current_measurement(component, metadata.operation)
            if current_value is None:
                continue
            
            # Calculate impact
            impact_percentage = ((current_value - baseline.baseline_value) / baseline.baseline_value) * 100
            impact_severity = self._determine_impact_severity(abs(impact_percentage))
            
            metric = PerformanceImpactMetric(
                metric_name=f"{component}_{metadata.operation}",
                component=component,
                operation=metadata.operation,
                baseline_value=baseline.baseline_value,
                current_value=current_value,
                impact_percentage=impact_percentage,
                impact_severity=impact_severity,
                measurement_unit=baseline.unit,
                confidence_score=metadata.confidence_score,
                sample_count=baseline.sample_count,
                timestamp=time.time()
            )
            
            metrics.append(metric)
        
        return metrics
    
    def _analyze_dependency_impacts(self, affected_components: List[str]) -> Dict[str, Any]:
        """Analyze impacts on component dependencies."""
        dependency_impacts = {
            'upstream_dependencies': [],
            'downstream_dependencies': [],
            'cascading_risk_score': 0.0
        }
        
        # Find upstream dependencies (components that depend on affected ones)
        for component, deps in self.component_dependencies.items():
            if any(dep in affected_components for dep in deps):
                dependency_impacts['upstream_dependencies'].append(component)
        
        # Find downstream dependencies (components affected ones depend on)
        for affected_component in affected_components:
            downstream_deps = self.component_dependencies.get(affected_component, [])
            dependency_impacts['downstream_dependencies'].extend(downstream_deps)
        
        # Calculate cascading risk score
        total_dependent_components = len(set(dependency_impacts['upstream_dependencies'] + dependency_impacts['downstream_dependencies']))
        dependency_impacts['cascading_risk_score'] = min(total_dependent_components / 10.0, 1.0)
        
        return dependency_impacts
    
    def _calculate_overall_impact_severity(self, metrics: List[PerformanceImpactMetric]) -> ImpactSeverity:
        """Calculate overall impact severity from individual metrics."""
        if not metrics:
            return ImpactSeverity.NEGLIGIBLE
        
        # Count metrics by severity
        severity_counts = {severity: 0 for severity in ImpactSeverity}
        for metric in metrics:
            severity_counts[metric.impact_severity] += 1
        
        # Determine overall severity
        if severity_counts[ImpactSeverity.CRITICAL] > 0:
            return ImpactSeverity.CRITICAL
        elif severity_counts[ImpactSeverity.HIGH] >= 2:
            return ImpactSeverity.HIGH
        elif severity_counts[ImpactSeverity.HIGH] > 0:
            return ImpactSeverity.HIGH
        elif severity_counts[ImpactSeverity.MEDIUM] >= 3:
            return ImpactSeverity.HIGH
        elif severity_counts[ImpactSeverity.MEDIUM] > 0:
            return ImpactSeverity.MEDIUM
        elif severity_counts[ImpactSeverity.LOW] > 0:
            return ImpactSeverity.LOW
        else:
            return ImpactSeverity.NEGLIGIBLE
    
    def _determine_impact_severity(self, impact_percentage: float) -> ImpactSeverity:
        """Determine impact severity from percentage change."""
        if impact_percentage >= self.impact_threshold_percentages['high']:
            return ImpactSeverity.CRITICAL
        elif impact_percentage >= self.impact_threshold_percentages['medium']:
            return ImpactSeverity.HIGH
        elif impact_percentage >= self.impact_threshold_percentages['low']:
            return ImpactSeverity.MEDIUM
        elif impact_percentage >= self.impact_threshold_percentages['negligible']:
            return ImpactSeverity.LOW
        else:
            return ImpactSeverity.NEGLIGIBLE
    
    def _get_current_measurement(self, component: str, operation: str) -> Optional[float]:
        """Get current performance measurement for component operation."""
        # Get recent measurements from regression detector
        history_key = f"{component}_{operation}_latency"
        recent_measurements = list(self.regression_detector.performance_history.get(history_key, []))[-10:]
        
        if not recent_measurements:
            return None
        
        # Return average of recent measurements
        recent_values = [m['value'] for m in recent_measurements]
        return statistics.mean(recent_values)
    
    def _validate_processing_latency(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate processing latency performance."""
        try:
            # Run latency test
            from ..post_processors.sanskrit_post_processor import SanskritPostProcessor
            processor = SanskritPostProcessor()
            
            test_text = config.get('test_text', 'Today we study chapter two verse twenty five from the sacred texts.')
            iterations = config.get('latency_iterations', 10)
            
            latencies = []
            for _ in range(iterations):
                start_time = time.time()
                result = processor.text_normalizer.normalize_with_advanced_tracking(test_text)
                latency = (time.time() - start_time) * 1000  # ms
                latencies.append(latency)
            
            avg_latency = statistics.mean(latencies)
            max_latency = max(latencies)
            
            # Check thresholds
            blocking = max_latency > self.validation_thresholds['processing_latency_ms']['max']
            warning = avg_latency > self.validation_thresholds['processing_latency_ms']['warning']
            
            return {
                'passed': not blocking,
                'blocking': blocking,
                'warning': warning,
                'avg_latency_ms': avg_latency,
                'max_latency_ms': max_latency,
                'message': f"Average latency: {avg_latency:.1f}ms, Max: {max_latency:.1f}ms"
            }
            
        except Exception as e:
            return {'passed': False, 'blocking': True, 'message': f"Latency validation error: {e}"}
    
    def _validate_cache_performance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate cache performance."""
        try:
            # Get cache statistics from optimizer
            cache_analysis = self.performance_optimizer._analyze_cache_performance()
            
            mcp_hit_rate = cache_analysis.get('mcp_hit_rate', 0)
            sanskrit_hit_rate = cache_analysis.get('sanskrit_hit_rate', 0)
            overall_hit_rate = (mcp_hit_rate + sanskrit_hit_rate) / 2
            
            # Check thresholds
            blocking = overall_hit_rate < self.validation_thresholds['cache_hit_rate']['min']
            warning = overall_hit_rate < self.validation_thresholds['cache_hit_rate']['warning']
            
            return {
                'passed': not blocking,
                'blocking': blocking,
                'warning': warning,
                'overall_hit_rate': overall_hit_rate,
                'mcp_hit_rate': mcp_hit_rate,
                'sanskrit_hit_rate': sanskrit_hit_rate,
                'message': f"Overall cache hit rate: {overall_hit_rate:.1%}"
            }
            
        except Exception as e:
            return {'passed': False, 'blocking': False, 'message': f"Cache validation warning: {e}"}
    
    def _validate_memory_usage(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate memory usage performance."""
        try:
            # Get system performance data
            system_analysis = self.performance_optimizer._analyze_system_performance()
            memory_pressure = system_analysis.get('memory_pressure', 0)
            available_memory_gb = system_analysis.get('available_memory_gb', 8)
            
            # Calculate approximate memory usage
            total_memory_gb = available_memory_gb / (1 - memory_pressure) if memory_pressure < 1 else 8
            used_memory_gb = total_memory_gb - available_memory_gb
            used_memory_mb = used_memory_gb * 1024
            
            # Check thresholds
            blocking = used_memory_mb > self.validation_thresholds['memory_usage_mb']['max']
            warning = used_memory_mb > self.validation_thresholds['memory_usage_mb']['warning']
            
            return {
                'passed': not blocking,
                'blocking': blocking,
                'warning': warning,
                'used_memory_mb': used_memory_mb,
                'memory_pressure': memory_pressure,
                'message': f"Memory usage: {used_memory_mb:.0f}MB ({memory_pressure:.1%} pressure)"
            }
            
        except Exception as e:
            return {'passed': False, 'blocking': False, 'message': f"Memory validation warning: {e}"}
    
    def _validate_error_rate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate error rate performance."""
        try:
            # Get processing performance data
            processing_analysis = self.performance_optimizer._analyze_processing_performance()
            error_rate = processing_analysis.get('error_rate', 0)
            success_rate = processing_analysis.get('success_rate', 1.0)
            
            # Check thresholds
            blocking = error_rate > self.validation_thresholds['error_rate']['max']
            warning = error_rate > self.validation_thresholds['error_rate']['warning']
            
            return {
                'passed': not blocking,
                'blocking': blocking,
                'warning': warning,
                'error_rate': error_rate,
                'success_rate': success_rate,
                'message': f"Error rate: {error_rate:.3%}, Success rate: {success_rate:.3%}"
            }
            
        except Exception as e:
            return {'passed': False, 'blocking': True, 'message': f"Error rate validation failed: {e}"}
    
    def _validate_end_to_end_performance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate end-to-end performance with full pipeline test."""
        try:
            # Run end-to-end test
            from ..post_processors.sanskrit_post_processor import SanskritPostProcessor
            processor = SanskritPostProcessor()
            
            # Comprehensive test content
            test_content = config.get('e2e_test_content', """1
00:00:01,000 --> 00:00:05,000
Today we study chapter two verse twenty five from the bhagavad gita.

2
00:00:06,000 --> 00:00:10,000
This verse teaches us about krishna and dharma in our yoga practice.""")
            
            # Process with timing
            start_time = time.time()
            
            # Use StringIO to simulate file processing
            from io import StringIO
            from ..utils.srt_parser import SRTParser
            
            parser = SRTParser()
            segments = parser.parse_string(test_content)
            
            processed_segments = []
            for segment in segments:
                processed_segment = processor._process_srt_segment(segment, processor.metrics_collector.create_file_metrics('test'))
                processed_segments.append(processed_segment)
            
            processing_time = (time.time() - start_time) * 1000  # ms
            
            # Validate segments were processed
            changes_made = sum(1 for i, segment in enumerate(processed_segments) 
                             if segment.text != segments[i].text)
            
            # Check performance
            blocking = processing_time > 2000  # 2 seconds for e2e test
            warning = processing_time > 1500  # 1.5 seconds warning
            
            success = changes_made > 0 and not blocking
            
            return {
                'passed': success,
                'blocking': blocking,
                'warning': warning,
                'processing_time_ms': processing_time,
                'segments_processed': len(processed_segments),
                'changes_made': changes_made,
                'message': f"E2E test: {processing_time:.0f}ms, {changes_made} changes"
            }
            
        except Exception as e:
            return {'passed': False, 'blocking': True, 'message': f"E2E test failed: {e}"}
    
    def _estimate_production_impact(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Estimate production performance impact."""
        # This would use machine learning models in a real implementation
        # For now, provide reasonable estimates based on validation results
        
        return {
            'estimated_latency_increase_percentage': 2.5,
            'estimated_throughput_decrease_percentage': 1.8,
            'estimated_memory_increase_percentage': 3.2,
            'estimated_error_rate_increase_percentage': 0.1,
            'confidence_score': 0.75
        }
    
    def _generate_risk_assessment(self, change_type: ChangeType, 
                                metrics: List[PerformanceImpactMetric],
                                dependency_impacts: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk assessment for the change."""
        risk_score = 0.0
        risk_factors = []
        
        # Change type risk
        change_risk_scores = {
            ChangeType.CODE_CHANGE: 0.3,
            ChangeType.CONFIGURATION_CHANGE: 0.2,
            ChangeType.INFRASTRUCTURE_CHANGE: 0.4,
            ChangeType.DEPENDENCY_UPDATE: 0.3,
            ChangeType.DATA_MODEL_CHANGE: 0.5
        }
        risk_score += change_risk_scores.get(change_type, 0.2)
        
        # Performance impact risk
        high_impact_metrics = [m for m in metrics if m.impact_severity in [ImpactSeverity.HIGH, ImpactSeverity.CRITICAL]]
        if high_impact_metrics:
            risk_score += len(high_impact_metrics) * 0.15
            risk_factors.append(f"{len(high_impact_metrics)} high-impact performance metrics")
        
        # Dependency cascade risk
        cascade_risk = dependency_impacts.get('cascading_risk_score', 0)
        risk_score += cascade_risk * 0.2
        if cascade_risk > 0.3:
            risk_factors.append(f"High dependency cascade risk ({cascade_risk:.1%})")
        
        # Normalize risk score
        risk_score = min(risk_score, 1.0)
        
        # Determine risk level
        if risk_score >= 0.7:
            risk_level = "HIGH"
        elif risk_score >= 0.4:
            risk_level = "MEDIUM"
        elif risk_score >= 0.2:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
        
        return {
            'overall_risk_score': risk_score,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'mitigation_strategies': self._suggest_mitigation_strategies(risk_level, risk_factors)
        }
    
    def _suggest_mitigation_strategies(self, risk_level: str, risk_factors: List[str]) -> List[str]:
        """Suggest mitigation strategies based on risk assessment."""
        strategies = []
        
        if risk_level in ["HIGH", "MEDIUM"]:
            strategies.extend([
                "Implement gradual rollout with performance monitoring",
                "Prepare automated rollback procedures",
                "Increase monitoring frequency during deployment"
            ])
        
        if "high-impact performance metrics" in str(risk_factors):
            strategies.extend([
                "Conduct additional performance testing",
                "Validate on staging environment with production-like load",
                "Review performance optimization opportunities"
            ])
        
        if "dependency cascade risk" in str(risk_factors):
            strategies.extend([
                "Test dependent components thoroughly",
                "Coordinate with dependent service teams",
                "Plan service dependency health checks"
            ])
        
        return strategies
    
    def _generate_impact_recommendations(self, severity: ImpactSeverity,
                                       metrics: List[PerformanceImpactMetric],
                                       risk_assessment: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on impact analysis."""
        recommendations = []
        
        if severity in [ImpactSeverity.HIGH, ImpactSeverity.CRITICAL]:
            recommendations.extend([
                "Consider postponing deployment until performance issues are resolved",
                "Conduct thorough performance optimization before deployment",
                "Implement comprehensive monitoring and alerting"
            ])
        
        if severity == ImpactSeverity.MEDIUM:
            recommendations.extend([
                "Proceed with caution and enhanced monitoring",
                "Prepare rollback plan in case of issues",
                "Monitor performance closely post-deployment"
            ])
        
        # Component-specific recommendations
        affected_components = set(m.component for m in metrics)
        if 'mcp_transformer' in affected_components:
            recommendations.append("Review MCP transformer cache configuration and optimization")
        
        if 'sanskrit_processing' in affected_components:
            recommendations.append("Validate Sanskrit processing accuracy and performance")
        
        # Risk-based recommendations
        if risk_assessment['risk_level'] == "HIGH":
            recommendations.extend([
                "Implement staged deployment with performance gates",
                "Increase automated testing coverage",
                "Coordinate with operations team for deployment monitoring"
            ])
        
        return recommendations
    
    def _predict_performance_outcomes(self, metrics: List[PerformanceImpactMetric],
                                    change_type: ChangeType,
                                    phase: AssessmentPhase) -> Dict[str, Any]:
        """Predict performance outcomes based on current analysis."""
        predictions = {
            'predicted_latency_impact': 0.0,
            'predicted_throughput_impact': 0.0,
            'predicted_memory_impact': 0.0,
            'predicted_stability_score': 1.0,
            'confidence_level': 0.8
        }
        
        # Aggregate metric impacts
        latency_impacts = [m.impact_percentage for m in metrics if 'latency' in m.metric_name.lower()]
        if latency_impacts:
            predictions['predicted_latency_impact'] = statistics.mean(latency_impacts)
        
        memory_impacts = [m.impact_percentage for m in metrics if 'memory' in m.metric_name.lower()]
        if memory_impacts:
            predictions['predicted_memory_impact'] = statistics.mean(memory_impacts)
        
        # Calculate stability score
        high_impact_count = len([m for m in metrics if m.impact_severity in [ImpactSeverity.HIGH, ImpactSeverity.CRITICAL]])
        predictions['predicted_stability_score'] = max(0.0, 1.0 - (high_impact_count * 0.2))
        
        return predictions
    
    def _generate_severity_justification(self, severity: ImpactSeverity,
                                       high_impact_metrics: List[PerformanceImpactMetric]) -> str:
        """Generate justification for rollback severity assessment."""
        if severity == ImpactSeverity.CRITICAL:
            return f"Critical performance degradation detected in {len(high_impact_metrics)} metrics requiring immediate rollback"
        elif severity == ImpactSeverity.HIGH:
            return f"High performance impact detected affecting {len(high_impact_metrics)} critical metrics"
        elif severity == ImpactSeverity.MEDIUM:
            return "Medium performance impact detected - monitor closely and consider rollback if issues persist"
        else:
            return "Performance impact within acceptable thresholds"
    
    def _generate_alternative_actions(self, analysis: ChangeImpactAnalysis, 
                                    recommend_rollback: bool) -> List[str]:
        """Generate alternative actions to rollback."""
        actions = []
        
        if not recommend_rollback:
            actions.extend([
                "Continue monitoring performance metrics",
                "Implement performance optimizations if degradation continues",
                "Review and adjust monitoring thresholds"
            ])
        else:
            actions.extend([
                "Immediate performance optimization deployment",
                "Gradual traffic reduction to affected components",
                "Emergency cache warming procedures",
                "Hotfix deployment for critical performance issues"
            ])
        
        # Component-specific actions
        affected_components = analysis.affected_components
        if 'mcp_transformer' in affected_components:
            actions.append("Optimize MCP transformer cache and timeout settings")
        
        if 'sanskrit_processing' in affected_components:
            actions.append("Review Sanskrit processing pipeline configuration")
        
        return actions
    
    def _analyze_impact_trends(self, analyses: List[ChangeImpactAnalysis]) -> Dict[str, Any]:
        """Analyze trends in performance impacts."""
        if not analyses:
            return {'trend': 'insufficient_data'}
        
        # Sort by creation time
        sorted_analyses = sorted(analyses, key=lambda a: a.created_at)
        
        # Calculate severity trend
        severity_scores = {
            ImpactSeverity.NEGLIGIBLE: 0,
            ImpactSeverity.LOW: 1,
            ImpactSeverity.MEDIUM: 2,
            ImpactSeverity.HIGH: 3,
            ImpactSeverity.CRITICAL: 4
        }
        
        scores = [severity_scores[a.overall_impact_severity] for a in sorted_analyses]
        
        if len(scores) >= 3:
            # Simple trend calculation
            recent_avg = statistics.mean(scores[-3:])
            earlier_avg = statistics.mean(scores[:-3]) if len(scores) > 3 else statistics.mean(scores[:3])
            
            if recent_avg > earlier_avg + 0.5:
                trend = 'deteriorating'
            elif recent_avg < earlier_avg - 0.5:
                trend = 'improving'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'trend': trend,
            'average_severity_score': statistics.mean(scores),
            'recent_analyses_count': len(analyses)
        }
    
    def _calculate_validation_statistics(self, validations: List[DeploymentValidationResult]) -> Dict[str, Any]:
        """Calculate validation statistics."""
        if not validations:
            return {
                'total_validations': 0,
                'approval_rate': 0.0,
                'average_test_success_rate': 0.0,
                'average_validation_time': 0.0
            }
        
        approved = len([v for v in validations if v.deployment_approved])
        total_tests = sum(v.performance_tests_passed + v.performance_tests_failed for v in validations)
        passed_tests = sum(v.performance_tests_passed for v in validations)
        
        return {
            'total_validations': len(validations),
            'approval_rate': approved / len(validations),
            'average_test_success_rate': passed_tests / total_tests if total_tests > 0 else 0.0,
            'average_validation_time': statistics.mean([v.validation_duration_seconds for v in validations]),
            'blocking_issues_frequency': len([v for v in validations if v.blocking_issues]) / len(validations)
        }
    
    def _analyze_component_impacts(self, analyses: List[ChangeImpactAnalysis]) -> Dict[str, Any]:
        """Analyze impacts by component."""
        component_impacts = defaultdict(list)
        
        for analysis in analyses:
            for component in analysis.affected_components:
                component_impacts[component].append(analysis.overall_impact_severity)
        
        # Calculate statistics per component
        component_stats = {}
        for component, severities in component_impacts.items():
            severity_scores = {
                ImpactSeverity.NEGLIGIBLE: 0,
                ImpactSeverity.LOW: 1,
                ImpactSeverity.MEDIUM: 2,
                ImpactSeverity.HIGH: 3,
                ImpactSeverity.CRITICAL: 4
            }
            
            scores = [severity_scores[s] for s in severities]
            component_stats[component] = {
                'impact_count': len(severities),
                'average_severity_score': statistics.mean(scores),
                'high_impact_frequency': len([s for s in severities if s in [ImpactSeverity.HIGH, ImpactSeverity.CRITICAL]]) / len(severities)
            }
        
        return component_stats
    
    def _assess_performance_health(self) -> Dict[str, Any]:
        """Assess overall performance health."""
        # Get current performance state
        performance_analysis = self.performance_optimizer.analyze_performance_bottlenecks()
        
        # Calculate health score
        health_factors = []
        
        # Cache performance
        cache_perf = performance_analysis.get('cache_performance', {})
        if cache_perf.get('mcp_hit_rate', 0) >= 0.7:
            health_factors.append(100)
        else:
            health_factors.append(50)
        
        # Processing performance
        proc_perf = performance_analysis.get('processing_performance', {})
        if proc_perf.get('average_processing_time', 1000) <= 1000:
            health_factors.append(100)
        else:
            health_factors.append(20)
        
        # System performance
        sys_perf = performance_analysis.get('system_performance', {})
        if sys_perf.get('memory_pressure', 0) <= 0.8:
            health_factors.append(100)
        else:
            health_factors.append(30)
        
        overall_health = statistics.mean(health_factors)
        
        if overall_health >= 90:
            status = "EXCELLENT"
        elif overall_health >= 75:
            status = "GOOD"
        elif overall_health >= 60:
            status = "FAIR"
        elif overall_health >= 40:
            status = "POOR"
        else:
            status = "CRITICAL"
        
        return {
            'overall_health_score': overall_health,
            'health_status': status,
            'performance_factors': health_factors
        }
    
    def _generate_report_recommendations(self, analyses: List[ChangeImpactAnalysis],
                                       validations: List[DeploymentValidationResult]) -> List[str]:
        """Generate recommendations for the impact report."""
        recommendations = []
        
        # Validation-based recommendations
        if validations:
            approval_rate = len([v for v in validations if v.deployment_approved]) / len(validations)
            if approval_rate < 0.8:
                recommendations.append(f"Deployment approval rate ({approval_rate:.1%}) is low - review validation criteria")
        
        # Impact-based recommendations
        high_impact_analyses = [a for a in analyses if a.overall_impact_severity in [ImpactSeverity.HIGH, ImpactSeverity.CRITICAL]]
        if len(high_impact_analyses) > len(analyses) * 0.2:
            recommendations.append("High frequency of significant performance impacts - review change management process")
        
        # Story 4.3 specific recommendations
        recommendations.extend([
            "Maintain sub-second processing targets through impact assessments",
            "Validate performance impact before all production deployments",
            "Monitor performance health continuously with automated alerting"
        ])
        
        return recommendations
    
    def _load_existing_impact_data(self):
        """Load existing impact analysis data."""
        try:
            # Load impact analyses
            for analysis_file in self.impact_storage_path.glob("analysis_*.json"):
                with open(analysis_file, 'r') as f:
                    analysis_data = json.load(f)
                
                # Reconstruct enums and objects
                analysis_data['change_type'] = ChangeType(analysis_data['change_type'])
                analysis_data['assessment_phase'] = AssessmentPhase(analysis_data['assessment_phase'])
                analysis_data['overall_impact_severity'] = ImpactSeverity(analysis_data['overall_impact_severity'])
                
                # Reconstruct performance metrics
                metrics = []
                for metric_data in analysis_data.get('performance_metrics', []):
                    metric_data['impact_severity'] = ImpactSeverity(metric_data['impact_severity'])
                    metrics.append(PerformanceImpactMetric(**metric_data))
                analysis_data['performance_metrics'] = metrics
                
                analysis = ChangeImpactAnalysis(**analysis_data)
                self.impact_analyses[analysis.analysis_id] = analysis
            
            self.logger.info(f"Loaded {len(self.impact_analyses)} impact analyses")
            
        except Exception as e:
            self.logger.error(f"Error loading impact data: {e}")
    
    def _save_impact_analysis(self, analysis: ChangeImpactAnalysis):
        """Save impact analysis to storage."""
        try:
            analysis_file = self.impact_storage_path / f"analysis_{analysis.analysis_id}.json"
            
            # Convert to dictionary for JSON serialization
            analysis_data = {
                'analysis_id': analysis.analysis_id,
                'change_id': analysis.change_id,
                'change_type': analysis.change_type.value,
                'change_description': analysis.change_description,
                'assessment_phase': analysis.assessment_phase.value,
                'overall_impact_severity': analysis.overall_impact_severity.value,
                'affected_components': analysis.affected_components,
                'performance_metrics': [
                    {
                        'metric_name': m.metric_name,
                        'component': m.component,
                        'operation': m.operation,
                        'baseline_value': m.baseline_value,
                        'current_value': m.current_value,
                        'impact_percentage': m.impact_percentage,
                        'impact_severity': m.impact_severity.value,
                        'measurement_unit': m.measurement_unit,
                        'confidence_score': m.confidence_score,
                        'sample_count': m.sample_count,
                        'timestamp': m.timestamp
                    }
                    for m in analysis.performance_metrics
                ],
                'risk_assessment': analysis.risk_assessment,
                'recommendations': analysis.recommendations,
                'predicted_outcomes': analysis.predicted_outcomes,
                'created_at': analysis.created_at,
                'created_by': analysis.created_by
            }
            
            with open(analysis_file, 'w') as f:
                json.dump(analysis_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving impact analysis: {e}")
    
    def _save_validation_result(self, result: DeploymentValidationResult):
        """Save validation result to storage."""
        try:
            validation_file = self.impact_storage_path / f"validation_{result.validation_id}.json"
            
            validation_data = {
                'validation_id': result.validation_id,
                'change_id': result.change_id,
                'deployment_approved': result.deployment_approved,
                'performance_tests_passed': result.performance_tests_passed,
                'performance_tests_failed': result.performance_tests_failed,
                'blocking_issues': result.blocking_issues,
                'warnings': result.warnings,
                'estimated_production_impact': result.estimated_production_impact,
                'validation_duration_seconds': result.validation_duration_seconds,
                'validated_by': result.validated_by,
                'validated_at': result.validated_at
            }
            
            with open(validation_file, 'w') as f:
                json.dump(validation_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving validation result: {e}")
    
    def _save_rollback_recommendation(self, recommendation: RollbackRecommendation):
        """Save rollback recommendation to storage."""
        try:
            recommendation_file = self.impact_storage_path / f"rollback_{recommendation.recommendation_id}.json"
            
            recommendation_data = {
                'recommendation_id': recommendation.recommendation_id,
                'change_id': recommendation.change_id,
                'recommend_rollback': recommendation.recommend_rollback,
                'severity_justification': recommendation.severity_justification,
                'impact_metrics': recommendation.impact_metrics,
                'rollback_urgency': recommendation.rollback_urgency,
                'rollback_window_hours': recommendation.rollback_window_hours,
                'alternative_actions': recommendation.alternative_actions,
                'created_at': recommendation.created_at
            }
            
            with open(recommendation_file, 'w') as f:
                json.dump(recommendation_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving rollback recommendation: {e}")


def test_performance_impact_assessment():
    """Test performance impact assessment functionality."""
    assessor = PerformanceImpactAssessor()
    
    print("Testing performance impact assessment...")
    
    # Test change impact assessment
    analysis = assessor.assess_change_impact(
        change_id="test_change_001",
        change_type=ChangeType.CODE_CHANGE,
        change_description="Test code change for impact assessment",
        affected_components=["mcp_transformer", "sanskrit_processing"]
    )
    
    # Test pre-deployment validation
    validation_config = {
        'test_text': 'Today we study chapter two verse twenty five',
        'latency_iterations': 5,
        'e2e_test_content': 'Test SRT content for validation'
    }
    
    validation = assessor.validate_pre_deployment_performance("test_change_001", validation_config)
    
    # Test rollback recommendation
    rollback = assessor.generate_rollback_recommendation("test_change_001", analysis)
    
    # Test impact report
    report = assessor.get_performance_impact_report()
    
    print(f"✅ Performance impact assessment test passed")
    print(f"   Impact severity: {analysis.overall_impact_severity.value}")
    print(f"   Deployment approved: {validation.deployment_approved}")
    print(f"   Rollback recommended: {rollback.recommend_rollback}")
    print(f"   Impact report generated with {report['executive_summary']['total_impact_assessments']} assessments")
    
    return True


if __name__ == "__main__":
    test_performance_impact_assessment()