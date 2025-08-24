"""
Professional Standards Validator for CEO Directive Compliance.

This module implements automated technical integrity enforcement per CEO mandate to
"ensure professional and honest work by the bmad team."

Extracted from advanced_text_normalizer.py to break circular import with mcp_client.py
"""

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional
from .performance_metrics import performance_context

logger = logging.getLogger(__name__)


# REMOVED: ProfessionalStandardsValidator - Replaced by TechnicalQualityGate
# This deprecated class has been removed per QA architecture review.
# All functionality migrated to TechnicalQualityGate for concrete quality enforcement.


class QualityGate(ABC):
    """Abstract base class for concrete quality gates"""
    
    @abstractmethod
    def evaluate(self, metrics: Dict[str, Any]) -> 'QualityGateResult':
        """Evaluate metrics against gate criteria"""
        pass


@dataclass
class QualityGateResult:
    """Result of quality gate evaluation"""
    gate_name: str
    passes: bool
    score: float
    threshold: float
    violations: List[str]
    metrics: Dict[str, Any]
    
    @property
    def severity(self) -> str:
        """Get severity level based on how far from threshold"""
        if self.passes:
            return "PASS"
        
        diff_pct = (self.threshold - self.score) / self.threshold * 100
        if diff_pct > 50:
            return "CRITICAL"
        elif diff_pct > 25:
            return "HIGH"
        elif diff_pct > 10:
            return "MEDIUM"
        else:
            return "LOW"


class CoverageGate(QualityGate):
    """Test coverage quality gate"""
    
    def __init__(self, min_threshold: float = 0.85):
        self.min_threshold = min_threshold
        
    def evaluate(self, metrics: Dict[str, Any]) -> QualityGateResult:
        # Fix: Handle case where metrics might be int or None
        if not isinstance(metrics, dict):
            metrics = {}
            
        coverage = metrics.get('test_coverage', 0.0)
        violations = []
        
        if coverage < self.min_threshold:
            violations.append(f"Test coverage {coverage:.1%} below minimum {self.min_threshold:.1%}")
            
        return QualityGateResult(
            gate_name="Test Coverage",
            passes=len(violations) == 0,
            score=coverage,
            threshold=self.min_threshold,
            violations=violations,
            metrics={'coverage': coverage}
        )


class ComplexityGate(QualityGate):
    """Code complexity quality gate"""
    
    def __init__(self, max_cyclomatic: int = 10):
        self.max_cyclomatic = max_cyclomatic
        
    def evaluate(self, metrics: Dict[str, Any]) -> QualityGateResult:
        # Fix: Handle case where metrics might be int or None
        if not isinstance(metrics, dict):
            metrics = {}
            
        complexity_data = metrics.get('cyclomatic_complexity', 0)
        violations = []
        
        # Handle both simple number and nested dict formats
        if isinstance(complexity_data, dict):
            avg_complexity = complexity_data.get('average', 0)
            max_complexity = complexity_data.get('max', 0)
            complexity_violations = complexity_data.get('violations', [])
            
            # Use max complexity for evaluation (worst case scenario)
            complexity = max_complexity
            violations.extend(complexity_violations[:3])  # First 3 violations
        else:
            complexity = complexity_data
        
        if complexity > self.max_cyclomatic:
            violations.append(f"Cyclomatic complexity {complexity} exceeds maximum {self.max_cyclomatic}")
            
        # Convert to 0-1 scale for scoring (inverse - lower complexity is better)
        score = 1.0 - min(complexity / (self.max_cyclomatic * 2), 1.0)
            
        return QualityGateResult(
            gate_name="Code Complexity",
            passes=len(violations) == 0,
            score=score,
            threshold=1.0 - (self.max_cyclomatic / (self.max_cyclomatic * 2)),
            violations=violations,
            metrics={'cyclomatic_complexity': complexity}
        )


class DuplicationGate(QualityGate):
    """Code duplication quality gate"""
    
    def __init__(self, max_percentage: float = 0.05):
        self.max_percentage = max_percentage
        
    def evaluate(self, metrics: Dict[str, Any]) -> QualityGateResult:
        # Fix: Handle case where metrics might be int or None
        if not isinstance(metrics, dict):
            metrics = {}
            
        duplication = metrics.get('code_duplication', 0.0)
        violations = []
        
        if duplication > self.max_percentage:
            violations.append(f"Code duplication {duplication:.1%} exceeds maximum {self.max_percentage:.1%}")
            
        # Convert to score (1.0 - duplication percentage)
        score = 1.0 - duplication
            
        return QualityGateResult(
            gate_name="Code Duplication",
            passes=len(violations) == 0,
            score=score,
            threshold=1.0 - self.max_percentage,
            violations=violations,
            metrics={'duplication_percentage': duplication}
        )


class SecurityGate(QualityGate):
    """Security vulnerability quality gate"""
    
    def __init__(self, vulnerability_scanner: bool = True, allow_low_severity: bool = True):
        self.vulnerability_scanner = vulnerability_scanner
        self.allow_low_severity = allow_low_severity
        
    def evaluate(self, metrics: Dict[str, Any]) -> QualityGateResult:
        # Fix: Handle case where metrics might be int or None
        if not isinstance(metrics, dict):
            # Create empty metrics dict if input is invalid
            vulnerabilities = {}
        else:
            vulnerabilities = metrics.get('security_vulnerabilities', {})
        
        # Ensure vulnerabilities is a dict
        if not isinstance(vulnerabilities, dict):
            vulnerabilities = {}
            
        violations = []
        
        critical_count = vulnerabilities.get('critical', 0)
        high_count = vulnerabilities.get('high', 0)
        medium_count = vulnerabilities.get('medium', 0)
        low_count = vulnerabilities.get('low', 0)
        
        if critical_count > 0:
            violations.append(f"{critical_count} critical security vulnerabilities found")
        if high_count > 0:
            violations.append(f"{high_count} high severity security vulnerabilities found")
        if medium_count > 0:
            violations.append(f"{medium_count} medium severity security vulnerabilities found")
        if not self.allow_low_severity and low_count > 0:
            violations.append(f"{low_count} low severity security vulnerabilities found")
            
        total_serious = critical_count + high_count + medium_count
        score = 1.0 if total_serious == 0 else 0.0
            
        return QualityGateResult(
            gate_name="Security Scan",
            passes=len(violations) == 0,
            score=score,
            threshold=1.0,
            violations=violations,
            metrics=vulnerabilities
        )


class PerformanceGate(QualityGate):
    """Performance quality gate"""
    
    def __init__(self, max_response_time_ms: int = 100, max_memory_mb: int = 2048):
        self.max_response_time_ms = max_response_time_ms
        self.max_memory_mb = max_memory_mb
        
    def evaluate(self, metrics: Dict[str, Any]) -> QualityGateResult:
        # Fix: Handle case where metrics might be int or None
        if not isinstance(metrics, dict):
            metrics = {}
            
        response_time = metrics.get('avg_response_time_ms', 0)
        memory_usage = metrics.get('peak_memory_mb', 0)
        violations = []
        
        if response_time > self.max_response_time_ms:
            violations.append(f"Average response time {response_time}ms exceeds {self.max_response_time_ms}ms")
            
        if memory_usage > self.max_memory_mb:
            violations.append(f"Peak memory usage {memory_usage}MB exceeds {self.max_memory_mb}MB")
            
        # Combined score based on both metrics
        time_score = 1.0 - min(response_time / (self.max_response_time_ms * 2), 1.0)
        memory_score = 1.0 - min(memory_usage / (self.max_memory_mb * 2), 1.0)
        score = (time_score + memory_score) / 2
            
        return QualityGateResult(
            gate_name="Performance",
            passes=len(violations) == 0,
            score=score,
            threshold=0.5,  # Both metrics should be under limit
            violations=violations,
            metrics={'response_time_ms': response_time, 'memory_mb': memory_usage}
        )


@dataclass
class QualityReport:
    """Overall quality assessment report"""
    passes: bool
    overall_score: float
    violations: List[QualityGateResult]
    gate_results: List[QualityGateResult]
    timestamp: float
    metrics_summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for serialization"""
        return {
            'passes': self.passes,
            'overall_score': self.overall_score,
            'total_gates': len(self.gate_results),
            'passed_gates': len([g for g in self.gate_results if g.passes]),
            'failed_gates': len(self.violations),
            'timestamp': self.timestamp,
            'violations': [
                {
                    'gate': v.gate_name,
                    'severity': v.severity,
                    'score': v.score,
                    'threshold': v.threshold,
                    'violations': v.violations
                }
                for v in self.violations
            ],
            'gate_scores': {
                g.gate_name: g.score for g in self.gate_results
            },
            'metrics_summary': self.metrics_summary
        }


class TechnicalQualityGate:
    """
    Concrete implementation of quality gates with measurable thresholds
    
    Replaces the hollow ProfessionalStandardsValidator with actual
    technical quality enforcement based on QA Architect recommendations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = self._setup_logging()
        self._setup_default_gates()
        
    def _setup_logging(self) -> Any:
        """Setup structured logging for quality gate operations"""
        try:
            import structlog
            return structlog.get_logger(__name__)
        except ImportError:
            import logging
            return logging.getLogger(__name__)
    
    def _setup_default_gates(self):
        """Initialize default quality gates"""
        self.gates = [
            CoverageGate(min_threshold=self.config.get('min_coverage', 0.85)),
            ComplexityGate(max_cyclomatic=self.config.get('max_complexity', 10)),
            DuplicationGate(max_percentage=self.config.get('max_duplication', 0.05)),
            SecurityGate(vulnerability_scanner=self.config.get('enable_security_scan', True)),
            PerformanceGate(
                max_response_time_ms=self.config.get('max_response_time_ms', 100),
                max_memory_mb=self.config.get('max_memory_mb', 2048)
            )
        ]
    
    def validate_code_quality(self, metrics: Dict[str, Any]) -> QualityReport:
        """
        Validate code quality against all configured gates
        
        Args:
            metrics: Dictionary containing quality metrics
            
        Returns:
            QualityReport with detailed results
        """
        # Use the new performance monitoring system
        with performance_context("TechnicalQualityGate", "validate_code_quality"):
            start_time = time.time()
            gate_results = []
            violations = []
            
            self.logger.info("Starting quality gate validation", 
                            extra={'gate_count': len(self.gates)})
            
            for gate in self.gates:
                with performance_context("TechnicalQualityGate", f"evaluate_{gate.__class__.__name__}"):
                    try:
                        result = gate.evaluate(metrics)
                        gate_results.append(result)
                        
                        if not result.passes:
                            violations.append(result)
                            
                        self.logger.info(f"Quality gate evaluation completed",
                                       extra={
                                           'gate': result.gate_name,
                                           'passes': result.passes,
                                           'score': result.score,
                                           'severity': result.severity
                                       })
                                       
                    except Exception as e:
                        self.logger.error(f"Quality gate evaluation failed",
                                        extra={'gate': gate.__class__.__name__, 'error': str(e)})
                        # Create failure result
                        failure_result = QualityGateResult(
                            gate_name=gate.__class__.__name__,
                            passes=False,
                            score=0.0,
                            threshold=1.0,
                            violations=[f"Gate evaluation failed: {str(e)}"],
                            metrics={}
                        )
                        gate_results.append(failure_result)
                        violations.append(failure_result)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(gate_results)
            passes = len(violations) == 0
            
            # Create comprehensive metrics summary
            metrics_summary = self._create_metrics_summary(gate_results, metrics)
            
            report = QualityReport(
                passes=passes,
                overall_score=overall_score,
                violations=violations,
                gate_results=gate_results,
                timestamp=time.time(),
                metrics_summary=metrics_summary
            )
            
            processing_time = time.time() - start_time
            self.logger.info("Quality gate validation completed",
                            extra={
                                'passes': passes,
                                'overall_score': overall_score,
                                'violation_count': len(violations),
                                'processing_time_ms': processing_time * 1000
                            })
            
            return report
    
    def _calculate_overall_score(self, gate_results: List[QualityGateResult]) -> float:
        """Calculate weighted overall quality score"""
        if not gate_results:
            return 0.0
            
        # Weight gates by importance (can be configured)
        weights = {
            'Security Scan': 0.3,      # Security is most critical
            'Test Coverage': 0.25,     # Testing is crucial
            'Performance': 0.2,        # Performance matters for production
            'Code Complexity': 0.15,   # Maintainability
            'Code Duplication': 0.1    # Code quality
        }
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for result in gate_results:
            weight = weights.get(result.gate_name, 0.1)  # Default weight
            total_weighted_score += result.score * weight
            total_weight += weight
            
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _create_metrics_summary(self, gate_results: List[QualityGateResult], 
                              original_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive metrics summary"""
        return {
            'input_metrics': original_metrics,
            'gate_summary': {
                'total_gates': len(gate_results),
                'passed_gates': len([g for g in gate_results if g.passes]),
                'failed_gates': len([g for g in gate_results if not g.passes]),
                'critical_violations': len([g for g in gate_results if not g.passes and g.severity == 'CRITICAL']),
                'high_violations': len([g for g in gate_results if not g.passes and g.severity == 'HIGH'])
            },
            'performance_requirements_met': all(
                g.passes for g in gate_results if g.gate_name == 'Performance'
            ),
            'security_clean': all(
                g.passes for g in gate_results if g.gate_name == 'Security Scan'
            ),
            'production_ready': len([g for g in gate_results if not g.passes]) == 0
        }
    
    def generate_quality_report(self, metrics: Dict[str, Any], 
                              format_type: str = 'detailed') -> str:
        """
        Generate human-readable quality report
        
        Args:
            metrics: Quality metrics to evaluate
            format_type: 'summary' or 'detailed'
            
        Returns:
            Formatted quality report string
        """
        report = self.validate_code_quality(metrics)
        
        if format_type == 'summary':
            return self._generate_summary_report(report)
        else:
            return self._generate_detailed_report(report)
    
    def _generate_summary_report(self, report: QualityReport) -> str:
        """Generate concise summary report"""
        status = "✅ PASS" if report.passes else "❌ FAIL"
        
        lines = [
            f"TECHNICAL QUALITY GATE REPORT - {status}",
            f"Overall Score: {report.overall_score:.1%}",
            f"Gates Passed: {len(report.gate_results) - len(report.violations)}/{len(report.gate_results)}",
        ]
        
        if report.violations:
            lines.append(f"\nCritical Issues ({len(report.violations)}):")
            for violation in report.violations:
                severity_icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "⚪"}.get(violation.severity, "⚪")
                lines.append(f"{severity_icon} {violation.gate_name}: {violation.violations[0] if violation.violations else 'Failed'}")
        
        return "\n".join(lines)
    
    def _generate_detailed_report(self, report: QualityReport) -> str:
        """Generate detailed quality report"""
        status = "✅ PASS" if report.passes else "❌ FAIL"
        
        lines = [
            "=" * 80,
            f"TECHNICAL QUALITY GATE REPORT - {status}",
            "=" * 80,
            f"Timestamp: {datetime.fromtimestamp(report.timestamp).isoformat()}",
            f"Overall Score: {report.overall_score:.1%}",
            f"Gates Evaluated: {len(report.gate_results)}",
            f"Gates Passed: {len(report.gate_results) - len(report.violations)}",
            f"Gates Failed: {len(report.violations)}",
            ""
        ]
        
        # Gate results summary
        lines.append("GATE RESULTS:")
        lines.append("-" * 40)
        for gate_result in report.gate_results:
            status_icon = "✅" if gate_result.passes else "❌"
            lines.append(f"{status_icon} {gate_result.gate_name}: {gate_result.score:.1%} "
                        f"(threshold: {gate_result.threshold:.1%})")
            
            if gate_result.violations:
                for violation in gate_result.violations:
                    lines.append(f"    ⚠️  {violation}")
        
        lines.append("")
        
        # Metrics summary
        if report.metrics_summary:
            lines.append("METRICS SUMMARY:")
            lines.append("-" * 40)
            summary = report.metrics_summary.get('gate_summary', {})
            lines.append(f"Production Ready: {'Yes' if report.metrics_summary.get('production_ready') else 'No'}")
            lines.append(f"Security Clean: {'Yes' if report.metrics_summary.get('security_clean') else 'No'}")
            lines.append(f"Performance Requirements Met: {'Yes' if report.metrics_summary.get('performance_requirements_met') else 'No'}")
            lines.append(f"Critical Violations: {summary.get('critical_violations', 0)}")
            lines.append(f"High Priority Violations: {summary.get('high_violations', 0)}")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def add_custom_gate(self, gate: QualityGate):
        """Add custom quality gate"""
        self.gates.append(gate)
        self.logger.info(f"Added custom quality gate: {gate.__class__.__name__}")
    
    def remove_gate(self, gate_name: str):
        """Remove quality gate by name"""
        original_count = len(self.gates)
        self.gates = [g for g in self.gates if g.__class__.__name__ != gate_name]
        removed_count = original_count - len(self.gates)
        
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} quality gate(s): {gate_name}")
        else:
            self.logger.warning(f"No quality gates removed for name: {gate_name}")
    
    def get_gate_configuration(self) -> Dict[str, Any]:
        """Get current gate configuration"""
        return {
            'gates': [
                {
                    'name': gate.__class__.__name__,
                    'type': gate.__class__.__name__,
                    'config': getattr(gate, 'config', {})
                }
                for gate in self.gates
            ],
            'total_gates': len(self.gates)
        }


# REMOVED: PerformanceValidator - Functionality migrated to TechnicalQualityGate PerformanceGate
# Use TechnicalQualityGate with PerformanceGate for concrete performance validation.
#
# Migration path:
# - Performance thresholds -> TechnicalQualityGate config
# - validate_performance_claims() -> TechnicalQualityGate.validate_code_quality() with performance metrics