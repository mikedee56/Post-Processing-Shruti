"""
Quality Gate Validator for Story 4.1: MCP Infrastructure Foundation

This module provides automated quality gate validation to prevent regression
of critical quality patterns during text processing.
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum

from .advanced_text_normalizer import AdvancedTextNormalizer, NumberContextType


class QualityGateSeverity(Enum):
    """Severity levels for quality gate violations."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class QualityGateViolation:
    """Represents a quality gate violation."""
    gate_name: str
    severity: QualityGateSeverity
    description: str
    input_text: str
    actual_result: str
    expected_result: Optional[str] = None
    confidence_score: Optional[float] = None
    context_type: Optional[NumberContextType] = None


@dataclass
class QualityGateResult:
    """Result of quality gate validation."""
    passed: bool
    violations: List[QualityGateViolation]
    total_checks: int
    processing_time_ms: float
    overall_confidence: float


class QualityGateValidator:
    """
    Quality gate validator for preventing regression of critical patterns.
    
    Implements AC3 requirements:
    - Comprehensive regression testing for "one by one" patterns
    - Quality gate validation for context classification confidence
    - Automated testing for idiomatic expression preservation
    - Confidence scoring system for processing decisions
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize quality gate validator."""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Quality thresholds
        self.min_idiomatic_confidence = self.config.get('min_idiomatic_confidence', 0.85)
        self.min_scriptural_confidence = self.config.get('min_scriptural_confidence', 0.80)
        self.min_temporal_confidence = self.config.get('min_temporal_confidence', 0.85)
        self.max_processing_time_ms = self.config.get('max_processing_time_ms', 1000)
        
        # Critical patterns that must NEVER be converted
        self.critical_preservation_patterns = [
            "one by one", "two by two", "step by step", "day by day",
            "hand in hand", "side by side", "piece by piece"
        ]
    
    def validate_text_processing(self, normalizer: AdvancedTextNormalizer, text: str) -> QualityGateResult:
        """
        Validate text processing against all quality gates.
        
        Args:
            normalizer: AdvancedTextNormalizer instance to test
            text: Text to process and validate
            
        Returns:
            QualityGateResult with comprehensive validation results
        """
        start_time = time.time()
        violations = []
        total_checks = 0
        
        try:
            # Process the text
            processed_text = normalizer.convert_numbers_with_context(text)
            context_type, confidence, segments = normalizer._classify_number_context_enhanced(text)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Gate 1: Critical pattern preservation
            preservation_violations = self._check_critical_pattern_preservation(
                text, processed_text, context_type, confidence
            )
            violations.extend(preservation_violations)
            total_checks += len(self.critical_preservation_patterns)
            
            # Gate 2: Context classification confidence
            confidence_violations = self._check_context_classification_confidence(
                text, context_type, confidence
            )
            violations.extend(confidence_violations)
            total_checks += 1
            
            # Gate 3: Processing performance
            performance_violations = self._check_processing_performance(
                text, processing_time_ms
            )
            violations.extend(performance_violations)
            total_checks += 1
            
            # Gate 4: Result quality validation
            quality_violations = self._check_result_quality(
                text, processed_text, context_type
            )
            violations.extend(quality_violations)
            total_checks += 1
            
            # Determine overall pass/fail
            critical_violations = [v for v in violations if v.severity == QualityGateSeverity.CRITICAL]
            passed = len(critical_violations) == 0
            
            return QualityGateResult(
                passed=passed,
                violations=violations,
                total_checks=total_checks,
                processing_time_ms=processing_time_ms,
                overall_confidence=confidence
            )
            
        except Exception as e:
            # Processing failure is a critical violation
            violation = QualityGateViolation(
                gate_name="processing_execution",
                severity=QualityGateSeverity.CRITICAL,
                description=f"Text processing failed with error: {str(e)}",
                input_text=text,
                actual_result="PROCESSING_FAILED"
            )
            
            return QualityGateResult(
                passed=False,
                violations=[violation],
                total_checks=1,
                processing_time_ms=(time.time() - start_time) * 1000,
                overall_confidence=0.0
            )
    
    def _check_critical_pattern_preservation(self, 
                                           original_text: str, 
                                           processed_text: str,
                                           context_type: NumberContextType,
                                           confidence: float) -> List[QualityGateViolation]:
        """Check that critical patterns are preserved."""
        violations = []
        original_lower = original_text.lower()
        processed_lower = processed_text.lower()
        
        for critical_pattern in self.critical_preservation_patterns:
            if critical_pattern in original_lower:
                if critical_pattern not in processed_lower:
                    # CRITICAL violation - pattern was lost
                    violation = QualityGateViolation(
                        gate_name="critical_pattern_preservation",
                        severity=QualityGateSeverity.CRITICAL,
                        description=f"Critical pattern '{critical_pattern}' was lost during processing",
                        input_text=original_text,
                        actual_result=processed_text,
                        expected_result=f"Text containing '{critical_pattern}'",
                        confidence_score=confidence,
                        context_type=context_type
                    )
                    violations.append(violation)
                    
                # Additional check: should be classified as IDIOMATIC
                elif context_type != NumberContextType.IDIOMATIC:
                    violation = QualityGateViolation(
                        gate_name="idiomatic_classification",
                        severity=QualityGateSeverity.ERROR,
                        description=f"Critical pattern '{critical_pattern}' was misclassified as {context_type.value}",
                        input_text=original_text,
                        actual_result=context_type.value,
                        expected_result="idiomatic",
                        confidence_score=confidence,
                        context_type=context_type
                    )
                    violations.append(violation)
        
        return violations
    
    def _check_context_classification_confidence(self,
                                               text: str,
                                               context_type: NumberContextType,
                                               confidence: float) -> List[QualityGateViolation]:
        """Check context classification confidence meets thresholds."""
        violations = []
        
        # Define minimum confidence by context type
        min_confidence_by_type = {
            NumberContextType.IDIOMATIC: self.min_idiomatic_confidence,
            NumberContextType.SCRIPTURAL: self.min_scriptural_confidence,
            NumberContextType.TEMPORAL: self.min_temporal_confidence,
            NumberContextType.MATHEMATICAL: 0.75,
            NumberContextType.EDUCATIONAL: 0.70,
            NumberContextType.ORDINAL: 0.70,
            NumberContextType.NARRATIVE: 0.75,
            NumberContextType.UNKNOWN: 0.30,  # Lower threshold for unknown
        }
        
        min_required = min_confidence_by_type.get(context_type, 0.50)
        
        if confidence < min_required:
            severity = QualityGateSeverity.CRITICAL if context_type == NumberContextType.IDIOMATIC else QualityGateSeverity.WARNING
            
            violation = QualityGateViolation(
                gate_name="context_classification_confidence",
                severity=severity,
                description=f"Context classification confidence {confidence:.3f} below threshold {min_required:.3f} for {context_type.value}",
                input_text=text,
                actual_result=f"{confidence:.3f}",
                expected_result=f">={min_required:.3f}",
                confidence_score=confidence,
                context_type=context_type
            )
            violations.append(violation)
        
        return violations
    
    def _check_processing_performance(self, text: str, processing_time_ms: float) -> List[QualityGateViolation]:
        """Check processing performance meets targets."""
        violations = []
        
        if processing_time_ms > self.max_processing_time_ms:
            violation = QualityGateViolation(
                gate_name="processing_performance",
                severity=QualityGateSeverity.WARNING,
                description=f"Processing time {processing_time_ms:.1f}ms exceeds target {self.max_processing_time_ms}ms",
                input_text=text,
                actual_result=f"{processing_time_ms:.1f}ms",
                expected_result=f"<{self.max_processing_time_ms}ms"
            )
            violations.append(violation)
        
        return violations
    
    def _check_result_quality(self,
                            original_text: str,
                            processed_text: str,
                            context_type: NumberContextType) -> List[QualityGateViolation]:
        """Check overall result quality."""
        violations = []
        
        # Basic sanity checks
        if not processed_text or len(processed_text.strip()) == 0:
            violation = QualityGateViolation(
                gate_name="result_quality",
                severity=QualityGateSeverity.CRITICAL,
                description="Processing produced empty or null result",
                input_text=original_text,
                actual_result=processed_text or "NULL",
                expected_result="Non-empty text"
            )
            violations.append(violation)
        
        # Check for inappropriate digit conversion in idiomatic contexts
        if context_type == NumberContextType.IDIOMATIC:
            # Look for digit patterns that shouldn't be there
            import re
            digit_patterns = [r'\b\d+\s+by\s+\d+\b', r'\b\d+\s+at\s+a\s+time\b']
            
            for pattern in digit_patterns:
                if re.search(pattern, processed_text):
                    violation = QualityGateViolation(
                        gate_name="idiomatic_preservation",
                        severity=QualityGateSeverity.CRITICAL,
                        description=f"Idiomatic expression was incorrectly converted to digits: {pattern}",
                        input_text=original_text,
                        actual_result=processed_text,
                        expected_result="Text with preserved words",
                        context_type=context_type
                    )
                    violations.append(violation)
        
        return violations
    
    def run_comprehensive_quality_audit(self, normalizer: AdvancedTextNormalizer) -> Dict[str, Any]:
        """
        Run comprehensive quality audit with predefined test cases.
        
        Returns:
            Detailed audit report with all validation results
        """
        audit_start_time = time.time()
        
        # Predefined test cases covering all critical scenarios
        test_cases = [
            # Critical idiomatic patterns
            "And one by one, he killed six of their children.",
            "Two by two, they entered the ark.",
            "Step by step, we learned the process.",
            
            # Scriptural patterns
            "Chapter two verse twenty five of the Bhagavad Gita.",
            "Verse three of chapter four explains dharma.",
            
            # Temporal patterns
            "Year two thousand five was significant.",
            "In two thousand six, we started this journey.",
            
            # Mathematical patterns
            "Two plus two equals four exactly.",
            "Three times five is fifteen.",
            
            # Educational patterns
            "Lesson two covers basic concepts.",
            "Page twenty five has the answer.",
            
            # Mixed contexts
            "One by one, we studied chapter two verse three.",
            "In year two thousand five, lesson three taught us about step by step processing.",
        ]
        
        audit_results = {
            'audit_timestamp': audit_start_time,
            'total_test_cases': len(test_cases),
            'passed_cases': 0,
            'failed_cases': 0,
            'critical_violations': 0,
            'warning_violations': 0,
            'detailed_results': [],
            'performance_summary': {
                'avg_processing_time_ms': 0.0,
                'max_processing_time_ms': 0.0,
                'min_processing_time_ms': float('inf')
            }
        }
        
        all_processing_times = []
        
        for test_case in test_cases:
            result = self.validate_text_processing(normalizer, test_case)
            
            # Update counters
            if result.passed:
                audit_results['passed_cases'] += 1
            else:
                audit_results['failed_cases'] += 1
            
            # Count violations by severity
            for violation in result.violations:
                if violation.severity == QualityGateSeverity.CRITICAL:
                    audit_results['critical_violations'] += 1
                elif violation.severity == QualityGateSeverity.WARNING:
                    audit_results['warning_violations'] += 1
            
            # Track performance
            all_processing_times.append(result.processing_time_ms)
            
            # Store detailed result
            audit_results['detailed_results'].append({
                'test_case': test_case,
                'passed': result.passed,
                'processing_time_ms': result.processing_time_ms,
                'confidence': result.overall_confidence,
                'violations': [
                    {
                        'gate': v.gate_name,
                        'severity': v.severity.value,
                        'description': v.description
                    }
                    for v in result.violations
                ]
            })
        
        # Calculate performance summary
        if all_processing_times:
            audit_results['performance_summary'] = {
                'avg_processing_time_ms': sum(all_processing_times) / len(all_processing_times),
                'max_processing_time_ms': max(all_processing_times),
                'min_processing_time_ms': min(all_processing_times)
            }
        
        # Overall audit assessment
        audit_results['audit_duration_ms'] = (time.time() - audit_start_time) * 1000
        audit_results['overall_passed'] = (
            audit_results['critical_violations'] == 0 and 
            audit_results['failed_cases'] == 0
        )
        
        return audit_results
    
    def validate_critical_patterns_batch(self, 
                                       normalizer: AdvancedTextNormalizer,
                                       test_texts: List[str]) -> Dict[str, bool]:
        """
        Batch validation of critical patterns for CI/CD integration.
        
        Args:
            normalizer: AdvancedTextNormalizer instance
            test_texts: List of texts to validate
            
        Returns:
            Dictionary mapping each text to pass/fail status
        """
        results = {}
        
        for text in test_texts:
            try:
                result = self.validate_text_processing(normalizer, text)
                
                # Pass if no critical violations
                critical_violations = [v for v in result.violations if v.severity == QualityGateSeverity.CRITICAL]
                results[text] = len(critical_violations) == 0
                
            except Exception as e:
                self.logger.error(f"Validation failed for '{text}': {e}")
                results[text] = False
        
        return results
    
    def generate_quality_report(self, audit_results: Dict[str, Any]) -> str:
        """Generate human-readable quality assurance report."""
        report_lines = [
            "=" * 60,
            "QUALITY ASSURANCE VALIDATION REPORT",
            "Story 4.1: MCP Infrastructure Foundation",
            "=" * 60,
            "",
            f"Audit Timestamp: {audit_results['audit_timestamp']:.0f}",
            f"Total Test Cases: {audit_results['total_test_cases']}",
            f"Passed Cases: {audit_results['passed_cases']}",
            f"Failed Cases: {audit_results['failed_cases']}",
            "",
            "VIOLATION SUMMARY:",
            f"  Critical Violations: {audit_results['critical_violations']}",
            f"  Warning Violations: {audit_results['warning_violations']}",
            "",
            "PERFORMANCE SUMMARY:",
            f"  Average Processing Time: {audit_results['performance_summary']['avg_processing_time_ms']:.2f}ms",
            f"  Maximum Processing Time: {audit_results['performance_summary']['max_processing_time_ms']:.2f}ms",
            f"  Performance Target (<1000ms): {'✅ PASS' if audit_results['performance_summary']['max_processing_time_ms'] < 1000 else '❌ FAIL'}",
            "",
        ]
        
        # Add detailed results for failed cases
        failed_cases = [r for r in audit_results['detailed_results'] if not r['passed']]
        if failed_cases:
            report_lines.extend([
                "DETAILED FAILURE ANALYSIS:",
                "-" * 40,
            ])
            
            for i, case in enumerate(failed_cases, 1):
                report_lines.extend([
                    f"{i}. Test Case: {case['test_case'][:50]}...",
                    f"   Processing Time: {case['processing_time_ms']:.2f}ms",
                    f"   Confidence: {case['confidence']:.3f}",
                    "   Violations:",
                ])
                
                for violation in case['violations']:
                    report_lines.append(f"     - {violation['severity'].upper()}: {violation['description']}")
                
                report_lines.append("")
        
        # Overall assessment
        overall_status = "✅ PASSED" if audit_results['overall_passed'] else "❌ FAILED"
        report_lines.extend([
            "=" * 60,
            f"OVERALL QUALITY GATE STATUS: {overall_status}",
            "=" * 60,
        ])
        
        return "\n".join(report_lines)


# Convenience functions for integration

def validate_single_text(text: str, config: Optional[Dict] = None) -> bool:
    """
    Quick validation for single text input.
    
    Returns True if text passes all quality gates, False otherwise.
    """
    normalizer = AdvancedTextNormalizer(config or {'enable_mcp_processing': True})
    validator = QualityGateValidator(config)
    
    result = validator.validate_text_processing(normalizer, text)
    return result.passed


def run_quality_audit(config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Run comprehensive quality audit with default configuration.
    
    Returns complete audit results for analysis.
    """
    normalizer = AdvancedTextNormalizer(config or {'enable_mcp_processing': True})
    validator = QualityGateValidator(config)
    
    return validator.run_comprehensive_quality_audit(normalizer)


if __name__ == "__main__":
    # Run quality audit for validation
    audit_results = run_quality_audit()
    validator = QualityGateValidator()
    
    print(validator.generate_quality_report(audit_results))
    
    # Return appropriate exit code
    exit_code = 0 if audit_results['overall_passed'] else 1
    exit(exit_code)