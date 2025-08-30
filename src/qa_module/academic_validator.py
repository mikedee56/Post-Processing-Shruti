"""
Academic Validator with IAST Compliance Integration.

This module integrates the QualityGate system with the existing text processing
pipeline to provide comprehensive academic validation for Story 3.2.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

from .quality_gate import QualityGate, ComplianceScore, QualityReport, QualityValidator
from utils.advanced_text_normalizer import AdvancedTextNormalizer, AdvancedCorrectionResult
from utils.iast_transliterator import IASTTransliterator
from utils.sanskrit_accuracy_validator import SanskritAccuracyValidator, IASTValidationResult


@dataclass
class AcademicValidationResult:
    """Result of comprehensive academic validation."""
    original_text: str
    validated_text: str
    quality_score: float
    compliance_score: float
    iast_compliance: IASTValidationResult
    quality_report: QualityReport
    processing_time_ms: float
    validation_passed: bool
    improvement_suggestions: List[str]


class AcademicValidator:
    """
    Academic Validator with IAST Compliance Integration.
    
    Integrates the QualityGate system with existing text processing to provide
    comprehensive academic validation as required by Story 3.2 Epic 3.
    
    Features:
    - IAST transliteration compliance checking
    - Academic quality gate validation
    - Integration with existing AdvancedTextNormalizer
    - Performance tracking to meet <50ms per segment target
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Academic Validator.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config or {}
        
        # Initialize core components
        self.quality_gate = QualityGate(config=self.config)
        self.text_normalizer = AdvancedTextNormalizer(config=self.config)
        self.iast_transliterator = IASTTransliterator(strict_mode=True)
        self.sanskrit_validator = SanskritAccuracyValidator()
        
        # Performance tracking
        self.validation_count = 0
        self.total_processing_time = 0.0
        
        # Quality thresholds from Epic 3 requirements
        self.quality_threshold = self.config.get('quality_threshold', 0.85)
        self.compliance_threshold = self.config.get('compliance_threshold', 0.80)
        self.performance_target_ms = self.config.get('performance_target_ms', 50)
        
        self.logger.info(f"AcademicValidator initialized with quality_threshold={self.quality_threshold}, "
                        f"compliance_threshold={self.compliance_threshold}, "
                        f"performance_target={self.performance_target_ms}ms")
    
    def validate_academic_quality(self, text: str, context: Optional[Dict] = None) -> AcademicValidationResult:
        """
        Perform comprehensive academic validation on text.
        
        This method integrates IAST compliance checking with the QualityGate
        system to provide complete academic validation as required by Epic 3.
        
        Args:
            text: Text to validate
            context: Optional context information
            
        Returns:
            AcademicValidationResult with comprehensive validation data
        """
        start_time = time.perf_counter()
        
        try:
            # Step 1: Apply advanced text normalization
            normalization_result = self.text_normalizer.normalize_with_advanced_tracking(text)
            
            # Step 2: Validate IAST compliance
            iast_result = self.sanskrit_validator.validate_iast_compliance(
                normalization_result.corrected_text
            )
            
            # Step 3: Apply quality gate validation
            quality_report = self.quality_gate.evaluate_quality(
                normalization_result.corrected_text,
                context or {}
            )
            
            # Step 4: Calculate composite scores
            quality_score = self._calculate_quality_score(normalization_result, quality_report)
            compliance_score = self._calculate_compliance_score(iast_result, quality_report)
            
            # Step 5: Determine validation result
            validation_passed = (
                quality_score >= self.quality_threshold and
                compliance_score >= self.compliance_threshold and
                iast_result.compliance_level.value >= 0.80
            )
            
            # Step 6: Generate improvement suggestions
            suggestions = self._generate_improvement_suggestions(
                normalization_result, iast_result, quality_report
            )
            
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Update performance tracking
            self.validation_count += 1
            self.total_processing_time += processing_time_ms
            
            # Log performance warning if target exceeded
            if processing_time_ms > self.performance_target_ms:
                self.logger.warning(
                    f"Validation exceeded performance target: {processing_time_ms:.2f}ms > {self.performance_target_ms}ms"
                )
            
            return AcademicValidationResult(
                original_text=text,
                validated_text=normalization_result.corrected_text,
                quality_score=quality_score,
                compliance_score=compliance_score,
                iast_compliance=iast_result,
                quality_report=quality_report,
                processing_time_ms=processing_time_ms,
                validation_passed=validation_passed,
                improvement_suggestions=suggestions
            )
            
        except Exception as e:
            self.logger.error(f"Academic validation failed: {e}")
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Return minimal result on error
            return AcademicValidationResult(
                original_text=text,
                validated_text=text,
                quality_score=0.0,
                compliance_score=0.0,
                iast_compliance=IASTValidationResult(
                    text=text,
                    compliance_level=self.sanskrit_validator.IASTComplianceLevel.NON_COMPLIANT,
                    compliance_score=0.0,
                    issues_found=[f"Validation error: {str(e)}"],
                    suggested_corrections={},
                    academic_notes=[]
                ),
                quality_report=QualityReport(
                    overall_score=0.0,
                    compliance_score=ComplianceScore(
                        iast_compliance=0.0,
                        sanskrit_accuracy=0.0,
                        proper_noun_capitalization=0.0,
                        terminology_consistency=0.0,
                        academic_formatting=0.0,
                        composite_score=0.0
                    ),
                    detailed_feedback=[f"Validation failed: {str(e)}"],
                    improvement_suggestions=["Fix validation error and retry"],
                    processing_time_ms=processing_time_ms,
                    academic_grade="F",
                    meets_publication_standards=False
                ),
                processing_time_ms=processing_time_ms,
                validation_passed=False,
                improvement_suggestions=[f"Fix validation error: {str(e)}"]
            )
    
    def _calculate_quality_score(
        self, 
        normalization_result: AdvancedCorrectionResult, 
        quality_report: QualityReport
    ) -> float:
        """Calculate composite quality score."""
        # Weight the different quality aspects
        normalization_weight = 0.3
        quality_gate_weight = 0.7
        
        normalization_score = normalization_result.quality_score
        quality_gate_score = quality_report.overall_score
        
        return (normalization_score * normalization_weight + 
                quality_gate_score * quality_gate_weight)
    
    def _calculate_compliance_score(
        self, 
        iast_result: IASTValidationResult, 
        quality_report: QualityReport
    ) -> float:
        """Calculate composite compliance score."""
        # Weight IAST and general compliance
        iast_weight = 0.6
        general_weight = 0.4
        
        iast_score = iast_result.compliance_score
        general_score = quality_report.compliance_score.composite_score
        
        return iast_score * iast_weight + general_score * general_weight
    
    def _generate_improvement_suggestions(
        self,
        normalization_result: AdvancedCorrectionResult,
        iast_result: IASTValidationResult,
        quality_report: QualityReport
    ) -> List[str]:
        """Generate actionable improvement suggestions."""
        suggestions = []
        
        # Add normalization suggestions
        if normalization_result.semantic_drift_score > 0.3:
            suggestions.append("Consider reducing semantic drift during normalization")
        
        # Add IAST compliance suggestions
        if iast_result.compliance_score < 0.9:
            suggestions.extend(iast_result.issues_found[:3])  # Top 3 IAST issues
        
        # Add quality gate suggestions
        suggestions.extend(quality_report.improvement_suggestions[:5])  # Top 5 suggestions
        
        # Remove duplicates while preserving order
        unique_suggestions = []
        seen = set()
        for suggestion in suggestions:
            if suggestion not in seen:
                unique_suggestions.append(suggestion)
                seen.add(suggestion)
        
        return unique_suggestions[:10]  # Limit to top 10 suggestions
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for the validator."""
        if self.validation_count == 0:
            return {
                'average_processing_time_ms': 0.0,
                'total_validations': 0,
                'performance_target_ms': self.performance_target_ms,
                'target_compliance_rate': 0.0
            }
        
        avg_time = self.total_processing_time / self.validation_count
        target_compliance_rate = (
            1.0 if avg_time <= self.performance_target_ms 
            else self.performance_target_ms / avg_time
        )
        
        return {
            'average_processing_time_ms': avg_time,
            'total_validations': self.validation_count,
            'performance_target_ms': self.performance_target_ms,
            'target_compliance_rate': target_compliance_rate
        }
    
    def create_batch_validator(self, batch_size: int = 10) -> 'BatchAcademicValidator':
        """Create a batch validator for processing multiple texts efficiently."""
        return BatchAcademicValidator(self, batch_size)


class BatchAcademicValidator:
    """Batch processor for efficient validation of multiple texts."""
    
    def __init__(self, validator: AcademicValidator, batch_size: int = 10):
        """Initialize batch validator."""
        self.validator = validator
        self.batch_size = batch_size
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate_batch(self, texts: List[str], contexts: Optional[List[Dict]] = None) -> List[AcademicValidationResult]:
        """
        Validate a batch of texts efficiently.
        
        Args:
            texts: List of texts to validate
            contexts: Optional list of context dictionaries
            
        Returns:
            List of validation results
        """
        if contexts is None:
            contexts = [None] * len(texts)
        
        if len(texts) != len(contexts):
            raise ValueError("Number of texts and contexts must match")
        
        results = []
        start_time = time.perf_counter()
        
        for i, (text, context) in enumerate(zip(texts, contexts)):
            try:
                result = self.validator.validate_academic_quality(text, context)
                results.append(result)
                
                # Log progress for large batches
                if len(texts) > 20 and (i + 1) % 10 == 0:
                    elapsed = time.perf_counter() - start_time
                    rate = (i + 1) / elapsed
                    self.logger.info(f"Processed {i + 1}/{len(texts)} texts ({rate:.1f} texts/sec)")
                    
            except Exception as e:
                self.logger.error(f"Failed to validate text {i}: {e}")
                # Create error result
                results.append(self._create_error_result(text, str(e)))
        
        total_time = time.perf_counter() - start_time
        self.logger.info(f"Batch validation complete: {len(texts)} texts in {total_time:.2f}s")
        
        return results
    
    def _create_error_result(self, text: str, error_msg: str) -> AcademicValidationResult:
        """Create an error result for failed validation."""
        return AcademicValidationResult(
            original_text=text,
            validated_text=text,
            quality_score=0.0,
            compliance_score=0.0,
            iast_compliance=IASTValidationResult(
                text=text,
                compliance_level=self.validator.sanskrit_validator.IASTComplianceLevel.NON_COMPLIANT,
                compliance_score=0.0,
                issues_found=[f"Validation error: {error_msg}"],
                suggested_corrections={},
                academic_notes=[]
            ),
            quality_report=QualityReport(
                overall_score=0.0,
                compliance_score=ComplianceScore(
                    iast_compliance=0.0,
                    sanskrit_accuracy=0.0,
                    proper_noun_capitalization=0.0,
                    terminology_consistency=0.0,
                    academic_formatting=0.0,
                    composite_score=0.0
                ),
                detailed_feedback=[f"Validation failed: {error_msg}"],
                improvement_suggestions=["Fix validation error and retry"],
                processing_time_ms=0.0,
                academic_grade="F",
                meets_publication_standards=False
            ),
            processing_time_ms=0.0,
            validation_passed=False,
            improvement_suggestions=[f"Fix validation error: {error_msg}"]
        )