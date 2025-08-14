"""
Sanskrit Accuracy Validator for Story 4.2 Research-Grade Enhancement

Implements academic validation standards for IAST transliteration quality
and 15% Sanskrit accuracy improvement measurement system.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import logging
import re
import json
from datetime import datetime
from pathlib import Path
import statistics


class IASTComplianceLevel(Enum):
    """IAST transliteration compliance levels."""
    FULL_COMPLIANCE = "full_compliance"      # 100% compliant
    HIGH_COMPLIANCE = "high_compliance"      # >90% compliant
    MEDIUM_COMPLIANCE = "medium_compliance"  # >75% compliant
    LOW_COMPLIANCE = "low_compliance"        # >50% compliant
    NON_COMPLIANT = "non_compliant"         # <=50% compliant


class AccuracyImprovementStatus(Enum):
    """Status of accuracy improvement measurements."""
    TARGET_EXCEEDED = "target_exceeded"      # >20% improvement
    TARGET_MET = "target_met"               # 15-20% improvement
    TARGET_APPROACHING = "target_approaching" # 10-15% improvement
    BELOW_TARGET = "below_target"           # <10% improvement
    BASELINE_NEEDED = "baseline_needed"     # No baseline measurement


@dataclass
class IASTValidationResult:
    """Result of IAST transliteration validation."""
    text: str
    is_compliant: bool
    compliance_level: IASTComplianceLevel
    compliance_score: float
    violations: List[Dict[str, str]]
    suggestions: List[str]
    academic_notes: List[str]


@dataclass
class AccuracyMeasurement:
    """Sanskrit processing accuracy measurement."""
    measurement_id: str
    timestamp: str
    total_terms_processed: int
    correctly_identified: int
    correctly_transliterated: int
    correctly_corrected: int
    false_positives: int
    false_negatives: int
    accuracy_score: float
    precision: float
    recall: float
    f1_score: float


@dataclass
class ImprovementAnalysis:
    """Analysis of Sanskrit processing improvements."""
    baseline_measurement: AccuracyMeasurement
    current_measurement: AccuracyMeasurement
    accuracy_improvement_percent: float
    precision_improvement_percent: float
    recall_improvement_percent: float
    f1_improvement_percent: float
    improvement_status: AccuracyImprovementStatus
    meets_target: bool
    statistical_significance: bool


class SanskritAccuracyValidator:
    """
    Sanskrit Accuracy Validator for research-grade validation.
    
    Provides:
    - IAST transliteration compliance validation
    - 15% Sanskrit accuracy improvement measurement
    - Academic validation standards
    - Statistical significance testing
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Sanskrit accuracy validator.
        
        Args:
            config: Configuration parameters
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # IAST character mappings for validation
        self.iast_vowels = {
            'a', 'ā', 'i', 'ī', 'u', 'ū', 'ṛ', 'ṝ', 'ḷ', 'ḹ', 'e', 'ai', 'o', 'au'
        }
        self.iast_consonants = {
            'k', 'kh', 'g', 'gh', 'ṅ',
            'c', 'ch', 'j', 'jh', 'ñ', 
            'ṭ', 'ṭh', 'ḍ', 'ḍh', 'ṇ',
            't', 'th', 'd', 'dh', 'n',
            'p', 'ph', 'b', 'bh', 'm',
            'y', 'r', 'l', 'v', 'ś', 'ṣ', 's', 'h'
        }
        self.iast_special_chars = {'ṃ', 'ḥ', '\''}
        
        # Measurement history
        self.measurement_history: List[AccuracyMeasurement] = []
        self.improvement_analyses: List[ImprovementAnalysis] = []
        
        # Load baseline if available
        self._load_baseline_measurements()

    def _get_default_config(self) -> Dict:
        """Get default configuration for accuracy validator."""
        return {
            'target_improvement_percent': 15.0,
            'iast_compliance_threshold': 0.9,
            'statistical_significance_threshold': 0.05,
            'min_sample_size': 100,
            'academic_review_required': True,
            'strict_iast_validation': True,
            'enable_statistical_testing': True
        }

    def _load_baseline_measurements(self):
        """Load baseline measurements from file if available."""
        try:
            baseline_path = Path("data/metrics/baseline_accuracy.json")
            if baseline_path.exists():
                with open(baseline_path, 'r', encoding='utf-8') as f:
                    baseline_data = json.load(f)
                
                # Convert to AccuracyMeasurement objects
                for measurement_data in baseline_data.get('measurements', []):
                    measurement = AccuracyMeasurement(**measurement_data)
                    self.measurement_history.append(measurement)
                
                self.logger.info(f"Loaded {len(self.measurement_history)} baseline measurements")
        except Exception as e:
            self.logger.warning(f"Could not load baseline measurements: {e}")

    def validate_iast_compliance(self, text: str, expected_iast: Optional[str] = None) -> IASTValidationResult:
        """
        Validate IAST transliteration compliance for Sanskrit text.
        
        Args:
            text: Text to validate
            expected_iast: Optional expected IAST transliteration for comparison
            
        Returns:
            IASTValidationResult with compliance analysis
        """
        try:
            violations = []
            suggestions = []
            academic_notes = []
            
            # Check for basic IAST character usage
            text_chars = set(text.lower())
            valid_iast_chars = (self.iast_vowels | self.iast_consonants | 
                              self.iast_special_chars | {' ', '-', '\''})
            
            # Find invalid characters
            invalid_chars = text_chars - valid_iast_chars - set('abcdefghijklmnopqrstuvwxyz0123456789.,!?;: ')
            
            if invalid_chars:
                violations.append({
                    'type': 'invalid_characters',
                    'details': f"Invalid IAST characters found: {', '.join(sorted(invalid_chars))}",
                    'severity': 'high'
                })
            
            # Check for common transliteration errors
            common_errors = {
                'ri': 'ṛ',
                'ree': 'ṝ',
                'lri': 'ḷ',
                'sh': 'ś',
                'Sh': 'Ṣ',
                'ng': 'ṅ',
                'ngh': 'ṅh',
                'chh': 'ch',
                'N': 'ṇ',
                'T': 'ṭ',
                'D': 'ḍ'
            }
            
            for incorrect, correct in common_errors.items():
                if incorrect in text:
                    violations.append({
                        'type': 'common_error',
                        'details': f"'{incorrect}' should be '{correct}'",
                        'severity': 'medium'
                    })
                    suggestions.append(f"Replace '{incorrect}' with '{correct}' for proper IAST")
            
            # Check for long vowel markings
            if re.search(r'[aeiou]{2,}', text.lower()):
                violations.append({
                    'type': 'vowel_length',
                    'details': 'Double vowels found - use diacritical marks for long vowels',
                    'severity': 'high'
                })
                suggestions.append('Use ā, ī, ū for long vowels instead of aa, ii, uu')
            
            # Check against expected IAST if provided
            if expected_iast:
                if text.lower() != expected_iast.lower():
                    violations.append({
                        'type': 'expected_mismatch',
                        'details': f"Text '{text}' does not match expected IAST '{expected_iast}'",
                        'severity': 'high'
                    })
            
            # Calculate compliance score
            total_checks = 5  # Number of validation checks
            violations_count = len([v for v in violations if v['severity'] == 'high'])
            compliance_score = max(0.0, 1.0 - (violations_count / total_checks))
            
            # Determine compliance level
            if compliance_score >= 1.0:
                compliance_level = IASTComplianceLevel.FULL_COMPLIANCE
            elif compliance_score >= 0.9:
                compliance_level = IASTComplianceLevel.HIGH_COMPLIANCE
            elif compliance_score >= 0.75:
                compliance_level = IASTComplianceLevel.MEDIUM_COMPLIANCE
            elif compliance_score >= 0.5:
                compliance_level = IASTComplianceLevel.LOW_COMPLIANCE
            else:
                compliance_level = IASTComplianceLevel.NON_COMPLIANT
            
            is_compliant = compliance_score >= self.config['iast_compliance_threshold']
            
            # Add academic notes
            if compliance_level == IASTComplianceLevel.FULL_COMPLIANCE:
                academic_notes.append("Text fully complies with IAST transliteration standards")
            elif violations:
                academic_notes.append(f"Text has {len(violations)} IAST compliance issues")
            
            return IASTValidationResult(
                text=text,
                is_compliant=is_compliant,
                compliance_level=compliance_level,
                compliance_score=compliance_score,
                violations=violations,
                suggestions=suggestions,
                academic_notes=academic_notes
            )
            
        except Exception as e:
            self.logger.error(f"Error validating IAST compliance for '{text}': {e}")
            
            return IASTValidationResult(
                text=text,
                is_compliant=False,
                compliance_level=IASTComplianceLevel.NON_COMPLIANT,
                compliance_score=0.0,
                violations=[{'type': 'validation_error', 'details': str(e), 'severity': 'high'}],
                suggestions=[],
                academic_notes=['Validation failed due to processing error']
            )

    def measure_sanskrit_accuracy(
        self,
        processed_terms: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        measurement_id: Optional[str] = None
    ) -> AccuracyMeasurement:
        """
        Measure Sanskrit processing accuracy against ground truth.
        
        Args:
            processed_terms: List of processed Sanskrit terms with metadata
            ground_truth: Ground truth data for comparison
            measurement_id: Optional measurement identifier
            
        Returns:
            AccuracyMeasurement with detailed accuracy metrics
        """
        try:
            measurement_id = measurement_id or f"accuracy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create lookup for ground truth
            ground_truth_lookup = {item['term'].lower(): item for item in ground_truth}
            
            # Initialize counters
            correctly_identified = 0
            correctly_transliterated = 0
            correctly_corrected = 0
            false_positives = 0
            false_negatives = 0
            
            processed_terms_set = set()
            
            # Evaluate processed terms
            for processed in processed_terms:
                term = processed['term'].lower()
                processed_terms_set.add(term)
                
                if term in ground_truth_lookup:
                    gt_item = ground_truth_lookup[term]
                    
                    # Check identification accuracy
                    if processed.get('identified_as_sanskrit', False) == gt_item.get('is_sanskrit', False):
                        correctly_identified += 1
                    
                    # Check transliteration accuracy (if applicable)
                    if 'transliteration' in processed and 'transliteration' in gt_item:
                        if processed['transliteration'].lower() == gt_item['transliteration'].lower():
                            correctly_transliterated += 1
                    
                    # Check correction accuracy
                    if 'corrected_form' in processed and 'correct_form' in gt_item:
                        if processed['corrected_form'].lower() == gt_item['correct_form'].lower():
                            correctly_corrected += 1
                else:
                    # Term processed but not in ground truth (potential false positive)
                    if processed.get('identified_as_sanskrit', False):
                        false_positives += 1
            
            # Check for false negatives (ground truth terms not processed)
            for gt_item in ground_truth:
                term = gt_item['term'].lower()
                if term not in processed_terms_set and gt_item.get('is_sanskrit', False):
                    false_negatives += 1
            
            # Calculate metrics
            total_terms = len(processed_terms)
            true_positives = correctly_identified
            
            # Accuracy = (TP + TN) / (TP + TN + FP + FN)
            # For simplicity, we'll use identification accuracy as primary metric
            accuracy_score = correctly_identified / total_terms if total_terms > 0 else 0.0
            
            # Precision = TP / (TP + FP)
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            
            # Recall = TP / (TP + FN)
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            
            # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            measurement = AccuracyMeasurement(
                measurement_id=measurement_id,
                timestamp=datetime.now().isoformat(),
                total_terms_processed=total_terms,
                correctly_identified=correctly_identified,
                correctly_transliterated=correctly_transliterated,
                correctly_corrected=correctly_corrected,
                false_positives=false_positives,
                false_negatives=false_negatives,
                accuracy_score=accuracy_score,
                precision=precision,
                recall=recall,
                f1_score=f1_score
            )
            
            # Add to history
            self.measurement_history.append(measurement)
            
            self.logger.info(f"Sanskrit accuracy measurement completed: {accuracy_score:.3f} accuracy")
            
            return measurement
            
        except Exception as e:
            self.logger.error(f"Error measuring Sanskrit accuracy: {e}")
            
            # Return empty measurement on error
            return AccuracyMeasurement(
                measurement_id=measurement_id or "error",
                timestamp=datetime.now().isoformat(),
                total_terms_processed=0,
                correctly_identified=0,
                correctly_transliterated=0,
                correctly_corrected=0,
                false_positives=0,
                false_negatives=0,
                accuracy_score=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0
            )

    def analyze_improvement(
        self,
        baseline_id: Optional[str] = None,
        current_id: Optional[str] = None
    ) -> Optional[ImprovementAnalysis]:
        """
        Analyze improvement between baseline and current measurements.
        
        Args:
            baseline_id: ID of baseline measurement (uses earliest if None)
            current_id: ID of current measurement (uses latest if None)
            
        Returns:
            ImprovementAnalysis if measurements available, None otherwise
        """
        try:
            if len(self.measurement_history) < 2:
                self.logger.warning("Need at least 2 measurements for improvement analysis")
                return None
            
            # Find baseline measurement
            if baseline_id:
                baseline = next((m for m in self.measurement_history if m.measurement_id == baseline_id), None)
            else:
                baseline = self.measurement_history[0]  # Use earliest
            
            # Find current measurement
            if current_id:
                current = next((m for m in self.measurement_history if m.measurement_id == current_id), None)
            else:
                current = self.measurement_history[-1]  # Use latest
            
            if not baseline or not current:
                self.logger.error("Could not find specified measurements for comparison")
                return None
            
            if baseline.measurement_id == current.measurement_id:
                self.logger.error("Cannot compare measurement to itself")
                return None
            
            # Calculate improvements (percentage)
            def calc_improvement(old_val: float, new_val: float) -> float:
                if old_val == 0:
                    return 100.0 if new_val > 0 else 0.0
                return ((new_val - old_val) / old_val) * 100
            
            accuracy_improvement = calc_improvement(baseline.accuracy_score, current.accuracy_score)
            precision_improvement = calc_improvement(baseline.precision, current.precision)
            recall_improvement = calc_improvement(baseline.recall, current.recall)
            f1_improvement = calc_improvement(baseline.f1_score, current.f1_score)
            
            # Determine improvement status
            target_percent = self.config['target_improvement_percent']
            
            if accuracy_improvement >= target_percent + 5:
                improvement_status = AccuracyImprovementStatus.TARGET_EXCEEDED
            elif accuracy_improvement >= target_percent:
                improvement_status = AccuracyImprovementStatus.TARGET_MET
            elif accuracy_improvement >= target_percent - 5:
                improvement_status = AccuracyImprovementStatus.TARGET_APPROACHING
            else:
                improvement_status = AccuracyImprovementStatus.BELOW_TARGET
            
            meets_target = accuracy_improvement >= target_percent
            
            # Statistical significance testing (simplified)
            statistical_significance = False
            if self.config.get('enable_statistical_testing', True):
                # Simple significance test based on sample size and improvement magnitude
                min_sample = self.config['min_sample_size']
                if (current.total_terms_processed >= min_sample and 
                    baseline.total_terms_processed >= min_sample and
                    abs(accuracy_improvement) >= 5.0):  # 5% minimum for significance
                    statistical_significance = True
            
            analysis = ImprovementAnalysis(
                baseline_measurement=baseline,
                current_measurement=current,
                accuracy_improvement_percent=accuracy_improvement,
                precision_improvement_percent=precision_improvement,
                recall_improvement_percent=recall_improvement,
                f1_improvement_percent=f1_improvement,
                improvement_status=improvement_status,
                meets_target=meets_target,
                statistical_significance=statistical_significance
            )
            
            self.improvement_analyses.append(analysis)
            
            self.logger.info(f"Improvement analysis: {accuracy_improvement:.1f}% accuracy improvement")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing improvement: {e}")
            return None

    def generate_research_grade_report(self) -> Dict[str, Any]:
        """Generate comprehensive research-grade accuracy report."""
        try:
            if not self.measurement_history:
                return {"error": "No measurements available for report"}
            
            latest_measurement = self.measurement_history[-1]
            
            # Calculate trends if multiple measurements
            trends = {}
            if len(self.measurement_history) > 1:
                accuracy_scores = [m.accuracy_score for m in self.measurement_history]
                trends['accuracy_trend'] = 'improving' if accuracy_scores[-1] > accuracy_scores[0] else 'declining'
                trends['avg_accuracy'] = statistics.mean(accuracy_scores)
                trends['accuracy_stdev'] = statistics.stdev(accuracy_scores) if len(accuracy_scores) > 1 else 0.0
            
            # Latest improvement analysis
            latest_improvement = self.improvement_analyses[-1] if self.improvement_analyses else None
            
            report = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'report_type': 'research_grade_sanskrit_accuracy',
                    'measurements_count': len(self.measurement_history),
                    'improvement_analyses_count': len(self.improvement_analyses)
                },
                'current_accuracy': {
                    'measurement_id': latest_measurement.measurement_id,
                    'accuracy_score': latest_measurement.accuracy_score,
                    'precision': latest_measurement.precision,
                    'recall': latest_measurement.recall,
                    'f1_score': latest_measurement.f1_score,
                    'total_terms_processed': latest_measurement.total_terms_processed
                },
                'improvement_analysis': {
                    'target_improvement_percent': self.config['target_improvement_percent'],
                    'current_improvement_percent': latest_improvement.accuracy_improvement_percent if latest_improvement else 0.0,
                    'improvement_status': latest_improvement.improvement_status.value if latest_improvement else 'baseline_needed',
                    'meets_target': latest_improvement.meets_target if latest_improvement else False,
                    'statistically_significant': latest_improvement.statistical_significance if latest_improvement else False
                },
                'trends': trends,
                'academic_compliance': {
                    'iast_compliance_threshold': self.config['iast_compliance_threshold'],
                    'strict_validation_enabled': self.config.get('strict_iast_validation', True),
                    'academic_review_required': self.config.get('academic_review_required', True)
                },
                'recommendations': self._generate_recommendations(latest_measurement, latest_improvement)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating research-grade report: {e}")
            return {"error": str(e)}

    def _generate_recommendations(
        self,
        latest_measurement: AccuracyMeasurement,
        latest_improvement: Optional[ImprovementAnalysis]
    ) -> List[str]:
        """Generate academic recommendations based on current performance."""
        recommendations = []
        
        # Accuracy-based recommendations
        if latest_measurement.accuracy_score < 0.8:
            recommendations.append("Consider expanding the Sanskrit lexicon with more comprehensive term coverage")
        
        if latest_measurement.precision < 0.85:
            recommendations.append("Review false positive cases to improve precision in Sanskrit term identification")
        
        if latest_measurement.recall < 0.85:
            recommendations.append("Investigate missed Sanskrit terms to improve recall performance")
        
        # Improvement-based recommendations
        if latest_improvement:
            if latest_improvement.improvement_status == AccuracyImprovementStatus.BELOW_TARGET:
                recommendations.append("Current improvement below target - consider ML model fine-tuning")
            elif latest_improvement.improvement_status == AccuracyImprovementStatus.TARGET_MET:
                recommendations.append("Target improvement achieved - monitor for consistency")
            
            if not latest_improvement.statistical_significance:
                recommendations.append("Increase sample size for statistically significant results")
        
        # General academic recommendations
        recommendations.extend([
            "Ensure IAST transliteration compliance for academic publication standards",
            "Validate results against multiple Sanskrit linguistic authorities",
            "Consider peer review of improved accuracy measurements"
        ])
        
        return recommendations

    def save_measurements(self, file_path: Optional[Path] = None) -> bool:
        """Save accuracy measurements to file."""
        try:
            save_path = file_path or Path("data/metrics/accuracy_measurements.json")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'measurements': [
                    {
                        'measurement_id': m.measurement_id,
                        'timestamp': m.timestamp,
                        'total_terms_processed': m.total_terms_processed,
                        'correctly_identified': m.correctly_identified,
                        'correctly_transliterated': m.correctly_transliterated,
                        'correctly_corrected': m.correctly_corrected,
                        'false_positives': m.false_positives,
                        'false_negatives': m.false_negatives,
                        'accuracy_score': m.accuracy_score,
                        'precision': m.precision,
                        'recall': m.recall,
                        'f1_score': m.f1_score
                    }
                    for m in self.measurement_history
                ],
                'config': self.config,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved {len(self.measurement_history)} measurements to {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving measurements: {e}")
            return False


# Test function for development
def test_sanskrit_accuracy_validator():
    """Test the Sanskrit accuracy validator."""
    validator = SanskritAccuracyValidator()
    
    # Test IAST validation
    test_texts = [
        "kṛṣṇa dharma yoga",  # Good IAST
        "krishna dharma yoga",  # Common errors
        "krsna dharma yog"  # Missing diacritics
    ]
    
    print("=== IAST Validation Tests ===")
    for text in test_texts:
        result = validator.validate_iast_compliance(text)
        print(f"Text: {text}")
        print(f"Compliant: {result.is_compliant}")
        print(f"Score: {result.compliance_score:.3f}")
        print(f"Violations: {len(result.violations)}")
        print("---")
    
    # Test accuracy measurement
    processed_terms = [
        {'term': 'dharma', 'identified_as_sanskrit': True, 'transliteration': 'dharma'},
        {'term': 'yoga', 'identified_as_sanskrit': True, 'transliteration': 'yoga'},
        {'term': 'practice', 'identified_as_sanskrit': False}
    ]
    
    ground_truth = [
        {'term': 'dharma', 'is_sanskrit': True, 'transliteration': 'dharma'},
        {'term': 'yoga', 'is_sanskrit': True, 'transliteration': 'yoga'},
        {'term': 'practice', 'is_sanskrit': False}
    ]
    
    measurement = validator.measure_sanskrit_accuracy(processed_terms, ground_truth)
    print(f"\n=== Accuracy Measurement ===")
    print(f"Accuracy: {measurement.accuracy_score:.3f}")
    print(f"Precision: {measurement.precision:.3f}")
    print(f"Recall: {measurement.recall:.3f}")


if __name__ == "__main__":
    test_sanskrit_accuracy_validator()