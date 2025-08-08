"""
Research Validation Metrics

Provides comprehensive research validation including Sanskrit linguistic accuracy,
IAST transliteration compliance, and academic standard validation tools.
"""

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from pathlib import Path
from enum import Enum
import json

from src.utils.logger_config import get_logger
from src.utils.iast_transliterator import IASTTransliterator
from src.utils.text_utils import normalize_unicode

logger = get_logger(__name__)


class AcademicStandard(Enum):
    """Supported academic standards for validation"""
    IAST = "international_alphabet_sanskrit_transliteration"
    ISO_15919 = "iso_15919_transliteration"
    HARVARD_KYOTO = "harvard_kyoto_convention"
    SCHOLARLY_CITATION = "scholarly_citation_format"


class ValidationSeverity(Enum):
    """Validation issue severity levels"""
    CRITICAL = "critical"
    WARNING = "warning" 
    INFO = "info"


@dataclass
class ValidationIssue:
    """Individual validation issue"""
    issue_type: str
    severity: ValidationSeverity
    message: str
    location: str
    original_text: str
    suggested_fix: Optional[str] = None
    academic_reference: Optional[str] = None


@dataclass
class IASTValidationResult:
    """Result of IAST transliteration validation"""
    text_validated: str
    is_compliant: bool
    compliance_score: float
    issues_found: List[ValidationIssue] = field(default_factory=list)
    corrected_text: Optional[str] = None
    academic_references: List[str] = field(default_factory=list)


@dataclass
class SanskritLinguisticValidation:
    """Result of Sanskrit linguistic processing validation"""
    text_analyzed: str
    linguistic_accuracy: float
    phonetic_accuracy: float
    morphological_accuracy: float
    sandhi_accuracy: float
    issues_found: List[ValidationIssue] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class AcademicValidationReport:
    """Comprehensive academic validation report"""
    validation_type: AcademicStandard
    text_segments_validated: int
    overall_compliance_score: float
    critical_issues: int
    warnings: int
    iast_validation: Optional[IASTValidationResult] = None
    linguistic_validation: Optional[SanskritLinguisticValidation] = None
    detailed_issues: List[ValidationIssue] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class ResearchValidationMetrics:
    """
    Comprehensive research validation metrics for academic accuracy validation.
    
    Provides Sanskrit linguistic accuracy validation, IAST transliteration compliance,
    and integration with golden dataset validation framework.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize IAST transliterator
        self.iast_transliterator = IASTTransliterator()
        
        # Sanskrit linguistic patterns for validation
        self.sanskrit_patterns = self._initialize_sanskrit_patterns()
        
        # Academic reference sources
        self.academic_references = self._load_academic_references()
        
    def _initialize_sanskrit_patterns(self) -> Dict[str, List[str]]:
        """Initialize Sanskrit linguistic validation patterns"""
        return {
            'vowels': ['अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ॠ', 'ऌ', 'ए', 'ऐ', 'ओ', 'औ'],
            'consonants': [
                'क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण',
                'त', 'थ', 'द', 'ध', 'न', 'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व',
                'श', 'ष', 'स', 'ह', 'क्ष', 'त्र', 'ज्ञ'
            ],
            'iast_vowels': ['a', 'ā', 'i', 'ī', 'u', 'ū', 'ṛ', 'ṝ', 'ḷ', 'e', 'ai', 'o', 'au'],
            'iast_consonants': [
                'k', 'kh', 'g', 'gh', 'ṅ', 'c', 'ch', 'j', 'jh', 'ñ', 'ṭ', 'ṭh', 'ḍ', 'ḍh', 'ṇ',
                't', 'th', 'd', 'dh', 'n', 'p', 'ph', 'b', 'bh', 'm', 'y', 'r', 'l', 'v',
                'ś', 'ṣ', 's', 'h', 'kṣ', 'tr', 'jñ'
            ]
        }
    
    def _load_academic_references(self) -> Dict[str, str]:
        """Load academic reference sources for validation"""
        return {
            'iast_standard': "International Alphabet of Sanskrit Transliteration (IAST)",
            'iso_15919': "ISO 15919:2001 Transliteration of Devanagari",
            'monier_williams': "Monier-Williams Sanskrit-English Dictionary",
            'whitney_grammar': "Whitney's Sanskrit Grammar",
            'macdonell_grammar': "Macdonell's Vedic Grammar"
        }
    
    def validate_iast_compliance(self, text: str, strict_mode: bool = True) -> IASTValidationResult:
        """
        Validate IAST transliteration compliance for given text.
        
        Args:
            text: Text to validate for IAST compliance
            strict_mode: Whether to apply strict validation rules
        
        Returns:
            Comprehensive IAST validation result
        """
        self.logger.debug(f"Validating IAST compliance for: {text[:50]}...")
        
        issues = []
        normalized_text = normalize_unicode(text)
        
        # Check for proper IAST diacritics usage
        iast_issues = self._check_iast_diacritics(normalized_text, strict_mode)
        issues.extend(iast_issues)
        
        # Check for mixed transliteration systems
        mixed_system_issues = self._check_mixed_transliteration(normalized_text)
        issues.extend(mixed_system_issues)
        
        # Check for common IAST errors
        common_error_issues = self._check_common_iast_errors(normalized_text)
        issues.extend(common_error_issues)
        
        # Calculate compliance score
        critical_issues = len([i for i in issues if i.severity == ValidationSeverity.CRITICAL])
        warning_issues = len([i for i in issues if i.severity == ValidationSeverity.WARNING])
        
        compliance_score = max(0.0, 1.0 - (critical_issues * 0.2) - (warning_issues * 0.05))
        is_compliant = compliance_score >= 0.8 and critical_issues == 0
        
        # Generate corrected text if possible
        corrected_text = self._generate_iast_corrections(normalized_text, issues) if issues else None
        
        return IASTValidationResult(
            text_validated=text,
            is_compliant=is_compliant,
            compliance_score=compliance_score,
            issues_found=issues,
            corrected_text=corrected_text,
            academic_references=[self.academic_references['iast_standard']]
        )
    
    def _check_iast_diacritics(self, text: str, strict_mode: bool) -> List[ValidationIssue]:
        """Check for proper IAST diacritic usage"""
        issues = []
        
        # Common incorrect diacritics patterns
        incorrect_patterns = {
            r'[āīūṛṝḷēōḥṃṇṅñṭḍśṣḻ]': 'Potential incorrect IAST diacritic usage',
            r'[àáâãäèéêëìíîïòóôõöùúûü]': 'Non-IAST diacritics found, should use IAST standard',
            r'\b[a-z]*[AEIOU][a-z]*\b': 'Mixed case in transliteration may indicate IAST violation'
        }
        
        for pattern, message in incorrect_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                severity = ValidationSeverity.CRITICAL if strict_mode else ValidationSeverity.WARNING
                issues.append(ValidationIssue(
                    issue_type="iast_diacritic",
                    severity=severity,
                    message=message,
                    location=f"Position {match.start()}-{match.end()}",
                    original_text=match.group(),
                    academic_reference=self.academic_references['iast_standard']
                ))
        
        return issues
    
    def _check_mixed_transliteration(self, text: str) -> List[ValidationIssue]:
        """Check for mixed transliteration systems"""
        issues = []
        
        # Harvard-Kyoto patterns that shouldn't appear in IAST
        hk_patterns = {
            r'[RMH]': 'Harvard-Kyoto capitals found, should use IAST lowercase with diacritics',
            r'\.([mnh])': 'Harvard-Kyoto dot notation found, should use IAST diacritics'
        }
        
        for pattern, message in hk_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                issues.append(ValidationIssue(
                    issue_type="mixed_transliteration",
                    severity=ValidationSeverity.WARNING,
                    message=message,
                    location=f"Position {match.start()}-{match.end()}",
                    original_text=match.group(),
                    academic_reference=self.academic_references['iso_15919']
                ))
        
        return issues
    
    def _check_common_iast_errors(self, text: str) -> List[ValidationIssue]:
        """Check for common IAST transliteration errors"""
        issues = []
        
        # Common error patterns
        error_patterns = {
            r'\bri\b': 'Should be "ṛ" in IAST, not "ri"',
            r'\bru\b': 'May need to be "rū" in IAST',
            r'sh': 'Should be "ś" or "ṣ" in IAST, not "sh"',
            r'ch(?![a-z])': 'Should be "c" in IAST, not "ch"',
            r'Rishi': 'Should be "ṛṣi" in IAST',
            r'Krishna': 'Should be "kṛṣṇa" in IAST'
        }
        
        for pattern, suggestion in error_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                issues.append(ValidationIssue(
                    issue_type="common_error",
                    severity=ValidationSeverity.WARNING,
                    message=f"Common IAST error: {suggestion}",
                    location=f"Position {match.start()}-{match.end()}",
                    original_text=match.group(),
                    suggested_fix=suggestion,
                    academic_reference=self.academic_references['iast_standard']
                ))
        
        return issues
    
    def _generate_iast_corrections(self, text: str, issues: List[ValidationIssue]) -> str:
        """Generate IAST corrected text based on identified issues"""
        corrected = text
        
        # Apply simple corrections based on issues
        corrections = {
            'Krishna': 'kṛṣṇa',
            'Rishi': 'ṛṣi', 
            'dharma': 'dharma',  # Already correct
            'yoga': 'yoga',      # Already correct
            'sh': 'ś',
            'ch': 'c'
        }
        
        for incorrect, correct in corrections.items():
            corrected = re.sub(r'\b' + re.escape(incorrect) + r'\b', correct, corrected, flags=re.IGNORECASE)
        
        return corrected if corrected != text else None
    
    def validate_sanskrit_linguistics(self, text: str, context: Optional[str] = None) -> SanskritLinguisticValidation:
        """
        Validate Sanskrit linguistic processing accuracy.
        
        Args:
            text: Sanskrit text to validate
            context: Optional context for better validation
        
        Returns:
            Comprehensive linguistic validation result
        """
        self.logger.debug(f"Validating Sanskrit linguistics for: {text[:50]}...")
        
        issues = []
        recommendations = []
        
        # Phonetic accuracy validation
        phonetic_score = self._validate_phonetic_accuracy(text, issues)
        
        # Morphological accuracy validation  
        morphological_score = self._validate_morphological_accuracy(text, issues)
        
        # Sandhi accuracy validation
        sandhi_score = self._validate_sandhi_accuracy(text, issues, context)
        
        # Overall linguistic accuracy
        linguistic_accuracy = (phonetic_score + morphological_score + sandhi_score) / 3.0
        
        # Generate recommendations
        if phonetic_score < 0.8:
            recommendations.append("Consider phonetic validation against Sanskrit sound patterns")
        if morphological_score < 0.8:
            recommendations.append("Review morphological analysis for grammatical accuracy")
        if sandhi_score < 0.8:
            recommendations.append("Validate sandhi rules application for compound words")
        
        return SanskritLinguisticValidation(
            text_analyzed=text,
            linguistic_accuracy=linguistic_accuracy,
            phonetic_accuracy=phonetic_score,
            morphological_accuracy=morphological_score,
            sandhi_accuracy=sandhi_score,
            issues_found=issues,
            recommendations=recommendations
        )
    
    def _validate_phonetic_accuracy(self, text: str, issues: List[ValidationIssue]) -> float:
        """Validate phonetic accuracy of Sanskrit text"""
        # Basic phonetic pattern validation
        valid_patterns = 0
        total_patterns = 0
        
        # Check for valid Sanskrit phonetic combinations
        sanskrit_words = re.findall(r'[a-zA-Zāīūṛṝḷēōḥṃṇṅñṭḍśṣḻ]+', text)
        
        for word in sanskrit_words:
            total_patterns += 1
            
            # Check for valid phonetic patterns (simplified)
            if self._is_valid_sanskrit_phonetic_pattern(word):
                valid_patterns += 1
            else:
                issues.append(ValidationIssue(
                    issue_type="phonetic_accuracy",
                    severity=ValidationSeverity.WARNING,
                    message=f"Questionable phonetic pattern in '{word}'",
                    location=f"Word: {word}",
                    original_text=word,
                    academic_reference=self.academic_references['whitney_grammar']
                ))
        
        return valid_patterns / max(total_patterns, 1)
    
    def _validate_morphological_accuracy(self, text: str, issues: List[ValidationIssue]) -> float:
        """Validate morphological accuracy of Sanskrit text"""
        # Simplified morphological validation
        valid_morphology = 0
        total_words = 0
        
        words = re.findall(r'[a-zA-Zāīūṛṝḷēōḥṃṇṅñṭḍśṣḻ]+', text)
        
        for word in words:
            total_words += 1
            
            # Basic morphological checks (simplified)
            if self._has_valid_sanskrit_morphology(word):
                valid_morphology += 1
            else:
                issues.append(ValidationIssue(
                    issue_type="morphological_accuracy",
                    severity=ValidationSeverity.INFO,
                    message=f"Uncertain morphological analysis for '{word}'",
                    location=f"Word: {word}",
                    original_text=word,
                    academic_reference=self.academic_references['macdonell_grammar']
                ))
        
        return valid_morphology / max(total_words, 1)
    
    def _validate_sandhi_accuracy(self, text: str, issues: List[ValidationIssue], 
                                 context: Optional[str] = None) -> float:
        """Validate sandhi rule accuracy"""
        # Simplified sandhi validation
        sandhi_applications = 0
        total_junctions = 0
        
        # Look for potential sandhi junctions
        words = text.split()
        for i in range(len(words) - 1):
            total_junctions += 1
            
            # Check if sandhi rules are properly applied (simplified)
            if self._is_valid_sandhi_junction(words[i], words[i+1]):
                sandhi_applications += 1
            else:
                issues.append(ValidationIssue(
                    issue_type="sandhi_accuracy", 
                    severity=ValidationSeverity.INFO,
                    message=f"Potential sandhi issue between '{words[i]}' and '{words[i+1]}'",
                    location=f"Junction: {words[i]} + {words[i+1]}",
                    original_text=f"{words[i]} {words[i+1]}",
                    academic_reference=self.academic_references['whitney_grammar']
                ))
        
        return sandhi_applications / max(total_junctions, 1) if total_junctions > 0 else 1.0
    
    def _is_valid_sanskrit_phonetic_pattern(self, word: str) -> bool:
        """Check if word follows valid Sanskrit phonetic patterns (simplified)"""
        # Basic validation - Sanskrit words typically don't end with consonant clusters
        if re.search(r'[bcdfghjklmnpqrstvwxyz]{3,}$', word, re.IGNORECASE):
            return False
        
        # Sanskrit words don't typically start with certain consonant combinations
        if re.match(r'^[bcdfghjklmnpqrstvwxyz]{3,}', word, re.IGNORECASE):
            return False
        
        return True
    
    def _has_valid_sanskrit_morphology(self, word: str) -> bool:
        """Check if word has valid Sanskrit morphological structure (simplified)"""
        # Basic check - most Sanskrit words have some vowels
        vowel_count = len(re.findall(r'[aeiouāīūṛṝḷēōḥ]', word, re.IGNORECASE))
        consonant_count = len(re.findall(r'[bcdfghjklmnpqrstvwxyzṃṇṅñṭḍśṣḻ]', word, re.IGNORECASE))
        
        # Reasonable vowel-to-consonant ratio
        return vowel_count > 0 and (vowel_count / max(consonant_count, 1)) > 0.2
    
    def _is_valid_sandhi_junction(self, word1: str, word2: str) -> bool:
        """Check if sandhi junction between words follows rules (simplified)"""
        # Very basic sandhi validation
        # This is a simplified implementation - full sandhi rules are complex
        
        if not word1 or not word2:
            return True
        
        last_char = word1[-1].lower()
        first_char = word2[0].lower()
        
        # Some basic sandhi patterns
        valid_junctions = {
            ('a', 'a'): True,  # a + a = ā
            ('a', 'i'): True,  # a + i = e  
            ('a', 'u'): True,  # a + u = o
        }
        
        return valid_junctions.get((last_char, first_char), True)  # Default to valid
    
    def generate_academic_validation_report(self, text_segments: List[str], 
                                          standard: AcademicStandard = AcademicStandard.IAST) -> AcademicValidationReport:
        """
        Generate comprehensive academic validation report for text segments.
        
        Args:
            text_segments: List of text segments to validate
            standard: Academic standard to validate against
        
        Returns:
            Comprehensive academic validation report
        """
        self.logger.info(f"Generating academic validation report for {len(text_segments)} segments")
        
        all_issues = []
        iast_results = []
        linguistic_results = []
        
        for i, segment in enumerate(text_segments):
            # IAST validation
            iast_result = self.validate_iast_compliance(segment)
            iast_results.append(iast_result)
            all_issues.extend(iast_result.issues_found)
            
            # Sanskrit linguistic validation
            linguistic_result = self.validate_sanskrit_linguistics(segment)
            linguistic_results.append(linguistic_result)
            all_issues.extend(linguistic_result.issues_found)
        
        # Calculate overall metrics
        overall_compliance = sum(r.compliance_score for r in iast_results) / len(iast_results)
        critical_issues = len([i for i in all_issues if i.severity == ValidationSeverity.CRITICAL])
        warnings = len([i for i in all_issues if i.severity == ValidationSeverity.WARNING])
        
        # Generate recommendations
        recommendations = []
        if overall_compliance < 0.8:
            recommendations.append("Consider comprehensive IAST compliance review")
        if critical_issues > 0:
            recommendations.append("Address critical validation issues before publication")
        if warnings > len(text_segments) * 0.1:
            recommendations.append("Review warning-level issues for academic accuracy")
        
        return AcademicValidationReport(
            validation_type=standard,
            text_segments_validated=len(text_segments),
            overall_compliance_score=overall_compliance,
            critical_issues=critical_issues,
            warnings=warnings,
            iast_validation=iast_results[0] if iast_results else None,
            linguistic_validation=linguistic_results[0] if linguistic_results else None,
            detailed_issues=all_issues,
            recommendations=recommendations
        )
    
    def export_validation_report(self, report: AcademicValidationReport, output_path: Path) -> None:
        """Export academic validation report to JSON file"""
        try:
            report_data = {
                'validation_type': report.validation_type.value,
                'timestamp': time.time(),
                'text_segments_validated': report.text_segments_validated,
                'overall_compliance_score': report.overall_compliance_score,
                'critical_issues': report.critical_issues,
                'warnings': report.warnings,
                'detailed_issues': [
                    {
                        'issue_type': issue.issue_type,
                        'severity': issue.severity.value,
                        'message': issue.message,
                        'location': issue.location,
                        'original_text': issue.original_text,
                        'suggested_fix': issue.suggested_fix,
                        'academic_reference': issue.academic_reference
                    }
                    for issue in report.detailed_issues
                ],
                'recommendations': report.recommendations,
                'academic_references': self.academic_references
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Academic validation report exported to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export validation report: {e}")
            raise