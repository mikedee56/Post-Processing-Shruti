"""
Academic Validator for Scripture Intelligence Enhancement.

This module provides comprehensive academic compliance validation and quality
assurance systems for research publication readiness, ensuring adherence to
academic standards and scholarly rigor.

Story 4.5: Scripture Intelligence Enhancement - Academic Validation Component
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from enum import Enum
from pathlib import Path
import logging
import re
import datetime
import json

from utils.logger_config import get_logger


class AcademicStandard(Enum):
    """Academic standards for validation."""
    UNDERGRADUATE = "undergraduate"
    GRADUATE = "graduate"
    DOCTORAL = "doctoral"
    PEER_REVIEW = "peer_review"
    JOURNAL_PUBLICATION = "journal_publication"
    BOOK_PUBLICATION = "book_publication"


class ComplianceLevel(Enum):
    """Compliance levels for academic validation."""
    BASIC = "basic"
    STANDARD = "standard"
    RIGOROUS = "rigorous"
    EXEMPLARY = "exemplary"


class ValidationCategory(Enum):
    """Categories of academic validation."""
    CITATION_COMPLIANCE = "citation_compliance"
    TRANSLITERATION_ACCURACY = "transliteration_accuracy"
    SCHOLARLY_RIGOR = "scholarly_rigor"
    METHODOLOGICAL_SOUNDNESS = "methodological_soundness"
    ACADEMIC_INTEGRITY = "academic_integrity"
    PUBLICATION_READINESS = "publication_readiness"


@dataclass
class ValidationRule:
    """Individual validation rule for academic compliance."""
    
    rule_id: str
    category: ValidationCategory
    description: str
    validation_function: str  # Name of validation method
    weight: float = 1.0
    severity: str = "warning"  # error, warning, info
    required_for_standards: List[AcademicStandard] = field(default_factory=list)
    
    # Rule configuration
    parameters: Dict[str, Any] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result of academic validation assessment."""
    
    rule_id: str
    category: ValidationCategory
    passed: bool
    score: float = 0.0
    
    # Detailed feedback
    message: str = ""
    suggestions: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    severity: str = "info"
    
    # Context information
    validated_content: str = ""
    rule_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComprehensiveValidationReport:
    """Comprehensive academic validation report."""
    
    # Overall assessment
    overall_compliance: ComplianceLevel
    overall_score: float = 0.0
    academic_standard_met: List[AcademicStandard] = field(default_factory=list)
    
    # Category-wise results
    category_scores: Dict[ValidationCategory, float] = field(default_factory=dict)
    validation_results: List[ValidationResult] = field(default_factory=list)
    
    # Summary statistics
    total_rules_evaluated: int = 0
    rules_passed: int = 0
    rules_failed: int = 0
    critical_issues: int = 0
    warnings: int = 0
    
    # Recommendations
    improvement_priorities: List[str] = field(default_factory=list)
    excellence_areas: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    
    # Metadata
    validation_timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    validator_version: str = "1.0"
    content_analyzed: Dict[str, int] = field(default_factory=dict)


class AcademicValidator:
    """
    Comprehensive Academic Validator for Scripture Intelligence Enhancement.
    
    Provides systematic validation of academic compliance, scholarly rigor,
    and publication readiness according to established academic standards.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Academic Validator.
        
        Args:
            config: Optional configuration parameters
        """
        self.logger = get_logger(__name__)
        self.config = config or {}
        
        # Initialize validation rules and standards
        self._initialize_validation_rules()
        self._initialize_academic_standards()
        self._initialize_reference_materials()
        
        # Performance tracking
        self.validation_stats = {
            'validations_performed': 0,
            'average_compliance_score': 0.0,
            'standards_distribution': {},
            'common_issues': {},
            'improvement_trends': []
        }
        
        self.logger.info("Academic Validator initialized with comprehensive rule set")
    
    def _initialize_validation_rules(self) -> None:
        """Initialize comprehensive set of academic validation rules."""
        
        self.validation_rules = [
            # Citation Compliance Rules
            ValidationRule(
                rule_id="cite_001",
                category=ValidationCategory.CITATION_COMPLIANCE,
                description="Citations follow established academic format standards",
                validation_function="_validate_citation_format",
                weight=1.5,
                severity="error",
                required_for_standards=[AcademicStandard.PEER_REVIEW, AcademicStandard.JOURNAL_PUBLICATION],
                parameters={'required_elements': ['source', 'reference', 'formatting']},
                examples=["BG 2.47", "Yoga Sūtras 1.14", "Muṇḍ. 3.2.9"]
            ),
            
            ValidationRule(
                rule_id="cite_002",
                category=ValidationCategory.CITATION_COMPLIANCE,
                description="Source abbreviations follow scholarly conventions",
                validation_function="_validate_source_abbreviations",
                weight=1.2,
                severity="warning",
                required_for_standards=[AcademicStandard.GRADUATE, AcademicStandard.DOCTORAL],
                parameters={'standard_abbreviations': True},
                examples=["BG (not Gita)", "YS (not Yoga)", "MuU (not Mundaka)"]
            ),
            
            ValidationRule(
                rule_id="cite_003",
                category=ValidationCategory.CITATION_COMPLIANCE,
                description="Citation consistency throughout document",
                validation_function="_validate_citation_consistency",
                weight=1.3,
                severity="warning",
                required_for_standards=list(AcademicStandard),
                parameters={'consistency_threshold': 0.9}
            ),
            
            # Transliteration Accuracy Rules
            ValidationRule(
                rule_id="trans_001",
                category=ValidationCategory.TRANSLITERATION_ACCURACY,
                description="IAST transliteration standards compliance",
                validation_function="_validate_iast_compliance",
                weight=1.4,
                severity="error",
                required_for_standards=[AcademicStandard.PEER_REVIEW, AcademicStandard.JOURNAL_PUBLICATION],
                parameters={'iast_strict': True, 'mixed_standards_penalty': 0.5},
                examples=["dharma (not dharama)", "yoga (not yog)", "ṛṣi (not rishi)"]
            ),
            
            ValidationRule(
                rule_id="trans_002",
                category=ValidationCategory.TRANSLITERATION_ACCURACY,
                description="Consistent transliteration system usage",
                validation_function="_validate_transliteration_consistency",
                weight=1.2,
                severity="warning",
                required_for_standards=list(AcademicStandard),
                parameters={'allow_mixed': False}
            ),
            
            ValidationRule(
                rule_id="trans_003",
                category=ValidationCategory.TRANSLITERATION_ACCURACY,
                description="Proper handling of Sanskrit compounds and technical terms",
                validation_function="_validate_sanskrit_technical_terms",
                weight=1.1,
                severity="info",
                required_for_standards=[AcademicStandard.GRADUATE, AcademicStandard.DOCTORAL],
                parameters={'technical_terms_database': True}
            ),
            
            # Scholarly Rigor Rules
            ValidationRule(
                rule_id="rigor_001",
                category=ValidationCategory.SCHOLARLY_RIGOR,
                description="Appropriate use of academic language and terminology",
                validation_function="_validate_academic_language",
                weight=1.3,
                severity="warning",
                required_for_standards=[AcademicStandard.PEER_REVIEW, AcademicStandard.JOURNAL_PUBLICATION],
                parameters={'academic_vocabulary_threshold': 0.7}
            ),
            
            ValidationRule(
                rule_id="rigor_002",
                category=ValidationCategory.SCHOLARLY_RIGOR,
                description="Evidence-based claims with proper substantiation",
                validation_function="_validate_evidence_based_claims",
                weight=1.5,
                severity="error",
                required_for_standards=[AcademicStandard.DOCTORAL, AcademicStandard.PEER_REVIEW],
                parameters={'citation_density_minimum': 0.05}
            ),
            
            ValidationRule(
                rule_id="rigor_003",
                category=ValidationCategory.SCHOLARLY_RIGOR,
                description="Balanced perspective and acknowledgment of complexity",
                validation_function="_validate_balanced_perspective",
                weight=1.1,
                severity="info",
                required_for_standards=[AcademicStandard.GRADUATE, AcademicStandard.DOCTORAL],
                parameters={'perspective_indicators': ['however', 'although', 'while', 'whereas']}
            ),
            
            # Methodological Soundness Rules
            ValidationRule(
                rule_id="method_001",
                category=ValidationCategory.METHODOLOGICAL_SOUNDNESS,
                description="Clear methodology description and rationale",
                validation_function="_validate_methodology_clarity",
                weight=1.4,
                severity="error",
                required_for_standards=[AcademicStandard.PEER_REVIEW, AcademicStandard.JOURNAL_PUBLICATION],
                parameters={'methodology_keywords': ['approach', 'method', 'analysis', 'framework']}
            ),
            
            ValidationRule(
                rule_id="method_002",
                category=ValidationCategory.METHODOLOGICAL_SOUNDNESS,
                description="Appropriate sample size and scope for claims",
                validation_function="_validate_scope_appropriateness",
                weight=1.2,
                severity="warning",
                required_for_standards=[AcademicStandard.DOCTORAL, AcademicStandard.PEER_REVIEW],
                parameters={'scope_indicators': ['comprehensive', 'systematic', 'thorough']}
            ),
            
            # Academic Integrity Rules
            ValidationRule(
                rule_id="integrity_001",
                category=ValidationCategory.ACADEMIC_INTEGRITY,
                description="Proper attribution and acknowledgment of sources",
                validation_function="_validate_source_attribution",
                weight=1.6,
                severity="error",
                required_for_standards=list(AcademicStandard),
                parameters={'attribution_required': True}
            ),
            
            ValidationRule(
                rule_id="integrity_002",
                category=ValidationCategory.ACADEMIC_INTEGRITY,
                description="Originality and appropriate use of existing scholarship",
                validation_function="_validate_originality",
                weight=1.3,
                severity="warning",
                required_for_standards=[AcademicStandard.GRADUATE, AcademicStandard.DOCTORAL],
                parameters={'originality_threshold': 0.8}
            ),
            
            # Publication Readiness Rules
            ValidationRule(
                rule_id="pub_001",
                category=ValidationCategory.PUBLICATION_READINESS,
                description="Complete and properly formatted academic apparatus",
                validation_function="_validate_academic_apparatus",
                weight=1.4,
                severity="error",
                required_for_standards=[AcademicStandard.JOURNAL_PUBLICATION, AcademicStandard.BOOK_PUBLICATION],
                parameters={'required_sections': ['abstract', 'bibliography', 'citations']}
            ),
            
            ValidationRule(
                rule_id="pub_002",
                category=ValidationCategory.PUBLICATION_READINESS,
                description="Adherence to specific journal or publisher guidelines",
                validation_function="_validate_publication_guidelines",
                weight=1.2,
                severity="warning",
                required_for_standards=[AcademicStandard.JOURNAL_PUBLICATION],
                parameters={'guideline_compliance': True}
            )
        ]
        
        # Create rule lookup dictionary
        self.rules_by_id = {rule.rule_id: rule for rule in self.validation_rules}
        self.rules_by_category = {}
        for rule in self.validation_rules:
            if rule.category not in self.rules_by_category:
                self.rules_by_category[rule.category] = []
            self.rules_by_category[rule.category].append(rule)

    def _get_validation_categories(self) -> List[str]:
        """Get list of all validation categories for external validation."""
        return [
            "citation_accuracy",
            "transliteration_compliance", 
            "scholarly_rigor",
            "academic_formatting",
            "publication_readiness"
        ]

    # Validation category constants for external test validation
    VALIDATION_CATEGORIES = {
        "citation_accuracy": "Accuracy of citations and references",
        "transliteration_compliance": "Compliance with transliteration standards", 
        "scholarly_rigor": "Academic rigor and methodology",
        "academic_formatting": "Proper academic formatting and style",
        "publication_readiness": "Readiness for academic publication"
    }
    
    def _initialize_academic_standards(self) -> None:
        """Initialize academic standards and their requirements."""
        
        self.academic_standards = {
            AcademicStandard.UNDERGRADUATE: {
                'minimum_score': 0.60,
                'required_categories': [ValidationCategory.CITATION_COMPLIANCE],
                'critical_rules': ['cite_001', 'integrity_001'],
                'description': 'Basic academic writing standards for undergraduate work'
            },
            AcademicStandard.GRADUATE: {
                'minimum_score': 0.75,
                'required_categories': [
                    ValidationCategory.CITATION_COMPLIANCE,
                    ValidationCategory.SCHOLARLY_RIGOR,
                    ValidationCategory.ACADEMIC_INTEGRITY
                ],
                'critical_rules': ['cite_001', 'cite_002', 'rigor_001', 'integrity_001'],
                'description': 'Advanced academic standards for graduate-level work'
            },
            AcademicStandard.DOCTORAL: {
                'minimum_score': 0.85,
                'required_categories': list(ValidationCategory),
                'critical_rules': [
                    'cite_001', 'cite_002', 'trans_001', 'rigor_001', 'rigor_002',
                    'method_001', 'integrity_001', 'integrity_002'
                ],
                'description': 'Rigorous standards for doctoral dissertation quality'
            },
            AcademicStandard.PEER_REVIEW: {
                'minimum_score': 0.90,
                'required_categories': list(ValidationCategory),
                'critical_rules': [
                    'cite_001', 'trans_001', 'rigor_001', 'rigor_002',
                    'method_001', 'integrity_001', 'pub_001'
                ],
                'description': 'Peer review publication standards'
            },
            AcademicStandard.JOURNAL_PUBLICATION: {
                'minimum_score': 0.95,
                'required_categories': list(ValidationCategory),
                'critical_rules': [
                    'cite_001', 'cite_002', 'trans_001', 'rigor_001', 'rigor_002',
                    'method_001', 'method_002', 'integrity_001', 'integrity_002',
                    'pub_001', 'pub_002'
                ],
                'description': 'High-quality journal publication standards'
            },
            AcademicStandard.BOOK_PUBLICATION: {
                'minimum_score': 0.95,
                'required_categories': list(ValidationCategory),
                'critical_rules': [
                    'cite_001', 'cite_002', 'trans_001', 'rigor_001', 'rigor_002',
                    'method_001', 'method_002', 'integrity_001', 'integrity_002',
                    'pub_001'
                ],
                'description': 'Academic book publication standards'
            }
        }
    
    def _initialize_reference_materials(self) -> None:
        """Initialize reference materials for validation."""
        
        # Standard abbreviations for scriptural sources
        self.standard_abbreviations = {
            'bhagavad_gita': ['BG', 'Bhag.', 'Gītā'],
            'yoga_sutras': ['YS', 'Yoga.', 'Sūtras'],
            'upanishads': {
                'mundaka': ['MuU', 'Muṇḍ.'],
                'katha': ['KaU', 'Kaṭh.'],
                'kena': ['KeU', 'Kena'],
                'isha': ['ĪU', 'Īśā']
            }
        }
        
        # IAST character validation patterns
        self.iast_patterns = {
            'vowels': r'[aāiīuūṛṝḷḹeēoō]',
            'consonants': r'[kgṅcjñṭḍṇtdnpbmyrlvśṣsh]',
            'special_marks': r'[ṃḥ]',
            'complete_iast': r'[aāiīuūṛṝḷḹeēoōkgṅcjñṭḍṇtdnpbmyrlvśṣshṃḥ]'
        }
        
        # Academic vocabulary indicators
        self.academic_vocabulary = {
            'methodology_terms': [
                'analysis', 'approach', 'framework', 'methodology', 'systematic',
                'comprehensive', 'rigorous', 'empirical', 'theoretical'
            ],
            'scholarly_qualifiers': [
                'according to', 'as demonstrated by', 'evidence suggests',
                'research indicates', 'scholars argue', 'studies show'
            ],
            'critical_thinking': [
                'however', 'nevertheless', 'although', 'while', 'whereas',
                'on the other hand', 'conversely', 'in contrast'
            ]
        }
    
    def validate_academic_compliance(
        self,
        content: str,
        target_standard: AcademicStandard = AcademicStandard.PEER_REVIEW,
        citations: List[Any] = None,
        metadata: Dict[str, Any] = None
    ) -> ComprehensiveValidationReport:
        """
        Perform comprehensive academic compliance validation.
        
        Args:
            content: Text content to validate
            target_standard: Target academic standard
            citations: Optional list of citations to validate
            metadata: Optional metadata for validation context
            
        Returns:
            Comprehensive validation report
        """
        start_time = datetime.datetime.now()
        metadata = metadata or {}
        citations = citations or []
        
        report = ComprehensiveValidationReport(
            overall_compliance=ComplianceLevel.BASIC,
            content_analyzed={
                'content_length': len(content),
                'citations_count': len(citations),
                'target_standard': target_standard.value
            }
        )
        
        try:
            # Get applicable rules for target standard
            applicable_rules = self._get_applicable_rules(target_standard)
            report.total_rules_evaluated = len(applicable_rules)
            
            # Execute validation rules
            for rule in applicable_rules:
                try:
                    validation_result = self._execute_validation_rule(
                        rule, content, citations, metadata
                    )
                    report.validation_results.append(validation_result)
                    
                    if validation_result.passed:
                        report.rules_passed += 1
                    else:
                        report.rules_failed += 1
                        if validation_result.severity == "error":
                            report.critical_issues += 1
                        elif validation_result.severity == "warning":
                            report.warnings += 1
                    
                except Exception as e:
                    self.logger.error(f"Error executing rule {rule.rule_id}: {e}")
                    error_result = ValidationResult(
                        rule_id=rule.rule_id,
                        category=rule.category,
                        passed=False,
                        score=0.0,
                        message=f"Validation error: {str(e)}",
                        severity="error"
                    )
                    report.validation_results.append(error_result)
                    report.rules_failed += 1
                    report.critical_issues += 1
            
            # Calculate category scores
            self._calculate_category_scores(report)
            
            # Calculate overall compliance
            self._determine_overall_compliance(report, target_standard)
            
            # Generate recommendations
            self._generate_recommendations(report, target_standard)
            
            # Update performance statistics
            self._update_validation_statistics(report, target_standard)
            
            processing_time = (datetime.datetime.now() - start_time).total_seconds()
            self.logger.info(
                f"Academic validation completed: score={report.overall_score:.3f}, "
                f"compliance={report.overall_compliance.value}, "
                f"processing_time={processing_time:.3f}s"
            )
            
        except Exception as e:
            self.logger.error(f"Error in academic validation: {e}")
            report.improvement_priorities.append(f"Validation system error: {str(e)}")
        
        return report
    
    def _get_applicable_rules(self, target_standard: AcademicStandard) -> List[ValidationRule]:
        """Get validation rules applicable to the target academic standard."""
        
        applicable_rules = []
        
        for rule in self.validation_rules:
            # Include rule if it's required for this standard or is generally applicable
            if (not rule.required_for_standards or 
                target_standard in rule.required_for_standards or
                len(rule.required_for_standards) == 0):
                applicable_rules.append(rule)
        
        return applicable_rules
    
    def _execute_validation_rule(
        self,
        rule: ValidationRule,
        content: str,
        citations: List[Any],
        metadata: Dict[str, Any]
    ) -> ValidationResult:
        """Execute a specific validation rule."""
        
        # Get the validation method
        validation_method = getattr(self, rule.validation_function, None)
        
        if not validation_method:
            return ValidationResult(
                rule_id=rule.rule_id,
                category=rule.category,
                passed=False,
                score=0.0,
                message=f"Validation method {rule.validation_function} not found",
                severity="error"
            )
        
        # Execute validation
        try:
            result = validation_method(content, citations, rule.parameters, metadata)
            result.rule_id = rule.rule_id
            result.category = rule.category
            result.severity = rule.severity
            result.rule_parameters = rule.parameters
            
            return result
            
        except Exception as e:
            return ValidationResult(
                rule_id=rule.rule_id,
                category=rule.category,
                passed=False,
                score=0.0,
                message=f"Rule execution error: {str(e)}",
                severity="error"
            )
    
    # Validation Rule Implementations
    
    def _validate_citation_format(
        self,
        content: str,
        citations: List[Any],
        parameters: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> ValidationResult:
        """Validate citation format standards."""
        
        # Check for citation patterns in content
        citation_patterns = [
            r'\b[A-Z]{2,4}\s+\d+\.\d+\b',  # Standard format: BG 2.47
            r'\b\w+\.\s+\d+\.\d+\.\d+\b',  # Upanishad format: Muṇḍ. 3.2.9
            r'\([A-Z]{2,4}\s+\d+\.\d+\)',   # Parenthetical: (BG 2.47)
        ]
        
        found_citations = 0
        formatted_citations = 0
        
        for pattern in citation_patterns:
            matches = re.findall(pattern, content)
            found_citations += len(matches)
            
            # Check if citations are properly formatted
            for match in matches:
                if self._is_properly_formatted_citation(match):
                    formatted_citations += 1
        
        # Calculate score based on formatting quality
        if found_citations == 0:
            score = 0.5  # Neutral score for no citations
            message = "No citations found to validate formatting"
        else:
            score = formatted_citations / found_citations
            message = f"Citation formatting: {formatted_citations}/{found_citations} properly formatted"
        
        passed = score >= 0.8  # 80% threshold for proper formatting
        
        return ValidationResult(
            rule_id="",  # Will be set by caller
            category=ValidationCategory.CITATION_COMPLIANCE,
            passed=passed,
            score=score,
            message=message,
            suggestions=[
                "Use standard abbreviations (BG, YS, MuU)",
                "Format as 'Source Chapter.Verse'",
                "Ensure consistent citation style throughout"
            ] if not passed else []
        )
    
    def _validate_source_abbreviations(
        self,
        content: str,
        citations: List[Any],
        parameters: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> ValidationResult:
        """Validate use of standard source abbreviations."""
        
        # Extract potential abbreviations
        abbreviation_pattern = r'\b[A-Z]{2,4}(?=\s+\d+\.\d+)'
        found_abbreviations = re.findall(abbreviation_pattern, content)
        
        if not found_abbreviations:
            return ValidationResult(
                rule_id="",
                category=ValidationCategory.CITATION_COMPLIANCE,
                passed=True,
                score=1.0,
                message="No abbreviations found to validate"
            )
        
        # Check against standard abbreviations
        valid_abbreviations = []
        all_standard_abbrevs = []
        
        for source_type, abbrevs in self.standard_abbreviations.items():
            if isinstance(abbrevs, list):
                all_standard_abbrevs.extend(abbrevs)
            elif isinstance(abbrevs, dict):
                for subsource, sub_abbrevs in abbrevs.items():
                    all_standard_abbrevs.extend(sub_abbrevs)
        
        for abbrev in found_abbreviations:
            if any(std_abbrev.replace('.', '') == abbrev for std_abbrev in all_standard_abbrevs):
                valid_abbreviations.append(abbrev)
        
        score = len(valid_abbreviations) / len(found_abbreviations) if found_abbreviations else 1.0
        passed = score >= 0.9  # 90% threshold for abbreviation compliance
        
        return ValidationResult(
            rule_id="",
            category=ValidationCategory.CITATION_COMPLIANCE,
            passed=passed,
            score=score,
            message=f"Standard abbreviations: {len(valid_abbreviations)}/{len(found_abbreviations)} recognized",
            suggestions=[
                "Use BG for Bhagavad Gītā",
                "Use YS for Yoga Sūtras",
                "Use MuU for Muṇḍaka Upaniṣad",
                "Follow established scholarly abbreviation conventions"
            ] if not passed else []
        )
    
    def _validate_citation_consistency(
        self,
        content: str,
        citations: List[Any],
        parameters: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> ValidationResult:
        """Validate consistency of citation format throughout document."""
        
        # Extract all citations and analyze format consistency
        citation_patterns = re.findall(r'\b[A-Z]{2,4}\s+\d+\.\d+\b', content)
        
        if len(citation_patterns) < 2:
            return ValidationResult(
                rule_id="",
                category=ValidationCategory.CITATION_COMPLIANCE,
                passed=True,
                score=1.0,
                message="Insufficient citations to assess consistency"
            )
        
        # Analyze format patterns
        format_styles = set()
        for citation in citation_patterns:
            # Determine format style based on structure
            if ' ' in citation and '.' in citation:
                format_styles.add('standard_format')
            # Add more format analysis as needed
        
        # Check abbreviation consistency
        abbreviations_used = set()
        for citation in citation_patterns:
            abbrev = citation.split()[0] if ' ' in citation else citation
            abbreviations_used.add(abbrev)
        
        # Calculate consistency score
        format_consistency = 1.0 if len(format_styles) == 1 else 0.7
        abbrev_consistency = len(abbreviations_used) / len(citation_patterns)
        
        overall_score = (format_consistency + abbrev_consistency) / 2
        threshold = parameters.get('consistency_threshold', 0.9)
        passed = overall_score >= threshold
        
        return ValidationResult(
            rule_id="",
            category=ValidationCategory.CITATION_COMPLIANCE,
            passed=passed,
            score=overall_score,
            message=f"Citation consistency score: {overall_score:.2f}",
            suggestions=[
                "Maintain consistent citation format throughout document",
                "Use the same abbreviation for each source consistently",
                "Apply uniform spacing and punctuation in citations"
            ] if not passed else []
        )
    
    def _validate_iast_compliance(
        self,
        content: str,
        citations: List[Any],
        parameters: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> ValidationResult:
        """Validate IAST transliteration standards compliance."""
        
        # Check for IAST characters in content
        iast_chars = re.findall(self.iast_patterns['complete_iast'], content)
        total_chars = len(content)
        
        if not iast_chars:
            return ValidationResult(
                rule_id="",
                category=ValidationCategory.TRANSLITERATION_ACCURACY,
                passed=True,
                score=0.8,  # Neutral score when no transliteration present
                message="No IAST transliteration found in content"
            )
        
        # Check for mixed transliteration standards (penalty)
        mixed_standards_penalty = 0.0
        if parameters.get('iast_strict', False):
            # Check for non-IAST transliteration patterns
            non_iast_patterns = [
                r'\b\w*[AIURLM]\w*\b',  # ITRANS/Harvard-Kyoto patterns
                r'\b\w*[aeiou]h\w*\b'   # Simplified transliteration
            ]
            
            for pattern in non_iast_patterns:
                if re.search(pattern, content):
                    mixed_standards_penalty = parameters.get('mixed_standards_penalty', 0.3)
                    break
        
        # Calculate IAST density and quality
        iast_density = len(iast_chars) / max(total_chars, 1) * 100
        
        # Score based on appropriate usage
        if iast_density >= 0.5:  # Good IAST usage
            base_score = 0.9
        elif iast_density >= 0.1:  # Moderate IAST usage
            base_score = 0.7
        else:  # Minimal IAST usage
            base_score = 0.5
        
        final_score = max(0.0, base_score - mixed_standards_penalty)
        passed = final_score >= 0.7
        
        return ValidationResult(
            rule_id="",
            category=ValidationCategory.TRANSLITERATION_ACCURACY,
            passed=passed,
            score=final_score,
            message=f"IAST compliance: {final_score:.2f} (density: {iast_density:.3f}%)",
            suggestions=[
                "Use consistent IAST transliteration for Sanskrit terms",
                "Avoid mixing transliteration standards",
                "Include proper diacritical marks (ā, ī, ū, ṛ, etc.)",
                "Follow IAST conventions for consonant clusters"
            ] if not passed else []
        )
    
    def _validate_transliteration_consistency(
        self,
        content: str,
        citations: List[Any],
        parameters: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> ValidationResult:
        """Validate consistency of transliteration system usage."""
        
        # Detect different transliteration systems in use
        systems_detected = {
            'iast': bool(re.search(self.iast_patterns['complete_iast'], content)),
            'harvard_kyoto': bool(re.search(r'[AIURLM]', content)),
            'itrans': bool(re.search(r'[~\^]', content)),
            'simplified': bool(re.search(r'\b\w*[aeiou]h\w*\b', content))
        }
        
        systems_in_use = sum(systems_detected.values())
        
        if systems_in_use == 0:
            return ValidationResult(
                rule_id="",
                category=ValidationCategory.TRANSLITERATION_ACCURACY,
                passed=True,
                score=1.0,
                message="No transliteration systems detected"
            )
        
        elif systems_in_use == 1:
            # Single system - check which one
            primary_system = next(system for system, detected in systems_detected.items() if detected)
            score = 1.0 if primary_system == 'iast' else 0.8
            message = f"Consistent use of {primary_system} transliteration"
            passed = True
        else:
            # Mixed systems
            score = 0.4  # Penalty for mixed systems
            message = f"Mixed transliteration systems detected: {[s for s, d in systems_detected.items() if d]}"
            passed = parameters.get('allow_mixed', False)
        
        return ValidationResult(
            rule_id="",
            category=ValidationCategory.TRANSLITERATION_ACCURACY,
            passed=passed,
            score=score,
            message=message,
            suggestions=[
                "Use a single transliteration system throughout",
                "IAST is preferred for academic publications",
                "Avoid mixing different transliteration conventions",
                "Be consistent with diacritical mark usage"
            ] if not passed else []
        )
    
    def _validate_sanskrit_technical_terms(
        self,
        content: str,
        citations: List[Any],
        parameters: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> ValidationResult:
        """Validate proper handling of Sanskrit technical terms and compounds."""
        
        # Common Sanskrit technical terms that should be properly transliterated
        technical_terms = {
            'yoga': ['yoga', 'yog'],
            'dharma': ['dharma', 'dharama', 'dharm'],
            'karma': ['karma', 'karm'],
            'moksha': ['moksha', 'mokṣa', 'moksa'],
            'samadhi': ['samadhi', 'samādhi', 'samadhi'],
            'atman': ['atman', 'ātman', 'atma'],
            'brahman': ['brahman', 'brahmān', 'brahma']
        }
        
        total_terms_found = 0
        properly_handled_terms = 0
        
        content_lower = content.lower()
        
        for correct_form, variations in technical_terms.items():
            for variation in variations:
                pattern = rf'\b{re.escape(variation)}\b'
                matches = re.findall(pattern, content_lower)
                if matches:
                    total_terms_found += len(matches)
                    # Check if it's the preferred form (first in list is preferred)
                    if variation == variations[0]:
                        properly_handled_terms += len(matches)
        
        if total_terms_found == 0:
            return ValidationResult(
                rule_id="",
                category=ValidationCategory.TRANSLITERATION_ACCURACY,
                passed=True,
                score=1.0,
                message="No Sanskrit technical terms found"
            )
        
        score = properly_handled_terms / total_terms_found
        passed = score >= 0.7
        
        return ValidationResult(
            rule_id="",
            category=ValidationCategory.TRANSLITERATION_ACCURACY,
            passed=passed,
            score=score,
            message=f"Technical terms: {properly_handled_terms}/{total_terms_found} properly handled",
            suggestions=[
                "Use standard transliteration for technical terms",
                "Prefer 'yoga' over 'yog'",
                "Use 'dharma' consistently (not 'dharama')",
                "Include proper diacritical marks where appropriate"
            ] if not passed else []
        )
    
    def _validate_academic_language(
        self,
        content: str,
        citations: List[Any],
        parameters: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> ValidationResult:
        """Validate appropriate use of academic language and terminology."""
        
        content_lower = content.lower()
        total_words = len(content.split())
        
        # Count academic vocabulary usage
        academic_word_count = 0
        for category, words in self.academic_vocabulary.items():
            for word in words:
                academic_word_count += len(re.findall(rf'\b{re.escape(word)}\b', content_lower))
        
        # Calculate academic vocabulary density
        academic_density = academic_word_count / max(total_words, 1)
        threshold = parameters.get('academic_vocabulary_threshold', 0.05)
        
        # Check for informal language (penalty)
        informal_patterns = [
            r'\bkinda\b', r'\bsorta\b', r'\bthing\b', r'\bstuff\b',
            r'\breally\b', r'\bpretty\s+\w+', r'\bvery\s+\w+'
        ]
        
        informal_count = sum(len(re.findall(pattern, content_lower)) for pattern in informal_patterns)
        informal_penalty = min(0.3, informal_count / max(total_words, 1) * 10)
        
        final_score = max(0.0, min(1.0, academic_density * 10) - informal_penalty)
        passed = final_score >= 0.6
        
        return ValidationResult(
            rule_id="",
            category=ValidationCategory.SCHOLARLY_RIGOR,
            passed=passed,
            score=final_score,
            message=f"Academic language score: {final_score:.2f} (density: {academic_density:.3f})",
            suggestions=[
                "Use more precise academic terminology",
                "Avoid informal language and qualifiers",
                "Employ scholarly vocabulary and expressions",
                "Use disciplinary-specific terminology appropriately"
            ] if not passed else []
        )
    
    def _validate_evidence_based_claims(
        self,
        content: str,
        citations: List[Any],
        parameters: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> ValidationResult:
        """Validate that claims are properly substantiated with evidence."""
        
        # Identify claim-making patterns
        claim_patterns = [
            r'\b(demonstrates?|shows?|proves?|indicates?|suggests?|argues?)\s+that\b',
            r'\b(research|studies|evidence|data)\s+(shows?|indicates?|suggests?)\b',
            r'\b(clearly|obviously|certainly|definitely)\s+\w+',
            r'\bthis\s+(shows?|proves?|demonstrates?)\b'
        ]
        
        claims_found = 0
        for pattern in claim_patterns:
            claims_found += len(re.findall(pattern, content, re.IGNORECASE))
        
        # Count citations and references
        citation_count = len(citations) if citations else 0
        content_citations = len(re.findall(r'\b[A-Z]{2,4}\s+\d+\.\d+\b', content))
        total_citations = citation_count + content_citations
        
        # Calculate citation density
        words = len(content.split())
        citation_density = total_citations / max(words, 1)
        minimum_density = parameters.get('citation_density_minimum', 0.01)
        
        # Score based on substantiation
        if claims_found == 0:
            score = 0.8  # Neutral score for descriptive content
            message = "No explicit claims requiring substantiation found"
        else:
            substantiation_ratio = total_citations / max(claims_found, 1)
            density_score = min(1.0, citation_density / minimum_density)
            score = (substantiation_ratio * 0.6 + density_score * 0.4) * 0.8  # Cap at 0.8 for good practice
            message = f"Evidence support: {total_citations} citations for {claims_found} claims"
        
        passed = score >= 0.6
        
        return ValidationResult(
            rule_id="",
            category=ValidationCategory.SCHOLARLY_RIGOR,
            passed=passed,
            score=score,
            message=message,
            suggestions=[
                "Support claims with appropriate citations",
                "Provide evidence for assertions",
                "Use qualified language for tentative conclusions",
                "Include sufficient citations to support arguments"
            ] if not passed else []
        )
    
    def _validate_balanced_perspective(
        self,
        content: str,
        citations: List[Any],
        parameters: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> ValidationResult:
        """Validate balanced perspective and acknowledgment of complexity."""
        
        content_lower = content.lower()
        
        # Look for perspective indicators
        perspective_indicators = parameters.get('perspective_indicators', [
            'however', 'although', 'while', 'whereas', 'nevertheless',
            'on the other hand', 'conversely', 'in contrast', 'alternatively'
        ])
        
        nuance_count = 0
        for indicator in perspective_indicators:
            nuance_count += len(re.findall(rf'\b{re.escape(indicator)}\b', content_lower))
        
        # Look for absolute statements (penalty)
        absolute_patterns = [
            r'\b(always|never|all|none|every|completely|totally|absolutely)\b',
            r'\b(clearly|obviously|certainly|definitely|undoubtedly)\b'
        ]
        
        absolute_count = 0
        for pattern in absolute_patterns:
            absolute_count += len(re.findall(pattern, content_lower))
        
        # Calculate balance score
        words = len(content.split())
        nuance_density = nuance_count / max(words, 1) * 100
        absolute_penalty = min(0.4, absolute_count / max(words, 1) * 50)
        
        # Base score from nuance indicators
        if nuance_density >= 0.5:
            base_score = 0.9
        elif nuance_density >= 0.2:
            base_score = 0.7
        else:
            base_score = 0.5
        
        final_score = max(0.0, base_score - absolute_penalty)
        passed = final_score >= 0.6
        
        return ValidationResult(
            rule_id="",
            category=ValidationCategory.SCHOLARLY_RIGOR,
            passed=passed,
            score=final_score,
            message=f"Perspective balance: {final_score:.2f} (nuance: {nuance_count}, absolutes: {absolute_count})",
            suggestions=[
                "Acknowledge complexity and multiple perspectives",
                "Use qualifying language where appropriate",
                "Avoid absolute statements without strong evidence",
                "Include contrasting viewpoints where relevant"
            ] if not passed else []
        )
    
    def _validate_methodology_clarity(
        self,
        content: str,
        citations: List[Any],
        parameters: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> ValidationResult:
        """Validate clear methodology description and rationale."""
        
        content_lower = content.lower()
        
        # Look for methodology keywords
        methodology_keywords = parameters.get('methodology_keywords', [
            'approach', 'method', 'methodology', 'analysis', 'framework',
            'procedure', 'technique', 'systematic', 'process', 'strategy'
        ])
        
        methodology_mentions = 0
        for keyword in methodology_keywords:
            methodology_mentions += len(re.findall(rf'\b{re.escape(keyword)}\b', content_lower))
        
        # Look for methodology section indicators
        section_indicators = [
            r'\bmethodology\b', r'\bapproach\b', r'\bframework\b',
            r'\banalysis\s+method\b', r'\bresearch\s+design\b'
        ]
        
        section_count = 0
        for indicator in section_indicators:
            section_count += len(re.findall(indicator, content_lower))
        
        # Calculate methodology clarity score
        words = len(content.split())
        methodology_density = methodology_mentions / max(words, 1) * 100
        
        if section_count > 0 and methodology_density >= 0.5:
            score = 0.9
        elif methodology_density >= 0.3:
            score = 0.7
        elif methodology_mentions > 0:
            score = 0.5
        else:
            score = 0.2
        
        passed = score >= 0.6
        
        return ValidationResult(
            rule_id="",
            category=ValidationCategory.METHODOLOGICAL_SOUNDNESS,
            passed=passed,
            score=score,
            message=f"Methodology clarity: {score:.2f} ({methodology_mentions} methodology references)",
            suggestions=[
                "Include clear methodology section",
                "Describe analytical approach explicitly",
                "Explain rationale for chosen methods",
                "Use precise methodological terminology"
            ] if not passed else []
        )
    
    def _validate_scope_appropriateness(
        self,
        content: str,
        citations: List[Any],
        parameters: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> ValidationResult:
        """Validate appropriate sample size and scope for claims."""
        
        content_lower = content.lower()
        
        # Look for scope indicators
        scope_indicators = parameters.get('scope_indicators', [
            'comprehensive', 'systematic', 'thorough', 'extensive',
            'complete', 'exhaustive', 'representative'
        ])
        
        scope_mentions = 0
        for indicator in scope_indicators:
            scope_mentions += len(re.findall(rf'\b{re.escape(indicator)}\b', content_lower))
        
        # Look for limitation acknowledgments
        limitation_patterns = [
            r'\blimited?\s+to\b', r'\brestricted?\s+to\b',
            r'\bscope\s+of\s+this\b', r'\bwithin\s+the\s+scope\b',
            r'\blimitations?\s+of\b', r'\bconstraints?\s+of\b'
        ]
        
        limitation_count = 0
        for pattern in limitation_patterns:
            limitation_count += len(re.findall(pattern, content_lower))
        
        # Calculate appropriateness score
        words = len(content.split())
        scope_density = scope_mentions / max(words, 1) * 100
        limitation_awareness = limitation_count > 0
        
        # Score based on scope claims and limitation awareness
        if scope_density >= 0.3 and limitation_awareness:
            score = 0.9  # Good scope with limitations
        elif scope_density >= 0.3:
            score = 0.6  # Claims scope but no limitations
        elif limitation_awareness:
            score = 0.8  # Acknowledges limitations
        else:
            score = 0.7  # Neutral
        
        passed = score >= 0.6
        
        return ValidationResult(
            rule_id="",
            category=ValidationCategory.METHODOLOGICAL_SOUNDNESS,
            passed=passed,
            score=score,
            message=f"Scope appropriateness: {score:.2f} (scope claims: {scope_mentions}, limitations: {limitation_count})",
            suggestions=[
                "Clearly define scope and limitations",
                "Avoid overgeneralization beyond evidence",
                "Acknowledge methodological constraints",
                "Match claims to available evidence"
            ] if not passed else []
        )
    
    def _validate_source_attribution(
        self,
        content: str,
        citations: List[Any],
        parameters: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> ValidationResult:
        """Validate proper attribution and acknowledgment of sources."""
        
        # Count different types of attributions
        attribution_patterns = [
            r'\baccording\s+to\b',
            r'\bas\s+(stated|noted|argued|demonstrated)\s+by\b',
            r'\b\w+\s+(argues?|suggests?|demonstrates?|shows?)\b',
            r'\bin\s+the\s+words\s+of\b',
            r'\bas\s+\w+\s+puts?\s+it\b'
        ]
        
        attribution_count = 0
        for pattern in attribution_patterns:
            attribution_count += len(re.findall(pattern, content, re.IGNORECASE))
        
        # Count citations and references
        citation_count = len(citations) if citations else 0
        inline_citations = len(re.findall(r'\b[A-Z]{2,4}\s+\d+\.\d+\b', content))
        total_references = citation_count + inline_citations
        
        # Calculate attribution score
        words = len(content.split())
        
        if total_references == 0 and attribution_count == 0:
            # Original work with no external sources
            score = 0.8
            message = "No external sources requiring attribution"
        else:
            attribution_density = (attribution_count + total_references) / max(words, 1) * 100
            if attribution_density >= 1.0:
                score = 0.9
            elif attribution_density >= 0.5:
                score = 0.8
            elif total_references > 0:
                score = 0.6
            else:
                score = 0.3
            
            message = f"Source attribution: {attribution_count} attributions, {total_references} citations"
        
        passed = score >= 0.6
        
        return ValidationResult(
            rule_id="",
            category=ValidationCategory.ACADEMIC_INTEGRITY,
            passed=passed,
            score=score,
            message=message,
            suggestions=[
                "Properly attribute all sources and influences",
                "Use clear attribution language",
                "Include appropriate citations for all claims",
                "Acknowledge intellectual debts explicitly"
            ] if not passed else []
        )
    
    def _validate_originality(
        self,
        content: str,
        citations: List[Any],
        parameters: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> ValidationResult:
        """Validate originality and appropriate use of existing scholarship."""
        
        words = len(content.split())
        citation_count = len(citations) if citations else 0
        inline_citations = len(re.findall(r'\b[A-Z]{2,4}\s+\d+\.\d+\b', content))
        total_citations = citation_count + inline_citations
        
        # Calculate citation density
        citation_density = total_citations / max(words, 1)
        
        # Look for original analysis indicators
        original_analysis_patterns = [
            r'\bour\s+(analysis|approach|findings|research)\b',
            r'\bthis\s+(study|research|analysis)\s+(demonstrates?|shows?|reveals?)\b',
            r'\bwe\s+(propose|suggest|argue|demonstrate)\b',
            r'\bnew\s+(approach|perspective|understanding|insight)\b'
        ]
        
        original_indicators = 0
        for pattern in original_analysis_patterns:
            original_indicators += len(re.findall(pattern, content, re.IGNORECASE))
        
        # Score based on balance of citations vs. original content
        threshold = parameters.get('originality_threshold', 0.8)
        
        if citation_density > 0.05:  # High citation density - may indicate over-reliance
            base_score = 0.6
        elif citation_density > 0.02:  # Moderate citation density
            base_score = 0.8
        else:  # Low citation density
            base_score = 0.9
        
        # Boost for original analysis indicators
        originality_boost = min(0.2, original_indicators * 0.05)
        final_score = min(1.0, base_score + originality_boost)
        
        passed = final_score >= threshold
        
        return ValidationResult(
            rule_id="",
            category=ValidationCategory.ACADEMIC_INTEGRITY,
            passed=passed,
            score=final_score,
            message=f"Originality score: {final_score:.2f} (citations: {total_citations}, original indicators: {original_indicators})",
            suggestions=[
                "Balance citations with original analysis",
                "Clearly distinguish your contributions from existing work",
                "Develop original insights and interpretations",
                "Avoid over-reliance on existing scholarship"
            ] if not passed else []
        )
    
    def _validate_academic_apparatus(
        self,
        content: str,
        citations: List[Any],
        parameters: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> ValidationResult:
        """Validate complete and properly formatted academic apparatus."""
        
        required_sections = parameters.get('required_sections', [
            'abstract', 'bibliography', 'citations'
        ])
        
        # Check for presence of required sections
        section_indicators = {
            'abstract': [r'\babstract\b', r'\bsummary\b'],
            'bibliography': [r'\bbibliography\b', r'\breferences\b', r'\bworks\s+cited\b'],
            'citations': [r'\b[A-Z]{2,4}\s+\d+\.\d+\b'],
            'introduction': [r'\bintroduction\b'],
            'conclusion': [r'\bconclusion\b', r'\bconcluding\s+remarks\b'],
            'methodology': [r'\bmethodology\b', r'\bmethods?\b']
        }
        
        sections_found = {}
        content_lower = content.lower()
        
        for section, patterns in section_indicators.items():
            sections_found[section] = any(
                re.search(pattern, content_lower) for pattern in patterns
            )
        
        # Calculate completeness score
        required_found = sum(
            sections_found.get(section, False) for section in required_sections
        )
        completeness_score = required_found / len(required_sections) if required_sections else 1.0
        
        # Check additional academic elements
        additional_elements = {
            'footnotes': bool(re.search(r'\[\d+\]|\(\d+\)', content)),
            'page_numbers': bool(re.search(r'\bp\.\s*\d+|\bpp\.\s*\d+-\d+', content)),
            'proper_headings': bool(re.search(r'^#{1,3}\s+\w+', content, re.MULTILINE))
        }
        
        additional_score = sum(additional_elements.values()) / len(additional_elements)
        
        # Final score combines required sections and additional elements
        final_score = (completeness_score * 0.7 + additional_score * 0.3)
        passed = final_score >= 0.8
        
        missing_sections = [
            section for section in required_sections
            if not sections_found.get(section, False)
        ]
        
        return ValidationResult(
            rule_id="",
            category=ValidationCategory.PUBLICATION_READINESS,
            passed=passed,
            score=final_score,
            message=f"Academic apparatus: {final_score:.2f} ({required_found}/{len(required_sections)} required sections)",
            suggestions=[
                f"Include missing required sections: {', '.join(missing_sections)}" if missing_sections else "",
                "Ensure proper academic formatting throughout",
                "Include complete bibliography or references section",
                "Add appropriate footnotes where needed"
            ]
        )
    
    def _validate_publication_guidelines(
        self,
        content: str,
        citations: List[Any],
        parameters: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> ValidationResult:
        """Validate adherence to specific journal or publisher guidelines."""
        
        # This is a simplified implementation - in practice, this would check
        # against specific publisher style guides
        
        guideline_checks = {
            'word_count_appropriate': self._check_word_count_guidelines(content, metadata),
            'citation_style_consistent': self._check_citation_style_consistency(content),
            'abstract_length_appropriate': self._check_abstract_length(content),
            'keyword_section_present': self._check_keywords_present(content),
            'author_information_complete': self._check_author_information(metadata)
        }
        
        passed_checks = sum(guideline_checks.values())
        total_checks = len(guideline_checks)
        
        score = passed_checks / total_checks if total_checks > 0 else 1.0
        passed = score >= 0.8
        
        failed_guidelines = [
            check for check, passed_check in guideline_checks.items()
            if not passed_check
        ]
        
        return ValidationResult(
            rule_id="",
            category=ValidationCategory.PUBLICATION_READINESS,
            passed=passed,
            score=score,
            message=f"Publication guidelines: {passed_checks}/{total_checks} checks passed",
            suggestions=[
                f"Address guideline issues: {', '.join(failed_guidelines)}" if failed_guidelines else "",
                "Review target journal's submission guidelines",
                "Ensure compliance with formatting requirements",
                "Check word count and section requirements"
            ]
        )
    
    # Helper methods for publication guidelines validation
    
    def _check_word_count_guidelines(self, content: str, metadata: Dict[str, Any]) -> bool:
        """Check if word count is within typical guidelines."""
        word_count = len(content.split())
        min_words = metadata.get('min_word_count', 3000)
        max_words = metadata.get('max_word_count', 10000)
        return min_words <= word_count <= max_words
    
    def _check_citation_style_consistency(self, content: str) -> bool:
        """Check for consistent citation style."""
        citations = re.findall(r'\b[A-Z]{2,4}\s+\d+\.\d+\b', content)
        if len(citations) < 2:
            return True
        
        # Simple consistency check - all citations should follow same pattern
        patterns = set()
        for citation in citations:
            if ' ' in citation and '.' in citation:
                patterns.add('standard')
            else:
                patterns.add('non_standard')
        
        return len(patterns) == 1
    
    def _check_abstract_length(self, content: str) -> bool:
        """Check if abstract is appropriate length."""
        # Look for abstract section
        abstract_match = re.search(r'\babstract\b.*?(?=\n\n|\n[A-Z])', content, re.IGNORECASE | re.DOTALL)
        if not abstract_match:
            return True  # No abstract found, assume it's optional
        
        abstract_words = len(abstract_match.group().split())
        return 100 <= abstract_words <= 300
    
    def _check_keywords_present(self, content: str) -> bool:
        """Check if keywords section is present."""
        return bool(re.search(r'\bkeywords?\b', content, re.IGNORECASE))
    
    def _check_author_information(self, metadata: Dict[str, Any]) -> bool:
        """Check if author information is complete."""
        return bool(metadata.get('authors', []))
    
    def _is_properly_formatted_citation(self, citation: str) -> bool:
        """Check if a citation follows proper formatting."""
        # Basic format check: should be "ABC 1.23" or similar
        pattern = r'^[A-Z]{2,4}\s+\d+\.\d+$'
        return bool(re.match(pattern, citation.strip()))
    
    # Report Generation Methods
    
    def _calculate_category_scores(self, report: ComprehensiveValidationReport) -> None:
        """Calculate scores for each validation category."""
        
        category_results = {}
        for result in report.validation_results:
            if result.category not in category_results:
                category_results[result.category] = []
            category_results[result.category].append(result.score)
        
        for category, scores in category_results.items():
            report.category_scores[category] = sum(scores) / len(scores) if scores else 0.0
    
    def _determine_overall_compliance(
        self,
        report: ComprehensiveValidationReport,
        target_standard: AcademicStandard
    ) -> None:
        """Determine overall compliance level and score."""
        
        # Calculate weighted overall score
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for result in report.validation_results:
            rule = self.rules_by_id.get(result.rule_id)
            weight = rule.weight if rule else 1.0
            total_weighted_score += result.score * weight
            total_weight += weight
        
        report.overall_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Determine compliance level
        standard_requirements = self.academic_standards[target_standard]
        minimum_score = standard_requirements['minimum_score']
        
        if report.overall_score >= 0.95:
            report.overall_compliance = ComplianceLevel.EXEMPLARY
        elif report.overall_score >= 0.85:
            report.overall_compliance = ComplianceLevel.RIGOROUS
        elif report.overall_score >= minimum_score:
            report.overall_compliance = ComplianceLevel.STANDARD
        else:
            report.overall_compliance = ComplianceLevel.BASIC
        
        # Check if specific academic standards are met
        if report.overall_score >= minimum_score:
            report.academic_standard_met.append(target_standard)
        
        # Check other standards this content might meet
        for standard, requirements in self.academic_standards.items():
            if (standard != target_standard and 
                report.overall_score >= requirements['minimum_score']):
                report.academic_standard_met.append(standard)
    
    def _generate_recommendations(
        self,
        report: ComprehensiveValidationReport,
        target_standard: AcademicStandard
    ) -> None:
        """Generate specific recommendations for improvement."""
        
        # Priority improvements based on failed critical rules
        critical_failures = [
            result for result in report.validation_results
            if not result.passed and result.severity == "error"
        ]
        
        for failure in critical_failures:
            report.improvement_priorities.append(
                f"Critical: {failure.message}"
            )
        
        # Category-specific improvements
        standard_requirements = self.academic_standards[target_standard]
        required_categories = standard_requirements.get('required_categories', [])
        
        for category in required_categories:
            if category in report.category_scores:
                score = report.category_scores[category]
                if score < 0.7:
                    report.improvement_priorities.append(
                        f"Improve {category.value}: current score {score:.2f}"
                    )
        
        # Excellence recognition
        for category, score in report.category_scores.items():
            if score >= 0.90:
                report.excellence_areas.append(
                    f"Excellence in {category.value}: {score:.2f}"
                )
        
        # Next steps based on compliance level
        if report.overall_compliance == ComplianceLevel.BASIC:
            report.next_steps.extend([
                "Focus on critical error resolution",
                "Improve citation compliance",
                "Enhance academic language usage"
            ])
        elif report.overall_compliance == ComplianceLevel.STANDARD:
            report.next_steps.extend([
                "Strengthen scholarly rigor",
                "Improve transliteration consistency",
                "Enhance methodological clarity"
            ])
        elif report.overall_compliance == ComplianceLevel.RIGOROUS:
            report.next_steps.extend([
                "Polish for publication excellence",
                "Ensure all guidelines compliance",
                "Prepare for peer review"
            ])
    
    def _update_validation_statistics(
        self,
        report: ComprehensiveValidationReport,
        target_standard: AcademicStandard
    ) -> None:
        """Update validation performance statistics."""
        
        self.validation_stats['validations_performed'] += 1
        
        # Update average compliance score
        current_avg = self.validation_stats['average_compliance_score']
        total_validations = self.validation_stats['validations_performed']
        new_avg = ((current_avg * (total_validations - 1)) + report.overall_score) / total_validations
        self.validation_stats['average_compliance_score'] = new_avg
        
        # Update standards distribution
        standard_key = target_standard.value
        if standard_key not in self.validation_stats['standards_distribution']:
            self.validation_stats['standards_distribution'][standard_key] = 0
        self.validation_stats['standards_distribution'][standard_key] += 1
        
        # Track common issues
        for result in report.validation_results:
            if not result.passed:
                issue_key = f"{result.category.value}_{result.rule_id}"
                if issue_key not in self.validation_stats['common_issues']:
                    self.validation_stats['common_issues'][issue_key] = 0
                self.validation_stats['common_issues'][issue_key] += 1
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics."""
        
        return {
            'performance_stats': self.validation_stats.copy(),
            'supported_standards': [standard.value for standard in AcademicStandard],
            'validation_categories': [category.value for category in ValidationCategory],
            'total_validation_rules': len(self.validation_rules),
            'rules_by_category': {
                category.value: len(rules) 
                for category, rules in self.rules_by_category.items()
            },
            'configuration': self.config.copy()
        }

    
    def validate_citation_accuracy(
        self,
        citations: List[Any],
        content: str = "",
        validation_level: str = "standard"
    ) -> Dict[str, Any]:
        """
        Validate citation accuracy and compliance with academic standards.
        
        Args:
            citations: List of citations to validate
            content: Optional content context for validation
            validation_level: Validation rigor level
            
        Returns:
            Citation accuracy validation report
        """
        try:
            validation_report = {
                'overall_accuracy': 0.0,
                'citations_validated': len(citations),
                'accurate_citations': 0,
                'format_compliance': 0.0,
                'source_verification': 0.0,
                'consistency_score': 0.0,
                'issues_found': [],
                'recommendations': []
            }
            
            if not citations:
                validation_report['overall_accuracy'] = 1.0
                validation_report['format_compliance'] = 1.0
                validation_report['source_verification'] = 1.0
                validation_report['consistency_score'] = 1.0
                return validation_report
            
            # Validate each citation
            format_scores = []
            source_scores = []
            consistency_issues = []
            
            for i, citation in enumerate(citations):
                # Format validation
                format_result = self._validate_single_citation_format(citation)
                format_scores.append(format_result['score'])
                
                if not format_result['valid']:
                    validation_report['issues_found'].append(
                        f"Citation {i+1}: {format_result['issue']}"
                    )
                else:
                    validation_report['accurate_citations'] += 1
                
                # Source verification
                source_result = self._verify_citation_source(citation)
                source_scores.append(source_result['score'])
                
                if source_result['issues']:
                    validation_report['issues_found'].extend(source_result['issues'])
            
            # Calculate overall scores
            validation_report['format_compliance'] = sum(format_scores) / len(format_scores)
            validation_report['source_verification'] = sum(source_scores) / len(source_scores)
            
            # Consistency analysis
            validation_report['consistency_score'] = self._analyze_citation_consistency(citations)
            
            # Overall accuracy score
            validation_report['overall_accuracy'] = (
                validation_report['format_compliance'] * 0.4 +
                validation_report['source_verification'] * 0.3 +
                validation_report['consistency_score'] * 0.3
            )
            
            # Generate recommendations
            if validation_report['overall_accuracy'] < 0.80:
                validation_report['recommendations'].extend([
                    "Review citation formatting standards",
                    "Verify source accuracy and accessibility",
                    "Ensure consistent citation style throughout"
                ])
            
            if validation_report['format_compliance'] < 0.70:
                validation_report['recommendations'].append(
                    "Focus on proper citation format compliance"
                )
            
            if validation_report['consistency_score'] < 0.80:
                validation_report['recommendations'].append(
                    "Standardize citation format across all references"
                )
            
            return validation_report
            
        except Exception as e:
            self.logger.error(f"Error in citation accuracy validation: {e}")
            return {
                'overall_accuracy': 0.0,
                'citations_validated': len(citations),
                'error': str(e)
            }
    
    def validate_transliteration_standards(
        self,
        content: str,
        target_standard: str = "iast",
        strict_mode: bool = True
    ) -> Dict[str, Any]:
        """
        Validate transliteration standards compliance.
        
        Args:
            content: Text content to validate
            target_standard: Target transliteration standard (iast, harvard_kyoto, etc.)
            strict_mode: Whether to apply strict validation rules
            
        Returns:
            Transliteration standards validation report
        """
        try:
            validation_report = {
                'overall_compliance': 0.0,
                'target_standard': target_standard,
                'strict_mode': strict_mode,
                'character_compliance': 0.0,
                'consistency_score': 0.0,
                'mixed_standards_detected': False,
                'standard_violations': [],
                'character_issues': [],
                'recommendations': []
            }
            
            if not content:
                validation_report['overall_compliance'] = 1.0
                return validation_report
            
            # Analyze transliteration character usage
            char_analysis = self._analyze_transliteration_characters(content, target_standard)
            validation_report['character_compliance'] = char_analysis['compliance_score']
            validation_report['character_issues'] = char_analysis['issues']
            
            # Check for mixed standards
            mixed_analysis = self._detect_mixed_transliteration_standards(content)
            validation_report['mixed_standards_detected'] = mixed_analysis['mixed_detected']
            validation_report['consistency_score'] = mixed_analysis['consistency_score']
            
            if mixed_analysis['mixed_detected']:
                validation_report['standard_violations'].extend(mixed_analysis['violations'])
            
            # Calculate overall compliance
            if validation_report['mixed_standards_detected'] and strict_mode:
                penalty = 0.3
            else:
                penalty = 0.0
            
            validation_report['overall_compliance'] = max(0.0, (
                validation_report['character_compliance'] * 0.6 +
                validation_report['consistency_score'] * 0.4 - penalty
            ))
            
            # Generate recommendations
            if validation_report['overall_compliance'] < 0.80:
                validation_report['recommendations'].append(
                    f"Improve compliance with {target_standard.upper()} transliteration standards"
                )
            
            if validation_report['mixed_standards_detected']:
                validation_report['recommendations'].append(
                    "Use consistent transliteration standard throughout document"
                )
            
            if validation_report['character_compliance'] < 0.70:
                validation_report['recommendations'].append(
                    "Review and correct transliteration character usage"
                )
            
            # Add specific character recommendations
            if target_standard.lower() == 'iast':
                validation_report['recommendations'].extend([
                    "Use proper IAST diacritical marks (ā, ī, ū, ṛ, etc.)",
                    "Ensure correct representation of Sanskrit consonant clusters",
                    "Follow IAST conventions for vowel and consonant transliteration"
                ])
            
            return validation_report
            
        except Exception as e:
            self.logger.error(f"Error in transliteration standards validation: {e}")
            return {
                'overall_compliance': 0.0,
                'target_standard': target_standard,
                'error': str(e)
            }
    
    def validate_scholarly_rigor(
        self,
        content: str,
        citations: List[Any] = None,
        target_level: str = "graduate"
    ) -> Dict[str, Any]:
        """
        Validate scholarly rigor and academic standards.
        
        Args:
            content: Text content to validate
            citations: Optional list of citations for context
            target_level: Target academic level (undergraduate, graduate, doctoral)
            
        Returns:
            Scholarly rigor validation report
        """
        try:
            citations = citations or []
            validation_report = {
                'overall_rigor_score': 0.0,
                'target_level': target_level,
                'academic_language_score': 0.0,
                'evidence_support_score': 0.0,
                'methodology_clarity_score': 0.0,
                'critical_thinking_score': 0.0,
                'originality_score': 0.0,
                'rigor_indicators': [],
                'weaknesses_identified': [],
                'recommendations': []
            }
            
            if not content:
                return validation_report
            
            # Assess academic language usage
            lang_analysis = self._assess_academic_language_usage(content)
            validation_report['academic_language_score'] = lang_analysis['score']
            validation_report['rigor_indicators'].extend(lang_analysis['indicators'])
            
            # Evaluate evidence support
            evidence_analysis = self._evaluate_evidence_support(content, citations)
            validation_report['evidence_support_score'] = evidence_analysis['score']
            
            if evidence_analysis['score'] < 0.60:
                validation_report['weaknesses_identified'].append(
                    "Insufficient evidence support for claims"
                )
            
            # Assess methodology clarity
            method_analysis = self._assess_methodology_presentation(content)
            validation_report['methodology_clarity_score'] = method_analysis['score']
            
            if method_analysis['score'] < 0.50:
                validation_report['weaknesses_identified'].append(
                    "Unclear or missing methodology description"
                )
            
            # Evaluate critical thinking
            critical_analysis = self._evaluate_critical_thinking(content)
            validation_report['critical_thinking_score'] = critical_analysis['score']
            validation_report['rigor_indicators'].extend(critical_analysis['indicators'])
            
            # Assess originality
            originality_analysis = self._assess_content_originality(content, citations)
            validation_report['originality_score'] = originality_analysis['score']
            
            # Calculate overall rigor score based on target level
            level_weights = self._get_rigor_weights_for_level(target_level)
            
            validation_report['overall_rigor_score'] = (
                validation_report['academic_language_score'] * level_weights['language'] +
                validation_report['evidence_support_score'] * level_weights['evidence'] +
                validation_report['methodology_clarity_score'] * level_weights['methodology'] +
                validation_report['critical_thinking_score'] * level_weights['critical_thinking'] +
                validation_report['originality_score'] * level_weights['originality']
            )
            
            # Generate recommendations based on weaknesses
            target_thresholds = self._get_rigor_thresholds_for_level(target_level)
            
            for component, score in [
                ('academic_language', validation_report['academic_language_score']),
                ('evidence_support', validation_report['evidence_support_score']),
                ('methodology_clarity', validation_report['methodology_clarity_score']),
                ('critical_thinking', validation_report['critical_thinking_score']),
                ('originality', validation_report['originality_score'])
            ]:
                threshold = target_thresholds.get(component, 0.70)
                if score < threshold:
                    validation_report['recommendations'].append(
                        f"Improve {component.replace('_', ' ')}: current {score:.2f}, target {threshold:.2f}"
                    )
            
            # Add level-specific recommendations
            if target_level == "doctoral" and validation_report['overall_rigor_score'] < 0.85:
                validation_report['recommendations'].extend([
                    "Enhance theoretical framework development",
                    "Strengthen methodological sophistication",
                    "Increase depth of critical analysis"
                ])
            elif target_level == "graduate" and validation_report['overall_rigor_score'] < 0.75:
                validation_report['recommendations'].extend([
                    "Develop more sophisticated argumentation",
                    "Integrate scholarly perspectives more effectively",
                    "Enhance analytical depth"
                ])
            
            return validation_report
            
        except Exception as e:
            self.logger.error(f"Error in scholarly rigor validation: {e}")
            return {
                'overall_rigor_score': 0.0,
                'target_level': target_level,
                'error': str(e)
            }
    
    def generate_comprehensive_report(
        self,
        validation_results: List[Any],
        target_standard: str = "peer_review",
        include_detailed_analysis: bool = True
    ) -> str:
        """
        Generate comprehensive validation report from validation results.
        
        Args:
            validation_results: List of validation results
            target_standard: Target academic standard
            include_detailed_analysis: Whether to include detailed analysis sections
            
        Returns:
            Formatted comprehensive validation report
        """
        try:
            # Generate report header
            report_lines = [
                "# Comprehensive Academic Validation Report",
                "",
                f"**Generated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Target Standard**: {target_standard.title()}",
                f"**Validation Rules Evaluated**: {len(validation_results)}",
                ""
            ]
            
            # Calculate summary statistics
            if validation_results:
                passed_count = sum(1 for result in validation_results if result.passed)
                total_count = len(validation_results)
                overall_score = sum(result.score for result in validation_results) / total_count
                
                report_lines.extend([
                    "## Executive Summary",
                    "",
                    f"- **Overall Validation Score**: {overall_score:.3f}",
                    f"- **Rules Passed**: {passed_count}/{total_count} ({passed_count/total_count*100:.1f}%)",
                    f"- **Compliance Level**: {self._determine_compliance_level(overall_score)}",
                    ""
                ])
            else:
                report_lines.extend([
                    "## Executive Summary",
                    "",
                    "No validation results available for analysis.",
                    ""
                ])
                return "\n".join(report_lines)
            
            # Categorize results by validation category
            categorized_results = {}
            for result in validation_results:
                category = result.category.value if hasattr(result.category, 'value') else str(result.category)
                if category not in categorized_results:
                    categorized_results[category] = []
                categorized_results[category].append(result)
            
            # Generate category-specific analysis
            report_lines.extend([
                "## Validation Results by Category",
                ""
            ])
            
            for category, category_results in categorized_results.items():
                category_passed = sum(1 for r in category_results if r.passed)
                category_total = len(category_results)
                category_score = sum(r.score for r in category_results) / category_total
                
                report_lines.extend([
                    f"### {category.title().replace('_', ' ')}",
                    "",
                    f"- **Category Score**: {category_score:.3f}",
                    f"- **Rules Passed**: {category_passed}/{category_total}",
                    ""
                ])
                
                if include_detailed_analysis:
                    # Failed rules in this category
                    failed_rules = [r for r in category_results if not r.passed]
                    if failed_rules:
                        report_lines.extend([
                            "**Issues Identified:**",
                            ""
                        ])
                        for rule in failed_rules:
                            severity_indicator = "🔴" if rule.severity == "error" else "🟡"
                            report_lines.append(f"- {severity_indicator} {rule.message}")
                        report_lines.append("")
                    
                    # Excellent performance in this category
                    excellent_rules = [r for r in category_results if r.passed and r.score >= 0.90]
                    if excellent_rules:
                        report_lines.extend([
                            "**Excellence Indicators:**",
                            ""
                        ])
                        for rule in excellent_rules:
                            report_lines.append(f"- ✅ {rule.message}")
                        report_lines.append("")
            
            # Critical issues section
            critical_issues = [r for r in validation_results if not r.passed and r.severity == "error"]
            if critical_issues:
                report_lines.extend([
                    "## Critical Issues Requiring Immediate Attention",
                    ""
                ])
                for i, issue in enumerate(critical_issues, 1):
                    report_lines.extend([
                        f"### {i}. {issue.rule_id}: {issue.message}",
                        ""
                    ])
                    if hasattr(issue, 'suggestions') and issue.suggestions:
                        report_lines.extend([
                            "**Recommendations:**",
                            ""
                        ])
                        for suggestion in issue.suggestions:
                            if suggestion.strip():  # Only add non-empty suggestions
                                report_lines.append(f"- {suggestion}")
                        report_lines.append("")
            
            # Excellence areas
            excellence_areas = [r for r in validation_results if r.passed and r.score >= 0.90]
            if excellence_areas:
                report_lines.extend([
                    "## Areas of Excellence",
                    ""
                ])
                for area in excellence_areas:
                    report_lines.append(f"- **{area.rule_id}**: {area.message} (Score: {area.score:.3f})")
                report_lines.append("")
            
            # Overall recommendations
            all_suggestions = []
            for result in validation_results:
                if hasattr(result, 'suggestions') and result.suggestions:
                    all_suggestions.extend([s for s in result.suggestions if s.strip()])
            
            if all_suggestions:
                # Deduplicate suggestions
                unique_suggestions = list(set(all_suggestions))
                
                report_lines.extend([
                    "## Recommendations for Improvement",
                    ""
                ])
                for suggestion in unique_suggestions[:10]:  # Limit to top 10 recommendations
                    report_lines.append(f"- {suggestion}")
                report_lines.append("")
            
            # Publication readiness assessment
            report_lines.extend([
                "## Publication Readiness Assessment",
                ""
            ])
            
            if overall_score >= 0.95:
                readiness_level = "Excellent - Ready for top-tier publication"
            elif overall_score >= 0.85:
                readiness_level = "Good - Ready for peer review with minor revisions"
            elif overall_score >= 0.75:
                readiness_level = "Acceptable - Requires moderate revision before submission"
            elif overall_score >= 0.60:
                readiness_level = "Needs Improvement - Significant revision required"
            else:
                readiness_level = "Not Ready - Major improvements needed before publication consideration"
            
            report_lines.extend([
                f"**Assessment**: {readiness_level}",
                "",
                f"**Next Steps**: ",
                ""
            ])
            
            if overall_score >= 0.85:
                report_lines.extend([
                    "- Final review and polishing",
                    "- Preparation for submission",
                    "- Consider consultant review for final validation"
                ])
            elif overall_score >= 0.70:
                report_lines.extend([
                    "- Address identified critical issues",
                    "- Strengthen weaker validation categories",
                    "- Conduct additional review cycle"
                ])
            else:
                report_lines.extend([
                    "- Focus on fundamental improvements",
                    "- Address all critical validation failures",
                    "- Consider comprehensive revision strategy"
                ])
            
            report_lines.extend([
                "",
                "---",
                "",
                f"*Report generated by Academic Validator v1.0 on {datetime.datetime.now().strftime('%Y-%m-%d')}*"
            ])
            
            return "\n".join(report_lines)
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {e}")
            return f"# Report Generation Error\n\nError occurred while generating report: {str(e)}"
    
    # Helper methods for the new validation functions
    
    def _validate_single_citation_format(self, citation: Any) -> Dict[str, Any]:
        """Validate format of a single citation."""
        result = {'valid': True, 'score': 1.0, 'issue': ''}
        
        try:
            # Extract citation text for analysis
            if hasattr(citation, 'citation_text'):
                citation_text = citation.citation_text
            elif isinstance(citation, str):
                citation_text = citation
            else:
                citation_text = str(citation)
            
            # Check for standard citation pattern
            if not re.match(r'^[A-Z]{2,4}\s+\d+\.\d+', citation_text.strip()):
                result['valid'] = False
                result['score'] = 0.3
                result['issue'] = 'Citation does not follow standard format (Source Chapter.Verse)'
            
            return result
            
        except Exception as e:
            return {'valid': False, 'score': 0.0, 'issue': f'Citation validation error: {str(e)}'}
    
    def _verify_citation_source(self, citation: Any) -> Dict[str, Any]:
        """Verify citation source accuracy."""
        result = {'score': 1.0, 'issues': []}
        
        try:
            # This is a simplified implementation - in practice would check against
            # authoritative source databases
            if hasattr(citation, 'verse_candidate') and citation.verse_candidate:
                # Basic verification of source availability
                source = citation.verse_candidate.source
                if source and hasattr(source, 'value'):
                    # Check if source is in recognized list
                    recognized_sources = ['BHAGAVAD_GITA', 'YOGA_SUTRAS', 'MUNDAKA_UPANISHAD']
                    if source.value in recognized_sources:
                        result['score'] = 1.0
                    else:
                        result['score'] = 0.8
                        result['issues'].append(f"Source {source.value} requires verification")
                else:
                    result['score'] = 0.5
                    result['issues'].append("Source information incomplete")
            else:
                result['score'] = 0.7
                result['issues'].append("Unable to verify source - limited citation metadata")
            
            return result
            
        except Exception as e:
            return {'score': 0.0, 'issues': [f'Source verification error: {str(e)}']}
    
    def _analyze_citation_consistency(self, citations: List[Any]) -> float:
        """Analyze consistency across all citations."""
        if len(citations) < 2:
            return 1.0
        
        try:
            # Analyze format consistency
            formats = []
            for citation in citations:
                if hasattr(citation, 'citation_text'):
                    citation_text = citation.citation_text
                    # Simple format analysis
                    if re.match(r'^[A-Z]{2,4}\s+\d+\.\d+', citation_text):
                        formats.append('standard')
                    else:
                        formats.append('non_standard')
            
            if not formats:
                return 0.5
            
            # Calculate consistency
            most_common_format = max(set(formats), key=formats.count)
            consistency_ratio = formats.count(most_common_format) / len(formats)
            
            return consistency_ratio
            
        except Exception:
            return 0.5
    
    def _analyze_transliteration_characters(self, content: str, target_standard: str) -> Dict[str, Any]:
        """Analyze transliteration character compliance."""
        analysis = {'compliance_score': 1.0, 'issues': []}
        
        try:
            if target_standard.lower() == 'iast':
                # Check for IAST-specific characters
                iast_chars = re.findall(self.iast_patterns['complete_iast'], content)
                total_sanskrit_words = len(re.findall(r'\b[a-zA-Zāīūṛḷēōṃḥśṣṇṭḍṅñ]+\b', content))
                
                if total_sanskrit_words > 0:
                    iast_density = len(iast_chars) / total_sanskrit_words
                    analysis['compliance_score'] = min(1.0, iast_density * 2)  # Scale appropriately
                    
                    if iast_density < 0.5:
                        analysis['issues'].append("Low IAST character usage for Sanskrit content")
                else:
                    analysis['compliance_score'] = 1.0  # No Sanskrit content to transliterate
            
            return analysis
            
        except Exception as e:
            return {'compliance_score': 0.0, 'issues': [f'Character analysis error: {str(e)}']}
    
    def _detect_mixed_transliteration_standards(self, content: str) -> Dict[str, Any]:
        """Detect mixed transliteration standards in content."""
        analysis = {
            'mixed_detected': False,
            'consistency_score': 1.0,
            'violations': []
        }
        
        try:
            # Detect different transliteration patterns
            standards_found = {
                'iast': bool(re.search(self.iast_patterns['complete_iast'], content)),
                'harvard_kyoto': bool(re.search(r'[AIURLM]', content)),
                'itrans': bool(re.search(r'[~\^]', content))
            }
            
            active_standards = sum(standards_found.values())
            
            if active_standards > 1:
                analysis['mixed_detected'] = True
                analysis['consistency_score'] = 0.4
                analysis['violations'].append("Multiple transliteration standards detected")
            
            return analysis
            
        except Exception:
            return analysis
    
    def _assess_academic_language_usage(self, content: str) -> Dict[str, Any]:
        """Assess academic language usage quality."""
        analysis = {'score': 0.5, 'indicators': []}
        
        try:
            content_lower = content.lower()
            word_count = len(content.split())
            
            # Count academic vocabulary
            academic_count = 0
            for category, terms in self.academic_vocabulary.items():
                for term in terms:
                    academic_count += len(re.findall(rf'\b{re.escape(term)}\b', content_lower))
            
            if word_count > 0:
                academic_density = academic_count / word_count
                analysis['score'] = min(1.0, academic_density * 10)  # Scale appropriately
                
                if academic_density >= 0.1:
                    analysis['indicators'].append("Strong academic vocabulary usage")
                elif academic_density >= 0.05:
                    analysis['indicators'].append("Moderate academic language")
            
            return analysis
            
        except Exception:
            return analysis
    
    def _evaluate_evidence_support(self, content: str, citations: List[Any]) -> Dict[str, Any]:
        """Evaluate evidence support quality."""
        analysis = {'score': 0.5}
        
        try:
            # Simple analysis based on citation density and claim patterns
            word_count = len(content.split())
            citation_count = len(citations)
            
            # Look for claim patterns
            claim_patterns = [
                r'\b(shows?|proves?|demonstrates?|indicates?)\s+that\b',
                r'\b(research|evidence|studies)\s+(shows?|indicates?)\b'
            ]
            
            claim_count = 0
            for pattern in claim_patterns:
                claim_count += len(re.findall(pattern, content, re.IGNORECASE))
            
            if claim_count > 0:
                support_ratio = citation_count / claim_count
                analysis['score'] = min(1.0, support_ratio * 0.8)
            elif citation_count > 0:
                # Has citations but no explicit claims
                analysis['score'] = 0.7
            
            return analysis
            
        except Exception:
            return analysis
    
    def _assess_methodology_presentation(self, content: str) -> Dict[str, Any]:
        """Assess methodology presentation quality."""
        analysis = {'score': 0.5}
        
        try:
            content_lower = content.lower()
            
            # Look for methodology indicators
            methodology_terms = [
                'methodology', 'approach', 'method', 'framework',
                'analysis', 'procedure', 'technique', 'systematic'
            ]
            
            method_count = 0
            for term in methodology_terms:
                method_count += len(re.findall(rf'\b{re.escape(term)}\b', content_lower))
            
            word_count = len(content.split())
            if word_count > 0:
                method_density = method_count / word_count
                analysis['score'] = min(1.0, method_density * 20)  # Scale appropriately
            
            return analysis
            
        except Exception:
            return analysis
    
    def _evaluate_critical_thinking(self, content: str) -> Dict[str, Any]:
        """Evaluate critical thinking indicators."""
        analysis = {'score': 0.5, 'indicators': []}
        
        try:
            content_lower = content.lower()
            
            # Look for critical thinking indicators
            critical_terms = [
                'however', 'although', 'while', 'whereas', 'nevertheless',
                'on the other hand', 'conversely', 'in contrast'
            ]
            
            critical_count = 0
            for term in critical_terms:
                if term in content_lower:
                    critical_count += 1
                    analysis['indicators'].append(f"Uses qualifier: '{term}'")
            
            # Simple scoring based on presence of critical thinking language
            analysis['score'] = min(1.0, critical_count / 10.0 + 0.5)
            
            return analysis
            
        except Exception:
            return analysis
    
    def _assess_content_originality(self, content: str, citations: List[Any]) -> Dict[str, Any]:
        """Assess content originality."""
        analysis = {'score': 0.7}
        
        try:
            word_count = len(content.split())
            citation_count = len(citations)
            
            # Simple originality assessment based on citation density
            if word_count > 0:
                citation_density = citation_count / word_count
                
                # Lower citation density suggests more original content
                if citation_density < 0.02:
                    analysis['score'] = 0.9
                elif citation_density < 0.05:
                    analysis['score'] = 0.8
                else:
                    analysis['score'] = 0.6
            
            return analysis
            
        except Exception:
            return analysis
    
    def _get_rigor_weights_for_level(self, level: str) -> Dict[str, float]:
        """Get rigor assessment weights for academic level."""
        weights = {
            'undergraduate': {
                'language': 0.3, 'evidence': 0.3, 'methodology': 0.2,
                'critical_thinking': 0.1, 'originality': 0.1
            },
            'graduate': {
                'language': 0.25, 'evidence': 0.25, 'methodology': 0.25,
                'critical_thinking': 0.15, 'originality': 0.1
            },
            'doctoral': {
                'language': 0.2, 'evidence': 0.2, 'methodology': 0.3,
                'critical_thinking': 0.15, 'originality': 0.15
            }
        }
        return weights.get(level, weights['graduate'])
    
    def _get_rigor_thresholds_for_level(self, level: str) -> Dict[str, float]:
        """Get rigor thresholds for academic level."""
        thresholds = {
            'undergraduate': {
                'academic_language': 0.60, 'evidence_support': 0.60,
                'methodology_clarity': 0.50, 'critical_thinking': 0.50,
                'originality': 0.60
            },
            'graduate': {
                'academic_language': 0.70, 'evidence_support': 0.70,
                'methodology_clarity': 0.65, 'critical_thinking': 0.60,
                'originality': 0.70
            },
            'doctoral': {
                'academic_language': 0.80, 'evidence_support': 0.80,
                'methodology_clarity': 0.80, 'critical_thinking': 0.75,
                'originality': 0.80
            }
        }
        return thresholds.get(level, thresholds['graduate'])
    
    def _determine_compliance_level(self, score: float) -> str:
        """Determine compliance level from score."""
        if score >= 0.95:
            return "Exemplary"
        elif score >= 0.85:
            return "Rigorous"
        elif score >= 0.75:
            return "Standard"
        elif score >= 0.60:
            return "Basic"
        else:
            return "Below Standard"


def create_academic_validator(
    target_standard: AcademicStandard = AcademicStandard.PEER_REVIEW,
    custom_config: Dict[str, Any] = None
) -> AcademicValidator:
    """
    Factory function to create an AcademicValidator with specified configuration.
    
    Args:
        target_standard: Default target academic standard
        custom_config: Optional custom configuration
        
    Returns:
        Configured AcademicValidator instance
    """
    config = custom_config or {}
    config['default_target_standard'] = target_standard.value
    
    return AcademicValidator(config=config)