"""
Academic Compliance Validator for Story 3.6
Enhanced academic standard compliance validation
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import logging
from pathlib import Path

from post_processors.academic_polish_processor import PolishIssue


class ComplianceLevel(Enum):
    """Academic compliance levels"""
    EXCELLENT = "excellent"          # 95%+
    GOOD = "good"                   # 85-94%
    ADEQUATE = "adequate"           # 75-84%
    NEEDS_IMPROVEMENT = "needs_improvement"  # <75%


@dataclass
class ComplianceRule:
    """Individual compliance rule definition"""
    rule_id: str
    category: str
    description: str
    pattern: Optional[str] = None
    validator_func: Optional[callable] = None
    weight: float = 1.0
    critical: bool = False
    academic_standard: str = ""


@dataclass
class ComplianceViolation:
    """Represents a compliance violation"""
    rule_id: str
    line_number: int
    violation_type: str
    description: str
    original_text: str
    suggested_correction: str
    severity: str
    academic_standard: str
    confidence: float = 1.0


@dataclass
class ComplianceReport:
    """Comprehensive compliance assessment report"""
    overall_compliance_score: float
    compliance_level: ComplianceLevel
    category_scores: Dict[str, float]
    violations: List[ComplianceViolation]
    recommendations: List[str]
    academic_standards_met: Dict[str, bool]
    total_rules_checked: int
    rules_passed: int


class AcademicComplianceValidator:
    """
    Enhanced academic standard compliance validator
    Validates content against comprehensive academic standards
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the academic compliance validator"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize compliance rules
        self.compliance_rules = self._initialize_compliance_rules()
        
        # Academic standards configuration
        self.standards_config = self.config.get('academic_standards', {
            'iast_strict_mode': True,
            'require_diacriticals': True,
            'enforce_capitalization': True,
            'validate_terminology_consistency': True,
            'check_spiritual_respectfulness': True
        })
        
        self.logger.info(f"Academic Compliance Validator initialized with {len(self.compliance_rules)} rules")
    
    def validate_academic_compliance(self, 
                                   content: str,
                                   context: Optional[Dict[str, Any]] = None) -> ComplianceReport:
        """
        Validate content against academic compliance standards
        
        Args:
            content: Content to validate
            context: Additional validation context
            
        Returns:
            Comprehensive compliance report
        """
        
        context = context or {}
        self.logger.info("Starting academic compliance validation")
        
        violations = []
        category_scores = {}
        rules_checked = 0
        rules_passed = 0
        
        # Group rules by category for organized validation
        rules_by_category = self._group_rules_by_category()
        
        for category, rules in rules_by_category.items():
            self.logger.debug(f"Validating {category} compliance ({len(rules)} rules)")
            
            category_violations = []
            category_rules_checked = 0
            category_rules_passed = 0
            
            for rule in rules:
                try:
                    rule_violations = self._validate_rule(content, rule, context)
                    category_violations.extend(rule_violations)
                    category_rules_checked += 1
                    
                    if not rule_violations:
                        category_rules_passed += 1
                    
                except Exception as e:
                    self.logger.error(f"Error validating rule {rule.rule_id}: {str(e)}")
            
            violations.extend(category_violations)
            rules_checked += category_rules_checked
            rules_passed += category_rules_passed
            
            # Calculate category score
            if category_rules_checked > 0:
                category_score = category_rules_passed / category_rules_checked
                category_scores[category] = category_score
            else:
                category_scores[category] = 1.0
        
        # Calculate overall compliance score
        overall_score = self._calculate_overall_compliance_score(category_scores, violations)
        compliance_level = self._determine_compliance_level(overall_score)
        
        # Generate recommendations
        recommendations = self._generate_compliance_recommendations(violations, category_scores)
        
        # Check academic standards compliance
        standards_met = self._check_academic_standards_compliance(violations, overall_score)
        
        report = ComplianceReport(
            overall_compliance_score=overall_score,
            compliance_level=compliance_level,
            category_scores=category_scores,
            violations=violations,
            recommendations=recommendations,
            academic_standards_met=standards_met,
            total_rules_checked=rules_checked,
            rules_passed=rules_passed
        )
        
        self.logger.info(f"Academic compliance validation completed. Score: {overall_score:.1%}")
        return report
    
    def _initialize_compliance_rules(self) -> List[ComplianceRule]:
        """Initialize comprehensive academic compliance rules"""
        
        rules = []
        
        # IAST Transliteration Compliance Rules
        rules.extend([
            ComplianceRule(
                rule_id="iast_001",
                category="iast_transliteration",
                description="Sanskrit terms must use proper IAST diacriticals",
                pattern=r'\b[a-zA-Z]*[āīūṛḷēōṃḥṇṭḍśṣṅñ][a-zA-Z]*\b',
                weight=1.5,
                critical=True,
                academic_standard="IAST Standard"
            ),
            ComplianceRule(
                rule_id="iast_002", 
                category="iast_transliteration",
                description="Consistent use of long vowels in Sanskrit terms",
                validator_func=self._validate_long_vowel_consistency,
                weight=1.2,
                academic_standard="IAST Standard"
            ),
            ComplianceRule(
                rule_id="iast_003",
                category="iast_transliteration", 
                description="Proper retroflex consonant usage",
                validator_func=self._validate_retroflex_consonants,
                weight=1.0,
                academic_standard="IAST Standard"
            )
        ])
        
        # Sanskrit Terminology Accuracy Rules
        rules.extend([
            ComplianceRule(
                rule_id="sanskrit_001",
                category="sanskrit_accuracy",
                description="Sanskrit deity names properly capitalized",
                validator_func=self._validate_deity_capitalization,
                weight=1.3,
                critical=True,
                academic_standard="Sanskrit Academic Convention"
            ),
            ComplianceRule(
                rule_id="sanskrit_002",
                category="sanskrit_accuracy", 
                description="Sacred text titles properly formatted",
                validator_func=self._validate_sacred_text_formatting,
                weight=1.2,
                academic_standard="Sanskrit Academic Convention"
            ),
            ComplianceRule(
                rule_id="sanskrit_003",
                category="sanskrit_accuracy",
                description="Philosophical terms contextually appropriate",
                validator_func=self._validate_philosophical_terms,
                weight=1.0,
                academic_standard="Sanskrit Academic Convention"
            )
        ])
        
        # Academic Formatting Rules
        rules.extend([
            ComplianceRule(
                rule_id="format_001",
                category="academic_formatting",
                description="Proper sentence capitalization",
                pattern=r'(?:^|[.!?]\s+)([a-z])',
                weight=1.0,
                academic_standard="Academic Writing Standards"
            ),
            ComplianceRule(
                rule_id="format_002",
                category="academic_formatting",
                description="Consistent punctuation usage",
                validator_func=self._validate_punctuation_consistency,
                weight=0.8,
                academic_standard="Academic Writing Standards"
            ),
            ComplianceRule(
                rule_id="format_003",
                category="academic_formatting",
                description="Proper spacing and formatting",
                pattern=r'  +|[.]{4,}|\s+[.,:;]',
                weight=0.6,
                academic_standard="Academic Writing Standards"
            )
        ])
        
        # Terminology Consistency Rules
        rules.extend([
            ComplianceRule(
                rule_id="terminology_001",
                category="terminology_consistency",
                description="Consistent spelling of technical terms",
                validator_func=self._validate_term_consistency,
                weight=1.1,
                academic_standard="Terminology Standards"
            ),
            ComplianceRule(
                rule_id="terminology_002",
                category="terminology_consistency",
                description="Appropriate capitalization of proper nouns",
                validator_func=self._validate_proper_noun_capitalization,
                weight=1.0,
                academic_standard="Terminology Standards"
            )
        ])
        
        # Spiritual Content Respectfulness Rules
        rules.extend([
            ComplianceRule(
                rule_id="spiritual_001",
                category="spiritual_respectfulness",
                description="Respectful treatment of religious concepts",
                validator_func=self._validate_spiritual_respectfulness,
                weight=1.5,
                critical=True,
                academic_standard="Spiritual Content Guidelines"
            ),
            ComplianceRule(
                rule_id="spiritual_002",
                category="spiritual_respectfulness",
                description="Appropriate tone for sacred content",
                validator_func=self._validate_sacred_content_tone,
                weight=1.2,
                academic_standard="Spiritual Content Guidelines"
            )
        ])
        
        return rules
    
    def _group_rules_by_category(self) -> Dict[str, List[ComplianceRule]]:
        """Group compliance rules by category"""
        
        grouped = {}
        for rule in self.compliance_rules:
            if rule.category not in grouped:
                grouped[rule.category] = []
            grouped[rule.category].append(rule)
        
        return grouped
    
    def _validate_rule(self, 
                      content: str, 
                      rule: ComplianceRule,
                      context: Dict[str, Any]) -> List[ComplianceViolation]:
        """Validate content against a specific compliance rule"""
        
        violations = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Skip SRT timestamp lines and empty lines
            if re.match(r'^\d+$', line.strip()) or re.match(r'^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$', line.strip()) or not line.strip():
                continue
            
            # Apply pattern-based validation
            if rule.pattern:
                matches = re.finditer(rule.pattern, line)
                for match in matches:
                    violations.append(ComplianceViolation(
                        rule_id=rule.rule_id,
                        line_number=line_num,
                        violation_type=rule.category,
                        description=rule.description,
                        original_text=match.group(),
                        suggested_correction="[Requires manual review]",
                        severity="critical" if rule.critical else "major",
                        academic_standard=rule.academic_standard,
                        confidence=0.8
                    ))
            
            # Apply function-based validation
            if rule.validator_func:
                try:
                    func_violations = rule.validator_func(line, line_num, rule, context)
                    violations.extend(func_violations)
                except Exception as e:
                    self.logger.error(f"Validator function failed for rule {rule.rule_id}: {str(e)}")
        
        return violations
    
    def _validate_long_vowel_consistency(self, line: str, line_num: int, rule: ComplianceRule, context: Dict[str, Any]) -> List[ComplianceViolation]:
        """Validate consistent use of long vowels in Sanskrit terms"""
        violations = []
        
        # Common Sanskrit words that should have long vowels
        long_vowel_terms = {
            'yoga': 'yoga',  # Should remain as is in common usage
            'karma': 'karma',  # Should remain as is in common usage
            'dharma': 'dharma',  # Should remain as is in common usage
            'krishna': 'Kṛṣṇa',  # Should use IAST in academic contexts
            'rama': 'Rāma',      # Should use IAST in academic contexts
            'gita': 'Gītā'       # Should use IAST in academic contexts
        }
        
        for term, correct_form in long_vowel_terms.items():
            if re.search(rf'\b{term}\b', line, re.IGNORECASE):
                # Check if academic context requires IAST
                if context.get('academic_context', False) and term != correct_form:
                    violations.append(ComplianceViolation(
                        rule_id=rule.rule_id,
                        line_number=line_num,
                        violation_type="iast_inconsistency",
                        description=f"Term '{term}' should use IAST form '{correct_form}' in academic context",
                        original_text=term,
                        suggested_correction=correct_form,
                        severity="major",
                        academic_standard=rule.academic_standard,
                        confidence=0.9
                    ))
        
        return violations
    
    def _validate_retroflex_consonants(self, line: str, line_num: int, rule: ComplianceRule, context: Dict[str, Any]) -> List[ComplianceViolation]:
        """Validate proper use of retroflex consonants"""
        violations = []
        
        # Common patterns requiring retroflex consonants
        retroflex_patterns = [
            (r'\bkrishna\b', 'Kṛṣṇa', 'Krishna should use retroflex consonants in IAST'),
            (r'\bsharma\b', 'Śarma', 'Sharma should use proper śa in IAST'),
            (r'\bshiva\b', 'Śiva', 'Shiva should use proper śa in IAST')
        ]
        
        for pattern, correction, description in retroflex_patterns:
            if re.search(pattern, line, re.IGNORECASE) and self.standards_config.get('iast_strict_mode', True):
                violations.append(ComplianceViolation(
                    rule_id=rule.rule_id,
                    line_number=line_num,
                    violation_type="retroflex_missing",
                    description=description,
                    original_text=re.search(pattern, line, re.IGNORECASE).group(),
                    suggested_correction=correction,
                    severity="major",
                    academic_standard=rule.academic_standard,
                    confidence=0.85
                ))
        
        return violations
    
    def _validate_deity_capitalization(self, line: str, line_num: int, rule: ComplianceRule, context: Dict[str, Any]) -> List[ComplianceViolation]:
        """Validate proper capitalization of deity names"""
        violations = []
        
        deity_names = ['krishna', 'rama', 'shiva', 'vishnu', 'hanuman', 'ganesha', 'durga', 'kali']
        
        for deity in deity_names:
            pattern = rf'\b{deity}\b'
            matches = re.finditer(pattern, line, re.IGNORECASE)
            
            for match in matches:
                if match.group().islower():
                    violations.append(ComplianceViolation(
                        rule_id=rule.rule_id,
                        line_number=line_num,
                        violation_type="deity_capitalization",
                        description=f"Deity name '{deity}' should be capitalized",
                        original_text=match.group(),
                        suggested_correction=match.group().capitalize(),
                        severity="critical",
                        academic_standard=rule.academic_standard,
                        confidence=0.95
                    ))
        
        return violations
    
    def _validate_sacred_text_formatting(self, line: str, line_num: int, rule: ComplianceRule, context: Dict[str, Any]) -> List[ComplianceViolation]:
        """Validate proper formatting of sacred text titles"""
        violations = []
        
        sacred_texts = {
            'bhagavad gita': 'Bhagavad Gītā',
            'ramayana': 'Rāmāyaṇa', 
            'mahabharata': 'Mahābhārata',
            'upanishads': 'Upaniṣads',
            'vedas': 'Vedas'
        }
        
        for text, correct_form in sacred_texts.items():
            pattern = rf'\b{text}\b'
            if re.search(pattern, line, re.IGNORECASE):
                match = re.search(pattern, line, re.IGNORECASE)
                if match.group() != correct_form:
                    violations.append(ComplianceViolation(
                        rule_id=rule.rule_id,
                        line_number=line_num,
                        violation_type="sacred_text_formatting",
                        description=f"Sacred text '{text}' should be formatted as '{correct_form}'",
                        original_text=match.group(),
                        suggested_correction=correct_form,
                        severity="major",
                        academic_standard=rule.academic_standard,
                        confidence=0.9
                    ))
        
        return violations
    
    def _validate_philosophical_terms(self, line: str, line_num: int, rule: ComplianceRule, context: Dict[str, Any]) -> List[ComplianceViolation]:
        """Validate contextually appropriate use of philosophical terms"""
        violations = []
        
        # This is a placeholder for more sophisticated contextual validation
        # In practice, this would analyze term usage in context
        
        philosophical_terms = ['dharma', 'karma', 'moksha', 'samsara', 'atman', 'brahman']
        
        for term in philosophical_terms:
            if re.search(rf'\b{term}\b', line, re.IGNORECASE):
                # Check for obvious misuse (this is simplified)
                if re.search(rf'\b{term}\s+is\s+(stupid|wrong|fake)', line, re.IGNORECASE):
                    violations.append(ComplianceViolation(
                        rule_id=rule.rule_id,
                        line_number=line_num,
                        violation_type="philosophical_term_misuse",
                        description=f"Inappropriate context for philosophical term '{term}'",
                        original_text=line.strip(),
                        suggested_correction="[Requires expert review]",
                        severity="critical",
                        academic_standard=rule.academic_standard,
                        confidence=0.7
                    ))
        
        return violations
    
    def _validate_punctuation_consistency(self, line: str, line_num: int, rule: ComplianceRule, context: Dict[str, Any]) -> List[ComplianceViolation]:
        """Validate consistent punctuation usage"""
        violations = []
        
        # Check for common punctuation issues
        issues = [
            (r'\s+[.,:;]', 'Space before punctuation should be removed'),
            (r'[.]{2,3}(?![.])', 'Use proper ellipsis (...) for omissions'),
            (r'[!]{2,}', 'Multiple exclamation marks not appropriate for academic text'),
            (r'[?]{2,}', 'Multiple question marks not appropriate for academic text')
        ]
        
        for pattern, description in issues:
            if re.search(pattern, line):
                violations.append(ComplianceViolation(
                    rule_id=rule.rule_id,
                    line_number=line_num,
                    violation_type="punctuation_inconsistency",
                    description=description,
                    original_text=re.search(pattern, line).group(),
                    suggested_correction="[Automatic correction available]",
                    severity="minor",
                    academic_standard=rule.academic_standard,
                    confidence=0.8
                ))
        
        return violations
    
    def _validate_term_consistency(self, line: str, line_num: int, rule: ComplianceRule, context: Dict[str, Any]) -> List[ComplianceViolation]:
        """Validate consistency in terminology usage"""
        violations = []
        
        # This would integrate with a term database to check for consistency
        # Placeholder implementation
        
        return violations
    
    def _validate_proper_noun_capitalization(self, line: str, line_num: int, rule: ComplianceRule, context: Dict[str, Any]) -> List[ComplianceViolation]:
        """Validate proper noun capitalization"""
        violations = []
        
        proper_nouns = [
            'vedanta', 'sankhya', 'advaita', 'yoga', 'ayurveda', 'sanskrit', 'hindi',
            'india', 'hinduism', 'buddhism', 'jainism'
        ]
        
        for noun in proper_nouns:
            pattern = rf'\b{noun}\b'
            matches = re.finditer(pattern, line, re.IGNORECASE)
            
            for match in matches:
                # Check if at start of sentence or should be capitalized
                if (match.start() == 0 or line[match.start()-1] in '.!?') and match.group().islower():
                    violations.append(ComplianceViolation(
                        rule_id=rule.rule_id,
                        line_number=line_num,
                        violation_type="proper_noun_capitalization",
                        description=f"Proper noun '{noun}' should be capitalized",
                        original_text=match.group(),
                        suggested_correction=match.group().capitalize(),
                        severity="major",
                        academic_standard=rule.academic_standard,
                        confidence=0.8
                    ))
        
        return violations
    
    def _validate_spiritual_respectfulness(self, line: str, line_num: int, rule: ComplianceRule, context: Dict[str, Any]) -> List[ComplianceViolation]:
        """Validate respectful treatment of spiritual concepts"""
        violations = []
        
        disrespectful_patterns = [
            (r'\b(god|krishna|rama|shiva|vishnu)\s+is\s+(just|only|merely)', 'Potentially diminishing language'),
            (r'\b(stupid|dumb|idiotic)\s+(hindu|sanskrit|yoga)', 'Offensive language toward tradition'),
            (r'\b(primitive|backward)\s+(belief|practice|teaching)', 'Dismissive language')
        ]
        
        for pattern, description in disrespectful_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                violations.append(ComplianceViolation(
                    rule_id=rule.rule_id,
                    line_number=line_num,
                    violation_type="spiritual_disrespect",
                    description=f"Potentially disrespectful language: {description}",
                    original_text=re.search(pattern, line, re.IGNORECASE).group(),
                    suggested_correction="[Requires human review for appropriate language]",
                    severity="critical",
                    academic_standard=rule.academic_standard,
                    confidence=0.9
                ))
        
        return violations
    
    def _validate_sacred_content_tone(self, line: str, line_num: int, rule: ComplianceRule, context: Dict[str, Any]) -> List[ComplianceViolation]:
        """Validate appropriate tone for sacred content"""
        violations = []
        
        # Check for casual/inappropriate language in sacred contexts
        sacred_context_indicators = ['scripture', 'verse', 'sacred', 'divine', 'holy', 'blessed']
        inappropriate_tone = ['whatever', 'stuff', 'things', 'like totally', 'awesome']
        
        has_sacred_context = any(re.search(rf'\b{indicator}\b', line, re.IGNORECASE) for indicator in sacred_context_indicators)
        
        if has_sacred_context:
            for inappropriate in inappropriate_tone:
                if re.search(rf'\b{inappropriate}\b', line, re.IGNORECASE):
                    violations.append(ComplianceViolation(
                        rule_id=rule.rule_id,
                        line_number=line_num,
                        violation_type="inappropriate_tone",
                        description=f"Casual language '{inappropriate}' inappropriate in sacred context",
                        original_text=inappropriate,
                        suggested_correction="[Use more formal academic language]",
                        severity="major",
                        academic_standard=rule.academic_standard,
                        confidence=0.7
                    ))
        
        return violations
    
    def _calculate_overall_compliance_score(self, category_scores: Dict[str, float], violations: List[ComplianceViolation]) -> float:
        """Calculate overall compliance score"""
        
        if not category_scores:
            return 0.0
        
        # Weight categories by importance
        category_weights = {
            'iast_transliteration': 1.5,
            'sanskrit_accuracy': 1.3,
            'academic_formatting': 1.0,
            'terminology_consistency': 1.1,
            'spiritual_respectfulness': 1.4
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for category, score in category_scores.items():
            weight = category_weights.get(category, 1.0)
            weighted_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            base_score = weighted_score / total_weight
        else:
            base_score = 1.0
        
        # Apply penalties for critical violations
        critical_violations = [v for v in violations if v.severity == 'critical']
        if critical_violations:
            penalty = min(len(critical_violations) * 0.05, 0.2)  # Max 20% penalty
            base_score = max(0.0, base_score - penalty)
        
        return base_score
    
    def _determine_compliance_level(self, score: float) -> ComplianceLevel:
        """Determine compliance level from score"""
        
        if score >= 0.95:
            return ComplianceLevel.EXCELLENT
        elif score >= 0.85:
            return ComplianceLevel.GOOD
        elif score >= 0.75:
            return ComplianceLevel.ADEQUATE
        else:
            return ComplianceLevel.NEEDS_IMPROVEMENT
    
    def _generate_compliance_recommendations(self, violations: List[ComplianceViolation], category_scores: Dict[str, float]) -> List[str]:
        """Generate recommendations for improving compliance"""
        
        recommendations = []
        
        # Category-specific recommendations
        for category, score in category_scores.items():
            if score < 0.8:
                if category == 'iast_transliteration':
                    recommendations.append("Improve IAST transliteration accuracy - consider expert linguistic review")
                elif category == 'sanskrit_accuracy':
                    recommendations.append("Enhance Sanskrit term recognition and validation")
                elif category == 'academic_formatting':
                    recommendations.append("Review and improve academic formatting standards")
                elif category == 'terminology_consistency':
                    recommendations.append("Standardize terminology usage across content")
                elif category == 'spiritual_respectfulness':
                    recommendations.append("Review content for appropriate spiritual tone and respect")
        
        # Violation-specific recommendations
        critical_violations = [v for v in violations if v.severity == 'critical']
        if critical_violations:
            recommendations.append(f"Address {len(critical_violations)} critical compliance violations immediately")
        
        major_violations = [v for v in violations if v.severity == 'major']
        if len(major_violations) > 10:
            recommendations.append("High number of major violations - consider comprehensive review")
        
        return recommendations
    
    def _check_academic_standards_compliance(self, violations: List[ComplianceViolation], overall_score: float) -> Dict[str, bool]:
        """Check compliance with specific academic standards"""
        
        standards_compliance = {}
        
        # IAST Standard compliance
        iast_violations = [v for v in violations if 'iast' in v.violation_type.lower()]
        standards_compliance['IAST Standard'] = len(iast_violations) == 0 and overall_score >= 0.85
        
        # Sanskrit Academic Convention compliance
        sanskrit_violations = [v for v in violations if 'sanskrit' in v.violation_type.lower()]
        standards_compliance['Sanskrit Academic Convention'] = len(sanskrit_violations) <= 2 and overall_score >= 0.8
        
        # Academic Writing Standards compliance
        format_violations = [v for v in violations if 'format' in v.violation_type.lower()]
        standards_compliance['Academic Writing Standards'] = len(format_violations) <= 5 and overall_score >= 0.75
        
        # Spiritual Content Guidelines compliance
        spiritual_violations = [v for v in violations if 'spiritual' in v.violation_type.lower()]
        standards_compliance['Spiritual Content Guidelines'] = len(spiritual_violations) == 0
        
        return standards_compliance


def create_academic_compliance_validator(config: Optional[Dict[str, Any]] = None) -> AcademicComplianceValidator:
    """Factory function to create academic compliance validator"""
    return AcademicComplianceValidator(config)