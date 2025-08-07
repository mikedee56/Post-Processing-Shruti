"""
Scripture Validator Module.

This module provides validation capabilities for scripture identification and
substitution operations to ensure academic accuracy.
"""

from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum
import logging
import re

from utils.logger_config import get_logger
from utils.fuzzy_matcher import FuzzyMatcher
from .scripture_identifier import VerseMatch, PassageType
from .canonical_text_manager import CanonicalVerse


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    severity: ValidationSeverity
    category: str
    message: str
    suggestion: Optional[str] = None
    confidence_impact: float = 0.0


@dataclass
class ValidationResult:
    """Result of scripture validation."""
    is_valid: bool
    confidence_score: float
    issues: List[ValidationIssue]
    warnings: List[str]
    errors: List[str]
    metadata: Dict[str, Any]


class ScriptureValidator:
    """
    Validator for scripture identification and substitution accuracy.
    
    Provides comprehensive validation to ensure academic rigor and accuracy
    in verse identification and canonical text substitution.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the Scripture Validator.
        
        Args:
            config: Configuration parameters
        """
        self.logger = get_logger(__name__)
        # Initialize fuzzy matcher with empty lexicon data for basic similarity calculations
        self.fuzzy_matcher = FuzzyMatcher({})
        
        # Configuration
        self.config = config or {}
        self.min_validation_confidence = self.config.get('min_validation_confidence', 0.7)
        self.max_length_ratio = self.config.get('max_length_ratio', 3.0)
        self.min_word_overlap = self.config.get('min_word_overlap', 0.3)
        self.require_sanskrit_markers = self.config.get('require_sanskrit_markers', True)
        
        # Validation rules
        self.sanskrit_markers = [
            r'[|।]',  # Sanskrit punctuation
            r'[aeiou]m\b',  # Sanskrit word endings
            r'\byam\b',  # Common Sanskrit words
            r'\btam\b',
            r'ā[a-zA-Z]*',  # IAST long vowels
        ]
        
        # Academic standards
        self.iast_patterns = [
            r'[āīūṛṝḷḹēōṃḥṅñṭḍṇtdnpbmyrlvśṣshḥ]',  # IAST characters
        ]
        
        self.logger.info("Scripture validator initialized")
    
    def validate_verse_identification(self, match: VerseMatch, original_text: str) -> ValidationResult:
        """
        Validate a verse identification result.
        
        Args:
            match: Verse match to validate
            original_text: Original text context
            
        Returns:
            Validation result
        """
        issues = []
        warnings = []
        errors = []
        metadata = {}
        
        # Confidence threshold check
        if match.confidence_score < self.min_validation_confidence:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="confidence",
                message=f"Low confidence score: {match.confidence_score:.3f}",
                suggestion="Consider manual review",
                confidence_impact=-0.2
            ))
        
        # Length ratio validation
        original_len = len(match.original_text)
        canonical_len = len(getattr(match.canonical_entry, 'canonical_text', ''))
        if canonical_len > 0:
            length_ratio = original_len / canonical_len
            metadata['length_ratio'] = length_ratio
            
            if length_ratio > self.max_length_ratio or length_ratio < (1.0 / self.max_length_ratio):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="length",
                    message=f"Significant length difference (ratio: {length_ratio:.2f})",
                    suggestion="Verify passage boundaries",
                    confidence_impact=-0.1
                ))
        
        # Word overlap validation
        word_overlap_score = self._calculate_word_overlap(
            match.original_text, 
            getattr(match.canonical_entry, 'canonical_text', '')
        )
        metadata['word_overlap'] = word_overlap_score
        
        if word_overlap_score < self.min_word_overlap:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="content",
                message=f"Low word overlap: {word_overlap_score:.3f}",
                suggestion="Check if this is the correct verse",
                confidence_impact=-0.3
            ))
        
        # Sanskrit marker validation
        if self.require_sanskrit_markers:
            sanskrit_marker_found = any(
                re.search(pattern, match.original_text, re.IGNORECASE)
                for pattern in self.sanskrit_markers
            )
            
            if not sanskrit_marker_found:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="sanskrit_markers",
                    message="No Sanskrit markers found in identified verse",
                    suggestion="Verify this is actually a Sanskrit verse",
                    confidence_impact=-0.15
                ))
        
        # Passage type validation
        type_validation = self._validate_passage_type(match)
        if type_validation:
            issues.extend(type_validation)
        
        # Calculate overall validation confidence
        base_confidence = match.confidence_score
        confidence_penalties = sum(issue.confidence_impact for issue in issues)
        validation_confidence = max(0.0, base_confidence + confidence_penalties)
        
        # Determine if validation passes
        is_valid = (
            validation_confidence >= self.min_validation_confidence and
            not any(issue.severity == ValidationSeverity.ERROR for issue in issues)
        )
        
        # Extract warnings and errors
        warnings = [issue.message for issue in issues if issue.severity == ValidationSeverity.WARNING]
        errors = [issue.message for issue in issues if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]
        
        return ValidationResult(
            is_valid=is_valid,
            confidence_score=validation_confidence,
            issues=issues,
            warnings=warnings,
            errors=errors,
            metadata=metadata
        )
    
    def validate_substitution(self, original_text: str, canonical_text: str, 
                            match: VerseMatch) -> ValidationResult:
        """
        Validate a verse substitution operation.
        
        Args:
            original_text: Original text to be replaced
            canonical_text: Canonical text for replacement
            match: Verse match information
            
        Returns:
            Validation result
        """
        issues = []
        warnings = []
        errors = []
        metadata = {}
        
        # Basic text validation
        if not canonical_text or not canonical_text.strip():
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="canonical_text",
                message="Empty canonical text",
                suggestion="Verify canonical text is available",
                confidence_impact=-1.0
            ))
        
        if not original_text or not original_text.strip():
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="original_text",
                message="Empty original text",
                confidence_impact=-1.0
            ))
        
        # IAST compliance validation
        iast_compliance = self._validate_iast_compliance(canonical_text)
        if not iast_compliance['is_compliant']:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="iast",
                message=f"IAST compliance issues: {', '.join(iast_compliance['issues'])}",
                suggestion="Verify canonical text follows IAST standards",
                confidence_impact=-0.1
            ))
        
        # Semantic consistency validation
        semantic_score = self._validate_semantic_consistency(original_text, canonical_text)
        metadata['semantic_consistency'] = semantic_score
        
        if semantic_score < 0.4:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="semantic",
                message=f"Low semantic consistency: {semantic_score:.3f}",
                suggestion="Verify this is the correct canonical text",
                confidence_impact=-0.4
            ))
        elif semantic_score < 0.7:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="semantic",
                message=f"Moderate semantic consistency: {semantic_score:.3f}",
                suggestion="Consider manual review",
                confidence_impact=-0.1
            ))
        
        # Character set validation
        char_validation = self._validate_character_sets(original_text, canonical_text)
        if char_validation:
            issues.extend(char_validation)
        
        # Length appropriateness
        length_validation = self._validate_substitution_length(original_text, canonical_text)
        if length_validation:
            issues.extend(length_validation)
        
        # Calculate validation confidence
        base_confidence = match.confidence_score
        confidence_penalties = sum(issue.confidence_impact for issue in issues)
        validation_confidence = max(0.0, base_confidence + confidence_penalties)
        
        # Determine validity
        is_valid = (
            validation_confidence >= self.min_validation_confidence and
            not any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] for issue in issues)
        )
        
        # Extract warnings and errors
        warnings = [issue.message for issue in issues if issue.severity == ValidationSeverity.WARNING]
        errors = [issue.message for issue in issues if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]
        
        return ValidationResult(
            is_valid=is_valid,
            confidence_score=validation_confidence,
            issues=issues,
            warnings=warnings,
            errors=errors,
            metadata=metadata
        )
    
    def validate_academic_standards(self, canonical_verse: CanonicalVerse) -> ValidationResult:
        """
        Validate a canonical verse meets academic standards.
        
        Args:
            canonical_verse: Canonical verse to validate
            
        Returns:
            Validation result
        """
        issues = []
        warnings = []
        errors = []
        metadata = {}
        
        # Source authority validation
        if not canonical_verse.source_authority or canonical_verse.source_authority == 'unknown':
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="authority",
                message="No source authority specified",
                suggestion="Add authoritative source reference",
                confidence_impact=-0.3
            ))
        
        # IAST transliteration validation
        if canonical_verse.transliteration:
            iast_result = self._validate_iast_compliance(canonical_verse.transliteration)
            if not iast_result['is_compliant']:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="transliteration",
                    message=f"Transliteration IAST issues: {', '.join(iast_result['issues'])}",
                    confidence_impact=-0.1
                ))
        else:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="transliteration",
                message="No IAST transliteration provided",
                suggestion="Add IAST transliteration for academic completeness"
            ))
        
        # Canonical text validation
        if not canonical_verse.canonical_text:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="canonical_text",
                message="No canonical text provided",
                confidence_impact=-1.0
            ))
        else:
            # Check for verse structure markers
            has_verse_markers = bool(re.search(r'[|।]|\|\|', canonical_verse.canonical_text))
            if not has_verse_markers:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="structure",
                    message="No verse structure markers found",
                    suggestion="Consider adding verse markers (| or ||)"
                ))
        
        # Metadata completeness
        required_fields = ['chapter', 'verse', 'source']
        for field in required_fields:
            if not getattr(canonical_verse, field, None):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="metadata",
                    message=f"Missing required field: {field}",
                    confidence_impact=-0.2
                ))
        
        # Calculate confidence
        base_confidence = 1.0
        confidence_penalties = sum(issue.confidence_impact for issue in issues)
        validation_confidence = max(0.0, base_confidence + confidence_penalties)
        
        is_valid = not any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] for issue in issues)
        
        warnings = [issue.message for issue in issues if issue.severity == ValidationSeverity.WARNING]
        errors = [issue.message for issue in issues if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]
        
        return ValidationResult(
            is_valid=is_valid,
            confidence_score=validation_confidence,
            issues=issues,
            warnings=warnings,
            errors=errors,
            metadata=metadata
        )
    
    def _calculate_word_overlap(self, text1: str, text2: str) -> float:
        """Calculate word overlap between two texts."""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _validate_passage_type(self, match: VerseMatch) -> List[ValidationIssue]:
        """Validate the passage type classification."""
        issues = []
        
        if match.passage_type == PassageType.VERSE:
            # Full verse should have reasonable length
            if len(match.original_text) < 20:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="passage_type",
                    message="Full verse seems too short",
                    suggestion="Verify this is a complete verse",
                    confidence_impact=-0.1
                ))
        elif match.passage_type == PassageType.PARTIAL_VERSE:
            # Partial verse should indicate uncertainty
            if match.confidence_score > 0.9:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="passage_type",
                    message="High confidence for partial verse match",
                    suggestion="Consider if this might be a full verse"
                ))
        
        return issues
    
    def _validate_iast_compliance(self, text: str) -> Dict[str, Any]:
        """Validate IAST transliteration compliance."""
        issues = []
        
        # Check for proper IAST characters
        has_iast_chars = bool(re.search('|'.join(self.iast_patterns), text))
        
        # Check for common non-IAST patterns that might indicate issues
        non_iast_patterns = [
            (r'[āēīōū][aeiou]', "Double vowels might indicate incorrect IAST"),
            (r'[kgcjṭḍtdpb]h(?![aeiouāīūṛṝḷḹēōṃḥ])', "Aspirated consonants without following vowel"),
            (r'[0-9]', "Numbers in transliteration might be incorrect")
        ]
        
        for pattern, message in non_iast_patterns:
            if re.search(pattern, text):
                issues.append(message)
        
        return {
            'is_compliant': has_iast_chars and len(issues) == 0,
            'has_iast_chars': has_iast_chars,
            'issues': issues
        }
    
    def _validate_semantic_consistency(self, original: str, canonical: str) -> float:
        """Validate semantic consistency between original and canonical text."""
        if not original or not canonical:
            return 0.0
        
        # Use fuzzy matching for basic semantic similarity
        similarity = self.fuzzy_matcher.calculate_similarity(original.lower(), canonical.lower())
        
        # Boost score for Sanskrit-specific similarities
        original_words = set(original.lower().split())
        canonical_words = set(canonical.lower().split())
        
        # Look for transliteration patterns
        transliteration_matches = 0
        for orig_word in original_words:
            for canon_word in canonical_words:
                if self._is_likely_transliteration_pair(orig_word, canon_word):
                    transliteration_matches += 1
                    break
        
        if len(original_words) > 0:
            transliteration_boost = min(0.3, transliteration_matches / len(original_words))
        else:
            transliteration_boost = 0.0
        
        return min(1.0, similarity + transliteration_boost)
    
    def _is_likely_transliteration_pair(self, word1: str, word2: str) -> bool:
        """Check if two words are likely transliteration pairs."""
        # Simple heuristic for transliteration similarity
        if len(word1) == 0 or len(word2) == 0:
            return False
        
        # Check if they start with same consonant
        if word1[0].lower() == word2[0].lower():
            # Check if length is similar
            length_ratio = len(word1) / len(word2)
            if 0.7 <= length_ratio <= 1.3:
                return True
        
        return False
    
    def _validate_character_sets(self, original: str, canonical: str) -> List[ValidationIssue]:
        """Validate appropriate character sets in texts."""
        issues = []
        
        # Check for mixed scripts that might indicate problems
        has_devanagari = bool(re.search(r'[\u0900-\u097F]', original))
        has_latin = bool(re.search(r'[a-zA-Z]', original))
        has_iast = bool(re.search('|'.join(self.iast_patterns), canonical))
        
        if has_devanagari and not has_iast:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="character_set",
                message="Original has Devanagari but canonical lacks IAST",
                suggestion="Ensure proper transliteration to IAST"
            ))
        
        return issues
    
    def _validate_substitution_length(self, original: str, canonical: str) -> List[ValidationIssue]:
        """Validate appropriateness of substitution length."""
        issues = []
        
        if len(canonical) > len(original) * 2:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="length",
                message="Canonical text much longer than original",
                suggestion="Verify this is the correct verse portion",
                confidence_impact=-0.1
            ))
        elif len(original) > len(canonical) * 2:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="length", 
                message="Original text much longer than canonical",
                suggestion="Check if original contains extra content",
                confidence_impact=-0.1
            ))
        
        return issues