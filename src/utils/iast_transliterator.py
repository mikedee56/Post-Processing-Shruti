"""
IAST Transliteration Enforcement Module.

This module provides functionality to enforce IAST (International Alphabet of
Sanskrit Transliteration) standards for Sanskrit and Hindi terms in transcripts.
It ensures academic rigor and consistency in transliteration.
"""

import re
import unicodedata
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum
import logging

from utils.logger_config import get_logger


class TransliterationStandard(Enum):
    """Supported transliteration standards."""
    IAST = "iast"
    ISO = "iso"
    HARVARD_KYOTO = "harvard_kyoto"
    ITRANS = "itrans"


@dataclass
class TransliterationRule:
    """Represents a transliteration rule."""
    source: str
    target: str
    context: Optional[str] = None
    priority: int = 1
    standard: TransliterationStandard = TransliterationStandard.IAST


@dataclass
class TransliterationResult:
    """Result of transliteration operation."""
    original_text: str
    transliterated_text: str
    changes_made: List[Tuple[str, str]]
    confidence: float
    rules_applied: List[TransliterationRule]
    issues_found: List[str]


class IASTTransliterator:
    """
    IAST Transliteration Enforcement System.
    
    Converts various transliteration schemes to IAST standard and validates
    existing IAST transliterations for consistency and correctness.
    """

    def __init__(self, strict_mode: bool = True):
        """
        Initialize the IAST transliterator.
        
        Args:
            strict_mode: If True, applies strict IAST rules; if False, more lenient
        """
        self.logger = get_logger(__name__)
        self.strict_mode = strict_mode
        
        # Initialize transliteration rules
        self._init_transliteration_rules()
        
        # Initialize IAST character sets
        self._init_iast_characters()
        
        # Initialize validation rules
        self._init_validation_rules()
        
        self.logger.info(f"IAST Transliterator initialized (strict_mode={strict_mode})")

    def _init_transliteration_rules(self) -> None:
        """Initialize transliteration rules for various schemes to IAST."""
        self.transliteration_rules = {
            # Harvard-Kyoto to IAST
            TransliterationStandard.HARVARD_KYOTO: [
                TransliterationRule("A", "ā", priority=1),
                TransliterationRule("I", "ī", priority=1),
                TransliterationRule("U", "ū", priority=1),
                TransliterationRule("R", "ṛ", priority=1),
                TransliterationRule("RR", "ṝ", priority=2),
                TransliterationRule("lR", "ḷ", priority=2),
                TransliterationRule("lRR", "ḹ", priority=2),
                TransliterationRule("M", "ṃ", priority=1),
                TransliterationRule("H", "ḥ", priority=1),
                TransliterationRule("G", "ṅ", priority=1),
                TransliterationRule("J", "ñ", priority=1),
                TransliterationRule("T", "ṭ", priority=1),
                TransliterationRule("D", "ḍ", priority=1),
                TransliterationRule("N", "ṇ", priority=1),
                TransliterationRule("z", "ś", priority=1),
                TransliterationRule("S", "ṣ", priority=1),
            ],
            
            # ITRANS to IAST
            TransliterationStandard.ITRANS: [
                TransliterationRule("aa", "ā", priority=1),
                TransliterationRule("ii", "ī", priority=1),
                TransliterationRule("uu", "ū", priority=1),
                TransliterationRule("RRi", "ṛ", priority=2),
                TransliterationRule("RRI", "ṝ", priority=2),
                TransliterationRule("LLi", "ḷ", priority=2),
                TransliterationRule("LLI", "ḹ", priority=2),
                TransliterationRule(".m", "ṃ", priority=2),
                TransliterationRule(".h", "ḥ", priority=2),
                TransliterationRule("~N", "ṅ", priority=2),
                TransliterationRule("~n", "ñ", priority=2),
                TransliterationRule(".t", "ṭ", priority=2),
                TransliterationRule(".d", "ḍ", priority=2),
                TransliterationRule(".n", "ṇ", priority=2),
                TransliterationRule("sh", "ś", priority=2),
                TransliterationRule("Sh", "ṣ", priority=2),
            ],
            
            # Common ASCII approximations to IAST
            TransliterationStandard.ISO: [
                TransliterationRule("aa", "ā", priority=1),
                TransliterationRule("ii", "ī", priority=1),
                TransliterationRule("uu", "ū", priority=1),
                TransliterationRule("ri", "ṛ", priority=1),
                TransliterationRule("rri", "ṝ", priority=2),
                TransliterationRule("li", "ḷ", priority=1),
                TransliterationRule("lli", "ḹ", priority=2),
            ]
        }

    def _init_iast_characters(self) -> None:
        """Initialize IAST character sets for validation."""
        # Standard IAST vowels
        self.iast_vowels = {
            'a', 'ā', 'i', 'ī', 'u', 'ū', 'ṛ', 'ṝ', 'ḷ', 'ḹ', 'e', 'ai', 'o', 'au'
        }
        
        # Standard IAST consonants
        self.iast_consonants = {
            # Stops
            'k', 'kh', 'g', 'gh', 'ṅ',
            'c', 'ch', 'j', 'jh', 'ñ',
            'ṭ', 'ṭh', 'ḍ', 'ḍh', 'ṇ',
            't', 'th', 'd', 'dh', 'n',
            'p', 'ph', 'b', 'bh', 'm',
            # Approximants
            'y', 'r', 'l', 'v',
            # Sibilants
            'ś', 'ṣ', 's', 'h'
        }
        
        # IAST diacritical marks
        self.iast_diacritics = {
            '\u0101': 'ā',  # ā (a with macron)
            '\u012B': 'ī',  # ī (i with macron)
            '\u016B': 'ū',  # ū (u with macron)
            '\u1E5B': 'ṛ',  # ṛ (r with dot below)
            '\u1E5D': 'ṝ',  # ṝ (r with dot below and macron)
            '\u1E37': 'ḷ',  # ḷ (l with dot below)
            '\u1E39': 'ḹ',  # ḹ (l with dot below and macron)
            '\u1E43': 'ṃ',  # ṃ (m with dot below)
            '\u1E25': 'ḥ',  # ḥ (h with dot below)
            '\u1E45': 'ṅ',  # ṅ (n with dot above)
            '\u00F1': 'ñ',  # ñ (n with tilde) - though not standard IAST
            '\u1E6D': 'ṭ',  # ṭ (t with dot below)
            '\u1E0D': 'ḍ',  # ḍ (d with dot below)
            '\u1E47': 'ṇ',  # ṇ (n with dot below)
            '\u015B': 'ś',  # ś (s with acute)
            '\u1E63': 'ṣ',  # ṣ (s with dot below)
        }

    def _init_validation_rules(self) -> None:
        """Initialize validation rules for IAST compliance."""
        self.validation_patterns = [
            # Invalid character combinations
            (r'[āīū]{2,}', "Double long vowels not valid in IAST"),
            (r'[ṛṝ][ṛṝ]', "Double r-vowels not valid"),
            (r'ṃṃ+', "Multiple anusvara marks not valid"),
            (r'ḥḥ+', "Multiple visarga marks not valid"),
            
            # Common mistakes
            (r'[^a-zA-Z\s\u0100-\u017F\u1E00-\u1EFF]', "Non-IAST characters found"),
            (r'[A-Z]{2,}', "All-caps words should be reviewed"),
            
            # Contextual rules
            (r'ṃ[kgṅcjñṭḍṇtdnpbm]', "Anusvara before stops (consider class nasal)"),
            (r'[aiueo]ḥ\s+[aiueo]', "Visarga sandhi rules may apply"),
        ]

    def transliterate_to_iast(self, text: str, source_standard: TransliterationStandard = None) -> TransliterationResult:
        """
        Transliterate text to IAST standard.
        
        Args:
            text: Input text to transliterate
            source_standard: Source transliteration standard (auto-detect if None)
            
        Returns:
            TransliterationResult with the transliterated text and metadata
        """
        if source_standard is None:
            source_standard = self._detect_transliteration_standard(text)
        
        changes_made = []
        rules_applied = []
        issues_found = []
        original_text = text
        
        # Apply transliteration rules
        if source_standard in self.transliteration_rules:
            rules = sorted(self.transliteration_rules[source_standard], key=lambda r: r.priority, reverse=True)
            
            for rule in rules:
                if rule.source in text:
                    # Apply context-sensitive rules if specified
                    if rule.context:
                        pattern = f"{rule.context}.*?{re.escape(rule.source)}"
                        if re.search(pattern, text):
                            old_text = text
                            text = re.sub(re.escape(rule.source), rule.target, text)
                            if text != old_text:
                                changes_made.append((rule.source, rule.target))
                                rules_applied.append(rule)
                    else:
                        old_text = text
                        text = text.replace(rule.source, rule.target)
                        if text != old_text:
                            changes_made.append((rule.source, rule.target))
                            rules_applied.append(rule)
        
        # Normalize Unicode characters
        text = self._normalize_unicode(text)
        
        # Validate result
        validation_issues = self.validate_iast(text)
        issues_found.extend(validation_issues)
        
        # Calculate confidence
        confidence = self._calculate_transliteration_confidence(original_text, text, changes_made)
        
        return TransliterationResult(
            original_text=original_text,
            transliterated_text=text,
            changes_made=changes_made,
            confidence=confidence,
            rules_applied=rules_applied,
            issues_found=issues_found
        )

    def transliterate_text(self, text: str) -> str:
        """
        Simple transliteration method for backward compatibility.
        
        Args:
            text: Input text to transliterate
            
        Returns:
            Transliterated text as string
        """
        try:
            result = self.transliterate_to_iast(text)
            return result.transliterated_text
        except Exception as e:
            self.logger.warning(f"Transliteration failed for '{text}': {e}")
            return text  # Return original text if transliteration fails

    def validate_iast(self, text: str) -> List[str]:
        """
        Validate text for IAST compliance.
        
        Args:
            text: Text to validate
            
        Returns:
            List of validation issues found
        """
        issues = []
        
        # Check for invalid patterns
        for pattern, message in self.validation_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(f"{message}: found in '{text}'")
        
        # Check for proper diacritical marks
        issues.extend(self._validate_diacritics(text))
        
        # Check for consistency
        issues.extend(self._validate_consistency(text))
        
        return issues

    def _detect_transliteration_standard(self, text: str) -> TransliterationStandard:
        """Auto-detect the transliteration standard used in text."""
        # Simple heuristics for detection
        if re.search(r'[AIUR](?![a-z])', text):  # Capitals for long vowels
            return TransliterationStandard.HARVARD_KYOTO
        elif re.search(r'(aa|ii|uu|\.m|\.h)', text):
            return TransliterationStandard.ITRANS
        elif re.search(r'[āīūṛṝḷḹṃḥṅñṭḍṇśṣ]', text):
            return TransliterationStandard.IAST
        else:
            return TransliterationStandard.ISO  # Default for ASCII approximations

    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters to proper IAST forms."""
        # Use Unicode normalization
        normalized = unicodedata.normalize('NFC', text)
        
        # Replace any alternative Unicode forms with canonical IAST
        for alt_form, canonical in self.iast_diacritics.items():
            if alt_form in normalized:
                normalized = normalized.replace(alt_form, canonical)
        
        return normalized

    def _validate_diacritics(self, text: str) -> List[str]:
        """Validate proper use of diacritical marks."""
        issues = []
        
        # Find all characters with diacritics
        diacritical_chars = re.findall(r'[āīūṛṝḷḹṃḥṅñṭḍṇśṣ]', text, re.IGNORECASE)
        
        for char in diacritical_chars:
            # Check if character is properly formed
            normalized = unicodedata.normalize('NFC', char)
            if normalized != char:
                issues.append(f"Improperly formed diacritical mark: {char}")
        
        return issues

    def _validate_consistency(self, text: str) -> List[str]:
        """Validate consistency in transliteration choices."""
        issues = []
        
        # Check for mixed transliteration schemes
        has_iast = bool(re.search(r'[āīūṛṝḷḹṃḥṅñṭḍṇśṣ]', text))
        has_ascii = bool(re.search(r'(aa|ii|uu)', text))
        
        if has_iast and has_ascii:
            issues.append("Mixed transliteration schemes detected (IAST and ASCII)")
        
        # Check for consistent capitalization of proper nouns
        words = re.findall(r'\b[A-Za-z][āīūṛṝḷḹṃḥṅñṭḍṇśṣa-z]*\b', text)
        proper_noun_patterns = []
        
        for word in words:
            if word[0].isupper():
                base_word = word.lower()
                # Check if same word appears with different capitalization
                for other_word in words:
                    if other_word.lower() == base_word and other_word != word:
                        issues.append(f"Inconsistent capitalization: {word} vs {other_word}")
        
        return issues

    def _calculate_transliteration_confidence(self, original: str, transliterated: str, changes: List[Tuple[str, str]]) -> float:
        """Calculate confidence score for transliteration result."""
        if original == transliterated:
            return 1.0  # No changes needed
        
        # Base confidence on number of successful rule applications
        base_confidence = 0.8
        
        # Boost confidence for each successful rule application
        confidence_boost = min(0.2, len(changes) * 0.05)
        
        # Reduce confidence for validation issues
        validation_issues = self.validate_iast(transliterated)
        confidence_penalty = min(0.3, len(validation_issues) * 0.1)
        
        final_confidence = base_confidence + confidence_boost - confidence_penalty
        return max(0.0, min(1.0, final_confidence))

    def get_iast_info(self, text: str) -> Dict[str, Any]:
        """
        Get comprehensive information about IAST usage in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with IAST analysis information
        """
        info = {
            'total_characters': len(text),
            'iast_characters': 0,
            'character_breakdown': {},
            'compliance_score': 0.0,
            'issues': [],
            'suggestions': []
        }
        
        # Count IAST characters
        for char in text:
            if char in ''.join(self.iast_diacritics.values()):
                info['iast_characters'] += 1
                info['character_breakdown'][char] = info['character_breakdown'].get(char, 0) + 1
        
        # Calculate compliance score
        issues = self.validate_iast(text)
        info['issues'] = issues
        info['compliance_score'] = max(0.0, 1.0 - (len(issues) * 0.1))
        
        # Generate suggestions
        info['suggestions'] = self._generate_suggestions(text, issues)
        
        return info

    def _generate_suggestions(self, text: str, issues: List[str]) -> List[str]:
        """Generate suggestions for improving IAST compliance."""
        suggestions = []
        
        # Analyze common patterns and suggest improvements
        if "Mixed transliteration schemes" in ' '.join(issues):
            suggestions.append("Consider converting all ASCII approximations to proper IAST diacritics")
        
        if any("Double" in issue for issue in issues):
            suggestions.append("Review vowel length markings for accuracy")
        
        if any("capitalization" in issue for issue in issues):
            suggestions.append("Establish consistent capitalization rules for proper nouns")
        
        # Check for common ASCII to IAST conversions
        if re.search(r'\b(aa|ii|uu)\b', text):
            suggestions.append("Convert ASCII vowel approximations (aa→ā, ii→ī, uu→ū)")
        
        if re.search(r'\b(sh|Sh)\b', text):
            suggestions.append("Convert sibilant approximations (sh→ś, Sh→ṣ)")
        
        return suggestions

    def batch_transliterate(self, texts: List[str], source_standard: TransliterationStandard = None) -> List[TransliterationResult]:
        """
        Transliterate multiple texts to IAST.
        
        Args:
            texts: List of texts to transliterate
            source_standard: Source standard (auto-detect if None)
            
        Returns:
            List of transliteration results
        """
        results = []
        
        for text in texts:
            result = self.transliterate_to_iast(text, source_standard)
            results.append(result)
        
        return results