"""
Scripture IAST Formatter Module.

This module extends the IAST transliteration system specifically for scriptural
verses with verse-specific formatting rules and metadata handling.
"""

import re
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum
import logging

from utils.logger_config import get_logger
from utils.iast_transliterator import IASTTransliterator, TransliterationResult
from .canonical_text_manager import CanonicalVerse, ScriptureSource


class VerseFormatting(Enum):
    """Verse formatting styles."""
    TRADITIONAL = "traditional"  # With verse markers
    ACADEMIC = "academic"        # Clean academic format
    LITURGICAL = "liturgical"    # For recitation
    SCHOLARLY = "scholarly"      # With detailed notation


@dataclass
class VerseMetadata:
    """Metadata for verse IAST formatting."""
    source: ScriptureSource
    chapter: int
    verse: int
    title: Optional[str] = None
    commentary_reference: Optional[str] = None
    variant_readings: List[str] = None
    
    def __post_init__(self):
        if self.variant_readings is None:
            self.variant_readings = []


@dataclass
class ScriptureIASTResult:
    """Result of scripture IAST formatting."""
    original_text: str
    formatted_text: str
    transliteration_result: TransliterationResult
    verse_metadata: VerseMetadata
    formatting_style: VerseFormatting
    consistency_issues: List[str]
    academic_compliance: float


class ScriptureIASTFormatter:
    """
    Specialized IAST formatter for scriptural verses.
    
    Extends the standard IAST transliterator with verse-specific formatting
    rules, metadata integration, and academic consistency validation.
    """
    
    def __init__(self, iast_transliterator: IASTTransliterator = None, config: Dict = None):
        """
        Initialize the Scripture IAST Formatter.
        
        Args:
            iast_transliterator: Base IAST transliterator
            config: Configuration parameters
        """
        self.logger = get_logger(__name__)
        self.iast_transliterator = iast_transliterator or IASTTransliterator(strict_mode=True)
        
        # Configuration
        self.config = config or {}
        self.default_formatting = VerseFormatting(
            self.config.get('default_formatting', 'academic')
        )
        self.preserve_verse_markers = self.config.get('preserve_verse_markers', True)
        self.add_source_attribution = self.config.get('add_source_attribution', True)
        self.validate_consistency = self.config.get('validate_consistency', True)
        
        # Initialize verse-specific formatting rules
        self._init_verse_formatting_rules()
        
        # Initialize consistency validation patterns
        self._init_consistency_patterns()
        
        self.logger.info("Scripture IAST formatter initialized")
    
    def _init_verse_formatting_rules(self) -> None:
        """Initialize verse-specific IAST formatting rules."""
        self.verse_formatting_rules = {
            # Verse structure markers
            'verse_markers': {
                '||': ' ||',  # Ensure space before verse end marker
                '।': ' ।',    # Ensure space before half-verse marker
                '|': ' |',    # Alternative verse marker
            },
            
            # Punctuation standardization
            'punctuation': {
                r'\s*\|\|\s*': ' ||',  # Standardize verse end markers
                r'\s*।\s*': ' ।',      # Standardize half-verse markers
                r'\s*\|\s*': ' |',     # Standardize alternative markers
            },
            
            # Sandhi and word boundary rules
            'word_boundaries': {
                r"(\w)'(\w)": r"\1'\2",  # Standardize apostrophes in sandhi
                r"(\w)-(\w)": r"\1-\2",  # Standardize hyphens
                r'\s+': ' ',             # Normalize whitespace
            },
            
            # Capitalization rules for verses
            'capitalization': {
                'sentence_start': True,    # Capitalize first word
                'proper_nouns': True,      # Capitalize proper nouns
                'verse_beginning': True,   # Capitalize verse beginning
            }
        }
    
    def _init_consistency_patterns(self) -> None:
        """Initialize patterns for consistency validation."""
        self.consistency_patterns = {
            # Common inconsistencies in Sanskrit transliteration
            'vowel_length': [
                (r'\ba\b', r'\bā\b', "Short 'a' vs long 'ā'"),
                (r'\bi\b', r'\bī\b', "Short 'i' vs long 'ī'"),
                (r'\bu\b', r'\bū\b', "Short 'u' vs long 'ū'"),
            ],
            
            # Retroflex consistency
            'retroflexes': [
                (r'\bt\b', r'\bṭ\b', "Dental 't' vs retroflex 'ṭ'"),
                (r'\bd\b', r'\bḍ\b', "Dental 'd' vs retroflex 'ḍ'"),
                (r'\bn\b', r'\bṇ\b', "Dental 'n' vs retroflex 'ṇ'"),
            ],
            
            # Aspirate consistency
            'aspirates': [
                (r'\bkh\b', r'\bkh\b', "Aspirated 'kh' consistency"),
                (r'\bgh\b', r'\bgh\b', "Aspirated 'gh' consistency"),
                (r'\bth\b', r'\bth\b', "Aspirated 'th' consistency"),
                (r'\bdh\b', r'\bdh\b', "Aspirated 'dh' consistency"),
            ],
            
            # Sibilant consistency
            'sibilants': [
                (r'\bś\b', r'\bṣ\b', "Palatal 'ś' vs retroflex 'ṣ'"),
                (r'\bs\b', r'\bś\b', "Dental 's' vs palatal 'ś'"),
            ]
        }
    
    def format_canonical_verse(self, canonical_verse: CanonicalVerse, 
                              formatting: VerseFormatting = None) -> ScriptureIASTResult:
        """
        Format a canonical verse with proper IAST standards.
        
        Args:
            canonical_verse: Canonical verse to format
            formatting: Formatting style to apply
            
        Returns:
            Complete IAST formatting result
        """
        formatting = formatting or self.default_formatting
        
        # Create verse metadata
        metadata = VerseMetadata(
            source=canonical_verse.source,
            chapter=canonical_verse.chapter,
            verse=canonical_verse.verse,
            title=getattr(canonical_verse, 'title', None)
        )
        
        # Apply IAST transliteration
        transliteration_result = self.iast_transliterator.transliterate_to_iast(
            canonical_verse.canonical_text
        )
        
        # Apply verse-specific formatting
        formatted_text = self._apply_verse_formatting(
            transliteration_result.transliterated_text,
            formatting,
            metadata
        )
        
        # Validate consistency
        consistency_issues = []
        if self.validate_consistency:
            consistency_issues = self._validate_verse_consistency(formatted_text)
        
        # Calculate academic compliance score
        academic_compliance = self._calculate_academic_compliance(
            formatted_text, transliteration_result, consistency_issues
        )
        
        return ScriptureIASTResult(
            original_text=canonical_verse.canonical_text,
            formatted_text=formatted_text,
            transliteration_result=transliteration_result,
            verse_metadata=metadata,
            formatting_style=formatting,
            consistency_issues=consistency_issues,
            academic_compliance=academic_compliance
        )
    
    def format_verse_with_metadata(self, verse_text: str, metadata: VerseMetadata,
                                  formatting: VerseFormatting = None) -> ScriptureIASTResult:
        """
        Format verse text with provided metadata.
        
        Args:
            verse_text: Verse text to format
            metadata: Verse metadata
            formatting: Formatting style
            
        Returns:
            Complete IAST formatting result
        """
        formatting = formatting or self.default_formatting
        
        # Apply IAST transliteration
        transliteration_result = self.iast_transliterator.transliterate_to_iast(verse_text)
        
        # Apply verse-specific formatting
        formatted_text = self._apply_verse_formatting(
            transliteration_result.transliterated_text,
            formatting,
            metadata
        )
        
        # Validate consistency
        consistency_issues = []
        if self.validate_consistency:
            consistency_issues = self._validate_verse_consistency(formatted_text)
        
        # Calculate academic compliance
        academic_compliance = self._calculate_academic_compliance(
            formatted_text, transliteration_result, consistency_issues
        )
        
        return ScriptureIASTResult(
            original_text=verse_text,
            formatted_text=formatted_text,
            transliteration_result=transliteration_result,
            verse_metadata=metadata,
            formatting_style=formatting,
            consistency_issues=consistency_issues,
            academic_compliance=academic_compliance
        )
    
    def _apply_verse_formatting(self, text: str, formatting: VerseFormatting,
                               metadata: VerseMetadata) -> str:
        """
        Apply verse-specific formatting rules.
        
        Args:
            text: Text to format
            formatting: Formatting style
            metadata: Verse metadata
            
        Returns:
            Formatted text
        """
        formatted = text
        
        # Apply punctuation standardization
        for pattern, replacement in self.verse_formatting_rules['punctuation'].items():
            formatted = re.sub(pattern, replacement, formatted)
        
        # Apply word boundary standardization
        for pattern, replacement in self.verse_formatting_rules['word_boundaries'].items():
            formatted = re.sub(pattern, replacement, formatted)
        
        # Apply formatting style specific rules
        if formatting == VerseFormatting.TRADITIONAL:
            formatted = self._apply_traditional_formatting(formatted, metadata)
        elif formatting == VerseFormatting.ACADEMIC:
            formatted = self._apply_academic_formatting(formatted, metadata)
        elif formatting == VerseFormatting.LITURGICAL:
            formatted = self._apply_liturgical_formatting(formatted, metadata)
        elif formatting == VerseFormatting.SCHOLARLY:
            formatted = self._apply_scholarly_formatting(formatted, metadata)
        
        # Apply capitalization rules
        formatted = self._apply_capitalization_rules(formatted)
        
        # Add source attribution if configured
        if self.add_source_attribution:
            formatted = self._add_source_attribution(formatted, metadata)
        
        return formatted.strip()
    
    def _apply_traditional_formatting(self, text: str, metadata: VerseMetadata) -> str:
        """Apply traditional verse formatting."""
        # Preserve traditional verse markers
        if self.preserve_verse_markers:
            # Ensure proper spacing around markers
            text = re.sub(r'\s*\|\|\s*', ' ||', text)
            text = re.sub(r'\s*।\s*', ' ।', text)
        
        # Traditional line breaks at half-verse
        if '।' in text:
            text = text.replace(' ।', ' ।\n')
        
        return text
    
    def _apply_academic_formatting(self, text: str, metadata: VerseMetadata) -> str:
        """Apply academic formatting (clean, standardized)."""
        # Clean formatting without extra line breaks
        text = re.sub(r'\s*\|\|\s*', ' ||', text)
        text = re.sub(r'\s*।\s*', ' ।', text)
        
        # Ensure consistent spacing
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def _apply_liturgical_formatting(self, text: str, metadata: VerseMetadata) -> str:
        """Apply liturgical formatting (for recitation)."""
        # Add pauses for recitation
        text = re.sub(r'\s*।\s*', ' । ', text)  # Longer pause at half-verse
        text = re.sub(r'\s*\|\|\s*', ' || ', text)  # Final pause
        
        # Ensure clear word boundaries
        text = re.sub(r"(\w)'(\w)", r"\1 ' \2", text)  # Space around sandhi
        
        return text
    
    def _apply_scholarly_formatting(self, text: str, metadata: VerseMetadata) -> str:
        """Apply scholarly formatting with detailed notation."""
        # Academic formatting as base
        formatted = self._apply_academic_formatting(text, metadata)
        
        # Add scholarly annotations (could be expanded)
        # For now, ensure very precise punctuation
        formatted = re.sub(r'\s*\|\|\s*$', ' ||', formatted)  # Verse end
        formatted = re.sub(r'\s*।\s*', ' ।', formatted)  # Half-verse
        
        return formatted
    
    def _apply_capitalization_rules(self, text: str) -> str:
        """Apply verse-specific capitalization rules."""
        if not text:
            return text
        
        # Capitalize first letter
        if self.verse_formatting_rules['capitalization']['sentence_start']:
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        # Capitalize after verse markers (new sentence)
        text = re.sub(r'(\|\||।)\s+([a-z])', lambda m: m.group(1) + ' ' + m.group(2).upper(), text)
        
        return text
    
    def _add_source_attribution(self, text: str, metadata: VerseMetadata) -> str:
        """Add source attribution to formatted text."""
        attribution = f"{metadata.source.value.replace('_', ' ').title()} {metadata.chapter}.{metadata.verse}"
        return text + f" — {attribution}"
    
    def _validate_verse_consistency(self, text: str) -> List[str]:
        """
        Validate IAST consistency in verse text.
        
        Args:
            text: Formatted verse text
            
        Returns:
            List of consistency issues found
        """
        issues = []
        
        for category, patterns in self.consistency_patterns.items():
            for pattern1, pattern2, description in patterns:
                # Check if both patterns exist in text (potential inconsistency)
                matches1 = re.findall(pattern1, text, re.IGNORECASE)
                matches2 = re.findall(pattern2, text, re.IGNORECASE)
                
                if matches1 and matches2:
                    issues.append(f"{description}: found both variants in same verse")
        
        # Check for mixing of transliteration schemes
        if self._has_mixed_schemes(text):
            issues.append("Mixed transliteration schemes detected")
        
        # Check for improper diacritics
        if self._has_improper_diacritics(text):
            issues.append("Improper or missing diacritics detected")
        
        return issues
    
    def _has_mixed_schemes(self, text: str) -> bool:
        """Check if text has mixed transliteration schemes."""
        # IAST indicators
        has_iast = bool(re.search(r'[āīūṛṝḷḹēōṃḥṅñṭḍṇśṣ]', text))
        
        # Harvard-Kyoto indicators
        has_hk = bool(re.search(r'[AIURM](?![a-z])', text))
        
        # ITRANS indicators  
        has_itrans = bool(re.search(r'(aa|ii|uu|R\^|L\^|M\^|H\^)', text))
        
        # Mixed if more than one scheme detected
        scheme_count = sum([has_iast, has_hk, has_itrans])
        return scheme_count > 1
    
    def _has_improper_diacritics(self, text: str) -> bool:
        """Check for improper diacritics usage."""
        # Look for common mistakes
        improper_patterns = [
            r'[aeiou][aeiou]',  # Double vowels (might be incorrect)
            r'[āīū][aeiou]',    # Mixed long/short vowels
            r'[^aeiouāīūṛṝḷḹēōṃḥṅñṭḍṇśṣ\s\|\।\'\-]',  # Non-IAST characters
        ]
        
        for pattern in improper_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _calculate_academic_compliance(self, text: str, transliteration_result: TransliterationResult,
                                     consistency_issues: List[str]) -> float:
        """
        Calculate academic compliance score.
        
        Args:
            text: Formatted text
            transliteration_result: Base transliteration result
            consistency_issues: Found consistency issues
            
        Returns:
            Compliance score (0.0 to 1.0)
        """
        base_score = transliteration_result.confidence
        
        # Penalize for consistency issues
        consistency_penalty = len(consistency_issues) * 0.1
        
        # Bonus for proper IAST characters
        has_proper_iast = bool(re.search(r'[āīūṛṝḷḹēōṃḥṅñṭḍṇśṣ]', text))
        iast_bonus = 0.1 if has_proper_iast else 0.0
        
        # Bonus for proper verse structure
        has_verse_structure = bool(re.search(r'[।|]', text))
        structure_bonus = 0.1 if has_verse_structure else 0.0
        
        # Calculate final score
        compliance_score = base_score + iast_bonus + structure_bonus - consistency_penalty
        
        return max(0.0, min(1.0, compliance_score))
    
    def validate_cross_verse_consistency(self, verses: List[ScriptureIASTResult]) -> Dict[str, Any]:
        """
        Validate consistency across multiple verses.
        
        Args:
            verses: List of formatted verses
            
        Returns:
            Cross-verse consistency report
        """
        if not verses:
            return {'is_consistent': True, 'issues': []}
        
        issues = []
        
        # Check for consistent transliteration patterns
        all_words = set()
        word_variants = {}
        
        for verse in verses:
            words = verse.formatted_text.lower().split()
            for word in words:
                # Clean word of punctuation
                clean_word = re.sub(r'[।|\|\s\'\-]', '', word)
                if clean_word:
                    all_words.add(clean_word)
                    
                    # Track variants
                    base_word = re.sub(r'[āīūṛṝḷḹēōṃḥṅñṭḍṇśṣ]', lambda m: m.group(0)[0], clean_word)
                    if base_word not in word_variants:
                        word_variants[base_word] = set()
                    word_variants[base_word].add(clean_word)
        
        # Find inconsistent variants
        for base, variants in word_variants.items():
            if len(variants) > 1:
                issues.append(f"Inconsistent transliterations for '{base}': {list(variants)}")
        
        # Check formatting consistency
        formatting_styles = [v.formatting_style for v in verses]
        if len(set(formatting_styles)) > 1:
            issues.append(f"Mixed formatting styles: {set(formatting_styles)}")
        
        # Check source consistency
        sources = [v.verse_metadata.source for v in verses]
        if len(set(sources)) > 1:
            issues.append("Mixed scripture sources in verse collection")
        
        return {
            'is_consistent': len(issues) == 0,
            'issues': issues,
            'total_verses': len(verses),
            'unique_words': len(all_words),
            'variant_count': len([v for variants in word_variants.values() if len(variants) > 1])
        }