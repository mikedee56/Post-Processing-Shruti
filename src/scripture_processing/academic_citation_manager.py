"""
Academic Citation Manager for Scripture Intelligence Enhancement.

This module implements comprehensive academic citation standards for scriptural
content, ensuring proper formatting for Sanskrit transliteration and verse
citations according to research publication standards.

Story 4.5: Scripture Intelligence Enhancement - Task 2 Implementation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from enum import Enum
from pathlib import Path
import logging
import re
import datetime

from scripture_processing.canonical_text_manager import CanonicalTextManager, VerseCandidate, ScriptureSource
from scripture_processing.scripture_iast_formatter import ScriptureIASTFormatter, VerseFormatting
from utils.logger_config import get_logger


class CitationStyle(Enum):
    """Academic citation styles supported."""
    MLA = "mla"
    APA = "apa"
    CHICAGO = "chicago"
    HARVARD = "harvard"
    CUSTOM_ACADEMIC = "custom_academic"
    INDOLOGICAL_STANDARD = "indological_standard"


class TransliterationStandard(Enum):
    """Transliteration standards for Sanskrit citations."""
    IAST = "iast"  # International Alphabet of Sanskrit Transliteration
    HARVARD_KYOTO = "harvard_kyoto"
    ITRANS = "itrans"
    DEVANAGARI = "devanagari"
    SIMPLIFIED = "simplified"


class CitationValidationLevel(Enum):
    """Citation validation strictness levels."""
    BASIC = "basic"
    STANDARD = "standard"
    RIGOROUS = "rigorous"
    PUBLICATION_GRADE = "publication_grade"


@dataclass
class CitationFormat:
    """Configuration for academic citation formatting."""
    
    # Citation style configuration
    style: CitationStyle = CitationStyle.INDOLOGICAL_STANDARD
    transliteration_standard: TransliterationStandard = TransliterationStandard.IAST
    validation_level: CitationValidationLevel = CitationValidationLevel.STANDARD
    
    # Format options
    include_original_text: bool = True
    include_transliteration: bool = True
    include_translation: bool = False
    include_commentary: bool = False
    
    # Academic requirements
    require_verse_numbers: bool = True
    require_source_attribution: bool = True
    require_edition_information: bool = False
    include_publication_details: bool = False
    
    # Formatting preferences
    use_parenthetical_citations: bool = True
    use_footnote_style: bool = False
    include_page_numbers: bool = False
    scholarly_abbreviations: bool = True


@dataclass
class AcademicCitation:
    """Represents a complete academic citation for a scriptural verse."""
    
    # Core citation elements
    verse_candidate: VerseCandidate
    citation_text: str
    original_passage: str
    
    # Academic metadata
    citation_style: CitationStyle
    transliteration_standard: TransliterationStandard
    validation_level: CitationValidationLevel
    
    # Citation components
    source_abbreviation: str = ""
    chapter_verse_reference: str = ""
    transliterated_text: str = ""
    original_script_text: str = ""
    translation_text: str = ""
    
    # Publication metadata
    edition_information: str = ""
    publication_details: str = ""
    scholarly_notes: List[str] = field(default_factory=list)
    
    # Validation results
    is_valid: bool = True
    validation_warnings: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    
    # Academic compliance
    meets_publication_standards: bool = False
    peer_review_ready: bool = False
    requires_verification: bool = True


@dataclass
class CitationValidationResult:
    """Result of academic citation validation."""
    
    citation: AcademicCitation
    overall_valid: bool = False
    validation_score: float = 0.0
    
    # Validation categories
    format_validation: Dict[str, bool] = field(default_factory=dict)
    content_validation: Dict[str, bool] = field(default_factory=dict)
    academic_compliance: Dict[str, bool] = field(default_factory=dict)
    
    # Recommendations
    improvement_suggestions: List[str] = field(default_factory=list)
    critical_issues: List[str] = field(default_factory=list)
    best_practices: List[str] = field(default_factory=list)


class AcademicCitationManager:
    """
    Academic Citation Manager for Scripture Intelligence Enhancement.
    
    Provides comprehensive academic citation management with support for
    multiple citation styles, transliteration standards, and validation
    levels for research publication readiness.
    """
    
    def __init__(
        self,
        canonical_manager: CanonicalTextManager,
        iast_formatter: Optional[ScriptureIASTFormatter] = None,
        default_format: Optional[CitationFormat] = None
    ):
        """
        Initialize the Academic Citation Manager.
        
        Args:
            canonical_manager: Canonical text management system
            iast_formatter: Optional IAST formatting system
            default_format: Default citation format configuration
        """
        self.logger = get_logger(__name__)
        self.canonical_manager = canonical_manager
        self.iast_formatter = iast_formatter or ScriptureIASTFormatter()
        self.default_format = default_format or CitationFormat()
        
        # Initialize citation templates and standards
        self._initialize_citation_templates()
        self._initialize_abbreviation_systems()
        self._initialize_validation_rules()
        
        # Performance tracking
        self.citation_stats = {
            'citations_generated': 0,
            'validations_performed': 0,
            'publication_grade_citations': 0,
            'average_validation_score': 0.0
        }
        
        self.logger.info("Academic Citation Manager initialized")
    
    def _initialize_citation_templates(self) -> None:
        """Initialize citation templates for different academic styles."""
        
        self.citation_templates = {
            CitationStyle.INDOLOGICAL_STANDARD: {
                'verse_citation': "{abbreviation} {chapter}.{verse}",
                'full_citation': "{source_title} {chapter}.{verse}",
                'with_text': "{abbreviation} {chapter}.{verse}: {transliterated_text}",
                'footnote': "{source_title}, {chapter}.{verse}."
            },
            CitationStyle.MLA: {
                'verse_citation': "({abbreviation} {chapter}.{verse})",
                'full_citation': "{author}. {source_title}. {chapter}.{verse}.",
                'with_text': '"{transliterated_text}" ({abbreviation} {chapter}.{verse})',
                'footnote': "{author}, {source_title}, {chapter}.{verse}."
            },
            CitationStyle.APA: {
                'verse_citation': "({abbreviation} {chapter}.{verse})",
                'full_citation': "{author} ({year}). {source_title}. {chapter}.{verse}.",
                'with_text': '"{transliterated_text}" ({abbreviation} {chapter}.{verse})',
                'footnote': "{author} ({year}). {source_title}, {chapter}.{verse}."
            },
            CitationStyle.CHICAGO: {
                'verse_citation': "({abbreviation} {chapter}.{verse})",
                'full_citation': "{author}, {source_title}, {chapter}.{verse}.",
                'with_text': '"{transliterated_text}" ({abbreviation} {chapter}.{verse})',
                'footnote': "{author}, {source_title}, {chapter}.{verse}."
            }
        }
    
    def _initialize_abbreviation_systems(self) -> None:
        """Initialize standard abbreviation systems for scriptural sources."""
        
        self.source_abbreviations = {
            ScriptureSource.BHAGAVAD_GITA: {
                'standard': 'BG',
                'scholarly': 'Bhag.',
                'full': 'Bhagavad Gītā',
                'devanagari': 'भगवद्गीता'
            },
            ScriptureSource.UPANISHADS: {
                'standard': 'Up',
                'scholarly': 'Upan.',
                'full': 'Upaniṣads',
                'devanagari': 'उपनिषद्'
            },
            ScriptureSource.YOGA_SUTRAS: {
                'standard': 'YS',
                'scholarly': 'Yoga.',
                'full': 'Yoga Sūtras',
                'devanagari': 'योग सूत्र'
            },
            ScriptureSource.VEDAS: {
                'standard': 'V',
                'scholarly': 'Ved.',
                'full': 'Vedas',
                'devanagari': 'वेद'
            },
            ScriptureSource.PURANAS: {
                'standard': 'P',
                'scholarly': 'Pur.',
                'full': 'Purāṇas',
                'devanagari': 'पुराण'
            }
        }
        
        # Traditional authorship and publication information
        self.source_metadata = {
            ScriptureSource.BHAGAVAD_GITA: {
                'traditional_author': 'Vyāsa',
                'part_of': 'Mahābhārata',
                'traditional_dating': 'c. 400 BCE - 400 CE',
                'modern_editions': ['Gita Press', 'Harvard Oriental Series']
            },
            ScriptureSource.UPANISHADS: {
                'traditional_author': 'Various Ṛṣis',
                'part_of': 'Vedic Literature',
                'traditional_dating': 'c. 800-200 BCE',
                'modern_editions': ['Ānandāśrama', 'Harvard Oriental Series']
            },
            ScriptureSource.YOGA_SUTRAS: {
                'traditional_author': 'Patañjali',
                'part_of': 'Yoga Darśana',
                'traditional_dating': 'c. 400 CE',
                'modern_editions': ['Yoga Institute', 'Harvard Oriental Series']
            },
            ScriptureSource.VEDAS: {
                'traditional_author': 'Various Ṛṣis',
                'part_of': 'Śruti Literature',
                'traditional_dating': 'c. 1500-500 BCE',
                'modern_editions': ['Vaidik Saṃśodhan Maṇḍal', 'Harvard Oriental Series']
            },
            ScriptureSource.PURANAS: {
                'traditional_author': 'Vyāsa and others',
                'part_of': 'Smṛti Literature', 
                'traditional_dating': 'c. 300-1500 CE',
                'modern_editions': ['Motilal Banarsidass', 'Sanskrit Series']
            }
        }
    
    def _initialize_validation_rules(self) -> None:
        """Initialize validation rules for different strictness levels."""
        
        self.validation_rules = {
            CitationValidationLevel.BASIC: {
                'require_source': True,
                'require_reference': True,
                'require_formatting': False,
                'require_transliteration': False,
                'min_validation_score': 0.5
            },
            CitationValidationLevel.STANDARD: {
                'require_source': True,
                'require_reference': True,
                'require_formatting': True,
                'require_transliteration': True,
                'require_consistency': True,
                'min_validation_score': 0.7
            },
            CitationValidationLevel.RIGOROUS: {
                'require_source': True,
                'require_reference': True,
                'require_formatting': True,
                'require_transliteration': True,
                'require_consistency': True,
                'require_scholarly_abbreviations': True,
                'require_complete_metadata': True,
                'min_validation_score': 0.85
            },
            CitationValidationLevel.PUBLICATION_GRADE: {
                'require_source': True,
                'require_reference': True,
                'require_formatting': True,
                'require_transliteration': True,
                'require_consistency': True,
                'require_scholarly_abbreviations': True,
                'require_complete_metadata': True,
                'require_edition_information': True,
                'require_peer_review_standards': True,
                'min_validation_score': 0.95
            }
        }
    
    def generate_citation(
        self,
        verse_candidate: VerseCandidate,
        original_passage: str,
        citation_format: Optional[CitationFormat] = None
    ) -> AcademicCitation:
        """
        Generate academic citation for a scriptural verse.
        
        Args:
            verse_candidate: Canonical verse to cite
            original_passage: Original text passage
            citation_format: Optional citation format (uses default if None)
            
        Returns:
            Complete academic citation
        """
        format_config = citation_format or self.default_format
        
        try:
            citation = AcademicCitation(
                verse_candidate=verse_candidate,
                citation_text="",
                original_passage=original_passage,
                citation_style=format_config.style,
                transliteration_standard=format_config.transliteration_standard,
                validation_level=format_config.validation_level
            )
            
            # Generate citation components
            self._generate_source_abbreviation(citation, format_config)
            self._generate_chapter_verse_reference(citation, format_config)
            self._generate_transliterated_text(citation, format_config)
            self._generate_original_script_text(citation, format_config)
            self._generate_publication_metadata(citation, format_config)
            
            # Assemble final citation text
            self._assemble_citation_text(citation, format_config)
            
            # Update statistics
            self.citation_stats['citations_generated'] += 1
            
            self.logger.info(
                f"Generated {format_config.style.value} citation for "
                f"{verse_candidate.source.value} {verse_candidate.chapter}.{verse_candidate.verse}"
            )
            
        except Exception as e:
            self.logger.error(f"Error generating citation: {e}")
            citation.is_valid = False
            citation.validation_errors.append(f"Citation generation error: {str(e)}")
        
        return citation
    
    def _generate_source_abbreviation(
        self,
        citation: AcademicCitation,
        format_config: CitationFormat
    ) -> None:
        """Generate appropriate source abbreviation."""
        
        source = citation.verse_candidate.source
        abbreviations = self.source_abbreviations.get(source, {})
        
        if format_config.scholarly_abbreviations:
            citation.source_abbreviation = abbreviations.get('scholarly', source.value)
        else:
            citation.source_abbreviation = abbreviations.get('standard', source.value)
    
    def _generate_chapter_verse_reference(
        self,
        citation: AcademicCitation,
        format_config: CitationFormat
    ) -> None:
        """Generate chapter and verse reference."""
        
        verse = citation.verse_candidate
        citation.chapter_verse_reference = f"{verse.chapter}.{verse.verse}"
    
    def _generate_transliterated_text(
        self,
        citation: AcademicCitation,
        format_config: CitationFormat
    ) -> None:
        """Generate transliterated text according to specified standard."""
        
        if not format_config.include_transliteration:
            return
        
        try:
            canonical_text = citation.verse_candidate.canonical_text
            
            if format_config.transliteration_standard == TransliterationStandard.IAST:
                # Use existing IAST formatter
                iast_result = self.iast_formatter.iast_transliterator.transliterate_to_iast(
                    canonical_text
                )
                citation.transliterated_text = iast_result.transliterated_text
            else:
                # For other standards, use simplified transliteration
                citation.transliterated_text = self._apply_transliteration_standard(
                    canonical_text, format_config.transliteration_standard
                )
                
        except Exception as e:
            self.logger.error(f"Error in transliteration: {e}")
            citation.validation_warnings.append(f"Transliteration error: {str(e)}")
    
    def _apply_transliteration_standard(
        self,
        text: str,
        standard: TransliterationStandard
    ) -> str:
        """Apply specific transliteration standard to text."""
        
        # Simplified implementation - in practice, you'd use specialized libraries
        if standard == TransliterationStandard.HARVARD_KYOTO:
            # Convert IAST to Harvard-Kyoto conventions
            conversions = {
                'ā': 'A', 'ī': 'I', 'ū': 'U', 'ṛ': 'R', 'ḷ': 'L',
                'ē': 'e', 'ō': 'o', 'ṃ': 'M', 'ḥ': 'H',
                'ṅ': 'G', 'ñ': 'J', 'ṭ': 'T', 'ḍ': 'D', 'ṇ': 'N',
                'ś': 'z', 'ṣ': 'S'
            }
            result = text
            for iast, hk in conversions.items():
                result = result.replace(iast, hk)
            return result
        
        elif standard == TransliterationStandard.SIMPLIFIED:
            # Remove diacritical marks for simplified reading
            import unicodedata
            normalized = unicodedata.normalize('NFD', text)
            ascii_text = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
            return ascii_text
        
        elif standard == TransliterationStandard.DEVANAGARI:
            # Keep original if already in Devanagari, otherwise return as-is
            return text
        
        return text
    
    def _generate_original_script_text(
        self,
        citation: AcademicCitation,
        format_config: CitationFormat
    ) -> None:
        """Generate original script text if requested."""
        
        if format_config.include_original_text:
            # For now, use canonical text as original
            citation.original_script_text = citation.verse_candidate.canonical_text
    
    def _generate_publication_metadata(
        self,
        citation: AcademicCitation,
        format_config: CitationFormat
    ) -> None:
        """Generate publication metadata for academic citations."""
        
        source = citation.verse_candidate.source
        metadata = self.source_metadata.get(source, {})
        
        if format_config.require_edition_information:
            editions = metadata.get('modern_editions', [])
            if editions:
                citation.edition_information = f"Ed. {editions[0]}"
        
        if format_config.include_publication_details:
            author = metadata.get('traditional_author', 'Unknown')
            dating = metadata.get('traditional_dating', 'Unknown')
            citation.publication_details = f"{author}, {dating}"
    
    def _assemble_citation_text(
        self,
        citation: AcademicCitation,
        format_config: CitationFormat
    ) -> None:
        """Assemble the final citation text."""
        
        template_key = 'verse_citation'
        if format_config.include_transliteration and citation.transliterated_text:
            template_key = 'with_text'
        
        template = self.citation_templates[format_config.style][template_key]
        
        # Prepare substitution values
        substitutions = {
            'abbreviation': citation.source_abbreviation,
            'chapter': citation.verse_candidate.chapter,
            'verse': citation.verse_candidate.verse,
            'transliterated_text': citation.transliterated_text,
            'source_title': self.source_abbreviations.get(
                citation.verse_candidate.source, {}
            ).get('full', citation.verse_candidate.source.value),
            'author': self.source_metadata.get(
                citation.verse_candidate.source, {}
            ).get('traditional_author', 'Unknown'),
            'year': 'Traditional'  # Simplified for ancient texts
        }
        
        try:
            citation.citation_text = template.format(**substitutions)
            
            # Add edition information if required
            if citation.edition_information and format_config.require_edition_information:
                citation.citation_text += f" [{citation.edition_information}]"
                
        except KeyError as e:
            citation.validation_errors.append(f"Template formatting error: missing {e}")
            citation.citation_text = f"{citation.source_abbreviation} {citation.chapter_verse_reference}"
    
    def validate_citation(
        self,
        citation: AcademicCitation,
        validation_level: Optional[CitationValidationLevel] = None
    ) -> CitationValidationResult:
        """
        Validate academic citation according to specified standards.
        
        Args:
            citation: Citation to validate
            validation_level: Optional validation level (uses citation's level if None)
            
        Returns:
            Comprehensive validation result
        """
        level = validation_level or citation.validation_level
        rules = self.validation_rules[level]
        
        validation_result = CitationValidationResult(citation=citation)
        
        try:
            # Format validation
            self._validate_citation_format(citation, validation_result, rules)
            
            # Content validation
            self._validate_citation_content(citation, validation_result, rules)
            
            # Academic compliance validation
            self._validate_academic_compliance(citation, validation_result, rules)
            
            # Calculate overall validation score
            self._calculate_validation_score(validation_result, rules)
            
            # Update citation status based on validation
            self._update_citation_status(citation, validation_result, rules)
            
            # Update statistics
            self.citation_stats['validations_performed'] += 1
            if citation.meets_publication_standards:
                self.citation_stats['publication_grade_citations'] += 1
            
            # Update average validation score
            current_avg = self.citation_stats['average_validation_score']
            total_validations = self.citation_stats['validations_performed']
            new_avg = ((current_avg * (total_validations - 1)) + validation_result.validation_score) / total_validations
            self.citation_stats['average_validation_score'] = new_avg
            
        except Exception as e:
            self.logger.error(f"Error in citation validation: {e}")
            validation_result.critical_issues.append(f"Validation error: {str(e)}")
            validation_result.overall_valid = False
        
        return validation_result
    
    def _validate_citation_format(
        self,
        citation: AcademicCitation,
        validation_result: CitationValidationResult,
        rules: Dict[str, Any]
    ) -> None:
        """Validate citation format according to academic standards."""
        
        format_checks = {
            'has_source_abbreviation': bool(citation.source_abbreviation),
            'has_chapter_verse_reference': bool(citation.chapter_verse_reference),
            'has_citation_text': bool(citation.citation_text),
            'proper_formatting': self._check_citation_formatting(citation),
            'consistent_style': self._check_style_consistency(citation)
        }
        
        if rules.get('require_transliteration', False):
            format_checks['has_transliteration'] = bool(citation.transliterated_text)
        
        if rules.get('require_scholarly_abbreviations', False):
            format_checks['scholarly_abbreviations'] = self._check_scholarly_abbreviations(citation)
        
        validation_result.format_validation = format_checks
        
        # Generate recommendations for failed checks
        for check, passed in format_checks.items():
            if not passed:
                validation_result.improvement_suggestions.append(
                    f"Format improvement needed: {check.replace('_', ' ')}"
                )
    
    def _validate_citation_content(
        self,
        citation: AcademicCitation,
        validation_result: CitationValidationResult,
        rules: Dict[str, Any]
    ) -> None:
        """Validate citation content accuracy and completeness."""
        
        content_checks = {
            'valid_verse_reference': self._validate_verse_reference(citation),
            'canonical_text_present': bool(citation.verse_candidate.canonical_text),
            'transliteration_accuracy': self._check_transliteration_accuracy(citation),
            'source_metadata_complete': self._check_source_metadata(citation)
        }
        
        if rules.get('require_complete_metadata', False):
            content_checks['complete_metadata'] = bool(
                citation.publication_details or citation.edition_information
            )
        
        validation_result.content_validation = content_checks
        
        # Generate recommendations for content issues
        for check, passed in content_checks.items():
            if not passed:
                validation_result.improvement_suggestions.append(
                    f"Content improvement needed: {check.replace('_', ' ')}"
                )
    
    def _validate_academic_compliance(
        self,
        citation: AcademicCitation,
        validation_result: CitationValidationResult,
        rules: Dict[str, Any]
    ) -> None:
        """Validate academic compliance standards."""
        
        compliance_checks = {
            'meets_style_guidelines': self._check_style_guidelines_compliance(citation),
            'proper_source_attribution': bool(citation.source_abbreviation),
            'consistent_transliteration': self._check_transliteration_consistency(citation),
            'academic_formatting': self._check_academic_formatting(citation)
        }
        
        if rules.get('require_peer_review_standards', False):
            compliance_checks['peer_review_ready'] = self._check_peer_review_standards(citation)
        
        if rules.get('require_edition_information', False):
            compliance_checks['has_edition_info'] = bool(citation.edition_information)
        
        validation_result.academic_compliance = compliance_checks
        
        # Generate best practices recommendations
        for check, passed in compliance_checks.items():
            if not passed:
                validation_result.best_practices.append(
                    f"Academic standard: {check.replace('_', ' ')}"
                )
    
    def _calculate_validation_score(
        self,
        validation_result: CitationValidationResult,
        rules: Dict[str, Any]
    ) -> None:
        """Calculate overall validation score."""
        
        all_checks = {
            **validation_result.format_validation,
            **validation_result.content_validation,
            **validation_result.academic_compliance
        }
        
        if all_checks:
            passed_checks = sum(all_checks.values())
            total_checks = len(all_checks)
            validation_result.validation_score = passed_checks / total_checks
        else:
            validation_result.validation_score = 0.0
        
        # Determine overall validity
        min_score = rules.get('min_validation_score', 0.7)
        validation_result.overall_valid = validation_result.validation_score >= min_score
    
    def _update_citation_status(
        self,
        citation: AcademicCitation,
        validation_result: CitationValidationResult,
        rules: Dict[str, Any]
    ) -> None:
        """Update citation status based on validation results."""
        
        citation.is_valid = validation_result.overall_valid
        citation.validation_errors = validation_result.critical_issues
        citation.validation_warnings = validation_result.improvement_suggestions
        
        # Determine publication readiness
        publication_threshold = 0.95
        peer_review_threshold = 0.85
        
        if validation_result.validation_score >= publication_threshold:
            citation.meets_publication_standards = True
            citation.peer_review_ready = True
            citation.requires_verification = False
        elif validation_result.validation_score >= peer_review_threshold:
            citation.peer_review_ready = True
            citation.requires_verification = True
        else:
            citation.requires_verification = True
    
    # Helper validation methods
    def _check_citation_formatting(self, citation: AcademicCitation) -> bool:
        """Check if citation follows proper formatting conventions."""
        text = citation.citation_text
        return bool(text and len(text.strip()) > 5 and citation.chapter_verse_reference in text)
    
    def _check_style_consistency(self, citation: AcademicCitation) -> bool:
        """Check if citation is consistent with specified style."""
        style = citation.citation_style
        text = citation.citation_text
        
        if style == CitationStyle.INDOLOGICAL_STANDARD:
            return bool(citation.source_abbreviation in text and '.' in citation.chapter_verse_reference)
        return True  # Simplified check for other styles
    
    def _check_scholarly_abbreviations(self, citation: AcademicCitation) -> bool:
        """Check if scholarly abbreviations are used correctly."""
        source = citation.verse_candidate.source
        expected_abbreviations = self.source_abbreviations.get(source, {})
        
        return citation.source_abbreviation in [
            expected_abbreviations.get('scholarly', ''),
            expected_abbreviations.get('standard', '')
        ]
    
    def _validate_verse_reference(self, citation: AcademicCitation) -> bool:
        """Validate that verse reference is accurate."""
        verse = citation.verse_candidate
        expected_ref = f"{verse.chapter}.{verse.verse}"
        return citation.chapter_verse_reference == expected_ref
    
    def _check_transliteration_accuracy(self, citation: AcademicCitation) -> bool:
        """Check transliteration accuracy (simplified)."""
        if not citation.transliterated_text:
            return True  # Not required, so valid
        
        # Basic check for transliteration markers
        has_diacritics = bool(re.search(r'[āīūṛḷēōṃḥṅñṭḍṇśṣ]', citation.transliterated_text))
        return has_diacritics or citation.transliteration_standard != TransliterationStandard.IAST
    
    def _check_source_metadata(self, citation: AcademicCitation) -> bool:
        """Check if source metadata is complete."""
        source = citation.verse_candidate.source
        return source in self.source_metadata
    
    def _check_style_guidelines_compliance(self, citation: AcademicCitation) -> bool:
        """Check compliance with style guidelines."""
        return bool(citation.citation_text and citation.source_abbreviation)
    
    def _check_transliteration_consistency(self, citation: AcademicCitation) -> bool:
        """Check transliteration consistency."""
        if not citation.transliterated_text:
            return True
        
        # Simple consistency check - no mixed standards
        has_iast = bool(re.search(r'[āīūṛḷēōṃḥ]', citation.transliterated_text))
        has_itrans = bool(re.search(r'[AIURLM]', citation.transliterated_text))
        
        return not (has_iast and has_itrans)  # Not both standards mixed
    
    def _check_academic_formatting(self, citation: AcademicCitation) -> bool:
        """Check academic formatting standards."""
        text = citation.citation_text
        
        # Check for proper punctuation and spacing
        has_proper_spacing = not bool(re.search(r'\s{2,}', text))  # No double spaces
        has_punctuation = bool(re.search(r'[.:]', text))
        
        return has_proper_spacing and has_punctuation
    
    def _check_peer_review_standards(self, citation: AcademicCitation) -> bool:
        """Check if citation meets peer review publication standards."""
        requirements = [
            bool(citation.source_abbreviation),
            bool(citation.chapter_verse_reference),
            bool(citation.transliterated_text),
            citation.validation_level in [
                CitationValidationLevel.RIGOROUS,
                CitationValidationLevel.PUBLICATION_GRADE
            ]
        ]
        
        return all(requirements)
    
    def format_bibliography_entry(
        self,
        citation: AcademicCitation,
        include_full_details: bool = True
    ) -> str:
        """
        Format a complete bibliography entry for the citation.
        
        Args:
            citation: Citation to format
            include_full_details: Whether to include full publication details
            
        Returns:
            Formatted bibliography entry
        """
        source = citation.verse_candidate.source
        metadata = self.source_metadata.get(source, {})
        abbreviations = self.source_abbreviations.get(source, {})
        
        if citation.citation_style == CitationStyle.INDOLOGICAL_STANDARD:
            title = abbreviations.get('full', source.value)
            
            if include_full_details:
                author = metadata.get('traditional_author', 'Unknown')
                dating = metadata.get('traditional_dating', 'Unknown')
                return f"{author}. {title}. {dating}."
            else:
                return f"{title}."
        
        # For other styles, use similar patterns adapted to their conventions
        return f"{abbreviations.get('full', source.value)}."
    
    def generate_citation_suggestions(
        self,
        verse_candidate: VerseCandidate,
        target_style: CitationStyle = None
    ) -> List[str]:
        """
        Generate multiple citation format suggestions.
        
        Args:
            verse_candidate: Verse to generate citations for
            target_style: Optional target style (generates all if None)
            
        Returns:
            List of citation format suggestions
        """
        suggestions = []
        
        styles_to_generate = [target_style] if target_style else list(CitationStyle)
        
        for style in styles_to_generate:
            if style is None:
                continue
                
            format_config = CitationFormat(style=style)
            citation = self.generate_citation(verse_candidate, "", format_config)
            
            if citation.is_valid:
                suggestions.append(citation.citation_text)
        
        return suggestions
    
    def get_citation_statistics(self) -> Dict[str, Any]:
        """Get citation management statistics."""
        
        return {
            'citation_generation_stats': self.citation_stats.copy(),
            'supported_styles': [style.value for style in CitationStyle],
            'supported_standards': [std.value for std in TransliterationStandard],
            'validation_levels': [level.value for level in CitationValidationLevel],
            'source_coverage': list(self.source_abbreviations.keys()),
            'configuration': {
                'default_style': self.default_format.style.value,
                'default_transliteration': self.default_format.transliteration_standard.value,
                'default_validation_level': self.default_format.validation_level.value
            }
        }


def create_academic_citation_manager(
    canonical_manager: CanonicalTextManager,
    default_style: CitationStyle = CitationStyle.INDOLOGICAL_STANDARD,
    default_transliteration: TransliterationStandard = TransliterationStandard.IAST,
    validation_level: CitationValidationLevel = CitationValidationLevel.STANDARD
) -> AcademicCitationManager:
    """
    Factory function to create an AcademicCitationManager with specified defaults.
    
    Args:
        canonical_manager: Canonical text management system
        default_style: Default citation style
        default_transliteration: Default transliteration standard
        validation_level: Default validation level
        
    Returns:
        Configured AcademicCitationManager instance
    """
    default_format = CitationFormat(
        style=default_style,
        transliteration_standard=default_transliteration,
        validation_level=validation_level
    )
    
    return AcademicCitationManager(
        canonical_manager=canonical_manager,
        default_format=default_format
    )