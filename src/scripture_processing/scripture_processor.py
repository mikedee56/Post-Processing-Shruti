"""
Main Scripture Processor Module.

This module integrates all scripture processing components into a unified
system for verse identification, canonical text substitution, and IAST formatting.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import logging

from utils.logger_config import get_logger
from utils.srt_parser import SRTSegment
from scripture_processing.scripture_identifier import ScriptureIdentifier
from scripture_processing.canonical_text_manager import CanonicalTextManager
from scripture_processing.verse_substitution_engine import VerseSubstitutionEngine, SubstitutionResult
from scripture_processing.scripture_validator import ScriptureValidator
from scripture_processing.scripture_iast_formatter import ScriptureIASTFormatter, VerseFormatting
from scripture_processing.verse_selection_system import VerseSelectionSystem, SelectionStrategy
from scripture_processing.hybrid_matching_engine import HybridMatchingEngine, HybridPipelineConfig
from scripture_processing.asr_scripture_matcher import ASRScriptureMatcher, MatchingStrategy


@dataclass
class ScriptureProcessingResult:
    """Comprehensive result of scripture processing."""
    original_text: str
    processed_text: str
    verses_identified: int
    substitutions_made: int
    iast_formatted: bool
    validation_passed: bool
    processing_metadata: Dict[str, Any]
    detailed_results: Dict[str, Any]
    
    # Story 2.4.3 - Hybrid matching results
    hybrid_matching_used: bool = False
    hybrid_confidence: float = 0.0
    hybrid_pipeline_stages: List[str] = None


class ScriptureProcessor:
    """
    Unified Scripture Processing System.
    
    Integrates all scripture processing components to provide complete
    verse identification, canonical substitution, and IAST formatting
    capabilities for ASR transcripts.
    """
    
    def __init__(self, config: Dict = None, scripture_dir: Path = None):
        """
        Initialize the Scripture Processor with standardized error handling.
        
        Args:
            config: Configuration parameters
            scripture_dir: Directory containing scripture databases
        """
        # Import ErrorHandler locally for consistent import pattern
        from utils.error_handler import ErrorHandler
        from utils.exception_hierarchy import ProcessingError, ConfigurationError, DependencyError
        
        # Initialize error handler first for consistent error handling
        try:
            self._error_handler = ErrorHandler(component="ScriptureProcessor")
            self.logger = self._error_handler.logger
            
            self._error_handler.log_operation_start("scripture_processor_initialization", {
                'component': 'ScriptureProcessor',
                'config_provided': config is not None,
                'scripture_dir_provided': scripture_dir is not None
            })
            
            self.config = config or {}
            
            # Story 2.4.3 - Hybrid matching configuration
            self.enable_hybrid_matching = self.config.get('enable_hybrid_matching', False)
            self.hybrid_engine = None
            
            # Epic 5 - ASR Scripture Matching (Digital Dharma implementation)
            self.enable_asr_matching = self.config.get('enable_asr_matching', True)
            self.asr_matcher = None
            
            # Story 4.5 - Academic enhancement configuration
            self.enable_academic_enhancement = self.config.get('enable_academic_enhancement', False)
            self.academic_citation_manager = None
            self.publication_formatter = None
            self.academic_validator = None
            
            # Initialize components with error handling
            try:
                self.canonical_manager = CanonicalTextManager(
                    scripture_dir=scripture_dir,
                    config=self.config.get('canonical_manager', {})
                )
                self._error_handler.log_operation_success("canonical_manager_initialization", {
                    'scripture_dir': str(scripture_dir) if scripture_dir else 'default'
                })
            except Exception as e:
                error_details = self._error_handler.handle_processing_error("canonical_manager_initialization", e, {
                    'scripture_dir': str(scripture_dir) if scripture_dir else 'None'
                })
                raise DependencyError(f"Failed to initialize CanonicalTextManager: {error_details}")
            
            try:
                self.scripture_identifier = ScriptureIdentifier(
                    lexicon_manager=None,  # Will use default
                    config=self.config.get('scripture_identifier', {})
                )
                self._error_handler.log_operation_success("scripture_identifier_initialization", {})
            except Exception as e:
                error_details = self._error_handler.handle_processing_error("scripture_identifier_initialization", e, {})
                raise DependencyError(f"Failed to initialize ScriptureIdentifier: {error_details}")
            
            try:
                self.validator = ScriptureValidator(
                    config=self.config.get('validator', {})
                )
                self._error_handler.log_operation_success("scripture_validator_initialization", {})
            except Exception as e:
                error_details = self._error_handler.handle_processing_error("scripture_validator_initialization", e, {})
                raise DependencyError(f"Failed to initialize ScriptureValidator: {error_details}")
            
            try:
                self.substitution_engine = VerseSubstitutionEngine(
                    scripture_identifier=self.scripture_identifier,
                    canonical_manager=self.canonical_manager,
                    validator=self.validator,
                    config=self.config.get('substitution_engine', {})
                )
                self._error_handler.log_operation_success("verse_substitution_engine_initialization", {})
            except Exception as e:
                error_details = self._error_handler.handle_processing_error("verse_substitution_engine_initialization", e, {})
                raise DependencyError(f"Failed to initialize VerseSubstitutionEngine: {error_details}")
            
            try:
                self.iast_formatter = ScriptureIASTFormatter(
                    config=self.config.get('iast_formatter', {})
                )
                self._error_handler.log_operation_success("iast_formatter_initialization", {})
            except Exception as e:
                error_details = self._error_handler.handle_processing_error("iast_formatter_initialization", e, {})
                raise DependencyError(f"Failed to initialize ScriptureIASTFormatter: {error_details}")
            
            try:
                self.selection_system = VerseSelectionSystem(
                    canonical_manager=self.canonical_manager,
                    scripture_identifier=self.scripture_identifier,
                    validator=self.validator,
                    config=self.config.get('selection_system', {})
                )
                self._error_handler.log_operation_success("verse_selection_system_initialization", {})
            except Exception as e:
                error_details = self._error_handler.handle_processing_error("verse_selection_system_initialization", e, {})
                raise DependencyError(f"Failed to initialize VerseSelectionSystem: {error_details}")
            
            # Processing configuration
            try:
                self.enable_substitution = self.config.get('enable_substitution', True)
                self.enable_iast_formatting = self.config.get('enable_iast_formatting', True)
                self.enable_validation = self.config.get('enable_validation', True)
                
                formatting_style_str = self.config.get('formatting_style', 'academic')
                self.formatting_style = VerseFormatting(formatting_style_str)
                
                self._error_handler.log_operation_success("processing_configuration", {
                    'enable_substitution': self.enable_substitution,
                    'enable_iast_formatting': self.enable_iast_formatting,
                    'enable_validation': self.enable_validation,
                    'formatting_style': formatting_style_str
                })
            except Exception as e:
                error_details = self._error_handler.handle_processing_error("processing_configuration", e, {
                    'config_keys': list(self.config.keys())
                })
                raise ConfigurationError(f"Invalid processing configuration: {error_details}")
            
            # Initialize hybrid matching engine if enabled (Story 2.4.3)
            if self.enable_hybrid_matching:
                try:
                    hybrid_config = HybridPipelineConfig()
                    if 'hybrid_pipeline' in self.config:
                        # Update config with user settings
                        for key, value in self.config['hybrid_pipeline'].items():
                            if hasattr(hybrid_config, key):
                                setattr(hybrid_config, key, value)
                    
                    self.hybrid_engine = HybridMatchingEngine(
                        canonical_manager=self.canonical_manager,
                        config=hybrid_config,
                        cache_dir=scripture_dir / "hybrid_cache" if scripture_dir else None
                    )
                    
                    self._error_handler.log_operation_success("hybrid_matching_engine_initialization", {
                        'cache_dir': str(scripture_dir / "hybrid_cache") if scripture_dir else 'None'
                    })
                    
                except Exception as e:
                    error_details = self._error_handler.handle_processing_error("hybrid_matching_engine_initialization", e, {
                        'enable_hybrid_matching': self.enable_hybrid_matching,
                        'scripture_dir': str(scripture_dir) if scripture_dir else 'None'
                    })
                    self.logger.error(f"Failed to initialize hybrid matching engine: {error_details}")
                    self.enable_hybrid_matching = False
            
            # Initialize ASR Scripture Matcher if enabled (Epic 5 - Digital Dharma)
            if self.enable_asr_matching:
                try:
                    self.asr_matcher = ASRScriptureMatcher(
                        scripture_data_path=scripture_dir if scripture_dir else Path("data/scriptures")
                    )
                    
                    self._error_handler.log_operation_success("asr_scripture_matcher_initialization", {
                        'scripture_dir': str(scripture_dir) if scripture_dir else 'default'
                    })
                    
                except Exception as e:
                    error_details = self._error_handler.handle_processing_error("asr_scripture_matcher_initialization", e, {
                        'enable_asr_matching': self.enable_asr_matching,
                        'scripture_dir': str(scripture_dir) if scripture_dir else 'None'
                    })
                    self.logger.error(f"Failed to initialize ASR Scripture Matcher: {error_details}")
                    self.enable_asr_matching = False
            
            # Initialize academic enhancement components if enabled (Story 4.5)
            if self.enable_academic_enhancement:
                try:
                    from scripture_processing.academic_citation_manager import AcademicCitationManager
                    from scripture_processing.publication_formatter import PublicationFormatter  
                    from utils.academic_validator import AcademicValidator
                    
                    # Initialize academic components
                    self.academic_citation_manager = AcademicCitationManager(
                        config=self.config.get('academic_citation', {})
                    )
                    
                    self.publication_formatter = PublicationFormatter(
                        canonical_manager=self.canonical_manager,
                        citation_manager=self.academic_citation_manager,
                        config=self.config.get('publication_formatter', {})
                    )
                    
                    self.academic_validator = AcademicValidator(
                        config=self.config.get('academic_validator', {})
                    )
                    
                    self._error_handler.log_operation_success("academic_enhancement_initialization", {
                        'citation_manager': self.academic_citation_manager is not None,
                        'publication_formatter': self.publication_formatter is not None,
                        'academic_validator': self.academic_validator is not None
                    })
                    
                except Exception as e:
                    error_details = self._error_handler.handle_processing_error("academic_enhancement_initialization", e, {
                        'enable_academic_enhancement': self.enable_academic_enhancement
                    })
                    self.logger.error(f"Failed to initialize academic enhancement components: {error_details}")
                    self.enable_academic_enhancement = False
            
            self._error_handler.log_operation_success("scripture_processor_initialization", {
                'components_initialized': [
                    'canonical_manager', 'scripture_identifier', 'validator',
                    'substitution_engine', 'iast_formatter', 'selection_system'
                ],
                'hybrid_matching_enabled': self.enable_hybrid_matching,
                'academic_enhancement_enabled': self.enable_academic_enhancement
            })
            
        except Exception as e:
            # If error handler initialization fails, use basic logging
            from utils.logger_config import get_logger
            self.logger = get_logger(__name__)
            self.logger.error(f"Failed to initialize ScriptureProcessor with error handler: {e}")
            
            self._error_handler = None
            raise ProcessingError(f"Critical failure initializing ScriptureProcessor: {str(e)}")
    
    def process_text(self, text: str, context: Dict = None) -> ScriptureProcessingResult:
        """
        Process text for scripture identification and substitution with standardized error handling.
        
        Args:
            text: Input text to process
            context: Additional context information
            
        Returns:
            Complete scripture processing result
        """
        if self._error_handler:
            self._error_handler.log_operation_start("scripture_text_processing", {
                'text_length': len(text),
                'text_preview': text[:100] + "..." if len(text) > 100 else text,
                'context_provided': context is not None,
                'context_keys': list(context.keys()) if context else []
            })
        
        context = context or {}
        processing_metadata = {'steps_completed': []}
        detailed_results = {}
        
        processed_text = text
        verses_identified = 0
        substitutions_made = 0
        iast_formatted = False
        validation_passed = True
        
        # Initialize hybrid matching variables
        hybrid_matching_used = False
        hybrid_confidence = 0.0
        hybrid_stages = []
        
        try:
            # Story 2.4.3 - Use hybrid matching if enabled
            if self.enable_hybrid_matching and self.hybrid_engine:
                try:
                    if self._error_handler:
                        self._error_handler.log_operation_start("hybrid_scripture_matching", {
                            'hybrid_engine_available': self.hybrid_engine is not None
                        })
                    
                    hybrid_result = self.hybrid_engine.match_verse_passage(text, context)
                    
                    if hybrid_result.pipeline_success and hybrid_result.matched_verse:
                        # Use hybrid matching result
                        verse_matches = [hybrid_result.traditional_match] if hybrid_result.traditional_match else []
                        verses_identified = 1
                        hybrid_matching_used = True
                        hybrid_confidence = hybrid_result.composite_confidence
                        hybrid_stages = [stage.value for stage in hybrid_result.stages_completed]
                        
                        detailed_results['hybrid_result'] = hybrid_result
                        processing_metadata['hybrid_matching'] = True
                        
                        if self._error_handler:
                            self._error_handler.log_operation_success("hybrid_scripture_matching", {
                                'confidence': hybrid_confidence,
                                'stages_completed': hybrid_stages,
                                'verses_found': verses_identified
                            })
                        
                    else:
                        # Fall back to traditional identification
                        if self._error_handler:
                            self._error_handler.log_operation_warning("hybrid_scripture_matching_fallback", {
                                'pipeline_success': hybrid_result.pipeline_success,
                                'matched_verse_found': hybrid_result.matched_verse is not None
                            })
                        
                        verse_matches = self.scripture_identifier.identify_scripture_passages(text)
                        verses_identified = len(verse_matches)
                        
                except Exception as e:
                    if self._error_handler:
                        error_details = self._error_handler.handle_processing_error("hybrid_scripture_matching", e, {
                            'text_preview': text[:50] + "..." if len(text) > 50 else text
                        })
                        self.logger.error(f"Error in hybrid matching: {error_details}, falling back to traditional")
                    
                    verse_matches = self.scripture_identifier.identify_scripture_passages(text)
                    verses_identified = len(verse_matches)
            else:
                # Step 1: Traditional scripture identification
                try:
                    if self._error_handler:
                        self._error_handler.log_operation_start("traditional_scripture_identification", {
                            'text_length': len(text)
                        })
                    
                    verse_matches = self.scripture_identifier.identify_scripture_passages(text)
                    verses_identified = len(verse_matches)
                    
                    if self._error_handler:
                        self._error_handler.log_operation_success("traditional_scripture_identification", {
                            'verses_identified': verses_identified
                        })
                        
                except Exception as e:
                    if self._error_handler:
                        error_details = self._error_handler.handle_processing_error("traditional_scripture_identification", e, {
                            'text_preview': text[:50] + "..." if len(text) > 50 else text
                        })
                        self.logger.error(f"Scripture identification failed: {error_details}")
                    
                    # Set empty results on identification failure
                    verse_matches = []
                    verses_identified = 0
            
            detailed_results['verse_matches'] = verse_matches
            processing_metadata['steps_completed'].append('identification')
            
            if verses_identified == 0:
                if self._error_handler:
                    self._error_handler.log_operation_success("scripture_text_processing_no_verses", {
                        'verses_identified': 0,
                        'text_preview': text[:50] + "..." if len(text) > 50 else text
                    })
                
                return ScriptureProcessingResult(
                    original_text=text,
                    processed_text=processed_text,
                    verses_identified=0,
                    substitutions_made=0,
                    iast_formatted=False,
                    validation_passed=True,
                    processing_metadata=processing_metadata,
                    detailed_results=detailed_results,
                    hybrid_matching_used=hybrid_matching_used,
                    hybrid_confidence=hybrid_confidence,
                    hybrid_pipeline_stages=hybrid_stages
                )
            
            # Step 2: Verse substitution (if enabled)
            if self.enable_substitution:
                try:
                    if self._error_handler:
                        self._error_handler.log_operation_start("verse_substitution", {
                            'verses_to_process': verses_identified
                        })
                    
                    substitution_result = self.substitution_engine.substitute_verses_in_text(processed_text)
                    processed_text = substitution_result.substituted_text
                    substitutions_made = len(substitution_result.operations_performed)
                    detailed_results['substitution_result'] = substitution_result
                    processing_metadata['steps_completed'].append('substitution')
                    
                    if self._error_handler:
                        self._error_handler.log_operation_success("verse_substitution", {
                            'substitutions_made': substitutions_made,
                            'text_length_change': len(processed_text) - len(text)
                        })
                        
                except Exception as e:
                    if self._error_handler:
                        error_details = self._error_handler.handle_processing_error("verse_substitution", e, {
                            'verses_identified': verses_identified,
                            'text_preview': processed_text[:50] + "..." if len(processed_text) > 50 else processed_text
                        })
                        self.logger.error(f"Verse substitution failed: {error_details}")
                    
                    # Continue with original text on substitution failure
                    substitutions_made = 0
            
            # Step 3: IAST formatting (if enabled)
            if self.enable_iast_formatting and substitutions_made > 0:
                try:
                    if self._error_handler:
                        self._error_handler.log_operation_start("iast_formatting", {
                            'substitutions_made': substitutions_made
                        })
                    
                    formatted_text = self._apply_iast_formatting_to_text(processed_text)
                    if formatted_text != processed_text:
                        processed_text = formatted_text
                        iast_formatted = True
                        processing_metadata['steps_completed'].append('iast_formatting')
                        
                        if self._error_handler:
                            self._error_handler.log_operation_success("iast_formatting", {
                                'formatting_applied': True,
                                'text_length_change': len(formatted_text) - len(text)
                            })
                    else:
                        if self._error_handler:
                            self._error_handler.log_operation_warning("iast_formatting_no_change", {
                                'text_preview': processed_text[:50] + "..." if len(processed_text) > 50 else processed_text
                            })
                            
                except Exception as e:
                    if self._error_handler:
                        error_details = self._error_handler.handle_processing_error("iast_formatting", e, {
                            'processed_text_preview': processed_text[:50] + "..." if len(processed_text) > 50 else processed_text
                        })
                        self.logger.error(f"IAST formatting failed: {error_details}")
                    
                    # Continue without IAST formatting on failure
                    iast_formatted = False
            
            # Step 4: Validation (if enabled)
            if self.enable_validation:
                try:
                    if self._error_handler:
                        self._error_handler.log_operation_start("scripture_validation", {
                            'substitutions_made': substitutions_made,
                            'iast_formatted': iast_formatted
                        })
                    
                    validation_results = self._validate_processing_results(
                        text, processed_text, detailed_results.get('substitution_result')
                    )
                    validation_passed = validation_results['overall_valid']
                    detailed_results['validation'] = validation_results
                    processing_metadata['steps_completed'].append('validation')
                    
                    if self._error_handler:
                        self._error_handler.log_operation_success("scripture_validation", {
                            'validation_passed': validation_passed,
                            'validation_results': validation_results
                        })
                        
                except Exception as e:
                    if self._error_handler:
                        error_details = self._error_handler.handle_processing_error("scripture_validation", e, {
                            'original_text_preview': text[:50] + "..." if len(text) > 50 else text,
                            'processed_text_preview': processed_text[:50] + "..." if len(processed_text) > 50 else processed_text
                        })
                        self.logger.error(f"Scripture validation failed: {error_details}")
                    
                    # Continue with failed validation status
                    validation_passed = False
            
            processing_metadata['success'] = True
            
            if self._error_handler:
                self._error_handler.log_operation_success("scripture_text_processing", {
                    'verses_identified': verses_identified,
                    'substitutions_made': substitutions_made,
                    'iast_formatted': iast_formatted,
                    'validation_passed': validation_passed,
                    'hybrid_matching_used': hybrid_matching_used,
                    'steps_completed': processing_metadata['steps_completed']
                })
            
        except Exception as e:
            if self._error_handler:
                error_details = self._error_handler.handle_processing_error("scripture_text_processing_critical", e, {
                    'original_text_preview': text[:50] + "..." if len(text) > 50 else text,
                    'context_provided': context is not None
                })
                self.logger.error(f"Critical failure in scripture processing: {error_details}")
            
            processing_metadata['error'] = str(e)
            processing_metadata['success'] = False
            validation_passed = False
        
        return ScriptureProcessingResult(
            original_text=text,
            processed_text=processed_text,
            verses_identified=verses_identified,
            substitutions_made=substitutions_made,
            iast_formatted=iast_formatted,
            validation_passed=validation_passed,
            processing_metadata=processing_metadata,
            detailed_results=detailed_results,
            hybrid_matching_used=hybrid_matching_used,
            hybrid_confidence=hybrid_confidence,
            hybrid_pipeline_stages=hybrid_stages
        )
    
    def process_srt_segment(self, segment: SRTSegment, 
                          context: Dict = None) -> Tuple[SRTSegment, ScriptureProcessingResult]:
        """
        Process an SRT segment for scripture handling.
        
        Args:
            segment: SRT segment to process
            context: Additional context
            
        Returns:
            Tuple of (processed_segment, processing_result)
        """
        # Process the segment text
        processing_result = self.process_text(segment.text, context)
        
        # Create new segment with processed text
        processed_segment = SRTSegment(
            index=segment.index,
            start_time=segment.start_time,
            end_time=segment.end_time,
            text=processing_result.processed_text
        )
        
        return processed_segment, processing_result
    
    def _apply_iast_formatting_to_text(self, text: str) -> str:
        """Apply IAST formatting to text containing verses."""
        # This is a simplified approach - in practice, you'd identify specific verses
        # and apply formatting to them individually
        
        # For now, apply basic IAST formatting if the text contains Sanskrit characteristics
        if self._contains_sanskrit_text(text):
            # Apply basic IAST transliteration
            try:
                transliteration_result = self.iast_formatter.iast_transliterator.transliterate_to_iast(text)
                return transliteration_result.transliterated_text
            except Exception as e:
                self.logger.warning(f"IAST formatting failed: {e}")
                return text
        
        return text
    
    def _contains_sanskrit_text(self, text: str) -> bool:
        """Check if text contains Sanskrit characteristics."""
        import re
        
        # Look for Sanskrit indicators
        sanskrit_indicators = [
            r'[|।]',  # Sanskrit punctuation
            r'[aeiou]m\b',  # Sanskrit word endings
            r'ā[a-zA-Z]*',  # IAST long vowels
            r'\b(karma|dharma|yoga|gita|sutra)\b',  # Common Sanskrit words
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in sanskrit_indicators)
    
    def _validate_processing_results(self, original: str, processed: str, 
                                   substitution_result: Optional[SubstitutionResult]) -> Dict[str, Any]:
        """Validate the overall processing results."""
        validation_results = {
            'overall_valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Basic sanity checks
        if not processed or len(processed.strip()) == 0:
            validation_results['overall_valid'] = False
            validation_results['issues'].append("Processed text is empty")
        
        # Length change validation
        length_ratio = len(processed) / len(original) if len(original) > 0 else 0
        if length_ratio > 3.0 or length_ratio < 0.3:
            validation_results['warnings'].append(
                f"Significant length change (ratio: {length_ratio:.2f})"
            )
        
        # Substitution validation
        if substitution_result:
            if substitution_result.validation_warnings:
                validation_results['warnings'].extend(substitution_result.validation_warnings)
            
            # Check substitution confidence
            if substitution_result.overall_confidence < 0.7:
                validation_results['warnings'].append(
                    f"Low substitution confidence: {substitution_result.overall_confidence:.3f}"
                )
        
        return validation_results
    
    def match_asr_to_scripture(self, asr_text: str, 
                              min_confidence: float = 0.3,
                              max_results: int = 5) -> Dict[str, Any]:
        """
        Match garbled ASR output to canonical scriptural verses.
        
        This method implements the Digital Dharma research insights for
        matching ASR approximations to exact scriptural quotes.
        Enhanced for Tulsi Ramayana (Hindi) support.
        
        Args:
            asr_text: The ASR-generated text (possibly garbled)
            min_confidence: Minimum confidence threshold for matches (default 0.3, balanced threshold)
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary containing:
                - matches: List of matched verses with confidence scores
                - original_text: The original ASR text
                - report: Human-readable match report
                - processing_metadata: Details about the matching process
        """
        if not self.enable_asr_matching or not self.asr_matcher:
            return {
                'matches': [],
                'original_text': asr_text,
                'report': 'ASR Scripture Matching is not enabled or initialized',
                'processing_metadata': {'enabled': False}
            }
        
        try:
            # Perform ASR-to-scripture matching with balanced confidence threshold
            matches = self.asr_matcher.match_asr_to_verse(
                asr_text, 
                min_confidence=min_confidence
            )
            
            # Format results
            formatted_matches = []
            for match in matches[:max_results]:
                formatted_matches.append({
                    'verse_reference': match.verse_reference,
                    'canonical_text': match.canonical_text,
                    'confidence': match.confidence_score,
                    'strategy': match.matching_strategy.value,
                    'translation': match.canonical_verse.get('translation', ''),
                    'source': match.canonical_verse.get('source', 'unknown'),
                    'details': match.match_details
                })
            
            # Generate human-readable report
            report = self.asr_matcher.format_match_report(
                matches, 
                asr_text, 
                max_results=max_results
            )
            
            return {
                'matches': formatted_matches,
                'original_text': asr_text,
                'report': report,
                'processing_metadata': {
                    'enabled': True,
                    'total_matches': len(matches),
                    'returned_matches': len(formatted_matches),
                    'min_confidence': min_confidence,
                    'strategies_used': list(set(m.matching_strategy.value for m in matches)),
                    'sources_searched': ['bhagavad_gita', 'yoga_sutras', 'upanishads', 'ramayana']
                }
            }
            
        except Exception as e:
            self.logger.error(f"ASR scripture matching failed: {e}")
            return {
                'matches': [],
                'original_text': asr_text,
                'report': f'ASR Scripture Matching failed: {str(e)}',
                'processing_metadata': {'enabled': True, 'error': str(e)}
            }

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics from all components."""
        return {
            'canonical_texts': self.canonical_manager.get_statistics(),
            'selection_system_config': {
                'strategy': self.selection_system.selection_strategy.value,
                'auto_threshold': self.selection_system.auto_select_threshold,
                'review_threshold': self.selection_system.human_review_threshold
            },
            'processing_config': {
                'substitution_enabled': self.enable_substitution,
                'iast_formatting_enabled': self.enable_iast_formatting,
                'validation_enabled': self.enable_validation,
                'formatting_style': self.formatting_style.value
            }
        }
    
    def preview_verse_substitutions(self, text: str, max_previews: int = 3) -> List[Dict[str, Any]]:
        """
        Preview potential verse substitutions without applying them.
        
        Args:
            text: Input text
            max_previews: Maximum number of previews to return
            
        Returns:
            List of substitution previews
        """
        return self.substitution_engine.get_substitution_preview(text, max_previews)
    
    def configure_selection_strategy(self, strategy: SelectionStrategy, 
                                   thresholds: Dict[str, float] = None) -> None:
        """
        Configure the verse selection strategy.
        
        Args:
            strategy: Selection strategy to use
            thresholds: Optional confidence thresholds
        """
        self.selection_system.selection_strategy = strategy
        
        if thresholds:
            if 'auto_select' in thresholds:
                self.selection_system.auto_select_threshold = thresholds['auto_select']
            if 'human_review' in thresholds:
                self.selection_system.human_review_threshold = thresholds['human_review']
        
        self.logger.info(f"Updated selection strategy to {strategy.value}")
    
    def enable_hybrid_matching_engine(self, enabled: bool = True, config: Dict = None) -> None:
        """
        Enable or disable the hybrid matching engine (Story 2.4.3).
        
        Args:
            enabled: Whether to enable hybrid matching
            config: Optional configuration for hybrid pipeline
        """
        if enabled and not self.hybrid_engine:
            try:
                hybrid_config = HybridPipelineConfig()
                if config:
                    for key, value in config.items():
                        if hasattr(hybrid_config, key):
                            setattr(hybrid_config, key, value)
                
                self.hybrid_engine = HybridMatchingEngine(
                    canonical_manager=self.canonical_manager,
                    config=hybrid_config
                )
                
                self.enable_hybrid_matching = True
                self.logger.info("Hybrid matching engine enabled")
                
            except Exception as e:
                self.logger.error(f"Failed to enable hybrid matching: {e}")
                self.enable_hybrid_matching = False
        else:
            self.enable_hybrid_matching = enabled
            self.logger.info(f"Hybrid matching {'enabled' if enabled else 'disabled'}")
    
    def get_hybrid_performance_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get performance statistics from hybrid matching engine.
        
        Returns:
            Hybrid engine statistics or None if not enabled
        """
        if self.enable_hybrid_matching and self.hybrid_engine:
            return self.hybrid_engine.get_performance_statistics()
        return None