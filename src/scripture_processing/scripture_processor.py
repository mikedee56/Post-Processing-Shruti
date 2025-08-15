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
from .scripture_identifier import ScriptureIdentifier
from .canonical_text_manager import CanonicalTextManager
from .verse_substitution_engine import VerseSubstitutionEngine, SubstitutionResult
from .scripture_validator import ScriptureValidator
from .scripture_iast_formatter import ScriptureIASTFormatter, VerseFormatting
from .verse_selection_system import VerseSelectionSystem, SelectionStrategy
from .hybrid_matching_engine import HybridMatchingEngine, HybridPipelineConfig


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
        Initialize the Scripture Processor.
        
        Args:
            config: Configuration parameters
            scripture_dir: Directory containing scripture databases
        """
        self.logger = get_logger(__name__)
        self.config = config or {}
        
        # Story 2.4.3 - Hybrid matching configuration
        self.enable_hybrid_matching = self.config.get('enable_hybrid_matching', False)
        self.hybrid_engine = None
        
        # Story 4.5 - Academic enhancement configuration
        self.enable_academic_enhancement = self.config.get('enable_academic_enhancement', False)
        self.academic_citation_manager = None
        self.publication_formatter = None
        self.academic_validator = None
        
        # Initialize components
        self.canonical_manager = CanonicalTextManager(
            scripture_dir=scripture_dir,
            config=self.config.get('canonical_manager', {})
        )
        
        self.scripture_identifier = ScriptureIdentifier(
            lexicon_manager=None,  # Will use default
            config=self.config.get('scripture_identifier', {})
        )
        
        self.validator = ScriptureValidator(
            config=self.config.get('validator', {})
        )
        
        self.substitution_engine = VerseSubstitutionEngine(
            scripture_identifier=self.scripture_identifier,
            canonical_manager=self.canonical_manager,
            validator=self.validator,
            config=self.config.get('substitution_engine', {})
        )
        
        self.iast_formatter = ScriptureIASTFormatter(
            config=self.config.get('iast_formatter', {})
        )
        
        self.selection_system = VerseSelectionSystem(
            canonical_manager=self.canonical_manager,
            scripture_identifier=self.scripture_identifier,
            validator=self.validator,
            config=self.config.get('selection_system', {})
        )
        
        # Processing configuration
        self.enable_substitution = self.config.get('enable_substitution', True)
        self.enable_iast_formatting = self.config.get('enable_iast_formatting', True)
        self.enable_validation = self.config.get('enable_validation', True)
        self.formatting_style = VerseFormatting(
            self.config.get('formatting_style', 'academic')
        )
        
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
                
                self.logger.info("Hybrid matching engine initialized (Story 2.4.3)")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize hybrid matching engine: {e}")
                self.enable_hybrid_matching = False
        
        # Initialize academic enhancement components if enabled (Story 4.5)
        if self.enable_academic_enhancement:
            try:
                from .academic_citation_manager import AcademicCitationManager
                from .publication_formatter import PublicationFormatter  
                from ..utils.academic_validator import AcademicValidator
                
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
                
                self.logger.info("Academic enhancement components initialized (Story 4.5)")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize academic enhancement components: {e}")
                self.enable_academic_enhancement = False
        
        self.logger.info("Scripture processor initialized with all components")
    
    def process_text(self, text: str, context: Dict = None) -> ScriptureProcessingResult:
        """
        Process text for scripture identification and substitution.
        
        Args:
            text: Input text to process
            context: Additional context information
            
        Returns:
            Complete scripture processing result
        """
        context = context or {}
        processing_metadata = {'steps_completed': []}
        detailed_results = {}
        
        processed_text = text
        verses_identified = 0
        substitutions_made = 0
        iast_formatted = False
        validation_passed = True
        
        try:
            # Story 2.4.3 - Use hybrid matching if enabled
            hybrid_matching_used = False
            hybrid_confidence = 0.0
            hybrid_stages = []
            
            if self.enable_hybrid_matching and self.hybrid_engine:
                self.logger.info("Using hybrid matching engine for scripture identification...")
                
                try:
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
                        
                        self.logger.info(
                            f"Hybrid matching succeeded: {hybrid_confidence:.3f} confidence, "
                            f"stages: {hybrid_stages}"
                        )
                    else:
                        # Fall back to traditional identification
                        self.logger.info("Hybrid matching failed, falling back to traditional identification")
                        verse_matches = self.scripture_identifier.identify_scripture_passages(text)
                        verses_identified = len(verse_matches)
                        
                except Exception as e:
                    self.logger.error(f"Error in hybrid matching: {e}, falling back to traditional")
                    verse_matches = self.scripture_identifier.identify_scripture_passages(text)
                    verses_identified = len(verse_matches)
            else:
                # Step 1: Traditional scripture identification
                self.logger.info("Identifying scripture passages...")
                verse_matches = self.scripture_identifier.identify_scripture_passages(text)
                verses_identified = len(verse_matches)
            
            detailed_results['verse_matches'] = verse_matches
            processing_metadata['steps_completed'].append('identification')
            
            if verses_identified == 0:
                self.logger.info("No scripture passages identified")
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
                self.logger.info(f"Processing {verses_identified} identified verses for substitution...")
                substitution_result = self.substitution_engine.substitute_verses_in_text(processed_text)
                processed_text = substitution_result.substituted_text
                substitutions_made = len(substitution_result.operations_performed)
                detailed_results['substitution_result'] = substitution_result
                processing_metadata['steps_completed'].append('substitution')
                
                self.logger.info(f"Made {substitutions_made} verse substitutions")
            
            # Step 3: IAST formatting (if enabled)
            if self.enable_iast_formatting and substitutions_made > 0:
                self.logger.info("Applying IAST formatting to processed verses...")
                formatted_text = self._apply_iast_formatting_to_text(processed_text)
                if formatted_text != processed_text:
                    processed_text = formatted_text
                    iast_formatted = True
                    processing_metadata['steps_completed'].append('iast_formatting')
            
            # Step 4: Validation (if enabled)
            if self.enable_validation:
                self.logger.info("Validating scripture processing results...")
                validation_results = self._validate_processing_results(
                    text, processed_text, detailed_results.get('substitution_result')
                )
                validation_passed = validation_results['overall_valid']
                detailed_results['validation'] = validation_results
                processing_metadata['steps_completed'].append('validation')
            
            processing_metadata['success'] = True
            
        except Exception as e:
            self.logger.error(f"Error during scripture processing: {e}")
            processing_metadata['error'] = str(e)
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