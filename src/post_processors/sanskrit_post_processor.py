"""
Sanskrit Post-Processor for ASR Transcript Correction.

This module provides the core functionality for processing ASR-generated
transcripts of Yoga Vedanta lectures, with specialized handling for
Sanskrit and Hindi terminology, IAST transliteration, and scriptural verses.
"""

import re
import json
import yaml
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

import pysrt
import pandas as pd
from fuzzywuzzy import fuzz, process

# Import new components
from utils.srt_parser import SRTParser, SRTSegment
from utils.text_normalizer import TextNormalizer
from utils.advanced_text_normalizer import AdvancedTextNormalizer
from utils.conversational_pattern_detector import ConversationalPatternDetector
from utils.contextual_number_processor import ContextualNumberProcessor
from utils.processing_quality_validator import ProcessingQualityValidator
from utils.metrics_collector import MetricsCollector, ProcessingMetrics
from utils.logger_config import get_logger

# Import Story 2.1 components
from sanskrit_hindi_identifier.word_identifier import SanskritHindiIdentifier
from sanskrit_hindi_identifier.lexicon_manager import LexiconManager
from sanskrit_hindi_identifier.correction_applier import CorrectionApplier
from utils.fuzzy_matcher import FuzzyMatcher, MatchingConfig
from utils.iast_transliterator import IASTTransliterator


@dataclass
class TranscriptSegment:
    """Represents an individual timestamped segment of the transcript."""
    id: str
    text: str
    start_time: float
    end_time: float
    confidence_score: float = 0.0
    is_flagged: bool = False
    flag_reason: str = ""
    correction_history: List[Dict[str, Any]] = None
    processing_metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.correction_history is None:
            self.correction_history = []
        if self.processing_metadata is None:
            self.processing_metadata = {}


@dataclass
class LexiconEntry:
    """Represents a single entry in the domain-specific lexicon."""
    original_term: str
    variations: List[str]
    transliteration: str
    is_proper_noun: bool = False
    is_verse: bool = False
    canonical_text: str = ""
    category: str = "general"
    confidence: float = 1.0
    source_authority: str = ""


class SanskritPostProcessor:
    """
    Main post-processing class for Sanskrit/Hindi term correction in ASR transcripts.
    
    This class handles the core functionality of identifying and correcting 
    misrecognized Sanskrit and Hindi terms, applying IAST transliteration,
    and processing scriptural verses.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the Sanskrit Post-Processor.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = self._load_config(config_path)
        self.logger = get_logger(__name__, self.config.get('logging', {}))
        
        # Initialize new components
        self.srt_parser = SRTParser()
        
        # Choose between basic and advanced text normalizer based on config
        use_advanced_normalization = self.config.get('use_advanced_normalization', True)
        if use_advanced_normalization:
            self.text_normalizer = AdvancedTextNormalizer(self.config.get('text_normalization', {}))
        else:
            self.text_normalizer = TextNormalizer(self.config.get('text_normalization', {}))
        
        # Initialize additional foundational correction components
        self.conversational_detector = ConversationalPatternDetector(self.config.get('conversational_patterns', {}))
        self.number_processor = ContextualNumberProcessor(self.config.get('contextual_numbers', {}))
        self.quality_validator = ProcessingQualityValidator(self.config.get('quality_validation', {}))
        
        self.metrics_collector = MetricsCollector(self.config.get('metrics', {}))
        
        # Initialize Story 2.1 components
        lexicon_dir = Path(self.config.get('lexicon_dir', 'data/lexicons'))
        
        # Initialize enhanced lexicon management
        self.lexicon_manager = LexiconManager(
            lexicon_dir=lexicon_dir, 
            enable_caching=self.config.get('enable_lexicon_caching', True)
        )
        
        # Initialize Sanskrit/Hindi word identifier
        self.word_identifier = SanskritHindiIdentifier(
            lexicon_dir=lexicon_dir,
            english_words_file=self.config.get('english_words_file')
        )
        
        # Initialize fuzzy matcher with enhanced configuration
        lexicon_data = self.lexicon_manager.get_all_entries()
        matching_config = MatchingConfig(
            min_confidence=self.config.get('fuzzy_min_confidence', 0.75),
            levenshtein_threshold=self.config.get('levenshtein_threshold', 0.80),
            phonetic_threshold=self.config.get('phonetic_threshold', 0.85),
            max_edit_distance=self.config.get('max_edit_distance', 3),
            enable_phonetic_matching=self.config.get('enable_phonetic_matching', True),
            enable_compound_matching=self.config.get('enable_compound_matching', True)
        )
        
        # Convert LexiconEntry objects to dict format for FuzzyMatcher
        lexicon_dict = {}
        for term, entry in lexicon_data.items():
            lexicon_dict[term] = {
                'transliteration': entry.transliteration,
                'variations': entry.variations,
                'is_proper_noun': entry.is_proper_noun,
                'category': entry.category,
                'confidence': entry.confidence,
                'source_authority': entry.source_authority
            }
        
        self.fuzzy_matcher = FuzzyMatcher(lexicon_dict, matching_config)
        
        # Initialize IAST transliterator
        self.iast_transliterator = IASTTransliterator(
            strict_mode=self.config.get('iast_strict_mode', True)
        )
        
        # Initialize correction applier
        self.correction_applier = CorrectionApplier(
            min_confidence=self.config.get('correction_min_confidence', 0.80),
            critical_confidence=self.config.get('correction_critical_confidence', 0.95),
            enable_context_validation=self.config.get('enable_context_validation', True),
            max_corrections_per_segment=self.config.get('max_corrections_per_segment', 10)
        )
        
        # Legacy lexicons for backward compatibility
        self.corrections: Dict[str, LexiconEntry] = {}
        self.proper_nouns: Dict[str, LexiconEntry] = {}
        self.phrases: Dict[str, LexiconEntry] = {}
        self.verses: Dict[str, LexiconEntry] = {}
        
        # Load legacy lexicons from external files
        self._load_lexicons()
        
        # Fuzzy matching threshold (legacy)
        self.fuzzy_threshold = self.config.get('fuzzy_threshold', 80)

    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_path and config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() == '.yaml':
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        
        # Default configuration
        return {
            'fuzzy_threshold': 80,
            'confidence_threshold': 0.6,
            'use_advanced_normalization': True,
            
            # Story 2.1: Lexicon-based correction system configuration
            'lexicon_dir': 'data/lexicons',
            'enable_lexicon_caching': True,
            'english_words_file': None,  # Optional English dictionary file
            
            # Fuzzy matching configuration
            'fuzzy_min_confidence': 0.75,
            'levenshtein_threshold': 0.80,
            'phonetic_threshold': 0.85,
            'max_edit_distance': 3,
            'enable_phonetic_matching': True,
            'enable_compound_matching': True,
            
            # IAST transliteration configuration
            'iast_strict_mode': True,
            
            # Correction application configuration
            'correction_min_confidence': 0.80,
            'correction_critical_confidence': 0.95,
            'enable_context_validation': True,
            'max_corrections_per_segment': 10,
            
            # Legacy lexicon paths (for backward compatibility)
            'lexicon_paths': {
                'corrections': 'data/lexicons/corrections.yaml',
                'proper_nouns': 'data/lexicons/proper_nouns.yaml',
                'phrases': 'data/lexicons/phrases.yaml',
                'verses': 'data/lexicons/verses.yaml'
            },
            'text_normalization': {
                'remove_fillers': True,
                'convert_numbers': True,
                'standardize_punctuation': True,
                'fix_capitalization': True,
                'preserve_meaningful_discourse': True,
                'semantic_drift_threshold': 0.3,
                'min_confidence_score': 0.7
            },
            'conversational_patterns': {
                'min_confidence_threshold': 0.7,
                'context_window_size': 50
            },
            'contextual_numbers': {
                'min_confidence_threshold': 0.7,
                'preserve_uncertainty': True
            },
            'quality_validation': {
                'validation_level': 'moderate',
                'max_semantic_drift': 0.3,
                'max_timestamp_deviation': 0.001,
                'min_quality_score': 0.7
            }
        }

    def _load_lexicons(self):
        """Load lexicons from external YAML/JSON files."""
        lexicon_paths = self.config.get('lexicon_paths', {})
        
        for lexicon_type, path in lexicon_paths.items():
            lexicon_path = Path(path)
            if lexicon_path.exists():
                self._load_lexicon_file(lexicon_type, lexicon_path)
            else:
                self.logger.warning(f"Lexicon file not found: {path}")

    def _load_lexicon_file(self, lexicon_type: str, file_path: Path):
        """Load a specific lexicon file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() == '.yaml':
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            target_dict = getattr(self, lexicon_type, {})
            
            for entry_data in data.get('entries', []):
                entry = LexiconEntry(**entry_data)
                target_dict[entry.original_term.lower()] = entry
                
                # Also index by variations
                for variation in entry.variations:
                    target_dict[variation.lower()] = entry
                    
            self.logger.info(f"Loaded {len(target_dict)} entries from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading lexicon {file_path}: {e}")


    def process_srt_file(self, input_path: Path, output_path: Path, session_id: Optional[str] = None) -> ProcessingMetrics:
        """
        Process an SRT file with comprehensive text corrections and metrics.
        
        Args:
            input_path: Path to input SRT file
            output_path: Path to output SRT file
            session_id: Optional session ID for metrics tracking
            
        Returns:
            ProcessingMetrics with detailed results
        """
        # Start timing
        start_time = time.time()
        
        # Create metrics object
        metrics = self.metrics_collector.create_file_metrics(str(input_path))
        
        try:
            self.logger.info(f"Starting processing: {input_path}")
            
            # Step 1: Parse SRT file
            self.metrics_collector.start_timer('parsing')
            segments = self.srt_parser.parse_file(str(input_path))
            metrics.parsing_time = self.metrics_collector.end_timer('parsing')
            
            if not segments:
                raise ValueError("No valid segments found in SRT file")
            
            metrics.total_segments = len(segments)
            self.logger.info(f"Parsed {len(segments)} segments")
            
            # Step 2: Validate timestamps
            self.metrics_collector.start_timer('validation')
            timestamp_valid = self.srt_parser.validate_timestamps(segments)
            metrics.timestamp_integrity_verified = timestamp_valid
            metrics.validation_time = self.metrics_collector.end_timer('validation')
            
            if not timestamp_valid:
                metrics.warnings_encountered.append("Timestamp integrity issues detected")
                self.logger.warning("Timestamp integrity issues detected")
            
            # Calculate original statistics
            metrics.original_word_count = sum(len(seg.text.split()) for seg in segments)
            metrics.original_char_count = sum(len(seg.text) for seg in segments)
            
            # Step 3: Process segments
            processed_segments = []
            
            for i, segment in enumerate(segments):
                try:
                    processed_segment = self._process_srt_segment(segment, metrics)
                    processed_segments.append(processed_segment)
                    
                    # Track confidence scores
                    if hasattr(processed_segment, 'confidence') and processed_segment.confidence is not None:
                        metrics.confidence_scores.append(processed_segment.confidence)
                    
                    # Count modifications
                    if segment.text != processed_segment.text:
                        metrics.segments_modified += 1
                    
                except Exception as e:
                    error_msg = f"Error processing segment {i}: {e}"
                    metrics.errors_encountered.append(error_msg)
                    self.logger.error(error_msg)
                    # Use original segment if processing fails
                    processed_segments.append(segment)
            
            # Calculate processed statistics
            metrics.processed_word_count = sum(len(seg.text.split()) for seg in processed_segments)
            metrics.processed_char_count = sum(len(seg.text) for seg in processed_segments)
            
            # Calculate quality metrics
            self.metrics_collector.calculate_quality_metrics(metrics)
            
            # Step 4: Generate output SRT
            output_srt = self.srt_parser.to_srt_string(processed_segments)
            
            # Save to file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output_srt)
            
            # Final timing
            metrics.processing_time = time.time() - start_time
            
            # Add to metrics collector
            self.metrics_collector.add_file_metrics(metrics)
            
            self.logger.info(f"Processing completed: {output_path}")
            self.logger.info(f"Segments: {metrics.total_segments}, Modified: {metrics.segments_modified}, "
                           f"Confidence: {metrics.average_confidence:.3f}")
            
            return metrics
            
        except Exception as e:
            error_msg = f"Error processing SRT file {input_path}: {e}"
            metrics.errors_encountered.append(error_msg)
            metrics.processing_time = time.time() - start_time
            self.logger.error(error_msg)
            
            # Still add metrics even if processing failed
            self.metrics_collector.add_file_metrics(metrics)
            raise

    def _process_srt_segment(self, segment: SRTSegment, metrics: ProcessingMetrics) -> SRTSegment:
        """
        Process a single SRT segment with the enhanced foundational corrections pipeline.
        
        Args:
            segment: SRT segment to process
            metrics: ProcessingMetrics object to update
            
        Returns:
            Processed SRT segment with foundational corrections applied
        """
        # Create a copy of the segment to avoid mutating the original
        import copy
        processed_segment = copy.deepcopy(segment)
        
        original_text = segment.text  # Keep reference to original text
        all_corrections_applied = []
        
        # Step 1: Enhanced Text Normalization with conversational nuance handling
        self.metrics_collector.start_timer('normalization')
        
        if isinstance(self.text_normalizer, AdvancedTextNormalizer):
            # Use advanced normalization with conversational pattern handling
            advanced_result = self.text_normalizer.normalize_with_advanced_tracking(processed_segment.text)
            processed_segment.text = advanced_result.corrected_text
            all_corrections_applied.extend(advanced_result.corrections_applied)
            
            # Track conversational fixes
            for conv_fix in advanced_result.conversational_fixes:
                self.metrics_collector.update_correction_count(metrics, f"conversational_{conv_fix.pattern_type}")
                all_corrections_applied.append(f"conversational_{conv_fix.pattern_type}")
        else:
            # Use basic normalization
            normalization_result = self.text_normalizer.normalize_with_tracking(processed_segment.text)
            processed_segment.text = normalization_result.normalized_text
            all_corrections_applied.extend(normalization_result.changes_applied)
        
        metrics.normalization_time += self.metrics_collector.end_timer('normalization')
        
        # Step 2: Enhanced contextual number processing for spiritual contexts
        self.metrics_collector.start_timer('number_processing')
        number_result = self.number_processor.process_numbers(processed_segment.text, context="spiritual")
        processed_segment.text = number_result.processed_text
        metrics.normalization_time += self.metrics_collector.end_timer('number_processing')
        
        # Track number conversions
        for conversion in number_result.conversions:
            if conversion.confidence_score >= 0.7:  # Only count high-confidence conversions
                self.metrics_collector.update_correction_count(metrics, f"contextual_number_{conversion.number_context.value}")
                all_corrections_applied.append(f"contextual_number_{conversion.number_context.value}")
        
        # Step 3: Enhanced Sanskrit/Hindi corrections (Story 2.1)
        self.metrics_collector.start_timer('sanskrit_hindi_correction')
        sanskrit_corrections = self._apply_enhanced_sanskrit_hindi_corrections(processed_segment.text)
        processed_segment.text = sanskrit_corrections['corrected_text']
        metrics.correction_time += self.metrics_collector.end_timer('sanskrit_hindi_correction')
        
        # Track Sanskrit/Hindi corrections
        for correction in sanskrit_corrections['corrections_applied']:
            correction_type = f"sanskrit_hindi_{correction.correction_type.value}"
            self.metrics_collector.update_correction_count(metrics, correction_type)
            all_corrections_applied.append(correction_type)
        
        # Step 4: Apply legacy Sanskrit/Hindi corrections (backward compatibility)
        self.metrics_collector.start_timer('legacy_correction')
        corrected_text, lexicon_corrections = self._apply_lexicon_corrections(processed_segment.text)
        processed_segment.text = corrected_text
        metrics.correction_time += self.metrics_collector.end_timer('legacy_correction')
        
        # Track legacy lexicon corrections
        for correction in lexicon_corrections:
            self.metrics_collector.update_correction_count(metrics, "legacy_lexicon_correction")
            all_corrections_applied.append("legacy_lexicon_correction")
        
        # Step 5: Apply proper noun capitalization (existing approach)
        processed_segment.text = self._apply_proper_noun_capitalization(processed_segment.text)
        
        # Step 6: Quality validation and semantic drift check
        if original_text != processed_segment.text:
            # Calculate semantic drift for this segment
            if isinstance(self.text_normalizer, AdvancedTextNormalizer):
                semantic_drift = self.text_normalizer.calculate_semantic_drift(original_text, processed_segment.text)
                
                # Flag segments with high semantic drift
                max_drift = self.config.get('quality_validation', {}).get('max_semantic_drift', 0.3)
                if semantic_drift > max_drift:
                    processed_segment.processing_flags.append("high_semantic_drift")
                    metrics.warnings_encountered.append(f"High semantic drift ({semantic_drift:.3f}) in segment")
            
            # Check for significant text length changes
            change_ratio = len(processed_segment.text) / len(original_text) if original_text else 1.0
            if abs(change_ratio - 1.0) > 0.2:  # More than 20% change
                processed_segment.processing_flags.append("significant_change")
        
        # Step 7: Calculate enhanced confidence score
        confidence = self._calculate_enhanced_confidence(
            processed_segment.text, 
            all_corrections_applied,
            original_text
        )
        processed_segment.confidence = confidence
        
        # Flag low confidence segments
        confidence_threshold = self.config.get('confidence_threshold', 0.6)
        if confidence < confidence_threshold:
            processed_segment.processing_flags.append("low_confidence")
            metrics.flagged_segments += 1
        
        # Track all applied corrections in metrics
        for correction_type in all_corrections_applied:
            self.metrics_collector.update_correction_count(metrics, correction_type)
        
        return processed_segment

    def _apply_enhanced_sanskrit_hindi_corrections(self, text: str) -> Dict[str, Any]:
        """
        Apply enhanced Sanskrit/Hindi corrections using Story 2.1 components.
        
        Args:
            text: Input text to correct
            
        Returns:
            Dictionary with corrected text and metadata
        """
        try:
            # Step 1: Identify Sanskrit/Hindi words
            identified_words = self.word_identifier.identify_words(text)
            
            # Step 2: Find fuzzy matches for potential corrections
            words = [word.strip() for word in text.split() if word.strip()]
            fuzzy_matches = []
            
            for word in words:
                matches = self.fuzzy_matcher.find_matches(word, context=text)
                fuzzy_matches.extend(matches)
            
            # Step 3: Create correction candidates
            correction_candidates = []
            
            # From identified words
            word_candidates = self.correction_applier.create_candidates_from_identified_words(
                identified_words, text
            )
            correction_candidates.extend(word_candidates)
            
            # From fuzzy matches
            fuzzy_candidates = self.correction_applier.create_candidates_from_fuzzy_matches(
                fuzzy_matches, text
            )
            correction_candidates.extend(fuzzy_candidates)
            
            # Step 4: Apply IAST transliteration if needed
            iast_result = self.iast_transliterator.transliterate_to_iast(text)
            if iast_result.changes_made:
                # Add IAST corrections as candidates
                from sanskrit_hindi_identifier.correction_applier import CorrectionCandidate, CorrectionType, CorrectionPriority
                
                for original, target in iast_result.changes_made:
                    # Find position of the change
                    position = text.find(original)
                    if position != -1:
                        candidate = CorrectionCandidate(
                            original_text=original,
                            corrected_text=target,
                            position=position,
                            length=len(original),
                            confidence=iast_result.confidence,
                            correction_type=CorrectionType.TRANSLITERATION,
                            priority=CorrectionPriority.HIGH,
                            source="iast_transliterator",
                            metadata={'rules_applied': len(iast_result.rules_applied)}
                        )
                        correction_candidates.append(candidate)
            
            # Step 5: Apply corrections
            correction_result = self.correction_applier.apply_corrections(text, correction_candidates)
            
            return {
                'original_text': text,
                'corrected_text': correction_result.corrected_text,
                'corrections_applied': correction_result.corrections_applied,
                'corrections_skipped': correction_result.corrections_skipped,
                'overall_confidence': correction_result.overall_confidence,
                'warnings': correction_result.warnings,
                'identified_words_count': len(identified_words),
                'fuzzy_matches_count': len(fuzzy_matches),
                'candidates_count': len(correction_candidates),
                'iast_changes': len(iast_result.changes_made)
            }
            
        except Exception as e:
            self.logger.error(f"Error in enhanced Sanskrit/Hindi corrections: {e}")
            # Return original text with no corrections on error
            return {
                'original_text': text,
                'corrected_text': text,
                'corrections_applied': [],
                'corrections_skipped': [],
                'overall_confidence': 1.0,
                'warnings': [f"Error in corrections: {str(e)}"],
                'identified_words_count': 0,
                'fuzzy_matches_count': 0,
                'candidates_count': 0,
                'iast_changes': 0
            }

    def _process_segment(self, segment: TranscriptSegment) -> TranscriptSegment:
        """Legacy method for processing TranscriptSegment objects."""
        original_text = segment.text
        
        # Use text normalizer
        segment.text = self.text_normalizer.normalize_text(segment.text)
        
        # Apply Sanskrit/Hindi corrections
        segment.text, corrections = self._apply_lexicon_corrections(segment.text)
        
        # Apply proper noun capitalization
        segment.text = self._apply_proper_noun_capitalization(segment.text)
        
        # Track changes
        if original_text != segment.text:
            segment.correction_history.append({
                'original': original_text,
                'corrected': segment.text,
                'corrections_applied': corrections,
                'timestamp': pd.Timestamp.now().isoformat()
            })
        
        # Calculate confidence score and flagging
        segment.confidence_score = self._calculate_confidence(segment.text, corrections)
        segment.is_flagged = segment.confidence_score < self.config.get('confidence_threshold', 0.6)
        
        if segment.is_flagged:
            segment.flag_reason = "Low confidence score"
        
        return segment

    def start_processing_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new processing session for batch operations.
        
        Args:
            session_id: Optional custom session ID
            
        Returns:
            Session ID
        """
        return self.metrics_collector.start_session(session_id)
    
    def end_processing_session(self) -> Optional[Dict[str, Any]]:
        """
        End the current processing session and return metrics.
        
        Returns:
            Session metrics dictionary or None
        """
        session = self.metrics_collector.end_session()
        if session:
            return self.metrics_collector.generate_session_report(session)
        return None

    def validate_processing_quality(
        self, 
        original_segments: List[SRTSegment], 
        processed_segments: List[SRTSegment],
        corrections_applied: List[str] = None
    ) -> Dict[str, Any]:
        """
        Validate the quality of processing results using the quality validator.
        
        Args:
            original_segments: Original SRT segments before processing
            processed_segments: Processed SRT segments
            corrections_applied: List of correction types applied
            
        Returns:
            Quality validation report dictionary
        """
        quality_report = self.quality_validator.validate_processing_quality(
            original_segments, 
            processed_segments, 
            corrections_applied or []
        )
        
        return {
            'overall_quality_score': quality_report.overall_quality_score,
            'timestamp_integrity_score': quality_report.timestamp_integrity_score,
            'semantic_preservation_score': quality_report.semantic_preservation_score,
            'correction_impact_score': quality_report.correction_impact_score,
            'validation_passed': quality_report.passed_validation,
            'total_issues': len(quality_report.validation_issues),
            'critical_issues': len([i for i in quality_report.validation_issues if i.severity.value == 'failed']),
            'warnings': len([i for i in quality_report.validation_issues if i.severity.value == 'warning']),
            'recommendations': quality_report.recommendations,
            'processing_metrics': quality_report.processing_metrics,
            'validation_details': {
                'issues': [
                    {
                        'type': issue.issue_type,
                        'severity': issue.severity.value,
                        'description': issue.description,
                        'segment_index': issue.segment_index,
                        'confidence': issue.confidence_score
                    } 
                    for issue in quality_report.validation_issues
                ]
            }
        }

    def _apply_lexicon_corrections(self, text: str) -> Tuple[str, List[str]]:
        """Apply lexicon-based corrections using fuzzy matching."""
        corrections_applied = []
        words = text.split()
        
        for i, word in enumerate(words):
            # Clean word for matching
            clean_word = re.sub(r'[^\w]', '', word.lower())
            
            # Try exact match first
            if clean_word in self.corrections:
                entry = self.corrections[clean_word]
                words[i] = self._preserve_case_and_punctuation(word, entry.transliteration)
                corrections_applied.append(f"{clean_word} -> {entry.transliteration}")
                continue
            
            # Try fuzzy matching
            best_match = self._fuzzy_match_lexicon(clean_word)
            if best_match:
                entry, score = best_match
                
                # Avoid replacing single words with multi-word phrases unless very high confidence
                entry_word_count = len(entry.transliteration.split())
                if entry_word_count > 1 and score < 95:
                    # Skip this fuzzy match - too risky
                    continue
                    
                words[i] = self._preserve_case_and_punctuation(word, entry.transliteration)
                corrections_applied.append(f"{clean_word} -> {entry.transliteration} (fuzzy: {score})")
        
        return ' '.join(words), corrections_applied

    def _fuzzy_match_lexicon(self, word: str) -> Optional[Tuple[LexiconEntry, int]]:
        """Find best fuzzy match in lexicons."""
        all_terms = list(self.corrections.keys()) + list(self.proper_nouns.keys()) + list(self.phrases.keys())
        
        if not all_terms:
            return None
        
        match = process.extractOne(word, all_terms, scorer=fuzz.ratio)
        
        if match and match[1] >= self.fuzzy_threshold:
            matched_term = match[0]
            
            # Find the entry
            for lexicon in [self.corrections, self.proper_nouns, self.phrases]:
                if matched_term in lexicon:
                    return lexicon[matched_term], match[1]
        
        return None

    def _apply_proper_noun_capitalization(self, text: str) -> str:
        """Apply proper noun capitalization."""
        for term, entry in self.proper_nouns.items():
            if entry.is_proper_noun:
                # Use word boundaries to avoid partial matches
                pattern = rf'\b{re.escape(term)}\b'
                text = re.sub(pattern, entry.transliteration, text, flags=re.IGNORECASE)
        
        return text

    def get_processing_report(self, metrics: ProcessingMetrics) -> Dict[str, Any]:
        """
        Generate a comprehensive processing report for a file.
        
        Args:
            metrics: ProcessingMetrics to report on
            
        Returns:
            Processing report dictionary
        """
        return self.metrics_collector.generate_processing_report(metrics)

    def _preserve_case_and_punctuation(self, original_word: str, replacement: str) -> str:
        """Preserve case and punctuation from original word."""
        # Extract punctuation from original
        leading_punct = re.match(r'^[^\w]*', original_word).group()
        trailing_punct = re.search(r'[^\w]*$', original_word).group()
        
        # Apply case pattern if original was capitalized
        if original_word and original_word[0].isupper():
            replacement = replacement.capitalize()
        
        return leading_punct + replacement + trailing_punct

    def _calculate_enhanced_confidence(self, text: str, corrections: List[str], original_text: str) -> float:
        """
        Calculate enhanced confidence score considering foundational corrections.
        
        Args:
            text: Processed text
            corrections: List of corrections applied
            original_text: Original text before processing
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence score
        base_confidence = 0.85
        
        # Categorize corrections by impact
        high_confidence_corrections = [
            'converted_numbers', 'standardized_punctuation', 'fixed_capitalization',
            'contextual_number_scriptural_reference', 'contextual_number_ordinal'
        ]
        
        medium_confidence_corrections = [
            'removed_filler_words', 'lexicon_correction',
            'contextual_number_date', 'contextual_number_time'
        ]
        
        low_confidence_corrections = [
            'conversational_rescinded', 'conversational_partial_phrase',
            'handled_conversational_nuances'
        ]
        
        # Calculate correction impact
        high_impact_count = sum(1 for c in corrections if any(hc in c for hc in high_confidence_corrections))
        medium_impact_count = sum(1 for c in corrections if any(mc in c for mc in medium_confidence_corrections))
        low_impact_count = sum(1 for c in corrections if any(lc in c for lc in low_confidence_corrections))
        
        # Apply penalties based on correction types
        confidence_penalty = (
            high_impact_count * 0.05 +    # Small penalty for high-confidence corrections
            medium_impact_count * 0.08 +  # Medium penalty for medium-confidence corrections
            low_impact_count * 0.12       # Higher penalty for risky corrections
        )
        
        # Check for remaining potential issues (legacy approach)
        words = text.split()
        unknown_words = 0
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if len(clean_word) > 3 and clean_word not in self.corrections and clean_word not in self.proper_nouns:
                unknown_words += 1
        
        unknown_penalty = (unknown_words / len(words)) * 0.2 if words else 0
        
        # Additional penalty for excessive text length changes
        if original_text:
            length_change_ratio = abs(len(text) - len(original_text)) / len(original_text)
            length_penalty = min(length_change_ratio * 0.3, 0.2)  # Cap at 0.2
        else:
            length_penalty = 0
        
        # Calculate final confidence
        final_confidence = base_confidence - confidence_penalty - unknown_penalty - length_penalty
        
        return max(0.0, min(1.0, final_confidence))
    
    def _calculate_confidence(self, text: str, corrections: List[str]) -> float:
        """Calculate confidence score for the processed segment (legacy method)."""
        # Simple confidence calculation - can be enhanced
        base_confidence = 0.8
        
        # Reduce confidence for each correction made
        confidence_penalty = len(corrections) * 0.1
        
        # Check for remaining potential issues
        words = text.split()
        unknown_words = 0
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if len(clean_word) > 3 and clean_word not in self.corrections and clean_word not in self.proper_nouns:
                unknown_words += 1
        
        unknown_penalty = (unknown_words / len(words)) * 0.3 if words else 0
        
        final_confidence = max(0.0, base_confidence - confidence_penalty - unknown_penalty)
        return min(1.0, final_confidence)

    def _time_to_seconds(self, time_obj) -> float:
        """Convert pysrt time object to seconds."""
        return time_obj.hours * 3600 + time_obj.minutes * 60 + time_obj.seconds + time_obj.milliseconds / 1000.0

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about loaded lexicons and processor state."""
        stats = {
            'legacy_lexicons': {
                'corrections': len(self.corrections),
                'proper_nouns': len(self.proper_nouns),
                'phrases': len(self.phrases),
                'verses': len(self.verses)
            },
            'config': self.config,
            'fuzzy_threshold': self.fuzzy_threshold
        }
        
        # Add Story 2.1 component statistics
        try:
            # Lexicon manager stats
            stats['enhanced_lexicons'] = self.lexicon_manager.get_lexicon_statistics()
            stats['lexicon_metadata'] = {
                name: {
                    'entries_count': meta.entries_count,
                    'version': meta.version,
                    'last_updated': meta.last_updated,
                    'categories': meta.categories
                }
                for name, meta in self.lexicon_manager.get_metadata().items()
            }
            
            # Word identifier stats
            stats['word_identifier'] = self.word_identifier.get_lexicon_stats()
            
            # Fuzzy matcher stats
            stats['fuzzy_matcher'] = self.fuzzy_matcher.get_matching_stats()
            
            # Correction applier stats
            stats['correction_applier'] = self.correction_applier.get_correction_stats()
            
        except Exception as e:
            self.logger.error(f"Error getting Story 2.1 statistics: {e}")
            stats['story_2_1_error'] = str(e)
        
        return stats

    def get_sanskrit_hindi_processing_report(self) -> Dict[str, Any]:
        """
        Get a detailed report on Sanskrit/Hindi processing capabilities.
        
        Returns:
            Comprehensive report on lexicon-based correction system
        """
        try:
            report = {
                'system_info': {
                    'story_version': '2.1',
                    'components': [
                        'SanskritHindiIdentifier',
                        'LexiconManager', 
                        'FuzzyMatcher',
                        'IASTTransliterator',
                        'CorrectionApplier'
                    ],
                    'capabilities': [
                        'Sanskrit/Hindi word identification',
                        'Fuzzy matching with Levenshtein distance',
                        'IAST transliteration enforcement', 
                        'High-confidence correction application',
                        'Externalized lexicon management'
                    ]
                },
                'lexicon_summary': self.lexicon_manager.get_lexicon_statistics(),
                'configuration': {
                    'fuzzy_min_confidence': self.config.get('fuzzy_min_confidence'),
                    'correction_min_confidence': self.config.get('correction_min_confidence'),
                    'iast_strict_mode': self.config.get('iast_strict_mode'),
                    'enable_phonetic_matching': self.config.get('enable_phonetic_matching'),
                    'max_corrections_per_segment': self.config.get('max_corrections_per_segment')
                },
                'validation_results': {}
            }
            
            # Validate lexicon integrity
            all_entries = self.lexicon_manager.get_all_entries()
            sample_entry = next(iter(all_entries.values())) if all_entries else None
            if sample_entry:
                validation_issues = self.word_identifier.validate_lexicon_integrity()
                report['validation_results'] = validation_issues
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating Sanskrit/Hindi processing report: {e}")
            return {
                'error': str(e),
                'system_info': {'story_version': '2.1', 'status': 'error'}
            }