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
from ..utils.srt_parser import SRTParser, SRTSegment
from ..utils.text_normalizer import TextNormalizer
from ..utils.metrics_collector import MetricsCollector, ProcessingMetrics
from ..utils.logger_config import get_logger


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
        self.text_normalizer = TextNormalizer(self.config.get('text_normalization', {}))
        self.metrics_collector = MetricsCollector(self.config.get('metrics', {}))
        
        # Initialize lexicons
        self.corrections: Dict[str, LexiconEntry] = {}
        self.proper_nouns: Dict[str, LexiconEntry] = {}
        self.phrases: Dict[str, LexiconEntry] = {}
        self.verses: Dict[str, LexiconEntry] = {}
        
        # Load lexicons from external files
        self._load_lexicons()
        
        # Fuzzy matching threshold
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
            'lexicon_paths': {
                'corrections': 'data/lexicons/corrections.yaml',
                'proper_nouns': 'data/lexicons/proper_nouns.yaml',
                'phrases': 'data/lexicons/phrases.yaml',
                'verses': 'data/lexicons/verses.yaml'
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
        Process a single SRT segment with the new pipeline.
        
        Args:
            segment: SRT segment to process
            metrics: ProcessingMetrics object to update
            
        Returns:
            Processed SRT segment
        """
        original_text = segment.text
        
        # Step 1: Text normalization using new TextNormalizer
        self.metrics_collector.start_timer('normalization')
        normalization_result = self.text_normalizer.normalize_with_tracking(segment.text)
        segment.text = normalization_result.normalized_text
        metrics.normalization_time += self.metrics_collector.end_timer('normalization')
        
        # Track normalization changes
        for change in normalization_result.changes_applied:
            self.metrics_collector.update_correction_count(metrics, f"normalization_{change}")
        
        # Step 2: Apply Sanskrit/Hindi corrections
        self.metrics_collector.start_timer('correction')
        corrected_text, corrections = self._apply_lexicon_corrections(segment.text)
        segment.text = corrected_text
        metrics.correction_time += self.metrics_collector.end_timer('correction')
        
        # Track lexicon corrections
        for correction in corrections:
            self.metrics_collector.update_correction_count(metrics, "lexicon_correction")
        
        # Step 3: Apply proper noun capitalization
        segment.text = self._apply_proper_noun_capitalization(segment.text)
        
        # Add processing flags for significant changes
        if original_text != segment.text:
            change_ratio = len(segment.text) / len(original_text) if original_text else 1.0
            if abs(change_ratio - 1.0) > 0.2:  # More than 20% change
                segment.processing_flags.append("significant_change")
        
        # Calculate confidence score
        confidence = self._calculate_confidence(segment.text, corrections)
        segment.confidence = confidence
        
        # Flag low confidence segments
        confidence_threshold = self.config.get('confidence_threshold', 0.6)
        if confidence < confidence_threshold:
            segment.processing_flags.append("low_confidence")
            metrics.flagged_segments += 1
        
        return segment

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

    def _calculate_confidence(self, text: str, corrections: List[str]) -> float:
        """Calculate confidence score for the processed segment."""
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
        """Get statistics about loaded lexicons and processor state."""
        return {
            'lexicons': {
                'corrections': len(self.corrections),
                'proper_nouns': len(self.proper_nouns),
                'phrases': len(self.phrases),
                'verses': len(self.verses)
            },
            'config': self.config,
            'fuzzy_threshold': self.fuzzy_threshold
        }