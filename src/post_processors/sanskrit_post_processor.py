"""
Sanskrit Post-Processor for ASR Transcript Correction.

This module provides the core functionality for processing ASR-generated
transcripts of Yoga Vedanta lectures, with specialized handling for
Sanskrit and Hindi terminology, IAST transliteration, and scriptural verses.
"""

import re
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

import pysrt
import pandas as pd
from fuzzywuzzy import fuzz, process


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
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        # Initialize lexicons
        self.corrections: Dict[str, LexiconEntry] = {}
        self.proper_nouns: Dict[str, LexiconEntry] = {}
        self.phrases: Dict[str, LexiconEntry] = {}
        self.verses: Dict[str, LexiconEntry] = {}
        
        # Load lexicons from external files
        self._load_lexicons()
        
        # Fuzzy matching threshold
        self.fuzzy_threshold = self.config.get('fuzzy_threshold', 80)
        
        # Common English filler words to remove
        self.filler_words = {
            'um', 'uh', 'uhm', 'er', 'ah', 'oh', 'you know', 
            'like', 'actually', 'basically', 'literally'
        }
        
        # Number word mappings for conversion
        self.number_words = self._build_number_mappings()

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

    def _build_number_mappings(self) -> Dict[str, str]:
        """Build mappings for converting spoken numbers to digits."""
        return {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
            'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
            'eighteen': '18', 'nineteen': '19', 'twenty': '20', 'thirty': '30',
            'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
            'eighty': '80', 'ninety': '90', 'hundred': '100', 'thousand': '1000',
            'million': '1000000', 'billion': '1000000000'
        }

    def process_srt_file(self, input_path: Path, output_path: Path) -> Dict[str, Any]:
        """
        Process an SRT file with Sanskrit/Hindi corrections.
        
        Args:
            input_path: Path to input SRT file
            output_path: Path to output SRT file
            
        Returns:
            Dictionary with processing results and metrics
        """
        try:
            # Load SRT file
            subs = pysrt.open(input_path, encoding='utf-8')
            
            processing_results = {
                'input_file': str(input_path),
                'output_file': str(output_path),
                'total_segments': len(subs),
                'corrections_made': 0,
                'flagged_segments': 0,
                'confidence_scores': []
            }
            
            # Process each subtitle segment
            for i, sub in enumerate(subs):
                segment = TranscriptSegment(
                    id=f"seg_{i:04d}",
                    text=sub.text,
                    start_time=self._time_to_seconds(sub.start),
                    end_time=self._time_to_seconds(sub.end)
                )
                
                # Apply corrections
                processed_segment = self._process_segment(segment)
                
                # Update subtitle text
                sub.text = processed_segment.text
                
                # Track metrics
                if processed_segment.correction_history:
                    processing_results['corrections_made'] += len(processed_segment.correction_history)
                
                if processed_segment.is_flagged:
                    processing_results['flagged_segments'] += 1
                
                processing_results['confidence_scores'].append(processed_segment.confidence_score)
            
            # Save processed SRT
            subs.save(output_path, encoding='utf-8')
            
            # Calculate average confidence
            if processing_results['confidence_scores']:
                processing_results['average_confidence'] = sum(processing_results['confidence_scores']) / len(processing_results['confidence_scores'])
            else:
                processing_results['average_confidence'] = 0.0
            
            self.logger.info(f"Processed {input_path} -> {output_path}")
            self.logger.info(f"Corrections: {processing_results['corrections_made']}, Flagged: {processing_results['flagged_segments']}")
            
            return processing_results
            
        except Exception as e:
            self.logger.error(f"Error processing SRT file {input_path}: {e}")
            raise

    def _process_segment(self, segment: TranscriptSegment) -> TranscriptSegment:
        """Process a single transcript segment."""
        original_text = segment.text
        
        # Step 1: Remove filler words
        segment.text = self._remove_filler_words(segment.text)
        
        # Step 2: Convert spoken numbers to digits
        segment.text = self._convert_numbers(segment.text)
        
        # Step 3: Apply Sanskrit/Hindi corrections
        segment.text, corrections = self._apply_lexicon_corrections(segment.text)
        
        # Step 4: Apply proper noun capitalization
        segment.text = self._apply_proper_noun_capitalization(segment.text)
        
        # Step 5: Normalize punctuation and spacing
        segment.text = self._normalize_punctuation(segment.text)
        
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

    def _remove_filler_words(self, text: str) -> str:
        """Remove common English filler words."""
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.filler_words]
        return ' '.join(filtered_words)

    def _convert_numbers(self, text: str) -> str:
        """Convert spoken numbers to digits."""
        # Simple implementation - can be enhanced for complex number phrases
        for word_num, digit in self.number_words.items():
            text = re.sub(rf'\b{word_num}\b', digit, text, flags=re.IGNORECASE)
        return text

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

    def _normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation and spacing."""
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Add space after punctuation before capital
        text = re.sub(r'\s+', ' ', text)  # Normalize multiple spaces
        
        return text.strip()

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