"""
Sanskrit/Hindi Word Identifier Module.

This module provides functionality to identify Sanskrit and Hindi words in text
by checking against externalized lexicons and standard English dictionaries.
It forms the core of the lexicon-based correction system for ASR transcripts.
"""

import re
import yaml
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

from utils.logger_config import get_logger
from .sandhi_preprocessor import SandhiPreprocessor, SandhiSplitResult


class WordCategory(Enum):
    """Categories for Sanskrit/Hindi words."""
    DEITY = "deity"
    SCRIPTURE = "scripture"
    CONCEPT = "concept"
    PRACTICE = "practice"
    PHILOSOPHY = "philosophy"
    CHARACTER = "character"
    TEACHER = "teacher"
    REFERENCE = "reference"
    TEMPORAL = "temporal"


@dataclass
class IdentifiedWord:
    """Represents an identified Sanskrit/Hindi word with metadata."""
    word: str
    position: int
    category: WordCategory
    confidence: float
    is_proper_noun: bool
    source_lexicon: str
    transliteration: Optional[str] = None
    variations: List[str] = None

    def __post_init__(self):
        if self.variations is None:
            self.variations = []


@dataclass
class LexiconEntry:
    """Represents a lexicon entry with all its metadata."""
    original_term: str
    variations: List[str]
    transliteration: str
    is_proper_noun: bool
    category: str
    confidence: float
    source_authority: str


class SanskritHindiIdentifier:
    """
    Core module for identifying Sanskrit and Hindi words in text.
    
    Uses externalized lexicons to identify non-English words that are likely
    to be Sanskrit or Hindi terms requiring correction or transliteration.
    """

    def __init__(self, lexicon_dir: Path = None, english_words_file: Path = None, 
                 enable_sandhi_preprocessing: bool = True):
        """
        Initialize the Sanskrit/Hindi identifier.
        
        Args:
            lexicon_dir: Directory containing lexicon files
            english_words_file: Path to English dictionary file (optional)
            enable_sandhi_preprocessing: Enable sandhi preprocessing for compound words
        """
        self.logger = get_logger(__name__)
        self.lexicon_dir = lexicon_dir or Path("data/lexicons")
        self.english_words_file = english_words_file
        self.enable_sandhi_preprocessing = enable_sandhi_preprocessing
        
        # Initialize data structures
        self.sanskrit_hindi_lexicon: Dict[str, LexiconEntry] = {}
        self.variation_lookup: Dict[str, str] = {}  # variation -> original_term
        self.english_words: Set[str] = set()
        
        # Initialize sandhi preprocessor
        self.sandhi_preprocessor = SandhiPreprocessor(enable_sandhi_preprocessing)
        
        # Load data
        self._load_lexicons()
        self._load_english_dictionary()
        
        self.logger.info(f"Loaded {len(self.sanskrit_hindi_lexicon)} Sanskrit/Hindi terms")
        self.logger.info(f"Loaded {len(self.english_words)} English words")
        if enable_sandhi_preprocessing:
            self.logger.info("Sandhi preprocessing enabled for compound word splitting")

    def _load_lexicons(self) -> None:
        """Load all lexicon files from the lexicon directory."""
        lexicon_files = [
            "corrections.yaml",
            "proper_nouns.yaml", 
            "phrases.yaml",
            "verses.yaml"
        ]
        
        for file_name in lexicon_files:
            file_path = self.lexicon_dir / file_name
            if file_path.exists():
                self._load_lexicon_file(file_path, file_name)
            else:
                self.logger.warning(f"Lexicon file not found: {file_path}")

    def _load_lexicon_file(self, file_path: Path, source_file: str) -> None:
        """Load a single lexicon file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if 'entries' not in data:
                self.logger.warning(f"No 'entries' section found in {file_path}")
                return
            
            for entry_data in data['entries']:
                entry = LexiconEntry(
                    original_term=entry_data.get('original_term', ''),
                    variations=entry_data.get('variations', []),
                    transliteration=entry_data.get('transliteration', ''),
                    is_proper_noun=entry_data.get('is_proper_noun', False),
                    category=entry_data.get('category', 'unknown'),
                    confidence=entry_data.get('confidence', 1.0),
                    source_authority=entry_data.get('source_authority', 'unknown')
                )
                
                # Store in main lexicon
                self.sanskrit_hindi_lexicon[entry.original_term.lower()] = entry
                
                # Build variation lookup
                for variation in entry.variations:
                    self.variation_lookup[variation.lower()] = entry.original_term.lower()
                
            self.logger.info(f"Loaded {len(data['entries'])} entries from {source_file}")
            
        except Exception as e:
            self.logger.error(f"Error loading lexicon file {file_path}: {e}")

    def _load_english_dictionary(self) -> None:
        """Load English dictionary for word filtering."""
        if self.english_words_file and self.english_words_file.exists():
            try:
                with open(self.english_words_file, 'r', encoding='utf-8') as f:
                    self.english_words = {word.strip().lower() for word in f}
                self.logger.info(f"Loaded English dictionary: {len(self.english_words)} words")
            except Exception as e:
                self.logger.error(f"Error loading English dictionary: {e}")
        else:
            # Use a basic set of common English words if no dictionary file
            self._load_basic_english_words()

    def _load_basic_english_words(self) -> None:
        """Load basic English words for filtering."""
        # Basic common English words to avoid false positives
        basic_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to',
            'was', 'will', 'with', 'have', 'had', 'this', 'but', 'his', 'not', 'or',
            'can', 'we', 'all', 'were', 'they', 'one', 'their', 'said', 'each',
            'which', 'she', 'do', 'how', 'her', 'my', 'me', 'would', 'could',
            'should', 'about', 'after', 'before', 'between', 'during', 'under',
            'over', 'through', 'into', 'onto', 'upon', 'within', 'without',
            'chapter', 'verse', 'verses', 'text', 'ancient', 'times', 'nature',
            'eternal', 'soul', 'today', 'discuss', 'study', 'speaks', 'about'
        }
        self.english_words = basic_words
        self.logger.info("Loaded basic English word set for filtering")

    def identify_words(self, text: str) -> List[IdentifiedWord]:
        """
        Identify Sanskrit/Hindi words in the given text.
        
        Enhanced with sandhi preprocessing to split compound words before lexicon matching.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of identified Sanskrit/Hindi words with metadata
        """
        identified = []
        
        # Apply sandhi preprocessing if enabled
        if self.enable_sandhi_preprocessing:
            processed_text = self._apply_sandhi_preprocessing(text)
        else:
            processed_text = text
        
        # Clean and tokenize text
        words = self._tokenize_text(processed_text)
        
        for word_info in words:
            word, position = word_info
            word_lower = word.lower()
            
            # Skip if it's a common English word
            if self._is_english_word(word_lower):
                continue
            
            # Check direct match in lexicon
            if word_lower in self.sanskrit_hindi_lexicon:
                entry = self.sanskrit_hindi_lexicon[word_lower]
                identified.append(self._create_identified_word(
                    word, position, entry, "direct_match"
                ))
                continue
            
            # Check variation match
            if word_lower in self.variation_lookup:
                original_term = self.variation_lookup[word_lower]
                entry = self.sanskrit_hindi_lexicon[original_term]
                identified.append(self._create_identified_word(
                    word, position, entry, "variation_match"
                ))
                continue
            
            # Check for partial matches or compound words
            partial_matches = self._find_partial_matches(word_lower)
            if partial_matches:
                for match_info in partial_matches:
                    identified.append(self._create_identified_word(
                        word, position, match_info['entry'], "partial_match",
                        confidence_modifier=match_info['confidence']
                    ))
        
        return identified

    def _apply_sandhi_preprocessing(self, text: str) -> str:
        """
        Apply sandhi preprocessing to split compound words.
        
        Args:
            text: Original text
            
        Returns:
            Text with compound words split into components
        """
        try:
            # Split text into sentences/phrases for processing
            phrases = re.split(r'[.!?;]', text)
            processed_phrases = []
            
            for phrase in phrases:
                if not phrase.strip():
                    continue
                
                # Apply sandhi splitting to the phrase
                sandhi_result = self.sandhi_preprocessor.preprocess_text(phrase.strip())
                
                if sandhi_result.preprocessing_successful and not sandhi_result.fallback_used:
                    # Use the primary candidate's segmentation
                    primary_candidate = sandhi_result.primary_candidate
                    processed_phrase = ' '.join(primary_candidate.segments)
                    processed_phrases.append(processed_phrase)
                    
                    # Log successful sandhi preprocessing
                    if len(primary_candidate.segments) > 1:
                        self.logger.debug(
                            f"Sandhi split: '{phrase.strip()}' → {primary_candidate.segments}"
                        )
                else:
                    # Use original phrase if preprocessing failed or used fallback
                    processed_phrases.append(phrase.strip())
            
            return '. '.join(processed_phrases) if processed_phrases else text
            
        except Exception as e:
            self.logger.warning(f"Error in sandhi preprocessing: {e}")
            return text  # Return original text on error

    def _tokenize_text(self, text: str) -> List[Tuple[str, int]]:
        """
        Tokenize text into words with their positions.
        
        Args:
            text: Input text
            
        Returns:
            List of (word, position) tuples
        """
        words = []
        # Use regex to find words and their positions
        for match in re.finditer(r'\b[a-zA-Z]+\b', text):
            words.append((match.group(), match.start()))
        return words

    def _is_english_word(self, word: str) -> bool:
        """Check if a word is likely an English word."""
        return word in self.english_words

    def _find_partial_matches(self, word: str) -> List[Dict[str, Any]]:
        """
        Find partial matches for compound words or variations.
        
        Args:
            word: Word to find matches for
            
        Returns:
            List of match information dictionaries
        """
        matches = []
        
        # Check if word contains any known Sanskrit/Hindi terms
        for term, entry in self.sanskrit_hindi_lexicon.items():
            if term in word or word in term:
                confidence = len(term) / len(word) if len(word) > len(term) else len(word) / len(term)
                if confidence > 0.6:  # Minimum confidence threshold
                    matches.append({
                        'entry': entry,
                        'confidence': confidence * 0.8  # Reduce confidence for partial matches
                    })
        
        return matches

    def _create_identified_word(self, word: str, position: int, entry: LexiconEntry, 
                              match_type: str, confidence_modifier: float = 1.0) -> IdentifiedWord:
        """Create an IdentifiedWord object from lexicon entry."""
        try:
            category = WordCategory(entry.category)
        except ValueError:
            category = WordCategory.CONCEPT  # Default fallback
        
        return IdentifiedWord(
            word=word,
            position=position,
            category=category,
            confidence=entry.confidence * confidence_modifier,
            is_proper_noun=entry.is_proper_noun,
            source_lexicon=entry.source_authority,
            transliteration=entry.transliteration,
            variations=entry.variations
        )

    def get_lexicon_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded lexicons."""
        stats = {
            'total_terms': len(self.sanskrit_hindi_lexicon),
            'total_variations': len(self.variation_lookup),
            'categories': {},
            'proper_nouns': 0,
            'common_terms': 0
        }
        
        for entry in self.sanskrit_hindi_lexicon.values():
            # Count by category
            category = entry.category
            stats['categories'][category] = stats['categories'].get(category, 0) + 1
            
            # Count proper nouns vs common terms
            if entry.is_proper_noun:
                stats['proper_nouns'] += 1
            else:
                stats['common_terms'] += 1
        
        return stats

    def validate_lexicon_integrity(self) -> Dict[str, List[str]]:
        """Validate lexicon integrity and return any issues found."""
        issues = {
            'missing_transliteration': [],
            'empty_variations': [],
            'invalid_confidence': [],
            'unknown_category': []
        }
        
        valid_categories = {cat.value for cat in WordCategory}
        
        for term, entry in self.sanskrit_hindi_lexicon.items():
            if not entry.transliteration:
                issues['missing_transliteration'].append(term)
            
            if not entry.variations:
                issues['empty_variations'].append(term)
            
            if not 0 <= entry.confidence <= 1:
                issues['invalid_confidence'].append(term)
            
            if entry.category not in valid_categories:
                issues['unknown_category'].append(term)
        
        return issues

    def get_sandhi_preprocessing_stats(self) -> Dict[str, Any]:
        """Get statistics about sandhi preprocessing operations."""
        return self.sandhi_preprocessor.get_processing_statistics()

    def reset_sandhi_preprocessing_stats(self) -> None:
        """Reset sandhi preprocessing statistics."""
        self.sandhi_preprocessor.reset_statistics()

    def validate_sandhi_preprocessing_config(self) -> Dict[str, Any]:
        """Validate sandhi preprocessing configuration."""
        return self.sandhi_preprocessor.validate_configuration()

    def set_sandhi_preprocessing_enabled(self, enabled: bool) -> None:
        """
        Enable or disable sandhi preprocessing.
        
        Args:
            enabled: True to enable, False to disable
        """
        self.enable_sandhi_preprocessing = enabled
        self.sandhi_preprocessor.enable_preprocessing = enabled
        
        if enabled:
            self.logger.info("Sandhi preprocessing enabled")
        else:
            self.logger.info("Sandhi preprocessing disabled")