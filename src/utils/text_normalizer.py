"""
Text Normalizer for basic text corrections and standardization.

This module provides comprehensive text normalization functionality including
number conversion, filler word removal, and punctuation standardization.
"""

import re
import logging
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass


@dataclass
class NormalizationResult:
    """Result of text normalization with tracking of applied changes."""
    original_text: str
    normalized_text: str
    changes_applied: List[str]
    word_count_before: int
    word_count_after: int


class TextNormalizer:
    """
    Comprehensive text normalizer for ASR transcript cleanup.
    
    Handles number conversion, filler word removal, punctuation standardization,
    and capitalization correction with detailed change tracking.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the text normalizer.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Initialize normalization components
        self._setup_filler_words()
        self._setup_number_mappings()
        self._setup_punctuation_rules()
        
        # Tracking enabled by default
        self.track_changes = self.config.get('track_changes', True)
    
    def normalize_text(self, text: str) -> str:
        """
        Apply complete text normalization pipeline.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        result = self.normalize_with_tracking(text)
        return result.normalized_text
    
    def normalize_with_tracking(self, text: str) -> NormalizationResult:
        """
        Apply normalization with detailed change tracking.
        
        Args:
            text: Input text to normalize
            
        Returns:
            NormalizationResult with detailed tracking
        """
        if not text or not text.strip():
            return NormalizationResult(
                original_text=text,
                normalized_text=text,
                changes_applied=[],
                word_count_before=0,
                word_count_after=0
            )
        
        original_text = text
        current_text = text
        changes_applied = []
        
        word_count_before = len(current_text.split())
        
        # Step 1: Remove filler words
        if self.config.get('remove_fillers', True):
            new_text = self.remove_filler_words(current_text)
            if new_text != current_text:
                changes_applied.append("removed_filler_words")
                current_text = new_text
        
        # Step 2: Convert numbers
        if self.config.get('convert_numbers', True):
            new_text = self.convert_numbers(current_text)
            if new_text != current_text:
                changes_applied.append("converted_numbers")
                current_text = new_text
        
        # Step 3: Standardize punctuation
        if self.config.get('standardize_punctuation', True):
            new_text = self.standardize_punctuation(current_text)
            if new_text != current_text:
                changes_applied.append("standardized_punctuation")
                current_text = new_text
        
        # Step 4: Remove word repetitions
        if self.config.get('remove_repetitions', True):
            new_text = self.remove_word_repetitions(current_text)
            if new_text != current_text:
                changes_applied.append("removed_repetitions")
                current_text = new_text
        
        # Step 5: Fix capitalization
        if self.config.get('fix_capitalization', True):
            new_text = self.fix_capitalization(current_text)
            if new_text != current_text:
                changes_applied.append("fixed_capitalization")
                current_text = new_text
        
        word_count_after = len(current_text.split())
        
        return NormalizationResult(
            original_text=original_text,
            normalized_text=current_text,
            changes_applied=changes_applied,
            word_count_before=word_count_before,
            word_count_after=word_count_after
        )
    
    def convert_numbers(self, text: str) -> str:
        """
        Convert spoken numbers to digits.
        
        Args:
            text: Input text
            
        Returns:
            Text with numbers converted to digits
        """
        if not text:
            return text
        
        result = text
        
        # Handle compound numbers FIRST (e.g., "twenty five" -> "25")
        result = self._convert_compound_numbers(result)
        
        # Handle years and large numbers
        result = self._convert_year_patterns(result)
        
        # Handle ordinals (e.g., "first" -> "1st", "second" -> "2nd")
        result = self._convert_ordinals(result)
        
        # Convert remaining basic number words
        for word_num, digit in self.basic_numbers.items():
            pattern = rf'\b{re.escape(word_num)}\b'
            result = re.sub(pattern, digit, result, flags=re.IGNORECASE)
        
        return result
    
    def remove_filler_words(self, text: str) -> str:
        """
        Remove common filler words and speech disfluencies.
        
        Args:
            text: Input text
            
        Returns:
            Text with filler words removed
        """
        if not text:
            return text
        
        # Split into words while preserving punctuation context
        words = text.split()
        filtered_words = []
        
        i = 0
        while i < len(words):
            word = words[i]
            clean_word = re.sub(r'[^\w]', '', word.lower())
            
            # Check for single-word fillers
            if clean_word in self.filler_words:
                # Skip this word
                pass
            # Check for multi-word fillers
            elif self._is_multiword_filler(words, i):
                # Skip the multi-word phrase
                i += self._get_multiword_filler_length(words, i) - 1
            else:
                filtered_words.append(word)
            
            i += 1
        
        # Join words and clean up spacing
        result = ' '.join(filtered_words)
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result
    
    def standardize_punctuation(self, text: str) -> str:
        """
        Standardize punctuation and spacing.
        
        Args:
            text: Input text
            
        Returns:
            Text with standardized punctuation
        """
        if not text:
            return text
        
        result = text
        
        # Fix spacing around punctuation
        result = re.sub(r'\s+([.!?,:;])', r'\1', result)  # Remove space before punctuation
        result = re.sub(r'([.!?])([A-Z])', r'\1 \2', result)  # Add space after sentence punctuation
        result = re.sub(r'([,:;])([^\s])', r'\1 \2', result)  # Add space after commas/semicolons
        
        # Normalize quotation marks - temporarily disabled due to regex issues
        # result = re.sub(r'[""]', '"', result)  # Normalize smart quotes
        # result = re.sub(r'['']', "'", result)  # Normalize smart apostrophes
        
        # Fix multiple punctuation
        result = re.sub(r'\.{2,}', '...', result)  # Multiple periods to ellipsis
        result = re.sub(r'[!]{2,}', '!', result)  # Multiple exclamations
        result = re.sub(r'[?]{2,}', '?', result)  # Multiple questions
        
        # Normalize spacing
        result = re.sub(r'\s+', ' ', result)  # Multiple spaces
        result = result.strip()
        
        return result
    
    def fix_capitalization(self, text: str) -> str:
        """
        Fix capitalization issues.
        
        Args:
            text: Input text
            
        Returns:
            Text with corrected capitalization
        """
        if not text:
            return text
        
        # Split into sentences
        sentences = re.split(r'([.!?]+)', text)
        
        fixed_sentences = []
        for i, sentence in enumerate(sentences):
            if i % 2 == 0:  # Text sentence (not punctuation)
                sentence = sentence.strip()
                if sentence:
                    # Capitalize first letter
                    sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                    
                    # Fix "i" -> "I"
                    sentence = re.sub(r'\bi\b', 'I', sentence)
                    
                    # Fix common proper nouns that should be capitalized
                    for proper_noun in self.common_proper_nouns:
                        pattern = rf'\b{re.escape(proper_noun)}\b'
                        sentence = re.sub(pattern, proper_noun.title(), sentence, flags=re.IGNORECASE)
                
                fixed_sentences.append(sentence)
            else:  # Punctuation
                fixed_sentences.append(sentence)
        
        return ''.join(fixed_sentences)
    
    def remove_word_repetitions(self, text: str) -> str:
        """
        Remove consecutive word repetitions like 'the the' or 'and and'.
        
        Args:
            text: Input text
            
        Returns:
            Text with word repetitions removed
        """
        if not text:
            return text
        
        # Pattern to match repeated words (case-insensitive)
        repetition_pattern = r'\b(\w+)\s+\1\b'
        
        # Remove repetitions
        result = re.sub(repetition_pattern, r'\1', text, flags=re.IGNORECASE)
        
        return result
    
    def _setup_filler_words(self):
        """Setup filler words for removal."""
        default_fillers = {
            # Single word fillers
            'um', 'uh', 'uhm', 'er', 'ah', 'oh', 'mm', 'hmm',
            'like', 'actually', 'basically', 'literally', 'really',
            'sort', 'kind', 'quite', 'rather', 'pretty',
            'just', 'simply', 'essentially', 'particularly',
            'obviously', 'clearly', 'certainly', 'definitely',
            'absolutely', 'totally', 'completely', 'entirely',
            'generally', 'usually', 'typically', 'normally',
            'probably', 'maybe', 'perhaps', 'possibly'
        }
        
        # Multi-word fillers
        self.multiword_fillers = {
            'you know': 2,
            'i mean': 2,
            'sort of': 2,
            'kind of': 2,
            'you see': 2,
            'let me see': 3,
            'how do i put it': 5,
            'what i mean is': 4,
            'you know what i mean': 5
        }
        
        # Combine with config overrides
        config_fillers = set(self.config.get('additional_fillers', []))
        excluded_fillers = set(self.config.get('excluded_fillers', []))
        
        self.filler_words = (default_fillers | config_fillers) - excluded_fillers
    
    def _setup_number_mappings(self):
        """Setup number word to digit mappings."""
        self.basic_numbers = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
            'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
            'eighteen': '18', 'nineteen': '19', 'twenty': '20', 'thirty': '30',
            'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
            'eighty': '80', 'ninety': '90'
        }
        
        self.scale_numbers = {
            'hundred': 100,
            'thousand': 1000,
            'million': 1000000,
            'billion': 1000000000
        }
        
        self.ordinals = {
            'first': '1st', 'second': '2nd', 'third': '3rd', 'fourth': '4th',
            'fifth': '5th', 'sixth': '6th', 'seventh': '7th', 'eighth': '8th',
            'ninth': '9th', 'tenth': '10th', 'eleventh': '11th', 'twelfth': '12th'
        }
    
    def _setup_punctuation_rules(self):
        """Setup punctuation standardization rules."""
        self.common_proper_nouns = {
            'english', 'hindi', 'sanskrit', 'yoga', 'vedanta', 
            'bhagavad', 'gita', 'upanishad', 'vedas'
        }
    
    def _convert_compound_numbers(self, text: str) -> str:
        """Convert compound numbers like 'twenty five' to '25'."""
        # Pattern for compound numbers (e.g., "twenty one", "thirty five")
        # Process compound numbers BEFORE single numbers to avoid "20 5" issue
        compound_pattern = r'\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s+(one|two|three|four|five|six|seven|eight|nine)\b'
        
        def replace_compound(match):
            tens = match.group(1)
            ones = match.group(2)
            tens_digit = self.basic_numbers.get(tens.lower(), tens)
            ones_digit = self.basic_numbers.get(ones.lower(), ones)
            
            if tens_digit.isdigit() and ones_digit.isdigit():
                return str(int(tens_digit) + int(ones_digit))
            return match.group(0)
        
        return re.sub(compound_pattern, replace_compound, text, flags=re.IGNORECASE)
    
    def _convert_ordinals(self, text: str) -> str:
        """Convert ordinal words to ordinal numbers."""
        for ordinal_word, ordinal_num in self.ordinals.items():
            pattern = rf'\b{re.escape(ordinal_word)}\b'
            text = re.sub(pattern, ordinal_num, text, flags=re.IGNORECASE)
        return text
    
    def _convert_year_patterns(self, text: str) -> str:
        """Convert year patterns like 'two thousand five' to '2005'."""
        # Pattern for years like "two thousand five"
        year_pattern = r'\b(nineteen|twenty|two thousand)\s+(hundred\s+)?(and\s+)?(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\b'
        
        # This is a simplified version - can be enhanced for more complex year patterns
        simple_patterns = {
            'two thousand': '2000',
            'two thousand and': '200',  # Will be completed by following number
        }
        
        result = text
        for pattern, replacement in simple_patterns.items():
            result = re.sub(rf'\b{re.escape(pattern)}\b', replacement, result, flags=re.IGNORECASE)
        
        return result
    
    def _is_multiword_filler(self, words: List[str], start_index: int) -> bool:
        """Check if words starting at index form a multi-word filler."""
        for filler_phrase, length in self.multiword_fillers.items():
            if start_index + length <= len(words):
                candidate = ' '.join(words[start_index:start_index + length]).lower()
                # Remove punctuation for comparison
                candidate_clean = re.sub(r'[^\w\s]', '', candidate)
                if candidate_clean == filler_phrase:
                    return True
        return False
    
    def _get_multiword_filler_length(self, words: List[str], start_index: int) -> int:
        """Get the length of the multi-word filler starting at index."""
        for filler_phrase, length in self.multiword_fillers.items():
            if start_index + length <= len(words):
                candidate = ' '.join(words[start_index:start_index + length]).lower()
                candidate_clean = re.sub(r'[^\w\s]', '', candidate)
                if candidate_clean == filler_phrase:
                    return length
        return 1