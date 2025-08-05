"""
Text processing utilities for ASR post-processing workflow.

This module provides common text processing functions including
normalization, fuzzy matching, and linguistic processing utilities.
"""

import re
import string
import unicodedata
from typing import List, Tuple, Dict, Any, Optional
import logging

from fuzzywuzzy import fuzz, process
import pandas as pd


logger = logging.getLogger(__name__)


def normalize_text(text: str, remove_diacritics: bool = False) -> str:
    """
    Normalize text for processing.
    
    Args:
        text: Input text
        remove_diacritics: Whether to remove diacritical marks
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Unicode normalization
    normalized = unicodedata.normalize('NFKC', text)
    
    # Remove diacritics if requested (be careful with Sanskrit/Hindi!)
    if remove_diacritics:
        normalized = ''.join(
            char for char in unicodedata.normalize('NFD', normalized)
            if unicodedata.category(char) != 'Mn'
        )
    
    return normalized


def clean_whitespace(text: str) -> str:
    """
    Clean and normalize whitespace in text.
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized whitespace
    """
    if not text:
        return ""
    
    # Replace multiple whitespace with single space
    cleaned = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    cleaned = cleaned.strip()
    
    return cleaned


def remove_punctuation(text: str, keep_chars: str = "") -> str:
    """
    Remove punctuation from text, optionally keeping specified characters.
    
    Args:
        text: Input text
        keep_chars: Characters to keep (e.g., "'-")
        
    Returns:
        Text with punctuation removed
    """
    if not text:
        return ""
    
    # Create translation table
    punctuation_to_remove = string.punctuation
    
    # Remove characters we want to keep
    for char in keep_chars:
        punctuation_to_remove = punctuation_to_remove.replace(char, '')
    
    translator = str.maketrans('', '', punctuation_to_remove)
    
    return text.translate(translator)


def split_into_words(text: str, preserve_punctuation: bool = True) -> List[str]:
    """
    Split text into words, optionally preserving punctuation.
    
    Args:
        text: Input text
        preserve_punctuation: Whether to keep punctuation attached to words
        
    Returns:
        List of words
    """
    if not text:
        return []
    
    if preserve_punctuation:
        # Split on whitespace only
        words = text.split()
    else:
        # Split on whitespace and punctuation
        words = re.findall(r'\b\w+\b', text)
    
    return [word for word in words if word.strip()]


def extract_numbers_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extract numbers (both digits and words) from text.
    
    Args:
        text: Input text
        
    Returns:
        List of dictionaries with number information
    """
    numbers_found = []
    
    # Pattern for digits
    digit_pattern = r'\b\d+(?:,\d{3})*(?:\.\d+)?\b'
    digit_matches = re.finditer(digit_pattern, text)
    
    for match in digit_matches:
        numbers_found.append({
            'type': 'digit',
            'text': match.group(),
            'start': match.start(),
            'end': match.end(),
            'value': float(match.group().replace(',', ''))
        })
    
    # Pattern for number words (basic set)
    number_words = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
        'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
        'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
        'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000, 'million': 1000000
    }
    
    for word, value in number_words.items():
        pattern = rf'\b{word}\b'
        matches = re.finditer(pattern, text, re.IGNORECASE)
        
        for match in matches:
            numbers_found.append({
                'type': 'word',
                'text': match.group(),
                'start': match.start(),
                'end': match.end(),
                'value': value
            })
    
    # Sort by position in text
    numbers_found.sort(key=lambda x: x['start'])
    
    return numbers_found


def fuzzy_match_terms(query: str, choices: List[str], threshold: int = 70) -> List[Tuple[str, int]]:
    """
    Find fuzzy matches for a query term in a list of choices.
    
    Args:
        query: Query string
        choices: List of possible matches
        threshold: Minimum similarity score (0-100)
        
    Returns:
        List of (match, score) tuples above threshold
    """
    if not query or not choices:
        return []
    
    # Get all matches above threshold
    matches = process.extract(query, choices, scorer=fuzz.ratio, limit=None)
    
    # Filter by threshold
    filtered_matches = [(match, score) for match, score in matches if score >= threshold]
    
    # Sort by score (highest first)
    filtered_matches.sort(key=lambda x: x[1], reverse=True)
    
    return filtered_matches


def phonetic_similarity(word1: str, word2: str) -> float:
    """
    Calculate phonetic similarity between two words.
    
    This is a simplified phonetic matching - for Sanskrit/Hindi,
    a more sophisticated approach would be needed.
    
    Args:
        word1: First word
        word2: Second word
        
    Returns:
        Similarity score (0.0 to 1.0)
    """
    if not word1 or not word2:
        return 0.0
    
    # Simple character-based similarity
    # In a real implementation, this would use proper phonetic algorithms
    return fuzz.ratio(word1.lower(), word2.lower()) / 100.0


def extract_potential_sanskrit_terms(text: str) -> List[Dict[str, Any]]:
    """
    Extract potential Sanskrit/Hindi terms from text.
    
    This uses heuristics to identify non-English words that might be
    Sanskrit or Hindi terms requiring correction.
    
    Args:
        text: Input text
        
    Returns:
        List of potential Sanskrit/Hindi terms with metadata
    """
    potential_terms = []
    
    # Common English words to exclude
    common_english = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'about', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
        'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now'
    }
    
    words = split_into_words(text, preserve_punctuation=False)
    
    for i, word in enumerate(words):
        word_lower = word.lower()
        
        # Skip common English words
        if word_lower in common_english:
            continue
        
        # Skip pure numbers
        if word.isdigit():
            continue
        
        # Skip very short words
        if len(word) < 3:
            continue
        
        # Heuristics for potential Sanskrit/Hindi terms
        potential_indicators = [
            # Contains non-ASCII characters (devanagari, diacritics)
            any(ord(char) > 127 for char in word),
            
            # Ends with common Sanskrit suffixes
            word_lower.endswith(('ama', 'ana', 'ya', 'ika', 'ini', 'ism', 'ist')),
            
            # Contains combinations uncommon in English
            bool(re.search(r'[aeiou]{3,}', word_lower)),  # Multiple vowels
            bool(re.search(r'dh|bh|gh|kh|th|ph', word_lower)),  # Aspirated consonants
            
            # Capitalized (might be proper noun)
            word[0].isupper() and len(word) > 4,
        ]
        
        if any(potential_indicators):
            potential_terms.append({
                'word': word,
                'position': i,
                'indicators': [
                    desc for desc, present in zip([
                        'non_ascii_chars',
                        'sanskrit_suffix', 
                        'multiple_vowels',
                        'aspirated_consonants',
                        'capitalized_proper_noun'
                    ], potential_indicators) if present
                ]
            })
    
    return potential_terms


def validate_iast_transliteration(text: str) -> Dict[str, Any]:
    """
    Validate IAST (International Alphabet of Sanskrit Transliteration) text.
    
    Args:
        text: Text to validate
        
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'is_valid_iast': True,
        'errors': [],
        'warnings': [],
        'character_count': len(text),
        'diacritic_count': 0
    }
    
    # IAST character set (basic set - would need expansion for complete validation)
    iast_chars = set(
        'abcdefghijklmnopqrstuvwxyz'
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        'āīūṛṝḷḹ'  # Long vowels
        'ĀĪŪṚṜḶḸ'
        'ṅñṭḍṇ'     # Retroflex and nasals
        'ṄÑṬḌṆ'
        'śṣḥṃ'      # Sibilants and anusvara/visarga
        'ŚṢḤṂ'
        ' \t\n\r.,;:!?()-[]{}"\''  # Whitespace and punctuation
        '0123456789'  # Numbers
    )
    
    # Count diacritics
    diacritics = 'āīūṛṝḷḹṅñṭḍṇśṣḥṃĀĪŪṚṜḶḸṄÑṪḌṆŚṢḤṂ'
    validation_result['diacritic_count'] = sum(1 for char in text if char in diacritics)
    
    # Check for invalid characters
    invalid_chars = set(char for char in text if char not in iast_chars)
    
    if invalid_chars:
        validation_result['is_valid_iast'] = False
        validation_result['errors'].append(f"Invalid IAST characters found: {sorted(invalid_chars)}")
    
    # Check for common mistakes
    if 'ri' in text.lower():
        validation_result['warnings'].append("Found 'ri' - should this be 'ṛi' or 'ṛ'?")
    
    if text.count('ṃ') == 0 and text.count('ḥ') == 0 and validation_result['diacritic_count'] > 0:
        validation_result['warnings'].append("Has diacritics but no anusvara/visarga - verify completeness")
    
    return validation_result


def sentence_similarity(sent1: str, sent2: str) -> float:
    """
    Calculate similarity between two sentences.
    
    Args:
        sent1: First sentence
        sent2: Second sentence
        
    Returns:
        Similarity score (0.0 to 1.0)
    """
    if not sent1 or not sent2:
        return 0.0
    
    # Normalize sentences
    norm1 = normalize_text(clean_whitespace(sent1.lower()))
    norm2 = normalize_text(clean_whitespace(sent2.lower()))
    
    # Use fuzzy ratio
    return fuzz.ratio(norm1, norm2) / 100.0


def extract_quoted_text(text: str) -> List[Dict[str, Any]]:
    """
    Extract quoted text (potential verses or exact quotations).
    
    Args:
        text: Input text
        
    Returns:
        List of quoted text segments with metadata
    """
    quoted_segments = []
    
    # Pattern for quoted text (various quote styles)
    quote_patterns = [
        r'"([^"]+)"',           # Double quotes
        r"'([^']+)'",           # Single quotes
        r'"([^"]+)"',           # Smart quotes
        r''([^']+)'',           # Smart single quotes
    ]
    
    for i, pattern in enumerate(quote_patterns):
        matches = re.finditer(pattern, text)
        
        for match in matches:
            quoted_segments.append({
                'text': match.group(1),
                'full_match': match.group(0),
                'start': match.start(),
                'end': match.end(),
                'quote_type': ['double', 'single', 'smart_double', 'smart_single'][i]
            })
    
    # Sort by position
    quoted_segments.sort(key=lambda x: x['start'])
    
    return quoted_segments