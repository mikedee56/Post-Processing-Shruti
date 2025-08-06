"""
Fuzzy Matching System for Sanskrit/Hindi Correction.

This module provides sophisticated fuzzy matching capabilities for identifying
and correcting misrecognized Sanskrit and Hindi terms using various string
similarity algorithms including Levenshtein distance.
"""

import re
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum
import logging

from fuzzywuzzy import fuzz, process
from Levenshtein import distance as levenshtein_distance, ratio as levenshtein_ratio
import yaml
from pathlib import Path

from utils.logger_config import get_logger


class MatchType(Enum):
    """Types of fuzzy matches."""
    EXACT = "exact"
    PHONETIC = "phonetic"
    LEVENSHTEIN = "levenshtein"
    PARTIAL = "partial"
    TOKEN_SORT = "token_sort"
    TOKEN_SET = "token_set"


@dataclass
class FuzzyMatch:
    """Represents a fuzzy match result."""
    original_word: str
    matched_term: str
    corrected_term: str
    transliteration: str
    confidence: float
    match_type: MatchType
    is_proper_noun: bool
    category: str
    source_lexicon: str
    position: int = 0
    distance: int = 0


@dataclass
class MatchingConfig:
    """Configuration for fuzzy matching parameters."""
    min_confidence: float = 0.75
    levenshtein_threshold: float = 0.80
    phonetic_threshold: float = 0.85
    partial_threshold: float = 0.70
    token_threshold: float = 0.75
    max_edit_distance: int = 3
    enable_phonetic_matching: bool = True
    enable_compound_matching: bool = True


class FuzzyMatcher:
    """
    Advanced fuzzy matching system for Sanskrit/Hindi correction.
    
    Uses multiple similarity algorithms to find the best corrections for
    misrecognized terms, with configurable confidence thresholds.
    """

    def __init__(self, lexicon_data: Dict[str, Any], config: MatchingConfig = None):
        """
        Initialize the fuzzy matcher.
        
        Args:
            lexicon_data: Dictionary of lexicon entries
            config: Matching configuration parameters
        """
        self.logger = get_logger(__name__)
        self.lexicon_data = lexicon_data
        self.config = config or MatchingConfig()
        
        # Build search structures
        self._build_search_indexes()
        
        # Phonetic mapping for common ASR confusions
        self._init_phonetic_mappings()
        
        self.logger.info(f"Fuzzy matcher initialized with {len(self.search_terms)} terms")

    def _build_search_indexes(self) -> None:
        """Build optimized search indexes for fast matching."""
        self.search_terms: Dict[str, dict] = {}  # term -> entry info
        self.variations_map: Dict[str, str] = {}  # variation -> original term
        self.term_list: List[str] = []  # for fuzzywuzzy process.extract
        
        for original_term, entry in self.lexicon_data.items():
            # Add original term
            self.search_terms[original_term] = {
                'original_term': original_term,
                'transliteration': entry.get('transliteration', ''),
                'is_proper_noun': entry.get('is_proper_noun', False),
                'category': entry.get('category', 'unknown'),
                'confidence': entry.get('confidence', 1.0),
                'source_authority': entry.get('source_authority', 'lexicon')
            }
            self.term_list.append(original_term)
            
            # Add variations
            for variation in entry.get('variations', []):
                variation_lower = variation.lower()
                self.variations_map[variation_lower] = original_term
                self.search_terms[variation_lower] = self.search_terms[original_term]
                self.term_list.append(variation_lower)

    def _init_phonetic_mappings(self) -> None:
        """Initialize phonetic mappings for common ASR confusions."""
        self.phonetic_patterns = [
            # Sanskrit phonetic confusions
            (r'v', 'w'),      # v/w confusion
            (r'w', 'v'),      
            (r'th', 't'),     # th/t confusion
            (r't', 'th'),
            (r'kh', 'k'),     # aspirated/unaspirated
            (r'k', 'kh'),
            (r'gh', 'g'),
            (r'g', 'gh'),
            (r'ch', 'c'),     # ch/c confusion
            (r'c', 'ch'),
            (r'sh', 's'),     # sh/s confusion
            (r's', 'sh'),
            (r'n', 'nn'),     # single/double consonants
            (r'nn', 'n'),
            (r'aa', 'a'),     # long/short vowels
            (r'a', 'aa'),
            (r'ii', 'i'),
            (r'i', 'ii'),
            (r'uu', 'u'),
            (r'u', 'uu'),
        ]
        
        # Common word endings confusion
        self.ending_patterns = [
            (r'a$', 'aa'),
            (r'aa$', 'a'),
            (r'i$', 'ii'),
            (r'ii$', 'i'),
            (r'u$', 'uu'),
            (r'uu$', 'u'),
        ]

    def find_matches(self, word: str, context: str = "", max_matches: int = 5) -> List[FuzzyMatch]:
        """
        Find fuzzy matches for a given word.
        
        Args:
            word: Word to find matches for
            context: Surrounding context for better matching
            max_matches: Maximum number of matches to return
            
        Returns:
            List of fuzzy matches sorted by confidence
        """
        matches = []
        word_lower = word.lower()
        
        # 1. Exact match check (highest priority)
        exact_matches = self._find_exact_matches(word_lower)
        matches.extend(exact_matches)
        
        # 2. Phonetic matches (for common ASR confusions)
        if self.config.enable_phonetic_matching:
            phonetic_matches = self._find_phonetic_matches(word_lower)
            matches.extend(phonetic_matches)
        
        # 3. Levenshtein-based matches
        levenshtein_matches = self._find_levenshtein_matches(word_lower)
        matches.extend(levenshtein_matches)
        
        # 4. Partial and token-based matches
        partial_matches = self._find_partial_matches(word_lower, context)
        matches.extend(partial_matches)
        
        # 5. Compound word matches
        if self.config.enable_compound_matching:
            compound_matches = self._find_compound_matches(word_lower)
            matches.extend(compound_matches)
        
        # Remove duplicates and sort by confidence
        matches = self._deduplicate_matches(matches)
        matches.sort(key=lambda m: m.confidence, reverse=True)
        
        # Filter by minimum confidence and limit results
        filtered_matches = [m for m in matches if m.confidence >= self.config.min_confidence]
        
        return filtered_matches[:max_matches]

    def _find_exact_matches(self, word: str) -> List[FuzzyMatch]:
        """Find exact matches in lexicon."""
        matches = []
        
        if word in self.search_terms:
            entry = self.search_terms[word]
            match = FuzzyMatch(
                original_word=word,
                matched_term=word,
                corrected_term=entry['original_term'],
                transliteration=entry['transliteration'],
                confidence=1.0,
                match_type=MatchType.EXACT,
                is_proper_noun=entry['is_proper_noun'],
                category=entry['category'],
                source_lexicon=entry['source_authority']
            )
            matches.append(match)
        
        return matches

    def _find_phonetic_matches(self, word: str) -> List[FuzzyMatch]:
        """Find matches using phonetic transformations."""
        matches = []
        
        # Generate phonetic variations
        phonetic_variants = self._generate_phonetic_variants(word)
        
        for variant in phonetic_variants:
            if variant in self.search_terms:
                entry = self.search_terms[variant]
                confidence = self._calculate_phonetic_confidence(word, variant)
                
                if confidence >= self.config.phonetic_threshold:
                    match = FuzzyMatch(
                        original_word=word,
                        matched_term=variant,
                        corrected_term=entry['original_term'],
                        transliteration=entry['transliteration'],
                        confidence=confidence,
                        match_type=MatchType.PHONETIC,
                        is_proper_noun=entry['is_proper_noun'],
                        category=entry['category'],
                        source_lexicon=entry['source_authority']
                    )
                    matches.append(match)
        
        return matches

    def _find_levenshtein_matches(self, word: str) -> List[FuzzyMatch]:
        """Find matches using Levenshtein distance."""
        matches = []
        
        for term in self.search_terms:
            distance = levenshtein_distance(word, term)
            
            if distance <= self.config.max_edit_distance:
                ratio = levenshtein_ratio(word, term)
                
                if ratio >= self.config.levenshtein_threshold:
                    entry = self.search_terms[term]
                    match = FuzzyMatch(
                        original_word=word,
                        matched_term=term,
                        corrected_term=entry['original_term'],
                        transliteration=entry['transliteration'],
                        confidence=ratio,
                        match_type=MatchType.LEVENSHTEIN,
                        is_proper_noun=entry['is_proper_noun'],
                        category=entry['category'],
                        source_lexicon=entry['source_authority'],
                        distance=distance
                    )
                    matches.append(match)
        
        return matches

    def _find_partial_matches(self, word: str, context: str = "") -> List[FuzzyMatch]:
        """Find partial and token-based matches."""
        matches = []
        
        # Use fuzzywuzzy for various matching strategies
        strategies = [
            (fuzz.partial_ratio, MatchType.PARTIAL, self.config.partial_threshold),
            (fuzz.token_sort_ratio, MatchType.TOKEN_SORT, self.config.token_threshold),
            (fuzz.token_set_ratio, MatchType.TOKEN_SET, self.config.token_threshold)
        ]
        
        for strategy_func, match_type, threshold in strategies:
            # Get top matches using fuzzywuzzy
            fuzzy_matches = process.extract(word, self.term_list, scorer=strategy_func, limit=5)
            
            for matched_term, score in fuzzy_matches:
                confidence = score / 100.0  # Convert to 0-1 range
                
                if confidence >= threshold:
                    entry = self.search_terms[matched_term]
                    match = FuzzyMatch(
                        original_word=word,
                        matched_term=matched_term,
                        corrected_term=entry['original_term'],
                        transliteration=entry['transliteration'],
                        confidence=confidence,
                        match_type=match_type,
                        is_proper_noun=entry['is_proper_noun'],
                        category=entry['category'],
                        source_lexicon=entry['source_authority']
                    )
                    matches.append(match)
        
        return matches

    def _find_compound_matches(self, word: str) -> List[FuzzyMatch]:
        """Find matches for compound words by breaking them down."""
        matches = []
        
        # Try to find Sanskrit/Hindi components in the word
        for term in self.search_terms:
            if len(term) >= 3 and term in word:  # Minimum component length
                # Calculate confidence based on component coverage
                coverage = len(term) / len(word)
                if coverage >= 0.4:  # At least 40% coverage
                    entry = self.search_terms[term]
                    confidence = coverage * 0.85  # Reduce confidence for partial matches
                    
                    match = FuzzyMatch(
                        original_word=word,
                        matched_term=term,
                        corrected_term=entry['original_term'],
                        transliteration=entry['transliteration'],
                        confidence=confidence,
                        match_type=MatchType.PARTIAL,
                        is_proper_noun=entry['is_proper_noun'],
                        category=entry['category'],
                        source_lexicon=entry['source_authority']
                    )
                    matches.append(match)
        
        return matches

    def _generate_phonetic_variants(self, word: str) -> Set[str]:
        """Generate phonetic variants of a word."""
        variants = {word}  # Include original
        
        # Apply phonetic transformations
        for pattern, replacement in self.phonetic_patterns:
            variant = re.sub(pattern, replacement, word)
            if variant != word:
                variants.add(variant)
        
        # Apply ending transformations
        for pattern, replacement in self.ending_patterns:
            variant = re.sub(pattern, replacement, word)
            if variant != word:
                variants.add(variant)
        
        return variants

    def _calculate_phonetic_confidence(self, original: str, variant: str) -> float:
        """Calculate confidence for phonetic matches."""
        # Use Levenshtein ratio as base confidence
        base_confidence = levenshtein_ratio(original, variant)
        
        # Boost confidence for common phonetic patterns
        if self._is_common_phonetic_pattern(original, variant):
            base_confidence = min(1.0, base_confidence * 1.1)
        
        return base_confidence

    def _is_common_phonetic_pattern(self, word1: str, word2: str) -> bool:
        """Check if the difference represents a common phonetic pattern."""
        # Simple heuristic: if words differ by only one character substitution
        # and it matches our known patterns
        if abs(len(word1) - len(word2)) <= 1:
            distance = levenshtein_distance(word1, word2)
            return distance <= 2
        return False

    def _deduplicate_matches(self, matches: List[FuzzyMatch]) -> List[FuzzyMatch]:
        """Remove duplicate matches, keeping the highest confidence ones."""
        seen = {}
        deduplicated = []
        
        for match in matches:
            key = (match.corrected_term, match.original_word)
            
            if key not in seen or match.confidence > seen[key].confidence:
                seen[key] = match
        
        return list(seen.values())

    def batch_match(self, words: List[str], context: str = "") -> Dict[str, List[FuzzyMatch]]:
        """
        Find matches for multiple words efficiently.
        
        Args:
            words: List of words to match
            context: Shared context for all words
            
        Returns:
            Dictionary mapping words to their matches
        """
        results = {}
        
        for word in words:
            matches = self.find_matches(word, context)
            if matches:
                results[word] = matches
        
        return results

    def get_matching_stats(self) -> Dict[str, Any]:
        """Get statistics about the matching system."""
        return {
            'total_search_terms': len(self.search_terms),
            'total_variations': len(self.variations_map),
            'phonetic_patterns': len(self.phonetic_patterns),
            'ending_patterns': len(self.ending_patterns),
            'config': {
                'min_confidence': self.config.min_confidence,
                'levenshtein_threshold': self.config.levenshtein_threshold,
                'phonetic_threshold': self.config.phonetic_threshold,
                'max_edit_distance': self.config.max_edit_distance
            }
        }