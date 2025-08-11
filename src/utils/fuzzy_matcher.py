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
    min_confidence: float = 0.88  # Increased from 0.75 - more conservative
    levenshtein_threshold: float = 0.88  # Increased from 0.80 - more conservative
    phonetic_threshold: float = 0.90  # Increased from 0.85 - more conservative
    partial_threshold: float = 0.85  # Increased from 0.70 - more conservative
    token_threshold: float = 0.88  # Increased from 0.75 - more conservative
    max_edit_distance: int = 2  # Decreased from 3 - less aggressive
    enable_phonetic_matching: bool = True
    enable_compound_matching: bool = True  # Will be controlled by improved logic


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
        
        # ULTRA-CONSERVATIVE ANTI-HALLUCINATION SAFEGUARDS
        # Skip very short words that are likely English - INCREASED THRESHOLD
        if len(word) <= 7:  # Increased from 5 to 7 - MUCH more conservative
            return matches
        
        # Comprehensive English protected words - identical to sanskrit_post_processor.py
        english_protected_words = {
            # Function words
            'who', 'what', 'when', 'where', 'why', 'how', 'and', 'the', 'is', 'are', 'was', 'were', 
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 
            'should', 'may', 'might', 'can', 'must', 'shall', 'ought',
            
            # Pronouns
            'i', 'me', 'my', 'mine', 'myself', 'you', 'your', 'yours', 'yourself', 'he', 'him', 
            'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'we', 'us', 
            'our', 'ours', 'ourselves', 'they', 'them', 'their', 'theirs', 'themselves',
            
            # Determiners and articles
            'this', 'that', 'these', 'those', 'a', 'an', 'some', 'any', 'all', 'every', 'each',
            
            # Prepositions
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 
            'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'under', 
            'over', 'across', 'beside', 'behind', 'beyond', 'within', 'without', 'against',
            
            # Conjunctions  
            'but', 'or', 'nor', 'so', 'yet', 'because', 'since', 'unless', 'while', 'although', 
            'though', 'if', 'when', 'where', 'whether',
            
            # Common adverbs
            'very', 'quite', 'rather', 'too', 'more', 'most', 'less', 'least', 'much', 'many', 
            'few', 'little', 'enough', 'only', 'just', 'even', 'also', 'already', 'still', 'yet', 
            'again', 'once', 'twice', 'here', 'there', 'now', 'then', 'today', 'tomorrow', 'yesterday',
            
            # Common verbs that could be mismatched
            'see', 'look', 'hear', 'listen', 'feel', 'think', 'know', 'understand', 'remember', 
            'forget', 'learn', 'teach', 'tell', 'say', 'speak', 'talk', 'ask', 'answer', 'call',
            'come', 'go', 'bring', 'take', 'get', 'give', 'put', 'make', 'let', 'help',
            
            # Spiritual/religious context words that should remain English
            'chapter', 'verse', 'entitled', 'text', 'scripture', 'book', 'page', 'line',
            'meditation', 'practice', 'teaching', 'lesson', 'study', 'read', 'recite',
            'prayer', 'worship', 'devotion', 'faith', 'belief', 'truth', 'wisdom',
            
            # Numbers and time
            'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
            'first', 'second', 'third', 'fourth', 'fifth', 'last', 'next', 'previous',
            'hour', 'minute', 'second', 'day', 'week', 'month', 'year', 'time',
            
            # Common adjectives
            'good', 'bad', 'big', 'small', 'great', 'little', 'long', 'short', 'high', 'low',
            'new', 'old', 'young', 'ancient', 'modern', 'early', 'late', 'fast', 'slow',
            'hot', 'cold', 'warm', 'cool', 'light', 'dark', 'bright', 'clear', 'clean', 'dirty'
        }
        
        # ABSOLUTE PROTECTION: Skip ALL protected English words - NO EXCEPTIONS EVER
        if word.lower() in english_protected_words:
            return matches
            
        # ABSOLUTE PROTECTION: Skip ANY word that contains protected substrings
        word_lower = word.lower()
        if any(protected in word_lower for protected in ['is', 'as', 'and', 'the', 'who', 'one', 'two', 'three', 'was', 'are', 'has', 'had']):
            return matches
        
        # Skip words that look like English based on patterns
        if self._is_likely_english_word(word):
            return matches
        
        # Use EXTREMELY conservative thresholds - PREVENT ALL HALLUCINATION
        conservative_thresholds = {
            MatchType.PARTIAL: max(0.98, self.config.partial_threshold),  # Increased to 98% - nearly exact
            MatchType.TOKEN_SORT: max(0.98, self.config.token_threshold),  # Increased to 98% - nearly exact
            MatchType.TOKEN_SET: max(0.98, self.config.token_threshold)    # Increased to 98% - nearly exact
        }
        
        # Use fuzzywuzzy for various matching strategies
        strategies = [
            (fuzz.partial_ratio, MatchType.PARTIAL, conservative_thresholds[MatchType.PARTIAL]),
            (fuzz.token_sort_ratio, MatchType.TOKEN_SORT, conservative_thresholds[MatchType.TOKEN_SORT]),
            (fuzz.token_set_ratio, MatchType.TOKEN_SET, conservative_thresholds[MatchType.TOKEN_SET])
        ]
        
        for strategy_func, match_type, threshold in strategies:
            # Get top matches using fuzzywuzzy - limit to 2 to avoid noise
            fuzzy_matches = process.extract(word, self.term_list, scorer=strategy_func, limit=2)
            
            for matched_term, score in fuzzy_matches:
                confidence = score / 100.0  # Convert to 0-1 range
                
                if confidence >= threshold:
                    # ULTRA-STRICT validation: ensure VERY similar length
                    length_ratio = min(len(word), len(matched_term)) / max(len(word), len(matched_term))
                    if length_ratio < 0.9:  # Increased from 0.6 to 0.9 - must be very similar length
                        continue
                        
                    # ADDITIONAL SAFETY: Character composition similarity
                    word_chars = set(word.lower())
                    term_chars = set(matched_term.lower())
                    char_overlap = len(word_chars.intersection(term_chars)) / len(word_chars.union(term_chars))
                    if char_overlap < 0.85:  # At least 85% character overlap required
                        continue
                    
                    entry = self.search_terms[matched_term]
                    
                    # Apply conservative confidence penalty
                    adjusted_confidence = confidence * 0.9  # 10% penalty for partial matches
                    
                    match = FuzzyMatch(
                        original_word=word,
                        matched_term=matched_term,
                        corrected_term=entry['original_term'],
                        transliteration=entry['transliteration'],
                        confidence=adjusted_confidence,
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
        
        # ULTRA-CONSERVATIVE ANTI-HALLUCINATION SAFEGUARDS  
        # DISABLE COMPOUND MATCHING COMPLETELY - too dangerous for hallucination
        return matches  # Completely disable compound matching to prevent false positives
        
        # Skip words that look like English
        if self._is_likely_english_word(word):
            return matches
        
        # Try to find Sanskrit/Hindi components in the word
        for term in self.search_terms:
            # Require minimum 5 character terms for compound matching to avoid short false positives
            if len(term) >= 5 and term in word:
                # Calculate confidence based on component coverage
                coverage = len(term) / len(word)
                
                # Require much higher coverage (75% minimum) to prevent false matches
                if coverage >= 0.75:
                    entry = self.search_terms[term]
                    
                    # Additional validation: term must be at word boundary or start/end
                    word_start_idx = word.find(term)
                    is_valid_boundary = (
                        word_start_idx == 0 or  # At start
                        word_start_idx + len(term) == len(word) or  # At end
                        not word[word_start_idx - 1].isalpha() or  # After non-letter
                        not word[word_start_idx + len(term)].isalpha()  # Before non-letter
                    )
                    
                    if is_valid_boundary:
                        # Much more conservative confidence calculation
                        confidence = coverage * 0.65  # Further reduced for safety
                        
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

    def _is_likely_english_word(self, word: str) -> bool:
        """
        Determine if a word is likely English based on letter patterns and structure.
        """
        word_lower = word.lower()
        
        # Very short words are often English function words
        if len(word_lower) <= 3:
            return True
        
        # Common English word endings
        english_endings = {
            'ed', 'ing', 'ly', 'er', 'est', 'tion', 'sion', 'ment', 'ness', 
            'able', 'ible', 'ful', 'less', 'ward', 'wise', 'like', 'ship'
        }
        
        # Check if word ends with common English suffixes
        for ending in english_endings:
            if word_lower.endswith(ending):
                return True
        
        # Common English prefixes
        english_prefixes = {
            'un', 're', 'pre', 'dis', 'mis', 'over', 'under', 'out', 'up', 'in', 'im'
        }
        
        # Check if word starts with common English prefixes
        for prefix in english_prefixes:
            if word_lower.startswith(prefix):
                return True
        
        # English-specific letter patterns
        if word_lower.count('th') > 0 or word_lower.count('ck') > 0 or word_lower.count('qu') > 0:
            return True
        
        # Double letters more common in English
        if any(word_lower.count(letter) > 1 for letter in 'llssttffpp'):
            return True
        
        return False

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