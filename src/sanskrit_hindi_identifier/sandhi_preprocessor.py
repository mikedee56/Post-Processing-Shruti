"""
Sanskrit Sandhi Preprocessing Module.

This module provides functionality to split Sanskrit compound words (sandhi) before
lexicon matching, enhancing word identification accuracy for the existing Story 2.1 
lexicon-based correction system.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from utils.logger_config import get_logger


class SegmentationConfidenceLevel(Enum):
    """Confidence levels for sandhi segmentation."""
    HIGH = "high"
    MEDIUM = "medium" 
    LOW = "low"
    FALLBACK = "fallback"


@dataclass
class SandhiSplitCandidate:
    """Represents a candidate segmentation of a Sanskrit compound word."""
    original_text: str
    segments: List[str]
    confidence_score: float
    confidence_level: SegmentationConfidenceLevel
    splitting_method: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SandhiSplitResult:
    """Result of sandhi splitting analysis containing all candidates."""
    original_text: str
    primary_candidate: SandhiSplitCandidate
    alternative_candidates: List[SandhiSplitCandidate]
    preprocessing_successful: bool
    fallback_used: bool
    processing_time_ms: float = 0.0
    
    @property
    def all_candidates(self) -> List[SandhiSplitCandidate]:
        """Get all candidates including primary."""
        return [self.primary_candidate] + self.alternative_candidates


class SandhiPreprocessor:
    """
    Sanskrit sandhi preprocessing component for splitting compound words.
    
    Uses sanskrit_parser library to analyze Sanskrit text and split compound words
    into their constituent parts before lexicon matching. Provides graceful fallback
    to basic tokenization when sanskrit_parser fails.
    """

    def __init__(self, enable_sandhi_preprocessing: bool = True):
        """
        Initialize the sandhi preprocessor with lazy loading for optimal performance.
        
        Args:
            enable_sandhi_preprocessing: Feature flag to enable/disable preprocessing
        """
        self.logger = get_logger(__name__)
        self.enable_preprocessing = enable_sandhi_preprocessing
        self.sanskrit_parser_available = False
        self.basic_tokenizer_fallback = True
        self._parser_initialized = False  # Lazy loading flag
        
        # PERFORMANCE OPTIMIZATION: Initialize cache for parsing results
        import functools
        self._parsing_cache = {}
        
        # Store parser classes for lazy initialization
        self._parser_classes = None
        
        # Attempt to check sanskrit_parser availability without loading Word2Vec
        try:
            import sanskrit_parser
            self.sanskrit_parser = sanskrit_parser
            self._parser_classes = {
                'Parser': sanskrit_parser.Parser,
                'LexicalSandhiAnalyzer': getattr(sanskrit_parser.api, 'LexicalSandhiAnalyzer', None)
            }
            self.sanskrit_parser_available = True
            self.logger.info("sanskrit_parser library available (lazy loading enabled)")
            
        except ImportError as e:
            self.logger.warning(f"sanskrit_parser library not available: {e}")
            self.logger.info("Will use basic tokenization fallback only")
        except Exception as e:
            self.logger.warning(f"sanskrit_parser check failed: {e}")
            self.sanskrit_parser_available = False
        
        # Initialize processing statistics
        self.stats = {
            'total_processed': 0,
            'successful_splits': 0,
            'fallback_used': 0,
            'processing_errors': 0,
            'cache_hits': 0,  # Track cache effectiveness
            'cache_misses': 0,
            'lazy_loads': 0   # Track when Word2Vec is actually loaded
        }

    def _lazy_initialize_parser(self):
        """
        Lazy initialization of parser with persistent Word2Vec model singleton caching.
        
        This method implements a production-grade solution to the 305% variance issue
        by using singleton pattern for Word2Vec model persistence across all instances.
        """
        if self._parser_initialized or not self.sanskrit_parser_available:
            return
            
        try:
            # CRITICAL PERFORMANCE FIX: Use singleton pattern for Word2Vec model
            # This eliminates the primary variance source (Word2Vec reloading)
            if not hasattr(SandhiPreprocessor, '_shared_parser_instance'):
                self.logger.info("Creating shared parser instance with Word2Vec model (one-time initialization)")
                
                # Create parser instance once and share across all instances
                SandhiPreprocessor._shared_parser_instance = self._parser_classes['Parser']()
                
                if self._parser_classes['LexicalSandhiAnalyzer']:
                    SandhiPreprocessor._shared_sandhi_analyzer = self._parser_classes['LexicalSandhiAnalyzer']()
                
                self.logger.info("Shared parser instance created - Word2Vec model loaded once")
            else:
                self.logger.debug("Using existing shared parser instance (no Word2Vec reload)")
            
            # Use the shared instance
            self.parser = SandhiPreprocessor._shared_parser_instance
            if hasattr(SandhiPreprocessor, '_shared_sandhi_analyzer'):
                self.sandhi_analyzer = SandhiPreprocessor._shared_sandhi_analyzer
            
            # PERFORMANCE OPTIMIZATION: Enhanced LRU cache for parsing results
            import functools
            
            # Check if shared cached method already exists
            if not hasattr(SandhiPreprocessor, '_shared_cached_parser_split'):
                @functools.lru_cache(maxsize=5000)  # Large cache for production
                def _shared_cached_parser_split(word: str):
                    """
                    Shared cached version of parser.split to maximize cache hits.
                    Uses class-level caching so all instances benefit from each other's work.
                    """
                    return SandhiPreprocessor._shared_parser_instance.split(word)
                
                SandhiPreprocessor._shared_cached_parser_split = _shared_cached_parser_split
                self.logger.info("Shared parsing cache initialized with 5000-item capacity")
            
            # Use the shared cached method
            self._cached_parser_split = SandhiPreprocessor._shared_cached_parser_split
            self._parser_initialized = True
            self.stats['lazy_loads'] += 1
            
            self.logger.info("Sanskrit parser lazy initialization completed with shared singleton caching")
            
        except Exception as e:
            self.logger.warning(f"Sanskrit parser lazy initialization failed: {e}")
            self.sanskrit_parser_available = False
            self._parser_initialized = False

    def preprocess_text(self, text: str) -> SandhiSplitResult:
        """
        Main preprocessing method to split Sanskrit compound words.
        
        Args:
            text: Input text containing potential Sanskrit compounds
            
        Returns:
            SandhiSplitResult with primary and alternative segmentations
        """
        import time
        start_time = time.time()
        
        self.stats['total_processed'] += 1
        
        # Handle None input
        if text is None:
            processing_time = (time.time() - start_time) * 1000
            return self._create_fallback_result("", processing_time)
        
        # If preprocessing is disabled, return fallback immediately
        if not self.enable_preprocessing:
            processing_time = (time.time() - start_time) * 1000
            return self._create_fallback_result(text, processing_time)
        
        # Clean and validate input
        cleaned_text = self._clean_input_text(text)
        if not self._is_likely_sanskrit(cleaned_text):
            processing_time = (time.time() - start_time) * 1000
            return self._create_fallback_result(text, processing_time)
        
        # Attempt sanskrit_parser splitting if available
        if self.sanskrit_parser_available:
            try:
                result = self._split_with_sanskrit_parser(cleaned_text)
                if result.preprocessing_successful:
                    self.stats['successful_splits'] += 1
                    processing_time = (time.time() - start_time) * 1000
                    result.processing_time_ms = processing_time
                    return result
            except Exception as e:
                self.logger.warning(f"sanskrit_parser failed for '{cleaned_text}': {e}")
                self.stats['processing_errors'] += 1
        
        # Fallback to basic tokenization
        self.stats['fallback_used'] += 1
        processing_time = (time.time() - start_time) * 1000
        return self._create_fallback_result(text, processing_time)

    def _clean_input_text(self, text: str) -> str:
        """Clean input text for sandhi analysis."""
        if text is None:
            return ""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Remove punctuation that might interfere with sandhi analysis
        cleaned = re.sub(r'[^\w\s\u0900-\u097F\u1CD0-\u1CFF]', '', cleaned)
        
        return cleaned

    def _is_likely_sanskrit(self, text: str) -> bool:
        """
        Determine if text is likely to contain Sanskrit requiring sandhi splitting.
        
        Args:
            text: Input text to analyze
            
        Returns:
            True if text appears to contain Sanskrit compounds
        """
        if not text or len(text) < 3:
            return False
        
        # Check for Devanagari Unicode ranges
        devanagari_pattern = r'[\u0900-\u097F\u1CD0-\u1CFF]+'
        if re.search(devanagari_pattern, text):
            return True
        
        # Check for transliterated Sanskrit patterns
        sanskrit_patterns = [
            r'\w*ā\w*',      # Long 'a' vowel
            r'\w*ṛ\w*',      # Vocalic 'r'
            r'\w*ṃ\w*',      # Anusvara
            r'\w*ḥ\w*',      # Visarga
            r'\w+[aeiou]\w+[aeiou]\w+',  # Multiple vowels suggesting compound
        ]
        
        for pattern in sanskrit_patterns:
            if re.search(pattern, text):
                return True
        
        # Check for long words that might be compounds
        words = text.split()
        for word in words:
            if len(word) > 12:  # Long words often indicate compounds
                return True
        
        return False

    def _split_with_sanskrit_parser(self, text: str) -> SandhiSplitResult:
        """
        Split text using sanskrit_parser library.
        
        Args:
            text: Cleaned Sanskrit text
            
        Returns:
            SandhiSplitResult with segmentation candidates
        """
        candidates = []
        
        try:
            # Try to get sandhi splits using sanskrit_parser
            words = text.split()
            
            for word in words:
                if len(word) > 6:  # Only process longer words likely to be compounds
                    word_candidates = self._get_word_sandhi_candidates(word)
                    candidates.extend(word_candidates)
            
            if candidates:
                # Sort candidates by confidence score
                candidates.sort(key=lambda x: x.confidence_score, reverse=True)
                
                primary = candidates[0]
                alternatives = candidates[1:5]  # Keep top 4 alternatives
                
                return SandhiSplitResult(
                    original_text=text,
                    primary_candidate=primary,
                    alternative_candidates=alternatives,
                    preprocessing_successful=True,
                    fallback_used=False
                )
        
        except Exception as e:
            self.logger.error(f"Error in sanskrit_parser splitting: {e}")
        
        # If no valid candidates found, return fallback
        return self._create_fallback_result(text, 0.0)

    def _get_word_sandhi_candidates(self, word: str) -> List[SandhiSplitCandidate]:
        """
        Get sandhi splitting candidates for a single word with lazy initialization.
        
        Args:
            word: Sanskrit word to analyze
            
        Returns:
            List of splitting candidates
        """
        candidates = []
        
        try:
            # Lazy initialize parser only when actually needed
            if self.sanskrit_parser_available and not self._parser_initialized:
                self._lazy_initialize_parser()
            
            # Try using actual sanskrit_parser functionality first
            if self.sanskrit_parser_available and self._parser_initialized and hasattr(self, 'parser'):
                try:
                    # PERFORMANCE OPTIMIZATION: Use cached parser to avoid repeated Word2Vec loading
                    if hasattr(self, '_cached_parser_split'):
                        # Check cache first
                        if word in self._parsing_cache:
                            parse_results = self._parsing_cache[word]
                            self.stats['cache_hits'] += 1
                        else:
                            parse_results = self._cached_parser_split(word)
                            self._parsing_cache[word] = parse_results
                            self.stats['cache_misses'] += 1
                    else:
                        # Fallback to direct parser call (no caching)
                        parse_results = self.parser.split(word)
                    
                    if parse_results:
                        for i, result in enumerate(parse_results[:5]):  # Top 5 results
                            # Extract segments from parse result
                            segments = self._extract_segments_from_parse_result(result)
                            if segments and len(segments) > 1:
                                confidence = 0.9 - (i * 0.1)  # Decrease confidence for later results
                                conf_level = SegmentationConfidenceLevel.HIGH if confidence >= 0.8 else SegmentationConfidenceLevel.MEDIUM
                                
                                candidate = SandhiSplitCandidate(
                                    original_text=word,
                                    segments=segments,
                                    confidence_score=confidence,
                                    confidence_level=conf_level,
                                    splitting_method='sanskrit_parser_split_cached',
                                    metadata={
                                        'word_length': len(word), 
                                        'segment_count': len(segments),
                                        'parse_rank': i,
                                        'cached': word in self._parsing_cache,
                                        'lazy_loaded': True
                                    }
                                )
                                candidates.append(candidate)
                except Exception as e:
                    self.logger.debug(f"sanskrit_parser split failed for '{word}': {e}")
            
            # If no candidates from sanskrit_parser, use heuristic methods
            if not candidates:
                splits = self._heuristic_sandhi_split(word)
                
                for split_result in splits:
                    segments, confidence, method = split_result
                    
                    # Determine confidence level
                    if confidence >= 0.8:
                        conf_level = SegmentationConfidenceLevel.HIGH
                    elif confidence >= 0.6:
                        conf_level = SegmentationConfidenceLevel.MEDIUM
                    else:
                        conf_level = SegmentationConfidenceLevel.LOW
                    
                    candidate = SandhiSplitCandidate(
                        original_text=word,
                        segments=segments,
                        confidence_score=confidence,
                        confidence_level=conf_level,
                        splitting_method=method,
                        metadata={'word_length': len(word), 'segment_count': len(segments)}
                    )
                    
                    candidates.append(candidate)
        
        except Exception as e:
            self.logger.warning(f"Error getting sandhi candidates for '{word}': {e}")
        
        return candidates

    def _extract_segments_from_parse_result(self, parse_result) -> List[str]:
        """
        Extract word segments from sanskrit_parser parse result.
        
        Args:
            parse_result: Result from sanskrit_parser.split()
            
        Returns:
            List of word segments
        """
        segments = []
        
        try:
            # The parse result structure depends on sanskrit_parser version
            # Handle different possible structures
            
            if hasattr(parse_result, 'splits'):
                # If it has splits attribute
                for split in parse_result.splits:
                    if hasattr(split, 'text'):
                        segments.append(str(split.text))
                    elif hasattr(split, '__str__'):
                        segments.append(str(split))
            elif hasattr(parse_result, '__iter__'):
                # If it's iterable
                for item in parse_result:
                    if hasattr(item, 'text'):
                        segments.append(str(item.text))
                    elif hasattr(item, '__str__'):
                        segments.append(str(item))
            elif hasattr(parse_result, '__str__'):
                # If it's a simple string representation
                segments = str(parse_result).split()
        
        except Exception as e:
            self.logger.debug(f"Error extracting segments from parse result: {e}")
        
        return segments

    def _heuristic_sandhi_split(self, word: str) -> List[Tuple[List[str], float, str]]:
        """
        Heuristic-based sandhi splitting as fallback/complement to sanskrit_parser.
        
        Args:
            word: Word to split
            
        Returns:
            List of (segments, confidence, method) tuples
        """
        splits = []
        
        # Example: yogaścittavṛttinirodhaḥ → ["yogaḥ", "citta", "vṛtti", "nirodhaḥ"]
        
        # Pattern 1: Split on common sandhi junctions
        common_junctions = ['ś', 'ṃ', 'ḥ']
        for junction in common_junctions:
            if junction in word:
                parts = word.split(junction)
                if len(parts) > 1:
                    # Reconstruct with proper endings
                    segments = []
                    for i, part in enumerate(parts[:-1]):
                        if part:
                            segments.append(part + junction)
                    if parts[-1]:
                        segments.append(parts[-1])
                    
                    if len(segments) > 1:
                        confidence = 0.7 if junction in ['ś', 'ḥ'] else 0.5
                        splits.append((segments, confidence, f'junction_split_{junction}'))
        
        # Pattern 2: Split on vowel patterns that suggest compound boundaries
        vowel_pattern = r'([aāiīuūṛṝeēoō])([kgcjṭḍtdpbmnlyrvś])'
        matches = list(re.finditer(vowel_pattern, word))
        
        if len(matches) > 1:
            # Try splitting at vowel-consonant boundaries
            split_points = [match.end(1) for match in matches[:-1]]
            segments = []
            start = 0
            
            for split_point in split_points:
                segments.append(word[start:split_point])
                start = split_point
            segments.append(word[start:])
            
            if len(segments) > 1 and all(len(seg) > 1 for seg in segments):
                confidence = 0.6
                splits.append((segments, confidence, 'vowel_consonant_split'))
        
        # Pattern 3: Basic length-based splitting for very long words
        if len(word) > 15 and not splits:
            mid_point = len(word) // 2
            # Find a good split point near the middle
            for offset in range(-3, 4):
                split_pos = mid_point + offset
                if 0 < split_pos < len(word):
                    segments = [word[:split_pos], word[split_pos:]]
                    confidence = 0.3
                    splits.append((segments, confidence, 'length_based_split'))
                    break
        
        return splits[:3]  # Return top 3 candidates

    def _create_fallback_result(self, text: str, processing_time: float) -> SandhiSplitResult:
        """
        Create fallback result using basic tokenization.
        
        Args:
            text: Original text
            processing_time: Processing time in milliseconds
            
        Returns:
            SandhiSplitResult using basic word tokenization
        """
        # Handle None or empty text
        if text is None:
            text = ""
        
        # Basic tokenization - split on whitespace
        segments = text.split() if text else []
        
        fallback_candidate = SandhiSplitCandidate(
            original_text=text,
            segments=segments,
            confidence_score=1.0,  # High confidence for basic tokenization
            confidence_level=SegmentationConfidenceLevel.FALLBACK,
            splitting_method='basic_tokenization',
            metadata={'fallback_reason': 'sanskrit_parser_unavailable_or_failed'}
        )
        
        return SandhiSplitResult(
            original_text=text,
            primary_candidate=fallback_candidate,
            alternative_candidates=[],
            preprocessing_successful=True,  # Fallback is considered successful
            fallback_used=True,
            processing_time_ms=processing_time
        )

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get statistics about sandhi preprocessing operations."""
        stats = self.stats.copy()
        stats.update({
            'sanskrit_parser_available': self.sanskrit_parser_available,
            'preprocessing_enabled': self.enable_preprocessing,
            'success_rate': (stats['successful_splits'] / max(stats['total_processed'], 1)) * 100
        })
        return stats

    
    def get_cache_performance_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics for monitoring performance improvements.
        
        Returns:
            Dictionary with cache performance metrics
        """
        cache_size = len(self._parsing_cache) if hasattr(self, '_parsing_cache') else 0
        total_requests = self.stats.get('cache_hits', 0) + self.stats.get('cache_misses', 0)
        hit_rate = (self.stats.get('cache_hits', 0) / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_enabled': hasattr(self, '_cached_parser_split'),
            'cache_size': cache_size,
            'cache_hits': self.stats.get('cache_hits', 0),
            'cache_misses': self.stats.get('cache_misses', 0),
            'cache_hit_rate_percent': round(hit_rate, 2),
            'total_requests': total_requests,
            'performance_improvement': f"{hit_rate:.1f}% of requests served from cache"
        }

    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self.stats = {
            'total_processed': 0,
            'successful_splits': 0,
            'fallback_used': 0,
            'processing_errors': 0
        }

    def validate_configuration(self) -> Dict[str, Any]:
        """Validate preprocessor configuration and dependencies."""
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        if not self.sanskrit_parser_available:
            validation_result['warnings'].append(
                'sanskrit_parser library not available - using fallback only'
            )
            validation_result['recommendations'].append(
                'Install sanskrit_parser for enhanced sandhi splitting: pip install sanskrit_parser'
            )
        
        if not self.enable_preprocessing:
            validation_result['warnings'].append(
                'Sandhi preprocessing is disabled via configuration'
            )
        
        return validation_result

    @classmethod
    def clear_shared_cache(cls):
        """
        Clear the shared parser cache and instances.
        Useful for testing or memory management.
        """
        if hasattr(cls, '_shared_cached_parser_split'):
            cls._shared_cached_parser_split.cache_clear()
        if hasattr(cls, '_shared_parser_instance'):
            delattr(cls, '_shared_parser_instance')
        if hasattr(cls, '_shared_sandhi_analyzer'):
            delattr(cls, '_shared_sandhi_analyzer')
        if hasattr(cls, '_shared_cached_parser_split'):
            delattr(cls, '_shared_cached_parser_split')
    
    @classmethod
    def get_shared_cache_stats(cls) -> Dict[str, Any]:
        """
        Get statistics for the shared parser cache across all instances.
        
        Returns:
            Dictionary with shared cache performance metrics
        """
        if hasattr(cls, '_shared_cached_parser_split'):
            cache_info = cls._shared_cached_parser_split.cache_info()
            total_calls = cache_info.hits + cache_info.misses
            hit_rate = (cache_info.hits / total_calls * 100) if total_calls > 0 else 0
            
            return {
                'shared_cache_enabled': True,
                'cache_hits': cache_info.hits,
                'cache_misses': cache_info.misses,
                'cache_maxsize': cache_info.maxsize,
                'cache_currsize': cache_info.currsize,
                'hit_rate_percent': round(hit_rate, 2),
                'total_calls': total_calls,
                'memory_efficiency': f"Shared across all SandhiPreprocessor instances",
                'variance_reduction': f"Eliminates Word2Vec reloading variance"
            }
        else:
            return {
                'shared_cache_enabled': False,
                'reason': 'Sanskrit parser not initialized or unavailable'
            }
