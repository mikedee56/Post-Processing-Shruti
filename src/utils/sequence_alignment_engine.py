"""
Sequence Alignment Engine for Story 2.4.3 - Stage 2 of Hybrid Matching Pipeline

This module implements Smith-Waterman local sequence alignment for precision matching
of noisy ASR text to canonical scripture verses. Optimized for Sanskrit text with 
IAST character handling and common ASR error patterns.

Key Features:
- Smith-Waterman local sequence alignment algorithm
- Sanskrit/IAST character-aware scoring
- ASR error pattern handling
- Optimized alignment for scripture matching
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass, field
import re
import logging
import time

from utils.logger_config import get_logger


class AlignmentScore(NamedTuple):
    """Alignment scoring parameters."""
    match: float = 2.0
    mismatch: float = -1.0
    gap_open: float = -2.0
    gap_extend: float = -0.5


@dataclass
class AlignmentResult:
    """Result of sequence alignment between two texts."""
    query_text: str
    target_text: str
    alignment_score: float
    normalized_score: float
    identity_percentage: float
    similarity_percentage: float
    alignment_length: int
    gaps: int
    query_start: int
    query_end: int
    target_start: int
    target_end: int
    aligned_query: str = ""
    aligned_target: str = ""
    alignment_string: str = ""
    computation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SanskritAlignmentConfig:
    """Configuration for Sanskrit-specific alignment parameters."""
    # Scoring parameters
    match_score: float = 2.0
    mismatch_score: float = -1.0
    gap_open_penalty: float = -2.0
    gap_extend_penalty: float = -0.5
    
    # Sanskrit-specific parameters
    iast_weight: float = 1.2  # Boost for IAST character matches
    vowel_mismatch_penalty: float = -0.5  # Lighter penalty for vowel mismatches
    consonant_cluster_bonus: float = 0.5  # Bonus for matching consonant clusters
    
    # ASR error tolerance
    aspiration_tolerance: bool = True  # kh/k, ph/p etc treated as similar
    vowel_length_tolerance: bool = True  # ā/a, ī/i etc treated as similar
    retroflex_dental_tolerance: bool = True  # ṭ/t, ḍ/d etc treated as similar
    
    # Performance optimization
    max_text_length: int = 1000
    enable_banded_alignment: bool = True
    band_width: int = 50


class SequenceAlignmentEngine:
    """
    Smith-Waterman sequence alignment engine optimized for Sanskrit scripture matching.
    
    This class implements Stage 2 of the hybrid matching pipeline, providing:
    1. Local sequence alignment for finding best matching regions
    2. Sanskrit-specific scoring that accounts for IAST transliteration
    3. ASR error pattern tolerance for improved matching
    4. Normalized scoring compatible with other pipeline stages
    """
    
    def __init__(self, config: Optional[SanskritAlignmentConfig] = None):
        """
        Initialize the sequence alignment engine.
        
        Args:
            config: Configuration for alignment parameters
        """
        self.logger = get_logger(__name__)
        self.config = config or SanskritAlignmentConfig()
        
        # Initialize character similarity maps
        self._initialize_character_similarity()
        
        # Performance tracking
        self.stats = {
            'alignments_performed': 0,
            'total_computation_time': 0.0,
            'average_computation_time': 0.0,
            'cache_hits': 0,
            'sequence_lengths': [],
        }
        
        # Simple cache for repeated alignments
        self._alignment_cache: Dict[str, AlignmentResult] = {}
        self._cache_max_size = 1000
        
        self.logger.info("Sequence alignment engine initialized with Sanskrit-specific scoring")
    
    def _initialize_character_similarity(self) -> None:
        """Initialize character similarity maps for Sanskrit/IAST."""
        
        # Character groups with high similarity
        self.similar_chars = {
            # Vowel length variations
            'vowel_short_long': {
                ('a', 'ā'), ('i', 'ī'), ('u', 'ū'),
                ('e', 'ē'), ('o', 'ō')
            },
            
            # Aspiration variations  
            'aspiration': {
                ('k', 'kh'), ('g', 'gh'), ('c', 'ch'), ('j', 'jh'),
                ('t', 'th'), ('d', 'dh'), ('p', 'ph'), ('b', 'bh'),
                ('ट', 'ठ'), ('ड', 'ढ')
            },
            
            # Retroflex/dental confusion
            'retroflex_dental': {
                ('ṭ', 't'), ('ḍ', 'd'), ('ṇ', 'n'),
                ('ṣ', 's'), ('ṛ', 'r')
            },
            
            # Sibilant variations
            'sibilants': {
                ('ś', 's'), ('ṣ', 's'), ('श', 'स')
            },
            
            # Nasalization variations
            'nasals': {
                ('ṃ', 'n'), ('ṃ', 'm'), ('ṅ', 'n'), ('ñ', 'n')
            }
        }
        
        # Build comprehensive similarity map
        self.char_similarity_map = {}
        
        for group_name, char_pairs in self.similar_chars.items():
            for char1, char2 in char_pairs:
                # Bidirectional similarity
                self.char_similarity_map[(char1, char2)] = 0.8  # High similarity
                self.char_similarity_map[(char2, char1)] = 0.8
        
        # IAST character priorities (for weighting)
        self.iast_chars = set('āīūṛṝḷḹēōṃḥṅñṭḍṇṣśĀĪŪṚṜḶḸĒŌṂḤṄÑṬḌṆṢŚ')
    
    def _get_char_similarity(self, char1: str, char2: str) -> float:
        """
        Get similarity score between two characters.
        
        Args:
            char1: First character
            char2: Second character
            
        Returns:
            Similarity score (0.0-1.0)
        """
        if char1 == char2:
            return 1.0
        
        # Check similarity map
        similarity = self.char_similarity_map.get((char1, char2), 0.0)
        
        # Apply IAST weighting
        if char1 in self.iast_chars or char2 in self.iast_chars:
            similarity *= self.config.iast_weight
        
        return min(similarity, 1.0)
    
    def _calculate_score(self, char1: str, char2: str, gap_type: str = None) -> float:
        """
        Calculate alignment score for character pair or gap.
        
        Args:
            char1: First character (or None for gap)
            char2: Second character (or None for gap)
            gap_type: Type of gap ('open' or 'extend')
            
        Returns:
            Alignment score
        """
        # Handle gaps
        if char1 is None or char2 is None:
            if gap_type == 'open':
                return self.config.gap_open_penalty
            else:
                return self.config.gap_extend_penalty
        
        # Character matching
        if char1 == char2:
            # Exact match
            score = self.config.match_score
            
            # Bonus for IAST characters
            if char1 in self.iast_chars:
                score *= self.config.iast_weight
            
            return score
        
        # Character similarity
        similarity = self._get_char_similarity(char1, char2)
        
        if similarity > 0.5:
            # Similar characters get reduced mismatch penalty
            return similarity * self.config.match_score * 0.7
        else:
            # Dissimilar characters get full mismatch penalty
            return self.config.mismatch_score
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for alignment.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Remove extra whitespace
        processed = re.sub(r'\s+', ' ', text.strip())
        
        # Remove punctuation (but keep Sanskrit punctuation markers)
        processed = re.sub(r'[^\w\s।|āīūṛṝḷḹēōṃḥṅñṭḍṇṣśĀĪŪṚṜḶḸĒŌṂḤṄÑṬḌṆṢŚ]', '', processed)
        
        return processed.lower()
    
    def calculate_sequence_alignment(
        self, 
        query: str, 
        target: str,
        local: bool = True
    ) -> AlignmentResult:
        """
        Calculate sequence alignment between query and target texts.
        
        Args:
            query: Query text (usually the ASR transcript)
            target: Target text (canonical verse)
            local: Use local alignment (Smith-Waterman) vs global
            
        Returns:
            AlignmentResult with detailed alignment information
        """
        start_time = time.time()
        
        # Check cache
        cache_key = f"{hash(query)}_{hash(target)}"
        if cache_key in self._alignment_cache:
            self.stats['cache_hits'] += 1
            cached_result = self._alignment_cache[cache_key]
            cached_result.computation_time = time.time() - start_time
            return cached_result
        
        # Preprocess texts
        query_processed = self._preprocess_text(query)
        target_processed = self._preprocess_text(target)
        
        # Check text length limits
        if (len(query_processed) > self.config.max_text_length or 
            len(target_processed) > self.config.max_text_length):
            self.logger.warning(
                f"Text length exceeds limit ({self.config.max_text_length}). "
                f"Query: {len(query_processed)}, Target: {len(target_processed)}"
            )
            # Truncate texts
            query_processed = query_processed[:self.config.max_text_length]
            target_processed = target_processed[:self.config.max_text_length]
        
        # Perform alignment
        if local:
            alignment_result = self._smith_waterman_alignment(query_processed, target_processed)
        else:
            alignment_result = self._needleman_wunsch_alignment(query_processed, target_processed)
        
        # Set original texts
        alignment_result.query_text = query
        alignment_result.target_text = target
        
        # Calculate computation time
        computation_time = time.time() - start_time
        alignment_result.computation_time = computation_time
        
        # Update statistics
        self.stats['alignments_performed'] += 1
        self.stats['total_computation_time'] += computation_time
        self.stats['average_computation_time'] = (
            self.stats['total_computation_time'] / self.stats['alignments_performed']
        )
        self.stats['sequence_lengths'].append((len(query_processed), len(target_processed)))
        
        # Cache result (with size limit)
        if len(self._alignment_cache) < self._cache_max_size:
            self._alignment_cache[cache_key] = alignment_result
        
        return alignment_result
    
    def _smith_waterman_alignment(self, query: str, target: str) -> AlignmentResult:
        """
        Smith-Waterman local alignment algorithm.
        
        Args:
            query: Preprocessed query sequence
            target: Preprocessed target sequence
            
        Returns:
            AlignmentResult with local alignment
        """
        m, n = len(query), len(target)
        
        # Initialize scoring matrix
        score_matrix = np.zeros((m + 1, n + 1), dtype=float)
        
        # Track best score position
        max_score = 0.0
        max_pos = (0, 0)
        
        # Fill scoring matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # Match/mismatch score
                match_score = score_matrix[i-1, j-1] + self._calculate_score(query[i-1], target[j-1])
                
                # Gap scores
                delete_score = score_matrix[i-1, j] + self._calculate_score(query[i-1], None, 'open')
                insert_score = score_matrix[i, j-1] + self._calculate_score(None, target[j-1], 'open')
                
                # Smith-Waterman: take max including 0
                score_matrix[i, j] = max(0, match_score, delete_score, insert_score)
                
                # Track maximum score
                if score_matrix[i, j] > max_score:
                    max_score = score_matrix[i, j]
                    max_pos = (i, j)
        
        # Traceback from maximum score position
        return self._traceback_smith_waterman(
            score_matrix, query, target, max_pos, max_score
        )
    
    def _needleman_wunsch_alignment(self, query: str, target: str) -> AlignmentResult:
        """
        Needleman-Wunsch global alignment algorithm.
        
        Args:
            query: Preprocessed query sequence
            target: Preprocessed target sequence
            
        Returns:
            AlignmentResult with global alignment
        """
        m, n = len(query), len(target)
        
        # Initialize scoring matrix
        score_matrix = np.zeros((m + 1, n + 1), dtype=float)
        
        # Initialize first row and column with gap penalties
        for i in range(1, m + 1):
            score_matrix[i, 0] = score_matrix[i-1, 0] + self._calculate_score(query[i-1], None, 'open')
        
        for j in range(1, n + 1):
            score_matrix[0, j] = score_matrix[0, j-1] + self._calculate_score(None, target[j-1], 'open')
        
        # Fill scoring matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # Match/mismatch score
                match_score = score_matrix[i-1, j-1] + self._calculate_score(query[i-1], target[j-1])
                
                # Gap scores
                delete_score = score_matrix[i-1, j] + self._calculate_score(query[i-1], None, 'open')
                insert_score = score_matrix[i, j-1] + self._calculate_score(None, target[j-1], 'open')
                
                # Needleman-Wunsch: take maximum
                score_matrix[i, j] = max(match_score, delete_score, insert_score)
        
        # Final score and traceback
        final_score = score_matrix[m, n]
        return self._traceback_needleman_wunsch(
            score_matrix, query, target, (m, n), final_score
        )
    
    def _traceback_smith_waterman(
        self, 
        score_matrix: np.ndarray, 
        query: str, 
        target: str,
        max_pos: Tuple[int, int], 
        max_score: float
    ) -> AlignmentResult:
        """Traceback for Smith-Waterman alignment."""
        
        aligned_query = ""
        aligned_target = ""
        alignment_string = ""
        
        i, j = max_pos
        query_end, target_end = i - 1, j - 1
        
        # Traceback until score becomes 0
        while i > 0 and j > 0 and score_matrix[i, j] > 0:
            current_score = score_matrix[i, j]
            
            # Determine which move led to current score
            match_score = score_matrix[i-1, j-1] + self._calculate_score(query[i-1], target[j-1])
            delete_score = score_matrix[i-1, j] + self._calculate_score(query[i-1], None, 'open')
            insert_score = score_matrix[i, j-1] + self._calculate_score(None, target[j-1], 'open')
            
            if current_score == match_score:
                # Match/mismatch
                aligned_query = query[i-1] + aligned_query
                aligned_target = target[j-1] + aligned_target
                
                if query[i-1] == target[j-1]:
                    alignment_string = "|" + alignment_string
                elif self._get_char_similarity(query[i-1], target[j-1]) > 0.5:
                    alignment_string = ":" + alignment_string  # Similar
                else:
                    alignment_string = " " + alignment_string  # Mismatch
                
                i -= 1
                j -= 1
                
            elif current_score == delete_score:
                # Deletion (gap in target)
                aligned_query = query[i-1] + aligned_query
                aligned_target = "-" + aligned_target
                alignment_string = " " + alignment_string
                i -= 1
                
            else:
                # Insertion (gap in query)
                aligned_query = "-" + aligned_query
                aligned_target = target[j-1] + aligned_target
                alignment_string = " " + alignment_string
                j -= 1
        
        query_start, target_start = i, j
        
        return self._create_alignment_result(
            query, target, aligned_query, aligned_target, alignment_string,
            max_score, query_start, query_end, target_start, target_end
        )
    
    def _traceback_needleman_wunsch(
        self, 
        score_matrix: np.ndarray, 
        query: str, 
        target: str,
        end_pos: Tuple[int, int], 
        final_score: float
    ) -> AlignmentResult:
        """Traceback for Needleman-Wunsch alignment."""
        
        aligned_query = ""
        aligned_target = ""
        alignment_string = ""
        
        i, j = end_pos
        
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                current_score = score_matrix[i, j]
                match_score = score_matrix[i-1, j-1] + self._calculate_score(query[i-1], target[j-1])
                delete_score = score_matrix[i-1, j] + self._calculate_score(query[i-1], None, 'open')
                insert_score = score_matrix[i, j-1] + self._calculate_score(None, target[j-1], 'open')
                
                if current_score == match_score:
                    aligned_query = query[i-1] + aligned_query
                    aligned_target = target[j-1] + aligned_target
                    alignment_string = ("|" if query[i-1] == target[j-1] else " ") + alignment_string
                    i -= 1
                    j -= 1
                elif current_score == delete_score:
                    aligned_query = query[i-1] + aligned_query
                    aligned_target = "-" + aligned_target
                    alignment_string = " " + alignment_string
                    i -= 1
                else:
                    aligned_query = "-" + aligned_query
                    aligned_target = target[j-1] + aligned_target
                    alignment_string = " " + alignment_string
                    j -= 1
            elif i > 0:
                aligned_query = query[i-1] + aligned_query
                aligned_target = "-" + aligned_target
                alignment_string = " " + alignment_string
                i -= 1
            else:
                aligned_query = "-" + aligned_query
                aligned_target = target[j-1] + aligned_target
                alignment_string = " " + alignment_string
                j -= 1
        
        return self._create_alignment_result(
            query, target, aligned_query, aligned_target, alignment_string,
            final_score, 0, len(query) - 1, 0, len(target) - 1
        )
    
    def _create_alignment_result(
        self, 
        query: str, 
        target: str,
        aligned_query: str, 
        aligned_target: str, 
        alignment_string: str,
        raw_score: float, 
        query_start: int, 
        query_end: int,
        target_start: int, 
        target_end: int
    ) -> AlignmentResult:
        """Create comprehensive alignment result."""
        
        # Calculate metrics
        alignment_length = len(aligned_query)
        gaps = aligned_query.count("-") + aligned_target.count("-")
        
        # Calculate identity and similarity
        matches = alignment_string.count("|")
        similarities = alignment_string.count(":")
        
        identity_percentage = (matches / max(alignment_length, 1)) * 100
        similarity_percentage = ((matches + similarities) / max(alignment_length, 1)) * 100
        
        # Normalize score
        max_possible_score = min(len(query), len(target)) * self.config.match_score
        normalized_score = max(0.0, min(1.0, raw_score / max(max_possible_score, 1.0)))
        
        return AlignmentResult(
            query_text=query,
            target_text=target,
            alignment_score=raw_score,
            normalized_score=normalized_score,
            identity_percentage=identity_percentage,
            similarity_percentage=similarity_percentage,
            alignment_length=alignment_length,
            gaps=gaps,
            query_start=query_start,
            query_end=query_end,
            target_start=target_start,
            target_end=target_end,
            aligned_query=aligned_query,
            aligned_target=aligned_target,
            alignment_string=alignment_string,
            metadata={
                'scoring_config': {
                    'match_score': self.config.match_score,
                    'mismatch_score': self.config.mismatch_score,
                    'gap_open_penalty': self.config.gap_open_penalty,
                    'gap_extend_penalty': self.config.gap_extend_penalty
                },
                'text_lengths': {
                    'query_length': len(query),
                    'target_length': len(target),
                    'alignment_length': alignment_length
                }
            }
        )
    
    def batch_align(
        self, 
        query: str, 
        targets: List[str],
        progress_callback: Optional[callable] = None
    ) -> List[AlignmentResult]:
        """
        Perform batch alignment of query against multiple targets.
        
        Args:
            query: Query text
            targets: List of target texts
            progress_callback: Optional progress callback
            
        Returns:
            List of alignment results sorted by normalized score
        """
        results = []
        total_targets = len(targets)
        
        for i, target in enumerate(targets):
            result = self.calculate_sequence_alignment(query, target)
            results.append(result)
            
            # Progress callback
            if progress_callback and (i + 1) % max(1, total_targets // 10) == 0:
                progress = (i + 1) / total_targets
                progress_callback(progress, i + 1, total_targets)
        
        # Sort by normalized score (descending)
        results.sort(key=lambda x: x.normalized_score, reverse=True)
        
        return results
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_query_length = 0.0
        avg_target_length = 0.0
        
        if self.stats['sequence_lengths']:
            avg_query_length = np.mean([lengths[0] for lengths in self.stats['sequence_lengths']])
            avg_target_length = np.mean([lengths[1] for lengths in self.stats['sequence_lengths']])
        
        return {
            'alignment_performance': {
                'alignments_performed': self.stats['alignments_performed'],
                'average_computation_time': f"{self.stats['average_computation_time']:.4f}s",
                'total_computation_time': f"{self.stats['total_computation_time']:.2f}s",
                'cache_hits': self.stats['cache_hits'],
                'cache_hit_rate': f"{(self.stats['cache_hits'] / max(self.stats['alignments_performed'], 1)) * 100:.1f}%"
            },
            'sequence_statistics': {
                'average_query_length': f"{avg_query_length:.1f}",
                'average_target_length': f"{avg_target_length:.1f}",
                'processed_sequences': len(self.stats['sequence_lengths'])
            },
            'configuration': {
                'match_score': self.config.match_score,
                'mismatch_score': self.config.mismatch_score,
                'gap_penalties': {
                    'open': self.config.gap_open_penalty,
                    'extend': self.config.gap_extend_penalty
                },
                'sanskrit_features': {
                    'iast_weight': self.config.iast_weight,
                    'aspiration_tolerance': self.config.aspiration_tolerance,
                    'vowel_length_tolerance': self.config.vowel_length_tolerance
                }
            }
        }
    
    def clear_cache(self) -> None:
        """Clear alignment cache."""
        self._alignment_cache.clear()
        self.logger.info("Cleared alignment cache")
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate configuration."""
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check scoring parameters
        if self.config.match_score <= 0:
            validation['errors'].append("Match score must be positive")
            validation['is_valid'] = False
        
        if self.config.gap_open_penalty >= 0:
            validation['warnings'].append("Gap open penalty should typically be negative")
        
        # Check similarity maps
        if not self.char_similarity_map:
            validation['errors'].append("Character similarity map not initialized")
            validation['is_valid'] = False
        
        return validation