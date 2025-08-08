"""
Enhanced Fuzzy Matcher for Story 2.4.4

This module provides performance-enhanced fuzzy matching using phonetic hashing
for 10-50x faster candidate filtering before expensive fuzzy operations.

Key Features:
- Phonetic hash-based first-pass filtering
- Integration with existing fuzzy matching pipeline
- Backward compatibility with Story 2.1 FuzzyMatcher
- Performance monitoring and optimization
"""

from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass
import time
import logging

from utils.fuzzy_matcher import FuzzyMatcher, FuzzyMatch, MatchingConfig, MatchType
from utils.sanskrit_phonetic_hasher import SanskritPhoneticHasher
from enhancement_integration.unified_confidence_scorer import (
    UnifiedConfidenceScorer, 
    ConfidenceSource, 
    ConfidenceScore
)
from utils.logger_config import get_logger


@dataclass
class EnhancedMatchResult:
    """Enhanced match result with phonetic and performance metadata."""
    match: FuzzyMatch
    phonetic_hash: str
    phonetic_score: float
    performance_data: Dict[str, Any]
    enhanced_confidence: float


@dataclass
class PerformanceMetrics:
    """Performance metrics for enhanced matching."""
    total_queries: int = 0
    phonetic_filter_time: float = 0.0
    fuzzy_matching_time: float = 0.0
    total_time: float = 0.0
    candidates_before_filter: int = 0
    candidates_after_filter: int = 0
    performance_improvement: float = 0.0


class EnhancedFuzzyMatcher:
    """
    Performance-enhanced fuzzy matcher using phonetic hashing for fast filtering.
    
    This component implements AC2 of Story 2.4.4, providing:
    - Phonetic hash first-pass filtering for 10-50x performance improvement
    - Backward compatibility with existing FuzzyMatcher API
    - Enhanced confidence scoring using unified system
    - Performance monitoring and optimization
    """
    
    def __init__(
        self, 
        lexicon_data: Dict[str, Any], 
        config: MatchingConfig = None,
        enable_phonetic_acceleration: bool = True
    ):
        """
        Initialize enhanced fuzzy matcher.
        
        Args:
            lexicon_data: Dictionary of lexicon entries
            config: Matching configuration parameters
            enable_phonetic_acceleration: Enable phonetic hash acceleration
        """
        self.logger = get_logger(__name__)
        self.lexicon_data = lexicon_data
        self.config = config or MatchingConfig()
        self.enable_phonetic_acceleration = enable_phonetic_acceleration
        
        # Initialize base fuzzy matcher for fallback
        self.base_fuzzy_matcher = FuzzyMatcher(lexicon_data, config)
        
        # Initialize phonetic hasher if acceleration enabled
        if self.enable_phonetic_acceleration:
            self.phonetic_hasher = SanskritPhoneticHasher()
            self._build_phonetic_index()
        else:
            self.phonetic_hasher = None
            self.phonetic_index = {}
        
        # Initialize unified confidence scorer
        self.confidence_scorer = UnifiedConfidenceScorer()
        
        # Performance tracking
        self.performance_metrics = PerformanceMetrics()
        
        self.logger.info(
            f"Enhanced fuzzy matcher initialized (phonetic acceleration: {enable_phonetic_acceleration})"
        )
    
    def _build_phonetic_index(self) -> None:
        """Build phonetic hash index for fast filtering."""
        start_time = time.time()
        
        self.phonetic_index: Dict[str, List[str]] = {}  # hash -> [terms]
        self.term_to_hash: Dict[str, str] = {}  # term -> hash
        
        # Index all terms and their variations
        all_terms = set()
        
        for original_term, entry in self.lexicon_data.items():
            all_terms.add(original_term.lower())
            # Add variations
            for variation in entry.get('variations', []):
                all_terms.add(variation.lower())
        
        # Generate phonetic hashes for all terms
        for term in all_terms:
            phonetic_hash = self.phonetic_hasher.generate_phonetic_hash(term)
            
            # Store in index
            if phonetic_hash not in self.phonetic_index:
                self.phonetic_index[phonetic_hash] = []
            self.phonetic_index[phonetic_hash].append(term)
            
            # Store reverse mapping
            self.term_to_hash[term] = phonetic_hash
        
        build_time = time.time() - start_time
        
        self.logger.info(
            f"Built phonetic index: {len(all_terms)} terms, "
            f"{len(self.phonetic_index)} unique hashes in {build_time:.3f}s"
        )
    
    def find_matches(
        self, 
        word: str, 
        context: str = "", 
        max_matches: int = 5,
        use_enhancement: bool = True
    ) -> List[EnhancedMatchResult]:
        """
        Find enhanced fuzzy matches with phonetic acceleration.
        
        Args:
            word: Word to find matches for
            context: Surrounding context for better matching
            max_matches: Maximum number of matches to return
            use_enhancement: Whether to use phonetic enhancement
            
        Returns:
            List of enhanced match results sorted by confidence
        """
        start_time = time.time()
        self.performance_metrics.total_queries += 1
        
        word_lower = word.lower()
        enhanced_matches = []
        
        if use_enhancement and self.enable_phonetic_acceleration:
            # Enhanced path with phonetic filtering
            phonetic_candidates = self._get_phonetic_candidates(word_lower)
            
            # Apply fuzzy matching only to filtered candidates
            fuzzy_start = time.time()
            
            if phonetic_candidates:
                # Create filtered lexicon for fuzzy matching
                filtered_lexicon = {
                    term: self.lexicon_data.get(term) or self._find_lexicon_entry(term)
                    for term in phonetic_candidates
                    if self.lexicon_data.get(term) or self._find_lexicon_entry(term)
                }
                
                if filtered_lexicon:
                    # Create temporary fuzzy matcher with filtered data
                    temp_matcher = FuzzyMatcher(filtered_lexicon, self.config)
                    base_matches = temp_matcher.find_matches(word, context, max_matches * 2)
                else:
                    base_matches = []
            else:
                # No phonetic candidates found, use full fuzzy matching as fallback
                base_matches = self.base_fuzzy_matcher.find_matches(word, context, max_matches * 2)
            
            fuzzy_time = time.time() - fuzzy_start
            self.performance_metrics.fuzzy_matching_time += fuzzy_time
            
            # Convert to enhanced results
            for base_match in base_matches:
                phonetic_hash = self.term_to_hash.get(base_match.matched_term.lower(), "")
                phonetic_score = self._calculate_phonetic_score(word_lower, base_match.matched_term.lower())
                
                # Calculate enhanced confidence using unified scorer
                enhanced_confidence = self._calculate_enhanced_confidence(base_match, phonetic_score)
                
                enhanced_match = EnhancedMatchResult(
                    match=base_match,
                    phonetic_hash=phonetic_hash,
                    phonetic_score=phonetic_score,
                    performance_data={
                        "phonetic_filtering_used": True,
                        "candidates_filtered": len(phonetic_candidates),
                        "fuzzy_matching_time": fuzzy_time
                    },
                    enhanced_confidence=enhanced_confidence
                )
                enhanced_matches.append(enhanced_match)
        
        else:
            # Fallback to base fuzzy matcher
            base_matches = self.base_fuzzy_matcher.find_matches(word, context, max_matches)
            
            for base_match in base_matches:
                enhanced_match = EnhancedMatchResult(
                    match=base_match,
                    phonetic_hash="",
                    phonetic_score=0.0,
                    performance_data={
                        "phonetic_filtering_used": False,
                        "fallback_used": True
                    },
                    enhanced_confidence=base_match.confidence
                )
                enhanced_matches.append(enhanced_match)
        
        # Sort by enhanced confidence
        enhanced_matches.sort(key=lambda m: m.enhanced_confidence, reverse=True)
        
        # Update performance metrics
        total_time = time.time() - start_time
        self.performance_metrics.total_time += total_time
        
        return enhanced_matches[:max_matches]
    
    def _get_phonetic_candidates(self, word: str) -> List[str]:
        """Get candidate terms using phonetic hash filtering."""
        filter_start = time.time()
        
        candidates = []
        word_hash = self.phonetic_hasher.generate_phonetic_hash(word)
        
        # Find exact phonetic hash matches
        if word_hash in self.phonetic_index:
            candidates.extend(self.phonetic_index[word_hash])
        
        # Find near matches within distance threshold
        max_distance = 2  # Allow up to 2 character differences in hash
        
        for hash_key, terms in self.phonetic_index.items():
            if hash_key != word_hash:
                distance = self.phonetic_hasher.calculate_hash_distance(word_hash, hash_key)
                if distance <= max_distance:
                    candidates.extend(terms)
        
        # Remove duplicates
        candidates = list(set(candidates))
        
        filter_time = time.time() - filter_start
        self.performance_metrics.phonetic_filter_time += filter_time
        self.performance_metrics.candidates_after_filter += len(candidates)
        self.performance_metrics.candidates_before_filter += len(self.base_fuzzy_matcher.term_list)
        
        self.logger.debug(
            f"Phonetic filtering: {len(self.base_fuzzy_matcher.term_list)} -> {len(candidates)} "
            f"candidates in {filter_time:.4f}s"
        )
        
        return candidates
    
    def _find_lexicon_entry(self, term: str) -> Optional[Dict[str, Any]]:
        """Find lexicon entry for a term (handles variations)."""
        # Check direct match
        if term in self.lexicon_data:
            return self.lexicon_data[term]
        
        # Check variations
        for original_term, entry in self.lexicon_data.items():
            if term in [v.lower() for v in entry.get('variations', [])]:
                return entry
        
        return None
    
    def _calculate_phonetic_score(self, word1: str, word2: str) -> float:
        """Calculate phonetic similarity score between two words."""
        if not self.phonetic_hasher:
            return 0.0
        
        hash1 = self.phonetic_hasher.generate_phonetic_hash(word1)
        hash2 = self.phonetic_hasher.generate_phonetic_hash(word2)
        
        if hash1 == hash2:
            return 1.0
        
        distance = self.phonetic_hasher.calculate_hash_distance(hash1, hash2)
        max_length = max(len(hash1), len(hash2))
        
        return max(0.0, 1.0 - (distance / max_length)) if max_length > 0 else 0.0
    
    def _calculate_enhanced_confidence(self, base_match: FuzzyMatch, phonetic_score: float) -> float:
        """Calculate enhanced confidence using unified confidence scorer."""
        # Create confidence scores from different sources
        confidence_scores = []
        
        # Base fuzzy matching confidence
        fuzzy_confidence = ConfidenceScore(
            value=base_match.confidence,
            source=ConfidenceSource.FUZZY_MATCHING,
            weight=1.0,
            metadata={"match_type": base_match.match_type.value}
        )
        confidence_scores.append(fuzzy_confidence)
        
        # Phonetic similarity confidence
        if phonetic_score > 0:
            phonetic_confidence = ConfidenceScore(
                value=phonetic_score,
                source=ConfidenceSource.PHONETIC_HASHING,
                weight=0.7,
                metadata={"phonetic_score": phonetic_score}
            )
            confidence_scores.append(phonetic_confidence)
        
        # Lexicon match confidence (if exact match)
        if base_match.match_type == MatchType.EXACT:
            lexicon_confidence = ConfidenceScore(
                value=1.0,
                source=ConfidenceSource.LEXICON_MATCH,
                weight=1.0,
                metadata={"exact_match": True}
            )
            confidence_scores.append(lexicon_confidence)
        
        # Combine confidences using unified scorer
        result = self.confidence_scorer.combine_confidence_scores(
            confidence_scores, 
            method="weighted_average"
        )
        
        return result.final_confidence
    
    def find_matches_legacy(
        self, 
        word: str, 
        context: str = "", 
        max_matches: int = 5
    ) -> List[FuzzyMatch]:
        """
        Legacy interface for backward compatibility with Story 2.1.
        
        Args:
            word: Word to find matches for
            context: Surrounding context
            max_matches: Maximum matches to return
            
        Returns:
            List of FuzzyMatch objects (original format)
        """
        enhanced_matches = self.find_matches(word, context, max_matches, use_enhancement=True)
        
        # Convert back to legacy format
        legacy_matches = []
        for enhanced_match in enhanced_matches:
            # Update confidence with enhanced value
            match = enhanced_match.match
            match.confidence = enhanced_match.enhanced_confidence
            legacy_matches.append(match)
        
        return legacy_matches
    
    def batch_match(
        self, 
        words: List[str], 
        context: str = ""
    ) -> Dict[str, List[EnhancedMatchResult]]:
        """
        Find enhanced matches for multiple words efficiently.
        
        Args:
            words: List of words to match
            context: Shared context for all words
            
        Returns:
            Dictionary mapping words to their enhanced matches
        """
        results = {}
        
        for word in words:
            matches = self.find_matches(word, context)
            if matches:
                results[word] = matches
        
        return results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance improvement metrics."""
        total_candidates_before = self.performance_metrics.candidates_before_filter
        total_candidates_after = self.performance_metrics.candidates_after_filter
        
        if total_candidates_before > 0:
            filter_ratio = total_candidates_after / total_candidates_before
            performance_improvement = 1.0 / filter_ratio if filter_ratio > 0 else 1.0
        else:
            performance_improvement = 1.0
        
        avg_phonetic_time = (
            self.performance_metrics.phonetic_filter_time / 
            max(self.performance_metrics.total_queries, 1)
        )
        avg_fuzzy_time = (
            self.performance_metrics.fuzzy_matching_time / 
            max(self.performance_metrics.total_queries, 1)
        )
        avg_total_time = (
            self.performance_metrics.total_time / 
            max(self.performance_metrics.total_queries, 1)
        )
        
        return {
            "performance_improvement": f"{performance_improvement:.1f}x",
            "total_queries": self.performance_metrics.total_queries,
            "average_phonetic_filter_time": f"{avg_phonetic_time:.4f}s",
            "average_fuzzy_matching_time": f"{avg_fuzzy_time:.4f}s", 
            "average_total_time": f"{avg_total_time:.4f}s",
            "candidates_reduction_ratio": f"{(1 - filter_ratio) * 100:.1f}%" if total_candidates_before > 0 else "0%",
            "phonetic_acceleration_enabled": self.enable_phonetic_acceleration,
            "phonetic_index_size": len(self.phonetic_index) if hasattr(self, 'phonetic_index') else 0
        }
    
    def validate_enhancement(self) -> Dict[str, Any]:
        """Validate enhancement configuration and performance."""
        validation = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "performance_status": "optimal"
        }
        
        # Check phonetic acceleration
        if self.enable_phonetic_acceleration:
            if not self.phonetic_hasher:
                validation["errors"].append("Phonetic hasher not initialized")
                validation["is_valid"] = False
            elif not hasattr(self, 'phonetic_index') or not self.phonetic_index:
                validation["errors"].append("Phonetic index not built")
                validation["is_valid"] = False
        
        # Check performance improvement
        if self.performance_metrics.total_queries > 10:
            total_candidates_before = self.performance_metrics.candidates_before_filter
            total_candidates_after = self.performance_metrics.candidates_after_filter
            
            if total_candidates_before > 0:
                improvement = total_candidates_before / max(total_candidates_after, 1)
                if improvement < 2.0:
                    validation["warnings"].append(
                        f"Performance improvement below expected (2x): {improvement:.1f}x"
                    )
                    validation["performance_status"] = "suboptimal"
                elif improvement >= 10.0:
                    validation["performance_status"] = "excellent"
        
        # Check base matcher
        if not self.base_fuzzy_matcher:
            validation["errors"].append("Base fuzzy matcher not initialized")
            validation["is_valid"] = False
        
        return validation