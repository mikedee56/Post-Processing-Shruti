"""
Scripture Identifier Module.

This module identifies longer Sanskrit/Hindi passages that correspond to known
scriptural verses in the lexicon, providing confidence scoring and passage boundaries.
"""

import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

from utils.logger_config import get_logger
from utils.fuzzy_matcher import FuzzyMatcher, MatchType
from sanskrit_hindi_identifier.lexicon_manager import LexiconManager
from sanskrit_hindi_identifier.word_identifier import LexiconEntry


class PassageType(Enum):
    """Types of scriptural passages."""
    VERSE = "verse"
    CHAPTER_REFERENCE = "chapter_reference"
    MULTI_VERSE = "multi_verse"
    PARTIAL_VERSE = "partial_verse"


@dataclass
class PassageSegment:
    """Represents a segment of text that might be a scriptural passage."""
    text: str
    start_pos: int
    end_pos: int
    word_count: int
    sanskrit_word_density: float
    line_breaks: int


@dataclass
class VerseMatch:
    """Represents a potential match between text and a scriptural verse."""
    original_text: str
    canonical_entry: LexiconEntry
    passage_type: PassageType
    confidence_score: float
    similarity_score: float
    match_start: int
    match_end: int
    word_matches: int


@dataclass
class ScriptureMatch:
    """Represents a match between text and a scriptural passage."""
    original_text: str
    matched_text: str
    scripture_reference: str
    confidence_score: float
    match_type: str
    start_position: int
    end_position: int
    metadata: Optional[Dict] = None
    

class ScriptureIdentifier:
    """
    Core module for identifying scriptural verses and passages in text.
    
    Uses fuzzy matching, similarity scoring, and contextual analysis to identify
    longer passages that correspond to known scriptural verses.
    """
    
    def __init__(self, lexicon_manager: LexiconManager = None, config: Dict = None):
        """
        Initialize the Scripture Identifier.
        
        Args:
            lexicon_manager: Manager for lexicon data
            config: Configuration parameters
        """
        self.logger = get_logger(__name__)
        self.lexicon_manager = lexicon_manager or LexiconManager()
        
        # Initialize fuzzy matcher with lexicon data
        lexicon_entries = self.lexicon_manager.get_all_entries()
        # Convert LexiconEntry objects to dictionary format expected by FuzzyMatcher
        lexicon_dict = {}
        for term, entry in lexicon_entries.items():
            lexicon_dict[term] = {
                'original_term': entry.original_term,
                'variations': getattr(entry, 'variations', []),
                'transliteration': getattr(entry, 'transliteration', ''),
                'is_proper_noun': getattr(entry, 'is_proper_noun', False),
                'category': getattr(entry, 'category', 'unknown'),
                'confidence': getattr(entry, 'confidence', 1.0)
            }
        self.fuzzy_matcher = FuzzyMatcher(lexicon_dict)
        
        # Configuration
        self.config = config or {}
        self.min_verse_length = self.config.get('min_verse_length', 10)
        self.min_confidence_threshold = self.config.get('min_confidence_threshold', 0.75)
        self.min_similarity_threshold = self.config.get('min_similarity_threshold', 0.6)
        self.min_sanskrit_density = self.config.get('min_sanskrit_density', 0.3)
        self.verse_boundary_patterns = self.config.get('verse_boundary_patterns', [
            r'\|\|',  # Sanskrit verse endings
            r'।।',   # Devanagari verse endings
            r'\n\s*\n',  # Double line breaks
        ])
        
        # Load scriptural entries
        self._load_scriptural_entries()
        
        self.logger.info(f"Scripture identifier initialized with {len(self.verse_entries)} verses")
    
    def _load_scriptural_entries(self) -> None:
        """Load scriptural entries from lexicon."""
        all_entries = self.lexicon_manager.get_all_entries()
        
        # Filter for verse entries
        self.verse_entries = {
            term: entry for term, entry in all_entries.items()
            if hasattr(entry, 'is_verse') and getattr(entry, 'is_verse', False)
        }
        
        # Build canonical text index
        self.canonical_texts = {}
        for entry in self.verse_entries.values():
            canonical = getattr(entry, 'canonical_text', '')
            if canonical:
                # Index by both the canonical text and normalized versions
                self.canonical_texts[canonical.lower()] = entry
                # Also index by first few words for partial matching
                words = canonical.lower().split()[:5]
                if len(words) >= 3:
                    partial_key = ' '.join(words)
                    self.canonical_texts[partial_key] = entry
    
    def identify_scripture_passages(self, text: str) -> List[VerseMatch]:
        """
        Identify potential scriptural passages in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of potential verse matches with confidence scores
        """
        matches = []
        
        # First, segment text into potential passages
        segments = self._segment_text_for_verses(text)
        
        for segment in segments:
            # Check if segment might contain a verse
            if self._is_potential_verse_segment(segment):
                verse_matches = self._match_segment_to_verses(segment, text)
                matches.extend(verse_matches)
        
        # Remove overlapping matches (keep highest confidence)
        matches = self._resolve_overlapping_matches(matches)
        
        # Sort by confidence
        matches.sort(key=lambda m: m.confidence_score, reverse=True)
        
        return matches
    
    def _segment_text_for_verses(self, text: str) -> List[PassageSegment]:
        """
        Segment text into potential verse passages.
        
        Args:
            text: Input text
            
        Returns:
            List of text segments that might contain verses
        """
        segments = []
        
        # Split by verse boundary patterns
        split_pattern = '|'.join(f'({pattern})' for pattern in self.verse_boundary_patterns)
        parts = re.split(split_pattern, text, flags=re.MULTILINE)
        
        current_pos = 0
        for part in parts:
            if not part or not part.strip():
                current_pos += len(part) if part else 0
                continue
            
            start_pos = text.find(part, current_pos)
            end_pos = start_pos + len(part)
            
            # Calculate metrics for this segment
            words = part.split()
            word_count = len(words)
            sanskrit_density = self._calculate_sanskrit_density(part)
            line_breaks = part.count('\n')
            
            segment = PassageSegment(
                text=part.strip(),
                start_pos=start_pos,
                end_pos=end_pos,
                word_count=word_count,
                sanskrit_word_density=sanskrit_density,
                line_breaks=line_breaks
            )
            segments.append(segment)
            current_pos = end_pos
        
        return segments
    
    def _is_potential_verse_segment(self, segment: PassageSegment) -> bool:
        """
        Check if a segment might contain a scriptural verse.
        
        Args:
            segment: Text segment to evaluate
            
        Returns:
            True if segment has verse-like characteristics
        """
        # Check minimum length
        if len(segment.text) < self.min_verse_length:
            return False
        
        # Check Sanskrit word density
        if segment.sanskrit_word_density < self.min_sanskrit_density:
            return False
        
        # Check for verse-like patterns
        verse_indicators = [
            r'[|।]',  # Verse markers
            r'\b(gita|upanishad|sutra)\b',  # Scripture references
            r'\b(chapter|verse)\b',  # Structural references
            r'[aeiou]m\b',  # Sanskrit word endings
        ]
        
        pattern_matches = sum(1 for pattern in verse_indicators 
                            if re.search(pattern, segment.text, re.IGNORECASE))
        
        return pattern_matches >= 2
    
    def _match_segment_to_verses(self, segment: PassageSegment, full_text: str) -> List[VerseMatch]:
        """
        Match a text segment against known verses.
        
        Args:
            segment: Text segment to match
            full_text: Full text for context
            
        Returns:
            List of verse matches for this segment
        """
        matches = []
        segment_text = segment.text.lower().strip()
        
        # Try exact and fuzzy matching against verse entries
        for verse_term, verse_entry in self.verse_entries.items():
            canonical_text = getattr(verse_entry, 'canonical_text', '')
            if not canonical_text:
                continue
            
            # Calculate similarity scores
            similarity_scores = self._calculate_verse_similarity(segment_text, canonical_text, verse_entry)
            
            if similarity_scores['overall'] >= self.min_similarity_threshold:
                passage_type = self._determine_passage_type(segment, verse_entry)
                confidence = self._calculate_confidence_score(segment, verse_entry, similarity_scores)
                
                if confidence >= self.min_confidence_threshold:
                    match = VerseMatch(
                        original_text=segment.text,
                        canonical_entry=verse_entry,
                        passage_type=passage_type,
                        confidence_score=confidence,
                        similarity_score=similarity_scores['overall'],
                        match_start=segment.start_pos,
                        match_end=segment.end_pos,
                        word_matches=similarity_scores['word_matches'],
                        total_words=similarity_scores['total_words'],
                        partial_match=similarity_scores['partial']
                    )
                    matches.append(match)
        
        return matches
    
    def _calculate_verse_similarity(self, text: str, canonical: str, entry: LexiconEntry) -> Dict[str, float]:
        """
        Calculate similarity between text and canonical verse.
        
        Args:
            text: Input text
            canonical: Canonical verse text
            entry: Lexicon entry
            
        Returns:
            Dictionary of similarity metrics
        """
        canonical_lower = canonical.lower()
        
        # Fuzzy string matching
        fuzzy_match = self.fuzzy_matcher.calculate_similarity(text, canonical_lower)
        
        # Word-level matching
        text_words = set(text.split())
        canonical_words = set(canonical_lower.split())
        common_words = text_words & canonical_words
        
        word_similarity = len(common_words) / max(len(canonical_words), 1)
        
        # Partial matching (first few words)
        text_start = ' '.join(text.split()[:5])
        canonical_start = ' '.join(canonical_lower.split()[:5])
        partial_match = self.fuzzy_matcher.calculate_similarity(text_start, canonical_start)
        
        # Check variations
        variation_match = 0.0
        variations = getattr(entry, 'variations', [])
        for variation in variations:
            var_similarity = self.fuzzy_matcher.calculate_similarity(text, variation.lower())
            variation_match = max(variation_match, var_similarity)
        
        # Overall score (weighted combination)
        overall = (
            fuzzy_match * 0.4 +
            word_similarity * 0.3 +
            partial_match.similarity_score * 0.2 +
            variation_match * 0.1
        )
        
        return {
            'overall': overall,
            'fuzzy': fuzzy_match,
            'word_similarity': word_similarity,
            'partial': partial_match.similarity_score > 0.7,
            'variation': variation_match,
            'word_matches': len(common_words),
            'total_words': len(canonical_words)
        }
    
    def _determine_passage_type(self, segment: PassageSegment, entry: LexiconEntry) -> PassageType:
        """
        Determine the type of scriptural passage.
        
        Args:
            segment: Text segment
            entry: Lexicon entry
            
        Returns:
            Type of passage identified
        """
        category = getattr(entry, 'category', '')
        canonical_text = getattr(entry, 'canonical_text', '')
        
        if category == 'verse_reference' and canonical_text:
            if segment.word_count >= 8:
                return PassageType.VERSE
            else:
                return PassageType.PARTIAL_VERSE
        elif category == 'chapter_reference':
            return PassageType.CHAPTER_REFERENCE
        else:
            return PassageType.VERSE
    
    def _calculate_confidence_score(self, segment: PassageSegment, entry: LexiconEntry, 
                                  similarity_scores: Dict[str, float]) -> float:
        """
        Calculate confidence score for a verse match.
        
        Args:
            segment: Text segment
            entry: Lexicon entry
            similarity_scores: Similarity metrics
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        base_confidence = similarity_scores['overall']
        
        # Boost confidence for high word overlap
        word_boost = min(0.1, similarity_scores['word_matches'] / 10.0)
        
        # Boost for Sanskrit density
        density_boost = min(0.1, segment.sanskrit_word_density * 0.2)
        
        # Boost for verse markers
        verse_marker_boost = 0.05 if re.search(r'[|।]', segment.text) else 0.0
        
        # Penalty for very short matches
        length_penalty = max(0.0, (self.min_verse_length - len(segment.text)) / 100.0)
        
        confidence = base_confidence + word_boost + density_boost + verse_marker_boost - length_penalty
        
        return min(1.0, max(0.0, confidence))
    
    def _calculate_sanskrit_density(self, text: str) -> float:
        """
        Calculate the density of Sanskrit/Hindi words in text.
        
        Args:
            text: Input text
            
        Returns:
            Ratio of Sanskrit/Hindi words to total words
        """
        words = text.split()
        if not words:
            return 0.0
        
        sanskrit_count = 0
        for word in words:
            # Simple heuristic for Sanskrit characteristics
            if (re.search(r'[aeiou]m$', word.lower()) or  # Sanskrit endings
                re.search(r'[|।]', word) or  # Sanskrit punctuation
                word.lower() in self.lexicon_manager.get_all_entries()):
                sanskrit_count += 1
        
        return sanskrit_count / len(words)
    
    def _resolve_overlapping_matches(self, matches: List[VerseMatch]) -> List[VerseMatch]:
        """
        Resolve overlapping verse matches by keeping highest confidence.
        
        Args:
            matches: List of potential matches
            
        Returns:
            List of non-overlapping matches
        """
        if not matches:
            return matches
        
        # Sort by confidence descending
        sorted_matches = sorted(matches, key=lambda m: m.confidence_score, reverse=True)
        resolved_matches = []
        
        for match in sorted_matches:
            # Check if this match overlaps with any already resolved match
            overlaps = False
            for resolved in resolved_matches:
                if (match.match_start < resolved.match_end and 
                    match.match_end > resolved.match_start):
                    overlaps = True
                    break
            
            if not overlaps:
                resolved_matches.append(match)
        
        return resolved_matches
    
    def get_passage_boundaries(self, text: str, match: VerseMatch) -> Tuple[int, int]:
        """
        Get precise boundaries for a verse passage.
        
        Args:
            text: Full text
            match: Verse match
            
        Returns:
            Tuple of (start_position, end_position)
        """
        # Use the match boundaries as starting point
        start = match.match_start
        end = match.match_end
        
        # Look for verse boundary markers to extend boundaries
        for pattern in self.verse_boundary_patterns:
            # Look backwards for start boundary
            before_text = text[max(0, start-50):start]
            before_match = list(re.finditer(pattern, before_text))
            if before_match:
                boundary_pos = before_match[-1].end()
                start = max(0, start - 50 + boundary_pos)
            
            # Look forwards for end boundary  
            after_text = text[end:min(len(text), end+50)]
            after_match = re.search(pattern, after_text)
            if after_match:
                end = min(len(text), end + after_match.start())
        
        return start, end