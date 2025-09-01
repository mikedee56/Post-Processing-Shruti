"""
High-Confidence Correction Application System.

This module provides functionality to apply high-confidence corrections to text
while maintaining transcript integrity and tracking all changes made.
"""

import re
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

from utils.logger_config import get_logger
from utils.fuzzy_matcher import FuzzyMatch, MatchType
from utils.iast_transliterator import TransliterationResult
from sanskrit_hindi_identifier.word_identifier import IdentifiedWord


class CorrectionType(Enum):
    """Types of corrections that can be applied."""
    DIRECT_REPLACEMENT = "direct_replacement"
    FUZZY_MATCH = "fuzzy_match"
    TRANSLITERATION = "transliteration"
    COMPOUND_WORD = "compound_word"
    CONTEXTUAL = "contextual"


class CorrectionPriority(Enum):
    """Priority levels for corrections."""
    CRITICAL = 1    # Must be applied (high confidence, known terms)
    HIGH = 2        # Should be applied (high confidence)
    MEDIUM = 3      # May be applied (medium confidence)
    LOW = 4         # Suggest only (low confidence)


@dataclass
class CorrectionCandidate:
    """Represents a potential correction to be applied."""
    original_text: str
    corrected_text: str
    position: int
    length: int
    confidence: float
    correction_type: CorrectionType
    priority: CorrectionPriority
    source: str
    metadata: Dict = field(default_factory=dict)
    context_before: str = ""
    context_after: str = ""
    

@dataclass
class AppliedCorrection:
    """Represents a correction that has been applied."""
    original_text: str
    corrected_text: str
    position: int
    confidence: float
    correction_type: CorrectionType
    source: str
    timestamp: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class CorrectionResult:
    """Result of applying corrections to text."""
    original_text: str
    corrected_text: str
    corrections_applied: List[AppliedCorrection]
    corrections_skipped: List[CorrectionCandidate]
    overall_confidence: float
    processing_time: float
    warnings: List[str]


class CorrectionApplier:
    """
    High-confidence correction application system.
    
    Applies corrections based on confidence thresholds while maintaining
    text integrity and providing detailed tracking of all changes.
    """

    def __init__(self, 
                 min_confidence: float = 0.80,
                 critical_confidence: float = 0.95,
                 enable_context_validation: bool = True,
                 max_corrections_per_segment: int = 10):
        """
        Initialize the correction applier.
        
        Args:
            min_confidence: Minimum confidence threshold for applying corrections
            critical_confidence: Threshold for critical priority corrections
            enable_context_validation: Whether to validate corrections in context
            max_corrections_per_segment: Maximum corrections per text segment
        """
        self.logger = get_logger(__name__)
        self.min_confidence = min_confidence
        self.critical_confidence = critical_confidence
        self.enable_context_validation = enable_context_validation
        self.max_corrections_per_segment = max_corrections_per_segment
        
        # Track statistics
        self.correction_stats = {
            'total_corrections_applied': 0,
            'corrections_by_type': {},
            'corrections_by_confidence': {},
            'conflicts_resolved': 0,
            'corrections_skipped': 0
        }
        
        self.logger.info(f"CorrectionApplier initialized (min_confidence={min_confidence})")

    def apply_corrections(self, text: str, candidates: List[CorrectionCandidate]) -> CorrectionResult:
        """
        Apply corrections to text based on candidates and confidence thresholds.
        
        Args:
            text: Original text to correct
            candidates: List of correction candidates
            
        Returns:
            CorrectionResult with applied corrections and metadata
        """
        import time
        start_time = time.time()
        
        original_text = text
        corrected_text = text
        applied_corrections = []
        skipped_corrections = []
        warnings = []
        
        # Filter and prioritize candidates
        filtered_candidates = self._filter_candidates(candidates, text)
        prioritized_candidates = self._prioritize_candidates(filtered_candidates)
        
        # Resolve conflicts between overlapping corrections
        resolved_candidates = self._resolve_conflicts(prioritized_candidates, text)
        
        # Limit number of corrections per segment
        if len(resolved_candidates) > self.max_corrections_per_segment:
            warnings.append(f"Limited corrections to {self.max_corrections_per_segment} per segment")
            resolved_candidates = resolved_candidates[:self.max_corrections_per_segment]
        
        # Apply corrections in order (from end to beginning to preserve positions)
        resolved_candidates.sort(key=lambda c: c.position, reverse=True)
        
        for candidate in resolved_candidates:
            if self._should_apply_correction(candidate, corrected_text):
                try:
                    # Apply the correction
                    correction_result = self._apply_single_correction(
                        corrected_text, candidate
                    )
                    
                    if correction_result:
                        corrected_text = correction_result['text']
                        applied_corrections.append(correction_result['correction'])
                        self._update_stats(candidate)
                    else:
                        skipped_corrections.append(candidate)
                        
                except Exception as e:
                    self.logger.error(f"Error applying correction: {e}")
                    skipped_corrections.append(candidate)
                    warnings.append(f"Failed to apply correction: {candidate.original_text} -> {candidate.corrected_text}")
            else:
                skipped_corrections.append(candidate)
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(applied_corrections, len(candidates))
        
        processing_time = time.time() - start_time
        
        return CorrectionResult(
            original_text=original_text,
            corrected_text=corrected_text,
            corrections_applied=applied_corrections,
            corrections_skipped=skipped_corrections,
            overall_confidence=overall_confidence,
            processing_time=processing_time,
            warnings=warnings
        )

    def _filter_candidates(self, candidates: List[CorrectionCandidate], text: str) -> List[CorrectionCandidate]:
        """Filter candidates based on confidence and validity."""
        filtered = []
        
        # CRITICAL ANTI-HALLUCINATION SAFEGUARDS - PROTECTION LIST
        protected_words = {
            'who', 'what', 'when', 'where', 'why', 'how', 'and', 'the', 'is', 'are', 'was', 'were',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'must', 'shall', 'ought', 'i', 'me', 'my', 'mine',
            'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its',
            'we', 'us', 'our', 'ours', 'they', 'them', 'their', 'theirs', 'this', 'that',
            'these', 'those', 'a', 'an', 'some', 'any', 'all', 'every', 'each', 'in', 'on',
            'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'between', 'among', 'under', 'over',
            'but', 'or', 'nor', 'so', 'yet', 'because', 'since', 'unless', 'while', 'although',
            'though', 'if', 'when', 'where', 'whether', 'one', 'two', 'three', 'four', 'five',
            'chapter', 'verse', 'entitled', 'text', 'scripture', 'book', 'as'
        }
        
        for candidate in candidates:
            # ABSOLUTE PROTECTION: Never touch protected English words
            original_word = candidate.original_text.lower().strip()
            if original_word in protected_words:
                self.logger.debug(f"Skipping protected word: {original_word}")
                continue
                
            # MODIFIED PROTECTION: Skip very short words (length <= 3) but allow proper nouns
            is_proper_noun = candidate.metadata.get('is_proper_noun', False)
            if len(original_word) <= 3 and not is_proper_noun:
                self.logger.debug(f"Skipping short non-proper-noun word: {original_word}")
                continue
            
            # ULTRA-STRICT confidence check - increased threshold
            if candidate.confidence < max(0.95, self.min_confidence):  # Minimum 95% confidence
                continue
            
            # Validate position and length
            if not self._validate_candidate_position(candidate, text):
                continue
            
            # Context validation if enabled
            if self.enable_context_validation:
                if not self._validate_context(candidate, text):
                    continue
            
            # Check for reasonable corrections (not too extreme)
            if not self._is_reasonable_correction(candidate):
                continue
            
            filtered.append(candidate)
        
        return filtered

    def _prioritize_candidates(self, candidates: List[CorrectionCandidate]) -> List[CorrectionCandidate]:
        """Prioritize candidates based on confidence and type."""
        # Update priorities based on confidence
        for candidate in candidates:
            if candidate.confidence >= self.critical_confidence:
                candidate.priority = CorrectionPriority.CRITICAL
            elif candidate.confidence >= 0.90:
                candidate.priority = CorrectionPriority.HIGH
            elif candidate.confidence >= 0.80:
                candidate.priority = CorrectionPriority.MEDIUM
            else:
                candidate.priority = CorrectionPriority.LOW
        
        # Sort by priority and confidence
        return sorted(candidates, key=lambda c: (c.priority.value, -c.confidence))

    def _resolve_conflicts(self, candidates: List[CorrectionCandidate], text: str) -> List[CorrectionCandidate]:
        """Resolve conflicts between overlapping corrections."""
        if not candidates:
            return []
        
        resolved = []
        used_positions = set()
        
        for candidate in candidates:
            position_range = set(range(candidate.position, candidate.position + candidate.length))
            
            # Check for overlap with already used positions
            if position_range.intersection(used_positions):
                # Skip this candidate due to conflict
                self.correction_stats['conflicts_resolved'] += 1
                continue
            
            # Add this candidate and mark positions as used
            resolved.append(candidate)
            used_positions.update(position_range)
        
        return resolved

    def _should_apply_correction(self, candidate: CorrectionCandidate, current_text: str) -> bool:
        """Determine if a correction should be applied."""
        # Critical corrections are always applied
        if candidate.priority == CorrectionPriority.CRITICAL:
            return True
        
        # Check confidence threshold
        if candidate.confidence < self.min_confidence:
            return False
        
        # Additional validation for non-critical corrections
        if candidate.priority in [CorrectionPriority.LOW, CorrectionPriority.MEDIUM]:
            # More stringent checks for lower priority corrections
            if len(candidate.original_text) < 3:  # Very short words need higher confidence
                return candidate.confidence >= 0.90
        
        return True

    def _apply_single_correction(self, text: str, candidate: CorrectionCandidate) -> Optional[Dict]:
        """Apply a single correction to text."""
        try:
            # Extract the text at the specified position
            start_pos = candidate.position
            end_pos = start_pos + candidate.length
            
            # Verify the text matches what we expect
            actual_text = text[start_pos:end_pos]
            if actual_text.lower() != candidate.original_text.lower():
                self.logger.warning(f"Text mismatch: expected '{candidate.original_text}', found '{actual_text}'")
                return None
            
            # Apply the correction
            new_text = text[:start_pos] + candidate.corrected_text + text[end_pos:]
            
            # Create applied correction record
            import datetime
            applied_correction = AppliedCorrection(
                original_text=candidate.original_text,
                corrected_text=candidate.corrected_text,
                position=candidate.position,
                confidence=candidate.confidence,
                correction_type=candidate.correction_type,
                source=candidate.source,
                timestamp=datetime.datetime.now().isoformat(),
                metadata=candidate.metadata
            )
            
            return {
                'text': new_text,
                'correction': applied_correction
            }
            
        except Exception as e:
            self.logger.error(f"Error in _apply_single_correction: {e}")
            return None

    def _validate_candidate_position(self, candidate: CorrectionCandidate, text: str) -> bool:
        """Validate that candidate position is valid for the text."""
        if candidate.position < 0 or candidate.position >= len(text):
            return False
        
        if candidate.position + candidate.length > len(text):
            return False
        
        return True

    def _validate_context(self, candidate: CorrectionCandidate, text: str) -> bool:
        """Validate correction makes sense in context."""
        # Extract context around the correction
        context_size = 20
        start = max(0, candidate.position - context_size)
        end = min(len(text), candidate.position + candidate.length + context_size)
        context = text[start:end]
        
        # Basic context validation - ensure it's not breaking words
        if candidate.position > 0:
            char_before = text[candidate.position - 1]
            if char_before.isalnum():  # Would break a word
                return False
        
        if candidate.position + candidate.length < len(text):
            char_after = text[candidate.position + candidate.length]
            if char_after.isalnum():  # Would break a word
                return False
        
        return True

    def _is_reasonable_correction(self, candidate: CorrectionCandidate) -> bool:
        """Check if correction is reasonable (not too extreme)."""
        original_len = len(candidate.original_text)
        corrected_len = len(candidate.corrected_text)
        
        # Don't allow corrections that change length too drastically
        if original_len > 0:
            length_ratio = corrected_len / original_len
            if length_ratio > 3.0 or length_ratio < 0.3:
                return False
        
        # Don't correct very short words unless confidence is very high
        if original_len <= 2 and candidate.confidence < 0.95:
            return False
        
        return True

    def _update_stats(self, candidate: CorrectionCandidate) -> None:
        """Update correction statistics."""
        self.correction_stats['total_corrections_applied'] += 1
        
        # Count by type
        correction_type = candidate.correction_type.value
        self.correction_stats['corrections_by_type'][correction_type] = \
            self.correction_stats['corrections_by_type'].get(correction_type, 0) + 1
        
        # Count by confidence range
        if candidate.confidence >= 0.95:
            confidence_range = '95-100%'
        elif candidate.confidence >= 0.90:
            confidence_range = '90-95%'
        elif candidate.confidence >= 0.80:
            confidence_range = '80-90%'
        else:
            confidence_range = '<80%'
        
        self.correction_stats['corrections_by_confidence'][confidence_range] = \
            self.correction_stats['corrections_by_confidence'].get(confidence_range, 0) + 1

    def _calculate_overall_confidence(self, applied_corrections: List[AppliedCorrection], total_candidates: int) -> float:
        """Calculate overall confidence for the correction process."""
        if not applied_corrections:
            return 1.0  # No corrections needed/applied
        
        # Average confidence of applied corrections
        avg_confidence = sum(c.confidence for c in applied_corrections) / len(applied_corrections)
        
        # Factor in the ratio of applied vs total candidates
        application_ratio = len(applied_corrections) / max(1, total_candidates)
        
        # Combine both factors
        overall_confidence = (avg_confidence * 0.7) + (application_ratio * 0.3)
        
        return min(1.0, overall_confidence)

    def create_candidates_from_fuzzy_matches(self, matches: List[FuzzyMatch], text: str) -> List[CorrectionCandidate]:
        """Create correction candidates from fuzzy matches."""
        candidates = []
        
        for match in matches:
            # Find the position of the match in the text
            position = self._find_word_position(match.original_word, text, match.position)
            
            if position is not None:
                candidate = CorrectionCandidate(
                    original_text=match.original_word,
                    corrected_text=match.transliteration or match.corrected_term,
                    position=position,
                    length=len(match.original_word),
                    confidence=match.confidence,
                    correction_type=self._match_type_to_correction_type(match.match_type),
                    priority=CorrectionPriority.MEDIUM,  # Will be updated in prioritization
                    source=f"fuzzy_match_{match.source_lexicon}",
                    metadata={
                        'match_type': match.match_type.value,
                        'category': match.category.value if hasattr(match.category, 'value') else str(match.category),
                        'is_proper_noun': match.is_proper_noun,
                        'distance': getattr(match, 'distance', 0)
                    }
                )
                candidates.append(candidate)
        
        return candidates

    def create_candidates_from_identified_words(self, identified_words: List[IdentifiedWord], text: str) -> List[CorrectionCandidate]:
        """Create correction candidates from identified words."""
        candidates = []
        
        for word in identified_words:
            if word.transliteration and word.transliteration != word.word:
                candidate = CorrectionCandidate(
                    original_text=word.word,
                    corrected_text=word.transliteration,
                    position=word.position,
                    length=len(word.word),
                    confidence=word.confidence,
                    correction_type=CorrectionType.TRANSLITERATION,
                    priority=CorrectionPriority.HIGH if word.confidence >= 0.90 else CorrectionPriority.MEDIUM,
                    source=f"word_identifier_{word.source_lexicon}",
                    metadata={
                        'category': word.category.value if hasattr(word.category, 'value') else str(word.category),
                        'is_proper_noun': word.is_proper_noun,
                        'variations': word.variations
                    }
                )
                candidates.append(candidate)
        
        return candidates

    def _find_word_position(self, word: str, text: str, hint_position: int = 0) -> Optional[int]:
        """Find the position of a word in text, using a hint position if provided."""
        # First try exact position from hint
        if hint_position < len(text) and hint_position >= 0:
            if text[hint_position:hint_position + len(word)].lower() == word.lower():
                return hint_position
        
        # Search for the word in the text
        word_lower = word.lower()
        text_lower = text.lower()
        
        # Use word boundary regex for better matching
        pattern = r'\b' + re.escape(word_lower) + r'\b'
        match = re.search(pattern, text_lower)
        
        if match:
            return match.start()
        
        return None

    def _match_type_to_correction_type(self, match_type: MatchType) -> CorrectionType:
        """Convert fuzzy match type to correction type."""
        mapping = {
            MatchType.EXACT: CorrectionType.DIRECT_REPLACEMENT,
            MatchType.LEVENSHTEIN: CorrectionType.FUZZY_MATCH,
            MatchType.PHONETIC: CorrectionType.FUZZY_MATCH,
            MatchType.PARTIAL: CorrectionType.COMPOUND_WORD,
            MatchType.TOKEN_SORT: CorrectionType.CONTEXTUAL,
            MatchType.TOKEN_SET: CorrectionType.CONTEXTUAL
        }
        return mapping.get(match_type, CorrectionType.FUZZY_MATCH)

    def get_correction_stats(self) -> Dict:
        """Get correction statistics."""
        return self.correction_stats.copy()

    def reset_stats(self) -> None:
        """Reset correction statistics."""
        self.correction_stats = {
            'total_corrections_applied': 0,
            'corrections_by_type': {},
            'corrections_by_confidence': {},
            'conflicts_resolved': 0,
            'corrections_skipped': 0
        }