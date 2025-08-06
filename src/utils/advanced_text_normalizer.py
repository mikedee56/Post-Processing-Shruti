"""
Advanced Text Normalizer with conversational nuance handling.

This module extends the basic TextNormalizer with advanced capabilities for handling
conversational speech patterns, partial phrases, and contextual corrections.
"""

import re
import logging
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from .text_normalizer import TextNormalizer, NormalizationResult


@dataclass
class ConversationalPattern:
    """Represents a detected conversational pattern and its correction."""
    pattern_type: str  # "partial_phrase", "rescinded", "filler_context"
    original_text: str
    corrected_text: str
    confidence_score: float
    context_clues: List[str]
    preservation_reason: Optional[str] = None


@dataclass
class AdvancedCorrectionResult:
    """Result of advanced text normalization with detailed tracking."""
    original_text: str
    corrected_text: str
    corrections_applied: List[str]
    conversational_fixes: List[ConversationalPattern]
    quality_score: float
    semantic_drift_score: float
    word_count_before: int
    word_count_after: int


class AdvancedTextNormalizer(TextNormalizer):
    """
    Advanced text normalizer with conversational nuance handling.
    
    Extends TextNormalizer with capabilities for:
    - Partial phrase detection and correction
    - Rescinded phrase identification
    - Context-aware filler word removal
    - Semantic preservation validation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the advanced text normalizer.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Setup advanced patterns
        self._setup_rescission_patterns()
        self._setup_partial_phrase_patterns()
        self._setup_meaningful_discourse_markers()
        
        # Advanced configuration
        self.preserve_meaningful_discourse = self.config.get('preserve_meaningful_discourse', True)
        self.semantic_drift_threshold = self.config.get('semantic_drift_threshold', 0.3)
        self.min_confidence_score = self.config.get('min_confidence_score', 0.7)
    
    def normalize_with_advanced_tracking(self, text: str) -> AdvancedCorrectionResult:
        """
        Apply advanced normalization with detailed conversational pattern tracking.
        
        Args:
            text: Input text to normalize
            
        Returns:
            AdvancedCorrectionResult with detailed tracking
        """
        if not text or not text.strip():
            return AdvancedCorrectionResult(
                original_text=text,
                corrected_text=text,
                corrections_applied=[],
                conversational_fixes=[],
                quality_score=1.0,
                semantic_drift_score=0.0,
                word_count_before=0,
                word_count_after=0
            )
        
        original_text = text
        current_text = text
        corrections_applied = []
        conversational_fixes = []
        
        word_count_before = len(current_text.split())
        
        # Step 1: Handle conversational nuances
        result = self.handle_conversational_nuances(current_text)
        if result.corrected_text != current_text:
            corrections_applied.append("handled_conversational_nuances")
            conversational_fixes.extend(result.patterns_detected)
            current_text = result.corrected_text
        
        # Step 2: Apply base normalization
        base_result = super().normalize_with_tracking(current_text)
        current_text = base_result.normalized_text
        corrections_applied.extend(base_result.changes_applied)
        
        # Step 3: Validate semantic preservation
        semantic_drift_score = self.calculate_semantic_drift(original_text, current_text)
        quality_score = self._calculate_quality_score(corrections_applied, semantic_drift_score)
        
        word_count_after = len(current_text.split())
        
        return AdvancedCorrectionResult(
            original_text=original_text,
            corrected_text=current_text,
            corrections_applied=corrections_applied,
            conversational_fixes=conversational_fixes,
            quality_score=quality_score,
            semantic_drift_score=semantic_drift_score,
            word_count_before=word_count_before,
            word_count_after=word_count_after
        )
    
    def handle_conversational_nuances(self, text: str) -> 'ConversationalCorrectionResult':
        """
        Handle conversational nuances including partial and rescinded phrases.
        
        Args:
            text: Input text
            
        Returns:
            ConversationalCorrectionResult with patterns detected and corrected text
        """
        current_text = text
        patterns_detected = []
        
        # Handle rescinded phrases first (highest priority)
        result = self.identify_and_correct_rescinded_phrases(current_text)
        current_text = result.corrected_text
        patterns_detected.extend(result.patterns_detected)
        
        # Handle partial phrases
        result = self.process_partial_phrases(current_text)
        current_text = result.corrected_text
        patterns_detected.extend(result.patterns_detected)
        
        # Handle meaningful discourse markers
        result = self.preserve_meaningful_discourse_markers(current_text)
        current_text = result.corrected_text
        patterns_detected.extend(result.patterns_detected)
        
        return ConversationalCorrectionResult(
            original_text=text,
            corrected_text=current_text,
            patterns_detected=patterns_detected
        )
    
    def identify_and_correct_rescinded_phrases(self, text: str) -> 'ConversationalCorrectionResult':
        """
        Identify and correct rescinded phrases like 'I mean', 'rather', 'actually'.
        
        Args:
            text: Input text
            
        Returns:
            ConversationalCorrectionResult with rescinded phrases corrected
        """
        patterns_detected = []
        current_text = text
        
        for pattern, replacement_strategy in self.rescission_patterns.items():
            matches = list(re.finditer(pattern, current_text, re.IGNORECASE))
            
            for match in reversed(matches):  # Process from end to preserve positions
                original_phrase = match.group(0)
                corrected_phrase = self._apply_rescission_strategy(
                    match, replacement_strategy, current_text
                )
                
                if corrected_phrase != original_phrase:
                    # Calculate confidence based on context
                    confidence = self._calculate_rescission_confidence(match, current_text)
                    
                    if confidence >= self.min_confidence_score:
                        pattern_info = ConversationalPattern(
                            pattern_type="rescinded",
                            original_text=original_phrase,
                            corrected_text=corrected_phrase,
                            confidence_score=confidence,
                            context_clues=self._extract_context_clues(match, current_text)
                        )
                        patterns_detected.append(pattern_info)
                        
                        # Apply the correction
                        current_text = (
                            current_text[:match.start()] + 
                            corrected_phrase + 
                            current_text[match.end():]
                        )
        
        return ConversationalCorrectionResult(
            original_text=text,
            corrected_text=current_text,
            patterns_detected=patterns_detected
        )
    
    def process_partial_phrases(self, text: str) -> 'ConversationalCorrectionResult':
        """
        Process partial phrases and incomplete thoughts.
        
        Args:
            text: Input text
            
        Returns:
            ConversationalCorrectionResult with partial phrases processed
        """
        patterns_detected = []
        current_text = text
        
        for pattern, completion_strategy in self.partial_phrase_patterns.items():
            matches = list(re.finditer(pattern, current_text, re.IGNORECASE))
            
            for match in reversed(matches):
                original_phrase = match.group(0)
                completed_phrase = self._apply_completion_strategy(
                    match, completion_strategy, current_text
                )
                
                if completed_phrase != original_phrase:
                    confidence = self._calculate_completion_confidence(match, current_text)
                    
                    if confidence >= self.min_confidence_score:
                        pattern_info = ConversationalPattern(
                            pattern_type="partial_phrase",
                            original_text=original_phrase,
                            corrected_text=completed_phrase,
                            confidence_score=confidence,
                            context_clues=self._extract_context_clues(match, current_text)
                        )
                        patterns_detected.append(pattern_info)
                        
                        current_text = (
                            current_text[:match.start()] + 
                            completed_phrase + 
                            current_text[match.end():]
                        )
        
        return ConversationalCorrectionResult(
            original_text=text,
            corrected_text=current_text,
            patterns_detected=patterns_detected
        )
    
    def preserve_meaningful_discourse_markers(self, text: str) -> 'ConversationalCorrectionResult':
        """
        Preserve meaningful discourse markers while removing pure fillers.
        
        Args:
            text: Input text
            
        Returns:
            ConversationalCorrectionResult with meaningful markers preserved
        """
        patterns_detected = []
        current_text = text
        
        # Override filler removal for meaningful discourse markers
        words = current_text.split()
        filtered_words = []
        
        i = 0
        while i < len(words):
            word = words[i]
            clean_word = re.sub(r'[^\w]', '', word.lower())
            
            # Check if this is a meaningful discourse marker in context
            if clean_word in self.potentially_meaningful_markers:
                is_meaningful = self._is_meaningful_in_context(words, i)
                
                if is_meaningful:
                    # Preserve this marker
                    filtered_words.append(word)
                    
                    pattern_info = ConversationalPattern(
                        pattern_type="filler_context",
                        original_text=word,
                        corrected_text=word,
                        confidence_score=0.8,
                        context_clues=self._extract_context_clues_from_words(words, i),
                        preservation_reason="meaningful_discourse_marker"
                    )
                    patterns_detected.append(pattern_info)
                else:
                    # Remove as filler
                    pass
            else:
                filtered_words.append(word)
            
            i += 1
        
        current_text = ' '.join(filtered_words)
        current_text = re.sub(r'\s+', ' ', current_text).strip()
        
        return ConversationalCorrectionResult(
            original_text=text,
            corrected_text=current_text,
            patterns_detected=patterns_detected
        )
    
    def calculate_semantic_drift(self, original: str, corrected: str) -> float:
        """
        Calculate semantic drift between original and corrected text.
        
        Args:
            original: Original text
            corrected: Corrected text
            
        Returns:
            Semantic drift score (0.0 = no drift, 1.0 = complete change)
        """
        # Simple implementation using word overlap and length differences
        original_words = set(original.lower().split())
        corrected_words = set(corrected.lower().split())
        
        if not original_words:
            return 0.0 if not corrected_words else 1.0
        
        # Calculate Jaccard similarity
        intersection = original_words.intersection(corrected_words)
        union = original_words.union(corrected_words)
        jaccard_similarity = len(intersection) / len(union) if union else 1.0
        
        # Calculate length ratio impact
        len_ratio = abs(len(original) - len(corrected)) / max(len(original), 1)
        
        # Combine metrics (1 - similarity gives us drift)
        semantic_drift = (1 - jaccard_similarity) * 0.7 + len_ratio * 0.3
        
        return min(semantic_drift, 1.0)
    
    def _setup_rescission_patterns(self):
        """Setup patterns for rescinded phrases."""
        self.rescission_patterns = {
            # "I mean" patterns
            r'\b(I\s+mean\s*,?\s*)(.*?)(?=\.|$)': 'keep_latter',
            
            # "Rather" corrections
            r'\b(.*?)\s*,?\s*(rather\s*,?\s*)(.*?)(?=\.|$)': 'keep_latter_if_complete',
            
            # "Actually" corrections  
            r'\b(.*?)\s*,?\s*(actually\s*,?\s*)(.*?)(?=\.|$)': 'keep_latter_if_complete',
            
            # "Let me rephrase" patterns
            r'\b(.*?)\s*,?\s*(let\s+me\s+rephrase\s*,?\s*)(.*?)(?=\.|$)': 'keep_latter',
            
            # "What I meant was" patterns
            r'\b(.*?)\s*,?\s*(what\s+I\s+meant\s+was\s*,?\s*)(.*?)(?=\.|$)': 'keep_latter',
        }
    
    def _setup_partial_phrase_patterns(self):
        """Setup patterns for partial phrases."""
        self.partial_phrase_patterns = {
            # Incomplete thoughts with trailing conjunctions
            r'\b(.*?)\s+(and|but|or|so)\s*\.?\s*$': 'remove_trailing_conjunction',
            
            # Interrupted speech
            r'\b(.*?)\s*-\s*$': 'clean_interruption',
            
            # Repeated starts - Use a different approach for this
            r'\b(\w+)\s+(\w+)\b': 'remove_repetition',
        }
    
    def _setup_meaningful_discourse_markers(self):
        """Setup meaningful discourse markers that should sometimes be preserved."""
        self.potentially_meaningful_markers = {
            'now', 'so', 'well', 'then', 'therefore', 'however', 'meanwhile',
            'furthermore', 'moreover', 'indeed', 'thus', 'hence', 'consequently'
        }
    
    def _apply_rescission_strategy(self, match, strategy: str, full_text: str) -> str:
        """Apply rescission correction strategy."""
        if strategy == 'keep_latter':
            # Keep only the part after the rescission marker
            groups = match.groups()
            if len(groups) >= 2:
                return groups[-1].strip()
        
        elif strategy == 'keep_latter_if_complete':
            # Keep latter part if it forms a complete thought
            groups = match.groups()
            if len(groups) >= 3:
                latter_part = groups[-1].strip()
                if self._is_complete_thought(latter_part):
                    return latter_part
                else:
                    # Keep original if latter part is incomplete
                    return match.group(0)
        
        return match.group(0)
    
    def _apply_completion_strategy(self, match, strategy: str, full_text: str) -> str:
        """Apply partial phrase completion strategy."""
        if strategy == 'remove_trailing_conjunction':
            # Remove trailing conjunction words
            groups = match.groups()
            if groups:
                return groups[0].strip()
        
        elif strategy == 'clean_interruption':
            # Clean up interrupted speech
            groups = match.groups()
            if groups:
                return groups[0].strip()
        
        elif strategy == 'remove_repetition':
            # Remove word repetitions - check if the two captured groups are the same
            groups = match.groups()
            if len(groups) >= 2 and groups[0].lower() == groups[1].lower():
                return groups[0]  # Return only one instance
            else:
                return match.group(0)  # No repetition found, return original
        
        return match.group(0)
    
    def _is_complete_thought(self, text: str) -> bool:
        """Check if text represents a complete thought."""
        text = text.strip()
        
        # Basic heuristics for complete thoughts
        if len(text.split()) < 3:  # Too short
            return False
        
        # Check for basic sentence structure (subject + verb indicators)
        has_verb_indicators = bool(re.search(r'\b(is|are|was|were|has|have|will|can|should|must|do|does|did)\b', text, re.IGNORECASE))
        has_meaningful_content = len(text.split()) >= 4
        
        return has_verb_indicators and has_meaningful_content
    
    def _is_meaningful_in_context(self, words: List[str], index: int) -> bool:
        """Check if a potential filler word is meaningful in its context."""
        if index >= len(words):
            return False
        
        word = re.sub(r'[^\w]', '', words[index].lower())
        
        # Context window
        start_idx = max(0, index - 2)
        end_idx = min(len(words), index + 3)
        context = ' '.join(words[start_idx:end_idx]).lower()
        
        # Rules for meaningful discourse markers
        if word == 'now':
            # "Now" is meaningful if it indicates transition
            return bool(re.search(r'\b(now\s+(let|we|this|that|here))', context))
        
        elif word == 'so':
            # "So" is meaningful as conclusion marker
            return bool(re.search(r'\b(so\s+(this|that|we|in))', context))
        
        elif word == 'well':
            # "Well" at sentence start can be meaningful
            return index == 0 or words[index-1].endswith(('.', '!', '?'))
        
        # Add more context-specific rules as needed
        return False
    
    def _calculate_rescission_confidence(self, match, text: str) -> float:
        """Calculate confidence score for rescission correction."""
        # Base confidence
        confidence = 0.7
        
        # Increase confidence for clear rescission markers
        rescission_text = match.group(0).lower()
        if 'i mean' in rescission_text:
            confidence += 0.2
        if 'rather' in rescission_text:
            confidence += 0.15
        if 'actually' in rescission_text:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _calculate_completion_confidence(self, match, text: str) -> float:
        """Calculate confidence score for partial phrase completion."""
        return 0.8  # Default confidence for partial phrase corrections
    
    def _extract_context_clues(self, match, text: str) -> List[str]:
        """Extract context clues around a match."""
        start = max(0, match.start() - 20)
        end = min(len(text), match.end() + 20)
        context = text[start:end]
        
        return [f"context: {context.strip()}"]
    
    def _extract_context_clues_from_words(self, words: List[str], index: int) -> List[str]:
        """Extract context clues from word list."""
        start_idx = max(0, index - 2)
        end_idx = min(len(words), index + 3)
        context = ' '.join(words[start_idx:end_idx])
        
        return [f"word_context: {context}"]
    
    def _calculate_quality_score(self, corrections: List[str], semantic_drift: float) -> float:
        """Calculate overall quality score for the corrections."""
        # Base quality score
        quality = 1.0
        
        # Reduce quality for high semantic drift
        quality -= semantic_drift * 0.6
        
        # Slight reduction for each correction (encouraging minimal changes)
        quality -= len(corrections) * 0.02
        
        return max(0.0, min(1.0, quality))


@dataclass
class ConversationalCorrectionResult:
    """Result of conversational pattern correction."""
    original_text: str
    corrected_text: str
    patterns_detected: List[ConversationalPattern]