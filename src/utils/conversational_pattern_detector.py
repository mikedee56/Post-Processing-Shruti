"""
Conversational Pattern Detector for identifying and correcting speech patterns.

This module provides specialized detection and correction capabilities for
conversational speech patterns including partial phrases, rescinded statements,
and discourse markers.
"""

import re
import logging
from typing import Dict, List, Set, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum


class PatternType(Enum):
    """Types of conversational patterns."""
    RESCINDED = "rescinded"
    PARTIAL_PHRASE = "partial_phrase"
    REPETITION = "repetition"
    INTERRUPTION = "interruption"
    DISCOURSE_MARKER = "discourse_marker"
    FILLER_CONTEXT = "filler_context"


@dataclass
class PatternMatch:
    """Represents a detected conversational pattern."""
    pattern_type: PatternType
    start_pos: int
    end_pos: int
    original_text: str
    suggested_correction: str
    confidence_score: float
    context_window: str
    reasoning: str


@dataclass
class DetectionResult:
    """Result of pattern detection analysis."""
    text: str
    patterns_found: List[PatternMatch]
    total_patterns: int
    high_confidence_patterns: int
    processing_notes: List[str]


class ConversationalPatternDetector:
    """
    Detects and analyzes conversational speech patterns for correction.
    
    This detector identifies various patterns in conversational speech that
    typically require correction or special handling in transcript processing.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the conversational pattern detector.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Detection thresholds
        self.min_confidence_threshold = self.config.get('min_confidence_threshold', 0.7)
        self.context_window_size = self.config.get('context_window_size', 50)
        
        # Initialize pattern definitions
        self._setup_rescission_patterns()
        self._setup_partial_phrase_patterns()
        self._setup_repetition_patterns()
        self._setup_interruption_patterns()
        self._setup_discourse_marker_patterns()
    
    def detect_patterns(self, text: str) -> DetectionResult:
        """
        Detect all conversational patterns in the given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            DetectionResult with all detected patterns
        """
        if not text or not text.strip():
            return DetectionResult(
                text=text,
                patterns_found=[],
                total_patterns=0,
                high_confidence_patterns=0,
                processing_notes=["Empty or whitespace-only text"]
            )
        
        patterns_found = []
        processing_notes = []
        
        # Detect each pattern type
        patterns_found.extend(self._detect_rescission_patterns(text))
        patterns_found.extend(self._detect_partial_phrase_patterns(text))
        patterns_found.extend(self._detect_repetition_patterns(text))
        patterns_found.extend(self._detect_interruption_patterns(text))
        patterns_found.extend(self._detect_discourse_marker_patterns(text))
        
        # Sort patterns by position for consistent processing
        patterns_found.sort(key=lambda p: p.start_pos)
        
        # Filter overlapping patterns (keep highest confidence)
        patterns_found = self._resolve_overlapping_patterns(patterns_found)
        
        # Calculate statistics
        high_confidence_patterns = sum(
            1 for p in patterns_found if p.confidence_score >= self.min_confidence_threshold
        )
        
        if patterns_found:
            processing_notes.append(f"Detected {len(patterns_found)} patterns")
            processing_notes.append(f"{high_confidence_patterns} high-confidence patterns")
        
        return DetectionResult(
            text=text,
            patterns_found=patterns_found,
            total_patterns=len(patterns_found),
            high_confidence_patterns=high_confidence_patterns,
            processing_notes=processing_notes
        )
    
    def apply_pattern_corrections(self, text: str, patterns: List[PatternMatch]) -> str:
        """
        Apply corrections for detected patterns.
        
        Args:
            text: Original text
            patterns: List of patterns to correct
            
        Returns:
            Text with pattern corrections applied
        """
        if not patterns:
            return text
        
        # Sort patterns by position (reverse order for safe string replacement)
        patterns_to_apply = [
            p for p in patterns 
            if p.confidence_score >= self.min_confidence_threshold
        ]
        patterns_to_apply.sort(key=lambda p: p.start_pos, reverse=True)
        
        corrected_text = text
        
        for pattern in patterns_to_apply:
            try:
                # Apply the correction
                corrected_text = (
                    corrected_text[:pattern.start_pos] +
                    pattern.suggested_correction +
                    corrected_text[pattern.end_pos:]
                )
                
                self.logger.debug(
                    f"Applied {pattern.pattern_type.value} correction: "
                    f"'{pattern.original_text}' -> '{pattern.suggested_correction}'"
                )
                
            except Exception as e:
                self.logger.warning(
                    f"Failed to apply pattern correction: {e}. "
                    f"Pattern: {pattern.pattern_type.value} at {pattern.start_pos}-{pattern.end_pos}"
                )
        
        return corrected_text
    
    def _detect_rescission_patterns(self, text: str) -> List[PatternMatch]:
        """Detect rescinded phrase patterns."""
        patterns = []
        
        for pattern_def in self.rescission_patterns:
            matches = list(re.finditer(pattern_def['regex'], text, re.IGNORECASE | re.MULTILINE))
            
            for match in matches:
                correction = self._generate_rescission_correction(match, pattern_def)
                confidence = self._calculate_rescission_confidence(match, text, pattern_def)
                
                if correction != match.group(0):
                    context = self._extract_context_window(text, match.start(), match.end())
                    
                    pattern_match = PatternMatch(
                        pattern_type=PatternType.RESCINDED,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        original_text=match.group(0),
                        suggested_correction=correction,
                        confidence_score=confidence,
                        context_window=context,
                        reasoning=pattern_def['reasoning']
                    )
                    patterns.append(pattern_match)
        
        return patterns
    
    def _detect_partial_phrase_patterns(self, text: str) -> List[PatternMatch]:
        """Detect partial phrase patterns."""
        patterns = []
        
        for pattern_def in self.partial_phrase_patterns:
            matches = list(re.finditer(pattern_def['regex'], text, re.IGNORECASE | re.MULTILINE))
            
            for match in matches:
                correction = self._generate_partial_phrase_correction(match, pattern_def)
                confidence = self._calculate_partial_phrase_confidence(match, text, pattern_def)
                
                if correction != match.group(0):
                    context = self._extract_context_window(text, match.start(), match.end())
                    
                    pattern_match = PatternMatch(
                        pattern_type=PatternType.PARTIAL_PHRASE,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        original_text=match.group(0),
                        suggested_correction=correction,
                        confidence_score=confidence,
                        context_window=context,
                        reasoning=pattern_def['reasoning']
                    )
                    patterns.append(pattern_match)
        
        return patterns
    
    def _detect_repetition_patterns(self, text: str) -> List[PatternMatch]:
        """Detect word and phrase repetition patterns."""
        patterns = []
        
        for pattern_def in self.repetition_patterns:
            matches = list(re.finditer(pattern_def['regex'], text, re.IGNORECASE))
            
            for match in matches:
                correction = self._generate_repetition_correction(match, pattern_def)
                confidence = pattern_def.get('base_confidence', 0.8)
                
                if correction != match.group(0):
                    context = self._extract_context_window(text, match.start(), match.end())
                    
                    pattern_match = PatternMatch(
                        pattern_type=PatternType.REPETITION,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        original_text=match.group(0),
                        suggested_correction=correction,
                        confidence_score=confidence,
                        context_window=context,
                        reasoning=pattern_def['reasoning']
                    )
                    patterns.append(pattern_match)
        
        return patterns
    
    def _detect_interruption_patterns(self, text: str) -> List[PatternMatch]:
        """Detect interruption and incomplete thought patterns."""
        patterns = []
        
        for pattern_def in self.interruption_patterns:
            matches = list(re.finditer(pattern_def['regex'], text, re.IGNORECASE | re.MULTILINE))
            
            for match in matches:
                correction = self._generate_interruption_correction(match, pattern_def)
                confidence = self._calculate_interruption_confidence(match, text, pattern_def)
                
                if correction != match.group(0):
                    context = self._extract_context_window(text, match.start(), match.end())
                    
                    pattern_match = PatternMatch(
                        pattern_type=PatternType.INTERRUPTION,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        original_text=match.group(0),
                        suggested_correction=correction,
                        confidence_score=confidence,
                        context_window=context,
                        reasoning=pattern_def['reasoning']
                    )
                    patterns.append(pattern_match)
        
        return patterns
    
    def _detect_discourse_marker_patterns(self, text: str) -> List[PatternMatch]:
        """Detect meaningful discourse markers that should be preserved."""
        patterns = []
        
        words = text.split()
        for i, word in enumerate(words):
            clean_word = re.sub(r'[^\w]', '', word.lower())
            
            if clean_word in self.discourse_markers:
                marker_def = self.discourse_markers[clean_word]
                
                # Check if this marker is meaningful in context
                is_meaningful, reasoning = self._is_discourse_marker_meaningful(words, i, marker_def)
                
                if is_meaningful:
                    # Find the position in the original text
                    word_start = self._find_word_position(text, word, i)
                    word_end = word_start + len(word)
                    
                    context = self._extract_context_window(text, word_start, word_end)
                    
                    pattern_match = PatternMatch(
                        pattern_type=PatternType.DISCOURSE_MARKER,
                        start_pos=word_start,
                        end_pos=word_end,
                        original_text=word,
                        suggested_correction=word,  # Preserve as-is
                        confidence_score=0.8,
                        context_window=context,
                        reasoning=f"Meaningful discourse marker: {reasoning}"
                    )
                    patterns.append(pattern_match)
        
        return patterns
    
    def _setup_rescission_patterns(self):
        """Setup rescission pattern definitions."""
        self.rescission_patterns = [
            {
                'regex': r'\b(I\s+mean\s*,?\s*)(.*?)(?=\.|,|$)',
                'strategy': 'keep_latter',
                'base_confidence': 0.85,
                'reasoning': 'Speaker correcting/clarifying their statement'
            },
            {
                'regex': r'\b(.*?)\s*,?\s*(rather\s*,?\s*)(.*?)(?=\.|,|$)',
                'strategy': 'keep_latter_if_complete',
                'base_confidence': 0.8,
                'reasoning': 'Speaker providing a correction with "rather"'
            },
            {
                'regex': r'\b(.*?)\s*,?\s*(actually\s*,?\s*)(.*?)(?=\.|,|$)',
                'strategy': 'keep_latter_if_complete',
                'base_confidence': 0.75,
                'reasoning': 'Speaker providing a correction with "actually"'
            },
            {
                'regex': r'\b(let\s+me\s+rephrase\s*,?\s*)(.*?)(?=\.|$)',
                'strategy': 'keep_latter',
                'base_confidence': 0.9,
                'reasoning': 'Explicit request to rephrase'
            },
            {
                'regex': r'\b(what\s+I\s+meant\s+(was|is)\s*,?\s*)(.*?)(?=\.|$)',
                'strategy': 'keep_latter',
                'base_confidence': 0.9,
                'reasoning': 'Explicit clarification of meaning'
            }
        ]
    
    def _setup_partial_phrase_patterns(self):
        """Setup partial phrase pattern definitions."""
        self.partial_phrase_patterns = [
            {
                'regex': r'\b(.*?)\s+(and|but|or|so)\s*\.?\s*$',
                'strategy': 'remove_trailing_conjunction',
                'base_confidence': 0.7,
                'reasoning': 'Incomplete thought ending with conjunction'
            },
            {
                'regex': r'\b(.*?)\s*-\s*$',
                'strategy': 'clean_interruption',
                'base_confidence': 0.8,
                'reasoning': 'Interrupted speech marked with dash'
            },
            {
                'regex': r'\b(.*?)\s*\.\.\.\s*$',
                'strategy': 'clean_ellipsis',
                'base_confidence': 0.6,
                'reasoning': 'Trailing off speech marked with ellipsis'
            }
        ]
    
    def _setup_repetition_patterns(self):
        """Setup repetition pattern definitions."""
        self.repetition_patterns = [
            {
                'regex': r'\b(\w+)\s+\1\b',
                'strategy': 'remove_duplicate',
                'base_confidence': 0.9,
                'reasoning': 'Immediate word repetition'
            },
            {
                'regex': r'\b(\w+\s+\w+)\s+\1\b',
                'strategy': 'remove_duplicate_phrase',
                'base_confidence': 0.8,
                'reasoning': 'Immediate phrase repetition'
            },
            {
                'regex': r'\b(the\s+the|a\s+a|an\s+an|and\s+and|or\s+or)\b',
                'strategy': 'remove_article_duplicate',
                'base_confidence': 0.95,
                'reasoning': 'Duplicate articles or conjunctions'
            }
        ]
    
    def _setup_interruption_patterns(self):
        """Setup interruption pattern definitions."""
        self.interruption_patterns = [
            {
                'regex': r'\b(.*?)\s*--\s*(.*?)(?=\.|$)',
                'strategy': 'join_interrupted_thought',
                'base_confidence': 0.7,
                'reasoning': 'Interrupted thought marked with double dash'
            },
            {
                'regex': r'\[\w+\]',  # [interruption] markers
                'strategy': 'remove_interruption_marker',
                'base_confidence': 0.9,
                'reasoning': 'Explicit interruption marker'
            }
        ]
    
    def _setup_discourse_marker_patterns(self):
        """Setup discourse marker definitions."""
        self.discourse_markers = {
            'now': {
                'meaningful_contexts': [r'now\s+(let|we|this|that|here)', r'now\s+\w+'],
                'filler_contexts': [r'now\s*,', r'now\s+um', r'now\s+uh']
            },
            'so': {
                'meaningful_contexts': [r'so\s+(this|that|we|in|therefore)', r'so\s+\w+'],
                'filler_contexts': [r'so\s*,\s*', r'so\s+um', r'so\s+like']
            },
            'well': {
                'meaningful_contexts': [r'^well\s+', r'\.well\s+'],  # Sentence-initial
                'filler_contexts': [r'well\s*,', r'well\s+um']
            },
            'then': {
                'meaningful_contexts': [r'then\s+(we|this|that)', r'and\s+then'],
                'filler_contexts': [r'then\s*,\s*um']
            }
        }
    
    def _generate_rescission_correction(self, match, pattern_def: Dict) -> str:
        """Generate correction for rescission pattern."""
        strategy = pattern_def['strategy']
        try:
            groups = match.groups()
        except AttributeError:
            return str(match)
        
        if strategy == 'keep_latter' and len(groups) >= 2:
            return groups[-1].strip()
        
        elif strategy == 'keep_latter_if_complete' and len(groups) >= 3:
            latter_part = groups[-1].strip()
            if self._is_complete_thought(latter_part):
                return latter_part
        
        return match.group(0)  # No change if strategy doesn't apply
    
    def _generate_partial_phrase_correction(self, match, pattern_def: Dict) -> str:
        """Generate correction for partial phrase pattern."""
        strategy = pattern_def['strategy']
        
        if strategy == 'remove_trailing_conjunction':
            try:
                groups = match.groups()
            except AttributeError:
                return str(match)
            if groups:
                return groups[0].strip()
        
        elif strategy == 'clean_interruption':
            try:
                return match.group(1).strip() if match.groups() else match.group(0).strip()
            except AttributeError:
                return str(match).strip()
        
        elif strategy == 'clean_ellipsis':
            try:
                return match.group(1).strip() if match.groups() else match.group(0).replace('...', '').strip()
            except AttributeError:
                return str(match).replace('...', '').strip()
        
        return match.group(0)
    
    def _generate_repetition_correction(self, match, pattern_def: Dict) -> str:
        """Generate correction for repetition pattern."""
        strategy = pattern_def['strategy']
        
        if strategy == 'remove_duplicate':
            try:
                groups = match.groups()
            except AttributeError:
                return str(match)
            if groups:
                return groups[0]
        
        elif strategy == 'remove_duplicate_phrase':
            try:
                groups = match.groups()
            except AttributeError:
                return str(match)
            if groups:
                return groups[0]
        
        elif strategy == 'remove_article_duplicate':
            # Extract the first occurrence of the repeated word
            repeated_text = match.group(0)
            words = repeated_text.split()
            if len(words) >= 2:
                return words[0]
        
        return match.group(0)
    
    def _generate_interruption_correction(self, match, pattern_def: Dict) -> str:
        """Generate correction for interruption pattern."""
        strategy = pattern_def['strategy']
        
        if strategy == 'join_interrupted_thought':
            try:
                groups = match.groups()
            except AttributeError:
                return str(match)
            if len(groups) >= 2:
                return f"{groups[0].strip()} {groups[1].strip()}"
        
        elif strategy == 'remove_interruption_marker':
            return ""  # Remove the interruption marker entirely
        
        return match.group(0)
    
    def _calculate_rescission_confidence(self, match, text: str, pattern_def: Dict) -> float:
        """Calculate confidence for rescission pattern."""
        base_confidence = pattern_def.get('base_confidence', 0.7)
        
        # Increase confidence for explicit rescission markers
        matched_text = match.group(0).lower()
        if 'i mean' in matched_text:
            base_confidence += 0.1
        if 'let me rephrase' in matched_text:
            base_confidence += 0.15
        if 'what i meant' in matched_text:
            base_confidence += 0.15
        
        return min(base_confidence, 1.0)
    
    def _calculate_partial_phrase_confidence(self, match, text: str, pattern_def: Dict) -> float:
        """Calculate confidence for partial phrase pattern."""
        return pattern_def.get('base_confidence', 0.7)
    
    def _calculate_interruption_confidence(self, match, text: str, pattern_def: Dict) -> float:
        """Calculate confidence for interruption pattern."""
        return pattern_def.get('base_confidence', 0.7)
    
    def _is_complete_thought(self, text: str) -> bool:
        """Check if text represents a complete thought."""
        text = text.strip()
        
        if len(text.split()) < 3:
            return False
        
        # Look for basic sentence structure indicators
        has_verb = bool(re.search(
            r'\b(is|are|was|were|has|have|had|will|would|can|could|should|must|do|does|did|being|been)\b',
            text, re.IGNORECASE
        ))
        
        has_subject = len(text.split()) >= 4
        
        return has_verb and has_subject
    
    def _is_discourse_marker_meaningful(self, words: List[str], index: int, marker_def: Dict) -> Tuple[bool, str]:
        """Check if discourse marker is meaningful in context."""
        if index >= len(words):
            return False, "Index out of range"
        
        # Create context window
        start_idx = max(0, index - 2)
        end_idx = min(len(words), index + 3)
        context = ' '.join(words[start_idx:end_idx]).lower()
        
        # Check meaningful contexts
        for meaningful_pattern in marker_def.get('meaningful_contexts', []):
            if re.search(meaningful_pattern, context, re.IGNORECASE):
                return True, f"Matched meaningful pattern: {meaningful_pattern}"
        
        # Check filler contexts
        for filler_pattern in marker_def.get('filler_contexts', []):
            if re.search(filler_pattern, context, re.IGNORECASE):
                return False, f"Matched filler pattern: {filler_pattern}"
        
        # Default to meaningful if no clear filler pattern
        return True, "No clear filler pattern detected"
    
    def _find_word_position(self, text: str, word: str, word_index: int) -> int:
        """Find the character position of a word in text by word index."""
        words_seen = 0
        pos = 0
        
        while pos < len(text) and words_seen <= word_index:
            # Skip whitespace
            while pos < len(text) and text[pos].isspace():
                pos += 1
            
            if pos >= len(text):
                break
            
            # Found start of a word
            if words_seen == word_index:
                return pos
            
            # Skip to end of current word
            while pos < len(text) and not text[pos].isspace():
                pos += 1
            
            words_seen += 1
        
        return pos
    
    def _extract_context_window(self, text: str, start_pos: int, end_pos: int) -> str:
        """Extract context window around a position."""
        window_start = max(0, start_pos - self.context_window_size)
        window_end = min(len(text), end_pos + self.context_window_size)
        
        return text[window_start:window_end].strip()
    
    def _resolve_overlapping_patterns(self, patterns: List[PatternMatch]) -> List[PatternMatch]:
        """Resolve overlapping patterns by keeping the highest confidence ones."""
        if not patterns:
            return patterns
        
        resolved_patterns = []
        sorted_patterns = sorted(patterns, key=lambda p: p.confidence_score, reverse=True)
        
        for pattern in sorted_patterns:
            # Check if this pattern overlaps with any already accepted pattern
            overlaps = False
            for accepted_pattern in resolved_patterns:
                if (pattern.start_pos < accepted_pattern.end_pos and 
                    pattern.end_pos > accepted_pattern.start_pos):
                    overlaps = True
                    break
            
            if not overlaps:
                resolved_patterns.append(pattern)
        
        # Sort by position for consistent processing
        resolved_patterns.sort(key=lambda p: p.start_pos)
        
        return resolved_patterns