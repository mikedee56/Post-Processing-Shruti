"""
Verse Substitution Engine Module.

This module handles the replacement of transcribed passages with canonical text,
including boundary detection, validation, and rollback capabilities.
"""

from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum
import logging
import re
from copy import deepcopy

from utils.logger_config import get_logger
from utils.srt_parser import SRTSegment
from .scripture_identifier import ScriptureIdentifier, VerseMatch, PassageType
from .canonical_text_manager import CanonicalTextManager, CanonicalVerse
from .scripture_validator import ScriptureValidator


class SubstitutionAction(Enum):
    """Types of substitution actions."""
    REPLACE = "replace"
    PREPEND = "prepend"
    APPEND = "append"
    SKIP = "skip"


@dataclass
class SubstitutionOperation:
    """Represents a verse substitution operation."""
    verse_match: VerseMatch
    original_text: str
    canonical_text: str
    action: SubstitutionAction
    start_pos: int
    end_pos: int
    confidence: float
    validation_passed: bool
    metadata: Dict[str, Any]


@dataclass
class SubstitutionResult:
    """Result of a verse substitution process."""
    original_text: str
    substituted_text: str
    operations_performed: List[SubstitutionOperation]
    operations_skipped: List[SubstitutionOperation]
    validation_warnings: List[str]
    overall_confidence: float
    rollback_data: Dict[str, Any]


class VerseSubstitutionEngine:
    """
    Engine for replacing transcribed passages with canonical verse text.
    
    Handles accurate boundary detection, substitution validation, and provides
    rollback capabilities for incorrect substitutions.
    """
    
    def __init__(self, scripture_identifier: ScriptureIdentifier = None,
                 canonical_manager: CanonicalTextManager = None,
                 validator: 'ScriptureValidator' = None,
                 config: Dict = None):
        """
        Initialize the Verse Substitution Engine.
        
        Args:
            scripture_identifier: Scripture identification component
            canonical_manager: Canonical text management
            validator: Scripture validation component
            config: Configuration parameters
        """
        self.logger = get_logger(__name__)
        self.scripture_identifier = scripture_identifier or ScriptureIdentifier()
        self.canonical_manager = canonical_manager or CanonicalTextManager()
        self.validator = validator
        
        # Configuration
        self.config = config or {}
        self.min_substitution_confidence = self.config.get('min_substitution_confidence', 0.8)
        self.require_validation = self.config.get('require_validation', True)
        self.preserve_formatting = self.config.get('preserve_formatting', True)
        self.max_substitutions_per_segment = self.config.get('max_substitutions_per_segment', 3)
        
        # Boundary detection patterns
        self.verse_boundary_markers = self.config.get('verse_boundary_markers', [
            r'\|\|',  # Sanskrit verse endings
            r'।।',   # Devanagari verse endings
            r'\.\s*\n',  # Period followed by newline
            r'[।|]\s*\n',  # Sanskrit punctuation followed by newline
        ])
        
        self.logger.info("Verse substitution engine initialized")
    
    def substitute_verses_in_text(self, text: str) -> SubstitutionResult:
        """
        Replace verses in text with canonical versions.
        
        Args:
            text: Input text with potential verses
            
        Returns:
            Substitution result with all operations
        """
        # Identify potential verses
        verse_matches = self.scripture_identifier.identify_scripture_passages(text)
        
        # Filter matches by confidence
        qualified_matches = [
            match for match in verse_matches 
            if match.confidence_score >= self.min_substitution_confidence
        ]
        
        # Limit substitutions per segment
        if len(qualified_matches) > self.max_substitutions_per_segment:
            qualified_matches = qualified_matches[:self.max_substitutions_per_segment]
        
        # Prepare substitution operations
        operations = []
        for match in qualified_matches:
            operation = self._prepare_substitution_operation(match, text)
            if operation:
                operations.append(operation)
        
        # Validate operations if validator is available
        if self.validator and self.require_validation:
            operations = self._validate_operations(operations, text)
        
        # Perform substitutions
        result = self._execute_substitutions(text, operations)
        
        return result
    
    def substitute_verses_in_segment(self, segment: SRTSegment) -> Tuple[SRTSegment, SubstitutionResult]:
        """
        Replace verses in an SRT segment.
        
        Args:
            segment: SRT segment to process
            
        Returns:
            Tuple of (modified_segment, substitution_result)
        """
        # Process the text
        substitution_result = self.substitute_verses_in_text(segment.text)
        
        # Create new segment with substituted text
        new_segment = SRTSegment(
            index=segment.index,
            start_time=segment.start_time,
            end_time=segment.end_time,
            text=substitution_result.substituted_text
        )
        
        return new_segment, substitution_result
    
    def _prepare_substitution_operation(self, match: VerseMatch, text: str) -> Optional[SubstitutionOperation]:
        """
        Prepare a substitution operation for a verse match.
        
        Args:
            match: Verse match to process
            text: Full text context
            
        Returns:
            Substitution operation if valid
        """
        try:
            # Get canonical text
            canonical_verse = self.canonical_manager.lookup_verse_by_reference(
                self._extract_verse_reference(match.canonical_entry)
            )
            
            if not canonical_verse or not canonical_verse.canonical_text:
                self.logger.warning(f"No canonical text found for match: {match.original_text}")
                return None
            
            # Determine substitution action
            action = self._determine_substitution_action(match)
            
            # Get precise boundaries
            start_pos, end_pos = self._get_substitution_boundaries(match, text)
            
            # Prepare canonical text with proper formatting
            formatted_canonical = self._format_canonical_text(
                canonical_verse.canonical_text, 
                match.original_text,
                action
            )
            
            operation = SubstitutionOperation(
                verse_match=match,
                original_text=text[start_pos:end_pos],
                canonical_text=formatted_canonical,
                action=action,
                start_pos=start_pos,
                end_pos=end_pos,
                confidence=match.confidence_score,
                validation_passed=False,  # Will be set during validation
                metadata={
                    'canonical_verse_id': canonical_verse.id,
                    'source': canonical_verse.source.value,
                    'chapter': canonical_verse.chapter,
                    'verse': canonical_verse.verse
                }
            )
            
            return operation
            
        except Exception as e:
            self.logger.error(f"Error preparing substitution operation: {e}")
            return None
    
    def _determine_substitution_action(self, match: VerseMatch) -> SubstitutionAction:
        """
        Determine the appropriate substitution action.
        
        Args:
            match: Verse match
            
        Returns:
            Substitution action to take
        """
        if match.passage_type == PassageType.VERSE and match.confidence_score >= 0.9:
            return SubstitutionAction.REPLACE
        elif match.passage_type == PassageType.PARTIAL_VERSE:
            if match.partial_match:
                return SubstitutionAction.REPLACE
            else:
                return SubstitutionAction.PREPEND
        elif match.passage_type == PassageType.CHAPTER_REFERENCE:
            return SubstitutionAction.SKIP  # Don't replace chapter references
        else:
            return SubstitutionAction.REPLACE
    
    def _get_substitution_boundaries(self, match: VerseMatch, text: str) -> Tuple[int, int]:
        """
        Get precise boundaries for substitution.
        
        Args:
            match: Verse match
            text: Full text
            
        Returns:
            Tuple of (start_position, end_position)
        """
        # Start with match boundaries
        start = match.match_start
        end = match.match_end
        
        # Extend to verse boundaries if possible
        extended_boundaries = self.scripture_identifier.get_passage_boundaries(text, match)
        if extended_boundaries:
            start, end = extended_boundaries
        
        # Ensure we don't go beyond text boundaries
        start = max(0, start)
        end = min(len(text), end)
        
        # Adjust boundaries to word boundaries to avoid cutting words
        while start > 0 and not text[start-1].isspace():
            start -= 1
        while end < len(text) and not text[end].isspace():
            end += 1
        
        return start, end
    
    def _format_canonical_text(self, canonical_text: str, original_text: str, 
                              action: SubstitutionAction) -> str:
        """
        Format canonical text for substitution.
        
        Args:
            canonical_text: Canonical verse text
            original_text: Original text being replaced
            action: Substitution action
            
        Returns:
            Formatted canonical text
        """
        if not self.preserve_formatting:
            return canonical_text
        
        formatted = canonical_text
        
        # Preserve capitalization patterns
        if original_text and original_text[0].isupper():
            formatted = formatted[0].upper() + formatted[1:] if len(formatted) > 1 else formatted.upper()
        
        # Handle punctuation
        original_ends_with_period = original_text.rstrip().endswith('.')
        canonical_ends_with_period = canonical_text.rstrip().endswith(('.', '।', '||'))
        
        if original_ends_with_period and not canonical_ends_with_period:
            formatted = formatted.rstrip() + '.'
        
        # Handle line breaks
        if '\n' in original_text:
            # Try to preserve line break structure
            original_lines = original_text.count('\n')
            if original_lines == 1 and '\n' not in formatted:
                # Add a line break in the middle for verse structure
                words = formatted.split()
                mid_point = len(words) // 2
                formatted = ' '.join(words[:mid_point]) + '\n' + ' '.join(words[mid_point:])
        
        return formatted
    
    def _validate_operations(self, operations: List[SubstitutionOperation], 
                           text: str) -> List[SubstitutionOperation]:
        """
        Validate substitution operations.
        
        Args:
            operations: List of operations to validate
            text: Original text
            
        Returns:
            List of validated operations
        """
        if not self.validator:
            # Mark all as validation passed if no validator
            for op in operations:
                op.validation_passed = True
            return operations
        
        validated_operations = []
        for operation in operations:
            try:
                validation_result = self.validator.validate_substitution(
                    operation.original_text,
                    operation.canonical_text,
                    operation.verse_match
                )
                
                operation.validation_passed = validation_result.is_valid
                if validation_result.is_valid:
                    validated_operations.append(operation)
                else:
                    self.logger.warning(f"Substitution validation failed: {validation_result.errors}")
                    
            except Exception as e:
                self.logger.error(f"Error validating substitution: {e}")
                operation.validation_passed = False
        
        return validated_operations
    
    def _execute_substitutions(self, text: str, operations: List[SubstitutionOperation]) -> SubstitutionResult:
        """
        Execute the substitution operations.
        
        Args:
            text: Original text
            operations: Validated operations to perform
            
        Returns:
            Complete substitution result
        """
        # Sort operations by position (reverse order for proper text replacement)
        operations.sort(key=lambda op: op.start_pos, reverse=True)
        
        modified_text = text
        performed_operations = []
        skipped_operations = []
        warnings = []
        
        # Create rollback data
        rollback_data = {
            'original_text': text,
            'operations': deepcopy(operations)
        }
        
        for operation in operations:
            try:
                if operation.action == SubstitutionAction.REPLACE:
                    # Adjust positions for already-made substitutions
                    adjusted_start, adjusted_end = self._adjust_positions_for_substitutions(
                        operation.start_pos, operation.end_pos, performed_operations
                    )
                    
                    # Perform replacement
                    before = modified_text[:adjusted_start]
                    after = modified_text[adjusted_end:]
                    modified_text = before + operation.canonical_text + after
                    
                    performed_operations.append(operation)
                    
                elif operation.action == SubstitutionAction.PREPEND:
                    adjusted_start, _ = self._adjust_positions_for_substitutions(
                        operation.start_pos, operation.end_pos, performed_operations
                    )
                    
                    before = modified_text[:adjusted_start]
                    after = modified_text[adjusted_start:]
                    modified_text = before + operation.canonical_text + " " + after
                    
                    performed_operations.append(operation)
                    
                elif operation.action == SubstitutionAction.APPEND:
                    _, adjusted_end = self._adjust_positions_for_substitutions(
                        operation.start_pos, operation.end_pos, performed_operations
                    )
                    
                    before = modified_text[:adjusted_end]
                    after = modified_text[adjusted_end:]
                    modified_text = before + " " + operation.canonical_text + after
                    
                    performed_operations.append(operation)
                    
                else:  # SKIP
                    skipped_operations.append(operation)
                    
            except Exception as e:
                self.logger.error(f"Error executing substitution: {e}")
                warnings.append(f"Failed to execute substitution: {str(e)}")
                skipped_operations.append(operation)
        
        # Calculate overall confidence
        if performed_operations:
            overall_confidence = sum(op.confidence for op in performed_operations) / len(performed_operations)
        else:
            overall_confidence = 0.0
        
        return SubstitutionResult(
            original_text=text,
            substituted_text=modified_text,
            operations_performed=performed_operations,
            operations_skipped=skipped_operations,
            validation_warnings=warnings,
            overall_confidence=overall_confidence,
            rollback_data=rollback_data
        )
    
    def _adjust_positions_for_substitutions(self, start: int, end: int, 
                                          performed_ops: List[SubstitutionOperation]) -> Tuple[int, int]:
        """
        Adjust text positions for already-performed substitutions.
        
        Args:
            start: Original start position
            end: Original end position
            performed_ops: Operations already performed
            
        Returns:
            Adjusted (start, end) positions
        """
        adjustment = 0
        
        for op in performed_ops:
            if op.end_pos <= start:
                # Operation was before our target, adjust by length difference
                original_length = op.end_pos - op.start_pos
                new_length = len(op.canonical_text)
                adjustment += new_length - original_length
        
        return start + adjustment, end + adjustment
    
    def _extract_verse_reference(self, lexicon_entry) -> Optional['VerseReference']:
        """Extract verse reference from lexicon entry."""
        try:
            # This is a placeholder - would need to implement based on lexicon structure
            from .canonical_text_manager import VerseReference, ScriptureSource
            
            # Try to parse from original_term or category
            original_term = getattr(lexicon_entry, 'original_term', '').lower()
            
            if 'bhagavad gita' in original_term or 'gita' in original_term:
                # Extract chapter and verse numbers
                chapter_match = re.search(r'chapter\s+(\d+)', original_term)
                verse_match = re.search(r'verse\s+(\d+)', original_term)
                
                if chapter_match and verse_match:
                    return VerseReference(
                        source=ScriptureSource.BHAGAVAD_GITA,
                        chapter=int(chapter_match.group(1)),
                        verse=int(verse_match.group(1))
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting verse reference: {e}")
            return None
    
    def rollback_substitutions(self, result: SubstitutionResult) -> str:
        """
        Rollback substitutions to original text.
        
        Args:
            result: Substitution result to rollback
            
        Returns:
            Original text
        """
        return result.rollback_data.get('original_text', result.substituted_text)
    
    def get_substitution_preview(self, text: str, max_operations: int = 5) -> List[Dict[str, Any]]:
        """
        Get a preview of potential substitutions without executing them.
        
        Args:
            text: Input text
            max_operations: Maximum operations to preview
            
        Returns:
            List of operation previews
        """
        verse_matches = self.scripture_identifier.identify_scripture_passages(text)
        qualified_matches = [
            match for match in verse_matches 
            if match.confidence_score >= self.min_substitution_confidence
        ][:max_operations]
        
        previews = []
        for match in qualified_matches:
            operation = self._prepare_substitution_operation(match, text)
            if operation:
                preview = {
                    'original_text': operation.original_text,
                    'canonical_text': operation.canonical_text,
                    'action': operation.action.value,
                    'confidence': operation.confidence,
                    'position': f"{operation.start_pos}-{operation.end_pos}",
                    'metadata': operation.metadata
                }
                previews.append(preview)
        
        return previews