"""
Verse Selection System Module.

This module provides intelligent verse candidate ranking and selection
capabilities for ambiguous scripture identification scenarios.
"""

from typing import List, Dict, Tuple, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import re
from datetime import datetime

from utils.logger_config import get_logger
from .canonical_text_manager import CanonicalVerse, CanonicalTextManager
from .scripture_identifier import VerseMatch, ScriptureIdentifier
from .scripture_validator import ScriptureValidator


class SelectionConfidence(Enum):
    """Confidence levels for automatic selection."""
    VERY_HIGH = "very_high"  # 0.95+
    HIGH = "high"           # 0.85-0.94
    MEDIUM = "medium"       # 0.70-0.84
    LOW = "low"            # 0.50-0.69
    VERY_LOW = "very_low"  # <0.50


class SelectionStrategy(Enum):
    """Strategies for verse selection."""
    AUTOMATIC = "automatic"              # Full automation
    INTERACTIVE = "interactive"          # User interaction required
    CONFIDENCE_BASED = "confidence_based" # Based on confidence thresholds
    HYBRID = "hybrid"                    # Mix of automatic and interactive


@dataclass
class VerseCandidateScore:
    """Scoring details for a verse candidate."""
    verse: CanonicalVerse
    verse_match: VerseMatch
    overall_score: float
    confidence_score: float
    similarity_score: float
    semantic_score: float
    structure_score: float
    source_authority_score: float
    context_score: float
    ranking: int
    selection_reasons: List[str] = field(default_factory=list)
    disqualification_reasons: List[str] = field(default_factory=list)


@dataclass
class SelectionResult:
    """Result of verse selection process."""
    selected_verse: Optional[CanonicalVerse]
    all_candidates: List[VerseCandidateScore]
    selection_method: SelectionStrategy
    confidence_level: SelectionConfidence
    requires_human_review: bool
    selection_metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class InteractivePrompt:
    """Interactive selection prompt for ambiguous cases."""
    original_text: str
    candidates: List[VerseCandidateScore]
    context_info: Dict[str, Any]
    prompt_message: str
    selection_options: List[Dict[str, Any]]


class VerseSelectionSystem:
    """
    Intelligent system for ranking and selecting verse candidates.
    
    Provides automatic selection with configurable confidence thresholds,
    interactive selection for ambiguous cases, and comprehensive candidate
    ranking with source attribution.
    """
    
    def __init__(self, canonical_manager: CanonicalTextManager = None,
                 scripture_identifier: ScriptureIdentifier = None,
                 validator: ScriptureValidator = None,
                 config: Dict = None):
        """
        Initialize the Verse Selection System.
        
        Args:
            canonical_manager: Canonical text management
            scripture_identifier: Scripture identification component
            validator: Scripture validation component
            config: Configuration parameters
        """
        self.logger = get_logger(__name__)
        self.canonical_manager = canonical_manager or CanonicalTextManager()
        self.scripture_identifier = scripture_identifier or ScriptureIdentifier()
        self.validator = validator or ScriptureValidator()
        
        # Configuration
        self.config = config or {}
        self.selection_strategy = SelectionStrategy(
            self.config.get('selection_strategy', 'hybrid')
        )
        self.confidence_thresholds = {
            SelectionConfidence.VERY_HIGH: self.config.get('very_high_threshold', 0.95),
            SelectionConfidence.HIGH: self.config.get('high_threshold', 0.85),
            SelectionConfidence.MEDIUM: self.config.get('medium_threshold', 0.70),
            SelectionConfidence.LOW: self.config.get('low_threshold', 0.50),
        }
        self.auto_select_threshold = self.config.get('auto_select_threshold', 0.90)
        self.human_review_threshold = self.config.get('human_review_threshold', 0.70)
        self.max_candidates = self.config.get('max_candidates', 5)
        
        # Scoring weights
        self.scoring_weights = {
            'confidence': self.config.get('confidence_weight', 0.30),
            'similarity': self.config.get('similarity_weight', 0.25),
            'semantic': self.config.get('semantic_weight', 0.20),
            'structure': self.config.get('structure_weight', 0.10),
            'authority': self.config.get('authority_weight', 0.10),
            'context': self.config.get('context_weight', 0.05)
        }
        
        # Interactive selection callback
        self.interactive_selector: Optional[Callable] = None
        
        self.logger.info(f"Verse selection system initialized with {self.selection_strategy.value} strategy")
    
    def select_best_verse(self, text: str, context: Dict = None) -> SelectionResult:
        """
        Select the best verse candidate for given text.
        
        Args:
            text: Input text to match against verses
            context: Additional context information
            
        Returns:
            Selection result with chosen verse and metadata
        """
        context = context or {}
        
        # Get verse candidates
        candidates = self._get_ranked_candidates(text, context)
        
        if not candidates:
            return SelectionResult(
                selected_verse=None,
                all_candidates=[],
                selection_method=self.selection_strategy,
                confidence_level=SelectionConfidence.VERY_LOW,
                requires_human_review=True,
                selection_metadata={'reason': 'No candidates found'}
            )
        
        # Determine selection method
        top_candidate = candidates[0]
        confidence_level = self._determine_confidence_level(top_candidate.overall_score)
        
        # Apply selection strategy
        if self.selection_strategy == SelectionStrategy.AUTOMATIC:
            selected = self._automatic_selection(candidates)
        elif self.selection_strategy == SelectionStrategy.INTERACTIVE:
            selected = self._interactive_selection(text, candidates, context)
        elif self.selection_strategy == SelectionStrategy.CONFIDENCE_BASED:
            selected = self._confidence_based_selection(candidates)
        else:  # HYBRID
            selected = self._hybrid_selection(text, candidates, context)
        
        return selected
    
    def rank_verse_candidates(self, text: str, candidates: List[CanonicalVerse],
                            context: Dict = None) -> List[VerseCandidateScore]:
        """
        Rank a list of verse candidates.
        
        Args:
            text: Input text
            candidates: List of candidate verses
            context: Additional context
            
        Returns:
            Ranked list of candidate scores
        """
        context = context or {}
        scored_candidates = []
        
        for verse in candidates:
            score = self._score_candidate(text, verse, context)
            scored_candidates.append(score)
        
        # Sort by overall score (descending)
        scored_candidates.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Assign rankings
        for i, candidate in enumerate(scored_candidates):
            candidate.ranking = i + 1
        
        return scored_candidates[:self.max_candidates]
    
    def _get_ranked_candidates(self, text: str, context: Dict) -> List[VerseCandidateScore]:
        """Get and rank all potential verse candidates."""
        # Identify verse matches
        verse_matches = self.scripture_identifier.identify_scripture_passages(text)
        
        if not verse_matches:
            return []
        
        # Get canonical verses for matches
        candidate_verses = []
        for match in verse_matches:
            verse_candidates = self.canonical_manager.get_verse_candidates(
                match.original_text,
                max_candidates=3  # Multiple candidates per match
            )
            candidate_verses.extend(verse_candidates)
        
        # Remove duplicates
        unique_candidates = list({v.id: v for v in candidate_verses}.values())
        
        # Rank candidates
        return self.rank_verse_candidates(text, unique_candidates, context)
    
    def _score_candidate(self, text: str, verse: CanonicalVerse, context: Dict) -> VerseCandidateScore:
        """
        Score a verse candidate comprehensively.
        
        Args:
            text: Input text
            verse: Candidate verse
            context: Additional context
            
        Returns:
            Comprehensive candidate score
        """
        # Create a dummy verse match for scoring
        verse_matches = self.scripture_identifier.identify_scripture_passages(text)
        verse_match = verse_matches[0] if verse_matches else None
        
        # Calculate component scores
        confidence_score = verse_match.confidence_score if verse_match else 0.5
        similarity_score = self._calculate_similarity_score(text, verse)
        semantic_score = self._calculate_semantic_score(text, verse)
        structure_score = self._calculate_structure_score(text, verse)
        authority_score = self._calculate_authority_score(verse)
        context_score = self._calculate_context_score(verse, context)
        
        # Calculate weighted overall score
        overall_score = (
            confidence_score * self.scoring_weights['confidence'] +
            similarity_score * self.scoring_weights['similarity'] +
            semantic_score * self.scoring_weights['semantic'] +
            structure_score * self.scoring_weights['structure'] +
            authority_score * self.scoring_weights['authority'] +
            context_score * self.scoring_weights['context']
        )
        
        # Generate selection reasons
        reasons = self._generate_selection_reasons(
            confidence_score, similarity_score, semantic_score,
            structure_score, authority_score, context_score
        )
        
        return VerseCandidateScore(
            verse=verse,
            verse_match=verse_match,
            overall_score=overall_score,
            confidence_score=confidence_score,
            similarity_score=similarity_score,
            semantic_score=semantic_score,
            structure_score=structure_score,
            source_authority_score=authority_score,
            context_score=context_score,
            ranking=0,  # Will be set during ranking
            selection_reasons=reasons
        )
    
    def _calculate_similarity_score(self, text: str, verse: CanonicalVerse) -> float:
        """Calculate text similarity score."""
        canonical_text = verse.canonical_text or ""
        if not canonical_text:
            return 0.0
        
        # Simple word overlap scoring
        text_words = set(text.lower().split())
        canonical_words = set(canonical_text.lower().split())
        
        if not canonical_words:
            return 0.0
        
        intersection = text_words & canonical_words
        union = text_words | canonical_words
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_semantic_score(self, text: str, verse: CanonicalVerse) -> float:
        """Calculate semantic similarity score."""
        # This is a simplified semantic scoring
        # In a full implementation, this could use embeddings or semantic analysis
        
        canonical_text = verse.canonical_text or ""
        if not canonical_text:
            return 0.0
        
        # Length similarity
        len_ratio = min(len(text), len(canonical_text)) / max(len(text), len(canonical_text))
        
        # Character overlap
        text_chars = set(text.lower())
        canonical_chars = set(canonical_text.lower())
        char_overlap = len(text_chars & canonical_chars) / len(text_chars | canonical_chars)
        
        return (len_ratio * 0.6 + char_overlap * 0.4)
    
    def _calculate_structure_score(self, text: str, verse: CanonicalVerse) -> float:
        """Calculate verse structure score."""
        canonical_text = verse.canonical_text or ""
        
        # Check for verse markers
        text_has_markers = bool(re.search(r'[|।]', text))
        canonical_has_markers = bool(re.search(r'[|।]', canonical_text))
        
        # Bonus for matching verse structure
        if text_has_markers and canonical_has_markers:
            return 1.0
        elif not text_has_markers and not canonical_has_markers:
            return 0.7
        else:
            return 0.3
    
    def _calculate_authority_score(self, verse: CanonicalVerse) -> float:
        """Calculate source authority score."""
        authority = verse.source_authority or "unknown"
        
        # Score based on source authority
        authority_scores = {
            'iast': 1.0,
            'scholar': 0.9,
            'traditional': 0.8,
            'academic': 0.9,
            'unknown': 0.5
        }
        
        return authority_scores.get(authority.lower(), 0.5)
    
    def _calculate_context_score(self, verse: CanonicalVerse, context: Dict) -> float:
        """Calculate contextual relevance score."""
        if not context:
            return 0.5
        
        score = 0.5  # Base score
        
        # Source context
        if 'preferred_source' in context:
            if verse.source.value == context['preferred_source']:
                score += 0.3
        
        # Chapter context
        if 'chapter_hint' in context:
            if verse.chapter == context['chapter_hint']:
                score += 0.2
        
        # Tag context
        if 'content_tags' in context:
            verse_tags = set(verse.tags)
            context_tags = set(context['content_tags'])
            if verse_tags & context_tags:
                score += 0.2
        
        return min(1.0, score)
    
    def _generate_selection_reasons(self, confidence: float, similarity: float,
                                  semantic: float, structure: float,
                                  authority: float, context: float) -> List[str]:
        """Generate human-readable selection reasons."""
        reasons = []
        
        if confidence >= 0.9:
            reasons.append("Very high confidence match")
        elif confidence >= 0.8:
            reasons.append("High confidence match")
        
        if similarity >= 0.8:
            reasons.append("Strong textual similarity")
        elif similarity >= 0.6:
            reasons.append("Good textual similarity")
        
        if semantic >= 0.8:
            reasons.append("Excellent semantic match")
        
        if structure >= 0.8:
            reasons.append("Matching verse structure")
        
        if authority >= 0.9:
            reasons.append("Authoritative source")
        
        if context >= 0.8:
            reasons.append("Strong contextual relevance")
        
        return reasons
    
    def _determine_confidence_level(self, score: float) -> SelectionConfidence:
        """Determine confidence level from score."""
        if score >= self.confidence_thresholds[SelectionConfidence.VERY_HIGH]:
            return SelectionConfidence.VERY_HIGH
        elif score >= self.confidence_thresholds[SelectionConfidence.HIGH]:
            return SelectionConfidence.HIGH
        elif score >= self.confidence_thresholds[SelectionConfidence.MEDIUM]:
            return SelectionConfidence.MEDIUM
        elif score >= self.confidence_thresholds[SelectionConfidence.LOW]:
            return SelectionConfidence.LOW
        else:
            return SelectionConfidence.VERY_LOW
    
    def _automatic_selection(self, candidates: List[VerseCandidateScore]) -> SelectionResult:
        """Perform automatic verse selection."""
        if not candidates:
            return SelectionResult(
                selected_verse=None,
                all_candidates=[],
                selection_method=SelectionStrategy.AUTOMATIC,
                confidence_level=SelectionConfidence.VERY_LOW,
                requires_human_review=True,
                selection_metadata={'reason': 'No candidates available'}
            )
        
        top_candidate = candidates[0]
        confidence_level = self._determine_confidence_level(top_candidate.overall_score)
        
        return SelectionResult(
            selected_verse=top_candidate.verse,
            all_candidates=candidates,
            selection_method=SelectionStrategy.AUTOMATIC,
            confidence_level=confidence_level,
            requires_human_review=confidence_level in [SelectionConfidence.LOW, SelectionConfidence.VERY_LOW],
            selection_metadata={
                'selection_reason': 'Highest scored candidate',
                'score': top_candidate.overall_score
            }
        )
    
    def _confidence_based_selection(self, candidates: List[VerseCandidateScore]) -> SelectionResult:
        """Perform confidence-based selection."""
        if not candidates:
            return SelectionResult(
                selected_verse=None,
                all_candidates=[],
                selection_method=SelectionStrategy.CONFIDENCE_BASED,
                confidence_level=SelectionConfidence.VERY_LOW,
                requires_human_review=True,
                selection_metadata={'reason': 'No candidates available'}
            )
        
        top_candidate = candidates[0]
        confidence_level = self._determine_confidence_level(top_candidate.overall_score)
        
        # Auto-select only if above threshold
        if top_candidate.overall_score >= self.auto_select_threshold:
            selected_verse = top_candidate.verse
            requires_review = False
            reason = 'Automatic selection - confidence threshold met'
        else:
            selected_verse = None
            requires_review = True
            reason = f'Below confidence threshold ({self.auto_select_threshold})'
        
        return SelectionResult(
            selected_verse=selected_verse,
            all_candidates=candidates,
            selection_method=SelectionStrategy.CONFIDENCE_BASED,
            confidence_level=confidence_level,
            requires_human_review=requires_review,
            selection_metadata={
                'selection_reason': reason,
                'threshold': self.auto_select_threshold,
                'score': top_candidate.overall_score
            }
        )
    
    def _interactive_selection(self, text: str, candidates: List[VerseCandidateScore],
                             context: Dict) -> SelectionResult:
        """Perform interactive selection (placeholder for user interaction)."""
        # This would integrate with a user interface in a full implementation
        # For now, return the top candidate with human review required
        
        top_candidate = candidates[0] if candidates else None
        confidence_level = self._determine_confidence_level(
            top_candidate.overall_score if top_candidate else 0.0
        )
        
        return SelectionResult(
            selected_verse=top_candidate.verse if top_candidate else None,
            all_candidates=candidates,
            selection_method=SelectionStrategy.INTERACTIVE,
            confidence_level=confidence_level,
            requires_human_review=True,
            selection_metadata={
                'selection_reason': 'Interactive selection required',
                'prompt_generated': True
            }
        )
    
    def _hybrid_selection(self, text: str, candidates: List[VerseCandidateScore],
                         context: Dict) -> SelectionResult:
        """Perform hybrid selection (automatic + interactive)."""
        if not candidates:
            return SelectionResult(
                selected_verse=None,
                all_candidates=[],
                selection_method=SelectionStrategy.HYBRID,
                confidence_level=SelectionConfidence.VERY_LOW,
                requires_human_review=True,
                selection_metadata={'reason': 'No candidates available'}
            )
        
        top_candidate = candidates[0]
        confidence_level = self._determine_confidence_level(top_candidate.overall_score)
        
        # Use automatic selection for high confidence
        if top_candidate.overall_score >= self.auto_select_threshold:
            return self._automatic_selection(candidates)
        
        # Use interactive selection for low confidence
        elif top_candidate.overall_score < self.human_review_threshold:
            return self._interactive_selection(text, candidates, context)
        
        # Medium confidence - provide recommendation but require review
        else:
            return SelectionResult(
                selected_verse=top_candidate.verse,
                all_candidates=candidates,
                selection_method=SelectionStrategy.HYBRID,
                confidence_level=confidence_level,
                requires_human_review=True,
                selection_metadata={
                    'selection_reason': 'Recommended selection - review advised',
                    'score': top_candidate.overall_score,
                    'threshold_status': 'medium_confidence'
                }
            )
    
    def generate_interactive_prompt(self, text: str, candidates: List[VerseCandidateScore],
                                  context: Dict = None) -> InteractivePrompt:
        """
        Generate an interactive prompt for verse selection.
        
        Args:
            text: Original text
            candidates: Candidate verses
            context: Additional context
            
        Returns:
            Interactive prompt for user selection
        """
        context = context or {}
        
        # Create selection options
        options = []
        for i, candidate in enumerate(candidates[:5]):  # Top 5 candidates
            option = {
                'index': i + 1,
                'verse': candidate.verse,
                'score': candidate.overall_score,
                'reasons': candidate.selection_reasons,
                'preview': candidate.verse.canonical_text[:100] + '...' if len(candidate.verse.canonical_text) > 100 else candidate.verse.canonical_text
            }
            options.append(option)
        
        # Add "none of the above" option
        options.append({
            'index': len(options) + 1,
            'verse': None,
            'score': 0.0,
            'reasons': ['None of the suggested verses match'],
            'preview': 'Skip verse substitution'
        })
        
        prompt_message = f"""
Original text: "{text}"

Please select the best matching canonical verse:
"""
        
        return InteractivePrompt(
            original_text=text,
            candidates=candidates,
            context_info=context,
            prompt_message=prompt_message,
            selection_options=options
        )
    
    def add_source_attribution(self, verse: CanonicalVerse, format_style: str = "academic") -> str:
        """
        Add source attribution to a verse.
        
        Args:
            verse: Verse to attribute
            format_style: Attribution format style
            
        Returns:
            Formatted attribution string
        """
        if format_style == "academic":
            return f"{verse.source.value.replace('_', ' ').title()} {verse.chapter}.{verse.verse}"
        elif format_style == "traditional":
            return f"({verse.source.value.replace('_', ' ').title()} {verse.chapter}:{verse.verse})"
        elif format_style == "scholarly":
            return f"[{verse.source.value.replace('_', ' ').title()} {verse.chapter}.{verse.verse} - {verse.source_authority}]"
        else:
            return f"{verse.source.value} {verse.chapter}.{verse.verse}"