"""
Enhanced Lexicon Manager for Story 4.2 Sanskrit Processing Enhancement

Extends the existing Story 2.1 lexicon management with ML-enhanced capabilities,
dynamic expansion, and research-grade quality validation.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
import logging
import json
import yaml
from datetime import datetime
from pathlib import Path
import asyncio

from .lexicon_manager import LexiconManager, LexiconEntry, LexiconValidationResult, WordCategory
try:
    from ..utils.mcp_transformer_client import MCPTransformerClient, SemanticClassification, CulturalContext
except ImportError:
    # Handle relative import issues during development
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.mcp_transformer_client import MCPTransformerClient, SemanticClassification, CulturalContext


class MLConfidenceLevel(Enum):
    """ML-based confidence levels for lexicon entries."""
    VERY_HIGH = "very_high"  # >0.95
    HIGH = "high"            # >0.85
    MEDIUM = "medium"        # >0.7
    LOW = "low"              # >0.5
    VERY_LOW = "very_low"    # <=0.5


class QualityValidationStatus(Enum):
    """Quality validation status for ML-suggested entries."""
    VALIDATED = "validated"
    PENDING_REVIEW = "pending_review"
    REQUIRES_ACADEMIC_REVIEW = "requires_academic_review"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


@dataclass
class MLEnhancedEntry:
    """Enhanced lexicon entry with ML-based metadata."""
    base_entry: LexiconEntry
    ml_confidence_score: float
    semantic_classification: Optional[SemanticClassification]
    suggested_by_ml: bool
    quality_validation_status: QualityValidationStatus
    academic_review_notes: List[str]
    ml_feedback_history: List[Dict[str, Any]]
    last_ml_update: str
    validation_metadata: Dict[str, Any]


@dataclass
class DynamicExpansionResult:
    """Result from dynamic lexicon expansion."""
    new_entries_suggested: List[MLEnhancedEntry]
    confidence_threshold_met: int
    requires_academic_review: List[MLEnhancedEntry]
    auto_approved: List[MLEnhancedEntry]
    processing_time_ms: float


@dataclass
class QualityValidationMetrics:
    """Quality validation metrics for lexicon entries."""
    total_entries: int
    ml_validated: int
    academically_reviewed: int
    pending_review: int
    auto_approved: int
    rejection_rate: float
    avg_confidence_score: float


class EnhancedLexiconManager:
    """
    Enhanced Lexicon Manager with ML capabilities.
    
    Extends the existing Story 2.1 LexiconManager with:
    - ML confidence scoring and feedback loops
    - Dynamic lexicon expansion with quality validation
    - Academic review workflow
    - Research-grade metrics and reporting
    """

    def __init__(
        self, 
        base_lexicon_manager: Optional[LexiconManager] = None,
        transformer_client: Optional[MCPTransformerClient] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize the enhanced lexicon manager.
        
        Args:
            base_lexicon_manager: Existing lexicon manager from Story 2.1
            transformer_client: MCP transformer client for ML processing
            config: Configuration parameters
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize base components
        self.base_manager = base_lexicon_manager or LexiconManager()
        self.transformer_client = transformer_client
        self.config = config or self._get_default_config()
        
        # Enhanced storage for ML metadata
        self.ml_enhanced_entries: Dict[str, MLEnhancedEntry] = {}
        self.expansion_history: List[DynamicExpansionResult] = []
        self.quality_metrics = QualityValidationMetrics(
            total_entries=0,
            ml_validated=0,
            academically_reviewed=0,
            pending_review=0,
            auto_approved=0,
            rejection_rate=0.0,
            avg_confidence_score=0.0
        )
        
        # Initialize ML enhancements
        self._initialize_ml_enhancements()

    def _get_default_config(self) -> Dict:
        """Get default configuration for enhanced lexicon manager."""
        return {
            'ml_confidence_threshold': 0.7,
            'auto_approval_threshold': 0.9,
            'academic_review_threshold': 0.6,
            'max_expansion_batch_size': 50,
            'enable_dynamic_expansion': True,
            'enable_quality_validation': True,
            'quality_validation_strictness': 'medium',  # low, medium, high
            'academic_review_required_for_new_terms': True,
            'ml_feedback_retention_days': 90
        }

    def _initialize_ml_enhancements(self):
        """Initialize ML enhancements for existing lexicon entries."""
        try:
            self.logger.info("Initializing ML enhancements for existing lexicon")
            
            # Load existing entries and enhance them with ML metadata
            base_entries = self.base_manager.get_all_entries()
            
            for term, entry in base_entries.items():
                enhanced_entry = self._create_enhanced_entry_from_base(entry)
                self.ml_enhanced_entries[term] = enhanced_entry
            
            self.logger.info(f"Enhanced {len(self.ml_enhanced_entries)} existing lexicon entries with ML metadata")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ML enhancements: {e}")

    def _create_enhanced_entry_from_base(self, base_entry: LexiconEntry) -> MLEnhancedEntry:
        """Create enhanced entry from existing base entry."""
        return MLEnhancedEntry(
            base_entry=base_entry,
            ml_confidence_score=base_entry.confidence,  # Use existing confidence as baseline
            semantic_classification=None,  # Will be populated on demand
            suggested_by_ml=False,  # Existing entries not ML-suggested
            quality_validation_status=QualityValidationStatus.VALIDATED,  # Assume existing are validated
            academic_review_notes=[],
            ml_feedback_history=[],
            last_ml_update=datetime.now().isoformat(),
            validation_metadata={}
        )

    async def enhance_entry_with_ml_classification(self, term: str) -> bool:
        """
        Enhance an existing entry with ML-based semantic classification.
        
        Args:
            term: Term to enhance
            
        Returns:
            True if enhancement was successful
        """
        if term not in self.ml_enhanced_entries:
            self.logger.warning(f"Term '{term}' not found in enhanced lexicon")
            return False
        
        if not self.transformer_client:
            self.logger.warning("No transformer client available for ML enhancement")
            return False
        
        try:
            enhanced_entry = self.ml_enhanced_entries[term]
            
            # Get semantic classification from transformer
            classification = self.transformer_client.classify_term_semantically(
                term, 
                context=enhanced_entry.base_entry.category
            )
            
            # Update the enhanced entry
            enhanced_entry.semantic_classification = classification
            enhanced_entry.ml_confidence_score = max(
                enhanced_entry.ml_confidence_score,
                classification.semantic_confidence
            )
            enhanced_entry.last_ml_update = datetime.now().isoformat()
            
            # Add feedback history
            enhanced_entry.ml_feedback_history.append({
                'timestamp': datetime.now().isoformat(),
                'operation': 'semantic_classification',
                'confidence_score': classification.semantic_confidence,
                'cultural_context': classification.cultural_context.value
            })
            
            self.logger.debug(f"Enhanced '{term}' with ML classification: {classification.primary_category}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to enhance entry '{term}' with ML: {e}")
            return False

    async def suggest_lexicon_expansions(
        self, 
        source_text: str,
        max_suggestions: int = 10
    ) -> DynamicExpansionResult:
        """
        Suggest new lexicon entries based on source text analysis.
        
        Args:
            source_text: Text to analyze for potential new entries
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            DynamicExpansionResult with suggested expansions
        """
        start_time = datetime.now()
        
        try:
            if not self.transformer_client:
                self.logger.warning("No transformer client available for expansion suggestions")
                return DynamicExpansionResult([], 0, [], [], 0.0)
            
            # Process text with transformer to identify potential Sanskrit/Hindi terms
            result = await self.transformer_client.process_sanskrit_text_with_context(source_text)
            
            # Extract potential terms (this is a simplified implementation)
            potential_terms = self._extract_potential_terms(source_text, result)
            
            suggested_entries = []
            requires_review = []
            auto_approved = []
            
            for term_info in potential_terms[:max_suggestions]:
                term = term_info['term']
                
                # Skip if already in lexicon
                if term.lower() in self.ml_enhanced_entries:
                    continue
                
                # Create ML-suggested entry
                enhanced_entry = await self._create_ml_suggested_entry(term, term_info, result)
                
                if enhanced_entry.ml_confidence_score >= self.config['auto_approval_threshold']:
                    auto_approved.append(enhanced_entry)
                    suggested_entries.append(enhanced_entry)
                elif enhanced_entry.ml_confidence_score >= self.config['academic_review_threshold']:
                    requires_review.append(enhanced_entry)
                    suggested_entries.append(enhanced_entry)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            expansion_result = DynamicExpansionResult(
                new_entries_suggested=suggested_entries,
                confidence_threshold_met=len([e for e in suggested_entries if e.ml_confidence_score >= self.config['ml_confidence_threshold']]),
                requires_academic_review=requires_review,
                auto_approved=auto_approved,
                processing_time_ms=processing_time
            )
            
            # Store in expansion history
            self.expansion_history.append(expansion_result)
            
            return expansion_result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.logger.error(f"Failed to suggest lexicon expansions: {e}")
            
            return DynamicExpansionResult([], 0, [], [], processing_time)

    def _extract_potential_terms(self, source_text: str, transformer_result: Any) -> List[Dict[str, Any]]:
        """Extract potential Sanskrit/Hindi terms from source text."""
        # Simplified implementation - would use more sophisticated NLP in production
        words = source_text.split()
        potential_terms = []
        
        # Look for capitalized words that might be Sanskrit/Hindi proper nouns
        sanskrit_indicators = ['krishna', 'dharma', 'yoga', 'karma', 'moksha', 'samsara', 'gita', 'veda']
        
        for word in words:
            word_clean = word.lower().strip('.,!?;:')
            
            # Check if word might be Sanskrit/Hindi
            if (any(indicator in word_clean for indicator in sanskrit_indicators) or
                word_clean.endswith('a') or word_clean.endswith('am') or
                len(word_clean) > 4 and word.istitle()):
                
                potential_terms.append({
                    'term': word_clean,
                    'original_form': word,
                    'context_confidence': transformer_result.confidence_score if transformer_result else 0.5,
                    'cultural_context': transformer_result.semantic_context.value if transformer_result else 'unknown'
                })
        
        return potential_terms

    async def _create_ml_suggested_entry(
        self, 
        term: str, 
        term_info: Dict[str, Any],
        transformer_result: Any
    ) -> MLEnhancedEntry:
        """Create ML-suggested lexicon entry."""
        try:
            # Get semantic classification for the term
            classification = self.transformer_client.classify_term_semantically(term)
            
            # Create base lexicon entry
            base_entry = LexiconEntry(
                original_term=term,
                variations=[term_info.get('original_form', term)],
                transliteration=classification.suggested_transliteration or term,
                is_proper_noun=classification.cultural_context != CulturalContext.UNKNOWN,
                category=classification.primary_category,
                confidence=classification.semantic_confidence,
                source_authority="ml_suggestion"
            )
            
            # Determine quality validation status
            if classification.semantic_confidence >= self.config['auto_approval_threshold']:
                validation_status = QualityValidationStatus.VALIDATED
            elif classification.semantic_confidence >= self.config['academic_review_threshold']:
                validation_status = QualityValidationStatus.REQUIRES_ACADEMIC_REVIEW
            else:
                validation_status = QualityValidationStatus.PENDING_REVIEW
            
            # Create enhanced entry
            enhanced_entry = MLEnhancedEntry(
                base_entry=base_entry,
                ml_confidence_score=classification.semantic_confidence,
                semantic_classification=classification,
                suggested_by_ml=True,
                quality_validation_status=validation_status,
                academic_review_notes=[],
                ml_feedback_history=[{
                    'timestamp': datetime.now().isoformat(),
                    'operation': 'initial_suggestion',
                    'confidence_score': classification.semantic_confidence,
                    'cultural_context': classification.cultural_context.value
                }],
                last_ml_update=datetime.now().isoformat(),
                validation_metadata={
                    'source_context': term_info.get('cultural_context', 'unknown'),
                    'transformer_confidence': transformer_result.confidence_score if transformer_result else 0.0
                }
            )
            
            return enhanced_entry
            
        except Exception as e:
            self.logger.error(f"Failed to create ML-suggested entry for '{term}': {e}")
            
            # Return minimal entry on failure
            base_entry = LexiconEntry(
                original_term=term,
                variations=[term],
                transliteration=term,
                is_proper_noun=False,
                category="unknown",
                confidence=0.1,
                source_authority="ml_suggestion_failed"
            )
            
            return MLEnhancedEntry(
                base_entry=base_entry,
                ml_confidence_score=0.1,
                semantic_classification=None,
                suggested_by_ml=True,
                quality_validation_status=QualityValidationStatus.REJECTED,
                academic_review_notes=["Failed to generate ML classification"],
                ml_feedback_history=[],
                last_ml_update=datetime.now().isoformat(),
                validation_metadata={}
            )

    def approve_ml_suggestion(self, term: str, academic_notes: Optional[List[str]] = None) -> bool:
        """
        Approve an ML-suggested entry and add it to the active lexicon.
        
        Args:
            term: Term to approve
            academic_notes: Optional academic review notes
            
        Returns:
            True if approval was successful
        """
        try:
            if term not in self.ml_enhanced_entries:
                self.logger.error(f"Term '{term}' not found in enhanced lexicon")
                return False
            
            enhanced_entry = self.ml_enhanced_entries[term]
            
            # Update validation status
            enhanced_entry.quality_validation_status = QualityValidationStatus.VALIDATED
            
            # Add academic notes if provided
            if academic_notes:
                enhanced_entry.academic_review_notes.extend(academic_notes)
            
            # Add to base lexicon manager
            success = self.base_manager.add_entry('ml_suggestions.yaml', enhanced_entry.base_entry)
            
            if success:
                self.logger.info(f"Approved and added ML-suggested term '{term}' to lexicon")
                return True
            else:
                self.logger.error(f"Failed to add approved term '{term}' to base lexicon")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to approve ML suggestion for '{term}': {e}")
            return False

    def reject_ml_suggestion(self, term: str, reason: str) -> bool:
        """
        Reject an ML-suggested entry.
        
        Args:
            term: Term to reject
            reason: Reason for rejection
            
        Returns:
            True if rejection was successful
        """
        try:
            if term not in self.ml_enhanced_entries:
                self.logger.error(f"Term '{term}' not found in enhanced lexicon")
                return False
            
            enhanced_entry = self.ml_enhanced_entries[term]
            enhanced_entry.quality_validation_status = QualityValidationStatus.REJECTED
            enhanced_entry.academic_review_notes.append(f"Rejected: {reason}")
            
            self.logger.info(f"Rejected ML-suggested term '{term}': {reason}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to reject ML suggestion for '{term}': {e}")
            return False

    def get_enhanced_entries_by_status(self, status: QualityValidationStatus) -> List[MLEnhancedEntry]:
        """Get enhanced entries by validation status."""
        return [
            entry for entry in self.ml_enhanced_entries.values()
            if entry.quality_validation_status == status
        ]

    def get_ml_confidence_distribution(self) -> Dict[MLConfidenceLevel, int]:
        """Get distribution of ML confidence levels across entries."""
        distribution = {level: 0 for level in MLConfidenceLevel}
        
        for entry in self.ml_enhanced_entries.values():
            confidence = entry.ml_confidence_score
            
            if confidence > 0.95:
                distribution[MLConfidenceLevel.VERY_HIGH] += 1
            elif confidence > 0.85:
                distribution[MLConfidenceLevel.HIGH] += 1
            elif confidence > 0.7:
                distribution[MLConfidenceLevel.MEDIUM] += 1
            elif confidence > 0.5:
                distribution[MLConfidenceLevel.LOW] += 1
            else:
                distribution[MLConfidenceLevel.VERY_LOW] += 1
        
        return distribution

    def generate_quality_metrics_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality metrics report."""
        try:
            total_entries = len(self.ml_enhanced_entries)
            
            if total_entries == 0:
                return {"error": "No enhanced entries available"}
            
            # Count by validation status
            status_counts = {}
            for status in QualityValidationStatus:
                status_counts[status.value] = len(self.get_enhanced_entries_by_status(status))
            
            # Calculate metrics
            ml_validated = status_counts.get('validated', 0)
            pending = status_counts.get('pending_review', 0)
            academic_review = status_counts.get('requires_academic_review', 0)
            rejected = status_counts.get('rejected', 0)
            
            # Average confidence
            total_confidence = sum(entry.ml_confidence_score for entry in self.ml_enhanced_entries.values())
            avg_confidence = total_confidence / total_entries if total_entries > 0 else 0.0
            
            # Rejection rate
            rejection_rate = rejected / total_entries if total_entries > 0 else 0.0
            
            # Update quality metrics
            self.quality_metrics = QualityValidationMetrics(
                total_entries=total_entries,
                ml_validated=ml_validated,
                academically_reviewed=academic_review,
                pending_review=pending,
                auto_approved=status_counts.get('validated', 0),  # Simplification
                rejection_rate=rejection_rate,
                avg_confidence_score=avg_confidence
            )
            
            return {
                'quality_metrics': asdict(self.quality_metrics),
                'confidence_distribution': {k.value: v for k, v in self.get_ml_confidence_distribution().items()},
                'expansion_history_count': len(self.expansion_history),
                'ml_suggestions_count': sum(1 for e in self.ml_enhanced_entries.values() if e.suggested_by_ml),
                'last_update': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate quality metrics report: {e}")
            return {"error": str(e)}

    def get_academic_review_queue(self) -> List[Dict[str, Any]]:
        """Get queue of entries requiring academic review."""
        review_entries = self.get_enhanced_entries_by_status(QualityValidationStatus.REQUIRES_ACADEMIC_REVIEW)
        
        queue = []
        for entry in review_entries:
            queue.append({
                'term': entry.base_entry.original_term,
                'transliteration': entry.base_entry.transliteration,
                'category': entry.base_entry.category,
                'ml_confidence_score': entry.ml_confidence_score,
                'cultural_context': entry.semantic_classification.cultural_context.value if entry.semantic_classification else 'unknown',
                'suggested_by_ml': entry.suggested_by_ml,
                'last_ml_update': entry.last_ml_update,
                'validation_metadata': entry.validation_metadata
            })
        
        # Sort by confidence score (highest first)
        queue.sort(key=lambda x: x['ml_confidence_score'], reverse=True)
        
        return queue

    def save_enhanced_lexicon_state(self, file_path: Optional[Path] = None) -> bool:
        """Save the enhanced lexicon state to file."""
        try:
            save_path = file_path or Path("data/lexicons/enhanced_lexicon_state.json")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare data for serialization
            state_data = {
                'enhanced_entries': {},
                'quality_metrics': asdict(self.quality_metrics),
                'config': self.config,
                'last_saved': datetime.now().isoformat()
            }
            
            # Serialize enhanced entries
            for term, entry in self.ml_enhanced_entries.items():
                entry_data = asdict(entry)
                # Handle enum serialization
                if entry.quality_validation_status:
                    entry_data['quality_validation_status'] = entry.quality_validation_status.value
                if entry.semantic_classification and entry.semantic_classification.cultural_context:
                    entry_data['semantic_classification']['cultural_context'] = entry.semantic_classification.cultural_context.value
                
                state_data['enhanced_entries'][term] = entry_data
            
            # Save to file
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved enhanced lexicon state to {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save enhanced lexicon state: {e}")
            return False

    # Maintain compatibility with base lexicon manager
    def get_all_entries(self) -> Dict[str, LexiconEntry]:
        """Get all entries (maintains compatibility with Story 2.1)."""
        return self.base_manager.get_all_entries()

    def get_entries_by_category(self, category: Union[str, WordCategory]) -> Dict[str, LexiconEntry]:
        """Get entries by category (maintains compatibility with Story 2.1)."""
        return self.base_manager.get_entries_by_category(category)

    def search_entries(self, query: str, max_results: int = 10) -> List[LexiconEntry]:
        """Search entries (maintains compatibility with Story 2.1)."""
        return self.base_manager.search_entries(query, max_results)