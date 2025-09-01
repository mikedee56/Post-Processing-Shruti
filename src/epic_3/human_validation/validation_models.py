"""
Data Models for Expert Validation - Story 3.3.1

This module defines the data structures for capturing expert decisions,
validation cases, and learning patterns in the knowledge capture system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import json
import uuid


class ValidationStatus(Enum):
    """Status of a validation case"""
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"
    EXPERT_ESCALATION = "expert_escalation"


class DecisionType(Enum):
    """Type of expert decision"""
    LEXICON_CORRECTION = "lexicon_correction"
    TRANSLITERATION_FIX = "transliteration_fix"
    CAPITALIZATION_CHANGE = "capitalization_change"
    SCRIPTURAL_IDENTIFICATION = "scriptural_identification"
    SEMANTIC_DISAMBIGUATION = "semantic_disambiguation"
    CONTEXTUAL_CORRECTION = "contextual_correction"
    FALSE_POSITIVE = "false_positive"
    MANUAL_OVERRIDE = "manual_override"


class ConfidenceLevel(Enum):
    """Confidence level for pattern application"""
    LOW = "low"          # <60% confidence
    MEDIUM = "medium"    # 60-80% confidence
    HIGH = "high"        # 80-95% confidence
    VERY_HIGH = "very_high"  # >95% confidence


class ExpertRecommendationType(Enum):
    """Type of expert recommendation for lexicon updates"""
    ACCEPT_AS_IS = "accept_as_is"
    ACCEPT_WITH_MODIFICATION = "accept_with_modification"
    REJECT = "reject"
    NEEDS_MORE_CONTEXT = "needs_more_context"


@dataclass
class ValidationCase:
    """
    Represents a case requiring expert validation
    
    This captures all the information needed for an expert to make
    a decision about a processing result.
    """
    case_id: str
    original_text: str
    processed_text: str
    processing_context: Dict[str, Any]
    flagging_reasons: List[str]
    confidence_scores: Dict[str, float]
    timestamp: datetime
    status: ValidationStatus = ValidationStatus.PENDING
    assigned_expert: Optional[str] = None
    priority_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert validation case to dictionary for storage"""
        return {
            'case_id': self.case_id,
            'original_text': self.original_text,
            'processed_text': self.processed_text,
            'processing_context': self.processing_context,
            'flagging_reasons': self.flagging_reasons,
            'confidence_scores': self.confidence_scores,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value,
            'assigned_expert': self.assigned_expert,
            'priority_score': self.priority_score,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationCase':
        """Create validation case from dictionary"""
        return cls(
            case_id=data['case_id'],
            original_text=data['original_text'],
            processed_text=data['processed_text'],
            processing_context=data['processing_context'],
            flagging_reasons=data['flagging_reasons'],
            confidence_scores=data['confidence_scores'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            status=ValidationStatus(data['status']),
            assigned_expert=data.get('assigned_expert'),
            priority_score=data.get('priority_score', 0.0),
            metadata=data.get('metadata', {})
        )


@dataclass
class ExpertDecision:
    """
    Captures an expert's decision on a validation case
    
    This includes the decision itself, reasoning, and any patterns
    that can be extracted for future learning.
    """
    case_id: str
    decision_type: DecisionType
    original_text: str
    corrected_text: str
    expert_id: str
    confidence_score: float
    reasoning: str
    created_at: datetime
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    approved_text: str = field(default="")
    confidence: ConfidenceLevel = field(default=ConfidenceLevel.MEDIUM)
    decision_timestamp: datetime = field(default_factory=datetime.now)
    processing_time_seconds: float = field(default=0.0)
    tags: List[str] = field(default_factory=list)
    pattern_hints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set derived fields after initialization"""
        import uuid
        if not self.approved_text:
            self.approved_text = self.corrected_text
        if not self.decision_timestamp or self.decision_timestamp == datetime.now():
            self.decision_timestamp = self.created_at
        # Convert confidence_score to ConfidenceLevel if needed
        if isinstance(self.confidence_score, float):
            if self.confidence_score >= 0.95:
                self.confidence = ConfidenceLevel.VERY_HIGH
            elif self.confidence_score >= 0.80:
                self.confidence = ConfidenceLevel.HIGH
            elif self.confidence_score >= 0.60:
                self.confidence = ConfidenceLevel.MEDIUM
            else:
                self.confidence = ConfidenceLevel.LOW
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert expert decision to dictionary for storage"""
        return {
            'decision_id': self.decision_id,
            'case_id': self.case_id,
            'expert_id': self.expert_id,
            'decision_type': self.decision_type.value,
            'approved_text': self.approved_text,
            'reasoning': self.reasoning,
            'confidence': self.confidence.value,
            'decision_timestamp': self.decision_timestamp.isoformat(),
            'processing_time_seconds': self.processing_time_seconds,
            'tags': self.tags,
            'pattern_hints': self.pattern_hints,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExpertDecision':
        """Create expert decision from dictionary"""
        return cls(
            decision_id=data['decision_id'],
            case_id=data['case_id'],
            expert_id=data['expert_id'],
            decision_type=DecisionType(data['decision_type']),
            approved_text=data['approved_text'],
            reasoning=data['reasoning'],
            confidence=ConfidenceLevel(data['confidence']),
            decision_timestamp=datetime.fromisoformat(data['decision_timestamp']),
            processing_time_seconds=data['processing_time_seconds'],
            tags=data.get('tags', []),
            pattern_hints=data.get('pattern_hints', {}),
            metadata=data.get('metadata', {})
        )


@dataclass
class LearningPattern:
    """
    Represents a pattern learned from expert decisions
    
    These patterns can be automatically applied to future processing
    to reduce the need for expert review.
    """
    pattern_id: str
    pattern_type: DecisionType
    condition: Dict[str, Any]  # Conditions under which pattern applies
    action: Dict[str, Any]     # What transformation to apply
    confidence_score: float
    support_count: int         # Number of expert decisions supporting pattern
    success_rate: float        # Success rate when applied
    created_timestamp: datetime
    last_updated: datetime
    version: int = 1
    is_active: bool = True
    expert_feedback: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert learning pattern to dictionary for storage"""
        return {
            'pattern_id': self.pattern_id,
            'pattern_type': self.pattern_type.value,
            'condition': self.condition,
            'action': self.action,
            'confidence_score': self.confidence_score,
            'support_count': self.support_count,
            'success_rate': self.success_rate,
            'created_timestamp': self.created_timestamp.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'version': self.version,
            'is_active': self.is_active,
            'expert_feedback': self.expert_feedback
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningPattern':
        """Create learning pattern from dictionary"""
        return cls(
            pattern_id=data['pattern_id'],
            pattern_type=DecisionType(data['pattern_type']),
            condition=data['condition'],
            action=data['action'],
            confidence_score=data['confidence_score'],
            support_count=data['support_count'],
            success_rate=data['success_rate'],
            created_timestamp=datetime.fromisoformat(data['created_timestamp']),
            last_updated=datetime.fromisoformat(data['last_updated']),
            version=data.get('version', 1),
            is_active=data.get('is_active', True),
            expert_feedback=data.get('expert_feedback', [])
        )
    
    def matches_context(self, context: Dict[str, Any]) -> bool:
        """Check if this pattern matches the given processing context"""
        try:
            # Simple pattern matching - can be enhanced with ML later
            for key, expected_value in self.condition.items():
                if key not in context:
                    return False
                
                actual_value = context[key]
                
                # Handle different types of pattern matching
                if isinstance(expected_value, dict):
                    if 'contains' in expected_value:
                        if expected_value['contains'].lower() not in str(actual_value).lower():
                            return False
                    elif 'regex' in expected_value:
                        import re
                        if not re.search(expected_value['regex'], str(actual_value)):
                            return False
                    elif 'range' in expected_value:
                        min_val, max_val = expected_value['range']
                        if not (min_val <= actual_value <= max_val):
                            return False
                elif isinstance(expected_value, list):
                    if actual_value not in expected_value:
                        return False
                else:
                    if actual_value != expected_value:
                        return False
            
            return True
        except Exception:
            return False


@dataclass
class KnowledgeCaptureMetrics:
    """Metrics for tracking knowledge capture effectiveness"""
    total_decisions_captured: int = 0
    patterns_identified: int = 0
    patterns_applied_successfully: int = 0
    expert_review_load_reduction: float = 0.0
    average_decision_confidence: float = 0.0
    processing_accuracy_improvement: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'total_decisions_captured': self.total_decisions_captured,
            'patterns_identified': self.patterns_identified,
            'patterns_applied_successfully': self.patterns_applied_successfully,
            'expert_review_load_reduction': self.expert_review_load_reduction,
            'average_decision_confidence': self.average_decision_confidence,
            'processing_accuracy_improvement': self.processing_accuracy_improvement,
            'last_updated': self.last_updated.isoformat()
        }