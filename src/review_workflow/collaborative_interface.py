"""
Collaborative Review Interface with Epic 4.5 Research-Grade Standards.

Implements Story 3.3 Task 1: Epic 4.5 Academic Consultant Integration
- Google Docs-style collaborative commenting system
- Research-grade review standards with real-time collaboration
- Academic citation integration for contextual review information
- Publication formatter output standards for review interface
"""

import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple, Union

# Epic 4 infrastructure imports
from scripture_processing.publication_formatter import PublicationFormatter
from scripture_processing.academic_citation_manager import AcademicCitationManager
from utils.performance_monitor import PerformanceMonitor
from monitoring.system_monitor import SystemMonitor
from utils.srt_parser import SRTSegment


class CommentType(Enum):
    """Types of review comments."""
    CORRECTION = "correction"
    SUGGESTION = "suggestion"
    QUESTION = "question"
    APPROVAL = "approval"
    REJECTION = "rejection"
    CITATION_REFERENCE = "citation_reference"
    ACADEMIC_NOTE = "academic_note"


class CommentStatus(Enum):
    """Comment resolution status."""
    OPEN = "open"
    ADDRESSED = "addressed"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class ReviewActionType(Enum):
    """Types of review actions."""
    TEXT_EDIT = "text_edit"
    FORMATTING_CHANGE = "formatting_change"
    CITATION_ADD = "citation_add"
    ACADEMIC_VALIDATION = "academic_validation"
    ESCALATION_REQUEST = "escalation_request"
    APPROVAL = "approval"


@dataclass
class ReviewComment:
    """Individual review comment with Epic 4.5 academic standards."""
    comment_id: str
    session_id: str
    segment_id: str
    comment_type: CommentType
    comment_text: str
    author_id: str
    author_role: str
    
    # Optional fields with defaults
    suggested_text: Optional[str] = None
    academic_citation: Optional[str] = None
    iast_compliance_note: Optional[str] = None
    publication_standard_reference: Optional[str] = None
    reply_to_comment_id: Optional[str] = None
    thread_depth: int = 0
    status: CommentStatus = CommentStatus.OPEN
    resolved_by: Optional[str] = None
    resolution_note: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    text_selection_start: int = 0
    text_selection_end: int = 0
    highlighted_text: str = ""
    requires_expert_review: bool = False
    consultant_validated: bool = False
    peer_review_relevant: bool = False


@dataclass
class ReviewAction:
    """Review action with Epic 4.5 academic tracking."""
    action_id: str
    session_id: str
    segment_id: str
    action_type: ReviewActionType
    description: str
    performed_by: str
    reviewer_role: str
    
    # Optional fields with defaults
    original_text: str = ""
    modified_text: str = ""
    approved_by: Optional[str] = None
    academic_justification: Optional[str] = None
    citation_support: List[str] = field(default_factory=list)
    iast_compliance_verified: bool = False
    publication_standard_met: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    approved_at: Optional[datetime] = None
    quality_impact_score: float = 0.0
    academic_impact_score: float = 0.0
    confidence_level: float = 1.0


@dataclass
class CollaborativeSession:
    """Real-time collaborative review session."""
    session_id: str
    active_reviewers: Set[str] = field(default_factory=set)
    
    # Real-time collaboration state
    concurrent_edits: Dict[str, Dict] = field(default_factory=dict)  # segment_id -> edit info
    edit_conflicts: List[Dict] = field(default_factory=list)
    last_activity: datetime = field(default_factory=datetime.now)
    
    # Comment threading
    comments: List[ReviewComment] = field(default_factory=list)
    comment_threads: Dict[str, List[str]] = field(default_factory=dict)  # comment_id -> reply_ids
    
    # Action tracking
    actions_history: List[ReviewAction] = field(default_factory=list)
    pending_approvals: List[str] = field(default_factory=list)  # action_ids
    
    # Academic collaboration (Epic 4.5)
    academic_discussion_active: bool = False
    citation_review_mode: bool = False
    consultant_participating: bool = False


class CollaborativeInterface:
    """
    Epic 4.5 Research-Grade Collaborative Review Interface.
    
    Implements Story 3.3 Task 1:
    - Google Docs-style collaborative commenting with academic standards
    - Research-grade review standards with citation integration
    - Academic context preservation and validation
    - Publication formatter integration for output standards
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize collaborative interface with Epic 4.5 academic standards."""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Epic 4.5 Academic Components
        self.publication_formatter = PublicationFormatter(
            self.config.get('publication_formatter', {})
        )
        self.citation_manager = AcademicCitationManager(
            self.config.get('citation_manager', {})
        )
        
        # Epic 4.3 Production Infrastructure
        self.system_monitor = SystemMonitor(self.config.get('monitoring', {}))
        self.performance_monitor = PerformanceMonitor(self.config.get('performance', {}))
        
        # Collaborative interface configuration
        self.interface_config = self.config.get('interface', {
            'max_concurrent_users': 10,
            'comment_thread_max_depth': 5,
            'auto_save_interval_seconds': 30,
            'conflict_resolution_timeout_seconds': 120,
            'academic_validation_required': True,
            'citation_suggestions_enabled': True
        })
        
        # Data structures for collaboration
        self.active_sessions: Dict[str, CollaborativeSession] = {}
        self.user_presence: Dict[str, Dict] = {}  # user_id -> presence info
        
        # Threading for real-time features
        self.lock = threading.RLock()
        
        # Performance tracking
        self.interface_statistics = defaultdict(int)
        
        self.logger.info("CollaborativeInterface initialized with Epic 4.5 academic integration")
    
    def create_collaborative_session(self, session_id: str) -> bool:
        """
        Create new collaborative review session.
        
        Args:
            session_id: Review session identifier
            
        Returns:
            bool: True if session created successfully
        """
        with self.lock:
            if session_id in self.active_sessions:
                self.logger.warning(f"Collaborative session already exists: {session_id}")
                return False
            
            self.active_sessions[session_id] = CollaborativeSession(session_id=session_id)
            self.interface_statistics['sessions_created'] += 1
            
            self.logger.info(f"Collaborative session created: {session_id}")
            return True
    
    def join_session(self, session_id: str, user_id: str, user_role: str) -> bool:
        """
        Add user to collaborative review session.
        
        Args:
            session_id: Review session identifier
            user_id: User identifier
            user_role: User role (gp, sme, consultant)
            
        Returns:
            bool: True if user joined successfully
        """
        with self.lock:
            if session_id not in self.active_sessions:
                self.logger.error(f"Collaborative session not found: {session_id}")
                return False
            
            session = self.active_sessions[session_id]
            
            # Check concurrent user limit
            if len(session.active_reviewers) >= self.interface_config['max_concurrent_users']:
                self.logger.warning(f"Session {session_id} at maximum capacity")
                return False
            
            # Add user to session
            session.active_reviewers.add(user_id)
            session.last_activity = datetime.now()
            
            # Track user presence
            self.user_presence[user_id] = {
                'session_id': session_id,
                'role': user_role,
                'joined_at': datetime.now(),
                'last_activity': datetime.now(),
                'active_segment': None
            }
            
            # Enable academic features for consultants
            if user_role == 'consultant':
                session.consultant_participating = True
                session.citation_review_mode = True
            
            self.interface_statistics['user_joins'] += 1
            
            self.logger.info(f"User {user_id} ({user_role}) joined session {session_id}")
            return True
    
    def add_comment(self, 
                   session_id: str,
                   segment_id: str,
                   author_id: str,
                   comment_text: str,
                   comment_type: CommentType = CommentType.SUGGESTION,
                   text_selection_start: int = 0,
                   text_selection_end: int = 0,
                   highlighted_text: str = "",
                   reply_to_comment_id: Optional[str] = None) -> Optional[str]:
        """
        Add comment to review session with Epic 4.5 academic standards.
        
        Args:
            session_id: Review session identifier
            segment_id: Segment being commented on
            author_id: Comment author
            comment_text: Comment content
            comment_type: Type of comment
            text_selection_start: Start position of highlighted text
            text_selection_end: End position of highlighted text
            highlighted_text: Selected text being commented on
            reply_to_comment_id: ID of comment being replied to
            
        Returns:
            str: Comment ID if successful, None otherwise
        """
        start_time = time.time()
        
        with self.performance_monitor.monitor_processing_operation("add_comment"):
            with self.lock:
                if session_id not in self.active_sessions:
                    return None
                
                session = self.active_sessions[session_id]
                
                # Generate unique comment ID
                comment_id = f"comment_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
                
                # Determine thread depth
                thread_depth = 0
                if reply_to_comment_id:
                    parent_comment = next(
                        (c for c in session.comments if c.comment_id == reply_to_comment_id), 
                        None
                    )
                    if parent_comment:
                        thread_depth = parent_comment.thread_depth + 1
                        
                        # Enforce thread depth limit
                        if thread_depth > self.interface_config['comment_thread_max_depth']:
                            self.logger.warning(f"Comment thread depth limit exceeded: {thread_depth}")
                            return None
                
                # Get author role and academic context
                author_role = self.user_presence.get(author_id, {}).get('role', 'unknown')
                
                # Create comment with Epic 4.5 academic enhancements
                comment = ReviewComment(
                    comment_id=comment_id,
                    session_id=session_id,
                    segment_id=segment_id,
                    comment_type=comment_type,
                    comment_text=comment_text,
                    author_id=author_id,
                    author_role=author_role,
                    reply_to_comment_id=reply_to_comment_id,
                    thread_depth=thread_depth,
                    text_selection_start=text_selection_start,
                    text_selection_end=text_selection_end,
                    highlighted_text=highlighted_text
                )
                
                # Add Epic 4.5 academic context
                self._enhance_comment_with_academic_context(comment, highlighted_text)
                
                # Add comment to session
                session.comments.append(comment)
                session.last_activity = datetime.now()
                
                # Update comment threading
                if reply_to_comment_id:
                    if reply_to_comment_id not in session.comment_threads:
                        session.comment_threads[reply_to_comment_id] = []
                    session.comment_threads[reply_to_comment_id].append(comment_id)
                
                # Update user activity
                if author_id in self.user_presence:
                    self.user_presence[author_id]['last_activity'] = datetime.now()
                
                self.interface_statistics['comments_added'] += 1
                
                processing_time = time.time() - start_time
                self.logger.info(f"Comment added: {comment_id} by {author_id} in session {session_id} ({processing_time:.3f}s)")
                
                return comment_id
    
    def resolve_comment(self, 
                       session_id: str,
                       comment_id: str,
                       resolver_id: str,
                       resolution_note: Optional[str] = None) -> bool:
        """
        Resolve comment with Epic 4.5 academic validation.
        
        Args:
            session_id: Review session identifier
            comment_id: Comment to resolve
            resolver_id: User resolving the comment
            resolution_note: Optional resolution note
            
        Returns:
            bool: True if comment resolved successfully
        """
        with self.lock:
            if session_id not in self.active_sessions:
                return False
            
            session = self.active_sessions[session_id]
            
            # Find comment
            comment = next((c for c in session.comments if c.comment_id == comment_id), None)
            if not comment:
                return False
            
            # Validate resolver authority
            resolver_role = self.user_presence.get(resolver_id, {}).get('role', 'unknown')
            if not self._can_resolve_comment(comment, resolver_role):
                self.logger.warning(f"User {resolver_id} cannot resolve comment {comment_id}")
                return False
            
            # Resolve comment
            comment.status = CommentStatus.RESOLVED
            comment.resolved_by = resolver_id
            comment.resolution_note = resolution_note
            comment.resolved_at = datetime.now()
            comment.updated_at = datetime.now()
            
            # Academic validation for consultant comments
            if comment.author_role == 'consultant' and resolver_role == 'consultant':
                comment.consultant_validated = True
            
            session.last_activity = datetime.now()
            
            self.interface_statistics['comments_resolved'] += 1
            
            self.logger.info(f"Comment resolved: {comment_id} by {resolver_id}")
            return True
    
    def record_action(self,
                     session_id: str,
                     segment_id: str,
                     action_type: ReviewActionType,
                     description: str,
                     performed_by: str,
                     original_text: str = "",
                     modified_text: str = "",
                     academic_justification: Optional[str] = None) -> Optional[str]:
        """
        Record review action with Epic 4.5 academic tracking.
        
        Args:
            session_id: Review session identifier
            segment_id: Segment being modified
            action_type: Type of action performed
            description: Action description
            performed_by: User performing action
            original_text: Text before modification
            modified_text: Text after modification
            academic_justification: Academic reasoning for change
            
        Returns:
            str: Action ID if successful, None otherwise
        """
        with self.lock:
            if session_id not in self.active_sessions:
                return None
            
            session = self.active_sessions[session_id]
            
            # Generate unique action ID
            action_id = f"action_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            # Get performer role
            performer_role = self.user_presence.get(performed_by, {}).get('role', 'unknown')
            
            # Create action with Epic 4.5 academic tracking
            action = ReviewAction(
                action_id=action_id,
                session_id=session_id,
                segment_id=segment_id,
                action_type=action_type,
                description=description,
                performed_by=performed_by,
                reviewer_role=performer_role,
                original_text=original_text,
                modified_text=modified_text,
                academic_justification=academic_justification
            )
            
            # Add Epic 4.5 academic validation
            self._validate_action_academic_compliance(action)
            
            # Calculate impact scores
            action.quality_impact_score = self._calculate_quality_impact(original_text, modified_text)
            action.academic_impact_score = self._calculate_academic_impact(action)
            
            # Add action to session
            session.actions_history.append(action)
            session.last_activity = datetime.now()
            
            # Check if action requires approval
            if self._requires_approval(action):
                session.pending_approvals.append(action_id)
            
            self.interface_statistics['actions_recorded'] += 1
            
            self.logger.info(f"Action recorded: {action_id} by {performed_by} in session {session_id}")
            return action_id
    
    def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive session state with Epic 4.5 academic context.
        
        Args:
            session_id: Review session identifier
            
        Returns:
            dict: Session state information
        """
        with self.lock:
            if session_id not in self.active_sessions:
                return None
            
            session = self.active_sessions[session_id]
            
            # Organize comments by thread
            comment_threads = {}
            root_comments = []
            
            for comment in session.comments:
                if comment.reply_to_comment_id is None:
                    root_comments.append(self._serialize_comment(comment))
                else:
                    parent_id = comment.reply_to_comment_id
                    if parent_id not in comment_threads:
                        comment_threads[parent_id] = []
                    comment_threads[parent_id].append(self._serialize_comment(comment))
            
            # Calculate academic metrics
            academic_comments = len([c for c in session.comments if c.comment_type == CommentType.ACADEMIC_NOTE])
            citation_references = len([c for c in session.comments if c.comment_type == CommentType.CITATION_REFERENCE])
            consultant_validations = len([c for c in session.comments if c.consultant_validated])
            
            return {
                'session_id': session_id,
                'active_reviewers': list(session.active_reviewers),
                'last_activity': session.last_activity.isoformat(),
                'collaboration_status': {
                    'concurrent_edits': len(session.concurrent_edits),
                    'edit_conflicts': len(session.edit_conflicts),
                    'academic_discussion_active': session.academic_discussion_active,
                    'citation_review_mode': session.citation_review_mode,
                    'consultant_participating': session.consultant_participating
                },
                'comments': {
                    'root_comments': root_comments,
                    'comment_threads': comment_threads,
                    'total_comments': len(session.comments),
                    'open_comments': len([c for c in session.comments if c.status == CommentStatus.OPEN]),
                    'resolved_comments': len([c for c in session.comments if c.status == CommentStatus.RESOLVED])
                },
                'actions': {
                    'total_actions': len(session.actions_history),
                    'pending_approvals': len(session.pending_approvals),
                    'recent_actions': [self._serialize_action(a) for a in session.actions_history[-5:]]
                },
                'academic_metrics': {
                    'academic_comments': academic_comments,
                    'citation_references': citation_references,
                    'consultant_validations': consultant_validations,
                    'iast_compliance_checks': len([a for a in session.actions_history if a.iast_compliance_verified]),
                    'publication_standard_validations': len([a for a in session.actions_history if a.publication_standard_met])
                }
            }
    
    def _enhance_comment_with_academic_context(self, comment: ReviewComment, highlighted_text: str) -> None:
        """Enhance comment with Epic 4.5 academic context."""
        try:
            # Check if comment relates to Sanskrit/Hindi terms
            if any(term in highlighted_text.lower() for term in ['yoga', 'dharma', 'karma', 'vedanta', 'krishna']):
                comment.requires_expert_review = True
                
                # Get relevant citations
                citations = self.citation_manager.get_citations_for_terms([highlighted_text])
                if citations:
                    comment.academic_citation = citations[0].get('formatted_citation', '')
                
                # Check IAST compliance
                if any(char in highlighted_text for char in 'āīūṛḷēōṃḥ'):
                    comment.iast_compliance_note = "IAST diacritics present - verify accuracy"
                else:
                    comment.iast_compliance_note = "Consider IAST transliteration"
            
            # Mark as peer review relevant for high-quality content
            if comment.comment_type in [CommentType.ACADEMIC_NOTE, CommentType.CITATION_REFERENCE]:
                comment.peer_review_relevant = True
                
        except Exception as e:
            self.logger.error(f"Failed to enhance comment with academic context: {e}")
    
    def _can_resolve_comment(self, comment: ReviewComment, resolver_role: str) -> bool:
        """Check if user can resolve comment based on role hierarchy."""
        # Comment author can always resolve their own comments
        if comment.author_id == resolver_role:
            return True
        
        # Role hierarchy for resolution authority
        role_hierarchy = {
            'gp': 1,
            'sme': 2,
            'consultant': 3,
            'admin': 4
        }
        
        author_level = role_hierarchy.get(comment.author_role, 0)
        resolver_level = role_hierarchy.get(resolver_role, 0)
        
        return resolver_level >= author_level
    
    def _validate_action_academic_compliance(self, action: ReviewAction) -> None:
        """Validate action for Epic 4.5 academic compliance."""
        try:
            # Check IAST compliance for Sanskrit terms
            if action.modified_text and any(term in action.modified_text.lower() for term in ['yoga', 'dharma', 'karma']):
                has_diacritics = any(char in action.modified_text for char in 'āīūṛḷēōṃḥ')
                action.iast_compliance_verified = has_diacritics
            
            # Check publication standards
            if action.action_type in [ReviewActionType.TEXT_EDIT, ReviewActionType.FORMATTING_CHANGE]:
                # Basic publication standard checks
                action.publication_standard_met = (
                    len(action.modified_text.strip()) > 0 and
                    not action.modified_text.startswith(' ') and
                    not action.modified_text.endswith(' ')
                )
            
            # Get citation support if academic justification provided
            if action.academic_justification:
                action.citation_support = self._extract_citation_references(action.academic_justification)
                
        except Exception as e:
            self.logger.error(f"Failed to validate academic compliance for action: {e}")
    
    def _calculate_quality_impact(self, original_text: str, modified_text: str) -> float:
        """Calculate quality impact score of text modification."""
        if not original_text or not modified_text:
            return 0.0
        
        # Simple quality metrics
        length_improvement = len(modified_text) / max(len(original_text), 1)
        
        # Bonus for fixing common issues
        quality_score = 0.5  # Base score
        
        if 'um' in original_text.lower() and 'um' not in modified_text.lower():
            quality_score += 0.2  # Filler word removal
        
        if any(term in modified_text.lower() for term in ['yoga', 'dharma', 'karma']):
            quality_score += 0.1  # Sanskrit term presence
        
        return min(quality_score * length_improvement, 1.0)
    
    def _calculate_academic_impact(self, action: ReviewAction) -> float:
        """Calculate academic impact score of action."""
        impact_score = 0.0
        
        # IAST compliance impact
        if action.iast_compliance_verified:
            impact_score += 0.3
        
        # Citation support impact
        if action.citation_support:
            impact_score += 0.2 * len(action.citation_support)
        
        # Academic justification impact
        if action.academic_justification:
            impact_score += 0.3
        
        # Publication standard impact
        if action.publication_standard_met:
            impact_score += 0.2
        
        return min(impact_score, 1.0)
    
    def _requires_approval(self, action: ReviewAction) -> bool:
        """Check if action requires approval from higher-level reviewer."""
        # Academic actions require approval
        if action.action_type in [ReviewActionType.ACADEMIC_VALIDATION, ReviewActionType.CITATION_ADD]:
            return True
        
        # High-impact actions require approval
        if action.quality_impact_score > 0.8 or action.academic_impact_score > 0.7:
            return True
        
        # GP actions on complex content require SME approval
        if action.reviewer_role == 'gp' and action.academic_impact_score > 0.5:
            return True
        
        return False
    
    def _extract_citation_references(self, text: str) -> List[str]:
        """Extract citation references from academic justification text."""
        # Simple regex-based extraction (would be enhanced in production)
        import re
        citations = re.findall(r'\(([^)]+\d{4}[^)]*)\)', text)
        return citations[:3]  # Limit to 3 citations
    
    def _serialize_comment(self, comment: ReviewComment) -> Dict[str, Any]:
        """Serialize comment for API response."""
        return {
            'comment_id': comment.comment_id,
            'comment_type': comment.comment_type.value,
            'comment_text': comment.comment_text,
            'suggested_text': comment.suggested_text,
            'author_id': comment.author_id,
            'author_role': comment.author_role,
            'status': comment.status.value,
            'created_at': comment.created_at.isoformat(),
            'highlighted_text': comment.highlighted_text,
            'academic_context': {
                'requires_expert_review': comment.requires_expert_review,
                'consultant_validated': comment.consultant_validated,
                'peer_review_relevant': comment.peer_review_relevant,
                'academic_citation': comment.academic_citation,
                'iast_compliance_note': comment.iast_compliance_note
            }
        }
    
    def _serialize_action(self, action: ReviewAction) -> Dict[str, Any]:
        """Serialize action for API response."""
        return {
            'action_id': action.action_id,
            'action_type': action.action_type.value,
            'description': action.description,
            'performed_by': action.performed_by,
            'reviewer_role': action.reviewer_role,
            'timestamp': action.timestamp.isoformat(),
            'quality_impact_score': action.quality_impact_score,
            'academic_impact_score': action.academic_impact_score,
            'academic_compliance': {
                'iast_compliance_verified': action.iast_compliance_verified,
                'publication_standard_met': action.publication_standard_met,
                'citation_support': action.citation_support,
                'academic_justification': action.academic_justification
            }
        }
    
    def get_interface_statistics(self) -> Dict[str, Any]:
        """Get collaborative interface statistics."""
        with self.lock:
            return {
                'sessions_created': self.interface_statistics['sessions_created'],
                'user_joins': self.interface_statistics['user_joins'],
                'comments_added': self.interface_statistics['comments_added'],
                'comments_resolved': self.interface_statistics['comments_resolved'],
                'actions_recorded': self.interface_statistics['actions_recorded'],
                'active_sessions': len(self.active_sessions),
                'active_users': len(self.user_presence)
            }