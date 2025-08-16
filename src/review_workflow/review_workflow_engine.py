"""
Core Review Workflow Engine with Epic 4.5 Academic Consultant Integration.

Implements Story 3.3 Task 1: Epic 4.5 Academic Consultant Integration
- GP-to-SME escalation leveraging Epic 4.5 academic consultant workflow
- Research-grade review standards with Google Docs-style collaboration  
- Academic citation management for contextual review information
- Publication formatter integration for review interface output standards
"""

import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple, Union

# Epic 4 infrastructure imports
from scripture_processing.publication_formatter import (
    PublicationFormatter, ConsultantReview, ConsultantReviewRequest
)
from scripture_processing.academic_citation_manager import AcademicCitationManager
from sanskrit_hindi_identifier.enhanced_lexicon_manager import EnhancedLexiconManager
from qa_module.qa_flagging_engine import QAFlaggingEngine, QAAnalysisResult
from utils.performance_monitor import PerformanceMonitor
from monitoring.system_monitor import SystemMonitor
from monitoring.telemetry_collector import TelemetryCollector
from utils.srt_parser import SRTSegment


class ReviewerRole(Enum):
    """Reviewer roles in the tiered system."""
    GENERAL_PROOFREADER = "general_proofreader"  # GP
    SUBJECT_MATTER_EXPERT = "subject_matter_expert"  # SME
    ACADEMIC_CONSULTANT = "academic_consultant"  # Epic 4.5 integration
    SENIOR_REVIEWER = "senior_reviewer"


class ReviewStatus(Enum):
    """Review session status."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_REVIEW = "in_review" 
    GP_COMPLETE = "gp_complete"
    SME_ESCALATED = "sme_escalated"
    SME_COMPLETE = "sme_complete"
    CONSULTANT_ESCALATED = "consultant_escalated"
    COMPLETED = "completed"
    REJECTED = "rejected"


class ReviewPriority(Enum):
    """Review priority levels aligned with Epic 4.5."""
    CRITICAL = "critical"
    HIGH = "high"
    STANDARD = "standard"
    LOW = "low"


@dataclass
class ReviewerSkill:
    """Reviewer skill assessment for expertise matching."""
    skill_area: str  # e.g., "sanskrit_linguistics", "vedanta_philosophy", "iast_transliteration"
    proficiency_level: float  # 0.0 - 1.0
    years_experience: int
    certification_level: str  # "novice", "intermediate", "expert", "master"
    specializations: List[str] = field(default_factory=list)


@dataclass  
class ReviewerProfile:
    """Comprehensive reviewer profile for Epic 4.5 integration."""
    reviewer_id: str
    name: str
    email: str
    role: ReviewerRole
    
    # Skills and expertise
    skills: List[ReviewerSkill] = field(default_factory=list)
    primary_expertise: str = ""
    secondary_expertise: List[str] = field(default_factory=list)
    
    # Performance metrics
    reviews_completed: int = 0
    average_review_time_hours: float = 0.0
    quality_rating: float = 1.0  # 0.0 - 1.0
    specialization_confidence: Dict[str, float] = field(default_factory=dict)
    
    # Availability and workload
    max_concurrent_reviews: int = 3
    current_review_load: int = 0
    available_hours_per_week: int = 20
    preferred_content_types: List[str] = field(default_factory=list)
    
    # Epic 4.5 academic integration
    academic_credentials: List[str] = field(default_factory=list)
    publication_history: List[str] = field(default_factory=list)
    institutional_affiliation: str = ""
    consultant_approval_level: str = ""  # Links to Epic 4.5 consultant workflow


@dataclass
class ReviewSegment:
    """Individual segment within a review session."""
    segment_id: str
    original_segment: SRTSegment
    review_status: str = "pending"  # pending, in_review, flagged, approved
    
    # Review annotations
    gp_corrections: List[str] = field(default_factory=list)
    sme_corrections: List[str] = field(default_factory=list)
    consultant_feedback: List[str] = field(default_factory=list)
    
    # Quality flags and complexity
    qa_flags: List[str] = field(default_factory=list)
    complexity_score: float = 0.0
    requires_escalation: bool = False
    escalation_reason: str = ""
    
    # Timestamps
    gp_review_start: Optional[datetime] = None
    gp_review_complete: Optional[datetime] = None
    sme_review_start: Optional[datetime] = None
    sme_review_complete: Optional[datetime] = None


@dataclass
class ReviewSession:
    """Complete review session integrating Epic 4.5 workflow."""
    session_id: str
    document_title: str
    priority: ReviewPriority
    
    # Content and segments
    segments: List[ReviewSegment] = field(default_factory=list)
    original_qa_result: Optional[QAAnalysisResult] = None
    
    # Reviewer assignments
    gp_reviewer_id: Optional[str] = None
    sme_reviewer_id: Optional[str] = None
    consultant_reviewer_id: Optional[str] = None
    
    # Review workflow state
    status: ReviewStatus = ReviewStatus.PENDING
    current_stage: str = "gp_assignment"
    escalation_triggers: List[str] = field(default_factory=list)
    
    # Epic 4.5 academic integration
    consultant_review: Optional[ConsultantReview] = None
    academic_citation_context: Dict[str, Any] = field(default_factory=dict)
    publication_requirements: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps and metrics
    created_at: datetime = field(default_factory=datetime.now)
    gp_deadline: Optional[datetime] = None
    sme_deadline: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Quality and performance tracking
    overall_quality_improvement: float = 0.0
    gp_review_time_hours: float = 0.0
    sme_review_time_hours: float = 0.0
    total_corrections_applied: int = 0


class ReviewWorkflowEngine:
    """
    Epic 4.5 Academic Consultant Integration for Tiered Human Review Workflow.
    
    Implements Story 3.3 Task 1:
    - GP-to-SME escalation using Epic 4.5 academic consultant workflow
    - Research-grade review standards with collaborative commenting
    - Academic citation management for contextual review information
    - Publication formatter integration for output standards
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize review workflow engine with Epic 4.5 integrations."""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Epic 4.5 Academic Components
        self.publication_formatter = PublicationFormatter(
            self.config.get('publication_formatter', {})
        )
        self.citation_manager = AcademicCitationManager(
            self.config.get('citation_manager', {})
        )
        self.enhanced_lexicon = EnhancedLexiconManager()
        
        # Epic 4.3 Production Infrastructure
        self.system_monitor = SystemMonitor(self.config.get('monitoring', {}))
        self.performance_monitor = PerformanceMonitor(self.config.get('performance', {}))
        self.telemetry_collector = TelemetryCollector(self.config.get('telemetry', {}))
        
        # Review workflow configuration
        self.workflow_config = self.config.get('workflow', {
            'gp_review_timeout_hours': 24,
            'sme_review_timeout_hours': 48,
            'max_concurrent_sessions': 20,
            'auto_escalation_threshold': 0.7,
            'consultant_escalation_threshold': 0.9,
            'quality_improvement_target': 0.15
        })
        
        # Data structures for session management
        self.active_sessions: Dict[str, ReviewSession] = {}
        self.reviewer_profiles: Dict[str, ReviewerProfile] = {}
        self.session_queue = deque()
        
        # Threading and concurrency (Epic 4.3)
        self.lock = threading.RLock()
        self.thread_pool = ThreadPoolExecutor(
            max_workers=8, thread_name_prefix="review_workflow"
        )
        
        # Performance and reliability tracking
        self.workflow_statistics = defaultdict(int)
        self.processing_history = deque(maxlen=1000)
        
        # Epic 4.1 Circuit breaker pattern
        self.circuit_breaker_failures = 0
        self.circuit_breaker_threshold = 3
        self.circuit_breaker_open = False
        
        # Initialize monitoring
        self.system_monitor.start_monitoring()
        
        self.logger.info("ReviewWorkflowEngine initialized with Epic 4.5 academic integration")
    
    def create_review_session(self, 
                            document_title: str,
                            segments: List[SRTSegment], 
                            qa_result: QAAnalysisResult,
                            priority: ReviewPriority = ReviewPriority.STANDARD) -> str:
        """
        Create new review session with Epic 4.5 academic standards.
        
        Args:
            document_title: Title of document being reviewed
            segments: SRT segments requiring review
            qa_result: QA analysis result from flagging engine
            priority: Review priority level
            
        Returns:
            session_id: Unique identifier for created session
        """
        start_time = time.time()
        
        with self.performance_monitor.monitor_processing_operation("create_review_session"):
            try:
                # Generate unique session ID
                session_id = f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
                
                # Create review segments from SRT segments and QA flags
                review_segments = []
                for i, segment in enumerate(segments):
                    # Find QA flags for this segment
                    segment_flags = [
                        flag for flag in qa_result.flags 
                        if flag.segment_index == i
                    ]
                    
                    review_segment = ReviewSegment(
                        segment_id=f"{session_id}_seg_{i}",
                        original_segment=segment,
                        qa_flags=[flag.message for flag in segment_flags],
                        complexity_score=self._calculate_segment_complexity(segment, segment_flags),
                        requires_escalation=any(flag.severity.value in ['critical', 'warning'] for flag in segment_flags)
                    )
                    review_segments.append(review_segment)
                
                # Create Epic 4.5 academic citation context
                academic_context = self._create_academic_context(document_title, segments)
                
                # Determine publication requirements based on content analysis
                publication_requirements = self._analyze_publication_requirements(segments, qa_result)
                
                # Create review session
                session = ReviewSession(
                    session_id=session_id,
                    document_title=document_title,
                    priority=priority,
                    segments=review_segments,
                    original_qa_result=qa_result,
                    academic_citation_context=academic_context,
                    publication_requirements=publication_requirements,
                    gp_deadline=datetime.now() + timedelta(hours=self.workflow_config['gp_review_timeout_hours']),
                    sme_deadline=datetime.now() + timedelta(hours=self.workflow_config['sme_review_timeout_hours'])
                )
                
                # Store session
                with self.lock:
                    self.active_sessions[session_id] = session
                    self.session_queue.append(session_id)
                    self.workflow_statistics['sessions_created'] += 1
                
                # Create Epic 4.5 consultant review if needed
                if self._requires_consultant_review(qa_result, publication_requirements):
                    consultant_review = self._create_consultant_review_request(session)
                    session.consultant_review = consultant_review
                    session.status = ReviewStatus.CONSULTANT_ESCALATED
                
                # Record telemetry
                processing_time = time.time() - start_time
                self.telemetry_collector.record_event("review_session_created", {
                    'session_id': session_id,
                    'segments_count': len(segments),
                    'priority': priority.value,
                    'processing_time_ms': processing_time * 1000,
                    'requires_consultant': session.consultant_review is not None
                })
                
                self.logger.info(f"Review session created: {session_id} ({len(segments)} segments, {priority.value} priority)")
                
                return session_id
                
            except Exception as e:
                self.circuit_breaker_failures += 1
                self.logger.error(f"Failed to create review session: {e}")
                raise
    
    def assign_gp_reviewer(self, session_id: str, gp_reviewer_id: str) -> bool:
        """
        Assign General Proofreader to review session.
        
        Args:
            session_id: Review session identifier
            gp_reviewer_id: GP reviewer identifier
            
        Returns:
            bool: True if assignment successful
        """
        with self.lock:
            if session_id not in self.active_sessions:
                self.logger.error(f"Session not found: {session_id}")
                return False
            
            session = self.active_sessions[session_id]
            
            # Validate GP reviewer
            if gp_reviewer_id not in self.reviewer_profiles:
                self.logger.error(f"GP reviewer not found: {gp_reviewer_id}")
                return False
            
            gp_profile = self.reviewer_profiles[gp_reviewer_id]
            if gp_profile.role != ReviewerRole.GENERAL_PROOFREADER:
                self.logger.error(f"Reviewer {gp_reviewer_id} is not a General Proofreader")
                return False
            
            # Check workload capacity
            if gp_profile.current_review_load >= gp_profile.max_concurrent_reviews:
                self.logger.warning(f"GP reviewer {gp_reviewer_id} at maximum capacity")
                return False
            
            # Assign reviewer
            session.gp_reviewer_id = gp_reviewer_id
            session.status = ReviewStatus.ASSIGNED
            session.current_stage = "gp_review"
            
            # Update reviewer workload
            gp_profile.current_review_load += 1
            
            # Initialize segment review timestamps
            for segment in session.segments:
                segment.gp_review_start = datetime.now()
            
            self.workflow_statistics['gp_assignments'] += 1
            
            self.logger.info(f"GP reviewer {gp_reviewer_id} assigned to session {session_id}")
            return True
    
    def submit_gp_review(self, session_id: str, segment_corrections: Dict[str, List[str]]) -> bool:
        """
        Submit GP review results and evaluate for SME escalation.
        
        Args:
            session_id: Review session identifier
            segment_corrections: Corrections by segment ID
            
        Returns:
            bool: True if submission successful
        """
        start_time = time.time()
        
        with self.performance_monitor.monitor_processing_operation("submit_gp_review"):
            with self.lock:
                if session_id not in self.active_sessions:
                    return False
                
                session = self.active_sessions[session_id]
                
                # Apply GP corrections
                total_corrections = 0
                for segment in session.segments:
                    if segment.segment_id in segment_corrections:
                        segment.gp_corrections = segment_corrections[segment.segment_id]
                        segment.gp_review_complete = datetime.now()
                        total_corrections += len(segment.gp_corrections)
                
                # Calculate GP review time
                gp_review_time = (datetime.now() - session.segments[0].gp_review_start).total_seconds() / 3600
                session.gp_review_time_hours = gp_review_time
                
                # Update GP reviewer statistics
                if session.gp_reviewer_id:
                    gp_profile = self.reviewer_profiles[session.gp_reviewer_id]
                    gp_profile.reviews_completed += 1
                    gp_profile.current_review_load -= 1
                    
                    # Update average review time
                    total_time = gp_profile.average_review_time_hours * (gp_profile.reviews_completed - 1)
                    gp_profile.average_review_time_hours = (total_time + gp_review_time) / gp_profile.reviews_completed
                
                # Evaluate for SME escalation using Epic 4.5 criteria
                escalation_needed = self._evaluate_sme_escalation(session, total_corrections)
                
                if escalation_needed:
                    session.status = ReviewStatus.SME_ESCALATED
                    session.current_stage = "sme_assignment"
                    
                    # Create Epic 4.5 academic citation context for SME review
                    self._enhance_academic_context_for_sme(session)
                    
                    self.workflow_statistics['sme_escalations'] += 1
                    self.logger.info(f"Session {session_id} escalated to SME review")
                else:
                    session.status = ReviewStatus.GP_COMPLETE
                    session.current_stage = "final_review"
                    self.workflow_statistics['gp_completions'] += 1
                
                session.total_corrections_applied = total_corrections
                
                # Record telemetry
                processing_time = time.time() - start_time
                self.telemetry_collector.record_event("gp_review_submitted", {
                    'session_id': session_id,
                    'corrections_count': total_corrections,
                    'review_time_hours': gp_review_time,
                    'escalated_to_sme': escalation_needed,
                    'processing_time_ms': processing_time * 1000
                })
                
                return True
    
    def escalate_to_sme(self, session_id: str, sme_reviewer_id: str, escalation_reason: str) -> bool:
        """
        Escalate session to Subject Matter Expert using Epic 4.5 standards.
        
        Args:
            session_id: Review session identifier  
            sme_reviewer_id: SME reviewer identifier
            escalation_reason: Reason for escalation
            
        Returns:
            bool: True if escalation successful
        """
        with self.lock:
            if session_id not in self.active_sessions:
                return False
            
            session = self.active_sessions[session_id]
            
            # Validate SME reviewer
            if sme_reviewer_id not in self.reviewer_profiles:
                return False
            
            sme_profile = self.reviewer_profiles[sme_reviewer_id]
            if sme_profile.role not in [ReviewerRole.SUBJECT_MATTER_EXPERT, ReviewerRole.ACADEMIC_CONSULTANT]:
                return False
            
            # Assign SME reviewer
            session.sme_reviewer_id = sme_reviewer_id
            session.status = ReviewStatus.SME_ESCALATED
            session.escalation_triggers.append(escalation_reason)
            
            # Initialize SME review timestamps
            for segment in session.segments:
                if segment.requires_escalation:
                    segment.sme_review_start = datetime.now()
            
            # Update SME workload
            sme_profile.current_review_load += 1
            
            # Create Epic 4.5 consultant review request if criteria met
            if self._meets_consultant_criteria(session, sme_profile):
                consultant_review = self._create_consultant_review_request(session)
                session.consultant_review = consultant_review
                session.status = ReviewStatus.CONSULTANT_ESCALATED
            
            self.workflow_statistics['sme_escalations'] += 1
            
            self.logger.info(f"Session {session_id} escalated to SME {sme_reviewer_id}: {escalation_reason}")
            return True
    
    def _calculate_segment_complexity(self, segment: SRTSegment, qa_flags: List) -> float:
        """Calculate complexity score for segment based on QA flags and content."""
        base_complexity = 0.3
        
        # Add complexity based on QA flags
        flag_complexity = len(qa_flags) * 0.1
        
        # Add complexity based on Sanskrit/Hindi content
        text = segment.text.lower()
        sanskrit_terms = sum(1 for term in ['yoga', 'dharma', 'karma', 'vedanta', 'upanishad', 'gita', 'krishna'] if term in text)
        sanskrit_complexity = min(sanskrit_terms * 0.05, 0.3)
        
        # Add complexity based on text length
        length_complexity = min(len(segment.text) / 1000, 0.2)
        
        return min(base_complexity + flag_complexity + sanskrit_complexity + length_complexity, 1.0)
    
    def _create_academic_context(self, document_title: str, segments: List[SRTSegment]) -> Dict[str, Any]:
        """Create Epic 4.5 academic citation context for review."""
        try:
            # Extract key terms for citation context
            key_terms = set()
            for segment in segments:
                words = segment.text.lower().split()
                for word in words:
                    if word in ['yoga', 'dharma', 'karma', 'vedanta', 'upanishad', 'gita', 'krishna', 'arjuna', 'bhagavad']:
                        key_terms.add(word)
            
            # Get academic citations for key terms
            citations = self.citation_manager.get_citations_for_terms(list(key_terms))
            
            return {
                'document_title': document_title,
                'key_terms': list(key_terms),
                'relevant_citations': citations,
                'citation_compliance_required': True,
                'academic_standards_level': 'research_grade'
            }
        except Exception as e:
            self.logger.error(f"Failed to create academic context: {e}")
            return {}
    
    def _analyze_publication_requirements(self, segments: List[SRTSegment], qa_result: QAAnalysisResult) -> Dict[str, Any]:
        """Analyze publication requirements based on content and QA results."""
        requirements = {
            'requires_iast_validation': False,
            'requires_peer_review': False,
            'publication_readiness_threshold': 0.85,
            'academic_compliance_required': True
        }
        
        # Check for Sanskrit content requiring IAST
        for segment in segments:
            if any(term in segment.text.lower() for term in ['yoga', 'dharma', 'karma', 'vedanta']):
                requirements['requires_iast_validation'] = True
                break
        
        # Check if quality warrants peer review
        if qa_result.overall_quality_score >= 0.9 and qa_result.academic_compliance_score >= 0.85:
            requirements['requires_peer_review'] = True
        
        return requirements
    
    def _requires_consultant_review(self, qa_result: QAAnalysisResult, publication_requirements: Dict[str, Any]) -> bool:
        """Determine if Epic 4.5 consultant review is required."""
        return (
            qa_result.academic_compliance_score >= self.workflow_config['consultant_escalation_threshold'] or
            publication_requirements.get('requires_peer_review', False) or
            qa_result.overall_quality_score >= 0.92
        )
    
    def _create_consultant_review_request(self, session: ReviewSession) -> ConsultantReview:
        """Create Epic 4.5 consultant review request."""
        try:
            # Use Epic 4.5 publication formatter to create consultant review
            document = self.publication_formatter._create_publication_document(
                title=session.document_title,
                content=" ".join([seg.original_segment.text for seg in session.segments]),
                metadata={'session_id': session.session_id}
            )
            
            consultant_review = self.publication_formatter.submit_for_consultant_review(
                document=document,
                priority_level=session.priority.value,
                specific_focus_areas=['sanskrit_accuracy', 'academic_standards', 'iast_compliance'],
                expedited=session.priority in [ReviewPriority.CRITICAL, ReviewPriority.HIGH]
            )
            
            return consultant_review
            
        except Exception as e:
            self.logger.error(f"Failed to create consultant review request: {e}")
            return None
    
    def _evaluate_sme_escalation(self, session: ReviewSession, total_corrections: int) -> bool:
        """Evaluate if session should be escalated to SME using Epic 4.5 criteria."""
        escalation_factors = []
        
        # High number of corrections suggests complexity
        if total_corrections > len(session.segments) * 0.5:
            escalation_factors.append("high_correction_rate")
        
        # Critical QA flags require SME attention
        critical_flags = sum(1 for seg in session.segments for flag in seg.qa_flags if 'critical' in flag.lower())
        if critical_flags > 0:
            escalation_factors.append("critical_qa_flags")
        
        # High complexity segments need expert review
        high_complexity_segments = sum(1 for seg in session.segments if seg.complexity_score > 0.7)
        if high_complexity_segments > len(session.segments) * 0.3:
            escalation_factors.append("high_complexity_content")
        
        # Academic standards requirements
        if session.publication_requirements.get('requires_peer_review', False):
            escalation_factors.append("peer_review_requirements")
        
        # Escalate if any significant factors present
        return len(escalation_factors) > 0
    
    def _enhance_academic_context_for_sme(self, session: ReviewSession) -> None:
        """Enhance academic context for SME review using Epic 4.5 standards."""
        try:
            # Get enhanced lexicon entries requiring academic review
            academic_queue = self.enhanced_lexicon.get_academic_review_queue()
            
            # Filter relevant entries
            relevant_entries = []
            session_text = " ".join([seg.original_segment.text for seg in session.segments]).lower()
            
            for entry in academic_queue:
                if entry['term'].lower() in session_text:
                    relevant_entries.append(entry)
            
            # Update academic context
            session.academic_citation_context.update({
                'sme_review_required': True,
                'relevant_lexicon_entries': relevant_entries,
                'academic_validation_level': 'expert_review',
                'enhanced_citation_context': True
            })
            
        except Exception as e:
            self.logger.error(f"Failed to enhance academic context for SME: {e}")
    
    def _meets_consultant_criteria(self, session: ReviewSession, sme_profile: ReviewerProfile) -> bool:
        """Check if session meets Epic 4.5 consultant escalation criteria."""
        return (
            session.priority in [ReviewPriority.CRITICAL, ReviewPriority.HIGH] or
            session.publication_requirements.get('requires_peer_review', False) or
            sme_profile.role == ReviewerRole.ACADEMIC_CONSULTANT or
            len(session.escalation_triggers) > 2
        )
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive session status with Epic 4.5 academic metrics."""
        with self.lock:
            if session_id not in self.active_sessions:
                return None
            
            session = self.active_sessions[session_id]
            
            return {
                'session_id': session_id,
                'status': session.status.value,
                'current_stage': session.current_stage,
                'priority': session.priority.value,
                'segments_count': len(session.segments),
                'gp_reviewer': session.gp_reviewer_id,
                'sme_reviewer': session.sme_reviewer_id,
                'consultant_review': session.consultant_review.review_id if session.consultant_review else None,
                'progress': {
                    'gp_complete': session.status.value in ['gp_complete', 'sme_escalated', 'completed'],
                    'sme_complete': session.status.value in ['sme_complete', 'completed'],
                    'consultant_complete': session.status == ReviewStatus.COMPLETED
                },
                'academic_context': session.academic_citation_context,
                'publication_requirements': session.publication_requirements,
                'escalation_triggers': session.escalation_triggers,
                'corrections_applied': session.total_corrections_applied,
                'created_at': session.created_at.isoformat(),
                'deadlines': {
                    'gp_deadline': session.gp_deadline.isoformat() if session.gp_deadline else None,
                    'sme_deadline': session.sme_deadline.isoformat() if session.sme_deadline else None
                }
            }
    
    def get_workflow_statistics(self) -> Dict[str, Any]:
        """Get comprehensive workflow statistics with Epic 4.5 metrics."""
        with self.lock:
            active_sessions_count = len(self.active_sessions)
            
            # Calculate performance metrics
            total_sessions = self.workflow_statistics['sessions_created']
            gp_completion_rate = (
                self.workflow_statistics['gp_completions'] / max(total_sessions, 1)
            )
            sme_escalation_rate = (
                self.workflow_statistics['sme_escalations'] / max(total_sessions, 1)
            )
            
            return {
                'overview': {
                    'active_sessions': active_sessions_count,
                    'total_sessions_created': total_sessions,
                    'sessions_completed': self.workflow_statistics.get('sessions_completed', 0),
                    'gp_assignments': self.workflow_statistics['gp_assignments'],
                    'sme_escalations': self.workflow_statistics['sme_escalations']
                },
                'performance': {
                    'gp_completion_rate': gp_completion_rate,
                    'sme_escalation_rate': sme_escalation_rate,
                    'average_gp_review_time_hours': self._calculate_average_gp_time(),
                    'circuit_breaker_status': 'open' if self.circuit_breaker_open else 'closed'
                },
                'epic_4_5_integration': {
                    'consultant_reviews_requested': self.workflow_statistics.get('consultant_reviews', 0),
                    'academic_citations_used': self.workflow_statistics.get('citations_used', 0),
                    'publication_formatter_active': True,
                    'academic_validation_enabled': True
                },
                'reviewer_metrics': {
                    'total_reviewers': len(self.reviewer_profiles),
                    'gp_reviewers': len([r for r in self.reviewer_profiles.values() if r.role == ReviewerRole.GENERAL_PROOFREADER]),
                    'sme_reviewers': len([r for r in self.reviewer_profiles.values() if r.role == ReviewerRole.SUBJECT_MATTER_EXPERT]),
                    'consultant_reviewers': len([r for r in self.reviewer_profiles.values() if r.role == ReviewerRole.ACADEMIC_CONSULTANT])
                }
            }
    
    def _calculate_average_gp_time(self) -> float:
        """Calculate average GP review time across all reviewers."""
        if not self.reviewer_profiles:
            return 0.0
        
        gp_reviewers = [
            r for r in self.reviewer_profiles.values() 
            if r.role == ReviewerRole.GENERAL_PROOFREADER and r.reviews_completed > 0
        ]
        
        if not gp_reviewers:
            return 0.0
        
        total_time = sum(r.average_review_time_hours for r in gp_reviewers)
        return total_time / len(gp_reviewers)
    
    def register_reviewer(self, profile: ReviewerProfile) -> bool:
        """Register new reviewer with Epic 4.5 academic validation."""
        with self.lock:
            if profile.reviewer_id in self.reviewer_profiles:
                self.logger.warning(f"Reviewer already registered: {profile.reviewer_id}")
                return False
            
            # Validate academic credentials for consultant roles
            if profile.role == ReviewerRole.ACADEMIC_CONSULTANT:
                if not profile.academic_credentials or not profile.institutional_affiliation:
                    self.logger.error(f"Insufficient academic credentials for consultant role: {profile.reviewer_id}")
                    return False
            
            self.reviewer_profiles[profile.reviewer_id] = profile
            self.workflow_statistics['reviewers_registered'] += 1
            
            self.logger.info(f"Reviewer registered: {profile.reviewer_id} ({profile.role.value})")
            return True
    
    def __del__(self):
        """Cleanup Epic 4.3 monitoring resources."""
        try:
            if hasattr(self, 'system_monitor'):
                self.system_monitor.stop_monitoring()
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=True)
        except:
            pass