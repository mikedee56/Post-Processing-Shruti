"""
Expert Review Queue System - Story 3.2.1
Async queue management for complex linguistic cases requiring expert review.
Integrates with existing ReviewWorkflowEngine for seamless review processing.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable, Awaitable
from uuid import uuid4, UUID
import json

from .quality_gate import QualityReport, QualityLevel
from ..review_workflow.review_workflow_engine import (
    ReviewWorkflowEngine, ReviewerProfile, ReviewerRole,
    ReviewSession, ReviewSegment
)


class TicketPriority(Enum):
    """Priority levels for expert review tickets"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class TicketStatus(Enum):
    """Status tracking for review tickets"""
    QUEUED = "queued"
    ASSIGNED = "assigned"
    IN_REVIEW = "in_review"
    ESCALATED = "escalated"
    COMPLETED = "completed"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


class ExpertiseType(Enum):
    """Types of expertise required for different linguistic issues"""
    SANSKRIT_GRAMMAR = "sanskrit_grammar"
    HINDI_PHONETICS = "hindi_phonetics"
    IAST_TRANSLITERATION = "iast_transliteration"
    SCRIPTURAL_CONTEXT = "scriptural_context"
    SEMANTIC_ANALYSIS = "semantic_analysis"
    GENERAL_LINGUISTIC = "general_linguistic"


@dataclass
class ReviewTicket:
    """
    Tracking system for expert review cases with comprehensive metadata.
    Integrates with existing ReviewSession while adding queue-specific functionality.
    """
    ticket_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.utcnow)
    priority: TicketPriority = TicketPriority.MEDIUM
    status: TicketStatus = TicketStatus.QUEUED
    expertise_required: Set[ExpertiseType] = field(default_factory=set)
    
    # Content and context
    segment_text: str = ""
    quality_report: Optional[QualityReport] = None
    processing_context: Dict[str, Any] = field(default_factory=dict)
    
    # Assignment tracking
    assigned_expert_id: Optional[str] = None
    assigned_at: Optional[datetime] = None
    
    # Review session integration
    review_session_id: Optional[UUID] = None
    
    # Timeline tracking
    estimated_completion: Optional[datetime] = None
    actual_completion: Optional[datetime] = None
    timeout_at: Optional[datetime] = None
    
    # Metadata
    retry_count: int = 0
    escalation_history: List[Dict[str, Any]] = field(default_factory=list)
    expert_notes: List[Dict[str, Any]] = field(default_factory=list)

    def is_expired(self) -> bool:
        """Check if ticket has exceeded timeout threshold"""
        if self.timeout_at:
            return datetime.utcnow() > self.timeout_at
        return False

    def get_age_minutes(self) -> float:
        """Get ticket age in minutes"""
        return (datetime.utcnow() - self.created_at).total_seconds() / 60

    def add_expert_note(self, expert_id: str, note: str, note_type: str = "general"):
        """Add expert note with timestamp"""
        self.expert_notes.append({
            "expert_id": expert_id,
            "note": note,
            "note_type": note_type,
            "timestamp": datetime.utcnow().isoformat()
        })


class ExpertReviewQueue:
    """
    Async queue system for managing expert review tickets.
    Integrates with existing ReviewWorkflowEngine for comprehensive review management.
    
    Features:
    - Non-blocking async task processing
    - Priority-based queue management
    - Expert assignment logic with expertise matching
    - Timeout and escalation handling
    - Integration with existing review workflow
    """
    
    def __init__(self, 
                 review_workflow: Optional[ReviewWorkflowEngine] = None,
                 max_queue_size: int = 1000,
                 default_timeout_minutes: int = 120,
                 max_retry_attempts: int = 3):
        
        self.logger = logging.getLogger(__name__)
        self.review_workflow = review_workflow or ReviewWorkflowEngine()
        
        # Queue configuration
        self.max_queue_size = max_queue_size
        self.default_timeout_minutes = default_timeout_minutes
        self.max_retry_attempts = max_retry_attempts
        
        # Queue storage
        self.tickets: Dict[UUID, ReviewTicket] = {}
        self.queued_tickets: asyncio.PriorityQueue = asyncio.PriorityQueue(max_queue_size)
        self.assigned_tickets: Dict[str, Set[UUID]] = {}  # expert_id -> ticket_ids
        
        # Expert availability tracking
        self.expert_availability: Dict[str, bool] = {}
        self.expert_capacity: Dict[str, int] = {}  # expert_id -> max concurrent tickets
        self.expert_expertise: Dict[str, Set[ExpertiseType]] = {}
        
        # Async processing
        self._queue_processor_task: Optional[asyncio.Task] = None
        self._timeout_monitor_task: Optional[asyncio.Task] = None
        self._is_running = False
        
        # Notification callbacks
        self.notification_callbacks: List[Callable[[ReviewTicket, str], Awaitable[None]]] = []
        
        # Performance metrics
        self.metrics = {
            "tickets_processed": 0,
            "tickets_completed": 0,
            "tickets_escalated": 0,
            "tickets_timeout": 0,
            "average_processing_time_minutes": 0.0,
            "queue_wait_time_minutes": 0.0
        }

    async def start(self):
        """Start async queue processing"""
        if self._is_running:
            return
            
        self._is_running = True
        self.logger.info("Starting ExpertReviewQueue async processing")
        
        # Start background tasks
        self._queue_processor_task = asyncio.create_task(self._process_queue())
        self._timeout_monitor_task = asyncio.create_task(self._monitor_timeouts())
        
        # Initialize expert profiles from existing review workflow
        await self._initialize_expert_profiles()

    async def stop(self):
        """Stop async queue processing gracefully"""
        self._is_running = False
        
        if self._queue_processor_task:
            self._queue_processor_task.cancel()
            try:
                await self._queue_processor_task
            except asyncio.CancelledError:
                pass
                
        if self._timeout_monitor_task:
            self._timeout_monitor_task.cancel()
            try:
                await self._timeout_monitor_task
            except asyncio.CancelledError:
                pass
                
        self.logger.info("ExpertReviewQueue stopped")

    async def submit_for_review(self, 
                               segment_text: str,
                               quality_report: QualityReport,
                               priority: TicketPriority = TicketPriority.MEDIUM,
                               required_expertise: Optional[Set[ExpertiseType]] = None,
                               processing_context: Optional[Dict[str, Any]] = None) -> UUID:
        """
        Submit a segment for expert review (non-blocking).
        
        Args:
            segment_text: Text requiring expert review
            quality_report: QA analysis results
            priority: Ticket priority level
            required_expertise: Specific expertise types needed
            processing_context: Additional context for reviewers
            
        Returns:
            UUID: Ticket ID for tracking
        """
        
        if self.queued_tickets.qsize() >= self.max_queue_size:
            raise RuntimeError(f"Review queue is full (max: {self.max_queue_size})")
        
        # Determine required expertise based on quality issues
        if required_expertise is None:
            required_expertise = self._analyze_expertise_requirements(quality_report)
        
        # Create review ticket
        ticket = ReviewTicket(
            segment_text=segment_text,
            quality_report=quality_report,
            priority=priority,
            expertise_required=required_expertise,
            processing_context=processing_context or {},
            timeout_at=datetime.utcnow() + timedelta(minutes=self.default_timeout_minutes)
        )
        
        # Store ticket
        self.tickets[ticket.ticket_id] = ticket
        
        # Add to priority queue (lower priority value = higher priority)
        priority_value = self._get_priority_value(priority)
        await self.queued_tickets.put((priority_value, ticket.created_at, ticket.ticket_id))
        
        self.logger.info(f"Submitted ticket {ticket.ticket_id} for expert review")
        await self._notify_ticket_event(ticket, "submitted")
        
        return ticket.ticket_id

    def get_ticket_status(self, ticket_id: UUID) -> Optional[Dict[str, Any]]:
        """Get current status of a review ticket"""
        ticket = self.tickets.get(ticket_id)
        if not ticket:
            return None
            
        return {
            "ticket_id": str(ticket.ticket_id),
            "status": ticket.status.value,
            "priority": ticket.priority.value,
            "created_at": ticket.created_at.isoformat(),
            "assigned_expert": ticket.assigned_expert_id,
            "age_minutes": ticket.get_age_minutes(),
            "is_expired": ticket.is_expired(),
            "expertise_required": [exp.value for exp in ticket.expertise_required]
        }

    def register_expert(self, 
                       expert_id: str,
                       expertise_types: Set[ExpertiseType],
                       max_concurrent_tickets: int = 5,
                       is_available: bool = True):
        """Register expert with their capabilities"""
        self.expert_expertise[expert_id] = expertise_types
        self.expert_capacity[expert_id] = max_concurrent_tickets
        self.expert_availability[expert_id] = is_available
        
        if expert_id not in self.assigned_tickets:
            self.assigned_tickets[expert_id] = set()
            
        self.logger.info(f"Registered expert {expert_id} with expertise: {[e.value for e in expertise_types]}")

    def set_expert_availability(self, expert_id: str, is_available: bool):
        """Update expert availability status"""
        if expert_id in self.expert_availability:
            self.expert_availability[expert_id] = is_available
            self.logger.info(f"Expert {expert_id} availability set to {is_available}")

    async def add_notification_callback(self, callback: Callable[[ReviewTicket, str], Awaitable[None]]):
        """Add notification callback for ticket events"""
        self.notification_callbacks.append(callback)

    def get_queue_metrics(self) -> Dict[str, Any]:
        """Get current queue performance metrics"""
        return {
            **self.metrics,
            "queue_size": self.queued_tickets.qsize(),
            "active_tickets": len([t for t in self.tickets.values() 
                                 if t.status in [TicketStatus.ASSIGNED, TicketStatus.IN_REVIEW]]),
            "total_tickets": len(self.tickets),
            "available_experts": len([e for e, available in self.expert_availability.items() if available]),
            "expert_workload": {expert_id: len(ticket_ids) 
                              for expert_id, ticket_ids in self.assigned_tickets.items()}
        }

    # Private methods for queue processing

    async def _process_queue(self):
        """Main queue processing loop"""
        while self._is_running:
            try:
                # Get next ticket with timeout to allow graceful shutdown
                try:
                    priority_value, created_at, ticket_id = await asyncio.wait_for(
                        self.queued_tickets.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                ticket = self.tickets.get(ticket_id)
                if not ticket or ticket.status != TicketStatus.QUEUED:
                    continue
                
                # Attempt expert assignment
                assigned_expert = await self._assign_expert(ticket)
                
                if assigned_expert:
                    # Successfully assigned
                    ticket.status = TicketStatus.ASSIGNED
                    ticket.assigned_expert_id = assigned_expert
                    ticket.assigned_at = datetime.utcnow()
                    
                    self.assigned_tickets[assigned_expert].add(ticket_id)
                    
                    # Create review session in workflow engine
                    await self._create_review_session(ticket, assigned_expert)
                    
                    await self._notify_ticket_event(ticket, "assigned")
                    self.logger.info(f"Assigned ticket {ticket_id} to expert {assigned_expert}")
                    
                else:
                    # No expert available, requeue with delay
                    await asyncio.sleep(5)  # Wait before retrying
                    await self.queued_tickets.put((priority_value, created_at, ticket_id))
                
            except Exception as e:
                self.logger.error(f"Error in queue processing: {e}")
                await asyncio.sleep(1)

    async def _assign_expert(self, ticket: ReviewTicket) -> Optional[str]:
        """
        Assign expert based on expertise matching and availability.
        Uses sophisticated matching algorithm for optimal assignment.
        """
        
        best_expert = None
        best_score = 0
        
        for expert_id, expert_expertise in self.expert_expertise.items():
            # Check availability
            if not self.expert_availability.get(expert_id, False):
                continue
            
            # Check capacity
            current_load = len(self.assigned_tickets.get(expert_id, set()))
            max_capacity = self.expert_capacity.get(expert_id, 5)
            if current_load >= max_capacity:
                continue
            
            # Calculate expertise match score
            expertise_overlap = len(ticket.expertise_required & expert_expertise)
            if expertise_overlap == 0 and ticket.expertise_required:
                continue  # No relevant expertise
                
            # Scoring algorithm
            expertise_score = expertise_overlap / max(len(ticket.expertise_required), 1)
            capacity_score = (max_capacity - current_load) / max_capacity
            
            total_score = expertise_score * 0.7 + capacity_score * 0.3
            
            if total_score > best_score:
                best_score = total_score
                best_expert = expert_id
        
        return best_expert

    async def _monitor_timeouts(self):
        """Monitor tickets for timeout and handle escalation"""
        while self._is_running:
            try:
                current_time = datetime.utcnow()
                
                for ticket in list(self.tickets.values()):
                    if ticket.is_expired() and ticket.status in [TicketStatus.ASSIGNED, TicketStatus.IN_REVIEW]:
                        await self._handle_timeout(ticket)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in timeout monitoring: {e}")
                await asyncio.sleep(10)

    async def _handle_timeout(self, ticket: ReviewTicket):
        """Handle ticket timeout with escalation logic"""
        self.logger.warning(f"Ticket {ticket.ticket_id} has timed out")
        
        ticket.retry_count += 1
        
        if ticket.retry_count <= self.max_retry_attempts:
            # Escalate priority and reassign
            if ticket.priority != TicketPriority.CRITICAL:
                old_priority = ticket.priority
                ticket.priority = TicketPriority.CRITICAL
                ticket.escalation_history.append({
                    "from_priority": old_priority.value,
                    "to_priority": ticket.priority.value,
                    "reason": "timeout_escalation",
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            # Remove from current assignment
            if ticket.assigned_expert_id:
                self.assigned_tickets[ticket.assigned_expert_id].discard(ticket.ticket_id)
            
            # Reset for reassignment
            ticket.status = TicketStatus.ESCALATED
            ticket.assigned_expert_id = None
            ticket.assigned_at = None
            ticket.timeout_at = datetime.utcnow() + timedelta(minutes=self.default_timeout_minutes)
            
            # Requeue with higher priority
            priority_value = self._get_priority_value(ticket.priority)
            await self.queued_tickets.put((priority_value, datetime.utcnow(), ticket.ticket_id))
            
            await self._notify_ticket_event(ticket, "escalated")
            self.metrics["tickets_escalated"] += 1
            
        else:
            # Max retries exceeded
            ticket.status = TicketStatus.TIMEOUT
            await self._notify_ticket_event(ticket, "timeout")
            self.metrics["tickets_timeout"] += 1

    async def _initialize_expert_profiles(self):
        """Initialize expert profiles from existing review workflow"""
        # Get reviewer profiles from existing workflow
        reviewer_profiles = self.review_workflow.get_reviewer_profiles()
        
        for profile in reviewer_profiles:
            # Map reviewer roles to expertise types
            expertise_types = self._map_reviewer_role_to_expertise(profile.role)
            
            self.register_expert(
                expert_id=profile.reviewer_id,
                expertise_types=expertise_types,
                max_concurrent_tickets=5,  # Default capacity
                is_available=profile.is_available
            )

    def _map_reviewer_role_to_expertise(self, role: ReviewerRole) -> Set[ExpertiseType]:
        """Map reviewer roles to expertise types"""
        role_mapping = {
            ReviewerRole.GENERAL_PURPOSE: {ExpertiseType.GENERAL_LINGUISTIC},
            ReviewerRole.SUBJECT_MATTER_EXPERT: {
                ExpertiseType.SANSKRIT_GRAMMAR,
                ExpertiseType.HINDI_PHONETICS,
                ExpertiseType.SCRIPTURAL_CONTEXT
            },
            ReviewerRole.ACADEMIC_CONSULTANT: {
                ExpertiseType.IAST_TRANSLITERATION,
                ExpertiseType.SEMANTIC_ANALYSIS,
                ExpertiseType.SCRIPTURAL_CONTEXT
            }
        }
        return role_mapping.get(role, {ExpertiseType.GENERAL_LINGUISTIC})

    def _analyze_expertise_requirements(self, quality_report: QualityReport) -> Set[ExpertiseType]:
        """Analyze quality report to determine required expertise"""
        required_expertise = set()
        
        # Analyze quality issues to determine expertise needs
        if quality_report.overall_quality == QualityLevel.POOR:
            required_expertise.add(ExpertiseType.GENERAL_LINGUISTIC)
        
        # Check for specific issues in the quality report
        # This would be expanded based on actual QualityReport structure
        if hasattr(quality_report, 'issues') and quality_report.issues:
            for issue in quality_report.issues:
                if 'iast' in issue.lower() or 'transliteration' in issue.lower():
                    required_expertise.add(ExpertiseType.IAST_TRANSLITERATION)
                elif 'sanskrit' in issue.lower():
                    required_expertise.add(ExpertiseType.SANSKRIT_GRAMMAR)
                elif 'hindi' in issue.lower():
                    required_expertise.add(ExpertiseType.HINDI_PHONETICS)
                elif 'verse' in issue.lower() or 'scripture' in issue.lower():
                    required_expertise.add(ExpertiseType.SCRIPTURAL_CONTEXT)
        
        # Default to general linguistic if no specific requirements
        if not required_expertise:
            required_expertise.add(ExpertiseType.GENERAL_LINGUISTIC)
            
        return required_expertise

    def _get_priority_value(self, priority: TicketPriority) -> int:
        """Convert priority enum to numeric value for queue ordering"""
        priority_values = {
            TicketPriority.CRITICAL: 1,
            TicketPriority.HIGH: 2,
            TicketPriority.MEDIUM: 3,
            TicketPriority.LOW: 4
        }
        return priority_values.get(priority, 3)

    async def _create_review_session(self, ticket: ReviewTicket, expert_id: str):
        """Create review session in the workflow engine"""
        try:
            # Create review segment from ticket
            review_segment = ReviewSegment(
                segment_id=str(uuid4()),
                original_text=ticket.segment_text,
                start_time=0.0,  # Will be populated from context if available
                end_time=0.0,
                confidence_score=0.5,
                processing_metadata=ticket.processing_context
            )
            
            # Create review session
            session_id = await asyncio.to_thread(
                self.review_workflow.create_review_session,
                segments=[review_segment],
                reviewer_id=expert_id,
                session_type="expert_queue_review",
                priority_level=ticket.priority.value
            )
            
            ticket.review_session_id = session_id
            self.logger.info(f"Created review session {session_id} for ticket {ticket.ticket_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to create review session for ticket {ticket.ticket_id}: {e}")

    async def _notify_ticket_event(self, ticket: ReviewTicket, event_type: str):
        """Send notifications for ticket events"""
        try:
            for callback in self.notification_callbacks:
                await callback(ticket, event_type)
        except Exception as e:
            self.logger.error(f"Error in notification callback: {e}")


# Factory function for easy instantiation
def create_expert_review_queue(review_workflow: Optional[ReviewWorkflowEngine] = None,
                              **config_kwargs) -> ExpertReviewQueue:
    """
    Factory function to create ExpertReviewQueue with sensible defaults.
    Integrates with existing ReviewWorkflowEngine if provided.
    """
    return ExpertReviewQueue(review_workflow=review_workflow, **config_kwargs)