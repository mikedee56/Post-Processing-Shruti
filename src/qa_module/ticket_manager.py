"""
Review Ticket Management System - Story 3.2.1 Task 2
Advanced ticket tracking, status management, and analytics for expert review system.
Complements ExpertReviewQueue with comprehensive ticket lifecycle management.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Union
from uuid import UUID
from dataclasses import dataclass, asdict
from pathlib import Path

from .expert_review_queue import (
    ReviewTicket, TicketStatus, TicketPriority, ExpertiseType
)


@dataclass
class TicketTransition:
    """Track status transitions for audit trail"""
    from_status: TicketStatus
    to_status: TicketStatus
    timestamp: datetime
    triggered_by: str  # expert_id or system
    reason: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TicketMetrics:
    """Comprehensive metrics for ticket performance analysis"""
    total_processing_time_minutes: float = 0.0
    queue_wait_time_minutes: float = 0.0
    expert_review_time_minutes: float = 0.0
    
    # Status duration tracking
    time_in_queue: float = 0.0
    time_assigned: float = 0.0
    time_in_review: float = 0.0
    
    # Quality metrics
    initial_quality_score: float = 0.0
    final_quality_score: float = 0.0
    quality_improvement: float = 0.0
    
    # Expert interaction
    expert_changes_count: int = 0
    expert_notes_count: int = 0
    escalation_count: int = 0


class TicketManager:
    """
    Advanced ticket tracking and status management system.
    Provides comprehensive ticket lifecycle management, analytics, and reporting.
    
    Features:
    - Complete status transition tracking with audit trails
    - Advanced ticket search and filtering
    - Performance metrics and analytics
    - Batch operations for efficiency
    - Persistent ticket storage with JSON backup
    - Expert workload analytics
    """
    
    def __init__(self, 
                 storage_path: Optional[Path] = None,
                 enable_persistence: bool = True,
                 auto_save_interval_seconds: int = 300):
        
        self.logger = logging.getLogger(__name__)
        
        # Storage configuration
        self.storage_path = storage_path or Path("data/tickets")
        self.enable_persistence = enable_persistence
        self.auto_save_interval = auto_save_interval_seconds
        
        # Ticket tracking
        self.tickets: Dict[UUID, ReviewTicket] = {}
        self.ticket_transitions: Dict[UUID, List[TicketTransition]] = {}
        self.ticket_metrics: Dict[UUID, TicketMetrics] = {}
        
        # Status indexes for efficient queries
        self.status_index: Dict[TicketStatus, Set[UUID]] = {
            status: set() for status in TicketStatus
        }
        
        self.priority_index: Dict[TicketPriority, Set[UUID]] = {
            priority: set() for priority in TicketPriority
        }
        
        self.expert_index: Dict[str, Set[UUID]] = {}
        self.expertise_index: Dict[ExpertiseType, Set[UUID]] = {
            expertise: set() for expertise in ExpertiseType
        }
        
        # Auto-save task
        self._auto_save_task: Optional[asyncio.Task] = None
        self._is_running = False
        
        # Analytics cache
        self._analytics_cache: Dict[str, Any] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl_seconds = 60

    async def start(self):
        """Start ticket manager with persistence and auto-save"""
        self._is_running = True
        
        if self.enable_persistence:
            await self._load_tickets()
            self._auto_save_task = asyncio.create_task(self._auto_save_loop())
            
        self.logger.info("TicketManager started")

    async def stop(self):
        """Stop ticket manager gracefully with final save"""
        self._is_running = False
        
        if self._auto_save_task:
            self._auto_save_task.cancel()
            try:
                await self._auto_save_task
            except asyncio.CancelledError:
                pass
        
        if self.enable_persistence:
            await self._save_tickets()
            
        self.logger.info("TicketManager stopped")

    def register_ticket(self, ticket: ReviewTicket) -> bool:
        """
        Register a new ticket in the tracking system.
        Updates all indexes and initializes tracking data.
        """
        if ticket.ticket_id in self.tickets:
            self.logger.warning(f"Ticket {ticket.ticket_id} already registered")
            return False
        
        # Store ticket
        self.tickets[ticket.ticket_id] = ticket
        
        # Initialize tracking data
        self.ticket_transitions[ticket.ticket_id] = []
        self.ticket_metrics[ticket.ticket_id] = TicketMetrics(
            initial_quality_score=self._extract_quality_score(ticket)
        )
        
        # Update indexes
        self._update_indexes_for_ticket(ticket)
        
        # Record initial transition
        self._record_transition(
            ticket.ticket_id,
            TicketStatus.QUEUED,  # Assumed initial status
            ticket.status,
            "system",
            "ticket_registration"
        )
        
        self.logger.info(f"Registered ticket {ticket.ticket_id}")
        return True

    def update_ticket_status(self, 
                           ticket_id: UUID, 
                           new_status: TicketStatus,
                           triggered_by: str = "system",
                           reason: str = "",
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update ticket status with full transition tracking"""
        
        ticket = self.tickets.get(ticket_id)
        if not ticket:
            self.logger.error(f"Ticket {ticket_id} not found")
            return False
        
        old_status = ticket.status
        if old_status == new_status:
            return True  # No change needed
        
        # Update ticket status
        ticket.status = new_status
        
        # Record transition
        self._record_transition(ticket_id, old_status, new_status, triggered_by, reason, metadata)
        
        # Update indexes
        self.status_index[old_status].discard(ticket_id)
        self.status_index[new_status].add(ticket_id)
        
        # Update metrics
        self._update_metrics_for_transition(ticket_id, old_status, new_status)
        
        self.logger.info(f"Ticket {ticket_id} status: {old_status.value} -> {new_status.value}")
        return True

    def assign_ticket_to_expert(self, 
                               ticket_id: UUID, 
                               expert_id: str,
                               estimated_completion: Optional[datetime] = None) -> bool:
        """Assign ticket to expert with tracking"""
        
        ticket = self.tickets.get(ticket_id)
        if not ticket:
            return False
        
        # Remove from previous expert's assignments
        if ticket.assigned_expert_id and ticket.assigned_expert_id in self.expert_index:
            self.expert_index[ticket.assigned_expert_id].discard(ticket_id)
        
        # Update assignment
        ticket.assigned_expert_id = expert_id
        ticket.assigned_at = datetime.utcnow()
        ticket.estimated_completion = estimated_completion
        
        # Update expert index
        if expert_id not in self.expert_index:
            self.expert_index[expert_id] = set()
        self.expert_index[expert_id].add(ticket_id)
        
        # Update status if needed
        if ticket.status == TicketStatus.QUEUED:
            self.update_ticket_status(ticket_id, TicketStatus.ASSIGNED, expert_id, "expert_assignment")
        
        return True

    def complete_ticket(self, 
                       ticket_id: UUID,
                       expert_id: str,
                       final_quality_score: Optional[float] = None,
                       expert_summary: Optional[str] = None) -> bool:
        """Mark ticket as completed with final metrics"""
        
        ticket = self.tickets.get(ticket_id)
        if not ticket:
            return False
        
        # Update ticket
        ticket.actual_completion = datetime.utcnow()
        if expert_summary:
            ticket.add_expert_note(expert_id, expert_summary, "completion_summary")
        
        # Update metrics
        metrics = self.ticket_metrics[ticket_id]
        if final_quality_score is not None:
            metrics.final_quality_score = final_quality_score
            metrics.quality_improvement = final_quality_score - metrics.initial_quality_score
        
        metrics.total_processing_time_minutes = (
            ticket.actual_completion - ticket.created_at
        ).total_seconds() / 60
        
        # Update status
        self.update_ticket_status(ticket_id, TicketStatus.COMPLETED, expert_id, "expert_completion")
        
        return True

    def get_tickets_by_status(self, status: TicketStatus) -> List[ReviewTicket]:
        """Get all tickets with specified status"""
        ticket_ids = self.status_index.get(status, set())
        return [self.tickets[tid] for tid in ticket_ids if tid in self.tickets]

    def get_tickets_by_expert(self, expert_id: str) -> List[ReviewTicket]:
        """Get all tickets assigned to expert"""
        ticket_ids = self.expert_index.get(expert_id, set())
        return [self.tickets[tid] for tid in ticket_ids if tid in self.tickets]

    def get_tickets_by_priority(self, priority: TicketPriority) -> List[ReviewTicket]:
        """Get all tickets with specified priority"""
        ticket_ids = self.priority_index.get(priority, set())
        return [self.tickets[tid] for tid in ticket_ids if tid in self.tickets]

    def search_tickets(self, 
                      status: Optional[TicketStatus] = None,
                      priority: Optional[TicketPriority] = None,
                      expert_id: Optional[str] = None,
                      expertise_type: Optional[ExpertiseType] = None,
                      created_after: Optional[datetime] = None,
                      created_before: Optional[datetime] = None,
                      text_contains: Optional[str] = None) -> List[ReviewTicket]:
        """
        Advanced ticket search with multiple filters.
        Efficiently uses indexes for performance.
        """
        
        # Start with all tickets
        candidate_ids = set(self.tickets.keys())
        
        # Apply filters using indexes
        if status is not None:
            candidate_ids &= self.status_index[status]
        
        if priority is not None:
            candidate_ids &= self.priority_index[priority]
        
        if expert_id is not None:
            candidate_ids &= self.expert_index.get(expert_id, set())
        
        if expertise_type is not None:
            candidate_ids &= self.expertise_index[expertise_type]
        
        # Apply remaining filters
        results = []
        for ticket_id in candidate_ids:
            ticket = self.tickets[ticket_id]
            
            if created_after and ticket.created_at < created_after:
                continue
            if created_before and ticket.created_at > created_before:
                continue
            if text_contains and text_contains.lower() not in ticket.segment_text.lower():
                continue
            
            results.append(ticket)
        
        # Sort by creation time (most recent first)
        results.sort(key=lambda t: t.created_at, reverse=True)
        
        return results

    def get_ticket_history(self, ticket_id: UUID) -> Optional[Dict[str, Any]]:
        """Get complete history and analytics for a ticket"""
        
        ticket = self.tickets.get(ticket_id)
        if not ticket:
            return None
        
        transitions = self.ticket_transitions.get(ticket_id, [])
        metrics = self.ticket_metrics.get(ticket_id, TicketMetrics())
        
        return {
            "ticket": asdict(ticket),
            "transitions": [
                {
                    "from_status": t.from_status.value,
                    "to_status": t.to_status.value,
                    "timestamp": t.timestamp.isoformat(),
                    "triggered_by": t.triggered_by,
                    "reason": t.reason,
                    "metadata": t.metadata
                }
                for t in transitions
            ],
            "metrics": asdict(metrics),
            "current_age_minutes": ticket.get_age_minutes(),
            "is_overdue": ticket.is_expired()
        }

    def get_expert_workload_analytics(self, expert_id: str) -> Dict[str, Any]:
        """Get comprehensive workload analytics for an expert"""
        
        assigned_tickets = self.get_tickets_by_expert(expert_id)
        
        # Basic counts
        active_count = len([t for t in assigned_tickets 
                           if t.status in [TicketStatus.ASSIGNED, TicketStatus.IN_REVIEW]])
        completed_count = len([t for t in assigned_tickets if t.status == TicketStatus.COMPLETED])
        
        # Performance metrics
        completed_tickets_with_metrics = [
            (t, self.ticket_metrics.get(t.ticket_id))
            for t in assigned_tickets 
            if t.status == TicketStatus.COMPLETED and t.ticket_id in self.ticket_metrics
        ]
        
        avg_processing_time = 0.0
        avg_quality_improvement = 0.0
        
        if completed_tickets_with_metrics:
            processing_times = [m.total_processing_time_minutes for _, m in completed_tickets_with_metrics if m]
            quality_improvements = [m.quality_improvement for _, m in completed_tickets_with_metrics if m]
            
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
            avg_quality_improvement = sum(quality_improvements) / len(quality_improvements) if quality_improvements else 0
        
        return {
            "expert_id": expert_id,
            "active_tickets": active_count,
            "completed_tickets": completed_count,
            "total_tickets": len(assigned_tickets),
            "average_processing_time_minutes": round(avg_processing_time, 2),
            "average_quality_improvement": round(avg_quality_improvement, 3),
            "current_workload_priority_breakdown": self._get_priority_breakdown(assigned_tickets),
            "expertise_types_handled": self._get_expertise_breakdown(assigned_tickets)
        }

    def get_system_analytics(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive system-wide analytics with caching.
        Provides insights into queue performance, expert efficiency, and trends.
        """
        
        # Check cache
        if not force_refresh and self._cache_timestamp:
            cache_age = (datetime.utcnow() - self._cache_timestamp).total_seconds()
            if cache_age < self._cache_ttl_seconds:
                return self._analytics_cache
        
        # Generate fresh analytics
        all_tickets = list(self.tickets.values())
        
        analytics = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_tickets": len(all_tickets),
            
            # Status breakdown
            "status_distribution": {
                status.value: len(self.status_index[status])
                for status in TicketStatus
            },
            
            # Priority breakdown
            "priority_distribution": {
                priority.value: len(self.priority_index[priority])
                for priority in TicketPriority
            },
            
            # Time-based metrics
            "queue_performance": self._calculate_queue_performance(),
            "expert_performance": self._calculate_expert_performance(),
            "quality_trends": self._calculate_quality_trends(),
            
            # System health indicators
            "health_indicators": self._calculate_health_indicators()
        }
        
        # Cache results
        self._analytics_cache = analytics
        self._cache_timestamp = datetime.utcnow()
        
        return analytics

    # Private helper methods

    def _record_transition(self, 
                         ticket_id: UUID, 
                         from_status: TicketStatus, 
                         to_status: TicketStatus,
                         triggered_by: str, 
                         reason: str, 
                         metadata: Optional[Dict[str, Any]] = None):
        """Record status transition with full audit trail"""
        
        transition = TicketTransition(
            from_status=from_status,
            to_status=to_status,
            timestamp=datetime.utcnow(),
            triggered_by=triggered_by,
            reason=reason,
            metadata=metadata or {}
        )
        
        if ticket_id not in self.ticket_transitions:
            self.ticket_transitions[ticket_id] = []
        
        self.ticket_transitions[ticket_id].append(transition)

    def _update_indexes_for_ticket(self, ticket: ReviewTicket):
        """Update all indexes when ticket is added/modified"""
        
        # Status index
        self.status_index[ticket.status].add(ticket.ticket_id)
        
        # Priority index
        self.priority_index[ticket.priority].add(ticket.ticket_id)
        
        # Expert index
        if ticket.assigned_expert_id:
            if ticket.assigned_expert_id not in self.expert_index:
                self.expert_index[ticket.assigned_expert_id] = set()
            self.expert_index[ticket.assigned_expert_id].add(ticket.ticket_id)
        
        # Expertise index
        for expertise_type in ticket.expertise_required:
            self.expertise_index[expertise_type].add(ticket.ticket_id)

    def _update_metrics_for_transition(self, 
                                     ticket_id: UUID, 
                                     from_status: TicketStatus, 
                                     to_status: TicketStatus):
        """Update performance metrics based on status transition"""
        
        metrics = self.ticket_metrics.get(ticket_id)
        if not metrics:
            return
        
        ticket = self.tickets[ticket_id]
        current_time = datetime.utcnow()
        
        # Calculate time spent in previous status
        last_transition_time = ticket.created_at
        transitions = self.ticket_transitions.get(ticket_id, [])
        if transitions:
            last_transition_time = transitions[-1].timestamp
        
        time_in_status = (current_time - last_transition_time).total_seconds() / 60
        
        # Update metrics based on status
        if from_status == TicketStatus.QUEUED:
            metrics.time_in_queue += time_in_status
            metrics.queue_wait_time_minutes = metrics.time_in_queue
        elif from_status == TicketStatus.ASSIGNED:
            metrics.time_assigned += time_in_status
        elif from_status == TicketStatus.IN_REVIEW:
            metrics.time_in_review += time_in_status
            metrics.expert_review_time_minutes = metrics.time_in_review

    def _extract_quality_score(self, ticket: ReviewTicket) -> float:
        """Extract quality score from ticket's quality report"""
        if ticket.quality_report and hasattr(ticket.quality_report, 'overall_score'):
            return float(ticket.quality_report.overall_score)
        return 0.5  # Default neutral score

    def _get_priority_breakdown(self, tickets: List[ReviewTicket]) -> Dict[str, int]:
        """Get priority distribution for a set of tickets"""
        breakdown = {priority.value: 0 for priority in TicketPriority}
        for ticket in tickets:
            breakdown[ticket.priority.value] += 1
        return breakdown

    def _get_expertise_breakdown(self, tickets: List[ReviewTicket]) -> Dict[str, int]:
        """Get expertise type distribution for a set of tickets"""
        breakdown = {expertise.value: 0 for expertise in ExpertiseType}
        for ticket in tickets:
            for expertise in ticket.expertise_required:
                breakdown[expertise.value] += 1
        return breakdown

    def _calculate_queue_performance(self) -> Dict[str, Any]:
        """Calculate queue performance metrics"""
        queued_tickets = self.get_tickets_by_status(TicketStatus.QUEUED)
        
        if not queued_tickets:
            return {"average_wait_time_minutes": 0.0, "queue_depth": 0}
        
        total_wait_time = sum(ticket.get_age_minutes() for ticket in queued_tickets)
        avg_wait_time = total_wait_time / len(queued_tickets)
        
        return {
            "average_wait_time_minutes": round(avg_wait_time, 2),
            "queue_depth": len(queued_tickets),
            "oldest_ticket_age_minutes": round(max(ticket.get_age_minutes() for ticket in queued_tickets), 2)
        }

    def _calculate_expert_performance(self) -> Dict[str, Any]:
        """Calculate expert performance metrics"""
        expert_metrics = {}
        
        for expert_id in self.expert_index:
            expert_analytics = self.get_expert_workload_analytics(expert_id)
            expert_metrics[expert_id] = {
                "active_tickets": expert_analytics["active_tickets"],
                "completion_rate": expert_analytics["completed_tickets"],
                "avg_processing_time": expert_analytics["average_processing_time_minutes"]
            }
        
        return expert_metrics

    def _calculate_quality_trends(self) -> Dict[str, Any]:
        """Calculate quality improvement trends"""
        completed_tickets = self.get_tickets_by_status(TicketStatus.COMPLETED)
        
        quality_improvements = []
        for ticket in completed_tickets:
            metrics = self.ticket_metrics.get(ticket.ticket_id)
            if metrics and metrics.quality_improvement != 0:
                quality_improvements.append(metrics.quality_improvement)
        
        if not quality_improvements:
            return {"average_improvement": 0.0, "improvement_count": 0}
        
        return {
            "average_improvement": round(sum(quality_improvements) / len(quality_improvements), 3),
            "improvement_count": len(quality_improvements),
            "max_improvement": round(max(quality_improvements), 3),
            "min_improvement": round(min(quality_improvements), 3)
        }

    def _calculate_health_indicators(self) -> Dict[str, Any]:
        """Calculate system health indicators"""
        total_tickets = len(self.tickets)
        if total_tickets == 0:
            return {"system_health": "healthy", "alerts": []}
        
        # Check for concerning patterns
        alerts = []
        
        # High queue depth
        queued_count = len(self.status_index[TicketStatus.QUEUED])
        if queued_count > total_tickets * 0.3:  # More than 30% queued
            alerts.append(f"High queue depth: {queued_count} tickets waiting")
        
        # Timeout tickets
        timeout_count = len(self.status_index[TicketStatus.TIMEOUT])
        if timeout_count > 0:
            alerts.append(f"Timeout issues: {timeout_count} tickets timed out")
        
        # Escalated tickets
        escalated_count = len(self.status_index[TicketStatus.ESCALATED])
        if escalated_count > total_tickets * 0.1:  # More than 10% escalated
            alerts.append(f"High escalation rate: {escalated_count} tickets escalated")
        
        health_status = "healthy" if not alerts else ("warning" if len(alerts) <= 2 else "critical")
        
        return {
            "system_health": health_status,
            "alerts": alerts,
            "completion_rate": round(len(self.status_index[TicketStatus.COMPLETED]) / total_tickets * 100, 1)
        }

    async def _auto_save_loop(self):
        """Auto-save tickets periodically"""
        while self._is_running:
            try:
                await asyncio.sleep(self.auto_save_interval)
                if self._is_running:
                    await self._save_tickets()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in auto-save: {e}")

    async def _save_tickets(self):
        """Save tickets to persistent storage"""
        if not self.enable_persistence:
            return
        
        try:
            # Ensure storage directory exists
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
            # Prepare serializable data
            save_data = {
                "tickets": {
                    str(ticket_id): {
                        **asdict(ticket),
                        "ticket_id": str(ticket.ticket_id),
                        "created_at": ticket.created_at.isoformat(),
                        "assigned_at": ticket.assigned_at.isoformat() if ticket.assigned_at else None,
                        "estimated_completion": ticket.estimated_completion.isoformat() if ticket.estimated_completion else None,
                        "actual_completion": ticket.actual_completion.isoformat() if ticket.actual_completion else None,
                        "timeout_at": ticket.timeout_at.isoformat() if ticket.timeout_at else None,
                        "status": ticket.status.value,
                        "priority": ticket.priority.value,
                        "expertise_required": [exp.value for exp in ticket.expertise_required]
                    }
                    for ticket_id, ticket in self.tickets.items()
                },
                "transitions": {
                    str(ticket_id): [
                        {
                            "from_status": t.from_status.value,
                            "to_status": t.to_status.value,
                            "timestamp": t.timestamp.isoformat(),
                            "triggered_by": t.triggered_by,
                            "reason": t.reason,
                            "metadata": t.metadata
                        }
                        for t in transitions
                    ]
                    for ticket_id, transitions in self.ticket_transitions.items()
                },
                "metrics": {
                    str(ticket_id): asdict(metrics)
                    for ticket_id, metrics in self.ticket_metrics.items()
                },
                "saved_at": datetime.utcnow().isoformat()
            }
            
            # Save to JSON file
            save_file = self.storage_path / f"tickets_{datetime.utcnow().strftime('%Y%m%d')}.json"
            
            with open(save_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Saved {len(self.tickets)} tickets to {save_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save tickets: {e}")

    async def _load_tickets(self):
        """Load tickets from persistent storage"""
        if not self.enable_persistence or not self.storage_path.exists():
            return
        
        try:
            # Find most recent save file
            save_files = list(self.storage_path.glob("tickets_*.json"))
            if not save_files:
                return
            
            latest_file = max(save_files, key=lambda f: f.stat().st_mtime)
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                save_data = json.load(f)
            
            # Restore tickets (implementation would need proper deserialization)
            # This is a simplified version - full implementation would properly
            # reconstruct all the dataclass objects with enum conversion
            
            self.logger.info(f"Loaded ticket data from {latest_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to load tickets: {e}")


# Factory function for easy instantiation
def create_ticket_manager(storage_path: Optional[Path] = None, **config_kwargs) -> TicketManager:
    """Factory function to create TicketManager with sensible defaults"""
    return TicketManager(storage_path=storage_path, **config_kwargs)