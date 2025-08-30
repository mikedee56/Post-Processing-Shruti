"""
Expert Assignment Engine - Story 3.2.1 Task 3
Advanced expert assignment logic with intelligent matching, load balancing, and expertise optimization.
Complements ExpertReviewQueue with sophisticated assignment algorithms.
"""

import asyncio
import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID

from .expert_review_queue import (
    ReviewTicket, TicketPriority, ExpertiseType, TicketStatus
)
from ..review_workflow.review_workflow_engine import ReviewerProfile, ReviewerRole


class AssignmentStrategy(Enum):
    """Different strategies for expert assignment"""
    ROUND_ROBIN = "round_robin"
    EXPERTISE_MATCH = "expertise_match"
    LOAD_BALANCED = "load_balanced"
    PRIORITY_WEIGHTED = "priority_weighted"
    HYBRID_OPTIMAL = "hybrid_optimal"


@dataclass
class ExpertCapacity:
    """Track expert capacity and workload"""
    expert_id: str
    max_concurrent_tickets: int = 5
    current_load: int = 0
    expertise_types: Set[ExpertiseType] = field(default_factory=set)
    availability_hours: Dict[int, Tuple[int, int]] = field(default_factory=dict)  # day_of_week -> (start_hour, end_hour)
    performance_rating: float = 1.0  # 0.0 to 2.0, where 1.0 is baseline
    specialization_bonus: Dict[ExpertiseType, float] = field(default_factory=dict)
    
    def get_utilization_percentage(self) -> float:
        """Get current utilization as percentage"""
        if self.max_concurrent_tickets == 0:
            return 100.0
        return (self.current_load / self.max_concurrent_tickets) * 100

    def can_accept_ticket(self, required_expertise: Set[ExpertiseType] = None) -> bool:
        """Check if expert can accept a new ticket"""
        if self.current_load >= self.max_concurrent_tickets:
            return False
        
        if required_expertise and not (required_expertise & self.expertise_types):
            return False
            
        return True

    def get_expertise_match_score(self, required_expertise: Set[ExpertiseType]) -> float:
        """Calculate expertise match score (0.0 to 1.0)"""
        if not required_expertise:
            return 1.0
        
        matched_expertise = required_expertise & self.expertise_types
        if not matched_expertise:
            return 0.0
        
        # Base match score
        base_score = len(matched_expertise) / len(required_expertise)
        
        # Apply specialization bonuses
        specialization_boost = 0.0
        for expertise in matched_expertise:
            bonus = self.specialization_bonus.get(expertise, 0.0)
            specialization_boost += bonus
        
        # Normalize specialization boost
        if matched_expertise:
            specialization_boost /= len(matched_expertise)
        
        return min(1.0, base_score + specialization_boost * 0.2)  # Max 20% boost from specialization

    def is_available_now(self) -> bool:
        """Check if expert is available during current time"""
        now = datetime.utcnow()
        day_of_week = now.weekday()  # 0 = Monday
        hour_of_day = now.hour
        
        if day_of_week not in self.availability_hours:
            return True  # Default to available if no schedule specified
        
        start_hour, end_hour = self.availability_hours[day_of_week]
        return start_hour <= hour_of_day < end_hour


@dataclass
class AssignmentResult:
    """Result of expert assignment operation"""
    success: bool
    assigned_expert_id: Optional[str] = None
    assignment_score: float = 0.0
    strategy_used: Optional[AssignmentStrategy] = None
    reasoning: str = ""
    alternative_experts: List[Tuple[str, float]] = field(default_factory=list)  # (expert_id, score)
    estimated_completion_time: Optional[datetime] = None


class ExpertAssignmentEngine:
    """
    Advanced expert assignment engine with intelligent matching algorithms.
    
    Features:
    - Multiple assignment strategies (round-robin, expertise-based, load-balanced, etc.)
    - Performance-based expert ranking
    - Availability scheduling with timezone support
    - Dynamic load balancing with capacity optimization
    - Historical performance analysis for assignment improvement
    - A/B testing framework for assignment strategy optimization
    """
    
    def __init__(self, 
                 default_strategy: AssignmentStrategy = AssignmentStrategy.HYBRID_OPTIMAL,
                 enable_performance_tracking: bool = True,
                 load_balancing_factor: float = 0.3):
        
        self.logger = logging.getLogger(__name__)
        self.default_strategy = default_strategy
        self.enable_performance_tracking = enable_performance_tracking
        self.load_balancing_factor = load_balancing_factor
        
        # Expert management
        self.expert_capacities: Dict[str, ExpertCapacity] = {}
        self.expert_availability: Dict[str, bool] = {}
        
        # Assignment tracking
        self.assignment_history: List[Dict[str, Any]] = []
        self.round_robin_pointer: int = 0
        
        # Performance metrics
        self.assignment_metrics = {
            "total_assignments": 0,
            "successful_assignments": 0,
            "failed_assignments": 0,
            "average_assignment_score": 0.0,
            "strategy_performance": {strategy.value: {"count": 0, "success_rate": 0.0} 
                                   for strategy in AssignmentStrategy}
        }
        
        # Strategy weights for hybrid optimization
        self.strategy_weights = {
            "expertise_match": 0.4,
            "load_balance": 0.3,
            "performance": 0.2,
            "availability": 0.1
        }

    def register_expert(self, 
                       expert_id: str,
                       expertise_types: Set[ExpertiseType],
                       max_concurrent_tickets: int = 5,
                       performance_rating: float = 1.0,
                       availability_schedule: Optional[Dict[int, Tuple[int, int]]] = None,
                       specialization_bonuses: Optional[Dict[ExpertiseType, float]] = None) -> bool:
        """
        Register expert with comprehensive capability profile.
        
        Args:
            expert_id: Unique expert identifier
            expertise_types: Set of expertise areas
            max_concurrent_tickets: Maximum concurrent ticket capacity
            performance_rating: Performance multiplier (0.0 to 2.0, 1.0 = baseline)
            availability_schedule: Weekly schedule {day_of_week: (start_hour, end_hour)}
            specialization_bonuses: Additional scoring for specific expertise areas
        """
        
        capacity = ExpertCapacity(
            expert_id=expert_id,
            max_concurrent_tickets=max_concurrent_tickets,
            expertise_types=expertise_types,
            availability_hours=availability_schedule or {},
            performance_rating=performance_rating,
            specialization_bonus=specialization_bonuses or {}
        )
        
        self.expert_capacities[expert_id] = capacity
        self.expert_availability[expert_id] = True
        
        self.logger.info(f"Registered expert {expert_id} with {len(expertise_types)} expertise areas")
        return True

    def update_expert_load(self, expert_id: str, new_load: int) -> bool:
        """Update expert's current workload"""
        if expert_id not in self.expert_capacities:
            return False
        
        self.expert_capacities[expert_id].current_load = max(0, new_load)
        return True

    def set_expert_availability(self, expert_id: str, is_available: bool) -> bool:
        """Update expert availability status"""
        if expert_id not in self.expert_availability:
            return False
        
        self.expert_availability[expert_id] = is_available
        self.logger.info(f"Expert {expert_id} availability: {is_available}")
        return True

    async def assign_expert(self, 
                           ticket: ReviewTicket,
                           strategy: Optional[AssignmentStrategy] = None,
                           force_assignment: bool = False) -> AssignmentResult:
        """
        Assign expert to ticket using specified or default strategy.
        
        Args:
            ticket: ReviewTicket requiring expert assignment
            strategy: Assignment strategy to use (defaults to instance default)
            force_assignment: Assign even if expert is at capacity (emergency mode)
        """
        
        strategy = strategy or self.default_strategy
        
        self.logger.debug(f"Assigning expert for ticket {ticket.ticket_id} using {strategy.value}")
        
        # Get available experts
        available_experts = self._get_available_experts(ticket.expertise_required, force_assignment)
        
        if not available_experts:
            return AssignmentResult(
                success=False,
                reasoning="No available experts with required expertise"
            )
        
        # Apply assignment strategy
        result = await self._apply_assignment_strategy(ticket, available_experts, strategy)
        
        # Update metrics and history
        self._record_assignment(ticket, result, strategy)
        
        # Update expert load if successful
        if result.success and result.assigned_expert_id:
            self._increment_expert_load(result.assigned_expert_id)
        
        return result

    def release_expert_assignment(self, expert_id: str, ticket_id: UUID) -> bool:
        """Release expert from ticket assignment (decrement load)"""
        if expert_id in self.expert_capacities:
            capacity = self.expert_capacities[expert_id]
            capacity.current_load = max(0, capacity.current_load - 1)
            
            self.logger.debug(f"Released expert {expert_id} from ticket {ticket_id}")
            return True
        return False

    def get_expert_recommendations(self, 
                                 ticket: ReviewTicket,
                                 max_recommendations: int = 5) -> List[Tuple[str, float, str]]:
        """
        Get ranked expert recommendations for a ticket.
        Returns list of (expert_id, score, reasoning) tuples.
        """
        
        available_experts = self._get_available_experts(ticket.expertise_required, force_assignment=True)
        
        if not available_experts:
            return []
        
        recommendations = []
        
        for expert_id in available_experts:
            capacity = self.expert_capacities[expert_id]
            
            # Calculate comprehensive score
            expertise_score = capacity.get_expertise_match_score(ticket.expertise_required)
            load_score = 1.0 - (capacity.get_utilization_percentage() / 100.0)
            performance_score = min(1.0, capacity.performance_rating / 2.0)
            availability_score = 1.0 if capacity.is_available_now() else 0.5
            
            # Weighted composite score
            composite_score = (
                expertise_score * self.strategy_weights["expertise_match"] +
                load_score * self.strategy_weights["load_balance"] +
                performance_score * self.strategy_weights["performance"] +
                availability_score * self.strategy_weights["availability"]
            )
            
            # Generate reasoning
            reasoning_parts = []
            if expertise_score > 0.8:
                reasoning_parts.append("Strong expertise match")
            if load_score > 0.7:
                reasoning_parts.append("Low current workload")
            if performance_score > 0.6:
                reasoning_parts.append("Above average performance")
            if availability_score == 1.0:
                reasoning_parts.append("Currently available")
            
            reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Basic qualification"
            
            recommendations.append((expert_id, composite_score, reasoning))
        
        # Sort by score (descending) and limit results
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:max_recommendations]

    def get_workload_analytics(self) -> Dict[str, Any]:
        """Get comprehensive workload analytics for all experts"""
        
        total_capacity = sum(cap.max_concurrent_tickets for cap in self.expert_capacities.values())
        total_load = sum(cap.current_load for cap in self.expert_capacities.values())
        
        expert_details = {}
        for expert_id, capacity in self.expert_capacities.items():
            expert_details[expert_id] = {
                "current_load": capacity.current_load,
                "max_capacity": capacity.max_concurrent_tickets,
                "utilization_percent": capacity.get_utilization_percentage(),
                "expertise_count": len(capacity.expertise_types),
                "performance_rating": capacity.performance_rating,
                "is_available": self.expert_availability.get(expert_id, False),
                "is_scheduled_available": capacity.is_available_now()
            }
        
        return {
            "system_utilization_percent": (total_load / total_capacity * 100) if total_capacity > 0 else 0,
            "total_experts": len(self.expert_capacities),
            "available_experts": len([e for e, available in self.expert_availability.items() if available]),
            "experts_at_capacity": len([c for c in self.expert_capacities.values() 
                                      if c.current_load >= c.max_concurrent_tickets]),
            "expert_details": expert_details,
            "assignment_metrics": self.assignment_metrics.copy()
        }

    def optimize_strategy_weights(self, target_metric: str = "success_rate") -> Dict[str, float]:
        """
        Analyze assignment history to optimize strategy weights.
        Uses simple performance analysis to suggest weight adjustments.
        """
        
        if len(self.assignment_history) < 50:  # Need sufficient data
            return self.strategy_weights.copy()
        
        # Analyze recent assignments (last 100)
        recent_assignments = self.assignment_history[-100:]
        
        # Calculate performance metrics by strategy component strength
        expertise_heavy = [a for a in recent_assignments if a.get("expertise_score", 0) > 0.8]
        load_balanced = [a for a in recent_assignments if a.get("load_score", 0) > 0.7]
        performance_heavy = [a for a in recent_assignments if a.get("performance_score", 0) > 0.7]
        
        # Calculate success rates
        expertise_success = sum(1 for a in expertise_heavy if a.get("success", False)) / len(expertise_heavy) if expertise_heavy else 0
        load_success = sum(1 for a in load_balanced if a.get("success", False)) / len(load_balanced) if load_balanced else 0
        performance_success = sum(1 for a in performance_heavy if a.get("success", False)) / len(performance_heavy) if performance_heavy else 0
        
        # Suggest weight adjustments (simple heuristic)
        suggested_weights = self.strategy_weights.copy()
        
        if expertise_success > 0.9:
            suggested_weights["expertise_match"] = min(0.5, suggested_weights["expertise_match"] + 0.1)
        if load_success > 0.9:
            suggested_weights["load_balance"] = min(0.4, suggested_weights["load_balance"] + 0.1)
        if performance_success > 0.9:
            suggested_weights["performance"] = min(0.3, suggested_weights["performance"] + 0.1)
        
        # Normalize weights to sum to 1.0
        total_weight = sum(suggested_weights.values())
        if total_weight > 0:
            for key in suggested_weights:
                suggested_weights[key] /= total_weight
        
        self.logger.info(f"Suggested weight optimization: {suggested_weights}")
        return suggested_weights

    # Private methods for assignment logic

    def _get_available_experts(self, 
                             required_expertise: Set[ExpertiseType],
                             force_assignment: bool = False) -> List[str]:
        """Get list of experts available for assignment"""
        
        available = []
        
        for expert_id, capacity in self.expert_capacities.items():
            # Check general availability
            if not self.expert_availability.get(expert_id, False):
                continue
            
            # Check capacity unless forcing
            if not force_assignment and not capacity.can_accept_ticket(required_expertise):
                continue
            
            # Check expertise overlap
            if required_expertise and not (required_expertise & capacity.expertise_types):
                continue
            
            available.append(expert_id)
        
        return available

    async def _apply_assignment_strategy(self, 
                                       ticket: ReviewTicket,
                                       available_experts: List[str],
                                       strategy: AssignmentStrategy) -> AssignmentResult:
        """Apply specific assignment strategy"""
        
        if strategy == AssignmentStrategy.ROUND_ROBIN:
            return self._assign_round_robin(ticket, available_experts)
        
        elif strategy == AssignmentStrategy.EXPERTISE_MATCH:
            return self._assign_expertise_match(ticket, available_experts)
        
        elif strategy == AssignmentStrategy.LOAD_BALANCED:
            return self._assign_load_balanced(ticket, available_experts)
        
        elif strategy == AssignmentStrategy.PRIORITY_WEIGHTED:
            return self._assign_priority_weighted(ticket, available_experts)
        
        elif strategy == AssignmentStrategy.HYBRID_OPTIMAL:
            return self._assign_hybrid_optimal(ticket, available_experts)
        
        else:
            # Default to expertise match
            return self._assign_expertise_match(ticket, available_experts)

    def _assign_round_robin(self, ticket: ReviewTicket, available_experts: List[str]) -> AssignmentResult:
        """Simple round-robin assignment"""
        
        if not available_experts:
            return AssignmentResult(success=False, reasoning="No available experts")
        
        # Get next expert in rotation
        expert_id = available_experts[self.round_robin_pointer % len(available_experts)]
        self.round_robin_pointer += 1
        
        return AssignmentResult(
            success=True,
            assigned_expert_id=expert_id,
            assignment_score=1.0,
            strategy_used=AssignmentStrategy.ROUND_ROBIN,
            reasoning="Round-robin assignment",
            estimated_completion_time=datetime.utcnow() + timedelta(hours=2)
        )

    def _assign_expertise_match(self, ticket: ReviewTicket, available_experts: List[str]) -> AssignmentResult:
        """Assignment based on expertise matching"""
        
        best_expert = None
        best_score = 0.0
        
        for expert_id in available_experts:
            capacity = self.expert_capacities[expert_id]
            expertise_score = capacity.get_expertise_match_score(ticket.expertise_required)
            
            if expertise_score > best_score:
                best_score = expertise_score
                best_expert = expert_id
        
        if best_expert:
            return AssignmentResult(
                success=True,
                assigned_expert_id=best_expert,
                assignment_score=best_score,
                strategy_used=AssignmentStrategy.EXPERTISE_MATCH,
                reasoning=f"Best expertise match (score: {best_score:.2f})",
                estimated_completion_time=self._estimate_completion_time(best_expert, ticket)
            )
        
        return AssignmentResult(success=False, reasoning="No expertise match found")

    def _assign_load_balanced(self, ticket: ReviewTicket, available_experts: List[str]) -> AssignmentResult:
        """Assignment based on load balancing"""
        
        best_expert = None
        lowest_utilization = float('inf')
        
        for expert_id in available_experts:
            capacity = self.expert_capacities[expert_id]
            utilization = capacity.get_utilization_percentage()
            
            if utilization < lowest_utilization:
                lowest_utilization = utilization
                best_expert = expert_id
        
        if best_expert:
            score = 1.0 - (lowest_utilization / 100.0)
            return AssignmentResult(
                success=True,
                assigned_expert_id=best_expert,
                assignment_score=score,
                strategy_used=AssignmentStrategy.LOAD_BALANCED,
                reasoning=f"Load balanced assignment (utilization: {lowest_utilization:.1f}%)",
                estimated_completion_time=self._estimate_completion_time(best_expert, ticket)
            )
        
        return AssignmentResult(success=False, reasoning="No suitable expert for load balancing")

    def _assign_priority_weighted(self, ticket: ReviewTicket, available_experts: List[str]) -> AssignmentResult:
        """Assignment with priority-based weighting"""
        
        # Priority multipliers
        priority_multipliers = {
            TicketPriority.CRITICAL: 2.0,
            TicketPriority.HIGH: 1.5,
            TicketPriority.MEDIUM: 1.0,
            TicketPriority.LOW: 0.8
        }
        
        multiplier = priority_multipliers.get(ticket.priority, 1.0)
        
        best_expert = None
        best_weighted_score = 0.0
        
        for expert_id in available_experts:
            capacity = self.expert_capacities[expert_id]
            
            # Base score from expertise and performance
            expertise_score = capacity.get_expertise_match_score(ticket.expertise_required)
            performance_score = min(1.0, capacity.performance_rating / 2.0)
            base_score = (expertise_score + performance_score) / 2
            
            # Apply priority weighting
            weighted_score = base_score * multiplier
            
            if weighted_score > best_weighted_score:
                best_weighted_score = weighted_score
                best_expert = expert_id
        
        if best_expert:
            return AssignmentResult(
                success=True,
                assigned_expert_id=best_expert,
                assignment_score=min(1.0, best_weighted_score),
                strategy_used=AssignmentStrategy.PRIORITY_WEIGHTED,
                reasoning=f"Priority-weighted assignment (priority: {ticket.priority.value})",
                estimated_completion_time=self._estimate_completion_time(best_expert, ticket)
            )
        
        return AssignmentResult(success=False, reasoning="No suitable expert for priority weighting")

    def _assign_hybrid_optimal(self, ticket: ReviewTicket, available_experts: List[str]) -> AssignmentResult:
        """Hybrid assignment using multiple factors with optimal weighting"""
        
        best_expert = None
        best_score = 0.0
        scoring_details = {}
        
        for expert_id in available_experts:
            capacity = self.expert_capacities[expert_id]
            
            # Calculate individual scores
            expertise_score = capacity.get_expertise_match_score(ticket.expertise_required)
            load_score = 1.0 - (capacity.get_utilization_percentage() / 100.0)
            performance_score = min(1.0, capacity.performance_rating / 2.0)
            availability_score = 1.0 if capacity.is_available_now() else 0.7
            
            # Apply priority boost for high-priority tickets
            priority_boost = 1.0
            if ticket.priority in [TicketPriority.HIGH, TicketPriority.CRITICAL]:
                if expertise_score > 0.8:  # Only boost if expertise is strong
                    priority_boost = 1.2
            
            # Weighted composite score
            composite_score = (
                expertise_score * self.strategy_weights["expertise_match"] +
                load_score * self.strategy_weights["load_balance"] +
                performance_score * self.strategy_weights["performance"] +
                availability_score * self.strategy_weights["availability"]
            ) * priority_boost
            
            scoring_details[expert_id] = {
                "expertise": expertise_score,
                "load": load_score,
                "performance": performance_score,
                "availability": availability_score,
                "composite": composite_score
            }
            
            if composite_score > best_score:
                best_score = composite_score
                best_expert = expert_id
        
        if best_expert:
            details = scoring_details[best_expert]
            reasoning = f"Hybrid optimal: expertise={details['expertise']:.2f}, load={details['load']:.2f}, performance={details['performance']:.2f}"
            
            return AssignmentResult(
                success=True,
                assigned_expert_id=best_expert,
                assignment_score=min(1.0, best_score),
                strategy_used=AssignmentStrategy.HYBRID_OPTIMAL,
                reasoning=reasoning,
                alternative_experts=[(eid, scores["composite"]) for eid, scores in 
                                   sorted(scoring_details.items(), key=lambda x: x[1]["composite"], reverse=True)[1:4]],
                estimated_completion_time=self._estimate_completion_time(best_expert, ticket)
            )
        
        return AssignmentResult(success=False, reasoning="No suitable expert found in hybrid analysis")

    def _estimate_completion_time(self, expert_id: str, ticket: ReviewTicket) -> datetime:
        """Estimate completion time based on expert performance and ticket complexity"""
        
        capacity = self.expert_capacities[expert_id]
        
        # Base time estimates by expertise (in hours)
        base_times = {
            ExpertiseType.GENERAL_LINGUISTIC: 1.5,
            ExpertiseType.SANSKRIT_GRAMMAR: 2.0,
            ExpertiseType.HINDI_PHONETICS: 1.8,
            ExpertiseType.IAST_TRANSLITERATION: 1.2,
            ExpertiseType.SCRIPTURAL_CONTEXT: 2.5,
            ExpertiseType.SEMANTIC_ANALYSIS: 3.0
        }
        
        # Calculate base time from required expertise
        if ticket.expertise_required:
            max_base_time = max(base_times.get(exp, 1.5) for exp in ticket.expertise_required)
        else:
            max_base_time = 1.5
        
        # Apply performance modifier
        performance_modifier = 2.0 - capacity.performance_rating  # Higher rating = faster completion
        
        # Apply load modifier (higher load = slower completion)
        load_modifier = 1.0 + (capacity.get_utilization_percentage() / 100.0) * 0.5
        
        # Apply priority modifier
        priority_modifiers = {
            TicketPriority.CRITICAL: 0.7,  # Rush job
            TicketPriority.HIGH: 0.85,
            TicketPriority.MEDIUM: 1.0,
            TicketPriority.LOW: 1.3  # Less urgent, may take longer
        }
        priority_modifier = priority_modifiers.get(ticket.priority, 1.0)
        
        # Calculate final estimated hours
        estimated_hours = max_base_time * performance_modifier * load_modifier * priority_modifier
        
        # Add current time
        return datetime.utcnow() + timedelta(hours=estimated_hours)

    def _increment_expert_load(self, expert_id: str):
        """Increment expert's current load"""
        if expert_id in self.expert_capacities:
            self.expert_capacities[expert_id].current_load += 1

    def _record_assignment(self, 
                          ticket: ReviewTicket, 
                          result: AssignmentResult, 
                          strategy: AssignmentStrategy):
        """Record assignment for metrics and learning"""
        
        assignment_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "ticket_id": str(ticket.ticket_id),
            "strategy": strategy.value,
            "success": result.success,
            "assigned_expert": result.assigned_expert_id,
            "assignment_score": result.assignment_score,
            "ticket_priority": ticket.priority.value,
            "expertise_required": [exp.value for exp in ticket.expertise_required]
        }
        
        self.assignment_history.append(assignment_record)
        
        # Keep only recent history (last 1000 assignments)
        if len(self.assignment_history) > 1000:
            self.assignment_history = self.assignment_history[-1000:]
        
        # Update metrics
        self.assignment_metrics["total_assignments"] += 1
        if result.success:
            self.assignment_metrics["successful_assignments"] += 1
        else:
            self.assignment_metrics["failed_assignments"] += 1
        
        # Update strategy performance
        strategy_stats = self.assignment_metrics["strategy_performance"][strategy.value]
        strategy_stats["count"] += 1
        strategy_stats["success_rate"] = (
            len([a for a in self.assignment_history[-100:] 
                if a["strategy"] == strategy.value and a["success"]]) /
            max(1, len([a for a in self.assignment_history[-100:] if a["strategy"] == strategy.value]))
        )
        
        # Update average assignment score
        successful_scores = [a["assignment_score"] for a in self.assignment_history[-100:] if a["success"]]
        if successful_scores:
            self.assignment_metrics["average_assignment_score"] = sum(successful_scores) / len(successful_scores)


# Factory function for easy instantiation
def create_expert_assignment_engine(**config_kwargs) -> ExpertAssignmentEngine:
    """Factory function to create ExpertAssignmentEngine with sensible defaults"""
    return ExpertAssignmentEngine(**config_kwargs)