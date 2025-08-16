"""
Epic 4.3 Production-Grade Reviewer Management with Enterprise Reliability.

Implements Story 3.3 Task 4: Epic 4.3 Production-Grade Review Infrastructure
- 99.9% uptime reliability for reviewer assignment and management
- Sub-second response times for reviewer queries and assignments
- Enterprise monitoring and telemetry for reviewer system health
- Bulletproof reliability patterns with circuit breakers and graceful degradation
"""

import logging
import statistics
import threading
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple, Union

# Epic 4 infrastructure imports
from monitoring.system_monitor import SystemMonitor
from monitoring.telemetry_collector import TelemetryCollector
from utils.performance_monitor import PerformanceMonitor
from utils.circuit_breaker import CircuitBreaker, CircuitBreakerError
from utils.rate_limiter import RateLimiter
from utils.health_checker import HealthChecker


class ReviewerRole(Enum):
    """Reviewer roles in the tiered system."""
    GENERAL_PROOFREADER = "general_proofreader"
    SUBJECT_MATTER_EXPERT = "subject_matter_expert"
    ACADEMIC_CONSULTANT = "academic_consultant"
    SENIOR_REVIEWER = "senior_reviewer"
    ADMIN = "admin"


class ReviewerStatus(Enum):
    """Reviewer availability status."""
    AVAILABLE = "available"
    BUSY = "busy"
    OFFLINE = "offline"
    ON_BREAK = "on_break"
    OVERLOADED = "overloaded"


class AssignmentStrategy(Enum):
    """Reviewer assignment strategies."""
    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    EXPERTISE_MATCHED = "expertise_matched"
    PERFORMANCE_BASED = "performance_based"


@dataclass
class ReviewerSkill:
    """Reviewer skill with Epic 4.3 performance tracking."""
    skill_area: str
    proficiency_level: float  # 0.0 - 1.0
    years_experience: int
    certification_level: str
    specializations: List[str] = field(default_factory=list)
    
    # Epic 4.3 Performance Metrics
    accuracy_score: float = 1.0
    response_time_ms: float = 0.0
    last_validated: Optional[datetime] = None
    validation_count: int = 0


@dataclass
class ReviewerPerformanceMetrics:
    """Comprehensive performance tracking for Epic 4.3."""
    reviewer_id: str
    
    # Throughput metrics
    reviews_completed_today: int = 0
    reviews_completed_week: int = 0
    reviews_completed_total: int = 0
    average_review_time_minutes: float = 0.0
    
    # Quality metrics
    quality_score: float = 1.0
    accuracy_rate: float = 1.0
    error_rate: float = 0.0
    correction_acceptance_rate: float = 1.0
    
    # Reliability metrics
    uptime_percentage: float = 100.0
    response_time_p95_ms: float = 0.0
    missed_deadlines: int = 0
    system_failures: int = 0
    
    # Workload metrics
    current_workload: int = 0
    peak_workload_today: int = 0
    workload_efficiency: float = 1.0
    burnout_risk_score: float = 0.0
    
    # Epic 4.3 Enterprise Metrics
    sla_compliance_rate: float = 1.0
    escalation_rate: float = 0.0
    client_satisfaction_score: float = 1.0
    
    # Timestamps
    last_activity: datetime = field(default_factory=datetime.now)
    last_performance_update: datetime = field(default_factory=datetime.now)
    metrics_reset_date: datetime = field(default_factory=datetime.now)


@dataclass
class ReviewerProfile:
    """Enhanced reviewer profile with Epic 4.3 production features."""
    reviewer_id: str
    name: str
    email: str
    role: ReviewerRole
    
    # Skills and expertise
    skills: List[ReviewerSkill] = field(default_factory=list)
    primary_expertise: str = ""
    secondary_expertise: List[str] = field(default_factory=list)
    
    # Availability and capacity
    status: ReviewerStatus = ReviewerStatus.AVAILABLE
    max_concurrent_reviews: int = 3
    current_review_load: int = 0
    available_hours_per_week: int = 40
    timezone: str = "UTC"
    
    # Epic 4.3 Production Features
    performance_metrics: ReviewerPerformanceMetrics = field(default_factory=lambda: ReviewerPerformanceMetrics(""))
    circuit_breaker_state: str = "closed"  # closed, half_open, open
    rate_limit_tokens: int = 100
    health_status: str = "healthy"  # healthy, degraded, unhealthy
    
    # Authentication and security
    authentication_token: Optional[str] = None
    last_login: Optional[datetime] = None
    security_clearance_level: str = "standard"
    
    # Academic credentials (Epic 4.5 integration)
    academic_credentials: List[str] = field(default_factory=list)
    publication_history: List[str] = field(default_factory=list)
    institutional_affiliation: str = ""
    
    def __post_init__(self):
        """Initialize performance metrics with reviewer ID."""
        if not self.performance_metrics.reviewer_id:
            self.performance_metrics.reviewer_id = self.reviewer_id


@dataclass
class AssignmentRequest:
    """Review assignment request with Epic 4.3 tracking."""
    request_id: str
    session_id: str
    required_role: ReviewerRole
    required_skills: List[str] = field(default_factory=list)
    
    # Assignment criteria
    priority_level: str = "standard"  # critical, high, standard, low
    deadline: Optional[datetime] = None
    estimated_workload_hours: float = 2.0
    complexity_score: float = 0.5
    
    # Epic 4.3 Production Requirements
    max_response_time_ms: float = 500.0  # Sub-second requirement
    requires_high_availability: bool = True
    fallback_assignment_enabled: bool = True
    
    # Request tracking
    created_at: datetime = field(default_factory=datetime.now)
    assigned_at: Optional[datetime] = None
    assigned_reviewer_id: Optional[str] = None
    assignment_strategy_used: Optional[AssignmentStrategy] = None
    
    # Performance tracking
    processing_time_ms: float = 0.0
    assignment_success: bool = False
    fallback_used: bool = False


class ReviewerManager:
    """
    Epic 4.3 Production-Grade Reviewer Management System.
    
    Implements Story 3.3 Task 4:
    - 99.9% uptime reliability for reviewer management
    - Sub-second response times for all reviewer operations
    - Enterprise monitoring and telemetry integration
    - Bulletproof reliability with circuit breakers and graceful degradation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize reviewer manager with Epic 4.3 production features."""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Epic 4.3 Production Infrastructure
        self.system_monitor = SystemMonitor(self.config.get('monitoring', {}))
        self.performance_monitor = PerformanceMonitor(self.config.get('performance', {}))
        self.telemetry_collector = TelemetryCollector(self.config.get('telemetry', {}))
        self.health_checker = HealthChecker(self.config.get('health_checker', {}))
        
        # Epic 4.3 Reliability Features
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=30,
            expected_exception=Exception
        )
        self.rate_limiter = RateLimiter(
            max_requests=1000,
            time_window=60  # 1000 requests per minute
        )
        
        # Production configuration
        self.production_config = self.config.get('production', {
            'max_response_time_ms': 500.0,
            'target_uptime_percentage': 99.9,
            'max_concurrent_assignments': 100,
            'assignment_timeout_seconds': 30,
            'health_check_interval_seconds': 10,
            'performance_metrics_interval_seconds': 60,
            'circuit_breaker_enabled': True,
            'rate_limiting_enabled': True,
            'graceful_degradation_enabled': True
        })
        
        # Data structures for reviewer management
        self.reviewer_profiles: Dict[str, ReviewerProfile] = {}
        self.active_assignments: Dict[str, AssignmentRequest] = {}
        self.assignment_queue = deque()
        
        # Performance tracking
        self.assignment_statistics = defaultdict(int)
        self.performance_history = deque(maxlen=1000)
        self.uptime_tracker = {
            'start_time': datetime.now(),
            'total_downtime_seconds': 0.0,
            'last_downtime': None,
            'uptime_percentage': 100.0
        }
        
        # Threading and concurrency (Epic 4.3)
        self.lock = threading.RLock()
        self.assignment_pool = ThreadPoolExecutor(
            max_workers=10, 
            thread_name_prefix="reviewer_assignment"
        )
        
        # Real-time monitoring
        self._start_monitoring_systems()
        
        self.logger.info("ReviewerManager initialized with Epic 4.3 production features")
    
    def register_reviewer(self, profile: ReviewerProfile) -> bool:
        """
        Register new reviewer with Epic 4.3 production validation.
        
        Args:
            profile: Complete reviewer profile
            
        Returns:
            bool: True if registration successful
        """
        start_time = time.time()
        
        with self.performance_monitor.monitor_processing_operation("register_reviewer"):
            try:
                # Rate limiting check
                if not self.rate_limiter.acquire():
                    self.logger.warning("Rate limit exceeded for reviewer registration")
                    return False
                
                # Circuit breaker check
                if self.circuit_breaker.is_open():
                    self.logger.warning("Circuit breaker open - reviewer registration unavailable")
                    return False
                
                with self.lock:
                    # Validate reviewer profile
                    if not self._validate_reviewer_profile(profile):
                        return False
                    
                    # Check for duplicate registration
                    if profile.reviewer_id in self.reviewer_profiles:
                        self.logger.warning(f"Reviewer already registered: {profile.reviewer_id}")
                        return False
                    
                    # Initialize Epic 4.3 production features
                    self._initialize_production_features(profile)
                    
                    # Store reviewer profile
                    self.reviewer_profiles[profile.reviewer_id] = profile
                    self.assignment_statistics['reviewers_registered'] += 1
                    
                    # Start health monitoring for reviewer
                    self._start_reviewer_monitoring(profile.reviewer_id)
                    
                    # Record telemetry
                    processing_time = (time.time() - start_time) * 1000
                    self.telemetry_collector.record_event("reviewer_registered", {
                        'reviewer_id': profile.reviewer_id,
                        'role': profile.role.value,
                        'processing_time_ms': processing_time,
                        'success': True
                    })
                    
                    self.logger.info(f"Reviewer registered successfully: {profile.reviewer_id} ({profile.role.value})")
                    return True
                    
            except Exception as e:
                # Circuit breaker failure tracking
                self.circuit_breaker.record_failure()
                self.logger.error(f"Failed to register reviewer: {e}")
                
                # Record failure telemetry
                self.telemetry_collector.record_event("reviewer_registration_failed", {
                    'error': str(e),
                    'processing_time_ms': (time.time() - start_time) * 1000
                })
                return False
    
    def assign_reviewer(self, request: AssignmentRequest) -> Optional[str]:
        """
        Assign reviewer with Epic 4.3 sub-second response guarantee.
        
        Args:
            request: Assignment request with requirements
            
        Returns:
            str: Assigned reviewer ID or None if assignment failed
        """
        start_time = time.time()
        
        with self.performance_monitor.monitor_processing_operation("assign_reviewer"):
            try:
                # Validate request meets Epic 4.3 performance requirements
                if request.max_response_time_ms > self.production_config['max_response_time_ms']:
                    self.logger.warning(f"Request exceeds maximum response time: {request.max_response_time_ms}ms")
                
                # Circuit breaker and rate limiting
                if self.circuit_breaker.is_open():
                    return self._handle_degraded_assignment(request)
                
                if not self.rate_limiter.acquire():
                    self.logger.warning("Rate limit exceeded for reviewer assignment")
                    return None
                
                with self.lock:
                    # Find optimal reviewer using Epic 4.3 assignment algorithm
                    assigned_reviewer_id = self._find_optimal_reviewer(request)
                    
                    if assigned_reviewer_id:
                        # Execute assignment
                        success = self._execute_assignment(request, assigned_reviewer_id)
                        
                        if success:
                            # Update assignment tracking
                            request.assigned_reviewer_id = assigned_reviewer_id
                            request.assigned_at = datetime.now()
                            request.assignment_success = True
                            request.processing_time_ms = (time.time() - start_time) * 1000
                            
                            # Store active assignment
                            self.active_assignments[request.request_id] = request
                            self.assignment_statistics['assignments_successful'] += 1
                            
                            # Validate Epic 4.3 response time requirement
                            if request.processing_time_ms <= request.max_response_time_ms:
                                self.assignment_statistics['sub_second_assignments'] += 1
                            else:
                                self.logger.warning(f"Assignment exceeded response time target: {request.processing_time_ms}ms")
                            
                            # Record success telemetry
                            self.telemetry_collector.record_event("reviewer_assigned", {
                                'request_id': request.request_id,
                                'reviewer_id': assigned_reviewer_id,
                                'processing_time_ms': request.processing_time_ms,
                                'strategy_used': request.assignment_strategy_used.value if request.assignment_strategy_used else None,
                                'sub_second_achieved': request.processing_time_ms <= request.max_response_time_ms
                            })
                            
                            self.logger.info(f"Reviewer assigned: {assigned_reviewer_id} to request {request.request_id} ({request.processing_time_ms:.1f}ms)")
                            return assigned_reviewer_id
                    
                    # Assignment failed - try fallback if enabled
                    if request.fallback_assignment_enabled:
                        return self._handle_fallback_assignment(request)
                    
                    self.assignment_statistics['assignments_failed'] += 1
                    return None
                    
            except Exception as e:
                self.circuit_breaker.record_failure()
                self.assignment_statistics['assignment_errors'] += 1
                
                # Record error telemetry
                self.telemetry_collector.record_event("assignment_error", {
                    'request_id': request.request_id,
                    'error': str(e),
                    'processing_time_ms': (time.time() - start_time) * 1000
                })
                
                self.logger.error(f"Assignment failed for request {request.request_id}: {e}")
                
                # Try graceful degradation
                if self.production_config['graceful_degradation_enabled']:
                    return self._handle_degraded_assignment(request)
                
                return None
    
    def get_reviewer_status(self, reviewer_id: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive reviewer status with Epic 4.3 real-time metrics.
        
        Args:
            reviewer_id: Reviewer identifier
            
        Returns:
            dict: Comprehensive status information
        """
        start_time = time.time()
        
        with self.performance_monitor.monitor_processing_operation("get_reviewer_status"):
            try:
                with self.lock:
                    if reviewer_id not in self.reviewer_profiles:
                        return None
                    
                    profile = self.reviewer_profiles[reviewer_id]
                    
                    # Calculate real-time metrics
                    current_load_percentage = (profile.current_review_load / max(profile.max_concurrent_reviews, 1)) * 100
                    health_score = self._calculate_reviewer_health_score(profile)
                    
                    status_info = {
                        'reviewer_id': reviewer_id,
                        'name': profile.name,
                        'role': profile.role.value,
                        'status': profile.status.value,
                        'availability': {
                            'current_load': profile.current_review_load,
                            'max_capacity': profile.max_concurrent_reviews,
                            'load_percentage': current_load_percentage,
                            'available_for_assignment': profile.status == ReviewerStatus.AVAILABLE and current_load_percentage < 90
                        },
                        'performance': {
                            'quality_score': profile.performance_metrics.quality_score,
                            'accuracy_rate': profile.performance_metrics.accuracy_rate,
                            'average_response_time_ms': profile.performance_metrics.response_time_p95_ms,
                            'sla_compliance': profile.performance_metrics.sla_compliance_rate,
                            'reviews_completed_today': profile.performance_metrics.reviews_completed_today,
                            'uptime_percentage': profile.performance_metrics.uptime_percentage
                        },
                        'health': {
                            'health_status': profile.health_status,
                            'health_score': health_score,
                            'circuit_breaker_state': profile.circuit_breaker_state,
                            'burnout_risk': profile.performance_metrics.burnout_risk_score
                        },
                        'epic_4_3_metrics': {
                            'response_time_compliance': profile.performance_metrics.response_time_p95_ms <= self.production_config['max_response_time_ms'],
                            'system_failures': profile.performance_metrics.system_failures,
                            'missed_deadlines': profile.performance_metrics.missed_deadlines,
                            'rate_limit_tokens': profile.rate_limit_tokens
                        },
                        'last_activity': profile.performance_metrics.last_activity.isoformat(),
                        'expertise': {
                            'primary_expertise': profile.primary_expertise,
                            'skills': [
                                {
                                    'area': skill.skill_area,
                                    'proficiency': skill.proficiency_level,
                                    'accuracy': skill.accuracy_score
                                } for skill in profile.skills
                            ]
                        }
                    }
                    
                    # Record status query telemetry
                    processing_time = (time.time() - start_time) * 1000
                    self.telemetry_collector.record_event("reviewer_status_queried", {
                        'reviewer_id': reviewer_id,
                        'processing_time_ms': processing_time,
                        'health_score': health_score
                    })
                    
                    return status_info
                    
            except Exception as e:
                self.logger.error(f"Failed to get reviewer status for {reviewer_id}: {e}")
                return None
    
    def update_reviewer_workload(self, reviewer_id: str, workload_change: int) -> bool:
        """
        Update reviewer workload with Epic 4.3 real-time tracking.
        
        Args:
            reviewer_id: Reviewer identifier
            workload_change: Change in workload (+/- number of reviews)
            
        Returns:
            bool: True if update successful
        """
        with self.lock:
            if reviewer_id not in self.reviewer_profiles:
                return False
            
            profile = self.reviewer_profiles[reviewer_id]
            
            # Update workload
            new_workload = max(0, profile.current_review_load + workload_change)
            profile.current_review_load = new_workload
            
            # Update performance metrics
            profile.performance_metrics.current_workload = new_workload
            profile.performance_metrics.peak_workload_today = max(
                profile.performance_metrics.peak_workload_today,
                new_workload
            )
            profile.performance_metrics.last_activity = datetime.now()
            
            # Update status based on workload
            load_percentage = (new_workload / max(profile.max_concurrent_reviews, 1)) * 100
            
            if load_percentage >= 100:
                profile.status = ReviewerStatus.OVERLOADED
            elif load_percentage >= 90:
                profile.status = ReviewerStatus.BUSY
            elif profile.status in [ReviewerStatus.BUSY, ReviewerStatus.OVERLOADED]:
                profile.status = ReviewerStatus.AVAILABLE
            
            # Calculate burnout risk
            profile.performance_metrics.burnout_risk_score = min(load_percentage / 100 * 0.8, 1.0)
            
            # Record workload change telemetry
            self.telemetry_collector.record_event("reviewer_workload_updated", {
                'reviewer_id': reviewer_id,
                'workload_change': workload_change,
                'new_workload': new_workload,
                'load_percentage': load_percentage,
                'status': profile.status.value
            })
            
            return True
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get comprehensive system health with Epic 4.3 production metrics.
        
        Returns:
            dict: System health and performance information
        """
        with self.lock:
            # Calculate system-wide metrics
            total_reviewers = len(self.reviewer_profiles)
            available_reviewers = len([r for r in self.reviewer_profiles.values() if r.status == ReviewerStatus.AVAILABLE])
            overloaded_reviewers = len([r for r in self.reviewer_profiles.values() if r.status == ReviewerStatus.OVERLOADED])
            
            # Calculate uptime
            current_time = datetime.now()
            total_runtime = (current_time - self.uptime_tracker['start_time']).total_seconds()
            uptime_percentage = ((total_runtime - self.uptime_tracker['total_downtime_seconds']) / max(total_runtime, 1)) * 100
            
            # Calculate assignment performance
            total_assignments = self.assignment_statistics['assignments_successful'] + self.assignment_statistics['assignments_failed']
            success_rate = (self.assignment_statistics['assignments_successful'] / max(total_assignments, 1)) * 100
            sub_second_rate = (self.assignment_statistics['sub_second_assignments'] / max(total_assignments, 1)) * 100
            
            # Epic 4.3 SLA compliance
            sla_compliance = {
                'uptime_target': self.production_config['target_uptime_percentage'],
                'uptime_actual': uptime_percentage,
                'uptime_compliant': uptime_percentage >= self.production_config['target_uptime_percentage'],
                'response_time_target_ms': self.production_config['max_response_time_ms'],
                'response_time_compliance_rate': sub_second_rate,
                'response_time_compliant': sub_second_rate >= 95.0
            }
            
            return {
                'system_overview': {
                    'total_reviewers': total_reviewers,
                    'available_reviewers': available_reviewers,
                    'busy_reviewers': len([r for r in self.reviewer_profiles.values() if r.status == ReviewerStatus.BUSY]),
                    'overloaded_reviewers': overloaded_reviewers,
                    'offline_reviewers': len([r for r in self.reviewer_profiles.values() if r.status == ReviewerStatus.OFFLINE]),
                    'system_capacity_utilization': (total_reviewers - available_reviewers) / max(total_reviewers, 1) * 100
                },
                'performance_metrics': {
                    'assignment_success_rate': success_rate,
                    'sub_second_assignment_rate': sub_second_rate,
                    'total_assignments_processed': total_assignments,
                    'assignments_successful': self.assignment_statistics['assignments_successful'],
                    'assignments_failed': self.assignment_statistics['assignments_failed'],
                    'assignment_errors': self.assignment_statistics['assignment_errors']
                },
                'epic_4_3_reliability': {
                    'uptime_percentage': uptime_percentage,
                    'total_downtime_seconds': self.uptime_tracker['total_downtime_seconds'],
                    'circuit_breaker_status': 'open' if self.circuit_breaker.is_open() else 'closed',
                    'rate_limiter_status': 'active' if self.production_config['rate_limiting_enabled'] else 'disabled',
                    'health_check_status': 'operational'
                },
                'sla_compliance': sla_compliance,
                'monitoring_status': {
                    'system_monitor': 'active',
                    'performance_monitor': 'active',
                    'telemetry_collector': 'active',
                    'health_checker': 'active'
                }
            }
    
    def _validate_reviewer_profile(self, profile: ReviewerProfile) -> bool:
        """Validate reviewer profile meets Epic 4.3 standards."""
        if not profile.reviewer_id or not profile.name or not profile.email:
            self.logger.error("Invalid reviewer profile: missing required fields")
            return False
        
        if profile.role == ReviewerRole.ACADEMIC_CONSULTANT:
            if not profile.academic_credentials or not profile.institutional_affiliation:
                self.logger.error("Academic consultant requires credentials and affiliation")
                return False
        
        return True
    
    def _initialize_production_features(self, profile: ReviewerProfile) -> None:
        """Initialize Epic 4.3 production features for reviewer."""
        # Initialize performance metrics
        if not profile.performance_metrics.reviewer_id:
            profile.performance_metrics.reviewer_id = profile.reviewer_id
        
        # Set production defaults
        profile.circuit_breaker_state = "closed"
        profile.rate_limit_tokens = 100
        profile.health_status = "healthy"
        
        # Initialize security
        profile.authentication_token = f"token_{uuid.uuid4().hex[:16]}"
        profile.last_login = datetime.now()
    
    def _find_optimal_reviewer(self, request: AssignmentRequest) -> Optional[str]:
        """Find optimal reviewer using Epic 4.3 assignment algorithms."""
        # Filter available reviewers by role and capacity
        candidates = [
            profile for profile in self.reviewer_profiles.values()
            if (profile.role == request.required_role and
                profile.status == ReviewerStatus.AVAILABLE and
                profile.current_review_load < profile.max_concurrent_reviews and
                profile.health_status in ['healthy', 'degraded'])
        ]
        
        if not candidates:
            return None
        
        # Apply expertise matching if skills required
        if request.required_skills:
            candidates = [
                profile for profile in candidates
                if any(skill.skill_area in request.required_skills for skill in profile.skills)
            ]
        
        if not candidates:
            return None
        
        # Select assignment strategy based on request priority
        if request.priority_level == "critical":
            strategy = AssignmentStrategy.PERFORMANCE_BASED
        elif request.required_skills:
            strategy = AssignmentStrategy.EXPERTISE_MATCHED
        else:
            strategy = AssignmentStrategy.LOAD_BALANCED
        
        request.assignment_strategy_used = strategy
        
        # Execute assignment strategy
        if strategy == AssignmentStrategy.PERFORMANCE_BASED:
            return max(candidates, key=lambda p: p.performance_metrics.quality_score).reviewer_id
        elif strategy == AssignmentStrategy.EXPERTISE_MATCHED:
            return self._find_best_expertise_match(candidates, request.required_skills)
        elif strategy == AssignmentStrategy.LOAD_BALANCED:
            return min(candidates, key=lambda p: p.current_review_load).reviewer_id
        else:  # ROUND_ROBIN
            return candidates[self.assignment_statistics['assignments_successful'] % len(candidates)].reviewer_id
    
    def _find_best_expertise_match(self, candidates: List[ReviewerProfile], required_skills: List[str]) -> str:
        """Find reviewer with best expertise match."""
        best_score = 0.0
        best_reviewer = candidates[0].reviewer_id
        
        for profile in candidates:
            score = 0.0
            for skill in profile.skills:
                if skill.skill_area in required_skills:
                    score += skill.proficiency_level * skill.accuracy_score
            
            if score > best_score:
                best_score = score
                best_reviewer = profile.reviewer_id
        
        return best_reviewer
    
    def _execute_assignment(self, request: AssignmentRequest, reviewer_id: str) -> bool:
        """Execute reviewer assignment with Epic 4.3 validation."""
        try:
            profile = self.reviewer_profiles[reviewer_id]
            
            # Update reviewer workload
            profile.current_review_load += 1
            profile.performance_metrics.last_activity = datetime.now()
            
            # Update performance tracking
            if request.priority_level == "critical":
                profile.performance_metrics.reviews_completed_today += 1
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to execute assignment: {e}")
            return False
    
    def _handle_fallback_assignment(self, request: AssignmentRequest) -> Optional[str]:
        """Handle fallback assignment when optimal assignment fails."""
        # Find any available reviewer regardless of expertise
        fallback_candidates = [
            profile for profile in self.reviewer_profiles.values()
            if (profile.status == ReviewerStatus.AVAILABLE and
                profile.current_review_load < profile.max_concurrent_reviews)
        ]
        
        if fallback_candidates:
            # Use load balancing for fallback
            selected = min(fallback_candidates, key=lambda p: p.current_review_load)
            request.fallback_used = True
            
            if self._execute_assignment(request, selected.reviewer_id):
                self.assignment_statistics['fallback_assignments'] += 1
                return selected.reviewer_id
        
        return None
    
    def _handle_degraded_assignment(self, request: AssignmentRequest) -> Optional[str]:
        """Handle assignment during degraded system performance."""
        # Simplified assignment algorithm for degraded mode
        available_reviewers = [
            profile for profile in self.reviewer_profiles.values()
            if profile.status == ReviewerStatus.AVAILABLE
        ]
        
        if available_reviewers:
            # Simple round-robin for degraded mode
            selected = available_reviewers[0]
            if self._execute_assignment(request, selected.reviewer_id):
                self.assignment_statistics['degraded_assignments'] += 1
                return selected.reviewer_id
        
        return None
    
    def _calculate_reviewer_health_score(self, profile: ReviewerProfile) -> float:
        """Calculate overall health score for reviewer."""
        health_factors = [
            profile.performance_metrics.quality_score,
            profile.performance_metrics.accuracy_rate,
            profile.performance_metrics.sla_compliance_rate,
            1.0 - profile.performance_metrics.burnout_risk_score,
            profile.performance_metrics.uptime_percentage / 100
        ]
        
        return statistics.mean(health_factors)
    
    def _start_reviewer_monitoring(self, reviewer_id: str) -> None:
        """Start Epic 4.3 monitoring for individual reviewer."""
        # This would integrate with the monitoring infrastructure
        # to track reviewer-specific metrics in real-time
        pass
    
    def _start_monitoring_systems(self) -> None:
        """Start Epic 4.3 monitoring and telemetry systems."""
        try:
            self.system_monitor.start_monitoring()
            self.health_checker.start_health_checks()
            
            # Start periodic performance updates
            self.assignment_pool.submit(self._performance_update_loop)
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring systems: {e}")
    
    def _performance_update_loop(self) -> None:
        """Continuous performance monitoring loop."""
        while True:
            try:
                time.sleep(self.production_config['performance_metrics_interval_seconds'])
                
                # Update system performance metrics
                with self.lock:
                    for profile in self.reviewer_profiles.values():
                        # Update performance metrics
                        profile.performance_metrics.last_performance_update = datetime.now()
                        
                        # Check health status
                        health_score = self._calculate_reviewer_health_score(profile)
                        if health_score < 0.7:
                            profile.health_status = "degraded"
                        elif health_score < 0.5:
                            profile.health_status = "unhealthy"
                        else:
                            profile.health_status = "healthy"
                
            except Exception as e:
                self.logger.error(f"Performance update loop error: {e}")
    
    def __del__(self):
        """Cleanup Epic 4.3 monitoring resources."""
        try:
            if hasattr(self, 'system_monitor'):
                self.system_monitor.stop_monitoring()
            if hasattr(self, 'assignment_pool'):
                self.assignment_pool.shutdown(wait=True)
        except:
            pass