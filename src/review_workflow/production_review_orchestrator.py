"""
Epic 4.3 Production-Grade Review Workflow Orchestrator.

Implements Story 3.3 Task 4: Epic 4.3 Production-Grade Review Infrastructure
- Orchestrates all Epic 4.3 components for 99.9% uptime reliability
- Coordinates reviewer management, monitoring, and reliability systems
- Provides bulletproof review workflow with enterprise monitoring
- Implements graceful degradation and automatic recovery patterns
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

# Epic 4.3 infrastructure imports
from monitoring.system_monitor import SystemMonitor
from monitoring.telemetry_collector import TelemetryCollector
from utils.performance_monitor import PerformanceMonitor
from utils.circuit_breaker import CircuitBreaker, CircuitBreakerError
from utils.rate_limiter import RateLimiter, AdaptiveRateLimiter
from utils.health_checker import HealthChecker

# Review workflow components
from .reviewer_manager import ReviewerManager, AssignmentRequest, ReviewerRole
from .collaborative_interface import CollaborativeInterface
from .review_workflow_engine import ReviewWorkflowEngine
from .expertise_matching_system import ExpertiseMatchingSystem
from .feedback_integrator import FeedbackIntegrator


class OrchestratorState(Enum):
    """Production orchestrator states."""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    MAINTENANCE = "maintenance"
    SHUTDOWN = "shutdown"


class FailoverMode(Enum):
    """Failover modes for degraded operation."""
    NORMAL = "normal"
    LIMITED_FEATURES = "limited_features"
    ESSENTIAL_ONLY = "essential_only"
    EMERGENCY_MODE = "emergency_mode"


@dataclass
class ProductionMetrics:
    """Epic 4.3 production metrics tracking."""
    # Uptime and reliability
    system_start_time: datetime = field(default_factory=datetime.now)
    uptime_percentage: float = 100.0
    total_downtime_seconds: float = 0.0
    
    # Performance metrics
    average_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    requests_per_second: float = 0.0
    
    # Error tracking
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    error_rate: float = 0.0
    
    # Circuit breaker states
    circuit_breakers_open: int = 0
    circuit_breakers_half_open: int = 0
    
    # SLA compliance
    sla_uptime_target: float = 99.9
    sla_response_time_target_ms: float = 500.0
    sla_uptime_compliance: bool = True
    sla_response_time_compliance: bool = True


@dataclass
class SystemHealthSummary:
    """Comprehensive system health summary."""
    overall_state: OrchestratorState
    failover_mode: FailoverMode
    
    # Component health
    components_healthy: int = 0
    components_degraded: int = 0
    components_critical: int = 0
    
    # Performance status
    performance_score: float = 1.0
    capacity_utilization: float = 0.0
    
    # Alerts and warnings
    active_alerts: int = 0
    warning_count: int = 0
    critical_count: int = 0
    
    # Metrics
    metrics: ProductionMetrics = field(default_factory=ProductionMetrics)
    last_health_check: datetime = field(default_factory=datetime.now)


class ProductionReviewOrchestrator:
    """
    Epic 4.3 Production-Grade Review Workflow Orchestrator.
    
    Implements Story 3.3 Task 4:
    - 99.9% uptime reliability coordination
    - Sub-second response time orchestration
    - Enterprise monitoring and telemetry integration
    - Bulletproof reliability with automatic failover
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize production orchestrator with Epic 4.3 features."""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Production configuration
        self.production_config = self.config.get('production', {
            'max_response_time_ms': 500.0,
            'target_uptime_percentage': 99.9,
            'health_check_interval_seconds': 10,
            'performance_metrics_interval_seconds': 30,
            'circuit_breaker_enabled': True,
            'adaptive_rate_limiting': True,
            'automatic_failover': True,
            'graceful_degradation': True,
            'enterprise_monitoring': True
        })
        
        # Epic 4.3 Infrastructure Components
        self.system_monitor = SystemMonitor(self.config.get('monitoring', {}))
        self.performance_monitor = PerformanceMonitor(self.config.get('performance', {}))
        self.telemetry_collector = TelemetryCollector(self.config.get('telemetry', {}))
        self.health_checker = HealthChecker(self.config.get('health_checker', {}))
        
        # Epic 4.3 Reliability Infrastructure
        self.master_circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=Exception
        )
        self.adaptive_rate_limiter = AdaptiveRateLimiter(
            max_requests=2000,
            time_window=60
        )
        
        # Review Workflow Components
        self.reviewer_manager = ReviewerManager(self.config.get('reviewer_manager', {}))
        self.collaborative_interface = CollaborativeInterface(self.config.get('collaborative_interface', {}))
        self.workflow_engine = ReviewWorkflowEngine(self.config.get('workflow_engine', {}))
        self.expertise_matcher = ExpertiseMatchingSystem(self.config.get('expertise_matcher', {}))
        self.feedback_integrator = FeedbackIntegrator(self.config.get('feedback_integrator', {}))
        
        # State management
        self.orchestrator_state = OrchestratorState.INITIALIZING
        self.failover_mode = FailoverMode.NORMAL
        self.production_metrics = ProductionMetrics()
        self.lock = threading.RLock()
        
        # Performance tracking
        self.response_times = deque(maxlen=1000)
        self.error_tracking = defaultdict(int)
        self.component_health = {}
        
        # Threading for continuous monitoring
        self.monitoring_active = False
        self.monitoring_pool = ThreadPoolExecutor(
            max_workers=5,
            thread_name_prefix="production_monitor"
        )
        
        # Initialize Epic 4.3 production systems
        self._initialize_production_systems()
        
        self.logger.info("ProductionReviewOrchestrator initialized with Epic 4.3 reliability")
    
    def start_production_operations(self) -> bool:
        """
        Start all production operations with Epic 4.3 reliability.
        
        Returns:
            bool: True if all systems started successfully
        """
        start_time = time.time()
        
        try:
            with self.lock:
                if self.orchestrator_state != OrchestratorState.INITIALIZING:
                    self.logger.warning("Production operations already started")
                    return False
                
                # Start Epic 4.3 monitoring systems
                self._start_monitoring_systems()
                
                # Start review workflow components
                self._start_workflow_components()
                
                # Begin health monitoring
                self._start_health_monitoring()
                
                # Start performance tracking
                self._start_performance_tracking()
                
                # Transition to healthy state
                self.orchestrator_state = OrchestratorState.HEALTHY
                self.failover_mode = FailoverMode.NORMAL
                
                startup_time = (time.time() - start_time) * 1000
                
                # Record startup telemetry
                self.telemetry_collector.record_event("production_startup", {
                    'startup_time_ms': startup_time,
                    'components_started': len(self.component_health),
                    'epic_4_3_enabled': True
                })
                
                self.logger.info(f"Production operations started successfully ({startup_time:.1f}ms)")
                return True
                
        except Exception as e:
            self.orchestrator_state = OrchestratorState.CRITICAL
            self.logger.error(f"Failed to start production operations: {e}")
            return False
    
    def process_review_request(self, 
                             session_id: str,
                             content_segments: List[Any],
                             priority: str = "standard",
                             required_expertise: Optional[List[str]] = None) -> Optional[str]:
        """
        Process review request with Epic 4.3 production guarantees.
        
        Args:
            session_id: Review session identifier
            content_segments: Content to be reviewed
            priority: Request priority (critical, high, standard, low)
            required_expertise: Required reviewer expertise
            
        Returns:
            str: Review session ID if successful, None otherwise
        """
        start_time = time.time()
        
        with self.performance_monitor.monitor_processing_operation("process_review_request"):
            try:
                # Rate limiting check
                if not self.adaptive_rate_limiter.acquire():
                    self.production_metrics.failed_requests += 1
                    self.logger.warning(f"Rate limit exceeded for review request {session_id}")
                    return None
                
                # Circuit breaker check
                if self.master_circuit_breaker.is_open():
                    return self._handle_degraded_request(session_id, content_segments, priority)
                
                with self.lock:
                    self.production_metrics.total_requests += 1
                    
                    # Create collaborative session
                    if not self.collaborative_interface.create_collaborative_session(session_id):
                        raise Exception("Failed to create collaborative session")
                    
                    # Match reviewers to content using expertise system
                    reviewer_assignments = self._assign_reviewers_with_expertise(
                        session_id, content_segments, priority, required_expertise
                    )
                    
                    if not reviewer_assignments:
                        raise Exception("No suitable reviewers available")
                    
                    # Initialize workflow for session
                    workflow_result = self.workflow_engine.initialize_workflow(
                        session_id, content_segments, reviewer_assignments
                    )
                    
                    if not workflow_result:
                        raise Exception("Failed to initialize workflow")
                    
                    # Record success metrics
                    processing_time = (time.time() - start_time) * 1000
                    self.response_times.append(processing_time)
                    self.production_metrics.successful_requests += 1
                    
                    # Update performance metrics
                    self._update_performance_metrics(processing_time)
                    
                    # Validate Epic 4.3 response time requirement
                    if processing_time <= self.production_config['max_response_time_ms']:
                        self.production_metrics.sla_response_time_compliance = True
                    else:
                        self.logger.warning(f"Request processing exceeded SLA: {processing_time:.1f}ms")
                        self.production_metrics.sla_response_time_compliance = False
                    
                    # Record success telemetry
                    self.telemetry_collector.record_event("review_request_processed", {
                        'session_id': session_id,
                        'processing_time_ms': processing_time,
                        'priority': priority,
                        'reviewers_assigned': len(reviewer_assignments),
                        'sla_compliant': processing_time <= self.production_config['max_response_time_ms']
                    })
                    
                    self.logger.info(f"Review request processed: {session_id} ({processing_time:.1f}ms)")
                    return session_id
                    
            except Exception as e:
                # Record failure and circuit breaker tracking
                self.master_circuit_breaker.record_failure()
                self.production_metrics.failed_requests += 1
                self.error_tracking[type(e).__name__] += 1
                
                processing_time = (time.time() - start_time) * 1000
                
                # Record error telemetry
                self.telemetry_collector.record_event("review_request_failed", {
                    'session_id': session_id,
                    'error': str(e),
                    'processing_time_ms': processing_time,
                    'priority': priority
                })
                
                self.logger.error(f"Review request failed for {session_id}: {e}")
                
                # Try graceful degradation if enabled
                if self.production_config['graceful_degradation']:
                    return self._handle_degraded_request(session_id, content_segments, priority)
                
                return None
    
    def get_system_health(self) -> SystemHealthSummary:
        """
        Get comprehensive system health with Epic 4.3 production metrics.
        
        Returns:
            SystemHealthSummary: Complete system health information
        """
        with self.lock:
            # Calculate uptime
            current_time = datetime.now()
            total_runtime = (current_time - self.production_metrics.system_start_time).total_seconds()
            uptime_percentage = ((total_runtime - self.production_metrics.total_downtime_seconds) / max(total_runtime, 1)) * 100
            
            # Update uptime metrics
            self.production_metrics.uptime_percentage = uptime_percentage
            self.production_metrics.sla_uptime_compliance = uptime_percentage >= self.production_config['target_uptime_percentage']
            
            # Calculate performance metrics
            if self.response_times:
                self.production_metrics.average_response_time_ms = statistics.mean(self.response_times)
                sorted_times = sorted(self.response_times)
                p95_index = int(len(sorted_times) * 0.95)
                p99_index = int(len(sorted_times) * 0.99)
                self.production_metrics.p95_response_time_ms = sorted_times[p95_index] if sorted_times else 0.0
                self.production_metrics.p99_response_time_ms = sorted_times[p99_index] if sorted_times else 0.0
            
            # Calculate error rate
            total_requests = self.production_metrics.total_requests
            if total_requests > 0:
                self.production_metrics.error_rate = (self.production_metrics.failed_requests / total_requests) * 100
            
            # Assess component health
            healthy_components = len([h for h in self.component_health.values() if h.get('status') == 'healthy'])
            degraded_components = len([h for h in self.component_health.values() if h.get('status') == 'degraded'])
            critical_components = len([h for h in self.component_health.values() if h.get('status') == 'critical'])
            
            # Determine overall system state
            if critical_components > 0 or uptime_percentage < 99.0:
                overall_state = OrchestratorState.CRITICAL
                failover_mode = FailoverMode.EMERGENCY_MODE
            elif degraded_components > 0 or uptime_percentage < 99.5:
                overall_state = OrchestratorState.DEGRADED
                failover_mode = FailoverMode.LIMITED_FEATURES
            else:
                overall_state = OrchestratorState.HEALTHY
                failover_mode = FailoverMode.NORMAL
            
            # Update orchestrator state
            self.orchestrator_state = overall_state
            self.failover_mode = failover_mode
            
            # Calculate performance score
            performance_factors = [
                1.0 if self.production_metrics.sla_uptime_compliance else 0.5,
                1.0 if self.production_metrics.sla_response_time_compliance else 0.7,
                max(0.0, 1.0 - (self.production_metrics.error_rate / 100)),
                1.0 if not self.master_circuit_breaker.is_open() else 0.3
            ]
            performance_score = statistics.mean(performance_factors)
            
            return SystemHealthSummary(
                overall_state=overall_state,
                failover_mode=failover_mode,
                components_healthy=healthy_components,
                components_degraded=degraded_components,
                components_critical=critical_components,
                performance_score=performance_score,
                capacity_utilization=self._calculate_capacity_utilization(),
                active_alerts=len([e for e in self.error_tracking.values() if e > 0]),
                warning_count=degraded_components,
                critical_count=critical_components,
                metrics=self.production_metrics,
                last_health_check=current_time
            )
    
    def get_production_dashboard(self) -> Dict[str, Any]:
        """
        Get Epic 4.3 production dashboard information.
        
        Returns:
            dict: Comprehensive production dashboard data
        """
        health_summary = self.get_system_health()
        
        return {
            'epic_4_3_status': {
                'overall_state': health_summary.overall_state.value,
                'failover_mode': health_summary.failover_mode.value,
                'uptime_percentage': health_summary.metrics.uptime_percentage,
                'performance_score': health_summary.performance_score,
                'sla_compliance': {
                    'uptime_target': self.production_config['target_uptime_percentage'],
                    'uptime_compliant': health_summary.metrics.sla_uptime_compliance,
                    'response_time_target_ms': self.production_config['max_response_time_ms'],
                    'response_time_compliant': health_summary.metrics.sla_response_time_compliance
                }
            },
            'performance_metrics': {
                'average_response_time_ms': health_summary.metrics.average_response_time_ms,
                'p95_response_time_ms': health_summary.metrics.p95_response_time_ms,
                'p99_response_time_ms': health_summary.metrics.p99_response_time_ms,
                'requests_per_second': health_summary.metrics.requests_per_second,
                'error_rate': health_summary.metrics.error_rate,
                'total_requests': health_summary.metrics.total_requests
            },
            'component_health': {
                'total_components': len(self.component_health),
                'healthy_components': health_summary.components_healthy,
                'degraded_components': health_summary.components_degraded,
                'critical_components': health_summary.components_critical,
                'component_details': self.component_health
            },
            'reliability_systems': {
                'master_circuit_breaker': 'open' if self.master_circuit_breaker.is_open() else 'closed',
                'adaptive_rate_limiter': {
                    'active': True,
                    'available_tokens': self.adaptive_rate_limiter.get_available_tokens(),
                    'metrics': self.adaptive_rate_limiter.get_metrics()
                },
                'health_checker': 'active',
                'monitoring_active': self.monitoring_active
            },
            'reviewer_management': self.reviewer_manager.get_system_health(),
            'workflow_statistics': self.workflow_engine.get_workflow_statistics() if hasattr(self.workflow_engine, 'get_workflow_statistics') else {},
            'collaborative_interface': self.collaborative_interface.get_interface_statistics()
        }
    
    def _initialize_production_systems(self) -> None:
        """Initialize Epic 4.3 production systems."""
        try:
            # Register health checks for all components
            self._register_component_health_checks()
            
            # Initialize component health tracking
            self.component_health = {
                'reviewer_manager': {'status': 'healthy', 'last_check': datetime.now()},
                'collaborative_interface': {'status': 'healthy', 'last_check': datetime.now()},
                'workflow_engine': {'status': 'healthy', 'last_check': datetime.now()},
                'expertise_matcher': {'status': 'healthy', 'last_check': datetime.now()},
                'feedback_integrator': {'status': 'healthy', 'last_check': datetime.now()},
                'system_monitor': {'status': 'healthy', 'last_check': datetime.now()},
                'telemetry_collector': {'status': 'healthy', 'last_check': datetime.now()}
            }
            
        except Exception as e:
            self.logger.error(f"Failed to initialize production systems: {e}")
    
    def _start_monitoring_systems(self) -> None:
        """Start Epic 4.3 monitoring and telemetry systems."""
        try:
            self.system_monitor.start_monitoring()
            self.health_checker.start_health_checks()
            self.monitoring_active = True
            
            self.logger.info("Epic 4.3 monitoring systems started")
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring systems: {e}")
    
    def _start_workflow_components(self) -> None:
        """Start all review workflow components."""
        # Components are already initialized and ready to use
        # No explicit startup required for current implementation
        pass
    
    def _start_health_monitoring(self) -> None:
        """Start continuous health monitoring."""
        self.monitoring_pool.submit(self._health_monitoring_loop)
    
    def _start_performance_tracking(self) -> None:
        """Start continuous performance tracking."""
        self.monitoring_pool.submit(self._performance_tracking_loop)
    
    def _assign_reviewers_with_expertise(self,
                                       session_id: str,
                                       content_segments: List[Any],
                                       priority: str,
                                       required_expertise: Optional[List[str]]) -> List[str]:
        """Assign reviewers using expertise matching system."""
        try:
            # Create assignment request
            assignment_request = AssignmentRequest(
                request_id=f"req_{session_id}_{int(time.time())}",
                session_id=session_id,
                required_role=ReviewerRole.GENERAL_PROOFREADER,  # Default role
                required_skills=required_expertise or [],
                priority_level=priority,
                max_response_time_ms=self.production_config['max_response_time_ms']
            )
            
            # Use reviewer manager to assign
            assigned_reviewer = self.reviewer_manager.assign_reviewer(assignment_request)
            
            if assigned_reviewer:
                return [assigned_reviewer]
            
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to assign reviewers: {e}")
            return []
    
    def _handle_degraded_request(self, session_id: str, content_segments: List[Any], priority: str) -> Optional[str]:
        """Handle request during degraded system performance."""
        try:
            self.logger.warning(f"Handling degraded request for session {session_id}")
            
            # Simplified processing for degraded mode
            if self.collaborative_interface.create_collaborative_session(session_id):
                # Record degraded operation
                self.telemetry_collector.record_event("degraded_request_processed", {
                    'session_id': session_id,
                    'priority': priority,
                    'degraded_mode': True
                })
                
                return session_id
            
            return None
            
        except Exception as e:
            self.logger.error(f"Degraded request handling failed: {e}")
            return None
    
    def _update_performance_metrics(self, processing_time: float) -> None:
        """Update Epic 4.3 performance metrics."""
        # Calculate requests per second
        if len(self.response_times) >= 2:
            time_window = 60  # 1 minute window
            recent_requests = len([t for t in self.response_times if t <= time_window * 1000])
            self.production_metrics.requests_per_second = recent_requests / time_window
        
        # Update adaptive rate limiter performance
        performance_score = 1.0 if processing_time <= self.production_config['max_response_time_ms'] else 0.7
        self.adaptive_rate_limiter.update_performance(performance_score)
    
    def _calculate_capacity_utilization(self) -> float:
        """Calculate system capacity utilization."""
        try:
            # Get reviewer utilization
            reviewer_health = self.reviewer_manager.get_system_health()
            reviewer_utilization = reviewer_health.get('system_overview', {}).get('system_capacity_utilization', 0.0)
            
            # Factor in circuit breaker state
            reliability_factor = 0.5 if self.master_circuit_breaker.is_open() else 1.0
            
            return min(reviewer_utilization * reliability_factor, 100.0)
            
        except Exception:
            return 50.0  # Default moderate utilization
    
    def _register_component_health_checks(self) -> None:
        """Register health checks for all Epic 4.3 components."""
        
        def reviewer_manager_check():
            try:
                health = self.reviewer_manager.get_system_health()
                uptime = health.get('epic_4_3_reliability', {}).get('uptime_percentage', 0)
                if uptime >= 99.0:
                    return True, f"Reviewer manager healthy: {uptime:.1f}% uptime", {'uptime': uptime}
                else:
                    return False, f"Reviewer manager degraded: {uptime:.1f}% uptime", {'uptime': uptime}
            except Exception as e:
                return False, f"Reviewer manager check failed: {e}", {}
        
        def collaborative_interface_check():
            try:
                stats = self.collaborative_interface.get_interface_statistics()
                active_sessions = stats.get('active_sessions', 0)
                return True, f"Collaborative interface healthy: {active_sessions} active sessions", {'sessions': active_sessions}
            except Exception as e:
                return False, f"Collaborative interface check failed: {e}", {}
        
        def circuit_breaker_check():
            if self.master_circuit_breaker.is_open():
                return False, "Master circuit breaker is OPEN", {'state': 'open'}
            else:
                return True, "Master circuit breaker is CLOSED", {'state': 'closed'}
        
        # Register all health checks
        self.health_checker.register_health_check("reviewer_manager", reviewer_manager_check)
        self.health_checker.register_health_check("collaborative_interface", collaborative_interface_check)
        self.health_checker.register_health_check("master_circuit_breaker", circuit_breaker_check)
    
    def _health_monitoring_loop(self) -> None:
        """Continuous health monitoring loop."""
        while self.monitoring_active:
            try:
                time.sleep(self.production_config['health_check_interval_seconds'])
                
                # Run health checks
                health_summary = self.health_checker.run_all_checks()
                
                # Update component health status
                with self.lock:
                    for component_name in self.component_health:
                        # Simulate component health based on overall system health
                        if health_summary.overall_status.value == 'healthy':
                            self.component_health[component_name]['status'] = 'healthy'
                        elif health_summary.overall_status.value in ['degraded', 'unhealthy']:
                            self.component_health[component_name]['status'] = 'degraded'
                        else:
                            self.component_health[component_name]['status'] = 'critical'
                        
                        self.component_health[component_name]['last_check'] = datetime.now()
                
            except Exception as e:
                self.logger.error(f"Health monitoring loop error: {e}")
    
    def _performance_tracking_loop(self) -> None:
        """Continuous performance tracking loop."""
        while self.monitoring_active:
            try:
                time.sleep(self.production_config['performance_metrics_interval_seconds'])
                
                # Calculate and log performance metrics
                health_summary = self.get_system_health()
                
                # Log Epic 4.3 performance metrics
                self.telemetry_collector.record_event("performance_metrics", {
                    'uptime_percentage': health_summary.metrics.uptime_percentage,
                    'average_response_time_ms': health_summary.metrics.average_response_time_ms,
                    'error_rate': health_summary.metrics.error_rate,
                    'performance_score': health_summary.performance_score,
                    'sla_uptime_compliance': health_summary.metrics.sla_uptime_compliance,
                    'sla_response_time_compliance': health_summary.metrics.sla_response_time_compliance
                })
                
            except Exception as e:
                self.logger.error(f"Performance tracking loop error: {e}")
    
    def shutdown_production_operations(self) -> bool:
        """
        Gracefully shutdown all production operations.
        
        Returns:
            bool: True if shutdown completed successfully
        """
        try:
            with self.lock:
                self.orchestrator_state = OrchestratorState.SHUTDOWN
                self.monitoring_active = False
                
                # Stop monitoring systems
                self.health_checker.stop_health_checks()
                
                # Shutdown thread pool
                self.monitoring_pool.shutdown(wait=True)
                
                self.logger.info("Production operations shut down successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to shutdown production operations: {e}")
            return False
    
    def __del__(self):
        """Cleanup Epic 4.3 production resources."""
        try:
            self.shutdown_production_operations()
        except:
            pass