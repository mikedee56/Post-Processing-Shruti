"""
Reliability Manager for Enterprise Production Excellence.

This module implements comprehensive reliability patterns, error handling, and
graceful degradation for 99.9% uptime targets in the Sanskrit processing system.
"""

import logging
import threading
import time
import traceback
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import weakref

# Import monitoring components  
import sys
sys.path.append(str(Path(__file__).parent.parent / "monitoring"))
from system_monitor import SystemMonitor
from telemetry_collector import TelemetryCollector


class ServiceState(Enum):
    """Service operational states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    UNAVAILABLE = "unavailable"
    RECOVERING = "recovering"


class FailureMode(Enum):
    """Types of failure modes."""
    TIMEOUT = "timeout"
    EXCEPTION = "exception"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    EXTERNAL_DEPENDENCY = "external_dependency"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Recovery strategies for different failure modes."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    FAIL_FAST = "fail_fast"


@dataclass
class FailureRecord:
    """Record of a failure occurrence."""
    failure_id: str
    failure_mode: FailureMode
    component: str
    error_message: str
    timestamp: float
    stack_trace: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None


@dataclass 
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout_seconds: int = 60
    half_open_max_calls: int = 3
    success_threshold: int = 2
    
    
@dataclass
class RetryConfig:
    """Retry mechanism configuration."""
    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    exponential_backoff: bool = True
    jitter: bool = True


@dataclass
class ServiceHealthMetrics:
    """Service health metrics."""
    component: str
    state: ServiceState
    uptime_percentage: float
    failure_count_24h: int
    last_failure_timestamp: Optional[float]
    recovery_count_24h: int
    average_response_time_ms: float
    current_error_rate: float


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Failing, reject requests
    HALF_OPEN = "half_open" # Testing recovery


class CircuitBreaker:
    """
    Circuit breaker implementation for external dependencies.
    
    Prevents cascading failures by temporarily stopping calls to failing services.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        
        # Failure tracking
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.half_open_attempts = 0
        
        # Thread safety
        self.lock = threading.Lock()
        
        self.logger = logging.getLogger(__name__)
    
    def call(self, func: Callable, *args, **kwargs):
        """Call function through circuit breaker."""
        with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                # Check if we should transition to half-open
                if time.time() - self.last_failure_time > self.config.recovery_timeout_seconds:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_attempts = 0
                    self.logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
                else:
                    raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.half_open_attempts >= self.config.half_open_max_calls:
                    raise CircuitBreakerOpenError(f"Circuit breaker {self.name} half-open limit exceeded")
                self.half_open_attempts += 1
        
        # Execute the function
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            raise
    
    def _record_success(self):
        """Record a successful call."""
        with self.lock:
            self.success_count += 1
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    self.logger.info(f"Circuit breaker {self.name} transitioning to CLOSED")
    
    def _record_failure(self):
        """Record a failed call."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitBreakerState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitBreakerState.OPEN
                    self.logger.warning(f"Circuit breaker {self.name} transitioning to OPEN")
            
            elif self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                self.logger.warning(f"Circuit breaker {self.name} returning to OPEN")
    
    def get_state(self) -> Tuple[CircuitBreakerState, Dict[str, Any]]:
        """Get current circuit breaker state and metrics."""
        with self.lock:
            return self.state, {
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'last_failure_time': self.last_failure_time,
                'half_open_attempts': self.half_open_attempts
            }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class ReliabilityManager:
    """
    Enterprise reliability manager for 99.9% uptime targets.
    
    Provides comprehensive reliability patterns:
    - Circuit breakers for external dependencies
    - Retry mechanisms with exponential backoff  
    - Graceful degradation strategies
    - Comprehensive error handling and recovery
    - Service health monitoring and alerting
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize reliability manager with enterprise configuration."""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Core reliability components
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.failure_records: deque = deque(maxlen=10000)
        self.service_states: Dict[str, ServiceState] = {}
        self.health_metrics: Dict[str, ServiceHealthMetrics] = {}
        
        # Monitoring integration
        self.system_monitor: Optional[SystemMonitor] = None
        self.telemetry_collector: Optional[TelemetryCollector] = None
        
        # Threading
        self.health_check_thread = None
        self.running = False
        self.lock = threading.Lock()
        
        # Configuration
        self.uptime_target = self.config.get('uptime_target', 0.999)  # 99.9%
        self.health_check_interval = self.config.get('health_check_interval_seconds', 30)
        self.failure_retention_hours = self.config.get('failure_retention_hours', 48)
        
        # Default configurations
        self.default_circuit_breaker_config = CircuitBreakerConfig(**self.config.get('circuit_breaker', {}))
        self.default_retry_config = RetryConfig(**self.config.get('retry', {}))
        
        # Recovery strategies mapping
        self.recovery_strategies: Dict[FailureMode, RecoveryStrategy] = {
            FailureMode.TIMEOUT: RecoveryStrategy.RETRY,
            FailureMode.EXCEPTION: RecoveryStrategy.FALLBACK,
            FailureMode.RESOURCE_EXHAUSTION: RecoveryStrategy.GRACEFUL_DEGRADATION,
            FailureMode.EXTERNAL_DEPENDENCY: RecoveryStrategy.CIRCUIT_BREAK,
            FailureMode.VALIDATION_ERROR: RecoveryStrategy.FAIL_FAST,
            FailureMode.UNKNOWN: RecoveryStrategy.FALLBACK
        }
        
        # Fallback handlers
        self.fallback_handlers: Dict[str, Callable] = {}
        
        self.logger.info("ReliabilityManager initialized with 99.9% uptime target")
    
    def set_monitoring_integration(self, system_monitor: SystemMonitor, 
                                 telemetry_collector: TelemetryCollector):
        """Set monitoring system integration."""
        self.system_monitor = system_monitor
        self.telemetry_collector = telemetry_collector
        self.logger.info("Monitoring integration configured")
    
    def start_health_monitoring(self):
        """Start health monitoring."""
        with self.lock:
            if self.running:
                return
            
            self.running = True
            self.health_check_thread = threading.Thread(target=self._health_monitoring_loop, daemon=True)
            self.health_check_thread.start()
            
            self.logger.info("Health monitoring started")
    
    def stop_health_monitoring(self):
        """Stop health monitoring."""
        with self.lock:
            self.running = False
        
        if self.health_check_thread:
            self.health_check_thread.join(timeout=5)
            
        self.logger.info("Health monitoring stopped")
    
    def reliable_operation(self, component: str, operation_name: str, 
                          circuit_breaker: bool = True,
                          retry_config: Optional[RetryConfig] = None,
                          fallback_handler: Optional[Callable] = None):
        """
        Decorator for reliable operation execution.
        
        Provides comprehensive reliability patterns including circuit breaking,
        retries, and fallback handling.
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Get or create circuit breaker
                cb_name = f"{component}_{operation_name}"
                if circuit_breaker and cb_name not in self.circuit_breakers:
                    self.circuit_breakers[cb_name] = CircuitBreaker(
                        cb_name, self.default_circuit_breaker_config
                    )
                
                # Configure retry
                retry_cfg = retry_config or self.default_retry_config
                
                # Execute with reliability patterns
                return self._execute_reliable_operation(
                    func, component, operation_name, cb_name if circuit_breaker else None,
                    retry_cfg, fallback_handler, args, kwargs
                )
            
            return wrapper
        return decorator
    
    def _execute_reliable_operation(self, func: Callable, component: str, operation_name: str,
                                  circuit_breaker_name: Optional[str], retry_config: RetryConfig,
                                  fallback_handler: Optional[Callable], args: tuple, kwargs: dict):
        """Execute operation with full reliability patterns."""
        start_time = time.time()
        last_exception = None
        
        for attempt in range(retry_config.max_attempts):
            try:
                # Execute through circuit breaker if configured
                if circuit_breaker_name and circuit_breaker_name in self.circuit_breakers:
                    result = self.circuit_breakers[circuit_breaker_name].call(func, *args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Record successful operation
                execution_time = (time.time() - start_time) * 1000
                self._record_successful_operation(component, operation_name, execution_time, attempt + 1)
                
                return result
                
            except CircuitBreakerOpenError as e:
                # Circuit breaker is open, try fallback
                if fallback_handler:
                    try:
                        result = fallback_handler(*args, **kwargs)
                        self._record_fallback_success(component, operation_name)
                        return result
                    except Exception as fallback_error:
                        self._record_fallback_failure(component, operation_name, str(fallback_error))
                        raise fallback_error
                else:
                    # No fallback available
                    self._record_circuit_breaker_rejection(component, operation_name)
                    raise e
                
            except Exception as e:
                last_exception = e
                failure_mode = self._classify_failure(e)
                
                # Record failure
                failure_record = self._record_failure(
                    component, operation_name, failure_mode, str(e), attempt + 1
                )
                
                # Determine if we should retry
                recovery_strategy = self.recovery_strategies.get(failure_mode, RecoveryStrategy.FALLBACK)
                
                if attempt < retry_config.max_attempts - 1 and recovery_strategy == RecoveryStrategy.RETRY:
                    # Calculate retry delay
                    delay = self._calculate_retry_delay(attempt, retry_config)
                    self.logger.warning(f"Operation {component}.{operation_name} failed (attempt {attempt + 1}), retrying in {delay:.2f}s")
                    time.sleep(delay)
                    continue
                elif recovery_strategy == RecoveryStrategy.FALLBACK and fallback_handler:
                    # Try fallback
                    try:
                        result = fallback_handler(*args, **kwargs)
                        self._record_fallback_success(component, operation_name)
                        return result
                    except Exception as fallback_error:
                        self._record_fallback_failure(component, operation_name, str(fallback_error))
                        # Fall through to final error handling
                else:
                    # Fail fast or no more retries
                    break
        
        # All retries exhausted or fail-fast strategy
        self._record_operation_failure(component, operation_name, str(last_exception))
        raise last_exception
    
    def register_fallback_handler(self, component: str, operation: str, handler: Callable):
        """Register a fallback handler for a specific operation."""
        key = f"{component}.{operation}"
        self.fallback_handlers[key] = handler
        self.logger.info(f"Registered fallback handler for {key}")
    
    def get_service_health(self, component: str) -> ServiceHealthMetrics:
        """Get health metrics for a service component."""
        if component not in self.health_metrics:
            return ServiceHealthMetrics(
                component=component,
                state=ServiceState.HEALTHY,
                uptime_percentage=100.0,
                failure_count_24h=0,
                last_failure_timestamp=None,
                recovery_count_24h=0,
                average_response_time_ms=0.0,
                current_error_rate=0.0
            )
        
        return self.health_metrics[component]
    
    def get_system_reliability_report(self) -> Dict[str, Any]:
        """Generate comprehensive system reliability report."""
        current_time = time.time()
        report_window = 24 * 3600  # 24 hours
        window_start = current_time - report_window
        
        # Filter recent failures
        recent_failures = [f for f in self.failure_records if f.timestamp >= window_start]
        
        # Calculate overall system metrics
        total_operations = len(recent_failures) + sum(
            getattr(metrics, 'success_count_24h', 0) 
            for metrics in self.health_metrics.values()
        )
        
        failure_count = len(recent_failures)
        success_rate = 1 - (failure_count / total_operations) if total_operations > 0 else 1.0
        
        # Component analysis
        component_analysis = {}
        for component in set(f.component for f in recent_failures) | set(self.health_metrics.keys()):
            component_failures = [f for f in recent_failures if f.component == component]
            component_analysis[component] = {
                'state': self.service_states.get(component, ServiceState.HEALTHY).value,
                'failure_count': len(component_failures),
                'most_common_failure_mode': self._get_most_common_failure_mode(component_failures),
                'uptime_percentage': self._calculate_component_uptime(component, window_start),
                'health_metrics': self.health_metrics.get(component)
            }
        
        # Circuit breaker status
        circuit_breaker_status = {}
        for name, cb in self.circuit_breakers.items():
            state, metrics = cb.get_state()
            circuit_breaker_status[name] = {
                'state': state.value,
                'metrics': metrics
            }
        
        report = {
            'report_metadata': {
                'generated_at': current_time,
                'window_hours': 24,
                'uptime_target': self.uptime_target
            },
            'system_overview': {
                'overall_success_rate': success_rate,
                'total_operations': total_operations,
                'failure_count': failure_count,
                'meets_uptime_target': success_rate >= self.uptime_target,
                'active_components': len(component_analysis)
            },
            'component_analysis': component_analysis,
            'circuit_breaker_status': circuit_breaker_status,
            'failure_analysis': self._analyze_failure_patterns(recent_failures),
            'recovery_analysis': self._analyze_recovery_patterns(recent_failures),
            'recommendations': self._generate_reliability_recommendations(recent_failures, success_rate)
        }
        
        return report
    
    def graceful_degradation(self, component: str, degradation_level: str):
        """
        Context manager for graceful degradation.
        
        Allows services to operate in degraded mode when dependencies fail.
        """
        @contextmanager
        def degraded_operation():
            try:
                # Set component to degraded state
                self.service_states[component] = ServiceState.DEGRADED
                self.logger.warning(f"Component {component} entering degraded mode: {degradation_level}")
                
                # Notify monitoring systems
                if self.telemetry_collector:
                    self.telemetry_collector.collect_event(
                        "service_degradation",
                        component,
                        {'degradation_level': degradation_level},
                        severity=AlertSeverity.WARNING
                    )
                
                yield
                
            finally:
                # Attempt to restore normal state
                self.service_states[component] = ServiceState.RECOVERING
                self.logger.info(f"Component {component} attempting recovery from degraded mode")
        
        return degraded_operation()
    
    def _classify_failure(self, exception: Exception) -> FailureMode:
        """Classify the type of failure based on exception."""
        if isinstance(exception, TimeoutError):
            return FailureMode.TIMEOUT
        elif isinstance(exception, (ConnectionError, OSError)):
            return FailureMode.EXTERNAL_DEPENDENCY
        elif isinstance(exception, (MemoryError, OverflowError)):
            return FailureMode.RESOURCE_EXHAUSTION  
        elif isinstance(exception, (ValueError, TypeError)):
            return FailureMode.VALIDATION_ERROR
        elif isinstance(exception, Exception):
            return FailureMode.EXCEPTION
        else:
            return FailureMode.UNKNOWN
    
    def _calculate_retry_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate retry delay with exponential backoff and jitter."""
        if config.exponential_backoff:
            delay = config.base_delay_seconds * (2 ** attempt)
        else:
            delay = config.base_delay_seconds
        
        delay = min(delay, config.max_delay_seconds)
        
        if config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # 50-100% of calculated delay
        
        return delay
    
    def _record_failure(self, component: str, operation: str, failure_mode: FailureMode,
                       error_message: str, attempt: int) -> FailureRecord:
        """Record a failure occurrence."""
        failure_record = FailureRecord(
            failure_id=f"{component}_{operation}_{int(time.time())}_{attempt}",
            failure_mode=failure_mode,
            component=component,
            error_message=error_message,
            timestamp=time.time(),
            stack_trace=traceback.format_exc()
        )
        
        self.failure_records.append(failure_record)
        
        # Update service state if needed
        self._update_service_state(component, failure_mode)
        
        # Notify monitoring systems
        if self.telemetry_collector:
            self.telemetry_collector.collect_event(
                "operation_failure",
                component,
                {
                    'operation': operation,
                    'failure_mode': failure_mode.value,
                    'error_message': error_message,
                    'attempt': attempt
                }
            )
        
        return failure_record
    
    def _record_successful_operation(self, component: str, operation: str, 
                                   execution_time_ms: float, attempts: int):
        """Record a successful operation."""
        # Update component state to healthy if it was degraded
        if self.service_states.get(component) in [ServiceState.DEGRADED, ServiceState.RECOVERING]:
            self.service_states[component] = ServiceState.HEALTHY
            self.logger.info(f"Component {component} recovered to healthy state")
        
        # Collect telemetry
        if self.telemetry_collector:
            self.telemetry_collector.collect_processing_telemetry(
                operation, component, execution_time_ms, True
            )
    
    def _record_fallback_success(self, component: str, operation: str):
        """Record successful fallback execution."""
        self.logger.info(f"Fallback successful for {component}.{operation}")
        
        if self.telemetry_collector:
            self.telemetry_collector.collect_event(
                "fallback_success",
                component,
                {'operation': operation}
            )
    
    def _record_fallback_failure(self, component: str, operation: str, error: str):
        """Record failed fallback execution."""
        self.logger.error(f"Fallback failed for {component}.{operation}: {error}")
        
        if self.telemetry_collector:
            self.telemetry_collector.collect_event(
                "fallback_failure", 
                component,
                {'operation': operation, 'error': error}
            )
    
    def _record_circuit_breaker_rejection(self, component: str, operation: str):
        """Record circuit breaker rejection."""
        self.logger.warning(f"Circuit breaker rejected call to {component}.{operation}")
        
        if self.telemetry_collector:
            self.telemetry_collector.collect_event(
                "circuit_breaker_rejection",
                component,
                {'operation': operation}
            )
    
    def _record_operation_failure(self, component: str, operation: str, error: str):
        """Record final operation failure after all recovery attempts."""
        self.logger.error(f"Operation {component}.{operation} failed permanently: {error}")
        
        # Set component to failing state
        self.service_states[component] = ServiceState.FAILING
        
        if self.telemetry_collector:
            self.telemetry_collector.collect_event(
                "operation_permanent_failure",
                component,
                {'operation': operation, 'error': error}
            )
    
    def _update_service_state(self, component: str, failure_mode: FailureMode):
        """Update service state based on failure patterns."""
        current_state = self.service_states.get(component, ServiceState.HEALTHY)
        
        # Simple state transition logic - could be enhanced
        if failure_mode in [FailureMode.RESOURCE_EXHAUSTION, FailureMode.EXTERNAL_DEPENDENCY]:
            if current_state == ServiceState.HEALTHY:
                self.service_states[component] = ServiceState.DEGRADED
            elif current_state == ServiceState.DEGRADED:
                self.service_states[component] = ServiceState.FAILING
        elif failure_mode == FailureMode.TIMEOUT:
            if current_state in [ServiceState.HEALTHY, ServiceState.DEGRADED]:
                self.service_states[component] = ServiceState.DEGRADED
    
    def _health_monitoring_loop(self):
        """Health monitoring loop."""
        while self.running:
            try:
                self._update_health_metrics()
                self._cleanup_old_failures()
                time.sleep(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(5)
    
    def _update_health_metrics(self):
        """Update health metrics for all components."""
        current_time = time.time()
        window_24h = current_time - (24 * 3600)
        
        # Get all components
        all_components = set()
        all_components.update(self.service_states.keys())
        all_components.update(f.component for f in self.failure_records)
        
        for component in all_components:
            # Get recent failures for this component
            component_failures = [f for f in self.failure_records if f.component == component and f.timestamp >= window_24h]
            
            # Calculate metrics
            failure_count = len(component_failures)
            last_failure = max([f.timestamp for f in component_failures], default=None)
            
            # Calculate uptime percentage
            uptime_percentage = self._calculate_component_uptime(component, window_24h)
            
            # Update health metrics
            self.health_metrics[component] = ServiceHealthMetrics(
                component=component,
                state=self.service_states.get(component, ServiceState.HEALTHY),
                uptime_percentage=uptime_percentage,
                failure_count_24h=failure_count,
                last_failure_timestamp=last_failure,
                recovery_count_24h=len([f for f in component_failures if f.recovery_successful]),
                average_response_time_ms=0.0,  # Would be calculated from successful operations
                current_error_rate=min(failure_count / 100.0, 1.0)  # Rough calculation
            )
    
    def _calculate_component_uptime(self, component: str, since_timestamp: float) -> float:
        """Calculate component uptime percentage."""
        # Simplified calculation - in production this would be more sophisticated
        component_failures = [f for f in self.failure_records if f.component == component and f.timestamp >= since_timestamp]
        
        total_period = time.time() - since_timestamp
        # Assume each failure represents 1 minute of downtime (simplified)
        downtime_seconds = len(component_failures) * 60
        
        uptime_percentage = max(0, (total_period - downtime_seconds) / total_period * 100)
        return min(100, uptime_percentage)
    
    def _cleanup_old_failures(self):
        """Clean up old failure records."""
        cutoff_time = time.time() - (self.failure_retention_hours * 3600)
        
        # Remove old failures
        while self.failure_records and self.failure_records[0].timestamp < cutoff_time:
            self.failure_records.popleft()
    
    def _get_most_common_failure_mode(self, failures: List[FailureRecord]) -> Optional[str]:
        """Get the most common failure mode from a list of failures."""
        if not failures:
            return None
        
        failure_counts = defaultdict(int)
        for failure in failures:
            failure_counts[failure.failure_mode.value] += 1
        
        return max(failure_counts, key=failure_counts.get)
    
    def _analyze_failure_patterns(self, failures: List[FailureRecord]) -> Dict[str, Any]:
        """Analyze failure patterns."""
        if not failures:
            return {'total_failures': 0}
        
        # Group by failure mode
        by_mode = defaultdict(int)
        by_component = defaultdict(int)
        
        for failure in failures:
            by_mode[failure.failure_mode.value] += 1
            by_component[failure.component] += 1
        
        return {
            'total_failures': len(failures),
            'by_failure_mode': dict(by_mode),
            'by_component': dict(by_component),
            'most_common_mode': max(by_mode, key=by_mode.get) if by_mode else None,
            'most_failing_component': max(by_component, key=by_component.get) if by_component else None
        }
    
    def _analyze_recovery_patterns(self, failures: List[FailureRecord]) -> Dict[str, Any]:
        """Analyze recovery patterns."""
        if not failures:
            return {'recovery_rate': 0.0}
        
        recovery_attempts = len([f for f in failures if f.recovery_attempted])
        successful_recoveries = len([f for f in failures if f.recovery_successful])
        
        return {
            'recovery_attempts': recovery_attempts,
            'successful_recoveries': successful_recoveries,
            'recovery_rate': successful_recoveries / recovery_attempts if recovery_attempts > 0 else 0.0
        }
    
    def _generate_reliability_recommendations(self, failures: List[FailureRecord], success_rate: float) -> List[str]:
        """Generate reliability recommendations."""
        recommendations = []
        
        # Check success rate against target
        if success_rate < self.uptime_target:
            recommendations.append(f"Success rate ({success_rate:.1%}) below target ({self.uptime_target:.1%}) - investigate failure patterns")
        
        # Check for common failure modes
        if failures:
            failure_modes = [f.failure_mode for f in failures]
            timeout_failures = len([f for f in failure_modes if f == FailureMode.TIMEOUT])
            external_failures = len([f for f in failure_modes if f == FailureMode.EXTERNAL_DEPENDENCY])
            
            if timeout_failures > len(failures) * 0.3:
                recommendations.append("High timeout failure rate - consider increasing timeouts or improving performance")
            
            if external_failures > len(failures) * 0.2:
                recommendations.append("Frequent external dependency failures - implement more circuit breakers")
        
        # Story 4.3 specific recommendations
        recommendations.extend([
            "Implement disaster recovery testing procedures",
            "Add automated health checks for all critical components",
            "Monitor uptime metrics continuously against 99.9% target"
        ])
        
        return recommendations
    
    def __del__(self):
        """Cleanup on destruction."""
        self.stop_health_monitoring()


def test_reliability_manager():
    """Test reliability manager functionality."""
    manager = ReliabilityManager()
    
    print("Testing reliability manager...")
    
    # Start health monitoring
    manager.start_health_monitoring()
    
    # Test reliable operation decorator
    @manager.reliable_operation("test_component", "test_operation")
    def test_operation(should_fail: bool = False):
        if should_fail:
            raise ValueError("Test failure")
        return "success"
    
    # Test successful operation
    result = test_operation(False)
    assert result == "success"
    
    # Test failed operation with retry
    try:
        test_operation(True)
        assert False, "Should have raised exception"
    except ValueError:
        pass  # Expected
    
    time.sleep(2)  # Let health monitoring run
    
    # Get reliability report
    report = manager.get_system_reliability_report()
    
    print(f"✅ Reliability manager test passed")
    print(f"   Success rate: {report['system_overview']['overall_success_rate']:.1%}")
    print(f"   Total operations: {report['system_overview']['total_operations']}")
    print(f"   Active components: {report['system_overview']['active_components']}")
    
    # Stop health monitoring
    manager.stop_health_monitoring()
    
    return True


if __name__ == "__main__":
    test_reliability_manager()