"""
Epic 4.3 Circuit Breaker Pattern for Production Reliability.

Implements production-grade circuit breaker pattern to prevent cascading failures
and ensure 99.9% uptime reliability for critical review workflow operations.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Any, Optional, Dict


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failures detected, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker performance metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    
    # Time tracking
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    state_change_time: datetime = field(default_factory=datetime.now)
    
    # Performance metrics
    failure_rate: float = 0.0
    average_response_time_ms: float = 0.0
    recovery_attempts: int = 0


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open and blocking requests."""
    pass


class CircuitBreaker:
    """
    Epic 4.3 Production-Grade Circuit Breaker.
    
    Prevents cascading failures and ensures system resilience by:
    - Monitoring failure rates and response times
    - Opening circuit when failure threshold exceeded
    - Allowing limited testing in half-open state
    - Automatic recovery when service stabilizes
    """
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: type = Exception,
                 timeout_seconds: float = 30.0,
                 monitoring_window_seconds: int = 300):
        """
        Initialize circuit breaker with Epic 4.3 production settings.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type that triggers circuit breaker
            timeout_seconds: Request timeout threshold
            monitoring_window_seconds: Window for calculating failure rates
        """
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.timeout_seconds = timeout_seconds
        self.monitoring_window_seconds = monitoring_window_seconds
        
        # State management
        self.state = CircuitBreakerState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self.lock = threading.RLock()
        
        # Failure tracking
        self.failure_times = []
        self.recent_failures = 0
        
        self.logger.info(f"CircuitBreaker initialized: threshold={failure_threshold}, timeout={recovery_timeout}s")
    
    def is_open(self) -> bool:
        """Check if circuit breaker is open (blocking requests)."""
        with self.lock:
            return self.state == CircuitBreakerState.OPEN
    
    def is_half_open(self) -> bool:
        """Check if circuit breaker is in half-open state (testing recovery)."""
        with self.lock:
            return self.state == CircuitBreakerState.HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
        """
        with self.lock:
            # Check circuit state
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    self.metrics.rejected_requests += 1
                    raise CircuitBreakerError("Circuit breaker is OPEN")
            
            elif self.state == CircuitBreakerState.HALF_OPEN:
                # Allow limited requests in half-open state
                if self.metrics.recovery_attempts >= 3:
                    self._transition_to_open()
                    raise CircuitBreakerError("Circuit breaker recovery failed")
        
        # Execute function with monitoring
        start_time = time.time()
        self.metrics.total_requests += 1
        
        try:
            result = func(*args, **kwargs)
            
            # Record success
            execution_time = (time.time() - start_time) * 1000
            self._record_success(execution_time)
            
            return result
            
        except self.expected_exception as e:
            # Record failure
            execution_time = (time.time() - start_time) * 1000
            self._record_failure(execution_time)
            raise
    
    def record_success(self) -> None:
        """Manually record a successful operation."""
        self._record_success(0.0)
    
    def record_failure(self) -> None:
        """Manually record a failed operation."""
        self._record_failure(0.0)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker performance metrics."""
        with self.lock:
            return {
                'state': self.state.value,
                'total_requests': self.metrics.total_requests,
                'successful_requests': self.metrics.successful_requests,
                'failed_requests': self.metrics.failed_requests,
                'rejected_requests': self.metrics.rejected_requests,
                'failure_rate': self.metrics.failure_rate,
                'average_response_time_ms': self.metrics.average_response_time_ms,
                'recent_failures': self.recent_failures,
                'recovery_attempts': self.metrics.recovery_attempts,
                'last_failure': self.metrics.last_failure_time.isoformat() if self.metrics.last_failure_time else None,
                'last_success': self.metrics.last_success_time.isoformat() if self.metrics.last_success_time else None,
                'state_change_time': self.metrics.state_change_time.isoformat()
            }
    
    def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        with self.lock:
            self.state = CircuitBreakerState.CLOSED
            self.recent_failures = 0
            self.failure_times.clear()
            self.metrics.recovery_attempts = 0
            self.metrics.state_change_time = datetime.now()
            
            self.logger.info("Circuit breaker manually reset to CLOSED state")
    
    def _record_success(self, execution_time_ms: float) -> None:
        """Record successful operation."""
        with self.lock:
            self.metrics.successful_requests += 1
            self.metrics.last_success_time = datetime.now()
            
            # Update average response time
            total_time = (self.metrics.average_response_time_ms * 
                         (self.metrics.successful_requests - 1) + execution_time_ms)
            self.metrics.average_response_time_ms = total_time / self.metrics.successful_requests
            
            # Handle state transitions
            if self.state == CircuitBreakerState.HALF_OPEN:
                # Success in half-open state - transition to closed
                self._transition_to_closed()
            elif self.state == CircuitBreakerState.CLOSED:
                # Reset recent failures on success
                self.recent_failures = max(0, self.recent_failures - 1)
    
    def _record_failure(self, execution_time_ms: float) -> None:
        """Record failed operation."""
        with self.lock:
            self.metrics.failed_requests += 1
            self.metrics.last_failure_time = datetime.now()
            self.failure_times.append(datetime.now())
            self.recent_failures += 1
            
            # Clean old failure times
            cutoff_time = datetime.now() - timedelta(seconds=self.monitoring_window_seconds)
            self.failure_times = [t for t in self.failure_times if t > cutoff_time]
            
            # Calculate failure rate
            total_recent = self.metrics.successful_requests + len(self.failure_times)
            if total_recent > 0:
                self.metrics.failure_rate = len(self.failure_times) / total_recent
            
            # Check if we should open the circuit
            if self.state == CircuitBreakerState.CLOSED:
                if (self.recent_failures >= self.failure_threshold or
                    len(self.failure_times) >= self.failure_threshold):
                    self._transition_to_open()
            
            elif self.state == CircuitBreakerState.HALF_OPEN:
                # Failure in half-open state - back to open
                self._transition_to_open()
                self.metrics.recovery_attempts += 1
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if not self.metrics.last_failure_time:
            return True
        
        time_since_failure = datetime.now() - self.metrics.last_failure_time
        return time_since_failure.total_seconds() >= self.recovery_timeout
    
    def _transition_to_open(self) -> None:
        """Transition circuit breaker to OPEN state."""
        self.state = CircuitBreakerState.OPEN
        self.metrics.state_change_time = datetime.now()
        self.logger.warning(f"Circuit breaker OPENED: {self.recent_failures} recent failures")
    
    def _transition_to_half_open(self) -> None:
        """Transition circuit breaker to HALF_OPEN state."""
        self.state = CircuitBreakerState.HALF_OPEN
        self.metrics.state_change_time = datetime.now()
        self.metrics.recovery_attempts = 0
        self.logger.info("Circuit breaker transitioned to HALF_OPEN - testing recovery")
    
    def _transition_to_closed(self) -> None:
        """Transition circuit breaker to CLOSED state."""
        self.state = CircuitBreakerState.CLOSED
        self.metrics.state_change_time = datetime.now()
        self.recent_failures = 0
        self.failure_times.clear()
        self.metrics.recovery_attempts = 0
        self.logger.info("Circuit breaker CLOSED - service recovered")


def circuit_breaker(failure_threshold: int = 5,
                   recovery_timeout: int = 60,
                   expected_exception: type = Exception):
    """
    Decorator for applying circuit breaker pattern to functions.
    
    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before attempting recovery
        expected_exception: Exception type that triggers circuit breaker
    
    Example:
        @circuit_breaker(failure_threshold=3, recovery_timeout=30)
        def risky_operation():
            # Function that might fail
            pass
    """
    def decorator(func):
        breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception
        )
        
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        
        # Attach breaker to wrapper for external access
        wrapper.circuit_breaker = breaker
        return wrapper
    
    return decorator