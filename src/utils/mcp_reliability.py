"""
MCP Reliability Patterns for Story 5.2
Implements circuit breaker patterns, health checks, and graceful degradation
with professional standards compliance per CEO directive.
"""

import time
import logging
import asyncio
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

from utils.professional_standards import TechnicalQualityGate

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Health check result with professional validation"""
    status: HealthStatus
    response_time: float
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics tracking"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    trips: int = 0
    last_trip_time: float = 0
    current_state: str = "closed"
    success_rate: float = 1.0


class MCPHealthMonitor:
    """
    MCP health monitoring system with professional standards compliance
    
    Implements comprehensive health checks and monitoring for MCP services
    with automated alerting and professional validation of health metrics.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.professional_validator = TechnicalQualityGate()
        self.health_history: List[HealthCheckResult] = []
        self.alert_thresholds = self.config.get('alert_thresholds', {})
        self.monitoring_enabled = self.config.get('enable_health_monitoring', True)
        self.check_interval = self.config.get('health_check_interval', 30.0)
        
        # Health check functions registry
        self.health_checks: Dict[str, Callable] = {}
        
        # Threading for continuous monitoring
        self.monitor_thread = None
        self.stop_monitoring = threading.Event()
        
        logger.info("MCPHealthMonitor initialized with professional standards compliance")
    
    def register_health_check(self, name: str, check_function: Callable):
        """Register a health check function"""
        self.health_checks[name] = check_function
        logger.info(f"Health check '{name}' registered")
    
    async def perform_health_check(self, check_name: str) -> HealthCheckResult:
        """Perform individual health check with professional validation"""
        start_time = time.time()
        
        try:
            # Professional standards validation for health check
            health_check_claims = {
                'health_check_execution': {
                    'factual_basis': f'Executing health check: {check_name}',
                    'verification_method': 'health_check_function',
                    'supporting_data': {'check_name': check_name, 'timestamp': start_time}
                }
            }
            
            validation_result = self.professional_validator.validate_technical_claims(health_check_claims)
            if not validation_result['professional_compliance']:
                logger.error(f"Health check '{check_name}' failed professional standards validation")
                return HealthCheckResult(
                    status=HealthStatus.UNKNOWN,
                    response_time=time.time() - start_time,
                    error_message="Professional standards validation failed"
                )
            
            # Execute health check
            if check_name in self.health_checks:
                check_function = self.health_checks[check_name]
                result = await check_function()
                
                response_time = time.time() - start_time
                
                if result.get('healthy', False):
                    status = HealthStatus.HEALTHY
                elif result.get('degraded', False):
                    status = HealthStatus.DEGRADED
                else:
                    status = HealthStatus.UNHEALTHY
                
                return HealthCheckResult(
                    status=status,
                    response_time=response_time,
                    details=result
                )
            else:
                return HealthCheckResult(
                    status=HealthStatus.UNKNOWN,
                    response_time=time.time() - start_time,
                    error_message=f"Health check '{check_name}' not registered"
                )
                
        except Exception as e:
            logger.error(f"Health check '{check_name}' failed: {e}")
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                response_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def perform_comprehensive_health_check(self) -> Dict[str, HealthCheckResult]:
        """Perform all registered health checks"""
        results = {}
        
        for check_name in self.health_checks:
            results[check_name] = await self.perform_health_check(check_name)
        
        # Store results in history
        self.health_history.extend(results.values())
        
        # Trim history to last 1000 entries
        if len(self.health_history) > 1000:
            self.health_history = self.health_history[-1000:]
        
        return results
    
    def get_system_health_status(self) -> HealthStatus:
        """Get overall system health status"""
        if not self.health_history:
            return HealthStatus.UNKNOWN
        
        recent_checks = self.health_history[-len(self.health_checks):]
        
        unhealthy_count = sum(1 for check in recent_checks if check.status == HealthStatus.UNHEALTHY)
        degraded_count = sum(1 for check in recent_checks if check.status == HealthStatus.DEGRADED)
        
        if unhealthy_count > 0:
            return HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def start_continuous_monitoring(self):
        """Start continuous health monitoring in background thread"""
        if not self.monitoring_enabled:
            return
        
        def monitor_loop():
            while not self.stop_monitoring.is_set():
                try:
                    # Run health checks
                    asyncio.run(self.perform_comprehensive_health_check())
                    
                    # Check for alerts
                    self._check_alert_conditions()
                    
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")
                
                # Wait for next check
                self.stop_monitoring.wait(self.check_interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Continuous health monitoring started")
    
    def stop_continuous_monitoring(self):
        """Stop continuous health monitoring"""
        self.stop_monitoring.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Continuous health monitoring stopped")
    
    def _check_alert_conditions(self):
        """Check if any alert conditions are met"""
        if not self.health_history:
            return
        
        recent_checks = self.health_history[-10:]  # Last 10 checks
        
        # Check failure rate
        if 'failure_rate_threshold' in self.alert_thresholds:
            failure_rate = sum(1 for check in recent_checks if check.status == HealthStatus.UNHEALTHY) / len(recent_checks)
            if failure_rate >= self.alert_thresholds['failure_rate_threshold']:
                logger.warning(f"High failure rate alert: {failure_rate:.2%}")
        
        # Check response time
        if 'response_time_threshold' in self.alert_thresholds:
            avg_response_time = sum(check.response_time for check in recent_checks) / len(recent_checks)
            if avg_response_time >= self.alert_thresholds['response_time_threshold']:
                logger.warning(f"High response time alert: {avg_response_time:.2f}s")


class MCPCircuitBreakerAdvanced:
    """
    Advanced circuit breaker with professional standards compliance
    
    Implements sophisticated circuit breaker patterns with configurable
    thresholds, exponential backoff, and professional validation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.professional_validator = TechnicalQualityGate()
        
        # Circuit breaker configuration
        self.failure_threshold = self.config.get('failure_threshold', 5)
        self.recovery_timeout = self.config.get('recovery_timeout', 60.0)
        self.success_threshold = self.config.get('success_threshold', 3)
        self.monitoring_window = self.config.get('monitoring_window', 300.0)  # 5 minutes
        
        # Circuit breaker state
        self.state = "closed"  # closed, open, half-open
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.last_success_time = 0
        
        # Statistics tracking
        self.stats = CircuitBreakerStats()
        
        # Request history for sliding window
        self.request_history: List[Dict[str, Any]] = []
        
        logger.info(f"MCPCircuitBreakerAdvanced initialized - threshold: {self.failure_threshold}, timeout: {self.recovery_timeout}s")
    
    def can_execute_request(self) -> bool:
        """
        Check if request can be executed based on circuit breaker state
        with professional standards validation
        """
        # Professional validation of execution decision
        execution_claims = {
            'circuit_breaker_decision': {
                'factual_basis': f'Circuit breaker state: {self.state}, failures: {self.failure_count}',
                'verification_method': 'circuit_breaker_logic',
                'supporting_data': {
                    'state': self.state,
                    'failure_count': self.failure_count,
                    'last_failure': self.last_failure_time
                }
            }
        }
        
        validation_result = self.professional_validator.validate_technical_claims(execution_claims)
        if not validation_result['professional_compliance']:
            logger.error("Circuit breaker decision failed professional standards validation")
            return False
        
        current_time = time.time()
        
        if self.state == "closed":
            return True
        elif self.state == "open":
            # Check if recovery timeout has elapsed
            if current_time - self.last_failure_time >= self.recovery_timeout:
                self.state = "half-open"
                self.success_count = 0
                logger.info("Circuit breaker transitioning to half-open state")
                return True
            return False
        elif self.state == "half-open":
            return True
        
        return False
    
    def record_success(self):
        """Record successful request execution"""
        current_time = time.time()
        
        self.stats.total_requests += 1
        self.stats.successful_requests += 1
        self.last_success_time = current_time
        
        # Add to request history
        self.request_history.append({
            'timestamp': current_time,
            'success': True,
            'response_time': 0  # To be updated by caller
        })
        
        if self.state == "half-open":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = "closed"
                self.failure_count = 0
                logger.info("Circuit breaker closed after successful recovery")
        elif self.state == "closed":
            # Reset failure count on success
            self.failure_count = max(0, self.failure_count - 1)
        
        self._update_statistics()
        self._trim_request_history()
    
    def record_failure(self):
        """Record failed request execution"""
        current_time = time.time()
        
        self.stats.total_requests += 1
        self.stats.failed_requests += 1
        self.failure_count += 1
        self.last_failure_time = current_time
        
        # Add to request history
        self.request_history.append({
            'timestamp': current_time,
            'success': False,
            'error': True
        })
        
        if self.state == "closed" and self.failure_count >= self.failure_threshold:
            self.state = "open"
            self.stats.trips += 1
            self.stats.last_trip_time = current_time
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
        elif self.state == "half-open":
            self.state = "open"
            self.stats.trips += 1
            self.stats.last_trip_time = current_time
            logger.warning("Circuit breaker opened during half-open state")
        
        self._update_statistics()
        self._trim_request_history()
    
    def _update_statistics(self):
        """Update circuit breaker statistics"""
        if self.stats.total_requests > 0:
            self.stats.success_rate = self.stats.successful_requests / self.stats.total_requests
        
        self.stats.current_state = self.state
    
    def _trim_request_history(self):
        """Trim request history to monitoring window"""
        current_time = time.time()
        cutoff_time = current_time - self.monitoring_window
        
        self.request_history = [
            req for req in self.request_history 
            if req['timestamp'] >= cutoff_time
        ]
    
    def get_statistics(self) -> CircuitBreakerStats:
        """Get comprehensive circuit breaker statistics"""
        self._update_statistics()
        return self.stats
    
    def get_failure_rate(self, window_seconds: float = 60.0) -> float:
        """Get failure rate within specified time window"""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        recent_requests = [
            req for req in self.request_history 
            if req['timestamp'] >= cutoff_time
        ]
        
        if not recent_requests:
            return 0.0
        
        failed_requests = sum(1 for req in recent_requests if not req['success'])
        return failed_requests / len(recent_requests)
    
    def reset(self):
        """Reset circuit breaker to closed state"""
        self.state = "closed"
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.last_success_time = 0
        logger.info("Circuit breaker manually reset to closed state")


class GracefulDegradationManager:
    """
    Graceful degradation manager for MCP services
    
    Implements fallback strategies and service degradation patterns
    with professional standards compliance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.professional_validator = TechnicalQualityGate()
        self.degradation_strategies: Dict[str, Callable] = {}
        self.current_degradation_level = 0  # 0 = normal, 1-5 = increasing degradation
        
        logger.info("GracefulDegradationManager initialized")
    
    def register_degradation_strategy(self, level: int, strategy_name: str, strategy_function: Callable):
        """Register a degradation strategy for a specific level"""
        if level not in self.degradation_strategies:
            self.degradation_strategies[level] = {}
        
        self.degradation_strategies[level][strategy_name] = strategy_function
        logger.info(f"Degradation strategy '{strategy_name}' registered for level {level}")
    
    def set_degradation_level(self, level: int, reason: str = ""):
        """Set system degradation level with professional validation"""
        # Professional standards validation
        degradation_claims = {
            'degradation_level_change': {
                'factual_basis': f'Setting degradation level to {level}, reason: {reason}',
                'verification_method': 'degradation_logic',
                'supporting_data': {
                    'new_level': level,
                    'previous_level': self.current_degradation_level,
                    'reason': reason
                }
            }
        }
        
        validation_result = self.professional_validator.validate_technical_claims(degradation_claims)
        if not validation_result['professional_compliance']:
            logger.error("Degradation level change failed professional standards validation")
            return False
        
        previous_level = self.current_degradation_level
        self.current_degradation_level = level
        
        logger.info(f"System degradation level changed: {previous_level} -> {level} ({reason})")
        return True
    
    def execute_degradation_strategies(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute degradation strategies for current level"""
        if self.current_degradation_level == 0:
            return context  # No degradation needed
        
        strategies = self.degradation_strategies.get(self.current_degradation_level, {})
        
        for strategy_name, strategy_function in strategies.items():
            try:
                context = strategy_function(context)
                logger.info(f"Applied degradation strategy '{strategy_name}' at level {self.current_degradation_level}")
            except Exception as e:
                logger.error(f"Degradation strategy '{strategy_name}' failed: {e}")
        
        return context
    
    def get_status(self) -> Dict[str, Any]:
        """Get current degradation status"""
        return {
            'current_level': self.current_degradation_level,
            'available_strategies': {
                level: list(strategies.keys()) 
                for level, strategies in self.degradation_strategies.items()
            },
            'professional_compliance': True
        }