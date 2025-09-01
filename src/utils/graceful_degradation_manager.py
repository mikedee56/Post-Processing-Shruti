"""
Graceful Degradation Manager - Story 3.4 Performance Optimization and Monitoring

This module provides comprehensive graceful degradation modes for semantic processing
services, ensuring system stability and acceptable performance when services are 
unavailable or experiencing failures.

Features:
- Multiple degradation levels with automatic switching
- Service health monitoring and automatic recovery
- Performance impact minimization
- Integration with existing processing pipeline
- Fallback to core functionality when semantic services fail

Author: Development Team
Date: 2025-01-30
Epic: 3 - Semantic Refinement & QA Framework
Story: 3.4 - Performance Optimization and Monitoring
"""

import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Callable, Union, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from ..utils.logger_config import get_logger
from ..utils.config_manager import ConfigManager
from .semantic_circuit_breaker import SemanticCircuitBreakerManager, SemanticServiceType


class DegradationLevel(Enum):
    """Levels of service degradation."""
    NORMAL = "normal"                    # All services operational
    PARTIAL = "partial"                  # Some services degraded, full functionality available
    LIMITED = "limited"                  # Limited functionality, core features available
    MINIMAL = "minimal"                  # Minimal functionality, essential features only
    EMERGENCY = "emergency"              # Emergency mode, basic processing only


class ServiceAvailability(Enum):
    """Service availability states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass
class DegradationConfig:
    """Configuration for graceful degradation."""
    # Degradation thresholds
    partial_threshold_failures: int = 2
    limited_threshold_failures: int = 4
    minimal_threshold_failures: int = 6
    emergency_threshold_failures: int = 8
    
    # Recovery thresholds
    recovery_success_threshold: int = 3
    recovery_check_interval_seconds: float = 30.0
    
    # Performance settings
    max_degradation_overhead_ms: float = 50.0
    enable_automatic_recovery: bool = True
    enable_performance_fallbacks: bool = True
    
    # Monitoring settings
    health_check_interval_seconds: float = 10.0
    metrics_retention_hours: int = 24


@dataclass
class ServiceHealth:
    """Health status for a service."""
    service_name: str
    service_type: str
    availability: ServiceAvailability = ServiceAvailability.UNKNOWN
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    response_time_ms: float = 0.0
    error_rate: float = 0.0
    last_health_check: Optional[datetime] = None


@dataclass
class DegradationState:
    """Current degradation state."""
    level: DegradationLevel = DegradationLevel.NORMAL
    activated_at: Optional[datetime] = None
    reason: Optional[str] = None
    affected_services: List[str] = field(default_factory=list)
    performance_impact_ms: float = 0.0
    fallback_strategies_active: List[str] = field(default_factory=list)


class GracefulDegradationManager:
    """
    Manager for graceful degradation of semantic processing services.
    
    This manager monitors service health and automatically switches between
    degradation levels to maintain system stability and acceptable performance.
    """
    
    def __init__(
        self,
        config: Optional[DegradationConfig] = None,
        config_manager: Optional[ConfigManager] = None,
        circuit_breaker_manager: Optional[SemanticCircuitBreakerManager] = None
    ):
        """
        Initialize graceful degradation manager.
        
        Args:
            config: Degradation configuration
            config_manager: Configuration manager
            circuit_breaker_manager: Circuit breaker manager integration
        """
        self.logger = get_logger(__name__)
        self.config = config or DegradationConfig()
        self.config_manager = config_manager or ConfigManager()
        self.circuit_breaker_manager = circuit_breaker_manager
        
        # State management
        self.current_state = DegradationState()
        self.service_health: Dict[str, ServiceHealth] = {}
        self.degradation_history: List[DegradationState] = []
        
        # Monitoring and recovery
        self.monitoring_active = False
        self.monitoring_thread = None
        self.recovery_attempts = 0
        
        # Fallback strategies by degradation level
        self.degradation_strategies = {
            DegradationLevel.NORMAL: self._normal_mode_strategy,
            DegradationLevel.PARTIAL: self._partial_degradation_strategy,
            DegradationLevel.LIMITED: self._limited_degradation_strategy,
            DegradationLevel.MINIMAL: self._minimal_degradation_strategy,
            DegradationLevel.EMERGENCY: self._emergency_degradation_strategy
        }
        
        # Performance tracking
        self.performance_metrics = {
            'degradation_activations': 0,
            'automatic_recoveries': 0,
            'manual_interventions': 0,
            'total_degraded_time_seconds': 0.0,
            'fallback_executions': 0
        }
        
        self.logger.info("Graceful degradation manager initialized for Story 3.4")
    
    def register_service(
        self,
        service_name: str,
        service_type: str,
        health_check_func: Optional[Callable[[], bool]] = None
    ) -> None:
        """
        Register a service for degradation monitoring.
        
        Args:
            service_name: Name of the service
            service_type: Type of service (semantic, cache, etc.)
            health_check_func: Optional health check function
        """
        self.service_health[service_name] = ServiceHealth(
            service_name=service_name,
            service_type=service_type
        )
        
        self.logger.info(f"Registered service for degradation monitoring: {service_name}")
    
    def record_service_call(
        self,
        service_name: str,
        success: bool,
        response_time_ms: float = 0.0,
        error_message: Optional[str] = None
    ) -> None:
        """
        Record a service call result for health monitoring.
        
        Args:
            service_name: Name of the service
            success: Whether the call was successful
            response_time_ms: Response time in milliseconds
            error_message: Error message if call failed
        """
        if service_name not in self.service_health:
            self.register_service(service_name, "unknown")
        
        health = self.service_health[service_name]
        current_time = datetime.now(timezone.utc)
        
        if success:
            health.last_success = current_time
            health.consecutive_successes += 1
            health.consecutive_failures = 0
            health.availability = ServiceAvailability.HEALTHY
        else:
            health.last_failure = current_time
            health.consecutive_failures += 1
            health.consecutive_successes = 0
            
            if health.consecutive_failures >= 3:
                health.availability = ServiceAvailability.UNAVAILABLE
            elif health.consecutive_failures >= 1:
                health.availability = ServiceAvailability.DEGRADED
        
        health.response_time_ms = response_time_ms
        health.last_health_check = current_time
        
        # Check if degradation level needs to change
        self._evaluate_degradation_level()
        
        if not success:
            self.logger.debug(f"Service call failed: {service_name} - {error_message}")
    
    def execute_with_degradation(
        self,
        service_name: str,
        primary_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with degradation-aware fallback strategies.
        
        Args:
            service_name: Name of the service
            primary_func: Primary function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or degraded result
        """
        start_time = time.time()
        
        try:
            # Get current degradation strategy
            strategy = self.degradation_strategies[self.current_state.level]
            
            # Execute with degradation awareness
            result = strategy(service_name, primary_func, *args, **kwargs)
            
            # Record successful execution
            execution_time = (time.time() - start_time) * 1000
            self.record_service_call(service_name, True, execution_time)
            
            return result
            
        except Exception as e:
            # Record failed execution
            execution_time = (time.time() - start_time) * 1000
            self.record_service_call(service_name, False, execution_time, str(e))
            
            # Try fallback strategies based on current degradation level
            fallback_result = self._execute_degraded_fallback(
                service_name, primary_func, *args, **kwargs
            )
            
            if fallback_result is not None:
                return fallback_result
            
            # Re-raise if no fallback worked
            raise e
    
    def _normal_mode_strategy(
        self,
        service_name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Normal mode - execute function directly."""
        return func(*args, **kwargs)
    
    def _partial_degradation_strategy(
        self,
        service_name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Partial degradation - some optimizations disabled, full functionality maintained.
        """
        try:
            # Try normal execution first
            return func(*args, **kwargs)
        except Exception:
            # Fall back to degraded execution
            return self._execute_with_reduced_features(service_name, func, *args, **kwargs)
    
    def _limited_degradation_strategy(
        self,
        service_name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Limited degradation - reduced functionality, core features available.
        """
        # Execute with limited features
        return self._execute_with_limited_features(service_name, func, *args, **kwargs)
    
    def _minimal_degradation_strategy(
        self,
        service_name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Minimal degradation - minimal functionality, essential features only.
        """
        return self._execute_with_minimal_features(service_name, func, *args, **kwargs)
    
    def _emergency_degradation_strategy(
        self,
        service_name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Emergency degradation - basic processing only.
        """
        return self._execute_emergency_mode(service_name, func, *args, **kwargs)
    
    def _execute_with_reduced_features(
        self,
        service_name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute with reduced feature set."""
        # Remove performance-intensive features
        if 'enable_caching' in kwargs:
            kwargs['enable_caching'] = False
        if 'parallel_processing' in kwargs:
            kwargs['parallel_processing'] = False
        
        return func(*args, **kwargs)
    
    def _execute_with_limited_features(
        self,
        service_name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute with limited feature set."""
        # Service-specific limited execution
        if 'semantic' in service_name.lower():
            return self._execute_limited_semantic_processing(*args, **kwargs)
        elif 'quality' in service_name.lower():
            return self._execute_limited_quality_gates(*args, **kwargs)
        elif 'cache' in service_name.lower():
            return self._execute_limited_caching(*args, **kwargs)
        else:
            return self._execute_basic_fallback(*args, **kwargs)
    
    def _execute_with_minimal_features(
        self,
        service_name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute with minimal feature set."""
        # Service-specific minimal execution
        if 'semantic' in service_name.lower():
            return self._execute_minimal_semantic_processing(*args, **kwargs)
        elif 'quality' in service_name.lower():
            return self._execute_minimal_quality_gates(*args, **kwargs)
        else:
            return self._execute_basic_fallback(*args, **kwargs)
    
    def _execute_emergency_mode(
        self,
        service_name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute in emergency mode with absolute minimal processing."""
        # Return basic/default results
        if 'semantic' in service_name.lower():
            return self._emergency_semantic_result(*args, **kwargs)
        elif 'quality' in service_name.lower():
            return True  # Pass everything in emergency mode
        else:
            return None  # Basic fallback
    
    def _execute_limited_semantic_processing(self, *args, **kwargs) -> Any:
        """Limited semantic processing fallback."""
        # Use basic text similarity instead of full semantic processing
        if len(args) >= 2:
            text1, text2 = str(args[0]), str(args[1])
            # Basic similarity calculation
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 and not words2:
                similarity = 1.0
            elif not words1 or not words2:
                similarity = 0.0
            else:
                intersection = words1.intersection(words2)
                union = words1.union(words2)
                similarity = len(intersection) / len(union)
            
            # Return similarity result object
            return type('LimitedSimilarityResult', (), {
                'similarity_score': similarity,
                'cache_hit': False,
                'processing_time_ms': 5.0,
                'degraded_mode': True
            })()
        
        return None
    
    def _execute_minimal_semantic_processing(self, *args, **kwargs) -> Any:
        """Minimal semantic processing fallback."""
        # Return neutral similarity
        return type('MinimalSimilarityResult', (), {
            'similarity_score': 0.5,
            'cache_hit': False,
            'processing_time_ms': 1.0,
            'degraded_mode': True
        })()
    
    def _execute_limited_quality_gates(self, *args, **kwargs) -> Any:
        """Limited quality gate processing."""
        if len(args) >= 1:
            text = str(args[0])
            # Basic validation
            return len(text.strip()) > 0 and len(text) < 10000
        return True
    
    def _execute_minimal_quality_gates(self, *args, **kwargs) -> Any:
        """Minimal quality gate processing."""
        # Just check for non-empty text
        if len(args) >= 1:
            return len(str(args[0]).strip()) > 0
        return True
    
    def _execute_limited_caching(self, *args, **kwargs) -> Any:
        """Limited caching fallback."""
        # Use in-memory cache only, no persistent storage
        return None
    
    def _execute_basic_fallback(self, *args, **kwargs) -> Any:
        """Basic fallback for unknown services."""
        return None
    
    def _emergency_semantic_result(self, *args, **kwargs) -> Any:
        """Emergency semantic processing result."""
        return type('EmergencyResult', (), {
            'similarity_score': 0.0,
            'cache_hit': False,
            'processing_time_ms': 0.1,
            'emergency_mode': True
        })()
    
    def _execute_degraded_fallback(
        self,
        service_name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute fallback based on current degradation level."""
        self.performance_metrics['fallback_executions'] += 1
        
        try:
            current_strategy = self.degradation_strategies[self.current_state.level]
            return current_strategy(service_name, func, *args, **kwargs)
        except Exception as e:
            self.logger.error(f"Degraded fallback failed for {service_name}: {e}")
            return None
    
    def _evaluate_degradation_level(self) -> None:
        """Evaluate and potentially change degradation level based on service health."""
        failed_services = []
        degraded_services = []
        
        for service_name, health in self.service_health.items():
            if health.availability == ServiceAvailability.UNAVAILABLE:
                failed_services.append(service_name)
            elif health.availability == ServiceAvailability.DEGRADED:
                degraded_services.append(service_name)
        
        total_failures = len(failed_services) + len(degraded_services)
        
        # Determine appropriate degradation level
        new_level = DegradationLevel.NORMAL
        
        if total_failures >= self.config.emergency_threshold_failures:
            new_level = DegradationLevel.EMERGENCY
        elif total_failures >= self.config.minimal_threshold_failures:
            new_level = DegradationLevel.MINIMAL
        elif total_failures >= self.config.limited_threshold_failures:
            new_level = DegradationLevel.LIMITED
        elif total_failures >= self.config.partial_threshold_failures:
            new_level = DegradationLevel.PARTIAL
        
        # Change degradation level if needed
        if new_level != self.current_state.level:
            self._change_degradation_level(
                new_level,
                f"Service failures: {total_failures} (failed: {len(failed_services)}, degraded: {len(degraded_services)})",
                failed_services + degraded_services
            )
    
    def _change_degradation_level(
        self,
        new_level: DegradationLevel,
        reason: str,
        affected_services: List[str]
    ) -> None:
        """Change degradation level."""
        old_level = self.current_state.level
        
        # Update degradation state
        self.current_state = DegradationState(
            level=new_level,
            activated_at=datetime.now(timezone.utc),
            reason=reason,
            affected_services=affected_services.copy()
        )
        
        # Update performance metrics
        if new_level != DegradationLevel.NORMAL:
            self.performance_metrics['degradation_activations'] += 1
        elif old_level != DegradationLevel.NORMAL:
            self.performance_metrics['automatic_recoveries'] += 1
        
        # Add to history
        self.degradation_history.append(self.current_state)
        
        # Log the change
        if new_level == DegradationLevel.NORMAL:
            self.logger.info(f"Recovered to normal operation from {old_level.value}")
        else:
            self.logger.warning(
                f"Degradation level changed: {old_level.value} → {new_level.value}. "
                f"Reason: {reason}"
            )
    
    def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.logger.info("Started degradation monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Stopped degradation monitoring")
    
    def _monitoring_loop(self) -> None:
        """Continuous monitoring loop."""
        while self.monitoring_active:
            try:
                # Perform health checks
                self._perform_health_checks()
                
                # Evaluate degradation level
                self._evaluate_degradation_level()
                
                # Check for automatic recovery opportunities
                if self.config.enable_automatic_recovery:
                    self._check_automatic_recovery()
                
                time.sleep(self.config.health_check_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.config.health_check_interval_seconds)
    
    def _perform_health_checks(self) -> None:
        """Perform health checks for all registered services."""
        for service_name, health in self.service_health.items():
            # Update health based on recent activity
            current_time = datetime.now(timezone.utc)
            
            # If no recent activity, mark as unknown
            if health.last_health_check:
                time_since_last_check = current_time - health.last_health_check
                if time_since_last_check.total_seconds() > 60:  # 1 minute
                    health.availability = ServiceAvailability.UNKNOWN
    
    def _check_automatic_recovery(self) -> None:
        """Check for automatic recovery opportunities."""
        if self.current_state.level == DegradationLevel.NORMAL:
            return
        
        # Count healthy services
        healthy_services = sum(
            1 for health in self.service_health.values()
            if health.availability == ServiceAvailability.HEALTHY
        )
        
        total_services = len(self.service_health)
        
        # If majority of services are healthy, attempt recovery
        if healthy_services >= total_services * 0.7:  # 70% healthy
            self._attempt_automatic_recovery()
    
    def _attempt_automatic_recovery(self) -> None:
        """Attempt automatic recovery to normal operation."""
        self.recovery_attempts += 1
        
        self.logger.info(f"Attempting automatic recovery (attempt {self.recovery_attempts})")
        
        # Gradual recovery - move up one level at a time
        current_level = self.current_state.level
        
        if current_level == DegradationLevel.EMERGENCY:
            new_level = DegradationLevel.MINIMAL
        elif current_level == DegradationLevel.MINIMAL:
            new_level = DegradationLevel.LIMITED
        elif current_level == DegradationLevel.LIMITED:
            new_level = DegradationLevel.PARTIAL
        elif current_level == DegradationLevel.PARTIAL:
            new_level = DegradationLevel.NORMAL
        else:
            return  # Already at normal
        
        self._change_degradation_level(
            new_level,
            f"Automatic recovery attempt {self.recovery_attempts}",
            []
        )
    
    def force_degradation_level(self, level: DegradationLevel, reason: str = "Manual override") -> None:
        """Force specific degradation level (for testing or manual intervention)."""
        self.performance_metrics['manual_interventions'] += 1
        
        self._change_degradation_level(level, reason, [])
        
        self.logger.warning(f"Forced degradation level to {level.value}: {reason}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system degradation status."""
        return {
            'current_degradation': {
                'level': self.current_state.level.value,
                'activated_at': self.current_state.activated_at.isoformat() if self.current_state.activated_at else None,
                'reason': self.current_state.reason,
                'affected_services': self.current_state.affected_services,
                'duration_seconds': (
                    (datetime.now(timezone.utc) - self.current_state.activated_at).total_seconds()
                    if self.current_state.activated_at else 0
                )
            },
            'service_health': {
                name: {
                    'availability': health.availability.value,
                    'consecutive_failures': health.consecutive_failures,
                    'consecutive_successes': health.consecutive_successes,
                    'last_success': health.last_success.isoformat() if health.last_success else None,
                    'last_failure': health.last_failure.isoformat() if health.last_failure else None,
                    'response_time_ms': health.response_time_ms
                }
                for name, health in self.service_health.items()
            },
            'performance_metrics': self.performance_metrics.copy(),
            'monitoring_active': self.monitoring_active,
            'recovery_attempts': self.recovery_attempts,
            'total_services': len(self.service_health),
            'healthy_services': sum(
                1 for h in self.service_health.values()
                if h.availability == ServiceAvailability.HEALTHY
            ),
            'degraded_services': sum(
                1 for h in self.service_health.values()
                if h.availability == ServiceAvailability.DEGRADED
            ),
            'unavailable_services': sum(
                1 for h in self.service_health.values()
                if h.availability == ServiceAvailability.UNAVAILABLE
            )
        }
    
    def export_degradation_report(self, output_file: Path) -> None:
        """Export degradation status report."""
        try:
            report = {
                'report_metadata': {
                    'generated_at': datetime.now(timezone.utc).isoformat(),
                    'story': '3.4 - Performance Optimization and Monitoring - Graceful Degradation',
                    'monitoring_active': self.monitoring_active
                },
                'system_status': self.get_system_status(),
                'degradation_history': [
                    {
                        'level': state.level.value,
                        'activated_at': state.activated_at.isoformat() if state.activated_at else None,
                        'reason': state.reason,
                        'affected_services': state.affected_services
                    }
                    for state in self.degradation_history[-20:]  # Last 20 events
                ],
                'configuration': {
                    'partial_threshold': self.config.partial_threshold_failures,
                    'limited_threshold': self.config.limited_threshold_failures,
                    'minimal_threshold': self.config.minimal_threshold_failures,
                    'emergency_threshold': self.config.emergency_threshold_failures,
                    'automatic_recovery_enabled': self.config.enable_automatic_recovery
                }
            }
            
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Degradation report exported to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error exporting degradation report: {e}")
            raise


# Global degradation manager
_global_degradation_manager = None


def initialize_graceful_degradation(
    config: Optional[DegradationConfig] = None,
    config_manager: Optional[ConfigManager] = None,
    circuit_breaker_manager: Optional[SemanticCircuitBreakerManager] = None
) -> GracefulDegradationManager:
    """Initialize global graceful degradation manager."""
    global _global_degradation_manager
    _global_degradation_manager = GracefulDegradationManager(
        config=config,
        config_manager=config_manager,
        circuit_breaker_manager=circuit_breaker_manager
    )
    return _global_degradation_manager


def get_degradation_manager() -> Optional[GracefulDegradationManager]:
    """Get global degradation manager."""
    return _global_degradation_manager


def execute_with_degradation(
    service_name: str,
    func: Callable,
    *args,
    **kwargs
) -> Any:
    """Execute function with global degradation manager."""
    manager = get_degradation_manager()
    if manager:
        return manager.execute_with_degradation(service_name, func, *args, **kwargs)
    else:
        return func(*args, **kwargs)


def record_service_result(
    service_name: str,
    success: bool,
    response_time_ms: float = 0.0,
    error_message: Optional[str] = None
) -> None:
    """Record service result with global degradation manager."""
    manager = get_degradation_manager()
    if manager:
        manager.record_service_call(service_name, success, response_time_ms, error_message)