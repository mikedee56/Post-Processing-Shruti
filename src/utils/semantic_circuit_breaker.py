"""
Semantic Circuit Breaker - Story 3.4 Performance Optimization and Monitoring

This module provides circuit breaker patterns specifically designed for semantic processing
external dependencies, ensuring graceful degradation when semantic services are unavailable
while maintaining system stability.

Features:
- Circuit breaker patterns for external dependencies
- Graceful degradation modes for service failures
- Automatic fallback to cached results
- Performance impact minimization
- Integration with existing circuit breaker infrastructure

Author: Development Team
Date: 2025-01-30
Epic: 3 - Semantic Refinement & QA Framework
Story: 3.4 - Performance Optimization and Monitoring
"""

import time
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Callable, Union, List
from dataclasses import dataclass, field
from enum import Enum
import threading
from functools import wraps
import logging

from ..utils.logger_config import get_logger
from ..utils.circuit_breaker import CircuitBreaker, CircuitBreakerState, CircuitBreakerError
from .cache_manager import CacheManager


class SemanticServiceType(Enum):
    """Types of semantic services that may require circuit breaking."""
    EMBEDDING_SERVICE = "embedding_service"
    SIMILARITY_SERVICE = "similarity_service" 
    QUALITY_GATE_SERVICE = "quality_gate_service"
    CACHE_SERVICE = "cache_service"
    EXTERNAL_API = "external_api"


@dataclass
class SemanticCircuitBreakerConfig:
    """Configuration for semantic-specific circuit breakers."""
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 60.0
    half_open_max_calls: int = 3
    timeout_seconds: float = 30.0
    
    # Semantic-specific settings
    enable_cache_fallback: bool = True
    enable_degraded_mode: bool = True
    cache_ttl_seconds: int = 3600
    degraded_mode_timeout_seconds: float = 300.0


@dataclass
class SemanticFallbackResult:
    """Result from fallback operations."""
    success: bool = False
    result: Any = None
    fallback_used: str = "none"
    performance_impact_ms: float = 0.0
    cached_result: bool = False
    error_message: Optional[str] = None


class SemanticCircuitBreaker:
    """
    Circuit breaker specifically designed for semantic processing services.
    
    This circuit breaker provides semantic-aware fallback strategies and
    graceful degradation modes to ensure system stability when external
    semantic services are unavailable.
    """
    
    def __init__(
        self,
        service_type: SemanticServiceType,
        service_name: str,
        config: Optional[SemanticCircuitBreakerConfig] = None,
        cache_manager: Optional[CacheManager] = None
    ):
        """
        Initialize semantic circuit breaker.
        
        Args:
            service_type: Type of semantic service
            service_name: Name identifier for the service
            config: Circuit breaker configuration
            cache_manager: Cache manager for fallback operations
        """
        self.service_type = service_type
        self.service_name = service_name
        self.config = config or SemanticCircuitBreakerConfig()
        self.cache_manager = cache_manager or CacheManager()
        self.logger = get_logger(__name__)
        
        # Initialize base circuit breaker
        self.circuit_breaker = CircuitBreaker(
            name=f"semantic_{service_type.value}_{service_name}",
            failure_threshold=self.config.failure_threshold,
            recovery_timeout=self.config.recovery_timeout_seconds,
            expected_exception=Exception
        )
        
        # Semantic-specific state
        self.degraded_mode_active = False
        self.degraded_mode_start_time = None
        self.fallback_strategies = {}
        self.performance_metrics = {
            'circuit_breaks': 0,
            'fallback_successes': 0,
            'fallback_failures': 0,
            'degraded_mode_activations': 0,
            'cache_hits_during_failure': 0
        }
        
        # Setup default fallback strategies
        self._setup_default_fallback_strategies()
        
        self.logger.info(f"Initialized semantic circuit breaker for {service_type.value}:{service_name}")
    
    def _setup_default_fallback_strategies(self) -> None:
        """Setup default fallback strategies based on service type."""
        if self.service_type == SemanticServiceType.EMBEDDING_SERVICE:
            self.fallback_strategies = {
                'cache_lookup': self._cache_embedding_fallback,
                'basic_embedding': self._basic_embedding_fallback,
                'null_embedding': self._null_embedding_fallback
            }
        elif self.service_type == SemanticServiceType.SIMILARITY_SERVICE:
            self.fallback_strategies = {
                'cache_lookup': self._cache_similarity_fallback,
                'basic_similarity': self._basic_similarity_fallback,
                'default_similarity': self._default_similarity_fallback
            }
        elif self.service_type == SemanticServiceType.QUALITY_GATE_SERVICE:
            self.fallback_strategies = {
                'cached_evaluation': self._cache_quality_gate_fallback,
                'basic_validation': self._basic_quality_gate_fallback,
                'pass_through': self._passthrough_quality_gate_fallback
            }
        elif self.service_type == SemanticServiceType.CACHE_SERVICE:
            self.fallback_strategies = {
                'memory_cache': self._memory_cache_fallback,
                'file_cache': self._file_cache_fallback,
                'no_cache': self._no_cache_fallback
            }
        else:
            self.fallback_strategies = {
                'cached_result': self._generic_cache_fallback,
                'default_result': self._generic_default_fallback
            }
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection and semantic fallbacks.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or fallback result
            
        Raises:
            CircuitBreakerError: When circuit is open and no fallback available
        """
        start_time = time.time()
        
        try:
            # Try to execute through circuit breaker
            result = self.circuit_breaker.call(func, *args, **kwargs)
            
            # If degraded mode was active, check if we can exit it
            if self.degraded_mode_active:
                self._check_degraded_mode_recovery()
            
            return result
            
        except CircuitBreakerError as e:
            self.performance_metrics['circuit_breaks'] += 1
            
            # Circuit is open - try fallback strategies
            fallback_result = self._execute_fallback_strategies(*args, **kwargs)
            
            if fallback_result.success:
                self.performance_metrics['fallback_successes'] += 1
                self.logger.info(
                    f"Fallback successful for {self.service_name} using {fallback_result.fallback_used}"
                )
                return fallback_result.result
            else:
                self.performance_metrics['fallback_failures'] += 1
                
                # Enter degraded mode if enabled
                if self.config.enable_degraded_mode and not self.degraded_mode_active:
                    self._enter_degraded_mode()
                
                # Re-raise the original error if no fallback worked
                raise e
        
        except Exception as e:
            # Regular function execution error - let circuit breaker handle it
            raise e
        
        finally:
            # Record performance impact
            execution_time = (time.time() - start_time) * 1000  # ms
            self._record_execution_time(execution_time)
    
    def _execute_fallback_strategies(self, *args, **kwargs) -> SemanticFallbackResult:
        """Execute fallback strategies in order of preference."""
        for strategy_name, strategy_func in self.fallback_strategies.items():
            try:
                self.logger.debug(f"Trying fallback strategy: {strategy_name}")
                result = strategy_func(*args, **kwargs)
                
                if result.success:
                    result.fallback_used = strategy_name
                    return result
                    
            except Exception as e:
                self.logger.debug(f"Fallback strategy {strategy_name} failed: {e}")
                continue
        
        # All fallback strategies failed
        return SemanticFallbackResult(
            success=False,
            error_message="All fallback strategies failed"
        )
    
    def _cache_embedding_fallback(self, text: str, language: str = "sa", **kwargs) -> SemanticFallbackResult:
        """Fallback strategy for embedding service using cache lookup."""
        if not self.config.enable_cache_fallback:
            return SemanticFallbackResult(success=False)
        
        cache_key = f"embedding_{hash(text)}_{language}"
        cached_result = self.cache_manager.get(cache_key)
        
        if cached_result is not None:
            self.performance_metrics['cache_hits_during_failure'] += 1
            return SemanticFallbackResult(
                success=True,
                result=cached_result,
                cached_result=True,
                performance_impact_ms=1.0  # Cache lookup is very fast
            )
        
        return SemanticFallbackResult(success=False)
    
    def _basic_embedding_fallback(self, text: str, language: str = "sa", **kwargs) -> SemanticFallbackResult:
        """Basic embedding fallback using simple text features."""
        try:
            # Create a basic embedding based on text characteristics
            # This is a simplified fallback - real implementation would use
            # a lightweight local model or text statistics
            
            text_features = [
                len(text),
                len(text.split()),
                text.count('ा'),  # Sanskrit vowel count as feature
                text.count(' '),
                hash(text[:50]) % 1000 / 1000.0  # Normalized hash as feature
            ]
            
            # Pad to standard embedding size (simplified)
            basic_embedding = text_features + [0.0] * (300 - len(text_features))
            
            return SemanticFallbackResult(
                success=True,
                result=basic_embedding,
                performance_impact_ms=5.0
            )
            
        except Exception as e:
            return SemanticFallbackResult(
                success=False,
                error_message=f"Basic embedding fallback failed: {e}"
            )
    
    def _null_embedding_fallback(self, text: str, language: str = "sa", **kwargs) -> SemanticFallbackResult:
        """Null embedding fallback - returns zero vector."""
        return SemanticFallbackResult(
            success=True,
            result=[0.0] * 300,  # Standard embedding size
            performance_impact_ms=0.1
        )
    
    def _cache_similarity_fallback(self, text1: str, text2: str, **kwargs) -> SemanticFallbackResult:
        """Cache-based similarity fallback."""
        if not self.config.enable_cache_fallback:
            return SemanticFallbackResult(success=False)
        
        cache_key = f"similarity_{hash(text1)}_{hash(text2)}"
        cached_result = self.cache_manager.get(cache_key)
        
        if cached_result is not None:
            self.performance_metrics['cache_hits_during_failure'] += 1
            return SemanticFallbackResult(
                success=True,
                result=cached_result,
                cached_result=True,
                performance_impact_ms=1.0
            )
        
        return SemanticFallbackResult(success=False)
    
    def _basic_similarity_fallback(self, text1: str, text2: str, **kwargs) -> SemanticFallbackResult:
        """Basic similarity fallback using text overlap."""
        try:
            # Simple similarity based on word overlap
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
            
            # Create similarity result object
            result = type('SimilarityResult', (), {
                'similarity_score': similarity,
                'cache_hit': False,
                'processing_time_ms': 2.0
            })()
            
            return SemanticFallbackResult(
                success=True,
                result=result,
                performance_impact_ms=2.0
            )
            
        except Exception as e:
            return SemanticFallbackResult(
                success=False,
                error_message=f"Basic similarity fallback failed: {e}"
            )
    
    def _default_similarity_fallback(self, text1: str, text2: str, **kwargs) -> SemanticFallbackResult:
        """Default similarity fallback - returns neutral similarity."""
        result = type('SimilarityResult', (), {
            'similarity_score': 0.5,  # Neutral similarity
            'cache_hit': False,
            'processing_time_ms': 0.1
        })()
        
        return SemanticFallbackResult(
            success=True,
            result=result,
            performance_impact_ms=0.1
        )
    
    def _cache_quality_gate_fallback(self, text: str, **kwargs) -> SemanticFallbackResult:
        """Cache-based quality gate fallback."""
        if not self.config.enable_cache_fallback:
            return SemanticFallbackResult(success=False)
        
        cache_key = f"quality_gate_{hash(text)}"
        cached_result = self.cache_manager.get(cache_key)
        
        if cached_result is not None:
            return SemanticFallbackResult(
                success=True,
                result=cached_result,
                cached_result=True,
                performance_impact_ms=1.0
            )
        
        return SemanticFallbackResult(success=False)
    
    def _basic_quality_gate_fallback(self, text: str, **kwargs) -> SemanticFallbackResult:
        """Basic quality gate fallback using simple validation."""
        try:
            # Basic quality checks
            is_valid = (
                len(text.strip()) > 0 and
                len(text) < 10000 and  # Reasonable length
                not any(char in text for char in ['<', '>', '{', '}'])  # No markup
            )
            
            return SemanticFallbackResult(
                success=True,
                result=is_valid,
                performance_impact_ms=1.0
            )
            
        except Exception as e:
            return SemanticFallbackResult(
                success=False,
                error_message=f"Basic quality gate fallback failed: {e}"
            )
    
    def _passthrough_quality_gate_fallback(self, text: str, **kwargs) -> SemanticFallbackResult:
        """Pass-through quality gate - always passes."""
        return SemanticFallbackResult(
            success=True,
            result=True,  # Always pass
            performance_impact_ms=0.1
        )
    
    def _memory_cache_fallback(self, key: str, **kwargs) -> SemanticFallbackResult:
        """Memory-based cache fallback."""
        # This would use an in-memory cache as fallback
        return SemanticFallbackResult(success=False)  # Simplified for now
    
    def _file_cache_fallback(self, key: str, **kwargs) -> SemanticFallbackResult:
        """File-based cache fallback."""
        # This would use file system as cache fallback
        return SemanticFallbackResult(success=False)  # Simplified for now
    
    def _no_cache_fallback(self, key: str, **kwargs) -> SemanticFallbackResult:
        """No cache fallback - just continue without caching."""
        return SemanticFallbackResult(
            success=True,
            result=None,  # No cached value
            performance_impact_ms=0.1
        )
    
    def _generic_cache_fallback(self, **kwargs) -> SemanticFallbackResult:
        """Generic cache-based fallback."""
        return SemanticFallbackResult(success=False)  # Simplified for now
    
    def _generic_default_fallback(self, **kwargs) -> SemanticFallbackResult:
        """Generic default fallback."""
        return SemanticFallbackResult(
            success=True,
            result=None,  # Default/empty result
            performance_impact_ms=0.1
        )
    
    def _enter_degraded_mode(self) -> None:
        """Enter degraded mode operation."""
        if self.degraded_mode_active:
            return
        
        self.degraded_mode_active = True
        self.degraded_mode_start_time = datetime.now(timezone.utc)
        self.performance_metrics['degraded_mode_activations'] += 1
        
        self.logger.warning(
            f"Entering degraded mode for {self.service_name} due to service failures"
        )
    
    def _check_degraded_mode_recovery(self) -> None:
        """Check if we can exit degraded mode."""
        if not self.degraded_mode_active:
            return
        
        # Check if degraded mode timeout has passed
        if self.degraded_mode_start_time:
            elapsed = datetime.now(timezone.utc) - self.degraded_mode_start_time
            if elapsed.total_seconds() > self.config.degraded_mode_timeout_seconds:
                self._exit_degraded_mode()
    
    def _exit_degraded_mode(self) -> None:
        """Exit degraded mode operation."""
        self.degraded_mode_active = False
        self.degraded_mode_start_time = None
        
        self.logger.info(f"Exiting degraded mode for {self.service_name} - service recovered")
    
    def _record_execution_time(self, execution_time_ms: float) -> None:
        """Record execution time for performance monitoring."""
        # This would integrate with the performance monitoring system
        pass
    
    def get_circuit_status(self) -> Dict[str, Any]:
        """Get comprehensive circuit breaker status."""
        return {
            'service_type': self.service_type.value,
            'service_name': self.service_name,
            'circuit_state': self.circuit_breaker.state.value,
            'degraded_mode_active': self.degraded_mode_active,
            'degraded_mode_duration_seconds': (
                (datetime.now(timezone.utc) - self.degraded_mode_start_time).total_seconds()
                if self.degraded_mode_start_time else 0
            ),
            'performance_metrics': self.performance_metrics.copy(),
            'fallback_strategies_available': list(self.fallback_strategies.keys()),
            'config': {
                'failure_threshold': self.config.failure_threshold,
                'recovery_timeout_seconds': self.config.recovery_timeout_seconds,
                'cache_fallback_enabled': self.config.enable_cache_fallback,
                'degraded_mode_enabled': self.config.enable_degraded_mode
            }
        }
    
    def force_circuit_state(self, state: CircuitBreakerState) -> None:
        """Force circuit breaker to specific state (for testing)."""
        self.circuit_breaker.state = state
        self.logger.info(f"Forced circuit state to {state.value} for {self.service_name}")
    
    def reset_performance_metrics(self) -> None:
        """Reset performance metrics."""
        self.performance_metrics = {
            'circuit_breaks': 0,
            'fallback_successes': 0,
            'fallback_failures': 0,
            'degraded_mode_activations': 0,
            'cache_hits_during_failure': 0
        }


class SemanticCircuitBreakerManager:
    """
    Manager for multiple semantic circuit breakers.
    
    Provides centralized management and monitoring of all semantic service
    circuit breakers in the system.
    """
    
    def __init__(self):
        """Initialize circuit breaker manager."""
        self.logger = get_logger(__name__)
        self.circuit_breakers: Dict[str, SemanticCircuitBreaker] = {}
        self.global_degraded_mode = False
    
    def register_circuit_breaker(
        self,
        service_type: SemanticServiceType,
        service_name: str,
        config: Optional[SemanticCircuitBreakerConfig] = None,
        cache_manager: Optional[CacheManager] = None
    ) -> SemanticCircuitBreaker:
        """
        Register a new semantic circuit breaker.
        
        Args:
            service_type: Type of semantic service
            service_name: Name identifier for the service
            config: Circuit breaker configuration
            cache_manager: Cache manager for fallback operations
            
        Returns:
            Registered circuit breaker
        """
        breaker_id = f"{service_type.value}_{service_name}"
        
        if breaker_id in self.circuit_breakers:
            self.logger.warning(f"Circuit breaker {breaker_id} already registered")
            return self.circuit_breakers[breaker_id]
        
        circuit_breaker = SemanticCircuitBreaker(
            service_type=service_type,
            service_name=service_name,
            config=config,
            cache_manager=cache_manager
        )
        
        self.circuit_breakers[breaker_id] = circuit_breaker
        self.logger.info(f"Registered circuit breaker for {breaker_id}")
        
        return circuit_breaker
    
    def get_circuit_breaker(
        self,
        service_type: SemanticServiceType,
        service_name: str
    ) -> Optional[SemanticCircuitBreaker]:
        """Get circuit breaker by service type and name."""
        breaker_id = f"{service_type.value}_{service_name}"
        return self.circuit_breakers.get(breaker_id)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status for all circuit breakers."""
        status = {
            'total_circuit_breakers': len(self.circuit_breakers),
            'global_degraded_mode': self.global_degraded_mode,
            'circuit_breakers': {}
        }
        
        open_circuits = 0
        degraded_services = 0
        
        for breaker_id, breaker in self.circuit_breakers.items():
            breaker_status = breaker.get_circuit_status()
            status['circuit_breakers'][breaker_id] = breaker_status
            
            if breaker_status['circuit_state'] == 'OPEN':
                open_circuits += 1
            
            if breaker_status['degraded_mode_active']:
                degraded_services += 1
        
        status['summary'] = {
            'open_circuits': open_circuits,
            'degraded_services': degraded_services,
            'healthy_services': len(self.circuit_breakers) - open_circuits - degraded_services
        }
        
        # Check if we should enter global degraded mode
        if open_circuits > len(self.circuit_breakers) * 0.5:  # More than 50% circuits open
            if not self.global_degraded_mode:
                self._enter_global_degraded_mode()
        else:
            if self.global_degraded_mode:
                self._exit_global_degraded_mode()
        
        return status
    
    def _enter_global_degraded_mode(self) -> None:
        """Enter global degraded mode when too many services are failing."""
        self.global_degraded_mode = True
        self.logger.error(
            "Entering GLOBAL degraded mode - multiple semantic services are failing"
        )
    
    def _exit_global_degraded_mode(self) -> None:
        """Exit global degraded mode when services recover."""
        self.global_degraded_mode = False
        self.logger.info("Exiting global degraded mode - services have recovered")
    
    def execute_with_circuit_breaker(
        self,
        service_type: SemanticServiceType,
        service_name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with appropriate circuit breaker protection.
        
        Args:
            service_type: Type of semantic service
            service_name: Service name
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or fallback result
        """
        breaker = self.get_circuit_breaker(service_type, service_name)
        
        if breaker is None:
            # No circuit breaker registered - execute directly
            self.logger.warning(
                f"No circuit breaker registered for {service_type.value}:{service_name}"
            )
            return func(*args, **kwargs)
        
        return breaker.execute(func, *args, **kwargs)


# Global circuit breaker manager
_global_circuit_breaker_manager = None


def initialize_semantic_circuit_breakers() -> SemanticCircuitBreakerManager:
    """Initialize global semantic circuit breaker manager."""
    global _global_circuit_breaker_manager
    _global_circuit_breaker_manager = SemanticCircuitBreakerManager()
    return _global_circuit_breaker_manager


def get_circuit_breaker_manager() -> Optional[SemanticCircuitBreakerManager]:
    """Get global circuit breaker manager."""
    return _global_circuit_breaker_manager


def semantic_circuit_breaker(
    service_type: SemanticServiceType,
    service_name: str,
    config: Optional[SemanticCircuitBreakerConfig] = None
):
    """
    Decorator for applying circuit breaker protection to semantic functions.
    
    Args:
        service_type: Type of semantic service
        service_name: Service name
        config: Circuit breaker configuration
    
    Example:
        @semantic_circuit_breaker(SemanticServiceType.EMBEDDING_SERVICE, "primary_embeddings")
        def compute_embedding(text, language):
            # Your embedding computation logic
            return embedding_result
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_circuit_breaker_manager()
            if manager is None:
                # No manager available - execute directly
                return func(*args, **kwargs)
            
            # Get or register circuit breaker
            breaker = manager.get_circuit_breaker(service_type, service_name)
            if breaker is None:
                breaker = manager.register_circuit_breaker(
                    service_type=service_type,
                    service_name=service_name,
                    config=config
                )
            
            return breaker.execute(func, *args, **kwargs)
        
        return wrapper
    return decorator