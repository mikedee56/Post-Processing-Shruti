"""
Feature Flags and Fallback System for Story 2.4.4

This module provides enhancement enable/disable control with graceful fallback
to original functionality when advanced features fail.

Key Features:
- Feature flag-based enhancement control
- Graceful degradation when advanced features fail
- Fallback to existing Story 2.1-2.3 implementations
- Comprehensive error handling with fallback logging
"""

from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from pathlib import Path
import json

from utils.logger_config import get_logger


class FeatureFlag(Enum):
    """Available feature flags for enhancements."""
    SANDHI_PREPROCESSING = "sandhi_preprocessing"
    PHONETIC_HASHING = "phonetic_hashing" 
    SEMANTIC_SIMILARITY = "semantic_similarity"
    PHONETIC_CONTEXTUAL = "phonetic_contextual"
    UNIFIED_CONFIDENCE = "unified_confidence"
    PROVENANCE_WEIGHTING = "provenance_weighting"
    ENHANCED_FUZZY_MATCHING = "enhanced_fuzzy_matching"
    SEMANTIC_CONTEXTUAL = "semantic_contextual"


class FallbackStrategy(Enum):
    """Strategies for handling feature failures."""
    IMMEDIATE = "immediate"      # Fallback immediately on error
    RETRY_ONCE = "retry_once"    # Retry once, then fallback
    GRADUAL = "gradual"          # Try progressively simpler approaches
    DISABLED = "disabled"        # Don't fallback, raise error


@dataclass
class FeatureConfig:
    """Configuration for a single feature flag."""
    enabled: bool = True
    fallback_strategy: FallbackStrategy = FallbackStrategy.IMMEDIATE
    retry_count: int = 1
    timeout_seconds: float = 5.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FallbackEvent:
    """Record of a fallback event."""
    feature: FeatureFlag
    error_message: str
    fallback_used: bool
    timestamp: float
    original_function: str
    fallback_function: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeatureFlagManager:
    """
    Manager for feature flags and fallback mechanisms.
    
    This component implements AC8 of Story 2.4.4, providing:
    - Feature flag-based enhancement control
    - Graceful degradation when advanced features fail
    - Fallback to existing Story implementations
    - Comprehensive error handling with fallback logging
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize feature flag manager.
        
        Args:
            config: Configuration dictionary with feature settings
        """
        self.logger = get_logger(__name__)
        self.config = config or {}
        
        # Initialize default feature configurations
        self.feature_configs: Dict[FeatureFlag, FeatureConfig] = {}
        self._initialize_default_configurations()
        
        # Override with user config
        self._apply_user_configuration()
        
        # Fallback tracking
        self.fallback_events: List[FallbackEvent] = []
        self.fallback_functions: Dict[FeatureFlag, Callable] = {}
        
        # Statistics
        self.feature_usage_stats = {
            feature: {
                'total_calls': 0,
                'successful_calls': 0,
                'failed_calls': 0,
                'fallback_calls': 0,
                'average_execution_time': 0.0
            }
            for feature in FeatureFlag
        }
        
        self.logger.info("Feature flag manager initialized")
    
    def _initialize_default_configurations(self) -> None:
        """Initialize default configurations for all features."""
        # Conservative defaults - enable features but with immediate fallback
        default_configs = {
            FeatureFlag.SANDHI_PREPROCESSING: FeatureConfig(
                enabled=True,
                fallback_strategy=FallbackStrategy.RETRY_ONCE,
                retry_count=1,
                timeout_seconds=3.0,
                metadata={'story': '2.4.1', 'performance_critical': False}
            ),
            FeatureFlag.PHONETIC_HASHING: FeatureConfig(
                enabled=True,
                fallback_strategy=FallbackStrategy.IMMEDIATE,
                retry_count=0,
                timeout_seconds=2.0,
                metadata={'story': '2.4.1', 'performance_critical': True}
            ),
            FeatureFlag.SEMANTIC_SIMILARITY: FeatureConfig(
                enabled=True,
                fallback_strategy=FallbackStrategy.RETRY_ONCE,
                retry_count=1,
                timeout_seconds=10.0,
                metadata={'story': '2.4.2', 'performance_critical': False}
            ),
            FeatureFlag.PHONETIC_CONTEXTUAL: FeatureConfig(
                enabled=True,
                fallback_strategy=FallbackStrategy.IMMEDIATE,
                retry_count=0,
                timeout_seconds=5.0,
                metadata={'story': '2.4.2', 'performance_critical': False}
            ),
            FeatureFlag.UNIFIED_CONFIDENCE: FeatureConfig(
                enabled=True,
                fallback_strategy=FallbackStrategy.IMMEDIATE,
                retry_count=0,
                timeout_seconds=1.0,
                metadata={'story': '2.4.4', 'performance_critical': True}
            ),
            FeatureFlag.PROVENANCE_WEIGHTING: FeatureConfig(
                enabled=True,
                fallback_strategy=FallbackStrategy.IMMEDIATE,
                retry_count=0,
                timeout_seconds=1.0,
                metadata={'story': '2.4.4', 'performance_critical': False}
            ),
            FeatureFlag.ENHANCED_FUZZY_MATCHING: FeatureConfig(
                enabled=True,
                fallback_strategy=FallbackStrategy.GRADUAL,
                retry_count=2,
                timeout_seconds=5.0,
                metadata={'story': '2.4.4', 'performance_critical': True}
            ),
            FeatureFlag.SEMANTIC_CONTEXTUAL: FeatureConfig(
                enabled=True,
                fallback_strategy=FallbackStrategy.RETRY_ONCE,
                retry_count=1,
                timeout_seconds=8.0,
                metadata={'story': '2.4.4', 'performance_critical': False}
            )
        }
        
        self.feature_configs.update(default_configs)
    
    def _apply_user_configuration(self) -> None:
        """Apply user configuration overrides."""
        feature_settings = self.config.get('features', {})
        
        for feature_name, settings in feature_settings.items():
            try:
                feature = FeatureFlag(feature_name)
                current_config = self.feature_configs.get(feature, FeatureConfig())
                
                # Update configuration fields
                if 'enabled' in settings:
                    current_config.enabled = bool(settings['enabled'])
                if 'fallback_strategy' in settings:
                    current_config.fallback_strategy = FallbackStrategy(settings['fallback_strategy'])
                if 'retry_count' in settings:
                    current_config.retry_count = int(settings['retry_count'])
                if 'timeout_seconds' in settings:
                    current_config.timeout_seconds = float(settings['timeout_seconds'])
                if 'metadata' in settings:
                    current_config.metadata.update(settings['metadata'])
                
                self.feature_configs[feature] = current_config
                
                self.logger.info(
                    f"Updated feature config for {feature.value}: enabled={current_config.enabled}"
                )
                
            except (ValueError, KeyError) as e:
                self.logger.warning(f"Invalid feature configuration for {feature_name}: {e}")
    
    def is_feature_enabled(self, feature: FeatureFlag) -> bool:
        """
        Check if a feature is enabled.
        
        Args:
            feature: Feature flag to check
            
        Returns:
            True if feature is enabled
        """
        return self.feature_configs.get(feature, FeatureConfig()).enabled
    
    def register_fallback_function(self, feature: FeatureFlag, fallback_func: Callable) -> None:
        """
        Register a fallback function for a feature.
        
        Args:
            feature: Feature flag
            fallback_func: Function to call when feature fails
        """
        self.fallback_functions[feature] = fallback_func
        self.logger.debug(f"Registered fallback function for {feature.value}")
    
    def execute_with_fallback(
        self, 
        feature: FeatureFlag,
        enhanced_func: Callable,
        fallback_func: Optional[Callable] = None,
        *args, **kwargs
    ) -> Any:
        """
        Execute function with automatic fallback on failure.
        
        Args:
            feature: Feature flag for the enhanced function
            enhanced_func: Enhanced function to try first
            fallback_func: Fallback function (uses registered if None)
            *args, **kwargs: Arguments to pass to functions
            
        Returns:
            Result from enhanced function or fallback
        """
        start_time = time.time()
        config = self.feature_configs.get(feature, FeatureConfig())
        stats = self.feature_usage_stats[feature]
        stats['total_calls'] += 1
        
        # Check if feature is disabled
        if not config.enabled:
            return self._execute_fallback(
                feature, fallback_func, "Feature disabled", *args, **kwargs
            )
        
        # Try enhanced function with retry logic
        for attempt in range(max(1, config.retry_count + 1)):
            try:
                # Execute with timeout if specified
                if config.timeout_seconds > 0:
                    result = self._execute_with_timeout(
                        enhanced_func, config.timeout_seconds, *args, **kwargs
                    )
                else:
                    result = enhanced_func(*args, **kwargs)
                
                # Success - update stats and return
                stats['successful_calls'] += 1
                execution_time = time.time() - start_time
                stats['average_execution_time'] = (
                    (stats['average_execution_time'] * (stats['total_calls'] - 1) + execution_time) /
                    stats['total_calls']
                )
                
                return result
                
            except Exception as e:
                self.logger.debug(
                    f"Attempt {attempt + 1}/{config.retry_count + 1} failed for {feature.value}: {e}"
                )
                
                # If this is the last attempt or strategy is immediate fallback
                if (attempt >= config.retry_count or 
                    config.fallback_strategy == FallbackStrategy.IMMEDIATE):
                    
                    stats['failed_calls'] += 1
                    
                    if config.fallback_strategy == FallbackStrategy.DISABLED:
                        # Re-raise the error
                        raise
                    
                    return self._execute_fallback(
                        feature, fallback_func, str(e), *args, **kwargs
                    )
        
        # Should not reach here, but fallback as safety
        return self._execute_fallback(
            feature, fallback_func, "Max retries exceeded", *args, **kwargs
        )
    
    def _execute_with_timeout(
        self, 
        func: Callable, 
        timeout_seconds: float, 
        *args, **kwargs
    ) -> Any:
        """Execute function with timeout (simplified implementation)."""
        # Note: This is a simplified timeout implementation
        # In production, you might want to use threading.Timer or asyncio
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Function execution exceeded {timeout_seconds}s")
        
        try:
            # Set timeout signal (Unix-like systems only)
            if hasattr(signal, 'SIGALRM'):
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(timeout_seconds))
            
            result = func(*args, **kwargs)
            
            # Clear timeout
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            
            return result
            
        except Exception as e:
            # Clear timeout on exception
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            raise
    
    def _execute_fallback(
        self, 
        feature: FeatureFlag,
        fallback_func: Optional[Callable],
        error_message: str,
        *args, **kwargs
    ) -> Any:
        """Execute fallback function and record the event."""
        stats = self.feature_usage_stats[feature]
        
        # Get fallback function
        if fallback_func is None:
            fallback_func = self.fallback_functions.get(feature)
        
        if fallback_func is None:
            self.logger.error(f"No fallback function registered for {feature.value}")
            raise RuntimeError(f"Feature {feature.value} failed and no fallback available")
        
        # Execute fallback
        try:
            result = fallback_func(*args, **kwargs)
            stats['fallback_calls'] += 1
            
            # Record fallback event
            event = FallbackEvent(
                feature=feature,
                error_message=error_message,
                fallback_used=True,
                timestamp=time.time(),
                original_function=getattr(enhanced_func, '__name__', 'unknown') if 'enhanced_func' in locals() else 'unknown',
                fallback_function=getattr(fallback_func, '__name__', 'unknown'),
                metadata={
                    'args_count': len(args),
                    'kwargs_count': len(kwargs)
                }
            )
            self.fallback_events.append(event)
            
            self.logger.warning(
                f"Feature {feature.value} failed, used fallback: {error_message}"
            )
            
            return result
            
        except Exception as fallback_error:
            self.logger.error(
                f"Fallback function also failed for {feature.value}: {fallback_error}"
            )
            raise
    
    def get_feature_status(self) -> Dict[str, Any]:
        """Get status of all features."""
        status = {}
        
        for feature, config in self.feature_configs.items():
            stats = self.feature_usage_stats[feature]
            
            status[feature.value] = {
                'enabled': config.enabled,
                'fallback_strategy': config.fallback_strategy.value,
                'total_calls': stats['total_calls'],
                'success_rate': f"{(stats['successful_calls'] / max(stats['total_calls'], 1)) * 100:.1f}%",
                'fallback_rate': f"{(stats['fallback_calls'] / max(stats['total_calls'], 1)) * 100:.1f}%",
                'average_execution_time': f"{stats['average_execution_time']:.4f}s",
                'has_fallback_function': feature in self.fallback_functions
            }
        
        return status
    
    def get_fallback_events(self, feature: Optional[FeatureFlag] = None) -> List[FallbackEvent]:
        """
        Get fallback events, optionally filtered by feature.
        
        Args:
            feature: Optional feature to filter by
            
        Returns:
            List of fallback events
        """
        if feature is None:
            return self.fallback_events.copy()
        else:
            return [event for event in self.fallback_events if event.feature == feature]
    
    def enable_feature(self, feature: FeatureFlag, enabled: bool = True) -> None:
        """
        Enable or disable a feature at runtime.
        
        Args:
            feature: Feature to modify
            enabled: Whether to enable or disable
        """
        if feature in self.feature_configs:
            self.feature_configs[feature].enabled = enabled
        else:
            self.feature_configs[feature] = FeatureConfig(enabled=enabled)
        
        self.logger.info(f"Feature {feature.value} {'enabled' if enabled else 'disabled'}")
    
    def disable_feature(self, feature: FeatureFlag) -> None:
        """
        Disable a feature at runtime.
        
        Args:
            feature: Feature to disable
        """
        self.enable_feature(feature, False)
    
    def export_configuration(self, file_path: Path) -> None:
        """
        Export current feature configuration to file.
        
        Args:
            file_path: Path to export configuration
        """
        try:
            config_data = {
                'features': {
                    feature.value: {
                        'enabled': config.enabled,
                        'fallback_strategy': config.fallback_strategy.value,
                        'retry_count': config.retry_count,
                        'timeout_seconds': config.timeout_seconds,
                        'metadata': config.metadata
                    }
                    for feature, config in self.feature_configs.items()
                },
                'export_timestamp': time.time(),
                'statistics': self.get_feature_status()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Exported feature configuration to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting configuration: {e}")
            raise
    
    def clear_statistics(self) -> None:
        """Clear all usage statistics and fallback events."""
        self.fallback_events.clear()
        
        for feature in FeatureFlag:
            self.feature_usage_stats[feature] = {
                'total_calls': 0,
                'successful_calls': 0,
                'failed_calls': 0,
                'fallback_calls': 0,
                'average_execution_time': 0.0
            }
        
        self.logger.info("Cleared all feature statistics")
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate feature flag configuration."""
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        for feature, config in self.feature_configs.items():
            # Check retry count
            if config.retry_count < 0:
                validation['errors'].append(f"{feature.value}: negative retry_count")
                validation['is_valid'] = False
            
            # Check timeout
            if config.timeout_seconds < 0:
                validation['errors'].append(f"{feature.value}: negative timeout_seconds")
                validation['is_valid'] = False
            elif config.timeout_seconds > 60:
                validation['warnings'].append(f"{feature.value}: very long timeout ({config.timeout_seconds}s)")
            
            # Check fallback function availability
            if config.enabled and feature not in self.fallback_functions:
                validation['warnings'].append(f"{feature.value}: no fallback function registered")
        
        return validation