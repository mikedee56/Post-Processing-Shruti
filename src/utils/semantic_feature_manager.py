"""
Story 3.5: Semantic Feature Flag Management System

Provides gradual rollout capabilities, A/B testing, and backward compatibility
for semantic processing features in the post-processing pipeline.

Author: Dev Agent James
Date: 2025-08-30
Epic: 3 - Semantic Refinement & QA Framework
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from src.utils.config_manager import ConfigurationManager


class SemanticFeature(Enum):
    """Available semantic processing features."""
    SEMANTIC_ANALYSIS = "semantic_analysis"
    DOMAIN_CLASSIFICATION = "domain_classification" 
    ACADEMIC_QA_FRAMEWORK = "academic_qa_framework"
    EXPERT_REVIEW_QUEUE = "expert_review_queue"
    TERM_RELATIONSHIP_MAPPING = "term_relationship_mapping"
    CONTEXTUAL_VALIDATION = "contextual_validation"
    PERFORMANCE_MONITORING = "performance_monitoring"


class RolloutStrategy(Enum):
    """Rollout strategies for semantic features."""
    DISABLED = "disabled"           # Feature completely off
    PERCENTAGE = "percentage"       # Percentage-based rollout
    USER_BASED = "user_based"      # Based on user/session ID
    CONTENT_BASED = "content_based" # Based on content characteristics
    FULL_ENABLED = "full_enabled"   # Feature completely on


@dataclass
class FeatureFlagResult:
    """Result of feature flag evaluation."""
    feature: SemanticFeature
    enabled: bool
    strategy_used: RolloutStrategy
    metadata: Dict[str, Any] = field(default_factory=dict)
    evaluation_time_ms: float = 0.0


@dataclass 
class SemanticProcessingContext:
    """Context information for semantic processing decisions."""
    segment_id: Optional[str] = None
    user_session_id: Optional[str] = None
    content_domain: Optional[str] = None
    processing_complexity: float = 0.0
    previous_errors: List[str] = field(default_factory=list)
    performance_history: Dict[str, float] = field(default_factory=dict)


class SemanticFeatureManager:
    """
    Manages semantic feature flags with gradual rollout capabilities.
    
    Provides backward compatibility, A/B testing, and performance-aware
    feature enablement for Epic 3 semantic processing features.
    """
    
    def __init__(self, config_manager: ConfigurationManager):
        """Initialize semantic feature manager."""
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        self._feature_cache: Dict[str, FeatureFlagResult] = {}
        self._cache_timeout = 300  # 5 minutes
        self._performance_history: Dict[str, List[float]] = {}
        
        # Load configuration
        self._load_semantic_config()
        
    def _load_semantic_config(self) -> None:
        """Load semantic processing configuration."""
        try:
            self.config = self.config_manager.get_config()
            self.semantic_config = self.config.get('semantic_features', {})
            self.academic_config = self.config.get('academic_validation', {})
            
            # Initialize performance tracking
            for feature in SemanticFeature:
                if feature.value not in self._performance_history:
                    self._performance_history[feature.value] = []
                    
        except Exception as e:
            self.logger.error(f"Failed to load semantic configuration: {e}")
            # Use safe defaults
            self.semantic_config = {'enable_semantic_features': False}
            self.academic_config = {'enabled': False}
    
    def is_feature_enabled(
        self, 
        feature: SemanticFeature, 
        context: Optional[SemanticProcessingContext] = None
    ) -> FeatureFlagResult:
        """
        Check if a specific semantic feature is enabled.
        
        Args:
            feature: The semantic feature to check
            context: Processing context for decision making
            
        Returns:
            FeatureFlagResult with enablement status and metadata
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(feature, context)
            if cache_key in self._feature_cache:
                cached_result = self._feature_cache[cache_key]
                if time.time() - cached_result.metadata.get('cached_at', 0) < self._cache_timeout:
                    return cached_result
            
            # Master feature flag check
            if not self.semantic_config.get('enable_semantic_features', False):
                result = FeatureFlagResult(
                    feature=feature,
                    enabled=False,
                    strategy_used=RolloutStrategy.DISABLED,
                    metadata={'reason': 'master_flag_disabled'},
                    evaluation_time_ms=(time.time() - start_time) * 1000
                )
                return self._cache_result(cache_key, result)
            
            # Individual feature flag check
            feature_flags = self.semantic_config.get('feature_flags', {})
            if not feature_flags.get(feature.value, False):
                result = FeatureFlagResult(
                    feature=feature,
                    enabled=False,
                    strategy_used=RolloutStrategy.DISABLED,
                    metadata={'reason': 'individual_flag_disabled'},
                    evaluation_time_ms=(time.time() - start_time) * 1000
                )
                return self._cache_result(cache_key, result)
            
            # Performance-based checks
            if not self._check_performance_constraints(feature):
                result = FeatureFlagResult(
                    feature=feature,
                    enabled=False,
                    strategy_used=RolloutStrategy.DISABLED,
                    metadata={'reason': 'performance_constraints_failed'},
                    evaluation_time_ms=(time.time() - start_time) * 1000
                )
                return self._cache_result(cache_key, result)
            
            # Rollout percentage check
            rollout_enabled, strategy = self._check_rollout_percentage(feature, context)
            
            result = FeatureFlagResult(
                feature=feature,
                enabled=rollout_enabled,
                strategy_used=strategy,
                metadata={
                    'rollout_percentage': self.semantic_config.get('rollout_percentages', {}).get(feature.value, 0),
                    'context_used': context is not None
                },
                evaluation_time_ms=(time.time() - start_time) * 1000
            )
            
            return self._cache_result(cache_key, result)
            
        except Exception as e:
            self.logger.error(f"Error evaluating feature flag for {feature.value}: {e}")
            # Safe fallback - disable feature on error
            result = FeatureFlagResult(
                feature=feature,
                enabled=False,
                strategy_used=RolloutStrategy.DISABLED,
                metadata={'reason': 'evaluation_error', 'error': str(e)},
                evaluation_time_ms=(time.time() - start_time) * 1000
            )
            return result
    
    def _check_rollout_percentage(
        self, 
        feature: SemanticFeature, 
        context: Optional[SemanticProcessingContext]
    ) -> tuple[bool, RolloutStrategy]:
        """Check rollout percentage for feature enablement."""
        rollout_percentages = self.semantic_config.get('rollout_percentages', {})
        percentage = rollout_percentages.get(feature.value, 0)
        
        # Full rollout
        if percentage >= 100:
            return True, RolloutStrategy.FULL_ENABLED
        
        # No rollout
        if percentage <= 0:
            return False, RolloutStrategy.DISABLED
        
        # Percentage-based rollout using consistent hashing
        if context and context.segment_id:
            hash_input = f"{feature.value}:{context.segment_id}"
        elif context and context.user_session_id:
            hash_input = f"{feature.value}:{context.user_session_id}"
        else:
            # Fallback to random-ish but consistent behavior
            hash_input = f"{feature.value}:{int(time.time() / 3600)}"  # Changes hourly
        
        # Create deterministic hash
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)
        hash_percentage = (hash_value % 100) + 1  # 1-100
        
        enabled = hash_percentage <= percentage
        return enabled, RolloutStrategy.PERCENTAGE
    
    def _check_performance_constraints(self, feature: SemanticFeature) -> bool:
        """Check if feature meets performance constraints."""
        try:
            performance_limits = self.semantic_config.get('performance_limits', {})
            
            # Check recent performance history
            if feature.value in self._performance_history:
                recent_times = self._performance_history[feature.value][-10:]  # Last 10 measurements
                if recent_times:
                    avg_time = sum(recent_times) / len(recent_times)
                    max_time = performance_limits.get('max_semantic_processing_time_ms', 100)
                    
                    if avg_time > max_time:
                        self.logger.warning(
                            f"Feature {feature.value} disabled due to performance: "
                            f"avg {avg_time:.1f}ms > limit {max_time}ms"
                        )
                        return False
            
            # Additional performance checks can be added here
            # (memory usage, cache hit ratios, etc.)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Performance constraint check failed for {feature.value}: {e}")
            return False  # Conservative approach - disable on error
    
    def record_performance_metrics(
        self, 
        feature: SemanticFeature, 
        processing_time_ms: float,
        success: bool = True
    ) -> None:
        """Record performance metrics for a feature."""
        try:
            if feature.value not in self._performance_history:
                self._performance_history[feature.value] = []
            
            # Store processing time
            self._performance_history[feature.value].append(processing_time_ms)
            
            # Keep only recent measurements (last 100)
            if len(self._performance_history[feature.value]) > 100:
                self._performance_history[feature.value] = self._performance_history[feature.value][-100:]
                
        except Exception as e:
            self.logger.error(f"Failed to record performance metrics for {feature.value}: {e}")
    
    def get_enabled_features(
        self, 
        context: Optional[SemanticProcessingContext] = None
    ) -> Dict[SemanticFeature, FeatureFlagResult]:
        """Get all enabled semantic features for current context."""
        enabled_features = {}
        
        for feature in SemanticFeature:
            result = self.is_feature_enabled(feature, context)
            if result.enabled:
                enabled_features[feature] = result
                
        return enabled_features
    
    def should_use_legacy_fallback(self) -> bool:
        """Check if legacy fallback should be used instead of semantic processing."""
        compatibility_config = self.semantic_config.get('compatibility', {})
        return compatibility_config.get('legacy_fallback_enabled', True)
    
    def get_backward_compatibility_settings(self) -> Dict[str, Any]:
        """Get backward compatibility configuration."""
        return self.semantic_config.get('compatibility', {
            'preserve_legacy_api': True,
            'legacy_fallback_enabled': True,
            'maintain_output_format': True,
            'performance_regression_threshold': 0.05
        })
    
    def _generate_cache_key(
        self, 
        feature: SemanticFeature, 
        context: Optional[SemanticProcessingContext]
    ) -> str:
        """Generate cache key for feature flag result."""
        key_parts = [feature.value]
        
        if context:
            if context.segment_id:
                key_parts.append(f"seg:{context.segment_id}")
            if context.user_session_id:
                key_parts.append(f"user:{context.user_session_id}")
            if context.content_domain:
                key_parts.append(f"domain:{context.content_domain}")
        
        return ":".join(key_parts)
    
    def _cache_result(self, cache_key: str, result: FeatureFlagResult) -> FeatureFlagResult:
        """Cache feature flag result."""
        result.metadata['cached_at'] = time.time()
        self._feature_cache[cache_key] = result
        
        # Clean up old cache entries
        if len(self._feature_cache) > 1000:
            # Remove entries older than cache timeout
            current_time = time.time()
            expired_keys = [
                key for key, cached_result in self._feature_cache.items()
                if current_time - cached_result.metadata.get('cached_at', 0) > self._cache_timeout
            ]
            for key in expired_keys:
                del self._feature_cache[key]
        
        return result
    
    def invalidate_cache(self, feature: Optional[SemanticFeature] = None) -> None:
        """Invalidate feature flag cache."""
        if feature:
            # Remove cache entries for specific feature
            keys_to_remove = [key for key in self._feature_cache.keys() if key.startswith(feature.value)]
            for key in keys_to_remove:
                del self._feature_cache[key]
        else:
            # Clear all cache
            self._feature_cache.clear()
            
        self.logger.info(f"Feature flag cache invalidated for {feature.value if feature else 'all features'}")


# Factory function for easy integration
def create_semantic_feature_manager(config_manager: ConfigurationManager) -> SemanticFeatureManager:
    """Create and configure semantic feature manager."""
    return SemanticFeatureManager(config_manager)