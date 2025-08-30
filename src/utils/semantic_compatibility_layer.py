"""
Story 3.5: Semantic Processing Backward Compatibility Layer

Ensures existing API contracts and workflows continue to function
while introducing semantic processing capabilities progressively.

Author: Dev Agent James  
Date: 2025-08-30
Epic: 3 - Semantic Refinement & QA Framework
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

from src.utils.semantic_feature_manager import SemanticFeatureManager, SemanticFeature


@dataclass
class CompatibilityResult:
    """Result of compatibility layer processing."""
    output_data: Any
    compatibility_warnings: List[str] = field(default_factory=list)
    performance_impact_ms: float = 0.0
    semantic_features_applied: List[str] = field(default_factory=list)
    fallback_used: bool = False


class SemanticCompatibilityLayer:
    """
    Provides backward compatibility for existing API contracts while
    enabling progressive semantic feature rollout.
    
    Ensures zero-regression for existing 79.7% Academic Excellence performance
    while adding semantic enhancement capabilities.
    """
    
    def __init__(self, semantic_feature_manager: Optional[SemanticFeatureManager] = None):
        """Initialize compatibility layer."""
        self.logger = logging.getLogger(__name__)
        self.semantic_feature_manager = semantic_feature_manager
        self._performance_tracking = {}
        
    def ensure_output_format_compatibility(
        self, 
        semantic_output: Dict[str, Any], 
        expected_legacy_format: str = "srt_segment"
    ) -> CompatibilityResult:
        """
        Ensure semantic processing output matches expected legacy format.
        
        Args:
            semantic_output: Output from semantic processing
            expected_legacy_format: Expected format type
            
        Returns:
            CompatibilityResult with compatible output
        """
        start_time = time.time()
        
        try:
            compatibility_warnings = []
            
            # Extract core data needed for legacy compatibility
            processed_text = semantic_output.get('processed_text', '')
            semantic_metrics = semantic_output.get('metrics', {})
            semantic_analysis = semantic_output.get('semantic_analysis', {})
            
            # Maintain legacy SRT segment structure
            if expected_legacy_format == "srt_segment":
                # Preserve original segment structure, add semantic data as metadata
                compatible_output = {
                    # Legacy required fields
                    'text': processed_text,
                    'confidence': semantic_output.get('confidence', 0.8),
                    'processing_flags': semantic_output.get('processing_flags', []),
                    
                    # Backward-compatible metadata (optional for legacy systems)
                    '_semantic_metadata': {
                        'features_applied': semantic_analysis.get('feature_flags_applied', []),
                        'processing_time_ms': semantic_metrics.get('processing_time', 0),
                        'enhancement_count': semantic_metrics.get('semantic_enhancements_applied', 0)
                    } if semantic_analysis.get('feature_flags_applied') else None
                }
                
                # Add compatibility warnings if format changes detected
                if semantic_metrics.get('semantic_enhancements_applied', 0) > 0:
                    compatibility_warnings.append("Semantic enhancements applied - verify output compatibility")
                
            elif expected_legacy_format == "processing_metrics":
                # Legacy ProcessingMetrics compatibility
                compatible_output = {
                    # Legacy required fields
                    'segments_processed': semantic_output.get('segments_processed', 1),
                    'corrections_applied': semantic_output.get('corrections_applied', 0),
                    'flagged_segments': semantic_output.get('flagged_segments', 0),
                    'processing_time': semantic_metrics.get('processing_time', 0) / 1000,  # Convert to seconds
                    
                    # Enhanced fields (backward compatible)
                    'semantic_terms_processed': semantic_metrics.get('terms_analyzed', 0),
                    'semantic_relationships_found': semantic_metrics.get('relationships_found', 0),
                    'semantic_validations_performed': semantic_metrics.get('validations_performed', 0)
                }
                
            else:
                # Generic compatibility - preserve all original structure
                compatible_output = semantic_output
                compatibility_warnings.append(f"Unknown legacy format '{expected_legacy_format}' - using generic compatibility")
            
            performance_impact = (time.time() - start_time) * 1000
            
            return CompatibilityResult(
                output_data=compatible_output,
                compatibility_warnings=compatibility_warnings,
                performance_impact_ms=performance_impact,
                semantic_features_applied=semantic_analysis.get('feature_flags_applied', []),
                fallback_used=semantic_output.get('fallback_used', False)
            )
            
        except Exception as e:
            self.logger.error(f"Compatibility layer failed: {e}")
            
            # Ultimate fallback - return original input in safe format
            return CompatibilityResult(
                output_data={'text': semantic_output.get('processed_text', ''), 'error': str(e)},
                compatibility_warnings=[f"Compatibility processing failed: {e}"],
                performance_impact_ms=(time.time() - start_time) * 1000,
                fallback_used=True
            )
    
    def validate_performance_regression(
        self, 
        processing_time_ms: float, 
        baseline_time_ms: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Validate that semantic processing doesn't cause performance regression.
        
        Args:
            processing_time_ms: Current processing time
            baseline_time_ms: Baseline processing time (optional)
            
        Returns:
            Performance validation result
        """
        try:
            # Get performance threshold from feature manager
            max_regression_threshold = 0.05  # 5% default
            if self.semantic_feature_manager:
                compatibility_settings = self.semantic_feature_manager.get_backward_compatibility_settings()
                max_regression_threshold = compatibility_settings.get('performance_regression_threshold', 0.05)
            
            result = {
                'performance_acceptable': True,
                'regression_percentage': 0.0,
                'threshold_exceeded': False,
                'baseline_time_ms': baseline_time_ms,
                'current_time_ms': processing_time_ms
            }
            
            if baseline_time_ms and baseline_time_ms > 0:
                regression_ratio = (processing_time_ms - baseline_time_ms) / baseline_time_ms
                result['regression_percentage'] = regression_ratio
                
                if regression_ratio > max_regression_threshold:
                    result['performance_acceptable'] = False
                    result['threshold_exceeded'] = True
                    
                    self.logger.warning(
                        f"Performance regression detected: {regression_ratio:.1%} "
                        f"(threshold: {max_regression_threshold:.1%})"
                    )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Performance regression validation failed: {e}")
            return {
                'performance_acceptable': False,
                'error': str(e),
                'current_time_ms': processing_time_ms
            }
    
    def create_legacy_api_wrapper(self, new_semantic_method):
        """
        Create a wrapper for new semantic methods that maintains legacy API compatibility.
        
        Args:
            new_semantic_method: New method with semantic capabilities
            
        Returns:
            Wrapped method that maintains legacy API
        """
        def legacy_compatible_wrapper(*args, **kwargs):
            """Wrapper that ensures legacy API compatibility."""
            start_time = time.time()
            
            try:
                # Call new semantic method
                semantic_result = new_semantic_method(*args, **kwargs)
                
                # Apply compatibility layer
                compatibility_result = self.ensure_output_format_compatibility(
                    semantic_result, 
                    expected_legacy_format=kwargs.get('_legacy_format', 'srt_segment')
                )
                
                # Validate performance
                processing_time_ms = (time.time() - start_time) * 1000
                performance_result = self.validate_performance_regression(
                    processing_time_ms, 
                    kwargs.get('_baseline_time_ms')
                )
                
                # Log compatibility issues
                if compatibility_result.compatibility_warnings:
                    for warning in compatibility_result.compatibility_warnings:
                        self.logger.warning(f"Compatibility warning: {warning}")
                
                if not performance_result['performance_acceptable']:
                    self.logger.warning(f"Performance regression detected in semantic processing")
                
                return compatibility_result.output_data
                
            except Exception as e:
                self.logger.error(f"Legacy API wrapper failed: {e}")
                
                # Ultimate fallback for legacy compatibility
                if hasattr(args[0], '_apply_legacy_processing'):
                    self.logger.info("Falling back to legacy processing")
                    return args[0]._apply_legacy_processing(*args[1:], **kwargs)
                else:
                    raise
        
        return legacy_compatible_wrapper
    
    def enable_graceful_degradation(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enable graceful degradation when semantic processing fails.
        
        Args:
            error: Exception that occurred
            context: Processing context
            
        Returns:
            Fallback processing result
        """
        self.logger.warning(f"Enabling graceful degradation due to: {error}")
        
        # Return minimal compatible structure
        fallback_result = {
            'text': context.get('original_text', ''),
            'processing_flags': ['semantic_processing_degraded'],
            'confidence': 0.5,  # Lower confidence due to degradation
            '_degradation_info': {
                'error': str(error),
                'fallback_used': True,
                'timestamp': time.time()
            }
        }
        
        return fallback_result
    
    def track_compatibility_metrics(self, operation: str, success: bool, performance_ms: float) -> None:
        """Track compatibility layer performance metrics."""
        if operation not in self._performance_tracking:
            self._performance_tracking[operation] = {
                'success_count': 0,
                'failure_count': 0,
                'total_time_ms': 0,
                'avg_time_ms': 0
            }
        
        tracking = self._performance_tracking[operation]
        
        if success:
            tracking['success_count'] += 1
        else:
            tracking['failure_count'] += 1
        
        tracking['total_time_ms'] += performance_ms
        total_operations = tracking['success_count'] + tracking['failure_count']
        tracking['avg_time_ms'] = tracking['total_time_ms'] / total_operations
    
    def get_compatibility_report(self) -> Dict[str, Any]:
        """Get compatibility layer performance and usage report."""
        return {
            'performance_tracking': self._performance_tracking,
            'semantic_features_available': self.semantic_feature_manager is not None,
            'total_operations': sum(
                data['success_count'] + data['failure_count'] 
                for data in self._performance_tracking.values()
            ),
            'overall_success_rate': self._calculate_overall_success_rate()
        }
    
    def _calculate_overall_success_rate(self) -> float:
        """Calculate overall success rate across all operations."""
        total_success = sum(data['success_count'] for data in self._performance_tracking.values())
        total_operations = sum(
            data['success_count'] + data['failure_count'] 
            for data in self._performance_tracking.values()
        )
        
        return total_success / total_operations if total_operations > 0 else 1.0


# Factory function for easy integration
def create_compatibility_layer(semantic_feature_manager: Optional[SemanticFeatureManager] = None) -> SemanticCompatibilityLayer:
    """Create semantic compatibility layer."""
    return SemanticCompatibilityLayer(semantic_feature_manager)