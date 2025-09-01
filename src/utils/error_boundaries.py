#!/usr/bin/env python3
"""
Error Boundaries for Academic Standards System
Implements error boundary patterns to prevent single component failures from crashing the entire system.
"""

import logging
from functools import wraps
from typing import Any, Callable, Optional, Dict, Union
import traceback

def academic_error_boundary(
    default_score: float = 0.0, 
    component_name: str = "Unknown",
    return_type: str = "score"
):
    """
    Decorator that provides error boundary for academic processing components.
    Prevents single component failures from crashing the entire system.
    
    Args:
        default_score: Default score to return on error (0.0-1.0 range)
        component_name: Name of the component for logging
        return_type: Type of return value expected ("score", "result", "list")
    
    Returns:
        Decorator function that wraps the original function with error handling
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                result = func(*args, **kwargs)
                # Record successful execution
                error_boundary_manager.component_status[component_name] = {
                    'status': 'success',
                    'last_execution': 'success',
                    'error_count': error_boundary_manager.error_count.get(component_name, 0)
                }
                return result
            except Exception as e:
                # Record the error
                error_boundary_manager.record_error(component_name, e)
                
                # Log the error with full traceback for debugging
                logging.error(f"Error in {component_name}: {str(e)}")
                logging.error(f"Function: {func.__name__}")
                logging.error(f"Args: {args}")
                logging.error(f"Kwargs: {kwargs}")
                logging.error(f"Traceback: {traceback.format_exc()}")
                logging.error(f"Returning safe default for return_type: {return_type}")
                
                # Return appropriate default structure based on expected return type
                if return_type == "list" or 'test_' in func.__name__:
                    # For test methods that expect list results, return empty list
                    return []
                elif return_type == "result" or 'result' in func.__name__ or 'evaluate' in func.__name__:
                    return {
                        'score': default_score,
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'component': component_name,
                        'status': 'error',
                        'function': func.__name__,
                        'graceful_degradation': True
                    }
                elif return_type == "dict":
                    # Return basic dict structure
                    return {
                        'error': str(e),
                        'component': component_name,
                        'status': 'error'
                    }
                else:
                    # Default to returning just the score
                    return default_score
                    
        return wrapper
    return decorator

def create_safe_academic_result(
    score: float = 0.0,
    component_name: str = "Unknown",
    error: Optional[Exception] = None,
    additional_fields: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Create a standardized safe result structure for academic components.
    
    Args:
        score: The score to return (0.0-1.0)
        component_name: Name of the component
        error: The exception that occurred (if any)
        additional_fields: Any additional fields to include
    
    Returns:
        Standardized result dictionary
    """
    result = {
        'score': score,
        'component': component_name,
        'status': 'error' if error else 'success',
        'graceful_degradation': error is not None,
        'timestamp': None
    }
    
    if error:
        result.update({
            'error': str(error),
            'error_type': type(error).__name__,
        })
    
    if additional_fields:
        result.update(additional_fields)
    
    try:
        from datetime import datetime
        result['timestamp'] = datetime.now().isoformat()
    except ImportError:
        pass
    
    return result

class AcademicErrorBoundaryManager:
    """
    Manager class for handling academic error boundaries and monitoring.
    """
    
    def __init__(self):
        self.error_count = {}
        self.component_status = {}
    
    def record_error(self, component_name: str, error: Exception):
        """Record an error for monitoring purposes."""
        if component_name not in self.error_count:
            self.error_count[component_name] = 0
        self.error_count[component_name] += 1
        
        self.component_status[component_name] = {
            'status': 'error',
            'last_error': str(error),
            'error_count': self.error_count[component_name],
            'error_type': type(error).__name__
        }
        
        # Log warning if error count gets high
        if self.error_count[component_name] > 5:
            logging.warning(f"Component {component_name} has failed {self.error_count[component_name]} times")
    
    def get_component_health(self) -> Dict[str, Any]:
        """Get overall component health status."""
        total_components = len(self.component_status)
        healthy_components = sum(1 for status in self.component_status.values() 
                               if status.get('status') != 'error')
        
        return {
            'total_components': total_components,
            'healthy_components': healthy_components,
            'error_components': total_components - healthy_components,
            'overall_health_percentage': (healthy_components / total_components * 100) if total_components > 0 else 100,
            'component_details': self.component_status.copy()
        }
    
    def reset_component_status(self, component_name: str):
        """Reset error status for a component."""
        if component_name in self.error_count:
            del self.error_count[component_name]
        if component_name in self.component_status:
            del self.component_status[component_name]

# Global error boundary manager instance
error_boundary_manager = AcademicErrorBoundaryManager()