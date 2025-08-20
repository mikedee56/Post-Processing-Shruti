"""
Standardized exception hierarchy for ASR Post-Processing system.

This module provides a comprehensive exception hierarchy with proper inheritance,
error categorization, and debugging support for all components.

Author: Dev Agent (Story 5.3)
Version: 1.0
"""

from typing import Optional, Dict, Any, List
from enum import Enum
import traceback
import time
import uuid


class ErrorSeverity(Enum):
    """Error severity levels for proper handling and escalation."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for proper classification and handling."""
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    DATA_PROCESSING = "data_processing"
    FILE_SYSTEM = "file_system"
    NETWORK = "network"
    MEMORY = "memory"
    PERFORMANCE = "performance"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"


class BaseProcessingException(Exception):
    """
    Base exception for all ASR post-processing errors.
    
    Provides structured error handling with categorization, severity levels,
    debugging information, and recovery suggestions.
    """
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.DATA_PROCESSING,
        component: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        recovery_suggestions: Optional[List[str]] = None,
        user_message: Optional[str] = None
    ):
        """
        Initialize base processing exception.
        
        Args:
            message: Technical error message for developers
            severity: Error severity level
            category: Error category for proper handling
            component: Component where error occurred
            details: Additional error details and context
            recovery_suggestions: Suggestions for error recovery
            user_message: User-friendly error message
        """
        super().__init__(message)
        
        self.error_id = str(uuid.uuid4())
        self.timestamp = time.time()
        self.message = message
        self.severity = severity
        self.category = category
        self.component = component or "unknown"
        self.details = details or {}
        self.recovery_suggestions = recovery_suggestions or []
        self.user_message = user_message or message
        
        # Capture stack trace for debugging
        self.stack_trace = traceback.format_exc()
        
        # Add debugging context
        self.details.update({
            'error_id': self.error_id,
            'timestamp': self.timestamp,
            'component': self.component,
            'severity': self.severity.value,
            'category': self.category.value
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging and reporting."""
        return {
            'error_id': self.error_id,
            'timestamp': self.timestamp,
            'message': self.message,
            'user_message': self.user_message,
            'severity': self.severity.value,
            'category': self.category.value,
            'component': self.component,
            'details': self.details,
            'recovery_suggestions': self.recovery_suggestions,
            'stack_trace': self.stack_trace
        }
    
    def get_recovery_guidance(self) -> str:
        """Get formatted recovery guidance for this error."""
        if not self.recovery_suggestions:
            return "No specific recovery guidance available."
        
        guidance = "Recovery suggestions:\n"
        for i, suggestion in enumerate(self.recovery_suggestions, 1):
            guidance += f"  {i}. {suggestion}\n"
        
        return guidance.strip()


# Configuration and Environment Exceptions
class ConfigurationError(BaseProcessingException):
    """Errors related to configuration loading, validation, or format issues."""
    
    def __init__(self, message: str, config_file: Optional[str] = None, **kwargs):
        # Extract parameters from kwargs to avoid conflicts
        severity = kwargs.pop('severity', ErrorSeverity.HIGH)
        component = kwargs.pop('component', 'configuration_manager')
        
        super().__init__(
            message,
            severity=severity,
            category=ErrorCategory.CONFIGURATION,
            component=component,
            **kwargs
        )
        self.config_file = config_file
        if config_file:
            self.details['config_file'] = config_file


class EnvironmentError(BaseProcessingException):
    """Errors related to environment setup, paths, or system configuration."""
    
    def __init__(self, message: str, environment_var: Optional[str] = None, **kwargs):
        # Extract parameters from kwargs to avoid conflicts
        severity = kwargs.pop('severity', ErrorSeverity.HIGH)
        component = kwargs.pop('component', 'environment')
        
        super().__init__(
            message,
            severity=severity,
            category=ErrorCategory.CONFIGURATION,
            component=component,
            **kwargs
        )
        self.environment_var = environment_var
        if environment_var:
            self.details['environment_var'] = environment_var


# Dependency and Library Exceptions
class DependencyError(BaseProcessingException):
    """Errors related to missing, incompatible, or failed dependencies."""
    
    def __init__(self, message: str, dependency_name: Optional[str] = None, 
                 required_version: Optional[str] = None, **kwargs):
        # Extract severity from kwargs to avoid duplicate parameter
        severity = kwargs.pop('severity', ErrorSeverity.CRITICAL)
        component = kwargs.pop('component', 'dependency_manager')
        
        super().__init__(
            message,
            severity=severity,
            category=ErrorCategory.DEPENDENCY,
            component=component,
            **kwargs
        )
        self.dependency_name = dependency_name
        self.required_version = required_version
        
        if dependency_name:
            self.details['dependency_name'] = dependency_name
        if required_version:
            self.details['required_version'] = required_version

# Enhanced error handling utilities
class ErrorHandler:
    """Standardized error handling utility using the exception hierarchy."""
    
    def __init__(self, logger, component_name: str):
        """
        Initialize error handler for a specific component.
        
        Args:
            logger: Logger instance for the component
            component_name: Name of the component for error tracking
        """
        self.logger = logger
        self.component_name = component_name
        self.correlation_id = None
    
    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for structured logging."""
        self.correlation_id = correlation_id
    
    def handle_processing_error(self, operation: str, error: Exception, 
                              context: Optional[Dict[str, Any]] = None) -> 'ProcessingError':
        """
        Handle processing errors with standardized logging and exception creation.
        
        Args:
            operation: Name of the operation that failed
            error: Original exception that occurred
            context: Additional context for debugging
            
        Returns:
            ProcessingError: Standardized processing error
        """
        error_msg = f"Error in {operation}: {str(error)}"
        
        # Create standardized exception
        processing_error = ProcessingError(
            message=error_msg,
            operation=operation,
            original_error=error,
            context=context,
            severity=ErrorSeverity.MEDIUM,
            component=self.component_name
        )
        
        # Structured logging with correlation ID
        log_context = {
            'operation': operation,
            'component': self.component_name,
            'error_type': type(error).__name__,
            'original_error': str(error)
        }
        
        if self.correlation_id:
            log_context['correlation_id'] = self.correlation_id
            
        if context:
            log_context.update(context)
            
        self.logger.error(error_msg, extra=log_context)
        
        return processing_error
    
    def handle_validation_error(self, field: str, value: Any, 
                              expected: str, context: Optional[Dict[str, Any]] = None) -> 'ValidationError':
        """
        Handle validation errors with standardized logging.
        
        Args:
            field: Field that failed validation
            value: Invalid value
            expected: Description of expected value
            context: Additional context for debugging
            
        Returns:
            ValidationError: Standardized validation error
        """
        error_msg = f"Validation failed for {field}: got {value}, expected {expected}"
        
        validation_error = ValidationError(
            message=error_msg,
            field=field,
            value=value,
            expected=expected,
            context=context,
            severity=ErrorSeverity.MEDIUM,
            component=self.component_name
        )
        
        # Structured logging
        log_context = {
            'validation_field': field,
            'invalid_value': str(value)[:200],  # Truncate long values
            'expected': expected,
            'component': self.component_name
        }
        
        if self.correlation_id:
            log_context['correlation_id'] = self.correlation_id
            
        if context:
            log_context.update(context)
            
        self.logger.warning(error_msg, extra=log_context)
        
        return validation_error
    
    def handle_dependency_error(self, dependency_name: str, error: Exception,
                              required_version: Optional[str] = None,
                              context: Optional[Dict[str, Any]] = None) -> 'DependencyError':
        """
        Handle dependency errors with standardized logging.
        
        Args:
            dependency_name: Name of the missing/failed dependency
            error: Original exception that occurred
            required_version: Required version if applicable
            context: Additional context for debugging
            
        Returns:
            DependencyError: Standardized dependency error
        """
        error_msg = f"Dependency error with {dependency_name}: {str(error)}"
        if required_version:
            error_msg += f" (required version: {required_version})"
        
        dependency_error = DependencyError(
            message=error_msg,
            dependency_name=dependency_name,
            required_version=required_version,
            original_error=error,
            context=context,
            severity=ErrorSeverity.CRITICAL,
            component=self.component_name
        )
        
        # Structured logging
        log_context = {
            'dependency_name': dependency_name,
            'error_type': type(error).__name__,
            'original_error': str(error),
            'component': self.component_name
        }
        
        if required_version:
            log_context['required_version'] = required_version
            
        if self.correlation_id:
            log_context['correlation_id'] = self.correlation_id
            
        if context:
            log_context.update(context)
            
        self.logger.error(error_msg, extra=log_context)
        
        return dependency_error
    
    def log_operation_start(self, operation: str, context: Optional[Dict[str, Any]] = None):
        """Log operation start with correlation ID."""
        log_context = {
            'operation': operation,
            'component': self.component_name,
            'phase': 'start'
        }
        
        if self.correlation_id:
            log_context['correlation_id'] = self.correlation_id
            
        if context:
            log_context.update(context)
            
        self.logger.info(f"Starting {operation}", extra=log_context)
    
    def log_operation_success(self, operation: str, metrics: Optional[Dict[str, Any]] = None):
        """Log successful operation completion with correlation ID."""
        log_context = {
            'operation': operation,
            'component': self.component_name,
            'phase': 'success'
        }
        
        if self.correlation_id:
            log_context['correlation_id'] = self.correlation_id
            
        if metrics:
            log_context.update(metrics)
            
        self.logger.info(f"Successfully completed {operation}", extra=log_context)
    
    def log_operation_warning(self, operation: str, warning_msg: str, 
                            context: Optional[Dict[str, Any]] = None):
        """Log operation warning with correlation ID."""
        log_context = {
            'operation': operation,
            'component': self.component_name,
            'warning': warning_msg
        }
        
        if self.correlation_id:
            log_context['correlation_id'] = self.correlation_id
            
        if context:
            log_context.update(context)
            
        self.logger.warning(f"Warning in {operation}: {warning_msg}", extra=log_context)


# Utility function to create error handlers
def create_error_handler(logger, component_name: str) -> ErrorHandler:
    """
    Create a standardized error handler for a component.
    
    Args:
        logger: Logger instance
        component_name: Name of the component
        
    Returns:
        ErrorHandler: Configured error handler
    """
    return ErrorHandler(logger, component_name)


# Decorator for standardized error handling
def handle_errors(component_name: str, operation: str = None):
    """
    Decorator for standardized error handling in methods.
    
    Args:
        component_name: Name of the component
        operation: Name of the operation (defaults to function name)
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            # Get operation name
            op_name = operation or func.__name__
            
            # Create error handler if not exists
            if not hasattr(self, '_error_handler'):
                self._error_handler = create_error_handler(self.logger, component_name)
            
            # Log operation start
            self._error_handler.log_operation_start(op_name)
            
            try:
                result = func(self, *args, **kwargs)
                self._error_handler.log_operation_success(op_name)
                return result
            except ProcessingError:
                # Re-raise our standardized errors
                raise
            except ValidationError:
                # Re-raise our standardized errors
                raise
            except DependencyError:
                # Re-raise our standardized errors
                raise
            except Exception as e:
                # Convert to standardized error
                standardized_error = self._error_handler.handle_processing_error(
                    op_name, e, {'args': args, 'kwargs': kwargs}
                )
                raise standardized_error
        
        return wrapper
    return decorator


class LibraryCompatibilityError(BaseProcessingException):
    """Errors related to library version compatibility and integration issues."""
    
    def __init__(self, message: str, library_name: str, current_version: Optional[str] = None, **kwargs):
        # Extract parameters from kwargs to avoid conflicts
        severity = kwargs.pop('severity', ErrorSeverity.HIGH)
        component = kwargs.pop('component', 'library_manager')
        
        super().__init__(
            message,
            severity=severity,
            category=ErrorCategory.DEPENDENCY,
            component=component,
            **kwargs
        )
        self.library_name = library_name
        self.current_version = current_version
        
        self.details.update({
            'library_name': library_name,
            'current_version': current_version
        })


# Data Processing Exceptions
class DataProcessingError(BaseProcessingException):
    """General data processing errors."""
    
    def __init__(self, message: str, data_type: Optional[str] = None, 
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, **kwargs):
        super().__init__(
            message,
            severity=severity,
            category=ErrorCategory.DATA_PROCESSING,
            **kwargs
        )
        self.data_type = data_type
        if data_type:
            self.details['data_type'] = data_type


class ProcessingError(DataProcessingError):
    """General processing error for ErrorHandler compatibility."""
    
    def __init__(self, message: str, operation: Optional[str] = None, 
                 original_error: Optional[Exception] = None, component: str = 'processor', 
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, **kwargs):
        super().__init__(
            message,
            component=component,
            severity=severity,
            **kwargs
        )
        self.operation = operation
        self.original_error = original_error
        
        if operation:
            self.details['operation'] = operation
        if original_error:
            self.details['original_error'] = str(original_error)
            self.details['original_error_type'] = type(original_error).__name__


class SRTProcessingError(DataProcessingError):
    """Errors specific to SRT file processing."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, 
                 segment_index: Optional[int] = None, **kwargs):
        super().__init__(
            message,
            data_type="SRT",
            component=kwargs.get('component', 'srt_processor'),
            **kwargs
        )
        self.file_path = file_path
        self.segment_index = segment_index
        
        if file_path:
            self.details['file_path'] = file_path
        if segment_index is not None:
            self.details['segment_index'] = segment_index


class TextNormalizationError(DataProcessingError):
    """Errors in text normalization and transformation."""
    
    def __init__(self, message: str, original_text: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            data_type="text",
            component=kwargs.get('component', 'text_normalizer'),
            **kwargs
        )
        self.original_text = original_text
        if original_text:
            self.details['original_text'] = original_text


class SanskritProcessingError(DataProcessingError):
    """Errors specific to Sanskrit/Hindi text processing."""
    
    def __init__(self, message: str, sanskrit_text: Optional[str] = None, 
                 processing_stage: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            data_type="sanskrit",
            component=kwargs.get('component', 'sanskrit_processor'),
            **kwargs
        )
        self.sanskrit_text = sanskrit_text
        self.processing_stage = processing_stage
        
        if sanskrit_text:
            self.details['sanskrit_text'] = sanskrit_text
        if processing_stage:
            self.details['processing_stage'] = processing_stage


# File System and I/O Exceptions
class FileProcessingError(BaseProcessingException):
    """Errors related to file operations and I/O."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, 
                 operation: Optional[str] = None, **kwargs):
        # Extract parameters from kwargs to avoid conflicts
        severity = kwargs.pop('severity', ErrorSeverity.MEDIUM)
        component = kwargs.pop('component', 'file_processor')
        
        super().__init__(
            message,
            severity=severity,
            category=ErrorCategory.FILE_SYSTEM,
            component=component,
            **kwargs
        )
        self.file_path = file_path
        self.operation = operation
        
        if file_path:
            self.details['file_path'] = file_path
        if operation:
            self.details['operation'] = operation


class FileFormatError(FileProcessingError):
    """Errors related to unsupported or invalid file formats."""
    
    def __init__(self, message: str, expected_format: Optional[str] = None, 
                 actual_format: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            operation="format_validation",
            **kwargs
        )
        self.expected_format = expected_format
        self.actual_format = actual_format
        
        if expected_format:
            self.details['expected_format'] = expected_format
        if actual_format:
            self.details['actual_format'] = actual_format


# Validation Exceptions  
class ValidationError(BaseProcessingException):
    """Errors in data validation and quality checks."""
    
    def __init__(self, message: str, validation_type: Optional[str] = None,
                 failed_value: Optional[Any] = None, **kwargs):
        # Extract parameters from kwargs to avoid conflicts
        severity = kwargs.pop('severity', ErrorSeverity.MEDIUM)
        component = kwargs.pop('component', 'validator')
        
        super().__init__(
            message,
            severity=severity,
            category=ErrorCategory.VALIDATION,
            component=component,
            **kwargs
        )
        self.validation_type = validation_type
        self.failed_value = failed_value
        
        if validation_type:
            self.details['validation_type'] = validation_type
        if failed_value is not None:
            self.details['failed_value'] = str(failed_value)


class QualityAssuranceError(ValidationError):
    """Errors in quality assurance and testing."""
    
    def __init__(self, message: str, quality_metric: Optional[str] = None, 
                 threshold: Optional[float] = None, actual_value: Optional[float] = None, **kwargs):
        super().__init__(
            message,
            validation_type="quality_assurance",
            component=kwargs.get('component', 'qa_validator'),
            **kwargs
        )
        self.quality_metric = quality_metric
        self.threshold = threshold
        self.actual_value = actual_value
        
        if quality_metric:
            self.details['quality_metric'] = quality_metric
        if threshold is not None:
            self.details['threshold'] = threshold
        if actual_value is not None:
            self.details['actual_value'] = actual_value


# Performance and Resource Exceptions
class PerformanceError(BaseProcessingException):
    """Errors related to performance issues and resource constraints."""
    
    def __init__(self, message: str, metric_name: Optional[str] = None,
                 expected_value: Optional[float] = None, actual_value: Optional[float] = None, **kwargs):
        # Extract parameters from kwargs to avoid conflicts
        severity = kwargs.pop('severity', ErrorSeverity.MEDIUM)
        component = kwargs.pop('component', 'performance_monitor')
        
        super().__init__(
            message,
            severity=severity,
            category=ErrorCategory.PERFORMANCE,
            component=component,
            **kwargs
        )
        self.metric_name = metric_name
        self.expected_value = expected_value
        self.actual_value = actual_value
        
        if metric_name:
            self.details['metric_name'] = metric_name
        if expected_value is not None:
            self.details['expected_value'] = expected_value
        if actual_value is not None:
            self.details['actual_value'] = actual_value


class MemoryError(BaseProcessingException):
    """Errors related to memory allocation and management."""
    
    def __init__(self, message: str, memory_usage: Optional[int] = None, 
                 memory_limit: Optional[int] = None, **kwargs):
        # Extract parameters from kwargs to avoid conflicts
        severity = kwargs.pop('severity', ErrorSeverity.HIGH)
        component = kwargs.pop('component', 'memory_manager')
        
        super().__init__(
            message,
            severity=severity,
            category=ErrorCategory.MEMORY,
            component=component,
            **kwargs
        )
        self.memory_usage = memory_usage
        self.memory_limit = memory_limit
        
        if memory_usage is not None:
            self.details['memory_usage'] = memory_usage
        if memory_limit is not None:
            self.details['memory_limit'] = memory_limit


# Network and Integration Exceptions
class NetworkError(BaseProcessingException):
    """Errors related to network operations and external integrations."""
    
    def __init__(self, message: str, endpoint: Optional[str] = None, 
                 status_code: Optional[int] = None, **kwargs):
        # Extract parameters from kwargs to avoid conflicts
        severity = kwargs.pop('severity', ErrorSeverity.MEDIUM)
        component = kwargs.pop('component', 'network_client')
        
        super().__init__(
            message,
            severity=severity,
            category=ErrorCategory.NETWORK,
            component=component,
            **kwargs
        )
        self.endpoint = endpoint
        self.status_code = status_code
        
        if endpoint:
            self.details['endpoint'] = endpoint
        if status_code is not None:
            self.details['status_code'] = status_code


class MCPIntegrationError(NetworkError):
    """Errors specific to MCP (Model Context Protocol) integration."""
    
    def __init__(self, message: str, mcp_operation: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            component=kwargs.get('component', 'mcp_client'),
            **kwargs
        )
        self.mcp_operation = mcp_operation
        if mcp_operation:
            self.details['mcp_operation'] = mcp_operation


# Utility Functions for Exception Handling
def handle_exception_with_recovery(
    exception: BaseProcessingException,
    recovery_attempts: int = 3,
    recovery_delay: float = 1.0
) -> Dict[str, Any]:
    """
    Handle exception with automatic recovery attempts.
    
    Args:
        exception: The exception to handle
        recovery_attempts: Number of recovery attempts
        recovery_delay: Delay between recovery attempts
    
    Returns:
        Recovery attempt results
    """
    recovery_log = []
    
    for attempt in range(recovery_attempts):
        recovery_log.append(f"Recovery attempt {attempt + 1}")
        
        # In a real implementation, this would attempt specific recovery strategies
        # based on the exception type and category
        
        if attempt < recovery_attempts - 1:
            time.sleep(recovery_delay)
    
    return {
        'exception_id': exception.error_id,
        'recovery_attempts': recovery_attempts,
        'recovery_log': recovery_log,
        'final_status': 'recovery_failed'  # Would be updated based on actual recovery
    }


def log_exception(exception: BaseProcessingException, logger) -> None:
    """Log exception with proper formatting and context."""
    exception_dict = exception.to_dict()
    
    log_message = (
        f"[{exception.category.value.upper()}] {exception.message} "
        f"(ID: {exception.error_id}, Component: {exception.component})"
    )
    
    if exception.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
        logger.error(log_message, extra={'exception_details': exception_dict})
    elif exception.severity == ErrorSeverity.MEDIUM:
        logger.warning(log_message, extra={'exception_details': exception_dict})
    else:
        logger.info(log_message, extra={'exception_details': exception_dict})


def create_exception_report(exceptions: List[BaseProcessingException]) -> Dict[str, Any]:
    """Create comprehensive exception report for analysis."""
    if not exceptions:
        return {'message': 'No exceptions to report'}
    
    # Categorize exceptions
    by_category = {}
    by_severity = {}
    by_component = {}
    
    for exc in exceptions:
        # By category
        category = exc.category.value
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(exc.to_dict())
        
        # By severity
        severity = exc.severity.value
        if severity not in by_severity:
            by_severity[severity] = 0
        by_severity[severity] += 1
        
        # By component
        component = exc.component
        if component not in by_component:
            by_component[component] = 0
        by_component[component] += 1
    
    return {
        'summary': {
            'total_exceptions': len(exceptions),
            'by_severity': by_severity,
            'by_component': by_component,
            'most_common_category': max(by_category.keys(), key=lambda k: len(by_category[k]))
        },
        'by_category': by_category,
        'recommendations': [
            "Review exceptions by category to identify patterns",
            "Address CRITICAL and HIGH severity exceptions first", 
            "Implement recovery strategies for common exception types",
            "Monitor exception trends over time for proactive maintenance"
        ]
    }


# Test function
def test_exception_hierarchy():
    """Test the exception hierarchy functionality."""
    import logging
    
    # Set up test logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("Testing exception hierarchy...")
    
    # Test different exception types
    exceptions_to_test = [
        ConfigurationError(
            "Invalid configuration format", 
            config_file="test.yaml",
            recovery_suggestions=["Check YAML syntax", "Validate configuration schema"]
        ),
        DependencyError(
            "Missing required dependency",
            dependency_name="indic-transliteration",
            required_version=">=2.3.0",
            severity=ErrorSeverity.CRITICAL
        ),
        SRTProcessingError(
            "Invalid timestamp format",
            file_path="test.srt",
            segment_index=5,
            recovery_suggestions=["Check SRT format", "Validate timestamp syntax"]
        ),
        PerformanceError(
            "Processing time exceeded threshold",
            metric_name="processing_time_ms",
            expected_value=1000.0,
            actual_value=2500.0
        )
    ]
    
    print(f"\nTesting {len(exceptions_to_test)} different exception types:")
    
    for exc in exceptions_to_test:
        print(f"\n--- {exc.__class__.__name__} ---")
        print(f"Message: {exc.message}")
        print(f"Severity: {exc.severity.value}")
        print(f"Category: {exc.category.value}")
        print(f"Recovery guidance: {exc.get_recovery_guidance()}")
        
        # Test logging
        log_exception(exc, logger)
    
    # Test exception report
    print("\n--- Exception Report ---")
    report = create_exception_report(exceptions_to_test)
    print(f"Total exceptions: {report['summary']['total_exceptions']}")
    print(f"By severity: {report['summary']['by_severity']}")
    print(f"Most common category: {report['summary']['most_common_category']}")
    
    print("\nException hierarchy test completed successfully!")


if __name__ == "__main__":
    test_exception_hierarchy()