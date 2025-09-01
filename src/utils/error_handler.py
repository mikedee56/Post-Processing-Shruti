"""
Error handling utilities for Post-Processing Shruti system.
Provides centralized error handling and logging capabilities.
"""
import logging
import traceback
from typing import Any, Optional, Dict


class ErrorHandler:
    """Centralized error handling and logging system."""
    
    def __init__(self, logger_name: str = "error_handler", component: str = None):
        """Initialize error handler with specified logger and component.
        
        Args:
            logger_name: Name of the logger to use
            component: Name of the component using this error handler
        """
        self.logger = logging.getLogger(logger_name)
        self.component = component or "unknown_component"
        
    def handle_error(self, error: Exception, context: str = "", 
                    additional_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Handle an error with proper logging and context.
        
        Args:
            error: The exception that occurred
            context: Contextual information about where/when the error occurred
            additional_info: Optional dictionary of additional debugging information
        """
        error_msg = f"[{self.component}] Error in {context}: {str(error)}"
        
        if additional_info:
            error_msg += f" | Additional info: {additional_info}"
            
        self.logger.error(error_msg)
        self.logger.debug(f"[{self.component}] Full traceback: {traceback.format_exc()}")
        
    def handle_processing_error(self, error: Exception, context: str = "", 
                              additional_info: Optional[Dict[str, Any]] = None) -> None:
        """Handle processing-specific errors with proper context."""
        self.handle_error(error, f"processing - {context}", additional_info)
        
    def log_warning(self, message: str, context: str = "") -> None:
        """Log a warning message with context."""
        warning_msg = f"[{self.component}] Warning in {context}: {message}" if context else f"[{self.component}] {message}"
        self.logger.warning(warning_msg)
        
    def log_info(self, message: str, context: str = "") -> None:
        """Log an info message with context."""
        info_msg = f"[{self.component}] Info in {context}: {message}" if context else f"[{self.component}] {message}"
        self.logger.info(info_msg)
        
    def log_operation_start(self, operation: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log the start of an operation."""
        msg = f"[{self.component}] Starting operation: {operation}"
        if details:
            msg += f" | Details: {details}"
        self.logger.info(msg)
        
    def log_operation_success(self, operation: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log successful completion of an operation."""
        msg = f"[{self.component}] Operation completed successfully: {operation}"
        if details:
            msg += f" | Details: {details}"
        self.logger.info(msg)
        
    def safe_execute(self, func, *args, **kwargs):
        """
        Safely execute a function with error handling.
        
        Returns:
            Tuple of (success: bool, result: Any, error: Optional[Exception])
        """
        try:
            result = func(*args, **kwargs)
            return True, result, None
        except Exception as e:
            self.handle_error(e, f"safe_execute({func.__name__})")
            return False, None, e