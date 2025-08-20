"""
Structured Logging Enhancement for Production Observability
Provides comprehensive structured logging with correlation IDs, context, and searchability
"""

import json
import logging
import logging.handlers
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextvars import ContextVar
from pathlib import Path

# Context variables for correlation
correlation_id_context: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
user_context: ContextVar[Optional[Dict[str, str]]] = ContextVar('user_context', default=None)
request_context: ContextVar[Optional[Dict[str, str]]] = ContextVar('request_context', default=None)


class LogLevel(Enum):
    """Enhanced log levels"""
    TRACE = "TRACE"
    DEBUG = "DEBUG"  
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    AUDIT = "AUDIT"  # Special level for audit logs


class LogCategory(Enum):
    """Log categories for better organization"""
    APPLICATION = "application"
    SECURITY = "security"
    PERFORMANCE = "performance"
    BUSINESS = "business"
    SYSTEM = "system"
    AUDIT = "audit"
    ERROR = "error"


@dataclass
class LogContext:
    """Structured log context"""
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    environment: str = "production"
    service_name: str = "asr-processor"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values"""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class StructuredLogEntry:
    """Structured log entry"""
    timestamp: datetime
    level: LogLevel
    message: str
    logger_name: str
    category: LogCategory = LogCategory.APPLICATION
    
    # Context information
    context: LogContext = field(default_factory=LogContext)
    
    # Structured fields
    fields: Dict[str, Any] = field(default_factory=dict)
    
    # Error information
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    stack_trace: Optional[str] = None
    
    # Performance metrics
    duration_ms: Optional[float] = None
    memory_mb: Optional[float] = None
    
    # Business metrics
    business_event: Optional[str] = None
    business_entity: Optional[str] = None
    business_entity_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        # Handle category - it could be an enum or string
        if hasattr(self.category, 'value'):
            category_value = self.category.value
        else:
            category_value = str(self.category)
            
        entry = {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.value,
            'message': self.message,
            'logger': self.logger_name,
            'category': category_value,
            **self.context.to_dict(),
            **self.fields
        }
        
        # Add error information if present
        if self.exception_type:
            entry['error'] = {
                'type': self.exception_type,
                'message': self.exception_message,
                'stack_trace': self.stack_trace
            }
        
        # Add performance metrics if present
        if self.duration_ms is not None:
            entry['performance'] = {
                'duration_ms': self.duration_ms,
                'memory_mb': self.memory_mb
            }
        
        # Add business context if present
        if self.business_event:
            entry['business'] = {
                'event': self.business_event,
                'entity': self.business_entity,
                'entity_id': self.business_entity_id
            }
        
        return entry
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False)


class StructuredLogFormatter(logging.Formatter):
    """Custom formatter for structured logging"""
    
    def __init__(self, include_context: bool = True):
        super().__init__()
        self.include_context = include_context
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""
        
        # Create log context
        context = LogContext(
            correlation_id=correlation_id_context.get(),
            service_name="asr-processor",
            environment="production"
        )
        
        # Add user context if available
        user_ctx = user_context.get()
        if user_ctx:
            context.user_id = user_ctx.get('user_id')
            context.session_id = user_ctx.get('session_id')
        
        # Add request context if available
        req_ctx = request_context.get()
        if req_ctx:
            context.request_id = req_ctx.get('request_id')
            context.trace_id = req_ctx.get('trace_id')
            context.span_id = req_ctx.get('span_id')
        
        # Extract structured fields from record
        fields = {}
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                          'module', 'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'stack_info', 'exc_info']:
                fields[key] = value
        
        # Create structured log entry
        entry = StructuredLogEntry(
            timestamp=datetime.fromtimestamp(record.created, timezone.utc),
            level=LogLevel(record.levelname),
            message=record.getMessage(),
            logger_name=record.name,
            category=getattr(record, 'category', LogCategory.APPLICATION),
            context=context if self.include_context else LogContext(),
            fields=fields
        )
        
        # Add exception information if present
        if record.exc_info:
            entry.exception_type = record.exc_info[0].__name__
            entry.exception_message = str(record.exc_info[1])
            entry.stack_trace = self.formatException(record.exc_info)
        
        return entry.to_json()


class StructuredLogger:
    """
    Enhanced structured logger for production observability
    Provides correlation IDs, context tracking, and structured fields
    """
    
    def __init__(self, name: str, config: Optional[Dict] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(name)
        
        # Setup logger if not already configured
        if not self.logger.handlers:
            self._setup_logger()
    
    def _setup_logger(self):
        """Setup structured logging configuration"""
        
        # Set log level
        log_level = self.config.get('level', 'INFO')
        self.logger.setLevel(getattr(logging, log_level))
        
        # Console handler with structured formatting
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(StructuredLogFormatter(include_context=True))
        self.logger.addHandler(console_handler)
        
        # File handler for structured logs
        if 'log_file' in self.config:
            file_handler = logging.handlers.RotatingFileHandler(
                self.config['log_file'],
                maxBytes=self.config.get('max_bytes', 50 * 1024 * 1024),  # 50MB
                backupCount=self.config.get('backup_count', 10)
            )
            file_handler.setFormatter(StructuredLogFormatter(include_context=True))
            self.logger.addHandler(file_handler)
        
        # Separate JSON file handler
        if 'json_log_file' in self.config:
            json_handler = logging.handlers.RotatingFileHandler(
                self.config['json_log_file'],
                maxBytes=self.config.get('max_bytes', 50 * 1024 * 1024),
                backupCount=self.config.get('backup_count', 10)
            )
            json_handler.setFormatter(StructuredLogFormatter(include_context=True))
            self.logger.addHandler(json_handler)
    
    def with_context(self, **context_fields) -> 'ContextualLogger':
        """Create contextual logger with additional fields"""
        return ContextualLogger(self, context_fields)
    
    def trace(self, message: str, **fields):
        """Log trace level message"""
        self._log(LogLevel.TRACE, message, LogCategory.APPLICATION, **fields)
    
    def debug(self, message: str, **fields):
        """Log debug level message"""
        self._log(LogLevel.DEBUG, message, LogCategory.APPLICATION, **fields)
    
    def info(self, message: str, **fields):
        """Log info level message"""
        self._log(LogLevel.INFO, message, LogCategory.APPLICATION, **fields)
    
    def warning(self, message: str, **fields):
        """Log warning level message"""
        self._log(LogLevel.WARNING, message, LogCategory.APPLICATION, **fields)
    
    def error(self, message: str, exception: Optional[Exception] = None, **fields):
        """Log error level message"""
        extra = fields.copy()
        if exception:
            extra['has_exception'] = True
        self._log(LogLevel.ERROR, message, LogCategory.ERROR, **extra)
    
    def critical(self, message: str, exception: Optional[Exception] = None, **fields):
        """Log critical level message"""
        extra = fields.copy()
        if exception:
            extra['has_exception'] = True
        self._log(LogLevel.CRITICAL, message, LogCategory.ERROR, **extra)
    
    def audit(self, message: str, **fields):
        """Log audit message"""
        self._log(LogLevel.AUDIT, message, LogCategory.AUDIT, **fields)
    
    def security_event(self, message: str, **fields):
        """Log security event"""
        self._log(LogLevel.WARNING, message, LogCategory.SECURITY, **fields)
    
    def performance(self, message: str, duration_ms: float, **fields):
        """Log performance metric"""
        fields['duration_ms'] = duration_ms
        self._log(LogLevel.INFO, message, LogCategory.PERFORMANCE, **fields)
    
    def business_event(self, event: str, entity: Optional[str] = None, 
                      entity_id: Optional[str] = None, **fields):
        """Log business event"""
        fields.update({
            'business_event': event,
            'business_entity': entity,
            'business_entity_id': entity_id
        })
        self._log(LogLevel.INFO, f"Business event: {event}", LogCategory.BUSINESS, **fields)
    
    def _log(self, level: LogLevel, message: str, category: LogCategory, **fields):
        """Internal logging method"""
        # Extract exc_info if present (it's a reserved parameter)
        exc_info = fields.pop('exc_info', None)
        
        # Add category to extra fields - convert enum to value for JSON serialization
        extra = fields.copy()
        extra['category'] = category.value
        
        # Map custom levels to standard logging levels
        if level == LogLevel.TRACE:
            log_level = logging.DEBUG
        elif level == LogLevel.AUDIT:
            log_level = logging.INFO
        else:
            log_level = getattr(logging, level.value)
        
        # Pass exc_info as a separate parameter if present
        if exc_info:
            self.logger.log(log_level, message, exc_info=exc_info, extra=extra)
        else:
            self.logger.log(log_level, message, extra=extra)


class ContextualLogger:
    """Logger with persistent context fields"""
    
    def __init__(self, base_logger: StructuredLogger, context_fields: Dict[str, Any]):
        self.base_logger = base_logger
        self.context_fields = context_fields
    
    def trace(self, message: str, **fields):
        """Log trace with context"""
        self.base_logger.trace(message, **{**self.context_fields, **fields})
    
    def debug(self, message: str, **fields):
        """Log debug with context"""
        self.base_logger.debug(message, **{**self.context_fields, **fields})
    
    def info(self, message: str, **fields):
        """Log info with context"""
        self.base_logger.info(message, **{**self.context_fields, **fields})
    
    def warning(self, message: str, **fields):
        """Log warning with context"""
        self.base_logger.warning(message, **{**self.context_fields, **fields})
    
    def error(self, message: str, exception: Optional[Exception] = None, **fields):
        """Log error with context"""
        self.base_logger.error(message, exception, **{**self.context_fields, **fields})
    
    def critical(self, message: str, exception: Optional[Exception] = None, **fields):
        """Log critical with context"""
        self.base_logger.critical(message, exception, **{**self.context_fields, **fields})
    
    def audit(self, message: str, **fields):
        """Log audit with context"""
        self.base_logger.audit(message, **{**self.context_fields, **fields})


class LoggingMiddleware:
    """Middleware for request/response logging"""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
    
    def log_request(self, method: str, path: str, headers: Dict[str, str],
                   user_id: Optional[str] = None):
        """Log incoming request"""
        correlation_id = headers.get('X-Correlation-ID', str(uuid.uuid4()))
        correlation_id_context.set(correlation_id)
        
        if user_id:
            user_context.set({'user_id': user_id})
        
        self.logger.info(
            f"Incoming request: {method} {path}",
            method=method,
            path=path,
            correlation_id=correlation_id,
            user_id=user_id
        )
    
    def log_response(self, status_code: int, duration_ms: float, 
                    response_size: Optional[int] = None):
        """Log outgoing response"""
        self.logger.performance(
            f"Response sent: {status_code}",
            duration_ms=duration_ms,
            status_code=status_code,
            response_size=response_size
        )


def timed_operation(logger: StructuredLogger, operation_name: str):
    """Decorator for timing operations"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                logger.performance(
                    f"Operation completed: {operation_name}",
                    duration_ms=duration_ms,
                    operation=operation_name,
                    success=True
                )
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"Operation failed: {operation_name}",
                    exception=e,
                    duration_ms=duration_ms,
                    operation=operation_name,
                    success=False
                )
                raise
        return wrapper
    return decorator


# Global structured logger
_global_logger: Optional[StructuredLogger] = None


def initialize_structured_logging(config: Optional[Dict] = None) -> StructuredLogger:
    """Initialize global structured logging"""
    global _global_logger
    _global_logger = StructuredLogger("asr_processor", config or {})
    return _global_logger


def get_logger(name: Optional[str] = None) -> StructuredLogger:
    """Get structured logger instance"""
    if _global_logger and not name:
        return _global_logger
    return StructuredLogger(name or "asr_processor")


def set_correlation_id(correlation_id: str):
    """Set correlation ID for current context"""
    correlation_id_context.set(correlation_id)


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID"""
    return correlation_id_context.get()


def set_user_context(user_id: str, session_id: Optional[str] = None):
    """Set user context for current request"""
    user_context.set({
        'user_id': user_id,
        'session_id': session_id or str(uuid.uuid4())
    })


def set_request_context(request_id: str, trace_id: Optional[str] = None, 
                       span_id: Optional[str] = None):
    """Set request context for current request"""
    request_context.set({
        'request_id': request_id,
        'trace_id': trace_id,
        'span_id': span_id
    })


def clear_context():
    """Clear all context variables"""
    correlation_id_context.set(None)
    user_context.set(None)
    request_context.set(None)