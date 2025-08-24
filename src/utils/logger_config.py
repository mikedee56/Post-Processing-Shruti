"""
Structured Logging Configuration for SRT Processing Pipeline.

This module provides comprehensive logging setup with configurable levels,
formatters, and handlers for file and console output.
"""

import os
import sys
import logging
import logging.config
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime


class ProcessingLoggerConfig:
    """
    Centralized logging configuration for the SRT processing pipeline.
    
    Provides structured logging with support for multiple handlers,
    configurable levels, and processing-specific formatters.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize logger configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            'level': 'INFO',
            'console_level': 'INFO',
            'file_level': 'DEBUG',
            'log_dir': 'logs',
            'max_file_size': 10 * 1024 * 1024,  # 10MB
            'backup_count': 5,
            'enable_console': True,
            'enable_file': True,
            'log_format': 'detailed',  # 'simple', 'detailed', 'json'
            'include_process_info': True,
            'include_thread_info': False
        }
        
        # Merge with provided config
        self.effective_config = {**self.default_config, **self.config}
        
        # Setup paths
        self.log_dir = Path(self.effective_config['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Logger setup state
        self._configured = False
    
    def setup_logging(self, logger_name: Optional[str] = None) -> logging.Logger:
        """
        Setup and configure logging for the application.
        
        Args:
            logger_name: Optional specific logger name
            
        Returns:
            Configured logger instance
        """
        if not self._configured:
            self._configure_logging()
            self._configured = True
        
        # Get logger
        if logger_name:
            logger = logging.getLogger(logger_name)
        else:
            logger = logging.getLogger('srt_processor')
        
        return logger
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger with the specified name.
        
        Args:
            name: Logger name
            
        Returns:
            Logger instance
        """
        return logging.getLogger(name)
    
    def _configure_logging(self) -> None:
        """Configure the logging system."""
        # Clear any existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Set root level
        root_level = getattr(logging, self.effective_config['level'].upper())
        root_logger.setLevel(root_level)
        
        # Setup formatters
        formatters = self._create_formatters()
        
        # Setup handlers
        handlers = []
        
        if self.effective_config['enable_console']:
            handlers.append(self._create_console_handler(formatters))
        
        if self.effective_config['enable_file']:
            handlers.extend(self._create_file_handlers(formatters))
        
        # Add handlers to root logger
        for handler in handlers:
            root_logger.addHandler(handler)
        
        # Configure specific loggers
        self._configure_specific_loggers()
        
        # Log configuration info
        logger = logging.getLogger(__name__)
        logger.info("Logging system configured successfully")
        logger.debug(f"Log directory: {self.log_dir}")
        logger.debug(f"Active handlers: {len(handlers)}")
    
    def _create_formatters(self) -> Dict[str, logging.Formatter]:
        """Create logging formatters."""
        formatters = {}
        
        # Base format components
        timestamp_fmt = '%(asctime)s'
        level_fmt = '%(levelname)-8s'
        name_fmt = '%(name)s'
        message_fmt = '%(message)s'
        
        # Optional components
        process_fmt = '[PID:%(process)d]' if self.effective_config['include_process_info'] else ''
        thread_fmt = '[TID:%(thread)d]' if self.effective_config['include_thread_info'] else ''
        
        # Simple formatter
        simple_format = f"{level_fmt} {message_fmt}"
        formatters['simple'] = logging.Formatter(simple_format)
        
        # Detailed formatter
        detailed_parts = [timestamp_fmt, level_fmt, name_fmt]
        if process_fmt:
            detailed_parts.append(process_fmt)
        if thread_fmt:
            detailed_parts.append(thread_fmt)
        detailed_parts.append(message_fmt)
        
        detailed_format = ' '.join(detailed_parts)
        formatters['detailed'] = logging.Formatter(
            detailed_format,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # JSON formatter (custom)
        formatters['json'] = self._create_json_formatter()
        
        # Error formatter (for error logs)
        error_format = f"{timestamp_fmt} {level_fmt} {name_fmt} {process_fmt} {message_fmt}"
        if self.effective_config['include_thread_info']:
            error_format = f"{timestamp_fmt} {level_fmt} {name_fmt} {process_fmt} {thread_fmt} {message_fmt}"
        
        formatters['error'] = logging.Formatter(
            error_format,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        return formatters
    
    def _create_json_formatter(self) -> logging.Formatter:
        """Create a JSON formatter for structured logging."""
        import json
        
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    'timestamp': self.formatTime(record, self.datefmt),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno
                }
                
                if self.effective_config['include_process_info']:
                    log_entry['process_id'] = record.process
                
                if self.effective_config['include_thread_info']:
                    log_entry['thread_id'] = record.thread
                
                # Add exception info if present
                if record.exc_info:
                    log_entry['exception'] = self.formatException(record.exc_info)
                
                # Add extra fields
                for key, value in record.__dict__.items():
                    if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                                   'module', 'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                                   'thread', 'threadName', 'processName', 'process', 'exc_info', 'exc_text',
                                   'stack_info', 'getMessage']:
                        log_entry[key] = value
                
                return json.dumps(log_entry, ensure_ascii=False)
        
        return JSONFormatter()
    
    def _create_console_handler(self, formatters: Dict[str, logging.Formatter]) -> logging.Handler:
        """Create console handler."""
        handler = logging.StreamHandler(sys.stdout)
        
        # Set level
        console_level = getattr(logging, self.effective_config['console_level'].upper())
        handler.setLevel(console_level)
        
        # Set formatter
        format_type = self.effective_config.get('console_format', 'simple')
        if format_type in formatters:
            handler.setFormatter(formatters[format_type])
        else:
            handler.setFormatter(formatters['simple'])
        
        return handler
    
    def _create_file_handlers(self, formatters: Dict[str, logging.Formatter]) -> list:
        """Create file handlers."""
        handlers = []
        
        try:
            # Main log file (rotating)
            main_log_file = self.log_dir / 'srt_processor.log'
            main_handler = logging.handlers.RotatingFileHandler(
                main_log_file,
                maxBytes=self.effective_config['max_file_size'],
                backupCount=self.effective_config['backup_count'],
                encoding='utf-8',
                delay=True  # Delay file creation until first emit
            )
            
            file_level = getattr(logging, self.effective_config['file_level'].upper())
            main_handler.setLevel(file_level)
            
            # Use detailed format for main log
            format_type = self.effective_config.get('file_format', 'detailed')
            if format_type in formatters:
                main_handler.setFormatter(formatters[format_type])
            else:
                main_handler.setFormatter(formatters['detailed'])
            
            handlers.append(main_handler)
            
        except (OSError, PermissionError) as e:
            # If file handler fails, log to console and continue
            console_logger = logging.getLogger(__name__)
            console_logger.warning(f"Could not create main log file handler: {e}")
        
        try:
            # Error log file (errors and warnings only)
            error_log_file = self.log_dir / 'errors.log'
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file,
                maxBytes=self.effective_config['max_file_size'],
                backupCount=self.effective_config['backup_count'],
                encoding='utf-8',
                delay=True  # Delay file creation until first emit
            )
            
            error_handler.setLevel(logging.WARNING)
            error_handler.setFormatter(formatters['error'])
            handlers.append(error_handler)
            
        except (OSError, PermissionError) as e:
            # If error handler fails, log to console and continue
            console_logger = logging.getLogger(__name__)
            console_logger.warning(f"Could not create error log file handler: {e}")
        
        # Processing metrics log (if JSON format enabled)
        if self.effective_config.get('log_format') == 'json' or self.effective_config.get('enable_metrics_log', False):
            try:
                metrics_log_file = self.log_dir / f"processing_metrics_{datetime.now().strftime('%Y%m%d')}.log"
                metrics_handler = logging.handlers.RotatingFileHandler(
                    metrics_log_file,
                    maxBytes=self.effective_config['max_file_size'],
                    backupCount=self.effective_config['backup_count'],
                    encoding='utf-8',
                    delay=True  # Delay file creation until first emit
                )
                
                metrics_handler.setLevel(logging.INFO)
                metrics_handler.setFormatter(formatters['json'])
                
                # Create filter for metrics-related logs
                class MetricsFilter(logging.Filter):
                    def filter(self, record):
                        return hasattr(record, 'metrics') or 'metrics' in record.getMessage().lower()
                
                metrics_handler.addFilter(MetricsFilter())
                handlers.append(metrics_handler)
                
            except (OSError, PermissionError) as e:
                # If metrics handler fails, log to console and continue
                console_logger = logging.getLogger(__name__)
                console_logger.warning(f"Could not create metrics log file handler: {e}")
        
        return handlers
    
    def _configure_specific_loggers(self) -> None:
        """Configure specific loggers with custom behavior."""
        # Configure third-party library loggers
        
        # Reduce verbosity of common libraries
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('pysrt').setLevel(logging.INFO)
        
        # Configure processing-specific loggers
        processing_loggers = [
            'srt_processor',
            'srt_parser',
            'text_normalizer',
            'metrics_collector',
            'sanskrit_post_processor'
        ]
        
        for logger_name in processing_loggers:
            logger = logging.getLogger(logger_name)
            # These loggers inherit from root, but we can set specific levels if needed
            logger.setLevel(logging.DEBUG)
    
    @staticmethod
    def create_processing_logger(name: str, config: Optional[Dict] = None) -> logging.Logger:
        """
        Convenience method to create a configured logger for processing components.
        
        Args:
            name: Logger name
            config: Optional configuration
            
        Returns:
            Configured logger
        """
        logger_config = ProcessingLoggerConfig(config)
        return logger_config.setup_logging(name)
    
    def add_file_handler(self, filepath: Union[str, Path], level: str = 'INFO', 
                        formatter_type: str = 'detailed') -> None:
        """
        Add an additional file handler.
        
        Args:
            filepath: Path to log file
            level: Logging level
            formatter_type: Type of formatter to use
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler(filepath, encoding='utf-8')
        handler.setLevel(getattr(logging, level.upper()))
        
        # Create formatter
        formatters = self._create_formatters()
        if formatter_type in formatters:
            handler.setFormatter(formatters[formatter_type])
        else:
            handler.setFormatter(formatters['detailed'])
        
        # Add to root logger
        logging.getLogger().addHandler(handler)
    
    def create_session_logger(self, session_id: str) -> logging.Logger:
        """
        Create a session-specific logger.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Session-specific logger
        """
        logger_name = f"session.{session_id}"
        logger = logging.getLogger(logger_name)
        
        # Add session-specific file handler
        session_log_file = self.log_dir / f"session_{session_id}.log"
        self.add_file_handler(session_log_file, 'DEBUG', 'detailed')
        
        return logger


# Import required modules at module level
import logging.handlers


# Convenience functions
def setup_logging(config: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Setup logging with default configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured root logger
    """
    logger_config = ProcessingLoggerConfig(config)
    return logger_config.setup_logging()


def get_logger(name: str, config: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Get a configured logger by name.
    
    Args:
        name: Logger name
        config: Optional configuration
        
    Returns:
        Configured logger
    """
    logger_config = ProcessingLoggerConfig(config)
    logger_config.setup_logging()  # Ensure logging is configured
    return logging.getLogger(name)


def setup_test_logging(log_level: Union[str, int] = "WARNING", log_dir: Optional[Path] = None) -> None:
    """
    Setup minimal logging configuration for tests.
    
    Configures logging to suppress verbose output during test execution
    while ensuring critical errors are still captured.
    
    Args:
        log_level: Logging level for tests (default: WARNING) - string or integer
        log_dir: Optional directory for test logs
    """
    # Convert log level to numeric level
    if isinstance(log_level, str):
        numeric_level = getattr(logging, log_level.upper(), logging.WARNING)
    else:
        numeric_level = log_level
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format='%(levelname)s: %(message)s',
        force=True  # Override any existing configuration
    )
    
    # Set specific test logger levels
    logging.getLogger('utils').setLevel(logging.ERROR)
    logging.getLogger('post_processors').setLevel(logging.ERROR)
    logging.getLogger('sanskrit_hindi_identifier').setLevel(logging.ERROR)
    logging.getLogger('ner_module').setLevel(logging.ERROR)
    
    # If log_dir is specified, create it
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)