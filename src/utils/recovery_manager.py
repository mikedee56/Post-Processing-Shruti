"""
Epic 4 - Story 4.1: Batch Processing Framework
RecoveryManager for handling failures and implementing recovery mechanisms

This module provides:
- Automatic failure detection and recovery
- Retry mechanisms with exponential backoff
- Checkpoint and resume functionality
- Error classification and handling strategies
- Recovery state persistence
"""

import os
import json
import logging
import time
import pickle
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import threading
from enum import Enum
import traceback


class ErrorType(Enum):
    """Classification of different error types."""
    TEMPORARY = "temporary"         # Network issues, temporary resource unavailability
    RECOVERABLE = "recoverable"     # File permission issues, disk space
    PERMANENT = "permanent"         # Invalid file format, corrupted data
    RESOURCE = "resource"           # Memory, CPU, disk space issues
    TIMEOUT = "timeout"             # Processing timeout
    UNKNOWN = "unknown"             # Unclassified errors


class RecoveryStrategy(Enum):
    """Different recovery strategies for error handling."""
    RETRY = "retry"                 # Retry with exponential backoff
    SKIP = "skip"                   # Skip the failed item
    QUARANTINE = "quarantine"       # Move to failed items directory
    ABORT = "abort"                 # Abort entire batch
    MANUAL = "manual"               # Require manual intervention


@dataclass
class FailureRecord:
    """Record of a processing failure."""
    item_id: str
    error_type: ErrorType
    error_message: str
    stack_trace: str
    timestamp: datetime
    attempt_count: int
    recovery_strategy: RecoveryStrategy
    metadata: Dict[str, Any]


@dataclass
class RecoveryConfig:
    """Configuration for recovery behavior."""
    max_retry_attempts: int = 3
    base_retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    exponential_backoff: bool = True
    
    # Error classification thresholds
    temporary_error_keywords: List[str] = None
    permanent_error_keywords: List[str] = None
    
    # Recovery strategies per error type
    strategy_map: Dict[ErrorType, RecoveryStrategy] = None
    
    # Checkpoint settings
    enable_checkpoints: bool = True
    checkpoint_interval: int = 100  # Items processed between checkpoints
    checkpoint_dir: str = "/tmp/recovery_checkpoints"
    
    # Persistence settings
    failure_log_path: str = "/tmp/failure_log.json"
    recovery_state_path: str = "/tmp/recovery_state.pkl"
    
    def __post_init__(self):
        """Initialize default values."""
        if self.temporary_error_keywords is None:
            self.temporary_error_keywords = [
                "connection", "network", "timeout", "temporary", "busy", 
                "unavailable", "rate limit", "throttle"
            ]
            
        if self.permanent_error_keywords is None:
            self.permanent_error_keywords = [
                "invalid format", "corrupted", "malformed", "parse error",
                "encoding error", "syntax error", "not found"
            ]
            
        if self.strategy_map is None:
            self.strategy_map = {
                ErrorType.TEMPORARY: RecoveryStrategy.RETRY,
                ErrorType.RECOVERABLE: RecoveryStrategy.RETRY,
                ErrorType.PERMANENT: RecoveryStrategy.QUARANTINE,
                ErrorType.RESOURCE: RecoveryStrategy.RETRY,
                ErrorType.TIMEOUT: RecoveryStrategy.RETRY,
                ErrorType.UNKNOWN: RecoveryStrategy.RETRY
            }


@dataclass
class CheckpointData:
    """Data saved at checkpoints for recovery."""
    checkpoint_id: str
    timestamp: datetime
    batch_id: str
    processed_items: List[str]
    failed_items: List[str]
    current_position: int
    total_items: int
    processing_state: Dict[str, Any]


class RecoveryManager:
    """
    Manages failure detection, classification, and recovery for batch processing.
    
    Features:
    - Automatic retry with exponential backoff
    - Error classification and appropriate handling
    - Checkpoint and resume functionality
    - Failure logging and analysis
    - Recovery state persistence
    """
    
    def __init__(self, config: Optional[RecoveryConfig] = None):
        self.config = config or RecoveryConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # State
        self.failure_records: List[FailureRecord] = []
        self.current_checkpoint: Optional[CheckpointData] = None
        self.recovery_state: Dict[str, Any] = {}
        
        # Statistics
        self.total_failures = 0
        self.recovered_items = 0
        self.permanent_failures = 0
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Ensure directories exist
        self._ensure_directories()
        
        # Load existing state
        self._load_recovery_state()
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        if self.config.enable_checkpoints:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        # Ensure parent directories for log files exist
        os.makedirs(os.path.dirname(self.config.failure_log_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.config.recovery_state_path), exist_ok=True)
    
    def _load_recovery_state(self):
        """Load previous recovery state if available."""
        try:
            if os.path.exists(self.config.recovery_state_path):
                with open(self.config.recovery_state_path, 'rb') as f:
                    self.recovery_state = pickle.load(f)
                self.logger.info("Loaded previous recovery state")
        except Exception as e:
            self.logger.warning(f"Could not load recovery state: {e}")
    
    def _save_recovery_state(self):
        """Save current recovery state."""
        try:
            with open(self.config.recovery_state_path, 'wb') as f:
                pickle.dump(self.recovery_state, f)
        except Exception as e:
            self.logger.warning(f"Could not save recovery state: {e}")
    
    def classify_error(self, error: Exception, context: Dict[str, Any] = None) -> ErrorType:
        """
        Classify an error to determine appropriate recovery strategy.
        
        Args:
            error: The exception that occurred
            context: Additional context about the error
            
        Returns:
            ErrorType classification
        """
        error_message = str(error).lower()
        
        # Check for temporary errors
        if any(keyword in error_message for keyword in self.config.temporary_error_keywords):
            return ErrorType.TEMPORARY
        
        # Check for permanent errors
        if any(keyword in error_message for keyword in self.config.permanent_error_keywords):
            return ErrorType.PERMANENT
        
        # Check error type
        if isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorType.TEMPORARY
        
        if isinstance(error, FileNotFoundError):
            return ErrorType.PERMANENT
        
        if isinstance(error, PermissionError):
            return ErrorType.RECOVERABLE
        
        if isinstance(error, MemoryError):
            return ErrorType.RESOURCE
        
        # Check context for additional clues
        if context:
            if context.get('timeout_occurred'):
                return ErrorType.TIMEOUT
            
            if context.get('resource_exhausted'):
                return ErrorType.RESOURCE
        
        return ErrorType.UNKNOWN
    
    def record_failure(
        self,
        item_id: str,
        error: Exception,
        context: Dict[str, Any] = None,
        attempt_count: int = 1
    ) -> FailureRecord:
        """
        Record a processing failure and determine recovery strategy.
        
        Args:
            item_id: Identifier for the failed item
            error: The exception that occurred
            context: Additional context about the failure
            attempt_count: Number of attempts made so far
            
        Returns:
            FailureRecord with error details and recovery strategy
        """
        with self._lock:
            error_type = self.classify_error(error, context)
            recovery_strategy = self.config.strategy_map.get(error_type, RecoveryStrategy.SKIP)
            
            failure_record = FailureRecord(
                item_id=item_id,
                error_type=error_type,
                error_message=str(error),
                stack_trace=traceback.format_exc(),
                timestamp=datetime.utcnow(),
                attempt_count=attempt_count,
                recovery_strategy=recovery_strategy,
                metadata=context or {}
            )
            
            self.failure_records.append(failure_record)
            self.total_failures += 1
            
            # Log the failure
            self.logger.warning(
                f"Recorded failure for {item_id}: {error_type.value} - {str(error)}"
            )
            
            # Persist failure log
            self._save_failure_log()
            
            return failure_record
    
    def should_retry(self, failure_record: FailureRecord) -> bool:
        """
        Determine if an item should be retried based on failure record.
        
        Args:
            failure_record: The failure record to evaluate
            
        Returns:
            True if the item should be retried
        """
        if failure_record.recovery_strategy != RecoveryStrategy.RETRY:
            return False
        
        if failure_record.attempt_count >= self.config.max_retry_attempts:
            self.logger.info(f"Max retry attempts reached for {failure_record.item_id}")
            return False
        
        # Don't retry permanent failures
        if failure_record.error_type == ErrorType.PERMANENT:
            return False
        
        return True
    
    def calculate_retry_delay(self, attempt_count: int) -> float:
        """
        Calculate delay before next retry attempt.
        
        Args:
            attempt_count: Number of previous attempts
            
        Returns:
            Delay in seconds
        """
        if not self.config.exponential_backoff:
            return self.config.base_retry_delay
        
        delay = self.config.base_retry_delay * (2 ** (attempt_count - 1))
        return min(delay, self.config.max_retry_delay)
    
    def execute_with_recovery(
        self,
        func: Callable,
        item_id: str,
        *args,
        **kwargs
    ) -> Union[Any, None]:
        """
        Execute a function with automatic retry and recovery.
        
        Args:
            func: Function to execute
            item_id: Identifier for the item being processed
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result or None if all retries failed
        """
        attempt_count = 0
        last_error = None
        
        while attempt_count < self.config.max_retry_attempts:
            attempt_count += 1
            
            try:
                result = func(*args, **kwargs)
                
                # Success - record recovery if this was a retry
                if attempt_count > 1:
                    with self._lock:
                        self.recovered_items += 1
                    self.logger.info(f"Recovered {item_id} on attempt {attempt_count}")
                
                return result
                
            except Exception as error:
                last_error = error
                
                # Record the failure
                failure_record = self.record_failure(
                    item_id=item_id,
                    error=error,
                    context={'function': func.__name__, 'args_count': len(args)},
                    attempt_count=attempt_count
                )
                
                # Check if we should retry
                if not self.should_retry(failure_record):
                    self.logger.error(f"Permanent failure for {item_id}: {str(error)}")
                    with self._lock:
                        self.permanent_failures += 1
                    break
                
                # Calculate retry delay
                if attempt_count < self.config.max_retry_attempts:
                    retry_delay = self.calculate_retry_delay(attempt_count)
                    self.logger.info(f"Retrying {item_id} in {retry_delay:.1f} seconds (attempt {attempt_count + 1})")
                    time.sleep(retry_delay)
        
        # All retries exhausted
        self.logger.error(f"All retry attempts exhausted for {item_id}: {str(last_error)}")
        return None
    
    def create_checkpoint(
        self,
        checkpoint_id: str,
        batch_id: str,
        processed_items: List[str],
        failed_items: List[str],
        current_position: int,
        total_items: int,
        processing_state: Dict[str, Any] = None
    ):
        """
        Create a checkpoint for recovery purposes.
        
        Args:
            checkpoint_id: Unique identifier for the checkpoint
            batch_id: Identifier for the current batch
            processed_items: List of successfully processed items
            failed_items: List of failed items
            current_position: Current position in processing
            total_items: Total number of items to process
            processing_state: Additional state information
        """
        if not self.config.enable_checkpoints:
            return
        
        checkpoint_data = CheckpointData(
            checkpoint_id=checkpoint_id,
            timestamp=datetime.utcnow(),
            batch_id=batch_id,
            processed_items=processed_items.copy(),
            failed_items=failed_items.copy(),
            current_position=current_position,
            total_items=total_items,
            processing_state=processing_state or {}
        )
        
        self.current_checkpoint = checkpoint_data
        
        # Save checkpoint to disk
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_{checkpoint_id}.pkl"
        )
        
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            self.logger.info(f"Created checkpoint {checkpoint_id} at position {current_position}")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint {checkpoint_id}: {e}")
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[CheckpointData]:
        """
        Load a checkpoint for recovery.
        
        Args:
            checkpoint_id: Identifier of the checkpoint to load
            
        Returns:
            CheckpointData or None if not found
        """
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_{checkpoint_id}.pkl"
        )
        
        try:
            if os.path.exists(checkpoint_path):
                with open(checkpoint_path, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                
                self.current_checkpoint = checkpoint_data
                self.logger.info(f"Loaded checkpoint {checkpoint_id}")
                return checkpoint_data
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
        
        return None
    
    def find_latest_checkpoint(self, batch_id: str = None) -> Optional[CheckpointData]:
        """
        Find the most recent checkpoint, optionally for a specific batch.
        
        Args:
            batch_id: Optional batch ID to filter by
            
        Returns:
            Latest CheckpointData or None
        """
        if not os.path.exists(self.config.checkpoint_dir):
            return None
        
        latest_checkpoint = None
        latest_time = None
        
        try:
            for filename in os.listdir(self.config.checkpoint_dir):
                if not filename.startswith('checkpoint_') or not filename.endswith('.pkl'):
                    continue
                
                filepath = os.path.join(self.config.checkpoint_dir, filename)
                
                with open(filepath, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                
                # Filter by batch ID if specified
                if batch_id and checkpoint_data.batch_id != batch_id:
                    continue
                
                # Check if this is the latest
                if latest_time is None or checkpoint_data.timestamp > latest_time:
                    latest_checkpoint = checkpoint_data
                    latest_time = checkpoint_data.timestamp
        
        except Exception as e:
            self.logger.error(f"Error finding latest checkpoint: {e}")
        
        return latest_checkpoint
    
    def cleanup_old_checkpoints(self, max_age_hours: int = 24):
        """
        Remove old checkpoint files.
        
        Args:
            max_age_hours: Maximum age of checkpoints to keep
        """
        if not os.path.exists(self.config.checkpoint_dir):
            return
        
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        removed_count = 0
        
        try:
            for filename in os.listdir(self.config.checkpoint_dir):
                if not filename.startswith('checkpoint_') or not filename.endswith('.pkl'):
                    continue
                
                filepath = os.path.join(self.config.checkpoint_dir, filename)
                
                # Check file modification time
                mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                
                if mtime < cutoff_time:
                    os.remove(filepath)
                    removed_count += 1
            
            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} old checkpoint files")
        
        except Exception as e:
            self.logger.error(f"Error cleaning up checkpoints: {e}")
    
    def _save_failure_log(self):
        """Save failure records to log file."""
        try:
            failure_data = [asdict(record) for record in self.failure_records]
            
            # Convert datetime objects to ISO strings
            for record in failure_data:
                if 'timestamp' in record:
                    record['timestamp'] = record['timestamp'].isoformat()
            
            with open(self.config.failure_log_path, 'w') as f:
                json.dump(failure_data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to save failure log: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get recovery statistics.
        
        Returns:
            Dictionary with recovery statistics
        """
        with self._lock:
            error_counts = {}
            strategy_counts = {}
            
            for record in self.failure_records:
                error_type = record.error_type.value
                strategy = record.recovery_strategy.value
                
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            return {
                'total_failures': self.total_failures,
                'recovered_items': self.recovered_items,
                'permanent_failures': self.permanent_failures,
                'recovery_rate': self.recovered_items / self.total_failures if self.total_failures > 0 else 0,
                'error_type_distribution': error_counts,
                'recovery_strategy_distribution': strategy_counts,
                'current_checkpoint': self.current_checkpoint.checkpoint_id if self.current_checkpoint else None
            }
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """
        Alias for get_statistics() for Epic 4 compatibility.
        
        Returns:
            Dictionary with recovery statistics
        """
        return self.get_statistics()

    def reset_statistics(self):
        """Reset recovery statistics."""
        with self._lock:
            self.failure_records = []
            self.total_failures = 0
            self.recovered_items = 0
            self.permanent_failures = 0
            self.logger.info("Recovery statistics reset")


# Utility decorators for easy recovery integration

def with_recovery(recovery_manager: RecoveryManager, item_id_param: str = 'item_id'):
    """
    Decorator to add automatic recovery to a function.
    
    Args:
        recovery_manager: RecoveryManager instance to use
        item_id_param: Name of parameter containing item ID
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract item ID from parameters
            item_id = kwargs.get(item_id_param, 'unknown')
            
            return recovery_manager.execute_with_recovery(
                func, item_id, *args, **kwargs
            )
        return wrapper
    return decorator