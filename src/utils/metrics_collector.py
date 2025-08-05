"""
Processing Metrics Collection and Reporting.

This module provides comprehensive metrics collection, analysis, and reporting
for SRT processing pipelines with detailed performance and quality tracking.
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import logging


@dataclass
class ProcessingMetrics:
    """Comprehensive processing metrics for a single file."""
    file_path: str
    processing_time: float
    total_segments: int
    segments_modified: int
    corrections_applied: Dict[str, int] = field(default_factory=dict)
    timestamp_integrity_verified: bool = True
    errors_encountered: List[str] = field(default_factory=list)
    warnings_encountered: List[str] = field(default_factory=list)
    
    # Text statistics
    original_word_count: int = 0
    processed_word_count: int = 0
    original_char_count: int = 0
    processed_char_count: int = 0
    
    # Quality metrics
    confidence_scores: List[float] = field(default_factory=list)
    average_confidence: float = 0.0
    flagged_segments: int = 0
    
    # Timing breakdown
    parsing_time: float = 0.0
    normalization_time: float = 0.0
    correction_time: float = 0.0
    validation_time: float = 0.0
    
    # Processing metadata
    processing_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    processor_version: str = "1.0.0"
    configuration_used: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionMetrics:
    """Aggregated metrics for a processing session."""
    session_id: str
    start_time: str
    end_time: Optional[str] = None
    total_files_processed: int = 0
    successful_files: int = 0
    failed_files: int = 0
    total_processing_time: float = 0.0
    total_segments_processed: int = 0
    total_corrections_applied: int = 0
    file_metrics: List[ProcessingMetrics] = field(default_factory=list)
    session_errors: List[str] = field(default_factory=list)


class MetricsCollector:
    """
    Comprehensive metrics collection and analysis system.
    
    Collects, aggregates, and reports processing metrics with support for
    real-time monitoring, historical analysis, and performance benchmarking.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the metrics collector.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Session tracking
        self.current_session: Optional[SessionMetrics] = None
        self.session_start_time: Optional[float] = None
        
        # Performance tracking
        self.operation_timers: Dict[str, float] = {}
        
        # Storage configuration
        self.metrics_dir = Path(self.config.get('metrics_dir', 'data/metrics'))
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-save configuration
        self.auto_save = self.config.get('auto_save', True)
        self.save_individual_files = self.config.get('save_individual_files', True)
    
    def start_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new processing session.
        
        Args:
            session_id: Optional custom session ID
            
        Returns:
            Session ID
        """
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session = SessionMetrics(
            session_id=session_id,
            start_time=datetime.now(timezone.utc).isoformat()
        )
        self.session_start_time = time.time()
        
        self.logger.info(f"Started processing session: {session_id}")
        return session_id
    
    def end_session(self) -> Optional[SessionMetrics]:
        """
        End the current processing session.
        
        Returns:
            Final session metrics
        """
        if not self.current_session:
            self.logger.warning("No active session to end")
            return None
        
        self.current_session.end_time = datetime.now(timezone.utc).isoformat()
        
        if self.session_start_time:
            self.current_session.total_processing_time = time.time() - self.session_start_time
        
        # Calculate aggregated metrics
        self._calculate_session_aggregates()
        
        # Auto-save if configured
        if self.auto_save:
            self.save_session_metrics(self.current_session)
        
        session = self.current_session
        self.current_session = None
        self.session_start_time = None
        
        self.logger.info(f"Ended processing session: {session.session_id}")
        return session
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.operation_timers[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """
        End timing an operation.
        
        Args:
            operation: Operation name
            
        Returns:
            Elapsed time in seconds
        """
        if operation not in self.operation_timers:
            self.logger.warning(f"Timer for operation '{operation}' was not started")
            return 0.0
        
        elapsed = time.time() - self.operation_timers[operation]
        del self.operation_timers[operation]
        return elapsed
    
    def create_file_metrics(self, file_path: str) -> ProcessingMetrics:
        """
        Create a new ProcessingMetrics instance for a file.
        
        Args:
            file_path: Path to the file being processed
            
        Returns:
            New ProcessingMetrics instance
        """
        return ProcessingMetrics(
            file_path=file_path,
            processing_time=0.0,
            total_segments=0,
            segments_modified=0,
            configuration_used=self.config.copy()
        )
    
    def add_file_metrics(self, metrics: ProcessingMetrics) -> None:
        """
        Add file metrics to the current session.
        
        Args:
            metrics: ProcessingMetrics to add
        """
        if not self.current_session:
            self.logger.warning("No active session - creating default session")
            self.start_session()
        
        self.current_session.file_metrics.append(metrics)
        self.current_session.total_files_processed += 1
        
        # Update session counters
        if metrics.errors_encountered:
            self.current_session.failed_files += 1
            self.current_session.session_errors.extend(metrics.errors_encountered)
        else:
            self.current_session.successful_files += 1
        
        # Save individual file metrics if configured
        if self.save_individual_files:
            self.save_file_metrics(metrics)
        
        self.logger.debug(f"Added metrics for file: {metrics.file_path}")
    
    def update_correction_count(self, metrics: ProcessingMetrics, correction_type: str, count: int = 1) -> None:
        """
        Update correction count for a specific type.
        
        Args:
            metrics: ProcessingMetrics to update
            correction_type: Type of correction applied
            count: Number of corrections (default: 1)
        """
        if correction_type not in metrics.corrections_applied:
            metrics.corrections_applied[correction_type] = 0
        metrics.corrections_applied[correction_type] += count
    
    def calculate_quality_metrics(self, metrics: ProcessingMetrics) -> None:
        """
        Calculate quality metrics from confidence scores.
        
        Args:
            metrics: ProcessingMetrics to update
        """
        if metrics.confidence_scores:
            metrics.average_confidence = sum(metrics.confidence_scores) / len(metrics.confidence_scores)
        else:
            metrics.average_confidence = 0.0
    
    def generate_processing_report(self, metrics: ProcessingMetrics) -> Dict[str, Any]:
        """
        Generate a comprehensive processing report for a file.
        
        Args:
            metrics: ProcessingMetrics to report on
            
        Returns:
            Dictionary containing the report
        """
        total_corrections = sum(metrics.corrections_applied.values())
        modification_rate = (metrics.segments_modified / metrics.total_segments * 100) if metrics.total_segments > 0 else 0
        
        report = {
            'file_summary': {
                'file_path': metrics.file_path,
                'processing_time': f"{metrics.processing_time:.2f}s",
                'total_segments': metrics.total_segments,
                'segments_modified': metrics.segments_modified,
                'modification_rate': f"{modification_rate:.1f}%"
            },
            'text_statistics': {
                'original_words': metrics.original_word_count,
                'processed_words': metrics.processed_word_count,
                'word_change': metrics.processed_word_count - metrics.original_word_count,
                'original_chars': metrics.original_char_count,
                'processed_chars': metrics.processed_char_count,
                'char_change': metrics.processed_char_count - metrics.original_char_count
            },
            'corrections': {
                'total_corrections': total_corrections,
                'by_type': metrics.corrections_applied.copy(),
                'average_per_segment': total_corrections / metrics.total_segments if metrics.total_segments > 0 else 0
            },
            'quality_metrics': {
                'average_confidence': f"{metrics.average_confidence:.3f}",
                'flagged_segments': metrics.flagged_segments,
                'flag_rate': f"{(metrics.flagged_segments / metrics.total_segments * 100):.1f}%" if metrics.total_segments > 0 else "0%",
                'timestamp_integrity': metrics.timestamp_integrity_verified
            },
            'performance': {
                'total_time': f"{metrics.processing_time:.2f}s",
                'parsing_time': f"{metrics.parsing_time:.2f}s",
                'normalization_time': f"{metrics.normalization_time:.2f}s",
                'correction_time': f"{metrics.correction_time:.2f}s",
                'validation_time': f"{metrics.validation_time:.2f}s",
                'segments_per_second': f"{(metrics.total_segments / metrics.processing_time):.1f}" if metrics.processing_time > 0 else "N/A"
            },
            'issues': {
                'errors': metrics.errors_encountered,
                'warnings': metrics.warnings_encountered,
                'error_count': len(metrics.errors_encountered),
                'warning_count': len(metrics.warnings_encountered)
            }
        }
        
        return report
    
    def generate_session_report(self, session: Optional[SessionMetrics] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive session report.
        
        Args:
            session: SessionMetrics to report on (defaults to current session)
            
        Returns:
            Dictionary containing the session report
        """
        if session is None:
            session = self.current_session
        
        if not session:
            return {'error': 'No session data available'}
        
        success_rate = (session.successful_files / session.total_files_processed * 100) if session.total_files_processed > 0 else 0
        
        # Aggregate statistics from all files
        total_segments = sum(fm.total_segments for fm in session.file_metrics)
        total_modified = sum(fm.segments_modified for fm in session.file_metrics)
        all_corrections = {}
        for fm in session.file_metrics:
            for correction_type, count in fm.corrections_applied.items():
                all_corrections[correction_type] = all_corrections.get(correction_type, 0) + count
        
        avg_confidence = 0.0
        if session.file_metrics:
            confidences = [fm.average_confidence for fm in session.file_metrics if fm.average_confidence > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        report = {
            'session_summary': {
                'session_id': session.session_id,
                'start_time': session.start_time,
                'end_time': session.end_time,
                'duration': f"{session.total_processing_time:.2f}s",
                'files_processed': session.total_files_processed,
                'success_rate': f"{success_rate:.1f}%"
            },
            'processing_statistics': {
                'successful_files': session.successful_files,
                'failed_files': session.failed_files,
                'total_segments': total_segments,
                'total_modified_segments': total_modified,
                'total_corrections': sum(all_corrections.values()),
                'average_confidence': f"{avg_confidence:.3f}"
            },
            'performance': {
                'total_processing_time': f"{session.total_processing_time:.2f}s",
                'average_time_per_file': f"{(session.total_processing_time / session.total_files_processed):.2f}s" if session.total_files_processed > 0 else "N/A",
                'segments_per_second': f"{(total_segments / session.total_processing_time):.1f}" if session.total_processing_time > 0 else "N/A"
            },
            'corrections_by_type': all_corrections,
            'session_errors': session.session_errors,
            'file_results': [
                {
                    'file': fm.file_path,
                    'segments': fm.total_segments,
                    'modified': fm.segments_modified,
                    'time': f"{fm.processing_time:.2f}s",
                    'confidence': f"{fm.average_confidence:.3f}",
                    'status': 'failed' if fm.errors_encountered else 'success'
                }
                for fm in session.file_metrics
            ]
        }
        
        return report
    
    def save_file_metrics(self, metrics: ProcessingMetrics) -> Path:
        """
        Save individual file metrics to disk.
        
        Args:
            metrics: ProcessingMetrics to save
            
        Returns:
            Path to saved file
        """
        # Create filename from file path and timestamp
        file_stem = Path(metrics.file_path).stem
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{file_stem}_metrics_{timestamp}.json"
        
        filepath = self.metrics_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(metrics), f, indent=2, ensure_ascii=False)
        
        self.logger.debug(f"Saved file metrics to: {filepath}")
        return filepath
    
    def save_session_metrics(self, session: SessionMetrics) -> Path:
        """
        Save session metrics to disk.
        
        Args:
            session: SessionMetrics to save
            
        Returns:
            Path to saved file
        """
        filename = f"{session.session_id}_metrics.json"
        filepath = self.metrics_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(session), f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved session metrics to: {filepath}")
        return filepath
    
    def load_session_metrics(self, session_id: str) -> Optional[SessionMetrics]:
        """
        Load session metrics from disk.
        
        Args:
            session_id: Session ID to load
            
        Returns:
            SessionMetrics if found, None otherwise
        """
        filename = f"{session_id}_metrics.json"
        filepath = self.metrics_dir / filename
        
        if not filepath.exists():
            self.logger.error(f"Session metrics file not found: {filepath}")
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert back to SessionMetrics
            session = SessionMetrics(**data)
            self.logger.info(f"Loaded session metrics: {session_id}")
            return session
            
        except Exception as e:
            self.logger.error(f"Error loading session metrics: {e}")
            return None
    
    def _calculate_session_aggregates(self) -> None:
        """Calculate aggregated session metrics."""
        if not self.current_session:
            return
        
        self.current_session.total_segments_processed = sum(
            fm.total_segments for fm in self.current_session.file_metrics
        )
        
        self.current_session.total_corrections_applied = sum(
            sum(fm.corrections_applied.values()) for fm in self.current_session.file_metrics
        )