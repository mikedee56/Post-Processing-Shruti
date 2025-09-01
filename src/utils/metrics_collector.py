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
    
    def record_batch_processing(self, processing_result) -> None:
        """
        Record batch processing results for Epic 4 integration.
        
        Args:
            processing_result: BatchProcessor ProcessingResult object
        """
        try:
            # Extract metrics from batch result
            batch_metrics = {
                'batch_id': processing_result.batch_id,
                'total_files': processing_result.total_files,
                'processed_files': processing_result.processed_files,
                'failed_files': processing_result.failed_files,
                'success_rate': processing_result.success_rate,
                'throughput': processing_result.throughput,
                'processing_time': processing_result.processing_time,
                'start_time': processing_result.start_time.isoformat() if processing_result.start_time else None,
                'end_time': processing_result.end_time.isoformat() if processing_result.end_time else None,
                'errors': processing_result.errors,
                'metrics': processing_result.metrics
            }
            
            # Save batch metrics to file for Epic 4 reporting
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"batch_{processing_result.batch_id}_metrics_{timestamp}.json"
            filepath = self.metrics_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(batch_metrics, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Recorded batch processing metrics: {processing_result.batch_id}")
            self.logger.debug(f"Batch metrics saved to: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error recording batch processing metrics: {e}")

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

@dataclass
class SemanticProcessingMetrics:
    """Metrics specific to semantic processing operations (Story 3.0)."""
    
    # Basic semantic processing metrics
    terms_identified: int = 0
    terms_processed: int = 0 
    embeddings_generated: int = 0
    embeddings_cached: int = 0
    relationships_discovered: int = 0
    
    # Performance metrics
    semantic_processing_time: float = 0.0
    embedding_generation_time: float = 0.0
    similarity_search_time: float = 0.0
    cache_lookup_time: float = 0.0
    
    # Quality metrics  
    domain_classification_accuracy: float = 0.0
    semantic_confidence_scores: List[float] = field(default_factory=list)
    average_semantic_confidence: float = 0.0
    
    # Cache performance
    cache_hit_ratio: float = 0.0
    cache_misses: int = 0
    cache_hits: int = 0
    
    # Infrastructure health
    vector_db_available: bool = False
    redis_cache_available: bool = False
    pgvector_extension_active: bool = False
    
    # Resource utilization
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Error tracking
    semantic_processing_errors: List[str] = field(default_factory=list)
    fallback_mode_activated: bool = False
    
    def calculate_performance_overhead(self, base_processing_time: float) -> float:
        """Calculate the performance overhead as a percentage."""
        if base_processing_time <= 0:
            return 0.0
        return (self.semantic_processing_time / base_processing_time) * 100
    
    def meets_performance_target(self, base_processing_time: float, target_overhead: float = 5.0) -> bool:
        """Check if semantic processing meets the <5% overhead target."""
        return self.calculate_performance_overhead(base_processing_time) <= target_overhead
    
    def get_cache_efficiency_score(self) -> float:
        """Calculate cache efficiency score (0-100)."""
        if self.cache_hits + self.cache_misses == 0:
            return 0.0
        return (self.cache_hits / (self.cache_hits + self.cache_misses)) * 100


class SemanticMetricsCollector:
    """
    Specialized metrics collector for semantic processing operations (Story 3.0).
    
    Monitors performance, quality, and infrastructure health for semantic features
    while ensuring the <5% overhead requirement is maintained.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Performance targets from Story 3.0
        self.target_overhead_percent = self.config.get('semantic_overhead_target', 5.0)
        self.target_cache_hit_ratio = self.config.get('semantic_cache_hit_target', 95.0)
        
        # Current semantic metrics
        self.current_semantic_metrics: Optional[SemanticProcessingMetrics] = None
        
        # Session aggregates
        self.session_semantic_metrics: List[SemanticProcessingMetrics] = []
        
        # Performance monitoring
        self.semantic_timers: Dict[str, float] = {}
        
        # Alert thresholds
        self.overhead_alert_threshold = self.config.get('overhead_alert_threshold', 7.0)  # Alert at 7%
        self.cache_alert_threshold = self.config.get('cache_alert_threshold', 85.0)  # Alert below 85%
    
    def start_semantic_processing(self) -> SemanticProcessingMetrics:
        """Start tracking semantic processing metrics for a segment."""
        self.current_semantic_metrics = SemanticProcessingMetrics()
        
        # Check infrastructure availability
        self.current_semantic_metrics.vector_db_available = self._check_vector_db_health()
        self.current_semantic_metrics.redis_cache_available = self._check_redis_health()
        self.current_semantic_metrics.pgvector_extension_active = self._check_pgvector_extension()
        
        return self.current_semantic_metrics
    
    def end_semantic_processing(self, base_processing_time: float) -> SemanticProcessingMetrics:
        """Complete semantic processing metrics and calculate performance overhead."""
        if not self.current_semantic_metrics:
            self.logger.warning("No active semantic metrics to end")
            return SemanticProcessingMetrics()
        
        # Calculate final metrics
        if self.current_semantic_metrics.semantic_confidence_scores:
            self.current_semantic_metrics.average_semantic_confidence = sum(
                self.current_semantic_metrics.semantic_confidence_scores
            ) / len(self.current_semantic_metrics.semantic_confidence_scores)
        
        # Calculate cache hit ratio
        total_cache_operations = (self.current_semantic_metrics.cache_hits + 
                                self.current_semantic_metrics.cache_misses)
        if total_cache_operations > 0:
            self.current_semantic_metrics.cache_hit_ratio = (
                self.current_semantic_metrics.cache_hits / total_cache_operations * 100
            )
        
        # Performance overhead validation
        overhead = self.current_semantic_metrics.calculate_performance_overhead(base_processing_time)
        if overhead > self.overhead_alert_threshold:
            self.logger.warning(
                f"Semantic processing overhead ({overhead:.1f}%) exceeds alert threshold "
                f"({self.overhead_alert_threshold}%)"
            )
        
        # Cache performance validation  
        if (self.current_semantic_metrics.cache_hit_ratio < self.cache_alert_threshold and 
            total_cache_operations > 0):
            self.logger.warning(
                f"Semantic cache hit ratio ({self.current_semantic_metrics.cache_hit_ratio:.1f}%) "
                f"below alert threshold ({self.cache_alert_threshold}%)"
            )
        
        # Store in session aggregates
        self.session_semantic_metrics.append(self.current_semantic_metrics)
        
        metrics = self.current_semantic_metrics
        self.current_semantic_metrics = None
        
        return metrics
    
    def start_semantic_timer(self, operation: str) -> None:
        """Start timing a semantic operation."""
        self.semantic_timers[f"semantic_{operation}"] = time.time()
    
    def end_semantic_timer(self, operation: str) -> float:
        """End timing a semantic operation and update metrics."""
        timer_key = f"semantic_{operation}"
        if timer_key not in self.semantic_timers:
            self.logger.warning(f"Semantic timer for '{operation}' was not started")
            return 0.0
        
        elapsed = time.time() - self.semantic_timers[timer_key]
        del self.semantic_timers[timer_key]
        
        # Update current metrics if active
        if self.current_semantic_metrics:
            if operation == 'processing':
                self.current_semantic_metrics.semantic_processing_time += elapsed
            elif operation == 'embedding_generation':
                self.current_semantic_metrics.embedding_generation_time += elapsed
            elif operation == 'similarity_search':
                self.current_semantic_metrics.similarity_search_time += elapsed
            elif operation == 'cache_lookup':
                self.current_semantic_metrics.cache_lookup_time += elapsed
        
        return elapsed
    
    def record_semantic_term_identified(self, term: str, confidence: float, domain: str = "") -> None:
        """Record identification of a semantic term."""
        if not self.current_semantic_metrics:
            return
            
        self.current_semantic_metrics.terms_identified += 1
        self.current_semantic_metrics.semantic_confidence_scores.append(confidence)
        
        # Update domain classification accuracy tracking
        if domain and confidence > 0.8:  # High confidence domain classification
            self.current_semantic_metrics.domain_classification_accuracy += 1
    
    def record_semantic_term_processed(self, processing_successful: bool) -> None:
        """Record processing of a semantic term."""
        if not self.current_semantic_metrics:
            return
            
        if processing_successful:
            self.current_semantic_metrics.terms_processed += 1
        else:
            self.current_semantic_metrics.semantic_processing_errors.append(
                f"Term processing failed at {datetime.now().isoformat()}"
            )
    
    def record_embedding_operation(self, generated: bool, cached: bool) -> None:
        """Record embedding generation or cache hit."""
        if not self.current_semantic_metrics:
            return
            
        if generated:
            self.current_semantic_metrics.embeddings_generated += 1
        if cached:
            self.current_semantic_metrics.embeddings_cached += 1
    
    def record_cache_operation(self, hit: bool) -> None:
        """Record cache hit or miss."""
        if not self.current_semantic_metrics:
            return
            
        if hit:
            self.current_semantic_metrics.cache_hits += 1
        else:
            self.current_semantic_metrics.cache_misses += 1
    
    def record_relationship_discovered(self, relationship_type: str, confidence: float) -> None:
        """Record discovery of a semantic relationship."""
        if not self.current_semantic_metrics:
            return
            
        self.current_semantic_metrics.relationships_discovered += 1
        self.current_semantic_metrics.semantic_confidence_scores.append(confidence)
    
    def record_fallback_mode(self, reason: str) -> None:
        """Record when semantic processing falls back to traditional processing."""
        if not self.current_semantic_metrics:
            return
            
        self.current_semantic_metrics.fallback_mode_activated = True
        self.current_semantic_metrics.semantic_processing_errors.append(
            f"Fallback activated: {reason}"
        )
        self.logger.info(f"Semantic processing fallback activated: {reason}")
    
    def generate_semantic_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive semantic processing performance report."""
        if not self.session_semantic_metrics:
            return {
                'semantic_processing': 'No semantic processing metrics available',
                'infrastructure_status': self._get_infrastructure_status()
            }
        
        # Aggregate session metrics
        total_terms = sum(m.terms_identified for m in self.session_semantic_metrics)
        total_processed = sum(m.terms_processed for m in self.session_semantic_metrics)
        total_semantic_time = sum(m.semantic_processing_time for m in self.session_semantic_metrics)
        
        # Performance analysis
        overhead_percentages = [
            m.calculate_performance_overhead(m.semantic_processing_time + 1.0)  # Avoid division by zero
            for m in self.session_semantic_metrics if m.semantic_processing_time > 0
        ]
        
        avg_overhead = sum(overhead_percentages) / len(overhead_percentages) if overhead_percentages else 0.0
        max_overhead = max(overhead_percentages) if overhead_percentages else 0.0
        
        # Cache performance
        total_cache_ops = sum(m.cache_hits + m.cache_misses for m in self.session_semantic_metrics)
        total_cache_hits = sum(m.cache_hits for m in self.session_semantic_metrics)
        session_cache_hit_ratio = (total_cache_hits / total_cache_ops * 100) if total_cache_ops > 0 else 0.0
        
        # Quality analysis
        all_confidence_scores = []
        for m in self.session_semantic_metrics:
            all_confidence_scores.extend(m.semantic_confidence_scores)
        
        avg_semantic_confidence = (sum(all_confidence_scores) / len(all_confidence_scores) 
                                 if all_confidence_scores else 0.0)
        
        report = {
            'semantic_processing_summary': {
                'total_terms_identified': total_terms,
                'total_terms_processed': total_processed,
                'processing_success_rate': f"{(total_processed / total_terms * 100):.1f}%" if total_terms > 0 else "0%",
                'total_semantic_processing_time': f"{total_semantic_time:.3f}s",
                'average_time_per_term': f"{(total_semantic_time / total_terms):.3f}s" if total_terms > 0 else "N/A"
            },
            'performance_analysis': {
                'average_overhead_percent': f"{avg_overhead:.2f}%",
                'maximum_overhead_percent': f"{max_overhead:.2f}%",
                'meets_target_overhead': avg_overhead <= self.target_overhead_percent,
                'target_overhead': f"{self.target_overhead_percent}%",
                'performance_grade': self._calculate_performance_grade(avg_overhead)
            },
            'cache_performance': {
                'session_cache_hit_ratio': f"{session_cache_hit_ratio:.1f}%",
                'total_cache_operations': total_cache_ops,
                'total_cache_hits': total_cache_hits,
                'meets_cache_target': session_cache_hit_ratio >= self.target_cache_hit_ratio,
                'target_cache_hit_ratio': f"{self.target_cache_hit_ratio}%"
            },
            'quality_metrics': {
                'average_semantic_confidence': f"{avg_semantic_confidence:.3f}",
                'total_confidence_samples': len(all_confidence_scores),
                'total_relationships_discovered': sum(m.relationships_discovered for m in self.session_semantic_metrics)
            },
            'infrastructure_status': self._get_infrastructure_status(),
            'error_analysis': {
                'total_errors': sum(len(m.semantic_processing_errors) for m in self.session_semantic_metrics),
                'fallback_activations': sum(1 for m in self.session_semantic_metrics if m.fallback_mode_activated),
                'error_rate': f"{(sum(len(m.semantic_processing_errors) for m in self.session_semantic_metrics) / total_terms * 100):.2f}%" if total_terms > 0 else "0%"
            }
        }
        
        return report
    
    def _check_vector_db_health(self) -> bool:
        """Check if vector database is available and healthy."""
        try:
            from database.vector_database import get_vector_database_manager
            vector_db = get_vector_database_manager()
            status = vector_db.get_health_status()
            return status.get('database_connected', False) and status.get('schema_initialized', False)
        except Exception as e:
            self.logger.debug(f"Vector database health check failed: {e}")
            return False
    
    def _check_redis_health(self) -> bool:
        """Check if Redis cache is available and healthy."""
        try:
            # This will be implemented when Redis is added in future stories
            # For now, return False to indicate Redis not yet implemented
            return False
        except Exception as e:
            self.logger.debug(f"Redis health check failed: {e}")
            return False
    
    def _check_pgvector_extension(self) -> bool:
        """Check if pgvector extension is active."""
        try:
            from database.vector_database import get_vector_database_manager
            vector_db = get_vector_database_manager()
            return vector_db._pgvector_available
        except Exception as e:
            self.logger.debug(f"pgvector extension check failed: {e}")
            return False
    
    def _get_infrastructure_status(self) -> Dict[str, Any]:
        """Get current infrastructure status for reporting."""
        return {
            'vector_database_available': self._check_vector_db_health(),
            'redis_cache_available': self._check_redis_health(),
            'pgvector_extension_active': self._check_pgvector_extension(),
            'infrastructure_health_score': self._calculate_infrastructure_health_score()
        }
    
    def _calculate_infrastructure_health_score(self) -> float:
        """Calculate infrastructure health score (0-100)."""
        components = [
            self._check_vector_db_health(),
            self._check_pgvector_extension(),
            # Redis will be worth more points when implemented in future stories
        ]
        
        active_components = sum(1 for component in components if component)
        return (active_components / len(components)) * 100
    
    def _calculate_performance_grade(self, overhead_percent: float) -> str:
        """Calculate performance grade based on overhead percentage."""
        if overhead_percent <= 2.0:
            return "A+ (Excellent)"
        elif overhead_percent <= 3.0:
            return "A (Very Good)" 
        elif overhead_percent <= 5.0:
            return "B (Good - Meets Target)"
        elif overhead_percent <= 7.0:
            return "C (Fair - Above Target)"
        elif overhead_percent <= 10.0:
            return "D (Poor)"
        else:
            return "F (Unacceptable)"
