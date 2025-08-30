"""
Pipeline Integration Adapter for Expert Review Queue System.

This module provides non-blocking integration points between the existing
SanskritPostProcessor pipeline and the new ExpertReviewQueue system,
enabling automatic routing of complex cases to expert review without
blocking normal processing flow.

Story 3.2.1: Expert Review Queue System - Task 4
Integration with existing processing pipeline is non-blocking.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json

from utils.srt_parser import SRTSegment
from .expert_review_queue import ExpertReviewQueue, ReviewTicket, TicketPriority, ExpertiseType


class ComplexityThreshold(Enum):
    """Thresholds for determining when cases require expert review."""
    LOW = 0.3
    MEDIUM = 0.6
    HIGH = 0.8
    CRITICAL = 0.9


@dataclass
class ComplexityAnalysis:
    """Results of analyzing segment complexity for expert review routing."""
    complexity_score: float
    requires_expert_review: bool
    complexity_factors: List[str] = field(default_factory=list)
    suggested_expertise: List[ExpertiseType] = field(default_factory=list)
    priority: TicketPriority = TicketPriority.MEDIUM
    reasoning: str = ""


@dataclass 
class IntegrationMetrics:
    """Metrics for pipeline integration performance."""
    segments_analyzed: int = 0
    segments_sent_to_queue: int = 0
    complexity_analysis_time: float = 0.0
    queue_submission_time: float = 0.0
    integration_errors: int = 0
    last_error: Optional[str] = None
    
    @property
    def expert_review_rate(self) -> float:
        """Calculate the percentage of segments sent to expert review."""
        return (self.segments_sent_to_queue / self.segments_analyzed * 100) if self.segments_analyzed > 0 else 0.0


class PipelineIntegrationAdapter:
    """
    Non-blocking integration adapter between SanskritPostProcessor and ExpertReviewQueue.
    
    This adapter analyzes processed segments for complexity indicators and automatically
    routes complex cases to the expert review queue without blocking the main pipeline.
    
    Features:
    - Complexity analysis based on processing flags and metrics
    - Asynchronous queue submission for non-blocking operation
    - Configurable thresholds for expert review routing
    - Integration metrics and monitoring
    - Graceful error handling and degradation
    """
    
    def __init__(self, expert_queue: ExpertReviewQueue, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the pipeline integration adapter.
        
        Args:
            expert_queue: The ExpertReviewQueue instance to route cases to
            config: Configuration dictionary for integration parameters
        """
        self.expert_queue = expert_queue
        self.config = config or self._get_default_config()
        self.metrics = IntegrationMetrics()
        self.logger = logging.getLogger(__name__)
        
        # Configuration validation
        self._validate_config()
        
        # Initialize async event loop for non-blocking operations
        self._loop = None
        self._submission_tasks = []  # Track background submission tasks
        
        self.logger.info(f"Pipeline integration adapter initialized with {len(self.config)} configuration parameters")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for pipeline integration."""
        return {
            # Complexity analysis thresholds
            'complexity_threshold': ComplexityThreshold.MEDIUM.value,
            'critical_threshold': ComplexityThreshold.HIGH.value,
            
            # Processing flag weights (higher = more complex)
            'flag_weights': {
                'high_semantic_drift': 0.4,
                'significant_change': 0.3,
                'academic_quality_below_threshold': 0.5,
                'low_quality_score': 0.3,
                'low_compliance_score': 0.4,
                'iast_non_compliant': 0.3,
                'multiple_corrections': 0.2,
                'rare_terms_detected': 0.3,
                'scriptural_content': 0.2,
                'complex_sanskrit': 0.4
            },
            
            # Metrics-based complexity indicators
            'metrics_thresholds': {
                'min_corrections_for_complexity': 3,
                'min_processing_time_ms': 100,
                'max_confidence_for_review': 0.7,
                'min_semantic_drift': 0.2
            },
            
            # Expert assignment preferences based on complexity type
            'expertise_mapping': {
                'iast_non_compliant': [ExpertiseType.SANSKRIT_LINGUISTICS],
                'scriptural_content': [ExpertiseType.SCRIPTURAL_ANALYSIS],
                'complex_sanskrit': [ExpertiseType.SANSKRIT_LINGUISTICS, ExpertiseType.CONTEXTUAL_ANALYSIS],
                'academic_quality_below_threshold': [ExpertiseType.ACADEMIC_REVIEW],
                'high_semantic_drift': [ExpertiseType.CONTEXTUAL_ANALYSIS]
            },
            
            # Performance and reliability settings
            'enable_async_submission': True,
            'max_concurrent_submissions': 5,
            'submission_timeout_seconds': 10.0,
            'enable_fallback_on_queue_failure': True,
            'log_all_complexity_analysis': False
        }
    
    def _validate_config(self):
        """Validate configuration parameters."""
        required_keys = ['complexity_threshold', 'flag_weights', 'metrics_thresholds']
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        # Validate threshold ranges
        threshold = self.config['complexity_threshold']
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Complexity threshold must be between 0.0 and 1.0, got {threshold}")
    
    def analyze_segment_complexity(self, segment: SRTSegment, processing_context: Dict[str, Any]) -> ComplexityAnalysis:
        """
        Analyze a processed segment to determine if it requires expert review.
        
        Args:
            segment: The processed SRT segment
            processing_context: Context from processing including metrics, flags, corrections
            
        Returns:
            ComplexityAnalysis with routing decision and recommendations
        """
        start_time = time.perf_counter()
        
        try:
            complexity_score = 0.0
            complexity_factors = []
            suggested_expertise = []
            
            # Analyze processing flags
            processing_flags = processing_context.get('processing_flags', [])
            for flag in processing_flags:
                if flag in self.config['flag_weights']:
                    weight = self.config['flag_weights'][flag]
                    complexity_score += weight
                    complexity_factors.append(f"processing_flag_{flag}")
                    
                    # Map flags to expertise types
                    if flag in self.config['expertise_mapping']:
                        suggested_expertise.extend(self.config['expertise_mapping'][flag])
            
            # Analyze correction metrics
            corrections_applied = processing_context.get('corrections_applied', [])
            if len(corrections_applied) >= self.config['metrics_thresholds']['min_corrections_for_complexity']:
                complexity_score += 0.2
                complexity_factors.append("multiple_corrections_applied")
            
            # Analyze processing time (indicating complex processing)
            processing_time_ms = processing_context.get('processing_time_ms', 0)
            if processing_time_ms >= self.config['metrics_thresholds']['min_processing_time_ms']:
                complexity_score += 0.1
                complexity_factors.append("high_processing_time")
            
            # Analyze confidence scores
            if hasattr(segment, 'academic_validation'):
                academic_data = segment.academic_validation
                
                # Low quality scores indicate complexity
                if academic_data.get('quality_score', 1.0) < self.config['metrics_thresholds']['max_confidence_for_review']:
                    complexity_score += 0.3
                    complexity_factors.append("low_quality_confidence")
                    suggested_expertise.append(ExpertiseType.ACADEMIC_REVIEW)
                
                # IAST compliance issues
                if academic_data.get('iast_compliance', 1.0) < 0.8:
                    complexity_score += 0.3
                    complexity_factors.append("iast_compliance_issues")
                    suggested_expertise.append(ExpertiseType.SANSKRIT_LINGUISTICS)
            
            # Analyze text characteristics
            text_analysis = self._analyze_text_complexity(segment.text)
            complexity_score += text_analysis['complexity_contribution']
            complexity_factors.extend(text_analysis['factors'])
            suggested_expertise.extend(text_analysis['suggested_expertise'])
            
            # Normalize complexity score to 0-1 range
            complexity_score = min(complexity_score, 1.0)
            
            # Determine if expert review is required
            requires_review = complexity_score >= self.config['complexity_threshold']
            
            # Determine priority based on complexity
            if complexity_score >= ComplexityThreshold.CRITICAL.value:
                priority = TicketPriority.URGENT
            elif complexity_score >= ComplexityThreshold.HIGH.value:
                priority = TicketPriority.HIGH
            elif complexity_score >= ComplexityThreshold.MEDIUM.value:
                priority = TicketPriority.MEDIUM
            else:
                priority = TicketPriority.LOW
            
            # Remove duplicate expertise suggestions
            suggested_expertise = list(set(suggested_expertise))
            
            # Create reasoning for the decision
            reasoning = self._generate_complexity_reasoning(complexity_score, complexity_factors, requires_review)
            
            analysis = ComplexityAnalysis(
                complexity_score=complexity_score,
                requires_expert_review=requires_review,
                complexity_factors=complexity_factors,
                suggested_expertise=suggested_expertise or [ExpertiseType.GENERAL_LINGUISTICS],
                priority=priority,
                reasoning=reasoning
            )
            
            # Update metrics
            analysis_time = time.perf_counter() - start_time
            self.metrics.complexity_analysis_time += analysis_time
            self.metrics.segments_analyzed += 1
            
            # Log complexity analysis if configured
            if self.config.get('log_all_complexity_analysis', False) or requires_review:
                self.logger.debug(f"Complexity analysis: score={complexity_score:.3f}, "
                                f"requires_review={requires_review}, factors={complexity_factors}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing segment complexity: {e}")
            self.metrics.integration_errors += 1
            self.metrics.last_error = str(e)
            
            # Return conservative analysis on error
            return ComplexityAnalysis(
                complexity_score=self.config['complexity_threshold'] + 0.1,
                requires_expert_review=True,
                complexity_factors=["analysis_error"],
                suggested_expertise=[ExpertiseType.GENERAL_LINGUISTICS],
                priority=TicketPriority.MEDIUM,
                reasoning=f"Error during complexity analysis: {str(e)}"
            )
    
    def _analyze_text_complexity(self, text: str) -> Dict[str, Any]:
        """Analyze text-based complexity indicators."""
        complexity_contribution = 0.0
        factors = []
        suggested_expertise = []
        
        # Check for Sanskrit/Hindi characters
        sanskrit_chars = sum(1 for char in text if '\u0900' <= char <= '\u097F')
        if sanskrit_chars > 0:
            complexity_contribution += min(sanskrit_chars / len(text), 0.2)
            factors.append("contains_sanskrit_characters")
            suggested_expertise.append(ExpertiseType.SANSKRIT_LINGUISTICS)
        
        # Check for IAST transliteration markers
        iast_markers = ['ā', 'ī', 'ū', 'ṛ', 'ḷ', 'ē', 'ō', 'ṃ', 'ḥ', 'ṅ', 'ñ', 'ṇ', 'ṭ', 'ḍ', 'ś', 'ṣ']
        iast_count = sum(text.count(marker) for marker in iast_markers)
        if iast_count > 2:
            complexity_contribution += min(iast_count / len(text) * 2, 0.2)
            factors.append("contains_iast_transliteration")
            suggested_expertise.append(ExpertiseType.SANSKRIT_LINGUISTICS)
        
        # Check for scriptural keywords
        scriptural_keywords = ['verse', 'chapter', 'gita', 'upanishad', 'sutra', 'mantra', 'sloka']
        scripture_matches = sum(1 for keyword in scriptural_keywords if keyword.lower() in text.lower())
        if scripture_matches > 0:
            complexity_contribution += min(scripture_matches * 0.1, 0.2)
            factors.append("contains_scriptural_references")
            suggested_expertise.append(ExpertiseType.SCRIPTURAL_ANALYSIS)
        
        # Check text length (very long segments may be complex)
        if len(text) > 200:
            complexity_contribution += 0.1
            factors.append("long_text_segment")
        
        return {
            'complexity_contribution': complexity_contribution,
            'factors': factors,
            'suggested_expertise': suggested_expertise
        }
    
    def _generate_complexity_reasoning(self, score: float, factors: List[str], requires_review: bool) -> str:
        """Generate human-readable reasoning for complexity analysis."""
        if requires_review:
            reason = f"Segment requires expert review (complexity score: {score:.3f}). "
        else:
            reason = f"Segment processed normally (complexity score: {score:.3f}). "
        
        if factors:
            top_factors = factors[:3]  # Show top 3 factors
            reason += f"Key factors: {', '.join(top_factors)}."
        
        return reason
    
    def integrate_with_pipeline(self, segment: SRTSegment, processing_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main integration point for the SanskritPostProcessor pipeline.
        
        This method should be called after segment processing to determine if the
        segment needs expert review and route it accordingly without blocking.
        
        Args:
            segment: The processed SRT segment
            processing_context: Context from processing including metrics, flags, corrections
            
        Returns:
            Dictionary with integration results and metadata
        """
        try:
            # Analyze complexity
            analysis = self.analyze_segment_complexity(segment, processing_context)
            
            integration_result = {
                'complexity_analysis': analysis,
                'expert_review_submitted': False,
                'submission_task_id': None,
                'integration_time_ms': 0.0
            }
            
            # If expert review is required, submit to queue (non-blocking)
            if analysis.requires_expert_review:
                if self.config['enable_async_submission']:
                    # Asynchronous submission
                    task_id = self._submit_to_queue_async(segment, analysis, processing_context)
                    integration_result['submission_task_id'] = task_id
                    integration_result['expert_review_submitted'] = True
                else:
                    # Synchronous submission (blocking)
                    success = self._submit_to_queue_sync(segment, analysis, processing_context)
                    integration_result['expert_review_submitted'] = success
            
            return integration_result
            
        except Exception as e:
            self.logger.error(f"Error in pipeline integration: {e}")
            self.metrics.integration_errors += 1
            self.metrics.last_error = str(e)
            
            return {
                'complexity_analysis': None,
                'expert_review_submitted': False,
                'error': str(e),
                'integration_time_ms': 0.0
            }
    
    def _submit_to_queue_async(self, segment: SRTSegment, analysis: ComplexityAnalysis, context: Dict[str, Any]) -> str:
        """Submit segment to expert review queue asynchronously (non-blocking)."""
        task_id = f"pipeline_{segment.index}_{int(time.time() * 1000)}"
        
        try:
            # Get or create event loop
            if self._loop is None or self._loop.is_closed():
                try:
                    self._loop = asyncio.get_event_loop()
                except RuntimeError:
                    # No event loop in current thread, create a new one
                    self._loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(self._loop)
            
            # Create coroutine for submission
            submission_coro = self._submit_ticket_async(segment, analysis, context, task_id)
            
            # Schedule for execution without blocking
            task = self._loop.create_task(submission_coro)
            self._submission_tasks.append(task)
            
            # Clean up completed tasks (prevent memory leak)
            self._submission_tasks = [t for t in self._submission_tasks if not t.done()]
            
            return task_id
            
        except Exception as e:
            self.logger.error(f"Error setting up async submission: {e}")
            self.metrics.integration_errors += 1
            return f"error_{task_id}"
    
    async def _submit_ticket_async(self, segment: SRTSegment, analysis: ComplexityAnalysis, 
                                 context: Dict[str, Any], task_id: str):
        """Async coroutine for submitting ticket to expert review queue."""
        start_time = time.perf_counter()
        
        try:
            # Create review ticket
            ticket = self._create_review_ticket(segment, analysis, context, task_id)
            
            # Submit to queue with timeout
            submission_timeout = self.config.get('submission_timeout_seconds', 10.0)
            await asyncio.wait_for(
                self.expert_queue.submit_ticket(ticket),
                timeout=submission_timeout
            )
            
            # Update metrics
            submission_time = time.perf_counter() - start_time
            self.metrics.queue_submission_time += submission_time
            self.metrics.segments_sent_to_queue += 1
            
            self.logger.debug(f"Successfully submitted ticket {task_id} to expert review queue")
            
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout submitting ticket {task_id} to expert review queue")
            self.metrics.integration_errors += 1
            self.metrics.last_error = f"Submission timeout for ticket {task_id}"
            
        except Exception as e:
            self.logger.error(f"Error submitting ticket {task_id} to expert review queue: {e}")
            self.metrics.integration_errors += 1
            self.metrics.last_error = str(e)
    
    def _submit_to_queue_sync(self, segment: SRTSegment, analysis: ComplexityAnalysis, context: Dict[str, Any]) -> bool:
        """Submit segment to expert review queue synchronously (blocking)."""
        start_time = time.perf_counter()
        
        try:
            task_id = f"pipeline_sync_{segment.index}_{int(time.time() * 1000)}"
            ticket = self._create_review_ticket(segment, analysis, context, task_id)
            
            # Submit synchronously using asyncio.run
            asyncio.run(self.expert_queue.submit_ticket(ticket))
            
            # Update metrics
            submission_time = time.perf_counter() - start_time
            self.metrics.queue_submission_time += submission_time
            self.metrics.segments_sent_to_queue += 1
            
            self.logger.debug(f"Successfully submitted ticket {task_id} to expert review queue (sync)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error submitting ticket to expert review queue (sync): {e}")
            self.metrics.integration_errors += 1
            self.metrics.last_error = str(e)
            return False
    
    def _create_review_ticket(self, segment: SRTSegment, analysis: ComplexityAnalysis, 
                            context: Dict[str, Any], task_id: str) -> ReviewTicket:
        """Create a ReviewTicket from segment analysis."""
        # Prepare ticket description
        description = f"Complex segment requiring expert review (complexity: {analysis.complexity_score:.3f})\n"
        description += f"Factors: {', '.join(analysis.complexity_factors)}\n"
        description += f"Reasoning: {analysis.reasoning}"
        
        # Prepare context data
        ticket_context = {
            'segment_index': segment.index,
            'segment_start_time': segment.start_time,
            'segment_end_time': segment.end_time,
            'original_text': getattr(segment, 'raw_text', segment.text),
            'processed_text': segment.text,
            'processing_flags': context.get('processing_flags', []),
            'corrections_applied': context.get('corrections_applied', []),
            'complexity_analysis': {
                'score': analysis.complexity_score,
                'factors': analysis.complexity_factors,
                'reasoning': analysis.reasoning
            },
            'academic_validation': getattr(segment, 'academic_validation', {}),
            'pipeline_task_id': task_id
        }
        
        # Create and return ticket
        return ReviewTicket(
            ticket_id=task_id,
            segment_text=segment.text,
            issue_description=description,
            context=ticket_context,
            required_expertise=analysis.suggested_expertise,
            priority=analysis.priority,
            metadata={
                'source': 'pipeline_integration',
                'complexity_score': analysis.complexity_score,
                'auto_routed': True,
                'processing_time_ms': context.get('processing_time_ms', 0)
            }
        )
    
    def get_integration_metrics(self) -> IntegrationMetrics:
        """Get current integration performance metrics."""
        return self.metrics
    
    def reset_metrics(self):
        """Reset integration metrics."""
        self.metrics = IntegrationMetrics()
        self.logger.info("Integration metrics reset")
    
    def cleanup_async_tasks(self):
        """Clean up any pending async submission tasks."""
        try:
            if self._submission_tasks:
                for task in self._submission_tasks:
                    if not task.done():
                        task.cancel()
                
                self._submission_tasks.clear()
                self.logger.debug("Cleaned up async submission tasks")
        except Exception as e:
            self.logger.error(f"Error cleaning up async tasks: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on integration adapter."""
        try:
            # Check expert queue health
            queue_health = self.expert_queue.get_queue_health()
            
            # Check async task status
            active_tasks = len([t for t in self._submission_tasks if not t.done()])
            
            # Calculate error rate
            error_rate = (self.metrics.integration_errors / self.metrics.segments_analyzed * 100) if self.metrics.segments_analyzed > 0 else 0
            
            health_status = {
                'integration_healthy': True,
                'queue_healthy': queue_health.get('is_healthy', False),
                'active_async_tasks': active_tasks,
                'segments_analyzed': self.metrics.segments_analyzed,
                'expert_review_rate': self.metrics.expert_review_rate,
                'error_rate_percent': error_rate,
                'last_error': self.metrics.last_error,
                'config_valid': True  # Validated during init
            }
            
            # Determine overall health
            if error_rate > 10 or not queue_health.get('is_healthy', False):
                health_status['integration_healthy'] = False
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Error during health check: {e}")
            return {
                'integration_healthy': False,
                'error': str(e),
                'queue_healthy': False
            }


def create_pipeline_integration_adapter(expert_queue: ExpertReviewQueue, 
                                      config_path: Optional[Path] = None) -> PipelineIntegrationAdapter:
    """
    Factory function to create a PipelineIntegrationAdapter with optional configuration.
    
    Args:
        expert_queue: ExpertReviewQueue instance to integrate with
        config_path: Optional path to JSON configuration file
        
    Returns:
        Configured PipelineIntegrationAdapter instance
    """
    config = None
    
    if config_path and config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load integration config from {config_path}: {e}")
            logging.info("Using default configuration")
    
    return PipelineIntegrationAdapter(expert_queue, config)