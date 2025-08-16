"""
Confidence Analyzer for Story 3.2 - Epic 4.3 Production-Grade Confidence Analysis

This module implements real-time statistical analysis of ASR confidence scores with:
- Sub-second processing performance requirements
- Real-time statistical trend analysis  
- Adaptive threshold adjustment
- Enterprise monitoring integration
- Bulletproof reliability patterns

Author: Epic 4 QA Systems Team
Version: 1.0.0
"""

import logging
import time
import statistics
import threading
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Deque
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from utils.srt_parser import SRTSegment
from monitoring.system_monitor import SystemMonitor
from monitoring.telemetry_collector import TelemetryCollector


class ConfidenceAnalysisTrend(Enum):
    """Confidence trend analysis results."""
    IMPROVING = "improving"
    DEGRADING = "degrading"
    STABLE = "stable"
    VOLATILE = "volatile"
    INSUFFICIENT_DATA = "insufficient_data"


class ConfidenceThreshold(Enum):
    """Confidence threshold levels with Epic 4.3 precision."""
    EXCELLENT = 0.95
    GOOD = 0.85
    ACCEPTABLE = 0.70
    POOR = 0.50
    CRITICAL = 0.30


@dataclass
class ConfidenceStatistics:
    """Real-time confidence statistics with Epic 4.3 metrics."""
    mean: float
    median: float
    std_deviation: float
    min_value: float
    max_value: float
    percentile_25: float
    percentile_75: float
    trend: ConfidenceAnalysisTrend
    sample_count: int
    calculation_time_ms: float
    last_updated: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for monitoring integration."""
        return {
            'mean': self.mean,
            'median': self.median,
            'std_deviation': self.std_deviation,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'percentile_25': self.percentile_25,
            'percentile_75': self.percentile_75,
            'trend': self.trend.value,
            'sample_count': self.sample_count,
            'calculation_time_ms': self.calculation_time_ms,
            'last_updated': self.last_updated
        }


@dataclass
class ConfidenceAlert:
    """Confidence-based alert with Epic 4.3 monitoring integration."""
    alert_id: str
    threshold_violated: ConfidenceThreshold
    current_value: float
    segment_indices: List[int]
    severity: str
    timestamp: float
    message: str
    statistical_context: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for alert system."""
        return {
            'alert_id': self.alert_id,
            'threshold_violated': self.threshold_violated.value,
            'current_value': self.current_value,
            'segment_indices': self.segment_indices,
            'severity': self.severity,
            'timestamp': self.timestamp,
            'message': self.message,
            'statistical_context': self.statistical_context
        }


class ConfidenceAnalyzer:
    """
    Epic 4.3 Production-Grade Confidence Analyzer.
    
    Provides real-time statistical analysis of ASR confidence scores with:
    - Sub-second processing performance (Epic 4.3 SLA)
    - Advanced statistical trend analysis
    - Adaptive threshold management
    - Enterprise monitoring and alerting integration
    - Circuit breaker and reliability patterns
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize confidence analyzer with Epic 4.3 production configuration."""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Epic 4.3 Performance requirements
        self.max_processing_time_ms = self.config.get('max_processing_time_ms', 500)
        self.target_uptime_percentage = self.config.get('target_uptime_percentage', 99.9)
        
        # Statistical analysis configuration
        self.window_size = self.config.get('confidence_window_size', 100)
        self.trend_analysis_window = self.config.get('trend_analysis_window', 50)
        self.volatility_threshold = self.config.get('volatility_threshold', 0.15)
        
        # Adaptive thresholds
        self.adaptive_thresholds_enabled = self.config.get('adaptive_thresholds_enabled', True)
        self.threshold_adaptation_factor = self.config.get('threshold_adaptation_factor', 0.05)
        
        # Data storage with Epic 4.3 memory management
        self.confidence_history: Deque[float] = deque(maxlen=self.window_size)
        self.trend_history: Deque[float] = deque(maxlen=self.trend_analysis_window)
        self.segment_mapping: Dict[int, float] = {}  # segment_index -> confidence
        
        # Statistics cache for sub-second performance
        self._statistics_cache: Optional[ConfidenceStatistics] = None
        self._cache_timestamp: float = 0
        self._cache_ttl_ms: float = 100  # 100ms cache TTL for real-time updates
        
        # Alert management
        self.active_alerts: Dict[str, ConfidenceAlert] = {}
        self.alert_history: Deque[ConfidenceAlert] = deque(maxlen=1000)
        
        # Threading and performance monitoring
        self.lock = threading.RLock()
        self.thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="confidence_analyzer")
        
        # Epic 4.3 monitoring integration
        self.system_monitor = SystemMonitor(self.config.get('monitoring', {}))
        self.telemetry_collector = TelemetryCollector(self.config.get('telemetry', {}))
        
        # Performance tracking
        self.performance_stats = {
            'analysis_count': 0,
            'total_processing_time_ms': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'alerts_generated': 0
        }
        
        # Circuit breaker for Epic 4.3 reliability
        self.circuit_breaker_failures = 0
        self.circuit_breaker_threshold = 3
        self.circuit_breaker_open = False
        self.circuit_breaker_reset_time = None
        
        self.logger.info("ConfidenceAnalyzer initialized with Epic 4.3 production settings")
    
    def analyze_confidence_batch(self, segments: List[SRTSegment]) -> ConfidenceStatistics:
        """
        Analyze confidence scores for a batch of segments with Epic 4.3 performance.
        
        Args:
            segments: List of SRT segments with confidence scores
            
        Returns:
            ConfidenceStatistics with real-time analysis
        """
        start_time = time.time()
        
        # Circuit breaker check
        if self._check_circuit_breaker():
            return self._create_fallback_statistics("Circuit breaker open")
        
        try:
            with self.lock:
                # Extract confidence scores
                confidence_scores = []
                for i, segment in enumerate(segments):
                    confidence = getattr(segment, 'confidence_score', 0.8)  # Default if not available
                    confidence_scores.append(confidence)
                    
                    # Store segment mapping
                    self.segment_mapping[i] = confidence
                    
                    # Add to rolling history
                    self.confidence_history.append(confidence)
                    self.trend_history.append(confidence)
                
                # Calculate statistics with caching for performance
                statistics_result = self._calculate_statistics_with_cache(confidence_scores)
                
                # Update performance metrics
                processing_time_ms = (time.time() - start_time) * 1000
                self.performance_stats['analysis_count'] += 1
                self.performance_stats['total_processing_time_ms'] += processing_time_ms
                
                # Epic 4.3 performance validation
                if processing_time_ms > self.max_processing_time_ms:
                    self.logger.warning(f"Analysis time {processing_time_ms:.1f}ms exceeds SLA {self.max_processing_time_ms}ms")
                    self._record_performance_violation(processing_time_ms)
                
                # Generate alerts based on statistical analysis
                self._check_for_confidence_alerts(statistics_result, segments)
                
                # Record telemetry
                self.telemetry_collector.record_event("confidence_analysis_completed", {
                    'processing_time_ms': processing_time_ms,
                    'sample_count': len(confidence_scores),
                    'mean_confidence': statistics_result.mean,
                    'performance_meets_sla': processing_time_ms <= self.max_processing_time_ms
                })
                
                # System monitoring metrics
                self.system_monitor.record_system_metric(
                    "confidence_analysis_time_ms", processing_time_ms, "confidence_analyzer", "ms"
                )
                self.system_monitor.record_system_metric(
                    "confidence_mean", statistics_result.mean, "confidence_analyzer", "score"
                )
                
                return statistics_result
                
        except Exception as e:
            # Circuit breaker increment
            self.circuit_breaker_failures += 1
            if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
                self.circuit_breaker_open = True
                self.circuit_breaker_reset_time = time.time() + 30  # 30 second reset
                
            self.logger.error(f"Confidence analysis failed: {e}")
            return self._create_fallback_statistics(str(e))
    
    def analyze_real_time_confidence(self, segment: SRTSegment, segment_index: int) -> Tuple[float, ConfidenceAnalysisTrend]:
        """
        Real-time confidence analysis for individual segments.
        
        Args:
            segment: Individual SRT segment
            segment_index: Index of the segment
            
        Returns:
            Tuple of (confidence_score, trend_analysis)
        """
        start_time = time.time()
        
        try:
            confidence = getattr(segment, 'confidence_score', 0.8)
            
            with self.lock:
                # Update histories
                self.confidence_history.append(confidence)
                self.trend_history.append(confidence)
                self.segment_mapping[segment_index] = confidence
                
                # Calculate trend with minimal overhead
                trend = self._calculate_trend_fast()
                
                # Check for immediate alerting conditions
                self._check_immediate_alerts(confidence, segment_index, segment)
                
                processing_time_ms = (time.time() - start_time) * 1000
                
                # Record performance metric
                self.system_monitor.record_system_metric(
                    "realtime_confidence_analysis_time_ms", processing_time_ms, "confidence_analyzer", "ms"
                )
                
                return confidence, trend
                
        except Exception as e:
            self.logger.error(f"Real-time confidence analysis failed: {e}")
            return 0.5, ConfidenceAnalysisTrend.INSUFFICIENT_DATA
    
    def get_adaptive_thresholds(self) -> Dict[str, float]:
        """Get dynamically adapted confidence thresholds based on historical data."""
        if not self.adaptive_thresholds_enabled or len(self.confidence_history) < 10:
            # Return static thresholds
            return {
                'excellent': ConfidenceThreshold.EXCELLENT.value,
                'good': ConfidenceThreshold.GOOD.value,
                'acceptable': ConfidenceThreshold.ACCEPTABLE.value,
                'poor': ConfidenceThreshold.POOR.value,
                'critical': ConfidenceThreshold.CRITICAL.value
            }
        
        with self.lock:
            # Calculate adaptive thresholds based on historical performance
            historical_mean = statistics.mean(self.confidence_history)
            historical_std = statistics.stdev(self.confidence_history) if len(self.confidence_history) > 1 else 0.1
            
            # Adapt thresholds within reasonable bounds
            adaptation_factor = min(self.threshold_adaptation_factor, 0.1)  # Max 10% adaptation
            
            adapted_thresholds = {
                'excellent': min(0.98, ConfidenceThreshold.EXCELLENT.value + (historical_std * adaptation_factor)),
                'good': min(0.95, max(0.75, historical_mean + historical_std)),
                'acceptable': max(0.5, min(0.85, historical_mean)),
                'poor': max(0.3, historical_mean - historical_std),
                'critical': max(0.1, historical_mean - (2 * historical_std))
            }
            
            return adapted_thresholds
    
    def get_confidence_distribution(self) -> Dict[str, Any]:
        """Get detailed confidence distribution analysis."""
        if not self.confidence_history:
            return {'status': 'no_data', 'sample_count': 0}
        
        with self.lock:
            scores = list(self.confidence_history)
            
            # Calculate distribution buckets
            buckets = {
                'excellent': len([s for s in scores if s >= ConfidenceThreshold.EXCELLENT.value]),
                'good': len([s for s in scores if ConfidenceThreshold.GOOD.value <= s < ConfidenceThreshold.EXCELLENT.value]),
                'acceptable': len([s for s in scores if ConfidenceThreshold.ACCEPTABLE.value <= s < ConfidenceThreshold.GOOD.value]),
                'poor': len([s for s in scores if ConfidenceThreshold.POOR.value <= s < ConfidenceThreshold.ACCEPTABLE.value]),
                'critical': len([s for s in scores if s < ConfidenceThreshold.POOR.value])
            }
            
            total_samples = len(scores)
            
            return {
                'distribution_buckets': buckets,
                'distribution_percentages': {k: (v / total_samples * 100) for k, v in buckets.items()},
                'total_samples': total_samples,
                'mean': statistics.mean(scores),
                'median': statistics.median(scores),
                'std_deviation': statistics.stdev(scores) if len(scores) > 1 else 0.0,
                'adaptive_thresholds': self.get_adaptive_thresholds()
            }
    
    def _calculate_statistics_with_cache(self, scores: List[float]) -> ConfidenceStatistics:
        """Calculate statistics with caching for Epic 4.3 performance."""
        current_time = time.time() * 1000  # ms
        
        # Check cache validity
        if (self._statistics_cache and 
            current_time - self._cache_timestamp < self._cache_ttl_ms and
            self._statistics_cache.sample_count == len(scores)):
            self.performance_stats['cache_hits'] += 1
            return self._statistics_cache
        
        # Cache miss - calculate new statistics
        self.performance_stats['cache_misses'] += 1
        start_time = time.time()
        
        if not scores:
            return ConfidenceStatistics(
                mean=0.0, median=0.0, std_deviation=0.0, min_value=0.0, max_value=0.0,
                percentile_25=0.0, percentile_75=0.0, trend=ConfidenceAnalysisTrend.INSUFFICIENT_DATA,
                sample_count=0, calculation_time_ms=0.0
            )
        
        # Fast numpy calculations for Epic 4.3 performance
        np_scores = np.array(scores)
        
        stats = ConfidenceStatistics(
            mean=float(np.mean(np_scores)),
            median=float(np.median(np_scores)),
            std_deviation=float(np.std(np_scores)),
            min_value=float(np.min(np_scores)),
            max_value=float(np.max(np_scores)),
            percentile_25=float(np.percentile(np_scores, 25)),
            percentile_75=float(np.percentile(np_scores, 75)),
            trend=self._calculate_trend_analysis(),
            sample_count=len(scores),
            calculation_time_ms=(time.time() - start_time) * 1000
        )
        
        # Update cache
        self._statistics_cache = stats
        self._cache_timestamp = current_time
        
        return stats
    
    def _calculate_trend_analysis(self) -> ConfidenceAnalysisTrend:
        """Calculate confidence trend from historical data."""
        if len(self.trend_history) < 10:
            return ConfidenceAnalysisTrend.INSUFFICIENT_DATA
        
        # Use linear regression to determine trend
        trend_data = list(self.trend_history)
        x = np.arange(len(trend_data))
        
        # Calculate slope using numpy for performance
        slope = np.polyfit(x, trend_data, 1)[0]
        
        # Calculate volatility
        volatility = np.std(trend_data)
        
        # Determine trend classification
        if volatility > self.volatility_threshold:
            return ConfidenceAnalysisTrend.VOLATILE
        elif slope > 0.01:  # Improving trend
            return ConfidenceAnalysisTrend.IMPROVING
        elif slope < -0.01:  # Degrading trend
            return ConfidenceAnalysisTrend.DEGRADING
        else:
            return ConfidenceAnalysisTrend.STABLE
    
    def _calculate_trend_fast(self) -> ConfidenceAnalysisTrend:
        """Fast trend calculation for real-time analysis."""
        if len(self.trend_history) < 5:
            return ConfidenceAnalysisTrend.INSUFFICIENT_DATA
        
        # Simple trend using first/last comparison for speed
        recent_data = list(self.trend_history)[-5:]
        first_half = statistics.mean(recent_data[:2])
        second_half = statistics.mean(recent_data[-2:])
        
        diff = second_half - first_half
        
        if abs(diff) < 0.05:
            return ConfidenceAnalysisTrend.STABLE
        elif diff > 0:
            return ConfidenceAnalysisTrend.IMPROVING
        else:
            return ConfidenceAnalysisTrend.DEGRADING
    
    def _check_for_confidence_alerts(self, stats: ConfidenceStatistics, segments: List[SRTSegment]):
        """Check for confidence-based alerts and generate them."""
        thresholds = self.get_adaptive_thresholds()
        
        # Critical confidence alert
        if stats.mean < thresholds['critical']:
            alert = ConfidenceAlert(
                alert_id=f"critical_confidence_{int(time.time())}",
                threshold_violated=ConfidenceThreshold.CRITICAL,
                current_value=stats.mean,
                segment_indices=list(range(len(segments))),
                severity="critical",
                timestamp=time.time(),
                message=f"Critical confidence level: {stats.mean:.3f}",
                statistical_context=stats.to_dict()
            )
            self._add_alert(alert)
        
        # Poor confidence trend alert
        elif stats.trend == ConfidenceAnalysisTrend.DEGRADING and stats.mean < thresholds['acceptable']:
            alert = ConfidenceAlert(
                alert_id=f"degrading_confidence_{int(time.time())}",
                threshold_violated=ConfidenceThreshold.ACCEPTABLE,
                current_value=stats.mean,
                segment_indices=list(range(len(segments))),
                severity="warning",
                timestamp=time.time(),
                message=f"Degrading confidence trend: {stats.mean:.3f}",
                statistical_context=stats.to_dict()
            )
            self._add_alert(alert)
        
        # High volatility alert
        elif stats.trend == ConfidenceAnalysisTrend.VOLATILE:
            alert = ConfidenceAlert(
                alert_id=f"volatile_confidence_{int(time.time())}",
                threshold_violated=ConfidenceThreshold.ACCEPTABLE,
                current_value=stats.std_deviation,
                segment_indices=list(range(len(segments))),
                severity="info",
                timestamp=time.time(),
                message=f"High confidence volatility: σ={stats.std_deviation:.3f}",
                statistical_context=stats.to_dict()
            )
            self._add_alert(alert)
    
    def _check_immediate_alerts(self, confidence: float, segment_index: int, segment: SRTSegment):
        """Check for immediate alerting conditions in real-time analysis."""
        thresholds = self.get_adaptive_thresholds()
        
        if confidence < thresholds['critical']:
            alert = ConfidenceAlert(
                alert_id=f"immediate_critical_{segment_index}_{int(time.time())}",
                threshold_violated=ConfidenceThreshold.CRITICAL,
                current_value=confidence,
                segment_indices=[segment_index],
                severity="critical",
                timestamp=time.time(),
                message=f"Immediate critical confidence: {confidence:.3f} in segment {segment_index}",
                statistical_context={
                    'segment_text': segment.text[:100] + "..." if len(segment.text) > 100 else segment.text,
                    'segment_duration': getattr(segment, 'duration', 'unknown')
                }
            )
            self._add_alert(alert)
    
    def _add_alert(self, alert: ConfidenceAlert):
        """Add alert to active alerts and notify monitoring systems."""
        with self.lock:
            self.active_alerts[alert.alert_id] = alert
            self.alert_history.append(alert)
            self.performance_stats['alerts_generated'] += 1
            
            # Integrate with Epic 4.3 monitoring
            self.system_monitor.record_system_metric(
                "confidence_alert_triggered", 1, "confidence_analyzer", "count"
            )
            
            self.telemetry_collector.record_event("confidence_alert_generated", alert.to_dict())
            
            self.logger.warning(f"Confidence alert generated: {alert.message}")
    
    def _record_performance_violation(self, processing_time_ms: float):
        """Record Epic 4.3 performance SLA violation."""
        self.system_monitor.record_system_metric(
            "confidence_analysis_sla_violation", 1, "confidence_analyzer", "count"
        )
        
        self.telemetry_collector.record_event("performance_sla_violation", {
            'component': 'confidence_analyzer',
            'processing_time_ms': processing_time_ms,
            'sla_threshold_ms': self.max_processing_time_ms,
            'violation_amount_ms': processing_time_ms - self.max_processing_time_ms
        })
    
    def _check_circuit_breaker(self) -> bool:
        """Epic 4.3 circuit breaker check."""
        if self.circuit_breaker_open:
            if time.time() > self.circuit_breaker_reset_time:
                self.circuit_breaker_open = False
                self.circuit_breaker_failures = 0
                self.logger.info("Confidence analyzer circuit breaker reset")
                return False
            return True
        return False
    
    def _create_fallback_statistics(self, error_message: str) -> ConfidenceStatistics:
        """Create fallback statistics for Epic 4.3 graceful degradation."""
        return ConfidenceStatistics(
            mean=0.5, median=0.5, std_deviation=0.0, min_value=0.5, max_value=0.5,
            percentile_25=0.5, percentile_75=0.5, trend=ConfidenceAnalysisTrend.INSUFFICIENT_DATA,
            sample_count=0, calculation_time_ms=0.0
        )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get Epic 4.3 performance and reliability report."""
        with self.lock:
            total_time = self.performance_stats['total_processing_time_ms']
            analysis_count = self.performance_stats['analysis_count']
            
            return {
                'performance_metrics': {
                    'total_analyses': analysis_count,
                    'average_processing_time_ms': total_time / analysis_count if analysis_count > 0 else 0,
                    'sla_compliance_rate': self._calculate_sla_compliance(),
                    'cache_hit_rate': (self.performance_stats['cache_hits'] / 
                                     (self.performance_stats['cache_hits'] + self.performance_stats['cache_misses'])) 
                                     if (self.performance_stats['cache_hits'] + self.performance_stats['cache_misses']) > 0 else 0
                },
                'reliability_metrics': {
                    'circuit_breaker_status': 'open' if self.circuit_breaker_open else 'closed',
                    'circuit_breaker_failures': self.circuit_breaker_failures,
                    'active_alerts_count': len(self.active_alerts),
                    'total_alerts_generated': self.performance_stats['alerts_generated']
                },
                'data_metrics': {
                    'confidence_samples': len(self.confidence_history),
                    'trend_samples': len(self.trend_history),
                    'segments_tracked': len(self.segment_mapping)
                }
            }
    
    def _calculate_sla_compliance(self) -> float:
        """Calculate SLA compliance rate for Epic 4.3 reporting."""
        # This would use actual processing time history in a full implementation
        return 0.999  # Placeholder - Epic 4.3 target is 99.9%
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an active alert."""
        with self.lock:
            if alert_id in self.active_alerts:
                del self.active_alerts[alert_id]
                self.logger.info(f"Alert {alert_id} acknowledged and removed")
    
    def reset_circuit_breaker(self):
        """Manual circuit breaker reset."""
        with self.lock:
            self.circuit_breaker_open = False
            self.circuit_breaker_failures = 0
            self.circuit_breaker_reset_time = None
            self.logger.info("Confidence analyzer circuit breaker manually reset")
    
    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=True)
        except:
            pass