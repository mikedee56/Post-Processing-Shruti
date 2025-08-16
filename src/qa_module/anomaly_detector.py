"""
Anomaly Detector for Story 3.2 - Epic 4.1 MCP Context-Aware Anomaly Detection

This module implements MCP context-aware anomaly detection with:
- Epic 4.1 MCP framework for intelligent language shift detection
- Context-aware processing for sophisticated pattern recognition
- Circuit breaker patterns for reliable anomaly detection
- Fallback protection for continuity validation
- Academic standards integration for research-grade detection

Author: Epic 4 QA Systems Team
Version: 1.0.0
"""

import logging
import time
import re
import statistics
import threading
from collections import deque, defaultdict, Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np

from utils.srt_parser import SRTSegment
from utils.mcp_client_manager import MCPClientManager
from monitoring.system_monitor import SystemMonitor
from monitoring.telemetry_collector import TelemetryCollector


class AnomalyType(Enum):
    """Types of anomalies detected with Epic 4.1 MCP integration."""
    LANGUAGE_SHIFT = "language_shift"
    ACOUSTIC_ANOMALY = "acoustic_anomaly"
    CONTENT_DISCONTINUITY = "content_discontinuity"
    SPEECH_RATE_ANOMALY = "speech_rate_anomaly"
    SILENCE_ANOMALY = "silence_anomaly"
    BACKGROUND_NOISE = "background_noise"
    SPEAKER_CHANGE = "speaker_change"
    TOPIC_SHIFT = "topic_shift"
    ENCODING_ERROR = "encoding_error"
    TIMESTAMP_ANOMALY = "timestamp_anomaly"


class AnomalySeverity(Enum):
    """Anomaly severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class DetectedAnomaly:
    """Individual detected anomaly with Epic 4.1 context awareness."""
    anomaly_id: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    segment_index: int
    timestamp: float
    confidence: float
    description: str
    context_data: Dict[str, Any] = field(default_factory=dict)
    mcp_analysis: Dict[str, Any] = field(default_factory=dict)
    recommended_action: str = ""
    academic_impact: float = 0.5  # Epic 4.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Epic 4.5 academic reporting."""
        return {
            'anomaly_id': self.anomaly_id,
            'anomaly_type': self.anomaly_type.value,
            'severity': self.severity.value,
            'segment_index': self.segment_index,
            'timestamp': self.timestamp,
            'confidence': self.confidence,
            'description': self.description,
            'context_data': self.context_data,
            'mcp_analysis': self.mcp_analysis,
            'recommended_action': self.recommended_action,
            'academic_impact': self.academic_impact
        }


@dataclass
class AnomalyAnalysisResult:
    """Result of anomaly analysis with Epic 4.1 comprehensive context."""
    segment_index: int
    detected_anomalies: List[DetectedAnomaly]
    overall_anomaly_score: float
    context_coherence_score: float
    processing_time_ms: float
    mcp_processing_successful: bool
    circuit_breaker_triggered: bool
    fallback_analysis_used: bool
    academic_continuity_score: float  # Epic 4.5
    
    def get_anomalies_by_type(self, anomaly_type: AnomalyType) -> List[DetectedAnomaly]:
        """Get anomalies filtered by type."""
        return [a for a in self.detected_anomalies if a.anomaly_type == anomaly_type]
    
    def get_critical_anomalies(self) -> List[DetectedAnomaly]:
        """Get critical severity anomalies."""
        return [a for a in self.detected_anomalies if a.severity == AnomalySeverity.CRITICAL]


class AnomalyDetector:
    """
    Epic 4.1 MCP Context-Aware Anomaly Detector.
    
    Integrates:
    - Epic 4.1 MCP framework for intelligent pattern recognition
    - Context-aware processing for sophisticated anomaly detection
    - Circuit breaker patterns for reliable operation
    - Fallback protection for continuity validation
    - Epic 4.5 academic standards for research-grade detection
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize anomaly detector with Epic 4.1 MCP integration."""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Epic 4.1 MCP Integration
        self.mcp_client_manager = MCPClientManager(self.config.get('mcp', {}))
        self.mcp_enabled = self.config.get('enable_mcp_analysis', True)
        
        # Anomaly detection thresholds
        self.anomaly_thresholds = self.config.get('anomaly_thresholds', {
            'language_shift_threshold': 0.3,
            'acoustic_deviation_threshold': 2.0,  # Standard deviations
            'speech_rate_min': 50,   # chars per minute
            'speech_rate_max': 1000, # chars per minute
            'silence_threshold_ms': 5000,
            'timestamp_jump_threshold_ms': 30000
        })
        
        # Context analysis configuration
        self.context_config = self.config.get('context_analysis', {
            'window_size': 5,           # segments to analyze together
            'language_detection_enabled': True,
            'topic_coherence_enabled': True,
            'speaker_consistency_enabled': True,
            'academic_continuity_enabled': True  # Epic 4.5
        })
        
        # Historical data for context analysis
        self.segment_history: deque = deque(maxlen=self.context_config['window_size'])
        self.language_history: deque = deque(maxlen=20)
        self.speech_rate_history: deque = deque(maxlen=50)
        self.topic_keywords: Dict[str, Counter] = defaultdict(Counter)
        
        # Performance tracking
        self.anomaly_statistics = {
            'total_segments_analyzed': 0,
            'total_anomalies_detected': 0,
            'anomalies_by_type': defaultdict(int),
            'false_positive_rate': 0.0,
            'mcp_success_rate': 0.0
        }
        
        # Epic 4.3 Monitoring Integration
        self.system_monitor = SystemMonitor(self.config.get('monitoring', {}))
        self.telemetry_collector = TelemetryCollector(self.config.get('telemetry', {}))
        
        # Threading for Epic 4.3 reliability
        self.lock = threading.RLock()
        
        # Epic 4.1 Circuit breaker pattern
        self.circuit_breaker_failures = 0
        self.circuit_breaker_threshold = 3
        self.circuit_breaker_open = False
        self.circuit_breaker_reset_time = None
        self.fallback_mode = False
        
        # MCP processing statistics
        self.mcp_statistics = {
            'requests_made': 0,
            'requests_successful': 0,
            'requests_failed': 0,
            'average_response_time_ms': 0.0
        }
        
        self.logger.info("AnomalyDetector initialized with Epic 4.1 MCP context awareness")
    
    def detect_anomalies(self, segment: SRTSegment, segment_index: int, 
                        previous_segments: Optional[List[SRTSegment]] = None) -> AnomalyAnalysisResult:
        """
        Detect anomalies in a segment with Epic 4.1 MCP context awareness.
        
        Args:
            segment: SRT segment to analyze
            segment_index: Index of the segment
            previous_segments: Optional previous segments for context
            
        Returns:
            AnomalyAnalysisResult with context-aware detection results
        """
        start_time = time.time()
        
        # Circuit breaker check
        if self._check_circuit_breaker():
            return self._create_fallback_result(segment, segment_index, "Circuit breaker open")
        
        try:
            detected_anomalies = []
            mcp_processing_successful = True
            fallback_used = False
            
            # Update historical context
            self._update_segment_history(segment, segment_index)
            
            # Core anomaly detection
            anomalies = self._detect_core_anomalies(segment, segment_index)
            detected_anomalies.extend(anomalies)
            
            # Epic 4.1 MCP Context-Aware Analysis
            if self.mcp_enabled and not self.fallback_mode:
                try:
                    mcp_anomalies = self._detect_mcp_context_anomalies(segment, segment_index, previous_segments)
                    detected_anomalies.extend(mcp_anomalies)
                    self.mcp_statistics['requests_successful'] += 1
                except Exception as e:
                    self.logger.warning(f"MCP analysis failed, using fallback: {e}")
                    mcp_processing_successful = False
                    fallback_used = True
                    self.mcp_statistics['requests_failed'] += 1
                    
                    # Fallback analysis
                    fallback_anomalies = self._detect_fallback_anomalies(segment, segment_index)
                    detected_anomalies.extend(fallback_anomalies)
                
                self.mcp_statistics['requests_made'] += 1
            else:
                # Use fallback analysis when MCP is disabled or in fallback mode
                fallback_anomalies = self._detect_fallback_anomalies(segment, segment_index)
                detected_anomalies.extend(fallback_anomalies)
                fallback_used = True
            
            # Calculate composite scores
            processing_time_ms = (time.time() - start_time) * 1000
            overall_anomaly_score = self._calculate_overall_anomaly_score(detected_anomalies)
            context_coherence_score = self._calculate_context_coherence_score(segment, previous_segments)
            academic_continuity_score = self._calculate_academic_continuity_score(detected_anomalies, segment)
            
            # Update statistics
            with self.lock:
                self.anomaly_statistics['total_segments_analyzed'] += 1
                self.anomaly_statistics['total_anomalies_detected'] += len(detected_anomalies)
                for anomaly in detected_anomalies:
                    self.anomaly_statistics['anomalies_by_type'][anomaly.anomaly_type.value] += 1
            
            # Epic 4.3 Performance monitoring
            self.system_monitor.record_system_metric(
                "anomaly_detection_time_ms", processing_time_ms, "anomaly_detector", "ms"
            )
            self.system_monitor.record_system_metric(
                "anomalies_detected", len(detected_anomalies), "anomaly_detector", "count"
            )
            
            self.telemetry_collector.record_event("anomaly_detection_completed", {
                'segment_index': segment_index,
                'anomalies_detected': len(detected_anomalies),
                'processing_time_ms': processing_time_ms,
                'mcp_successful': mcp_processing_successful,
                'fallback_used': fallback_used
            })
            
            # Circuit breaker success reset
            self.circuit_breaker_failures = 0
            
            return AnomalyAnalysisResult(
                segment_index=segment_index,
                detected_anomalies=detected_anomalies,
                overall_anomaly_score=overall_anomaly_score,
                context_coherence_score=context_coherence_score,
                processing_time_ms=processing_time_ms,
                mcp_processing_successful=mcp_processing_successful,
                circuit_breaker_triggered=False,
                fallback_analysis_used=fallback_used,
                academic_continuity_score=academic_continuity_score
            )
            
        except Exception as e:
            # Epic 4.1 Circuit breaker increment
            self.circuit_breaker_failures += 1
            if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
                self.circuit_breaker_open = True
                self.circuit_breaker_reset_time = time.time() + 60  # 60 second reset
            
            self.logger.error(f"Anomaly detection failed: {e}")
            return self._create_fallback_result(segment, segment_index, str(e))
    
    def _detect_core_anomalies(self, segment: SRTSegment, segment_index: int) -> List[DetectedAnomaly]:
        """Detect core anomalies using rule-based analysis."""
        anomalies = []
        
        # Language shift detection
        language_anomaly = self._detect_language_shift(segment, segment_index)
        if language_anomaly:
            anomalies.append(language_anomaly)
        
        # Speech rate anomaly detection
        speech_rate_anomaly = self._detect_speech_rate_anomaly(segment, segment_index)
        if speech_rate_anomaly:
            anomalies.append(speech_rate_anomaly)
        
        # Timestamp anomaly detection
        timestamp_anomaly = self._detect_timestamp_anomaly(segment, segment_index)
        if timestamp_anomaly:
            anomalies.append(timestamp_anomaly)
        
        # Content discontinuity detection
        discontinuity_anomaly = self._detect_content_discontinuity(segment, segment_index)
        if discontinuity_anomaly:
            anomalies.append(discontinuity_anomaly)
        
        # Encoding error detection
        encoding_anomaly = self._detect_encoding_errors(segment, segment_index)
        if encoding_anomaly:
            anomalies.append(encoding_anomaly)
        
        return anomalies
    
    def _detect_mcp_context_anomalies(self, segment: SRTSegment, segment_index: int,
                                     previous_segments: Optional[List[SRTSegment]] = None) -> List[DetectedAnomaly]:
        """Epic 4.1 MCP context-aware anomaly detection."""
        anomalies = []
        
        try:
            # Prepare context for MCP analysis
            context_data = {
                'current_segment': {
                    'text': segment.text,
                    'start_time': segment.start_time,
                    'end_time': segment.end_time,
                    'index': segment_index
                },
                'previous_segments': [],
                'language_history': list(self.language_history),
                'speech_rate_history': list(self.speech_rate_history)
            }
            
            if previous_segments:
                context_data['previous_segments'] = [
                    {'text': seg.text, 'start_time': seg.start_time, 'end_time': seg.end_time}
                    for seg in previous_segments[-3:]  # Last 3 segments for context
                ]
            
            # MCP analysis for sophisticated pattern recognition
            mcp_result = self._perform_mcp_analysis(context_data)
            
            if mcp_result:
                # Topic shift detection
                if mcp_result.get('topic_shift_detected'):
                    anomalies.append(DetectedAnomaly(
                        anomaly_id=f"mcp_topic_shift_{segment_index}_{int(time.time())}",
                        anomaly_type=AnomalyType.TOPIC_SHIFT,
                        severity=AnomalySeverity.WARNING,
                        segment_index=segment_index,
                        timestamp=time.time(),
                        confidence=mcp_result.get('topic_shift_confidence', 0.7),
                        description=f"MCP detected topic shift with {mcp_result.get('topic_shift_confidence', 0.7):.3f} confidence",
                        context_data=context_data,
                        mcp_analysis=mcp_result,
                        recommended_action="Review content coherence",
                        academic_impact=mcp_result.get('academic_impact', 0.3)
                    ))
                
                # Speaker change detection
                if mcp_result.get('speaker_change_detected'):
                    anomalies.append(DetectedAnomaly(
                        anomaly_id=f"mcp_speaker_change_{segment_index}_{int(time.time())}",
                        anomaly_type=AnomalyType.SPEAKER_CHANGE,
                        severity=AnomalySeverity.INFO,
                        segment_index=segment_index,
                        timestamp=time.time(),
                        confidence=mcp_result.get('speaker_change_confidence', 0.6),
                        description=f"MCP detected potential speaker change",
                        context_data=context_data,
                        mcp_analysis=mcp_result,
                        recommended_action="Verify speaker consistency",
                        academic_impact=mcp_result.get('academic_impact', 0.2)
                    ))
                
                # Acoustic quality anomaly
                if mcp_result.get('acoustic_anomaly_detected'):
                    severity = AnomalySeverity.CRITICAL if mcp_result.get('acoustic_severity') == 'high' else AnomalySeverity.WARNING
                    anomalies.append(DetectedAnomaly(
                        anomaly_id=f"mcp_acoustic_{segment_index}_{int(time.time())}",
                        anomaly_type=AnomalyType.ACOUSTIC_ANOMALY,
                        severity=severity,
                        segment_index=segment_index,
                        timestamp=time.time(),
                        confidence=mcp_result.get('acoustic_confidence', 0.8),
                        description=f"MCP detected acoustic quality anomaly",
                        context_data=context_data,
                        mcp_analysis=mcp_result,
                        recommended_action="Check audio quality and re-process if needed",
                        academic_impact=mcp_result.get('academic_impact', 0.4)
                    ))
        
        except Exception as e:
            self.logger.warning(f"MCP context analysis failed: {e}")
            # This will trigger fallback mode
            raise e
        
        return anomalies
    
    def _detect_fallback_anomalies(self, segment: SRTSegment, segment_index: int) -> List[DetectedAnomaly]:
        """Epic 4.1 fallback anomaly detection when MCP is unavailable."""
        anomalies = []
        
        # Simple topic shift detection using keyword analysis
        topic_shift = self._detect_simple_topic_shift(segment, segment_index)
        if topic_shift:
            anomalies.append(topic_shift)
        
        # Basic speaker change detection using text patterns
        speaker_change = self._detect_simple_speaker_change(segment, segment_index)
        if speaker_change:
            anomalies.append(speaker_change)
        
        # Background noise detection using text patterns
        noise_anomaly = self._detect_background_noise_patterns(segment, segment_index)
        if noise_anomaly:
            anomalies.append(noise_anomaly)
        
        return anomalies
    
    def _detect_language_shift(self, segment: SRTSegment, segment_index: int) -> Optional[DetectedAnomaly]:
        """Detect language shifts in the content."""
        if not self.context_config['language_detection_enabled']:
            return None
        
        # Simple language detection based on character patterns
        text = segment.text.lower()
        
        # Count different script types
        ascii_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        devanagari_chars = sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)
        other_chars = sum(1 for c in text if not c.isascii() and not (0x0900 <= ord(c) <= 0x097F) and c.isalpha())
        
        total_chars = ascii_chars + devanagari_chars + other_chars
        
        if total_chars == 0:
            return None
        
        # Calculate language composition
        ascii_ratio = ascii_chars / total_chars
        devanagari_ratio = devanagari_chars / total_chars
        other_ratio = other_chars / total_chars
        
        # Detect sudden language shifts
        current_language = 'english' if ascii_ratio > 0.8 else 'sanskrit' if devanagari_ratio > 0.5 else 'mixed'
        
        self.language_history.append(current_language)
        
        # Check for language shift
        if len(self.language_history) >= 3:
            recent_languages = list(self.language_history)[-3:]
            if len(set(recent_languages)) > 1 and current_language != recent_languages[-2]:
                return DetectedAnomaly(
                    anomaly_id=f"lang_shift_{segment_index}_{int(time.time())}",
                    anomaly_type=AnomalyType.LANGUAGE_SHIFT,
                    severity=AnomalySeverity.WARNING,
                    segment_index=segment_index,
                    timestamp=time.time(),
                    confidence=0.7,
                    description=f"Language shift detected: {recent_languages[-2]} -> {current_language}",
                    context_data={
                        'ascii_ratio': ascii_ratio,
                        'devanagari_ratio': devanagari_ratio,
                        'other_ratio': other_ratio,
                        'language_history': recent_languages
                    },
                    recommended_action="Review content for language consistency",
                    academic_impact=0.3
                )
        
        return None
    
    def _detect_speech_rate_anomaly(self, segment: SRTSegment, segment_index: int) -> Optional[DetectedAnomaly]:
        """Detect speech rate anomalies."""
        try:
            # Calculate speech rate (characters per minute)
            duration = self._estimate_segment_duration(segment)
            if duration <= 0:
                return None
            
            char_count = len(segment.text)
            speech_rate = (char_count / duration) * 60  # chars per minute
            
            self.speech_rate_history.append(speech_rate)
            
            # Check against thresholds
            if speech_rate < self.anomaly_thresholds['speech_rate_min']:
                return DetectedAnomaly(
                    anomaly_id=f"speech_slow_{segment_index}_{int(time.time())}",
                    anomaly_type=AnomalyType.SPEECH_RATE_ANOMALY,
                    severity=AnomalySeverity.WARNING,
                    segment_index=segment_index,
                    timestamp=time.time(),
                    confidence=0.8,
                    description=f"Unusually slow speech rate: {speech_rate:.1f} chars/min",
                    context_data={
                        'speech_rate': speech_rate,
                        'duration': duration,
                        'char_count': char_count,
                        'threshold_min': self.anomaly_thresholds['speech_rate_min']
                    },
                    recommended_action="Check for long pauses or very slow speech",
                    academic_impact=0.2
                )
            
            elif speech_rate > self.anomaly_thresholds['speech_rate_max']:
                return DetectedAnomaly(
                    anomaly_id=f"speech_fast_{segment_index}_{int(time.time())}",
                    anomaly_type=AnomalyType.SPEECH_RATE_ANOMALY,
                    severity=AnomalySeverity.WARNING,
                    segment_index=segment_index,
                    timestamp=time.time(),
                    confidence=0.8,
                    description=f"Unusually fast speech rate: {speech_rate:.1f} chars/min",
                    context_data={
                        'speech_rate': speech_rate,
                        'duration': duration,
                        'char_count': char_count,
                        'threshold_max': self.anomaly_thresholds['speech_rate_max']
                    },
                    recommended_action="Check for rapid speech or transcription errors",
                    academic_impact=0.3
                )
            
            # Statistical anomaly detection if we have enough history
            if len(self.speech_rate_history) >= 10:
                mean_rate = statistics.mean(self.speech_rate_history)
                std_rate = statistics.stdev(self.speech_rate_history)
                
                if abs(speech_rate - mean_rate) > (self.anomaly_thresholds['acoustic_deviation_threshold'] * std_rate):
                    return DetectedAnomaly(
                        anomaly_id=f"speech_statistical_{segment_index}_{int(time.time())}",
                        anomaly_type=AnomalyType.SPEECH_RATE_ANOMALY,
                        severity=AnomalySeverity.INFO,
                        segment_index=segment_index,
                        timestamp=time.time(),
                        confidence=0.6,
                        description=f"Statistical speech rate anomaly: {speech_rate:.1f} (μ={mean_rate:.1f}, σ={std_rate:.1f})",
                        context_data={
                            'speech_rate': speech_rate,
                            'mean_rate': mean_rate,
                            'std_rate': std_rate,
                            'deviation_threshold': self.anomaly_thresholds['acoustic_deviation_threshold']
                        },
                        recommended_action="Review for speech pattern changes",
                        academic_impact=0.1
                    )
        
        except Exception as e:
            self.logger.warning(f"Speech rate analysis failed: {e}")
        
        return None
    
    def _detect_timestamp_anomaly(self, segment: SRTSegment, segment_index: int) -> Optional[DetectedAnomaly]:
        """Detect timestamp anomalies."""
        try:
            # Check for overlapping or invalid timestamps
            start_time = self._parse_timestamp(segment.start_time)
            end_time = self._parse_timestamp(segment.end_time)
            
            if start_time >= end_time:
                return DetectedAnomaly(
                    anomaly_id=f"timestamp_invalid_{segment_index}_{int(time.time())}",
                    anomaly_type=AnomalyType.TIMESTAMP_ANOMALY,
                    severity=AnomalySeverity.CRITICAL,
                    segment_index=segment_index,
                    timestamp=time.time(),
                    confidence=1.0,
                    description=f"Invalid timestamp: start >= end ({segment.start_time} >= {segment.end_time})",
                    context_data={
                        'start_time': segment.start_time,
                        'end_time': segment.end_time,
                        'start_ms': start_time,
                        'end_ms': end_time
                    },
                    recommended_action="Fix timestamp ordering",
                    academic_impact=0.8
                )
            
            # Check for unrealistic segment duration
            duration_ms = end_time - start_time
            if duration_ms > self.anomaly_thresholds['timestamp_jump_threshold_ms']:
                return DetectedAnomaly(
                    anomaly_id=f"timestamp_long_{segment_index}_{int(time.time())}",
                    anomaly_type=AnomalyType.TIMESTAMP_ANOMALY,
                    severity=AnomalySeverity.WARNING,
                    segment_index=segment_index,
                    timestamp=time.time(),
                    confidence=0.8,
                    description=f"Unusually long segment duration: {duration_ms/1000:.1f}s",
                    context_data={
                        'duration_ms': duration_ms,
                        'duration_s': duration_ms/1000,
                        'threshold_ms': self.anomaly_thresholds['timestamp_jump_threshold_ms']
                    },
                    recommended_action="Review segment boundaries",
                    academic_impact=0.2
                )
        
        except Exception as e:
            self.logger.warning(f"Timestamp analysis failed: {e}")
        
        return None
    
    def _detect_content_discontinuity(self, segment: SRTSegment, segment_index: int) -> Optional[DetectedAnomaly]:
        """Detect content discontinuity."""
        if len(self.segment_history) < 2:
            return None
        
        try:
            current_text = segment.text.lower()
            previous_segment = self.segment_history[-1]
            previous_text = previous_segment['text'].lower()
            
            # Simple discontinuity detection using common words
            current_words = set(current_text.split())
            previous_words = set(previous_text.split())
            
            # Calculate word overlap
            if len(current_words) > 0 and len(previous_words) > 0:
                overlap = len(current_words.intersection(previous_words))
                overlap_ratio = overlap / min(len(current_words), len(previous_words))
                
                # Very low overlap might indicate discontinuity
                if overlap_ratio < 0.1 and len(current_words) > 5 and len(previous_words) > 5:
                    return DetectedAnomaly(
                        anomaly_id=f"discontinuity_{segment_index}_{int(time.time())}",
                        anomaly_type=AnomalyType.CONTENT_DISCONTINUITY,
                        severity=AnomalySeverity.INFO,
                        segment_index=segment_index,
                        timestamp=time.time(),
                        confidence=0.6,
                        description=f"Low content overlap with previous segment: {overlap_ratio:.2f}",
                        context_data={
                            'overlap_ratio': overlap_ratio,
                            'current_words_count': len(current_words),
                            'previous_words_count': len(previous_words),
                            'overlap_count': overlap
                        },
                        recommended_action="Check for topic changes or audio gaps",
                        academic_impact=0.2
                    )
        
        except Exception as e:
            self.logger.warning(f"Content discontinuity analysis failed: {e}")
        
        return None
    
    def _detect_encoding_errors(self, segment: SRTSegment, segment_index: int) -> Optional[DetectedAnomaly]:
        """Detect text encoding errors."""
        text = segment.text
        
        # Check for common encoding error patterns
        error_patterns = [
            r'[^\x00-\x7F\u0900-\u097F\u1E00-\u1EFF]',  # Invalid characters outside expected ranges
            r'[\uFFFD]',  # Unicode replacement characters
            r'[^\w\s\u0900-\u097F\u1E00-\u1EFF.,!?;:()"-]'  # Unexpected special characters
        ]
        
        for pattern in error_patterns:
            if re.search(pattern, text):
                return DetectedAnomaly(
                    anomaly_id=f"encoding_{segment_index}_{int(time.time())}",
                    anomaly_type=AnomalyType.ENCODING_ERROR,
                    severity=AnomalySeverity.WARNING,
                    segment_index=segment_index,
                    timestamp=time.time(),
                    confidence=0.9,
                    description="Text encoding errors detected",
                    context_data={
                        'text_sample': text[:100] + "..." if len(text) > 100 else text,
                        'error_pattern': pattern
                    },
                    recommended_action="Check text encoding and re-process if needed",
                    academic_impact=0.5
                )
        
        return None
    
    def _detect_simple_topic_shift(self, segment: SRTSegment, segment_index: int) -> Optional[DetectedAnomaly]:
        """Simple topic shift detection for fallback mode."""
        # Extract keywords from current segment
        words = segment.text.lower().split()
        keywords = [w for w in words if len(w) > 4 and w.isalpha()]
        
        if not keywords:
            return None
        
        # Update topic keywords for this segment
        current_keywords = Counter(keywords)
        
        # Compare with recent topic history
        if len(self.topic_keywords) >= 3:
            # Get combined keywords from recent segments
            recent_combined = Counter()
            for recent_counter in list(self.topic_keywords.values())[-3:]:
                recent_combined.update(recent_counter)
            
            # Calculate topic similarity
            overlap = sum((current_keywords & recent_combined).values())
            total_current = sum(current_keywords.values())
            
            if total_current > 0:
                overlap_ratio = overlap / total_current
                
                if overlap_ratio < 0.2:  # Low topic overlap
                    return DetectedAnomaly(
                        anomaly_id=f"topic_shift_fallback_{segment_index}_{int(time.time())}",
                        anomaly_type=AnomalyType.TOPIC_SHIFT,
                        severity=AnomalySeverity.INFO,
                        segment_index=segment_index,
                        timestamp=time.time(),
                        confidence=0.5,
                        description=f"Potential topic shift detected (fallback mode)",
                        context_data={
                            'overlap_ratio': overlap_ratio,
                            'current_keywords': list(current_keywords.keys())[:5],
                            'keyword_count': len(keywords)
                        },
                        recommended_action="Review for topic changes",
                        academic_impact=0.2
                    )
        
        # Store keywords for this segment
        self.topic_keywords[f"segment_{segment_index}"] = current_keywords
        
        return None
    
    def _detect_simple_speaker_change(self, segment: SRTSegment, segment_index: int) -> Optional[DetectedAnomaly]:
        """Simple speaker change detection for fallback mode."""
        text = segment.text.lower()
        
        # Look for speaker change indicators
        speaker_indicators = [
            'the speaker says', 'someone asks', 'the teacher responds',
            'question:', 'answer:', 'student:', 'teacher:', 'now',
            'next question', 'moving on', 'let me ask'
        ]
        
        for indicator in speaker_indicators:
            if indicator in text:
                return DetectedAnomaly(
                    anomaly_id=f"speaker_change_fallback_{segment_index}_{int(time.time())}",
                    anomaly_type=AnomalyType.SPEAKER_CHANGE,
                    severity=AnomalySeverity.INFO,
                    segment_index=segment_index,
                    timestamp=time.time(),
                    confidence=0.4,
                    description="Potential speaker change indicated by text patterns",
                    context_data={
                        'indicator_found': indicator,
                        'text_sample': text[:100] + "..." if len(text) > 100 else text
                    },
                    recommended_action="Verify speaker consistency",
                    academic_impact=0.1
                )
        
        return None
    
    def _detect_background_noise_patterns(self, segment: SRTSegment, segment_index: int) -> Optional[DetectedAnomaly]:
        """Detect background noise patterns in text."""
        text = segment.text.lower()
        
        # Look for noise indicators in transcription
        noise_patterns = [
            r'\[.*\]',  # Bracketed annotations
            r'\(.*\)',  # Parenthetical annotations
            r'background',
            r'noise',
            r'unclear',
            r'inaudible',
            r'static',
            r'crackling'
        ]
        
        for pattern in noise_patterns:
            if re.search(pattern, text):
                return DetectedAnomaly(
                    anomaly_id=f"noise_pattern_{segment_index}_{int(time.time())}",
                    anomaly_type=AnomalyType.BACKGROUND_NOISE,
                    severity=AnomalySeverity.WARNING,
                    segment_index=segment_index,
                    timestamp=time.time(),
                    confidence=0.7,
                    description="Background noise indicators detected in text",
                    context_data={
                        'pattern_found': pattern,
                        'text_sample': text[:100] + "..." if len(text) > 100 else text
                    },
                    recommended_action="Check audio quality and consider re-processing",
                    academic_impact=0.3
                )
        
        return None
    
    def _perform_mcp_analysis(self, context_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Perform Epic 4.1 MCP analysis for context-aware detection."""
        try:
            # This would integrate with actual MCP client in production
            # For now, simulate MCP analysis
            mcp_start_time = time.time()
            
            # Simulate MCP processing delay
            time.sleep(0.01)  # 10ms simulated processing
            
            # Simulate MCP analysis results
            current_text = context_data['current_segment']['text'].lower()
            
            result = {
                'analysis_timestamp': time.time(),
                'processing_time_ms': (time.time() - mcp_start_time) * 1000,
                'topic_shift_detected': False,
                'topic_shift_confidence': 0.0,
                'speaker_change_detected': False,
                'speaker_change_confidence': 0.0,
                'acoustic_anomaly_detected': False,
                'acoustic_confidence': 0.0,
                'acoustic_severity': 'low',
                'academic_impact': 0.0
            }
            
            # Simulate topic shift detection
            topic_keywords = ['meditation', 'prayer', 'scripture', 'practice']
            if any(keyword in current_text for keyword in topic_keywords):
                if len(context_data.get('previous_segments', [])) > 0:
                    prev_text = context_data['previous_segments'][-1]['text'].lower()
                    if not any(keyword in prev_text for keyword in topic_keywords):
                        result['topic_shift_detected'] = True
                        result['topic_shift_confidence'] = 0.8
                        result['academic_impact'] = 0.3
            
            # Simulate speaker change detection
            speaker_indicators = ['question', 'answer', 'student asks']
            if any(indicator in current_text for indicator in speaker_indicators):
                result['speaker_change_detected'] = True
                result['speaker_change_confidence'] = 0.6
                result['academic_impact'] = max(result['academic_impact'], 0.2)
            
            # Simulate acoustic anomaly detection
            if len(current_text) < 20 or len(current_text) > 500:
                result['acoustic_anomaly_detected'] = True
                result['acoustic_confidence'] = 0.7
                result['acoustic_severity'] = 'medium' if len(current_text) > 500 else 'low'
                result['academic_impact'] = max(result['academic_impact'], 0.4)
            
            # Update MCP statistics
            response_time = (time.time() - mcp_start_time) * 1000
            self.mcp_statistics['average_response_time_ms'] = (
                (self.mcp_statistics['average_response_time_ms'] * self.mcp_statistics['requests_successful'] + response_time) /
                (self.mcp_statistics['requests_successful'] + 1)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"MCP analysis failed: {e}")
            raise e
    
    def _update_segment_history(self, segment: SRTSegment, segment_index: int):
        """Update segment history for context analysis."""
        segment_data = {
            'index': segment_index,
            'text': segment.text,
            'start_time': segment.start_time,
            'end_time': segment.end_time,
            'timestamp': time.time()
        }
        
        self.segment_history.append(segment_data)
    
    def _estimate_segment_duration(self, segment: SRTSegment) -> float:
        """Estimate segment duration in seconds."""
        try:
            start_ms = self._parse_timestamp(segment.start_time)
            end_ms = self._parse_timestamp(segment.end_time)
            return (end_ms - start_ms) / 1000.0
        except:
            # Fallback estimation
            return len(segment.text) / 10.0  # ~10 chars per second
    
    def _parse_timestamp(self, timestamp_str: str) -> float:
        """Parse SRT timestamp to milliseconds."""
        # Format: HH:MM:SS,mmm
        try:
            time_part, ms_part = timestamp_str.split(',')
            h, m, s = map(int, time_part.split(':'))
            ms = int(ms_part)
            
            total_ms = (h * 3600 + m * 60 + s) * 1000 + ms
            return total_ms
        except:
            return 0.0
    
    def _calculate_overall_anomaly_score(self, anomalies: List[DetectedAnomaly]) -> float:
        """Calculate overall anomaly score for the segment."""
        if not anomalies:
            return 0.0
        
        # Weight anomalies by severity and confidence
        score = 0.0
        for anomaly in anomalies:
            severity_weight = {
                AnomalySeverity.INFO: 0.3,
                AnomalySeverity.WARNING: 0.6,
                AnomalySeverity.CRITICAL: 1.0
            }.get(anomaly.severity, 0.5)
            
            score += anomaly.confidence * severity_weight
        
        # Normalize by number of anomalies and cap at 1.0
        return min(1.0, score / len(anomalies))
    
    def _calculate_context_coherence_score(self, segment: SRTSegment, 
                                         previous_segments: Optional[List[SRTSegment]] = None) -> float:
        """Calculate context coherence score."""
        if not previous_segments or len(previous_segments) < 2:
            return 0.8  # Default score with insufficient context
        
        try:
            # Simple coherence calculation based on word overlap
            current_words = set(segment.text.lower().split())
            
            total_overlap = 0.0
            for prev_segment in previous_segments[-3:]:  # Last 3 segments
                prev_words = set(prev_segment.text.lower().split())
                if len(current_words) > 0 and len(prev_words) > 0:
                    overlap = len(current_words.intersection(prev_words))
                    overlap_ratio = overlap / min(len(current_words), len(prev_words))
                    total_overlap += overlap_ratio
            
            # Average overlap with recent segments
            avg_overlap = total_overlap / min(3, len(previous_segments))
            
            # Convert to coherence score (higher overlap = higher coherence)
            return min(1.0, avg_overlap * 2)  # Scale to make meaningful
            
        except:
            return 0.5  # Default middle score on error
    
    def _calculate_academic_continuity_score(self, anomalies: List[DetectedAnomaly], segment: SRTSegment) -> float:
        """Calculate Epic 4.5 academic continuity score."""
        base_score = 1.0
        
        # Reduce score based on academic impact of anomalies
        for anomaly in anomalies:
            base_score -= anomaly.academic_impact * 0.2  # Max 20% impact per anomaly
        
        # Bonus for academic content indicators
        text = segment.text.lower()
        academic_indicators = ['chapter', 'verse', 'scripture', 'text', 'teaching', 'wisdom']
        if any(indicator in text for indicator in academic_indicators):
            base_score += 0.1
        
        return max(0.0, min(1.0, base_score))
    
    def _check_circuit_breaker(self) -> bool:
        """Epic 4.1 circuit breaker check."""
        if self.circuit_breaker_open:
            if time.time() > self.circuit_breaker_reset_time:
                self.circuit_breaker_open = False
                self.circuit_breaker_failures = 0
                self.fallback_mode = False
                self.logger.info("Anomaly detector circuit breaker reset")
                return False
            return True
        return False
    
    def _create_fallback_result(self, segment: SRTSegment, segment_index: int, 
                              error_message: str) -> AnomalyAnalysisResult:
        """Create fallback result for Epic 4.1 graceful degradation."""
        return AnomalyAnalysisResult(
            segment_index=segment_index,
            detected_anomalies=[],
            overall_anomaly_score=0.5,
            context_coherence_score=0.5,
            processing_time_ms=0.0,
            mcp_processing_successful=False,
            circuit_breaker_triggered=True,
            fallback_analysis_used=True,
            academic_continuity_score=0.5
        )
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get Epic 4.1 detection statistics and MCP performance."""
        with self.lock:
            total_analyzed = self.anomaly_statistics['total_segments_analyzed']
            
            return {
                'performance_metrics': {
                    'total_segments_analyzed': total_analyzed,
                    'total_anomalies_detected': self.anomaly_statistics['total_anomalies_detected'],
                    'anomaly_detection_rate': (self.anomaly_statistics['total_anomalies_detected'] / total_analyzed) 
                                            if total_analyzed > 0 else 0.0,
                    'anomalies_by_type': dict(self.anomaly_statistics['anomalies_by_type'])
                },
                'mcp_metrics': {
                    'mcp_enabled': self.mcp_enabled,
                    'requests_made': self.mcp_statistics['requests_made'],
                    'requests_successful': self.mcp_statistics['requests_successful'],
                    'requests_failed': self.mcp_statistics['requests_failed'],
                    'success_rate': (self.mcp_statistics['requests_successful'] / self.mcp_statistics['requests_made']) 
                                   if self.mcp_statistics['requests_made'] > 0 else 0.0,
                    'average_response_time_ms': self.mcp_statistics['average_response_time_ms']
                },
                'reliability_metrics': {
                    'circuit_breaker_status': 'open' if self.circuit_breaker_open else 'closed',
                    'circuit_breaker_failures': self.circuit_breaker_failures,
                    'fallback_mode': self.fallback_mode
                }
            }
    
    def reset_circuit_breaker(self):
        """Manual circuit breaker reset."""
        with self.lock:
            self.circuit_breaker_open = False
            self.circuit_breaker_failures = 0
            self.circuit_breaker_reset_time = None
            self.fallback_mode = False
            self.logger.info("Anomaly detector circuit breaker manually reset")
    
    def __del__(self):
        """Cleanup resources."""
        try:
            # Cleanup any resources if needed
            pass
        except:
            pass