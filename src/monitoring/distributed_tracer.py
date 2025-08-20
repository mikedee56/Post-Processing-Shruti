"""
Distributed Tracing System for Production Observability
Implements OpenTelemetry-compatible tracing for microservices monitoring
"""

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from contextvars import ContextVar
from functools import wraps
import threading

logger = logging.getLogger(__name__)

# Context variable to store current trace context
current_trace_context: ContextVar[Optional['TraceContext']] = ContextVar('trace_context', default=None)


class SpanKind(Enum):
    """Span types following OpenTelemetry conventions"""
    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(Enum):
    """Span status following OpenTelemetry conventions"""
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class TraceContext:
    """Distributed trace context"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    trace_flags: int = 1  # 1 = sampled
    trace_state: str = ""
    baggage: Dict[str, str] = field(default_factory=dict)
    
    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers for propagation"""
        return {
            'traceparent': f"00-{self.trace_id}-{self.span_id}-{self.trace_flags:02x}",
            'tracestate': self.trace_state,
            'baggage': ','.join([f"{k}={v}" for k, v in self.baggage.items()])
        }
    
    @classmethod
    def from_headers(cls, headers: Dict[str, str]) -> Optional['TraceContext']:
        """Create trace context from HTTP headers"""
        traceparent = headers.get('traceparent')
        if not traceparent:
            return None
        
        try:
            # Parse traceparent: version-trace_id-parent_id-flags
            parts = traceparent.split('-')
            if len(parts) != 4 or parts[0] != '00':
                return None
            
            trace_id = parts[1]
            parent_span_id = parts[2]
            trace_flags = int(parts[3], 16)
            
            # Generate new span ID for this service
            span_id = generate_span_id()
            
            # Parse baggage
            baggage = {}
            baggage_header = headers.get('baggage', '')
            if baggage_header:
                for item in baggage_header.split(','):
                    if '=' in item:
                        key, value = item.split('=', 1)
                        baggage[key.strip()] = value.strip()
            
            return cls(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                trace_flags=trace_flags,
                trace_state=headers.get('tracestate', ''),
                baggage=baggage
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse trace context: {e}")
            return None


@dataclass
class SpanEvent:
    """Span event following OpenTelemetry model"""
    name: str
    timestamp: datetime
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """Distributed tracing span"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    service_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    kind: SpanKind = SpanKind.INTERNAL
    status: SpanStatus = SpanStatus.UNSET
    status_message: str = ""
    
    # Attributes and metadata
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)
    
    # Performance metrics
    duration_ms: Optional[float] = None
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add event to span"""
        event = SpanEvent(
            name=name,
            timestamp=datetime.now(timezone.utc),
            attributes=attributes or {}
        )
        self.events.append(event)
    
    def set_attribute(self, key: str, value: Any):
        """Set span attribute"""
        self.attributes[key] = value
    
    def set_status(self, status: SpanStatus, message: str = ""):
        """Set span status"""
        self.status = status
        self.status_message = message
    
    def finish(self):
        """Finish the span"""
        if self.end_time is None:
            self.end_time = datetime.now(timezone.utc)
            if self.start_time:
                self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for export"""
        return {
            'trace_id': self.trace_id,
            'span_id': self.span_id,
            'parent_span_id': self.parent_span_id,
            'operation_name': self.operation_name,
            'service_name': self.service_name,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_ms': self.duration_ms,
            'kind': self.kind.value,
            'status': self.status.value,
            'status_message': self.status_message,
            'attributes': self.attributes,
            'events': [
                {
                    'name': event.name,
                    'timestamp': event.timestamp.isoformat(),
                    'attributes': event.attributes
                }
                for event in self.events
            ]
        }


class SpanExporter:
    """Interface for span exporters"""
    
    def export(self, spans: List[Span]) -> bool:
        """Export spans to external system"""
        raise NotImplementedError


class JaegerExporter(SpanExporter):
    """Export spans to Jaeger tracing system"""
    
    def __init__(self, endpoint: str):
        self.endpoint = endpoint.rstrip('/')
        
    def export(self, spans: List[Span]) -> bool:
        """Export spans to Jaeger"""
        try:
            import requests
            
            # Convert spans to Jaeger format
            jaeger_spans = []
            for span in spans:
                jaeger_span = {
                    'traceID': span.trace_id,
                    'spanID': span.span_id,
                    'parentSpanID': span.parent_span_id,
                    'operationName': span.operation_name,
                    'startTime': int(span.start_time.timestamp() * 1_000_000),  # microseconds
                    'duration': int((span.duration_ms or 0) * 1000),  # microseconds
                    'tags': [{'key': k, 'value': v} for k, v in span.attributes.items()],
                    'process': {
                        'serviceName': span.service_name,
                        'tags': []
                    }
                }
                jaeger_spans.append(jaeger_span)
            
            # Send to Jaeger
            payload = {'data': [{'spans': jaeger_spans}]}
            response = requests.post(
                f"{self.endpoint}/api/traces",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            response.raise_for_status()
            return True
            
        except Exception as e:
            logger.error(f"Failed to export spans to Jaeger: {e}")
            return False


class ConsoleExporter(SpanExporter):
    """Export spans to console for debugging"""
    
    def export(self, spans: List[Span]) -> bool:
        """Export spans to console"""
        try:
            for span in spans:
                logger.info(f"TRACE: {span.operation_name} [{span.duration_ms:.2f}ms] - {span.status.value}")
            return True
        except Exception as e:
            logger.error(f"Failed to export spans to console: {e}")
            return False


class FileExporter(SpanExporter):
    """Export spans to JSON file"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.lock = threading.Lock()
    
    def export(self, spans: List[Span]) -> bool:
        """Export spans to file"""
        try:
            with self.lock:
                with open(self.file_path, 'a', encoding='utf-8') as f:
                    for span in spans:
                        f.write(json.dumps(span.to_dict()) + '\n')
            return True
        except Exception as e:
            logger.error(f"Failed to export spans to file: {e}")
            return False


class DistributedTracer:
    """
    Distributed tracing system for production observability
    Follows OpenTelemetry conventions for compatibility
    """
    
    def __init__(self, service_name: str, config: Optional[Dict] = None):
        self.service_name = service_name
        self.config = config or {}
        
        # Configuration
        self.sampling_ratio = self.config.get('sampling_ratio', 1.0)  # 1.0 = trace everything
        self.max_spans_per_trace = self.config.get('max_spans_per_trace', 1000)
        self.span_timeout_seconds = self.config.get('span_timeout_seconds', 300)
        
        # Active spans storage
        self.active_spans: Dict[str, Span] = {}
        self.completed_spans: List[Span] = []
        self.lock = threading.Lock()
        
        # Exporters
        self.exporters: List[SpanExporter] = []
        self._setup_exporters()
        
        # Background thread for span management
        self.cleanup_thread = threading.Thread(target=self._cleanup_spans, daemon=True)
        self.cleanup_thread.start()
        
        logger.info(f"DistributedTracer initialized for service: {service_name}")
    
    def start_span(self, operation_name: str, kind: SpanKind = SpanKind.INTERNAL,
                   parent_context: Optional[TraceContext] = None,
                   attributes: Optional[Dict[str, Any]] = None) -> Span:
        """Start a new span"""
        
        # Get or create trace context
        if parent_context:
            trace_id = parent_context.trace_id
            parent_span_id = parent_context.span_id
        else:
            # Check current context
            current_context = current_trace_context.get()
            if current_context:
                trace_id = current_context.trace_id
                parent_span_id = current_context.span_id
            else:
                # Start new trace
                trace_id = generate_trace_id()
                parent_span_id = None
        
        # Create span
        span = Span(
            trace_id=trace_id,
            span_id=generate_span_id(),
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            service_name=self.service_name,
            start_time=datetime.now(timezone.utc),
            kind=kind,
            attributes=attributes or {}
        )
        
        # Store active span
        with self.lock:
            self.active_spans[span.span_id] = span
        
        # Set as current context
        new_context = TraceContext(
            trace_id=span.trace_id,
            span_id=span.span_id,
            parent_span_id=span.parent_span_id
        )
        current_trace_context.set(new_context)
        
        return span
    
    def finish_span(self, span: Span):
        """Finish and export a span"""
        span.finish()
        
        with self.lock:
            # Remove from active spans
            if span.span_id in self.active_spans:
                del self.active_spans[span.span_id]
            
            # Add to completed spans
            self.completed_spans.append(span)
            
            # Export if batch is ready
            if len(self.completed_spans) >= 10:  # Batch size
                self._export_spans()
    
    def add_span_event(self, span: Span, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add event to span"""
        span.add_event(name, attributes)
    
    def record_exception(self, span: Span, exception: Exception):
        """Record exception in span"""
        span.set_status(SpanStatus.ERROR, str(exception))
        span.add_event('exception', {
            'exception.type': type(exception).__name__,
            'exception.message': str(exception),
            'exception.stacktrace': str(exception.__traceback__) if exception.__traceback__ else ''
        })
    
    def create_trace_context(self, headers: Dict[str, str]) -> Optional[TraceContext]:
        """Create trace context from incoming headers"""
        return TraceContext.from_headers(headers)
    
    def inject_context(self, span: Span) -> Dict[str, str]:
        """Inject trace context into headers for outgoing requests"""
        context = TraceContext(
            trace_id=span.trace_id,
            span_id=span.span_id,
            parent_span_id=span.parent_span_id
        )
        return context.to_headers()
    
    def trace(self, operation_name: str, kind: SpanKind = SpanKind.INTERNAL):
        """Decorator for tracing function calls"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                span = self.start_span(
                    operation_name=operation_name or func.__name__,
                    kind=kind,
                    attributes={
                        'function.name': func.__name__,
                        'function.module': func.__module__
                    }
                )
                
                try:
                    result = func(*args, **kwargs)
                    span.set_status(SpanStatus.OK)
                    return result
                except Exception as e:
                    self.record_exception(span, e)
                    raise
                finally:
                    self.finish_span(span)
            
            return wrapper
        return decorator
    
    def get_trace_metrics(self) -> Dict[str, Any]:
        """Get tracing metrics"""
        with self.lock:
            active_count = len(self.active_spans)
            completed_count = len(self.completed_spans)
        
        return {
            'active_spans': active_count,
            'completed_spans': completed_count,
            'total_exporters': len(self.exporters),
            'service_name': self.service_name,
            'sampling_ratio': self.sampling_ratio
        }
    
    def _setup_exporters(self):
        """Setup span exporters based on configuration"""
        
        # Console exporter for development
        if self.config.get('console_exporter', False):
            self.exporters.append(ConsoleExporter())
        
        # File exporter
        if 'file_exporter_path' in self.config:
            self.exporters.append(FileExporter(self.config['file_exporter_path']))
        
        # Jaeger exporter
        if 'jaeger_endpoint' in self.config:
            self.exporters.append(JaegerExporter(self.config['jaeger_endpoint']))
        
        # Default to console if no exporters configured
        if not self.exporters:
            self.exporters.append(ConsoleExporter())
    
    def _export_spans(self):
        """Export completed spans"""
        if not self.completed_spans:
            return
        
        spans_to_export = self.completed_spans.copy()
        self.completed_spans.clear()
        
        for exporter in self.exporters:
            try:
                exporter.export(spans_to_export)
            except Exception as e:
                logger.error(f"Exporter failed: {e}")
    
    def _cleanup_spans(self):
        """Background cleanup of timed-out spans"""
        while True:
            try:
                time.sleep(60)  # Check every minute
                
                now = datetime.now(timezone.utc)
                timeout_threshold = now.timestamp() - self.span_timeout_seconds
                
                with self.lock:
                    timed_out_spans = []
                    for span_id, span in list(self.active_spans.items()):
                        if span.start_time.timestamp() < timeout_threshold:
                            timed_out_spans.append(span_id)
                    
                    for span_id in timed_out_spans:
                        span = self.active_spans.pop(span_id)
                        span.set_status(SpanStatus.ERROR, "Span timed out")
                        span.finish()
                        self.completed_spans.append(span)
                        logger.warning(f"Span {span.operation_name} timed out")
                    
                    # Export any remaining spans periodically
                    if self.completed_spans:
                        self._export_spans()
                        
            except Exception as e:
                logger.error(f"Span cleanup error: {e}")


def generate_trace_id() -> str:
    """Generate 128-bit trace ID"""
    return f"{uuid.uuid4().hex}{uuid.uuid4().hex}"[:32]


def generate_span_id() -> str:
    """Generate 64-bit span ID"""
    return f"{uuid.uuid4().hex}"[:16]


# Global tracer instance
_global_tracer: Optional[DistributedTracer] = None


def initialize_tracing(service_name: str, config: Optional[Dict] = None) -> DistributedTracer:
    """Initialize global tracing"""
    global _global_tracer
    _global_tracer = DistributedTracer(service_name, config)
    return _global_tracer


def get_tracer() -> Optional[DistributedTracer]:
    """Get global tracer instance"""
    return _global_tracer


def trace(operation_name: str, kind: SpanKind = SpanKind.INTERNAL):
    """Decorator for tracing using global tracer"""
    def decorator(func):
        if _global_tracer:
            return _global_tracer.trace(operation_name, kind)(func)
        return func
    return decorator