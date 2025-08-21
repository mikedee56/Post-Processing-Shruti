"""
Enterprise Telemetry and Alerting System for Story 4.3.

Comprehensive telemetry collection, real-time monitoring, and intelligent alerting
for production excellence in Sanskrit ASR post-processing pipeline.
"""

import json
import logging
import psutil
import queue
import smtplib
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict, deque
from enum import Enum
import sqlite3

from utils.performance_monitor import AlertSeverity, MetricType


class TelemetryEventType(Enum):
    """Types of telemetry events."""
    PERFORMANCE_METRIC = "performance_metric"
    SYSTEM_METRIC = "system_metric"
    ERROR_EVENT = "error_event"
    WARNING_EVENT = "warning_event"
    BUSINESS_METRIC = "business_metric"
    SECURITY_EVENT = "security_event"


class AlertChannel(Enum):
    """Alert delivery channels."""
    LOG = "log"
    EMAIL = "email"
    WEBHOOK = "webhook"
    DATABASE = "database"
    FILE = "file"


@dataclass
class TelemetryEvent:
    """Individual telemetry event with full context."""
    event_type: TelemetryEventType
    timestamp: float
    component: str
    metric_name: str
    value: Union[float, int, str, bool]
    unit: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    severity: AlertSeverity = AlertSeverity.INFO
    source_hostname: Optional[str] = None
    process_id: Optional[int] = None


@dataclass
class AlertRule:
    """Alert rule configuration."""
    rule_id: str
    name: str
    metric_name: str
    condition: str  # 'greater_than', 'less_than', 'equals', 'not_equals'
    threshold: Union[float, int, str]
    severity: AlertSeverity
    channels: List[AlertChannel]
    cooldown_seconds: int = 300  # 5 minutes default
    enabled: bool = True
    component_filter: Optional[str] = None
    tags_filter: Dict[str, str] = field(default_factory=dict)


@dataclass
class AlertNotification:
    """Alert notification with delivery tracking."""
    alert_id: str
    rule_id: str
    event: TelemetryEvent
    triggered_at: float
    severity: AlertSeverity
    message: str
    delivery_attempts: List[Dict[str, Any]] = field(default_factory=list)
    resolved_at: Optional[float] = None
    acknowledged_at: Optional[float] = None
    acknowledged_by: Optional[str] = None


class TelemetryDatabase:
    """SQLite database for telemetry data persistence."""
    
    def __init__(self, db_path: str = "telemetry.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize telemetry database schema."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                # Events table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_type TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        component TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        value TEXT NOT NULL,
                        unit TEXT,
                        tags TEXT,
                        context TEXT,
                        severity TEXT,
                        source_hostname TEXT,
                        process_id INTEGER,
                        created_at REAL DEFAULT (julianday('now'))
                    )
                """)
                
                # Alerts table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        alert_id TEXT UNIQUE NOT NULL,
                        rule_id TEXT NOT NULL,
                        triggered_at REAL NOT NULL,
                        severity TEXT NOT NULL,
                        message TEXT NOT NULL,
                        event_data TEXT NOT NULL,
                        resolved_at REAL,
                        acknowledged_at REAL,
                        acknowledged_by TEXT,
                        created_at REAL DEFAULT (julianday('now'))
                    )
                """)
                
                # Create indices for performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_events_component ON events(component)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_events_metric ON events(metric_name)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_triggered ON alerts(triggered_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity)")
                
                conn.commit()
            finally:
                conn.close()
    
    def store_event(self, event: TelemetryEvent):
        """Store telemetry event in database."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("""
                    INSERT INTO events (
                        event_type, timestamp, component, metric_name, value,
                        unit, tags, context, severity, source_hostname, process_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_type.value,
                    event.timestamp,
                    event.component,
                    event.metric_name,
                    json.dumps(event.value) if not isinstance(event.value, str) else str(event.value),
                    event.unit,
                    json.dumps(event.tags),
                    json.dumps(event.context),
                    event.severity.value,
                    event.source_hostname,
                    event.process_id
                ))
                conn.commit()
            finally:
                conn.close()
    
    def store_alert(self, alert: AlertNotification):
        """Store alert notification in database."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO alerts (
                        alert_id, rule_id, triggered_at, severity, message,
                        event_data, resolved_at, acknowledged_at, acknowledged_by
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert.alert_id,
                    alert.rule_id,
                    alert.triggered_at,
                    alert.severity.value,
                    alert.message,
                    json.dumps(asdict(alert.event)),
                    alert.resolved_at,
                    alert.acknowledged_at,
                    alert.acknowledged_by
                ))
                conn.commit()
            finally:
                conn.close()
    
    def get_recent_events(self, hours: int = 24, component: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent telemetry events."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                query = "SELECT * FROM events WHERE timestamp > ?"
                params = [cutoff_time]
                
                if component:
                    query += " AND component = ?"
                    params.append(component)
                
                query += " ORDER BY timestamp DESC LIMIT 1000"
                
                cursor = conn.execute(query, params)
                columns = [description[0] for description in cursor.description]
                
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
            finally:
                conn.close()
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active (unresolved) alerts."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute("""
                    SELECT * FROM alerts 
                    WHERE resolved_at IS NULL 
                    ORDER BY triggered_at DESC
                """)
                columns = [description[0] for description in cursor.description]
                
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
            finally:
                conn.close()
    
    def close(self):
        """Close database connections and cleanup resources."""
        # SQLite connections are closed after each operation in this implementation
        # This method exists for compatibility with cleanup routines
        pass


class AlertDeliveryManager:
    """Manages alert delivery across multiple channels."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.delivery_queue = queue.Queue()
        self.delivery_thread = None
        self.active = False
        
        # Email configuration
        self.email_config = self.config.get('email', {})
        
        # Webhook configuration
        self.webhook_config = self.config.get('webhook', {})
        
        # File output configuration
        self.file_config = self.config.get('file', {})
    
    def start(self):
        """Start alert delivery manager."""
        if self.active:
            return
        
        self.active = True
        self.delivery_thread = threading.Thread(target=self._delivery_worker, daemon=True)
        self.delivery_thread.start()
        
        self.logger.info("Alert delivery manager started")
    
    def stop(self):
        """Stop alert delivery manager."""
        self.active = False
        if self.delivery_thread:
            self.delivery_thread.join(timeout=5.0)
    
    def deliver_alert(self, alert: AlertNotification, channels: List[AlertChannel]):
        """Queue alert for delivery."""
        self.delivery_queue.put((alert, channels))
    
    def _delivery_worker(self):
        """Worker thread for alert delivery."""
        while self.active:
            try:
                alert, channels = self.delivery_queue.get(timeout=1.0)
                self._deliver_alert_to_channels(alert, channels)
                self.delivery_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Alert delivery error: {e}")
    
    def _deliver_alert_to_channels(self, alert: AlertNotification, channels: List[AlertChannel]):
        """Deliver alert to specified channels."""
        for channel in channels:
            try:
                delivery_start = time.time()
                
                if channel == AlertChannel.LOG:
                    self._deliver_to_log(alert)
                elif channel == AlertChannel.EMAIL:
                    self._deliver_to_email(alert)
                elif channel == AlertChannel.WEBHOOK:
                    self._deliver_to_webhook(alert)
                elif channel == AlertChannel.FILE:
                    self._deliver_to_file(alert)
                elif channel == AlertChannel.DATABASE:
                    self._deliver_to_database(alert)
                
                delivery_time = time.time() - delivery_start
                
                # Record successful delivery
                alert.delivery_attempts.append({
                    'channel': channel.value,
                    'success': True,
                    'delivery_time': delivery_time,
                    'timestamp': time.time()
                })
                
            except Exception as e:
                # Record failed delivery
                alert.delivery_attempts.append({
                    'channel': channel.value,
                    'success': False,
                    'error': str(e),
                    'timestamp': time.time()
                })
                self.logger.error(f"Failed to deliver alert {alert.alert_id} to {channel.value}: {e}")
    
    def _deliver_to_log(self, alert: AlertNotification):
        """Deliver alert to log."""
        severity_map = {
            AlertSeverity.INFO: self.logger.info,
            AlertSeverity.WARNING: self.logger.warning,
            AlertSeverity.CRITICAL: self.logger.critical,
            AlertSeverity.EMERGENCY: self.logger.critical
        }
        
        log_func = severity_map.get(alert.severity, self.logger.info)
        log_func(f"ALERT [{alert.severity.value.upper()}] {alert.message}")
    
    def _deliver_to_email(self, alert: AlertNotification):
        """Deliver alert via email."""
        if not self.email_config.get('enabled', False):
            return
        
        smtp_server = self.email_config.get('smtp_server')
        smtp_port = self.email_config.get('smtp_port', 587)
        username = self.email_config.get('username')
        password = self.email_config.get('password')
        recipients = self.email_config.get('recipients', [])
        
        if not all([smtp_server, username, password, recipients]):
            self.logger.warning("Email configuration incomplete, skipping email delivery")
            return
        
        # Create email message
        subject = f"[{alert.severity.value.upper()}] Production Alert: {alert.event.metric_name}"
        
        body = f"""
Production Alert Notification

Alert ID: {alert.alert_id}
Rule ID: {alert.rule_id}
Severity: {alert.severity.value.upper()}
Triggered: {datetime.fromtimestamp(alert.triggered_at).isoformat()}

Message: {alert.message}

Event Details:
- Component: {alert.event.component}
- Metric: {alert.event.metric_name}
- Value: {alert.event.value}
- Timestamp: {datetime.fromtimestamp(alert.event.timestamp).isoformat()}

Context: {json.dumps(alert.event.context, indent=2)}

This is an automated message from the Sanskrit ASR Post-Processing Production Excellence System.
"""
        
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = username
        msg['To'] = ', '.join(recipients)
        
        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
    
    def _deliver_to_webhook(self, alert: AlertNotification):
        """Deliver alert via webhook."""
        import urllib.request
        import urllib.parse
        
        webhook_url = self.webhook_config.get('url')
        if not webhook_url:
            return
        
        # Prepare webhook payload
        payload = {
            'alert_id': alert.alert_id,
            'rule_id': alert.rule_id,
            'severity': alert.severity.value,
            'message': alert.message,
            'triggered_at': alert.triggered_at,
            'event': asdict(alert.event)
        }
        
        # Send webhook
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            webhook_url,
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        
        with urllib.request.urlopen(req, timeout=10) as response:
            response.read()
    
    def _deliver_to_file(self, alert: AlertNotification):
        """Deliver alert to file."""
        output_file = self.file_config.get('path', 'alerts.log')
        
        alert_data = {
            'timestamp': datetime.fromtimestamp(alert.triggered_at).isoformat(),
            'alert_id': alert.alert_id,
            'severity': alert.severity.value,
            'message': alert.message,
            'event': asdict(alert.event)
        }
        
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(alert_data) + '\n')
    
    def _deliver_to_database(self, alert: AlertNotification):
        """Deliver alert to database (handled by TelemetryDatabase)."""
        # This is handled by the main telemetry system
        pass


class EnterpriseTelemetrySystem:
    """
    Enterprise-grade telemetry and alerting system for Story 4.3.
    
    Provides comprehensive monitoring, alerting, and analytics for production
    Sanskrit ASR post-processing pipeline.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize enterprise telemetry system."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.database = TelemetryDatabase(self.config.get('database_path', 'telemetry.db'))
        self.alert_delivery = AlertDeliveryManager(self.config.get('alert_delivery', {}))
        
        # Event collection
        self.event_queue = queue.Queue(maxsize=10000)
        self.collection_thread = None
        self.active = False
        
        # Alert rules
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alert_cooldowns: Dict[str, float] = {}
        
        # In-memory metrics for fast access
        self.current_metrics: Dict[str, TelemetryEvent] = {}
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # System info
        self.hostname = self.config.get('hostname', 'unknown')
        self.process_id = psutil.Process().pid
        
        # Initialize default alert rules
        self._initialize_default_alert_rules()
        
        self.logger.info("Enterprise telemetry system initialized")
    
    def start(self):
        """Start telemetry collection and alerting."""
        if self.active:
            return
        
        self.active = True
        
        # Start alert delivery
        self.alert_delivery.start()
        
        # Start collection thread
        self.collection_thread = threading.Thread(target=self._collection_worker, daemon=True)
        self.collection_thread.start()
        
        # Start system metrics collection
        self._start_system_metrics_collection()
        
        self.logger.info("Enterprise telemetry system started")
    
    def stop(self):
        """Stop telemetry collection and alerting."""
        self.active = False
        
        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
        
        self.alert_delivery.stop()
        
        self.logger.info("Enterprise telemetry system stopped")
    
    def record_event(self, 
                    event_type: TelemetryEventType,
                    component: str,
                    metric_name: str,
                    value: Union[float, int, str, bool],
                    unit: Optional[str] = None,
                    tags: Optional[Dict[str, str]] = None,
                    context: Optional[Dict[str, Any]] = None,
                    severity: AlertSeverity = AlertSeverity.INFO):
        """Record telemetry event."""
        
        event = TelemetryEvent(
            event_type=event_type,
            timestamp=time.time(),
            component=component,
            metric_name=metric_name,
            value=value,
            unit=unit,
            tags=tags or {},
            context=context or {},
            severity=severity,
            source_hostname=self.hostname,
            process_id=self.process_id
        )
        
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            self.logger.warning("Telemetry event queue full, dropping event")
    
    def _collection_worker(self):
        """Worker thread for event collection and processing."""
        while self.active:
            try:
                event = self.event_queue.get(timeout=1.0)
                self._process_event(event)
                self.event_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Event processing error: {e}")
    
    def _process_event(self, event: TelemetryEvent):
        """Process individual telemetry event."""
        try:
            # Store in database
            self.database.store_event(event)
            
            # Update in-memory metrics
            metric_key = f"{event.component}.{event.metric_name}"
            self.current_metrics[metric_key] = event
            self.metric_history[metric_key].append(event)
            
            # Check alert rules
            self._check_alert_rules(event)
            
        except Exception as e:
            self.logger.error(f"Failed to process event: {e}")
    
    def _check_alert_rules(self, event: TelemetryEvent):
        """Check event against configured alert rules."""
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue
            
            # Check if rule applies to this event
            if not self._rule_matches_event(rule, event):
                continue
            
            # Check cooldown
            cooldown_key = f"{rule.rule_id}.{event.component}"
            if cooldown_key in self.alert_cooldowns:
                if time.time() - self.alert_cooldowns[cooldown_key] < rule.cooldown_seconds:
                    continue
            
            # Evaluate rule condition
            if self._evaluate_rule_condition(rule, event):
                self._trigger_alert(rule, event)
                self.alert_cooldowns[cooldown_key] = time.time()
    
    def _rule_matches_event(self, rule: AlertRule, event: TelemetryEvent) -> bool:
        """Check if alert rule matches the event."""
        # Check metric name
        if rule.metric_name != event.metric_name:
            return False
        
        # Check component filter
        if rule.component_filter and rule.component_filter != event.component:
            return False
        
        # Check tag filters
        for tag_key, tag_value in rule.tags_filter.items():
            if event.tags.get(tag_key) != tag_value:
                return False
        
        return True
    
    def _evaluate_rule_condition(self, rule: AlertRule, event: TelemetryEvent) -> bool:
        """Evaluate alert rule condition against event."""
        try:
            event_value = event.value
            threshold = rule.threshold
            
            # Convert values for comparison if needed
            if isinstance(event_value, str) and isinstance(threshold, (int, float)):
                try:
                    event_value = float(event_value)
                except ValueError:
                    return False
            
            if rule.condition == 'greater_than':
                return event_value > threshold
            elif rule.condition == 'less_than':
                return event_value < threshold
            elif rule.condition == 'equals':
                return event_value == threshold
            elif rule.condition == 'not_equals':
                return event_value != threshold
            else:
                self.logger.warning(f"Unknown alert condition: {rule.condition}")
                return False
                
        except Exception as e:
            self.logger.warning(f"Failed to evaluate rule condition: {e}")
            return False
    
    def _trigger_alert(self, rule: AlertRule, event: TelemetryEvent):
        """Trigger alert notification."""
        alert_id = f"{rule.rule_id}_{event.component}_{int(time.time())}"
        
        message = f"{rule.name}: {event.metric_name} = {event.value}"
        if event.unit:
            message += f" {event.unit}"
        message += f" (threshold: {rule.threshold})"
        
        alert = AlertNotification(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            event=event,
            triggered_at=time.time(),
            severity=rule.severity,
            message=message
        )
        
        # Store alert in database
        self.database.store_alert(alert)
        
        # Queue for delivery
        self.alert_delivery.deliver_alert(alert, rule.channels)
        
        self.logger.info(f"Alert triggered: {alert_id} - {message}")
    
    def _initialize_default_alert_rules(self):
        """Initialize default alert rules for production monitoring."""
        # Processing time alert
        self.add_alert_rule(AlertRule(
            rule_id="processing_time_critical",
            name="Processing Time Critical",
            metric_name="processing_time_ms",
            condition="greater_than",
            threshold=1000.0,  # 1 second
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.LOG, AlertChannel.DATABASE],
            cooldown_seconds=300
        ))
        
        # Error rate alert
        self.add_alert_rule(AlertRule(
            rule_id="error_rate_warning",
            name="Error Rate Warning",
            metric_name="error_rate_percentage",
            condition="greater_than",
            threshold=1.0,  # 1%
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG, AlertChannel.DATABASE],
            cooldown_seconds=600
        ))
        
        # Memory usage alert
        self.add_alert_rule(AlertRule(
            rule_id="memory_usage_critical",
            name="Memory Usage Critical",
            metric_name="memory_usage_mb",
            condition="greater_than",
            threshold=512.0,  # 512 MB
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.LOG, AlertChannel.DATABASE],
            cooldown_seconds=300
        ))
        
        # Throughput alert
        self.add_alert_rule(AlertRule(
            rule_id="throughput_degradation",
            name="Throughput Degradation",
            metric_name="throughput_segments_per_sec",
            condition="less_than",
            threshold=10.0,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG, AlertChannel.DATABASE],
            cooldown_seconds=300
        ))
    
    def add_alert_rule(self, rule: AlertRule):
        """Add or update alert rule."""
        self.alert_rules[rule.rule_id] = rule
        self.logger.debug(f"Alert rule added/updated: {rule.rule_id}")
    
    def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove alert rule."""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            self.logger.debug(f"Alert rule removed: {rule_id}")
            return True
        return False
    
    def _start_system_metrics_collection(self):
        """Start system metrics collection."""
        def collect_system_metrics():
            while self.active:
                try:
                    # CPU metrics
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.record_event(
                        TelemetryEventType.SYSTEM_METRIC,
                        "system",
                        "cpu_usage_percentage",
                        cpu_percent,
                        unit="percent"
                    )
                    
                    # Memory metrics
                    memory = psutil.virtual_memory()
                    self.record_event(
                        TelemetryEventType.SYSTEM_METRIC,
                        "system",
                        "memory_usage_percentage",
                        memory.percent,
                        unit="percent"
                    )
                    
                    self.record_event(
                        TelemetryEventType.SYSTEM_METRIC,
                        "system",
                        "memory_usage_mb",
                        memory.used / 1024 / 1024,
                        unit="MB"
                    )
                    
                    # Disk metrics
                    disk = psutil.disk_usage('/')
                    self.record_event(
                        TelemetryEventType.SYSTEM_METRIC,
                        "system", 
                        "disk_usage_percentage",
                        disk.percent,
                        unit="percent"
                    )
                    
                    time.sleep(30)  # Collect every 30 seconds
                    
                except Exception as e:
                    self.logger.warning(f"System metrics collection error: {e}")
                    time.sleep(60)
        
        metrics_thread = threading.Thread(target=collect_system_metrics, daemon=True)
        metrics_thread.start()
        
        self.logger.debug("System metrics collection started")
    
    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        cutoff_time = time.time() - (hours * 3600)
        
        summary = {
            'collection_period_hours': hours,
            'current_metrics': {},
            'historical_summary': {},
            'active_alerts': len(self.database.get_active_alerts()),
            'system_status': self._get_system_status()
        }
        
        # Current metrics
        for metric_key, event in self.current_metrics.items():
            if event.timestamp >= cutoff_time:
                summary['current_metrics'][metric_key] = {
                    'value': event.value,
                    'unit': event.unit,
                    'timestamp': event.timestamp,
                    'component': event.component
                }
        
        # Historical summary
        for metric_key, history in self.metric_history.items():
            recent_events = [e for e in history if e.timestamp >= cutoff_time]
            
            if recent_events and isinstance(recent_events[0].value, (int, float)):
                values = [e.value for e in recent_events]
                summary['historical_summary'][metric_key] = {
                    'count': len(values),
                    'average': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'latest': values[-1] if values else None
                }
        
        return summary
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'active': self.active,
            'hostname': self.hostname,
            'process_id': self.process_id,
            'event_queue_size': self.event_queue.qsize(),
            'alert_rules_count': len(self.alert_rules),
            'cooldowns_active': len(self.alert_cooldowns)
        }
    
    def get_alerts_report(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive alerts report."""
        cutoff_time = time.time() - (hours * 3600)
        
        # Get recent events from database
        recent_events = self.database.get_recent_events(hours)
        active_alerts = self.database.get_active_alerts()
        
        # Alert statistics
        alert_events = [e for e in recent_events if e.get('event_type') == 'error_event' or 
                       any(severity in str(e.get('severity', '')).lower() 
                           for severity in ['warning', 'critical', 'emergency'])]
        
        severity_counts = defaultdict(int)
        component_alerts = defaultdict(int)
        
        for event in alert_events:
            severity = event.get('severity', 'info')
            component = event.get('component', 'unknown')
            severity_counts[severity] += 1
            component_alerts[component] += 1
        
        return {
            'report_period_hours': hours,
            'total_events': len(recent_events),
            'alert_events': len(alert_events),
            'active_alerts': len(active_alerts),
            'severity_breakdown': dict(severity_counts),
            'component_breakdown': dict(component_alerts),
            'alert_rate_per_hour': len(alert_events) / hours if hours > 0 else 0,
            'system_health_score': self._calculate_health_score(recent_events, alert_events)
        }
    
    def _calculate_health_score(self, recent_events: List[Dict], alert_events: List[Dict]) -> float:
        """Calculate system health score (0-100)."""
        if not recent_events:
            return 100.0
        
        # Base score starts at 100
        health_score = 100.0
        
        # Deduct points for alert rate
        alert_rate = len(alert_events) / len(recent_events)
        health_score -= min(alert_rate * 100, 50)  # Max 50 points deduction
        
        # Deduct points for critical alerts
        critical_alerts = [e for e in alert_events if 'critical' in str(e.get('severity', '')).lower()]
        health_score -= len(critical_alerts) * 10  # 10 points per critical alert
        
        # Deduct points for emergency alerts
        emergency_alerts = [e for e in alert_events if 'emergency' in str(e.get('severity', '')).lower()]
        health_score -= len(emergency_alerts) * 20  # 20 points per emergency alert
        
        return max(health_score, 0.0)
    
    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics snapshot.
        
        Returns:
            Dict containing current system metrics like CPU and memory usage.
        """
        try:
            # Get current system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics = {
                'cpu_usage_percent': cpu_percent,
                'memory_usage_percent': memory.percent,
                'memory_available_mb': memory.available / 1024 / 1024,
                'disk_usage_percent': disk.percent,
                'disk_free_gb': disk.free / 1024 / 1024 / 1024
            }
            
            self.logger.debug(f"Collected system metrics: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Failed to collect system metrics: {e}")
            return {
                'cpu_usage_percent': 0.0,
                'memory_usage_percent': 0.0,
                'memory_available_mb': 0.0,
                'disk_usage_percent': 0.0,
                'disk_free_gb': 0.0
            }

    def shutdown(self):
        """Shutdown the telemetry system and cleanup resources."""
        try:
            self.logger.info("Shutting down EnterpriseTelemetrySystem...")
            self.stop()
            
            # Wait for threads to finish
            if hasattr(self, 'collection_thread') and self.collection_thread and self.collection_thread.is_alive():
                self.collection_thread.join(timeout=5.0)
            
            # Close database connections
            if hasattr(self, 'database'):
                self.database.close()
            
            # Shutdown alert delivery
            if hasattr(self, 'alert_delivery'):
                self.alert_delivery.stop()
            
            self.logger.info("EnterpriseTelemetrySystem shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during telemetry system shutdown: {e}")


# Global telemetry instance
_global_telemetry: Optional[EnterpriseTelemetrySystem] = None


def get_global_telemetry() -> EnterpriseTelemetrySystem:
    """Get or create global telemetry system instance."""
    global _global_telemetry
    if _global_telemetry is None:
        _global_telemetry = EnterpriseTelemetrySystem()
    return _global_telemetry


def record_performance_metric(component: str, metric_name: str, value: Union[float, int], 
                            unit: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
    """Convenience function to record performance metrics."""
    telemetry = get_global_telemetry()
    telemetry.record_event(
        TelemetryEventType.PERFORMANCE_METRIC,
        component,
        metric_name,
        value,
        unit=unit,
        context=context
    )


def record_error_event(component: str, error_message: str, context: Optional[Dict[str, Any]] = None):
    """Convenience function to record error events."""
    telemetry = get_global_telemetry()
    telemetry.record_event(
        TelemetryEventType.ERROR_EVENT,
        component,
        "error_occurred",
        error_message,
        context=context,
        severity=AlertSeverity.CRITICAL
    )