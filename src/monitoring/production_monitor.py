"""
Production-Grade Monitoring Enhancement
Extends existing SystemMonitor with production-specific capabilities
"""

import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import uuid
import requests
from urllib.parse import urlparse

from monitoring.system_monitor import SystemMonitor, SystemHealthMetric, AlertSeverity
from security.audit_logger import get_audit_logger, AuditEventType


class ProductionAlertChannel(Enum):
    """Production alert channels"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    PAGERDUTY = "pagerduty"


@dataclass
class ProductionAlert:
    """Production alert with escalation"""
    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    component: str
    metric_value: float
    threshold: float
    triggered_at: datetime
    escalation_level: int = 0
    channels_notified: List[ProductionAlertChannel] = None
    incident_id: Optional[str] = None
    
    def __post_init__(self):
        if self.channels_notified is None:
            self.channels_notified = []


@dataclass
class ProductionMetric:
    """Enhanced production metric"""
    name: str
    value: float
    timestamp: datetime
    component: str
    environment: str
    instance_id: str
    tags: Dict[str, str]
    unit: str = ""
    severity: AlertSeverity = AlertSeverity.INFO


class ProductionMonitor(SystemMonitor):
    """
    Production-grade monitoring system extending SystemMonitor
    Adds production-specific features like distributed tracing, health checks, and alerting
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        
        self.production_config = config.get('production', {}) if config else {}
        self.environment = self.production_config.get('environment', 'production')
        self.instance_id = self.production_config.get('instance_id', 'asr-processor-01')
        
        # Production alerting
        self.alert_channels = self._initialize_alert_channels()
        self.escalation_rules = self._initialize_escalation_rules()
        self.production_alerts: Dict[str, ProductionAlert] = {}
        
        # Distributed tracing
        self.tracing_enabled = self.production_config.get('tracing_enabled', True)
        self.trace_endpoint = self.production_config.get('trace_endpoint', 'http://jaeger:14268/api/traces')
        
        # Health check endpoints
        self.health_checks = self._initialize_health_checks()
        
        # External monitoring integrations
        self.prometheus_endpoint = self.production_config.get('prometheus_endpoint')
        self.grafana_dashboard_url = self.production_config.get('grafana_dashboard_url')
        
        # SLA tracking
        self.sla_targets = self.production_config.get('sla_targets', {
            'availability': 99.9,
            'response_time_ms': 1000,
            'error_rate': 0.01
        })
        
        self.audit_logger = get_audit_logger()
        self.logger.info(f"ProductionMonitor initialized for environment: {self.environment}")
    
    def record_production_metric(self, name: str, value: float, component: str,
                               tags: Optional[Dict[str, str]] = None, unit: str = ""):
        """Record production metric with enhanced metadata"""
        metric = ProductionMetric(
            name=name,
            value=value,
            timestamp=datetime.now(timezone.utc),
            component=component,
            environment=self.environment,
            instance_id=self.instance_id,
            tags=tags or {},
            unit=unit
        )
        
        # Store in parent SystemMonitor
        self.record_system_metric(name, value, component, unit, tags)
        
        # Send to external monitoring systems
        self._export_metric_to_prometheus(metric)
        self._send_trace_data(metric)
        
        # Audit log for critical metrics
        if name in ['error_rate', 'response_time_ms', 'availability']:
            self.audit_logger.log_event(
                event_type=AuditEventType.SYSTEM_START,
                action=f"metric_recorded:{name}",
                result="success",
                severity=AlertSeverity.INFO,
                details={'metric_name': name, 'value': value, 'component': component}
            )
    
    def check_health_endpoints(self) -> Dict[str, Any]:
        """Check all configured health endpoints"""
        health_results = {}
        
        for check_name, check_config in self.health_checks.items():
            try:
                result = self._perform_health_check(check_name, check_config)
                health_results[check_name] = result
                
                # Record health metric
                self.record_production_metric(
                    f"health_check_{check_name}",
                    1.0 if result['status'] == 'healthy' else 0.0,
                    'health_checker',
                    {'check_type': check_config.get('type', 'http')}
                )
                
            except Exception as e:
                self.logger.error(f"Health check {check_name} failed: {e}")
                health_results[check_name] = {
                    'status': 'unhealthy',
                    'error': str(e),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
                # Record failure metric
                self.record_production_metric(
                    f"health_check_{check_name}",
                    0.0,
                    'health_checker',
                    {'error': str(e)[:100]}
                )
        
        return health_results
    
    def trigger_production_alert(self, alert_id: str, title: str, description: str,
                               severity: AlertSeverity, component: str,
                               metric_value: float, threshold: float):
        """Trigger production alert with escalation"""
        
        alert = ProductionAlert(
            alert_id=alert_id,
            title=title,
            description=description,
            severity=severity,
            component=component,
            metric_value=metric_value,
            threshold=threshold,
            triggered_at=datetime.now(timezone.utc),
            incident_id=str(uuid.uuid4())
        )
        
        self.production_alerts[alert_id] = alert
        
        # Send notifications based on severity
        self._send_alert_notifications(alert)
        
        # Audit log critical alerts
        if severity in [AlertSeverity.CRITICAL, AlertSeverity.ERROR]:
            self.audit_logger.log_event(
                event_type=AuditEventType.ERROR_OCCURRED,
                action="production_alert_triggered",
                result="alert_sent",
                severity=severity,
                details={
                    'alert_id': alert_id,
                    'title': title,
                    'component': component,
                    'metric_value': metric_value,
                    'threshold': threshold
                }
            )
        
        self.logger.warning(f"Production alert triggered: {title} [{severity.value}]")
    
    def generate_sla_report(self, hours_back: int = 24) -> Dict[str, Any]:
        """Generate SLA compliance report"""
        end_time = datetime.now(timezone.utc)
        start_time = end_time.replace(hour=end_time.hour - hours_back)
        
        # Calculate availability
        total_minutes = hours_back * 60
        downtime_minutes = self._calculate_downtime_minutes(start_time, end_time)
        availability = ((total_minutes - downtime_minutes) / total_minutes) * 100
        
        # Get performance metrics
        avg_response_time = self._calculate_average_response_time(start_time, end_time)
        error_rate = self._calculate_error_rate(start_time, end_time)
        
        sla_report = {
            'report_period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'hours': hours_back
            },
            'sla_metrics': {
                'availability': {
                    'actual': availability,
                    'target': self.sla_targets['availability'],
                    'met': availability >= self.sla_targets['availability']
                },
                'response_time': {
                    'actual_ms': avg_response_time,
                    'target_ms': self.sla_targets['response_time_ms'],
                    'met': avg_response_time <= self.sla_targets['response_time_ms']
                },
                'error_rate': {
                    'actual': error_rate,
                    'target': self.sla_targets['error_rate'],
                    'met': error_rate <= self.sla_targets['error_rate']
                }
            },
            'incidents': self._get_incidents_summary(start_time, end_time),
            'recommendations': self._generate_sla_recommendations(availability, avg_response_time, error_rate)
        }
        
        return sla_report
    
    def export_metrics_to_prometheus(self):
        """Export current metrics to Prometheus format"""
        if not self.prometheus_endpoint:
            return
        
        prometheus_metrics = []
        
        for metric_name, metrics_deque in self.system_metrics.items():
            if metrics_deque:
                latest_metric = metrics_deque[-1]
                
                # Convert to Prometheus format
                prom_metric = f"""# HELP {metric_name} System metric from ASR processor
# TYPE {metric_name} gauge
{metric_name}{{component="{latest_metric.component}",environment="{self.environment}",instance="{self.instance_id}"}} {latest_metric.value}
"""
                prometheus_metrics.append(prom_metric)
        
        # Send to Prometheus pushgateway
        try:
            response = requests.post(
                f"{self.prometheus_endpoint}/metrics/job/asr-processor/instance/{self.instance_id}",
                data='\n'.join(prometheus_metrics),
                headers={'Content-Type': 'text/plain'}
            )
            response.raise_for_status()
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics to Prometheus: {e}")
    
    def get_production_dashboard_url(self) -> Optional[str]:
        """Get production dashboard URL"""
        if self.grafana_dashboard_url:
            return f"{self.grafana_dashboard_url}?var-environment={self.environment}&var-instance={self.instance_id}"
        return None
    
    def _initialize_alert_channels(self) -> Dict[ProductionAlertChannel, Dict]:
        """Initialize alert channel configurations"""
        channels = {}
        
        # Slack configuration
        if 'slack_webhook' in self.production_config:
            channels[ProductionAlertChannel.SLACK] = {
                'webhook_url': self.production_config['slack_webhook'],
                'enabled': True
            }
        
        # Email configuration
        if 'email_config' in self.production_config:
            channels[ProductionAlertChannel.EMAIL] = {
                **self.production_config['email_config'],
                'enabled': True
            }
        
        # Webhook configuration
        if 'alert_webhook' in self.production_config:
            channels[ProductionAlertChannel.WEBHOOK] = {
                'url': self.production_config['alert_webhook'],
                'enabled': True
            }
        
        return channels
    
    def _initialize_escalation_rules(self) -> Dict[AlertSeverity, List[ProductionAlertChannel]]:
        """Initialize alert escalation rules"""
        return {
            AlertSeverity.INFO: [ProductionAlertChannel.SLACK],
            AlertSeverity.WARNING: [ProductionAlertChannel.SLACK, ProductionAlertChannel.EMAIL],
            AlertSeverity.ERROR: [ProductionAlertChannel.SLACK, ProductionAlertChannel.EMAIL, ProductionAlertChannel.WEBHOOK],
            AlertSeverity.CRITICAL: [ProductionAlertChannel.SLACK, ProductionAlertChannel.EMAIL, ProductionAlertChannel.WEBHOOK, ProductionAlertChannel.SMS]
        }
    
    def _initialize_health_checks(self) -> Dict[str, Dict]:
        """Initialize health check configurations"""
        return {
            'api_endpoint': {
                'type': 'http',
                'url': 'http://localhost:8080/health',
                'timeout': 5,
                'expected_status': 200
            },
            'database': {
                'type': 'database',
                'connection_string': self.production_config.get('db_connection'),
                'timeout': 10
            },
            'redis': {
                'type': 'redis',
                'connection_string': self.production_config.get('redis_connection'),
                'timeout': 5
            },
            'file_system': {
                'type': 'filesystem',
                'paths': ['/app/data', '/app/logs'],
                'check_writable': True
            }
        }
    
    def _perform_health_check(self, check_name: str, config: Dict) -> Dict[str, Any]:
        """Perform individual health check"""
        check_type = config.get('type', 'http')
        
        if check_type == 'http':
            return self._http_health_check(config)
        elif check_type == 'database':
            return self._database_health_check(config)
        elif check_type == 'redis':
            return self._redis_health_check(config)
        elif check_type == 'filesystem':
            return self._filesystem_health_check(config)
        else:
            raise ValueError(f"Unknown health check type: {check_type}")
    
    def _http_health_check(self, config: Dict) -> Dict[str, Any]:
        """Perform HTTP health check"""
        url = config['url']
        timeout = config.get('timeout', 5)
        expected_status = config.get('expected_status', 200)
        
        try:
            response = requests.get(url, timeout=timeout)
            
            return {
                'status': 'healthy' if response.status_code == expected_status else 'unhealthy',
                'response_time_ms': response.elapsed.total_seconds() * 1000,
                'status_code': response.status_code,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def _database_health_check(self, config: Dict) -> Dict[str, Any]:
        """Perform database health check"""
        # Simplified - in production would use actual DB connection
        try:
            # Mock database check
            return {
                'status': 'healthy',
                'connection_time_ms': 10,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def _redis_health_check(self, config: Dict) -> Dict[str, Any]:
        """Perform Redis health check"""
        # Simplified - in production would use actual Redis connection
        try:
            return {
                'status': 'healthy',
                'ping_time_ms': 5,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def _filesystem_health_check(self, config: Dict) -> Dict[str, Any]:
        """Perform filesystem health check"""
        paths = config.get('paths', [])
        check_writable = config.get('check_writable', False)
        
        try:
            for path in paths:
                path_obj = Path(path)
                if not path_obj.exists():
                    return {
                        'status': 'unhealthy',
                        'error': f"Path {path} does not exist",
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                
                if check_writable:
                    test_file = path_obj / f".health_check_{int(time.time())}"
                    try:
                        test_file.touch()
                        test_file.unlink()
                    except Exception:
                        return {
                            'status': 'unhealthy',
                            'error': f"Path {path} is not writable",
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        }
            
            return {
                'status': 'healthy',
                'paths_checked': len(paths),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def _send_alert_notifications(self, alert: ProductionAlert):
        """Send alert notifications through configured channels"""
        channels = self.escalation_rules.get(alert.severity, [])
        
        for channel in channels:
            if channel in self.alert_channels:
                try:
                    self._send_notification(channel, alert)
                    alert.channels_notified.append(channel)
                    
                except Exception as e:
                    self.logger.error(f"Failed to send alert via {channel.value}: {e}")
    
    def _send_notification(self, channel: ProductionAlertChannel, alert: ProductionAlert):
        """Send notification via specific channel"""
        if channel == ProductionAlertChannel.SLACK:
            self._send_slack_notification(alert)
        elif channel == ProductionAlertChannel.EMAIL:
            self._send_email_notification(alert)
        elif channel == ProductionAlertChannel.WEBHOOK:
            self._send_webhook_notification(alert)
    
    def _send_slack_notification(self, alert: ProductionAlert):
        """Send Slack notification"""
        slack_config = self.alert_channels[ProductionAlertChannel.SLACK]
        webhook_url = slack_config['webhook_url']
        
        color_map = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ff9800",
            AlertSeverity.ERROR: "#f44336",
            AlertSeverity.CRITICAL: "#9c27b0"
        }
        
        payload = {
            "text": f"🚨 Production Alert: {alert.title}",
            "attachments": [
                {
                    "color": color_map.get(alert.severity, "#607d8b"),
                    "fields": [
                        {"title": "Severity", "value": alert.severity.value, "short": True},
                        {"title": "Component", "value": alert.component, "short": True},
                        {"title": "Value", "value": f"{alert.metric_value:.3f}", "short": True},
                        {"title": "Threshold", "value": f"{alert.threshold:.3f}", "short": True},
                        {"title": "Environment", "value": self.environment, "short": True},
                        {"title": "Instance", "value": self.instance_id, "short": True},
                        {"title": "Description", "value": alert.description, "short": False}
                    ],
                    "footer": "ASR Production Monitor",
                    "ts": int(alert.triggered_at.timestamp())
                }
            ]
        }
        
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
    
    def _send_email_notification(self, alert: ProductionAlert):
        """Send email notification (simplified implementation)"""
        # In production, would integrate with actual email service
        self.logger.info(f"EMAIL ALERT: {alert.title} - {alert.description}")
    
    def _send_webhook_notification(self, alert: ProductionAlert):
        """Send webhook notification"""
        webhook_config = self.alert_channels[ProductionAlertChannel.WEBHOOK]
        webhook_url = webhook_config['url']
        
        payload = {
            "alert_id": alert.alert_id,
            "title": alert.title,
            "description": alert.description,
            "severity": alert.severity.value,
            "component": alert.component,
            "metric_value": alert.metric_value,
            "threshold": alert.threshold,
            "environment": self.environment,
            "instance_id": self.instance_id,
            "triggered_at": alert.triggered_at.isoformat(),
            "incident_id": alert.incident_id
        }
        
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
    
    def _export_metric_to_prometheus(self, metric: ProductionMetric):
        """Export individual metric to Prometheus"""
        if not self.prometheus_endpoint:
            return
        
        # This would integrate with actual Prometheus client in production
        pass
    
    def _send_trace_data(self, metric: ProductionMetric):
        """Send tracing data if enabled"""
        if not self.tracing_enabled or not self.trace_endpoint:
            return
        
        # This would send trace data to Jaeger or similar in production
        pass
    
    def _calculate_downtime_minutes(self, start_time: datetime, end_time: datetime) -> float:
        """Calculate downtime in minutes for SLA calculation"""
        downtime = 0.0
        
        for alert in self.production_alerts.values():
            if (alert.severity == AlertSeverity.CRITICAL and 
                alert.triggered_at >= start_time and 
                alert.triggered_at <= end_time):
                
                # Calculate downtime duration
                resolution_time = datetime.now(timezone.utc)  # Assume ongoing if not resolved
                downtime_duration = (resolution_time - alert.triggered_at).total_seconds() / 60
                downtime += downtime_duration
        
        return downtime
    
    def _calculate_average_response_time(self, start_time: datetime, end_time: datetime) -> float:
        """Calculate average response time for SLA calculation"""
        # Simplified - would integrate with actual performance metrics
        return 500.0  # milliseconds
    
    def _calculate_error_rate(self, start_time: datetime, end_time: datetime) -> float:
        """Calculate error rate for SLA calculation"""
        # Simplified - would integrate with actual error metrics
        return 0.005  # 0.5%
    
    def _get_incidents_summary(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Get incidents summary for SLA report"""
        incidents = []
        
        for alert in self.production_alerts.values():
            if (alert.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL] and
                alert.triggered_at >= start_time and
                alert.triggered_at <= end_time):
                
                incidents.append({
                    'incident_id': alert.incident_id,
                    'title': alert.title,
                    'severity': alert.severity.value,
                    'component': alert.component,
                    'triggered_at': alert.triggered_at.isoformat(),
                    'duration_minutes': (datetime.now(timezone.utc) - alert.triggered_at).total_seconds() / 60
                })
        
        return incidents
    
    def _generate_sla_recommendations(self, availability: float, response_time: float, error_rate: float) -> List[str]:
        """Generate SLA improvement recommendations"""
        recommendations = []
        
        if availability < self.sla_targets['availability']:
            recommendations.append(f"Availability ({availability:.2f}%) below target - investigate frequent outages")
        
        if response_time > self.sla_targets['response_time_ms']:
            recommendations.append(f"Response time ({response_time:.0f}ms) above target - optimize performance")
        
        if error_rate > self.sla_targets['error_rate']:
            recommendations.append(f"Error rate ({error_rate:.3f}) above target - improve error handling")
        
        if not recommendations:
            recommendations.append("All SLA targets are being met - maintain current performance")
        
        return recommendations