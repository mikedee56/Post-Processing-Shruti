"""
Incident Response Management
Production incident detection, escalation, and resolution tracking
"""

import logging
import threading
import time
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class IncidentSeverity(Enum):
    """Incident severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IncidentStatus(Enum):
    """Incident status values"""
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    CLOSED = "closed"


@dataclass
class EscalationLevel:
    """Escalation level configuration"""
    level: int
    response_time_minutes: int
    contacts: List[str]
    
    
@dataclass
class Incident:
    """Incident record"""
    id: str
    title: str
    description: str
    severity: IncidentSeverity
    status: IncidentStatus
    created_at: datetime
    updated_at: datetime
    assigned_to: Optional[str] = None
    resolved_at: Optional[datetime] = None
    escalation_level: int = 1
    tags: List[str] = field(default_factory=list)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IncidentMetrics:
    """Incident response metrics"""
    total_incidents: int = 0
    open_incidents: int = 0
    resolved_incidents: int = 0
    average_resolution_time_minutes: float = 0.0
    mttr: float = 0.0  # Mean Time To Resolve
    mtbf: float = 0.0  # Mean Time Between Failures
    incidents_by_severity: Dict[str, int] = field(default_factory=dict)


class IncidentManager:
    """Production incident management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Incident storage (in production, use database)
        self.incidents: Dict[str, Incident] = {}
        self.incident_lock = threading.RLock()
        
        # Escalation configuration
        self.escalation_levels = self._parse_escalation_matrix(
            config.get('escalation_matrix', [])
        )
        
        # Notification configuration
        self.slack_webhook = config.get('slack_webhook')
        self.email_config = config.get('email_config', {})
        self.alert_webhook = config.get('alert_webhook')
        
        # Monitoring thread
        self._monitoring_thread = None
        self._shutdown = False
        self._start_monitoring()
        
    def _parse_escalation_matrix(self, matrix: List[Dict]) -> List[EscalationLevel]:
        """Parse escalation matrix from config"""
        levels = []
        for level_config in matrix:
            level = EscalationLevel(
                level=level_config['level'],
                response_time_minutes=level_config['response_time_minutes'],
                contacts=level_config['contacts']
            )
            levels.append(level)
        return sorted(levels, key=lambda x: x.level)
        
    def _start_monitoring(self):
        """Start incident monitoring thread"""
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitoring_thread.start()
        
    def _monitoring_loop(self):
        """Monitor incidents for escalation and automated actions"""
        while not self._shutdown:
            try:
                self._check_incident_escalations()
                self._check_incident_stale_status()
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error("Incident monitoring error", exception=e)
                time.sleep(300)  # Wait 5 minutes on error
                
    def _check_incident_escalations(self):
        """Check for incidents that need escalation"""
        with self.incident_lock:
            for incident in self.incidents.values():
                if incident.status in [IncidentStatus.OPEN, IncidentStatus.ACKNOWLEDGED]:
                    time_since_created = datetime.utcnow() - incident.created_at
                    
                    # Check if we need to escalate
                    current_level = incident.escalation_level
                    if current_level <= len(self.escalation_levels):
                        level_config = self.escalation_levels[current_level - 1]
                        
                        if time_since_created.total_seconds() > level_config.response_time_minutes * 60:
                            self._escalate_incident(incident)
                            
    def _check_incident_stale_status(self):
        """Check for incidents with stale status"""
        stale_threshold = timedelta(hours=24)
        
        with self.incident_lock:
            for incident in self.incidents.values():
                if incident.status == IncidentStatus.INVESTIGATING:
                    time_since_update = datetime.utcnow() - incident.updated_at
                    
                    if time_since_update > stale_threshold:
                        self._add_timeline_entry(
                            incident,
                            "system",
                            "Incident marked as stale - no updates in 24 hours"
                        )
                        self._send_stale_incident_notification(incident)
                        
    def create_incident(self, 
                       title: str,
                       description: str,
                       severity: IncidentSeverity,
                       tags: Optional[List[str]] = None) -> str:
        """Create new incident"""
        incident_id = str(uuid.uuid4())
        
        incident = Incident(
            id=incident_id,
            title=title,
            description=description,
            severity=severity,
            status=IncidentStatus.OPEN,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags=tags or []
        )
        
        with self.incident_lock:
            self.incidents[incident_id] = incident
            
        # Add creation to timeline
        self._add_timeline_entry(incident, "system", "Incident created")
        
        # Send initial notifications
        self._send_incident_notification(incident, "created")
        
        self.logger.info(
            f"Incident created: {incident_id}",
            title=title,
            severity=severity.value
        )
        
        return incident_id
        
    def update_incident(self,
                       incident_id: str,
                       status: Optional[IncidentStatus] = None,
                       assigned_to: Optional[str] = None,
                       update_message: Optional[str] = None) -> bool:
        """Update incident"""
        with self.incident_lock:
            if incident_id not in self.incidents:
                return False
                
            incident = self.incidents[incident_id]
            old_status = incident.status
            
            if status:
                incident.status = status
                
            if assigned_to:
                incident.assigned_to = assigned_to
                
            if status == IncidentStatus.RESOLVED:
                incident.resolved_at = datetime.utcnow()
                
            incident.updated_at = datetime.utcnow()
            
            # Add timeline entry
            if update_message:
                self._add_timeline_entry(incident, "user", update_message)
            elif status and status != old_status:
                self._add_timeline_entry(
                    incident, 
                    "system", 
                    f"Status changed from {old_status.value} to {status.value}"
                )
                
        # Send status change notification
        if status and status != old_status:
            self._send_incident_notification(incident, "updated")
            
        self.logger.info(
            f"Incident updated: {incident_id}",
            status=incident.status.value,
            assigned_to=incident.assigned_to
        )
        
        return True
        
    def _escalate_incident(self, incident: Incident):
        """Escalate incident to next level"""
        if incident.escalation_level < len(self.escalation_levels):
            incident.escalation_level += 1
            incident.updated_at = datetime.utcnow()
            
            level_config = self.escalation_levels[incident.escalation_level - 1]
            
            self._add_timeline_entry(
                incident,
                "system",
                f"Escalated to level {incident.escalation_level}"
            )
            
            self._send_escalation_notification(incident, level_config)
            
            self.logger.warning(
                f"Incident escalated: {incident.id}",
                level=incident.escalation_level,
                contacts=level_config.contacts
            )
            
    def _add_timeline_entry(self, incident: Incident, actor: str, message: str):
        """Add entry to incident timeline"""
        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'actor': actor,
            'message': message
        }
        incident.timeline.append(entry)
        
    def _send_incident_notification(self, incident: Incident, action: str):
        """Send incident notification"""
        message = self._format_incident_message(incident, action)
        
        # Send Slack notification
        if self.slack_webhook and REQUESTS_AVAILABLE:
            self._send_slack_notification(message)
            
        # Send webhook notification
        if self.alert_webhook and REQUESTS_AVAILABLE:
            self._send_webhook_notification(incident, action)
            
    def _send_escalation_notification(self, incident: Incident, level_config: EscalationLevel):
        """Send escalation notification"""
        message = (
            f"🚨 ESCALATION ALERT 🚨\n"
            f"Incident {incident.id} escalated to Level {level_config.level}\n"
            f"Title: {incident.title}\n"
            f"Severity: {incident.severity.value.upper()}\n"
            f"Response required within {level_config.response_time_minutes} minutes\n"
            f"Contacts: {', '.join(level_config.contacts)}"
        )
        
        if self.slack_webhook and REQUESTS_AVAILABLE:
            self._send_slack_notification(message)
            
    def _send_stale_incident_notification(self, incident: Incident):
        """Send stale incident notification"""
        message = (
            f"⚠️ STALE INCIDENT ALERT ⚠️\n"
            f"Incident {incident.id} has no updates for 24+ hours\n"
            f"Title: {incident.title}\n"
            f"Status: {incident.status.value}\n"
            f"Please review and update status"
        )
        
        if self.slack_webhook and REQUESTS_AVAILABLE:
            self._send_slack_notification(message)
            
    def _format_incident_message(self, incident: Incident, action: str) -> str:
        """Format incident message for notifications"""
        severity_emoji = {
            IncidentSeverity.CRITICAL: "🔴",
            IncidentSeverity.HIGH: "🟠", 
            IncidentSeverity.MEDIUM: "🟡",
            IncidentSeverity.LOW: "🟢"
        }
        
        emoji = severity_emoji.get(incident.severity, "🔵")
        
        return (
            f"{emoji} Incident {action.title()}: {incident.id}\n"
            f"Title: {incident.title}\n"
            f"Severity: {incident.severity.value.upper()}\n"
            f"Status: {incident.status.value.title()}\n"
            f"Created: {incident.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"Assigned: {incident.assigned_to or 'Unassigned'}"
        )
        
    def _send_slack_notification(self, message: str):
        """Send Slack notification"""
        try:
            payload = {"text": message}
            response = requests.post(
                self.slack_webhook,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
        except Exception as e:
            self.logger.error("Failed to send Slack notification", exception=e)
            
    def _send_webhook_notification(self, incident: Incident, action: str):
        """Send webhook notification"""
        try:
            payload = {
                "incident_id": incident.id,
                "action": action,
                "title": incident.title,
                "description": incident.description,
                "severity": incident.severity.value,
                "status": incident.status.value,
                "created_at": incident.created_at.isoformat(),
                "updated_at": incident.updated_at.isoformat(),
                "escalation_level": incident.escalation_level,
                "tags": incident.tags
            }
            
            response = requests.post(
                self.alert_webhook,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
        except Exception as e:
            self.logger.error("Failed to send webhook notification", exception=e)
            
    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Get incident by ID"""
        with self.incident_lock:
            return self.incidents.get(incident_id)
            
    def list_incidents(self, 
                      status: Optional[IncidentStatus] = None,
                      severity: Optional[IncidentSeverity] = None,
                      limit: int = 50) -> List[Incident]:
        """List incidents with optional filters"""
        with self.incident_lock:
            incidents = list(self.incidents.values())
            
            # Apply filters
            if status:
                incidents = [i for i in incidents if i.status == status]
            if severity:
                incidents = [i for i in incidents if i.severity == severity]
                
            # Sort by created_at (newest first) and limit
            incidents.sort(key=lambda x: x.created_at, reverse=True)
            return incidents[:limit]
            
    def get_metrics(self) -> IncidentMetrics:
        """Get incident response metrics"""
        with self.incident_lock:
            incidents = list(self.incidents.values())
            
            metrics = IncidentMetrics()
            metrics.total_incidents = len(incidents)
            
            open_incidents = [i for i in incidents if i.status != IncidentStatus.CLOSED]
            resolved_incidents = [i for i in incidents if i.resolved_at]
            
            metrics.open_incidents = len(open_incidents)
            metrics.resolved_incidents = len(resolved_incidents)
            
            # Calculate resolution times
            if resolved_incidents:
                resolution_times = []
                for incident in resolved_incidents:
                    if incident.resolved_at:
                        resolution_time = (incident.resolved_at - incident.created_at).total_seconds() / 60
                        resolution_times.append(resolution_time)
                        
                metrics.average_resolution_time_minutes = sum(resolution_times) / len(resolution_times)
                metrics.mttr = metrics.average_resolution_time_minutes
                
            # Incidents by severity
            for incident in incidents:
                severity = incident.severity.value
                metrics.incidents_by_severity[severity] = metrics.incidents_by_severity.get(severity, 0) + 1
                
            return metrics
            
    def get_health_status(self) -> Dict[str, Any]:
        """Get incident management health status"""
        metrics = self.get_metrics()
        
        # Determine health based on open critical/high incidents
        critical_incidents = len([
            i for i in self.incidents.values() 
            if i.severity == IncidentSeverity.CRITICAL and i.status != IncidentStatus.CLOSED
        ])
        
        high_incidents = len([
            i for i in self.incidents.values()
            if i.severity == IncidentSeverity.HIGH and i.status != IncidentStatus.CLOSED
        ])
        
        if critical_incidents > 0:
            health = "critical"
        elif high_incidents > 2:
            health = "degraded"
        else:
            health = "healthy"
            
        return {
            'status': health,
            'open_incidents': metrics.open_incidents,
            'critical_incidents': critical_incidents,
            'high_incidents': high_incidents,
            'average_resolution_time_minutes': metrics.average_resolution_time_minutes,
            'monitoring_active': self._monitoring_thread and self._monitoring_thread.is_alive()
        }
        
    def shutdown(self):
        """Shutdown incident manager"""
        self._shutdown = True
        
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5)
            
        self.logger.info("Incident manager shutdown completed")


# Global incident manager instance
_incident_manager = None


def initialize_incident_management(config: Dict[str, Any]) -> IncidentManager:
    """Initialize incident management system"""
    global _incident_manager
    _incident_manager = IncidentManager(config)
    return _incident_manager


def get_incident_manager() -> Optional[IncidentManager]:
    """Get the global incident manager instance"""
    return _incident_manager