"""
Operations Coordinator
Centralized coordination and management of all operational procedures
"""

import logging
import threading
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid

from .incident_response import IncidentManager, IncidentSeverity, IncidentStatus
from .maintenance_manager import MaintenanceManager, MaintenanceType, MaintenanceStatus
from .runbook_manager import RunbookManager, RunbookCategory


class OperationalHealth(Enum):
    """Overall operational health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    DOWN = "down"


@dataclass
class OperationalMetrics:
    """Comprehensive operational metrics"""
    timestamp: datetime
    overall_health: OperationalHealth
    
    # Incident metrics
    open_incidents: int = 0
    critical_incidents: int = 0
    average_resolution_time_minutes: float = 0.0
    
    # Maintenance metrics
    active_maintenance: int = 0
    scheduled_maintenance: int = 0
    
    # Runbook metrics
    runbook_executions_today: int = 0
    successful_runbook_executions: int = 0
    
    # System metrics
    uptime_percentage: float = 100.0
    components_healthy: int = 0
    components_total: int = 0
    
    # Performance metrics
    response_time_ms: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0


@dataclass
class OperationalAlert:
    """Operational alert definition"""
    id: str
    type: str  # incident, maintenance, system, performance
    severity: str  # critical, high, medium, low
    title: str
    message: str
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False
    tags: List[str] = field(default_factory=list)


class OperationsCoordinator:
    """Central operations coordination system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Component managers
        self.incident_manager: Optional[IncidentManager] = None
        self.maintenance_manager: Optional[MaintenanceManager] = None
        self.runbook_manager: Optional[RunbookManager] = None
        
        # Operational state
        self.alerts: Dict[str, OperationalAlert] = {}
        self.metrics_history: List[OperationalMetrics] = []
        self.alert_lock = threading.RLock()
        
        # Monitoring configuration
        self.monitoring_interval = config.get('monitoring_interval_seconds', 60)
        self.metrics_retention_hours = config.get('metrics_retention_hours', 24)
        
        # Alerting configuration
        self.alert_thresholds = config.get('alert_thresholds', {})
        self.notification_config = config.get('notifications', {})
        
        # Coordination thread
        self._coordination_thread = None
        self._shutdown = False
        
        # Health check intervals
        self._last_health_check = datetime.utcnow()
        self._last_metrics_cleanup = datetime.utcnow()
        
    def initialize_components(self, 
                            incident_manager: IncidentManager,
                            maintenance_manager: MaintenanceManager,
                            runbook_manager: RunbookManager):
        """Initialize component managers"""
        self.incident_manager = incident_manager
        self.maintenance_manager = maintenance_manager
        self.runbook_manager = runbook_manager
        
        self.logger.info("Operations coordinator components initialized")
        
        # Start coordination monitoring
        self._start_coordination_monitoring()
        
    def _start_coordination_monitoring(self):
        """Start coordination monitoring thread"""
        self._coordination_thread = threading.Thread(
            target=self._coordination_loop,
            daemon=True
        )
        self._coordination_thread.start()
        self.logger.info("Operations coordination monitoring started")
        
    def _coordination_loop(self):
        """Main coordination monitoring loop"""
        while not self._shutdown:
            try:
                # Collect operational metrics
                self._collect_operational_metrics()
                
                # Check alert conditions
                self._check_alert_conditions()
                
                # Coordinate cross-component activities
                self._coordinate_operations()
                
                # Cleanup old data
                self._cleanup_old_data()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error("Coordination monitoring error", exception=e)
                time.sleep(300)  # Wait 5 minutes on error
                
    def _collect_operational_metrics(self):
        """Collect comprehensive operational metrics"""
        try:
            metrics = OperationalMetrics(timestamp=datetime.utcnow())
            
            # Collect incident metrics
            if self.incident_manager:
                incident_metrics = self.incident_manager.get_metrics()
                metrics.open_incidents = incident_metrics.open_incidents
                metrics.critical_incidents = len([
                    i for i in self.incident_manager.list_incidents()
                    if i.severity == IncidentSeverity.CRITICAL and i.status != IncidentStatus.CLOSED
                ])
                metrics.average_resolution_time_minutes = incident_metrics.average_resolution_time_minutes
                
            # Collect maintenance metrics
            if self.maintenance_manager:
                maintenance_windows = self.maintenance_manager.list_maintenance_windows()
                metrics.active_maintenance = len([
                    w for w in maintenance_windows 
                    if w.status == MaintenanceStatus.IN_PROGRESS
                ])
                metrics.scheduled_maintenance = len([
                    w for w in maintenance_windows
                    if w.status == MaintenanceStatus.SCHEDULED
                ])
                
            # Collect runbook metrics
            if self.runbook_manager:
                today = datetime.utcnow().date()
                executions = self.runbook_manager.list_executions()
                today_executions = [e for e in executions if e.started_at.date() == today]
                successful_executions = [e for e in today_executions if e.status == "completed"]
                
                metrics.runbook_executions_today = len(today_executions)
                metrics.successful_runbook_executions = len(successful_executions)
                
            # Determine overall health
            metrics.overall_health = self._calculate_overall_health(metrics)
            
            # Store metrics
            self.metrics_history.append(metrics)
            
            # Limit metrics history
            cutoff_time = datetime.utcnow() - timedelta(hours=self.metrics_retention_hours)
            self.metrics_history = [
                m for m in self.metrics_history if m.timestamp > cutoff_time
            ]
            
            self.logger.debug(
                "Operational metrics collected",
                overall_health=metrics.overall_health.value,
                open_incidents=metrics.open_incidents,
                active_maintenance=metrics.active_maintenance
            )
            
        except Exception as e:
            self.logger.error("Failed to collect operational metrics", exception=e)
            
    def _calculate_overall_health(self, metrics: OperationalMetrics) -> OperationalHealth:
        """Calculate overall operational health based on metrics"""
        if metrics.critical_incidents > 0:
            return OperationalHealth.CRITICAL
            
        if (metrics.open_incidents > 5 or 
            metrics.active_maintenance > 0 or 
            metrics.error_rate > 0.05):
            return OperationalHealth.DEGRADED
            
        if metrics.open_incidents > 0:
            return OperationalHealth.DEGRADED
            
        return OperationalHealth.HEALTHY
        
    def _check_alert_conditions(self):
        """Check for conditions that require alerts"""
        if not self.metrics_history:
            return
            
        latest_metrics = self.metrics_history[-1]
        
        # Check incident thresholds
        self._check_incident_alerts(latest_metrics)
        
        # Check maintenance alerts
        self._check_maintenance_alerts(latest_metrics)
        
        # Check system health alerts
        self._check_health_alerts(latest_metrics)
        
    def _check_incident_alerts(self, metrics: OperationalMetrics):
        """Check for incident-related alerts"""
        # Critical incidents alert
        if metrics.critical_incidents > 0:
            self._create_alert(
                "critical_incidents",
                "critical",
                "Critical Incidents Active",
                f"{metrics.critical_incidents} critical incidents are currently open and require immediate attention",
                ["incidents", "critical"]
            )
            
        # High incident volume alert
        incident_threshold = self.alert_thresholds.get('max_open_incidents', 10)
        if metrics.open_incidents > incident_threshold:
            self._create_alert(
                "high_incident_volume",
                "high",
                "High Incident Volume",
                f"{metrics.open_incidents} incidents are open (threshold: {incident_threshold})",
                ["incidents", "volume"]
            )
            
    def _check_maintenance_alerts(self, metrics: OperationalMetrics):
        """Check for maintenance-related alerts"""
        if metrics.active_maintenance > 0:
            self._create_alert(
                "active_maintenance",
                "medium",
                "Active Maintenance Windows",
                f"{metrics.active_maintenance} maintenance windows are currently active",
                ["maintenance", "active"]
            )
            
    def _check_health_alerts(self, metrics: OperationalMetrics):
        """Check for system health alerts"""
        if metrics.overall_health == OperationalHealth.CRITICAL:
            self._create_alert(
                "system_critical",
                "critical",
                "System Health Critical",
                "Overall system health is critical - immediate action required",
                ["system", "health", "critical"]
            )
        elif metrics.overall_health == OperationalHealth.DEGRADED:
            self._create_alert(
                "system_degraded",
                "medium",
                "System Health Degraded",
                "Overall system health is degraded - investigation recommended",
                ["system", "health", "degraded"]
            )
            
    def _create_alert(self, alert_type: str, severity: str, title: str, message: str, tags: List[str]):
        """Create operational alert"""
        alert_id = f"{alert_type}_{int(datetime.utcnow().timestamp())}"
        
        with self.alert_lock:
            # Check if similar alert already exists
            existing_alert = None
            for alert in self.alerts.values():
                if alert.type == alert_type and not alert.resolved:
                    existing_alert = alert
                    break
                    
            if existing_alert:
                # Update existing alert
                existing_alert.timestamp = datetime.utcnow()
                existing_alert.message = message
                self.logger.debug(f"Updated existing alert: {existing_alert.id}")
            else:
                # Create new alert
                alert = OperationalAlert(
                    id=alert_id,
                    type=alert_type,
                    severity=severity,
                    title=title,
                    message=message,
                    timestamp=datetime.utcnow(),
                    tags=tags
                )
                
                self.alerts[alert_id] = alert
                self.logger.warning(
                    f"Operational alert created: {title}",
                    alert_id=alert_id,
                    severity=severity,
                    type=alert_type
                )
                
                # Send notifications for new alerts
                self._send_alert_notification(alert)
                
    def _send_alert_notification(self, alert: OperationalAlert):
        """Send alert notification"""
        try:
            # This would integrate with notification systems
            # For now, just log the alert
            self.logger.info(
                f"OPERATIONAL ALERT: {alert.title}",
                alert_id=alert.id,
                severity=alert.severity,
                message=alert.message
            )
            
        except Exception as e:
            self.logger.error("Failed to send alert notification", exception=e)
            
    def _coordinate_operations(self):
        """Coordinate cross-component operational activities"""
        try:
            # Coordinate incident and maintenance scheduling
            self._coordinate_incident_maintenance()
            
            # Coordinate runbook executions with system status
            self._coordinate_runbook_executions()
            
            # Update component health status
            self._update_component_health()
            
        except Exception as e:
            self.logger.error("Failed to coordinate operations", exception=e)
            
    def _coordinate_incident_maintenance(self):
        """Coordinate incident response with maintenance windows"""
        if not (self.incident_manager and self.maintenance_manager):
            return
            
        # Check for conflicts between active incidents and scheduled maintenance
        critical_incidents = [
            i for i in self.incident_manager.list_incidents()
            if i.severity == IncidentSeverity.CRITICAL and i.status != IncidentStatus.CLOSED
        ]
        
        if critical_incidents:
            # Defer non-emergency maintenance during critical incidents
            scheduled_maintenance = [
                w for w in self.maintenance_manager.list_maintenance_windows()
                if w.status == MaintenanceStatus.SCHEDULED and w.maintenance_type != MaintenanceType.EMERGENCY
            ]
            
            for maintenance in scheduled_maintenance:
                # Check if maintenance is scheduled soon (within 2 hours)
                if maintenance.scheduled_start:
                    time_to_maintenance = maintenance.scheduled_start - datetime.utcnow()
                    if time_to_maintenance.total_seconds() < 7200:  # 2 hours
                        self.logger.warning(
                            f"Critical incident active - consider deferring maintenance {maintenance.id}",
                            incident_count=len(critical_incidents),
                            maintenance_id=maintenance.id
                        )
                        
    def _coordinate_runbook_executions(self):
        """Coordinate runbook executions with system status"""
        if not self.runbook_manager:
            return
            
        # Get current operational status
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        if not latest_metrics:
            return
            
        # Check for running runbook executions during critical state
        if latest_metrics.overall_health == OperationalHealth.CRITICAL:
            running_executions = [
                e for e in self.runbook_manager.list_executions()
                if e.status == "running"
            ]
            
            for execution in running_executions:
                runbook = self.runbook_manager.get_runbook(execution.runbook_id)
                if runbook and runbook.category != RunbookCategory.INCIDENT_RESPONSE:
                    self.logger.warning(
                        f"Non-incident runbook executing during critical state: {execution.id}",
                        runbook_id=execution.runbook_id,
                        runbook_title=runbook.title
                    )
                    
    def _update_component_health(self):
        """Update health status of operational components"""
        component_health = {}
        
        # Check incident manager health
        if self.incident_manager:
            incident_health = self.incident_manager.get_health_status()
            component_health['incident_manager'] = incident_health
            
        # Check maintenance manager health
        if self.maintenance_manager:
            maintenance_health = self.maintenance_manager.get_health_status()
            component_health['maintenance_manager'] = maintenance_health
            
        # Check runbook manager health
        if self.runbook_manager:
            runbook_health = self.runbook_manager.get_health_status()
            component_health['runbook_manager'] = runbook_health
            
        # Store component health for reporting
        self.component_health = component_health
        
    def _cleanup_old_data(self):
        """Cleanup old operational data"""
        now = datetime.utcnow()
        
        # Cleanup every hour
        if (now - self._last_metrics_cleanup).total_seconds() < 3600:
            return
            
        self._last_metrics_cleanup = now
        
        try:
            # Cleanup resolved alerts older than 24 hours
            cutoff_time = now - timedelta(hours=24)
            
            with self.alert_lock:
                alerts_to_remove = []
                for alert_id, alert in self.alerts.items():
                    if alert.resolved and alert.timestamp < cutoff_time:
                        alerts_to_remove.append(alert_id)
                        
                for alert_id in alerts_to_remove:
                    del self.alerts[alert_id]
                    
                if alerts_to_remove:
                    self.logger.debug(f"Cleaned up {len(alerts_to_remove)} old resolved alerts")
                    
        except Exception as e:
            self.logger.error("Failed to cleanup old data", exception=e)
            
    def create_operational_incident(self, 
                                  title: str, 
                                  description: str, 
                                  severity: IncidentSeverity,
                                  source_component: str = "operations") -> Optional[str]:
        """Create incident from operational monitoring"""
        if not self.incident_manager:
            self.logger.error("Cannot create incident - incident manager not initialized")
            return None
            
        incident_id = self.incident_manager.create_incident(
            title=title,
            description=description,
            severity=severity,
            tags=["operational", source_component]
        )
        
        self.logger.info(
            f"Operational incident created: {incident_id}",
            title=title,
            severity=severity.value,
            source=source_component
        )
        
        return incident_id
        
    def schedule_operational_maintenance(self,
                                       title: str,
                                       description: str,
                                       maintenance_type: MaintenanceType,
                                       scheduled_start: datetime,
                                       duration_minutes: int,
                                       services_affected: List[str]) -> Optional[str]:
        """Schedule maintenance from operational procedures"""
        if not self.maintenance_manager:
            self.logger.error("Cannot schedule maintenance - maintenance manager not initialized")
            return None
            
        maintenance_id = self.maintenance_manager.schedule_maintenance(
            title=title,
            description=description,
            maintenance_type=maintenance_type,
            scheduled_start=scheduled_start,
            duration_minutes=duration_minutes,
            services_affected=services_affected
        )
        
        self.logger.info(
            f"Operational maintenance scheduled: {maintenance_id}",
            title=title,
            maintenance_type=maintenance_type.value,
            scheduled_start=scheduled_start.isoformat()
        )
        
        return maintenance_id
        
    def execute_operational_runbook(self, 
                                   runbook_id: str, 
                                   executor: str,
                                   context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Execute runbook from operational procedures"""
        if not self.runbook_manager:
            self.logger.error("Cannot execute runbook - runbook manager not initialized")
            return None
            
        execution_id = self.runbook_manager.execute_runbook(
            runbook_id=runbook_id,
            executor=executor,
            context=context or {}
        )
        
        self.logger.info(
            f"Operational runbook execution started: {execution_id}",
            runbook_id=runbook_id,
            executor=executor
        )
        
        return execution_id
        
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge operational alert"""
        with self.alert_lock:
            if alert_id not in self.alerts:
                return False
                
            alert = self.alerts[alert_id]
            alert.acknowledged = True
            
            self.logger.info(
                f"Operational alert acknowledged: {alert_id}",
                acknowledged_by=acknowledged_by,
                alert_title=alert.title
            )
            
            return True
            
    def resolve_alert(self, alert_id: str, resolved_by: str) -> bool:
        """Resolve operational alert"""
        with self.alert_lock:
            if alert_id not in self.alerts:
                return False
                
            alert = self.alerts[alert_id]
            alert.resolved = True
            
            self.logger.info(
                f"Operational alert resolved: {alert_id}",
                resolved_by=resolved_by,
                alert_title=alert.title
            )
            
            return True
            
    def get_operational_status(self) -> Dict[str, Any]:
        """Get comprehensive operational status"""
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        status = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_health': latest_metrics.overall_health.value if latest_metrics else 'unknown',
            'components': {
                'incident_manager': self.incident_manager is not None,
                'maintenance_manager': self.maintenance_manager is not None,
                'runbook_manager': self.runbook_manager is not None
            },
            'metrics': {
                'open_incidents': latest_metrics.open_incidents if latest_metrics else 0,
                'critical_incidents': latest_metrics.critical_incidents if latest_metrics else 0,
                'active_maintenance': latest_metrics.active_maintenance if latest_metrics else 0,
                'scheduled_maintenance': latest_metrics.scheduled_maintenance if latest_metrics else 0
            },
            'alerts': {
                'total': len(self.alerts),
                'unacknowledged': len([a for a in self.alerts.values() if not a.acknowledged]),
                'unresolved': len([a for a in self.alerts.values() if not a.resolved])
            }
        }
        
        return status
        
    def get_operational_metrics(self, hours: int = 24) -> List[OperationalMetrics]:
        """Get operational metrics for specified time period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
    def get_active_alerts(self) -> List[OperationalAlert]:
        """Get all active (unresolved) alerts"""
        with self.alert_lock:
            return [alert for alert in self.alerts.values() if not alert.resolved]
            
    def get_health_status(self) -> Dict[str, Any]:
        """Get operations coordinator health status"""
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        return {
            'status': latest_metrics.overall_health.value if latest_metrics else 'unknown',
            'monitoring_active': self._coordination_thread and self._coordination_thread.is_alive(),
            'components_initialized': {
                'incident_manager': self.incident_manager is not None,
                'maintenance_manager': self.maintenance_manager is not None,
                'runbook_manager': self.runbook_manager is not None
            },
            'metrics_collected': len(self.metrics_history),
            'active_alerts': len([a for a in self.alerts.values() if not a.resolved]),
            'last_health_check': self._last_health_check.isoformat()
        }
        
    def shutdown(self):
        """Shutdown operations coordinator"""
        self._shutdown = True
        
        if self._coordination_thread and self._coordination_thread.is_alive():
            self._coordination_thread.join(timeout=5)
            
        self.logger.info("Operations coordinator shutdown completed")


# Global operations coordinator instance
_operations_coordinator = None


def initialize_operations_management(config: Dict[str, Any]) -> OperationsCoordinator:
    """Initialize operations coordination system"""
    global _operations_coordinator
    _operations_coordinator = OperationsCoordinator(config)
    return _operations_coordinator


def get_operations_coordinator() -> Optional[OperationsCoordinator]:
    """Get the global operations coordinator instance"""
    return _operations_coordinator