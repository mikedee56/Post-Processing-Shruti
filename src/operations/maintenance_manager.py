"""
Maintenance Management System
Scheduled maintenance windows, emergency procedures, and system coordination
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
    import croniter
    CRONITER_AVAILABLE = True
except ImportError:
    CRONITER_AVAILABLE = False


class MaintenanceType(Enum):
    """Maintenance types"""
    SCHEDULED = "scheduled"
    EMERGENCY = "emergency" 
    SECURITY = "security"
    UPGRADE = "upgrade"
    PATCHING = "patching"


class MaintenanceStatus(Enum):
    """Maintenance status"""
    PLANNED = "planned"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class MaintenanceWindow:
    """Maintenance window definition"""
    id: str
    title: str
    description: str
    maintenance_type: MaintenanceType
    status: MaintenanceStatus
    scheduled_start: datetime
    scheduled_end: datetime
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None
    responsible_team: str = "operations"
    impact_level: str = "low"  # low, medium, high
    services_affected: List[str] = field(default_factory=list)
    checklist: List[Dict[str, Any]] = field(default_factory=list)
    rollback_plan: str = ""
    communication_plan: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class MaintenanceMetrics:
    """Maintenance metrics"""
    total_maintenance_windows: int = 0
    completed_maintenance: int = 0
    failed_maintenance: int = 0
    average_duration_minutes: float = 0.0
    emergency_maintenance_count: int = 0
    adherence_to_schedule_percent: float = 0.0


class MaintenanceManager:
    """Production maintenance management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Maintenance storage (in production, use database)
        self.maintenance_windows: Dict[str, MaintenanceWindow] = {}
        self.maintenance_lock = threading.RLock()
        
        # Configuration
        self.weekly_window = config.get('weekly_window', {})
        self.emergency_contact = config.get('emergency_contact')
        
        # Notification settings
        self.notification_channels = config.get('notification_channels', [])
        
        # Scheduler thread
        self._scheduler_thread = None
        self._shutdown = False
        
        if CRONITER_AVAILABLE:
            self._start_scheduler()
        else:
            self.logger.warning("croniter not available, maintenance scheduling disabled")
            
    def _start_scheduler(self):
        """Start maintenance scheduler thread"""
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True
        )
        self._scheduler_thread.start()
        
    def _scheduler_loop(self):
        """Maintenance scheduler loop"""
        while not self._shutdown:
            try:
                self._check_scheduled_maintenance()
                self._send_upcoming_maintenance_notifications()
                time.sleep(300)  # Check every 5 minutes
            except Exception as e:
                self.logger.error("Maintenance scheduler error", exception=e)
                time.sleep(600)  # Wait 10 minutes on error
                
    def _check_scheduled_maintenance(self):
        """Check for maintenance windows that should start"""
        current_time = datetime.utcnow()
        
        with self.maintenance_lock:
            for maintenance in self.maintenance_windows.values():
                if (maintenance.status == MaintenanceStatus.APPROVED and
                    maintenance.scheduled_start <= current_time and
                    not maintenance.actual_start):
                    
                    self._start_maintenance_window(maintenance)
                    
    def _send_upcoming_maintenance_notifications(self):
        """Send notifications for upcoming maintenance"""
        current_time = datetime.utcnow()
        notification_windows = [
            timedelta(hours=24),  # 24 hours before
            timedelta(hours=4),   # 4 hours before
            timedelta(hours=1),   # 1 hour before
        ]
        
        with self.maintenance_lock:
            for maintenance in self.maintenance_windows.values():
                if maintenance.status == MaintenanceStatus.APPROVED:
                    time_until_start = maintenance.scheduled_start - current_time
                    
                    for window in notification_windows:
                        # Check if we should send notification for this window
                        if (window - timedelta(minutes=5) <= time_until_start <= window + timedelta(minutes=5) and
                            not self._notification_sent(maintenance.id, window)):
                            
                            self._send_maintenance_notification(maintenance, window)
                            self._mark_notification_sent(maintenance.id, window)
                            
    def schedule_maintenance(self,
                           title: str,
                           description: str,
                           maintenance_type: MaintenanceType,
                           scheduled_start: datetime,
                           duration_minutes: int,
                           services_affected: List[str],
                           impact_level: str = "low",
                           responsible_team: str = "operations") -> str:
        """Schedule new maintenance window"""
        
        maintenance_id = str(uuid.uuid4())
        scheduled_end = scheduled_start + timedelta(minutes=duration_minutes)
        
        # Check for conflicts
        if self._has_scheduling_conflict(scheduled_start, scheduled_end):
            raise ValueError("Maintenance window conflicts with existing scheduled maintenance")
            
        maintenance = MaintenanceWindow(
            id=maintenance_id,
            title=title,
            description=description,
            maintenance_type=maintenance_type,
            status=MaintenanceStatus.PLANNED,
            scheduled_start=scheduled_start,
            scheduled_end=scheduled_end,
            responsible_team=responsible_team,
            impact_level=impact_level,
            services_affected=services_affected
        )
        
        with self.maintenance_lock:
            self.maintenance_windows[maintenance_id] = maintenance
            
        self.logger.info(
            f"Maintenance scheduled: {maintenance_id}",
            title=title,
            type=maintenance_type.value,
            start=scheduled_start.isoformat(),
            duration=duration_minutes
        )
        
        return maintenance_id
        
    def approve_maintenance(self, maintenance_id: str) -> bool:
        """Approve scheduled maintenance"""
        with self.maintenance_lock:
            if maintenance_id not in self.maintenance_windows:
                return False
                
            maintenance = self.maintenance_windows[maintenance_id]
            if maintenance.status != MaintenanceStatus.PLANNED:
                return False
                
            maintenance.status = MaintenanceStatus.APPROVED
            
        self.logger.info(f"Maintenance approved: {maintenance_id}")
        return True
        
    def start_emergency_maintenance(self,
                                  title: str,
                                  description: str,
                                  services_affected: List[str],
                                  estimated_duration_minutes: int = 120) -> str:
        """Start emergency maintenance immediately"""
        
        maintenance_id = str(uuid.uuid4())
        current_time = datetime.utcnow()
        estimated_end = current_time + timedelta(minutes=estimated_duration_minutes)
        
        maintenance = MaintenanceWindow(
            id=maintenance_id,
            title=title,
            description=description,
            maintenance_type=MaintenanceType.EMERGENCY,
            status=MaintenanceStatus.IN_PROGRESS,
            scheduled_start=current_time,
            scheduled_end=estimated_end,
            actual_start=current_time,
            impact_level="high",
            services_affected=services_affected,
            responsible_team="oncall"
        )
        
        with self.maintenance_lock:
            self.maintenance_windows[maintenance_id] = maintenance
            
        # Send emergency notifications
        self._send_emergency_maintenance_notification(maintenance)
        
        self.logger.warning(
            f"Emergency maintenance started: {maintenance_id}",
            title=title,
            services=services_affected
        )
        
        return maintenance_id
        
    def _start_maintenance_window(self, maintenance: MaintenanceWindow):
        """Start a scheduled maintenance window"""
        maintenance.status = MaintenanceStatus.IN_PROGRESS
        maintenance.actual_start = datetime.utcnow()
        
        # Send start notification
        self._send_maintenance_start_notification(maintenance)
        
        self.logger.info(
            f"Maintenance window started: {maintenance.id}",
            title=maintenance.title
        )
        
    def complete_maintenance(self, 
                           maintenance_id: str,
                           success: bool = True,
                           completion_notes: str = "") -> bool:
        """Complete maintenance window"""
        
        with self.maintenance_lock:
            if maintenance_id not in self.maintenance_windows:
                return False
                
            maintenance = self.maintenance_windows[maintenance_id]
            if maintenance.status != MaintenanceStatus.IN_PROGRESS:
                return False
                
            maintenance.status = MaintenanceStatus.COMPLETED if success else MaintenanceStatus.FAILED
            maintenance.actual_end = datetime.utcnow()
            
            # Add completion notes to description
            if completion_notes:
                maintenance.description += f"\n\nCompletion Notes: {completion_notes}"
                
        # Send completion notification
        self._send_maintenance_completion_notification(maintenance, success)
        
        self.logger.info(
            f"Maintenance completed: {maintenance_id}",
            success=success,
            duration_minutes=self._calculate_actual_duration(maintenance)
        )
        
        return True
        
    def cancel_maintenance(self, maintenance_id: str, reason: str = "") -> bool:
        """Cancel scheduled maintenance"""
        
        with self.maintenance_lock:
            if maintenance_id not in self.maintenance_windows:
                return False
                
            maintenance = self.maintenance_windows[maintenance_id]
            if maintenance.status == MaintenanceStatus.IN_PROGRESS:
                return False  # Cannot cancel in-progress maintenance
                
            maintenance.status = MaintenanceStatus.CANCELLED
            if reason:
                maintenance.description += f"\n\nCancellation Reason: {reason}"
                
        self.logger.info(f"Maintenance cancelled: {maintenance_id}", reason=reason)
        return True
        
    def _has_scheduling_conflict(self, start: datetime, end: datetime) -> bool:
        """Check if proposed time conflicts with existing maintenance"""
        with self.maintenance_lock:
            for maintenance in self.maintenance_windows.values():
                if maintenance.status in [MaintenanceStatus.APPROVED, MaintenanceStatus.IN_PROGRESS]:
                    # Check for overlap
                    if (start < maintenance.scheduled_end and 
                        end > maintenance.scheduled_start):
                        return True
        return False
        
    def _calculate_actual_duration(self, maintenance: MaintenanceWindow) -> float:
        """Calculate actual maintenance duration in minutes"""
        if maintenance.actual_start and maintenance.actual_end:
            duration = maintenance.actual_end - maintenance.actual_start
            return duration.total_seconds() / 60
        return 0.0
        
    def _notification_sent(self, maintenance_id: str, window: timedelta) -> bool:
        """Check if notification was already sent for this window"""
        # In production, track this in database or cache
        return False
        
    def _mark_notification_sent(self, maintenance_id: str, window: timedelta):
        """Mark notification as sent for this window"""
        # In production, store this in database or cache
        pass
        
    def _send_maintenance_notification(self, maintenance: MaintenanceWindow, time_window: timedelta):
        """Send upcoming maintenance notification"""
        hours = int(time_window.total_seconds() / 3600)
        
        message = (
            f"🔧 SCHEDULED MAINTENANCE NOTIFICATION\n"
            f"Maintenance in {hours} hour(s): {maintenance.title}\n"
            f"Start: {maintenance.scheduled_start.strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"Duration: {(maintenance.scheduled_end - maintenance.scheduled_start).total_seconds() / 60:.0f} minutes\n"
            f"Impact: {maintenance.impact_level.upper()}\n"
            f"Services affected: {', '.join(maintenance.services_affected)}\n"
            f"Description: {maintenance.description}"
        )
        
        self.logger.info(f"Sending maintenance notification for {maintenance.id}")
        
    def _send_emergency_maintenance_notification(self, maintenance: MaintenanceWindow):
        """Send emergency maintenance notification"""
        message = (
            f"🚨 EMERGENCY MAINTENANCE ALERT 🚨\n"
            f"Emergency maintenance started: {maintenance.title}\n"
            f"Started: {maintenance.actual_start.strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"Estimated duration: {(maintenance.scheduled_end - maintenance.scheduled_start).total_seconds() / 60:.0f} minutes\n"
            f"Services affected: {', '.join(maintenance.services_affected)}\n"
            f"Emergency contact: {self.emergency_contact}\n"
            f"Description: {maintenance.description}"
        )
        
        self.logger.warning(f"Sending emergency maintenance notification for {maintenance.id}")
        
    def _send_maintenance_start_notification(self, maintenance: MaintenanceWindow):
        """Send maintenance start notification"""
        message = (
            f"🔧 MAINTENANCE WINDOW STARTED\n"
            f"Maintenance: {maintenance.title}\n"
            f"Started: {maintenance.actual_start.strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"Expected end: {maintenance.scheduled_end.strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"Services affected: {', '.join(maintenance.services_affected)}"
        )
        
        self.logger.info(f"Sending maintenance start notification for {maintenance.id}")
        
    def _send_maintenance_completion_notification(self, maintenance: MaintenanceWindow, success: bool):
        """Send maintenance completion notification"""
        status_emoji = "✅" if success else "❌"
        status_text = "COMPLETED" if success else "FAILED"
        
        duration = self._calculate_actual_duration(maintenance)
        
        message = (
            f"{status_emoji} MAINTENANCE {status_text}\n"
            f"Maintenance: {maintenance.title}\n"
            f"Completed: {maintenance.actual_end.strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"Actual duration: {duration:.0f} minutes\n"
            f"Services affected: {', '.join(maintenance.services_affected)}"
        )
        
        self.logger.info(f"Sending maintenance completion notification for {maintenance.id}")
        
    def get_maintenance_window(self, maintenance_id: str) -> Optional[MaintenanceWindow]:
        """Get maintenance window by ID"""
        with self.maintenance_lock:
            return self.maintenance_windows.get(maintenance_id)
            
    def list_maintenance_windows(self,
                                status: Optional[MaintenanceStatus] = None,
                                maintenance_type: Optional[MaintenanceType] = None,
                                days_ahead: int = 30) -> List[MaintenanceWindow]:
        """List maintenance windows"""
        with self.maintenance_lock:
            windows = list(self.maintenance_windows.values())
            
            # Apply filters
            if status:
                windows = [w for w in windows if w.status == status]
            if maintenance_type:
                windows = [w for w in windows if w.maintenance_type == maintenance_type]
                
            # Filter by time range
            cutoff_date = datetime.utcnow() + timedelta(days=days_ahead)
            windows = [w for w in windows if w.scheduled_start <= cutoff_date]
            
            # Sort by scheduled start time
            windows.sort(key=lambda x: x.scheduled_start)
            return windows
            
    def get_metrics(self) -> MaintenanceMetrics:
        """Get maintenance metrics"""
        with self.maintenance_lock:
            windows = list(self.maintenance_windows.values())
            
            metrics = MaintenanceMetrics()
            metrics.total_maintenance_windows = len(windows)
            
            completed = [w for w in windows if w.status == MaintenanceStatus.COMPLETED]
            failed = [w for w in windows if w.status == MaintenanceStatus.FAILED]
            emergency = [w for w in windows if w.maintenance_type == MaintenanceType.EMERGENCY]
            
            metrics.completed_maintenance = len(completed)
            metrics.failed_maintenance = len(failed)
            metrics.emergency_maintenance_count = len(emergency)
            
            # Calculate average duration
            durations = [self._calculate_actual_duration(w) for w in completed if w.actual_end]
            if durations:
                metrics.average_duration_minutes = sum(durations) / len(durations)
                
            # Calculate schedule adherence
            scheduled = [w for w in windows if w.status in [MaintenanceStatus.COMPLETED, MaintenanceStatus.FAILED]]
            on_time = 0
            
            for window in scheduled:
                if window.actual_start and window.scheduled_start:
                    # Consider "on time" if started within 15 minutes of scheduled time
                    delay = abs((window.actual_start - window.scheduled_start).total_seconds() / 60)
                    if delay <= 15:
                        on_time += 1
                        
            if scheduled:
                metrics.adherence_to_schedule_percent = (on_time / len(scheduled)) * 100
                
            return metrics
            
    def get_health_status(self) -> Dict[str, Any]:
        """Get maintenance management health status"""
        metrics = self.get_metrics()
        
        # Check for active emergency maintenance
        active_emergency = len([
            w for w in self.maintenance_windows.values()
            if w.maintenance_type == MaintenanceType.EMERGENCY and w.status == MaintenanceStatus.IN_PROGRESS
        ])
        
        # Check for overdue maintenance
        current_time = datetime.utcnow()
        overdue = len([
            w for w in self.maintenance_windows.values()
            if w.status == MaintenanceStatus.IN_PROGRESS and current_time > w.scheduled_end
        ])
        
        # Determine health
        if active_emergency > 0 or overdue > 0:
            health = "degraded"
        else:
            health = "healthy"
            
        return {
            'status': health,
            'active_emergency_maintenance': active_emergency,
            'overdue_maintenance': overdue,
            'completed_maintenance_count': metrics.completed_maintenance,
            'average_duration_minutes': metrics.average_duration_minutes,
            'schedule_adherence_percent': metrics.adherence_to_schedule_percent,
            'scheduler_active': self._scheduler_thread and self._scheduler_thread.is_alive()
        }
        
    def shutdown(self):
        """Shutdown maintenance manager"""
        self._shutdown = True
        
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=5)
            
        self.logger.info("Maintenance manager shutdown completed")


# Global maintenance manager instance
_maintenance_manager = None


def initialize_maintenance_management(config: Dict[str, Any]) -> MaintenanceManager:
    """Initialize maintenance management system"""
    global _maintenance_manager
    _maintenance_manager = MaintenanceManager(config)
    return _maintenance_manager


def get_maintenance_manager() -> Optional[MaintenanceManager]:
    """Get the global maintenance manager instance"""
    return _maintenance_manager