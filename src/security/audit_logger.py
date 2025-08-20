"""
Audit Logging System for Production Security and Compliance
Provides comprehensive audit trails for all production operations
"""

import json
import logging
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path
from queue import Queue
import uuid


class AuditEventType(Enum):
    """Types of audit events"""
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    FILE_UPLOAD = "file_upload"
    FILE_PROCESS = "file_process"
    FILE_DOWNLOAD = "file_download"
    FILE_DELETE = "file_delete"
    CONFIG_CHANGE = "config_change"
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    ERROR_OCCURRED = "error_occurred"
    SECURITY_VIOLATION = "security_violation"
    API_ACCESS = "api_access"
    DATABASE_ACCESS = "database_access"


class AuditSeverity(Enum):
    """Severity levels for audit events"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Audit event data structure"""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str]
    username: Optional[str]
    source_ip: Optional[str]
    user_agent: Optional[str]
    resource: Optional[str]
    action: str
    result: str  # success, failure, error
    severity: AuditSeverity
    details: Dict[str, Any]
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['event_type'] = self.event_type.value
        data['severity'] = self.severity.value
        return data
    
    def to_json(self) -> str:
        """Convert audit event to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False)


class AuditLogger:
    """Thread-safe audit logging system"""
    
    def __init__(self, log_directory: str = "logs/audit", max_queue_size: int = 10000):
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        self.max_queue_size = max_queue_size
        self.audit_queue = Queue(maxsize=max_queue_size)
        self.is_running = False
        self.worker_thread = None
        
        # Set up audit logger
        self.logger = logging.getLogger('audit')
        self.logger.setLevel(logging.INFO)
        
        # Create rotating file handler for audit logs
        from logging.handlers import RotatingFileHandler
        audit_file = self.log_directory / "audit.log"
        handler = RotatingFileHandler(
            audit_file,
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=10
        )
        
        # JSON formatter for structured logging
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Start worker thread
        self.start()
    
    def start(self):
        """Start audit logging worker thread"""
        if not self.is_running:
            self.is_running = True
            self.worker_thread = threading.Thread(target=self._worker, daemon=True)
            self.worker_thread.start()
    
    def stop(self):
        """Stop audit logging worker thread"""
        if self.is_running:
            self.is_running = False
            if self.worker_thread:
                self.worker_thread.join(timeout=5)
    
    def _worker(self):
        """Worker thread to process audit events"""
        while self.is_running:
            try:
                event = self.audit_queue.get(timeout=1)
                if event:
                    self._write_audit_event(event)
                    self.audit_queue.task_done()
            except:
                continue  # Timeout is expected when queue is empty
    
    def _write_audit_event(self, event: AuditEvent):
        """Write audit event to log file"""
        try:
            self.logger.info(event.to_json())
            
            # For critical events, also write to separate file
            if event.severity == AuditSeverity.CRITICAL:
                critical_file = self.log_directory / "critical_audit.log"
                with open(critical_file, 'a', encoding='utf-8') as f:
                    f.write(event.to_json() + '\n')
                    
        except Exception as e:
            # Fallback logging to prevent audit loss
            fallback_logger = logging.getLogger('audit_fallback')
            fallback_logger.error(f"Failed to write audit event: {e}")
    
    def log_event(self, 
                  event_type: AuditEventType,
                  action: str,
                  result: str,
                  severity: AuditSeverity = AuditSeverity.MEDIUM,
                  user_id: Optional[str] = None,
                  username: Optional[str] = None,
                  source_ip: Optional[str] = None,
                  user_agent: Optional[str] = None,
                  resource: Optional[str] = None,
                  details: Optional[Dict[str, Any]] = None,
                  session_id: Optional[str] = None,
                  correlation_id: Optional[str] = None):
        """Log an audit event"""
        
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            username=username,
            source_ip=source_ip,
            user_agent=user_agent,
            resource=resource,
            action=action,
            result=result,
            severity=severity,
            details=details or {},
            session_id=session_id,
            correlation_id=correlation_id
        )
        
        try:
            self.audit_queue.put_nowait(event)
        except:
            # Queue full, log immediately
            self._write_audit_event(event)
    
    def log_user_login(self, username: str, user_id: str, source_ip: str, 
                       result: str, user_agent: str = None, details: Dict = None):
        """Log user login event"""
        self.log_event(
            event_type=AuditEventType.USER_LOGIN,
            action="login",
            result=result,
            severity=AuditSeverity.MEDIUM if result == "success" else AuditSeverity.HIGH,
            user_id=user_id,
            username=username,
            source_ip=source_ip,
            user_agent=user_agent,
            details=details
        )
    
    def log_file_operation(self, operation: str, filename: str, user_id: str,
                          username: str, result: str, details: Dict = None):
        """Log file operation event"""
        event_type_map = {
            "upload": AuditEventType.FILE_UPLOAD,
            "process": AuditEventType.FILE_PROCESS,
            "download": AuditEventType.FILE_DOWNLOAD,
            "delete": AuditEventType.FILE_DELETE
        }
        
        self.log_event(
            event_type=event_type_map.get(operation, AuditEventType.FILE_PROCESS),
            action=operation,
            result=result,
            severity=AuditSeverity.MEDIUM,
            user_id=user_id,
            username=username,
            resource=filename,
            details=details
        )
    
    def log_security_violation(self, violation_type: str, user_id: str = None,
                             username: str = None, source_ip: str = None,
                             details: Dict = None):
        """Log security violation event"""
        self.log_event(
            event_type=AuditEventType.SECURITY_VIOLATION,
            action=violation_type,
            result="violation",
            severity=AuditSeverity.CRITICAL,
            user_id=user_id,
            username=username,
            source_ip=source_ip,
            details=details
        )
    
    def log_api_access(self, endpoint: str, method: str, user_id: str,
                      username: str, source_ip: str, result: str,
                      response_code: int, details: Dict = None):
        """Log API access event"""
        severity = AuditSeverity.LOW
        if response_code >= 400:
            severity = AuditSeverity.MEDIUM if response_code < 500 else AuditSeverity.HIGH
        
        api_details = {"method": method, "response_code": response_code}
        if details:
            api_details.update(details)
        
        self.log_event(
            event_type=AuditEventType.API_ACCESS,
            action=f"{method} {endpoint}",
            result=result,
            severity=severity,
            user_id=user_id,
            username=username,
            source_ip=source_ip,
            resource=endpoint,
            details=api_details
        )
    
    def log_system_event(self, event: str, result: str, details: Dict = None):
        """Log system event"""
        event_type = AuditEventType.SYSTEM_START if "start" in event.lower() else AuditEventType.SYSTEM_STOP
        
        self.log_event(
            event_type=event_type,
            action=event,
            result=result,
            severity=AuditSeverity.HIGH,
            details=details
        )
    
    def get_audit_report(self, start_date: datetime, end_date: datetime,
                        event_types: List[AuditEventType] = None) -> List[Dict]:
        """Generate audit report for specified time range"""
        report = []
        audit_file = self.log_directory / "audit.log"
        
        if not audit_file.exists():
            return report
        
        try:
            with open(audit_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        event_data = json.loads(line.strip())
                        event_time = datetime.fromisoformat(event_data['timestamp'])
                        
                        if start_date <= event_time <= end_date:
                            if not event_types or event_data['event_type'] in [et.value for et in event_types]:
                                report.append(event_data)
                                
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
                        
        except Exception as e:
            logging.error(f"Failed to generate audit report: {e}")
        
        return sorted(report, key=lambda x: x['timestamp'])


# Global audit logger instance
audit_logger = None


def initialize_audit_logging(log_directory: str = "logs/audit") -> AuditLogger:
    """Initialize global audit logger"""
    global audit_logger
    audit_logger = AuditLogger(log_directory)
    return audit_logger


def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance"""
    if audit_logger is None:
        return initialize_audit_logging()
    return audit_logger


def audit_log(event_type: AuditEventType, action: str, result: str, **kwargs):
    """Convenience function for audit logging"""
    logger = get_audit_logger()
    logger.log_event(event_type, action, result, **kwargs)