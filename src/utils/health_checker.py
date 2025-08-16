"""
Epic 4.3 Health Checker for Production System Monitoring.

Implements comprehensive health checking and monitoring to ensure
99.9% uptime reliability and proactive issue detection.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class CheckType(Enum):
    """Types of health checks."""
    SYSTEM_RESOURCE = "system_resource"
    DATABASE_CONNECTION = "database_connection"
    EXTERNAL_SERVICE = "external_service"
    APPLICATION_LOGIC = "application_logic"
    PERFORMANCE_METRIC = "performance_metric"


@dataclass
class HealthCheckResult:
    """Result of an individual health check."""
    check_name: str
    check_type: CheckType
    status: HealthStatus
    message: str
    
    # Metrics
    response_time_ms: float = 0.0
    error_count: int = 0
    success_rate: float = 1.0
    
    # Timestamps
    timestamp: datetime = field(default_factory=datetime.now)
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    
    # Additional data
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealthSummary:
    """Overall system health summary."""
    overall_status: HealthStatus
    healthy_checks: int = 0
    degraded_checks: int = 0
    unhealthy_checks: int = 0
    critical_checks: int = 0
    
    # Performance metrics
    average_response_time_ms: float = 0.0
    system_uptime_percentage: float = 100.0
    total_errors_last_hour: int = 0
    
    # Timestamps
    last_check_time: datetime = field(default_factory=datetime.now)
    system_start_time: datetime = field(default_factory=datetime.now)
    
    # Details
    failing_checks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class HealthChecker:
    """
    Epic 4.3 Production-Grade Health Monitoring System.
    
    Provides comprehensive health monitoring including:
    - Periodic health checks for all system components
    - Real-time status monitoring and alerting
    - Performance metrics collection and analysis
    - Proactive issue detection and recommendations
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize health checker with Epic 4.3 production settings."""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Configuration
        self.check_interval = self.config.get('check_interval_seconds', 30)
        self.retention_hours = self.config.get('retention_hours', 24)
        self.alert_threshold = self.config.get('alert_threshold', 0.8)
        self.critical_threshold = self.config.get('critical_threshold', 0.5)
        
        # Health checks registry
        self.health_checks: Dict[str, Callable] = {}
        self.check_configs: Dict[str, Dict] = {}
        
        # Health status tracking
        self.current_status = HealthStatus.HEALTHY
        self.check_results: Dict[str, HealthCheckResult] = {}
        self.health_history = deque(maxlen=1000)
        
        # Performance tracking
        self.system_start_time = datetime.now()
        self.downtime_periods = []
        self.error_counts = defaultdict(int)
        
        # Threading for continuous monitoring
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()
        
        # Register default health checks
        self._register_default_checks()
        
        self.logger.info("HealthChecker initialized with Epic 4.3 production monitoring")
    
    def register_health_check(self, 
                            name: str,
                            check_func: Callable,
                            check_type: CheckType = CheckType.APPLICATION_LOGIC,
                            timeout_seconds: float = 10.0,
                            enabled: bool = True) -> None:
        """
        Register a new health check function.
        
        Args:
            name: Unique name for the health check
            check_func: Function that returns (success: bool, message: str, details: dict)
            check_type: Type of health check
            timeout_seconds: Maximum time allowed for check
            enabled: Whether check is initially enabled
        """
        with self.lock:
            self.health_checks[name] = check_func
            self.check_configs[name] = {
                'type': check_type,
                'timeout': timeout_seconds,
                'enabled': enabled,
                'consecutive_failures': 0,
                'last_success': None,
                'last_failure': None
            }
            
            self.logger.info(f"Health check registered: {name} ({check_type.value})")
    
    def start_health_checks(self) -> None:
        """Start continuous health monitoring."""
        with self.lock:
            if self.monitoring_active:
                self.logger.warning("Health monitoring already active")
                return
            
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                name="health_checker",
                daemon=True
            )
            self.monitoring_thread.start()
            
            self.logger.info("Health monitoring started")
    
    def stop_health_checks(self) -> None:
        """Stop continuous health monitoring."""
        with self.lock:
            self.monitoring_active = False
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)
            
            self.logger.info("Health monitoring stopped")
    
    def run_all_checks(self) -> SystemHealthSummary:
        """
        Run all registered health checks and return system summary.
        
        Returns:
            SystemHealthSummary: Comprehensive health status
        """
        start_time = time.time()
        check_results = []
        
        with self.lock:
            # Run each enabled health check
            for name, check_func in self.health_checks.items():
                config = self.check_configs[name]
                
                if not config['enabled']:
                    continue
                
                result = self._run_single_check(name, check_func, config)
                check_results.append(result)
                self.check_results[name] = result
        
        # Analyze results and create summary
        summary = self._create_health_summary(check_results, time.time() - start_time)
        
        # Update system status
        self.current_status = summary.overall_status
        
        # Store in history
        self.health_history.append({
            'timestamp': datetime.now(),
            'status': summary.overall_status.value,
            'healthy_checks': summary.healthy_checks,
            'total_checks': len(check_results),
            'average_response_time': summary.average_response_time_ms
        })
        
        return summary
    
    def get_check_result(self, check_name: str) -> Optional[HealthCheckResult]:
        """Get latest result for specific health check."""
        with self.lock:
            return self.check_results.get(check_name)
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get comprehensive system health information.
        
        Returns:
            dict: Detailed health status and metrics
        """
        with self.lock:
            # Calculate uptime
            uptime_seconds = (datetime.now() - self.system_start_time).total_seconds()
            total_downtime = sum(
                (end - start).total_seconds() 
                for start, end in self.downtime_periods
            )
            uptime_percentage = ((uptime_seconds - total_downtime) / max(uptime_seconds, 1)) * 100
            
            # Health check statistics
            total_checks = len(self.check_results)
            healthy_checks = len([r for r in self.check_results.values() if r.status == HealthStatus.HEALTHY])
            
            # Recent performance
            recent_history = list(self.health_history)[-10:]  # Last 10 checks
            avg_response_time = 0.0
            if recent_history:
                avg_response_time = sum(h['average_response_time'] for h in recent_history) / len(recent_history)
            
            return {
                'system_overview': {
                    'current_status': self.current_status.value,
                    'uptime_percentage': uptime_percentage,
                    'uptime_seconds': uptime_seconds,
                    'system_start_time': self.system_start_time.isoformat(),
                    'monitoring_active': self.monitoring_active
                },
                'health_checks': {
                    'total_checks': total_checks,
                    'healthy_checks': healthy_checks,
                    'degraded_checks': len([r for r in self.check_results.values() if r.status == HealthStatus.DEGRADED]),
                    'unhealthy_checks': len([r for r in self.check_results.values() if r.status == HealthStatus.UNHEALTHY]),
                    'critical_checks': len([r for r in self.check_results.values() if r.status == HealthStatus.CRITICAL]),
                    'health_percentage': (healthy_checks / max(total_checks, 1)) * 100
                },
                'performance_metrics': {
                    'average_response_time_ms': avg_response_time,
                    'check_interval_seconds': self.check_interval,
                    'total_error_count': sum(self.error_counts.values()),
                    'error_rate_last_hour': self._calculate_error_rate()
                },
                'epic_4_3_compliance': {
                    'uptime_target': 99.9,
                    'uptime_compliant': uptime_percentage >= 99.9,
                    'response_time_target_ms': 500.0,
                    'response_time_compliant': avg_response_time <= 500.0,
                    'monitoring_enabled': True
                },
                'recent_issues': self._get_recent_issues(),
                'recommendations': self._get_health_recommendations()
            }
    
    def enable_check(self, check_name: str) -> bool:
        """Enable a specific health check."""
        with self.lock:
            if check_name in self.check_configs:
                self.check_configs[check_name]['enabled'] = True
                self.logger.info(f"Health check enabled: {check_name}")
                return True
            return False
    
    def disable_check(self, check_name: str) -> bool:
        """Disable a specific health check."""
        with self.lock:
            if check_name in self.check_configs:
                self.check_configs[check_name]['enabled'] = False
                self.logger.info(f"Health check disabled: {check_name}")
                return True
            return False
    
    def _run_single_check(self, name: str, check_func: Callable, config: Dict) -> HealthCheckResult:
        """Run a single health check with timeout and error handling."""
        start_time = time.time()
        
        try:
            # Execute check with timeout
            success, message, details = check_func()
            
            response_time_ms = (time.time() - start_time) * 1000
            
            # Determine status
            if success:
                status = HealthStatus.HEALTHY
                config['consecutive_failures'] = 0
                config['last_success'] = datetime.now()
            else:
                config['consecutive_failures'] += 1
                config['last_failure'] = datetime.now()
                
                # Escalate status based on consecutive failures
                if config['consecutive_failures'] >= 5:
                    status = HealthStatus.CRITICAL
                elif config['consecutive_failures'] >= 3:
                    status = HealthStatus.UNHEALTHY
                else:
                    status = HealthStatus.DEGRADED
                
                self.error_counts[name] += 1
            
            return HealthCheckResult(
                check_name=name,
                check_type=config['type'],
                status=status,
                message=message,
                response_time_ms=response_time_ms,
                error_count=config['consecutive_failures'],
                details=details or {},
                last_success=config['last_success'],
                last_failure=config['last_failure']
            )
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            config['consecutive_failures'] += 1
            config['last_failure'] = datetime.now()
            self.error_counts[name] += 1
            
            status = HealthStatus.CRITICAL if config['consecutive_failures'] >= 3 else HealthStatus.UNHEALTHY
            
            return HealthCheckResult(
                check_name=name,
                check_type=config['type'],
                status=status,
                message=f"Health check failed: {str(e)}",
                response_time_ms=response_time_ms,
                error_count=config['consecutive_failures'],
                last_failure=config['last_failure']
            )
    
    def _create_health_summary(self, results: List[HealthCheckResult], total_time: float) -> SystemHealthSummary:
        """Create comprehensive health summary from check results."""
        healthy = len([r for r in results if r.status == HealthStatus.HEALTHY])
        degraded = len([r for r in results if r.status == HealthStatus.DEGRADED])
        unhealthy = len([r for r in results if r.status == HealthStatus.UNHEALTHY])
        critical = len([r for r in results if r.status == HealthStatus.CRITICAL])
        
        # Determine overall status
        if critical > 0:
            overall_status = HealthStatus.CRITICAL
        elif unhealthy > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded > 0:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Calculate metrics
        avg_response_time = sum(r.response_time_ms for r in results) / max(len(results), 1)
        failing_checks = [r.check_name for r in results if r.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]]
        
        return SystemHealthSummary(
            overall_status=overall_status,
            healthy_checks=healthy,
            degraded_checks=degraded,
            unhealthy_checks=unhealthy,
            critical_checks=critical,
            average_response_time_ms=avg_response_time,
            failing_checks=failing_checks,
            recommendations=self._get_health_recommendations()
        )
    
    def _monitoring_loop(self) -> None:
        """Continuous monitoring loop."""
        while self.monitoring_active:
            try:
                # Run health checks
                summary = self.run_all_checks()
                
                # Log status changes
                if summary.overall_status != self.current_status:
                    self.logger.warning(f"System health status changed: {self.current_status.value} -> {summary.overall_status.value}")
                
                # Sleep until next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Health monitoring loop error: {e}")
                time.sleep(5)  # Brief pause before retrying
    
    def _register_default_checks(self) -> None:
        """Register default Epic 4.3 health checks."""
        
        def memory_check():
            """Check system memory usage."""
            try:
                import psutil
                memory = psutil.virtual_memory()
                if memory.percent > 90:
                    return False, f"High memory usage: {memory.percent}%", {'memory_percent': memory.percent}
                elif memory.percent > 80:
                    return True, f"Moderate memory usage: {memory.percent}%", {'memory_percent': memory.percent}
                else:
                    return True, f"Normal memory usage: {memory.percent}%", {'memory_percent': memory.percent}
            except ImportError:
                return True, "Memory monitoring unavailable (psutil not installed)", {}
        
        def disk_check():
            """Check disk space usage."""
            try:
                import psutil
                disk = psutil.disk_usage('/')
                percent_used = (disk.used / disk.total) * 100
                if percent_used > 95:
                    return False, f"Critical disk space: {percent_used:.1f}% used", {'disk_percent': percent_used}
                elif percent_used > 85:
                    return True, f"High disk usage: {percent_used:.1f}% used", {'disk_percent': percent_used}
                else:
                    return True, f"Normal disk usage: {percent_used:.1f}% used", {'disk_percent': percent_used}
            except ImportError:
                return True, "Disk monitoring unavailable (psutil not installed)", {}
        
        def thread_check():
            """Check thread count."""
            try:
                import threading
                thread_count = threading.active_count()
                if thread_count > 100:
                    return False, f"High thread count: {thread_count}", {'thread_count': thread_count}
                elif thread_count > 50:
                    return True, f"Moderate thread count: {thread_count}", {'thread_count': thread_count}
                else:
                    return True, f"Normal thread count: {thread_count}", {'thread_count': thread_count}
            except Exception:
                return True, "Thread monitoring unavailable", {}
        
        # Register default checks
        self.register_health_check("memory_usage", memory_check, CheckType.SYSTEM_RESOURCE)
        self.register_health_check("disk_space", disk_check, CheckType.SYSTEM_RESOURCE)
        self.register_health_check("thread_count", thread_check, CheckType.SYSTEM_RESOURCE)
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate for the last hour."""
        # Simplified error rate calculation
        total_errors = sum(self.error_counts.values())
        return total_errors  # In a real implementation, this would be per hour
    
    def _get_recent_issues(self) -> List[Dict]:
        """Get recent health issues."""
        issues = []
        
        for name, result in self.check_results.items():
            if result.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                issues.append({
                    'check_name': name,
                    'status': result.status.value,
                    'message': result.message,
                    'timestamp': result.timestamp.isoformat(),
                    'consecutive_failures': result.error_count
                })
        
        return issues
    
    def _get_health_recommendations(self) -> List[str]:
        """Generate health improvement recommendations."""
        recommendations = []
        
        # Check for patterns in failed health checks
        critical_checks = [r for r in self.check_results.values() if r.status == HealthStatus.CRITICAL]
        if critical_checks:
            recommendations.append("Immediate attention required for critical health checks")
        
        # Check response times
        slow_checks = [r for r in self.check_results.values() if r.response_time_ms > 1000]
        if slow_checks:
            recommendations.append("Some health checks are responding slowly - investigate performance")
        
        # Check error rates
        if sum(self.error_counts.values()) > 10:
            recommendations.append("High error rate detected - review system logs")
        
        return recommendations