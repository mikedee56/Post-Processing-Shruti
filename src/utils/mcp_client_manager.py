"""
MCP Client Management and Monitoring System for Story 4.1
Enterprise-grade MCP infrastructure with comprehensive monitoring and reliability patterns.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from enum import Enum
import json

from .advanced_text_normalizer import MCPClient, CircuitBreakerState, MCPServerConfig


class MCPServerStatus(Enum):
    """MCP server operational status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"


class MCPOperationType(Enum):
    """Types of MCP operations for telemetry."""
    CONTEXT_ANALYSIS = "context_analysis"
    TEXT_PROCESSING = "text_processing"
    FALLBACK_ROUTING = "fallback_routing"
    HEALTH_CHECK = "health_check"


@dataclass
class MCPOperationMetrics:
    """Metrics for individual MCP operations."""
    operation_type: MCPOperationType
    server_name: str
    start_time: float
    end_time: Optional[float] = None
    success: bool = False
    error_message: Optional[str] = None
    fallback_used: bool = False
    response_size_bytes: int = 0
    
    @property
    def duration_ms(self) -> float:
        """Calculate operation duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0


@dataclass
class MCPServerHealth:
    """Health status for MCP servers."""
    server_name: str
    status: MCPServerStatus
    last_health_check: float
    response_time_ms: float
    error_rate: float
    uptime_percentage: float
    consecutive_failures: int = 0
    last_error: Optional[str] = None


@dataclass
class MCPPerformanceAlert:
    """Performance alert configuration and tracking."""
    alert_type: str
    threshold_value: float
    current_value: float
    triggered_at: float
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    description: str
    auto_resolution: bool = False


class MCPClientManager:
    """
    Enterprise-grade MCP client manager with comprehensive monitoring.
    
    Provides:
    - Circuit breaker patterns for service reliability
    - Performance monitoring and telemetry
    - Health status tracking for all MCP servers
    - Automated alerting and recovery systems
    - Graceful degradation and fallback routing
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize MCP client manager with enterprise configurations."""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Initialize base MCP client
        self.mcp_client = MCPClient(config)
        
        # Enterprise-grade monitoring components
        self.server_health: Dict[str, MCPServerHealth] = {}
        self.operation_metrics: List[MCPOperationMetrics] = []
        self.performance_alerts: List[MCPPerformanceAlert] = []
        
        # Configuration for enterprise features
        self.health_check_interval = self.config.get('health_check_interval_seconds', 30)
        self.metrics_retention_hours = self.config.get('metrics_retention_hours', 24)
        self.alert_thresholds = self.config.get('alert_thresholds', self._get_default_alert_thresholds())
        
        # Performance targets
        self.target_response_time_ms = self.config.get('target_response_time_ms', 1000)
        self.target_success_rate = self.config.get('target_success_rate', 0.99)
        self.target_uptime = self.config.get('target_uptime', 0.999)
        
        # Initialize server health tracking
        self._initialize_server_health()
        
        # Start background monitoring (if in async context)
        self._monitoring_task = None
        
        self.logger.info(f"MCPClientManager initialized with {len(self.mcp_client.servers)} servers")
    
    def _get_default_alert_thresholds(self) -> Dict[str, Dict]:
        """Get default alert threshold configurations."""
        return {
            'response_time': {
                'WARNING': 500,  # 500ms
                'CRITICAL': 1000  # 1 second
            },
            'error_rate': {
                'WARNING': 0.05,  # 5%
                'CRITICAL': 0.15  # 15%
            },
            'uptime': {
                'WARNING': 0.95,  # 95%
                'CRITICAL': 0.90   # 90%
            }
        }
    
    def _initialize_server_health(self):
        """Initialize health tracking for all configured servers."""
        current_time = time.time()
        
        for server_name in self.mcp_client.servers.keys():
            self.server_health[server_name] = MCPServerHealth(
                server_name=server_name,
                status=MCPServerStatus.HEALTHY,
                last_health_check=current_time,
                response_time_ms=0.0,
                error_rate=0.0,
                uptime_percentage=100.0
            )
    
    async def process_text_with_monitoring(self, text: str, operation_type: MCPOperationType = MCPOperationType.CONTEXT_ANALYSIS) -> Any:
        """
        Process text with comprehensive monitoring and telemetry.
        
        Args:
            text: Text to process
            operation_type: Type of operation for metrics
            
        Returns:
            Processing result with telemetry
        """
        operation_start = time.time()
        
        # Record operation start
        metrics = MCPOperationMetrics(
            operation_type=operation_type,
            server_name="primary",
            start_time=operation_start
        )
        
        try:
            # Execute the operation through enhanced MCP client
            if operation_type == MCPOperationType.CONTEXT_ANALYSIS:
                result = await self.mcp_client.analyze_number_context(text)
            else:
                # Add other operation types as needed
                result = await self.mcp_client.analyze_number_context(text)
            
            # Record successful operation
            metrics.end_time = time.time()
            metrics.success = True
            metrics.response_size_bytes = len(str(result))
            
            # Update server health based on performance
            await self._update_server_health_from_operation(metrics)
            
        except Exception as e:
            # Record failed operation
            metrics.end_time = time.time()
            metrics.success = False
            metrics.error_message = str(e)
            metrics.fallback_used = True
            
            self.logger.warning(f"MCP operation failed: {e}")
            
            # Still return a result using fallback
            result = await self.mcp_client._fallback_context_analysis(text, operation_start)
        
        # Store metrics for analysis
        self.operation_metrics.append(metrics)
        
        # Clean up old metrics
        await self._cleanup_old_metrics()
        
        # Check for performance alerts
        await self._check_performance_alerts(metrics)
        
        return result
    
    async def _update_server_health_from_operation(self, metrics: MCPOperationMetrics):
        """Update server health status based on operation metrics."""
        server_name = metrics.server_name
        
        if server_name in self.server_health:
            health = self.server_health[server_name]
            
            # Update response time (moving average)
            if health.response_time_ms == 0:
                health.response_time_ms = metrics.duration_ms
            else:
                health.response_time_ms = (health.response_time_ms * 0.7) + (metrics.duration_ms * 0.3)
            
            # Update error rate based on recent operations
            recent_operations = [m for m in self.operation_metrics[-100:] if m.server_name == server_name]
            if recent_operations:
                failed_operations = sum(1 for m in recent_operations if not m.success)
                health.error_rate = failed_operations / len(recent_operations)
            
            # Update consecutive failure count
            if not metrics.success:
                health.consecutive_failures += 1
                health.last_error = metrics.error_message
            else:
                health.consecutive_failures = 0
                health.last_error = None
            
            # Determine health status
            health.status = self._determine_server_status(health)
            health.last_health_check = time.time()
    
    def _determine_server_status(self, health: MCPServerHealth) -> MCPServerStatus:
        """Determine server status based on health metrics."""
        # Critical failures
        if health.consecutive_failures >= 5:
            return MCPServerStatus.UNHEALTHY
        
        # Performance degradation
        if health.response_time_ms > self.target_response_time_ms or health.error_rate > 0.1:
            return MCPServerStatus.DEGRADED
        
        # Healthy operation
        return MCPServerStatus.HEALTHY
    
    async def _cleanup_old_metrics(self):
        """Clean up metrics older than retention period."""
        cutoff_time = time.time() - (self.metrics_retention_hours * 3600)
        
        self.operation_metrics = [
            m for m in self.operation_metrics 
            if m.start_time > cutoff_time
        ]
    
    async def _check_performance_alerts(self, metrics: MCPOperationMetrics):
        """Check if operation metrics trigger any performance alerts."""
        current_time = time.time()
        
        # Response time alerts
        if metrics.duration_ms > self.alert_thresholds['response_time']['CRITICAL']:
            alert = MCPPerformanceAlert(
                alert_type='response_time',
                threshold_value=self.alert_thresholds['response_time']['CRITICAL'],
                current_value=metrics.duration_ms,
                triggered_at=current_time,
                severity='CRITICAL',
                description=f'Server {metrics.server_name} response time {metrics.duration_ms:.1f}ms exceeds critical threshold'
            )
            self.performance_alerts.append(alert)
            self.logger.critical(alert.description)
        
        elif metrics.duration_ms > self.alert_thresholds['response_time']['WARNING']:
            alert = MCPPerformanceAlert(
                alert_type='response_time',
                threshold_value=self.alert_thresholds['response_time']['WARNING'],
                current_value=metrics.duration_ms,
                triggered_at=current_time,
                severity='WARNING',
                description=f'Server {metrics.server_name} response time {metrics.duration_ms:.1f}ms exceeds warning threshold'
            )
            self.performance_alerts.append(alert)
            self.logger.warning(alert.description)
    
    async def perform_health_check(self) -> Dict[str, MCPServerHealth]:
        """Perform comprehensive health check on all MCP servers."""
        self.logger.info("Performing MCP server health check")
        
        health_results = {}
        
        for server_name, server_config in self.mcp_client.servers.items():
            try:
                # Test server connectivity with simple ping
                start_time = time.time()
                
                # Create a lightweight test request
                test_result = await self._test_server_connectivity(server_name, server_config)
                
                end_time = time.time()
                response_time_ms = (end_time - start_time) * 1000
                
                # Update health status
                if server_name in self.server_health:
                    health = self.server_health[server_name]
                    health.response_time_ms = response_time_ms
                    health.last_health_check = end_time
                    
                    if test_result:
                        health.status = MCPServerStatus.HEALTHY
                        health.consecutive_failures = 0
                    else:
                        health.consecutive_failures += 1
                        health.status = self._determine_server_status(health)
                
                health_results[server_name] = self.server_health[server_name]
                
            except Exception as e:
                self.logger.error(f"Health check failed for {server_name}: {e}")
                
                if server_name in self.server_health:
                    health = self.server_health[server_name]
                    health.consecutive_failures += 1
                    health.last_error = str(e)
                    health.status = MCPServerStatus.UNHEALTHY
                    health.last_health_check = time.time()
                    
                    health_results[server_name] = health
        
        return health_results
    
    async def _test_server_connectivity(self, server_name: str, server_config: MCPServerConfig) -> bool:
        """Test basic connectivity to an MCP server."""
        try:
            # Simple connectivity test - this would be implemented based on actual MCP protocol
            # For now, simulate a basic connectivity check
            
            import httpx
            async with httpx.AsyncClient(timeout=2.0) as client:
                # Try to connect to server endpoint (simplified for demonstration)
                response = await client.get(f"{server_config.endpoint}/health")
                return response.status_code == 200
                
        except Exception:
            return False
    
    def get_comprehensive_status_report(self) -> Dict[str, Any]:
        """Generate comprehensive status report for MCP infrastructure."""
        current_time = time.time()
        
        # Calculate aggregate metrics
        total_operations = len(self.operation_metrics)
        successful_operations = sum(1 for m in self.operation_metrics if m.success)
        
        success_rate = successful_operations / total_operations if total_operations > 0 else 0.0
        
        # Calculate average response time
        successful_metrics = [m for m in self.operation_metrics if m.success]
        avg_response_time = sum(m.duration_ms for m in successful_metrics) / len(successful_metrics) if successful_metrics else 0.0
        
        # Server health summary
        server_health_summary = {}
        for server_name, health in self.server_health.items():
            server_health_summary[server_name] = {
                'status': health.status.value,
                'response_time_ms': health.response_time_ms,
                'error_rate': health.error_rate,
                'uptime_percentage': health.uptime_percentage,
                'consecutive_failures': health.consecutive_failures,
                'last_health_check': health.last_health_check
            }
        
        # Recent alerts
        recent_alerts = [
            {
                'type': alert.alert_type,
                'severity': alert.severity,
                'description': alert.description,
                'triggered_at': alert.triggered_at
            }
            for alert in self.performance_alerts[-10:]  # Last 10 alerts
        ]
        
        return {
            'report_timestamp': current_time,
            'infrastructure_health': {
                'overall_success_rate': success_rate,
                'average_response_time_ms': avg_response_time,
                'total_operations': total_operations,
                'servers_healthy': sum(1 for h in self.server_health.values() if h.status == MCPServerStatus.HEALTHY),
                'servers_total': len(self.server_health)
            },
            'server_health': server_health_summary,
            'recent_alerts': recent_alerts,
            'circuit_breaker_states': {
                name: breaker.state 
                for name, breaker in self.mcp_client.circuit_breakers.items()
            },
            'performance_targets': {
                'target_response_time_ms': self.target_response_time_ms,
                'target_success_rate': self.target_success_rate,
                'target_uptime': self.target_uptime
            }
        }
    
    async def start_monitoring(self):
        """Start background monitoring tasks."""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.logger.info("MCP monitoring started")
    
    async def stop_monitoring(self):
        """Stop background monitoring tasks."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
            self.logger.info("MCP monitoring stopped")
    
    async def _monitoring_loop(self):
        """Background monitoring loop for health checks and metrics cleanup."""
        while True:
            try:
                # Perform health checks
                await self.perform_health_check()
                
                # Clean up old metrics and alerts
                await self._cleanup_old_alerts()
                
                # Generate automated reports (if configured)
                if self.config.get('enable_automated_reporting', False):
                    await self._generate_automated_report()
                
                # Wait for next health check interval
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)  # Brief pause on error
    
    async def _cleanup_old_alerts(self):
        """Clean up old performance alerts."""
        cutoff_time = time.time() - (24 * 3600)  # 24 hours
        
        self.performance_alerts = [
            alert for alert in self.performance_alerts
            if alert.triggered_at > cutoff_time
        ]
    
    async def _generate_automated_report(self):
        """Generate automated performance reports."""
        try:
            report = self.get_comprehensive_status_report()
            
            # Save report to file (if configured)
            if self.config.get('save_reports_to_file', False):
                reports_dir = Path('logs/mcp_reports')
                reports_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = int(time.time())
                report_file = reports_dir / f'mcp_status_{timestamp}.json'
                
                with open(report_file, 'w') as f:
                    json.dump(report, f, indent=2)
                
                self.logger.info(f"Automated report saved to {report_file}")
                
        except Exception as e:
            self.logger.error(f"Failed to generate automated report: {e}")
    
    def force_circuit_breaker_reset(self, server_name: str) -> bool:
        """Force reset circuit breaker for a specific server."""
        if server_name in self.mcp_client.circuit_breakers:
            breaker = self.mcp_client.circuit_breakers[server_name]
            breaker.state = "CLOSED"
            breaker.failure_count = 0
            breaker.last_failure_time = 0
            
            self.logger.info(f"Circuit breaker for {server_name} manually reset")
            return True
        
        return False
    
    def get_server_recommendations(self) -> List[Dict[str, Any]]:
        """Get operational recommendations based on current server health."""
        recommendations = []
        
        for server_name, health in self.server_health.items():
            if health.status == MCPServerStatus.UNHEALTHY:
                recommendations.append({
                    'server': server_name,
                    'priority': 'HIGH',
                    'action': 'investigate_server_health',
                    'description': f'Server {server_name} is unhealthy with {health.consecutive_failures} consecutive failures'
                })
            
            elif health.status == MCPServerStatus.DEGRADED:
                recommendations.append({
                    'server': server_name,
                    'priority': 'MEDIUM',
                    'action': 'optimize_performance',
                    'description': f'Server {server_name} showing performance degradation (response time: {health.response_time_ms:.1f}ms)'
                })
            
            if health.error_rate > self.alert_thresholds['error_rate']['WARNING']:
                recommendations.append({
                    'server': server_name,
                    'priority': 'MEDIUM',
                    'action': 'reduce_error_rate',
                    'description': f'Server {server_name} error rate {health.error_rate:.1%} exceeds warning threshold'
                })
        
        return recommendations
    
    def export_metrics_for_analysis(self, format: str = 'json') -> Union[str, Dict]:
        """Export metrics in various formats for external analysis."""
        metrics_data = {
            'export_timestamp': time.time(),
            'server_health': {
                name: {
                    'status': health.status.value,
                    'response_time_ms': health.response_time_ms,
                    'error_rate': health.error_rate,
                    'uptime_percentage': health.uptime_percentage,
                    'consecutive_failures': health.consecutive_failures
                }
                for name, health in self.server_health.items()
            },
            'operation_metrics': [
                {
                    'operation_type': m.operation_type.value,
                    'server_name': m.server_name,
                    'duration_ms': m.duration_ms,
                    'success': m.success,
                    'fallback_used': m.fallback_used,
                    'response_size_bytes': m.response_size_bytes,
                    'timestamp': m.start_time
                }
                for m in self.operation_metrics
            ],
            'performance_alerts': [
                {
                    'alert_type': alert.alert_type,
                    'severity': alert.severity,
                    'description': alert.description,
                    'triggered_at': alert.triggered_at,
                    'threshold_value': alert.threshold_value,
                    'current_value': alert.current_value
                }
                for alert in self.performance_alerts
            ]
        }
        
        if format == 'json':
            return json.dumps(metrics_data, indent=2)
        else:
            return metrics_data
    
    def validate_system_integration(self) -> Dict[str, Any]:
        """Validate MCP system integration and readiness."""
        validation_results = {
            'is_valid': True,
            'checks_passed': [],
            'checks_failed': [],
            'warnings': []
        }
        
        # Check 1: MCP client initialization
        try:
            if self.mcp_client and self.mcp_client.servers:
                validation_results['checks_passed'].append('mcp_client_initialized')
            else:
                validation_results['checks_failed'].append('mcp_client_not_initialized')
                validation_results['is_valid'] = False
        except Exception as e:
            validation_results['checks_failed'].append(f'mcp_client_error: {e}')
            validation_results['is_valid'] = False
        
        # Check 2: Circuit breaker configuration
        if self.mcp_client.circuit_breakers:
            validation_results['checks_passed'].append('circuit_breakers_configured')
        else:
            validation_results['warnings'].append('circuit_breakers_not_configured')
        
        # Check 3: Performance monitoring ready
        if self.server_health:
            validation_results['checks_passed'].append('health_monitoring_ready')
        else:
            validation_results['warnings'].append('health_monitoring_not_initialized')
        
        # Check 4: Configuration validation
        required_configs = ['health_check_interval_seconds', 'metrics_retention_hours']
        missing_configs = [cfg for cfg in required_configs if cfg not in self.config]
        
        if not missing_configs:
            validation_results['checks_passed'].append('configuration_complete')
        else:
            validation_results['warnings'].append(f'missing_configurations: {missing_configs}')
        
        return validation_results


# Convenience functions for easy integration

def create_enterprise_mcp_manager(config_path: Optional[str] = None) -> MCPClientManager:
    """Create enterprise MCP manager with configuration loading."""
    config = {}
    
    if config_path:
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Could not load config from {config_path}: {e}")
    
    return MCPClientManager(config)


async def test_mcp_client_manager():
    """Test function for MCP client manager functionality."""
    print("Testing MCP Client Manager Enterprise Features...")
    
    # Initialize manager
    config = {
        'health_check_interval_seconds': 10,
        'metrics_retention_hours': 1,
        'target_response_time_ms': 500,
        'enable_automated_reporting': False
    }
    
    manager = MCPClientManager(config)
    
    # Test basic functionality
    test_text = "Today we study chapter two verse twenty five"
    result = await manager.process_text_with_monitoring(test_text)
    
    print(f"✅ Processed text: '{test_text}' -> Context: {result.context_type.value}")
    
    # Test health check
    health_status = await manager.perform_health_check()
    print(f"✅ Health check completed for {len(health_status)} servers")
    
    # Test status report
    status_report = manager.get_comprehensive_status_report()
    print(f"✅ Status report generated with {status_report['infrastructure_health']['total_operations']} operations")
    
    # Test system validation
    validation = manager.validate_system_integration()
    print(f"✅ System validation: {'PASSED' if validation['is_valid'] else 'FAILED'}")
    
    return manager


if __name__ == "__main__":
    asyncio.run(test_mcp_client_manager())