"""
Production Monitoring and Observability Package
Comprehensive monitoring system with metrics, tracing, logging, and dashboards
"""

from monitoring.production_monitor import ProductionMonitor
from monitoring.distributed_tracer import DistributedTracer, initialize_tracing, get_tracer, trace
from monitoring.structured_logger import StructuredLogger, initialize_structured_logging, get_logger
from monitoring.performance_metrics_collector import (
    PerformanceMetricsCollector, 
    initialize_metrics_collection, 
    get_metrics_collector,
    record_metric
)
from monitoring.dashboard_integration import (
    DashboardIntegrationManager,
    initialize_dashboard_integration,
    get_dashboard_integration
)

__all__ = [
    # Core monitoring
    'ProductionMonitor',
    
    # Distributed tracing
    'DistributedTracer',
    'initialize_tracing',
    'get_tracer', 
    'trace',
    
    # Structured logging
    'StructuredLogger',
    'initialize_structured_logging',
    'get_logger',
    
    # Performance metrics
    'PerformanceMetricsCollector',
    'initialize_metrics_collection',
    'get_metrics_collector',
    'record_metric',
    
    # Dashboard integration
    'DashboardIntegrationManager',
    'initialize_dashboard_integration',
    'get_dashboard_integration',
    
    # Utilities
    'initialize_full_observability_stack'
]


def initialize_full_observability_stack(config: dict) -> dict:
    """
    Initialize the complete observability stack
    Returns handles to all monitoring components
    """
    components = {}
    
    try:
        # Initialize structured logging
        logging_config = config.get('logging', {})
        structured_logger = initialize_structured_logging(logging_config)
        components['logger'] = structured_logger
        structured_logger.info("Structured logging initialized")
        
        # Initialize distributed tracing
        tracing_config = config.get('tracing', {})
        if tracing_config.get('enabled', True):
            service_name = config.get('service_name', 'asr-processor')
            tracer = initialize_tracing(service_name, tracing_config)
            components['tracer'] = tracer
            structured_logger.info("Distributed tracing initialized")
        
        # Initialize performance metrics collection
        metrics_config = config.get('metrics_collection', {})
        if metrics_config.get('enabled', True):
            metrics_collector = initialize_metrics_collection(metrics_config)
            components['metrics'] = metrics_collector
            structured_logger.info("Performance metrics collection initialized")
        
        # Initialize dashboard integration
        dashboard_config = config.get('dashboard_integration', {})
        if dashboard_config:
            dashboard_integration = initialize_dashboard_integration(dashboard_config)
            components['dashboards'] = dashboard_integration
            structured_logger.info("Dashboard integration initialized")
        
        # Initialize production monitor (main orchestrator)
        monitor_config = config.get('monitoring', {})
        production_monitor = ProductionMonitor(monitor_config)
        components['monitor'] = production_monitor
        structured_logger.info("Production monitor initialized")
        
        # Create standard dashboards and alerts
        if 'dashboards' in components:
            dashboard_integration.create_production_dashboard()
            dashboard_integration.create_performance_alerts()
            structured_logger.info("Standard dashboards and alerts created")
        
        structured_logger.info(
            "Full observability stack initialized successfully",
            components_count=len(components),
            components=list(components.keys())
        )
        
        return components
        
    except Exception as e:
        if 'logger' in components:
            components['logger'].error("Failed to initialize observability stack", exception=e)
        raise