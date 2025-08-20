"""
Scalability Infrastructure Package
Production-ready load balancing, auto-scaling, and distributed processing
"""

from .load_balancer import (
    LoadBalancer,
    LoadBalancingAlgorithm,
    RoundRobinAlgorithm,
    LeastConnectionsAlgorithm,
    WeightedRoundRobinAlgorithm,
    ResponseTimeAlgorithm,
    CPUBasedAlgorithm,
    HealthChecker,
    AutoScaler,
    initialize_load_balancer,
    get_load_balancer
)
from .distributed_processor import (
    DistributedProcessor,
    TaskQueue,
    WorkerManager,
    RedisTaskQueue,
    ProcessingTask,
    TaskStatus,
    TaskPriority,
    initialize_distributed_processing,
    get_distributed_processor
)

__all__ = [
    # Load Balancing
    'LoadBalancer',
    'LoadBalancingAlgorithm',
    'RoundRobinAlgorithm', 
    'LeastConnectionsAlgorithm',
    'WeightedRoundRobinAlgorithm',
    'ResponseTimeAlgorithm',
    'CPUBasedAlgorithm',
    'HealthChecker',
    'AutoScaler',
    'initialize_load_balancer',
    'get_load_balancer',
    
    # Distributed Processing
    'DistributedProcessor',
    'TaskQueue',
    'WorkerManager',
    'RedisTaskQueue',
    'ProcessingTask',
    'TaskStatus',
    'TaskPriority',
    'initialize_distributed_processing',
    'get_distributed_processor',
    
    # Utilities
    'initialize_scalability_infrastructure'
]


def initialize_scalability_infrastructure(config: dict) -> dict:
    """
    Initialize the complete scalability infrastructure
    Returns handles to all scalability components
    """
    components = {}
    
    try:
        # Initialize load balancing
        load_balancing_config = config.get('load_balancing', {})
        if load_balancing_config.get('enabled', True):
            load_balancer = initialize_load_balancer(load_balancing_config)
            components['load_balancer'] = load_balancer
            
        # Initialize distributed processing
        distributed_config = config.get('distributed_processing', {})
        if distributed_config.get('enabled', True):
            distributed_processor = initialize_distributed_processing(distributed_config)
            components['distributed_processor'] = distributed_processor
            
        # Start auto-scaling monitoring if enabled
        if 'load_balancer' in components:
            auto_scaling_config = load_balancing_config.get('auto_scaling', {})
            if auto_scaling_config.get('enabled', True):
                load_balancer = components['load_balancer']
                load_balancer.start_auto_scaling()
                
        # Start distributed worker management if enabled
        if 'distributed_processor' in components:
            worker_config = distributed_config.get('worker_management', {})
            if worker_config.get('enabled', True):
                distributed_processor = components['distributed_processor']
                distributed_processor.start_worker_management()
                
        return components
        
    except Exception as e:
        # Cleanup any partially initialized components
        for component_name, component in components.items():
            try:
                if hasattr(component, 'shutdown'):
                    component.shutdown()
            except:
                pass
        raise RuntimeError(f"Failed to initialize scalability infrastructure: {e}")


def shutdown_scalability_infrastructure(components: dict) -> None:
    """
    Gracefully shutdown all scalability components
    """
    for component_name, component in components.items():
        try:
            if hasattr(component, 'shutdown'):
                component.shutdown()
                print(f"Shutdown {component_name} successfully")
        except Exception as e:
            print(f"Error shutting down {component_name}: {e}")


def get_scalability_health_status(components: dict) -> dict:
    """
    Get health status of all scalability components
    """
    health_status = {
        'overall_health': 'healthy',
        'components': {}
    }
    
    overall_healthy = True
    
    for component_name, component in components.items():
        try:
            if hasattr(component, 'get_health_status'):
                status = component.get_health_status()
                health_status['components'][component_name] = status
                if status.get('status') != 'healthy':
                    overall_healthy = False
            else:
                health_status['components'][component_name] = {
                    'status': 'unknown',
                    'message': 'Health check not implemented'
                }
        except Exception as e:
            health_status['components'][component_name] = {
                'status': 'error',
                'message': str(e)
            }
            overall_healthy = False
    
    if not overall_healthy:
        health_status['overall_health'] = 'degraded'
        
    return health_status


def get_scalability_metrics(components: dict) -> dict:
    """
    Get comprehensive metrics from all scalability components
    """
    metrics = {
        'timestamp': __import__('datetime').datetime.utcnow().isoformat(),
        'components': {}
    }
    
    for component_name, component in components.items():
        try:
            if hasattr(component, 'get_metrics'):
                component_metrics = component.get_metrics()
                metrics['components'][component_name] = component_metrics
        except Exception as e:
            metrics['components'][component_name] = {
                'error': str(e)
            }
    
    return metrics