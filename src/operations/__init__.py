"""
Operational Procedures Package
Production operations, incident response, maintenance, and runbook management
"""

from .incident_response import (
    IncidentManager,
    IncidentSeverity,
    IncidentStatus,
    initialize_incident_management
)
from .maintenance_manager import (
    MaintenanceManager,
    MaintenanceWindow,
    MaintenanceType,
    initialize_maintenance_management
)
from .runbook_manager import (
    RunbookManager,
    RunbookCategory,
    initialize_runbook_management
)
from .operations_coordinator import (
    OperationsCoordinator,
    initialize_operations_management
)

__all__ = [
    # Incident Management
    'IncidentManager',
    'IncidentSeverity',
    'IncidentStatus',
    'initialize_incident_management',
    
    # Maintenance Management
    'MaintenanceManager', 
    'MaintenanceWindow',
    'MaintenanceType',
    'initialize_maintenance_management',
    
    # Runbook Management
    'RunbookManager',
    'RunbookCategory',
    'initialize_runbook_management',
    
    # Operations Coordination
    'OperationsCoordinator',
    'initialize_operations_management',
    
    # Utilities
    'initialize_operational_procedures'
]


def initialize_operational_procedures(config: dict) -> dict:
    """
    Initialize the complete operational procedures system
    Returns handles to all operational components
    """
    components = {}
    
    try:
        # Initialize incident management
        incident_config = config.get('incident_response', {})
        if incident_config:
            incident_manager = initialize_incident_management(incident_config)
            components['incident_manager'] = incident_manager
            
        # Initialize maintenance management
        maintenance_config = config.get('maintenance', {})
        if maintenance_config:
            maintenance_manager = initialize_maintenance_management(maintenance_config)
            components['maintenance_manager'] = maintenance_manager
            
        # Initialize runbook management
        runbook_config = config.get('runbooks', {})
        if runbook_config:
            runbook_manager = initialize_runbook_management(runbook_config)
            components['runbook_manager'] = runbook_manager
            
        # Initialize operations coordinator
        operations_config = config
        operations_coordinator = initialize_operations_management(operations_config)
        components['operations_coordinator'] = operations_coordinator
        
        return components
        
    except Exception as e:
        # Cleanup any partially initialized components
        for component_name, component in components.items():
            try:
                if hasattr(component, 'shutdown'):
                    component.shutdown()
            except:
                pass
        raise RuntimeError(f"Failed to initialize operational procedures: {e}")


def shutdown_operational_procedures(components: dict) -> None:
    """
    Gracefully shutdown all operational components
    """
    for component_name, component in components.items():
        try:
            if hasattr(component, 'shutdown'):
                component.shutdown()
                print(f"Shutdown {component_name} successfully")
        except Exception as e:
            print(f"Error shutting down {component_name}: {e}")


def get_operational_health_status(components: dict) -> dict:
    """
    Get health status of all operational components
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