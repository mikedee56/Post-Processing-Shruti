"""
Database and Storage Production Package
Production-ready database configurations, connection pooling, and data management
"""

from .production_database import (
    ProductionDatabaseManager,
    DatabaseConnectionPool,
    initialize_database_production
)
from .storage_manager import (
    ProductionStorageManager,
    BackupManager, 
    initialize_storage_production
)
from .data_migration import (
    DatabaseMigrator,
    initialize_migration_system
)

__all__ = [
    # Database Management
    'ProductionDatabaseManager',
    'DatabaseConnectionPool', 
    'initialize_database_production',
    
    # Storage Management
    'ProductionStorageManager',
    'BackupManager',
    'initialize_storage_production',
    
    # Data Migration
    'DatabaseMigrator',
    'initialize_migration_system',
    
    # Utilities
    'initialize_database_storage_infrastructure'
]


def initialize_database_storage_infrastructure(config: dict) -> dict:
    """
    Initialize the complete database and storage infrastructure
    Returns handles to all database and storage components
    """
    components = {}
    
    try:
        # Initialize database production setup
        database_config = config.get('database', {})
        if database_config:
            db_manager = initialize_database_production(database_config)
            components['database'] = db_manager
            
        # Initialize storage production setup  
        storage_config = config.get('storage', {})
        if storage_config:
            storage_manager = initialize_storage_production(storage_config)
            components['storage'] = storage_manager
            
        # Initialize migration system
        migration_config = config.get('migration', {})
        if migration_config.get('enabled', True):
            migrator = initialize_migration_system(migration_config)
            components['migrator'] = migrator
            
        return components
        
    except Exception as e:
        # Cleanup any partially initialized components
        for component_name, component in components.items():
            try:
                if hasattr(component, 'close'):
                    component.close()
            except:
                pass
        raise RuntimeError(f"Failed to initialize database/storage infrastructure: {e}")