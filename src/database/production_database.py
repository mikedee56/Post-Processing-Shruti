"""
Production Database Management
High-availability database setup with connection pooling, monitoring, and failover
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime, timedelta
import json

try:
    import psycopg2
    from psycopg2 import pool, sql
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import sqlalchemy
    from sqlalchemy import create_engine, text
    from sqlalchemy.pool import QueuePool
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    host: str
    port: int
    database: str
    username: str
    password: str
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    health_check_interval: int = 60
    
    
@dataclass
class ConnectionMetrics:
    """Database connection metrics"""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0
    average_response_time: float = 0.0
    last_health_check: Optional[datetime] = None


class DatabaseConnectionPool:
    """Production-ready database connection pool with monitoring"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics = ConnectionMetrics()
        self.pool = None
        self.redis_pool = None
        self._lock = threading.RLock()
        self._health_check_thread = None
        self._shutdown = False
        
        self._initialize_pools()
        self._start_health_monitoring()
        
    def _initialize_pools(self):
        """Initialize database connection pools"""
        try:
            # PostgreSQL connection pool
            if POSTGRESQL_AVAILABLE and SQLALCHEMY_AVAILABLE:
                connection_string = (
                    f"postgresql://{self.config.username}:{self.config.password}"
                    f"@{self.config.host}:{self.config.port}/{self.config.database}"
                )
                
                self.engine = create_engine(
                    connection_string,
                    poolclass=QueuePool,
                    pool_size=self.config.pool_size,
                    max_overflow=self.config.max_overflow,
                    pool_timeout=self.config.pool_timeout,
                    pool_recycle=self.config.pool_recycle,
                    echo=False,
                    pool_pre_ping=True,  # Verify connections before use
                )
                
                self.logger.info(
                    "PostgreSQL connection pool initialized",
                    pool_size=self.config.pool_size,
                    max_overflow=self.config.max_overflow
                )
                
        except Exception as e:
            self.logger.error("Failed to initialize database pools", exception=e)
            raise
            
    def _start_health_monitoring(self):
        """Start health monitoring thread"""
        self._health_check_thread = threading.Thread(
            target=self._health_monitoring_loop,
            daemon=True
        )
        self._health_check_thread.start()
        
    def _health_monitoring_loop(self):
        """Continuous health monitoring of database connections"""
        while not self._shutdown:
            try:
                self._perform_health_check()
                time.sleep(self.config.health_check_interval)
            except Exception as e:
                self.logger.error("Health check failed", exception=e)
                time.sleep(min(self.config.health_check_interval, 30))
                
    def _perform_health_check(self):
        """Perform health check on database connections"""
        start_time = time.time()
        
        try:
            with self.get_connection() as conn:
                # Simple health check query
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
                
            response_time = time.time() - start_time
            
            with self._lock:
                self.metrics.average_response_time = (
                    (self.metrics.average_response_time + response_time) / 2
                )
                self.metrics.last_health_check = datetime.utcnow()
                
                # Update connection metrics from pool
                if hasattr(self.engine.pool, 'size'):
                    pool_status = self.engine.pool.status()
                    self.metrics.total_connections = self.engine.pool.size()
                    self.metrics.active_connections = pool_status.get('pool_size', 0) - pool_status.get('checked_in', 0)
                    self.metrics.idle_connections = pool_status.get('checked_in', 0)
                    
        except Exception as e:
            with self._lock:
                self.metrics.failed_connections += 1
            self.logger.warning("Database health check failed", exception=e)
            
    @contextmanager
    def get_connection(self):
        """Get database connection from pool"""
        connection = None
        try:
            connection = self.engine.connect()
            yield connection
        except Exception as e:
            with self._lock:
                self.metrics.failed_connections += 1
            self.logger.error("Database connection failed", exception=e)
            raise
        finally:
            if connection:
                connection.close()
                
    def execute_query(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """Execute query with connection pooling"""
        with self.get_connection() as conn:
            result = conn.execute(text(query), parameters or {})
            return [dict(row._mapping) for row in result]
            
    def execute_transaction(self, operations: List[Dict[str, Any]]) -> bool:
        """Execute multiple operations in a transaction"""
        with self.get_connection() as conn:
            trans = conn.begin()
            try:
                for operation in operations:
                    query = operation.get('query')
                    parameters = operation.get('parameters', {})
                    conn.execute(text(query), parameters)
                    
                trans.commit()
                return True
            except Exception as e:
                trans.rollback()
                self.logger.error("Transaction failed", exception=e)
                raise
                
    def get_metrics(self) -> Dict[str, Any]:
        """Get connection pool metrics"""
        with self._lock:
            return {
                'total_connections': self.metrics.total_connections,
                'active_connections': self.metrics.active_connections,
                'idle_connections': self.metrics.idle_connections,
                'failed_connections': self.metrics.failed_connections,
                'average_response_time': self.metrics.average_response_time,
                'last_health_check': self.metrics.last_health_check.isoformat() if self.metrics.last_health_check else None,
                'pool_size': self.config.pool_size,
                'max_overflow': self.config.max_overflow,
            }
            
    def close(self):
        """Close connection pools and cleanup"""
        self._shutdown = True
        
        if self._health_check_thread and self._health_check_thread.is_alive():
            self._health_check_thread.join(timeout=5)
            
        if self.engine:
            self.engine.dispose()
            
        self.logger.info("Database connection pools closed")


class ProductionDatabaseManager:
    """Production database manager with high availability features"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.connection_pools = {}
        self.redis_client = None
        
        self._initialize_databases()
        self._initialize_redis()
        
    def _initialize_databases(self):
        """Initialize all database connections"""
        # Primary database
        primary_config = self.config.get('primary', {})
        if primary_config:
            # Validate required configuration parameters
            required_fields = ['host', 'port', 'database', 'username', 'password']
            missing_fields = [field for field in required_fields if field not in primary_config]
            
            if missing_fields:
                raise ValueError(f"Missing required database configuration fields: {missing_fields}")
            
            db_config = DatabaseConfig(
                host=primary_config['host'],
                port=primary_config['port'],
                database=primary_config['database'],
                username=primary_config['username'],
                password=primary_config['password'],
                pool_size=primary_config.get('pool_size', 20),
                max_overflow=primary_config.get('max_overflow', 30),
                pool_timeout=primary_config.get('pool_timeout', 30),
            )
            
            self.connection_pools['primary'] = DatabaseConnectionPool(db_config)
            self.logger.info("Primary database connection pool initialized")
        else:
            self.logger.warning("No primary database configuration provided")
            
        # Read replicas (if configured)
        replicas_config = self.config.get('replicas', [])
        for i, replica_config in enumerate(replicas_config):
            # Validate required configuration parameters for replicas
            required_fields = ['host', 'port', 'database', 'username', 'password']
            missing_fields = [field for field in required_fields if field not in replica_config]
            
            if missing_fields:
                self.logger.error(f"Missing required fields for replica {i}: {missing_fields}")
                continue
                
            replica_db_config = DatabaseConfig(
                host=replica_config['host'],
                port=replica_config['port'],
                database=replica_config['database'],
                username=replica_config['username'],
                password=replica_config['password'],
                pool_size=replica_config.get('pool_size', 10),
                max_overflow=replica_config.get('max_overflow', 20),
            )
            
            self.connection_pools[f'replica_{i}'] = DatabaseConnectionPool(replica_db_config)
            self.logger.info(f"Read replica {i} connection pool initialized")
            
    def _initialize_redis(self):
        """Initialize Redis connection for caching"""
        if not REDIS_AVAILABLE:
            self.logger.warning("Redis not available, caching disabled")
            return
            
        redis_config = self.config.get('redis', {})
        if redis_config:
            try:
                self.redis_client = redis.ConnectionPool(
                    host=redis_config['host'],
                    port=redis_config.get('port', 6379),
                    password=redis_config.get('password'),
                    db=redis_config.get('db', 0),
                    max_connections=redis_config.get('max_connections', 20),
                    decode_responses=True
                )
                
                # Test Redis connection
                r = redis.Redis(connection_pool=self.redis_client)
                r.ping()
                
                self.logger.info("Redis connection pool initialized")
                
            except Exception as e:
                self.logger.error("Failed to initialize Redis", exception=e)
                self.redis_client = None
                
    def get_database_connection(self, database: str = 'primary'):
        """Get database connection from specified pool"""
        if database not in self.connection_pools:
            raise ValueError(f"Database '{database}' not configured")
            
        return self.connection_pools[database].get_connection()
        
    def execute_read_query(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """Execute read query on best available database"""
        # Try read replicas first, fallback to primary
        for pool_name in ['replica_0', 'replica_1', 'primary']:
            if pool_name in self.connection_pools:
                try:
                    return self.connection_pools[pool_name].execute_query(query, parameters)
                except Exception as e:
                    self.logger.warning(f"Read query failed on {pool_name}", exception=e)
                    continue
                    
        raise Exception("All database connections failed for read query")
        
    def execute_write_query(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """Execute write query on primary database"""
        if 'primary' not in self.connection_pools:
            raise Exception("Primary database not configured")
            
        return self.connection_pools['primary'].execute_query(query, parameters)
        
    def execute_transaction(self, operations: List[Dict[str, Any]]) -> bool:
        """Execute transaction on primary database"""
        if 'primary' not in self.connection_pools:
            raise Exception("Primary database not configured")
            
        return self.connection_pools['primary'].execute_transaction(operations)
        
    def cache_get(self, key: str) -> Optional[str]:
        """Get value from Redis cache"""
        if not self.redis_client:
            return None
            
        try:
            r = redis.Redis(connection_pool=self.redis_client)
            return r.get(key)
        except Exception as e:
            self.logger.warning("Cache get failed", key=key, exception=e)
            return None
            
    def cache_set(self, key: str, value: str, ttl_seconds: Optional[int] = None):
        """Set value in Redis cache"""
        if not self.redis_client:
            return
            
        try:
            r = redis.Redis(connection_pool=self.redis_client)
            r.set(key, value, ex=ttl_seconds)
        except Exception as e:
            self.logger.warning("Cache set failed", key=key, exception=e)
            
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all database connections"""
        status = {
            'overall_health': 'healthy',
            'databases': {},
            'redis': {'status': 'disabled' if not self.redis_client else 'healthy'}
        }
        
        overall_healthy = True
        
        for pool_name, pool in self.connection_pools.items():
            try:
                pool_metrics = pool.get_metrics()
                last_check = pool_metrics.get('last_health_check')
                
                if last_check:
                    last_check_time = datetime.fromisoformat(last_check.replace('Z', '+00:00'))
                    time_since_check = datetime.utcnow() - last_check_time.replace(tzinfo=None)
                    
                    if time_since_check > timedelta(minutes=5):
                        pool_status = 'stale'
                        overall_healthy = False
                    else:
                        pool_status = 'healthy'
                else:
                    pool_status = 'unknown'
                    overall_healthy = False
                    
                status['databases'][pool_name] = {
                    'status': pool_status,
                    'metrics': pool_metrics
                }
                
            except Exception as e:
                status['databases'][pool_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                overall_healthy = False
                
        # Check Redis health
        if self.redis_client:
            try:
                r = redis.Redis(connection_pool=self.redis_client)
                r.ping()
                status['redis'] = {'status': 'healthy'}
            except Exception as e:
                status['redis'] = {'status': 'error', 'error': str(e)}
                overall_healthy = False
                
        if not overall_healthy:
            status['overall_health'] = 'degraded'
            
        return status
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive database metrics"""
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'databases': {},
            'redis': {}
        }
        
        for pool_name, pool in self.connection_pools.items():
            metrics['databases'][pool_name] = pool.get_metrics()
            
        if self.redis_client:
            try:
                r = redis.Redis(connection_pool=self.redis_client)
                info = r.info()
                metrics['redis'] = {
                    'connected_clients': info.get('connected_clients', 0),
                    'used_memory': info.get('used_memory', 0),
                    'total_connections_received': info.get('total_connections_received', 0),
                    'total_commands_processed': info.get('total_commands_processed', 0),
                }
            except Exception as e:
                metrics['redis'] = {'error': str(e)}
                
        return metrics
        
    def close(self):
        """Close all database connections"""
        for pool in self.connection_pools.values():
            pool.close()
            
        self.logger.info("All database connections closed")


# Global database manager instance
_database_manager = None


def initialize_database_production(config: Dict[str, Any]) -> ProductionDatabaseManager:
    """Initialize production database manager"""
    global _database_manager
    _database_manager = ProductionDatabaseManager(config)
    return _database_manager


def get_database_manager() -> Optional[ProductionDatabaseManager]:
    """Get the global database manager instance"""
    return _database_manager