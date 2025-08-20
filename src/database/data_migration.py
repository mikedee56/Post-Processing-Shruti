"""
Database Migration System
Production-ready database schema management and data migration tools
"""

import logging
import hashlib
import json
import os
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import sqlalchemy
    from sqlalchemy import create_engine, text, inspect
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


@dataclass
class Migration:
    """Database migration definition"""
    version: str
    name: str
    description: str
    up_sql: str
    down_sql: str
    checksum: str
    created_at: datetime
    dependencies: List[str]


@dataclass
class MigrationResult:
    """Result of migration execution"""
    version: str
    success: bool
    error_message: Optional[str] = None
    execution_time_seconds: float = 0.0
    applied_at: Optional[datetime] = None


class DatabaseMigrator:
    """Production database migration system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.engine = None
        self.migrations_dir = config.get('migrations_dir', 'migrations')
        self.migrations_table = config.get('migrations_table', 'schema_migrations')
        
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("SQLAlchemy is required for database migrations")
            
        self._initialize_database_connection()
        self._ensure_migrations_table()
        
    def _initialize_database_connection(self):
        """Initialize database connection for migrations"""
        db_config = self.config.get('database', {})
        connection_string = self._build_connection_string(db_config)
        
        self.engine = create_engine(
            connection_string,
            echo=self.config.get('verbose', False),
            pool_pre_ping=True
        )
        
        # Test connection
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            
        self.logger.info("Database connection established for migrations")
        
    def _build_connection_string(self, db_config: Dict[str, Any]) -> str:
        """Build database connection string"""
        db_type = db_config.get('type', 'postgresql')
        host = db_config.get('host', 'localhost')
        port = db_config.get('port', 5432)
        database = db_config.get('database')
        username = db_config.get('username')
        password = db_config.get('password')
        
        return f"{db_type}://{username}:{password}@{host}:{port}/{database}"
        
    def _ensure_migrations_table(self):
        """Create migrations table if it doesn't exist"""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.migrations_table} (
            version VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            checksum VARCHAR(64) NOT NULL,
            applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            execution_time_seconds FLOAT
        )
        """
        
        with self.engine.connect() as conn:
            trans = conn.begin()
            try:
                conn.execute(text(create_table_sql))
                trans.commit()
                self.logger.info(f"Migrations table '{self.migrations_table}' ensured")
            except Exception as e:
                trans.rollback()
                self.logger.error("Failed to create migrations table", exception=e)
                raise
                
    def load_migrations_from_directory(self) -> List[Migration]:
        """Load migration files from directory"""
        migrations = []
        migrations_path = Path(self.migrations_dir)
        
        if not migrations_path.exists():
            self.logger.warning(f"Migrations directory not found: {self.migrations_dir}")
            return migrations
            
        # Load migration files (expected format: V001__migration_name.sql)
        for migration_file in sorted(migrations_path.glob("V*__*.sql")):
            try:
                migration = self._parse_migration_file(migration_file)
                migrations.append(migration)
                self.logger.debug(f"Loaded migration: {migration.version}")
            except Exception as e:
                self.logger.error(f"Failed to load migration {migration_file}", exception=e)
                
        return migrations
        
    def _parse_migration_file(self, file_path: Path) -> Migration:
        """Parse migration file"""
        filename = file_path.name
        
        # Extract version and name from filename (V001__migration_name.sql)
        parts = filename.split('__', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid migration filename format: {filename}")
            
        version = parts[0][1:]  # Remove 'V' prefix
        name = parts[1].replace('.sql', '').replace('_', ' ').title()
        
        # Read migration content
        content = file_path.read_text(encoding='utf-8')
        
        # Parse migration sections (-- +migrate Up / -- +migrate Down)
        up_sql, down_sql, description = self._parse_migration_content(content)
        
        # Calculate checksum
        checksum = hashlib.sha256(content.encode()).hexdigest()
        
        return Migration(
            version=version,
            name=name,
            description=description,
            up_sql=up_sql,
            down_sql=down_sql,
            checksum=checksum,
            created_at=datetime.fromtimestamp(file_path.stat().st_mtime),
            dependencies=[]  # Could be parsed from comments
        )
        
    def _parse_migration_content(self, content: str) -> tuple[str, str, str]:
        """Parse migration content into up/down sections"""
        lines = content.split('\n')
        
        description = ""
        up_sql = []
        down_sql = []
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('-- Description:'):
                description = line.replace('-- Description:', '').strip()
            elif line == '-- +migrate Up':
                current_section = 'up'
            elif line == '-- +migrate Down':
                current_section = 'down'
            elif current_section == 'up' and not line.startswith('--'):
                up_sql.append(line)
            elif current_section == 'down' and not line.startswith('--'):
                down_sql.append(line)
                
        return '\n'.join(up_sql).strip(), '\n'.join(down_sql).strip(), description
        
    def get_applied_migrations(self) -> List[str]:
        """Get list of applied migration versions"""
        with self.engine.connect() as conn:
            result = conn.execute(text(f"SELECT version FROM {self.migrations_table} ORDER BY version"))
            return [row[0] for row in result]
            
    def get_pending_migrations(self, migrations: List[Migration]) -> List[Migration]:
        """Get list of migrations that need to be applied"""
        applied_versions = set(self.get_applied_migrations())
        return [m for m in migrations if m.version not in applied_versions]
        
    def validate_migration_integrity(self, migrations: List[Migration]) -> List[str]:
        """Validate migration integrity and checksums"""
        issues = []
        
        # Check for applied migrations with different checksums
        with self.engine.connect() as conn:
            result = conn.execute(text(f"SELECT version, checksum FROM {self.migrations_table}"))
            applied_checksums = {row[0]: row[1] for row in result}
            
        for migration in migrations:
            if migration.version in applied_checksums:
                stored_checksum = applied_checksums[migration.version]
                if stored_checksum != migration.checksum:
                    issues.append(
                        f"Migration {migration.version} checksum mismatch. "
                        f"Expected: {migration.checksum}, Stored: {stored_checksum}"
                    )
                    
        # Check for version gaps
        versions = [m.version for m in migrations]
        expected_versions = [f"{i:03d}" for i in range(1, len(versions) + 1)]
        
        if versions != expected_versions:
            missing = set(expected_versions) - set(versions)
            if missing:
                issues.append(f"Missing migration versions: {sorted(missing)}")
                
        return issues
        
    def apply_migration(self, migration: Migration) -> MigrationResult:
        """Apply a single migration"""
        start_time = datetime.now()
        
        try:
            with self.engine.connect() as conn:
                trans = conn.begin()
                
                try:
                    # Execute migration SQL
                    if migration.up_sql:
                        for statement in migration.up_sql.split(';'):
                            statement = statement.strip()
                            if statement:
                                conn.execute(text(statement))
                                
                    # Record migration in migrations table
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    insert_sql = text(f"""
                        INSERT INTO {self.migrations_table} 
                        (version, name, description, checksum, applied_at, execution_time_seconds)
                        VALUES (:version, :name, :description, :checksum, :applied_at, :execution_time)
                    """)
                    
                    conn.execute(insert_sql, {
                        'version': migration.version,
                        'name': migration.name,
                        'description': migration.description,
                        'checksum': migration.checksum,
                        'applied_at': start_time,
                        'execution_time': execution_time
                    })
                    
                    trans.commit()
                    
                    self.logger.info(
                        f"Applied migration {migration.version}: {migration.name}",
                        execution_time=execution_time
                    )
                    
                    return MigrationResult(
                        version=migration.version,
                        success=True,
                        execution_time_seconds=execution_time,
                        applied_at=start_time
                    )
                    
                except Exception as e:
                    trans.rollback()
                    raise e
                    
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(
                f"Failed to apply migration {migration.version}",
                exception=e,
                execution_time=execution_time
            )
            
            return MigrationResult(
                version=migration.version,
                success=False,
                error_message=str(e),
                execution_time_seconds=execution_time
            )
            
    def rollback_migration(self, version: str, migration: Migration) -> MigrationResult:
        """Rollback a specific migration"""
        start_time = datetime.now()
        
        try:
            with self.engine.connect() as conn:
                trans = conn.begin()
                
                try:
                    # Execute rollback SQL
                    if migration.down_sql:
                        for statement in migration.down_sql.split(';'):
                            statement = statement.strip()
                            if statement:
                                conn.execute(text(statement))
                                
                    # Remove from migrations table
                    delete_sql = text(f"DELETE FROM {self.migrations_table} WHERE version = :version")
                    conn.execute(delete_sql, {'version': version})
                    
                    trans.commit()
                    
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    self.logger.info(
                        f"Rolled back migration {version}",
                        execution_time=execution_time
                    )
                    
                    return MigrationResult(
                        version=version,
                        success=True,
                        execution_time_seconds=execution_time,
                        applied_at=start_time
                    )
                    
                except Exception as e:
                    trans.rollback()
                    raise e
                    
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(
                f"Failed to rollback migration {version}",
                exception=e,
                execution_time=execution_time
            )
            
            return MigrationResult(
                version=version,
                success=False,
                error_message=str(e),
                execution_time_seconds=execution_time
            )
            
    def migrate_up(self, target_version: Optional[str] = None) -> List[MigrationResult]:
        """Apply all pending migrations up to target version"""
        migrations = self.load_migrations_from_directory()
        
        # Validate migration integrity
        issues = self.validate_migration_integrity(migrations)
        if issues:
            raise ValueError(f"Migration integrity issues found: {issues}")
            
        # Get pending migrations
        pending = self.get_pending_migrations(migrations)
        
        if target_version:
            # Filter to only apply up to target version
            pending = [m for m in pending if m.version <= target_version]
            
        if not pending:
            self.logger.info("No pending migrations to apply")
            return []
            
        self.logger.info(f"Applying {len(pending)} migrations")
        
        results = []
        for migration in pending:
            result = self.apply_migration(migration)
            results.append(result)
            
            if not result.success:
                self.logger.error(f"Migration failed, stopping at version {migration.version}")
                break
                
        return results
        
    def migrate_down(self, target_version: str) -> List[MigrationResult]:
        """Rollback migrations down to target version"""
        applied_versions = self.get_applied_migrations()
        migrations = self.load_migrations_from_directory()
        migration_map = {m.version: m for m in migrations}
        
        # Find migrations to rollback (in reverse order)
        to_rollback = []
        for version in reversed(applied_versions):
            if version > target_version:
                if version in migration_map:
                    to_rollback.append((version, migration_map[version]))
                else:
                    self.logger.warning(f"Migration file not found for version {version}")
                    
        if not to_rollback:
            self.logger.info("No migrations to rollback")
            return []
            
        self.logger.info(f"Rolling back {len(to_rollback)} migrations")
        
        results = []
        for version, migration in to_rollback:
            result = self.rollback_migration(version, migration)
            results.append(result)
            
            if not result.success:
                self.logger.error(f"Rollback failed, stopping at version {version}")
                break
                
        return results
        
    def get_migration_status(self) -> Dict[str, Any]:
        """Get comprehensive migration status"""
        migrations = self.load_migrations_from_directory()
        applied_versions = set(self.get_applied_migrations())
        pending = self.get_pending_migrations(migrations)
        
        # Get applied migration details
        with self.engine.connect() as conn:
            result = conn.execute(text(f"""
                SELECT version, name, applied_at, execution_time_seconds 
                FROM {self.migrations_table} 
                ORDER BY version
            """))
            applied_details = [
                {
                    'version': row[0],
                    'name': row[1],
                    'applied_at': row[2].isoformat() if row[2] else None,
                    'execution_time_seconds': row[3]
                }
                for row in result
            ]
            
        return {
            'total_migrations': len(migrations),
            'applied_count': len(applied_versions),
            'pending_count': len(pending),
            'applied_migrations': applied_details,
            'pending_migrations': [
                {
                    'version': m.version,
                    'name': m.name,
                    'description': m.description
                }
                for m in pending
            ],
            'integrity_issues': self.validate_migration_integrity(migrations)
        }
        
    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            self.logger.info("Migration database connection closed")


# Global migrator instance
_migrator = None


def initialize_migration_system(config: Dict[str, Any]) -> DatabaseMigrator:
    """Initialize database migration system"""
    global _migrator
    _migrator = DatabaseMigrator(config)
    return _migrator


def get_migrator() -> Optional[DatabaseMigrator]:
    """Get the global migrator instance"""
    return _migrator